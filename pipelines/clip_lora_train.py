"""Tri-modal LoRA fine-tuning of CLIP to bridge the drone<->satellite gap.

For each training drone image we have a free positive pair (drone view, GT
satellite crop) plus a view-invariant caption of that location (produced offline
by ``caption_crops.py``). We LoRA-fine-tune HuggingFace CLIP so that all three
land close in one embedding space, using a symmetric InfoNCE over the three
pairings: drone<->text, sat<->text, drone<->sat. Each pairing has a CLI weight
(``--w-dt --w-st --w-ds``); ``--w-dt 0 --w-st 0`` trains the image-only control
that separates "text is a useful bridge" from "drone<->sat finetuning alone
helps". When train-band drone captions exist (``caption_crops.py --target drone
--band train``) a fourth drone<->own-caption term (``--w-ddt``) exposes the text
encoder to query-style captions, not just satellite-crop ones.

Consecutive drone frames overlap heavily, so two batch rows a few tens of
meters apart show the same ground and are *false negatives* for InfoNCE. Pairs
whose GT positions are closer than ``--neg-mask-m`` meters are therefore masked
out of the in-batch negatives of every pairing.

Backbone is any HF CLIP-family model id (peft can target its separate
q/k/v/out_proj + fc1/fc2 Linears cleanly, unlike open_clip's packed attention):
``openai/clip-vit-base-patch32`` / ``-large-patch14`` and SigLIP/SigLIP2 ids
(e.g. ``google/siglip2-base-patch16-384``) — input resolution, normalization,
and text padding are derived from the model config. Only the LoRA adapter is
saved (to ``weights/clip_lora/``); an existing adapter in --out-dir aborts the
run unless --overwrite is given.

Example (smoke test):
    python pipelines/clip_lora_train.py --flights 01 --limit 50 --epochs 1
Image-only control (no text losses, same training rows):
    python pipelines/clip_lora_train.py --flights all --w-dt 0 --w-st 0 \
        --out-dir weights/clip_lora_imgonly
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import (
    FLIGHTS_AVAILABLE, crop_gt_patch, get_flight_paths, load_flight,
    split_flight_rows,
)

torch.manual_seed(0)

BACKBONE      = "openai/clip-vit-base-patch32"
CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)
CAPTION_DIR   = "cache/captions"
PAIRS_DIR     = "cache/pairs"
OUT_DIR       = "weights/clip_lora"


# ---------- pair index (pre-crop satellite patches once) -------------------

def _load_captions(flight, caption_dir, target="sat"):
    path = os.path.join(caption_dir, f"{flight}_{target}.jsonl")
    caps = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    caps[r["filename"]] = r["caption"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return caps


def build_pair_index(flights, caption_dir, pairs_dir, limit, split):
    """Return [(drone_path, sat_patch_path, sat_caption, drone_caption, lat,
    lon), ...], pre-cropping sat patches to disk on first use so training reads
    plain image files. drone_caption is "" when the train band has not been
    captioned with --target drone --band train. Uses only the TRAIN spatial
    band of each flight (see split_flight_rows)."""
    import pandas as pd
    index = []
    for flight in flights:
        caps = _load_captions(flight, caption_dir, "sat")
        if not caps:
            print(f"  WARN flight {flight}: no captions in {caption_dir}; skipping")
            continue
        dcaps = _load_captions(flight, caption_dir, "drone")
        out_dir = os.path.join(pairs_dir, flight)
        os.makedirs(out_dir, exist_ok=True)
        _, drone_dir, drone_csv, _ = get_flight_paths(flight)
        df = pd.read_csv(drone_csv)
        df = split_flight_rows(df, which="train", test_frac=split["frac"],
                               axis=split["axis"], buffer_frac=split["buffer"])
        if limit is not None:
            df = df.iloc[:limit]

        tiles = None  # lazily load the (large) satellite only if a crop is missing
        n = n_dcap = 0
        for _, row in df.iterrows():
            fname = row["filename"]
            cap = caps.get(fname)
            if cap is None:
                continue
            sat_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".jpg")
            if not os.path.isfile(sat_path):
                if tiles is None:
                    tiles = load_flight(flight)[0]
                patch = crop_gt_patch(tiles, float(row["lat"]), float(row["lon"]),
                                     float(row["height"]), yaw_deg=0.0, flight=flight)
                if patch is None:
                    continue
                cv2.imwrite(sat_path, patch)
            dcap = dcaps.get(fname, "")
            n_dcap += bool(dcap)
            index.append((os.path.join(drone_dir, fname), sat_path, cap, dcap,
                          float(row["lat"]), float(row["lon"])))
            n += 1
        del tiles
        print(f"  flight {flight}: {n} training triples "
              f"({n_dcap} with a drone caption)")
    return index


# ---------- dataset --------------------------------------------------------

def _to_tensor(pil):
    arr = np.asarray(pil.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


class PairDataset(Dataset):
    """Drone image gets light augmentation; the satellite crop gets a mild
    scale jitter (sat_aug) so training covers the scale gap between GT
    footprint crops and the fixed-size gallery tiles used at test time."""

    def __init__(self, index, res, mean, std, train=True, sat_aug=True):
        self.index = index
        self.train = train
        self.res = res
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1)
        from torchvision import transforms
        bicubic = transforms.InterpolationMode.BICUBIC
        eval_tf = transforms.Compose([
            transforms.Resize(res, interpolation=bicubic),
            transforms.CenterCrop(res),
        ])
        if train:
            self.drone_tf = transforms.Compose([
                transforms.RandomResizedCrop(res, scale=(0.6, 1.0),
                                             interpolation=bicubic),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            ])
            self.sat_tf = transforms.RandomResizedCrop(
                res, scale=(0.7, 1.0), interpolation=bicubic) if sat_aug else eval_tf
        else:
            self.drone_tf = eval_tf
            self.sat_tf = eval_tf

    def __len__(self):
        return len(self.index)

    def _img(self, path, tf):
        bgr = cv2.imread(path)
        if bgr is None:
            return torch.zeros(3, self.res, self.res)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        t = _to_tensor(tf(pil))
        return (t - self.mean) / self.std

    def __getitem__(self, i):
        drone_path, sat_path, cap, dcap, lat, lon = self.index[i]
        drone = self._img(drone_path, self.drone_tf)
        sat   = self._img(sat_path, self.sat_tf)
        return drone, sat, cap, dcap, torch.tensor([lat, lon], dtype=torch.float64)


def make_collate(tokenizer, max_len, padding):
    def _tok(texts):
        return tokenizer(texts, padding=padding, truncation=True,
                         max_length=max_len, return_tensors="pt")

    def collate(batch):
        drone  = torch.stack([b[0] for b in batch])
        sat    = torch.stack([b[1] for b in batch])
        tok    = _tok([b[2] for b in batch])
        # Drone-caption channel: tokenize all rows ("" for missing) and carry a
        # validity mask so the loss can restrict to captioned rows.
        dtok   = _tok([b[3] if b[3] else "" for b in batch])
        dmask  = torch.tensor([bool(b[3]) for b in batch])
        coords = torch.stack([b[4] for b in batch])
        return (drone, sat, tok["input_ids"], tok["attention_mask"],
                dtok["input_ids"], dtok["attention_mask"], dmask, coords)
    return collate


# ---------- model ----------------------------------------------------------

def build_model(device, rank, alpha, dropout, grad_ckpt=False):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer
    clip = AutoModel.from_pretrained(BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    mtype = clip.config.model_type                       # clip | siglip | siglip2
    res = clip.config.vision_config.image_size
    max_len = clip.config.text_config.max_position_embeddings
    if "siglip" in mtype:
        mean = std = (0.5, 0.5, 0.5)
        padding = "max_length"   # SigLIP is trained with max-length padding
    else:
        mean, std = CLIP_MEAN, CLIP_STD
        padding = True
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    peft_model = get_peft_model(clip, cfg)
    if grad_ckpt:
        # Recompute activations in backward -> big VRAM cut (lets ViT-L/14 train at
        # a real batch on 40GB). use_reentrant=False backprops through the in-layer
        # LoRA params, so enable_input_require_grads is unnecessary (and CLIPModel
        # doesn't support it — no single input-embedding across its two encoders).
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.to(device)
    peft_model.print_trainable_parameters()
    print(f"  Backbone {BACKBONE} ({mtype}) | res {res} | text max_len {max_len}")
    # base_model.model is the backbone with LoRA injected in place; call its
    # feature heads directly (PeftModel has no get_image_features wrapper).
    return (peft_model, peft_model.base_model.model, tokenizer,
            {"res": res, "mean": mean, "std": std,
             "max_len": max_len, "padding": padding, "model_type": mtype})


def overlap_neg_mask(coords, thresh_m):
    """Boolean (B,B) mask, True where two batch rows look at (nearly) the same
    ground — GT positions closer than thresh_m meters — so they must not act as
    InfoNCE negatives of each other. Diagonal (the positives) stays False."""
    if thresh_m <= 0:
        return None
    lat, lon = coords[:, 0], coords[:, 1]
    mid = torch.deg2rad(0.5 * (lat[:, None] + lat[None, :]))
    dx = (lon[:, None] - lon[None, :]) * torch.cos(mid) * 111320.0
    dy = (lat[:, None] - lat[None, :]) * 110540.0
    mask = (dx * dx + dy * dy) < (thresh_m * thresh_m)
    mask.fill_diagonal_(False)
    return mask


def info_nce(a, b, logit_scale, neg_mask=None):
    """Symmetric CLIP-style contrastive loss between aligned rows of a and b.
    neg_mask=True entries are excluded from the negatives (false negatives
    from overlapping drone frames)."""
    logits = logit_scale * a @ b.t()
    if neg_mask is not None:
        logits = logits.masked_fill(neg_mask, -1e4)
    target = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) +
                  F.cross_entropy(logits.t(), target))


# ---------- train loop -----------------------------------------------------

def train(args):
    existing = [f for f in ("adapter_model.safetensors", "adapter_model.bin")
                if os.path.isfile(os.path.join(args.out_dir, f))]
    if existing and not args.overwrite:
        sys.exit(f"Refusing to overwrite existing adapter in {args.out_dir} "
                 f"({existing[0]} present). Use a fresh --out-dir or --overwrite.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    split = {"frac": args.test_frac, "axis": args.split_axis, "buffer": args.split_buffer}
    print(f"  Device: {device} | backbone {BACKBONE} | train band "
          f"(test_frac={args.test_frac}, axis={args.split_axis}, "
          f"buffer={args.split_buffer})")
    print(f"  Loss weights: dt={args.w_dt} st={args.w_st} ds={args.w_ds} "
          f"ddt={args.w_ddt} | neg-mask {args.neg_mask_m} m | "
          f"sat_aug={not args.no_sat_aug}")

    index = build_pair_index(flights, args.caption_dir, args.pairs_dir,
                             args.limit, split)
    if not index:
        sys.exit("No training triples found. Run caption_crops.py first.")
    n_dcap = sum(1 for r in index if r[3])
    print(f"  Total triples: {len(index)} ({n_dcap} with drone captions)")
    if args.w_ddt > 0 and n_dcap == 0:
        print("  NOTE: --w-ddt > 0 but no train-band drone captions found; the "
              "drone<->own-caption term is inactive. Produce them with "
              "caption_crops.py --target drone --band train.")

    peft_model, clip, tokenizer, info = build_model(
        device, args.rank, args.alpha, args.dropout, args.grad_ckpt)
    ds = PairDataset(index, info["res"], info["mean"], info["std"],
                     train=True, sat_aug=not args.no_sat_aug)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=make_collate(tokenizer, info["max_len"],
                                                info["padding"]),
                        pin_memory=True)

    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(loader) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    logit_scale = clip.logit_scale.exp().clamp(max=100.0).detach()
    use_text = args.w_dt > 0 or args.w_st > 0

    peft_model.train()
    for ep in range(args.epochs):
        pbar = tqdm(loader, desc=f"  epoch {ep+1}/{args.epochs}", unit="batch")
        for drone, sat, ids, attn, dids, dattn, dmask, coords in pbar:
            drone, sat = drone.to(device), sat.to(device)
            ids, attn = ids.to(device), attn.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                neg = overlap_neg_mask(coords.to(device), args.neg_mask_m)
                d = F.normalize(clip.get_image_features(pixel_values=drone), dim=-1)
                s = F.normalize(clip.get_image_features(pixel_values=sat),   dim=-1)
                loss, post = 0.0, {}
                if args.w_ds > 0:
                    l = info_nce(d, s, logit_scale, neg)
                    loss = loss + args.w_ds * l
                    post["ds"] = f"{l.item():.3f}"
                if use_text:
                    t = F.normalize(clip.get_text_features(
                        input_ids=ids, attention_mask=attn), dim=-1)
                    if args.w_dt > 0:
                        l = info_nce(d, t, logit_scale, neg)
                        loss = loss + args.w_dt * l
                        post["dt"] = f"{l.item():.3f}"
                    if args.w_st > 0:
                        l = info_nce(s, t, logit_scale, neg)
                        loss = loss + args.w_st * l
                        post["st"] = f"{l.item():.3f}"
                if args.w_ddt > 0 and int(dmask.sum()) >= 2:
                    sub = dmask.to(device)
                    td = F.normalize(clip.get_text_features(
                        input_ids=dids[dmask].to(device),
                        attention_mask=dattn[dmask].to(device)), dim=-1)
                    nsub = neg[sub][:, sub] if neg is not None else None
                    l = info_nce(d[sub], td, logit_scale, nsub)
                    loss = loss + args.w_ddt * l
                    post["ddt"] = f"{l.item():.3f}"
            if not torch.is_tensor(loss):   # every term weighted 0 / inactive
                continue
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                sched.step()
            pbar.set_postfix(post, refresh=False)

    os.makedirs(args.out_dir, exist_ok=True)
    peft_model.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "train_meta.json"), "w") as f:
        json.dump({"backbone": BACKBONE, "rank": args.rank, "alpha": args.alpha,
                   "epochs": args.epochs, "n_triples": len(index),
                   "n_drone_captions": n_dcap, "flights": args.flights,
                   "test_frac": args.test_frac, "split_axis": args.split_axis,
                   "split_buffer": args.split_buffer,
                   "loss_weights": {"dt": args.w_dt, "st": args.w_st,
                                    "ds": args.w_ds, "ddt": args.w_ddt},
                   "neg_mask_m": args.neg_mask_m,
                   "sat_aug": not args.no_sat_aug}, f, indent=2)
    print(f"  Saved LoRA adapter -> {args.out_dir}")


def main():
    global BACKBONE
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", nargs="+", default=["all"])
    ap.add_argument("--backbone", default=BACKBONE,
                    help="HF model id (openai/clip-vit-base-patch32 | -large-patch14 "
                         "| google/siglip2-base-patch16-384 | ...).")
    ap.add_argument("--caption-dir", default=CAPTION_DIR)
    ap.add_argument("--pairs-dir", default=PAIRS_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting an existing adapter in --out-dir.")
    ap.add_argument("--test-frac", type=float, default=0.25,
                    help="Spatial test band fraction held out per flight.")
    ap.add_argument("--split-axis", choices=["auto", "lat", "lon"], default="auto")
    ap.add_argument("--split-buffer", type=float, default=0.05,
                    help="Guard band fraction dropped between train and test "
                         "(removes seam overlap; the test band is unaffected).")
    ap.add_argument("--w-dt", type=float, default=1.0,
                    help="Weight of the drone<->sat-caption InfoNCE term.")
    ap.add_argument("--w-st", type=float, default=1.0,
                    help="Weight of the sat<->sat-caption InfoNCE term.")
    ap.add_argument("--w-ds", type=float, default=1.0,
                    help="Weight of the drone<->sat image InfoNCE term.")
    ap.add_argument("--w-ddt", type=float, default=1.0,
                    help="Weight of the drone<->own-caption term; only active for "
                         "rows with a train-band drone caption (see caption_crops.py "
                         "--target drone --band train). 0 disables.")
    ap.add_argument("--neg-mask-m", type=float, default=100.0,
                    help="Mask in-batch negatives whose GT positions are closer "
                         "than this many meters (overlapping frames are false "
                         "negatives). 0 disables.")
    ap.add_argument("--no-sat-aug", action="store_true",
                    help="Disable the satellite-crop scale jitter (RandomResizedCrop).")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="Gradient checkpointing: big VRAM cut, ~30%% slower (for ViT-L/14 on 40GB).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows per flight (smoke test).")
    args = ap.parse_args()
    BACKBONE = args.backbone
    train(args)


if __name__ == "__main__":
    main()
