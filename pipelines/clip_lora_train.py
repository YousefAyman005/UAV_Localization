"""Tri-modal LoRA fine-tuning of CLIP to bridge the drone<->satellite gap.

For each training drone image we have a free positive pair (drone view, GT
satellite crop) plus a view-invariant caption of that location (produced offline
by ``caption_crops.py``). We LoRA-fine-tune HuggingFace CLIP so that all three
land close in one embedding space, using a symmetric InfoNCE over the three
pairings: drone<->text, sat<->text, drone<->sat.

Backbone is HF ``openai/clip-vit-base-patch32`` (peft can target its separate
q/k/v/out_proj + fc1/fc2 Linears cleanly, unlike open_clip's packed attention).
Only the LoRA adapter is saved (to ``weights/clip_lora/``).

Example (smoke test):
    python pipelines/clip_lora_train.py --flights 01 --limit 50 --epochs 1
Full:
    python pipelines/clip_lora_train.py --flights 01 02 03 04 05 06 08 09
"""

import argparse
import json
import math
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

from helpers.utils import crop_gt_patch, get_flight_paths, load_flight

torch.manual_seed(0)

BACKBONE      = "openai/clip-vit-base-patch32"
CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)
RES           = 224
CAPTION_DIR   = "cache/captions"
PAIRS_DIR     = "cache/pairs"
OUT_DIR       = "weights/clip_lora"
TRAIN_FLIGHTS = ["01", "02", "03", "04", "05", "06", "08", "09"]


# ---------- pair index (pre-crop satellite patches once) -------------------

def _load_captions(flight, caption_dir):
    path = os.path.join(caption_dir, f"{flight}_sat.jsonl")
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


def build_pair_index(flights, caption_dir, pairs_dir, limit):
    """Return [(drone_path, sat_patch_path, caption), ...], pre-cropping sat
    patches to disk on first use so training reads plain image files."""
    import pandas as pd
    index = []
    for flight in flights:
        caps = _load_captions(flight, caption_dir)
        if not caps:
            print(f"  WARN flight {flight}: no captions in {caption_dir}; skipping")
            continue
        out_dir = os.path.join(pairs_dir, flight)
        os.makedirs(out_dir, exist_ok=True)
        _, drone_dir, drone_csv, _ = get_flight_paths(flight)
        df = pd.read_csv(drone_csv)
        if limit is not None:
            df = df.iloc[:limit]

        tiles = None  # lazily load the (large) satellite only if a crop is missing
        n = 0
        for _, row in df.iterrows():
            fname = row["filename"]
            cap = caps.get(fname)
            if cap is None:
                continue
            sat_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".jpg")
            if not os.path.isfile(sat_path):
                if tiles is None:
                    tiles = load_flight(flight)[0]
                yaw = float(row["Phi1"]) if "Phi1" in row.index else 0.0
                patch = crop_gt_patch(tiles, float(row["lat"]), float(row["lon"]),
                                     float(row["height"]), yaw_deg=yaw, flight=flight)
                if patch is None:
                    continue
                cv2.imwrite(sat_path, patch)
            index.append((os.path.join(drone_dir, fname), sat_path, cap))
            n += 1
        del tiles
        print(f"  flight {flight}: {n} training triples")
    return index


# ---------- dataset --------------------------------------------------------

def _normalize(t):
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std  = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (t - mean) / std


def _to_tensor(pil):
    arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


class PairDataset(Dataset):
    """Drone image gets light augmentation; satellite crop is deterministic."""

    def __init__(self, index, tokenizer, train=True):
        self.index = index
        self.tok = tokenizer
        self.train = train
        from torchvision import transforms
        if train:
            self.drone_tf = transforms.Compose([
                transforms.RandomResizedCrop(
                    RES, scale=(0.6, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            ])
        else:
            self.drone_tf = transforms.Compose([
                transforms.Resize(RES, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(RES),
            ])
        self.sat_tf = transforms.Compose([
            transforms.Resize(RES, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(RES),
        ])

    def __len__(self):
        return len(self.index)

    def _img(self, path, tf):
        bgr = cv2.imread(path)
        if bgr is None:
            return torch.zeros(3, RES, RES)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return _normalize(_to_tensor(tf(pil)))

    def __getitem__(self, i):
        drone_path, sat_path, caption = self.index[i]
        return self._img(drone_path, self.drone_tf), self._img(sat_path, self.sat_tf), caption


def make_collate(tokenizer):
    def collate(batch):
        drone = torch.stack([b[0] for b in batch])
        sat   = torch.stack([b[1] for b in batch])
        tok = tokenizer([b[2] for b in batch], padding=True, truncation=True,
                        max_length=77, return_tensors="pt")
        return drone, sat, tok["input_ids"], tok["attention_mask"]
    return collate


# ---------- model ----------------------------------------------------------

def build_model(device, rank, alpha, dropout):
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPModel, CLIPTokenizer
    clip = CLIPModel.from_pretrained(BACKBONE)
    tokenizer = CLIPTokenizer.from_pretrained(BACKBONE)
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    peft_model = get_peft_model(clip, cfg)
    peft_model.to(device)
    peft_model.print_trainable_parameters()
    # base_model.model is the CLIPModel with LoRA injected in place; call its
    # feature heads directly (PeftModel has no get_image_features wrapper).
    return peft_model, peft_model.base_model.model, tokenizer


def clip_features(clip, drone, sat, input_ids, attn):
    d = F.normalize(clip.get_image_features(pixel_values=drone), dim=-1)
    s = F.normalize(clip.get_image_features(pixel_values=sat),   dim=-1)
    t = F.normalize(clip.get_text_features(input_ids=input_ids,
                                           attention_mask=attn), dim=-1)
    return d, s, t


def info_nce(a, b, logit_scale):
    """Symmetric CLIP-style contrastive loss between aligned rows of a and b."""
    logits = logit_scale * a @ b.t()
    target = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) +
                  F.cross_entropy(logits.t(), target))


# ---------- train loop -----------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device} | backbone {BACKBONE}")

    index = build_pair_index(args.flights, args.caption_dir, args.pairs_dir, args.limit)
    if not index:
        sys.exit("No training triples found. Run caption_crops.py first.")
    print(f"  Total triples: {len(index)}")

    peft_model, clip, tokenizer = build_model(
        device, args.rank, args.alpha, args.dropout)
    ds = PairDataset(index, tokenizer, train=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=make_collate(tokenizer), pin_memory=True)

    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(loader) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    logit_scale = clip.logit_scale.exp().clamp(max=100.0).detach()

    peft_model.train()
    for ep in range(args.epochs):
        pbar = tqdm(loader, desc=f"  epoch {ep+1}/{args.epochs}", unit="batch")
        for drone, sat, ids, attn in pbar:
            drone, sat = drone.to(device), sat.to(device)
            ids, attn = ids.to(device), attn.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                d, s, t = clip_features(clip, drone, sat, ids, attn)
                l_dt = info_nce(d, t, logit_scale)
                l_st = info_nce(s, t, logit_scale)
                l_ds = info_nce(d, s, logit_scale)
                loss = l_dt + l_st + l_ds
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            pbar.set_postfix(dt=f"{l_dt.item():.3f}", st=f"{l_st.item():.3f}",
                             ds=f"{l_ds.item():.3f}", refresh=False)

    os.makedirs(args.out_dir, exist_ok=True)
    peft_model.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "train_meta.json"), "w") as f:
        json.dump({"backbone": BACKBONE, "rank": args.rank, "alpha": args.alpha,
                   "epochs": args.epochs, "n_triples": len(index),
                   "flights": args.flights}, f, indent=2)
    print(f"  Saved LoRA adapter -> {args.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", nargs="+", default=TRAIN_FLIGHTS)
    ap.add_argument("--caption-dir", default=CAPTION_DIR)
    ap.add_argument("--pairs-dir", default=PAIRS_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows per flight (smoke test).")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
