#!/usr/bin/env python
"""LoRA-finetune Efficient-LoFTR on teacher-distilled drone<->satellite pairs.

We adapt ELoFTR to the UAV->satellite appearance gap by distilling the
``roma_extre`` teacher: ``gen_eloftr_pairs.py`` produced, per TRAIN-band drone
image, a satellite crop + filtered teacher correspondences + a geo-homography.
Here we LoRA-finetune ELoFTR's coarse transformer to reproduce the teacher's
per-pair geometry, supervising with the SAME coarse-focal + fine-L2 loss the
upstream repo uses (``src/losses/loftr_loss.py``) — only the ground-truth
construction is swapped: instead of warping a grid through depth+pose
(``spvs_coarse``/``spvs_fine``'s ``warp_kpts``), we warp it through a per-pair
homography H (drone_px -> crop_px), forward H for 0->1 and inverse H for 1->0.

Why this is sound:
  - ELoFTR ships training code; we reuse its loss and mirror its supervision math.
  - The full model already uses skip_softmax=False / fp16matmul=False, so the
    stock checkpoint produces a real dual-softmax ``conf_matrix`` and is trainable
    as-is. We do NOT reparameterize during training (RepVGG must keep its
    multi-branch form); reparameterization is applied only at eval, after merging
    LoRA into the (disjoint) coarse-transformer Linears.
  - Padding: inputs are padded to a multiple of 32 like eval; we keep the model
    mask-free (identical to eval) and prevent spurious pad-region GT by bounding
    grid/warp validity to the real (unpadded) dims inside our supervision.

Outputs the PEFT adapter + train_meta.json to weights/eloftr_lora/. Load it back
at eval via ``eloftr_pipeline.py --lora-ckpt weights/eloftr_lora``.

Smoke test (needs /opt/EfficientLoFTR + torch + peft):
    python pipelines/eloftr_lora_train.py --flights 03 --limit 50 --epochs 1 \
           --batch-size 1 --dump-modules
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from kornia.utils import create_meshgrid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, "/opt/EfficientLoFTR")
from src.loftr import LoFTR, full_default_cfg  # noqa: E402  (do NOT import reparameter)
from src.losses.loftr_loss import LoFTRLoss  # noqa: E402
from src.config.default import get_cfg_defaults  # noqa: E402
from src.utils.misc import lower_config  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.utils import FLIGHTS_AVAILABLE, SZ_H, SZ_W, get_flight_paths, _make_clahe  # noqa: E402

torch.manual_seed(0)

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "merge"]
MLP_TARGETS = ["mlp.0", "mlp.2"]
OUT_DIR = "weights/eloftr_lora"
PAIRS_DIR = "cache/eloftr_pairs"
VAL_PAIRS_DIR = "cache/eloftr_pairs_val"


# ── config (uppercase yacs for supervision; lower_config for the loss) ─────────

def build_cfg():
    """Top-level yacs cfg whose LOFTR/LOSS fields the supervision + loss read.
    Reconcile the few fields that differ from the full inference model."""
    cfg = get_cfg_defaults()
    cfg.LOFTR.RESOLUTION = (8, 1)
    cfg.LOFTR.FINE_WINDOW_SIZE = 8
    cfg.LOFTR.ALIGN_CORNER = False
    cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = True
    cfg.LOFTR.MATCH_FINE.SPARSE_SPVS = True
    cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0   # match full_default_cfg
    cfg.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = False
    cfg.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = False
    cfg.LOFTR.LOSS.LOCAL_WEIGHT = 0.25                      # ELoFTR optimized weight
    return cfg


# ── homography-driven supervision (mirrors src/loftr/utils/supervision.py) ─────

def _warp_homo(pts, H):
    """pts [N, P, 2], H [N, 3, 3] (drone->crop) -> warped [N, P, 2]."""
    ones = torch.ones_like(pts[..., :1])
    ph = torch.cat([pts, ones], dim=-1)               # [N, P, 3]
    w = torch.einsum("nij,npj->npi", H, ph)           # [N, P, 3]
    return w[..., :2] / w[..., 2:3].clamp(min=1e-8)


def _ob(pt, w, h):
    return (pt[..., 0] < 0) | (pt[..., 0] >= w) | (pt[..., 1] < 0) | (pt[..., 1] >= h)


@torch.no_grad()
def spvs_coarse_homo(data, cfg):
    """Build conf_matrix_gt + spv_* from per-pair homographies (replaces warp_kpts).
    Validity is bounded by the REAL (unpadded) coarse dims so the replicate-pad
    band produces no ground-truth."""
    device = data["image0"].device
    N, _, H0, W0 = data["image0"].shape
    _, _, H1, W1 = data["image1"].shape
    scale = cfg.LOFTR.RESOLUTION[0]
    h0, w0, h1, w1 = H0 // scale, W0 // scale, H1 // scale, W1 // scale
    rh0c, rw0c = data["real_h"] // scale, data["real_w"] // scale
    rh1c, rw1c = rh0c, rw0c

    grid0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)
    grid1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
    grid0_i, grid1_i = scale * grid0_c, scale * grid1_c

    w_pt0_c = _warp_homo(grid0_i, data["H_0to1"]) / scale
    w_pt1_c = _warp_homo(grid1_i, data["H_1to0"]) / scale
    w_pt0_round = w_pt0_c.round().long()
    w_pt1_round = w_pt1_c.round().long()
    nearest_index1 = w_pt0_round[..., 0] + w_pt0_round[..., 1] * w1
    nearest_index0 = w_pt1_round[..., 0] + w_pt1_round[..., 1] * w0
    nearest_index1[_ob(w_pt0_round, rw1c, rh1c)] = 0
    nearest_index0[_ob(w_pt1_round, rw0c, rh0c)] = 0

    loop_back = torch.gather(nearest_index0, 1, nearest_index1.clamp(0, h1 * w1 - 1))
    correct = loop_back == torch.arange(h0 * w0, device=device)[None].repeat(N, 1)
    correct[:, 0] = False
    src_valid = (grid0_c[..., 0] < rw0c) & (grid0_c[..., 1] < rh0c)
    correct = correct & src_valid

    conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)
    b_ids, i_ids = torch.where(correct)
    j_ids = nearest_index1[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data["conf_matrix_gt"] = conf_matrix_gt
    if len(b_ids) == 0:  # avoid empty-gt crashes downstream
        b_ids = i_ids = j_ids = torch.zeros(1, dtype=torch.long, device=device)
    data.update({"spv_b_ids": b_ids, "spv_i_ids": i_ids, "spv_j_ids": j_ids,
                 "spv_w_pt0_i": _warp_homo(grid0_i, data["H_0to1"]),
                 "spv_pt1_i": grid1_i})


@torch.no_grad()
def spvs_fine_homo(data, cfg):
    """Build expec_f_gt + conf_matrix_f_gt + m_ids_f/... from homographies
    (replaces warp_kpts in spvs_fine). Runs AFTER the forward (uses data['b_ids'])."""
    pt1_i = data["spv_pt1_i"]
    W = cfg.LOFTR.FINE_WINDOW_SIZE
    WW = W * W
    device = data["image0"].device
    N = data["image0"].shape[0]
    hf0, wf0 = data["hw0_f"][0], data["hw0_f"][1]
    rh1, rw1 = data["real_h"], data["real_w"]
    b_ids, i_ids, j_ids = data["b_ids"], data["i_ids"], data["j_ids"]
    m = b_ids.shape[0]
    if m == 0:  # never happens in training (coarse pads with gt), but stay safe
        data.update({"conf_matrix_f_gt": torch.zeros(m, WW, WW, device=device),
                     "expec_f": torch.zeros(1, 2, device=device),
                     "expec_f_gt": torch.zeros(1, 2, device=device)})
        return

    grid_pt0_f = create_meshgrid(hf0, wf0, False, device) - W // 2 + 0.5
    grid_pt0_f = rearrange(grid_pt0_f, "n h w c -> n c h w")
    grid_unfold = F.unfold(grid_pt0_f, kernel_size=(W, W), stride=W, padding=0)
    grid_unfold = rearrange(grid_unfold, "n (c ww) l -> n l ww c", ww=WW)
    grid_unfold = repeat(grid_unfold[0], "l ww c -> N l ww c", N=N)
    grid_unfold = grid_unfold[b_ids, i_ids]                      # [m, WW, 2]

    correct = torch.zeros(m, WW, device=device, dtype=torch.bool)
    w_pt0_i = torch.zeros(m, WW, 2, device=device)
    for b in range(N):
        sel = b_ids == b
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        wp = _warp_homo(grid_unfold[sel].reshape(1, -1, 2), data["H_0to1"][[b]])
        wp = wp.reshape(cnt, WW, 2)
        correct[sel] = (wp[..., 0] >= 0) & (wp[..., 0] < rw1) & (wp[..., 1] >= 0) & (wp[..., 1] < rh1)
        w_pt0_i[sel] = wp

    delta_i = w_pt0_i - pt1_i[b_ids, j_ids][:, None, :]          # [m, WW, 2]
    delta_f = delta_i + W // 2 - 0.5                             # scalei1 == 1
    delta_round = delta_f.round()
    delta_round_l = delta_round.long()
    nearest1 = delta_round_l[..., 0] + delta_round_l[..., 1] * W
    ob = _ob(delta_round_l, W, W)
    nearest1[ob] = 0
    correct[ob] = 0

    m_ids, i_ids_f = torch.where(correct)
    j_ids_f = nearest1[m_ids, i_ids_f]
    expec_f_gt = delta_f - delta_round
    if m_ids.numel() == 0:
        data.update({"expec_f": torch.zeros(1, 2, device=device),
                     "expec_f_gt": torch.zeros(1, 2, device=device)})
    else:
        data.update({"expec_f_gt": expec_f_gt[m_ids, i_ids_f],
                     "m_ids_f": m_ids.long(), "i_ids_f": i_ids_f.long(),
                     "j_ids_f_di": (j_ids_f // W).long(), "j_ids_f_dj": (j_ids_f % W).long()})
    conf_f_gt = torch.zeros(m, WW, WW, device=device)
    conf_f_gt[m_ids, i_ids_f, j_ids_f] = 1
    data["conf_matrix_f_gt"] = conf_f_gt


# ── image / homography prep ────────────────────────────────────────────────────

def _pad32(gray):
    h, w = gray.shape
    nh, nw = -(-h // 32) * 32, -(-w // 32) * 32
    if (nh, nw) == (h, w):
        return gray
    return cv2.copyMakeBorder(gray, 0, nh - h, 0, nw - w, cv2.BORDER_REPLICATE)


def _to_input(bgr, long_side):
    """BGR (SZ_W x SZ_H) -> (gray tensor-ready [Hp, Wp], scale s, real (h, w))."""
    s = 1.0
    if long_side and long_side > 0:
        s = long_side / SZ_W
        bgr = cv2.resize(bgr, (int(round(SZ_W * s)), int(round(SZ_H * s))),
                         interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape
    return _pad32(gray), s, rh, rw


def _scale_H(H, s):
    if s == 1.0:
        return H
    S = np.diag([s, s, 1.0]).astype(np.float64)
    return (S @ H @ np.linalg.inv(S)).astype(np.float64)


def _fit_H(d, gt_mode):
    """Per-pair homography drone->crop in the native SZ_W x SZ_H frame."""
    H_geo = d["H_geo"].astype(np.float64)
    if gt_mode == "geo":
        return H_geo
    H_teacher = d["H_teacher"].astype(np.float64) if bool(d["has_teacher"]) else None
    if gt_mode == "teacher":
        return H_teacher if H_teacher is not None else H_geo
    xd, yd, xs, ys = d["xd"], d["yd"], d["xs"], d["ys"]   # gt_mode == "homo"
    if len(xd) >= 8:
        src = np.stack([xd, yd], 1).reshape(-1, 1, 2).astype(np.float32)
        dst = np.stack([xs, ys], 1).reshape(-1, 1, 2).astype(np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0,
                                  maxIters=5000, confidence=0.9999)
        if H is not None:
            return H.astype(np.float64)
    return H_teacher if H_teacher is not None else H_geo


def build_index(pairs_dir, flights, gt_mode, limit):
    """[(drone_path, crop_png, H_3x3, name), ...] from per-flight manifests."""
    index = []
    for flight in flights:
        fdir = os.path.join(pairs_dir, flight)
        manifest = os.path.join(fdir, "manifest.jsonl")
        if not os.path.isfile(manifest):
            print(f"  WARN: no manifest for flight {flight} in {pairs_dir}")
            continue
        _, drone_dir, _, _ = get_flight_paths(flight)
        with open(manifest) as fh:
            metas = [json.loads(line) for line in fh if line.strip()]
        if limit is not None:
            metas = metas[:limit]
        n = 0
        for meta in metas:
            npz_path = os.path.join(pairs_dir, meta["npz"])
            png_path = os.path.join(pairs_dir, meta["png"])
            if not (os.path.isfile(npz_path) and os.path.isfile(png_path)):
                continue
            with np.load(npz_path, allow_pickle=True) as d:
                H = _fit_H(d, gt_mode)
            index.append((os.path.join(drone_dir, meta["filename"]), png_path,
                          H.astype(np.float32), f"{flight}/{meta['filename']}"))
            n += 1
        print(f"  flight {flight}: {n} pairs ({gt_mode})")
    return index


class PairDataset(Dataset):
    """Returns grayscale drone + cached (already-CLAHE'd) crop + homography."""

    def __init__(self, index, long_side, clahe):
        self.index = index
        self.long_side = long_side
        self.clahe_fn = _make_clahe(clahe)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        drone_path, crop_path, H, name = self.index[i]
        drone = cv2.imread(drone_path)
        crop = cv2.imread(crop_path)
        if drone is None or crop is None:
            return None
        drone = cv2.resize(drone, (SZ_W, SZ_H), interpolation=cv2.INTER_AREA)
        if self.clahe_fn:
            drone = self.clahe_fn(drone)           # crop PNG is already CLAHE'd
        g0, s, rh, rw = _to_input(drone, self.long_side)
        g1, _, _, _ = _to_input(crop, self.long_side)
        Hs = _scale_H(H.astype(np.float64), s)
        return (torch.from_numpy(g0).float().div(255.)[None],
                torch.from_numpy(g1).float().div(255.)[None],
                torch.from_numpy(Hs).float(), rh, rw, name)


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    img0 = torch.stack([b[0] for b in batch])
    img1 = torch.stack([b[1] for b in batch])
    H = torch.stack([b[2] for b in batch])
    Hinv = torch.inverse(H)
    return {"image0": img0, "image1": img1, "H_0to1": H, "H_1to0": Hinv,
            "real_h": batch[0][3], "real_w": batch[0][4],
            "pair_names": [b[5] for b in batch]}


# ── model (LoRA) ───────────────────────────────────────────────────────────────

def build_model(ckpt_path, device, rank, alpha, dropout, targets, grad_ckpt, dump):
    from copy import deepcopy
    from peft import LoraConfig, get_peft_model

    matcher = LoFTR(config=deepcopy(full_default_cfg))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    matcher.load_state_dict(sd)                              # NO reparameter (training)

    if dump:
        names = [n for n, mod in matcher.named_modules()
                 if isinstance(mod, torch.nn.Linear) and n.startswith("loftr_coarse")]
        print("  loftr_coarse Linear modules:\n    " + "\n    ".join(names))

    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=targets)
    peft_model = get_peft_model(matcher, cfg)
    peft_model.print_trainable_parameters()
    inner = peft_model.base_model.model                     # LoRA injected in place
    if grad_ckpt:
        import torch.utils.checkpoint as cp

        def wrap(layer):
            orig = layer.forward

            def fwd(x, source, x_mask=None, source_mask=None, _o=orig):
                return cp.checkpoint(_o, x, source, x_mask, source_mask, use_reentrant=False)
            layer.forward = fwd
        for layer in inner.loftr_coarse.layers:
            wrap(layer)
    peft_model.to(device)
    return peft_model, inner


# ── train / val ────────────────────────────────────────────────────────────────

def _move(batch, device):
    for k in ("image0", "image1", "H_0to1", "H_1to0"):
        batch[k] = batch[k].to(device)
    return batch


@torch.no_grad()
def validate(peft_model, inner, loader, device):
    """Median reprojection error (px) of predicted matches vs the per-pair geo-H,
    plus mean #matches, on the held-out test band. Lower error = better."""
    peft_model.eval()
    errs, nmatches = [], []
    for batch in loader:
        if batch is None:
            continue
        batch = _move(batch, device)
        inner(batch)
        kp0, kp1 = batch["mkpts0_f"], batch["mkpts1_f"]
        rh, rw = batch["real_h"], batch["real_w"]
        inb = (kp0[:, 0] < rw) & (kp0[:, 1] < rh) & (kp1[:, 0] < rw) & (kp1[:, 1] < rh)
        kp0, kp1 = kp0[inb], kp1[inb]
        nmatches.append(int(len(kp0)))
        if len(kp0) == 0:
            errs.append(float(max(rh, rw)))                 # penalize no-match
            continue
        proj = _warp_homo(kp0[None].float(), batch["H_0to1"][:1])[0]
        errs.append(float((proj - kp1.float()).norm(dim=1).median()))
    peft_model.train()
    return (float(np.median(errs)) if errs else float("inf"),
            float(np.mean(nmatches)) if nmatches else 0.0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    cfg = build_cfg()
    print(f"  Device: {device} | flights {' '.join(flights)} | gt-mode {args.gt_mode} | "
          f"long-side {args.long_side or 'native'} | bs {args.batch_size} | amp {args.amp}")

    index = build_index(args.pairs_dir, flights, args.gt_mode, args.limit)
    if not index:
        sys.exit(f"No training pairs in {args.pairs_dir}. Run gen_eloftr_pairs.py first.")
    print(f"  Total training pairs: {len(index)}")

    peft_model, inner = build_model(args.weights, device, args.rank, args.alpha,
                                    args.dropout, args.targets, args.grad_ckpt, args.dump_modules)
    loss_fn = LoFTRLoss(lower_config(cfg)).to(device)
    loss_fn.train()

    ds = PairDataset(index, args.long_side, not args.no_clahe)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, pin_memory=True)

    val_loader = None
    if os.path.isdir(args.val_pairs_dir):
        val_index = build_index(args.val_pairs_dir, flights, "geo", args.val_limit)
        if val_index:
            val_loader = DataLoader(PairDataset(val_index, args.long_side, not args.no_clahe),
                                    batch_size=1, shuffle=False, num_workers=args.workers,
                                    collate_fn=collate, pin_memory=True)
            print(f"  Validation pairs: {len(val_index)} (test band)")
    if val_loader is None:
        print(f"  No val set at {args.val_pairs_dir} (gen with --split test --no-teacher); "
              f"skipping validation/early-stop.")

    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.1)
    total_steps = max(1, len(loader) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    peft_model.train()
    for ep in range(args.epochs):
        pbar = tqdm(loader, desc=f"  epoch {ep+1}/{args.epochs}", unit="batch")
        run = 0.0
        for step, batch in enumerate(pbar):
            if batch is None:
                continue
            batch = _move(batch, device)
            opt.zero_grad(set_to_none=True)
            spvs_coarse_homo(batch, cfg)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                inner(batch)
                spvs_fine_homo(batch, cfg)
                loss_fn(batch)
            loss = batch["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            old = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old:
                sched.step()
            run += float(loss.detach())
            ls = batch["loss_scalars"]
            pbar.set_postfix(loss=f"{run/(step+1):.3f}",
                             c=f"{float(ls['loss_c']):.3f}", f=f"{float(ls['loss_f']):.3f}",
                             l=f"{float(ls['loss_l']):.3f}", refresh=False)

        if val_loader is not None:
            reproj, nm = validate(peft_model, inner, val_loader, device)
            print(f"  [val] epoch {ep+1}: median reproj {reproj:.2f}px | mean matches {nm:.0f}")
            if reproj < best_val:
                best_val = reproj
                _save(peft_model, args, cfg, len(index), flights, best_val, tag="best")
                print(f"    new best -> saved to {args.out_dir}")

    if val_loader is None:                       # no early-stop signal: keep the final epoch
        _save(peft_model, args, cfg, len(index), flights, None, tag="final")
    else:                                        # best already in out_dir; stash the last epoch
        _save(peft_model, args, cfg, len(index), flights, best_val, tag="last",
              out_dir=args.out_dir + "_last")
    print(f"  Done. Best val reproj: {best_val if best_val < float('inf') else 'n/a'} | "
          f"adapter -> {args.out_dir}")


def _save(peft_model, args, cfg, n_pairs, flights, best_val, tag, out_dir=None):
    out_dir = out_dir or args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_meta.json"), "w") as fh:
        json.dump({"base_ckpt": os.path.basename(args.weights), "rank": args.rank,
                   "alpha": args.alpha, "dropout": args.dropout, "targets": args.targets,
                   "epochs": args.epochs, "lr": args.lr, "n_pairs": n_pairs,
                   "flights": flights, "gt_mode": args.gt_mode, "long_side": args.long_side,
                   "amp": args.amp, "local_weight": cfg.LOFTR.LOSS.LOCAL_WEIGHT,
                   "best_val_reproj_px": best_val, "saved_at": tag}, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flights", nargs="+", default=["all"])
    ap.add_argument("--weights", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "eloftr_outdoor.ckpt"))
    ap.add_argument("--pairs-dir", default=PAIRS_DIR)
    ap.add_argument("--val-pairs-dir", default=VAL_PAIRS_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--gt-mode", choices=["homo", "teacher", "geo"], default="homo",
                    help="homo: per-pair homography fit to teacher corr (default); "
                         "teacher: stored RoMa similarity; geo: GPS-prior homography (ablation).")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lora-mlp", action="store_true", help="Also LoRA the coarse MLP linears.")
    ap.add_argument("--long-side", type=int, default=0,
                    help="Resize long side to this (0 = native 1024x680, like eval). "
                         "Lower (e.g. 832) cuts memory; correspondences/H are rescaled.")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true", help="Checkpoint coarse layers (VRAM cut).")
    ap.add_argument("--amp", action="store_true",
                    help="Mixed precision (upstream trains fp32 / MP=False; off by default).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Cap pairs/flight (smoke test).")
    ap.add_argument("--val-limit", type=int, default=200, help="Cap val pairs/flight.")
    ap.add_argument("--dump-modules", action="store_true",
                    help="Print coarse Linear module names (to lock LoRA target_modules).")
    args = ap.parse_args()
    args.targets = DEFAULT_TARGETS + (MLP_TARGETS if args.lora_mlp else [])
    train(args)


if __name__ == "__main__":
    main()
