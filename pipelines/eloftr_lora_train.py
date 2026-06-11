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
from helpers.utils import (FLIGHTS_AVAILABLE, MIN_INL, SEARCH_FACTOR, SZ_H, SZ_W,  # noqa: E402
                           _make_clahe, fit_similarity, get_flight_paths)

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

    w_pt0_i = _warp_homo(grid0_i, data["H_0to1"])
    w_pt0_c = w_pt0_i / scale
    w_pt1_c = _warp_homo(grid1_i, data["H_1to0"]) / scale
    w_pt0_round = w_pt0_c.round().long()
    w_pt1_round = w_pt1_c.round().long()
    nearest_index1 = w_pt0_round[..., 0] + w_pt0_round[..., 1] * w1
    nearest_index0 = w_pt1_round[..., 0] + w_pt1_round[..., 1] * w0
    nearest_index1[_ob(w_pt0_round, rw1c, rh1c)] = 0
    nearest_index0[_ob(w_pt1_round, rw0c, rh0c)] = 0
    if "S1_inv" in data:
        # Crop-side aug: replicate-border bands carry no true content. A point
        # whose pre-aug position S1^-1·y falls outside the real frame cannot be
        # ground truth on either leg (no-op when S1 = I).
        rw1i, rh1i = data["real_w"], data["real_h"]
        nearest_index1[_ob(_warp_homo(w_pt0_i, data["S1_inv"]), rw1i, rh1i)] = 0
        nearest_index0[_ob(_warp_homo(grid1_i, data["S1_inv"]), rw1i, rh1i)] = 0

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
                 "spv_w_pt0_i": w_pt0_i,
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
        ok = (wp[..., 0] >= 0) & (wp[..., 0] < rw1) & (wp[..., 1] >= 0) & (wp[..., 1] < rh1)
        if "S1_inv" in data:  # exclude replicate-border content (see spvs_coarse_homo)
            orig = _warp_homo(wp.reshape(1, -1, 2), data["S1_inv"][[b]]).reshape(cnt, WW, 2)
            ok &= ~_ob(orig, rw1, rh1)
        correct[sel] = ok
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


def _photometric(img):
    """Mild photometric jitter (uint8 BGR, post-CLAHE): gamma, contrast/
    brightness, occasional blur/noise. Applied independently per image so the
    drone<->satellite appearance gap itself is varied each epoch."""
    g = np.random.uniform(0.7, 1.4)
    lut = np.clip((np.linspace(0.0, 1.0, 256) ** g) * 255.0, 0, 255).astype(np.uint8)
    img = cv2.LUT(img, lut)
    img = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.85, 1.15),
                              beta=np.random.uniform(-20.0, 20.0))
    if np.random.rand() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), np.random.uniform(0.3, 1.0))
    if np.random.rand() < 0.3:
        noise = np.random.normal(0.0, np.random.uniform(2.0, 6.0), img.shape)
        img = np.clip(img.astype(np.float32) + noise.astype(np.float32),
                      0, 255).astype(np.uint8)
    return img


def _rand_similarity(max_rot_deg, max_scale, max_trans_px):
    """Random similarity about the crop centre, in cv2.warpAffine's FORWARD
    convention (no WARP_INVERSE_MAP): a feature at src px c lands at S·c."""
    ang = np.random.uniform(-max_rot_deg, max_rot_deg)
    sc = np.random.uniform(1.0 - max_scale, 1.0 + max_scale)
    M = cv2.getRotationMatrix2D((SZ_W / 2.0, SZ_H / 2.0), ang, sc)
    M[0, 2] += np.random.uniform(-max_trans_px, max_trans_px)
    M[1, 2] += np.random.uniform(-max_trans_px, max_trans_px)
    S = np.eye(3, dtype=np.float64)
    S[:2] = M
    return S


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


def build_index(pairs_dir, flights, gt_mode, limit, min_corr=0, max_teacher_geo_m=0.0):
    """[{drone_path, png_path, H, name, m_per_px, true_gps_px, flight}, ...].

    Quality filter (train indexes): drop pairs with fewer than `min_corr`
    teacher inliers, or whose teacher-H disagrees with the geo-H at the drone
    centre by more than `max_teacher_geo_m` meters (alias-suspect labels on
    repetitive terrain; <=0 disables). Pairs generated at a different
    SEARCH_FACTOR than the current one are a hard error (stale cache)."""
    index, no_sf = [], 0
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
        n = n_corr_drop = n_disagree_drop = 0
        for meta in metas:
            npz_path = os.path.join(pairs_dir, meta["npz"])
            png_path = os.path.join(pairs_dir, meta["png"])
            if not (os.path.isfile(npz_path) and os.path.isfile(png_path)):
                continue
            sf = meta.get("search_factor")
            if sf is None:
                no_sf += 1
            elif abs(float(sf) - SEARCH_FACTOR) > 1e-6:
                sys.exit(f"Pairs in {pairs_dir} were generated at SEARCH_FACTOR={sf}, "
                         f"current is {SEARCH_FACTOR} — regenerate them.")
            m_per_px = float(meta["m_per_px"])
            if min_corr > 0 and meta.get("teacher") and meta.get("n_corr", 0) < min_corr:
                n_corr_drop += 1; continue
            with np.load(npz_path, allow_pickle=True) as d:
                H = _fit_H(d, gt_mode)
                if max_teacher_geo_m > 0 and bool(d["has_teacher"]):
                    cd_px = meta.get("center_disagree_px")
                    if cd_px is None:  # old cache: recompute from the stored Hs
                        c = np.array([[[SZ_W / 2.0, SZ_H / 2.0]]], dtype=np.float64)
                        pt_t = cv2.perspectiveTransform(c, d["H_teacher"]).reshape(2)
                        pt_g = cv2.perspectiveTransform(c, d["H_geo"]).reshape(2)
                        cd_px = float(np.hypot(*(pt_t - pt_g)))
                    if cd_px * m_per_px > max_teacher_geo_m:
                        n_disagree_drop += 1; continue
            index.append(dict(drone_path=os.path.join(drone_dir, meta["filename"]),
                              png_path=png_path, H=H.astype(np.float64),
                              name=f"{flight}/{meta['filename']}",
                              m_per_px=m_per_px,
                              true_gps_px=tuple(meta["true_gps_px"]),
                              flight=flight))
            n += 1
        drops = (f", dropped {n_corr_drop} low-corr + {n_disagree_drop} "
                 f"teacher-vs-geo>{max_teacher_geo_m:g}m"
                 if (n_corr_drop or n_disagree_drop) else "")
        print(f"  flight {flight}: {n} pairs ({gt_mode}{drops})")
    if no_sf:
        print(f"  WARN: {no_sf} pairs lack search_factor metadata (old cache?) — "
              f"cannot verify crop scale matches SEARCH_FACTOR={SEARCH_FACTOR}.")
    return index


class PairDataset(Dataset):
    """Grayscale drone + cached (already-CLAHE'd) crop + homography.

    With aug=True (train band only): independent photometric jitter on both
    images plus a random similarity S on the crop with the label updated as
    H' = S @ H — label-exact geometric variation mimicking the yaw/K/prior
    noise the benchmark exhibits."""

    def __init__(self, index, long_side, clahe, aug=False,
                 aug_rot=8.0, aug_scale=0.1, aug_trans=48.0):
        self.index = index
        self.long_side = long_side
        self.clahe_fn = _make_clahe(clahe)
        self.aug = aug
        self.aug_rot, self.aug_scale, self.aug_trans = aug_rot, aug_scale, aug_trans

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e = self.index[i]
        drone = cv2.imread(e["drone_path"])
        crop = cv2.imread(e["png_path"])
        if drone is None or crop is None:
            return None
        drone = cv2.resize(drone, (SZ_W, SZ_H), interpolation=cv2.INTER_AREA)
        if self.clahe_fn:
            drone = self.clahe_fn(drone)           # crop PNG is already CLAHE'd
        H = e["H"].astype(np.float64)
        S = np.eye(3, dtype=np.float64)
        if self.aug:
            drone = _photometric(drone)
            crop = _photometric(crop)
            S = _rand_similarity(self.aug_rot, self.aug_scale, self.aug_trans)
            crop = cv2.warpAffine(crop, S[:2].astype(np.float32), (SZ_W, SZ_H),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            H = S @ H                              # feature c -> S·c in the aug crop
        g0, s, rh, rw = _to_input(drone, self.long_side)
        g1, _, _, _ = _to_input(crop, self.long_side)
        return dict(image0=torch.from_numpy(g0).float().div(255.)[None],
                    image1=torch.from_numpy(g1).float().div(255.)[None],
                    H=torch.from_numpy(_scale_H(H, s)),
                    S1=torch.from_numpy(_scale_H(S, s)),
                    rh=rh, rw=rw, scale=s, name=e["name"],
                    m_per_px=e["m_per_px"],
                    true_gps_px=torch.tensor(e["true_gps_px"], dtype=torch.float64))


def _worker_init(_wid):
    # helpers.utils seeds np.random globally at import; forked DataLoader
    # workers would otherwise share identical augmentation streams.
    # torch.initial_seed() is per-worker and per-epoch, yet reproducible
    # under torch.manual_seed(0).
    np.random.seed(torch.initial_seed() % 2 ** 32)


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    H = torch.stack([b["H"] for b in batch])       # float64 for stable inverses
    S1 = torch.stack([b["S1"] for b in batch])
    return {"image0": torch.stack([b["image0"] for b in batch]),
            "image1": torch.stack([b["image1"] for b in batch]),
            "H_0to1": H.float(), "H_1to0": torch.inverse(H).float(),
            "S1_inv": torch.inverse(S1).float(),
            "real_h": batch[0]["rh"], "real_w": batch[0]["rw"],
            "scale_in": batch[0]["scale"],
            "m_per_px": [b["m_per_px"] for b in batch],
            "true_gps_px": torch.stack([b["true_gps_px"] for b in batch]),
            "pair_names": [b["name"] for b in batch]}


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
    for k in ("image0", "image1", "H_0to1", "H_1to0", "S1_inv"):
        batch[k] = batch[k].to(device)
    return batch


@torch.no_grad()
def validate(peft_model, inner, loader, device):
    """Benchmark-aligned validation on the held-out val band.

    Per pair: predicted matches -> helpers.utils.fit_similarity (the SAME 4-DOF
    estimator every eval pipeline uses) -> project the drone centre -> error in
    METERS vs the true-GPS pixel. Gated like the benchmark: a failed fit or
    < MIN_INL inliers counts as a miss (err = inf). The stored m_per_px is the
    CROP-frame GSD, so err_px * m_per_px is meters directly. Also returns the
    legacy median match-reprojection error vs the geo-H for continuity."""
    peft_model.eval()
    errs_m, reproj, nmatches = [], [], []
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
        if len(kp0):
            proj = _warp_homo(kp0[None].float(), batch["H_0to1"][:1])[0]
            reproj.append(float((proj - kp1.float()).norm(dim=1).median()))
        else:
            reproj.append(float(max(rh, rw)))               # penalize no-match
        H_pred, ninl = fit_similarity(kp0.cpu().numpy(), kp1.cpu().numpy())
        if H_pred is None or ninl < MIN_INL:
            errs_m.append(float("inf"))
            continue
        s = float(batch["scale_in"])
        p = H_pred @ np.array([rw / 2.0, rh / 2.0, 1.0])
        p = p[:2] / max(float(p[2]), 1e-12)
        gt = batch["true_gps_px"][0].numpy() * s            # native crop px -> input px
        err_px = float(np.hypot(p[0] - gt[0], p[1] - gt[1]))
        errs_m.append(err_px * batch["m_per_px"][0] / s)    # m_per_px is per NATIVE px
    peft_model.train()
    errs_m = np.asarray(errs_m if errs_m else [float("inf")], dtype=float)
    return dict(med_err_m=float(np.median(errs_m)),
                acc25=float(np.mean(errs_m <= 25.0)),
                fail_frac=float(np.mean(np.isinf(errs_m))),
                reproj_px=float(np.median(reproj)) if reproj else float("inf"),
                mean_matches=float(np.mean(nmatches)) if nmatches else 0.0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    cfg = build_cfg()
    print(f"  Device: {device} | flights {' '.join(flights)} | gt-mode {args.gt_mode} | "
          f"long-side {args.long_side or 'native'} | bs {args.batch_size} | amp {args.amp}")

    index = build_index(args.pairs_dir, flights, args.gt_mode, args.limit,
                        min_corr=args.min_corr,
                        max_teacher_geo_m=args.max_teacher_geo_m)
    if not index:
        sys.exit(f"No training pairs in {args.pairs_dir}. Run gen_eloftr_pairs.py first.")
    print(f"  Total training pairs: {len(index)} | aug {'off' if args.no_aug else 'on'}")

    peft_model, inner = build_model(args.weights, device, args.rank, args.alpha,
                                    args.dropout, args.targets, args.grad_ckpt, args.dump_modules)
    loss_fn = LoFTRLoss(lower_config(cfg)).to(device)
    loss_fn.train()

    ds = PairDataset(index, args.long_side, not args.no_clahe, aug=not args.no_aug,
                     aug_rot=args.aug_rot, aug_scale=args.aug_scale,
                     aug_trans=args.aug_trans)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, pin_memory=True,
                        worker_init_fn=_worker_init)

    val_loader = None
    if os.path.isdir(args.val_pairs_dir):
        val_index = build_index(args.val_pairs_dir, flights, "geo", args.val_limit)
        if val_index:
            val_loader = DataLoader(PairDataset(val_index, args.long_side, not args.no_clahe),
                                    batch_size=1, shuffle=False, num_workers=args.workers,
                                    collate_fn=collate, pin_memory=True)
            print(f"  Validation pairs: {len(val_index)} (val band)")
    if val_loader is None:
        print(f"  No val set at {args.val_pairs_dir} (gen with --split val --no-teacher); "
              f"skipping validation/early-stop.")

    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.1)
    total_steps = max(1, len(loader) * args.epochs)
    warmup = max(0, min(args.warmup_steps, total_steps - 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup))
    sched = (torch.optim.lr_scheduler.SequentialLR(
                 opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05,
                                                         total_iters=warmup), cosine],
                 milestones=[warmup])
             if warmup else cosine)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    best_key, best, history = (float("inf"), 0.0), None, []
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
            vm = validate(peft_model, inner, val_loader, device)
            vm["epoch"] = ep + 1
            vm["train_loss"] = run / max(1, len(loader))
            history.append(vm)
            print(f"  [val] epoch {ep+1}: median err {vm['med_err_m']:.1f}m | "
                  f"A@25 {100*vm['acc25']:.1f}% | fit-fail {100*vm['fail_frac']:.0f}% | "
                  f"reproj {vm['reproj_px']:.1f}px | matches {vm['mean_matches']:.0f}")
            key = (vm["med_err_m"], -vm["acc25"])
            if key < best_key:                   # strict < keeps the earlier epoch on ties
                best_key, best = key, dict(vm)
                _save(peft_model, args, cfg, len(index), flights, tag="best",
                      best=best, history=history)
                print(f"    new best -> saved to {args.out_dir}")

    if val_loader is None:                       # no early-stop signal: keep the final epoch
        _save(peft_model, args, cfg, len(index), flights, tag="final")
    else:                                        # best already in out_dir; stash the last epoch
        _save(peft_model, args, cfg, len(index), flights, tag="last",
              out_dir=args.out_dir + "_last", best=best, history=history)
    msg = (f"median err {best['med_err_m']:.1f}m @ epoch {best['epoch']}"
           if best else "n/a")
    print(f"  Done. Best val: {msg} | adapter -> {args.out_dir}")


def _save(peft_model, args, cfg, n_pairs, flights, tag, out_dir=None,
          best=None, history=None):
    out_dir = out_dir or args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_meta.json"), "w") as fh:
        json.dump({"base_ckpt": os.path.basename(args.weights), "rank": args.rank,
                   "alpha": args.alpha, "dropout": args.dropout, "targets": args.targets,
                   "epochs": args.epochs, "lr": args.lr, "warmup_steps": args.warmup_steps,
                   "n_pairs": n_pairs,
                   "flights": flights, "gt_mode": args.gt_mode, "long_side": args.long_side,
                   "amp": args.amp, "local_weight": cfg.LOFTR.LOSS.LOCAL_WEIGHT,
                   "aug": (None if args.no_aug else
                           dict(rot=args.aug_rot, scale=args.aug_scale,
                                trans=args.aug_trans)),
                   "min_corr": args.min_corr,
                   "max_teacher_geo_m": args.max_teacher_geo_m,
                   "search_factor": float(SEARCH_FACTOR),
                   "pairs_dir": args.pairs_dir, "val_pairs_dir": args.val_pairs_dir,
                   "val_metric": "fit_similarity centre error vs true-GPS px (m), "
                                 f"gated at MIN_INL={MIN_INL}",
                   "best": best, "val_history": history,
                   "saved_at": tag}, fh, indent=2)


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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=100,
                    help="Linear LR warmup steps before the cosine decay (0 disables).")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lora-mlp", action="store_true", help="Also LoRA the coarse MLP linears.")
    ap.add_argument("--no-aug", action="store_true",
                    help="Disable train-time augmentation (ablation).")
    ap.add_argument("--aug-rot", type=float, default=8.0,
                    help="Max |rotation| in degrees for the crop-side similarity jitter.")
    ap.add_argument("--aug-scale", type=float, default=0.1,
                    help="Max |scale-1| for the crop-side similarity jitter.")
    ap.add_argument("--aug-trans", type=float, default=48.0,
                    help="Max |translation| in px for the crop-side similarity jitter.")
    ap.add_argument("--min-corr", type=int, default=16,
                    help="Drop teacher pairs with fewer inlier correspondences.")
    ap.add_argument("--max-teacher-geo-m", type=float, default=64.0,
                    help="Drop teacher pairs whose teacher-H vs geo-H drone-centre "
                         "disagreement exceeds this many meters (alias-suspect "
                         "labels on repetitive terrain). <=0 disables.")
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
