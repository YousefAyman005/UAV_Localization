import argparse
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from visloc_utils import (
    MIN_INL, RANSAC_THRESH, SAT_TIF, SAT_CSV, DRONE_CSV, DRONE_DIR,
    load_satellite, run_pipeline, save_dense_viz,
)

OUT_CSV = "visloc_jamma_results.csv"
VIZ_DIR = "visloc_jamma_visualizations"

JAMMA_URL = "https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt"

# Verbatim from upstream demo/utlis.py — do not add or remove keys.
JAMMA_CFG = {
    "coarse": {"d_model": 256},
    "fine": {
        "d_model": 64,
        "dsmax_temperature": 0.1,
        "thr": 0.1,
        "inference": True,
    },
    "match_coarse": {
        "thr": 0.2,
        "use_sm": True,
        "border_rm": 2,
        "dsmax_temperature": 0.1,
        "inference": True,
    },
    "fine_window_size": 5,
    "resolution": [8, 2],
}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def jamma_preprocess(bgr, resize, df, device):
    """BGR ndarray -> (ImageNet-normalised RGB tensor [1,3,S,S], coarse mask [1,S/8,S/8],
    (sx, sy)) where S = max(h_new, w_new) — square pad matches upstream training."""
    h0, w0 = bgr.shape[:2]
    scale = resize / max(h0, w0)
    # Floor to df multiple — matches upstream get_divisible_wh (int(x // df * df)).
    w_new = int(w0 * scale) // df * df
    h_new = int(h0 * scale) // df * df
    rgb = cv2.cvtColor(cv2.resize(bgr, (w_new, h_new)), cv2.COLOR_BGR2RGB)
    # Square pad — matches upstream pad_bottom_right(pad_to=max(h_new, w_new)).
    pad_to = max(h_new, w_new)
    padded = np.zeros((pad_to, pad_to, 3), dtype=np.uint8)
    padded[:h_new, :w_new] = rgb
    t = torch.from_numpy(padded).float().div(255.).permute(2, 0, 1).unsqueeze(0)
    t = ((t - _IMAGENET_MEAN) / _IMAGENET_STD).to(device)
    # Mask marks the valid (non-padded) region at coarse (stride-8) resolution.
    m = torch.zeros((1, pad_to // 8, pad_to // 8), dtype=torch.bool, device=device)
    m[0, :h_new // 8, :w_new // 8] = True
    return t, m, (w0 / w_new, h0 / h_new)


def match_jamma(t0, m0, s0, t1, m1, s1, backbone, matcher, conf_thresh, ransac_thresh):
    data = {"imagec_0": t0, "imagec_1": t1, "mask0": m0, "mask1": m1}
    with torch.inference_mode():
        backbone(data)
        matcher(data)

    kp0  = data["mkpts0_f"].cpu().numpy().astype(np.float32)
    kp1  = data["mkpts1_f"].cpu().numpy().astype(np.float32)
    conf = data["mconf_f"].cpu().numpy()
    kp0[:, 0] *= s0[0]; kp0[:, 1] *= s0[1]
    kp1[:, 0] *= s1[0]; kp1[:, 1] *= s1[1]

    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=len(kp0), good=0, inliers=0,
             H=None, _kp0=kp0, _kp1=kp1, _conf=conf, _mask=None)
    mask = conf >= conf_thresh
    r["good"], r["_mask"] = int(mask.sum()), mask
    if r["good"] < 4:
        return r
    H, mh = cv2.findHomography(kp0[mask].reshape(-1, 1, 2),
                               kp1[mask].reshape(-1, 1, 2),
                               cv2.USAC_MAGSAC, ransac_thresh,
                               maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


def _ts(label):
    print(f"  [{time.strftime('%H:%M:%S')}] {label}", flush=True)


def _load_jamma(device):
    _ts("importing JamMa modules (triggers mamba-ssm CUDA build if first run)...")
    from src.jamma.jamma import JamMa as JamMaMatcher
    from src.jamma.backbone import CovNextV2_nano
    _ts("imports done")

    _ts("building model skeleton...")
    class _Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = CovNextV2_nano()
            self.matcher  = JamMaMatcher(JAMMA_CFG, profiler=None)

    wrapper = _Wrapper()
    _ts("skeleton built")

    _ts(f"downloading/loading weights from {JAMMA_URL} ...")
    state_dict = torch.hub.load_state_dict_from_url(
        JAMMA_URL, map_location="cpu", file_name="jamma.ckpt")["state_dict"]
    _ts("weights loaded, applying state_dict...")
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=True)
    _ts(f"state_dict applied — moving to {device}...")

    backbone = wrapper.backbone.eval().to(device)
    matcher  = wrapper.matcher.eval().to(device)
    _ts("model on device, ready")
    return backbone, matcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",         type=int,   default=400)
    ap.add_argument("--dist",          type=float, default=25.0)
    ap.add_argument("--conf",          type=float, default=0.2)
    ap.add_argument("--resize",        type=int,   default=832)
    ap.add_argument("--df",            type=int,   default=16)
    ap.add_argument("--ransac-thresh", type=float, default=None)
    ap.add_argument("--min-inl",       type=int,   default=None)
    ap.add_argument("--clahe",         action="store_true")
    ap.add_argument("--visualize",     action="store_true")
    args = ap.parse_args()

    if args.df <= 0 or args.df % 8 != 0:
        raise ValueError(f"--df must be a positive multiple of 8 (got {args.df}); "
                         "JamMa's coarse stride is 8.")
    if args.resize <= 0:
        raise ValueError(f"--resize must be positive (got {args.resize}).")

    ransac_t = args.ransac_thresh if args.ransac_thresh is not None else RANSAC_THRESH
    min_inl  = args.min_inl       if args.min_inl       is not None else MIN_INL

    if not torch.cuda.is_available():
        raise RuntimeError(
            "JamMa requires CUDA (mamba-ssm has no CPU/MPS kernels). "
            "Run this pipeline on Kaggle/Colab with a GPU enabled."
        )

    device = torch.device("cuda")
    print(f"  Device: {device}")
    print(f"  Loading JamMa ({args.resize}px) ... ", end="", flush=True)
    backbone, matcher = _load_jamma(device)
    print("done")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None
    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    print(f"  Method: JamMa | Resize: {args.resize} | DF: {args.df} | "
          f"CLAHE: {args.clahe} | Conf: {args.conf} | RANSAC: {ransac_t}px | "
          f"MinInl: {min_inl} | Dist: {args.dist}m | {len(df)} images\n")

    def match_factory(drone):
        t0, m0, s0 = jamma_preprocess(drone, args.resize, args.df, device)
        return lambda p: match_jamma(
            t0, m0, s0,
            *jamma_preprocess(p, args.resize, args.df, device),
            backbone, matcher, args.conf, ransac_t,
        )

    run_pipeline(sat, geo, df, match_factory, OUT_CSV, args.dist,
                 min_inl=min_inl, clahe=clahe, drone_dir=DRONE_DIR,
                 viz_fn=save_dense_viz if args.visualize else None,
                 viz_dir=VIZ_DIR if args.visualize else None)


if __name__ == "__main__":
    main()
