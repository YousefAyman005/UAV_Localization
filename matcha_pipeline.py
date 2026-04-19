import argparse
import contextlib
import os
import cv2
import numpy as np
import pandas as pd
import torch

# MATCHA hardcodes many CUDA calls. Shim them to no-ops on CPU-only machines.
if not torch.cuda.is_available():
    torch.cuda.synchronize      = lambda *a, **k: None
    torch.cuda.empty_cache      = lambda *a, **k: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.max_memory_allocated    = lambda *a, **k: 0
    torch.Tensor.cuda = lambda self, *a, **k: self                   # tensor.cuda() -> self
    torch.nn.Module.cuda = lambda self, *a, **k: self                # module.cuda() -> self

from visloc_utils import (
    MIN_INL, SZ_W, SZ_H, RANSAC_THRESH, SAT_TIF, SAT_CSV, DRONE_CSV, DRONE_DIR,
    load_satellite, run_pipeline, save_dense_viz,
)
from matcha.feature.matcha_feature import MatchaFeature
from matcha.matcher.base_matcher import BaseMatcher
from matcha.utils.device import to_numpy

OUT_CSV = "visloc_matcha_results.csv"
VIZ_DIR = "visloc_matcha_visualizations"


def bgr_to_tensor(bgr, img_w, img_h, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rs  = cv2.resize(rgb, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(rs / 255.).float().permute(2, 0, 1).unsqueeze(0).to(device)


def cuda_cleanup(device):
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def amp_context(device, enabled):
    if enabled and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def oom_hint(img_w, img_h):
    return (f"MATCHA ran out of CUDA memory at {img_w}x{img_h}. "
            "Restart the Kaggle session to clear old models, then retry with "
            "--img-w 512 --img-h 352 --amp. Raise the size only if it fits.")


def extract_matcha_features(bgr, matcher, img_w, img_h, device, use_amp):
    t = None
    try:
        t = bgr_to_tensor(bgr, img_w, img_h, device)
        with torch.inference_mode(), amp_context(device, use_amp):
            kpts, desc = matcher.model.detect_and_describe(img=t)
    except torch.cuda.OutOfMemoryError as exc:
        cuda_cleanup(device)
        raise RuntimeError(oom_hint(img_w, img_h)) from exc
    finally:
        if t is not None:
            del t
        cuda_cleanup(device)
    return kpts, desc


def match_matcha_features(kpts0, desc0, kpts1, desc1, img_w, img_h, conf_thresh, ransac_thresh, device):
    try:
        with torch.inference_mode():
            matches = matcher_matches(desc0, desc1)
    except torch.cuda.OutOfMemoryError as exc:
        cuda_cleanup(device)
        raise RuntimeError(oom_hint(img_w, img_h)) from exc

    kp0 = np.asarray(to_numpy(kpts0[0])).reshape(-1, 2)
    kp1 = np.asarray(to_numpy(kpts1[0])).reshape(-1, 2)
    m   = np.asarray(to_numpy(matches[0])).reshape(-1).astype(np.int64)
    scr = None

    valid = m >= 0
    mid0, mid1 = np.where(valid)[0], m[valid]
    conf = scr[mid0] if scr is not None else np.ones(mid0.shape, dtype=np.float32)

    scale = np.array([SZ_W / img_w, SZ_H / img_h], dtype=np.float32)
    kp0_f = (kp0[mid0] * scale).astype(np.float32)
    kp1_f = (kp1[mid1] * scale).astype(np.float32)

    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=int(valid.sum()), good=0, inliers=0,
             H=None, _kp0=kp0_f, _kp1=kp1_f, _conf=conf, _mask=None)
    mask = conf >= conf_thresh
    r["good"], r["_mask"] = int(mask.sum()), mask
    if r["good"] < 4:
        cuda_cleanup(device)
        return r
    H, mh = cv2.findHomography(kp0_f[mask].reshape(-1, 1, 2),
                               kp1_f[mask].reshape(-1, 1, 2),
                               cv2.USAC_MAGSAC, ransac_thresh,
                               maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    cuda_cleanup(device)
    return r


def matcher_matches(desc0, desc1):
    return BaseMatcher.nearest_neighbor_matching(x0=desc0, x1=desc1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",           type=int,   default=400)
    ap.add_argument("--dist",            type=float, default=25.0)
    ap.add_argument("--conf",            type=float, default=0.0)
    ap.add_argument("--weights",         type=str,   default="weights/matcha_pretrained.pth")
    ap.add_argument("--img-w",           type=int,   default=512, help="must be divisible by 32")
    ap.add_argument("--img-h",           type=int,   default=352, help="must be divisible by 32")
    ap.add_argument("--keypoint-method", choices=["disk"], default="disk")
    ap.add_argument("--ransac-thresh",   type=float, default=None)
    ap.add_argument("--min-inl",         type=int,   default=None)
    ap.add_argument("--amp",             action="store_true",
                    help="use CUDA fp16 autocast during MATCHA feature extraction")
    ap.add_argument("--clahe",           action="store_true")
    ap.add_argument("--visualize",       action="store_true")
    args = ap.parse_args()

    ransac_t = args.ransac_thresh if args.ransac_thresh is not None else RANSAC_THRESH
    min_inl  = args.min_inl       if args.min_inl       is not None else MIN_INL

    if args.img_w % 32 or args.img_h % 32:
        raise ValueError(f"--img-w/--img-h must be divisible by 32 (got {args.img_w}x{args.img_h})")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"MATCHA weights not found at {args.weights}. "
            "Download matcha_pretrained.pth from https://github.com/nv-dvl/matcha "
            "and place it in ./weights/."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Loading MATCHA ({args.keypoint_method}, {args.img_w}x{args.img_h}) ... ",
          end="", flush=True)
    model = MatchaFeature(config={"keypoint_method": args.keypoint_method,
                                   "image_size": (args.img_w, args.img_h)})
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)
    matcher = BaseMatcher(model, device)
    print("done")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None
    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    print(f"  Method: MATCHA ({args.keypoint_method}) | Size: {args.img_w}x{args.img_h} | "
          f"AMP: {args.amp} | CLAHE: {args.clahe} | Conf: {args.conf} | RANSAC: {ransac_t}px | "
          f"MinInl: {min_inl} | Dist: {args.dist}m | {len(df)} images\n")

    def match_factory(drone):
        kpd, descd = extract_matcha_features(drone, matcher, args.img_w, args.img_h, device, args.amp)
        def match_fn(p):
            kps, descs = extract_matcha_features(p, matcher, args.img_w, args.img_h, device, args.amp)
            return match_matcha_features(kpd, descd, kps, descs,
                                         args.img_w, args.img_h, args.conf, ransac_t, device)
        return match_fn

    run_pipeline(sat, geo, df, match_factory, OUT_CSV, args.dist,
                 min_inl=min_inl, clahe=clahe, drone_dir=DRONE_DIR,
                 viz_fn=save_dense_viz if args.visualize else None,
                 viz_dir=VIZ_DIR if args.visualize else None)


if __name__ == "__main__":
    main()
