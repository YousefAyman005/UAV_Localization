import argparse
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
    MIN_INL, SZ_W, SZ_H, RANSAC_THRESH, SAT_TIF, SAT_CSV, DRONE_CSV,
    load_satellite, run_pipeline, save_dense_viz,
)
from matcha.feature.matcha_feature import MatchaFeature
from matcha.matcher.base_matcher import BaseMatcher
from matcha.utils.device import to_numpy

OUT_CSV = "visloc_matcha_results.csv"
VIZ_DIR = "visloc_matcha_visualizations"


def bgr_to_tensor(bgr, img_size, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rs  = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(rs / 255.).float().permute(2, 0, 1).unsqueeze(0).to(device)


def match_matcha(drone_bgr, sat_bgr, matcher, img_size, device, conf_thresh, ransac_thresh):
    t0 = bgr_to_tensor(drone_bgr, img_size, device)
    t1 = bgr_to_tensor(sat_bgr,   img_size, device)
    with torch.inference_mode():
        out = matcher(data0={"image": t0}, data1={"image": t1})

    kp0 = np.asarray(to_numpy(out["keypoints0"])).reshape(-1, 2)
    kp1 = np.asarray(to_numpy(out["keypoints1"])).reshape(-1, 2)
    m   = np.asarray(to_numpy(out["matches"])).reshape(-1).astype(np.int64)
    scr_t = out.get("scores")
    scr = np.asarray(to_numpy(scr_t)).reshape(-1) if isinstance(scr_t, torch.Tensor) else None

    valid = m >= 0
    mid0, mid1 = np.where(valid)[0], m[valid]
    conf = scr[mid0] if scr is not None else np.ones(mid0.shape, dtype=np.float32)

    scale = np.array([SZ_W / img_size, SZ_H / img_size], dtype=np.float32)
    kp0_f = (kp0[mid0] * scale).astype(np.float32)
    kp1_f = (kp1[mid1] * scale).astype(np.float32)

    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=int(valid.sum()), good=0, inliers=0,
             H=None, _kp0=kp0_f, _kp1=kp1_f, _conf=conf, _mask=None)
    mask = conf >= conf_thresh
    r["good"], r["_mask"] = int(mask.sum()), mask
    if r["good"] < 4:
        return r
    H, mh = cv2.findHomography(kp0_f[mask].reshape(-1, 1, 2),
                               kp1_f[mask].reshape(-1, 1, 2),
                               cv2.USAC_MAGSAC, ransac_thresh,
                               maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",           type=int,   default=400)
    ap.add_argument("--dist",            type=float, default=25.0)
    ap.add_argument("--conf",            type=float, default=0.0)
    ap.add_argument("--weights",         type=str,   default="weights/matcha_pretrained.pth")
    ap.add_argument("--img-size",        type=int,   default=512)
    ap.add_argument("--keypoint-method", choices=["disk"], default="disk")
    ap.add_argument("--ransac-thresh",   type=float, default=None)
    ap.add_argument("--min-inl",         type=int,   default=None)
    ap.add_argument("--clahe",           action="store_true")
    ap.add_argument("--visualize",       action="store_true")
    args = ap.parse_args()

    ransac_t = args.ransac_thresh if args.ransac_thresh is not None else RANSAC_THRESH
    min_inl  = args.min_inl       if args.min_inl       is not None else MIN_INL

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"MATCHA weights not found at {args.weights}. "
            "Download matcha_pretrained.pth from https://github.com/nv-dvl/matcha "
            "and place it in ./weights/."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Loading MATCHA ({args.keypoint_method}, {args.img_size}x{args.img_size}) ... ",
          end="", flush=True)
    model = MatchaFeature(config={"keypoint_method": args.keypoint_method,
                                   "image_size": (args.img_size, args.img_size)})
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)
    matcher = BaseMatcher(model, device)
    print("done")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None
    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    print(f"  Method: MATCHA ({args.keypoint_method}) | Size: {args.img_size} | "
          f"CLAHE: {args.clahe} | Conf: {args.conf} | RANSAC: {ransac_t}px | "
          f"MinInl: {min_inl} | Dist: {args.dist}m | {len(df)} images\n")

    def match_factory(drone):
        return lambda p: match_matcha(drone, p, matcher, args.img_size, device, args.conf, ransac_t)

    run_pipeline(sat, geo, df, match_factory, OUT_CSV, args.dist,
                 min_inl=min_inl, clahe=clahe,
                 viz_fn=save_dense_viz if args.visualize else None,
                 viz_dir=VIZ_DIR if args.visualize else None)


if __name__ == "__main__":
    main()
