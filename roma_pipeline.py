import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from visloc_utils import (
    MIN_INL, RANSAC_THRESH, SAT_TIF, SAT_CSV, DRONE_CSV,
    load_satellite, run_pipeline, save_dense_viz,
)
from romatch import roma_outdoor, roma_indoor

OUT_CSV = "visloc_roma_results.csv"
VIZ_DIR = "visloc_roma_visualizations"


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def match_roma(drone_bgr, sat_bgr, matcher, conf_thresh, ransac_thresh, device, num_samples):
    H_img, W_img = drone_bgr.shape[:2]
    with torch.inference_mode():
        warp, cert  = matcher.match(bgr_to_pil(drone_bgr), bgr_to_pil(sat_bgr), device=device)
        matches, c  = matcher.sample(warp, cert, num=num_samples)
        kp_a, kp_b  = matcher.to_pixel_coordinates(matches, H_img, W_img, H_img, W_img)
    kp0  = kp_a.cpu().numpy().astype(np.float32)
    kp1  = kp_b.cpu().numpy().astype(np.float32)
    c_np = c.cpu().numpy()
    r = dict(sat_kp=len(kp0), drone_kp=len(kp0), raw=len(kp0), good=0, inliers=0,
             H=None, _kp0=kp0, _kp1=kp1, _conf=c_np, _mask=None)
    mask = c_np >= conf_thresh
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",         type=int,   default=400)
    ap.add_argument("--dist",          type=float, default=25.0)
    ap.add_argument("--conf",          type=float, default=0.0)
    ap.add_argument("--pretrained",    choices=["outdoor", "indoor"], default="outdoor")
    ap.add_argument("--num-matches",   type=int,   default=5000)
    ap.add_argument("--ransac-thresh", type=float, default=None)
    ap.add_argument("--min-inl",       type=int,   default=None)
    ap.add_argument("--clahe",         action="store_true")
    ap.add_argument("--visualize",     action="store_true")
    args = ap.parse_args()

    ransac_t = args.ransac_thresh if args.ransac_thresh is not None else RANSAC_THRESH
    min_inl  = args.min_inl       if args.min_inl       is not None else MIN_INL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("highest")
    print(f"  Device: {device}")
    print(f"  Loading RoMa ({args.pretrained}) ... ", end="", flush=True)
    kw = {} if device == "cuda" else {"amp_dtype": torch.float32}
    matcher = (roma_outdoor(device=device, **kw) if args.pretrained == "outdoor"
               else roma_indoor(device=device, **kw))
    print("done")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None
    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    print(f"  Method: RoMa ({args.pretrained}) | CLAHE: {args.clahe} | Conf: {args.conf} | "
          f"NumMatches: {args.num_matches} | RANSAC: {ransac_t}px | MinInl: {min_inl} | "
          f"Dist: {args.dist}m | {len(df)} images\n")

    def match_factory(drone):
        return lambda p: match_roma(drone, p, matcher, args.conf, ransac_t, device, args.num_matches)

    run_pipeline(sat, geo, df, match_factory, OUT_CSV, args.dist,
                 min_inl=min_inl, clahe=clahe,
                 viz_fn=save_dense_viz if args.visualize else None,
                 viz_dir=VIZ_DIR if args.visualize else None)


if __name__ == "__main__":
    main()
