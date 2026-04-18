import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from visloc_utils import (
    RANSAC_THRESH, SAT_TIF, SAT_CSV, DRONE_CSV,
    load_satellite, run_pipeline, save_dense_viz,
)
from kornia.feature import LoFTR

OUT_CSV = "visloc_loftr_results.csv"
VIZ_DIR = "visloc_loftr_visualizations"


def img_to_tensor(bgr, device):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(gray).float().div(255.).unsqueeze(0).unsqueeze(0).to(device)


def match_loftr(drone_t, sat_t, matcher, conf_thresh):
    with torch.inference_mode():
        out = matcher({"image0": drone_t, "image1": sat_t})
    kp0  = out["keypoints0"].cpu().numpy()
    kp1  = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=len(kp0), good=0, inliers=0,
             H=None, _kp0=kp0, _kp1=kp1, _conf=conf, _mask=None)
    mask = conf >= conf_thresh
    r["good"], r["_mask"] = int(mask.sum()), mask
    if r["good"] < 4:
        return r
    H, mh = cv2.findHomography(kp0[mask].reshape(-1, 1, 2).astype(np.float32),
                               kp1[mask].reshape(-1, 1, 2).astype(np.float32),
                               cv2.RANSAC, RANSAC_THRESH)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",      type=int,   default=400)
    ap.add_argument("--dist",       type=float, default=25.0)
    ap.add_argument("--conf",       type=float, default=0.0)
    ap.add_argument("--pretrained", choices=["outdoor", "indoor"], default="outdoor")
    ap.add_argument("--visualize",  action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Loading LoFTR ({args.pretrained}) ... ", end="", flush=True)
    matcher = LoFTR(pretrained=args.pretrained).eval().to(device)
    print("done")

    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    print(f"  Method: LoFTR ({args.pretrained}) | Conf: {args.conf} | Dist: {args.dist}m | {len(df)} images\n")

    def match_factory(drone):
        drone_t = img_to_tensor(drone, device)
        return lambda p: match_loftr(drone_t, img_to_tensor(p, device), matcher, args.conf)

    run_pipeline(sat, geo, df, match_factory, OUT_CSV, args.dist,
                 viz_fn=save_dense_viz if args.visualize else None,
                 viz_dir=VIZ_DIR if args.visualize else None)


if __name__ == "__main__":
    main()
