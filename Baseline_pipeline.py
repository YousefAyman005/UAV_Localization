import argparse
import cv2
import numpy as np
import pandas as pd
from visloc_utils import (
    RANSAC_THRESH, TOP_MATCHES, SAT_TIF, SAT_CSV, DRONE_CSV,
    load_satellite, run_pipeline, draw_and_save,
)

OUT_CSV = "visloc_sift_results.csv"
VIZ_DIR = "visloc_visualizations"
LOWE = 0.75
FLANN_TREES, FLANN_CHECKS = 5, 50


def run_match(sg, dg, detector, method, clahe=None, rootsift=False):
    if clahe is not None:
        sg = clahe.apply(sg)
    kps, ds = detector.detectAndCompute(sg, None)
    kpd, dd = detector.detectAndCompute(dg, None)
    if rootsift and ds is not None and dd is not None:
        for d in (ds, dd):
            d /= d.sum(axis=1, keepdims=True) + 1e-7
            np.sqrt(d, out=d)
    r = dict(sat_kp=len(kps), drone_kp=len(kpd), raw=0, good=0, inliers=0, H=None,
             _kps=kps, _kpd=kpd, _matches=[], _sg=sg)
    if ds is None or dd is None or len(kps) < 4 or len(kpd) < 4:
        return r
    matcher = (cv2.FlannBasedMatcher({"algorithm": 1, "trees": FLANN_TREES}, {"checks": FLANN_CHECKS})
               if method == "sift" else cv2.BFMatcher(cv2.NORM_HAMMING))
    matches = matcher.knnMatch(dd, ds, k=2)
    good = [m for m, n in matches if m.distance < LOWE * n.distance]
    r["raw"], r["good"], r["_matches"] = len(matches), len(good), good
    if len(good) >= 4:
        src = np.float32([kpd[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
        if H is not None and mask is not None:
            r["inliers"], r["H"] = int(mask.sum()), H
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",     type=int,   default=400)
    ap.add_argument("--dist",      type=float, default=25.0)
    ap.add_argument("--method",    choices=["sift", "orb", "brisk"], default="sift")
    ap.add_argument("--clahe",     action="store_true")
    ap.add_argument("--rootsift",  action="store_true")
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()

    if args.rootsift and args.method != "sift":
        print(f"  WARNING: --rootsift ignored with --method {args.method}")

    detector = {"sift":  lambda: cv2.SIFT_create(),
                "orb":   lambda: cv2.ORB_create(5000),
                "brisk": lambda: cv2.BRISK_create()}[args.method]()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    flags = " | ".join(f for f, on in [("CLAHE", args.clahe),
                                        ("RootSIFT", args.rootsift and args.method == "sift")] if on) or "none"
    print(f"  Method: {args.method.upper()} | Preprocessing: {flags} | Dist: {args.dist}m | {len(df)} images\n")

    def match_factory(drone):
        dg = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
        if clahe is not None:
            dg = clahe.apply(dg)
        return lambda p: run_match(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), dg, detector,
                                   args.method, clahe=clahe, rootsift=args.rootsift)

    def viz_fn(drone, patch, best, filename, viz_dir):
        if not best["_matches"]:
            return
        dg = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
        if clahe is not None:
            dg = clahe.apply(dg)
        top = sorted(best["_matches"], key=lambda m: m.distance)[:TOP_MATCHES]
        draw_and_save(dg, best["_kpd"], best["_sg"], best["_kps"], top, filename, viz_dir)

    run_pipeline(sat, geo, df, match_factory, OUT_CSV, args.dist,
                 viz_fn=viz_fn if args.visualize else None,
                 viz_dir=VIZ_DIR if args.visualize else None)


if __name__ == "__main__":
    main()
