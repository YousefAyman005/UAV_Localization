import argparse
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from visloc_utils import (
    MIN_INL, CROP_W, CROP_H, SZ_W, SZ_H, RANSAC_THRESH, TOP_MATCHES, JPEG_QUALITY,
    load_satellite, gps_to_px, crop_sat, pred_offset_m, print_summary, altitude_scales,
)

FLIGHT    = "03"
BASE      = f"UAV_Visloc_example/{FLIGHT}"
SAT_TIF   = f"{BASE}/satellite{FLIGHT}.tif"
DRONE_DIR = f"{BASE}/drone"
DRONE_CSV = f"{BASE}/{FLIGHT}.csv"
SAT_CSV   = "UAV_Visloc_example/satellite_ coordinates_range.csv"
OUT_CSV   = "visloc_sift_results.csv"
VIZ_DIR   = "visloc_visualizations"

LOWE         = 0.75   # Lowe's ratio test threshold
FLANN_TREES  = 5      # number of KD-trees for FLANN index
FLANN_CHECKS = 50     # number of leaf checks during FLANN search


def run_match(sg, dg, detector, method, clahe=None, rootsift=False):
    if clahe is not None:
        sg = clahe.apply(sg)
    kps, ds = detector.detectAndCompute(sg, None)
    kpd, dd = detector.detectAndCompute(dg, None)
    if rootsift and ds is not None and dd is not None:
        for d in (ds, dd):
            d /= d.sum(axis=1, keepdims=True) + 1e-7  # L1-normalize in place
            np.sqrt(d, out=d)                          # element-wise sqrt in place
    r = dict(sat_kp=len(kps), drone_kp=len(kpd), raw=0, good=0,
             inliers=0, H=None, _kps=kps, _kpd=kpd, _matches=[])
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
    ap.add_argument("--limit",    type=int,   default=400)
    ap.add_argument("--dist",     type=float, default=25.0,  help="Success radius in metres")
    ap.add_argument("--method",   choices=["sift", "orb", "brisk"], default="sift")
    ap.add_argument("--clahe",    action="store_true", help="CLAHE contrast enhancement")
    ap.add_argument("--rootsift", action="store_true", help="RootSIFT normalisation (SIFT only)")
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()

    if args.rootsift and args.method != "sift":
        print(f"  WARNING: --rootsift ignored with --method {args.method}")

    detectors = {
        "sift":  lambda: cv2.SIFT_create(),
        "orb":   lambda: cv2.ORB_create(5000),
        "brisk": lambda: cv2.BRISK_create(),
    }
    detector = detectors[args.method]()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    flags = " | ".join(f for f, on in [("CLAHE", args.clahe),
                                        ("RootSIFT", args.rootsift and args.method == "sift")] if on) or "none"
    print(f"  Method: {args.method.upper()} | Preprocessing: {flags} | Dist: {args.dist}m | {len(df)} images\n")

    if args.visualize:
        os.makedirs(VIZ_DIR, exist_ok=True)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        f, lat, lon = row["filename"], float(row["lat"]), float(row["lon"])
        drone = cv2.imread(os.path.join(DRONE_DIR, f))
        if drone is None:
            rows.append(dict(filename=f, skipped=True))
            continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        dg = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
        if clahe is not None:
            dg = clahe.apply(dg)
        cx, cy = gps_to_px(lat, lon, geo)

        best, best_crop, patch = None, None, None
        for s in altitude_scales(float(row["height"]), geo):
            crop_w = max(SZ_W, int(CROP_W * s))
            crop_h = max(SZ_H, int(CROP_H * s))
            p = crop_sat(sat, cx, cy, geo, crop_w, crop_h)
            if p is None:
                continue
            r = run_match(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), dg, detector,
                          args.method, clahe=clahe, rootsift=args.rootsift)
            if best is None or r["inliers"] > best["inliers"]:
                best, best_crop, patch = r, (crop_w, crop_h), p

        if best is None:
            rows.append(dict(filename=f, skipped=True))
            continue

        r = best
        off = pred_offset_m(r["H"], cx, cy, *best_crop, geo, lat, lon) if r["inliers"] >= MIN_INL else None
        off_m, plat, plon = off if off else (None, None, None)
        success = off_m is not None and off_m <= args.dist

        rows.append(dict(filename=f, lat=lat, lon=lon, height=float(row["height"]),
                         skipped=False, crop_w=best_crop[0], crop_h=best_crop[1],
                         sat_kp=r["sat_kp"], drone_kp=r["drone_kp"],
                         raw=r["raw"], good=r["good"], inliers=r["inliers"],
                         inlier_ratio=round(r["inliers"]/r["good"], 4) if r["good"] else 0,
                         pred_lat=round(plat, 7) if plat is not None else None,
                         pred_lon=round(plon, 7) if plon is not None else None,
                         offset_m=round(off_m, 2) if off_m is not None else None,
                         success=success))

        if args.visualize and r["_matches"]:
            top = sorted(r["_matches"], key=lambda m: m.distance)[:TOP_MATCHES]
            viz = cv2.drawMatches(dg, r["_kpd"],
                                  cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), r["_kps"],
                                  top, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            stem = os.path.splitext(f)[0]
            cv2.imwrite(os.path.join(VIZ_DIR, f"{stem}_matches.jpg"), viz,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    if out.empty or "skipped" not in out.columns:
        print("\n  No images processed."); return
    print_summary(out[~out["skipped"].fillna(False)], args.dist, OUT_CSV)


if __name__ == "__main__":
    main()
