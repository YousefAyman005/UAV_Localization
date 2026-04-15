import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from visloc_utils import (
    MIN_INL, CROP_W, CROP_H, SZ_W, SZ_H, RANSAC_THRESH, TOP_MATCHES, JPEG_QUALITY,
    load_satellite, gps_to_px, crop_sat, pred_offset_m, print_summary, altitude_scales,
)
from kornia.feature import LoFTR

FLIGHT    = "03"
BASE      = f"UAV_Visloc_example/{FLIGHT}"
SAT_TIF   = f"{BASE}/satellite{FLIGHT}.tif"
DRONE_DIR = f"{BASE}/drone"
DRONE_CSV = f"{BASE}/{FLIGHT}.csv"
SAT_CSV   = "UAV_Visloc_example/satellite_ coordinates_range.csv"
OUT_CSV   = "visloc_loftr_results.csv"
VIZ_DIR   = "visloc_loftr_visualizations"


def img_to_tensor(bgr, device):
    """Convert BGR image to grayscale tensor [1, 1, H, W] in [0, 1]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t = torch.from_numpy(gray).float().div(255.).unsqueeze(0).unsqueeze(0)
    return t.to(device)


def match_loftr(drone_t, sat_t, matcher, conf_thresh):
    """Run LoFTR on a drone/satellite tensor pair and estimate homography."""
    with torch.inference_mode():
        out = matcher({"image0": drone_t, "image1": sat_t})

    kp0 = out["keypoints0"].cpu().numpy()  # (N, 2) drone keypoints
    kp1 = out["keypoints1"].cpu().numpy()  # (N, 2) satellite keypoints
    conf = out["confidence"].cpu().numpy()  # (N,)

    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=len(kp0), good=0,
             inliers=0, H=None, _kp0=kp0, _kp1=kp1, _conf=conf, _mask=None)

    mask = conf >= conf_thresh
    r["good"] = int(mask.sum())

    if r["good"] < 4:
        return r

    src = kp0[mask].reshape(-1, 1, 2).astype(np.float32)
    dst = kp1[mask].reshape(-1, 1, 2).astype(np.float32)
    H, mask_h = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    if H is not None and mask_h is not None:
        r["inliers"], r["H"] = int(mask_h.sum()), H
    r["_mask"] = mask
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",      type=int,   default=400)
    ap.add_argument("--dist",       type=float, default=25.0,  help="Success radius in metres")
    ap.add_argument("--conf",       type=float, default=0.0,   help="LoFTR confidence threshold")
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
        cx, cy = gps_to_px(lat, lon, geo)

        drone_t = img_to_tensor(drone, device)

        best, best_crop, patch = None, None, None
        for s in altitude_scales(float(row["height"]), geo):
            crop_w = max(SZ_W, int(CROP_W * s))
            crop_h = max(SZ_H, int(CROP_H * s))
            p = crop_sat(sat, cx, cy, geo, crop_w, crop_h)
            if p is None:
                continue
            sat_t = img_to_tensor(p, device)
            r = match_loftr(drone_t, sat_t, matcher, args.conf)
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
                         pred_lat=round(plat, 7) if plat else None,
                         pred_lon=round(plon, 7) if plon else None,
                         offset_m=round(off_m, 2) if off_m else None,
                         success=success))

        if args.visualize and r["_mask"] is not None and r["good"] > 0:
            kp0, kp1, conf = r["_kp0"], r["_kp1"], r["_conf"]
            mask = r["_mask"]
            kpd_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0[mask]]
            kps_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1[mask]]
            matches_cv = [cv2.DMatch(i, i, 1.0 - c) for i, c in enumerate(conf[mask])]
            top = sorted(matches_cv, key=lambda m: m.distance)[:TOP_MATCHES]
            viz = cv2.drawMatches(drone, kpd_cv,
                                  patch, kps_cv,
                                  top, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            stem = os.path.splitext(f)[0]
            cv2.imwrite(os.path.join(VIZ_DIR, f"{stem}_matches.jpg"), viz,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    if out.empty or "skipped" not in out.columns:
        print("\n  No images processed."); return
    v = out[~out["skipped"].fillna(False)]
    print_summary(v, args.dist, OUT_CSV)


if __name__ == "__main__":
    main()
