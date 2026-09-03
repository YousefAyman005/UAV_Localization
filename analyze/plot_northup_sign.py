"""Decisive sign check for north_up_drone (CLIP-line query alignment).

For sampled frames, applies helpers.utils.north_up_drone(drone,
corrected_yaw) — exactly what clip_lora_train/clip_pipeline --north-up do —
and SIFT-matches the result against the NORTH-UP (yaw_deg=0) GT metric crop
through the shared fit_similarity. If the implementation is correct the
recovered residual rotation is ~0 deg; a sign error shows up as ~2*yaw.
The -yaw variant is reported as the control. Writes a small CSV + prints a
summary (run via slurm/run_plot_fig.sh; needs the dataset + cv2).

Usage:
    python analyze/plot_northup_sign.py --flight 08 --n 12
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from helpers import utils as U  # noqa: E402

LOWE, FLANN_TREES, FLANN_CHECKS = 0.75, 5, 50  # baseline_pipeline.py values


_CLAHE = U._make_clahe(True)  # the harness's default preprocessing


def sift_residual_angle(drone_bgr, patch_bgr):
    """SIFT + ratio test + fit_similarity; returns (angle_deg, inliers)."""
    sift = cv2.SIFT_create()
    g1 = cv2.cvtColor(_CLAHE(drone_bgr), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(_CLAHE(patch_bgr), cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": FLANN_TREES},
                                  {"checks": FLANN_CHECKS})
    good = [m for m, n in flann.knnMatch(d1, d2, k=2)
            if m.distance < LOWE * n.distance]
    if len(good) < 8:
        return None, len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = U.fit_similarity(src, dst)
    if H is None:
        return None, 0
    ang = math.degrees(math.atan2(H[1, 0], H[0, 0]))
    return ang, inl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="08")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    tiles, drone_dir, drone_csv, _ = U.load_flight(args.flight)
    df = pd.read_csv(drone_csv)
    idxs = np.linspace(0, len(df) - 1, args.n, dtype=int)

    def center_sq(img, frac=0.6):
        h, w = img.shape[:2]
        side = int(min(h, w) * frac)
        y0, x0 = (h - side) // 2, (w - side) // 2
        return img[y0:y0 + side, x0:x0 + side]

    def ncc(a, b):
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
        a -= a.mean(); b -= b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 0 else 0.0

    rows = []
    geom = {"+yaw": [], "-yaw": []}
    for i in idxs:
        row = df.iloc[i]
        yaw = U.corrected_yaw(args.flight, float(row["Phi1"]))
        sat, geo, cx, cy, ok = U.tile_for_gps(tiles, row["lat"], row["lon"])
        if not ok:
            continue
        patch, _ = U.metric_crop(sat, geo, cx, cy, row["height"], yaw_deg=0.0,
                                 flight=args.flight)
        drone = cv2.imread(os.path.join(drone_dir, row["filename"]))
        if patch is None or drone is None:
            continue
        # Pure-geometry test (immune to scene symmetry): the heading-up
        # metric crop, rotated by north_up_drone with each sign, vs the
        # north-up metric crop of the SAME spot. Correct sign -> NCC ~1.
        patch_h, _ = U.metric_crop(sat, geo, cx, cy, row["height"],
                                   yaw_deg=yaw, flight=args.flight)
        if patch_h is not None:
            ref = center_sq(U.north_up_drone(patch, 0.0))  # square-crop ref
            for sign, ang in (("+yaw", yaw), ("-yaw", -yaw)):
                test = center_sq(U.north_up_drone(patch_h, ang))
                geom[sign].append(ncc(test, ref))
        drone = cv2.resize(drone, (U.SZ_W, U.SZ_H), interpolation=cv2.INTER_AREA)
        for sign, img in (("+yaw", U.north_up_drone(drone, yaw)),
                          ("-yaw", U.north_up_drone(drone, -yaw))):
            ang, inl = sift_residual_angle(img, patch)
            rows.append({"row": int(i), "yaw": round(yaw, 1), "variant": sign,
                         "residual_deg": None if ang is None else round(ang, 1),
                         "inliers": inl})
            print(f"  row {i:4d} yaw {yaw:7.1f} {sign}: "
                  f"residual {ang if ang is None else round(ang, 1)} deg, "
                  f"{inl} inliers", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, f"northup_sign_{args.flight}.csv"),
               index=False)
    for v in ("+yaw", "-yaw"):
        sub = out[(out.variant == v) & out.residual_deg.notna()
                  & (out.inliers >= U.MIN_INL)]
        if len(sub):
            med = sub.residual_deg.abs().median()
            print(f"{v}: {len(sub)} fits | median |residual| {med:.1f} deg "
                  f"| median inliers {sub.inliers.median():.0f}")
        else:
            print(f"{v}: no accepted fits")
    for v in ("+yaw", "-yaw"):
        if geom[v]:
            print(f"GEOMETRY {v}: NCC mean {np.mean(geom[v]):.3f} "
                  f"min {np.min(geom[v]):.3f} (n={len(geom[v])})")


if __name__ == "__main__":
    main()
