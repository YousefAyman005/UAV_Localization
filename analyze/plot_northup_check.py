"""Diagnostic: which rotation sign turns a drone image north-up?

For sampled frames of one flight, shows [north-up GT sat crop | drone rotated
by +corrected_yaw | drone rotated by -corrected_yaw] so the correct convention
for CLIP-line query alignment can be verified visually (a sign error is a
~180 deg mistake on cardinal legs). Rotation = center square crop, then
cv2.getRotationMatrix2D(center, angle, 1.0) in the same canvas — the
transform the CLIP north-up retrain will use.

Usage:
    python analyze/plot_northup_check.py --flight 08 --indices 100 664 1100
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from helpers import utils as U  # noqa: E402


def square_rotate(bgr, angle_deg):
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    sq = bgr[y0:y0 + side, x0:x0 + side]
    M = cv2.getRotationMatrix2D((side / 2, side / 2), angle_deg, 1.0)
    return cv2.warpAffine(sq, M, (side, side))


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="08")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="drone CSV rows (default: 3 spread over the flight)")
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args()

    tiles, drone_dir, drone_csv, _ = U.load_flight(args.flight)
    df = pd.read_csv(drone_csv)
    idxs = args.indices or [len(df) // 4, len(df) // 2, (3 * len(df)) // 4]

    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idxs):
        row = df.iloc[i]
        yaw = U.corrected_yaw(args.flight, float(row["Phi1"]))
        sat, geo, cx, cy, _ = U.tile_for_gps(tiles, row["lat"], row["lon"])
        patch, _ = U.metric_crop(sat, geo, cx, cy, row["height"], yaw_deg=0.0,
                                 flight=args.flight)
        drone = cv2.imread(os.path.join(drone_dir, row["filename"]))
        drone = cv2.resize(drone, (U.SZ_W, U.SZ_H), interpolation=cv2.INTER_AREA)
        for c, (img, title) in enumerate([
                (patch, f"GT sat crop, north-up (row {i})"),
                (square_rotate(drone, yaw), f"drone rot +yaw ({yaw:.0f})"),
                (square_rotate(drone, -yaw), f"drone rot -yaw ({-yaw:.0f})")]):
            axes[r, c].imshow(rgb(img))
            axes[r, c].set_title(title, fontsize=8)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"northup_check_{args.flight}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main()
