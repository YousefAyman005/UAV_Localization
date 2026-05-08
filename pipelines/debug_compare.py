"""4-panel diagnostic: drone vs current-pipeline crop vs metric-isotropic crop
vs metric-isotropic + heading-rotated crop.

Usage:
    .venv/bin/python3 pipelines/debug_compare.py --flight 03 --start 1 --count 20

Output: <repo>/debug_compare/<flight>/<filename>_compare.jpg
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visloc_utils import (
    CROP_H,
    CROP_W,
    SZ_H,
    SZ_W,
    _tile_for_gps,
    altitude_scales,
    crop_sat as legacy_crop_sat,
    load_flight,
    metric_crop,
)


def _annotate(img, label):
    out = img.copy() if img is not None else np.zeros((SZ_H, SZ_W, 3), np.uint8)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", required=True)
    ap.add_argument("--start", type=int, default=1, help="1-based row index in CSV")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--step", type=int, default=1, help="row stride between samples")
    ap.add_argument("--yaw-sign", type=float, default=1.0,
                    help="multiply Phi1 by this before passing to metric_crop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(repo_root, "debug_compare", args.flight)
    os.makedirs(out_dir, exist_ok=True)

    tiles, drone_dir, drone_csv, _ = load_flight(args.flight)
    df = pd.read_csv(drone_csv)
    sl = slice(args.start - 1, args.start - 1 + args.count * args.step, args.step)
    df = df.iloc[sl]

    print(f"flight={args.flight}  rows {args.start}..{args.start + args.count * args.step - 1}"
          f"  step={args.step}  yaw_sign={args.yaw_sign}  out={out_dir}")

    for _, row in df.iterrows():
        f = row["filename"]
        lat = float(row["lat"]); lon = float(row["lon"])
        height = float(row["height"]); yaw = float(row["Phi1"]) * args.yaw_sign

        drone = cv2.imread(os.path.join(drone_dir, f))
        if drone is None:
            print(f"  skip (no image): {f}")
            continue
        drone_r = cv2.resize(drone, (SZ_W, SZ_H))

        sat, geo, cx, cy = _tile_for_gps(tiles, lat, lon)

        s = altitude_scales(height, geo)[0]
        crop_w, crop_h = max(SZ_W, int(CROP_W * s)), max(SZ_H, int(CROP_H * s))
        legacy = legacy_crop_sat(sat, cx, cy, geo, crop_w, crop_h)

        metric_north, _ = metric_crop(sat, geo, cx, cy, height, yaw_deg=0.0)
        metric_rot, _ = metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw)

        top = np.hstack([
            _annotate(drone_r, "1. drone (resized 1024x680)"),
            _annotate(legacy,  f"2. legacy crop (scale={s:.2f}, {crop_w}x{crop_h})"),
        ])
        bot = np.hstack([
            _annotate(metric_north, "3. metric crop (north-up, isotropic m/px)"),
            _annotate(metric_rot,   f"4. metric crop + yaw={yaw:+.1f}°"),
        ])
        grid = np.vstack([top, bot])

        out_path = os.path.join(out_dir, f"{os.path.splitext(f)[0]}_compare.jpg")
        cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {f}  yaw={yaw:+.2f}°  alt={height:.0f}m  saved")


if __name__ == "__main__":
    main()
