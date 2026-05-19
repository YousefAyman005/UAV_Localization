"""4-panel diagnostic: drone vs legacy-axis-aligned crop vs metric-isotropic crop
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.utils import (
    SZ_H, SZ_W,
    load_flight, metric_crop, metric_m_per_px, tile_for_gps,
)


# ── Legacy axis-aligned crop (only kept here for visual comparison) ─────────

CROP_W = 2048
CROP_H = CROP_W * SZ_H // SZ_W   # 1360
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]


def legacy_crop_sat(sat, cx, cy, g, crop_w, crop_h):
    """Old axis-aligned, fixed-aspect crop (pre-metric_crop)."""
    if not (0 <= cx < g["w"] and 0 <= cy < g["h"]):
        return None
    cxi = min(max(int(round(cx)), 0), g["w"] - 1)
    cyi = min(max(int(round(cy)), 0), g["h"] - 1)
    sx, sy = crop_w / SZ_W, crop_h / SZ_H
    x0 = max(0, cxi - crop_w // 2 - 1)
    y0 = max(0, cyi - crop_h // 2 - 1)
    x1 = min(g["w"], cxi + crop_w // 2 + 1)
    y1 = min(g["h"], cyi + crop_h // 2 + 1)
    M = np.float32([[sx, 0, cxi - crop_w // 2 + (sx - 1) / 2 - x0],
                    [0, sy, cyi - crop_h // 2 + (sy - 1) / 2 - y0]])
    return cv2.warpAffine(sat[y0:y1, x0:x1], M, (SZ_W, SZ_H),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT)


def altitude_scales(height_m, geo, flight):
    """Pick a legacy scale closest to what altitude implies."""
    m_per_px = metric_m_per_px(height_m, flight=flight)
    sat_m_per_px = (math.cos(math.radians((geo["lt_lat"] + geo["rb_lat"]) / 2))
                    * 111_320 / geo["pplon"])
    target = m_per_px * SZ_W / sat_m_per_px / CROP_W
    return sorted(SCALES, key=lambda s: abs(s - target))


# ── Drawing ─────────────────────────────────────────────────────────────────

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
    ap.add_argument("--step",  type=int, default=1, help="row stride between samples")
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

    print(f"flight={args.flight}  "
          f"rows {args.start}..{args.start + args.count * args.step - 1}  "
          f"step={args.step}  yaw_sign={args.yaw_sign}  out={out_dir}")

    for _, row in df.iterrows():
        f      = row["filename"]
        lat    = float(row["lat"])
        lon    = float(row["lon"])
        height = float(row["height"])
        yaw    = float(row["Phi1"]) * args.yaw_sign

        drone_path = os.path.join(drone_dir, f)
        drone = cv2.imread(drone_path)
        if drone is None:
            print(f"  skip (no image): {f}"); continue
        drone_r = cv2.resize(drone, (SZ_W, SZ_H))

        sat, geo, cx, cy, _ = tile_for_gps(tiles, lat, lon)
        s = altitude_scales(height, geo, args.flight)[0]
        crop_w, crop_h = max(SZ_W, int(CROP_W * s)), max(SZ_H, int(CROP_H * s))
        legacy = legacy_crop_sat(sat, cx, cy, geo, crop_w, crop_h)

        metric_north, _ = metric_crop(sat, geo, cx, cy, height, yaw_deg=0.0,
                                      flight=args.flight)
        metric_rot,   _ = metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw,
                                      flight=args.flight)

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
