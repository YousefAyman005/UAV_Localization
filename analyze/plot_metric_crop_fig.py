"""Figure: what the metric-isotropic, heading-aligned crop buys (Sec. 4.3.3).

Four panels for one drone image: (a) the drone image at working resolution,
(b) a raw north-up satellite slice of the same area at native map scale,
(c) the standardized metric crop (fixed GSD, heading rotated up), and (d) the
full satellite map with the rotated crop footprint. Reuses helpers.utils so
the crop, the seeded GPS-prior offset, and the footprint are exactly the
pipeline's — must run inside the container (cv2 etc.).

`--crop-zoom` widens the illustrated crop (footprint, slice, and metric crop
together) beyond the pipeline's 1.75 SEARCH_FACTOR — purely for a clearer,
more zoomed-out illustration; 1.0 reproduces the true pipeline crop.

Usage (dataset bound at the default DATASET_DIR location):
    python analyze/plot_metric_crop_fig.py --flight 08
"""
import argparse
import math
import os
import sys
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from helpers import utils as U  # noqa: E402


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="08")
    ap.add_argument("--index", type=int, default=None,
                    help="drone CSV row (default: middle of the flight)")
    ap.add_argument("--yaw-offset", type=float, default=0.0,
                    help="extra degrees added on top of the pipeline's "
                         "corrected_yaw (manual experiments only)")
    ap.add_argument("--no-yaw-cal", action="store_true",
                    help="use raw Phi1 (the pre-calibration pipeline)")
    ap.add_argument("--crop-zoom", type=float, default=1.5,
                    help="widen the illustrated crop by this factor on top of "
                         "the pipeline span (1.0 = true pipeline crop)")
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args()

    tiles, drone_dir, drone_csv, _ = U.load_flight(args.flight)
    df = pd.read_csv(drone_csv)
    row = df.iloc[args.index if args.index is not None else len(df) // 2]
    fname, lat, lon = row["filename"], row["lat"], row["lon"]
    height, yaw = row["height"], float(row["Phi1"])
    if not args.no_yaw_cal:
        yaw = U.corrected_yaw(args.flight, yaw)
    yaw += args.yaw_offset

    drone = cv2.imread(os.path.join(drone_dir, fname))
    drone = cv2.resize(drone, (U.SZ_W, U.SZ_H), interpolation=cv2.INTER_AREA)

    sat, geo, cx, cy, in_bounds = U.tile_for_gps(tiles, lat, lon)
    assert in_bounds, f"{fname}: prior outside map"

    # pipeline-identical seeded prior offset (collect_pipeline_rows_multitile)
    seed = zlib.crc32(f"{args.flight}/{fname}".encode())
    dx_m, dy_m = np.random.default_rng(seed).normal(0.0, U.PRIOR_OFFSET_STD_M, 2)
    mid_lat = (geo["lt_lat"] + geo["rb_lat"]) / 2
    sx_per_m = geo["pplon"] / (math.cos(math.radians(mid_lat)) * U.DEG_TO_M)
    sy_per_m = geo["pplat"] / U.DEG_TO_M
    cx += dx_m * sx_per_m
    cy += dy_m * sy_per_m

    # widen the illustrated crop beyond the pipeline's 1.75 SEARCH_FACTOR
    # (k_override scales the target GSD, so the patch spans more ground)
    k_flight = U.K_PER_FLIGHT.get(str(args.flight), U.K_DEFAULT)
    patch, M = U.metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw,
                             flight=args.flight,
                             k_override=k_flight * args.crop_zoom)
    assert patch is not None, f"{fname}: crop rejected"

    # crop footprint in satellite px: patch corners through M
    corners = np.array([[0, 0], [U.SZ_W, 0], [U.SZ_W, U.SZ_H], [0, U.SZ_H]],
                       dtype=np.float64)
    foot = corners @ M[:, :2].T + M[:, 2]

    # raw north-up slice covering the footprint extent, cut to the same
    # aspect as the drone/crop panels so the three panels tile cleanly
    half = int(math.ceil(max(np.ptp(foot[:, 0]), np.ptp(foot[:, 1])) / 2))
    half_w = int(round(half * U.SZ_W / U.SZ_H))
    cxi, cyi = int(round(cx)), int(round(cy))
    x0, y0 = max(0, cxi - half_w), max(0, cyi - half)
    x1, y1 = min(geo["w"], cxi + half_w), min(geo["h"], cyi + half)
    raw = sat[y0:y1, x0:x1]

    # downsampled full map for the context panel
    s = 1500.0 / max(geo["w"], geo["h"])
    overview = cv2.resize(sat, (int(geo["w"] * s), int(geo["h"] * s)),
                          interpolation=cv2.INTER_AREA)

    # row heights derived from the image aspects so every panel fills its
    # cell exactly (no floating/centered panels): the full-width map bar on
    # top (pipeline order: map -> slice -> crop -> drone), three 3:2 panels
    # below
    panel_h = (1.0 / 3.0) * U.SZ_H / U.SZ_W
    map_h = geo["h"] / geo["w"]
    fig_w = 11.0
    fig = plt.figure(figsize=(fig_w, fig_w * (panel_h + map_h) + 0.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[map_h, panel_h],
                          wspace=0.04, hspace=0.16,
                          left=0.005, right=0.995, top=0.93, bottom=0.01)
    ax_m = fig.add_subplot(gs[0, :])
    ax_r = fig.add_subplot(gs[1, 0])
    ax_p = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    axes = (ax_m, ax_r, ax_p, ax_d)
    ax_m.imshow(rgb(overview))
    ax_m.plot(np.append(foot[:, 0], foot[0, 0]) * s,
              np.append(foot[:, 1], foot[0, 1]) * s, "-", c="#d62728", lw=2.2)
    ax_m.set_title("(a) satellite map with crop footprint", fontsize=9)
    ax_r.imshow(rgb(raw))
    ax_r.set_title("(b) raw slice (north-up, native scale)", fontsize=9)
    ax_p.imshow(rgb(patch))
    ax_p.set_title("(c) metric crop (heading-up, fixed GSD)", fontsize=9)
    ax_d.imshow(rgb(drone))
    ax_d.set_title(f"(d) drone image ({U.SZ_W}×{U.SZ_H})", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(args.out, exist_ok=True)
    suffix = f"_off{args.yaw_offset:+.0f}" if args.yaw_offset else ""
    for ext in ("pdf", "png"):
        path = os.path.join(args.out, f"metric_crop_{args.flight}{suffix}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("wrote", path, f"({fname}, yaw {yaw:.1f}°, alt {height:.0f} m, "
              f"prior offset {math.hypot(dx_m, dy_m):.0f} m, "
              f"crop-zoom {args.crop_zoom:g}×)")


if __name__ == "__main__":
    main()
