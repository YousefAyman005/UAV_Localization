"""Figure: the view gap — a drone image (oblique perspective) beside the
satellite view of the same ground (orthographic top-down).

Two panels for one drone frame: (a) the drone image at working resolution
(off-nadir camera → building facades, perspective), and (b) a metric-isotropic,
heading-up satellite crop centred on the drone's TRUE GPS (no simulated prior
offset) so it is exactly the same patch of ground, seen straight down. Also
writes the two panels as standalone images. Reuses helpers.utils so the crop is
the pipeline's — must run inside the container (cv2 etc.).

`--crop-zoom` scales the satellite crop's ground span relative to the pipeline
crop. Default matches the drone footprint (1 / SEARCH_FACTOR); 1.0 reproduces
the wider pipeline search crop.

Usage (dataset bound at the default DATASET_DIR location):
    python analyze/plot_oblique_topdown_pair.py --flight 01 --index 281
    # or via the cluster wrapper:
    sbatch slurm/run_plot_fig.sh analyze/plot_oblique_topdown_pair.py \
        --flight 01 --index 281
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


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="01")
    ap.add_argument("--index", type=int, default=None,
                    help="drone CSV row (df.iloc; default: middle of the flight)")
    ap.add_argument("--no-yaw-cal", action="store_true",
                    help="use raw Phi1 instead of the calibrated corrected_yaw")
    ap.add_argument("--crop-zoom", type=float, default=1.0 / U.SEARCH_FACTOR,
                    help="satellite ground span relative to the pipeline crop "
                         "(default 1/SEARCH_FACTOR ~ matches the drone footprint; "
                         "1.0 = full pipeline search crop)")
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args()

    tiles, drone_dir, drone_csv, _ = U.load_flight(args.flight)
    df = pd.read_csv(drone_csv)
    row = df.iloc[args.index if args.index is not None else len(df) // 2]
    fname, lat, lon = row["filename"], row["lat"], row["lon"]
    height, yaw = row["height"], float(row["Phi1"])
    if not args.no_yaw_cal:
        yaw = U.corrected_yaw(args.flight, yaw)

    drone = cv2.imread(os.path.join(drone_dir, fname))
    drone = cv2.resize(drone, (U.SZ_W, U.SZ_H), interpolation=cv2.INTER_AREA)

    # centre on the TRUE GPS (no prior offset): we want the same ground, not the
    # noisy pipeline prior.
    sat, geo, cx, cy, in_bounds = U.tile_for_gps(tiles, lat, lon)
    assert in_bounds, f"{fname}: GPS outside map"

    k_flight = U.K_PER_FLIGHT.get(str(args.flight), U.K_DEFAULT)
    patch, _ = U.metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw,
                             flight=args.flight,
                             k_override=k_flight * args.crop_zoom)
    assert patch is not None, f"{fname}: crop rejected"

    os.makedirs(args.out, exist_ok=True)
    stem = f"{args.flight}_{os.path.splitext(fname)[0]}"

    # standalone panels
    for name, img in (("drone_oblique", drone), ("sat_topdown", patch)):
        p = os.path.join(args.out, f"{name}_{stem}.png")
        cv2.imwrite(p, img)
        print("wrote", p)

    # side-by-side pair
    fig, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(11, 11 * U.SZ_H / U.SZ_W / 2))
    ax_d.imshow(rgb(drone))
    ax_d.set_title("(a) drone image — oblique perspective", fontsize=10)
    ax_s.imshow(rgb(patch))
    ax_s.set_title("(b) satellite view — orthographic top-down", fontsize=10)
    for ax in (ax_d, ax_s):
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01, wspace=0.04)

    for ext in ("png", "pdf"):
        p = os.path.join(args.out, f"oblique_topdown_{stem}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p, f"({fname}, yaw {yaw:.1f}°, alt {height:.0f} m, "
              f"crop-zoom {args.crop_zoom:g}×)")


if __name__ == "__main__":
    main()
