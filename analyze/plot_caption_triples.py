"""Qualitative check: is the caption a view-invariant bridge?

For sampled frames that have BOTH a drone caption and a sat-crop caption,
shows [drone image | north-up GT sat crop] with the two captions printed
underneath, so caption<->image faithfulness and drone-caption<->sat-caption
agreement can be judged. Needs the dataset and the FULL caption set bound
(run via slurm/run_plot_fig.sh).

Usage:
    python analyze/plot_caption_triples.py --flight 08 --n 3
"""
import argparse
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from helpers import utils as U  # noqa: E402

CAPTION_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cache", "captions")


def load_caps(flight, target):
    path = os.path.join(CAPTION_DIR, f"{flight}_{target}.jsonl")
    caps = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    caps[r["filename"]] = r["caption"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return caps


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="08")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args()

    dcaps = load_caps(args.flight, "drone")
    scaps = load_caps(args.flight, "sat")
    tiles, drone_dir, drone_csv, _ = U.load_flight(args.flight)
    df = pd.read_csv(drone_csv)
    cand = df[df["filename"].isin(set(dcaps) & set(scaps))]
    print(f"  {len(dcaps)} drone caps | {len(scaps)} sat caps | "
          f"{len(cand)} rows with both")
    if cand.empty:
        sys.exit("no rows with both captions")
    picks = cand.iloc[np.linspace(0, len(cand) - 1, args.n, dtype=int)]

    fig, axes = plt.subplots(args.n, 2, figsize=(11, 5.6 * args.n))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(picks.iterrows()):
        f = row["filename"]
        drone = cv2.imread(os.path.join(drone_dir, f))
        drone = cv2.resize(drone, (U.SZ_W, U.SZ_H), interpolation=cv2.INTER_AREA)
        sat, geo, cx, cy, _ = U.tile_for_gps(tiles, row["lat"], row["lon"])
        patch, _ = U.metric_crop(sat, geo, cx, cy, row["height"], yaw_deg=0.0,
                                 flight=args.flight)
        axes[r, 0].imshow(rgb(drone))
        axes[r, 0].set_title(f"{f} — drone", fontsize=9)
        axes[r, 1].imshow(rgb(patch))
        axes[r, 1].set_title("GT sat crop (north-up)", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
        cap = (f"DRONE: {dcaps[f]}\n\nSAT:   {scaps[f]}")
        axes[r, 0].set_xlabel("\n".join(textwrap.wrap(cap, 150,
                              replace_whitespace=False)), fontsize=7.5,
                              loc="left")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"caption_triples_{args.flight}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
