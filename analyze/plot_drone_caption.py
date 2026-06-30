"""Qualitative figure(s): drone frame(s) with the VLM caption underneath.

Single panel:   --flight 08 --filename 08_0370.JPG
Selection grid: --frames 08_0335.JPG,03_0071.JPG,...   (flight inferred from prefix)

By default the frame is rotated north-up (matching how caption_crops.py captioned
it, so the caption's compass words line up). Pass --raw to show the original
unrotated frame instead. Run via slurm/run_plot_fig.sh (dataset + captions bound).
"""
import argparse
import json
import math
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 10,
})

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
from helpers import utils as U  # noqa: E402

CAPTION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "cache", "captions")

_CAP_CACHE, _CSV_CACHE = {}, {}


def caps_for(flight):
    if flight not in _CAP_CACHE:
        caps = {}
        with open(os.path.join(CAPTION_DIR, f"{flight}_drone.jsonl")) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    caps[r["filename"]] = r["caption"]
                except Exception:
                    pass
        _CAP_CACHE[flight] = caps
    return _CAP_CACHE[flight]


def paths_for(flight):
    if flight not in _CSV_CACHE:
        _, drone_dir, drone_csv, _ = U.get_flight_paths(flight)
        _CSV_CACHE[flight] = (drone_dir, pd.read_csv(drone_csv))
    return _CSV_CACHE[flight]


def load_frame(fname, raw):
    flight = fname.split("_")[0]
    drone_dir, df = paths_for(flight)
    caption = caps_for(flight).get(fname, "(no caption)")
    bgr = cv2.imread(os.path.join(drone_dir, fname))
    if bgr is None:
        raise FileNotFoundError(os.path.join(drone_dir, fname))
    bgr = cv2.resize(bgr, (U.SZ_W, U.SZ_H), interpolation=cv2.INTER_AREA)
    if not raw:
        row = df[df["filename"] == fname]
        yaw = U.corrected_yaw(flight, float(row.iloc[0]["Phi1"])) if not row.empty else 0.0
        bgr = U.north_up_drone(bgr, yaw)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="08")
    ap.add_argument("--filename", default=None)
    ap.add_argument("--frames", default=None, help="comma-separated filenames for a grid")
    ap.add_argument("--raw", action="store_true", help="show original (unrotated) frame")
    ap.add_argument("--caption", default=None, help="override the caption text shown")
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args()
    tag = "raw" if args.raw else "northup"
    os.makedirs(args.out, exist_ok=True)

    if args.frames:
        frames = [s.strip() for s in args.frames.split(",") if s.strip()]
        cols = min(3, len(frames))
        rows = math.ceil(len(frames) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 4.2))
        axes = [axes] if len(frames) == 1 else axes.flatten()
        for i, fname in enumerate(frames):
            rgb, cap = load_frame(fname, args.raw)
            ax = axes[i]
            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title(fname, fontsize=8)
            ax.text(0.5, -0.03, textwrap.fill(cap, 38), transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, style="italic")
        for j in range(len(frames), len(axes)):
            axes[j].axis("off")
        fig.subplots_adjust(hspace=0.6, wspace=0.08, top=0.95, bottom=0.04)
        stem = f"drone_caption_grid_{tag}"
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out, f"{stem}.{ext}"), dpi=150)
        print(f"  [fig] {args.out}/{stem}.png|pdf  ({len(frames)} frames)")
        return

    fname = args.filename or next(iter(caps_for(args.flight)))
    rgb, cap = load_frame(fname, args.raw)
    if args.caption:
        cap = args.caption
    wrapped = textwrap.fill(cap, 46)
    n_lines = wrapped.count("\n") + 1
    h, w = rgb.shape[:2]
    fig_w = 4.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * h / w + 0.34 * n_lines + 0.15))
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():            # thin neat frame, no ticks/title
        sp.set_edgecolor("0.55"); sp.set_linewidth(0.7)
    fig.text(0.5, 0.018, wrapped, ha="center", va="bottom", fontsize=10.5,
             style="italic", color="0.15")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.99,
                        bottom=0.03 + 0.058 * n_lines)
    stem = f"drone_caption_{fname.split('_')[0]}_{os.path.splitext(fname)[0]}_{tag}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    print(f"  caption: {cap}")
    print(f"  [fig] {args.out}/{stem}.png|pdf")


if __name__ == "__main__":
    main()
