"""Figure: spatial train/buffer/val/test bands per flight (Methodology Sec. 4.4.4).

Scatters each flight's drone GPS positions colored by split band. The band
assignment mirrors helpers.utils.split_flight_rows exactly (argsort along the
wider-spread geographic axis, contiguous bands bottom -> top:
train | buffer | val | test) but is reimplemented here so the script runs
without the container deps (helpers.utils imports cv2/torch).

Usage:
    python analyze/plot_split_bands.py \
        --dataset-dir $DATAPOOL3/datasets/Visloc --out thesis/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FLIGHTS = ["01", "02", "03", "04", "05", "06", "08", "10", "11"]

# Bottom -> top fractions, as in eloftr fine-tuning (Sec. 4.4.4). The CLIP
# variant (Sec. 4.6.2) differs only in merging val into train (no val band).
TEST_FRAC, VAL_FRAC, BUF_FRAC = 0.25, 0.10, 0.05

BAND_STYLE = {  # label -> (color, zorder)
    "train":  ("#1f77b4", 1),
    "buffer": ("#9e9e9e", 2),
    "val":    ("#ff7f0e", 3),
    "test":   ("#d62728", 4),
}


def band_labels(df):
    """Per-row band label, replicating split_flight_rows' slicing arithmetic."""
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    axis = "lat" if (lat.max() - lat.min()) >= (lon.max() - lon.min()) else "lon"
    order = np.argsort(df[axis].to_numpy(dtype=float))
    n = len(df)
    n_test = int(round(n * TEST_FRAC))
    n_val = int(round(n * VAL_FRAC))
    n_buf = int(round(n * BUF_FRAC))
    labels = np.empty(n, dtype=object)
    labels[order[: n - n_test - n_val - n_buf]] = "train"
    labels[order[n - n_test - n_val - n_buf: n - n_test - n_val]] = "buffer"
    labels[order[n - n_test - n_val: n - n_test]] = "val"
    labels[order[n - n_test:]] = "test"
    return labels, axis


def plot_flight(ax, df, flight, marker_size):
    labels, axis = band_labels(df)
    for band, (color, z) in BAND_STYLE.items():
        m = labels == band
        ax.scatter(df["lon"][m], df["lat"][m], s=marker_size, c=color,
                   zorder=z, linewidths=0, label=band)
    # meters-isotropic aspect: 1 deg lon = cos(lat) * 1 deg lat
    ax.set_aspect(1.0 / np.cos(np.radians(df["lat"].mean())))
    ax.set_title(f"flight {flight} (split axis: {axis})", fontsize=9)
    ax.ticklabel_format(useOffset=False, style="plain")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir",
                    default=os.path.join(os.environ.get("DATAPOOL3", ""),
                                         "datasets", "Visloc"))
    ap.add_argument("--out", default="thesis/figures")
    ap.add_argument("--flights", nargs="+", default=FLIGHTS)
    args = ap.parse_args()

    n = len(args.flights)
    if n == 1:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        axes, msize = [ax], 14
    else:
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, grid = plt.subplots(nrows, ncols, figsize=(9.5, 3.2 * nrows))
        axes, msize = grid.ravel(), 4
        for ax in axes[n:]:
            ax.set_visible(False)
    for ax, flight in zip(axes, args.flights):
        csv = os.path.join(args.dataset_dir, flight, f"{flight}.csv")
        df = pd.read_csv(csv)
        plot_flight(ax, df, flight, msize)
        ax.tick_params(labelsize=6 if n > 1 else 8)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4, frameon=False,
               markerscale=3 if n > 1 else 1.5, fontsize=10)
    fig.tight_layout(rect=(0, 0.07 if n == 1 else 0.04, 1, 1))

    suffix = "" if args.flights == FLIGHTS else "_" + "_".join(args.flights)
    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(args.out, f"split_bands{suffix}.{ext}")
        fig.savefig(path, dpi=200)
        print("wrote", path)


if __name__ == "__main__":
    main()
