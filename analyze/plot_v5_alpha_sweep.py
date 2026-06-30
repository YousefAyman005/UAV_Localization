"""v5 image+text fusion alpha-sweep figure: Recall@{1,3,5,10} vs fusion weight
alpha (image share), pooled over the spatial test band, for the LoRA-v5 CLIP
adapter.

REPRODUCIBILITY: the v5 adapter is trained north_up=True, so the input CSVs MUST
come from a fusion run launched WITH --north-up (otherwise the drone queries are
fed un-rotated to a north-up encoder and recall collapses ~12 pp). Generate the
inputs with:
    sbatch slurm/run_clip_fusion.sh clip_lora_v5 --north-up \
        --fuse-alpha 0.70 0.75 0.80 0.85 0.90 0.95 1.00
then render in-container via:
    sbatch slurm/run_plot_fig.sh analyze/plot_v5_alpha_sweep.py \
        --results-dir /opt/uav_localization/results/v5_sweep_387204_northup
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEST_FRAC = 0.25                     # split_flight_rows default
KS = (1, 3, 5, 10)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
})
SINGLE_W = 3.4


def test_band_mask(df, test_frac=TEST_FRAC):
    """Held-out spatial band per flight (mirrors plot_matcher_figures)."""
    mask = pd.Series(False, index=df.index)
    for _, g in df.groupby("flight"):
        lat = g["lat"].to_numpy(dtype=float); lon = g["lon"].to_numpy(dtype=float)
        axis = "lat" if (lat.max() - lat.min()) >= (lon.max() - lon.min()) else "lon"
        order = np.argsort(g[axis].to_numpy(dtype=float))
        n_test = int(round(len(g) * test_frac))
        mask.loc[g.index[order[len(g) - n_test:]]] = True
    return mask


def _recall(df, k, col="gt_tile_rank"):
    r = pd.to_numeric(df[col], errors="coerce")
    return (r < k).mean() * 100.0     # NaN / -1 (GT outside gallery) = miss


def load(results_dir):
    rows = []
    paths = glob.glob(os.path.join(results_dir, "visloc_*v5*_a*_results.csv"))
    for p in sorted(paths, key=lambda x: float(re.search(r"_a([\d.]+)_", x).group(1))):
        alpha = float(re.search(r"_a([\d.]+)_", os.path.basename(p)).group(1))
        df = pd.read_csv(p)
        df["flight"] = df["flight"].astype(str).str.zfill(2)
        df = df[df["skipped"] != True]            # noqa: E712
        # NB: the fusion pipeline already restricts to the 25% spatial test band
        # (clip_fusion_pipeline.py:268), so these CSVs ARE the test band (~1270).
        # Do NOT re-apply test_band_mask here — that double-splits to ~318 (6.25%).
        rows.append((alpha, {k: round(_recall(df, k), 1) for k in KS}, len(df)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/v5_sweep_387204_northup")
    ap.add_argument("--out", default="thesis/figures/matchers")
    args = ap.parse_args()

    rows = load(args.results_dir)
    if len(rows) < 2:
        raise SystemExit(f"need >=2 alphas, found {len(rows)} in {args.results_dir}")
    os.makedirs(args.out, exist_ok=True)
    xs = [a for a, _, _ in rows]
    n = rows[0][2]

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.7))
    shades = [plt.cm.viridis(v) for v in np.linspace(0.2, 0.85, len(KS))]
    for i, k in enumerate(KS):
        ys = [r[1][k] for r in rows]
        ax.plot(xs, ys, color=shades[i], marker="o", markersize=3, label=f"R@{k}")
    ax.set_xlabel(r"fusion weight $\alpha$ (image share)")
    ax.set_ylabel(f"recall (%) — test band ($N={n}$)")
    ax.set_xlim(min(xs) - 0.02, max(xs) + 0.02); ax.set_ylim(0, 100)
    ax.legend(framealpha=0.9, ncols=2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"retrieval_v5_alpha_sweep.{ext}"))
    plt.close(fig)

    # companion table
    cols = ["alpha", "N"] + [f"R{k}" for k in KS]
    df = pd.DataFrame([{"alpha": a, "N": n2, **{f"R{k}": v[k] for k in KS}}
                       for a, v, n2 in rows], columns=cols)
    df.to_csv(os.path.join(args.out, "summary_v5_alpha_sweep.csv"), index=False)
    print(df.to_string(index=False))
    print(f"  [fig] {args.out}/retrieval_v5_alpha_sweep.pdf|png")
    print(f"  [tab] {args.out}/summary_v5_alpha_sweep.csv")


if __name__ == "__main__":
    main()
