"""Week-1 retrieval-recall gate analyzer.

Reads one or more visloc_<model>_results.csv files produced by
clip_pipeline.py (with --gps-radii), computes Recall@1/5/10 per
(model, flight, mode) where mode ∈ {denied, r<R>}, prints a table, and
writes recall_summary.csv.

Usage:
    .venv/bin/python3 analyze/retrieval_recall.py \\
        --csvs visloc_clip_results.csv visloc_geoclip_results.csv \\
               visloc_satclip_results.csv visloc_mobileclip_results.csv \\
        --out recall_summary.csv
"""

import argparse
import os
import re

import pandas as pd

RECALL_KS = (1, 5, 10)


def _model_name(path):
    m = re.search(r"visloc_(.+?)_results\.csv$", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def _mode_columns(df):
    """Yield (mode_label, column_name) for every rank column present."""
    if "gt_tile_rank" in df.columns:
        yield "denied", "gt_tile_rank"
    for col in sorted(c for c in df.columns if re.fullmatch(r"gt_rank_r\d+", c)):
        yield col.replace("gt_rank_", ""), col


def _recall(series, k):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    valid = valid[valid >= 0]
    if valid.empty:
        return float("nan")
    return float((valid < k).mean())


def aggregate(csv_paths):
    rows = []
    for path in csv_paths:
        if not os.path.isfile(path):
            print(f"  WARN: missing {path}, skipping"); continue
        df = pd.read_csv(path)
        if "skipped" in df.columns:
            df = df[~df["skipped"].fillna(False)]
        if df.empty:
            print(f"  WARN: {path} has no non-skipped rows"); continue
        model = _model_name(path)
        flights = df["flight"].unique() if "flight" in df.columns else ["all"]
        for flight in sorted(flights):
            sub = df if "flight" not in df.columns else df[df["flight"] == flight]
            for mode, col in _mode_columns(sub):
                row = {"model": model, "flight": str(flight), "mode": mode,
                       "N": len(sub)}
                for k in RECALL_KS:
                    row[f"R@{k}"] = round(_recall(sub[col], k), 4)
                rows.append(row)
    return pd.DataFrame(rows)


def print_table(df):
    if df.empty:
        print("  (no data)"); return
    for mode in df["mode"].unique():
        print(f"\n## Mode: {mode}")
        sub = df[df["mode"] == mode].sort_values(["model", "flight"])
        cols = ["model", "flight", "N"] + [f"R@{k}" for k in RECALL_KS]
        print(sub[cols].to_string(index=False))
        # Per-model average across flights.
        agg = sub.groupby("model")[[f"R@{k}" for k in RECALL_KS]].mean().round(4)
        print("\n  Mean across flights:")
        print(agg.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="visloc_<model>_results.csv files to aggregate.")
    ap.add_argument("--out", default="recall_summary.csv",
                    help="Where to write the aggregated summary CSV.")
    args = ap.parse_args()

    df = aggregate(args.csvs)
    print_table(df)
    df.to_csv(args.out, index=False)
    print(f"\n  Wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
