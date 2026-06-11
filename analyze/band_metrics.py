#!/usr/bin/env python
"""Band-filtered benchmark summaries (spatial train | buffer | val | test bands).

Matcher eval runs score FULL flights, but the ELoFTR-LoRA retrain only learns
from the train band of each flight — honest comparisons (adapted model vs
baselines) must therefore be restricted to the held-out test band. This script
recomputes band membership for any visloc_*_results.csv from the CSV's own
filename/lat/lon columns via ``helpers.utils.split_flight_rows`` (the single
source of truth for the split), then recomputes the gated + ungated accuracy
summary per flight and overall on that band only.

``success_*`` columns are recomputed from offset_m/inliers (gated) and
raw_err_m (ungated): old CSVs predate schema additions (no success_30 /
gt_in_patch) and store booleans as strings, so stored flags are never trusted.

Needs the container (helpers.utils imports cv2), but no GPU/dataset:
    apptainer exec uav_localization.sif python3 analyze/band_metrics.py \\
        --csvs visloc_eloftr_lora_results.csv visloc_eloftr_results.csv \\
               visloc_roma_extre_results.csv --band test
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.results import print_summary  # noqa: E402
from helpers.utils import ACC_THRESHOLDS, MIN_INL, split_flight_rows  # noqa: E402


def _method_name(path):
    m = re.search(r"visloc_(.+?)_results\.csv$", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def _load(path):
    df = pd.read_csv(path, dtype={"flight": str, "filename": str})
    n_skip = 0
    if "skipped" in df.columns:
        skipped = df["skipped"].astype(str).str.lower().eq("true")
        n_skip = int(skipped.sum())
        df = df[~skipped]
    for col in ("offset_m", "raw_err_m", "inliers", "inlier_ratio", "lat", "lon"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True), n_skip


def band_rows(df, band, args):
    """Concat of per-flight band slices, via the shared split logic."""
    parts = []
    for flight in sorted(df["flight"].unique()):
        sub = df[df["flight"] == flight].reset_index(drop=True)
        parts.append(split_flight_rows(sub, which=band, test_frac=args.test_frac,
                                       axis=args.axis, buffer_frac=args.buffer_frac,
                                       val_frac=args.val_frac))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]


def recompute_success(df, min_inl):
    df = df.copy()
    gate = df["inliers"].fillna(0) >= min_inl
    for t in ACC_THRESHOLDS:
        df[f"success_{t}"] = (df["offset_m"] <= t) & gate     # NaN -> False
    return df


def _row(label, flight, band, sub, min_inl):
    n = len(sub)
    acc = sub[sub["inliers"].fillna(0) >= min_inl]["offset_m"].dropna()
    raw = sub["raw_err_m"].dropna() if "raw_err_m" in sub else pd.Series(dtype=float)
    row = {"method": label, "flight": flight, "band": band, "N": n}
    for t in ACC_THRESHOLDS:
        row[f"A@{t}"] = round(100.0 * sub[f"success_{t}"].mean(), 2) if n else None
    for t in ACC_THRESHOLDS:
        row[f"U@{t}"] = round(100.0 * int((raw <= t).sum()) / n, 2) if n else None
    row["med_offset_m"] = round(float(acc.median()), 2) if len(acc) else None
    row["med_raw_err_m"] = round(float(raw.median()), 2) if len(raw) else None
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="visloc_<method>_results.csv files to band-filter.")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Method names (default: parsed from filenames).")
    ap.add_argument("--band", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--buffer-frac", type=float, default=0.05)
    ap.add_argument("--axis", choices=["auto", "lat", "lon"], default="auto")
    ap.add_argument("--min-inliers", type=int, default=MIN_INL)
    ap.add_argument("--out", default="band_summary.csv")
    args = ap.parse_args()

    labels = args.labels or [_method_name(p) for p in args.csvs]
    if len(labels) != len(args.csvs):
        sys.exit("--labels must match --csvs in length")

    banded = {}
    for path, label in zip(args.csvs, labels):
        if not os.path.isfile(path):
            print(f"  WARN: missing {path}, skipping"); continue
        df, n_skip = _load(path)
        if n_skip:
            print(f"  WARN: {label}: {n_skip} skipped rows excluded before banding — "
                  f"band boundaries may shift by up to that many rows.")
        banded[label] = band_rows(df, args.band, args)
    if not banded:
        sys.exit("No readable CSVs.")

    # Row-identity across methods: comparison tables must score identical rows.
    keysets = {lb: set(zip(d["flight"], d["filename"])) for lb, d in banded.items()}
    common = set.intersection(*keysets.values())
    for label, keys in keysets.items():
        if keys != common:
            print(f"  WARN: {label} band differs ({len(keys)} rows vs {len(common)} "
                  f"common) — intersecting all methods to identical rows.")
    banded = {lb: d[[k in common for k in zip(d["flight"], d["filename"])]]
              for lb, d in banded.items()}

    out_rows = []
    for label, df in banded.items():
        df = recompute_success(df, args.min_inliers)
        print(f"\n════ {label} | band={args.band} | N={len(df)} ════")
        for flight in sorted(df["flight"].unique()):
            out_rows.append(_row(label, flight, args.band,
                                 df[df["flight"] == flight], args.min_inliers))
        print_summary(df, f"{label} ({args.band} band)", min_inl=args.min_inliers)
        out_rows.append(_row(label, "ALL", args.band, df, args.min_inliers))

    out = pd.DataFrame(out_rows)
    out.to_csv(args.out, index=False)
    print(f"\n  Wrote {args.out} ({len(out)} rows)")

    print("\n  A@25m gated, by flight:")
    piv = out[out["flight"] != "ALL"].pivot(index="flight", columns="method",
                                            values="A@25")
    piv.loc["ALL"] = out[out["flight"] == "ALL"].set_index("method")["A@25"]
    print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
