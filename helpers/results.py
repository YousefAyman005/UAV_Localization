"""Per-flight / overall summary of a results CSV.

`print_summary` takes the non-skipped rows of one run and prints the gated and
ungated A@Xm, error statistics, oracle GT-in-patch rate and match latency;
`summarize_rows` is the row-list wrapper `run_pipeline` calls per flight.
"""

import math

import numpy as np
import pandas as pd

from helpers.utils import ACC_THRESHOLDS, MIN_INL


def print_summary(v, label, min_inl=MIN_INL, n_skipped=0):
    if v.empty:
        print("\n  All images skipped."); return
    n = len(v)
    accepted     = v[v["inliers"] >= min_inl]
    accepted_err = accepted["offset_m"].dropna() if "offset_m" in accepted else pd.Series(dtype=float)
    raw_err      = (pd.to_numeric(v["raw_err_m"], errors="coerce")
                    if "raw_err_m" in v else pd.Series(dtype=float, index=v.index))
    print(f"\n  Results saved to {label}")
    if n_skipped:
        print(f"  Skipped:             {n_skipped} images (no drone img / GT off-map "
              f"/ <20% crop coverage) — excluded from the {n} scored below")
    for t in ACC_THRESHOLDS:
        col = f"success_{t}"
        s = int(v[col].fillna(False).sum()) if col in v.columns else 0
        print(f"  A@{t:2d}m:              {s}/{n} ({100 * s / n:.1f}%)")
    # Ungated A@t: raw_err_m is the centre-projection error vs TRUE GT for any
    # estimated H, before the inlier gate. NaN (no H) counts as failure.
    if raw_err.notna().any():
        ung = " | ".join(f"@{t} {100 * int((raw_err <= t).sum()) / n:.1f}%"
                         for t in ACC_THRESHOLDS)
        print(f"  A ungated (any H):   {ung}")
    if "gt_in_patch" in v.columns:
        g = v["gt_in_patch"].fillna(False)
        print(f"  GT inside patch:     {int(g.sum())}/{n} ({100 * g.mean():.1f}%) "
              f"— oracle ceiling (GT outside the searched patch is unsolvable)")
    if len(accepted_err):
        rmse = math.sqrt(float(np.mean(np.square(accepted_err))))
        print(f"  Error accepted:      mean {accepted_err.mean():.1f}m | "
              f"median {accepted_err.median():.1f}m | RMSE {rmse:.1f}m | "
              f"P90 {np.percentile(accepted_err, 90):.1f}m | max {accepted_err.max():.1f}m")
    if raw_err.notna().any():
        re = raw_err.dropna()
        print(f"  Ungated err vs GT:   median {re.median():.1f}m | "
              f"P90 {np.percentile(re, 90):.1f}m | max {re.max():.1f}m")
    # Per-image matching latency (model fwd + robust fit). Guarded: CSVs
    # written before this column existed must still summarize cleanly.
    if "t_match_ms" in v.columns:
        tm = pd.to_numeric(v["t_match_ms"], errors="coerce").dropna()
        if len(tm):
            print(f"  Match time:          median {tm.median():.1f} ms | "
                  f"mean {tm.mean():.1f} ms | "
                  f"P90 {np.percentile(tm, 90):.1f} ms")
    print(f"  Homography accepted: {len(accepted)}/{n} ({100 * len(accepted) / n:.1f}%)")
    print(f"  Median inliers: {v['inliers'].median():.0f} | "
          f"ratio: {v['inlier_ratio'].median():.3f}")


def summarize_rows(rows, label, min_inl):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "skipped" not in df.columns:
        return
    skipped = df["skipped"].fillna(False)
    valid = df[~skipped]
    if valid.empty:
        print(f"\n  {label}: all {len(df)} images skipped.")
        return
    print_summary(valid, label, min_inl=min_inl, n_skipped=int(skipped.sum()))
