"""Result-row construction and CSV summarization.

`_build_row` / `_skip_row` produce the per-image dicts that get written to
the results CSV; `print_summary` / `_summarize` compute the per-flight and
overall accuracy/error stats printed at the end of a run.
"""

import math

import numpy as np
import pandas as pd

from helpers.utils import ACC_THRESHOLDS, MIN_INL, SZ_H, SZ_W


def _round(x, n):
    return None if x is None else round(x, n)


def _build_row(filename, lat, lon, height, flight, match,
               raw_pred_px, raw_err_px, raw_err_m, plat, plon, off_m, m_per_px):
    row = dict(filename=filename, lat=lat, lon=lon, height=height, skipped=False,
               crop_w=SZ_W, crop_h=SZ_H,
               sat_kp=match["sat_kp"], drone_kp=match["drone_kp"],
               raw=match["raw"], good=match["good"], inliers=match["inliers"],
               inlier_ratio=round(match["inliers"] / match["good"], 4)
                            if match["good"] else 0,
               raw_pred_x=_round(raw_pred_px[0], 2) if raw_pred_px else None,
               raw_pred_y=_round(raw_pred_px[1], 2) if raw_pred_px else None,
               raw_err_px=_round(raw_err_px, 2),
               raw_err_m=_round(raw_err_m, 2),
               m_per_px=_round(m_per_px, 4),
               pred_lat=_round(plat, 7), pred_lon=_round(plon, 7),
               offset_m=_round(off_m, 2))
    for t in ACC_THRESHOLDS:
        row[f"success_{t}"] = off_m is not None and off_m <= t
    if flight:
        row["flight"] = flight
    return row


def _skip_row(filename, flight):
    r = {"filename": filename, "skipped": True}
    if flight:
        r["flight"] = flight
    return r


def print_summary(v, label, min_inl=MIN_INL):
    if v.empty:
        print("\n  All images skipped."); return
    n = len(v)
    accepted     = v[v["inliers"] >= min_inl]
    accepted_err = accepted["offset_m"].dropna() if "offset_m" in accepted else pd.Series(dtype=float)
    raw_err      = v["raw_err_m"].dropna()       if "raw_err_m" in v       else pd.Series(dtype=float)
    print(f"\n  Results saved to {label}")
    for t in ACC_THRESHOLDS:
        col = f"success_{t}"
        s = int(v[col].fillna(False).sum()) if col in v.columns else 0
        print(f"  A@{t:2d}m:              {s}/{n} ({100 * s / n:.1f}%)")
    if len(accepted_err):
        rmse = math.sqrt(float(np.mean(np.square(accepted_err))))
        print(f"  Error accepted:      mean {accepted_err.mean():.1f}m | "
              f"median {accepted_err.median():.1f}m | RMSE {rmse:.1f}m | "
              f"P90 {np.percentile(accepted_err, 90):.1f}m | max {accepted_err.max():.1f}m")
    if len(raw_err):
        print(f"  Raw center error:    median {raw_err.median():.1f}m | "
              f"P90 {np.percentile(raw_err, 90):.1f}m | max {raw_err.max():.1f}m")
    print(f"  Homography accepted: {len(accepted)}/{n} ({100 * len(accepted) / n:.1f}%)")
    print(f"  Median inliers: {v['inliers'].median():.0f} | "
          f"ratio: {v['inlier_ratio'].median():.3f}")


def _summarize(rows, label, min_inl):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "skipped" not in df.columns:
        return
    valid = df[~df["skipped"].fillna(False)]
    if not valid.empty:
        print_summary(valid, label, min_inl=min_inl)
