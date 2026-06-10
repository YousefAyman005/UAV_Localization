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
               raw_pred_px, raw_err_px, raw_err_m, plat, plon, off_m, m_per_px,
               gt_in_patch=None):
    # Intrinsic scale of the estimated homography (drone→patch). For a similarity
    # H this is exactly the drone↔patch scale ratio; used by the H-scale K
    # calibration in pipelines/calibrate_k.py (K = h_scale · SEARCH_FACTOR · K_used).
    _H = match.get("H")
    h_scale = (math.sqrt(abs(float(np.linalg.det(np.asarray(_H, float)[:2, :2]))))
               if _H is not None else None)
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
               offset_m=_round(off_m, 2), h_scale=_round(h_scale, 5),
               gt_in_patch=gt_in_patch)
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
    print(f"  Homography accepted: {len(accepted)}/{n} ({100 * len(accepted) / n:.1f}%)")
    print(f"  Median inliers: {v['inliers'].median():.0f} | "
          f"ratio: {v['inlier_ratio'].median():.3f}")


def _summarize(rows, label, min_inl):
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
