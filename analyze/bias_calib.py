#!/usr/bin/env python
"""Train-band bias calibration: the no-NN control for the GT-supervised finetune.

Estimates, per flight, the median prediction-error vector of a matcher on the
TRAIN band only (accepted rows: inliers >= MIN_INL and a predicted position),
then subtracts it from every accepted prediction and rewrites offset_m. Two
frames: world East/North (catches flight 08's fixed offset) and along/cross
track (track bearing from consecutive GT rows in the CSV; catches the
gimbal-pitch/GPS-lag along-track bias). Writes corrected CSV copies
(visloc_<method>_calib_{en,track}_results.csv) whose schema matches the input,
so analyze/band_metrics.py consumes them unchanged. raw_* columns are left
untouched (no ungated prediction vector is stored in the CSVs) — compare the
calibrated variants on GATED metrics only.

Run (container, login node — no GPU/dataset needed):
    apptainer exec --bind /data/cluster uav_localization.sif \\
        python3 analyze/bias_calib.py --csvs visloc_eloftr_results.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# helpers.utils first: utils' back-import of results breaks if results goes first.
from helpers.utils import MIN_INL, haversine_m  # noqa: E402
from band_metrics import _load, band_rows  # noqa: E402

M_PER_DEG_LAT = 110_540.0
SPLIT_DEFAULTS = argparse.Namespace(test_frac=0.25, val_frac=0.10,
                                    buffer_frac=0.05, axis="auto")


def _m_per_deg_lon(lat):
    return 111_320.0 * np.cos(np.radians(lat))


def _en_errors(df):
    """Per-row prediction-error vector in meters (East, North); NaN w/o pred."""
    e = (df["pred_lon"] - df["lon"]) * _m_per_deg_lon(df["lat"])
    n = (df["pred_lat"] - df["lat"]) * M_PER_DEG_LAT
    return e, n


def _track_dir(df):
    """Per-row unit direction of motion (East, North) from consecutive TRUE
    positions in CSV order within each flight (forward difference; the last
    row reuses the previous direction)."""
    te = np.zeros(len(df))
    tn = np.ones(len(df))
    for _, g in df.groupby("flight", sort=False):
        loc = df.index.get_indexer(g.index)
        de = np.diff(g["lon"].to_numpy()) * _m_per_deg_lon(g["lat"].to_numpy()[:-1])
        dn = np.diff(g["lat"].to_numpy()) * M_PER_DEG_LAT
        if len(de) == 0:
            continue
        de = np.append(de, de[-1])
        dn = np.append(dn, dn[-1])
        norm = np.hypot(de, dn)
        norm[norm < 1e-6] = 1.0
        te[loc] = de / norm
        tn[loc] = dn / norm
    return te, tn


def _apply(df, corr_e, corr_n):
    """Subtract the per-row (East, North) meter correction from the prediction
    and recompute offset_m vs GT. Rows without a prediction stay unchanged."""
    out = df.copy()
    ok = out["pred_lat"].notna() & out["pred_lon"].notna()
    out.loc[ok, "pred_lon"] = (out.loc[ok, "pred_lon"]
                               - corr_e[ok] / _m_per_deg_lon(out.loc[ok, "lat"]))
    out.loc[ok, "pred_lat"] = out.loc[ok, "pred_lat"] - corr_n[ok] / M_PER_DEG_LAT
    off = [haversine_m(la, lo, pla, plo) for la, lo, pla, plo in
           zip(out.loc[ok, "lat"], out.loc[ok, "lon"],
               out.loc[ok, "pred_lat"], out.loc[ok, "pred_lon"])]
    out.loc[ok, "offset_m"] = np.round(off, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--min-inliers", type=int, default=MIN_INL)
    args = ap.parse_args()
    for path in args.csvs:
        df, n_skip = _load(path)
        for col in ("pred_lat", "pred_lon"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        tr = band_rows(df, "train", SPLIT_DEFAULTS)
        train_keys = set(zip(tr["flight"], tr["filename"]))
        in_train = pd.Series([k in train_keys for k in
                              zip(df["flight"], df["filename"])], index=df.index)
        acc = (df["inliers"].fillna(0) >= args.min_inliers) & df["pred_lat"].notna()
        fit = in_train & acc
        e, n = _en_errors(df)
        te, tn = _track_dir(df)
        along = e * te + n * tn
        cross = -e * tn + n * te
        corr = {"en": (np.zeros(len(df)), np.zeros(len(df))),
                "track": (np.zeros(len(df)), np.zeros(len(df)))}
        print(f"\n== {path} (train-band fit rows: {int(fit.sum())}, "
              f"{n_skip} skipped excluded)")
        for fl in sorted(df["flight"].unique()):
            m = (fit & (df["flight"] == fl)).to_numpy()
            if m.sum() < 10:
                print(f"  flight {fl}: <10 accepted train rows — no correction")
                continue
            be, bn = float(e[m].median()), float(n[m].median())
            ba, bc = float(along[m].median()), float(cross[m].median())
            print(f"  flight {fl}: EN bias ({be:+6.1f} E, {bn:+6.1f} N) m | "
                  f"track bias ({ba:+6.1f} along, {bc:+6.1f} cross) m")
            row = (df["flight"] == fl).to_numpy()
            corr["en"][0][row], corr["en"][1][row] = be, bn
            # Reconstruct the EN vector of the track-frame bias per row:
            # bias = ba·t̂ + bc·ĉ with t̂=(te,tn), ĉ=(-tn,te).
            corr["track"][0][row] = ba * te[row] - bc * tn[row]
            corr["track"][1][row] = ba * tn[row] + bc * te[row]
        for tag, (ce, cn) in corr.items():
            out = _apply(df, pd.Series(ce, index=df.index),
                         pd.Series(cn, index=df.index))
            dst = path.replace("_results.csv", f"_calib_{tag}_results.csv")
            out.to_csv(dst, index=False)
            print(f"  wrote {dst}")


if __name__ == "__main__":
    main()
