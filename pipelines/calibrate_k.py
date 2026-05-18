"""Aggregate the k-sweep CSVs and write pipelines/k_calibration.json.

Expected input: a directory of CSVs produced by running a pipeline (e.g.
lightglue) with --k-override <K> --results-suffix k<K>_f<FLIGHT>. Filenames
must match the pattern visloc_<name>_k<K>_f<FLIGHT>_results.csv.

For each native drone resolution (derived from the flight's first drone JPEG),
this picks the K maximising median inlier count across the sampled images,
tie-broken by A@25m. The chosen K per native_res is written to
pipelines/k_calibration.json.
"""

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.utils import DATASET_DIR

_CSV_RE = re.compile(r"visloc_.+_k(?P<k>[0-9.]+)_f(?P<flight>[0-9]+)_results\.csv$")


def _native_res_for_flight(flight):
    drone_dir = os.path.join(DATASET_DIR, flight, "drone")
    sample = next((f for f in sorted(os.listdir(drone_dir))
                   if f.lower().endswith(".jpg")), None)
    if sample is None:
        raise FileNotFoundError(f"No drone JPGs under {drone_dir}")
    with Image.open(os.path.join(drone_dir, sample)) as im:
        return im.size  # (w, h)


def _summarise_csv(path):
    df = pd.read_csv(path)
    if "skipped" in df.columns:
        df = df[~df["skipped"].fillna(False)]
    if df.empty or "inliers" not in df.columns:
        return None
    n = len(df)
    med_inl = float(df["inliers"].median())
    succ_25 = int(df.get("success_25", pd.Series(dtype=bool)).fillna(False).sum())
    a25 = succ_25 / n if n else 0.0
    return dict(n=n, median_inliers=med_inl, a25=a25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="Directory containing the sweep CSVs.")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "k_calibration.json"),
                    help="Where to write k_calibration.json.")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.results_dir, "visloc_*_k*_f*_results.csv")))
    if not csvs:
        sys.exit(f"No sweep CSVs found in {args.results_dir}")

    # rows: (native_res_str, k, flight, n, median_inliers, a25)
    rows = []
    flight_to_res = {}
    for path in csvs:
        m = _CSV_RE.search(os.path.basename(path))
        if not m:
            continue
        k = float(m["k"])
        flight = m["flight"]
        if flight not in flight_to_res:
            w, h = _native_res_for_flight(flight)
            flight_to_res[flight] = f"{w}x{h}"
        res = flight_to_res[flight]
        s = _summarise_csv(path)
        if s is None:
            print(f"  skip (empty): {os.path.basename(path)}")
            continue
        rows.append((res, k, flight, s["n"], s["median_inliers"], s["a25"]))

    if not rows:
        sys.exit("All sweep CSVs were empty after filtering.")

    df = pd.DataFrame(rows, columns=["native_res", "k", "flight",
                                     "n", "median_inliers", "a25"])
    print("\n=== Sweep summary ===")
    print(df.sort_values(["native_res", "k"]).to_string(index=False))

    # If a native_res has multiple calibration flights, average per (res, k).
    agg = (df.groupby(["native_res", "k"])
             .agg(median_inliers=("median_inliers", "mean"),
                  a25=("a25", "mean"))
             .reset_index())

    chosen = {}
    print("\n=== Selection (max median_inliers, tie-break a25) ===")
    for res, sub in agg.groupby("native_res"):
        sub = sub.sort_values(["median_inliers", "a25"], ascending=False)
        best = sub.iloc[0]
        chosen[res] = float(best["k"])
        print(f"  {res}: k* = {best['k']}  "
              f"(median_inliers={best['median_inliers']:.1f}, a25={best['a25']:.3f})")
        # Warn if at the edge of the sweep grid:
        ks = sorted(sub["k"].unique())
        if best["k"] in (ks[0], ks[-1]):
            print(f"    WARNING: k* sits at sweep edge {ks}; widen the grid.")

    with open(args.out, "w") as f:
        json.dump(chosen, f, indent=2, sort_keys=True)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
