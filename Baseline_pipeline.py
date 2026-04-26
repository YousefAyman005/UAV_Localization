import multiprocessing as mp
import os
import argparse
import cv2
import numpy as np
import pandas as pd
from visloc_utils import (
    RANSAC_THRESH, TOP_MATCHES,
    FLIGHTS_AVAILABLE, load_flight, collect_pipeline_rows_multitile,
    print_summary, draw_and_save, TeeLogger,
)

OUT_CSV_TEMPLATE = "visloc_{method}_results.csv"
VIZ_DIR_TEMPLATE = "visloc_{method}_visualizations"
LOWE = 0.75
FLANN_TREES, FLANN_CHECKS = 5, 50


def run_match(sg, kpd, dd, detector, matcher):
    kps, ds = detector.detectAndCompute(sg, None)
    r = dict(sat_kp=len(kps), drone_kp=len(kpd), raw=0, good=0, inliers=0, H=None,
             _kps=kps, _kpd=kpd, _matches=[], _sg=sg)
    if ds is None or dd is None or len(kps) < 4 or len(kpd) < 4:
        return r
    matches = matcher.knnMatch(dd, ds, k=2)
    good = [m for pair in matches if len(pair) == 2
            for m, n in [pair] if m.distance < LOWE * n.distance]
    r["raw"], r["good"], r["_matches"] = len(matches), len(good), good
    if len(good) >= 4:
        src = np.float32([kpd[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
        if H is not None and mask is not None:
            r["inliers"], r["H"] = int(mask.sum()), H
    return r


def _make_detector(method):
    return {"sift": cv2.SIFT_create,
            "orb":  lambda: cv2.ORB_create(5000),
            "brisk": cv2.BRISK_create}[method]()


def _make_match_factory(method):
    detector = _make_detector(method)
    matcher = (cv2.FlannBasedMatcher({"algorithm": 1, "trees": FLANN_TREES}, {"checks": FLANN_CHECKS})
               if method == "sift" else cv2.BFMatcher(cv2.NORM_HAMMING))

    def match_factory(drone):
        kpd, dd = detector.detectAndCompute(cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY), None)
        return lambda p: run_match(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), kpd, dd, detector, matcher)

    return match_factory


def save_baseline_viz(drone, patch, best, filename, viz_dir):
    matches = best.get("_matches") or []
    top = sorted(matches, key=lambda m: m.distance)[:TOP_MATCHES] if matches else []
    draw_and_save(drone, best.get("_kpd", []), patch, best.get("_kps", []),
                  top, filename, viz_dir, H=best.get("H"))


def collect_rows(tiles, df, method, dist, drone_dir, flight, viz_dir, progress=True):
    return collect_pipeline_rows_multitile(
        tiles, df, _make_match_factory(method), dist, drone_dir=drone_dir, flight=flight,
        viz_fn=save_baseline_viz if viz_dir else None, viz_dir=viz_dir, progress=progress)


def _process_chunk(args):
    chunk_df, tiles, drone_dir, method, dist, flight, viz_dir = args
    return collect_rows(tiles, chunk_df, method, dist, drone_dir, flight, viz_dir, progress=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist",      type=float, default=25.0)
    ap.add_argument("--method",    choices=["sift", "orb", "brisk"], default="sift")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--flights",   nargs="+", default=["all"],
                    help="Flight IDs to evaluate, e.g. 01 03 05, or 'all' (default)")
    ap.add_argument("--workers",   type=int, default=None,
                    help="Number of parallel workers (default: cpu_count)")
    args = ap.parse_args()

    flights   = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    n_workers = args.workers or os.cpu_count() or 1
    OUT_CSV   = OUT_CSV_TEMPLATE.format(method=args.method)
    VIZ_DIR   = VIZ_DIR_TEMPLATE.format(method=args.method)

    print(f"  Method: {args.method.upper()} | Dist: {args.dist}m | "
          f"Workers: {n_workers} | Flights: {' '.join(flights)}")

    log_path = OUT_CSV.replace(".csv", ".log")
    with TeeLogger(log_path):
        all_rows = []
        for flight in flights:
            tiles, drone_dir, drone_csv, _ = load_flight(flight)
            df = pd.read_csv(drone_csv)
            print(f"\n=== Flight {flight}: {len(df)} images ===")

            chunks = [c.reset_index(drop=True) for c in np.array_split(df, min(n_workers, max(len(df), 1)))]
            viz_dir_arg = VIZ_DIR if args.visualize else None

            if len(chunks) == 1:
                rows = collect_rows(tiles, df, args.method, args.dist, drone_dir, flight, viz_dir_arg)
            else:
                chunk_args = [(c, tiles, drone_dir, args.method, args.dist, flight, viz_dir_arg)
                              for c in chunks]
                with mp.Pool(len(chunks)) as pool:
                    results = pool.map(_process_chunk, chunk_args)
                rows = [r for chunk_rows in results for r in chunk_rows]

            all_rows.extend(rows)
            flight_df = pd.DataFrame(rows)
            valid = flight_df[~flight_df["skipped"].fillna(False)]
            if not valid.empty:
                print_summary(valid, f"flight {flight}")

        out = pd.DataFrame(all_rows)
        out.to_csv(OUT_CSV, index=False)
        if len(flights) > 1:
            print(f"\n=== Overall ({len(flights)} flights) ===")
            valid_all = out[~out["skipped"].fillna(False)]
            if not valid_all.empty:
                print_summary(valid_all, OUT_CSV)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
