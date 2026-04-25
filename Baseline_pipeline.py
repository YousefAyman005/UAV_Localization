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


def run_match(sg, dg, detector, method):
    kps, ds = detector.detectAndCompute(sg, None)
    kpd, dd = detector.detectAndCompute(dg, None)
    r = dict(sat_kp=len(kps), drone_kp=len(kpd), raw=0, good=0, inliers=0, H=None,
             _kps=kps, _kpd=kpd, _matches=[], _sg=sg)
    if ds is None or dd is None or len(kps) < 4 or len(kpd) < 4:
        return r
    matcher = (cv2.FlannBasedMatcher({"algorithm": 1, "trees": FLANN_TREES}, {"checks": FLANN_CHECKS})
               if method == "sift" else cv2.BFMatcher(cv2.NORM_HAMMING))
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


def _process_chunk(args):
    """Worker: runs one chunk of drone images. Creates its own detector (not thread-safe)."""
    chunk_df, tiles, drone_dir, method, dist, flight, viz_dir = args
    detector = _make_detector(method)

    def match_factory(drone):
        dg = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
        return lambda p: run_match(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), dg, detector, method)

    return collect_pipeline_rows_multitile(
        tiles, chunk_df, match_factory, dist,
        drone_dir=drone_dir, flight=flight,
        viz_dir=viz_dir, progress=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist",      type=float, default=25.0)
    ap.add_argument("--method",    choices=["sift", "orb", "brisk"], default="sift")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--flights",   nargs="+", default=["all"],
                    help="Flight IDs to evaluate, e.g. 01 03 05, or 'all' (default)")
    args = ap.parse_args()

    flights   = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    n_workers = os.cpu_count() or 1
    OUT_CSV   = OUT_CSV_TEMPLATE.format(method=args.method)
    VIZ_DIR   = VIZ_DIR_TEMPLATE.format(method=args.method)

    print(f"  Method: {args.method.upper()} | Dist: {args.dist}m | "
          f"Workers: {n_workers} | Flights: {' '.join(flights)}")

    def viz_fn(drone, patch, best, filename, viz_dir):
        if not best["_matches"]: return
        dg = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
        top = sorted(best["_matches"], key=lambda m: m.distance)[:TOP_MATCHES]
        draw_and_save(dg, best["_kpd"], best["_sg"], best["_kps"], top, filename, viz_dir)

    log_path = OUT_CSV.replace(".csv", ".log")
    with TeeLogger(log_path):
        all_rows = []
        for flight in flights:
            tiles, drone_dir, drone_csv, _ = load_flight(flight)
            df = pd.read_csv(drone_csv)
            print(f"\n=== Flight {flight}: {len(df)} images ===")

            chunks = [c.reset_index(drop=True) for c in np.array_split(df, n_workers) if len(c) > 0]
            viz_dir_arg = VIZ_DIR if args.visualize else None

            if len(chunks) == 1:
                # Single chunk — run sequentially with tqdm progress bar.
                detector = _make_detector(args.method)
                def match_factory(drone):
                    dg = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
                    return lambda p: run_match(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), dg, detector, args.method)
                rows = collect_pipeline_rows_multitile(
                    tiles, df, match_factory, args.dist,
                    drone_dir=drone_dir, flight=flight,
                    viz_fn=viz_fn if args.visualize else None,
                    viz_dir=viz_dir_arg)
            else:
                chunk_args = [(c, tiles, drone_dir, args.method, args.dist, flight, viz_dir_arg)
                              for c in chunks]
                with mp.Pool(n_workers) as pool:
                    results = pool.map(_process_chunk, chunk_args)
                rows = [r for chunk_rows in results for r in chunk_rows]

            all_rows.extend(rows)
            flight_df = pd.DataFrame(rows)
            valid = flight_df[~flight_df["skipped"].fillna(False)]
            if not valid.empty:
                print_summary(valid, args.dist, f"flight {flight}")

        out = pd.DataFrame(all_rows)
        out.to_csv(OUT_CSV, index=False)
        if len(flights) > 1:
            print(f"\n=== Overall ({len(flights)} flights) ===")
            valid_all = out[~out["skipped"].fillna(False)]
            if not valid_all.empty:
                print_summary(valid_all, args.dist, OUT_CSV)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
