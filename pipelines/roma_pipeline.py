import argparse
import multiprocessing
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from visloc_utils import (
    MIN_INL, SZ_W, SZ_H, RANSAC_THRESH,
    FLIGHTS_AVAILABLE, load_flight, collect_pipeline_rows_multitile,
    print_summary, save_dense_viz, TeeLogger,
)

from romatch import roma_outdoor, roma_indoor

OUT_CSV = "visloc_roma_results.csv"
VIZ_DIR = "visloc_roma_visualizations"


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def match_roma(drone_pil, sat_bgr, matcher, conf_thresh, ransac_thresh, device, num_samples):
    with torch.inference_mode():
        warp, cert  = matcher.match(drone_pil, bgr_to_pil(sat_bgr), device=device)
        matches, c  = matcher.sample(warp, cert, num=num_samples)
        kp_a, kp_b  = matcher.to_pixel_coordinates(matches, SZ_H, SZ_W, SZ_H, SZ_W)
    kp0  = kp_a.cpu().numpy().astype(np.float32)
    kp1  = kp_b.cpu().numpy().astype(np.float32)
    c_np = c.cpu().numpy()
    r = dict(sat_kp=len(kp0), drone_kp=len(kp0), raw=len(kp0), good=0, inliers=0,
             H=None, _kp0=kp0, _kp1=kp1, _conf=c_np, _mask=None)
    mask = c_np >= conf_thresh
    r["good"], r["_mask"] = int(mask.sum()), mask
    if r["good"] < 4:
        return r
    H, mh = cv2.findHomography(kp0[mask].reshape(-1, 1, 2),
                               kp1[mask].reshape(-1, 1, 2),
                               cv2.USAC_MAGSAC, ransac_thresh,
                               maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


def _load_model(device, pretrained):
    torch.set_float32_matmul_precision("highest")
    kw = {} if device.type == "cuda" else {"amp_dtype": torch.float32}
    return (roma_outdoor(device=device, **kw) if pretrained == "outdoor"
            else roma_indoor(device=device, **kw))


def _make_match_factory(matcher, device, num_matches):
    def match_factory(drone):
        drone_pil = bgr_to_pil(drone)
        return lambda p: match_roma(drone_pil, p, matcher, 0.0, RANSAC_THRESH, device, num_matches)
    return match_factory


def collect_flight_rows(flight, match_factory, dist, viz_dir, clahe_arg,
                        limit=None, progress=True):
    tiles, drone_dir, drone_csv, _ = load_flight(flight)
    df = pd.read_csv(drone_csv)
    if limit is not None:
        df = df.head(limit)
    if progress: print(f"\n=== Flight {flight}: {len(df)} images ===")
    return collect_pipeline_rows_multitile(
        tiles, df, match_factory, dist, min_inl=MIN_INL,
        drone_dir=drone_dir, flight=flight,
        viz_fn=save_dense_viz if viz_dir else None, viz_dir=viz_dir, progress=progress,
        clahe=clahe_arg)


def summarize_rows(rows, label):
    if not rows: return
    df = pd.DataFrame(rows)
    if "skipped" not in df: return
    valid = df[~df["skipped"].fillna(False)]
    if not valid.empty: print_summary(valid, label, min_inl=MIN_INL)


def _worker(args):
    flight_group, gpu_id, pretrained, dist, num_matches, viz_dir, limit, clahe_arg = args
    device = torch.device(f"cuda:{gpu_id}")
    matcher = _load_model(device, pretrained)
    match_factory = _make_match_factory(matcher, device, num_matches)
    return [r for f in flight_group
            for r in collect_flight_rows(f, match_factory, dist, viz_dir,
                                         clahe_arg, limit, False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist",        type=float, default=25.0)
    ap.add_argument("--pretrained",  choices=["outdoor", "indoor"], default="outdoor")
    ap.add_argument("--num-matches", type=int,   default=5000)
    ap.add_argument("--visualize",   action="store_true")
    ap.add_argument("--flights",     nargs="+", default=["all"],
                    help="Flight IDs to evaluate, e.g. 01 03 05, or 'all' (default)")
    ap.add_argument("--limit",       type=int,   default=None,
                    help="Cap number of drone images per flight (for quick tests)")
    ap.add_argument("--no-clahe",    action="store_true",
                    help="Disable CLAHE preprocessing (on by default)")
    args = ap.parse_args()

    flights   = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    n_gpus    = max(1, torch.cuda.device_count())
    viz_dir   = VIZ_DIR if args.visualize else None
    clahe_arg = None if args.no_clahe else "auto"
    print(f"  Method: RoMa ({args.pretrained}) | NumMatches: {args.num_matches} | "
          f"RANSAC: {RANSAC_THRESH}px | MinInl: {MIN_INL} | "
          f"CLAHE: {'off' if args.no_clahe else 'on'} | "
          f"Dist: {args.dist}m | Flights: {' '.join(flights)} | GPUs: {n_gpus}")

    groups = [g for g in [flights[i::n_gpus] for i in range(n_gpus)] if g]

    log_path = OUT_CSV.replace(".csv", ".log")
    with TeeLogger(log_path):
        if len(groups) == 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"  Device: {device}")
            print(f"  Loading RoMa ({args.pretrained}) ... ", end="", flush=True)
            matcher = _load_model(device, args.pretrained)
            print("done")

            match_factory = _make_match_factory(matcher, device, args.num_matches)
            all_rows = []
            for flight in flights:
                rows = collect_flight_rows(flight, match_factory, args.dist, viz_dir,
                                           clahe_arg, args.limit)
                all_rows.extend(rows)
                summarize_rows(rows, f"flight {flight}")
        else:
            ctx = multiprocessing.get_context("spawn")
            worker_args = [(g, i, args.pretrained, args.dist, args.num_matches, viz_dir,
                            args.limit, clahe_arg)
                           for i, g in enumerate(groups)]
            with ctx.Pool(len(groups)) as pool:
                results = pool.map(_worker, worker_args)
            all_rows = [r for chunk in results for r in chunk]
            for flight in flights:
                summarize_rows([r for r in all_rows if r.get("flight") == flight],
                               f"flight {flight}")

        out = pd.DataFrame(all_rows)
        out.to_csv(OUT_CSV, index=False)
        if len(flights) > 1:
            print(f"\n=== Overall ({len(flights)} flights) ===")
            summarize_rows(all_rows, OUT_CSV)
            if "07" in flights:
                print(f"\n=== Overall (without flight 07) ===")
                summarize_rows([r for r in all_rows if r.get("flight") != "07"], OUT_CSV)


if __name__ == "__main__":
    main()
