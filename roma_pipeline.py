import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from visloc_utils import (
    MIN_INL, RANSAC_THRESH,
    FLIGHTS_AVAILABLE, load_flight, collect_pipeline_rows_multitile,
    print_summary, save_dense_viz, TeeLogger,
)

from romatch import roma_outdoor, roma_indoor

OUT_CSV = "visloc_roma_results.csv"
VIZ_DIR = "visloc_roma_visualizations"


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def match_roma(drone_bgr, sat_bgr, matcher, conf_thresh, ransac_thresh, device, num_samples):
    H_img, W_img = drone_bgr.shape[:2]
    with torch.inference_mode():
        warp, cert  = matcher.match(bgr_to_pil(drone_bgr), bgr_to_pil(sat_bgr), device=device)
        matches, c  = matcher.sample(warp, cert, num=num_samples)
        kp_a, kp_b  = matcher.to_pixel_coordinates(matches, H_img, W_img, H_img, W_img)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist",        type=float, default=25.0)
    ap.add_argument("--pretrained",  choices=["outdoor", "indoor"], default="outdoor")
    ap.add_argument("--num-matches", type=int,   default=5000)
    ap.add_argument("--visualize",   action="store_true")
    ap.add_argument("--flights",     nargs="+", default=["all"],
                    help="Flight IDs to evaluate, e.g. 01 03 05, or 'all' (default)")
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("highest")
    print(f"  Device: {device}")
    print(f"  Loading RoMa ({args.pretrained}) ... ", end="", flush=True)
    kw = {} if device == "cuda" else {"amp_dtype": torch.float32}
    matcher = (roma_outdoor(device=device, **kw) if args.pretrained == "outdoor"
               else roma_indoor(device=device, **kw))
    print("done")

    print(f"  Method: RoMa ({args.pretrained}) | NumMatches: {args.num_matches} | "
          f"RANSAC: {RANSAC_THRESH}px | MinInl: {MIN_INL} | "
          f"Dist: {args.dist}m | Flights: {' '.join(flights)}")

    def match_factory(drone):
        return lambda p: match_roma(drone, p, matcher, 0.0, RANSAC_THRESH, device, args.num_matches)

    log_path = OUT_CSV.replace(".csv", ".log")
    with TeeLogger(log_path):
        all_rows = []
        for flight in flights:
            tiles, drone_dir, drone_csv, _ = load_flight(flight)
            df = pd.read_csv(drone_csv)
            print(f"\n=== Flight {flight}: {len(df)} images ===")

            rows = collect_pipeline_rows_multitile(tiles, df, match_factory, args.dist,
                                                    min_inl=MIN_INL,
                                                    drone_dir=drone_dir, flight=flight,
                                                    viz_fn=save_dense_viz if args.visualize else None,
                                                    viz_dir=VIZ_DIR if args.visualize else None)
            all_rows.extend(rows)

            flight_df = pd.DataFrame(rows)
            valid = flight_df[~flight_df["skipped"].fillna(False)]
            if not valid.empty:
                print_summary(valid, args.dist, f"flight {flight}", min_inl=MIN_INL)

        out = pd.DataFrame(all_rows)
        out.to_csv(OUT_CSV, index=False)
        if len(flights) > 1:
            print(f"\n=== Overall ({len(flights)} flights) ===")
            valid_all = out[~out["skipped"].fillna(False)]
            if not valid_all.empty:
                print_summary(valid_all, args.dist, OUT_CSV, min_inl=MIN_INL)


if __name__ == "__main__":
    main()
