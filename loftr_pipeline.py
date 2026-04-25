import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from visloc_utils import (
    RANSAC_THRESH,
    FLIGHTS_AVAILABLE, load_flight, collect_pipeline_rows_multitile,
    print_summary, save_dense_viz, TeeLogger,
)
from kornia.feature import LoFTR

OUT_CSV = "visloc_loftr_results.csv"
VIZ_DIR = "visloc_loftr_visualizations"


def img_to_tensor(bgr, device):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(gray).float().div(255.).unsqueeze(0).unsqueeze(0).to(device)


def match_loftr(drone_t, sat_t, matcher, conf_thresh):
    with torch.inference_mode():
        out = matcher({"image0": drone_t, "image1": sat_t})
    kp0  = out["keypoints0"].cpu().numpy()
    kp1  = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=len(kp0), good=0, inliers=0,
             H=None, _kp0=kp0, _kp1=kp1, _conf=conf, _mask=None)
    mask = conf >= conf_thresh
    r["good"], r["_mask"] = int(mask.sum()), mask
    if r["good"] < 4:
        return r
    H, mh = cv2.findHomography(kp0[mask].reshape(-1, 1, 2).astype(np.float32),
                               kp1[mask].reshape(-1, 1, 2).astype(np.float32),
                               cv2.USAC_MAGSAC, RANSAC_THRESH,
                               maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist",       type=float, default=25.0)
    ap.add_argument("--pretrained", choices=["outdoor", "indoor"], default="outdoor")
    ap.add_argument("--visualize",  action="store_true")
    ap.add_argument("--flights",    nargs="+", default=["all"],
                    help="Flight IDs to evaluate, e.g. 01 03 05, or 'all' (default)")
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Loading LoFTR ({args.pretrained}) ... ", end="", flush=True)
    matcher = LoFTR(pretrained=args.pretrained).eval().to(device)
    print("done")
    print(f"  Method: LoFTR ({args.pretrained}) | Dist: {args.dist}m | Flights: {' '.join(flights)}")

    def match_factory(drone):
        drone_t = img_to_tensor(drone, device)
        return lambda p: match_loftr(drone_t, img_to_tensor(p, device), matcher, 0.0)

    log_path = OUT_CSV.replace(".csv", ".log")
    with TeeLogger(log_path):
        all_rows = []
        for flight in flights:
            tiles, drone_dir, drone_csv, _ = load_flight(flight)
            df = pd.read_csv(drone_csv)
            print(f"\n=== Flight {flight}: {len(df)} images ===")

            rows = collect_pipeline_rows_multitile(tiles, df, match_factory, args.dist,
                                                    drone_dir=drone_dir, flight=flight,
                                                    viz_fn=save_dense_viz if args.visualize else None,
                                                    viz_dir=VIZ_DIR if args.visualize else None)
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
    main()
