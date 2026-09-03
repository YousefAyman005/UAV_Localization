"""Diagnostic: residual drone->patch rotation under the pipeline's yaw handling.

For sampled frames of one flight, builds (drone, metric crop) exactly as the
benchmark does, runs the SIFT baseline matcher + the shared 4-DOF
fit_similarity, and reports the rotation angle the similarity recovers.
If Phi1 truly rotates the patch heading-up, the residual angle is ~0.
Run for yaw = +Phi1 (pipeline) and yaw = -Phi1 (sign-flip hypothesis).

Usage:
    python analyze/check_crop_rotation.py --flight 08 --n 24
"""
import argparse
import json
import math
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from helpers import utils as U  # noqa: E402

LOWE, FLANN_TREES, FLANN_CHECKS = 0.75, 5, 50  # baseline_pipeline.py values


def circ_dist(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circ_mean(angles):
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(s, c))


def cluster_legs(rows, thresh=45.0, min_members=3):
    """Greedy circular clustering of (heading, residual) rows into flight legs.
    Returns [[leg_heading, median_residual], ...] for stable legs."""
    clusters = []  # each: list of (heading, residual) members
    for h, t in rows:
        best = next((cl for cl in clusters
                     if circ_dist(h, circ_mean([m[0] for m in cl])) < thresh),
                    None)
        if best is None:
            clusters.append([(h, t)])
        else:
            best.append((h, t))
    legs = []
    for cl in clusters:
        if len(cl) < min_members:
            continue
        legs.append([round(circ_mean([m[0] for m in cl]), 1),
                     round(float(np.median([m[1] for m in cl])), 1)])
    return legs


def sift_similarity(drone_bgr, patch_bgr, sift, flann):
    kpd, dd = sift.detectAndCompute(cv2.cvtColor(drone_bgr, cv2.COLOR_BGR2GRAY), None)
    kps, ds = sift.detectAndCompute(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY), None)
    if dd is None or ds is None or len(kpd) < 4 or len(kps) < 4:
        return None, 0
    pairs = flann.knnMatch(dd, ds, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < LOWE * n.distance]
    if len(good) < 4:
        return None, 0
    src = np.float32([kpd[m.queryIdx].pt for m in good])
    dst = np.float32([kps[m.trainIdx].pt for m in good])
    return U.fit_similarity(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", nargs="+", default=["08"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit CSV row indices (overrides --n sampling)")
    ap.add_argument("--matcher", choices=["sift", "roma"], default="sift")
    ap.add_argument("--yaw-source", choices=["phi1", "track", "cal"],
                    default="phi1",
                    help="track: compass bearing of GPS motion (central "
                         "difference); cal: Phi1 + the calibrated YAW_OFFSET "
                         "(validation of helpers.utils.corrected_yaw)")
    ap.add_argument("--both-signs", action="store_true",
                    help="also measure yaw=-Phi1 (sign-flip hypothesis); "
                         "default measures only the pipeline's +yaw")
    ap.add_argument("--extre-weights", default="/data/weights/roma_extre.pth")
    ap.add_argument("--calibrate", metavar="OUT_JSON", default=None,
                    help="cluster frames into flight legs (45-deg circular "
                         "threshold on yaw, clusters with <3 gated frames "
                         "dropped) and write {flight: [[leg_heading, "
                         "median_residual_deg], ...]} for helpers YAW_OFFSET")
    args = ap.parse_args()

    clahe = U._make_clahe(True)
    if args.matcher == "roma":
        import torch
        from pipelines import roma_pipeline as RP
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rargs = argparse.Namespace(pretrained="extre",
                                   extre_weights=args.extre_weights,
                                   num_matches=5000)
        model = RP.load_model(device, rargs)
        factory = RP.make_match_factory(model, device, rargs)
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": FLANN_TREES},
                                  {"checks": FLANN_CHECKS})

    signs = ("+", "-") if args.both_signs else ("+",)
    calib = {}
    for flight in args.flight:
        tiles, drone_dir, drone_csv, _ = U.load_flight(flight)
        df = pd.read_csv(drone_csv)
        idxs = (np.asarray(args.indices) if args.indices is not None
                else np.linspace(0, len(df) - 1, args.n).round().astype(int))
        print(f"flight {flight} (yaw={args.yaw_source}): residual rotation of "
              f"the 4-DOF similarity (deg, + = CCW), gate inl>={U.MIN_INL}",
              flush=True)
        print(f"{'frame':>12} {'yaw':>7} | " + " | ".join(
            f"{s}yaw: {'inl':>6} {'theta':>7}" for s in signs))
        stats = {s: [] for s in signs}
        cal_rows = []
        for i in idxs:
            row = df.iloc[i]
            f, lat, lon = row["filename"], float(row["lat"]), float(row["lon"])
            height, phi1 = float(row["height"]), float(row["Phi1"])
            if args.yaw_source == "cal":
                phi1 = U.corrected_yaw(flight, phi1)
            elif args.yaw_source == "track":
                j0, j1 = max(0, i - 1), min(len(df) - 1, i + 1)
                d_north = float(df["lat"].iloc[j1]) - float(df["lat"].iloc[j0])
                d_east = ((float(df["lon"].iloc[j1]) - float(df["lon"].iloc[j0]))
                          * math.cos(math.radians(lat)))
                phi1 = math.degrees(math.atan2(d_east, d_north))
            drone = cv2.imread(os.path.join(drone_dir, f))
            if drone is None:
                continue
            drone = clahe(cv2.resize(drone, (U.SZ_W, U.SZ_H),
                                     interpolation=cv2.INTER_AREA))
            sat, geo, cx, cy, inb = U.tile_for_gps(tiles, lat, lon)
            if not inb:
                continue
            seed = zlib.crc32(f"{flight}/{f}".encode())
            dx_m, dy_m = np.random.default_rng(seed).normal(
                0.0, U.PRIOR_OFFSET_STD_M, 2)
            mid_lat = (geo["lt_lat"] + geo["rb_lat"]) / 2
            cx += dx_m * geo["pplon"] / (math.cos(math.radians(mid_lat))
                                         * U.DEG_TO_M)
            cy += dy_m * geo["pplat"] / U.DEG_TO_M

            out = {}
            for sign in signs:
                yaw = phi1 if sign == "+" else -phi1
                patch, _ = U.metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw,
                                         flight=flight)
                if patch is None:
                    out[sign] = (0, float("nan"))
                    continue
                if args.matcher == "roma":
                    r = factory(drone)(clahe(patch))
                    H, inl = r["H"], r["inliers"]
                else:
                    H, inl = sift_similarity(drone, clahe(patch), sift, flann)
                theta = (math.degrees(math.atan2(H[1, 0], H[0, 0]))
                         if H is not None else float("nan"))
                out[sign] = (inl, theta)
                if inl >= U.MIN_INL:
                    stats[sign].append(theta)
                    if sign == "+" and not math.isnan(theta):
                        cal_rows.append((phi1, theta))
            print(f"{f:>12} {phi1:7.1f} | " + " | ".join(
                f"{out[s][0]:11d} {out[s][1]:7.1f}" for s in signs), flush=True)

        for sign in signs:
            v = np.array(stats[sign])
            if len(v):
                print(f"flight {flight} yaw={sign}{args.yaw_source}: {len(v)} "
                      f"gated frames, median theta {np.median(v):+.1f} deg, "
                      f"IQR [{np.percentile(v, 25):+.1f}, "
                      f"{np.percentile(v, 75):+.1f}]")
            else:
                print(f"flight {flight} yaw={sign}{args.yaw_source}: "
                      f"no frames passed the gate")
        if args.calibrate:
            calib[flight] = cluster_legs(cal_rows)
            print(f"flight {flight} legs: {calib[flight]}", flush=True)
        del tiles

    if args.calibrate:
        with open(args.calibrate, "w") as fh:
            json.dump(calib, fh, indent=1)
        print("wrote", args.calibrate)



if __name__ == "__main__":
    main()
