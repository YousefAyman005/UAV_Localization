import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from visloc_utils import (
    MIN_INL, CROP_W, CROP_H, SZ_W, SZ_H, RANSAC_THRESH, TOP_MATCHES, JPEG_QUALITY,
    load_satellite, gps_to_px, crop_sat, pred_offset_m, print_summary, altitude_scales,
)
from romatch import roma_outdoor, roma_indoor

FLIGHT    = "03"
BASE      = f"UAV_Visloc_example/{FLIGHT}"
SAT_TIF   = f"{BASE}/satellite{FLIGHT}.tif"
DRONE_DIR = f"{BASE}/drone"
DRONE_CSV = f"{BASE}/{FLIGHT}.csv"
SAT_CSV   = "UAV_Visloc_example/satellite_ coordinates_range.csv"
OUT_CSV   = "visloc_roma_results.csv"
VIZ_DIR   = "visloc_roma_visualizations"


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def apply_clahe(bgr, clahe):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def match_roma(drone_bgr, sat_bgr, matcher, conf_thresh, ransac_thresh, device, num_samples):
    drone_pil = bgr_to_pil(drone_bgr)
    sat_pil   = bgr_to_pil(sat_bgr)
    H_img, W_img = drone_bgr.shape[:2]  # both are SZ_H x SZ_W after resize

    with torch.inference_mode():
        warp, certainty = matcher.match(drone_pil, sat_pil, device=device)
        matches, cert   = matcher.sample(warp, certainty, num=num_samples)
        kp_a, kp_b      = matcher.to_pixel_coordinates(matches, H_img, W_img, H_img, W_img)

    kp0     = kp_a.cpu().numpy().astype(np.float32)
    kp1     = kp_b.cpu().numpy().astype(np.float32)
    cert_np = cert.cpu().numpy()

    n_raw = len(kp0)
    r = dict(sat_kp=n_raw, drone_kp=n_raw, raw=n_raw, good=0,
             inliers=0, H=None, _kp0=kp0, _kp1=kp1, _conf=cert_np, _mask=None)

    mask = cert_np >= conf_thresh
    r["good"] = int(mask.sum())
    r["_mask"] = mask
    if r["good"] < 4:
        return r

    H, mask_h = cv2.findHomography(
        kp0[mask].reshape(-1, 1, 2),
        kp1[mask].reshape(-1, 1, 2),
        cv2.USAC_MAGSAC, ransac_thresh,
        maxIters=5000, confidence=0.9999,
    )
    if H is not None and mask_h is not None:
        r["inliers"], r["H"] = int(mask_h.sum()), H
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",         type=int,   default=400)
    ap.add_argument("--dist",          type=float, default=25.0,
                    help="GPS success radius (metres)")
    ap.add_argument("--conf",          type=float, default=0.0,
                    help="RoMa certainty threshold [0-1]; 0.0 = no filter (LoFTR default)")
    ap.add_argument("--pretrained",    choices=["outdoor", "indoor"], default="outdoor")
    ap.add_argument("--num-matches",   type=int,   default=5000,
                    help="Correspondence pairs sampled from the dense warp")
    ap.add_argument("--ransac-thresh", type=float, default=None,
                    help="RANSAC reprojection threshold (px); default: visloc_utils.RANSAC_THRESH")
    ap.add_argument("--min-inl",       type=int,   default=None,
                    help="Min inliers for GPS prediction; default: visloc_utils.MIN_INL")
    ap.add_argument("--clahe",         action="store_true")
    ap.add_argument("--visualize",     action="store_true")
    args = ap.parse_args()

    ransac_t = args.ransac_thresh if args.ransac_thresh is not None else RANSAC_THRESH
    min_inl  = args.min_inl       if args.min_inl       is not None else MIN_INL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("highest")  # required by RoMa

    print(f"  Device: {device}")
    print(f"  Loading RoMa ({args.pretrained}) ... ", end="", flush=True)
    roma_kwargs = {} if device == "cuda" else {"amp_dtype": torch.float32}
    matcher = (roma_outdoor(device=device, **roma_kwargs)
               if args.pretrained == "outdoor"
               else roma_indoor(device=device, **roma_kwargs))
    print("done")

    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    print(
        f"  Method: RoMa ({args.pretrained}) | CLAHE: {args.clahe} | Conf: {args.conf} | "
        f"NumMatches: {args.num_matches} | RANSAC: {ransac_t}px | MinInl: {min_inl} | "
        f"Dist: {args.dist}m | {len(df)} images\n"
    )

    if args.visualize:
        os.makedirs(VIZ_DIR, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        f, lat, lon = row["filename"], float(row["lat"]), float(row["lon"])
        drone = cv2.imread(os.path.join(DRONE_DIR, f))
        if drone is None:
            rows.append(dict(filename=f, skipped=True)); continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        if clahe is not None:
            drone = apply_clahe(drone, clahe)
        cx, cy = gps_to_px(lat, lon, geo)

        best, best_crop, patch = None, None, None
        for s in altitude_scales(float(row["height"]), geo):
            crop_w = max(SZ_W, int(CROP_W * s))
            crop_h = max(SZ_H, int(CROP_H * s))
            p = crop_sat(sat, cx, cy, geo, crop_w, crop_h)
            if p is None:
                continue
            if clahe is not None:
                p = apply_clahe(p, clahe)
            r = match_roma(drone, p, matcher, args.conf, ransac_t, device, args.num_matches)
            if best is None or r["inliers"] > best["inliers"]:
                best, best_crop, patch = r, (crop_w, crop_h), p

        if best is None:
            rows.append(dict(filename=f, skipped=True)); continue

        r = best
        off = pred_offset_m(r["H"], cx, cy, *best_crop, geo, lat, lon) if r["inliers"] >= min_inl else None
        off_m, plat, plon = off if off else (None, None, None)
        success = off_m is not None and off_m <= args.dist

        rows.append(dict(
            filename=f, lat=lat, lon=lon, height=float(row["height"]),
            skipped=False, crop_w=best_crop[0], crop_h=best_crop[1],
            sat_kp=r["sat_kp"], drone_kp=r["drone_kp"],
            raw=r["raw"], good=r["good"], inliers=r["inliers"],
            inlier_ratio=round(r["inliers"] / r["good"], 4) if r["good"] else 0,
            pred_lat=round(plat, 7)  if plat  is not None else None,
            pred_lon=round(plon, 7)  if plon  is not None else None,
            offset_m=round(off_m, 2) if off_m is not None else None,
            success=success,
        ))

        if args.visualize and r["_mask"] is not None and r["good"] > 0:
            kp0, kp1, conf, mask = r["_kp0"], r["_kp1"], r["_conf"], r["_mask"]
            kpd_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0[mask]]
            kps_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1[mask]]
            top = sorted(
                [cv2.DMatch(i, i, 1.0 - c) for i, c in enumerate(conf[mask])],
                key=lambda m: m.distance,
            )[:TOP_MATCHES]
            viz = cv2.drawMatches(drone, kpd_cv, patch, kps_cv, top, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(
                os.path.join(VIZ_DIR, f"{os.path.splitext(f)[0]}_matches.jpg"),
                viz, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    if out.empty or "skipped" not in out.columns:
        print("\n  No images processed."); return
    print_summary(out[~out["skipped"].fillna(False)], args.dist, OUT_CSV, min_inl=min_inl)


if __name__ == "__main__":
    main()
