import argparse
import math
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

FLIGHT    = "03"
BASE      = f"UAV_Visloc_example/{FLIGHT}"
SAT_TIF   = f"{BASE}/satellite{FLIGHT}.tif"
DRONE_DIR = f"{BASE}/drone"
DRONE_CSV = f"{BASE}/{FLIGHT}.csv"
SAT_CSV   = "UAV_Visloc_example/satellite_ coordinates_range.csv"
OUT_CSV   = "visloc_sift_results.csv"
VIZ_DIR   = "visloc_visualizations"

LOWE           = 0.75   # Lowe's ratio test threshold
MIN_INL        = 10     # minimum inliers required to attempt GPS prediction
CROP           = 2048   # base crop size on the satellite tile (pixels)
SZ             = 1024   # working resolution for both satellite crops and drone images
SCALES         = [0.5, 0.75, 1.0, 1.25, 1.5]
RANSAC_THRESH  = 5.0    # RANSAC reprojection error threshold (pixels)
FLANN_TREES    = 5      # number of KD-trees for FLANN index
FLANN_CHECKS   = 50     # number of leaf checks during FLANN search
TOP_MATCHES    = 50     # max matches drawn in visualizations
JPEG_QUALITY   = 85     # output quality for saved visualization images


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    # Clamp to [0, 1] to guard against floating-point values slightly outside the valid asin domain
    return R * 2 * math.asin(math.sqrt(max(0.0, min(a, 1.0))))


def load_satellite():
    print(f"Loading {SAT_TIF} ... ", end="", flush=True)
    Image.MAX_IMAGE_PIXELS = None  # satellite TIFs exceed PIL's default decompression-bomb limit
    img = cv2.cvtColor(np.array(Image.open(SAT_TIF).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    print(f"{w}x{h} px")
    m = pd.read_csv(SAT_CSV).iloc[0]
    geo = dict(lt_lat=m["LT_lat_map"], lt_lon=m["LT_lon_map"],
               rb_lat=m["RB_lat_map"], rb_lon=m["RB_lon_map"], w=w, h=h)
    geo["pplat"] = h / (geo["lt_lat"] - geo["rb_lat"])
    geo["pplon"] = w / (geo["rb_lon"] - geo["lt_lon"])
    return img, geo


def gps_to_px(lat, lon, g):
    return int((lon - g["lt_lon"]) * g["pplon"]), int((g["lt_lat"] - lat) * g["pplat"])


def crop_sat(sat, cx, cy, g, sz):
    if not (0 <= cx < g["w"] and 0 <= cy < g["h"]):
        return None
    half = sz // 2
    x0, y0 = cx - half, cy - half
    xc, yc = max(0, x0), max(0, y0)
    patch = sat[yc:min(g["h"], y0+sz), xc:min(g["w"], x0+sz)]
    if patch.shape[0] != sz or patch.shape[1] != sz:
        patch = cv2.copyMakeBorder(patch, yc-y0, sz-patch.shape[0]-(yc-y0),
                                   xc-x0, sz-patch.shape[1]-(xc-x0), cv2.BORDER_REFLECT)
    return cv2.resize(patch, (SZ, SZ))


def run_match(sg, dg, detector, method, clahe=None, rootsift=False):
    if clahe is not None:
        sg, dg = clahe.apply(sg), clahe.apply(dg)
    kps, ds = detector.detectAndCompute(sg, None)
    kpd, dd = detector.detectAndCompute(dg, None)
    if rootsift and ds is not None and dd is not None:
        for d in (ds, dd):
            d /= d.sum(axis=1, keepdims=True) + 1e-7  # L1-normalize in place
            np.sqrt(d, out=d)                          # element-wise sqrt in place
    r = dict(sat_kp=len(kps), drone_kp=len(kpd), raw=0, good=0,
             inliers=0, H=None, _kps=kps, _kpd=kpd, _matches=[])
    if ds is None or dd is None or len(kps) < 4 or len(kpd) < 4:
        return r
    matcher = (cv2.FlannBasedMatcher({"algorithm": 1, "trees": FLANN_TREES}, {"checks": FLANN_CHECKS})
               if method == "sift" else cv2.BFMatcher(cv2.NORM_HAMMING))
    matches = matcher.knnMatch(dd, ds, k=2)
    good = [m for m, n in matches if m.distance < LOWE * n.distance]
    r["raw"], r["good"], r["_matches"] = len(matches), len(good), good
    if len(good) >= 4:
        src = np.float32([kpd[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
        if H is not None and mask is not None:
            r["inliers"], r["H"] = int(mask.sum()), H
    return r


def pred_offset_m(H, cx, cy, crop_sz, geo, lat, lon):
    """Return (offset_m, pred_lat, pred_lon) from the homography-predicted position,
    or None if H is None."""
    if H is None:
        return None
    px_crop, py_crop = cv2.perspectiveTransform(
        np.float32([[SZ/2, SZ/2]]).reshape(-1, 1, 2), H).reshape(2)
    scale = crop_sz / SZ
    px_sat = (cx - crop_sz/2) + px_crop * scale
    py_sat = (cy - crop_sz/2) + py_crop * scale
    pred_lat = geo["lt_lat"] - py_sat / geo["pplat"]
    pred_lon = geo["lt_lon"] + px_sat / geo["pplon"]
    return haversine_m(lat, lon, pred_lat, pred_lon), pred_lat, pred_lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",    type=int,   default=400)
    ap.add_argument("--dist",     type=float, default=25.0,  help="Success radius in metres")
    ap.add_argument("--method",   choices=["sift", "orb", "brisk"], default="sift")
    ap.add_argument("--clahe",    action="store_true", help="CLAHE contrast enhancement")
    ap.add_argument("--rootsift", action="store_true", help="RootSIFT normalisation (SIFT only)")
    ap.add_argument("--visualize",action="store_true")
    args = ap.parse_args()

    if args.rootsift and args.method != "sift":
        print(f"  WARNING: --rootsift ignored with --method {args.method}")

    detectors = {
        "sift":  lambda: cv2.SIFT_create(),
        "orb":   lambda: cv2.ORB_create(5000),
        "brisk": lambda: cv2.BRISK_create(),
    }
    detector = detectors[args.method]()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    sat, geo = load_satellite()
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    flags = " | ".join(f for f, on in [("CLAHE", args.clahe),
                                        ("RootSIFT", args.rootsift and args.method == "sift")] if on) or "none"
    print(f"  Method: {args.method.upper()} | Preprocessing: {flags} | Dist: {args.dist}m | {len(df)} images\n")

    if args.visualize:
        os.makedirs(VIZ_DIR, exist_ok=True)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        f, lat, lon = row["filename"], float(row["lat"]), float(row["lon"])
        drone = cv2.imread(os.path.join(DRONE_DIR, f))
        if drone is None:
            rows.append(dict(filename=f, skipped=True))
            continue
        dg = cv2.cvtColor(cv2.resize(drone, (SZ, SZ)), cv2.COLOR_BGR2GRAY)
        cx, cy = gps_to_px(lat, lon, geo)

        best, best_sz, patch = None, None, None
        for s in SCALES:
            crop_sz = max(256, int(CROP * s))
            p = crop_sat(sat, cx, cy, geo, crop_sz)
            if p is None:
                continue
            r = run_match(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), dg, detector,
                          args.method, clahe=clahe, rootsift=args.rootsift)
            if best is None or r["inliers"] > best["inliers"]:
                best, best_sz, patch = r, crop_sz, p

        if best is None:
            rows.append(dict(filename=f, skipped=True))
            continue

        r = best
        off = pred_offset_m(r["H"], cx, cy, best_sz, geo, lat, lon) if r["inliers"] >= MIN_INL else None
        off_m, plat, plon = off if off else (None, None, None)
        success = off_m is not None and off_m <= args.dist

        rows.append(dict(filename=f, lat=lat, lon=lon, height=float(row["height"]),
                         skipped=False, sat_kp=r["sat_kp"], drone_kp=r["drone_kp"],
                         raw=r["raw"], good=r["good"], inliers=r["inliers"],
                         inlier_ratio=round(r["inliers"]/r["good"], 4) if r["good"] else 0,
                         pred_lat=round(plat, 7) if plat else None,
                         pred_lon=round(plon, 7) if plon else None,
                         offset_m=round(off_m, 2) if off_m else None,
                         success=success))

        if args.visualize and r["_matches"]:
            top = sorted(r["_matches"], key=lambda m: m.distance)[:TOP_MATCHES]
            viz = cv2.drawMatches(cv2.cvtColor(cv2.resize(drone, (SZ, SZ)), cv2.COLOR_BGR2GRAY), r["_kpd"],
                                  cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), r["_kps"],
                                  top, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            stem = os.path.splitext(f)[0]
            cv2.imwrite(os.path.join(VIZ_DIR, f"{stem}_matches.jpg"), viz,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    if out.empty or "skipped" not in out.columns:
        print("\n  No images processed."); return
    v = out[~out["skipped"].fillna(False)]
    if v.empty:
        print("\n  All images skipped."); return

    n = len(v)
    with_H      = v[v["inliers"] >= MIN_INL]
    succeeded   = v[v["success"].fillna(False)]
    s, h        = len(succeeded), len(with_H)
    fp          = with_H[~with_H["success"].fillna(False)]

    print(f"\n  Success (≤{args.dist}m):    {s}/{n} ({100*s/n:.1f}%)")
    print(f"  Homography found:       {h}/{n} ({100*h/n:.1f}%)")
    if h:
        print(f"  Incorrect matches:      {len(fp)}/{h} ({100*len(fp)/h:.1f}%) — offset > {args.dist}m")
    if s:
        print(f"  Offset (successes):     mean {succeeded['offset_m'].mean():.1f}m  "
              f"median {succeeded['offset_m'].median():.1f}m  max {succeeded['offset_m'].max():.1f}m")
    print(f"  Median inliers: {v['inliers'].median():.0f} | ratio: {v['inlier_ratio'].median():.3f}")


if __name__ == "__main__":
    main()
