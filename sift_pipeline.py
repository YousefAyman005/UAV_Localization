import argparse, math, os
import cv2, numpy as np, pandas as pd
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

LOWE      = 0.75
MIN_INL   = 10
CROP      = 2048
SZ        = 1024
SCALES    = [0.5, 0.75, 1.0, 1.25, 1.5]


def load_satellite():
    print(f"Loading {SAT_TIF} ... ", end="", flush=True)
    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open(SAT_TIF).convert("RGB"))
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


def crop_sat(sat, cx, cy, g, sz=CROP):
    if not (0 <= cx < g["w"] and 0 <= cy < g["h"]):
        return None
    half = sz // 2
    x0, y0 = cx - half, cy - half
    xc, yc = max(0, x0), max(0, y0)
    patch = sat[yc:min(g["h"], y0 + sz), xc:min(g["w"], x0 + sz)]
    if patch.shape[0] != sz or patch.shape[1] != sz:
        patch = cv2.copyMakeBorder(patch, yc - y0, sz - patch.shape[0] - (yc - y0),
                                   xc - x0, sz - patch.shape[1] - (xc - x0), cv2.BORDER_REFLECT)
    return cv2.resize(patch, (SZ, SZ))


def sift_match(sat_img, drone_img, preprocess=False):
    sg = cv2.cvtColor(sat_img, cv2.COLOR_BGR2GRAY)
    dg = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
    if preprocess:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        sg, dg = clahe.apply(sg), clahe.apply(dg)

    sift = cv2.SIFT_create()
    kps, ds = sift.detectAndCompute(sg, None)
    kpd, dd = sift.detectAndCompute(dg, None)

    if preprocess and ds is not None and dd is not None:
        for d in (ds, dd):
            d /= d.sum(axis=1, keepdims=True) + 1e-7
            np.sqrt(d, out=d)

    r = dict(sat_kp=len(kps), drone_kp=len(kpd), raw=0, good=0,
             inliers=0, H=None, _kps=kps, _kpd=kpd, _matches=[])
    if ds is None or dd is None or len(kps) < 4 or len(kpd) < 4:
        return r

    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5},
                                    {"checks": 50}).knnMatch(dd, ds, k=2)
    good = [m for m, n in matches if m.distance < LOWE * n.distance]
    r["raw"], r["good"], r["_matches"] = len(matches), len(good), good

    if len(good) >= 4:
        src = np.float32([kpd[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None and mask is not None:
            r["inliers"], r["H"] = int(mask.sum()), H
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--preprocess", action="store_true", help="CLAHE + RootSIFT")
    args = ap.parse_args()

    sat, geo = load_satellite()
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    print(f"  Crop {CROP}px | Preprocess {'ON' if args.preprocess else 'OFF'} | {len(df)} images\n")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        f, lat, lon = row["filename"], float(row["lat"]), float(row["lon"])
        drone = cv2.imread(os.path.join(DRONE_DIR, f))
        if drone is None:
            rows.append(dict(filename=f, skipped=True))
            continue
        drone = cv2.resize(drone, (SZ, SZ))
        cx, cy = gps_to_px(lat, lon, geo)
        # Multi-scale: try each scale, keep best by inlier count
        best = None
        for s in SCALES:
            p = crop_sat(sat, cx, cy, geo, max(256, int(CROP * s)))
            if p is None:
                continue
            r = sift_match(p, drone, args.preprocess)
            if best is None or r["inliers"] > best["inliers"]:
                best, patch = r, p
        if best is None:
            rows.append(dict(filename=f, skipped=True))
            continue
        r = best
        ok = r["H"] is not None and r["inliers"] >= MIN_INL

        rows.append(dict(filename=f, lat=lat, lon=lon, height=float(row["height"]),
                         skipped=False, sat_kp=r["sat_kp"], drone_kp=r["drone_kp"],
                         raw=r["raw"], good=r["good"], inliers=r["inliers"],
                         inlier_ratio=round(r["inliers"] / r["good"], 4) if r["good"] else 0,
                         success=ok))

        if args.visualize and r["_matches"]:
            os.makedirs(VIZ_DIR, exist_ok=True)
            top = sorted(r["_matches"], key=lambda m: m.distance)[:50]
            viz = cv2.drawMatches(cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY), r["_kpd"],
                                  cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), r["_kps"],
                                  top, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(f"{VIZ_DIR}/{f.replace('.JPG', '_matches.jpg')}",
                        viz, [cv2.IMWRITE_JPEG_QUALITY, 85])

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    v = out[~out["skipped"].fillna(False)]
    s = v["success"].sum()
    print(f"\n  {s}/{len(v)} ({100*s/len(v):.1f}%) matched | median inliers {v['inliers'].median():.0f}"
          f" | ratio {v['inlier_ratio'].median():.3f}")


if __name__ == "__main__":
    main()
