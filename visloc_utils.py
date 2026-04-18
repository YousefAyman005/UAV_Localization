import math
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

MIN_INL       = 10      # minimum RANSAC inliers to attempt GPS prediction
CROP_W        = 2048    # base satellite crop width (px on the full sat tile)
SZ_W          = 1024    # working width (preserves native 3:2 UAV aspect)
SZ_H          = 680
CROP_H        = CROP_W * SZ_H // SZ_W   # = 1360; keeps scale_x == scale_y
SCALES        = [0.5, 0.75, 1.0, 1.25, 1.5]
RANSAC_THRESH = 5.0
TOP_MATCHES   = 50
JPEG_QUALITY  = 85

# Default UAV_VisLoc dataset paths (flight 03).
FLIGHT    = "03"
_HERE     = os.path.dirname(os.path.abspath(__file__))
BASE      = os.path.join(_HERE, f"UAV_Visloc_example/{FLIGHT}")
SAT_TIF   = os.path.join(BASE, f"satellite{FLIGHT}.tif")
DRONE_DIR = os.path.join(BASE, "drone")
DRONE_CSV = os.path.join(BASE, f"{FLIGHT}.csv")
SAT_CSV   = os.path.join(_HERE, "UAV_Visloc_example/satellite_ coordinates_range.csv")

_UAV_HFOV_DEG = 70.0  # approximate UAV horizontal FOV for altitude-based scale prior


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(max(0.0, min(a, 1.0))))


def load_satellite(sat_tif, sat_csv):
    print(f"Loading {sat_tif} ... ", end="", flush=True)
    Image.MAX_IMAGE_PIXELS = None
    img = cv2.cvtColor(np.array(Image.open(sat_tif).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    print(f"{w}x{h} px")
    m = pd.read_csv(sat_csv).iloc[0]
    lt_lat, lt_lon = m["LT_lat_map"], m["LT_lon_map"]
    rb_lat, rb_lon = m["RB_lat_map"], m["RB_lon_map"]
    return img, dict(lt_lat=lt_lat, lt_lon=lt_lon, rb_lat=rb_lat, rb_lon=rb_lon, w=w, h=h,
                     pplat=h / (lt_lat - rb_lat), pplon=w / (rb_lon - lt_lon))


def gps_to_px(lat, lon, g):
    return int((lon - g["lt_lon"]) * g["pplon"]), int((g["lt_lat"] - lat) * g["pplat"])


def crop_sat(sat, cx, cy, g, crop_w, crop_h):
    """Crop crop_w×crop_h centred at (cx,cy) and resize to SZ_W×SZ_H."""
    if not (0 <= cx < g["w"] and 0 <= cy < g["h"]):
        return None
    x0, y0 = cx - crop_w // 2, cy - crop_h // 2
    xc, yc = max(0, x0), max(0, y0)
    patch = sat[yc:min(g["h"], y0 + crop_h), xc:min(g["w"], x0 + crop_w)]
    ph, pw = patch.shape[:2]
    if ph != crop_h or pw != crop_w:
        patch = cv2.copyMakeBorder(patch,
                                   yc - y0, crop_h - ph - (yc - y0),
                                   xc - x0, crop_w - pw - (xc - x0),
                                   cv2.BORDER_REFLECT)
    return cv2.resize(patch, (SZ_W, SZ_H))


def pred_gps(H, cx, cy, crop_w, crop_h, geo):
    """Return (pred_lat, pred_lon) from homography, or None."""
    if H is None:
        return None
    px_crop, py_crop = cv2.perspectiveTransform(
        np.float32([[SZ_W / 2, SZ_H / 2]]).reshape(-1, 1, 2), H).reshape(2)
    px_sat = (cx - crop_w / 2) + px_crop * (crop_w / SZ_W)
    py_sat = (cy - crop_h / 2) + py_crop * (crop_h / SZ_H)
    return geo["lt_lat"] - py_sat / geo["pplat"], geo["lt_lon"] + px_sat / geo["pplon"]


def pred_offset_m(H, cx, cy, crop_w, crop_h, geo, lat, lon):
    """Return (offset_m, pred_lat, pred_lon) from homography, or None."""
    gps = pred_gps(H, cx, cy, crop_w, crop_h, geo)
    if gps is None:
        return None
    return haversine_m(lat, lon, *gps), *gps


def altitude_scales(height_m, geo):
    """SCALES sorted by proximity to the altitude-predicted footprint (best first)."""
    lat_mid = (geo["lt_lat"] + geo["rb_lat"]) / 2
    m_per_px = math.cos(math.radians(lat_mid)) * 111_320 / geo["pplon"]
    target_s = 2 * height_m * math.tan(math.radians(_UAV_HFOV_DEG / 2)) / m_per_px / CROP_W
    return sorted(SCALES, key=lambda s: abs(s - target_s))


def scale_sweep(sat, cx, cy, geo, height_m, match_fn, clahe_fn=None, early_stop_inliers=None):
    """Iterate altitude-prioritised scales; return (best_r, (crop_w, crop_h), patch)."""
    best, best_crop, best_patch = None, None, None
    for s in altitude_scales(height_m, geo):
        crop_w = max(SZ_W, int(CROP_W * s))
        crop_h = max(SZ_H, int(CROP_H * s))
        p = crop_sat(sat, cx, cy, geo, crop_w, crop_h)
        if p is None:
            continue
        if clahe_fn is not None:
            p = clahe_fn(p)
        r = match_fn(p)
        if best is None or r["inliers"] > best["inliers"]:
            best, best_crop, best_patch = r, (crop_w, crop_h), p
            if early_stop_inliers is not None and best["inliers"] >= early_stop_inliers:
                break
    return best, best_crop, best_patch


def apply_clahe_lab(bgr, clahe):
    """Apply CLAHE to the L channel of bgr in LAB space."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def make_result_row(filename, lat, lon, height, r, best_crop, off_m, plat, plon, success):
    return dict(
        filename=filename, lat=lat, lon=lon, height=height,
        skipped=False, crop_w=best_crop[0], crop_h=best_crop[1],
        sat_kp=r["sat_kp"], drone_kp=r["drone_kp"],
        raw=r["raw"], good=r["good"], inliers=r["inliers"],
        inlier_ratio=round(r["inliers"] / r["good"], 4) if r["good"] else 0,
        pred_lat=round(plat, 7)  if plat  is not None else None,
        pred_lon=round(plon, 7)  if plon  is not None else None,
        offset_m=round(off_m, 2) if off_m is not None else None,
        success=success,
    )


def draw_and_save(drone, kpd, patch, kps, matches, filename, viz_dir):
    """drawMatches + imwrite with the standard JPEG output path."""
    viz = cv2.drawMatches(drone, kpd, patch, kps, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(viz_dir, f"{os.path.splitext(filename)[0]}_matches.jpg"),
                viz, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def save_dense_viz(drone, patch, best, filename, viz_dir):
    """Viz for dense matchers that expose _kp0/_kp1/_conf/_mask (LoFTR, RoMa, MATCHA)."""
    if best.get("_mask") is None or best.get("good", 0) <= 0:
        return
    kp0, kp1, conf, mask = best["_kp0"], best["_kp1"], best["_conf"], best["_mask"]
    kpd = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0[mask]]
    kps = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1[mask]]
    top = sorted([cv2.DMatch(i, i, 1.0 - c) for i, c in enumerate(conf[mask])],
                 key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, kpd, patch, kps, top, filename, viz_dir)


def run_pipeline(sat, geo, df, match_factory, out_csv, dist, min_inl=MIN_INL,
                 clahe=None, viz_fn=None, viz_dir=None, drone_dir=None):
    """Per-image loop: build match_fn via match_factory(drone_bgr), scale_sweep, predict GPS,
    append row, optionally visualize; then save CSV and print the standard summary.
    If `clahe` is given, LAB-CLAHE is applied to both the drone and every sat patch.
    """
    if drone_dir is None:
        drone_dir = DRONE_DIR
    clahe_fn = (lambda p: apply_clahe_lab(p, clahe)) if clahe is not None else None
    if viz_fn is not None and viz_dir is not None:
        os.makedirs(viz_dir, exist_ok=True)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        f = row["filename"]
        lat, lon, height = float(row["lat"]), float(row["lon"]), float(row["height"])
        drone = cv2.imread(os.path.join(drone_dir, f))
        if drone is None:
            rows.append(dict(filename=f, skipped=True)); continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        if clahe_fn is not None:
            drone = clahe_fn(drone)
        cx, cy = gps_to_px(lat, lon, geo)
        match_fn = match_factory(drone)
        best, best_crop, patch = scale_sweep(sat, cx, cy, geo, height, match_fn, clahe_fn=clahe_fn)
        if best is None:
            rows.append(dict(filename=f, skipped=True)); continue
        off = pred_offset_m(best["H"], cx, cy, *best_crop, geo, lat, lon) if best["inliers"] >= min_inl else None
        off_m, plat, plon = off if off else (None, None, None)
        rows.append(make_result_row(f, lat, lon, height, best, best_crop,
                                    off_m, plat, plon, off_m is not None and off_m <= dist))
        if viz_fn is not None:
            viz_fn(drone, patch, best, f, viz_dir)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    if out.empty or "skipped" not in out.columns:
        print("\n  No images processed."); return
    print_summary(out[~out["skipped"].fillna(False)], dist, out_csv, min_inl=min_inl)


def print_summary(v, dist, out_csv, min_inl=None):
    if min_inl is None:
        min_inl = MIN_INL
    if v.empty:
        print("\n  All images skipped."); return
    n = len(v)
    accepted  = v[v["inliers"] >= min_inl]
    succeeded = v[v["success"].fillna(False)]
    s, h      = len(succeeded), len(accepted)
    fp        = accepted[~accepted["success"].fillna(False)]
    print(f"\n  Results saved to {out_csv}")
    print(f"  Success (≤{dist}m):    {s}/{n} ({100*s/n:.1f}%)")
    print(f"  Homography accepted:    {h}/{n} ({100*h/n:.1f}%)")
    if h:
        print(f"  Incorrect matches:      {len(fp)}/{h} ({100*len(fp)/h:.1f}%) — offset > {dist}m")
    if s:
        print(f"  Offset (successes):     mean {succeeded['offset_m'].mean():.1f}m  "
              f"median {succeeded['offset_m'].median():.1f}m  max {succeeded['offset_m'].max():.1f}m")
    print(f"  Median inliers: {v['inliers'].median():.0f} | ratio: {v['inlier_ratio'].median():.3f}")
