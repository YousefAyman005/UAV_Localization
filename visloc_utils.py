import math
import cv2
import numpy as np
import pandas as pd
from PIL import Image

MIN_INL       = 10      # minimum RANSAC inliers required to attempt GPS prediction
CROP_W        = 2048    # base satellite crop width  (pixels, on the full satellite tile)
CROP_H        = 1360    # must equal CROP_W * SZ_H/SZ_W — keeps scale_x==scale_y in pred_offset_m
SZ_W          = 1024    # working width  (pixels) — preserves native 3:2 UAV aspect ratio
SZ_H          = 680     # working height (pixels)
SCALES        = [0.5, 0.75, 1.0, 1.25, 1.5]
RANSAC_THRESH = 5.0     # RANSAC reprojection error threshold (pixels)
TOP_MATCHES   = 50      # max matches drawn in visualizations
JPEG_QUALITY  = 85      # output quality for saved visualization images

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
    geo = dict(lt_lat=m["LT_lat_map"], lt_lon=m["LT_lon_map"],
               rb_lat=m["RB_lat_map"], rb_lon=m["RB_lon_map"], w=w, h=h)
    geo["pplat"] = h / (geo["lt_lat"] - geo["rb_lat"])
    geo["pplon"] = w / (geo["rb_lon"] - geo["lt_lon"])
    return img, geo


def gps_to_px(lat, lon, g):
    return int((lon - g["lt_lon"]) * g["pplon"]), int((g["lt_lat"] - lat) * g["pplat"])


def crop_sat(sat, cx, cy, g, crop_w, crop_h):
    """Crop a crop_w×crop_h rectangle from sat centred at (cx,cy), resize to SZ_W×SZ_H."""
    if not (0 <= cx < g["w"] and 0 <= cy < g["h"]):
        return None
    half_w, half_h = crop_w // 2, crop_h // 2
    x0, y0 = cx - half_w, cy - half_h
    xc, yc = max(0, x0), max(0, y0)
    patch = sat[yc:min(g["h"], y0 + crop_h), xc:min(g["w"], x0 + crop_w)]
    ph, pw = patch.shape[:2]
    if ph != crop_h or pw != crop_w:
        patch = cv2.copyMakeBorder(patch,
                                   yc - y0,           crop_h - ph - (yc - y0),
                                   xc - x0,           crop_w - pw - (xc - x0),
                                   cv2.BORDER_REFLECT)
    return cv2.resize(patch, (SZ_W, SZ_H))


def pred_offset_m(H, cx, cy, crop_w, crop_h, geo, lat, lon):
    """Return (offset_m, pred_lat, pred_lon) from the homography, or None if H is None."""
    if H is None:
        return None
    px_crop, py_crop = cv2.perspectiveTransform(
        np.float32([[SZ_W / 2, SZ_H / 2]]).reshape(-1, 1, 2), H).reshape(2)
    scale_x = crop_w / SZ_W
    scale_y = crop_h / SZ_H
    px_sat = (cx - crop_w / 2) + px_crop * scale_x
    py_sat = (cy - crop_h / 2) + py_crop * scale_y
    pred_lat = geo["lt_lat"] - py_sat / geo["pplat"]
    pred_lon = geo["lt_lon"] + px_sat / geo["pplon"]
    return haversine_m(lat, lon, pred_lat, pred_lon), pred_lat, pred_lon


def altitude_scales(height_m, geo):
    """Return SCALES sorted by proximity to the altitude-predicted satellite footprint."""
    lat_mid = (geo["lt_lat"] + geo["rb_lat"]) / 2
    m_per_px = math.cos(math.radians(lat_mid)) * 111_320 / geo["pplon"]
    footprint_px = 2 * height_m * math.tan(math.radians(_UAV_HFOV_DEG / 2)) / m_per_px
    target_s = footprint_px / CROP_W
    return sorted(SCALES, key=lambda s: abs(s - target_s))


def print_summary(v, dist, out_csv, min_inl=None):
    """Print standard success/offset summary stats from a results DataFrame."""
    if min_inl is None:
        min_inl = MIN_INL
    if v.empty:
        print("\n  All images skipped."); return
    n = len(v)
    with_H    = v[v["inliers"] >= min_inl]
    succeeded = v[v["success"].fillna(False)]
    s, h      = len(succeeded), len(with_H)
    fp        = with_H[~with_H["success"].fillna(False)]
    print(f"\n  Results saved to {out_csv}")
    print(f"  Success (≤{dist}m):    {s}/{n} ({100*s/n:.1f}%)")
    print(f"  Homography found:       {h}/{n} ({100*h/n:.1f}%)")
    if h:
        print(f"  Incorrect matches:      {len(fp)}/{h} ({100*len(fp)/h:.1f}%) — offset > {dist}m")
    if s:
        print(f"  Offset (successes):     mean {succeeded['offset_m'].mean():.1f}m  "
              f"median {succeeded['offset_m'].median():.1f}m  max {succeeded['offset_m'].max():.1f}m")
    print(f"  Median inliers: {v['inliers'].median():.0f} | ratio: {v['inlier_ratio'].median():.3f}")
