"""Core utilities: constants, geo helpers, satellite loading, metric crop,
CLAHE preprocessing, and the per-flight matching loop.

Visualization, parallel orchestration, and CSV summarization live in their
own modules (visualization.py, workers.py, results.py).
"""

import heapq
import json
import math
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # large satellite TIFs

# Deterministic RNGs for every pipeline that imports this module.
random.seed(0); np.random.seed(0); cv2.setRNGSeed(0)

# ── Constants ───────────────────────────────────────────────────────────────

SZ_W, SZ_H         = 1024, 680
RANSAC_THRESH      = 5.0
MIN_INL            = 7
TOP_MATCHES        = 50
MIN_PATCH_COVERAGE = 0.2     # skip mostly-outside crops; edge samples still evaluate
ACC_THRESHOLDS     = [5, 10, 15, 20, 25]
BEST_N, WORST_N    = 10, 10
JPEG_QUALITY       = 85
EARTH_R_M          = 6_371_000.0
DEG_TO_M           = 111_320.0  # meters per degree latitude (flat-earth approx)

# patch_span_m = K * altitude_m, then m_per_px = patch_span_m / SZ_W.
# Default = SAT_ZOOM(1.75) * 2 * tan(35°) ≈ 2.451, preserves legacy 70°/1.75x behaviour.
K_DEFAULT          = 1.75 * 2.0 * math.tan(math.radians(35.0))

_HERE       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(_HERE), "UAV_VisLoc_dataset")
# Flight 07's satellite is 3000×170 — too narrow for metric_crop; dropped.
FLIGHTS_AVAILABLE = [f"{i:02d}" for i in range(1, 12) if i != 7]

# Calibrated per native drone resolution; populated by pipelines/calibrate_k.py.
# Key format: "WxH" (e.g. "3976x2652"). Falls back to K_DEFAULT on miss.
_K_JSON = os.path.join(_HERE, "k_calibration.json")
try:
    with open(_K_JSON) as _f:
        K_PER_RES = {tuple(int(x) for x in k.split("x")): float(v)
                     for k, v in json.load(_f).items()}
except FileNotFoundError:
    K_PER_RES = {}

# CLI override (set by run_pipeline when --k-override is passed); None = use lookup.
K_OVERRIDE = None


# stores terminal to log files
class TeeLogger:
    def __init__(self, path):
        self._f, self._out = open(path, "w", buffering=1), sys.stdout
    def write(self, m):  self._out.write(m); self._f.write(m)
    def flush(self):     self._out.flush();  self._f.flush()
    def __enter__(self): sys.stdout = self;  return self
    def __exit__(self, *_): sys.stdout = self._out; self._f.close()


# ── Geo helpers ─────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return EARTH_R_M * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def gps_to_px(lat, lon, g):
    return (lon - g["lt_lon"]) * g["pplon"], (g["lt_lat"] - lat) * g["pplat"]


def sat_px_to_gps(sx, sy, g):
    return g["lt_lat"] - sy / g["pplat"], g["lt_lon"] + sx / g["pplon"]


def _resolve_k(native_res):
    if K_OVERRIDE is not None:
        return K_OVERRIDE
    if native_res is not None and native_res in K_PER_RES:
        return K_PER_RES[native_res]
    return K_DEFAULT


def metric_m_per_px(height_m, native_res=None, sz_w=SZ_W):
    """Target ground-sampling distance (m/px) for the output patch.

    patch_span_m = K * height_m; m_per_px = patch_span_m / sz_w.
    K is looked up from K_PER_RES by native drone resolution (or K_OVERRIDE
    if set, or K_DEFAULT). This replaces the old HFOV-based formula since
    UAV-VisLoc JPEGs are EXIF-stripped — K is calibrated empirically per
    native resolution by pipelines/calibrate_k.py.
    """
    return _resolve_k(native_res) * height_m / sz_w


# ── Satellite / flight loading ──────────────────────────────────────────────

def _load_bgr(path):
    with Image.open(path) as im:
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _image_size(path):
    with Image.open(path) as im:
        return im.size


def _make_geo(lt_lat, lt_lon, rb_lat, rb_lon, w, h):
    return dict(lt_lat=lt_lat, lt_lon=lt_lon, rb_lat=rb_lat, rb_lon=rb_lon,
                w=w, h=h, pplat=h / (lt_lat - rb_lat), pplon=w / (rb_lon - lt_lon))


def get_flight_paths(flight, dataset_dir=None):
    root = dataset_dir or DATASET_DIR
    base = os.path.join(root, flight)
    return (os.path.join(base, f"satellite{flight}.tif"),
            os.path.join(base, "drone"),
            os.path.join(base, f"{flight}.csv"),
            os.path.join(root, "satellite_ coordinates_range.csv"))


def get_flight09_tile_paths(dataset_dir=None):
    root = dataset_dir or DATASET_DIR
    base = os.path.join(root, "09")
    tiles = sorted(os.path.join(base, f) for f in os.listdir(base)
                   if f.startswith("satellite09_") and f.endswith(".tif"))
    return (tiles, os.path.join(base, "drone"), os.path.join(base, "09.csv"),
            os.path.join(root, "satellite_ coordinates_range.csv"))


def load_satellite(sat_tif, sat_csv):
    print(f"Loading {sat_tif} ... ", end="", flush=True)
    img = _load_bgr(sat_tif)
    h, w = img.shape[:2]
    print(f"{w}x{h} px")
    rows = pd.read_csv(sat_csv)
    rows = rows[rows["mapname"] == os.path.basename(sat_tif)]
    if rows.empty:
        raise KeyError(f"No entry for '{os.path.basename(sat_tif)}' in {sat_csv}")
    m = rows.iloc[0]
    return img, _make_geo(m["LT_lat_map"], m["LT_lon_map"],
                          m["RB_lat_map"], m["RB_lon_map"], w, h)


def _parse_tile_rc(path):
    r, c = os.path.basename(path).split("_")[1].split(".")[0].split("-")
    return int(r), int(c)


def load_flight09_tiles(tile_paths, sat_csv):
    row = pd.read_csv(sat_csv)
    row = row[row["mapname"] == "satellite09.tif"].iloc[0]
    lt_lat, lt_lon = row["LT_lat_map"], row["LT_lon_map"]
    rb_lat, rb_lon = row["RB_lat_map"], row["RB_lon_map"]

    rc      = {_parse_tile_rc(p): p for p in tile_paths}
    rows_i  = sorted({r for r, _ in rc})
    cols_i  = sorted({c for _, c in rc})
    sizes   = {k: _image_size(p) for k, p in rc.items()}
    pplat   = sum(sizes[(r, cols_i[0])][1] for r in rows_i) / (lt_lat - rb_lat)
    pplon   = sum(sizes[(rows_i[0], c)][0] for c in cols_i) / (rb_lon - lt_lon)

    tiles, y_off = [], 0
    for r in rows_i:
        x_off = 0
        for c in cols_i:
            tw, th = sizes[(r, c)]
            tll, tlo = lt_lat - y_off / pplat, lt_lon + x_off / pplon
            geo = dict(lt_lat=tll, lt_lon=tlo,
                       rb_lat=tll - th / pplat, rb_lon=tlo + tw / pplon,
                       w=tw, h=th, pplat=pplat, pplon=pplon)
            print(f"  Loaded {os.path.basename(rc[(r, c)])}: {tw}x{th} px  "
                  f"lat[{tll:.6f}, {geo['rb_lat']:.6f}]  "
                  f"lon[{tlo:.6f}, {geo['rb_lon']:.6f}]")
            tiles.append((_load_bgr(rc[(r, c)]), geo))
            x_off += tw
        y_off += sizes[(r, cols_i[0])][1]
    return tiles


def load_flight(flight, dataset_dir=None):
    """Return (tiles, drone_dir, drone_csv, sat_csv); tiles = [(img, geo), ...]."""
    if flight == "09":
        tp, drone_dir, drone_csv, sat_csv = get_flight09_tile_paths(dataset_dir)
        return load_flight09_tiles(tp, sat_csv), drone_dir, drone_csv, sat_csv
    sat_tif, drone_dir, drone_csv, sat_csv = get_flight_paths(flight, dataset_dir)
    sat, geo = load_satellite(sat_tif, sat_csv)
    return [(sat, geo)], drone_dir, drone_csv, sat_csv


# ── Cropping ────────────────────────────────────────────────────────────────

def tile_for_gps(tiles, lat, lon):
    """Pick the tile containing (lat, lon); fall back to nearest with clamped px.
    Returns (sat, geo, cx, cy, in_bounds)."""
    cand = [(sat, geo, *gps_to_px(lat, lon, geo)) for sat, geo in tiles]
    hit  = next((c for c in cand
                 if 0 <= c[2] < c[1]["w"] and 0 <= c[3] < c[1]["h"]), None)
    if hit:
        return (*hit, True)
    sat, geo, cx, cy = min(cand, key=lambda c: sum(max(0, d) for d in
        (-c[2], c[2] - c[1]["w"], -c[3], c[3] - c[1]["h"])))
    return (sat, geo,
            min(max(cx, 0), geo["w"] - 1),
            min(max(cy, 0), geo["h"] - 1), False)


def _metric_affine(geo, cx, cy, height_m, yaw_deg, sz_w, sz_h, native_res):
    """Build the 2×3 affine M mapping output-patch px → satellite px.
    The output patch is metric-isotropic with `m_per_px` GSD."""
    m_per_px   = metric_m_per_px(height_m, native_res=native_res, sz_w=sz_w)
    mid_lat    = (geo["lt_lat"] + geo["rb_lat"]) / 2
    sx_per_m   = geo["pplon"] / (math.cos(math.radians(mid_lat)) * DEG_TO_M)
    sy_per_m   = geo["pplat"] / DEG_TO_M
    ct, st     = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    a, b       =  m_per_px * sx_per_m * ct, -m_per_px * sx_per_m * st
    c, d       =  m_per_px * sy_per_m * st,  m_per_px * sy_per_m * ct
    tx         = cx - a * sz_w / 2 - b * sz_h / 2
    ty         = cy - c * sz_w / 2 - d * sz_h / 2
    # float64 so patch_px_to_gps keeps sub-cm precision near big satellite coords.
    return np.array([[a, b, tx], [c, d, ty]], dtype=np.float64), m_per_px


def metric_crop(sat, geo, cx, cy, height_m, yaw_deg=0.0,
                sz_w=SZ_W, sz_h=SZ_H, native_res=None):
    """Sample a metric-isotropic, optionally heading-rotated patch around (cx,cy).

    yaw_deg is compass-convention (CW from north); pass `Phi1` directly.
    native_res = (native_w, native_h) of the source drone image, used to look
    up the calibrated K (patch-span-per-altitude) constant.
    Returns (patch, M) where M is the 2×3 affine output_px → satellite_px,
    or (None, None) when the source rectangle barely overlaps the tile.
    """
    M, _ = _metric_affine(geo, cx, cy, height_m, yaw_deg, sz_w, sz_h, native_res)
    a, b, tx = M[0]; c, d, ty = M[1]

    # Reject crops where most of the source rectangle is outside the tile.
    xs = a * np.array([0, sz_w - 1, 0, sz_w - 1]) \
        + b * np.array([0, 0, sz_h - 1, sz_h - 1]) + tx
    ys = c * np.array([0, sz_w - 1, 0, sz_w - 1]) \
        + d * np.array([0, 0, sz_h - 1, sz_h - 1]) + ty
    bbox  = (xs.max() - xs.min()) * (ys.max() - ys.min())
    inter = (max(0.0, min(geo["w"], xs.max()) - max(0.0, xs.min()))
             * max(0.0, min(geo["h"], ys.max()) - max(0.0, ys.min())))
    if bbox <= 0 or inter / bbox < MIN_PATCH_COVERAGE:
        return None, None

    half = int(math.ceil(math.hypot(xs.max() - xs.min(),
                                    ys.max() - ys.min()) / 2)) + 4
    cxi, cyi = int(round(cx)), int(round(cy))
    x0, y0 = max(0, cxi - half),       max(0, cyi - half)
    x1, y1 = min(geo["w"], cxi + half), min(geo["h"], cyi + half)
    if x1 <= x0 or y1 <= y0:
        return None, None

    M_roi = M.copy(); M_roi[0, 2] -= x0; M_roi[1, 2] -= y0
    patch = cv2.warpAffine(sat[y0:y1, x0:x1], M_roi.astype(np.float32),
                           (sz_w, sz_h),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
    return patch, M


def patch_px_to_gps(px, py, M, geo):
    """Apply the metric_crop affine and convert to (lat, lon)."""
    sx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
    sy = M[1, 0] * px + M[1, 1] * py + M[1, 2]
    return sat_px_to_gps(sx, sy, geo)


# ── CLAHE ───────────────────────────────────────────────────────────────────

def _make_clahe(enabled):
    if not enabled:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    def apply(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return apply


# ── Per-flight match collection ─────────────────────────────────────────────

# Imported here (not at top) to break a circular dependency: results.py
# pulls constants from this module.
from helpers.results import _build_row, _skip_row  # noqa: E402

_PATCH_CENTRE = np.float32([[SZ_W / 2, SZ_H / 2]]).reshape(-1, 1, 2)


def _predict_from_H(H, m_per_px):
    """(px, py), err_px, err_m for the centre projected through H."""
    px, py = cv2.perspectiveTransform(_PATCH_CENTRE, H).reshape(2)
    err_px = math.hypot(float(px) - SZ_W / 2, float(py) - SZ_H / 2)
    return (float(px), float(py)), err_px, err_px * m_per_px


def collect_pipeline_rows_multitile(tiles, df, match_factory, *, drone_dir,
                                    flight=None, min_inl=MIN_INL, clahe=True,
                                    viz_fn=None, viz_dir=None, progress=True):
    """Iterate `df`: load drone → pick tile → metric_crop → match → record row."""
    if drone_dir is None:
        raise ValueError("drone_dir is required")
    if viz_fn is not None and viz_dir is None:
        raise ValueError("viz_dir is required when viz_fn is set")
    clahe_fn = _make_clahe(clahe)
    if viz_dir is not None:
        os.makedirs(viz_dir, exist_ok=True)

    rows, best_heap, worst_heap = [], [], []
    sample_idx, n_valid = 0, 0
    running = {t: 0 for t in ACC_THRESHOLDS}
    pbar = tqdm(df.iterrows(), total=len(df), unit="img", disable=not progress)

    for _, row in pbar:
        f = row["filename"]
        lat, lon, height = float(row["lat"]), float(row["lon"]), float(row["height"])
        yaw = float(row["Phi1"]) if "Phi1" in row.index else 0.0

        drone_path = os.path.join(drone_dir, f)
        try:
            with Image.open(drone_path) as _im:
                native_res = _im.size  # (w, h) — header-only read
        except (FileNotFoundError, OSError):
            rows.append(_skip_row(f, flight)); continue
        drone = cv2.imread(drone_path)
        if drone is None:
            rows.append(_skip_row(f, flight)); continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        if clahe_fn:
            drone = clahe_fn(drone)

        sat, geo, cx, cy, in_bounds = tile_for_gps(tiles, lat, lon)
        if not in_bounds:
            rows.append(_skip_row(f, flight)); continue
        patch, M = metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw,
                               native_res=native_res)
        if patch is None:
            rows.append(_skip_row(f, flight)); continue
        if clahe_fn:
            patch = clahe_fn(patch)

        best = match_factory(drone)(patch)
        if best is None:
            rows.append(_skip_row(f, flight)); continue

        m_per_px = metric_m_per_px(height, native_res=native_res)
        best["_m_per_px"] = m_per_px

        raw_pred_px = raw_err_px = raw_err_m = None
        plat = plon = off_m = None
        H = best.get("H")
        if H is not None:
            raw_pred_px, raw_err_px, raw_err_m = _predict_from_H(H, m_per_px)
            if best.get("inliers", 0) >= min_inl:
                plat, plon = patch_px_to_gps(raw_pred_px[0], raw_pred_px[1], M, geo)
                off_m = raw_err_m

        r = _build_row(f, lat, lon, height, flight, best,
                       raw_pred_px, raw_err_px, raw_err_m, plat, plon,
                       off_m, m_per_px)
        rows.append(r)

        if off_m is not None:
            n_valid += 1
            for t in ACC_THRESHOLDS:
                if r[f"success_{t}"]:
                    running[t] += 1
            if progress:
                pbar.set_postfix({f"A@{t}": f"{100 * running[t] / n_valid:.0f}%"
                                  for t in ACC_THRESHOLDS}, refresh=False)

        if viz_fn is not None:
            sample_idx += 1
            sample = (best["inliers"], sample_idx,
                      drone.copy(), patch.copy(), r.copy(), best)
            heapq.heappush(best_heap, sample)
            if len(best_heap) > BEST_N:
                heapq.heappop(best_heap)
            # worst: negate score so smallest-inlier sample is on top
            worst_sample = (-best["inliers"], sample_idx,
                            drone.copy(), patch.copy(), r.copy(), best)
            heapq.heappush(worst_heap, worst_sample)
            if len(worst_heap) > WORST_N:
                heapq.heappop(worst_heap)

    if viz_fn is not None:
        tag = flight or "all"
        for label, samples in (("best",  sorted(best_heap,  key=lambda x: -x[0])),
                               ("worst", sorted(worst_heap, key=lambda x: -x[0]))):
            for rank, (_, _, dr, pa, rd, bd) in enumerate(samples, 1):
                base = os.path.splitext(rd["filename"])[0]
                viz_fn(dr, pa, bd,
                       f"{tag}_{label}{rank:02d}_inl{rd['inliers']}_{base}.jpg",
                       viz_dir)
        print(f"  Saved {len(best_heap)} best + {len(worst_heap)} worst "
              f"visualizations → {viz_dir}")
    return rows
