"""Shared utilities for the UAV-VisLoc benchmark pipelines.

Public API (used by pipelines/ and kaggle_benchmark.ipynb):
  * Constants: SZ_W, SZ_H, RANSAC_THRESH, MIN_INL, TOP_MATCHES,
               SAT_ZOOM, ACC_THRESHOLDS, JPEG_QUALITY, FLIGHTS_AVAILABLE
  * Geo:       haversine_m, gps_to_px, metric_m_per_px
  * Loading:   get_flight_paths, get_flight09_tile_paths,
               load_satellite, load_flight09_tiles, load_flight
  * Cropping:  tile_for_gps, metric_crop
  * Viz:       draw_and_save, save_dense_viz, setup_viz_dir
  * Driver:    collect_pipeline_rows_multitile, run_pipeline
  * Misc:      TeeLogger
"""

import argparse
import heapq
import math
import multiprocessing
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # large satellite TIFs

# ── Constants ───────────────────────────────────────────────────────────────

SZ_W, SZ_H         = 1024, 680
RANSAC_THRESH      = 5.0
MIN_INL            = 7
TOP_MATCHES        = 50
MIN_PATCH_COVERAGE = 0.2     # skip mostly-outside crops; edge samples still evaluate
SAT_ZOOM           = 1.75    # patch covers SAT_ZOOM × UAV linear FOV
ACC_THRESHOLDS     = [5, 10, 15, 20, 25]
BEST_N, WORST_N    = 10, 10
JPEG_QUALITY       = 85
UAV_HFOV_DEG       = 70.0
EARTH_R_M          = 6_371_000.0
DEG_TO_M           = 111_320.0  # meters per degree latitude (flat-earth approx)

_HERE       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(_HERE), "UAV_VisLoc_dataset")
# Flight 07's satellite is 3000×170 — too narrow for metric_crop; dropped.
FLIGHTS_AVAILABLE = [f"{i:02d}" for i in range(1, 12) if i != 7]


# ── TeeLogger ───────────────────────────────────────────────────────────────

class TeeLogger:
    """Duplicate stdout to a log file inside a `with` block."""
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


def metric_m_per_px(height_m, sz_w=SZ_W, sat_zoom=SAT_ZOOM):
    """Target ground-sampling distance (m/px) for the output patch."""
    return 2.0 * height_m * math.tan(math.radians(UAV_HFOV_DEG / 2)) / sz_w * sat_zoom


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


def _metric_affine(geo, cx, cy, height_m, yaw_deg, sz_w, sz_h, sat_zoom):
    """Build the 2×3 affine M mapping output-patch px → satellite px.
    The output patch is metric-isotropic with `m_per_px` GSD."""
    m_per_px   = metric_m_per_px(height_m, sz_w=sz_w, sat_zoom=sat_zoom)
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
                sz_w=SZ_W, sz_h=SZ_H, sat_zoom=SAT_ZOOM):
    """Sample a metric-isotropic, optionally heading-rotated patch around (cx,cy).

    yaw_deg is compass-convention (CW from north); pass `Phi1` directly.
    Returns (patch, M) where M is the 2×3 affine output_px → satellite_px,
    or (None, None) when the source rectangle barely overlaps the tile.
    """
    M, _ = _metric_affine(geo, cx, cy, height_m, yaw_deg, sz_w, sz_h, sat_zoom)
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


# ── Visualization ───────────────────────────────────────────────────────────

def _draw_overlays(patch, H, m_per_px=None):
    """Green cross+circle = GT (patch centre); yellow/red = predicted point."""
    out = (patch if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)).copy()
    gx, gy, arm = SZ_W // 2, SZ_H // 2, 18
    cv2.line(out, (gx - arm, gy), (gx + arm, gy), (0, 220, 0), 3)
    cv2.line(out, (gx, gy - arm), (gx, gy + arm), (0, 220, 0), 3)
    cv2.circle(out, (gx, gy), arm + 6, (0, 220, 0), 2)

    if m_per_px and m_per_px > 0:
        for metres, col in ((20, (255, 255, 0)), (25, (255, 255, 255))):
            r = max(1, int(round(metres / m_per_px)))
            cv2.circle(out, (gx, gy), r, col, 1, cv2.LINE_AA)
            cv2.putText(out, f"{metres}m", (gx + r + 4, gy - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    if H is None:
        return out
    px = cv2.perspectiveTransform(
        np.float32([[SZ_W / 2, SZ_H / 2]]).reshape(-1, 1, 2), H).reshape(2)
    pxi = (int(round(float(px[0]))), int(round(float(px[1]))))
    cv2.line(out, (gx, gy), pxi, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.circle(out, pxi, 8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(out, pxi, 4, (0, 0, 255), -1, cv2.LINE_AA)
    if m_per_px and m_per_px > 0:
        err_m = math.hypot(float(px[0]) - SZ_W / 2, float(px[1]) - SZ_H / 2) * m_per_px
        cv2.putText(out, f"{err_m:.1f}m", (pxi[0] + 10, pxi[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def draw_and_save(drone, kpd, patch, kps, matches, filename, viz_dir,
                  H=None, m_per_px=None):
    patch = _draw_overlays(patch, H, m_per_px=m_per_px)
    viz = cv2.drawMatches(drone, kpd, patch, kps, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    sep = drone.shape[1]
    cv2.line(viz, (sep, 0), (sep, viz.shape[0] - 1), (255, 255, 255), 3)
    cv2.imwrite(os.path.join(viz_dir, f"{os.path.splitext(filename)[0]}_matches.jpg"),
                viz, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def save_dense_viz(drone, patch, best, filename, viz_dir):
    """Dense-matcher viz (LoFTR/RoMa/MATCHA)."""
    kp0, kp1 = best.get("_kp0"), best.get("_kp1")
    if kp0 is None or kp1 is None:
        return
    mask = best.get("_mask")
    if mask is None or mask.sum() == 0:
        kpd, kps, top = [], [], []
    else:
        conf = best["_conf"]
        kpd  = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0[mask]]
        kps  = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1[mask]]
        top  = sorted([cv2.DMatch(i, i, 1.0 - c) for i, c in enumerate(conf[mask])],
                      key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, kpd, patch, kps, top, filename, viz_dir,
                  H=best.get("H"), m_per_px=best.get("_m_per_px"))


def setup_viz_dir(viz_dir):
    if viz_dir is None:
        return
    shutil.rmtree(viz_dir, ignore_errors=True)
    os.makedirs(viz_dir, exist_ok=True)


# ── Result rows / summaries ─────────────────────────────────────────────────

def _round(x, n):
    return None if x is None else round(x, n)


def _build_row(filename, lat, lon, height, flight, match,
               raw_pred_px, raw_err_px, raw_err_m, plat, plon, off_m, m_per_px):
    row = dict(filename=filename, lat=lat, lon=lon, height=height, skipped=False,
               crop_w=SZ_W, crop_h=SZ_H,
               sat_kp=match["sat_kp"], drone_kp=match["drone_kp"],
               raw=match["raw"], good=match["good"], inliers=match["inliers"],
               inlier_ratio=round(match["inliers"] / match["good"], 4)
                            if match["good"] else 0,
               raw_pred_x=_round(raw_pred_px[0], 2) if raw_pred_px else None,
               raw_pred_y=_round(raw_pred_px[1], 2) if raw_pred_px else None,
               raw_err_px=_round(raw_err_px, 2),
               raw_err_m=_round(raw_err_m, 2),
               m_per_px=_round(m_per_px, 4),
               pred_lat=_round(plat, 7), pred_lon=_round(plon, 7),
               offset_m=_round(off_m, 2))
    for t in ACC_THRESHOLDS:
        row[f"success_{t}"] = off_m is not None and off_m <= t
    if flight:
        row["flight"] = flight
    return row


def _skip_row(filename, flight):
    r = {"filename": filename, "skipped": True}
    if flight:
        r["flight"] = flight
    return r


def print_summary(v, label, min_inl=MIN_INL):
    if v.empty:
        print("\n  All images skipped."); return
    n = len(v)
    accepted     = v[v["inliers"] >= min_inl]
    accepted_err = accepted["offset_m"].dropna() if "offset_m" in accepted else pd.Series(dtype=float)
    raw_err      = v["raw_err_m"].dropna()       if "raw_err_m" in v       else pd.Series(dtype=float)
    print(f"\n  Results saved to {label}")
    for t in ACC_THRESHOLDS:
        col = f"success_{t}"
        s = int(v[col].fillna(False).sum()) if col in v.columns else 0
        print(f"  A@{t:2d}m:              {s}/{n} ({100 * s / n:.1f}%)")
    if len(accepted_err):
        rmse = math.sqrt(float(np.mean(np.square(accepted_err))))
        print(f"  Error accepted:      mean {accepted_err.mean():.1f}m | "
              f"median {accepted_err.median():.1f}m | RMSE {rmse:.1f}m | "
              f"P90 {np.percentile(accepted_err, 90):.1f}m | max {accepted_err.max():.1f}m")
    if len(raw_err):
        print(f"  Raw center error:    median {raw_err.median():.1f}m | "
              f"P90 {np.percentile(raw_err, 90):.1f}m | max {raw_err.max():.1f}m")
    print(f"  Homography accepted: {len(accepted)}/{n} ({100 * len(accepted) / n:.1f}%)")
    print(f"  Median inliers: {v['inliers'].median():.0f} | "
          f"ratio: {v['inlier_ratio'].median():.3f}")


def _summarize(rows, label, min_inl):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "skipped" not in df.columns:
        return
    valid = df[~df["skipped"].fillna(False)]
    if not valid.empty:
        print_summary(valid, label, min_inl=min_inl)


# ── Per-flight match collection ─────────────────────────────────────────────

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

        drone = cv2.imread(os.path.join(drone_dir, f))
        if drone is None:
            rows.append(_skip_row(f, flight)); continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        if clahe_fn:
            drone = clahe_fn(drone)

        sat, geo, cx, cy, in_bounds = tile_for_gps(tiles, lat, lon)
        if not in_bounds:
            rows.append(_skip_row(f, flight)); continue
        patch, M = metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw)
        if patch is None:
            rows.append(_skip_row(f, flight)); continue
        if clahe_fn:
            patch = clahe_fn(patch)

        best = match_factory(drone)(patch)
        if best is None:
            rows.append(_skip_row(f, flight)); continue

        m_per_px = metric_m_per_px(height)
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


# ── Pipeline orchestrator ───────────────────────────────────────────────────

def _collect_flight(flight, match_factory, viz_fn, viz_dir, clahe, limit,
                    min_inl, progress):
    tiles, drone_dir, drone_csv, _ = load_flight(flight)
    df = pd.read_csv(drone_csv)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    if progress:
        print(f"\n=== Flight {flight}: {len(df)} images ===")
    return collect_pipeline_rows_multitile(
        tiles, df, match_factory,
        drone_dir=drone_dir, flight=flight, clahe=clahe, min_inl=min_inl,
        viz_fn=viz_fn if viz_dir else None, viz_dir=viz_dir, progress=progress)


def _gpu_worker(args):
    flight_group, gpu_id, spec, run = args
    import torch  # imported lazily so CPU-only specs don't need torch
    device        = torch.device(f"cuda:{gpu_id}")
    model         = spec["load_model"](device, run["args"])
    match_factory = spec["make_match_factory"](model, device, run["args"])
    return [r for f in flight_group
            for r in _collect_flight(
                f, match_factory, spec["viz_fn"], run["viz_dir"],
                run["clahe"], run["limit"], run["min_inl"], progress=False)]


def _cpu_worker(args):
    chunk_df, tiles, drone_dir, flight, spec, run = args
    match_factory = spec["make_match_factory"](None, None, run["args"])
    return collect_pipeline_rows_multitile(
        tiles, chunk_df, match_factory,
        drone_dir=drone_dir, flight=flight, clahe=run["clahe"],
        min_inl=run["min_inl"],
        viz_fn=spec["viz_fn"] if run["viz_dir"] else None,
        viz_dir=run["viz_dir"], progress=False)


def _run_gpu_flights(flights, spec, run):
    import torch
    n_gpus = max(1, torch.cuda.device_count())
    groups = [g for g in [flights[i::n_gpus] for i in range(n_gpus)] if g]

    if len(groups) <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")
        model         = spec["load_model"](device, run["args"])
        match_factory = spec["make_match_factory"](model, device, run["args"])
        return [r for f in flights
                for r in _collect_flight(
                    f, match_factory, spec["viz_fn"], run["viz_dir"],
                    run["clahe"], run["limit"], run["min_inl"], progress=True)]

    ctx = multiprocessing.get_context("spawn")
    worker_args = [(g, i, spec, run) for i, g in enumerate(groups)]
    with ctx.Pool(len(groups)) as pool:
        chunks = pool.map(_gpu_worker, worker_args)
    return [r for chunk in chunks for r in chunk]


def _run_cpu_chunks(flights, workers, spec, run):
    n_workers = workers or os.cpu_count() or 1
    rows = []
    for flight in flights:
        tiles, drone_dir, drone_csv, _ = load_flight(flight)
        df = pd.read_csv(drone_csv)
        if run["limit"] is not None:
            df = df.iloc[:run["limit"]].reset_index(drop=True)
        print(f"\n=== Flight {flight}: {len(df)} images ===")
        n_chunks = min(n_workers, max(len(df), 1))
        edges = np.linspace(0, len(df), n_chunks + 1, dtype=int)
        chunks = [df.iloc[edges[i]:edges[i + 1]].reset_index(drop=True)
                  for i in range(n_chunks)]

        if len(chunks) == 1:
            match_factory = spec["make_match_factory"](None, None, run["args"])
            rows.extend(collect_pipeline_rows_multitile(
                tiles, df, match_factory,
                drone_dir=drone_dir, flight=flight, clahe=run["clahe"],
                min_inl=run["min_inl"],
                viz_fn=spec["viz_fn"] if run["viz_dir"] else None,
                viz_dir=run["viz_dir"]))
        else:
            chunk_args = [(c, tiles, drone_dir, flight, spec, run) for c in chunks]
            with multiprocessing.Pool(len(chunks)) as pool:
                results = pool.map(_cpu_worker, chunk_args)
            rows.extend(r for chunk in results for r in chunk)
    return rows


def _add_common_args(parser):
    parser.add_argument("--dist",        type=float, default=25.0,
                        help="Display-only top-k distance threshold.")
    parser.add_argument("--visualize",   action="store_true")
    parser.add_argument("--flights",     nargs="+", default=["all"],
                        help="Flight IDs (e.g. 01 03 05) or 'all'.")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Cap drone images per flight (quick tests).")
    parser.add_argument("--min-inliers", type=int, default=MIN_INL,
                        help=f"Minimum RANSAC inliers to accept H (default: {MIN_INL}).")
    parser.add_argument("--no-clahe",    action="store_true",
                        help="Disable CLAHE preprocessing (on by default).")


def run_pipeline(*, name, load_model, make_match_factory,
                 add_args=None, viz_fn=save_dense_viz, banner=None,
                 parallelism="gpu_flights"):
    """Run a feature-matching benchmark.

    parallelism:
      'gpu_flights' — one worker per CUDA device, each handling a flight subset.
                      Single-GPU / CPU-only runs in-process.
      'cpu_chunks'  — for each flight, split rows across forked CPU workers.
                      Adds `--workers` to the CLI.
    """
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    if parallelism == "cpu_chunks":
        parser.add_argument("--workers", type=int, default=None,
                            help="Parallel workers (default: cpu_count).")
    if add_args:
        add_args(parser)
    args = parser.parse_args()

    flights  = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    name_str = name(args) if callable(name) else name
    out_csv  = f"visloc_{name_str}_results.csv"
    viz_dir  = f"visloc_{name_str}_visualizations" if args.visualize else None
    setup_viz_dir(viz_dir)

    if banner:
        print(banner(args))

    spec = dict(load_model=load_model, make_match_factory=make_match_factory,
                viz_fn=viz_fn)
    run  = dict(args=args, viz_dir=viz_dir, clahe=not args.no_clahe,
                limit=args.limit, min_inl=args.min_inliers)

    with TeeLogger(out_csv.replace(".csv", ".log")):
        if parallelism == "gpu_flights":
            all_rows = _run_gpu_flights(flights, spec, run)
        elif parallelism == "cpu_chunks":
            all_rows = _run_cpu_chunks(flights, args.workers, spec, run)
        else:
            raise ValueError(f"unknown parallelism={parallelism!r}")

        for flight in flights:
            _summarize([r for r in all_rows if r.get("flight") == flight],
                       f"flight {flight}", run["min_inl"])
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        if len(flights) > 1:
            print(f"\n=== Overall ({len(flights)} flights) ===")
            _summarize(all_rows, out_csv, run["min_inl"])
