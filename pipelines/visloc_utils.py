import math
import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # allow large satellite TIFs

MIN_INL        = 12
MIN_PATCH_COVERAGE = 0.5  # skip metric_crop when <50% of source bbox lies inside the satellite tile
CROP_W         = 2048    # legacy crop width (used by debug_compare.py)
SZ_W, SZ_H    = 1024, 680
CROP_H         = CROP_W * SZ_H // SZ_W   # legacy: 1360
SCALES         = [0.5, 0.75, 1.0, 1.25, 1.5]   # legacy fallback list
RANSAC_THRESH  = 5.0
TOP_MATCHES    = 50
JPEG_QUALITY   = 85
ACC_THRESHOLDS = [5, 10, 15, 20, 25]   # metres for A@N accuracy columns
BEST_N         = 10                 # images in the per-flight best-matches grid
WORST_N        = 10                 # images in the per-flight worst-matches grid
SAT_ZOOM       = 2.0                # satellite patch covers SAT_ZOOM× the linear UAV FOV

_HERE             = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR       = os.path.join(os.path.dirname(_HERE), "UAV_VisLoc_dataset")
# Flight 07's satellite is a 3000x170 strip — too narrow for metric_crop, dropped.
FLIGHTS_AVAILABLE = [f"{i:02d}" for i in range(1, 12) if i != 7]
_UAV_HFOV_DEG     = 70.0


class TeeLogger:
    def __init__(self, path):   self._f, self._out = open(path, "w", buffering=1), sys.stdout
    def write(self, m):         self._out.write(m);  self._f.write(m)
    def flush(self):            self._out.flush();   self._f.flush()
    def __enter__(self):        sys.stdout = self;   return self
    def __exit__(self, *_):     sys.stdout = self._out; self._f.close()


def get_flight_paths(flight, dataset_dir=None):
    root = dataset_dir or DATASET_DIR
    base = os.path.join(root, flight)
    return (os.path.join(base, f"satellite{flight}.tif"), os.path.join(base, "drone"),
            os.path.join(base, f"{flight}.csv"),
            os.path.join(root, "satellite_ coordinates_range.csv"))


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(max(0.0, min(a, 1.0))))


def _image_size(path):
    with Image.open(path) as im: return im.size


def _load_bgr(path):
    with Image.open(path) as im: return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_satellite(sat_tif, sat_csv):
    print(f"Loading {sat_tif} ... ", end="", flush=True)
    img = _load_bgr(sat_tif)
    h, w = img.shape[:2]
    print(f"{w}x{h} px")
    df = pd.read_csv(sat_csv); df = df[df["mapname"] == os.path.basename(sat_tif)]
    if df.empty: raise KeyError(f"No entry for '{os.path.basename(sat_tif)}' in {sat_csv}")
    m = df.iloc[0]
    return img, dict(lt_lat=m["LT_lat_map"], lt_lon=m["LT_lon_map"],
                     rb_lat=m["RB_lat_map"], rb_lon=m["RB_lon_map"], w=w, h=h,
                     pplat=h/(m["LT_lat_map"]-m["RB_lat_map"]),
                     pplon=w/(m["RB_lon_map"]-m["LT_lon_map"]))


def gps_to_px(lat, lon, g):
    return (lon-g["lt_lon"])*g["pplon"], (g["lt_lat"]-lat)*g["pplat"]


def _tile_for_gps(tiles, lat, lon):
    cand = [(sat, geo, *gps_to_px(lat, lon, geo)) for sat, geo in tiles]
    hit = next((c for c in cand if 0 <= c[2] < c[1]["w"] and 0 <= c[3] < c[1]["h"]), None)
    if hit: return hit
    sat, geo, cx, cy = min(cand, key=lambda c: sum(max(0, d) for d in (-c[2], c[2]-c[1]["w"],
                                                                       -c[3], c[3]-c[1]["h"])))
    return sat, geo, min(max(cx, 0), geo["w"]-1), min(max(cy, 0), geo["h"]-1)


def metric_crop(sat, geo, cx, cy, height_m, yaw_deg=0.0,
                sz_w=SZ_W, sz_h=SZ_H, sat_zoom=SAT_ZOOM):
    """Sample a metric-isotropic, optionally heading-rotated patch from the satellite.

    yaw_deg is compass-convention (positive = CW from north). Pass `Phi1` directly.
    sat_zoom > 1 widens the satellite footprint relative to the UAV FOV (UAV's
    image content occupies ~1/sat_zoom of the patch's linear extent).

    Returns (patch, pred_to_gps) — pred_to_gps(px, py) maps an output-patch pixel
    back to (lat, lon). Returns (None, None) if the crop falls outside the satellite.
    """
    m_per_px = 2.0 * height_m * math.tan(math.radians(_UAV_HFOV_DEG / 2)) / sz_w * sat_zoom
    mid_lat = (geo["lt_lat"] + geo["rb_lat"]) / 2
    px_per_m_x = geo["pplon"] / (math.cos(math.radians(mid_lat)) * 111_320)
    px_per_m_y = geo["pplat"] / 111_320
    src_w_px = sz_w * m_per_px * px_per_m_x
    src_h_px = sz_h * m_per_px * px_per_m_y

    theta = math.radians(yaw_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    a =  m_per_px * px_per_m_x * cos_t
    b = -m_per_px * px_per_m_x * sin_t
    c =  m_per_px * px_per_m_y * sin_t
    d =  m_per_px * px_per_m_y * cos_t
    tx = cx - a * (sz_w / 2) - b * (sz_h / 2)
    ty = cy - c * (sz_w / 2) - d * (sz_h / 2)

    # Skip when too little of the requested source rectangle lies inside the tile,
    # otherwise warpAffine fills the rest from BORDER_CONSTANT and the patch is
    # mostly black (and previously, with BORDER_REFLECT, mostly fabricated).
    cx_corners = np.array([0.0, sz_w - 1, 0.0,        sz_w - 1])
    cy_corners = np.array([0.0, 0.0,      sz_h - 1,   sz_h - 1])
    sx_corners = a * cx_corners + b * cy_corners + tx
    sy_corners = c * cx_corners + d * cy_corners + ty
    bbox_w = sx_corners.max() - sx_corners.min()
    bbox_h = sy_corners.max() - sy_corners.min()
    inter_w = max(0.0, min(geo["w"], sx_corners.max()) - max(0.0, sx_corners.min()))
    inter_h = max(0.0, min(geo["h"], sy_corners.max()) - max(0.0, sy_corners.min()))
    if bbox_w * bbox_h <= 0 or (inter_w * inter_h) / (bbox_w * bbox_h) < MIN_PATCH_COVERAGE:
        return None, None

    half_diag = int(math.ceil(math.hypot(src_w_px, src_h_px) / 2)) + 4
    cx_i, cy_i = int(round(cx)), int(round(cy))
    x0 = max(0, cx_i - half_diag)
    y0 = max(0, cy_i - half_diag)
    x1 = min(geo["w"], cx_i + half_diag)
    y1 = min(geo["h"], cy_i + half_diag)
    if x1 <= x0 or y1 <= y0:
        return None, None
    roi = sat[y0:y1, x0:x1]
    M_roi = np.float32([[a, b, tx - x0], [c, d, ty - y0]])
    patch = cv2.warpAffine(
        roi, M_roi, (sz_w, sz_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )

    def pred_to_gps(px, py):
        sx = a * px + b * py + tx
        sy = c * px + d * py + ty
        plat = geo["lt_lat"] - sy / geo["pplat"]
        plon = geo["lt_lon"] + sx / geo["pplon"]
        return plat, plon

    return patch, pred_to_gps


# ---- legacy helpers retained for debug_compare.py ---------------------------

def crop_sat(sat, cx, cy, g, crop_w, crop_h):
    """Legacy axis-aligned, fixed-aspect crop. Used only by debug_compare.py
    to render the "before-fix" panel; not in the active matching pipeline."""
    if not (0 <= cx < g["w"] and 0 <= cy < g["h"]): return None
    cx = min(max(int(round(cx)), 0), g["w"] - 1)
    cy = min(max(int(round(cy)), 0), g["h"] - 1)
    sx, sy = crop_w / SZ_W, crop_h / SZ_H
    x0 = max(0, cx - crop_w // 2 - 1)
    y0 = max(0, cy - crop_h // 2 - 1)
    x1 = min(g["w"], cx + crop_w // 2 + 1)
    y1 = min(g["h"], cy + crop_h // 2 + 1)
    roi = sat[y0:y1, x0:x1]
    M = np.float32([[sx, 0, cx - crop_w // 2 + (sx - 1) / 2 - x0],
                    [0, sy, cy - crop_h // 2 + (sy - 1) / 2 - y0]])
    return cv2.warpAffine(roi, M, (SZ_W, SZ_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT)


def altitude_scales(height_m, geo):
    """Legacy altitude→scale heuristic. Used only by debug_compare.py."""
    target_s = (2*height_m*math.tan(math.radians(_UAV_HFOV_DEG/2))
                / (math.cos(math.radians((geo["lt_lat"]+geo["rb_lat"])/2)) * 111_320 / geo["pplon"])
                / CROP_W)
    return sorted(SCALES, key=lambda s: abs(s-target_s))


def apply_clahe_lab(bgr, clahe):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def make_result_row(filename, lat, lon, height, r, best_crop, off_m, plat, plon):
    _r = lambda x, n: round(x, n) if x is not None else None
    row = dict(filename=filename, lat=lat, lon=lon, height=height, skipped=False,
               crop_w=best_crop[0], crop_h=best_crop[1],
               sat_kp=r["sat_kp"], drone_kp=r["drone_kp"],
               raw=r["raw"], good=r["good"], inliers=r["inliers"],
               inlier_ratio=round(r["inliers"]/r["good"], 4) if r["good"] else 0,
               pred_lat=_r(plat, 7), pred_lon=_r(plon, 7), offset_m=_r(off_m, 2))
    for t in ACC_THRESHOLDS:
        row[f"success_{t}"] = off_m is not None and off_m <= t
    return row


def _ensure_bgr(img):
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _draw_pred_circle(patch, H):
    if H is None: return patch
    out = _ensure_bgr(patch).copy()
    px = cv2.perspectiveTransform(
        np.float32([[SZ_W/2, SZ_H/2]]).reshape(-1, 1, 2), H).reshape(2)
    cx, cy = int(round(px[0])), int(round(px[1]))
    cv2.circle(out, (cx, cy), 30, (0, 255, 255), 3)   # yellow ring: predicted location
    cv2.circle(out, (cx, cy), 4,  (0, 0, 255),  -1)   # red dot: predicted center
    return out


def draw_and_save(drone, kpd, patch, kps, matches, filename, viz_dir, H=None):
    patch = _draw_pred_circle(patch, H)
    viz = cv2.drawMatches(drone, kpd, patch, kps, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(viz_dir, f"{os.path.splitext(filename)[0]}_matches.jpg"),
                viz, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def save_dense_viz(drone, patch, best, filename, viz_dir):
    kp0, kp1 = best.get("_kp0"), best.get("_kp1")
    if kp0 is None or kp1 is None:
        return
    mask = best.get("_mask")
    if mask is None or mask.sum() == 0:
        kpd, kps, top = [], [], []
    else:
        conf = best["_conf"]
        kpd = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0[mask]]
        kps = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1[mask]]
        top = sorted([cv2.DMatch(i, i, 1.0-c) for i, c in enumerate(conf[mask])],
                     key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, kpd, patch, kps, top, filename, viz_dir, H=best.get("H"))

def get_flight09_tile_paths(dataset_dir=None):
    root = dataset_dir or DATASET_DIR
    base = os.path.join(root, "09")
    tiles = sorted(os.path.join(base, f) for f in os.listdir(base)
                   if f.startswith("satellite09_") and f.endswith(".tif"))
    return tiles, os.path.join(base, "drone"), os.path.join(base, "09.csv"), \
           os.path.join(root, "satellite_ coordinates_range.csv")


def _parse_tile_rc(path):
    r, c = os.path.basename(path).split("_")[1].split(".")[0].split("-")
    return int(r), int(c)


def load_flight09_tiles(tile_paths, sat_csv):
    row = pd.read_csv(sat_csv); row = row[row["mapname"] == "satellite09.tif"].iloc[0]
    lt_lat, lt_lon, rb_lat, rb_lon = row["LT_lat_map"], row["LT_lon_map"], row["RB_lat_map"], row["RB_lon_map"]

    rc_to_path  = {_parse_tile_rc(p): p for p in tile_paths}
    unique_rows = sorted({rc[0] for rc in rc_to_path})
    unique_cols = sorted({rc[1] for rc in rc_to_path})
    size_map    = {rc: _image_size(p) for rc, p in rc_to_path.items()}

    pplat = sum(size_map[(r, unique_cols[0])][1] for r in unique_rows) / (lt_lat - rb_lat)
    pplon = sum(size_map[(unique_rows[0], c)][0] for c in unique_cols) / (rb_lon - lt_lon)

    tiles, y_off = [], 0
    for r in unique_rows:
        x_off = 0
        for c in unique_cols:
            tw, th = size_map[(r, c)]
            img = _load_bgr(rc_to_path[(r,c)])
            tll, tlo = lt_lat - y_off/pplat, lt_lon + x_off/pplon
            geo = dict(lt_lat=tll, lt_lon=tlo, rb_lat=tll-th/pplat, rb_lon=tlo+tw/pplon,
                       w=tw, h=th, pplat=pplat, pplon=pplon)
            print(f"  Loaded {os.path.basename(rc_to_path[(r,c)])}: {tw}x{th} px  "
                  f"lat[{tll:.6f}, {tll-th/pplat:.6f}]  lon[{tlo:.6f}, {tlo+tw/pplon:.6f}]")
            tiles.append((img, geo)); x_off += tw
        y_off += size_map[(r, unique_cols[0])][1]
    return tiles


def _skip_row(f, flight):
    return {"filename": f, "skipped": True, **({} if flight is None else {"flight": flight})}


def setup_viz_dir(viz_dir):
    """Clear the visualization directory at pipeline startup so old runs do not accumulate."""
    if viz_dir is None:
        return
    import shutil
    shutil.rmtree(viz_dir, ignore_errors=True)
    os.makedirs(viz_dir, exist_ok=True)


def load_flight(flight, dataset_dir=None):
    if flight == "09":
        tp, drone_dir, drone_csv, sat_csv = get_flight09_tile_paths(dataset_dir)
        return load_flight09_tiles(tp, sat_csv), drone_dir, drone_csv, sat_csv
    sat_tif, drone_dir, drone_csv, sat_csv = get_flight_paths(flight, dataset_dir)
    sat, geo = load_satellite(sat_tif, sat_csv)
    return [(sat, geo)], drone_dir, drone_csv, sat_csv


def collect_pipeline_rows_multitile(tiles, df, match_factory, dist, min_inl=MIN_INL,
                                     clahe="auto", viz_fn=None, viz_dir=None,
                                     drone_dir=None, flight=None, progress=True):
    import heapq
    if drone_dir is None: raise ValueError("drone_dir is required")
    if viz_fn is not None and viz_dir is None: raise ValueError("viz_dir is required when viz_fn is set")
    if clahe == "auto": clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_fn = (lambda p: apply_clahe_lab(p, clahe)) if clahe is not None else None
    if viz_dir is not None: os.makedirs(viz_dir, exist_ok=True)
    rows, _best, _worst = [], [], []
    sample_counter = 0
    running = {t: 0 for t in ACC_THRESHOLDS}
    n_valid = 0
    pbar = tqdm(df.iterrows(), total=len(df), unit="img", disable=not progress)
    for _, row in pbar:
        f = row["filename"]
        lat, lon, height = float(row["lat"]), float(row["lon"]), float(row["height"])
        yaw = float(row["Phi1"]) if "Phi1" in row.index else 0.0
        drone = cv2.imread(os.path.join(drone_dir, f))
        if drone is None: rows.append(_skip_row(f, flight)); continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        if clahe_fn is not None: drone = clahe_fn(drone)

        sat, geo, cx, cy = _tile_for_gps(tiles, lat, lon)
        patch, pred_to_gps = metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw)
        if patch is None: rows.append(_skip_row(f, flight)); continue
        if clahe_fn is not None: patch = clahe_fn(patch)
        best = match_factory(drone)(patch)
        if best is None: rows.append(_skip_row(f, flight)); continue

        if best.get("inliers", 0) >= min_inl and best.get("H") is not None:
            px_c, py_c = cv2.perspectiveTransform(
                np.float32([[SZ_W/2, SZ_H/2]]).reshape(-1, 1, 2),
                best["H"]).reshape(2)
            plat, plon = pred_to_gps(float(px_c), float(py_c))
            off_m = haversine_m(lat, lon, plat, plon)
        else:
            plat = plon = off_m = None
        r = make_result_row(f, lat, lon, height, best, (SZ_W, SZ_H), off_m, plat, plon)
        if flight is not None: r["flight"] = flight
        rows.append(r)

        if off_m is not None:
            n_valid += 1
            for t in ACC_THRESHOLDS:
                if r[f"success_{t}"]: running[t] += 1
            if progress and n_valid > 0:
                pbar.set_postfix({f"A@{t}": f"{100*running[t]/n_valid:.0f}%"
                                  for t in ACC_THRESHOLDS}, refresh=False)

        if viz_fn is not None and patch is not None:
            sample_counter += 1
            # Best: min-heap on inliers, pop minimum → keeps top BEST_N by inliers
            heapq.heappush(_best, (best["inliers"], sample_counter,
                                   drone.copy(), patch.copy(), r.copy(), best))
            if len(_best) > BEST_N:
                heapq.heappop(_best)
            # Worst: max-heap (negate), pop maximum → keeps bottom WORST_N by inliers
            heapq.heappush(_worst, (-best["inliers"], sample_counter,
                                    drone.copy(), patch.copy(), r.copy(), best))
            if len(_worst) > WORST_N:
                heapq.heappop(_worst)

    if viz_fn is not None:
        flight_tag = flight or "all"
        for label, samples in (("best", sorted(_best, key=lambda x: -x[0])),
                               ("worst", sorted(_worst, key=lambda x: -x[0]))):
            for rank, (_, _, dr, pa, rd, bd) in enumerate(samples, 1):
                base = os.path.splitext(rd["filename"])[0]
                fname = f"{flight_tag}_{label}{rank:02d}_inl{rd['inliers']}_{base}.jpg"
                viz_fn(dr, pa, bd, fname, viz_dir)
        print(f"  Saved {len(_best)} best + {len(_worst)} worst match visualizations → {viz_dir}")
    return rows


def print_summary(v, label, min_inl=MIN_INL):
    if v.empty: print("\n  All images skipped."); return
    n        = len(v)
    accepted = v[v["inliers"] >= min_inl]
    h        = len(accepted)
    print(f"\n  Results saved to {label}")
    for t in ACC_THRESHOLDS:
        col = f"success_{t}"
        s = int(v[col].fillna(False).sum()) if col in v.columns else 0
        print(f"  A@{t:2d}m:              {s}/{n} ({100*s/n:.1f}%)")
    under20 = v[v["offset_m"].fillna(9999) <= 20]["offset_m"]
    if len(under20):
        print(f"  Mean error (≤20m):   {under20.mean():.1f}m")
    print(f"  Homography accepted: {h}/{n} ({100*h/n:.1f}%)")
    print(f"  Median inliers: {v['inliers'].median():.0f} | ratio: {v['inlier_ratio'].median():.3f}")
