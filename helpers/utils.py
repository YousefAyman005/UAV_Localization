import heapq
import math
import os
import random
import sys
import time
import zlib

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Optional: only needed for accurate GPU timing (cuda sync around the match
# call). The CPU baseline and local tooling must work without torch.
try:
    import torch
except ImportError:
    torch = None

Image.MAX_IMAGE_PIXELS = None  # large satellite TIFs

# Deterministic RNGs for every pipeline that imports this module.
random.seed(0); np.random.seed(0); cv2.setRNGSeed(0)

# ── Constants ───────────────────────────────────────────────────────────────

SZ_W, SZ_H         = 1024, 680
RANSAC_THRESH      = 5.0
MIN_INL            = 7
TOP_MATCHES        = 50
MIN_PATCH_COVERAGE = 0.2     # skip mostly-outside crops; edge samples still evaluate
ACC_THRESHOLDS     = [5, 10, 15, 20, 25, 30]
BEST_N, WORST_N    = 3, 3
JPEG_QUALITY       = 85
EARTH_R_M          = 6_371_000.0
DEG_TO_M           = 111_320.0  # meters per degree latitude (flat-earth approx)

# patch_span_m = SEARCH_FACTOR * K * altitude_m; m_per_px = patch_span_m / SZ_W.
# K_PER_FLIGHT is the calibrated *drone-footprint* GSD per flight; SEARCH_FACTOR
# makes the satellite patch larger than the drone view so there's room to
# localize. PRIOR_OFFSET_STD_M is the σ of the simulated GPS prior added to the
# patch center, so the drone is not at the trivial dead-center of the patch.
# 01/02/03/08 are the established hand-tuned anchors (H-scale calibration recovers
# them within ~5%); 04/05/06/10/11 are from the H-scale calibration (job 371932,
# pipelines/calibrate_k.py --mode hscale) — see [[project-k-calibration]].
K_PER_FLIGHT = {
    "01": 1.00, "02": 1.00, "03": 0.95, "04": 0.99, "05": 0.167,
    "06": 0.352, "08": 1.00, "10": 0.361, "11": 0.297,
}
K_DEFAULT          = 1.75 * 2.0 * math.tan(math.radians(35.0))
# patch_span = SEARCH_FACTOR * K * alt; margin/side = 0.5*(SEARCH_FACTOR-1)*K*alt.
# 1.75 = sweep-chosen sweet spot: lifts the coverage-starved flights (06/10/01) at
# no overall cost (A@25 60.0, same as 1.5), while 1.8–2.0 bleed inliers and hurt the
# scale-sensitive flights (05/08). Overridable per-run via env UAV_SEARCH_FACTOR.
# K is invariant to SEARCH_FACTOR (true footprint ratio). See [[project-k-calibration]].
SEARCH_FACTOR      = float(os.environ.get("UAV_SEARCH_FACTOR", "1.75"))
PRIOR_OFFSET_STD_M = float(os.environ.get("UAV_PRIOR_STD_M", "80.0"))  # σ of GPS prior; env-overridable for the sensitivity sweep

# Per-flight, per-leg yaw correction (degrees ADDED to Phi1 before metric_crop).
# The recorded yaw only approximates the camera's image orientation: the
# deviation is constant within a flight leg but differs between legs (sign
# flips with flight direction — wind-crab + mount-offset signature).
# Calibrated like K_PER_FLIGHT from matcher geometry: median residual rotation
# of the 4-DOF similarity on RoMa-extre matches, 40 frames/flight
# (analyze/check_crop_rotation.py --calibrate, 2026-06-11; raw legs in
# cache/yaw_calibration.json). {flight: [(leg_heading_deg, offset_deg), ...]}
YAW_OFFSET = {
    "01": [(166.4, -5.3), (29.2, -12.9)],
    "02": [(173.1, -2.1), (6.9, 2.5)],
    "03": [(-40.4, -1.7), (122.6, -7.0)],
    "04": [(-9.6, 1.1), (168.9, -3.8)],
    "05": [(104.7, -4.5), (-102.6, 3.9)],
    "06": [(171.1, -8.8), (-21.2, 2.9)],
    "08": [(109.1, -23.4), (-81.8, 14.2)],
    "10": [(171.9, 6.4), (-34.3, -7.6)],
    "11": [(90.3, -1.9), (-88.8, -0.5)],
}


def corrected_yaw(flight, yaw_deg):
    """Phi1 + the calibrated camera-vs-yaw offset of the nearest flight leg."""
    legs = YAW_OFFSET.get(str(flight))
    if not legs:
        return yaw_deg
    leg = min(legs, key=lambda l: abs((yaw_deg - l[0] + 180.0) % 360.0 - 180.0))
    return yaw_deg + leg[1]


def north_up_drone(bgr, yaw_deg):
    """Rotate a (heading-up) drone image to north-up via its compass yaw.

    Center square crop, then content rotation by -yaw: validated empirically
    (analyze/plot_northup_sign.py geometry test — north_up_drone applied to
    metric_crop(yaw) reproduces metric_crop(0) at NCC 0.97; the +yaw sign
    scores ~0). NB caption_crops.iter_drone_images rotates by +Phi1, i.e.
    the OPPOSITE sign — compass words in drone captions are unreliable.
    Pass corrected_yaw(). Used by the CLIP line (--north-up) so query and
    gallery share orientation; the matcher line instead rotates the
    satellite patch (metric_crop)."""
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    sq = bgr[y0:y0 + side, x0:x0 + side]
    M = cv2.getRotationMatrix2D((side / 2, side / 2), -yaw_deg, 1.0)
    return cv2.warpAffine(sq, M, (side, side))


_HERE       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(_HERE), "UAV_VisLoc_dataset")
# 07 has no flight folder (its satellite is 3000×170 — too narrow for metric_crop).
# 09's satellite is split into 4 tiles, unsupported by the single-tile loader.
FLIGHTS_AVAILABLE = ["01", "02", "03", "04", "05", "06", "08", "10", "11"]


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


def metric_m_per_px(height_m, flight=None, sz_w=SZ_W, k_override=None):
    """Target ground-sampling distance (m/px) for the output patch.
    patch_span_m = SEARCH_FACTOR * K * height_m; m_per_px = patch_span_m / sz_w.
    k_override (if not None) bypasses the per-flight K lookup — used by the
    K-calibration sweep (pipelines/calibrate_k.py)."""
    k = k_override if k_override is not None else K_PER_FLIGHT.get(str(flight), K_DEFAULT)
    return SEARCH_FACTOR * k * height_m / sz_w


# ── Satellite / flight loading ──────────────────────────────────────────────

def _load_bgr(path):
    with Image.open(path) as im:
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


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


def load_flight(flight, dataset_dir=None):
    """Return (tiles, drone_dir, drone_csv, sat_csv); tiles = [(img, geo), ...]."""
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


def _metric_affine(geo, cx, cy, height_m, yaw_deg, sz_w, sz_h, flight, k_override=None):
    """Build the 2×3 affine M mapping output-patch px → satellite px.
    The output patch is metric-isotropic with `m_per_px` GSD."""
    m_per_px   = metric_m_per_px(height_m, flight=flight, sz_w=sz_w, k_override=k_override)
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
                sz_w=SZ_W, sz_h=SZ_H, flight=None, k_override=None):
    """Sample a metric-isotropic, optionally heading-rotated patch around (cx,cy).

    yaw_deg is compass-convention (CW from north); pass `Phi1` directly.
    `flight` selects the calibrated K (patch-span-per-altitude) from K_PER_FLIGHT;
    `k_override` (if not None) forces K directly (K-calibration sweep).
    Returns (patch, M) where M is the 2×3 affine output_px → satellite_px,
    or (None, None) when the source rectangle barely overlaps the tile.
    """
    M, _ = _metric_affine(geo, cx, cy, height_m, yaw_deg, sz_w, sz_h, flight, k_override)
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


def split_flight_rows(df, which="train", test_frac=0.25, axis="auto",
                      buffer_frac=0.0, val_frac=0.0):
    """Deterministic SPATIAL split of a flight's drone rows.

    Drone frames along a flight overlap heavily, so a random split leaks ground
    between train and test. Instead we sort by the wider-spread geographic axis
    (lat or lon) and slice contiguous bands, bottom → top:
    ``train | buffer | val | test``. The test band (top `test_frac`) is
    unaffected by `val_frac`/`buffer_frac`, so existing test splits stay
    bit-identical. `buffer_frac` drops a guard band directly above train —
    protecting train↔val when a val band exists, train↔test otherwise.
    `which` ∈ {train, val, test, all}; returns the filtered df (row order
    preserved). With val_frac=0 behavior is identical to before val existed."""
    if which == "all" or df.empty or (test_frac <= 0 and val_frac <= 0):
        return df.reset_index(drop=True)
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    if axis == "auto":
        axis = "lat" if (lat.max() - lat.min()) >= (lon.max() - lon.min()) else "lon"
    order  = np.argsort(df[axis].to_numpy(dtype=float))
    n      = len(df)
    n_test = int(round(n * test_frac))
    n_val  = int(round(n * val_frac))
    n_buf  = int(round(n * buffer_frac))
    test_idx  = order[n - n_test:]                               # top band → test
    val_idx   = order[n - n_test - n_val: n - n_test]            # directly below test
    train_idx = order[: max(0, n - n_test - n_val - n_buf)]      # bottom (minus buffer)
    keep = {"train": train_idx, "val": val_idx, "test": test_idx}[which]
    return df.iloc[np.sort(keep)].reset_index(drop=True)


def crop_gt_patch(tiles, lat, lon, height_m, yaw_deg=0.0, flight=None):
    """Satellite patch centered on the *true* GPS of a drone image (no prior noise).

    Shared by the captioner and the CLIP LoRA trainer so both operate on the
    identical positive crop (both pass yaw_deg=0: north-up, matching the
    gallery tiles). NB: a caller passing a Phi1-derived yaw must apply
    corrected_yaw() itself. Returns a BGR patch or None when the location is
    out of bounds / barely overlaps the tile."""
    sat, geo, cx, cy, in_bounds = tile_for_gps(tiles, lat, lon)
    if not in_bounds:
        return None
    patch, _ = metric_crop(sat, geo, cx, cy, height_m, yaw_deg=yaw_deg, flight=flight)
    return patch


# ── Robust fit (shared by every matcher pipeline) ───────────────────────────

def fit_similarity(kp0, kp1):
    """4-DOF RANSAC similarity fit, drone px → patch px.

    Every matcher pipeline uses this same robust-fit stage so that A@t
    differences between methods come from match quality, not the estimator.
    The patch is metric-isotropic and yaw-aligned, so the true drone→patch map
    is ≈ translation + scale (+ small residual rotation); 4 DOF cannot
    hallucinate perspective from a handful of inliers the way an 8-DOF
    homography can. Returns (3×3 float64 H, inlier_count) or (None, 0).
    """
    kp0 = np.asarray(kp0, dtype=np.float32).reshape(-1, 2)
    kp1 = np.asarray(kp1, dtype=np.float32).reshape(-1, 2)
    if len(kp0) < 4 or len(kp0) != len(kp1):
        return None, 0
    M, mask = cv2.estimateAffinePartial2D(
        kp0, kp1, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH,
        maxIters=5000, confidence=0.9999, refineIters=10)
    if M is None or mask is None:
        return None, 0
    return np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64), int(mask.sum())


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

# Centre of the (resized) DRONE image — what H projects into the patch. It
# happens to equal the patch centre because both share SZ_W×SZ_H; if the two
# sizes ever diverge, this must stay the drone-image centre.
_DRONE_CENTRE = np.float32([[SZ_W / 2, SZ_H / 2]]).reshape(-1, 1, 2)


def _predict_from_H(H, m_per_px, gt_px):
    """Project the drone-image centre through H into the patch.

    Returns ((px, py), err_px, err_m) where err is the distance to the TRUE GT
    patch pixel `gt_px` — not to the patch centre, which is the noisy GPS prior.
    Computed for any H regardless of the inlier gate, so it doubles as the
    ungated localization error."""
    px, py = cv2.perspectiveTransform(_DRONE_CENTRE, H).reshape(2)
    err_px = math.hypot(float(px) - gt_px[0], float(py) - gt_px[1])
    return (float(px), float(py)), err_px, err_px * m_per_px


def _cuda_sync():
    """Barrier for accurate GPU timing; no-op without torch/CUDA."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_pipeline_rows_multitile(tiles, df, match_factory, *, drone_dir,
                                    flight=None, min_inl=MIN_INL, clahe=True,
                                    viz_fn=None, viz_dir=None, progress=True,
                                    k_override=None, yaw_cal=True):
    """Iterate `df`: load drone → pick tile → metric_crop → match → record row."""
    if drone_dir is None:
        raise ValueError("drone_dir is required")
    if viz_fn is not None and viz_dir is None:
        raise ValueError("viz_dir is required when viz_fn is set")
    clahe_fn = _make_clahe(clahe)
    if viz_dir is not None:
        os.makedirs(viz_dir, exist_ok=True)

    rows, best_heap, worst_heap = [], [], []
    sample_idx, n_scored = 0, 0
    warmed = False
    running = {t: 0 for t in ACC_THRESHOLDS}
    pbar = tqdm(df.iterrows(), total=len(df), unit="img", disable=not progress)

    for _, row in pbar:
        f = row["filename"]
        lat, lon, height = float(row["lat"]), float(row["lon"]), float(row["height"])
        yaw = float(row["Phi1"]) if "Phi1" in row.index else 0.0
        if yaw_cal:
            yaw = corrected_yaw(flight, yaw)

        drone_path = os.path.join(drone_dir, f)
        drone = cv2.imread(drone_path)
        if drone is None:
            rows.append(_skip_row(f, flight)); continue
        # INTER_AREA: correct filter for the ~4x downscale (INTER_LINEAR aliases)
        drone = cv2.resize(drone, (SZ_W, SZ_H), interpolation=cv2.INTER_AREA)
        if clahe_fn:
            drone = clahe_fn(drone)

        sat, geo, cx, cy, in_bounds = tile_for_gps(tiles, lat, lon)
        if not in_bounds:
            rows.append(_skip_row(f, flight)); continue

        # Simulate a noisy GPS prior: offset the patch center by N(0, σ²) so
        # the drone is NOT at the trivial dead-center of the patch. crc32 hash
        # is stable across processes (unlike builtin hash, which uses a per-
        # process random seed) so per-row offsets are reproducible.
        seed       = zlib.crc32(f"{flight or ''}/{f}".encode())
        dx_m, dy_m = np.random.default_rng(seed).normal(0.0, PRIOR_OFFSET_STD_M, 2)
        mid_lat    = (geo["lt_lat"] + geo["rb_lat"]) / 2
        sx_per_m   = geo["pplon"] / (math.cos(math.radians(mid_lat)) * DEG_TO_M)
        sy_per_m   = geo["pplat"] / DEG_TO_M
        cx        += dx_m * sx_per_m
        cy        += dy_m * sy_per_m

        patch, M = metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw,
                               flight=flight, k_override=k_override)
        if patch is None:
            rows.append(_skip_row(f, flight)); continue
        if clahe_fn:
            patch = clahe_fn(patch)

        # t_match_ms covers everything method-specific: drone-side encoding,
        # matching, and the robust fit inside the factory. One untimed call
        # on the first row that gets here absorbs CUDA/cuDNN/lazy-init cost.
        if not warmed:
            match_factory(drone)(patch)
            warmed = True
        _cuda_sync()
        _t0 = time.perf_counter()
        best = match_factory(drone)(patch)
        _cuda_sync()
        t_match_ms = (time.perf_counter() - _t0) * 1e3
        if best is None:
            rows.append(_skip_row(f, flight)); continue

        # True GT location in patch px: GT gps → satellite px → patch px via
        # M⁻¹. The patch is centred on the *prior* (GT + offset), so the true GT
        # is off-centre; it is the reference for raw_err and the viz pin.
        gt_sat = gps_to_px(lat, lon, geo)
        gt_vec = cv2.invertAffineTransform(M) @ np.array([gt_sat[0], gt_sat[1], 1.0])
        gt_px  = (float(gt_vec[0]), float(gt_vec[1]))
        best["_gt_px"] = gt_px
        # Oracle solvability flag: if the prior offset pushed the true GT outside
        # the searched patch, no matcher can succeed on this row by construction.
        gt_in_patch = (0.0 <= gt_px[0] < SZ_W) and (0.0 <= gt_px[1] < SZ_H)

        m_per_px = metric_m_per_px(height, flight=flight, k_override=k_override)
        best["_m_per_px"] = m_per_px

        raw_pred_px = raw_err_px = raw_err_m = None
        plat = plon = off_m = None
        H = best.get("H")
        if H is not None:
            raw_pred_px, raw_err_px, raw_err_m = _predict_from_H(H, m_per_px, gt_px)
            if best.get("inliers", 0) >= min_inl:
                plat, plon = patch_px_to_gps(raw_pred_px[0], raw_pred_px[1], M, geo)
                off_m = haversine_m(lat, lon, plat, plon)

        r = _build_row(f, lat, lon, height, flight, best,
                       raw_pred_px, raw_err_px, raw_err_m, plat, plon,
                       off_m, m_per_px, gt_in_patch, t_match_ms=t_match_ms)
        rows.append(r)

        # Live A@t over ALL scored rows (gate rejections count as failures) —
        # the same denominator print_summary uses, so the bar matches the log.
        n_scored += 1
        for t in ACC_THRESHOLDS:
            if r[f"success_{t}"]:
                running[t] += 1
        if progress:
            pbar.set_postfix({f"A@{t}": f"{100 * running[t] / n_scored:.0f}%"
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
