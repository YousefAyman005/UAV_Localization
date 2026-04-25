import math
import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None  # allow large satellite TIFs

MIN_INL       = 10
CROP_W        = 2048    # base satellite crop width (px)
SZ_W, SZ_H   = 1024, 680
CROP_H        = CROP_W * SZ_H // SZ_W   # 1360; keeps scale_x == scale_y
SCALES        = [0.5, 0.75, 1.0, 1.25, 1.5]
RANSAC_THRESH = 5.0
TOP_MATCHES   = 50
JPEG_QUALITY  = 85

_HERE             = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR       = os.path.join(_HERE, "UAV_VisLoc_dataset")
FLIGHTS_AVAILABLE = [f"{i:02d}" for i in range(1, 12)]
_UAV_HFOV_DEG     = 70.0


class TeeLogger:
    """Mirrors stdout to a log file."""
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


def load_satellite(sat_tif, sat_csv):
    print(f"Loading {sat_tif} ... ", end="", flush=True)
    img = cv2.cvtColor(np.array(Image.open(sat_tif).convert("RGB")), cv2.COLOR_RGB2BGR)
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
    return int((lon-g["lt_lon"])*g["pplon"]), int((g["lt_lat"]-lat)*g["pplat"])


def crop_sat(sat, cx, cy, g, crop_w, crop_h):
    """Crop crop_w×crop_h centred at (cx,cy) and resize to SZ_W×SZ_H."""
    if not (0 <= cx < g["w"] and 0 <= cy < g["h"]): return None
    x0, y0 = cx - crop_w//2, cy - crop_h//2
    xc, yc = max(0, x0), max(0, y0)
    patch = sat[yc:min(g["h"], y0+crop_h), xc:min(g["w"], x0+crop_w)]
    ph, pw = patch.shape[:2]
    if ph != crop_h or pw != crop_w:
        patch = cv2.copyMakeBorder(patch, yc-y0, crop_h-ph-(yc-y0),
                                   xc-x0, crop_w-pw-(xc-x0), cv2.BORDER_REFLECT)
    return cv2.resize(patch, (SZ_W, SZ_H))


def pred_offset_m(H, cx, cy, crop_w, crop_h, geo, lat, lon):
    """Return (offset_m, pred_lat, pred_lon) from homography, or None."""
    if H is None: return None
    px_c, py_c = cv2.perspectiveTransform(
        np.float32([[SZ_W/2, SZ_H/2]]).reshape(-1,1,2), H).reshape(2)
    plat = geo["lt_lat"] - ((cy-crop_h/2) + py_c*(crop_h/SZ_H)) / geo["pplat"]
    plon = geo["lt_lon"] + ((cx-crop_w/2) + px_c*(crop_w/SZ_W)) / geo["pplon"]
    return haversine_m(lat, lon, plat, plon), plat, plon


def altitude_scales(height_m, geo):
    """SCALES sorted by proximity to the altitude-predicted footprint (best first)."""
    target_s = (2*height_m*math.tan(math.radians(_UAV_HFOV_DEG/2))
                / (math.cos(math.radians((geo["lt_lat"]+geo["rb_lat"])/2)) * 111_320 / geo["pplon"])
                / CROP_W)
    return sorted(SCALES, key=lambda s: abs(s-target_s))


def scale_sweep(sat, cx, cy, geo, height_m, match_fn, clahe_fn=None, early_stop_inliers=None):
    """Iterate altitude-prioritised scales; return (best_r, (crop_w, crop_h), patch)."""
    best, best_crop, best_patch = None, None, None
    for s in altitude_scales(height_m, geo):
        crop_w, crop_h = max(SZ_W, int(CROP_W*s)), max(SZ_H, int(CROP_H*s))
        p = crop_sat(sat, cx, cy, geo, crop_w, crop_h)
        if p is None: continue
        if clahe_fn is not None: p = clahe_fn(p)
        r = match_fn(p)
        if best is None or r["inliers"] > best["inliers"]:
            best, best_crop, best_patch = r, (crop_w, crop_h), p
            if early_stop_inliers is not None and best["inliers"] >= early_stop_inliers: break
    return best, best_crop, best_patch


def apply_clahe_lab(bgr, clahe):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def make_result_row(filename, lat, lon, height, r, best_crop, off_m, plat, plon, success):
    _r = lambda x, n: round(x, n) if x is not None else None
    return dict(filename=filename, lat=lat, lon=lon, height=height, skipped=False,
                crop_w=best_crop[0], crop_h=best_crop[1],
                sat_kp=r["sat_kp"], drone_kp=r["drone_kp"],
                raw=r["raw"], good=r["good"], inliers=r["inliers"],
                inlier_ratio=round(r["inliers"]/r["good"], 4) if r["good"] else 0,
                pred_lat=_r(plat, 7), pred_lon=_r(plon, 7), offset_m=_r(off_m, 2),
                success=success)


def draw_and_save(drone, kpd, patch, kps, matches, filename, viz_dir):
    viz = cv2.drawMatches(drone, kpd, patch, kps, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(viz_dir, f"{os.path.splitext(filename)[0]}_matches.jpg"),
                viz, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def save_dense_viz(drone, patch, best, filename, viz_dir):
    """Viz for dense matchers exposing _kp0/_kp1/_conf/_mask."""
    if best.get("_mask") is None or best.get("good", 0) <= 0: return
    kp0, kp1, conf, mask = best["_kp0"], best["_kp1"], best["_conf"], best["_mask"]
    kpd = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0[mask]]
    kps = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1[mask]]
    top = sorted([cv2.DMatch(i, i, 1.0-c) for i, c in enumerate(conf[mask])],
                 key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, kpd, patch, kps, top, filename, viz_dir)


# ---------------------------------------------------------------------------
# Flight 09 support — 4-tile mosaic
# ---------------------------------------------------------------------------

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
    """Load flight-09 tiles; return [(img, geo), ...] in row-major order."""
    row = pd.read_csv(sat_csv); row = row[row["mapname"] == "satellite09.tif"].iloc[0]
    lt_lat, lt_lon, rb_lat, rb_lon = row["LT_lat_map"], row["LT_lon_map"], row["RB_lat_map"], row["RB_lon_map"]

    rc_to_path  = {_parse_tile_rc(p): p for p in tile_paths}
    unique_rows = sorted({rc[0] for rc in rc_to_path})
    unique_cols = sorted({rc[1] for rc in rc_to_path})
    size_map    = {rc: Image.open(p).size for rc, p in rc_to_path.items()}

    pplat = sum(size_map[(r, unique_cols[0])][1] for r in unique_rows) / (lt_lat - rb_lat)
    pplon = sum(size_map[(unique_rows[0], c)][0] for c in unique_cols) / (rb_lon - lt_lon)

    tiles, y_off = [], 0
    for r in unique_rows:
        x_off = 0
        for c in unique_cols:
            tw, th = size_map[(r, c)]
            img = cv2.cvtColor(np.array(Image.open(rc_to_path[(r,c)]).convert("RGB")), cv2.COLOR_RGB2BGR)
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


def load_flight(flight, dataset_dir=None):
    """Return (tiles, drone_dir, drone_csv, sat_csv); tiles is always [(img, geo), ...]."""
    if flight == "09":
        tp, drone_dir, drone_csv, sat_csv = get_flight09_tile_paths(dataset_dir)
        return load_flight09_tiles(tp, sat_csv), drone_dir, drone_csv, sat_csv
    sat_tif, drone_dir, drone_csv, sat_csv = get_flight_paths(flight, dataset_dir)
    sat, geo = load_satellite(sat_tif, sat_csv)
    return [(sat, geo)], drone_dir, drone_csv, sat_csv


def collect_pipeline_rows_multitile(tiles, df, match_factory, dist, min_inl=MIN_INL,
                                     clahe=None, viz_fn=None, viz_dir=None,
                                     drone_dir=None, flight=None, progress=True):
    """Routes each drone image to its tile by GPS, runs scale-sweep matching."""
    if drone_dir is None: raise ValueError("drone_dir is required")
    clahe_fn = (lambda p: apply_clahe_lab(p, clahe)) if clahe is not None else None
    if viz_fn is not None and viz_dir is not None: os.makedirs(viz_dir, exist_ok=True)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), unit="img", disable=not progress):
        f = row["filename"]
        lat, lon, height = float(row["lat"]), float(row["lon"]), float(row["height"])
        drone = cv2.imread(os.path.join(drone_dir, f))
        if drone is None: rows.append(_skip_row(f, flight)); continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        if clahe_fn is not None: drone = clahe_fn(drone)

        sat, geo, cx, cy = None, None, None, None
        for t_sat, t_geo in tiles:
            _cx, _cy = gps_to_px(lat, lon, t_geo)
            if 0 <= _cx < t_geo["w"] and 0 <= _cy < t_geo["h"]:
                sat, geo, cx, cy = t_sat, t_geo, _cx, _cy; break
        if sat is None:
            best_d = float("inf")
            for t_sat, t_geo in tiles:
                _cx, _cy = gps_to_px(lat, lon, t_geo)
                d = max(0,-_cx)+max(0,_cx-t_geo["w"])+max(0,-_cy)+max(0,_cy-t_geo["h"])
                if d < best_d: best_d, sat, geo, cx, cy = d, t_sat, t_geo, _cx, _cy

        best, best_crop, patch = scale_sweep(sat, cx, cy, geo, height,
                                             match_factory(drone), clahe_fn=clahe_fn)
        if best is None: rows.append(_skip_row(f, flight)); continue
        off = pred_offset_m(best["H"], cx, cy, *best_crop, geo, lat, lon) if best["inliers"] >= min_inl else None
        off_m, plat, plon = off if off else (None, None, None)
        r = make_result_row(f, lat, lon, height, best, best_crop,
                            off_m, plat, plon, off_m is not None and off_m <= dist)
        if flight is not None: r["flight"] = flight
        if viz_fn is not None: viz_fn(drone, patch, best, f, viz_dir)
        rows.append(r)
    return rows


def print_summary(v, dist, label, min_inl=MIN_INL):
    if v.empty: print("\n  All images skipped."); return
    n        = len(v)
    accepted = v[v["inliers"] >= min_inl]
    succeeded = v[v["success"].fillna(False)]
    s, h     = len(succeeded), len(accepted)
    fp   = accepted[~accepted["success"].fillna(False)]
    print(f"\n  Results saved to {label}")
    print(f"  Success (≤{dist}m):    {s}/{n} ({100*s/n:.1f}%)")
    print(f"  Homography accepted:    {h}/{n} ({100*h/n:.1f}%)")
    if h: print(f"  Incorrect matches:      {len(fp)}/{h} ({100*len(fp)/h:.1f}%) — offset > {dist}m")
    if s:
        print(f"  Offset (successes):     mean {succeeded['offset_m'].mean():.1f}m  "
              f"median {succeeded['offset_m'].median():.1f}m  max {succeeded['offset_m'].max():.1f}m")
    print(f"  Median inliers: {v['inliers'].median():.0f} | ratio: {v['inlier_ratio'].median():.3f}")
