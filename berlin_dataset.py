#!/usr/bin/env python3
"""
Berlin UAV-Satellite Grid Dataset Generator
============================================
Generates synthetic UAV/satellite image pairs over Berlin using a uniform
grid and Google Maps Static API for a bachelor thesis on UAV localization
via aerial-to-satellite image matching.

Approach:
  - Tile a bounding box over Berlin into a uniform grid
  - Per grid cell: download 1 satellite image + 3 UAV images at altitudes
    50m, 80m, 110m with random offsets from the tile center
  - Ground truth in CSV: tile corners, UAV offset, altitude, zoom levels

Setup:
  1. pip install requests Pillow pandas tqdm numpy
  2. Run:  python berlin_dataset.py --dry-run   (verify grid & CSV first)
          python berlin_dataset.py              (download images)
          python berlin_dataset.py --limit 10   (test with 10 tiles)
"""

import argparse
import io
import math
import os
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


GOOGLE_MAPS_API_KEY = "AIzaSyAfN5_FKHLxoD1r27hJAofNT228bf6LAF8"
RANDOM_SEED = 24

# ---- Grid Parameters (easy to adjust) ------------------------------------
BBOX_LAT_MIN, BBOX_LAT_MAX = 52.46, 52.56
BBOX_LON_MIN, BBOX_LON_MAX = 13.30, 13.46

SAT_ZOOM = 17                # fixed satellite zoom level
OVERLAP_FRACTION = 0.10      # 10% overlap between neighboring tiles

# UAV altitudes (m) -> ground coverage (m), 1.5x ratio (~73deg FOV)
UAV_ALTITUDES = {65: 97, 100: 150, 150: 225}

IMG_SIZE = 1024              # output image size in pixels
REQUEST_SIZE = 640           # Google Maps API request size (scale=2 -> 1280px)
WATERMARK_PX = 50            # watermark strip at bottom of Google Maps images

OUTPUT_DIR = "berlin_grid_dataset"
UAV_DIR = os.path.join(OUTPUT_DIR, "uav")
SAT_DIR = os.path.join(OUTPUT_DIR, "satellite")
CSV_FILENAME = "berlin_grid_dataset.csv"

MAX_WORKERS = 16
PREVIEW_AT = 50
GMAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
MAX_RETRIES = 3


# ---- Coordinate helpers ---------------------------------------------------

def meters_to_deg_lat(m):
    return m / 111320.0


def meters_to_deg_lon(m, lat):
    return m / (111320.0 * math.cos(math.radians(lat)))


def coverage_to_zoom(coverage_m, lat, size_px=256):
    """Calculate Google Maps zoom level for a given ground coverage."""
    meters_per_pixel = coverage_m / size_px
    m_per_px_z0 = (40075016.686 * math.cos(math.radians(lat))) / 256
    zoom = math.log2(m_per_px_z0 / meters_per_pixel)
    return int(round(zoom))


def actual_coverage(zoom, lat):
    """Compute actual ground coverage (meters) of the final 1024x1024 image
    after downloading at scale=2 (1280px), stripping watermark, and center-cropping.
    """
    m_per_px_z0 = (40075016.686 * math.cos(math.radians(lat))) / 256
    m_per_px = m_per_px_z0 / (2 ** zoom)
    # scale=2 halves the effective m/px
    m_per_px_scaled = m_per_px / 2
    return m_per_px_scaled * IMG_SIZE


# ---- Grid generation -------------------------------------------------------

def generate_grid():
    """Generate grid cell centers, row-by-row, left-to-right.

    Cell size is derived from the actual satellite image coverage at SAT_ZOOM.
    Returns list of dicts with tile_id, center coords, and corner coords.
    """
    center_lat = (BBOX_LAT_MIN + BBOX_LAT_MAX) / 2
    cell_size_m = actual_coverage(SAT_ZOOM, center_lat)
    step_m = cell_size_m * (1 - OVERLAP_FRACTION)
    step_lat = meters_to_deg_lat(step_m)
    step_lon = meters_to_deg_lon(step_m, center_lat)
    half_cell_lat = meters_to_deg_lat(cell_size_m / 2)
    half_cell_lon = meters_to_deg_lon(cell_size_m / 2, center_lat)

    # Start from top-left (northwest) so row 00 = northernmost (matrix convention)
    start_lat = BBOX_LAT_MAX - half_cell_lat
    start_lon = BBOX_LON_MIN + half_cell_lon

    grid = []
    row = 0
    lat = start_lat
    while lat >= BBOX_LAT_MIN + half_cell_lat:
        col = 0
        lon = start_lon
        while lon <= BBOX_LON_MAX - half_cell_lon:
            grid.append({
                "row": row,
                "col": col,
                "tile_id": f"{row:02d}_{col:02d}",
                "center_lat": lat,
                "center_lon": lon,
                "tl_lat": lat + half_cell_lat,
                "tl_lon": lon - half_cell_lon,
                "br_lat": lat - half_cell_lat,
                "br_lon": lon + half_cell_lon,
            })
            col += 1
            lon = start_lon + col * step_lon
        row += 1
        lat = start_lat - row * step_lat

    return grid


# ---- Image download -------------------------------------------------------

def fetch_google_image(lat, lon, coverage_m, size_px=1024):
    """Download a satellite image from Google Maps Static API.

    Requests 640x640 at scale=2 (1280x1280), strips the watermark (~50px),
    then center-crops to size_px.
    """
    zoom = coverage_to_zoom(coverage_m, lat, REQUEST_SIZE)

    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{REQUEST_SIZE}x{REQUEST_SIZE}",
        "scale": 2,
        "maptype": "satellite",
        "key": GOOGLE_MAPS_API_KEY,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(GMAPS_URL, params=params, timeout=30)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type:
                raise ValueError(f"Expected image, got {content_type}: {resp.text[:200]}")

            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img = img.crop((0, 0, img.width, img.height - WATERMARK_PX))
            left = (img.width - size_px) // 2
            top = (img.height - size_px) // 2
            img = img.crop((left, top, left + size_px, top + size_px))
            return img, zoom

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                raise


# ---- CSV helpers -----------------------------------------------------------

def load_existing_csv(csv_path):
    """Load existing CSV and return set of completed tile IDs."""
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
        completed = df.groupby("tile_id").size()
        return set(completed[completed >= len(UAV_ALTITUDES)].index)
    return set()


def save_csv_rows(csv_path, rows):
    """Append rows to CSV, creating with header if needed."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# ---- Preview ---------------------------------------------------------------

def generate_preview_html(records, output_path):
    """Generate HTML preview showing UAV/satellite pairs."""
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Berlin Grid Dataset Preview</title>",
        "<style>",
        "body{font-family:sans-serif;background:#1a1a1a;color:#eee;margin:20px}",
        "h1{text-align:center}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(640px,1fr));gap:16px}",
        ".pair{background:#2a2a2a;border-radius:8px;padding:12px;display:flex;gap:12px;align-items:center}",
        ".pair img{border-radius:4px;object-fit:cover}",
        ".sat-img{width:300px;height:300px}",
        ".uav-img{width:250px;height:250px}",
        ".label{font-size:12px;text-align:center;margin-top:4px;color:#aaa}",
        ".meta{font-size:11px;color:#888;min-width:120px}",
        ".col{display:flex;flex-direction:column;align-items:center}",
        "</style></head><body>",
        f"<h1>Berlin Grid Dataset Preview ({len(records)} pairs)</h1>",
        "<div class='grid'>",
    ]

    for rec in records:
        uav_rel = os.path.relpath(
            os.path.join(UAV_DIR, rec["uav_filename"]),
            os.path.dirname(output_path),
        )
        sat_rel = os.path.relpath(
            os.path.join(SAT_DIR, rec["satellite_filename"]),
            os.path.dirname(output_path),
        )
        html_parts.append(
            f"<div class='pair'>"
            f"<div class='col'><img class='sat-img' src='{sat_rel}'>"
            f"<div class='label'>Satellite z{rec['satellite_zoom_level']}</div></div>"
            f"<div class='col'><img class='uav-img' src='{uav_rel}'>"
            f"<div class='label'>UAV {rec['uav_altitude']}m (z{rec['uav_zoom_level']})</div></div>"
            f"<div class='meta'>"
            f"tile: {rec['tile_id']}<br>"
            f"offset: ({rec['offset_east_m']:.0f}, {rec['offset_north_m']:.0f})m<br>"
            f"alt: {rec['uav_altitude']}m<br>"
            f"coverage: {rec['uav_coverage_m']}m"
            f"</div></div>"
        )

    html_parts.append("</div></body></html>")
    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))
    print(f"Preview saved to {output_path}")
    webbrowser.open(f"file://{os.path.abspath(output_path)}")


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Berlin UAV-Satellite grid dataset from Google Maps"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate CSV and print summary without downloading")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N tiles (for testing)")
    args = parser.parse_args()

    os.makedirs(UAV_DIR, exist_ok=True)
    os.makedirs(SAT_DIR, exist_ok=True)

    # ---- Generate grid --------------------------------------------------------
    grid = generate_grid()
    if args.limit > 0:
        grid = grid[:args.limit]

    total_tiles = len(grid)
    uavs_per_tile = len(UAV_ALTITUDES)
    total_pairs = total_tiles * uavs_per_tile

    # Compute actual satellite coverage from zoom level
    sample_lat = (BBOX_LAT_MIN + BBOX_LAT_MAX) / 2
    sat_actual_cov = actual_coverage(SAT_ZOOM, sample_lat)
    step_m = sat_actual_cov * (1 - OVERLAP_FRACTION)

    # ---- RNG for UAV offsets ---------------------------------------------------
    offset_rng = np.random.default_rng(RANDOM_SEED)

    # ---- Load existing progress ------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, CSV_FILENAME)
    already_done = load_existing_csv(csv_path)

    # ---- Build schedule --------------------------------------------------------
    altitudes_sorted = sorted(UAV_ALTITUDES.keys())
    schedule = []

    for cell in grid:
        tile_id = cell["tile_id"]
        sat_filename = f"tile_{tile_id}_sat.png"

        uav_entries = []
        for alt in altitudes_sorted:
            uav_coverage = UAV_ALTITUDES[alt]

            # Random offset — UAV center stays within satellite image,
            # but UAV footprint may spill into neighbor tiles
            max_offset = sat_actual_cov / 2 - 20  # small margin from edge
            offset_east = float(offset_rng.uniform(-max_offset, max_offset))
            offset_north = float(offset_rng.uniform(-max_offset, max_offset))

            uav_lat = cell["center_lat"] + meters_to_deg_lat(offset_north)
            uav_lon = cell["center_lon"] + meters_to_deg_lon(offset_east, cell["center_lat"])

            uav_filename = f"tile_{tile_id}_uav_{alt}m.png"
            uav_zoom = coverage_to_zoom(uav_coverage, cell["center_lat"], REQUEST_SIZE)

            uav_entries.append({
                "tile_id": tile_id,
                "tile_center_lat": round(cell["center_lat"], 6),
                "tile_center_lon": round(cell["center_lon"], 6),
                "tile_top_left_lat": round(cell["tl_lat"], 6),
                "tile_top_left_lon": round(cell["tl_lon"], 6),
                "tile_bottom_right_lat": round(cell["br_lat"], 6),
                "tile_bottom_right_lon": round(cell["br_lon"], 6),
                "tile_ground_size_meters": round(sat_actual_cov, 2),
                "uav_id": f"tile_{tile_id}_uav_{alt}m",
                "uav_center_lat": round(uav_lat, 6),
                "uav_center_lon": round(uav_lon, 6),
                "uav_altitude": alt,
                "uav_coverage_m": uav_coverage,
                "offset_east_m": round(offset_east, 2),
                "offset_north_m": round(offset_north, 2),
                "satellite_filename": sat_filename,
                "uav_filename": uav_filename,
                "satellite_zoom_level": SAT_ZOOM,
                "uav_zoom_level": uav_zoom,
            })

        schedule.append({
            "tile_id": tile_id,
            "sat_filename": sat_filename,
            "center_lat": cell["center_lat"],
            "center_lon": cell["center_lon"],
            "uav_entries": uav_entries,
        })

    # ---- Print plan ------------------------------------------------------------
    n_rows = max(c["row"] for c in grid) + 1 if grid else 0
    n_cols = max(c["col"] for c in grid) + 1 if grid else 0
    todo = [e for e in schedule if e["tile_id"] not in already_done]
    skipped = len(schedule) - len(todo)
    total_api = len(todo) * (1 + uavs_per_tile)

    print(f"Grid: {n_rows} rows x {n_cols} cols = {total_tiles} tiles")
    print(f"  Cell size: {sat_actual_cov:.0f}m, overlap: {OVERLAP_FRACTION*100:.0f}%, step: {step_m:.0f}m")
    print(f"  Satellite zoom: z{SAT_ZOOM} (actual coverage: {sat_actual_cov:.0f}m)")
    print(f"  UAV altitudes: {altitudes_sorted}")
    print(f"  UAVs per tile: {uavs_per_tile}, total pairs: {total_pairs}")
    print(f"Output:")
    print(f"  Satellite: {os.path.abspath(SAT_DIR)}/")
    print(f"  UAV:       {os.path.abspath(UAV_DIR)}/")
    print(f"  CSV:       {csv_path}")
    print(f"  To download: {len(todo)} tiles ({total_api} API calls)")
    print(f"  Skipped (resume): {skipped}")
    print()

    if args.dry_run:
        # Show sample CSV rows
        if schedule:
            sample = schedule[0]["uav_entries"][:2]
            print("Sample CSV rows:")
            for row in sample:
                print(f"  {row}")
        print("\n[DRY RUN] No images downloaded.")
        return

    # ---- Download --------------------------------------------------------------
    def process_tile(tile_entry):
        """Download 1 satellite + 3 UAV images for a grid cell."""
        sat_path = os.path.join(SAT_DIR, tile_entry["sat_filename"])

        sat_img, _ = fetch_google_image(
            tile_entry["center_lat"], tile_entry["center_lon"],
            sat_actual_cov, IMG_SIZE,
        )
        sat_img.save(sat_path)

        rows = []
        for uav in tile_entry["uav_entries"]:
            uav_img, _ = fetch_google_image(
                uav["uav_center_lat"], uav["uav_center_lon"],
                uav["uav_coverage_m"], IMG_SIZE,
            )
            uav_img.save(os.path.join(UAV_DIR, uav["uav_filename"]))
            rows.append(uav)

        return tile_entry["tile_id"], rows

    downloaded = 0
    errors = 0
    error_details = []
    preview_records = []
    csv_buffer = []

    if todo:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_tile, e): e for e in todo}
            with tqdm(total=len(todo), desc="Downloading tiles") as pbar:
                for future in as_completed(futures):
                    tile_entry = futures[future]
                    try:
                        _, rows = future.result()
                        csv_buffer.extend(rows)
                        if len(preview_records) < PREVIEW_AT * uavs_per_tile:
                            preview_records.extend(rows)
                        downloaded += 1
                        if len(csv_buffer) >= 20:
                            save_csv_rows(csv_path, csv_buffer)
                            csv_buffer = []
                    except Exception as e:
                        errors += 1
                        error_details.append(f"{tile_entry['tile_id']}: {e}")
                        tqdm.write(f"  ERROR {tile_entry['tile_id']}: {e}")
                    pbar.update(1)

    save_csv_rows(csv_path, csv_buffer)

    # ---- Summary ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Grid:               {n_rows} x {n_cols} = {total_tiles} tiles")
    print(f"UAVs per tile:      {uavs_per_tile}")
    print(f"Total pairs:        {total_pairs}")
    print(f"Downloaded:         {downloaded}")
    print(f"Errors:             {errors}")
    print(f"Skipped (resume):   {skipped}")
    print()
    print("Altitudes:")
    for alt in altitudes_sorted:
        cov = UAV_ALTITUDES[alt]
        uav_z = coverage_to_zoom(cov, sample_lat, REQUEST_SIZE)
        uav_actual = actual_coverage(uav_z, sample_lat)
        print(f"  {alt:>3d}m:  {cov}m nominal, z{uav_z} ({uav_actual:.0f}m actual)")
    print(f"\nSatellite: z{SAT_ZOOM}, {sat_actual_cov:.0f}m actual, {IMG_SIZE}x{IMG_SIZE}px")
    print(f"Images: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"CSV:    {csv_path}")

    if error_details:
        print(f"\nFailed ({len(error_details)}) -- re-run to retry:")
        for e in error_details[:10]:
            print(f"  {e}")

    if preview_records:
        preview_path = os.path.join(script_dir, "preview.html")
        generate_preview_html(preview_records, preview_path)

    if downloaded > 0:
        uav_n = len([f for f in os.listdir(UAV_DIR) if f.endswith(".png")])
        sat_n = len([f for f in os.listdir(SAT_DIR) if f.endswith(".png")])
        print(f"\nVerification:")
        print(f"  Satellite files: {sat_n}")
        print(f"  UAV files:       {uav_n}")


if __name__ == "__main__":
    main()
