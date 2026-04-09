#!/usr/bin/env python3
"""
Berlin UAV-Satellite Image Pair Dataset Generator
==================================================
Generates synthetic UAV/satellite image pairs over Berlin using
Google Maps Static API (~0.3m/pixel) for a bachelor thesis on UAV
localization via aerial-to-satellite image matching.

Approach:
  - Download 1,000 satellite images (600m coverage, 1024x1024)
  - For each satellite, download 3 UAV images at random altitudes,
    with random offsets (NOT centered) and random rotation
  - Total: 3,000 UAV-satellite pairs from 4,000 API calls
  - Ground truth in CSV: offset in meters + rotation angle

Setup:
  1. pip install requests Pillow pandas tqdm numpy
  2. Run:  python berlin_dataset.py --dry-run   (verify CSV first)
          python berlin_dataset.py              (download images)
"""

import argparse
import io
import math
import os
import time
import webbrowser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


GOOGLE_MAPS_API_KEY = "AIzaSyD0qT6_bqBNj3vwNCjOeu2yXng9LDF__NA"  # set via .env or env var
RANDOM_SEED = 24  # for reproducibility
# ---- Constants -------------------------------------------------------------
# Urban core of Berlin (excludes rural outskirts and farmland)
BERLIN_LON_MIN, BERLIN_LON_MAX = 13.25, 13.55
BERLIN_LAT_MIN, BERLIN_LAT_MAX = 52.44, 52.57

# Altitude (m) -> ground coverage (m) for the UAV crop
ALTITUDE_COVERAGE = {80: 120, 100: 150, 150: 220, 200: 300}

SAT_COVERAGE_M = 600       # satellite crop ground coverage
IMG_SIZE = 1024             # output image size in pixels

NUM_SATELLITE = 1000        # full dataset
UAVS_PER_SAT = 3           # UAV images per satellite
TOTAL_PAIRS = NUM_SATELLITE * UAVS_PER_SAT  # 3,000

OUTPUT_DIR = "berlin_uav_dataset"
UAV_DIR = os.path.join(OUTPUT_DIR, "uav")
SAT_DIR = os.path.join(OUTPUT_DIR, "satellite")
CSV_FILENAME = "berlin_pairs.csv"

MAX_WORKERS = 16           # parallel download threads
RATE_LIMIT_EVERY = 50
RATE_LIMIT_SLEEP = 0.5
PREVIEW_AT = 50             # preview after this many satellite images

# Google Maps Static API
GMAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
MAX_RETRIES = 3


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


def fetch_google_image(lat, lon, coverage_m, size_px=1024):
    """Download a satellite image from Google Maps Static API.

    Requests 640x640 at scale=2 (1280x1280), strips the watermark (~50px),
    then center-crops to size_px — no resize, no quality loss.
    """
    request_size = 640
    zoom = coverage_to_zoom(coverage_m, lat, request_size)

    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{request_size}x{request_size}",
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
            # Crop watermark (~50px at scale=2) then center-crop to target size
            img = img.crop((0, 0, img.width, img.height - 50))
            left = (img.width - size_px) // 2
            top = (img.height - size_px) // 2
            img = img.crop((left, top, left + size_px, top + size_px))
            return img

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                raise


def rotate_uav_image(img, angle_deg):
    """Rotate image by angle_deg without black corners.

    Achieves this by rotating without expand (corners wrap to background),
    then center-cropping to the inscribed square — the largest axis-aligned
    square that fits inside the rotated circle.  The caller is responsible for
    fetching at sqrt(2) * coverage_m so that inscribed-square area still
    represents the intended ground footprint.
    """
    if abs(angle_deg) < 0.5:
        # No rotation: crop to inscribed square of the (already oversized) image
        w, h = img.size
        target = int(round(min(w, h) / math.sqrt(2)))
        left = (w - target) // 2
        top = (h - target) // 2
        img = img.crop((left, top, left + target, top + target))
        return img.resize((w, h), Image.LANCZOS)

    w, h = img.size
    # Rotate in place (no expand) — corners become background (black), but
    # we'll crop them away entirely in the next step.
    rotated = img.rotate(angle_deg, resample=Image.BICUBIC, expand=False)
    # Inscribed square side for a w×h image rotated by angle_deg:
    #   side = w / (|cos θ| + |sin θ|)   (for square images where w == h)
    theta = math.radians(angle_deg % 90)  # symmetry: worst case at 45°
    side = int(w / (math.cos(theta) + math.sin(theta)))
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = rotated.crop((left, top, left + side, top + side))
    return cropped.resize((w, h), Image.LANCZOS)



def generate_points(rng, n):
    """Generate n random (lat, lon) points within Berlin bbox."""
    lats = rng.uniform(BERLIN_LAT_MIN, BERLIN_LAT_MAX, n)
    lons = rng.uniform(BERLIN_LON_MIN, BERLIN_LON_MAX, n)
    return lats, lons


def load_existing_csv(csv_path):
    """Load existing CSV and return set of already-done satellite names."""
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
        return set(df["sat_image"].tolist())
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


def generate_preview_html(records, output_path):
    """Generate HTML preview showing UAV/satellite pairs."""
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Berlin UAV Dataset Preview</title>",
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
        f"<h1>Berlin UAV Dataset Preview ({len(records)} pairs)</h1>",
        "<div class='grid'>",
    ]

    for rec in records:
        uav_rel = os.path.relpath(
            os.path.join(UAV_DIR, rec["uav_image"] + ".png"),
            os.path.dirname(output_path),
        )
        sat_rel = os.path.relpath(
            os.path.join(SAT_DIR, rec["sat_image"] + ".png"),
            os.path.dirname(output_path),
        )
        alt = rec["altitude_m"]
        cov = rec["uav_coverage_m"]

        html_parts.append(
            f"<div class='pair'>"
            f"<div class='col'><img class='sat-img' src='{sat_rel}'>"
            f"<div class='label'>Satellite {SAT_COVERAGE_M}m</div></div>"
            f"<div class='col'><img class='uav-img' src='{uav_rel}'>"
            f"<div class='label'>UAV alt{alt}m ({cov}m)</div></div>"
            f"<div class='meta'>"
            f"{rec['uav_image']}<br>"
            f"offset: ({rec['offset_east_m']:.0f}, {rec['offset_north_m']:.0f})m<br>"
            f"rotation: {rec['rotation_deg']:.1f}&deg;<br>"
            f"from: {rec['sat_image']}"
            f"</div></div>"
        )

    html_parts.append("</div></body></html>")
    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))
    print(f"Preview saved to {output_path}")
    webbrowser.open(f"file://{os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Berlin UAV-Satellite dataset from Google Maps"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate CSV and print summary without downloading")
    args = parser.parse_args()

    os.makedirs(UAV_DIR, exist_ok=True)
    os.makedirs(SAT_DIR, exist_ok=True)

    # ---- Generate satellite center points ----------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    lats, lons = generate_points(rng, NUM_SATELLITE)

    # ---- RNGs for UAV params -----------------------------------------------
    offset_rng = np.random.default_rng(RANDOM_SEED + 1000)
    alt_rng = np.random.default_rng(RANDOM_SEED + 2000)
    rot_rng = np.random.default_rng(RANDOM_SEED + 3000)
    altitudes_list = sorted(ALTITUDE_COVERAGE.keys())

    # ---- Load existing progress --------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, CSV_FILENAME)
    already_done = load_existing_csv(csv_path)

    # ---- Build full schedule -----------------------------------------------
    schedule = []
    for sat_idx in range(NUM_SATELLITE):
        sat_lat = float(lats[sat_idx])
        sat_lon = float(lons[sat_idx])
        sat_name = f"berlin_sat_{sat_idx + 1:04d}"

        uav_entries = []
        for uav_j in range(UAVS_PER_SAT):
            alt = int(alt_rng.choice(altitudes_list))
            uav_coverage = ALTITUDE_COVERAGE[alt]

            # Random offset: UAV must be fully within satellite coverage
            max_offset = (SAT_COVERAGE_M - uav_coverage) / 2
            offset_east = float(offset_rng.uniform(-max_offset, max_offset))
            offset_north = float(offset_rng.uniform(-max_offset, max_offset))

            uav_lat = sat_lat + meters_to_deg_lat(offset_north)
            uav_lon = sat_lon + meters_to_deg_lon(offset_east, sat_lat)

            # Random rotation: 0-360 degrees
            rotation = float(rot_rng.uniform(0, 360))

            uav_name = f"berlin_uav_{sat_idx + 1:04d}_{uav_j + 1}_alt{alt}"

            uav_entries.append({
                "uav_image": uav_name,
                "sat_image": sat_name,
                "sat_latitude": round(sat_lat, 6),
                "sat_longitude": round(sat_lon, 6),
                "uav_latitude": round(uav_lat, 6),
                "uav_longitude": round(uav_lon, 6),
                "offset_east_m": round(offset_east, 2),
                "offset_north_m": round(offset_north, 2),
                "altitude_m": alt,
                "uav_coverage_m": uav_coverage,
                "sat_coverage_m": SAT_COVERAGE_M,
                "rotation_deg": round(rotation, 1),
            })

        schedule.append({
            "sat_name": sat_name,
            "sat_lat": sat_lat,
            "sat_lon": sat_lon,
            "uav_entries": uav_entries,
        })

    # ---- Print plan --------------------------------------------------------
    to_download = sum(1 for s in schedule if s["sat_name"] not in already_done)
    total_api = to_download + to_download * UAVS_PER_SAT
    print(f"Output directories:")
    print(f"  UAV:       {os.path.abspath(UAV_DIR)}/")
    print(f"  Satellite: {os.path.abspath(SAT_DIR)}/")
    print(f"  CSV:       {csv_path}")
    print(f"  Satellite images: {NUM_SATELLITE} ({to_download} to download)")
    print(f"  UAV images per satellite: {UAVS_PER_SAT}")
    print(f"  Total pairs: {TOTAL_PAIRS}")
    if not args.dry_run:
        print(f"  API calls needed: {total_api} ({to_download} sat + {to_download * UAVS_PER_SAT} uav)")
    print()

    # ---- Download ----------------------------------------------------------
    downloaded_sat = 0
    skipped = 0
    errors = 0
    error_details = []
    preview_records = []
    csv_buffer = []
    api_calls = 0

    def process_satellite(sat_entry):
        sat_name = sat_entry["sat_name"]
        sat_path = os.path.join(SAT_DIR, sat_name + ".png")

        sat_img = fetch_google_image(
            sat_entry["sat_lat"], sat_entry["sat_lon"],
            SAT_COVERAGE_M, IMG_SIZE,
        )
        sat_img.save(sat_path)

        rows = []
        for uav_entry in sat_entry["uav_entries"]:
            uav_img = fetch_google_image(
                uav_entry["uav_latitude"], uav_entry["uav_longitude"],
                uav_entry["uav_coverage_m"] * math.sqrt(2), IMG_SIZE,
            )
            uav_img = rotate_uav_image(uav_img, uav_entry["rotation_deg"])
            uav_img.save(os.path.join(UAV_DIR, uav_entry["uav_image"] + ".png"))
            rows.append({k: v for k, v in uav_entry.items() if not k.startswith("_")})

        return sat_name, rows

    todo = [
        e for e in schedule
        if not (e["sat_name"] in already_done and
                os.path.exists(os.path.join(SAT_DIR, e["sat_name"] + ".png")))
    ]
    skipped = len(schedule) - len(todo)

    if not args.dry_run and todo:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_satellite, e): e for e in todo}
            with tqdm(total=len(todo), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    sat_entry = futures[future]
                    sat_name = sat_entry["sat_name"]
                    try:
                        _, rows = future.result()
                        csv_buffer.extend(rows)
                        if len(preview_records) < PREVIEW_AT * UAVS_PER_SAT:
                            preview_records.extend(rows)
                        downloaded_sat += 1
                        api_calls += 1 + UAVS_PER_SAT
                        if len(csv_buffer) >= 20:
                            save_csv_rows(csv_path, csv_buffer)
                            csv_buffer = []
                    except Exception as e:
                        errors += 1
                        error_details.append(f"{sat_name}: {e}")
                        tqdm.write(f"  ERROR {sat_name}: {e}")
                    pbar.update(1)

    if not args.dry_run:
        save_csv_rows(csv_path, csv_buffer)

    # ---- Summary -----------------------------------------------------------
    alt_counts = Counter()
    for s in schedule:
        for u in s["uav_entries"]:
            alt_counts[u["altitude_m"]] += 1

    print("\n" + "=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Satellite images:   {NUM_SATELLITE}")
    print(f"UAV per satellite:  {UAVS_PER_SAT}")
    print(f"Total pairs:        {TOTAL_PAIRS}")
    if not args.dry_run:
        print(f"Sat downloaded:     {downloaded_sat}")
        print(f"API calls made:     {api_calls}")
        print(f"Errors:             {errors}")
    else:
        print(f"To download:        {to_download}")
    print(f"Skipped (resume):   {skipped}")
    print()
    print("By altitude (randomly assigned):")
    for alt in altitudes_list:
        cov = ALTITUDE_COVERAGE[alt]
        print(f"  {alt:>3d}m:  ~{alt_counts[alt]} pairs, {cov}m coverage")
    print(f"\nSatellite: {SAT_COVERAGE_M}m, {IMG_SIZE}x{IMG_SIZE}px")
    print(f"UAV: random offset + rotation, {IMG_SIZE}x{IMG_SIZE}px")
    print(f"Images: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"CSV:    {csv_path}")

    if error_details:
        print(f"\nFailed ({len(error_details)}) — re-run to retry:")
        for e in error_details[:10]:
            print(f"  {e}")

    if args.dry_run:
        print("\n[DRY RUN] No images downloaded.")

    if not args.dry_run and downloaded_sat > 0:
        uav_n = len([f for f in os.listdir(UAV_DIR) if f.endswith(".png")])
        sat_n = len([f for f in os.listdir(SAT_DIR) if f.endswith(".png")])
        print(f"\nVerification:")
        print(f"  UAV files:       {uav_n}")
        print(f"  Satellite files: {sat_n}")


if __name__ == "__main__":
    main()
