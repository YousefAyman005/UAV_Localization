"""Figure: drone image vs. its corresponding satellite crop (the season/sensor gap).

For each requested flight, one or more clean two-panel figures:
  (left)  the drone image at working resolution (raw, no CLAHE), and
  (right) the satellite basemap cropped to the SAME ground footprint and rotated
          to the drone's heading, so the two panels show the identical scene in
          the identical orientation.

The only thing that differs between the two panels is *when/how* they were
captured: the drone frame carries a capture date (its season is printed), while
the satellite is a single static basemap from a different, unlabeled time. The
figure is meant to motivate the drone->satellite domain gap.

Same-footprint trick: the pipeline's search patch spans SEARCH_FACTOR * K * alt
meters (1.75x wider than the drone footprint, to leave room for the GPS prior).
Passing k_override = K / SEARCH_FACTOR makes metric_crop span exactly K * alt --
the true drone footprint. The crop is centred on the TRUE GPS (no prior offset),
so it lines up with the drone image. Reuses helpers.utils, so the crop/yaw are
exactly the pipeline's -- must run inside the container (cv2 etc.).

--per-flight N samples N evenly-spaced frames per flight (walking to the nearest
frame whose footprint crop is accepted), so one job yields several candidates.

Usage (dataset bound at the default DATASET_DIR location):
    python analyze/plot_season_pair.py --flights 08 11 --per-flight 3
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from helpers import utils as U  # noqa: E402

# Northern-hemisphere meteorological seasons (all flights are in China).
_SEASON = {12: "winter", 1: "winter", 2: "winter",
           3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer",
           9: "autumn", 10: "autumn", 11: "autumn"}


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def region_for(sat_csv, flight):
    """Human-readable region name for a flight, or '' if unavailable."""
    try:
        rows = pd.read_csv(sat_csv)
        m = rows[rows["mapname"] == f"satellite{flight}.tif"]
        return "" if m.empty else str(m.iloc[0]["region"])
    except Exception:
        return ""


def season_of(date_str):
    """('2018-10-23', 'autumn') from an ISO-ish date field; ('', '') on failure."""
    day = str(date_str).split("T")[0]
    try:
        return day, _SEASON.get(int(day.split("-")[1]), "")
    except (IndexError, ValueError):
        return day, ""


def crop_for(row, flight, tiles, no_yaw_cal):
    """(patch, yaw) for a drone row's footprint, or (None, yaw) if unusable."""
    lat, lon, height = float(row["lat"]), float(row["lon"]), float(row["height"])
    yaw = float(row["Phi1"])
    if not no_yaw_cal:
        yaw = U.corrected_yaw(flight, yaw)
    sat, geo, cx, cy, in_bounds = U.tile_for_gps(tiles, lat, lon)
    if not in_bounds:
        return None, yaw
    k_flight = U.K_PER_FLIGHT.get(str(flight), U.K_DEFAULT)
    patch, _ = U.metric_crop(sat, geo, cx, cy, height, yaw_deg=yaw,
                             flight=flight, k_override=k_flight / U.SEARCH_FACTOR)
    return patch, yaw


def choose_indices(df, flight, tiles, n, no_yaw_cal):
    """N evenly-spaced row indices, each walked outward to the nearest frame
    whose footprint crop is accepted (avoids map-edge / rejected frames)."""
    N = len(df)
    targets = [int((k + 0.5) * N / n) for k in range(n)]
    chosen = []
    for t in targets:
        for i in sorted(range(N), key=lambda i: abs(i - t)):
            if i in chosen:
                continue
            patch, _ = crop_for(df.iloc[i], flight, tiles, no_yaw_cal)
            if patch is not None:
                chosen.append(i)
                break
    return sorted(set(chosen))


def build_pair(flight, row, tiles, drone_dir, no_yaw_cal):
    """Return (drone_bgr, patch_bgr, yaw, height, date, season) at SZ_W x SZ_H.

    Both panels share the same footprint / orientation / pixel grid, so the
    split panels and the composite are always the identical two images."""
    fname = row["filename"]
    patch, yaw = crop_for(row, flight, tiles, no_yaw_cal)
    assert patch is not None, f"{fname}: satellite crop rejected"
    height = float(row["height"])
    date, season = season_of(row["date"]) if "date" in row.index else ("", "")
    drone = cv2.imread(os.path.join(drone_dir, fname))
    assert drone is not None, f"{fname}: drone image not found"
    drone = cv2.resize(drone, (U.SZ_W, U.SZ_H), interpolation=cv2.INTER_AREA)
    return drone, patch, yaw, height, date, season


def write_panels(flight, row, region, tiles, drone_dir, no_yaw_cal, out_dir, stem):
    """Save the two panels as standalone, undecorated full-crop PNGs."""
    drone, patch, yaw, height, date, season = build_pair(
        flight, row, tiles, drone_dir, no_yaw_cal)
    os.makedirs(out_dir, exist_ok=True)
    d_path = os.path.join(out_dir, f"drone_{stem}.png")
    s_path = os.path.join(out_dir, f"sat_{stem}.png")
    cv2.imwrite(d_path, drone)
    cv2.imwrite(s_path, patch)
    print(f"wrote drone_{stem}.png + sat_{stem}.png  "
          f"({row['filename']}, {date or 'no-date'} {season}, {U.SZ_W}x{U.SZ_H}, "
          f"yaw {yaw:.1f} deg, alt {height:.0f} m, region '{region}')")


def render_pair(flight, row, region, tiles, drone_dir, no_yaw_cal, out_dir, basename):
    fname = row["filename"]
    drone, patch, yaw, height, date, season = build_pair(
        flight, row, tiles, drone_dir, no_yaw_cal)

    fig = plt.figure(figsize=(11.0, 11.0 * (U.SZ_H / U.SZ_W) / 2 + 1.1))
    gs = fig.add_gridspec(1, 2, wspace=0.03,
                          left=0.005, right=0.995, top=0.86, bottom=0.02)
    ax_d, ax_s = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_d.imshow(rgb(drone))
    date_txt = f" - {date}" if date else ""
    season_txt = f" ({season})" if season else ""
    ax_d.set_title(f"Drone image{date_txt}{season_txt}", fontsize=11)
    ax_s.imshow(rgb(patch))
    ax_s.set_title("Satellite basemap (same footprint)", fontsize=11)
    for ax in (ax_d, ax_s):
        ax.set_xticks([]); ax.set_yticks([])
    region_txt = f"{region}  -  " if region else ""
    fig.suptitle(f"Flight {flight}  -  {region_txt}drone view vs. satellite reference",
                 fontsize=12, y=0.97)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{basename}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {basename}.png/pdf  ({fname}, {date or 'no-date'} {season}, "
          f"yaw {yaw:.1f} deg, alt {height:.0f} m, region '{region}')")


def make_flight(flight, index, per_flight, frame_ids, split, no_yaw_cal, out_dir):
    tiles, drone_dir, drone_csv, sat_csv = U.load_flight(flight)
    df = pd.read_csv(drone_csv)
    region = region_for(sat_csv, flight)

    stems = [os.path.splitext(str(f))[0] for f in df["filename"]]
    if frame_ids:
        want = set(frame_ids)
        idxs = [i for i, s in enumerate(stems) if s in want]
        single = False
    elif index is not None:
        idxs, single = [index], False
    else:
        idxs = choose_indices(df, flight, tiles, per_flight, no_yaw_cal)
        single = (per_flight == 1)

    for i in idxs:
        row, stem = df.iloc[i], stems[i]
        if split:
            write_panels(flight, row, region, tiles, drone_dir, no_yaw_cal, out_dir, stem)
        else:
            basename = f"season_pair_{flight}" if single else f"season_pair_{stem}"
            render_pair(flight, row, region, tiles, drone_dir, no_yaw_cal, out_dir, basename)

    del tiles  # free the big satellite array before the next flight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", nargs="+",
                    default=["01", "02", "03", "04", "05", "06", "08", "10", "11"],
                    help="flights to render drone/satellite pairs for")
    ap.add_argument("--per-flight", type=int, default=1,
                    help="how many evenly-spaced frames to render per flight")
    ap.add_argument("--index", type=int, default=None,
                    help="exact drone CSV row (single flight only; overrides "
                         "--per-flight sampling)")
    ap.add_argument("--frame-ids", nargs="+", default=None,
                    help="explicit drone frame stems (e.g. 10_0025) to render; "
                         "overrides --index/--per-flight")
    ap.add_argument("--split", action="store_true",
                    help="write the two panels as standalone drone_<id>.png and "
                         "sat_<id>.png (no titles) instead of the composite")
    ap.add_argument("--no-yaw-cal", action="store_true",
                    help="use raw Phi1 instead of the calibrated heading")
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args()

    for flight in args.flights:
        make_flight(flight, args.index, args.per_flight, args.frame_ids,
                    args.split, args.no_yaw_cal, args.out)


if __name__ == "__main__":
    main()
