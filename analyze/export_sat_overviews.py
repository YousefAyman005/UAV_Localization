"""Export a downsampled PNG overview of every flight's full satellite TIF.

One PNG per flight (long side --long-side px, default 4000) plus a
sat_overviews_meta.csv with the native dimensions and ground resolution,
so figure captions can state the true map size. Must run inside the
container with the dataset bound (cv2 + DATAPOOL3).

Usage:
    python analyze/export_sat_overviews.py --out /data/job_results/sat_overviews
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from helpers import utils as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", nargs="+", default=["all"])
    ap.add_argument("--long-side", type=int, default=4000,
                    help="output long side in px (native TIFs are 10-30k px)")
    ap.add_argument("--out", default="sat_overviews")
    args = ap.parse_args()

    flights = (U.FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights)
    os.makedirs(args.out, exist_ok=True)
    meta = []
    for fl in flights:
        tiles, _, _, _ = U.load_flight(fl)
        sat, geo = tiles[0]
        h, w = sat.shape[:2]
        s = args.long_side / max(w, h)
        small = cv2.resize(sat, (round(w * s), round(h * s)),
                           interpolation=cv2.INTER_AREA) if s < 1.0 else sat
        out_png = os.path.join(args.out, f"satellite{fl}_overview.png")
        cv2.imwrite(out_png, small, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        # native ground resolution from the geo headers (m per satellite px)
        mid_lat = (geo["lt_lat"] + geo["rb_lat"]) / 2
        mx = math.cos(math.radians(mid_lat)) * U.DEG_TO_M / abs(geo["pplon"])
        my = U.DEG_TO_M / abs(geo["pplat"])
        meta.append({"flight": fl, "width_px": w, "height_px": h,
                     "m_per_px_x": round(mx, 3), "m_per_px_y": round(my, 3),
                     "ground_w_km": round(w * mx / 1000, 2),
                     "ground_h_km": round(h * my / 1000, 2),
                     "overview_scale": round(min(s, 1.0), 4)})
        print(f"  {fl}: {w}x{h} px ({mx:.2f} m/px) -> {out_png}", flush=True)

    with open(os.path.join(args.out, "sat_overviews_meta.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(meta[0].keys()))
        wr.writeheader()
        wr.writerows(meta)
    print(f"Wrote {len(meta)} overviews + sat_overviews_meta.csv to {args.out}")


if __name__ == "__main__":
    main()
