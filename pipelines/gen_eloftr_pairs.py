#!/usr/bin/env python
"""Generate teacher-distilled training pairs for Efficient-LoFTR LoRA finetuning.

For each spatial-band drone image we:
  - crop the satellite around the (optionally jittered) GPS prior, exactly like the
    eval pipeline (``metric_crop`` + CLAHE),
  - run the ``roma_extre`` teacher (AerialExtreMatch RoMa, ``weights/roma_extre.pth``)
    to get dense drone<->crop correspondences,
  - filter them (confidence -> one-to-one dedup -> RANSAC inliers),
  - store the crop + filtered correspondences + a geo-homography prior.

``eloftr_lora_train.py`` then LoRA-finetunes ELoFTR to reproduce the teacher's
geometry on these appearance-hard drone<->satellite pairs.

Coordinate frame: both drone image and satellite crop are SZ_W x SZ_H (1024x680);
all stored keypoint coords live in that frame (the same one the student consumes),
and the stored homographies map ``drone_px -> crop_px`` (the convention
``helpers.utils._predict_from_H`` expects: it warps the drone centre into the crop).

Train/test leakage: labels are generated per spatial band via
``helpers.utils.split_flight_rows`` (the SAME split the eval pipelines use). Use
``--split train`` for the LoRA training set and ``--split test`` (usually with
``--no-teacher``) to make a held-out validation set of crops + geo-homographies.

Example (cluster, via slurm/run_gen_pairs.sh):
    python pipelines/gen_eloftr_pairs.py --flights all --teacher extre
    python pipelines/gen_eloftr_pairs.py --flights all --split test --no-teacher \
           --offset-mode jitter --out-dir cache/eloftr_pairs_val
Smoke test:
    python pipelines/gen_eloftr_pairs.py --flights 01 --limit 30 --num-samples 2000
"""

import argparse
import glob
import json
import math
import os
import sys
import zlib
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import (  # noqa: E402
    DEG_TO_M, FLIGHTS_AVAILABLE, PRIOR_OFFSET_STD_M, RANSAC_THRESH,
    SEARCH_FACTOR, SZ_H, SZ_W, _make_clahe, get_flight_paths, gps_to_px,
    load_flight, metric_crop, metric_m_per_px, split_flight_rows, tile_for_gps,
)

# RoMa teacher helpers. Deferred-safe: romatch needs torch+GPU, so a local
# `python -m py_compile` or a --no-teacher run still works via the fallback.
try:
    from pipelines.roma_pipeline import bgr_to_pil, match_roma  # noqa: E402
except Exception:  # pragma: no cover - romatch unavailable locally
    match_roma = None

    def bgr_to_pil(bgr):
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ── teacher (lazy import so --no-teacher / py_compile need no romatch) ─────────

def load_teacher(device, args):
    """Build the RoMa teacher (reusing roma_pipeline.load_model)."""
    import torch
    from pipelines.roma_pipeline import load_model as load_roma
    ns = SimpleNamespace(pretrained=args.teacher, extre_weights=args.extre_weights)
    return load_roma(torch.device(device), ns)


# ── correspondence filtering ──────────────────────────────────────────────────

def dedup_one_to_one(kp0, kp1, conf, q=8.0):
    """Keep at most one (highest-confidence) correspondence per quantized drone
    cell AND per quantized sat cell. q=8 matches ELoFTR's coarse cell size, so the
    survivors map ~one-per-coarse-cell — ideal for the coarse matching GT."""
    order = np.argsort(-conf)
    seen0, seen1, keep = set(), set(), []
    for idx in order:
        c0 = (int(kp0[idx, 0] // q), int(kp0[idx, 1] // q))
        c1 = (int(kp1[idx, 0] // q), int(kp1[idx, 1] // q))
        if c0 in seen0 or c1 in seen1:
            continue
        seen0.add(c0); seen1.add(c1); keep.append(idx)
    keep = np.asarray(keep, dtype=np.int64)
    return kp0[keep], kp1[keep], conf[keep]


def filter_correspondences(res, args):
    """conf threshold (via res['_mask']) -> one-to-one dedup -> MAGSAC inliers.
    Returns (kp0, kp1, conf, H_teacher) with kp in the SZ_W x SZ_H crop frame, or
    (None, ...) when too few survive RANSAC."""
    mask = res["_mask"]
    kp0, kp1, conf = res["_kp0"][mask], res["_kp1"][mask], res["_conf"][mask]
    if len(kp0) >= 4 and not args.no_mutual_nn:
        kp0, kp1, conf = dedup_one_to_one(kp0, kp1, conf)
    if len(kp0) < 4:
        return None, None, None, None
    H, mh = cv2.findHomography(
        kp0.reshape(-1, 1, 2).astype(np.float32),
        kp1.reshape(-1, 1, 2).astype(np.float32),
        cv2.USAC_MAGSAC, args.ransac_thresh, maxIters=5000, confidence=0.9999)
    if H is None or mh is None:
        return None, None, None, None
    inl = mh.ravel().astype(bool)
    kp0, kp1, conf = kp0[inl], kp1[inl], conf[inl]
    if len(kp0) > args.max_correspondences:  # keep the highest-confidence subset
        top = np.argsort(-conf)[: args.max_correspondences]
        kp0, kp1, conf = kp0[top], kp1[top], conf[top]
    return kp0, kp1, conf, H.astype(np.float64)


# ── per-row geometry ──────────────────────────────────────────────────────────

def choose_offset(mode, flight, fname, m_per_px, std_m, margin_px, mix_true_pct):
    """Return (dx_m, dy_m, label) for the GPS-prior offset added to the crop centre.
    `true` = centred on true GPS; `jitter` = eval-style N(0, std) (clamped so the
    true-GPS pixel stays >= margin from the crop edge); `mix` = per-row hash pick."""
    if mode == "mix":
        h = zlib.crc32(f"mix/{flight}/{fname}".encode())
        mode = "true" if (h % 100) < mix_true_pct else "jitter"
    if mode == "true":
        return 0.0, 0.0, "true"
    seed = zlib.crc32(f"{flight}/{fname}".encode())  # eval-compatible seed
    dx_m, dy_m = np.random.default_rng(seed).normal(0.0, std_m, 2)
    max_off_m = (min(SZ_W, SZ_H) / 2.0 - margin_px) * m_per_px  # keep footprint in-frame
    norm = math.hypot(dx_m, dy_m)
    if max_off_m > 0 and norm > max_off_m:
        s = max_off_m / norm
        dx_m, dy_m = dx_m * s, dy_m * s
    return float(dx_m), float(dy_m), "jitter"


def geo_homography(M, geo, lat, lon):
    """Build the geo-prior homography (drone_px -> crop_px) from the crop affine M.

    The crop is metric-isotropic + heading-rotated, so drone->crop is a pure
    similarity: rotation ~0, scale s = drone_GSD/crop_GSD = 1/SEARCH_FACTOR, and the
    drone centre (SZ_W/2, SZ_H/2) maps to the true-GPS pixel inside the crop."""
    cxt, cyt = gps_to_px(lat, lon, geo)            # true GPS in full-sat px
    Minv = cv2.invertAffineTransform(M)            # sat_px -> crop_px (2x3)
    gx = Minv[0, 0] * cxt + Minv[0, 1] * cyt + Minv[0, 2]
    gy = Minv[1, 0] * cxt + Minv[1, 1] * cyt + Minv[1, 2]
    s = 1.0 / SEARCH_FACTOR
    H_geo = np.array([[s, 0.0, gx - s * SZ_W / 2.0],
                      [0.0, s, gy - s * SZ_H / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    return H_geo, (float(gx), float(gy))


# ── per-flight generation ─────────────────────────────────────────────────────

def gen_flight(flight, matcher, device, args):
    _, drone_dir, drone_csv, _ = get_flight_paths(flight)
    df = pd.read_csv(drone_csv)
    df = split_flight_rows(df, which=args.split, test_frac=args.test_frac,
                           axis=args.split_axis, buffer_frac=args.split_buffer)
    if args.limit is not None:
        df = df.iloc[: args.limit]

    out_dir = os.path.join(args.out_dir, flight)
    os.makedirs(out_dir, exist_ok=True)
    clahe_fn = _make_clahe(not args.no_clahe)
    tiles = None
    st = dict(rows=len(df), written=0, skip_no_drone=0, skip_oob=0,
              skip_crop_none=0, dropped_few_corr=0, n_true=0, n_jitter=0,
              n_corr=[], teacher_conf=[], center_disagree_px=[])

    for _, row in tqdm(df.iterrows(), total=len(df), unit="img",
                       desc=f"  flight {flight} [{args.split}]"):
        f = row["filename"]
        lat, lon, height = float(row["lat"]), float(row["lon"]), float(row["height"])
        yaw = float(row["Phi1"]) if "Phi1" in row.index else 0.0
        stem = os.path.splitext(f)[0]
        npz_path = os.path.join(out_dir, stem + ".npz")
        png_path = os.path.join(out_dir, stem + ".png")
        drop_path = os.path.join(out_dir, stem + ".dropped")
        if (os.path.isfile(npz_path) and os.path.isfile(png_path)) or os.path.isfile(drop_path):
            continue  # resumable

        drone = cv2.imread(os.path.join(drone_dir, f))
        if drone is None:
            st["skip_no_drone"] += 1; continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        if clahe_fn:
            drone = clahe_fn(drone)

        if tiles is None:
            tiles = load_flight(flight)[0]
        sat, geo, cx, cy, in_bounds = tile_for_gps(tiles, lat, lon)
        if not in_bounds:
            st["skip_oob"] += 1; continue

        m_per_px = metric_m_per_px(height, flight=flight)
        dx_m, dy_m, label = choose_offset(args.offset_mode, flight, f, m_per_px,
                                          args.jitter_std_m, args.jitter_margin_px,
                                          args.mix_true_pct)
        mid_lat = (geo["lt_lat"] + geo["rb_lat"]) / 2.0
        sx_per_m = geo["pplon"] / (math.cos(math.radians(mid_lat)) * DEG_TO_M)
        sy_per_m = geo["pplat"] / DEG_TO_M
        cx_off, cy_off = cx + dx_m * sx_per_m, cy + dy_m * sy_per_m

        patch, M = metric_crop(sat, geo, cx_off, cy_off, height, yaw_deg=yaw, flight=flight)
        if patch is None:
            st["skip_crop_none"] += 1; continue
        if clahe_fn:
            patch = clahe_fn(patch)

        H_geo, gps_px = geo_homography(M, geo, lat, lon)

        kp0 = kp1 = conf = H_teacher = None
        if not args.no_teacher:
            res = match_roma(bgr_to_pil(drone), patch, matcher, device,
                             args.num_samples, conf_thresh=args.conf_thresh)
            kp0, kp1, conf, H_teacher = filter_correspondences(res, args)
            if kp0 is None or len(kp0) < args.min_correspondences:
                open(drop_path, "w").close()  # marker so resume skips
                st["dropped_few_corr"] += 1; continue
            st["n_corr"].append(len(kp0))
            st["teacher_conf"].append(float(np.median(conf)))
            # teacher-H vs geo-H agreement at the drone centre
            c = np.array([[SZ_W / 2.0, SZ_H / 2.0]], dtype=np.float64).reshape(-1, 1, 2)
            pt_t = cv2.perspectiveTransform(c, H_teacher).reshape(2)
            pt_g = cv2.perspectiveTransform(c, H_geo).reshape(2)
            st["center_disagree_px"].append(float(np.hypot(*(pt_t - pt_g))))

        st["n_true" if label == "true" else "n_jitter"] += 1
        meta = dict(flight=flight, filename=f, stem=stem, split=args.split,
                    offset_mode=label, offset_m=[dx_m, dy_m],
                    n_corr=0 if kp0 is None else int(len(kp0)),
                    median_conf=None if kp0 is None else float(np.median(conf)),
                    m_per_px=float(m_per_px), true_gps_px=gps_px,
                    crop_w=SZ_W, crop_h=SZ_H, clahe=not args.no_clahe,
                    teacher=None if args.no_teacher else args.teacher,
                    num_samples=args.num_samples, conf_thresh=args.conf_thresh,
                    png=os.path.join(flight, stem + ".png"),
                    npz=os.path.join(flight, stem + ".npz"))
        cv2.imwrite(png_path, patch)
        np.savez_compressed(
            npz_path,
            xd=np.empty(0, np.float32) if kp0 is None else kp0[:, 0].astype(np.float32),
            yd=np.empty(0, np.float32) if kp0 is None else kp0[:, 1].astype(np.float32),
            xs=np.empty(0, np.float32) if kp1 is None else kp1[:, 0].astype(np.float32),
            ys=np.empty(0, np.float32) if kp1 is None else kp1[:, 1].astype(np.float32),
            conf=np.empty(0, np.float32) if conf is None else conf.astype(np.float32),
            H_teacher=(np.eye(3) if H_teacher is None else H_teacher).astype(np.float64),
            has_teacher=np.array(H_teacher is not None),
            H_geo=H_geo, M_affine=M.astype(np.float64),
            m_per_px=np.float64(m_per_px),
            true_gps_px=np.array(gps_px, np.float64),
            meta=np.array(json.dumps(meta)))
        st["written"] += 1
    del tiles
    return st


def write_manifest(flight, out_dir):
    """Rebuild manifest.jsonl from the per-row npz meta (resumable, dedup-free)."""
    fdir = os.path.join(out_dir, flight)
    lines = []
    for p in sorted(glob.glob(os.path.join(fdir, "*.npz"))):
        try:
            meta = json.loads(str(np.load(p, allow_pickle=True)["meta"]))
            lines.append(json.dumps(meta))
        except Exception:
            pass
    with open(os.path.join(fdir, "manifest.jsonl"), "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flights", nargs="+", default=["all"])
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="Spatial band: train labels for finetuning, test for the held-out val set.")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows/flight (smoke test).")
    ap.add_argument("--teacher", choices=["extre", "outdoor", "indoor"], default="extre")
    ap.add_argument("--extre-weights", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "roma_extre.pth"))
    ap.add_argument("--num-samples", type=int, default=8000,
                    help="RoMa dense samples per pair (teacher is denser than the 5000 eval default).")
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--ransac-thresh", type=float, default=RANSAC_THRESH)
    ap.add_argument("--min-correspondences", type=int, default=16,
                    help="Drop+log pairs with fewer inlier correspondences (teacher mode).")
    ap.add_argument("--max-correspondences", type=int, default=1024)
    ap.add_argument("--offset-mode", choices=["true", "jitter", "mix"], default="mix")
    ap.add_argument("--mix-true-pct", type=int, default=50)
    ap.add_argument("--jitter-std-m", type=float, default=PRIOR_OFFSET_STD_M)
    ap.add_argument("--jitter-margin-px", type=int, default=64)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--no-mutual-nn", action="store_true", help="Skip one-to-one dedup of teacher matches.")
    ap.add_argument("--no-teacher", action="store_true",
                    help="Skip RoMa; write crop + geo-homography only (fast; for the val set).")
    ap.add_argument("--out-dir", default="cache/eloftr_pairs")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--split-axis", choices=["auto", "lat", "lon"], default="auto")
    ap.add_argument("--split-buffer", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"  Teacher: {'<none>' if args.no_teacher else 'roma_' + args.teacher} | "
          f"split={args.split} | offset={args.offset_mode} | flights={' '.join(flights)}")

    matcher = None
    if not args.no_teacher:
        matcher = load_teacher(args.device, args)

    agg = dict(rows=0, written=0, skip_no_drone=0, skip_oob=0, skip_crop_none=0,
               dropped_few_corr=0, n_true=0, n_jitter=0,
               n_corr=[], teacher_conf=[], center_disagree_px=[])
    per_flight = {}
    for flight in flights:
        st = gen_flight(flight, matcher, args.device, args)
        n_manifest = write_manifest(flight, args.out_dir)
        st["manifest"] = n_manifest
        per_flight[flight] = {k: (len(v) if isinstance(v, list) else v) for k, v in st.items()}
        for k, v in st.items():
            if isinstance(v, list):
                agg[k].extend(v)
            elif k in agg:
                agg[k] += v
        print(f"  flight {flight}: wrote {st['written']}, dropped {st['dropped_few_corr']}, "
              f"manifest {n_manifest} rows")

    summary = dict(
        split=args.split, flights=flights, teacher=None if args.no_teacher else args.teacher,
        offset_mode=args.offset_mode, num_samples=args.num_samples,
        conf_thresh=args.conf_thresh, min_correspondences=args.min_correspondences,
        totals=dict(rows=agg["rows"], written=agg["written"],
                    skip_no_drone=agg["skip_no_drone"], skip_oob=agg["skip_oob"],
                    skip_crop_none=agg["skip_crop_none"],
                    dropped_few_corr=agg["dropped_few_corr"],
                    n_true=agg["n_true"], n_jitter=agg["n_jitter"]),
        n_corr=dict(median=_pct(agg["n_corr"], 50), p10=_pct(agg["n_corr"], 10),
                    p90=_pct(agg["n_corr"], 90)),
        median_teacher_conf=_pct(agg["teacher_conf"], 50),
        center_disagree_px=dict(median=_pct(agg["center_disagree_px"], 50),
                                p90=_pct(agg["center_disagree_px"], 90)),
        per_flight=per_flight)
    with open(os.path.join(args.out_dir, f"qc_summary_{args.split}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n  ── QC summary ──")
    print(json.dumps(summary["totals"], indent=2))
    print(f"  n_corr  median/p10/p90: {summary['n_corr']}")
    print(f"  teacher median conf:    {summary['median_teacher_conf']}")
    print(f"  geo-vs-teacher centre disagreement px: {summary['center_disagree_px']}")
    dropped = agg["dropped_few_corr"]
    seen = max(1, agg["written"] + dropped)
    if dropped / seen > 0.4:
        print(f"  WARN: dropped {100*dropped/seen:.0f}% of rows (< {args.min_correspondences} "
              f"correspondences) — teacher struggling on the appearance gap; consider "
              f"--offset-mode true or a lower --conf-thresh before training.")
    print(f"  Wrote pairs -> {args.out_dir}")


if __name__ == "__main__":
    main()
