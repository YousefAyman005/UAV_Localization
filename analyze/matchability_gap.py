#!/usr/bin/env python3
"""
analyze/matchability_gap.py  —  CLIP retrieval vs. RoMA matchability gap

For N sampled UAV-VisLoc queries, retrieves top-K CLIP candidates and runs
RoMA dense matching on every one.  Logs (clip_rank, clip_sim, inliers,
candidate_dist_m) then produces three diagnostic plots.

Modes
-----
  --collect    GPU phase: build galleries, embed queries, run RoMA × topk
  --plot       CPU phase: read raw CSV → three plots + qualitative panels
  (default)    both phases in sequence

Expected runtime  (A100, topk=50, n_queries=300):  ~40–60 min
Resumable: re-run --collect at any time; completed queries are skipped.

Usage
-----
  python analyze/matchability_gap.py --collect --n-queries 300 --topk 50
  python analyze/matchability_gap.py --plot
  python analyze/matchability_gap.py          # collect then plot
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "pipelines"))

from helpers.utils import (
    SZ_W, SZ_H, RANSAC_THRESH,
    FLIGHTS_AVAILABLE, load_flight, tile_for_gps,
    metric_crop, metric_m_per_px, haversine_m,
    TeeLogger, _make_clahe,
)
from clip_pipeline import (
    load_bundle as load_clip_bundle,
    build_flight_gallery,
    CACHE_DIR, SATCLIP_CKPT,
)

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_CSV_COLS = [
    "query_key", "flight", "filename",
    "gt_lat", "gt_lon", "height",
    "clip_rank", "clip_sim",
    "cand_lat", "cand_lon", "cand_dist_m",
    "inliers", "good_matches",
]

# ── RoMA helpers ──────────────────────────────────────────────────────────────

def _load_roma(device, pretrained):
    try:
        from romatch import roma_outdoor, roma_indoor
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "romatch is not installed in this environment. "
            "Run on the cluster (Apptainer image) where romatch is available, "
            "or use --dry-run to test the CLIP retrieval part locally."
        ) from None
    torch.set_float32_matmul_precision("highest")
    kw = {} if device.type == "cuda" else {"amp_dtype": torch.float32}
    return (roma_outdoor if pretrained == "outdoor" else roma_indoor)(device=device, **kw)


def _bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _run_roma(drone_pil, patch_bgr, matcher, device, num_samples):
    """Return (inliers, good_matches). Swallows errors so one bad patch can't abort the run."""
    try:
        with torch.inference_mode():
            warp, cert = matcher.match(drone_pil, _bgr_to_pil(patch_bgr), device=device)
            pts, c     = matcher.sample(warp, cert, num=num_samples)
            kp_a, kp_b = matcher.to_pixel_coordinates(pts, SZ_H, SZ_W, SZ_H, SZ_W)
        kp0  = kp_a.cpu().numpy().astype(np.float32)
        kp1  = kp_b.cpu().numpy().astype(np.float32)
        good = len(kp0)
        if good < 3:
            return 0, good
        M, mh = cv2.estimateAffinePartial2D(
            kp0, kp1, method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESH,
            maxIters=5000, confidence=0.9999, refineIters=10)
        inliers = int(mh.sum()) if (M is not None and mh is not None) else 0
        return inliers, good
    except Exception:
        return 0, 0


# ── CLIP embed ────────────────────────────────────────────────────────────────

def _embed(drone_bgr, clip_bundle):
    encode, preprocess, _ = clip_bundle
    rgb = cv2.cvtColor(drone_bgr, cv2.COLOR_BGR2RGB)
    t   = preprocess(Image.fromarray(rgb)).unsqueeze(0)
    with torch.inference_mode():
        return F.normalize(encode(t), dim=-1).cpu().numpy().squeeze(0)


# ── Query sampling ────────────────────────────────────────────────────────────

def _sample_queries(flights, n_total, drone_csvs):
    """Deterministic uniform sample across flights, total capped at n_total."""
    rng   = np.random.default_rng(42)
    n_per = max(1, math.ceil(n_total / len(flights)))
    out   = []
    for flight in flights:
        df  = pd.read_csv(drone_csvs[flight])
        idx = rng.choice(len(df), min(n_per, len(df)), replace=False)
        out.extend((flight, df.iloc[int(i)]) for i in sorted(idx))
    return out[:n_total]


# ── Collection phase ──────────────────────────────────────────────────────────

def collect(args, device):
    os.makedirs(args.out_dir, exist_ok=True)
    raw_csv = os.path.join(args.out_dir, "raw_results.csv")

    # Identify already-completed queries so we can resume
    done = set()
    if os.path.isfile(raw_csv):
        prev   = pd.read_csv(raw_csv)
        counts = prev.groupby("query_key")["clip_rank"].count()
        done   = set(counts[counts >= args.topk].index)
        print(f"  Resuming: {len(done)} completed queries already on disk")

    print(f"  Loading CLIP '{args.clip_model}' …", end=" ", flush=True)
    clip_bundle = load_clip_bundle(args.clip_model, device, args.satclip_ckpt)
    print("done")
    if args.dry_run:
        roma = None
        print("  --dry-run: RoMA skipped, all inliers will be 0")
    else:
        print(f"  Loading RoMA ({args.pretrained}) …", end=" ", flush=True)
        roma = _load_roma(device, args.pretrained)
        print("done")

    clahe_fn = _make_clahe(not args.no_clahe)

    # Build / load CLIP gallery and satellite tiles per flight
    print("\nBuilding / loading CLIP galleries …")
    flight_data = {}
    for flight in args.flights:
        emb, _, centers, drone_dir, drone_csv, _, _ = build_flight_gallery(
            flight, clip_bundle, args.tile_size, args.stride,
            args.batch_size, device, args.cache_dir,
            args.clip_model, args.rebuild_cache)
        sat_tiles, _, _, _ = load_flight(flight)
        flight_data[flight] = dict(emb=emb, centers=centers,
                                   drone_dir=drone_dir, drone_csv=drone_csv,
                                   sat_tiles=sat_tiles)

    queries = _sample_queries(
        args.flights, args.n_queries,
        {f: flight_data[f]["drone_csv"] for f in args.flights})
    pending = [(fl, r) for fl, r in queries
               if f"{fl}/{r['filename']}" not in done]
    print(f"\n  {len(pending)} queries pending  "
          f"({args.topk} RoMA calls each = {len(pending) * args.topk:,} total)\n")

    write_header = not os.path.isfile(raw_csv)
    fh = open(raw_csv, "a", buffering=1)
    if write_header:
        fh.write(",".join(RAW_CSV_COLS) + "\n")

    try:
        pbar = tqdm(pending, unit="query", dynamic_ncols=True)
        for flight, row in pbar:
            fname     = row["filename"]
            gt_lat    = float(row["lat"])
            gt_lon    = float(row["lon"])
            height    = float(row["height"])
            yaw       = float(row["Phi1"]) if "Phi1" in row.index else 0.0
            query_key = f"{flight}/{fname}"
            fd        = flight_data[flight]

            drone = cv2.imread(os.path.join(fd["drone_dir"], fname))
            if drone is None:
                continue
            drone = cv2.resize(drone, (SZ_W, SZ_H))
            if clahe_fn:
                drone = clahe_fn(drone)
            drone_pil = _bgr_to_pil(drone)

            sims    = fd["emb"] @ _embed(drone, clip_bundle)
            top_idx = np.argsort(-sims)[:args.topk]

            for rank, cidx in enumerate(top_idx, 1):
                clat   = float(fd["centers"][cidx][0])
                clon   = float(fd["centers"][cidx][1])
                dist_m = haversine_m(gt_lat, gt_lon, clat, clon)

                sat, geo, cx, cy, _ = tile_for_gps(fd["sat_tiles"], clat, clon)
                patch, _ = metric_crop(sat, geo, cx, cy, height,
                                        yaw_deg=yaw, flight=flight)
                if patch is None or args.dry_run:
                    inliers, good = 0, 0
                else:
                    if clahe_fn:
                        patch = clahe_fn(patch)
                    inliers, good = _run_roma(
                        drone_pil, patch, roma, device, args.num_matches)

                fh.write(
                    f"{query_key},{flight},{fname},"
                    f"{gt_lat},{gt_lon},{height},"
                    f"{rank},{sims[cidx]:.6f},"
                    f"{clat:.7f},{clon:.7f},{dist_m:.2f},"
                    f"{inliers},{good}\n")

            pbar.set_postfix(flight=flight)
    finally:
        fh.close()

    print(f"\n  Collection done → {raw_csv}")


# ── Per-query statistics ──────────────────────────────────────────────────────

def _per_query_stats(df):
    """Aggregate per-candidate rows into one row per query."""
    rows = []
    for qk, g in df.groupby("query_key", sort=False):
        c1  = g[g["clip_rank"] == 1].iloc[0]           # CLIP top-1 candidate
        bm  = g.loc[g["inliers"].idxmax()]              # most RoMA-matchable candidate
        # How many candidates beat CLIP top-1 by inlier count?
        n_better = int((g["inliers"] > int(c1["inliers"])).sum())
        rows.append(dict(
            query_key                 = qk,
            flight                    = str(c1["flight"]),
            filename                  = str(c1["filename"]),
            clip_top1_inliers         = int(c1["inliers"]),
            clip_top1_sim             = float(c1["clip_sim"]),
            clip_top1_dist_m          = float(c1["cand_dist_m"]),
            best_inliers              = int(bm["inliers"]),
            best_inlier_clip_rank     = int(bm["clip_rank"]),
            best_matchable_dist_m     = float(bm["cand_dist_m"]),
            matchability_rank_of_top1 = n_better + 1,
            inlier_gap                = int(bm["inliers"]) - int(c1["inliers"]),
            success_25_clip           = float(c1["cand_dist_m"]) <= 25,
            success_25_oracle         = float(bm["cand_dist_m"]) <= 25,
        ))
    return pd.DataFrame(rows)


# ── Plot phase ────────────────────────────────────────────────────────────────

def make_plots(raw_csv, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    df        = pd.read_csv(raw_csv)
    pq        = _per_query_stats(df)
    topk      = int(df["clip_rank"].max())
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Plot 1: histogram — matchability rank of CLIP top-1 ──────────────────
    # For each query, we sort all topk candidates by RoMA inliers descending.
    # This plot shows where the CLIP top-1 candidate ends up in that ranking.
    # Rank 1 = CLIP and RoMA agree. Rank 20 = 19 better-matchable candidates exist.
    pct1 = 100 * (pq["matchability_rank_of_top1"] == 1).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0.5, topk + 1.5, 1)
    ax.hist(pq["matchability_rank_of_top1"], bins=bins,
            color="#4477AA", edgecolor="white", linewidth=0.3)
    ax.axvline(1, color="#CC3311", linestyle="--", linewidth=1.5,
               label=f"Rank 1  ({pct1:.0f}% of queries)")
    ax.set_xlabel("Matchability rank of CLIP top-1 candidate\n"
                  "(rank 1 = CLIP top-1 also has most RoMA inliers)", fontsize=11)
    ax.set_ylabel("Number of queries", fontsize=11)
    ax.set_title("How often does the most visually similar tile = the most geometrically matchable?",
                 fontsize=11)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.legend(fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(plots_dir, "plot1_matchability_rank_hist.png")
    fig.savefig(p1, dpi=150); plt.close(fig)
    print(f"  [Plot 1] {p1}")
    print(f"           CLIP top-1 is also RoMA top-1 in {pct1:.1f}% of queries")

    # ── Plot 2: scatter — CLIP similarity vs. RoMA inlier count ──────────────
    is_top1 = df["clip_rank"] == 1
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(df["clip_sim"], df["inliers"])[0, 1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.loc[~is_top1, "clip_sim"], df.loc[~is_top1, "inliers"],
               s=2, alpha=0.25, color="#AAAAAA", rasterized=True,
               label=f"Candidates 2–{topk}")
    ax.scatter(df.loc[is_top1,  "clip_sim"], df.loc[is_top1,  "inliers"],
               s=18, alpha=0.75, color="#CC3311", zorder=3, label="CLIP top-1")
    ax.set_xlabel("CLIP cosine similarity", fontsize=11)
    ax.set_ylabel("RoMA inlier count", fontsize=11)
    ax.set_title(f"CLIP similarity vs. RoMA inliers  (Pearson r = {r:.3f})", fontsize=12)
    ax.legend(markerscale=3, fontsize=10)
    fig.tight_layout()
    p2 = os.path.join(plots_dir, "plot2_sim_vs_inliers.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [Plot 2] {p2}   (r = {r:.3f})")

    # ── Print summary stats ───────────────────────────────────────────────────
    a25_clip   = 100 * pq["success_25_clip"].mean()
    a25_oracle = 100 * pq["success_25_oracle"].mean()
    gap_pp     = a25_oracle - a25_clip
    print(f"\n  Queries           : {len(pq)}")
    print(f"  A@25m  CLIP top-1 : {a25_clip:.1f}%")
    print(f"  A@25m  RoMA oracle: {a25_oracle:.1f}%")
    print(f"  Matchability gap  : +{gap_pp:.1f}pp")
    print(f"  Queries with inlier gap ≥ 5: "
          f"{100 * (pq['inlier_gap'] >= 5).mean():.0f}%")

    return pq


def save_qualitative(raw_csv, pq, out_dir, n=20, min_inlier_gap=5):
    """Save side-by-side [drone | CLIP top-1 patch | RoMA top-1 patch] panels.

    Each saved filename has a CATEGORY field to fill in manually:
    canopy / repetition / scale / occlusion / other
    """
    import matplotlib.pyplot as plt

    df       = pd.read_csv(raw_csv)
    qual_dir = os.path.join(out_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    gap_cases = (pq[
        (pq["inlier_gap"] >= min_inlier_gap) &
        (pq["matchability_rank_of_top1"] > 1)
    ].nlargest(n, "inlier_gap"))
    print(f"\n  Saving {len(gap_cases)} qualitative panels → {qual_dir}")

    # Lazy-load satellite tiles only for flights we actually need
    sat_cache = {}
    def _load(flight):
        key = f"{int(flight):02d}"
        if key not in sat_cache:
            tiles, drone_dir, _, _ = load_flight(key)
            sat_cache[key] = (tiles, drone_dir)
        return sat_cache[key]

    def _patch(r, sat_tiles, flight, height):
        sat, geo, cx, cy, _ = tile_for_gps(
            sat_tiles, float(r["cand_lat"]), float(r["cand_lon"]))
        patch, _ = metric_crop(sat, geo, cx, cy, height, flight=flight)
        return patch

    saved = 0
    for i, (_, qrow) in enumerate(gap_cases.iterrows(), 1):
        qkey    = qrow["query_key"]
        flight  = f"{int(qrow['flight']):02d}"
        fname   = qrow["filename"]
        sat_tiles, drone_dir = _load(flight)

        drone = cv2.imread(os.path.join(drone_dir, fname))
        if drone is None:
            continue
        drone  = cv2.resize(drone, (SZ_W, SZ_H))
        q_df   = df[df["query_key"] == qkey]
        r_c1   = q_df[q_df["clip_rank"] == 1].iloc[0]
        r_bm   = q_df.loc[q_df["inliers"].idxmax()]
        height = float(r_c1["height"])

        c1_patch = _patch(r_c1, sat_tiles, flight, height)
        bm_patch = _patch(r_bm, sat_tiles, flight, height)
        if c1_patch is None or bm_patch is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        panels = [
            (drone,    f"Drone  [{flight}]\n{fname}"),
            (c1_patch, f"CLIP top-1  (rank 1)\n"
                       f"sim={r_c1['clip_sim']:.3f}  "
                       f"inliers={int(r_c1['inliers'])}  "
                       f"dist={r_c1['cand_dist_m']:.0f} m"),
            (bm_patch, f"RoMA top-1  (CLIP rank {int(r_bm['clip_rank'])})\n"
                       f"sim={r_bm['clip_sim']:.3f}  "
                       f"inliers={int(r_bm['inliers'])}  "
                       f"dist={r_bm['cand_dist_m']:.0f} m"),
        ]
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=9, pad=4)
            ax.axis("off")
        fig.suptitle(
            f"Gap case {i}/{len(gap_cases)}  |  flight {flight}  |  "
            f"inlier gap = {int(qrow['inlier_gap'])}  |  "
            "CATEGORY: ___________",
            fontsize=10, y=1.01)
        fig.tight_layout()
        safe = fname.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(
            qual_dir,
            f"{i:02d}_{flight}_{safe}_gap{int(qrow['inlier_gap'])}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    print(f"  Saved {saved} panels.")
    print("  Rename each file by inserting a category label where CATEGORY is:")
    print("  canopy / repetition / scale / occlusion / other")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="CLIP-retrieval vs. RoMA-matchability gap analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--collect",       action="store_true",
                    help="Run GPU data-collection phase.")
    ap.add_argument("--plot",          action="store_true",
                    help="Run plot/qualitative phase from saved CSV.")
    ap.add_argument("--n-queries",     type=int,  default=300,
                    help="Total queries sampled uniformly across flights.")
    ap.add_argument("--topk",          type=int,  default=50,
                    help="CLIP candidates to verify with RoMA per query.")
    ap.add_argument("--clip-model",    default="clip",
                    choices=["clip","geoclip","satclip","mobileclip","dinov2"])
    ap.add_argument("--pretrained",    default="outdoor",
                    choices=["outdoor","indoor"])
    ap.add_argument("--num-matches",   type=int,  default=5000,
                    help="RoMA sampled correspondences per (drone, patch) pair.")
    ap.add_argument("--tile-size",     type=int,  default=1024)
    ap.add_argument("--stride",        type=int,  default=512)
    ap.add_argument("--batch-size",    type=int,  default=64)
    ap.add_argument("--flights",       nargs="+", default=FLIGHTS_AVAILABLE)
    ap.add_argument("--cache-dir",     default=CACHE_DIR)
    ap.add_argument("--satclip-ckpt",  default=SATCLIP_CKPT)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--no-clahe",      action="store_true")
    ap.add_argument("--out-dir",       default="matchability_gap_results")
    ap.add_argument("--n-qualitative", type=int,  default=20)
    ap.add_argument("--min-inlier-gap",type=int,  default=5,
                    help="Min gap in inliers to include in qualitative output.")
    ap.add_argument("--dry-run",       action="store_true",
                    help="Skip RoMA (records 0 inliers). Useful for testing "
                         "CLIP gallery + retrieval locally without romatch.")
    return ap.parse_args()


def main():
    args    = parse_args()
    do_col  = args.collect or not args.plot
    do_plt  = args.plot    or not args.collect
    raw_csv = os.path.join(args.out_dir, "raw_results.csv")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "matchability_gap.log")

    with TeeLogger(log_path):
        if do_col:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  Device: {device}  |  n_queries: {args.n_queries}  |  "
                  f"topk: {args.topk}  |  CLIP: {args.clip_model}  |  "
                  f"RoMA: {args.pretrained}  |  "
                  f"flights: {' '.join(args.flights)}")
            collect(args, device)

        if do_plt:
            if not os.path.isfile(raw_csv):
                sys.exit(f"ERROR: {raw_csv} not found — run with --collect first.")
            pq = make_plots(raw_csv, args.out_dir)
            save_qualitative(raw_csv, pq, args.out_dir,
                             n=args.n_qualitative,
                             min_inlier_gap=args.min_inlier_gap)


if __name__ == "__main__":
    main()
