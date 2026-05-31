"""Embedding-retrieval baselines: classic CLIP / GeoCLIP / SatCLIP.

Tiles the satellite image into a gallery, embeds every tile, then for each drone
image picks the top-1 tile by cosine similarity. No homography or RANSAC.
"""

import argparse
import math
import multiprocessing
import os
import sys
import time
import zlib

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import (
    SZ_W, SZ_H, DEG_TO_M, PRIOR_OFFSET_STD_M,
    FLIGHTS_AVAILABLE, get_flight_paths, load_satellite, haversine_m, TeeLogger,
)

torch.manual_seed(0)

OUT_CSV_TEMPLATE = "visloc_{model}_results.csv"
CACHE_DIR        = "cache/clip_gallery"
SATCLIP_CKPT     = "weights/satclip-vit16-l40.ckpt"
ACC_THRS         = [5, 10, 15, 20]

MODELS = ("clip", "geoclip", "satclip", "mobileclip", "dinov2")


# ---------- model loaders --------------------------------------------------

def load_clip(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval().to(device)
    return (lambda t: model.encode_image(t.to(device))), preprocess, 512


def load_mobileclip(device, ckpt=None):
    import open_clip
    # Use a local weights file when provided (cluster nodes are offline).
    # Download: huggingface-cli download apple/MobileCLIP-S2-OpenCLIP open_clip_pytorch_model.bin
    pretrained = ckpt if (ckpt and os.path.isfile(ckpt)) else "datacompdr"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP-S2", pretrained=pretrained)
    model.eval().to(device)
    with torch.inference_mode():
        dim = model.encode_image(torch.zeros(1, 3, 256, 256, device=device)).shape[-1]
    return (lambda t: model.encode_image(t.to(device))), preprocess, dim


def load_geoclip(device):
    from geoclip.model.image_encoder import ImageEncoder
    m = ImageEncoder().to(device).eval()
    def preprocess(pil_img):
        t = m.preprocess_image(pil_img)
        return t.squeeze(0) if t.dim() == 4 else t
    return (lambda t: m(t.to(device))), preprocess, 512


def load_satclip(device, ckpt):
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"SatCLIP checkpoint not found at {ckpt}. Download from "
            "https://huggingface.co/microsoft/SatCLIP-ViT16-L40 (file "
            "satclip-vit16-l40.ckpt).")
    # satclip.__init__ imports 'datamodules.s2geo_dataset' which is a training-
    # only dependency not present in the inference container. Inject a stub so
    # the import chain succeeds without the full training environment.
    if "datamodules" not in sys.modules:
        import types as _types
        _stub_pkg  = _types.ModuleType("datamodules")
        _stub_ds   = _types.ModuleType("datamodules.s2geo_dataset")
        class _S2GeoDataModule: pass
        _stub_ds.S2GeoDataModule        = _S2GeoDataModule
        sys.modules["datamodules"]               = _stub_pkg
        sys.modules["datamodules.s2geo_dataset"] = _stub_ds
        setattr(_stub_pkg, "s2geo_dataset", _stub_ds)
    from satclip.load import get_satclip
    from torchvision import transforms
    m = get_satclip(ckpt, device=device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                              std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    with torch.no_grad():
        dim = m.visual(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
    return (lambda t: m.visual(t.to(device))), preprocess, dim


def load_dinov2(device):
    """Self-supervised ViT-B/14 — non-CLIP baseline for retrieval.

    Prefers a locally cached `facebookresearch_dinov2_main` repo under
    $TORCH_HOME/hub/ so cluster compute nodes (no internet) work too. Falls
    back to a network fetch from GitHub when the local copy is absent.
    """
    from torchvision import transforms
    local_repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if os.path.isdir(local_repo):
        model = torch.hub.load(local_repo, "dinov2_vitb14", source="local")
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    with torch.inference_mode():
        dim = model(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
    return (lambda t: model(t.to(device))), preprocess, dim


def load_bundle(name, device, satclip_ckpt, mobileclip_ckpt=None):
    if name == "clip":       return load_clip(device)
    if name == "geoclip":    return load_geoclip(device)
    if name == "satclip":    return load_satclip(device, satclip_ckpt)
    if name == "mobileclip": return load_mobileclip(device, mobileclip_ckpt)
    if name == "dinov2":     return load_dinov2(device)
    raise ValueError(f"unknown model: {name}")


# ---------- tiling & gallery -----------------------------------------------

def iter_tiles(geo, tile_size, stride):
    """Yield (tile_id, x0, y0, lat_center, lon_center) clamped to fit inside the satellite."""
    W, H = geo["w"], geo["h"]
    if W < tile_size or H < tile_size:
        raise ValueError(f"Satellite {W}x{H} smaller than tile {tile_size}")
    xs = list(range(0, W - tile_size + 1, stride))
    ys = list(range(0, H - tile_size + 1, stride))
    if xs[-1] + tile_size < W: xs.append(W - tile_size)
    if ys[-1] + tile_size < H: ys.append(H - tile_size)
    tid = 0
    for y0 in ys:
        for x0 in xs:
            cx, cy = x0 + tile_size / 2, y0 + tile_size / 2
            yield (tid, x0, y0,
                   geo["lt_lat"] - cy / geo["pplat"],
                   geo["lt_lon"] + cx / geo["pplon"])
            tid += 1


def cache_path(cache_dir, model_name, tile_size, stride, sat_tif):
    mtime = int(os.path.getmtime(sat_tif))
    return os.path.join(cache_dir, f"{model_name}_ts{tile_size}_st{stride}_m{mtime}.npz")


def build_gallery(sat, geo, bundle, tile_size, stride, batch_size, device):
    encode, preprocess, dim = bundle
    tiles = list(iter_tiles(geo, tile_size, stride))
    n = len(tiles)
    emb      = np.empty((n, dim), dtype=np.float32)
    centers  = np.empty((n, 2),   dtype=np.float64)
    tile_ids = np.empty(n,        dtype=np.int32)

    for i in tqdm(range(0, n, batch_size), desc="  gallery", unit="batch"):
        batch = tiles[i:i + batch_size]
        imgs = []
        for _tid, x0, y0, _lat, _lon in batch:
            patch = sat[y0:y0 + tile_size, x0:x0 + tile_size]
            imgs.append(preprocess(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))))
        with torch.inference_mode():
            e = F.normalize(encode(torch.stack(imgs)), dim=-1).cpu().numpy()
        for j, (tid, _x0, _y0, lat, lon) in enumerate(batch):
            emb[i + j]      = e[j]
            centers[i + j]  = (lat, lon)
            tile_ids[i + j] = tid
    return emb, tile_ids, centers


def load_or_build_gallery(cpath, build_fn):
    if os.path.isfile(cpath):
        z = np.load(cpath)
        return (z["emb"].astype(np.float32), z["tile_ids"].astype(np.int32),
                z["centers"].astype(np.float64), True)
    emb, ids, centers = build_fn()
    os.makedirs(os.path.dirname(cpath) or ".", exist_ok=True)
    # Save as float32 — keeps cosine similarities exact (was float16 before).
    np.savez_compressed(cpath, emb=emb.astype(np.float32),
                        tile_ids=ids.astype(np.int32),
                        centers=centers.astype(np.float64))
    return emb, ids, centers, False


def build_flight_gallery(flight, bundle, tile_size, stride, batch_size,
                         device, cache_dir, model_name, rebuild_cache):
    """Return (emb, tile_ids, centers, drone_dir, drone_csv, t_gallery, all_cached)."""
    sat_tif, drone_dir, drone_csv, sat_csv = get_flight_paths(flight)
    sat, geo = load_satellite(sat_tif, sat_csv)
    cpath = cache_path(cache_dir, model_name, tile_size, stride, sat_tif)
    if rebuild_cache and os.path.isfile(cpath):
        os.remove(cpath)
    t0 = time.time()
    emb, ids, ctr, was_cached = load_or_build_gallery(
        cpath, lambda: build_gallery(sat, geo, bundle, tile_size, stride,
                                      batch_size, device))
    return emb, ids, ctr, drone_dir, drone_csv, time.time() - t0, was_cached


# ---------- retrieval ------------------------------------------------------

def haversine_m_vec(lat, lon, lats, lons):
    R = 6_371_000.0
    lat_r  = math.radians(lat)
    lats_r = np.radians(lats)
    a = (np.sin((lats_r - lat_r) / 2) ** 2
         + math.cos(lat_r) * np.cos(lats_r) * np.sin(np.radians(lons - lon) / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def retrieve(flight, bundle, emb, tile_ids, centers, df, drone_dir,
             dist, topk, gps_radii=(), flight_tag=None):
    """Run top-k retrieval; returns (rows, floors, t_retrieval).

    For each radius R in `gps_radii`, also computes the GT tile's rank
    within tiles ≤ R m of a noisy prior (GT + N(0, PRIOR_OFFSET_STD_M²)),
    stored as column `gt_rank_r<R>`. Sentinel -1 = GT outside radius."""
    encode, preprocess, _ = bundle
    lats_g, lons_g = centers[:, 0], centers[:, 1]
    topk_col = f"top{topk}_hit"
    rows, floors = [], []
    running, n_valid = {t: 0 for t in ACC_THRS}, 0

    t0 = time.time()
    pbar = tqdm(df.iterrows(), total=len(df), unit="img")
    for _, row in pbar:
        f = row["filename"]
        lat, lon, height = float(row["lat"]), float(row["lon"]), float(row["height"])
        drone = cv2.imread(os.path.join(drone_dir, f))
        if drone is None:
            r = {"filename": f, "skipped": True}
            if flight_tag: r["flight"] = flight_tag
            rows.append(r); continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        rgb = cv2.cvtColor(drone, cv2.COLOR_BGR2RGB)
        t = preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with torch.inference_mode():
            q = F.normalize(encode(t), dim=-1).cpu().numpy().squeeze(0)

        sims = emb @ q
        sorted_idx = np.argsort(-sims)
        top_idx    = sorted_idx[:topk]
        plat, plon = centers[top_idx[0]]
        off_m      = haversine_m(lat, lon, float(plat), float(plon))

        all_dists = haversine_m_vec(lat, lon, lats_g, lons_g)
        floors.append(float(all_dists.min()))
        gt_tile = int(np.argmin(all_dists))
        gt_rank = int(np.where(sorted_idx == gt_tile)[0][0])

        r = {
            "filename": f, "lat": lat, "lon": lon, "height": height,
            "pred_lat":  round(float(plat), 7),
            "pred_lon":  round(float(plon), 7),
            "offset_m":  round(off_m, 2),
            "top1_tile_id": int(tile_ids[top_idx[0]]),
            "top1_sim":     round(float(sims[top_idx[0]]), 4),
            "gt_tile_rank": gt_rank,
            "skipped": False,
            **{f"success_{thr}": off_m <= thr for thr in ACC_THRS},
            topk_col: bool((all_dists[top_idx] <= dist).any()),
        }

        # GPS-degraded variants: rank GT tile within tiles ≤ R m of a noisy
        # prior. Per-row seed matches helpers.utils.collect_pipeline_rows_multitile.
        if gps_radii:
            seed       = zlib.crc32(f"{flight_tag or ''}/{f}".encode())
            dx_m, dy_m = np.random.default_rng(seed).normal(0.0, PRIOR_OFFSET_STD_M, 2)
            prior_lat  = lat + dy_m / DEG_TO_M
            prior_lon  = lon + dx_m / (DEG_TO_M * math.cos(math.radians(lat)))
            prior_d    = haversine_m_vec(prior_lat, prior_lon, lats_g, lons_g)
            for R in gps_radii:
                keep = prior_d <= R
                if not keep.any() or not bool(keep[gt_tile]):
                    r[f"gt_rank_r{R}"] = -1
                else:
                    sub = np.where(keep)[0]
                    order = sub[np.argsort(-sims[sub])]
                    r[f"gt_rank_r{R}"] = int(np.where(order == gt_tile)[0][0])

        if flight_tag: r["flight"] = flight_tag
        rows.append(r)

        n_valid += 1
        for thr in ACC_THRS:
            if r[f"success_{thr}"]: running[thr] += 1
        pbar.set_postfix({f"A@{thr}": f"{100 * running[thr] / n_valid:.0f}%"
                          for thr in ACC_THRS}, refresh=False)

    return rows, np.array(floors), time.time() - t0


# ---------- per-flight runner ---------------------------------------------

def run_flight(flight, bundle, args, model_name, device):
    emb, tile_ids, centers, drone_dir, drone_csv, t_gallery, cached = \
        build_flight_gallery(flight, bundle, args.tile_size, args.stride,
                             args.batch_size, device, args.cache_dir,
                             model_name, args.rebuild_cache)
    print(f"  Gallery: {len(tile_ids)} tiles | dim={emb.shape[1]} | "
          f"{'cached' if cached else f'built in {t_gallery:.1f}s'}")
    df = pd.read_csv(drone_csv)
    if args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)
    rows, floors, t_retrieval = retrieve(
        flight, bundle, emb, tile_ids, centers, df, drone_dir,
        args.dist, args.topk, gps_radii=tuple(args.gps_radii),
        flight_tag=flight)
    return rows, floors, t_gallery, t_retrieval, cached


def print_retrieval_summary(out, dist, label, t_gallery, t_retrieval,
                             floors, cached, topk):
    if out.empty or "skipped" not in out.columns:
        print("\n  No images processed."); return
    v = out[~out["skipped"].fillna(False)]
    if v.empty:
        print("\n  All images skipped."); return
    n = len(v)
    topk_col = f"top{topk}_hit"
    t_hit = int(v[topk_col].fillna(False).sum()) if topk_col in v.columns else 0
    print(f"\n  Results saved to {label}")
    for thr in ACC_THRS:
        col = f"success_{thr}"
        s = int(v[col].fillna(False).sum()) if col in v.columns else 0
        print(f"  A@{thr:2d}m:              {s}/{n} ({100 * s / n:.1f}%)")
    under20 = v[v["offset_m"].fillna(9999) <= 20]["offset_m"]
    if len(under20):
        print(f"  Mean error (≤20m):   {under20.mean():.1f}m")
    print(f"  Top-{topk} within {dist}m:  {t_hit}/{n} ({100 * t_hit / n:.1f}%)")
    if floors.size:
        print(f"  Error floor (grid):  median {np.median(floors):.1f}m  "
              f"P90 {np.percentile(floors, 90):.1f}m")
    print(f"  Offset (all):        mean {v['offset_m'].mean():.1f}m  "
          f"median {v['offset_m'].median():.1f}m  "
          f"P90 {np.percentile(v['offset_m'].dropna(), 90):.1f}m  "
          f"max {v['offset_m'].max():.1f}m")
    succ = v[v["success_20"].fillna(False)] if "success_20" in v else v.iloc[:0]
    fail = v[~v["success_20"].fillna(False)] if "success_20" in v else v
    if len(succ) and len(fail):
        print(f"  Top-1 sim:           A@20 {succ['top1_sim'].mean():.3f}  "
              f"fail {fail['top1_sim'].mean():.3f}")
    print(f"  Median GT tile rank: {v['gt_tile_rank'].median():.0f}")
    if t_gallery is not None and t_retrieval is not None:
        tag = "cached" if cached else f"built {t_gallery:.1f}s"
        print(f"  Time: gallery {tag} | retrieval {t_retrieval:.1f}s "
              f"({1000 * t_retrieval / n:.0f} ms/img)")


# ---------- multi-GPU worker ----------------------------------------------

def _worker(worker_args):
    flight_group, gpu_id, model_name, args = worker_args
    torch.manual_seed(0)
    device = torch.device(f"cuda:{gpu_id}")
    bundle = load_bundle(model_name, device, args.satclip_ckpt, args.mobileclip_ckpt)
    return [(f, *run_flight(f, bundle, args, model_name, device)) for f in flight_group]


# ---------- main -----------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        choices=[*MODELS, "all"], default="all")
    ap.add_argument("--dist",          type=float, default=25.0)
    ap.add_argument("--tile-size",     type=int,   default=1024)
    ap.add_argument("--stride",        type=int,   default=512)
    ap.add_argument("--cache-dir",     type=str,   default=CACHE_DIR)
    ap.add_argument("--out-dir",       type=str,   default=".")
    ap.add_argument("--batch-size",    type=int,   default=64)
    ap.add_argument("--topk",          type=int,   default=5)
    ap.add_argument("--limit",         type=int,   default=None,
                    help="Cap drone images per flight (for quick tests).")
    ap.add_argument("--satclip-ckpt",    type=str, default=SATCLIP_CKPT)
    ap.add_argument("--mobileclip-ckpt", type=str, default=None,
                    help="Local path to MobileCLIP-S2 open_clip_pytorch_model.bin "
                         "(needed on offline cluster nodes). Download with: "
                         "huggingface-cli download apple/MobileCLIP-S2-OpenCLIP "
                         "open_clip_pytorch_model.bin --local-dir weights/")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--flights",       nargs="+", default=["all"])
    ap.add_argument("--gps-radii",     nargs="*", type=int, default=[1000, 5000],
                    help="Radii (m) for GPS-degraded GT-rank columns; pass "
                         "no values to disable.")
    return ap.parse_args()


def main():
    args = parse_args()
    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    n_gpus  = max(1, torch.cuda.device_count())
    print(f"  GPUs: {n_gpus} | Tile: {args.tile_size}px | Stride: {args.stride}px | "
          f"Dist: {args.dist}m | Flights: {' '.join(flights)}")

    models = MODELS if args.model == "all" else (args.model,)
    os.makedirs(args.out_dir, exist_ok=True)

    for mname in models:
        print(f"\n=== {mname.upper()} ===")
        out_csv  = os.path.join(args.out_dir, OUT_CSV_TEMPLATE.format(model=mname))
        log_path = out_csv.replace(".csv", ".log")
        groups   = [g for g in [flights[i::n_gpus] for i in range(n_gpus)] if g]

        with TeeLogger(log_path):
            if len(groups) == 1:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"  Loading {mname} ... ", end="", flush=True)
                bundle = load_bundle(mname, device, args.satclip_ckpt, args.mobileclip_ckpt)
                print("done")
                results = [(f, *run_flight(f, bundle, args, mname, device)) for f in flights]
                del bundle
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            else:
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(len(groups)) as pool:
                    chunks = pool.map(_worker, [(g, i, mname, args)
                                                 for i, g in enumerate(groups)])
                results = [r for chunk in chunks for r in chunk]

            all_rows, all_floors = [], []
            for flight, rows, floors, t_gal, t_ret, cached in results:
                fdf = pd.DataFrame(rows)
                valid = fdf[~fdf["skipped"].fillna(False)]
                if not valid.empty:
                    print(f"\n--- Flight {flight}: {len(fdf)} images ---")
                    print_retrieval_summary(valid, args.dist, f"flight {flight}",
                                             t_gal, t_ret, floors, cached, args.topk)
                all_rows.extend(rows)
                all_floors.append(floors)

            pd.DataFrame(all_rows).to_csv(out_csv, index=False)
            if len(flights) > 1:
                print(f"\n=== {mname.upper()} Overall ({len(flights)} flights) ===")
                valid_all = pd.DataFrame(all_rows)
                valid_all = valid_all[~valid_all["skipped"].fillna(False)]
                if not valid_all.empty:
                    floors_arr = np.concatenate(all_floors) if all_floors else np.array([])
                    # Gallery time only meaningful per-flight; suppress in overall summary.
                    print_retrieval_summary(valid_all, args.dist, out_csv,
                                             None, None, floors_arr, True, args.topk)


if __name__ == "__main__":
    main()
