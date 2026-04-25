import argparse
import math
import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from visloc_utils import (
    SZ_W, SZ_H,
    FLIGHTS_AVAILABLE, get_flight_paths, get_flight09_tile_paths, load_flight09_tiles,
    load_satellite, haversine_m, TeeLogger,
)

OUT_CSV_TEMPLATE = "visloc_{model}_results.csv"
CACHE_DIR        = "cache/clip_gallery"
SATCLIP_CKPT     = "weights/satclip-vit16-l40.ckpt"

MODELS = ("clip", "geoclip", "satclip")


# ---------- model loaders --------------------------------------------------

def load_clip(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval().to(device)
    def encode(t):
        return model.encode_image(t.to(device))
    return encode, preprocess, 512


def load_geoclip(device):
    from geoclip.model.image_encoder import ImageEncoder
    m = ImageEncoder().to(device).eval()
    def preprocess(pil_img):
        t = m.preprocess_image(pil_img)
        return t.squeeze(0) if t.dim() == 4 else t
    def encode(t):
        return m(t.to(device))
    return encode, preprocess, 512


def load_satclip(device, ckpt):
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"SatCLIP checkpoint not found at {ckpt}. Download from "
            f"https://huggingface.co/microsoft/SatCLIP-ViT16-L40 (file "
            f"satclip-vit16-l40.ckpt) and place it there, or pass "
            f"--satclip-ckpt <path>."
        )
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
    def encode(t):
        return m.visual(t.to(device))
    return encode, preprocess, dim


# ---------- tiling ---------------------------------------------------------

def iter_tiles(geo, tile_size, stride):
    """Yield (tile_id, x0, y0, lat_center, lon_center). Positions clamped so
    every tile fits fully inside the satellite — no edge padding needed."""
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
            lat = geo["lt_lat"] - cy / geo["pplat"]
            lon = geo["lt_lon"] + cx / geo["pplon"]
            yield tid, x0, y0, lat, lon
            tid += 1


# ---------- gallery (cached embeddings) ------------------------------------

def cache_path(cache_dir, model_name, tile_size, stride, sat_tif):
    mtime = int(os.path.getmtime(sat_tif))
    return os.path.join(cache_dir,
                        f"{model_name}_ts{tile_size}_st{stride}_m{mtime}.npz")


def build_gallery(sat, geo, bundle, tile_size, stride, batch_size, device):
    encode, preprocess, dim = bundle
    tiles = list(iter_tiles(geo, tile_size, stride))
    n = len(tiles)
    emb      = np.empty((n, dim),  dtype=np.float32)
    centers  = np.empty((n, 2),    dtype=np.float64)
    tile_ids = np.empty(n,         dtype=np.int32)

    for i in tqdm(range(0, n, batch_size), desc="  gallery", unit="batch"):
        batch = tiles[i:i + batch_size]
        imgs = []
        for tid, x0, y0, lat, lon in batch:
            patch = sat[y0:y0 + tile_size, x0:x0 + tile_size]
            rgb   = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            imgs.append(preprocess(Image.fromarray(rgb)))
        t = torch.stack(imgs)
        with torch.inference_mode():
            e = F.normalize(encode(t), dim=-1).cpu().numpy()
        for j, (tid, x0, y0, lat, lon) in enumerate(batch):
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
    np.savez_compressed(cpath,
                        emb=emb.astype(np.float16),
                        tile_ids=ids.astype(np.int32),
                        centers=centers.astype(np.float64))
    return emb, ids, centers, False


# ---------- retrieval ------------------------------------------------------

def haversine_m_vec(lat, lon, lats, lons):
    R = 6_371_000.0
    lat_r = math.radians(lat)
    lats_r = np.radians(lats)
    dlat = lats_r - lat_r
    dlon = np.radians(lons - lon)
    a = np.sin(dlat / 2) ** 2 + math.cos(lat_r) * np.cos(lats_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def run_model(model_name, bundle, sat, geo, df, dist,
              tile_size, stride, cache_dir, sat_tif, topk, batch_size, device,
              rebuild_cache, drone_dir, flight=None):
    topk_col = f"top{topk}_hit"
    """Run retrieval for one flight; return (rows, floors, t_gallery, t_retrieval, cached)."""
    cpath = cache_path(cache_dir, model_name, tile_size, stride, sat_tif)
    if rebuild_cache and os.path.isfile(cpath):
        os.remove(cpath)

    t0 = time.time()
    emb, tile_ids, centers, cached = load_or_build_gallery(
        cpath, lambda: build_gallery(sat, geo, bundle, tile_size, stride,
                                     batch_size, device))
    t_gallery = time.time() - t0
    print(f"  Gallery: {len(tile_ids)} tiles | dim={emb.shape[1]} | "
          f"{'cached' if cached else f'built in {t_gallery:.1f}s'}")

    encode, preprocess, _ = bundle
    lats_g, lons_g = centers[:, 0], centers[:, 1]
    rows, floors = [], []
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        f   = row["filename"]
        lat = float(row["lat"]); lon = float(row["lon"])
        height = float(row["height"])
        drone = cv2.imread(os.path.join(drone_dir, f))
        if drone is None:
            r = dict(filename=f, skipped=True)
            if flight is not None:
                r["flight"] = flight
            rows.append(r)
            continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        rgb = cv2.cvtColor(drone, cv2.COLOR_BGR2RGB)
        t = preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with torch.inference_mode():
            q = F.normalize(encode(t), dim=-1).cpu().numpy().squeeze(0)

        sims = emb @ q
        sorted_idx = np.argsort(-sims)
        top_idx    = sorted_idx[:topk]
        plat, plon = centers[top_idx[0]]
        off_m = haversine_m(lat, lon, float(plat), float(plon))

        all_dists = haversine_m_vec(lat, lon, lats_g, lons_g)
        floors.append(float(all_dists.min()))
        top_dists = all_dists[top_idx]
        gt_tile   = int(np.argmin(all_dists))
        gt_rank   = int(np.where(sorted_idx == gt_tile)[0][0])

        r = dict(
            filename=f, lat=lat, lon=lon, height=height,
            pred_lat=round(float(plat), 7),
            pred_lon=round(float(plon), 7),
            offset_m=round(off_m, 2),
            top1_tile_id=int(tile_ids[top_idx[0]]),
            top1_sim=round(float(sims[top_idx[0]]), 4),
            gt_tile_rank=gt_rank,
            success=off_m <= dist,
            skipped=False,
        )
        r[topk_col] = bool((top_dists <= dist).any())
        if flight is not None:
            r["flight"] = flight
        rows.append(r)
    t_retrieval = time.time() - t0
    return rows, np.array(floors), t_gallery, t_retrieval, cached


def run_model_multitile(model_name, bundle, tile_paths, sat_csv, df, dist,
                        tile_size, stride, cache_dir, topk, batch_size, device,
                        rebuild_cache, drone_dir, flight=None):
    """Flight-09 variant: build one gallery per tile, merge them, then retrieve."""
    topk_col = f"top{topk}_hit"
    tiles_data = load_flight09_tiles(tile_paths, sat_csv)

    t0 = time.time()
    all_cached = True
    emb_parts, ids_parts, ctr_parts = [], [], []
    id_offset = 0
    for (sat, geo), tif_path in zip(tiles_data, tile_paths):
        cpath = cache_path(cache_dir, model_name, tile_size, stride, tif_path)
        if rebuild_cache and os.path.isfile(cpath):
            os.remove(cpath)
        emb_t, ids_t, ctr_t, was_cached = load_or_build_gallery(
            cpath, lambda s=sat, g=geo: build_gallery(s, g, bundle, tile_size,
                                                       stride, batch_size, device))
        if not was_cached:
            all_cached = False
        emb_parts.append(emb_t)
        ids_parts.append(ids_t + id_offset)
        ctr_parts.append(ctr_t)
        id_offset += len(ids_t)
    emb      = np.concatenate(emb_parts)
    tile_ids = np.concatenate(ids_parts)
    centers  = np.concatenate(ctr_parts)
    t_gallery = time.time() - t0
    print(f"  Gallery (4 tiles merged): {len(tile_ids)} tiles | dim={emb.shape[1]} | "
          f"{'all cached' if all_cached else f'built in {t_gallery:.1f}s'}")

    encode, preprocess, _ = bundle
    lats_g, lons_g = centers[:, 0], centers[:, 1]
    rows, floors = [], []
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        f   = row["filename"]
        lat = float(row["lat"]); lon = float(row["lon"])
        height = float(row["height"])
        drone = cv2.imread(os.path.join(drone_dir, f))
        if drone is None:
            r = dict(filename=f, skipped=True)
            if flight is not None:
                r["flight"] = flight
            rows.append(r)
            continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        rgb = cv2.cvtColor(drone, cv2.COLOR_BGR2RGB)
        t = preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with torch.inference_mode():
            q = F.normalize(encode(t), dim=-1).cpu().numpy().squeeze(0)

        sims = emb @ q
        sorted_idx = np.argsort(-sims)
        top_idx    = sorted_idx[:topk]
        plat, plon = centers[top_idx[0]]
        off_m = haversine_m(lat, lon, float(plat), float(plon))

        all_dists = haversine_m_vec(lat, lon, lats_g, lons_g)
        floors.append(float(all_dists.min()))

        top_dists = all_dists[top_idx]
        gt_tile   = int(np.argmin(all_dists))
        gt_rank   = int(np.where(sorted_idx == gt_tile)[0][0])

        r = dict(
            filename=f, lat=lat, lon=lon, height=height,
            pred_lat=round(float(plat), 7),
            pred_lon=round(float(plon), 7),
            offset_m=round(off_m, 2),
            top1_tile_id=int(tile_ids[top_idx[0]]),
            top1_sim=round(float(sims[top_idx[0]]), 4),
            gt_tile_rank=gt_rank,
            success=off_m <= dist,
            skipped=False,
        )
        r[topk_col] = bool((top_dists <= dist).any())
        if flight is not None:
            r["flight"] = flight
        rows.append(r)
    t_retrieval = time.time() - t0
    return rows, np.array(floors), t_gallery, t_retrieval, all_cached


def print_retrieval_summary(out, dist, label, t_gallery, t_retrieval,
                            floors, cached, topk):
    if out.empty or "skipped" not in out.columns:
        print("\n  No images processed."); return
    v = out[~out["skipped"].fillna(False)]
    if v.empty:
        print("\n  All images skipped."); return
    topk_col = f"top{topk}_hit"
    n  = len(v)
    s  = int(v["success"].fillna(False).sum())
    t5 = int(v[topk_col].fillna(False).sum()) if topk_col in v.columns else 0
    succ = v[v["success"].fillna(False)]
    fail = v[~v["success"].fillna(False)]
    print(f"\n  Results saved to {label}")
    print(f"  Success (≤{dist}m):      {s}/{n} ({100*s/n:.1f}%)")
    print(f"  Top-{topk} within {dist}m:     {t5}/{n} ({100*t5/n:.1f}%)")
    if floors.size:
        print(f"  Error floor (grid):     median {np.median(floors):.1f}m  "
              f"P90 {np.percentile(floors, 90):.1f}m")
    print(f"  Offset (all):            mean {v['offset_m'].mean():.1f}m  "
          f"median {v['offset_m'].median():.1f}m  "
          f"P90 {np.percentile(v['offset_m'].dropna(), 90):.1f}m  "
          f"max {v['offset_m'].max():.1f}m")
    if len(succ) and len(fail):
        print(f"  Top-1 sim:               success {succ['top1_sim'].mean():.3f}  "
              f"failure {fail['top1_sim'].mean():.3f}")
    print(f"  Median GT tile rank:     {v['gt_tile_rank'].median():.0f}")
    tag = "cached" if cached else f"built {t_gallery:.1f}s"
    print(f"  Time: gallery {tag} | retrieval {t_retrieval:.1f}s "
          f"({1000*t_retrieval/n:.0f} ms/img)")


# ---------- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        choices=["clip", "geoclip", "satclip", "all"],
                                       default="all")
    ap.add_argument("--dist",          type=float, default=25.0)
    ap.add_argument("--tile-size",     type=int,   default=1024)
    ap.add_argument("--stride",        type=int,   default=512)
    ap.add_argument("--cache-dir",     type=str,   default=CACHE_DIR)
    ap.add_argument("--out-dir",       type=str,   default=".")
    ap.add_argument("--batch-size",    type=int,   default=64)
    ap.add_argument("--topk",          type=int,   default=5)
    ap.add_argument("--satclip-ckpt",  type=str,   default=SATCLIP_CKPT)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--flights",       nargs="+", default=["all"],
                    help="Flight IDs to evaluate, e.g. 01 03 05, or 'all' (default)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    print(f"  Device: {device} | Tile: {args.tile_size}px | Stride: {args.stride}px | "
          f"Dist: {args.dist}m | Flights: {' '.join(flights)}")

    models_to_run = MODELS if args.model == "all" else (args.model,)

    for mname in models_to_run:
        print(f"\n=== {mname.upper()} ===")
        print(f"  Loading {mname} ... ", end="", flush=True)
        if   mname == "clip":    bundle = load_clip(device)
        elif mname == "geoclip": bundle = load_geoclip(device)
        else:                    bundle = load_satclip(device, args.satclip_ckpt)
        print("done")

        os.makedirs(args.out_dir, exist_ok=True)
        out_csv = os.path.join(args.out_dir, OUT_CSV_TEMPLATE.format(model=mname))
        log_path = out_csv.replace(".csv", ".log")

        with TeeLogger(log_path):
            all_rows = []
            all_floors = []
            for flight in flights:
                if flight == "09":
                    tile_paths, drone_dir, drone_csv, sat_csv = get_flight09_tile_paths()
                    df = pd.read_csv(drone_csv)
                    print(f"\n--- Flight {flight}: {len(df)} images ---")
                    rows, floors, t_gal, t_ret, cached = run_model_multitile(
                        mname, bundle, tile_paths, sat_csv, df, args.dist,
                        args.tile_size, args.stride, args.cache_dir,
                        args.topk, args.batch_size, device, args.rebuild_cache,
                        drone_dir=drone_dir, flight=flight)
                else:
                    sat_tif, drone_dir, drone_csv, sat_csv = get_flight_paths(flight)
                    sat, geo = load_satellite(sat_tif, sat_csv)
                    df = pd.read_csv(drone_csv)
                    print(f"\n--- Flight {flight}: {len(df)} images ---")
                    rows, floors, t_gal, t_ret, cached = run_model(
                        mname, bundle, sat, geo, df, args.dist,
                        args.tile_size, args.stride, args.cache_dir, sat_tif,
                        args.topk, args.batch_size, device, args.rebuild_cache,
                        drone_dir=drone_dir, flight=flight)
                all_rows.extend(rows)
                all_floors.append(floors)

                flight_df = pd.DataFrame(rows)
                valid = flight_df[~flight_df["skipped"].fillna(False)]
                if not valid.empty:
                    print_retrieval_summary(valid, args.dist, f"flight {flight}",
                                            t_gal, t_ret, floors, cached, args.topk)

            out = pd.DataFrame(all_rows)
            out.to_csv(out_csv, index=False)
            if len(flights) > 1:
                print(f"\n=== {mname.upper()} Overall ({len(flights)} flights) ===")
                valid_all = out[~out["skipped"].fillna(False)]
                if not valid_all.empty:
                    combined_floors = np.concatenate(all_floors) if all_floors else np.array([])
                    print_retrieval_summary(valid_all, args.dist, out_csv,
                                            0, 0, combined_floors, True, args.topk)

        # Release model to free memory before loading the next one.
        del bundle
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
