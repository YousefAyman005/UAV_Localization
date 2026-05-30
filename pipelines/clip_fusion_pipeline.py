"""Image+text fusion retrieval with the LoRA-fine-tuned CLIP.

Builds a satellite-tile gallery with the (optionally fine-tuned) CLIP visual
encoder, then for each test drone image forms a fused query

    q = normalize( alpha * image_emb + (1 - alpha) * text_emb )

where text_emb comes from the drone image's cached caption, and ranks tiles by
cosine similarity. alpha=0 -> image-only, alpha=1 -> text-only; the sweep in
between is the headline "match with both image and description" result.

Reuses the gallery / retrieval / summary machinery from clip_pipeline.py so the
output CSV schema is identical and analyze/retrieval_recall.py works unchanged.

Example:
    python pipelines/clip_fusion_pipeline.py --flights 10 11 \
        --lora-ckpt weights/clip_lora --fuse-alpha 0.0 0.5 0.7 1.0
"""

import argparse
import json
import math
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
    SZ_W, SZ_H, DEG_TO_M, PRIOR_OFFSET_STD_M, FLIGHTS_AVAILABLE,
    haversine_m, TeeLogger,
)
from pipelines.clip_pipeline import (
    ACC_THRS, CACHE_DIR, build_flight_gallery, haversine_m_vec,
    print_retrieval_summary,
)

torch.manual_seed(0)

BACKBONE     = "openai/clip-vit-base-patch32"
CLIP_MEAN    = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD     = (0.26862954, 0.26130258, 0.27577711)
RES          = 224
CAPTION_DIR  = "cache/captions"
OUT_TEMPLATE = "visloc_clipfusion_a{alpha}_results.csv"


# ---------- model bundle (HF CLIP, optional merged LoRA) -------------------

def load_ft_clip(device, lora_ckpt):
    from transformers import CLIPModel, CLIPTokenizer
    from torchvision import transforms
    clip = CLIPModel.from_pretrained(BACKBONE)
    if lora_ckpt:
        from peft import PeftModel
        clip = PeftModel.from_pretrained(clip, lora_ckpt).merge_and_unload()
        print(f"  Loaded + merged LoRA adapter from {lora_ckpt}")
    clip.eval().to(device)
    tokenizer = CLIPTokenizer.from_pretrained(BACKBONE)
    preprocess = transforms.Compose([
        transforms.Resize(RES, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RES),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    encode = lambda t: clip.get_image_features(pixel_values=t.to(device))  # noqa: E731
    with torch.inference_mode():
        dim = encode(torch.zeros(1, 3, RES, RES, device=device)).shape[-1]
    return (encode, preprocess, dim), clip, tokenizer


def encode_text(clip, tokenizer, captions):
    device = next(clip.parameters()).device
    tok = tokenizer(captions, padding=True, truncation=True, max_length=77,
                    return_tensors="pt").to(device)
    with torch.inference_mode():
        return F.normalize(clip.get_text_features(**tok), dim=-1).cpu().numpy()


def load_drone_captions(flight, caption_dir):
    path = os.path.join(caption_dir, f"{flight}_drone.jsonl")
    caps = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    caps[r["filename"]] = r["caption"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return caps


# ---------- fused retrieval (parallels clip_pipeline.retrieve) -------------

def retrieve_fusion(flight, bundle, clip, tokenizer, emb, tile_ids, centers, df,
                    drone_dir, captions, alpha, dist, topk, gps_radii=(),
                    flight_tag=None):
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
            img_emb = F.normalize(encode(t), dim=-1).cpu().numpy().squeeze(0)

        # Fuse with the caption embedding. Missing caption -> image-only for this
        # row (keeps N constant across the alpha sweep); flagged via has_caption.
        cap = captions.get(f)
        if cap and alpha >= 1.0:
            q = encode_text(clip, tokenizer, [cap]).squeeze(0)
        elif cap:
            txt_emb = encode_text(clip, tokenizer, [cap]).squeeze(0)
            q = alpha * img_emb + (1.0 - alpha) * txt_emb
        else:
            q = img_emb
        q = q / (np.linalg.norm(q) + 1e-12)

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
            "pred_lat": round(float(plat), 7), "pred_lon": round(float(plon), 7),
            "offset_m": round(off_m, 2),
            "top1_tile_id": int(tile_ids[top_idx[0]]),
            "top1_sim": round(float(sims[top_idx[0]]), 4),
            "gt_tile_rank": gt_rank, "has_caption": bool(cap), "skipped": False,
            **{f"success_{thr}": off_m <= thr for thr in ACC_THRS},
            topk_col: bool((all_dists[top_idx] <= dist).any()),
        }

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


# ---------- per-flight / main ----------------------------------------------

def run_flight(flight, bundle, clip, tokenizer, args, model_name, device):
    emb, tile_ids, centers, drone_dir, drone_csv, t_gallery, cached = \
        build_flight_gallery(flight, bundle, args.tile_size, args.stride,
                             args.batch_size, device, args.cache_dir,
                             model_name, args.rebuild_cache)
    print(f"  Gallery: {len(tile_ids)} tiles | dim={emb.shape[1]} | "
          f"{'cached' if cached else f'built in {t_gallery:.1f}s'}")
    df = pd.read_csv(drone_csv)
    if args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)
    captions = load_drone_captions(flight, args.caption_dir)
    if not captions and args.fuse_alpha_val < 1.0:
        print(f"  WARN: no drone captions for flight {flight}; alpha<1 falls back "
              f"to image-only.")
    rows, floors, t_ret = retrieve_fusion(
        flight, bundle, clip, tokenizer, emb, tile_ids, centers, df, drone_dir,
        captions, args.fuse_alpha_val, args.dist, args.topk,
        gps_radii=tuple(args.gps_radii), flight_tag=flight)
    return rows, floors, t_gallery, t_ret, cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", nargs="+", default=["10", "11"])
    ap.add_argument("--lora-ckpt", default=None,
                    help="LoRA adapter dir; omit for the stock-CLIP baseline.")
    ap.add_argument("--fuse-alpha", nargs="+", type=float, default=[0.0, 0.5, 0.7, 1.0],
                    help="Sweep of image weights (0=text-only end is 1).")
    ap.add_argument("--caption-dir", default=CAPTION_DIR)
    ap.add_argument("--dist", type=float, default=25.0)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--gps-radii", nargs="*", type=int, default=[1000, 5000])
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Gallery embeddings depend only on the encoder identity, not alpha -> shared
    # across the sweep so it is built once.
    model_name = "cliphf_lora" if args.lora_ckpt else "cliphf_base"
    print(f"  Device: {device} | encoder: {model_name} | "
          f"alphas: {args.fuse_alpha} | flights: {' '.join(flights)}")

    print("  Loading CLIP ... ", end="", flush=True)
    bundle, clip, tokenizer = load_ft_clip(device, args.lora_ckpt)
    print("done")
    os.makedirs(args.out_dir, exist_ok=True)

    for alpha in args.fuse_alpha:
        args.fuse_alpha_val = alpha
        out_csv  = os.path.join(args.out_dir, OUT_TEMPLATE.format(alpha=alpha))
        log_path = out_csv.replace(".csv", ".log")
        print(f"\n=== FUSION alpha={alpha} ({model_name}) ===")
        with TeeLogger(log_path):
            all_rows, all_floors = [], []
            for flight in flights:
                rows, floors, t_gal, t_ret, cached = run_flight(
                    flight, bundle, clip, tokenizer, args, model_name, device)
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
                print(f"\n=== Overall alpha={alpha} ({len(flights)} flights) ===")
                va = pd.DataFrame(all_rows)
                va = va[~va["skipped"].fillna(False)]
                if not va.empty:
                    floors_arr = np.concatenate(all_floors) if all_floors else np.array([])
                    print_retrieval_summary(va, args.dist, out_csv, None, None,
                                             floors_arr, True, args.topk)


if __name__ == "__main__":
    main()
