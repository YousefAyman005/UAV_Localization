"""Image+text fusion retrieval with the LoRA-fine-tuned CLIP.

Builds a satellite-tile gallery with the (optionally fine-tuned) CLIP visual
encoder, then for each test drone image forms a fused query

    q = normalize( alpha * image_emb + (1 - alpha) * text_emb )

where text_emb comes from the drone image's cached caption, and ranks tiles by
cosine similarity. alpha is the IMAGE weight on both sides: alpha=1 ->
image-only, alpha=0 -> text-only; the sweep in between is the headline "match
with both image and description" result. --gallery-alpha decouples the gallery
blend from the query blend, e.g. ``--fuse-alpha 1.0 --gallery-alpha 0.7``
retrieves with an image-only query against a text-fused gallery — tile captions
are computed once offline, so no VLM is needed at query time.

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
    haversine_m, split_flight_rows, TeeLogger,
)
from pipelines.clip_pipeline import (
    ACC_THRS, CACHE_DIR, build_flight_gallery, haversine_m_vec,
    print_retrieval_summary,
)

torch.manual_seed(0)

BACKBONE     = "openai/clip-vit-base-patch32"
CLIP_MEAN    = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD     = (0.26862954, 0.26130258, 0.27577711)
CAPTION_DIR  = "cache/captions"
# Encoder identity (adapter name + backbone) is part of the filename so runs
# with different adapters can never overwrite or shadow each other.
OUT_TEMPLATE = "visloc_{model}_{tag}_results.csv"

# Tokenizer kwargs derived from the loaded backbone (set by load_ft_clip):
# CLIP pads to the longest caption (max 77); SigLIP needs max-length padding.
_TXT_KW = {"padding": True, "max_length": 77}

TILE_SIZE    = 1024   # must match caption_crops.py
TILE_STRIDE  = 512    # must match caption_crops.py
TEST_FRAC    = 0.25   # must match clip_lora_train.py
DIST_THRESH  = 25.0
TOPK         = 5
BATCH_SIZE   = 64
GPS_RADII    = (1000, 5000)


# ---------- model bundle (HF CLIP, optional merged LoRA) -------------------

def load_ft_clip(device, lora_ckpt):
    from transformers import AutoModel, AutoTokenizer
    from torchvision import transforms
    clip = AutoModel.from_pretrained(BACKBONE)
    if lora_ckpt:
        from peft import PeftModel
        clip = PeftModel.from_pretrained(clip, lora_ckpt).merge_and_unload()
        print(f"  Loaded + merged LoRA adapter from {lora_ckpt}")
    clip.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    res = clip.config.vision_config.image_size
    if "siglip" in clip.config.model_type:
        mean = std = (0.5, 0.5, 0.5)
        _TXT_KW["padding"] = "max_length"  # SigLIP is trained with max-length padding
    else:
        mean, std = CLIP_MEAN, CLIP_STD
    _TXT_KW["max_length"] = clip.config.text_config.max_position_embeddings
    preprocess = transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    encode = lambda t: clip.get_image_features(pixel_values=t.to(device))  # noqa: E731
    with torch.inference_mode():
        dim = encode(torch.zeros(1, 3, res, res, device=device)).shape[-1]
    return (encode, preprocess, dim), clip, tokenizer


def encode_text(clip, tokenizer, captions):
    device = next(clip.parameters()).device
    tok = tokenizer(captions, truncation=True, return_tensors="pt",
                    **_TXT_KW).to(device)
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


def load_tile_captions(flight):
    path = os.path.join(CAPTION_DIR,
                        f"{flight}_tile_ts{TILE_SIZE}_st{TILE_STRIDE}.jsonl")
    caps = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    caps[int(r["tile_id"])] = r["caption"]
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
    return caps


def fuse_gallery(emb, tile_ids, tilecaps, clip, tokenizer, alpha):
    """Blend each gallery tile's image embedding with its caption embedding:
    fused = normalize(alpha*img + (1-alpha)*text). Tiles without a caption keep
    their image embedding."""
    present = [(i, tilecaps[int(t)]) for i, t in enumerate(tile_ids)
               if int(t) in tilecaps]
    if not present:
        print("  WARN: no tile captions found; gallery stays image-only.")
        return emb
    idx = [i for i, _ in present]
    txt = encode_text(clip, tokenizer, [c for _, c in present])
    fused = emb.copy()
    fused[idx] = alpha * emb[idx] + (1.0 - alpha) * txt
    norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12
    print(f"  Gallery fused with text on {len(idx)}/{len(tile_ids)} tiles "
          f"(alpha={alpha}).")
    return (fused / norms).astype(np.float32)


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
        drone = cv2.resize(drone, (SZ_W, SZ_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(drone, cv2.COLOR_BGR2RGB)
        t = preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with torch.inference_mode():
            img_emb = F.normalize(encode(t), dim=-1).cpu().numpy().squeeze(0)

        # Fuse with the caption embedding (alpha = image weight). At alpha>=1
        # the query is the true image-only endpoint. Missing caption ->
        # image-only for this row (keeps N constant across the alpha sweep);
        # flagged via has_caption.
        cap = captions.get(f)
        if cap and alpha < 1.0:
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

def run_flight(flight, bundle, clip, tokenizer, alpha, gallery_alpha, limit,
               model_name, device):
    emb, tile_ids, centers, drone_dir, drone_csv, t_gallery, cached = \
        build_flight_gallery(flight, bundle, TILE_SIZE, TILE_STRIDE,
                             BATCH_SIZE, device, CACHE_DIR, model_name, False)
    print(f"  Gallery: {len(tile_ids)} tiles | dim={emb.shape[1]} | "
          f"{'cached' if cached else f'built in {t_gallery:.1f}s'}")

    g_alpha = alpha if gallery_alpha is None else gallery_alpha
    gallery = emb
    if g_alpha < 1.0:
        tilecaps = load_tile_captions(flight)
        gallery = fuse_gallery(emb, tile_ids, tilecaps, clip, tokenizer, g_alpha)

    df = pd.read_csv(drone_csv)
    df = split_flight_rows(df, which="test", test_frac=TEST_FRAC,
                           axis="auto", buffer_frac=0.0)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    captions = load_drone_captions(flight, CAPTION_DIR)
    if not captions and alpha < 1.0:
        print(f"  WARN: no drone captions for flight {flight}; alpha<1 falls back "
              f"to image-only.")
    rows, floors, t_ret = retrieve_fusion(
        flight, bundle, clip, tokenizer, gallery, tile_ids, centers, df, drone_dir,
        captions, alpha, DIST_THRESH, TOPK, gps_radii=GPS_RADII, flight_tag=flight)
    return rows, floors, t_gallery, t_ret, cached


def main():
    global CAPTION_DIR, CACHE_DIR, BACKBONE
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", nargs="+", default=["all"])
    ap.add_argument("--lora-ckpt", default=None,
                    help="LoRA adapter dir; omit for the stock-CLIP baseline.")
    ap.add_argument("--fuse-alpha", nargs="+", type=float, default=[0.0, 0.5, 0.7, 1.0],
                    help="Image weight of the QUERY (and, unless --gallery-alpha is "
                         "given, of the gallery); 0=text-only, 1=image-only.")
    ap.add_argument("--gallery-alpha", type=float, default=None,
                    help="Image weight of the GALLERY blend, decoupled from the "
                         "query. E.g. --fuse-alpha 1.0 --gallery-alpha 0.7 = "
                         "image-only query vs text-fused gallery (no VLM at "
                         "query time). Default: follow --fuse-alpha.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows per flight (smoke test).")
    ap.add_argument("--caption-dir", default=CAPTION_DIR,
                    help="Dir with {flight}_drone / _tile caption JSONLs.")
    ap.add_argument("--cache-dir", default=CACHE_DIR,
                    help="Gallery embedding cache dir (passed to build_flight_gallery).")
    ap.add_argument("--backbone", default=BACKBONE,
                    help="HF CLIP model id; must match the --lora-ckpt base model.")
    args = ap.parse_args()

    # Honor the absolute dirs the SLURM wrapper passes: run_flight reads these
    # module globals (CAPTION_DIR via load_drone/tile_captions; CACHE_DIR via
    # build_flight_gallery), so set them once here before any flight runs.
    CAPTION_DIR = args.caption_dir
    CACHE_DIR   = args.cache_dir
    BACKBONE    = args.backbone

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb_tag = BACKBONE.split("/")[-1].replace("clip-vit-", "").replace("-patch", "p")
    # Adapter NAME (not just "lora") keys the gallery cache and the output CSVs:
    # different adapters must never share cached gallery embeddings.
    ad_tag = (os.path.basename(os.path.normpath(args.lora_ckpt))
              if args.lora_ckpt else "base")
    model_name = f"cliphf_{ad_tag}_{bb_tag}"
    g_str = "follows query" if args.gallery_alpha is None else args.gallery_alpha
    print(f"  Device: {device} | encoder: {model_name} | "
          f"alphas: {args.fuse_alpha} (gallery: {g_str}) | "
          f"flights: {' '.join(flights)}")

    print("  Loading CLIP ... ", end="", flush=True)
    bundle, clip, tokenizer = load_ft_clip(device, args.lora_ckpt)
    print("done")

    for alpha in args.fuse_alpha:
        tag = f"a{alpha}"
        if args.gallery_alpha is not None and args.gallery_alpha != alpha:
            tag += f"_g{args.gallery_alpha}"
        out_csv  = OUT_TEMPLATE.format(model=model_name, tag=tag)
        log_path = out_csv.replace(".csv", ".log")
        print(f"\n=== FUSION {tag} ({model_name}) ===")
        with TeeLogger(log_path):
            all_rows, all_floors = [], []
            for flight in flights:
                rows, floors, t_gal, t_ret, cached = run_flight(
                    flight, bundle, clip, tokenizer, alpha, args.gallery_alpha,
                    args.limit, model_name, device)
                fdf = pd.DataFrame(rows)
                valid = fdf[~fdf["skipped"].fillna(False)]
                if not valid.empty:
                    print(f"\n--- Flight {flight}: {len(fdf)} images ---")
                    print_retrieval_summary(valid, DIST_THRESH, f"flight {flight}",
                                             t_gal, t_ret, floors, cached, TOPK)
                all_rows.extend(rows)
                all_floors.append(floors)

            pd.DataFrame(all_rows).to_csv(out_csv, index=False)
            if len(flights) > 1:
                print(f"\n=== Overall alpha={alpha} ({len(flights)} flights) ===")
                va = pd.DataFrame(all_rows)
                va = va[~va["skipped"].fillna(False)]
                if not va.empty:
                    floors_arr = np.concatenate(all_floors) if all_floors else np.array([])
                    print_retrieval_summary(va, DIST_THRESH, out_csv, None, None,
                                             floors_arr, True, TOPK)


if __name__ == "__main__":
    main()
