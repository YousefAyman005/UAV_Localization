"""Cross-view alignment diagnostic: did text close the drone<->satellite gap?

For each drone image we embed it and its ground-truth satellite crop with CLIP,
and measure their cosine similarity. Higher = the two views of the same place sit
closer in embedding space. Comparing stock CLIP vs the LoRA-fine-tuned adapter
shows directly whether the text-bridged training pulled matched views together —
the honest success metric, independent of whether retrieval accuracy improves.

Usage:
    python analyze/crossview_cosine.py --lora-ckpt weights/clip_lora --flights all
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import (
    SZ_W, SZ_H, FLIGHTS_AVAILABLE, crop_gt_patch, load_flight, split_flight_rows,
)
from pipelines.clip_fusion_pipeline import load_ft_clip


def build_pairs(flight, preprocess, which, test_frac, axis, limit):
    """Preprocessed (drone, sat-crop) tensor pairs for the chosen spatial band."""
    import pandas as pd
    tiles, drone_dir, drone_csv, _ = load_flight(flight)
    df = split_flight_rows(pd.read_csv(drone_csv), which=which,
                           test_frac=test_frac, axis=axis)
    if limit is not None:
        df = df.iloc[:limit]
    drone_t, sat_t = [], []
    for _, row in df.iterrows():
        drone = cv2.imread(os.path.join(drone_dir, row["filename"]))
        if drone is None:
            continue
        yaw = float(row["Phi1"]) if "Phi1" in row.index else 0.0
        patch = crop_gt_patch(tiles, float(row["lat"]), float(row["lon"]),
                              float(row["height"]), yaw_deg=yaw, flight=flight)
        if patch is None:
            continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        drone_t.append(preprocess(Image.fromarray(cv2.cvtColor(drone, cv2.COLOR_BGR2RGB))))
        sat_t.append(preprocess(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))))
    if not drone_t:
        return None
    return torch.stack(drone_t), torch.stack(sat_t)


@torch.inference_mode()
def cosines(encode, drone, sat, device, batch=64):
    out = []
    for i in range(0, len(drone), batch):
        d = F.normalize(encode(drone[i:i+batch].to(device)), dim=-1)
        s = F.normalize(encode(sat[i:i+batch].to(device)),   dim=-1)
        out.append((d * s).sum(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-ckpt", default="weights/clip_lora",
                    help="Adapter to compare against stock CLIP (set '' to skip).")
    ap.add_argument("--flights", nargs="+", default=["all"])
    ap.add_argument("--which", choices=["test", "train", "all"], default="test",
                    help="Spatial band to evaluate (test = held-out, the honest one).")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--split-axis", choices=["auto", "lat", "lon"], default="auto")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("  Loading stock CLIP ...")
    (enc0, preprocess, _), _, _ = load_ft_clip(device, None)
    enc1 = None
    if args.lora_ckpt:
        print("  Loading fine-tuned CLIP ...")
        (enc1, _, _), _, _ = load_ft_clip(device, args.lora_ckpt)

    print(f"\n  {'flight':8s} {'N':>5} {'stock':>8}" +
          ("   finetuned     delta" if enc1 else ""))
    print("  " + "-" * (40 if enc1 else 24))
    all0, all1 = [], []
    for flight in flights:
        pair = build_pairs(flight, preprocess, args.which, args.test_frac,
                           args.split_axis, args.limit)
        if pair is None:
            continue
        drone, sat = pair
        c0 = cosines(enc0, drone, sat, device); all0.append(c0)
        line = f"  {flight:8s} {len(c0):>5} {c0.mean():>8.4f}"
        if enc1:
            c1 = cosines(enc1, drone, sat, device); all1.append(c1)
            line += f"   {c1.mean():>8.4f}  {c1.mean()-c0.mean():>+8.4f}"
        print(line)

    a0 = np.concatenate(all0)
    print("  " + "-" * (40 if enc1 else 24))
    summary = f"  {'OVERALL':8s} {len(a0):>5} {a0.mean():>8.4f}"
    if enc1:
        a1 = np.concatenate(all1)
        summary += f"   {a1.mean():>8.4f}  {a1.mean()-a0.mean():>+8.4f}"
        print(summary)
        print(f"\n  Mean cross-view cosine rose {a0.mean():.4f} -> {a1.mean():.4f} "
              f"({a1.mean()-a0.mean():+.4f}). Positive = text-bridged training pulled "
              f"matched drone/satellite views closer.")
    else:
        print(summary)


if __name__ == "__main__":
    main()
