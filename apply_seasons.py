#!/usr/bin/env python3
"""
Seasonal / Lighting Augmentation for UAV Images
================================================
Applies one of four conditions to every UAV image in the dataset,
leaving satellite images untouched. Run AFTER dataset generation.

Usage:
    .venv/bin/python3 apply_seasons.py --season summer   # in-place
    .venv/bin/python3 apply_seasons.py --season autumn --output berlin_uav_dataset_autumn
    .venv/bin/python3 apply_seasons.py --season all      # creates four output dirs

Seasons: summer | autumn | winter | night | all
"""

import argparse
import os
import random
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# ---- Paths (mirror berlin_dataset.py) --------------------------------------
INPUT_DATASET = "berlin_uav_dataset"
UAV_SUBDIR    = "uav"
SAT_SUBDIR    = "satellite"

RANDOM_SEED = 42  # reproducible augmentation


# ============================================================================
# Helpers
# ============================================================================

def shift_channels(arr: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """Add per-channel offsets and clip to [0, 255]."""
    out = arr.astype(np.int16)
    out[:, :, 0] = np.clip(out[:, :, 0] + r, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] + g, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] + b, 0, 255)
    return out.astype(np.uint8)


def add_gaussian_noise(arr: np.ndarray, std: float, rng: random.Random) -> np.ndarray:
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, std, arr.shape)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ============================================================================
# Season transforms
# ============================================================================

def apply_summer(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.95, 1.05)
    img = ImageEnhance.Brightness(img).enhance(factor)
    return img


def apply_autumn(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.85, 1.0)
    img = ImageEnhance.Brightness(img).enhance(factor)

    arr = np.array(img)
    r_shift = rng.randint(20, 40)
    g_shift = rng.randint(5, 15)
    b_shift = -rng.randint(10, 20)
    arr = shift_channels(arr, r_shift, g_shift, b_shift)
    img = Image.fromarray(arr)

    blur_radius = rng.uniform(0.5, 1.0)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return img


def apply_winter(img: Image.Image, rng: random.Random) -> Image.Image:
    desat = rng.uniform(0.60, 0.80)
    gray = img.convert("L").convert("RGB")
    img = Image.blend(img, gray, desat)

    factor = rng.uniform(1.05, 1.2)
    img = ImageEnhance.Brightness(img).enhance(factor)

    arr = np.array(img)
    b_shift = rng.randint(10, 25)
    arr = shift_channels(arr, 0, 0, b_shift)
    noise_std = rng.uniform(5, 10)
    arr = add_gaussian_noise(arr, noise_std, rng)
    return Image.fromarray(arr)


def apply_night(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.15, 0.35)
    img = ImageEnhance.Brightness(img).enhance(factor)

    contrast = rng.uniform(1.3, 1.8)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    arr = np.array(img)
    noise_std = rng.uniform(15, 25)
    arr = add_gaussian_noise(arr, noise_std, rng)
    g_shift = rng.randint(5, 10)
    b_shift = rng.randint(10, 20)
    arr = shift_channels(arr, 0, g_shift, b_shift)
    return Image.fromarray(arr)


SEASON_FN = {
    "summer": apply_summer,
    "autumn": apply_autumn,
    "winter": apply_winter,
    "night":  apply_night,
}


# ============================================================================
# Core
# ============================================================================

def output_dir_for(season: str, base_output: str | None) -> str:
    if base_output:
        return base_output
    return f"{INPUT_DATASET}_{season}"


def process_season(season: str, output_root: str):
    fn = SEASON_FN[season]
    rng = random.Random(RANDOM_SEED)

    src_uav = os.path.join(INPUT_DATASET, UAV_SUBDIR)
    dst_uav = os.path.join(output_root, UAV_SUBDIR)
    dst_sat = os.path.join(output_root, SAT_SUBDIR)

    os.makedirs(dst_uav, exist_ok=True)

    # Copy satellite images unchanged
    src_sat = os.path.join(INPUT_DATASET, SAT_SUBDIR)
    if not os.path.isdir(dst_sat):
        print(f"  Linking/copying satellite images to {dst_sat} ...")
        shutil.copytree(src_sat, dst_sat)

    uav_files = sorted(f for f in os.listdir(src_uav) if f.lower().endswith(".png"))
    if not uav_files:
        print(f"  No UAV images found in {src_uav}")
        return

    print(f"  Applying '{season}' to {len(uav_files)} UAV images → {dst_uav}")
    for fname in tqdm(uav_files, desc=season):
        src_path = os.path.join(src_uav, fname)
        dst_path = os.path.join(dst_uav, fname)
        img = Image.open(src_path).convert("RGB")
        img = fn(img, rng)
        img.save(dst_path)

    # Write CSV with season column prepended
    csv_src = "berlin_pairs.csv"
    if os.path.isfile(csv_src):
        import pandas as pd
        df = pd.read_csv(csv_src)
        df.insert(0, "season", season)
        dst_csv = os.path.join(output_root, "berlin_pairs.csv")
        df.to_csv(dst_csv, index=False)
        print(f"  Wrote {dst_csv} (+ season column)")


def main():
    parser = argparse.ArgumentParser(description="Apply seasonal augmentation to UAV images.")
    parser.add_argument(
        "--season",
        choices=["summer", "autumn", "winter", "night", "all"],
        required=True,
        help="Which season/condition to apply. Use 'all' to generate all four variants.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (only valid with a single season, not 'all').",
    )
    args = parser.parse_args()

    if args.output and args.season == "all":
        parser.error("--output cannot be used with --season all (would overwrite itself).")

    seasons = list(SEASON_FN.keys()) if args.season == "all" else [args.season]

    for season in seasons:
        out = output_dir_for(season, args.output)
        print(f"\n[{season}] → {out}")
        process_season(season, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
