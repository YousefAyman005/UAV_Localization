#!/usr/bin/env python3
"""
UAV Image Augmentation Pipeline
================================
Augments UAV images with effects that simulate real UAV camera characteristics,
creating a more realistic and challenging matching problem vs. the satellite images.

Augmentations applied (in order):
  1. Perspective warp  — slight non-nadir tilt (max ±10% of image size per corner)
  2. Scale jitter      — random zoom in/out ±15%
  3. Motion blur       — directional blur to simulate UAV movement
  4. Gaussian noise    — sensor noise characteristics
  5. Brightness/contrast jitter — lighting and sensor response variation

Input:  berlin_grid_dataset/uav/   (3168 PNGs, 1024x1024)
Output: Augmented_UAV/             (same filenames, augmented)

Usage:
  .venv/bin/python3 augment_uav.py                  # all 3168 images
  .venv/bin/python3 augment_uav.py --limit 10       # first 10 (quick test)
  .venv/bin/python3 augment_uav.py --workers 4      # parallel workers (default: 4)
"""

import argparse
import os
import glob
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

# ---- Constants ---------------------------------------------------------------
RANDOM_SEED     = 24
UAV_DIR         = "berlin_grid_dataset/uav"
OUTPUT_DIR      = "Augmented_UAV"
IMG_SIZE        = 1024

OBLIQUE_SQUEEZE_RANGE = (0.15, 0.30)     # fraction of image width to squeeze (simulates ~30-50° tilt)
SCALE_RANGE     = (0.85, 1.15)
BLUR_LEN_RANGE  = (3, 12)                # motion blur kernel length in pixels
NOISE_STD_RANGE = (5, 20)               # Gaussian noise std
ALPHA_RANGE     = (0.75, 1.25)          # contrast multiplier
BETA_RANGE      = (-40, 40)             # brightness offset


# ---- Augmentation steps ------------------------------------------------------

def perspective_warp(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a directional oblique perspective warp to simulate tilted UAV camera.

    Squeezes one edge of the image inward (15-30% of width) to simulate a
    camera tilted ~30-50° from nadir, with a random heading direction.
    """
    h, w = img.shape[:2]
    squeeze_frac = rng.uniform(*OBLIQUE_SQUEEZE_RANGE)
    squeeze_px = int(w * squeeze_frac)

    # Squeeze the top edge inward (simulates forward tilt)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [squeeze_px, 0],
        [w - squeeze_px, 0],
        [w, h],
        [0, h],
    ])

    # Rotate the squeeze direction by a random heading so tilt varies
    heading = rng.uniform(0, 360)
    center = (w / 2, h / 2)
    R = cv2.getRotationMatrix2D(center, heading, 1.0)

    def rotate_pts(pts, M):
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([pts, ones])
        return (M @ pts_h.T).T.astype(np.float32)

    src_rot = rotate_pts(src, R)
    dst_rot = rotate_pts(dst, R)

    M_persp = cv2.getPerspectiveTransform(src_rot, dst_rot)
    return cv2.warpPerspective(img, M_persp, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)


def scale_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly zoom in or out ±15%, then restore to IMG_SIZE×IMG_SIZE."""
    scale = rng.uniform(*SCALE_RANGE)
    new_size = int(round(IMG_SIZE * scale))
    resized = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

    if scale >= 1.0:
        # Zoom in: center-crop back to IMG_SIZE
        start = (new_size - IMG_SIZE) // 2
        return resized[start:start + IMG_SIZE, start:start + IMG_SIZE]
    else:
        # Zoom out: pad with reflection to IMG_SIZE
        pad = (IMG_SIZE - new_size) // 2
        pad_r = IMG_SIZE - new_size - pad
        return cv2.copyMakeBorder(resized, pad, pad_r, pad, pad_r,
                                  borderType=cv2.BORDER_REFLECT)


def motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply directional motion blur to simulate UAV movement during capture."""
    length = int(rng.integers(BLUR_LEN_RANGE[0], BLUR_LEN_RANGE[1] + 1))
    angle  = float(rng.uniform(0, 360))

    # Build a horizontal line kernel then rotate it
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length

    cx, cy = length / 2, length / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    # Re-normalise after rotation (warpAffine may redistribute weights slightly)
    total = kernel.sum()
    if total > 0:
        kernel /= total

    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian sensor noise with random standard deviation."""
    sigma = rng.uniform(*NOISE_STD_RANGE)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def brightness_contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random brightness and contrast shift to simulate lighting/sensor variation."""
    alpha = float(rng.uniform(*ALPHA_RANGE))
    beta  = float(rng.uniform(*BETA_RANGE))
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ---- Main augment function ---------------------------------------------------

def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the full augmentation pipeline in order."""
    img = perspective_warp(img, rng)
    img = scale_jitter(img, rng)
    img = motion_blur(img, rng)
    img = gaussian_noise(img, rng)
    img = brightness_contrast(img, rng)
    return img


# ---- Worker ------------------------------------------------------------------

def process_one(args: tuple) -> str | None:
    """Load one UAV image, augment it, save to OUTPUT_DIR. Returns error string or None."""
    idx, src_path = args
    filename = os.path.basename(src_path)
    dst_path = os.path.join(OUTPUT_DIR, filename)

    img = cv2.imread(src_path)
    if img is None:
        return f"Could not read {src_path}"

    # Per-image seed: reproducible but unique per file
    rng = np.random.default_rng(RANDOM_SEED + idx)
    augmented = augment_image(img, rng)

    cv2.imwrite(dst_path, augmented)
    return None


# ---- Entry point -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Augment UAV images with realistic camera effects")
    parser.add_argument("--limit",   type=int, default=None, help="Process only first N images")
    parser.add_argument("--workers", type=int, default=4,    help="Parallel worker threads (default: 4)")
    args = parser.parse_args()

    if not os.path.isdir(UAV_DIR):
        sys.exit(f"ERROR: UAV directory not found: {UAV_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(UAV_DIR, "*.png")))
    if not paths:
        sys.exit(f"ERROR: No PNG files found in {UAV_DIR}")

    if args.limit:
        paths = paths[: args.limit]

    print(f"Augmenting {len(paths)} UAV images → {OUTPUT_DIR}/")
    print(f"  Oblique warp     : squeeze {OBLIQUE_SQUEEZE_RANGE[0]*100:.0f}–{OBLIQUE_SQUEEZE_RANGE[1]*100:.0f}% (simulates ~30-50° tilt)")
    print(f"  Scale jitter     : ±15%  [{SCALE_RANGE[0]:.2f}–{SCALE_RANGE[1]:.2f}x]")
    print(f"  Motion blur      : {BLUR_LEN_RANGE[0]}–{BLUR_LEN_RANGE[1]}px, random angle")
    print(f"  Gaussian noise   : σ ∈ [{NOISE_STD_RANGE[0]}, {NOISE_STD_RANGE[1]}]")
    print(f"  Brightness/contrast: α ∈ {ALPHA_RANGE}, β ∈ {BETA_RANGE}")
    print(f"  Workers          : {args.workers}")
    print()

    tasks = list(enumerate(paths))
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        with tqdm(total=len(tasks), unit="img") as pbar:
            for future in as_completed(futures):
                err = future.result()
                if err:
                    errors.append(err)
                pbar.update(1)

    print(f"\nDone. {len(tasks) - len(errors)}/{len(tasks)} images saved to {OUTPUT_DIR}/")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
