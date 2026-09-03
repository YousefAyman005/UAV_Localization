"""Qualitative check: what the LoRA trainer actually feeds into CLIP.

Reproduces pipelines/clip_lora_train.py's PairDataset transforms exactly, so
the drone and satellite views can be compared in the SAME frame the encoder
sees. Four columns per sample train-band row:

  1. drone raw (heading-up, as captured)
  2. drone -> CLIP, north-up, deterministic (north_up_drone + Resize/CenterCrop)
  3. satellite GT crop -> CLIP, north-up (crop_gt_patch(yaw=0) + Resize/CenterCrop)
  4. drone -> CLIP, one train-time augmented draw (RandomResizedCrop + ColorJitter)

Columns 2 and 3 are the orientation-matched views the model aligns; column 1
shows the raw heading-up frame the fix corrects; column 4 shows train-time aug.
Needs the dataset bound (cv2 + torchvision). Run via a short srun, e.g.:
  srun -p testing --cpus-per-task=2 apptainer exec \
      --bind $DATAPOOL3/datasets/Visloc:/mnt/visloc:ro uav_localization.sif \
      python analyze/plot_clip_train_inputs.py --dataset-root /mnt/visloc --flight 08
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from helpers.utils import (  # noqa: E402
    SZ_W, SZ_H, corrected_yaw, crop_gt_patch, load_flight,
    north_up_drone, split_flight_rows,
)

TEST_FRAC, BUFFER_FRAC = 0.25, 0.05   # clip_lora_train split defaults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="08")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--res", type=int, default=224,
                    help="CLIP input res (openai/clip-vit-large-patch14 = 224)")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args()

    from torchvision import transforms
    bicubic = transforms.InterpolationMode.BICUBIC
    eval_tf = transforms.Compose([
        transforms.Resize(args.res, interpolation=bicubic),
        transforms.CenterCrop(args.res),
    ])
    drone_aug = transforms.Compose([                      # train-time drone_tf
        transforms.RandomResizedCrop(args.res, scale=(0.6, 1.0),
                                     interpolation=bicubic),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    ])

    def rgb(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    tiles, drone_dir, drone_csv, _ = load_flight(args.flight, args.dataset_root)
    df = pd.read_csv(drone_csv)
    df = split_flight_rows(df, which="train", test_frac=TEST_FRAC,
                           axis="auto", buffer_frac=BUFFER_FRAC)
    picks = df.iloc[np.linspace(0, len(df) - 1, args.n, dtype=int)]

    titles = ["drone (raw, heading-up)", "drone → CLIP (north-up)",
              "satellite → CLIP (north-up)", "drone → CLIP (train aug)"]
    fig, axes = plt.subplots(args.n, 4, figsize=(11, 3.0 * args.n))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(picks.iterrows()):
        f = row["filename"]
        yaw = corrected_yaw(args.flight, float(row["Phi1"]))
        bgr = cv2.imread(os.path.join(drone_dir, f))
        raw = cv2.resize(bgr, (SZ_W, SZ_H), interpolation=cv2.INTER_AREA)
        nu = north_up_drone(bgr, yaw)                     # exact trainer transform
        nu_pil = Image.fromarray(rgb(nu))
        sat = crop_gt_patch(tiles, float(row["lat"]), float(row["lon"]),
                            float(row["height"]), yaw_deg=0.0, flight=args.flight)
        sat_pil = Image.fromarray(rgb(sat))

        panels = [rgb(raw), np.asarray(eval_tf(nu_pil)),
                  np.asarray(eval_tf(sat_pil)), np.asarray(drone_aug(nu_pil))]
        for c, (ax, img) in enumerate(zip(axes[r], panels)):
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        axes[r, 0].set_ylabel(f"{f}\nyaw {yaw:.0f}°", fontsize=8)

    fig.suptitle(f"flight {args.flight} — LoRA trainer inputs to CLIP "
                 f"(res {args.res})", fontsize=10)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"clip_train_inputs_{args.flight}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
