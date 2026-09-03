"""Probe candidate caption prompts on a few crops before a full re-caption.

Captions the drone (north-up, via north_up_drone+corrected_yaw) and the GT
satellite crop for a handful of frames with each candidate SYSTEM_PROMPT, and
prints them so the style / cross-view agreement can be judged cheaply (~a
dozen VLM calls) instead of re-captioning thousands. Reuses caption_crops's
Ollama backend; needs an Ollama server reachable at $OLLAMA_HOST (run via
slurm/run_caption_probe.sh, which brings one up in-container).

Usage (inside the probe slurm job):
    python analyze/caption_prompt_probe.py --flight 08
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

import pipelines.caption_crops as CC  # noqa: E402
from helpers.utils import (  # noqa: E402
    corrected_yaw, crop_gt_patch, load_flight, north_up_drone,
)

# Candidate prompts, both targeting the natural spatial-layout style:
# "road goes down the middle between two fields", "water on the west, farm
# field on the right and houses", "water from south to east, dense buildings".
# They differ ONLY on whether colour / field-state words are allowed.
_BASE = (
    "List the distinctive layout of this ground patch in UNDER 18 words, as "
    "short comma-separated phrases, so it can be told apart from nearby "
    "patches. The patch is already rotated so up = north; use north/south/"
    "east/west for directions. Cover only what is present: where the main road "
    "runs; which side water is on or how it flows (e.g. south to east); where "
    "fields and buildings sit (left, right, a named side, centre). Be specific "
    "about position. No full sentences, no preamble; do not start with 'a', "
    "'an' or 'the'. Never use these words: north-up, aerial, view, image, "
    "patch, photo, satellite, drone, shadow. "
)
PROMPTS = {
    "layout": _BASE + "Do not mention colour or brightness.",
    "field_state": _BASE + ("You may note whether fields are bare or vegetated, "
                            "but use no colour names."),
}

# Reference captions the user wrote for these flight-08 frames (style target).
TARGETS = {
    "08_0333.JPG": "road goes down middle between two fields, patches of beige and green",
    "08_0649.JPG": "water on the west, northwest, farm field on right and houses",
    "08_1033.JPG": "water going from south to east surrounded by dense buildings",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="08")
    ap.add_argument("--files", nargs="+",
                    default=["08_0333.JPG", "08_0649.JPG", "08_1033.JPG"])
    ap.add_argument("--dataset-root", default=None)
    args = ap.parse_args()

    import pandas as pd
    tiles, drone_dir, drone_csv, _ = load_flight(args.flight, args.dataset_root)
    df = pd.read_csv(drone_csv).set_index("filename")
    caption = CC.make_ollama(CC.OLLAMA_MODEL, 48, None)  # ~18-word cap

    def cap_with(prompt, pil):
        CC.SYSTEM_PROMPT = prompt
        return CC._clean_caption(caption(pil))

    for f in args.files:
        row = df.loc[f]
        yaw = corrected_yaw(args.flight, float(row["Phi1"]))
        drone = north_up_drone(cv2.imread(os.path.join(drone_dir, f)), yaw)
        sat = crop_gt_patch(tiles, float(row["lat"]), float(row["lon"]),
                            float(row["height"]), yaw_deg=0.0, flight=args.flight)
        d_pil, s_pil = CC._bgr_to_pil(drone), CC._bgr_to_pil(sat)
        print(f"\n=== {f}  (yaw {yaw:.0f}°) ===", flush=True)
        if f in TARGETS:
            print(f"  TARGET (user)      : {TARGETS[f]}")
        for name, prompt in PROMPTS.items():
            print(f"  drone [{name:10s}] : {cap_with(prompt, d_pil)}", flush=True)
            print(f"  sat   [{name:10s}] : {cap_with(prompt, s_pil)}", flush=True)


if __name__ == "__main__":
    main()
