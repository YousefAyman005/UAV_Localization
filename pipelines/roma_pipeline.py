"""RoMa (Robust dense matcher) + RANSAC."""

import os
import sys

import cv2
import torch
from PIL import Image

torch.manual_seed(0)

from romatch import roma_outdoor, roma_indoor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import SZ_W, SZ_H, dense_match_result
from helpers.workers import run_pipeline


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def match_roma(drone_pil, sat_bgr, matcher, device, num_samples):
    with torch.inference_mode():
        warp, cert = matcher.match(drone_pil, bgr_to_pil(sat_bgr), device=device)
        matches, c = matcher.sample(warp, cert, num=num_samples)
        kp_a, kp_b = matcher.to_pixel_coordinates(matches, SZ_H, SZ_W, SZ_H, SZ_W)
    return dense_match_result(kp_a.cpu().numpy(), kp_b.cpu().numpy(), c.cpu().numpy())


def load_model(device, args):
    torch.set_float32_matmul_precision("highest")
    kw = {} if device.type == "cuda" else {"amp_dtype": torch.float32}
    # AerialExtreMatch fine-tune: same RoMa architecture as roma_outdoor, just a
    # different checkpoint. Load the .pth state_dict into the stock outdoor model.
    if args.pretrained == "extre":
        weights = torch.load(args.extre_weights, map_location=device)
        if isinstance(weights, dict) and "model" in weights:
            weights = weights["model"]
        return roma_outdoor(device=device, weights=weights, **kw)
    builder = roma_outdoor if args.pretrained == "outdoor" else roma_indoor
    return builder(device=device, **kw)


def make_match_factory(matcher, device, args):
    def match_factory(drone):
        drone_pil = bgr_to_pil(drone)
        return lambda p: match_roma(drone_pil, p, matcher, device, args.num_matches)
    return match_factory


def add_args(p):
    p.add_argument("--pretrained",  choices=["outdoor", "indoor", "extre"], default="outdoor")
    p.add_argument("--extre-weights", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "weights", "roma_extre.pth"),
        help="Path to AerialExtreMatch fine-tuned RoMa checkpoint (used when --pretrained extre)")
    p.add_argument("--num-matches", type=int, default=5000)


def main():
    run_pipeline(
        name=lambda a: f"roma_{a.pretrained}",
        label=lambda a: f"RoMa ({a.pretrained}, {a.num_matches} samples)",
        add_args=add_args,
        load_model=load_model,
        make_match_factory=make_match_factory,
    )


if __name__ == "__main__":
    main()
