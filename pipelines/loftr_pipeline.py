"""LoFTR (Detector-free local feature matching) + RANSAC."""

import os
import sys

import cv2
import torch

torch.manual_seed(0)

from kornia.feature import LoFTR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import dense_match_result
from helpers.workers import run_pipeline


def img_to_tensor(bgr, device):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return (torch.from_numpy(gray).float().div(255.)
            .unsqueeze(0).unsqueeze(0).to(device))


def match_loftr(drone_t, sat_t, matcher):
    with torch.inference_mode():
        out = matcher({"image0": drone_t, "image1": sat_t})
    return dense_match_result(out["keypoints0"].cpu().numpy(),
                              out["keypoints1"].cpu().numpy(),
                              out["confidence"].cpu().numpy())


def load_model(device, args):
    return LoFTR(pretrained=args.pretrained).eval().to(device)


def make_match_factory(matcher, device, _args):
    def match_factory(drone):
        drone_t = img_to_tensor(drone, device)
        return lambda p: match_loftr(drone_t, img_to_tensor(p, device), matcher)
    return match_factory


def main():
    run_pipeline(
        name="loftr",
        label=lambda a: f"LoFTR ({a.pretrained})",
        add_args=lambda p: p.add_argument("--pretrained",
                                           choices=["outdoor", "indoor"],
                                           default="outdoor"),
        load_model=load_model,
        make_match_factory=make_match_factory,
    )


if __name__ == "__main__":
    main()
