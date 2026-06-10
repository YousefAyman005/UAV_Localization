"""LoFTR (Detector-free local feature matching) + RANSAC."""

import os
import sys

import cv2
import numpy as np
import torch

torch.manual_seed(0)

from kornia.feature import LoFTR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import RANSAC_THRESH, fit_similarity
from helpers.workers import run_pipeline


def img_to_tensor(bgr, device):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return (torch.from_numpy(gray).float().div(255.)
            .unsqueeze(0).unsqueeze(0).to(device))


def match_loftr(drone_t, sat_t, matcher, conf_thresh=0.0):
    with torch.inference_mode():
        out = matcher({"image0": drone_t, "image1": sat_t})
    kp0  = out["keypoints0"].cpu().numpy()
    kp1  = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    mask = conf >= conf_thresh
    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=len(kp0),
             good=int(mask.sum()), inliers=0, H=None,
             _kp0=kp0, _kp1=kp1, _conf=conf, _mask=mask)
    if r["good"] >= 4:
        H, ninl = fit_similarity(kp0[mask], kp1[mask])
        if H is not None:
            r["inliers"], r["H"] = ninl, H
    return r


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
        add_args=lambda p: p.add_argument("--pretrained",
                                           choices=["outdoor", "indoor"],
                                           default="outdoor"),
        load_model=load_model,
        make_match_factory=make_match_factory,
        banner=lambda a: (f"  Method: LoFTR ({a.pretrained}) | RANSAC(sim-4dof): {RANSAC_THRESH} | "
                          f"MinInl: {a.min_inliers} | Dist: {a.dist}m | "
                          f"CLAHE: {'off' if a.no_clahe else 'on'} | "
                          f"Flights: {' '.join(a.flights)}"),
    )


if __name__ == "__main__":
    main()
