"""RoMa (Robust dense matcher) + RANSAC."""

import cv2
import numpy as np
import torch
from PIL import Image

from romatch import roma_outdoor, roma_indoor

from visloc_utils import SZ_W, SZ_H, RANSAC_THRESH, run_pipeline


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def match_roma(drone_pil, sat_bgr, matcher, device, num_samples, conf_thresh=0.0):
    with torch.inference_mode():
        warp, cert = matcher.match(drone_pil, bgr_to_pil(sat_bgr), device=device)
        matches, c = matcher.sample(warp, cert, num=num_samples)
        kp_a, kp_b = matcher.to_pixel_coordinates(matches, SZ_H, SZ_W, SZ_H, SZ_W)
    kp0  = kp_a.cpu().numpy().astype(np.float32)
    kp1  = kp_b.cpu().numpy().astype(np.float32)
    conf = c.cpu().numpy()
    mask = conf >= conf_thresh
    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=len(kp0),
             good=int(mask.sum()), inliers=0, H=None,
             _kp0=kp0, _kp1=kp1, _conf=conf, _mask=mask)
    if r["good"] >= 4:
        H, mh = cv2.findHomography(kp0[mask].reshape(-1, 1, 2),
                                    kp1[mask].reshape(-1, 1, 2),
                                    cv2.USAC_MAGSAC, RANSAC_THRESH,
                                    maxIters=5000, confidence=0.9999)
        if H is not None and mh is not None:
            r["inliers"], r["H"] = int(mh.sum()), H
    return r


def load_model(device, args):
    torch.set_float32_matmul_precision("highest")
    kw = {} if device.type == "cuda" else {"amp_dtype": torch.float32}
    builder = roma_outdoor if args.pretrained == "outdoor" else roma_indoor
    return builder(device=device, **kw)


def make_match_factory(matcher, device, args):
    def match_factory(drone):
        drone_pil = bgr_to_pil(drone)
        return lambda p: match_roma(drone_pil, p, matcher, device, args.num_matches)
    return match_factory


def add_args(p):
    p.add_argument("--pretrained",  choices=["outdoor", "indoor"], default="outdoor")
    p.add_argument("--num-matches", type=int, default=5000)


def main():
    run_pipeline(
        name="roma",
        add_args=add_args,
        load_model=load_model,
        make_match_factory=make_match_factory,
        banner=lambda a: (f"  Method: RoMa ({a.pretrained}) | NumMatches: {a.num_matches} | "
                          f"RANSAC: {RANSAC_THRESH}px | MinInl: {a.min_inliers} | "
                          f"Dist: {a.dist}m | "
                          f"CLAHE: {'off' if a.no_clahe else 'on'} | "
                          f"Flights: {' '.join(a.flights)}"),
    )


if __name__ == "__main__":
    main()
