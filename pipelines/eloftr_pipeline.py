"""Efficient LoFTR (zju3dv/EfficientLoFTR), full model / FP32, + RANSAC.

EfficientLoFTR is the faster, more accurate successor to LoFTR. It is a research
repo (src/ layout) cloned into the container at /opt/EfficientLoFTR rather than
pip-installed, so we add it to sys.path here — kept off the global PYTHONPATH so
its generic top-level `src` package can't shadow anything else.

Differs from loftr_pipeline.py only where the upstream API forces it:
  - input must be grayscale, /255, with H & W divisible by 32 (we pad the
    bottom/right with BORDER_REPLICATE; padding keeps the top-left origin so
    keypoint coordinates stay in the original SZ_W x SZ_H frame — no rescaling),
  - forward mutates a batch dict in place; matches come back as
    mkpts0_f / mkpts1_f / mconf.
The RANSAC/homography step and the returned match-dict contract are identical to
LoFTR's.
"""

import os
import sys
from copy import deepcopy

import cv2
import numpy as np
import torch

torch.manual_seed(0)

sys.path.insert(0, "/opt/EfficientLoFTR")
from src.loftr import LoFTR, full_default_cfg, reparameter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import SZ_W, SZ_H, RANSAC_THRESH  # noqa: E402
from helpers.workers import run_pipeline  # noqa: E402


def _pad32(gray):
    """Pad bottom/right so H, W are multiples of 32 (eLoFTR requirement)."""
    h, w = gray.shape
    nh, nw = -(-h // 32) * 32, -(-w // 32) * 32
    if (nh, nw) == (h, w):
        return gray
    return cv2.copyMakeBorder(gray, 0, nh - h, 0, nw - w, cv2.BORDER_REPLICATE)


def img_to_tensor(bgr, device):
    gray = _pad32(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return (torch.from_numpy(gray).float().div(255.)
            .unsqueeze(0).unsqueeze(0).to(device))


def match_eloftr(drone_t, sat_t, matcher, conf_thresh=0.0):
    batch = {"image0": drone_t, "image1": sat_t}
    with torch.no_grad():
        matcher(batch)
    kp0  = batch["mkpts0_f"].cpu().numpy()
    kp1  = batch["mkpts1_f"].cpu().numpy()
    conf = batch["mconf"].cpu().numpy()
    # Drop matches landing in the replicated pad band so every coordinate fed to
    # RANSAC lives in the real SZ_W x SZ_H crop.
    inb  = ((kp0[:, 0] < SZ_W) & (kp0[:, 1] < SZ_H)
            & (kp1[:, 0] < SZ_W) & (kp1[:, 1] < SZ_H))
    kp0, kp1, conf = kp0[inb], kp1[inb], conf[inb]
    mask = conf >= conf_thresh
    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=len(kp0),
             good=int(mask.sum()), inliers=0, H=None,
             _kp0=kp0, _kp1=kp1, _conf=conf, _mask=mask)
    if r["good"] >= 4:
        H, mh = cv2.findHomography(
            kp0[mask].reshape(-1, 1, 2).astype(np.float32),
            kp1[mask].reshape(-1, 1, 2).astype(np.float32),
            cv2.USAC_MAGSAC, RANSAC_THRESH, maxIters=5000, confidence=0.9999)
        if H is not None and mh is not None:
            r["inliers"], r["H"] = int(mh.sum()), H
    return r


def load_model(device, args):
    matcher = LoFTR(config=deepcopy(full_default_cfg))
    ckpt = torch.load(args.weights, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    matcher.load_state_dict(sd)
    matcher = reparameter(matcher)  # fuse RepVGG branches for inference
    return matcher.eval().to(device)


def make_match_factory(matcher, device, _args):
    def match_factory(drone):
        drone_t = img_to_tensor(drone, device)
        return lambda p: match_eloftr(drone_t, img_to_tensor(p, device), matcher)
    return match_factory


def add_args(p):
    p.add_argument("--weights", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "weights", "eloftr_outdoor.ckpt"),
        help="Path to the EfficientLoFTR checkpoint (eloftr_outdoor.ckpt).")


def main():
    run_pipeline(
        name="eloftr",
        add_args=add_args,
        load_model=load_model,
        make_match_factory=make_match_factory,
        banner=lambda a: (f"  Method: EfficientLoFTR (full/fp32) | RANSAC: {RANSAC_THRESH} | "
                          f"MinInl: {a.min_inliers} | Dist: {a.dist}m | "
                          f"CLAHE: {'off' if a.no_clahe else 'on'} | "
                          f"Flights: {' '.join(a.flights)}"),
    )


if __name__ == "__main__":
    main()
