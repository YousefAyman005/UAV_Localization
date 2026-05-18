"""MATCHA (DISK keypoints + learned descriptors) + RANSAC."""

import contextlib
import os
import sys

import cv2
import numpy as np
import torch

torch.manual_seed(0)

# MATCHA hardcodes many CUDA calls. Shim them to no-ops on CPU-only machines.
if not torch.cuda.is_available():
    torch.cuda.synchronize             = lambda *a, **k: None
    torch.cuda.empty_cache             = lambda *a, **k: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.max_memory_allocated    = lambda *a, **k: 0
    torch.Tensor.cuda    = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self

from matcha.feature.matcha_feature import MatchaFeature
from matcha.matcher.base_matcher import BaseMatcher
from matcha.utils.device import to_numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import SZ_W, SZ_H, RANSAC_THRESH
from helpers.workers import run_pipeline


def _bgr_to_tensor(bgr, img_w, img_h, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rs  = cv2.resize(rgb, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    return (torch.from_numpy(rs).float().div(255.)
            .permute(2, 0, 1).unsqueeze(0).to(device))


def _amp_context(device, enabled):
    if enabled and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _oom_hint(img_w, img_h):
    return (f"MATCHA ran out of CUDA memory at {img_w}x{img_h}. "
            "Restart the Kaggle session to clear old models, then retry with "
            "--img-w 512 --img-h 352 --amp. Raise the size only if it fits.")


def extract_features(bgr, matcher, img_w, img_h, device, use_amp):
    try:
        t = _bgr_to_tensor(bgr, img_w, img_h, device)
        with torch.inference_mode(), _amp_context(device, use_amp):
            return matcher.model.detect_and_describe(img=t)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(_oom_hint(img_w, img_h)) from exc


def _mutual_nn_match(desc0, desc1):
    """Mutual-nearest-neighbour matching on L2-normalized descriptors."""
    if desc0.shape[-1] == 0 or desc1.shape[-1] == 0:
        shape = desc0.shape[0], desc0.shape[-1]
        return (torch.full(shape, -1, dtype=torch.long, device=desc0.device),
                torch.zeros(shape, dtype=desc0.dtype, device=desc0.device))
    sim = torch.einsum("bdn,bdm->bnm", desc0, desc1)
    scores, matches = sim.max(dim=-1)

    reverse = sim.transpose(1, 2).argmax(dim=-1)
    src_ids = torch.arange(matches.shape[-1], device=matches.device).expand_as(matches)
    mutual  = src_ids == torch.gather(reverse, -1, matches)
    matches = torch.where(mutual, matches, matches.new_tensor(-1))
    scores  = ((scores + 1.0) * 0.5).clamp(0.0, 1.0)  # cosine → [0, 1]
    scores  = torch.where(mutual, scores, scores.new_zeros(()))
    return matches, scores


def match_features(kpts0, desc0, kpts1, desc1, img_w, img_h, device, conf_thresh=0.0):
    try:
        with torch.inference_mode():
            matches, scores = _mutual_nn_match(desc0, desc1)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(_oom_hint(img_w, img_h)) from exc

    kp0 = to_numpy(kpts0[0]).reshape(-1, 2)
    kp1 = to_numpy(kpts1[0]).reshape(-1, 2)
    m   = to_numpy(matches[0]).reshape(-1).astype(np.int64)
    scr = to_numpy(scores[0]).reshape(-1).astype(np.float32)

    valid = m >= 0
    mid0, mid1 = np.where(valid)[0], m[valid]
    conf = scr[mid0]

    scale = np.array([SZ_W / img_w, SZ_H / img_h], dtype=np.float32)
    kp0_f = (kp0[mid0] * scale).astype(np.float32)
    kp1_f = (kp1[mid1] * scale).astype(np.float32)

    mask = conf >= conf_thresh
    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=int(valid.sum()),
             good=int(mask.sum()), inliers=0, H=None,
             _kp0=kp0_f, _kp1=kp1_f, _conf=conf, _mask=mask)
    if r["good"] >= 4:
        H, mh = cv2.findHomography(kp0_f[mask].reshape(-1, 1, 2),
                                    kp1_f[mask].reshape(-1, 1, 2),
                                    cv2.USAC_MAGSAC, RANSAC_THRESH,
                                    maxIters=5000, confidence=0.9999)
        if H is not None and mh is not None:
            r["inliers"], r["H"] = int(mh.sum()), H
    return r


def load_model(device, args):
    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"MATCHA weights not found at {args.weights}. Download "
            "matcha_pretrained.pth from https://github.com/nv-dvl/matcha and place "
            "it in ./weights/.")
    model = MatchaFeature(config={"keypoint_method": "disk",
                                   "image_size": (args.img_w, args.img_h)})
    incompat = model.load_state_dict(torch.load(args.weights, map_location="cpu"),
                                      strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        print(f"  WARNING — MATCHA weights mismatch: "
              f"missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}")
    return BaseMatcher(model, device)


def make_match_factory(matcher, device, args):
    iw, ih, amp = args.img_w, args.img_h, args.amp

    def match_factory(drone):
        kpd, descd = extract_features(drone, matcher, iw, ih, device, amp)
        def match_fn(p):
            kps, descs = extract_features(p, matcher, iw, ih, device, amp)
            return match_features(kpd, descd, kps, descs, iw, ih, device)
        return match_fn
    return match_factory


def add_args(p):
    p.add_argument("--weights", default="weights/matcha_pretrained.pth")
    p.add_argument("--img-w",   type=int, default=512, help="must be divisible by 32")
    p.add_argument("--img-h",   type=int, default=352, help="must be divisible by 32")
    p.add_argument("--amp",     action="store_true",
                   help="use CUDA fp16 autocast during MATCHA feature extraction")


def main():
    # Sanity-check sizes via a one-shot parser before run_pipeline does its own parse.
    import argparse, sys
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--img-w", type=int, default=512)
    pre.add_argument("--img-h", type=int, default=352)
    pre_args, _ = pre.parse_known_args()
    if pre_args.img_w % 32 or pre_args.img_h % 32:
        sys.exit(f"--img-w/--img-h must be divisible by 32 "
                 f"(got {pre_args.img_w}x{pre_args.img_h})")

    run_pipeline(
        name="matcha",
        add_args=add_args,
        load_model=load_model,
        make_match_factory=make_match_factory,
        banner=lambda a: (f"  Method: MATCHA | Size: {a.img_w}x{a.img_h} | AMP: {a.amp} | "
                          f"RANSAC: {RANSAC_THRESH}px | MinInl: {a.min_inliers} | "
                          f"Dist: {a.dist}m | "
                          f"CLAHE: {'off' if a.no_clahe else 'on'} | "
                          f"Flights: {' '.join(a.flights)}"),
    )


if __name__ == "__main__":
    main()
