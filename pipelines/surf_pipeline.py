"""SURF (Speeded-Up Robust Features) + RANSAC pipeline.

Requires opencv-contrib-python (cv2.xfeatures2d).  SURF descriptors are
float32, so FLANN KD-tree matching is used — same as SIFT.

Run e.g.: python pipelines/surf_pipeline.py --flights 03 --limit 5
          python pipelines/surf_pipeline.py --hessian-threshold 800 --extended
"""

import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import RANSAC_THRESH, TOP_MATCHES
from helpers.visualization import draw_and_save
from helpers.workers import run_pipeline

LOWE = 0.75
FLANN_TREES, FLANN_CHECKS = 5, 50

DEFAULT_HESSIAN = 400


def _make_surf(hessian_threshold, extended, upright):
    try:
        surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            extended=extended,
            upright=upright,
        )
    except AttributeError:
        raise RuntimeError(
            "cv2.xfeatures2d not found — install opencv-contrib-python."
        )
    return surf


def _make_flann():
    return cv2.FlannBasedMatcher(
        {"algorithm": 1, "trees": FLANN_TREES},
        {"checks": FLANN_CHECKS},
    )


def run_match(sat_gray, kpd, dd, surf, flann):
    kps, ds = surf.detectAndCompute(sat_gray, None)
    r = dict(sat_kp=len(kps), drone_kp=len(kpd), raw=0, good=0, inliers=0, H=None,
             _kps=kps, _kpd=kpd, _matches=[])
    if ds is None or dd is None or len(kps) < 4 or len(kpd) < 4:
        return r
    # FLANN requires contiguous float32
    dd_f = np.ascontiguousarray(dd, dtype=np.float32)
    ds_f = np.ascontiguousarray(ds, dtype=np.float32)
    pairs = flann.knnMatch(dd_f, ds_f, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < LOWE * n.distance]
    r["raw"], r["good"], r["_matches"] = len(pairs), len(good), good
    if len(good) >= 4:
        src = np.float32([kpd[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
        if H is not None and mask is not None:
            r["inliers"], r["H"] = int(mask.sum()), H
    return r


def load_model(_device, _args):
    return None  # OpenCV detectors constructed per-worker in make_match_factory


def add_args(p):
    p.add_argument("--hessian-threshold", type=int, default=DEFAULT_HESSIAN,
                   help="SURF Hessian threshold (default: %(default)s)")
    p.add_argument("--extended", action="store_true",
                   help="Use 128-dim extended SURF descriptors (default: 64-dim)")
    p.add_argument("--upright", action="store_true",
                   help="Disable rotation invariance (faster, upright scenes only)")


def make_match_factory(_model, _device, args):
    surf  = _make_surf(args.hessian_threshold, args.extended, args.upright)
    flann = _make_flann()

    def match_factory(drone):
        gray = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)
        kpd, dd = surf.detectAndCompute(gray, None)
        return lambda p: run_match(
            cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), kpd, dd, surf, flann
        )
    return match_factory


def surf_viz(drone, patch, best, filename, viz_dir):
    matches = best.get("_matches") or []
    top = sorted(matches, key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, best.get("_kpd", []), patch, best.get("_kps", []),
                  top, filename, viz_dir, H=best.get("H"),
                  m_per_px=best.get("_m_per_px"))


def banner(a):
    desc = "128-dim" if a.extended else "64-dim"
    orient = "upright" if a.upright else "rotation-invariant"
    return (f"  Method: SURF | Hessian: {a.hessian_threshold} | "
            f"Desc: {desc} | Orient: {orient} | "
            f"CLAHE: {'off' if a.no_clahe else 'on'} | "
            f"Workers: {a.workers or 'auto'} | Flights: {' '.join(a.flights)}")


def main():
    run_pipeline(
        name=lambda _: "surf",
        add_args=add_args,
        load_model=load_model,
        make_match_factory=make_match_factory,
        viz_fn=surf_viz,
        banner=banner,
        parallelism="cpu_chunks",
    )


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
