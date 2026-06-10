"""Classical-feature baselines: SIFT / ORB / BRISK + RANSAC.

Run e.g.: ./.venv/bin/python3 Baseline_pipeline.py --method sift --flights 03 --limit 5
"""

import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.utils import TOP_MATCHES, fit_similarity
from helpers.visualization import draw_and_save
from helpers.workers import run_pipeline

LOWE = 0.75
FLANN_TREES, FLANN_CHECKS = 5, 50


def _make_detector(method):
    return {"sift":  cv2.SIFT_create,
            "orb":   lambda: cv2.ORB_create(5000),
            "brisk": cv2.BRISK_create}[method]()


def _make_matcher(method):
    if method == "sift":
        return cv2.FlannBasedMatcher({"algorithm": 1, "trees": FLANN_TREES},
                                      {"checks":   FLANN_CHECKS})
    return cv2.BFMatcher(cv2.NORM_HAMMING)


def run_match(sat_gray, kpd, dd, detector, matcher):
    kps, ds = detector.detectAndCompute(sat_gray, None)
    r = dict(sat_kp=len(kps), drone_kp=len(kpd), raw=0, good=0, inliers=0, H=None,
             _kps=kps, _kpd=kpd, _matches=[])
    if ds is None or dd is None or len(kps) < 4 or len(kpd) < 4:
        return r
    pairs = matcher.knnMatch(dd, ds, k=2)
    good  = [m for pair in pairs if len(pair) == 2
             for m, n in [pair] if m.distance < LOWE * n.distance]
    r["raw"], r["good"], r["_matches"] = len(pairs), len(good), good
    if len(good) >= 4:
        src = np.float32([kpd[m.queryIdx].pt for m in good])
        dst = np.float32([kps[m.trainIdx].pt for m in good])
        H, ninl = fit_similarity(src, dst)
        if H is not None:
            r["inliers"], r["H"] = ninl, H
    return r


def load_model(_device, _args):
    return None  # OpenCV detectors are constructed per-worker in make_match_factory


def add_args(p):
    p.add_argument("--method", choices=["sift", "orb", "brisk"], default="sift")


def make_match_factory(_model, _device, args):
    detector = _make_detector(args.method)
    matcher  = _make_matcher(args.method)

    def match_factory(drone):
        kpd, dd = detector.detectAndCompute(cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY), None)
        return lambda p: run_match(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY),
                                    kpd, dd, detector, matcher)
    return match_factory


def baseline_viz(drone, patch, best, filename, viz_dir):
    matches = best.get("_matches") or []
    top = sorted(matches, key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, best.get("_kpd", []), patch, best.get("_kps", []),
                  top, filename, viz_dir, H=best.get("H"),
                  m_per_px=best.get("_m_per_px"), gt_px=best.get("_gt_px"))


def banner(a):
    return (f"  Method: {a.method.upper()} | Dist: {a.dist}m | "
            f"CLAHE: {'off' if a.no_clahe else 'on'} | "
            f"Workers: {a.workers or 'auto'} | Flights: {' '.join(a.flights)}")


def main():
    run_pipeline(
        name=lambda a: a.method,
        add_args=add_args,
        load_model=load_model,
        make_match_factory=make_match_factory,
        viz_fn=baseline_viz,
        banner=banner,
        parallelism="cpu_chunks",
    )


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
