"""Classical-feature baselines: SIFT / ORB / BRISK + RANSAC.

Run e.g.: python pipelines/baseline_pipeline.py --method sift --flights 03 --limit 5
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
N_FEATURES = 5000  # common keypoint budget so the speed column is apples-to-apples


def _make_detector(method):
    # SIFT/ORB cap internally via nfeatures; BRISK has no such arg and is capped
    # post-hoc in _detect. Without a shared budget BRISK fires ~28k keypoints and
    # its brute-force Hamming match (O(N_drone*N_sat)) dwarfs the others.
    return {"sift":  lambda: cv2.SIFT_create(N_FEATURES),
            "orb":   lambda: cv2.ORB_create(N_FEATURES),
            "brisk": cv2.BRISK_create}[method]()


def _detect(detector, gray):
    """detectAndCompute capped to the top N_FEATURES keypoints by response, so
    every baseline matches the same budget (no-op for SIFT/ORB, which cap inside)."""
    kps, ds = detector.detectAndCompute(gray, None)
    if ds is not None and len(kps) > N_FEATURES:
        idx = np.argsort([-k.response for k in kps])[:N_FEATURES]
        kps = [kps[i] for i in idx]
        ds  = ds[idx]
    return kps, ds


def _make_matcher(method):
    if method == "sift":
        return cv2.FlannBasedMatcher({"algorithm": 1, "trees": FLANN_TREES},
                                      {"checks":   FLANN_CHECKS})
    return cv2.BFMatcher(cv2.NORM_HAMMING)


def run_match(sat_gray, kpd, dd, detector, matcher):
    kps, ds = _detect(detector, sat_gray)
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
        kpd, dd = _detect(detector, cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY))
        return lambda p: run_match(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY),
                                    kpd, dd, detector, matcher)
    return match_factory


def baseline_viz(drone, patch, best, filename, viz_dir):
    matches = best.get("_matches") or []
    top = sorted(matches, key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, best.get("_kpd", []), patch, best.get("_kps", []),
                  top, filename, viz_dir, H=best.get("H"),
                  m_per_px=best.get("_m_per_px"), gt_px=best.get("_gt_px"))


def main():
    run_pipeline(
        name=lambda a: a.method,
        label=lambda a: a.method.upper(),
        add_args=add_args,
        load_model=load_model,
        make_match_factory=make_match_factory,
        viz_fn=baseline_viz,
        parallelism="cpu_chunks",
    )


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
