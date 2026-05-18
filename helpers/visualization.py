"""Visualization helpers for the feature-matching pipelines.

Draws GT/predicted markers on cropped patches and stitches drone↔patch
match views. Used by every matcher pipeline (sparse + dense) via the
`viz_fn` argument of `collect_pipeline_rows_multitile`.
"""

import math
import os
import shutil

import cv2
import numpy as np

from helpers.utils import JPEG_QUALITY, SZ_H, SZ_W, TOP_MATCHES


def _draw_overlays(patch, H, m_per_px=None):
    """Green cross+circle = GT (patch centre); yellow/red = predicted point."""
    out = (patch if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)).copy()
    gx, gy, arm = SZ_W // 2, SZ_H // 2, 18
    cv2.line(out, (gx - arm, gy), (gx + arm, gy), (0, 220, 0), 3)
    cv2.line(out, (gx, gy - arm), (gx, gy + arm), (0, 220, 0), 3)
    cv2.circle(out, (gx, gy), arm + 6, (0, 220, 0), 2)

    if m_per_px and m_per_px > 0:
        for metres, col in ((20, (255, 255, 0)), (25, (255, 255, 255))):
            r = max(1, int(round(metres / m_per_px)))
            cv2.circle(out, (gx, gy), r, col, 1, cv2.LINE_AA)
            cv2.putText(out, f"{metres}m", (gx + r + 4, gy - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    if H is None:
        return out
    px = cv2.perspectiveTransform(
        np.float32([[SZ_W / 2, SZ_H / 2]]).reshape(-1, 1, 2), H).reshape(2)
    pxi = (int(round(float(px[0]))), int(round(float(px[1]))))
    cv2.line(out, (gx, gy), pxi, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.circle(out, pxi, 8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(out, pxi, 4, (0, 0, 255), -1, cv2.LINE_AA)
    if m_per_px and m_per_px > 0:
        err_m = math.hypot(float(px[0]) - SZ_W / 2, float(px[1]) - SZ_H / 2) * m_per_px
        cv2.putText(out, f"{err_m:.1f}m", (pxi[0] + 10, pxi[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def draw_and_save(drone, kpd, patch, kps, matches, filename, viz_dir,
                  H=None, m_per_px=None):
    patch = _draw_overlays(patch, H, m_per_px=m_per_px)
    viz = cv2.drawMatches(drone, kpd, patch, kps, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    sep = drone.shape[1]
    cv2.line(viz, (sep, 0), (sep, viz.shape[0] - 1), (255, 255, 255), 3)
    cv2.imwrite(os.path.join(viz_dir, f"{os.path.splitext(filename)[0]}_matches.jpg"),
                viz, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def save_dense_viz(drone, patch, best, filename, viz_dir):
    """Dense-matcher viz (LoFTR/RoMa/MATCHA)."""
    kp0, kp1 = best.get("_kp0"), best.get("_kp1")
    if kp0 is None or kp1 is None:
        return
    mask = best.get("_mask")
    if mask is None or mask.sum() == 0:
        kpd, kps, top = [], [], []
    else:
        conf = best["_conf"]
        kpd  = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0[mask]]
        kps  = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1[mask]]
        top  = sorted([cv2.DMatch(i, i, 1.0 - c) for i, c in enumerate(conf[mask])],
                      key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, kpd, patch, kps, top, filename, viz_dir,
                  H=best.get("H"), m_per_px=best.get("_m_per_px"))


def setup_viz_dir(viz_dir):
    if viz_dir is None:
        return
    shutil.rmtree(viz_dir, ignore_errors=True)
    os.makedirs(viz_dir, exist_ok=True)
