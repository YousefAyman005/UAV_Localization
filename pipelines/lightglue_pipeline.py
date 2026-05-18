"""LightGlue matcher with three feature backbones (DISK / DeDoDe / SIFT) + RANSAC."""

import cv2
import numpy as np
import torch

torch.manual_seed(0)

from kornia.feature import LightGlue, DISK, DeDoDe

from visloc_utils import (
    SZ_W, SZ_H, RANSAC_THRESH, TOP_MATCHES,
    draw_and_save, run_pipeline,
)


def bgr_to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (torch.from_numpy(rgb).float().div(255.)
            .permute(2, 0, 1).unsqueeze(0).to(device))


def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# ---- feature extractors ---------------------------------------------------

def extract_disk(bgr, extractor, device, **_kw):
    with torch.inference_mode():
        feats = extractor(bgr_to_tensor(bgr, device), n=4096, pad_if_not_divisible=True)[0]
    return feats.keypoints, feats.descriptors, {}


def extract_dedode(bgr, extractor, device, **_kw):
    with torch.inference_mode():
        kp, _, desc = extractor(bgr_to_tensor(bgr, device), n=4096)
    return kp[0], desc[0], {}


def _rootsift(descs):
    descs = descs.astype(np.float32)
    descs /= descs.sum(axis=1, keepdims=True) + 1e-8
    return np.sqrt(descs)


def extract_sift(bgr, _extractor, _device, *, sift_det, **_kw):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kps, descs = sift_det.detectAndCompute(gray, None)
    if descs is None or len(kps) < 4:
        return torch.zeros(0, 2), torch.zeros(0, 128), {}
    kp_t  = torch.tensor([k.pt    for k in kps], dtype=torch.float32)
    sc_t  = torch.tensor([k.size  for k in kps], dtype=torch.float32)
    ori_t = torch.deg2rad(torch.tensor([k.angle for k in kps], dtype=torch.float32))
    return kp_t, torch.from_numpy(_rootsift(descs)), {"scales": sc_t, "oris": ori_t}


# ---- matching -------------------------------------------------------------

def match_and_ransac(kpd, descd, extd, kps, descs, exts,
                     matcher, device, conf_thresh=0.0):
    kpd_np, kps_np = _to_np(kpd), _to_np(kps)
    r = dict(sat_kp=len(kps_np), drone_kp=len(kpd_np),
             raw=0, good=0, inliers=0, H=None,
             _kpd_np=kpd_np, _kps_np=kps_np, _valid=None)
    if len(kpd_np) < 4 or len(kps_np) < 4:
        return r

    img_size = torch.tensor([[SZ_W, SZ_H]], device=device)
    d0 = {"keypoints":   kpd.unsqueeze(0).to(device),
          "descriptors": descd.unsqueeze(0).to(device), "image_size": img_size}
    d1 = {"keypoints":   kps.unsqueeze(0).to(device),
          "descriptors": descs.unsqueeze(0).to(device), "image_size": img_size}
    for extras, d in [(extd, d0), (exts, d1)]:
        for k in ("scales", "oris"):
            if k in extras:
                d[k] = extras[k].unsqueeze(0).to(device)

    with torch.inference_mode():
        out = matcher({"image0": d0, "image1": d1})

    mi = out["matches0"][0].cpu().numpy()
    sc = out["matching_scores0"][0].cpu().numpy()
    r["raw"] = int((mi >= 0).sum())
    valid = (mi >= 0) & (sc >= conf_thresh)
    d_idx = np.where(valid)[0]
    s_idx = mi[d_idx].astype(np.intp)
    conf  = sc[d_idx]
    r["good"]   = len(d_idx)
    r["_valid"] = (d_idx, s_idx, conf)
    if len(d_idx) < 4:
        return r

    src = kpd_np[d_idx].reshape(-1, 1, 2).astype(np.float32)
    dst = kps_np[s_idx].reshape(-1, 1, 2).astype(np.float32)
    H, mh = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, RANSAC_THRESH,
                                maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


# ---- model loader & match factory ----------------------------------------

_BACKBONES = {
    "disk":    ("disk",    extract_disk,
                lambda dev: DISK.from_pretrained("depth").eval().to(dev)),
    "dedodeb": ("dedodeb", extract_dedode,
                lambda dev: DeDoDe.from_pretrained(detector_weights="L-upright",
                                                    descriptor_weights="B-upright").eval().to(dev)),
    "sift":    ("sift",    extract_sift, lambda _dev: None),
}


def load_model(device, args):
    feat_name, extract_fn, build = _BACKBONES[args.method]
    extractor = build(device)
    matcher   = LightGlue(feat_name, filter_threshold=0.0,
                          depth_confidence=-1, width_confidence=-1).eval().to(device)
    sift_det  = cv2.SIFT_create(nfeatures=8192) if args.method == "sift" else None
    return extractor, matcher, extract_fn, sift_det


def make_match_factory(model, device, _args):
    extractor, matcher, extract_fn, sift_det = model

    def match_factory(drone):
        d0 = extract_fn(drone, extractor, device, sift_det=sift_det)
        return lambda p: match_and_ransac(
            *d0, *extract_fn(p, extractor, device, sift_det=sift_det),
            matcher, device)
    return match_factory


# ---- visualization --------------------------------------------------------

def lg_viz(drone, patch, best, filename, viz_dir):
    kpd_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in best["_kpd_np"]]
    kps_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in best["_kps_np"]]
    valid  = best.get("_valid")
    if valid is None or not valid[0].size:
        top = []
    else:
        d_idx, s_idx, conf = valid
        top = sorted([cv2.DMatch(int(i), int(j), 1.0 - c)
                      for i, j, c in zip(d_idx, s_idx, conf)],
                     key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, kpd_cv, patch, kps_cv, top, filename, viz_dir,
                  H=best.get("H"), m_per_px=best.get("_m_per_px"))


def main():
    run_pipeline(
        name=lambda a: f"lightglue_{a.method}",
        add_args=lambda p: p.add_argument("--method",
                                           choices=["disk", "dedodeb", "sift"],
                                           default="disk"),
        load_model=load_model,
        make_match_factory=make_match_factory,
        viz_fn=lg_viz,
        banner=lambda a: (f"  Method: LightGlue/{a.method.upper()} | "
                          f"RANSAC: {RANSAC_THRESH} | MinInl: {a.min_inliers} | "
                          f"Dist: {a.dist}m | "
                          f"CLAHE: {'off' if a.no_clahe else 'on'} | "
                          f"Flights: {' '.join(a.flights)}"),
    )


if __name__ == "__main__":
    main()
