import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from visloc_utils import (
    MIN_INL, SZ_W, SZ_H, RANSAC_THRESH, TOP_MATCHES, SAT_TIF, SAT_CSV, DRONE_CSV, DRONE_DIR,
    load_satellite, run_pipeline, draw_and_save,
)
from kornia.feature import LightGlue, DISK, DeDoDe

OUT_CSV = "visloc_lightglue_results.csv"
VIZ_DIR = "visloc_lightglue_visualizations"


def bgr_to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)


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


def extract_sift(bgr, _extractor, _device, *, sift_det=None, **_kw):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kps, descs = sift_det.detectAndCompute(gray, None)
    if descs is None or len(kps) < 4:
        return torch.zeros(0, 2), torch.zeros(0, 128), {}
    kp_t  = torch.tensor([k.pt    for k in kps], dtype=torch.float32)
    sc_t  = torch.tensor([k.size  for k in kps], dtype=torch.float32)
    ori_t = torch.deg2rad(torch.tensor([k.angle for k in kps], dtype=torch.float32))
    return kp_t, torch.from_numpy(_rootsift(descs)), {"scales": sc_t, "oris": ori_t}


def match_and_ransac(kpd, descd, extd, kps, descs, exts,
                     matcher, conf_thresh, ransac_thresh, device):
    kpd_np = kpd.cpu().numpy() if kpd.is_cuda else kpd.numpy()
    kps_np = kps.cpu().numpy() if kps.is_cuda else kps.numpy()
    r = dict(sat_kp=len(kps_np), drone_kp=len(kpd_np), raw=0, good=0, inliers=0, H=None,
             _kpd_np=kpd_np, _kps_np=kps_np, _valid=None)
    if len(kpd_np) < 4 or len(kps_np) < 4:
        return r

    img_size = torch.tensor([[SZ_W, SZ_H]], device=device)
    d0 = {"keypoints": kpd.unsqueeze(0).to(device),
          "descriptors": descd.unsqueeze(0).to(device), "image_size": img_size}
    d1 = {"keypoints": kps.unsqueeze(0).to(device),
          "descriptors": descs.unsqueeze(0).to(device), "image_size": img_size}
    for extras, d in [(extd, d0), (exts, d1)]:
        for k in ("scales", "oris"):
            if k in extras:
                d[k] = extras[k].unsqueeze(0).to(device)

    with torch.inference_mode():
        out = matcher({"image0": d0, "image1": d1})

    mi   = out["matches0"][0].cpu().numpy()
    sc   = out["matching_scores0"][0].cpu().numpy()
    mask = (mi >= 0) & (sc >= conf_thresh)
    d_idx = np.where(mask)[0]
    s_idx = mi[d_idx].astype(np.intp)
    conf  = sc[d_idx]

    r["raw"] = r["good"] = len(d_idx)
    r["_valid"] = (d_idx, s_idx, conf)
    if len(d_idx) < 4:
        return r

    src = kpd_np[d_idx].reshape(-1, 1, 2).astype(np.float32)
    dst = kps_np[s_idx].reshape(-1, 1, 2).astype(np.float32)
    H, mh = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_thresh,
                               maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",         type=int,   default=400)
    ap.add_argument("--dist",          type=float, default=25.0)
    ap.add_argument("--method",        choices=["disk", "dedodeb", "sift"], default="disk")
    ap.add_argument("--conf",          type=float, default=0.0)
    ap.add_argument("--clahe",         action="store_true")
    ap.add_argument("--ransac-thresh", type=float, default=None)
    ap.add_argument("--min-inl",       type=int,   default=None)
    ap.add_argument("--visualize",     action="store_true")
    args = ap.parse_args()

    ransac_t = args.ransac_thresh if args.ransac_thresh is not None else RANSAC_THRESH
    min_inl  = args.min_inl       if args.min_inl       is not None else MIN_INL
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print(f"  Loading models ({args.method}) ... ", end="", flush=True)
    sift_det = None
    if args.method == "disk":
        extractor = DISK.from_pretrained("depth").eval().to(device)
        extract_fn, lg_feat = extract_disk, "disk"
    elif args.method == "dedodeb":
        extractor = DeDoDe.from_pretrained(detector_weights="L-upright",
                                           descriptor_weights="B-upright").eval().to(device)
        extract_fn, lg_feat = extract_dedode, "dedodeb"
    else:
        sift_det  = cv2.SIFT_create(nfeatures=8192)
        extractor = None
        extract_fn, lg_feat = extract_sift, "sift"
    matcher = LightGlue(lg_feat, filter_threshold=args.conf,
                        depth_confidence=-1, width_confidence=-1).eval().to(device)
    print("done")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None
    sat, geo = load_satellite(SAT_TIF, SAT_CSV)
    df = pd.read_csv(DRONE_CSV).head(args.limit)
    print(f"  Method: {args.method.upper()} | CLAHE: {args.clahe} | Conf: {args.conf} | "
          f"RANSAC: {ransac_t} | MinInl: {min_inl} | Dist: {args.dist}m | {len(df)} images\n")

    def match_factory(drone):
        kpd, descd, extd = extract_fn(drone, extractor, device, sift_det=sift_det)
        def match_fn(p):
            kps, descs, exts = extract_fn(p, extractor, device, sift_det=sift_det)
            return match_and_ransac(kpd, descd, extd, kps, descs, exts,
                                    matcher, args.conf, ransac_t, device)
        return match_fn

    def viz_fn(drone, patch, best, filename, viz_dir):
        if best["_valid"] is None or not best["_valid"][0].size:
            return
        d_idx, s_idx, conf = best["_valid"]
        kpd_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in best["_kpd_np"]]
        kps_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in best["_kps_np"]]
        top = sorted([cv2.DMatch(int(i), int(j), 1.0 - c) for i, j, c in zip(d_idx, s_idx, conf)],
                     key=lambda m: m.distance)[:TOP_MATCHES]
        draw_and_save(drone, kpd_cv, patch, kps_cv, top, filename, viz_dir)

    run_pipeline(sat, geo, df, match_factory, OUT_CSV, args.dist,
                 min_inl=min_inl, clahe=clahe, drone_dir=DRONE_DIR,
                 viz_fn=viz_fn if args.visualize else None,
                 viz_dir=VIZ_DIR if args.visualize else None)


if __name__ == "__main__":
    main()
