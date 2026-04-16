import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from visloc_utils import (
    MIN_INL, CROP_W, CROP_H, SZ_W, SZ_H, RANSAC_THRESH, TOP_MATCHES, JPEG_QUALITY,
    load_satellite, gps_to_px, crop_sat, pred_offset_m, print_summary, altitude_scales,
)
from kornia.feature import LightGlue, DISK, DeDoDe

FLIGHT    = "03"
BASE      = f"UAV_Visloc_example/{FLIGHT}"
SAT_TIF   = f"{BASE}/satellite{FLIGHT}.tif"
DRONE_DIR = f"{BASE}/drone"
DRONE_CSV = f"{BASE}/{FLIGHT}.csv"
SAT_CSV   = "UAV_Visloc_example/satellite_ coordinates_range.csv"
OUT_CSV   = "visloc_lightglue_results.csv"
VIZ_DIR   = "visloc_lightglue_visualizations"


def apply_clahe(bgr, clahe):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def bgr_to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)


def extract_disk(bgr, extractor, device, **_kw):
    t = bgr_to_tensor(bgr, device)
    with torch.inference_mode():
        feats = extractor(t, n=4096, pad_if_not_divisible=True)[0]
    return feats.keypoints, feats.descriptors, {}


def extract_dedode(bgr, extractor, device, **_kw):
    t = bgr_to_tensor(bgr, device)
    with torch.inference_mode():
        kp, _, desc = extractor(t, n=4096)
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

    r = dict(sat_kp=len(kps_np), drone_kp=len(kpd_np), raw=0, good=0,
             inliers=0, H=None, _kpd_np=kpd_np, _kps_np=kps_np, _valid=None)

    if len(kpd_np) < 4 or len(kps_np) < 4:
        return r

    img_size = torch.tensor([[SZ_W, SZ_H]], device=device)
    d0 = {"keypoints": kpd.unsqueeze(0).to(device),
          "descriptors": descd.unsqueeze(0).to(device),
          "image_size": img_size}
    d1 = {"keypoints": kps.unsqueeze(0).to(device),
          "descriptors": descs.unsqueeze(0).to(device),
          "image_size": img_size}
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
    H, mask_h = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_thresh,
                                   maxIters=5000, confidence=0.9999)
    if H is not None and mask_h is not None:
        r["inliers"], r["H"] = int(mask_h.sum()), H
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",         type=int,   default=400)
    ap.add_argument("--dist",          type=float, default=25.0,  help="Success radius (m)")
    ap.add_argument("--method",        choices=["disk", "dedodeb", "sift"], default="disk")
    ap.add_argument("--conf",          type=float, default=0.0,   help="LightGlue conf threshold")
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
        extractor = DeDoDe.from_pretrained(
            detector_weights="L-upright", descriptor_weights="B-upright"
        ).eval().to(device)
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
    print(f"  Method: {args.method.upper()} | CLAHE: {args.clahe} | "
          f"Conf: {args.conf} | RANSAC: {ransac_t} | MinInl: {min_inl} | "
          f"Dist: {args.dist}m | {len(df)} images\n")

    if args.visualize:
        os.makedirs(VIZ_DIR, exist_ok=True)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        f, lat, lon = row["filename"], float(row["lat"]), float(row["lon"])
        drone = cv2.imread(os.path.join(DRONE_DIR, f))
        if drone is None:
            rows.append(dict(filename=f, skipped=True)); continue
        drone = cv2.resize(drone, (SZ_W, SZ_H))
        if clahe is not None:
            drone = apply_clahe(drone, clahe)
        cx, cy = gps_to_px(lat, lon, geo)
        kpd, descd, extd = extract_fn(drone, extractor, device, sift_det=sift_det)

        best, best_crop, patch = None, None, None
        for s in altitude_scales(float(row["height"]), geo):
            crop_w = max(SZ_W, int(CROP_W * s))
            crop_h = max(SZ_H, int(CROP_H * s))
            p = crop_sat(sat, cx, cy, geo, crop_w, crop_h)
            if p is None:
                continue
            if clahe is not None:
                p = apply_clahe(p, clahe)
            kps, descs, exts = extract_fn(p, extractor, device, sift_det=sift_det)
            r = match_and_ransac(kpd, descd, extd, kps, descs, exts,
                                 matcher, args.conf, ransac_t, device)
            if best is None or r["inliers"] > best["inliers"]:
                best, best_crop, patch = r, (crop_w, crop_h), p

        if best is None:
            rows.append(dict(filename=f, skipped=True)); continue

        r = best
        off = pred_offset_m(r["H"], cx, cy, *best_crop, geo, lat, lon) if r["inliers"] >= min_inl else None
        off_m, plat, plon = off if off else (None, None, None)
        success = off_m is not None and off_m <= args.dist

        rows.append(dict(filename=f, lat=lat, lon=lon, height=float(row["height"]),
                         skipped=False, crop_w=best_crop[0], crop_h=best_crop[1],
                         sat_kp=r["sat_kp"], drone_kp=r["drone_kp"],
                         raw=r["raw"], good=r["good"], inliers=r["inliers"],
                         inlier_ratio=round(r["inliers"]/r["good"], 4) if r["good"] else 0,
                         pred_lat=round(plat, 7) if plat is not None else None,
                         pred_lon=round(plon, 7) if plon is not None else None,
                         offset_m=round(off_m, 2) if off_m is not None else None,
                         success=success))

        if args.visualize and r["_valid"] is not None and r["_valid"][0].size:
            d_idx, s_idx, conf = r["_valid"]
            kpd_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in r["_kpd_np"]]
            kps_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in r["_kps_np"]]
            top = sorted([cv2.DMatch(int(i), int(j), 1.0 - c) for i, j, c in zip(d_idx, s_idx, conf)],
                         key=lambda m: m.distance)[:TOP_MATCHES]
            viz = cv2.drawMatches(drone, kpd_cv, patch, kps_cv, top, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(os.path.join(VIZ_DIR, f"{os.path.splitext(f)[0]}_matches.jpg"),
                        viz, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    if out.empty or "skipped" not in out.columns:
        print("\n  No images processed."); return
    print_summary(out[~out["skipped"].fillna(False)], args.dist, OUT_CSV, min_inl=min_inl)


if __name__ == "__main__":
    main()
