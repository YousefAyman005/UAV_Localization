import argparse
import multiprocessing
import cv2
import numpy as np
import pandas as pd
import torch
from visloc_utils import (
    MIN_INL, SZ_W, SZ_H, RANSAC_THRESH, TOP_MATCHES,
    FLIGHTS_AVAILABLE, load_flight, collect_pipeline_rows_multitile,
    print_summary, draw_and_save, TeeLogger,
)
from kornia.feature import LightGlue, DISK, DeDoDe

OUT_CSV_TEMPLATE = "visloc_lightglue_{method}_results.csv"
VIZ_DIR_TEMPLATE = "visloc_lightglue_{method}_visualizations"


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
    r["raw"] = int((mi >= 0).sum())
    mask = (mi >= 0) & (sc >= conf_thresh)
    d_idx = np.where(mask)[0]
    s_idx = mi[d_idx].astype(np.intp)
    conf  = sc[d_idx]

    r["good"] = len(d_idx)
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


def _load_model(device, method):
    sift_det = None
    if method == "disk":
        extractor = DISK.from_pretrained("depth").eval().to(device)
        extract_fn, lg_feat = extract_disk, "disk"
    elif method == "dedodeb":
        extractor = DeDoDe.from_pretrained(detector_weights="L-upright",
                                           descriptor_weights="B-upright").eval().to(device)
        extract_fn, lg_feat = extract_dedode, "dedodeb"
    else:
        sift_det  = cv2.SIFT_create(nfeatures=8192)
        extractor = None
        extract_fn, lg_feat = extract_sift, "sift"
    matcher = LightGlue(lg_feat, filter_threshold=0.0,
                        depth_confidence=-1, width_confidence=-1).eval().to(device)
    return extractor, matcher, extract_fn, sift_det


def _lg_viz_fn(drone, patch, best, filename, viz_dir):
    if best["_valid"] is None or not best["_valid"][0].size:
        return
    d_idx, s_idx, conf = best["_valid"]
    kpd_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in best["_kpd_np"]]
    kps_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in best["_kps_np"]]
    top = sorted([cv2.DMatch(int(i), int(j), 1.0 - c) for i, j, c in zip(d_idx, s_idx, conf)],
                 key=lambda m: m.distance)[:TOP_MATCHES]
    draw_and_save(drone, kpd_cv, patch, kps_cv, top, filename, viz_dir)


def _make_match_factory(extractor, matcher, extract_fn, sift_det, device):
    def match_factory(drone):
        d0 = extract_fn(drone, extractor, device, sift_det=sift_det)
        return lambda p: match_and_ransac(
            *d0, *extract_fn(p, extractor, device, sift_det=sift_det),
            matcher, 0.0, RANSAC_THRESH, device)
    return match_factory


def collect_flight_rows(flight, match_factory, dist, viz_dir, progress=True):
    tiles, drone_dir, drone_csv, _ = load_flight(flight)
    df = pd.read_csv(drone_csv)
    if progress: print(f"\n=== Flight {flight}: {len(df)} images ===")
    return collect_pipeline_rows_multitile(
        tiles, df, match_factory, dist, min_inl=MIN_INL,
        drone_dir=drone_dir, flight=flight,
        viz_fn=_lg_viz_fn if viz_dir else None, viz_dir=viz_dir, progress=progress)


def summarize_rows(rows, label):
    if not rows: return
    df = pd.DataFrame(rows)
    if "skipped" not in df: return
    valid = df[~df["skipped"].fillna(False)]
    if not valid.empty: print_summary(valid, label, min_inl=MIN_INL)


def _worker(args):
    flight_group, gpu_id, method, dist, viz_dir = args
    device = torch.device(f"cuda:{gpu_id}")
    extractor, matcher, extract_fn, sift_det = _load_model(device, method)
    match_factory = _make_match_factory(extractor, matcher, extract_fn, sift_det, device)
    return [r for f in flight_group for r in collect_flight_rows(f, match_factory, dist, viz_dir, False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist",      type=float, default=25.0)
    ap.add_argument("--method",    choices=["disk", "dedodeb", "sift"], default="disk")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--flights",   nargs="+", default=["all"],
                    help="Flight IDs to evaluate, e.g. 01 03 05, or 'all' (default)")
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    n_gpus  = max(1, torch.cuda.device_count())
    OUT_CSV = OUT_CSV_TEMPLATE.format(method=args.method)
    VIZ_DIR = VIZ_DIR_TEMPLATE.format(method=args.method)
    viz_dir = VIZ_DIR if args.visualize else None
    print(f"  Method: {args.method.upper()} | RANSAC: {RANSAC_THRESH} | MinInl: {MIN_INL} | "
          f"Dist: {args.dist}m | Flights: {' '.join(flights)} | GPUs: {n_gpus}")

    groups = [g for g in [flights[i::n_gpus] for i in range(n_gpus)] if g]

    log_path = OUT_CSV.replace(".csv", ".log")
    with TeeLogger(log_path):
        if len(groups) == 1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"  Device: {device}")
            print(f"  Loading models ({args.method}) ... ", end="", flush=True)
            extractor, matcher, extract_fn, sift_det = _load_model(device, args.method)
            print("done")

            match_factory = _make_match_factory(extractor, matcher, extract_fn, sift_det, device)
            all_rows = []
            for flight in flights:
                rows = collect_flight_rows(flight, match_factory, args.dist, viz_dir)
                all_rows.extend(rows)
                summarize_rows(rows, f"flight {flight}")
        else:
            ctx = multiprocessing.get_context("spawn")
            worker_args = [(g, i, args.method, args.dist, viz_dir)
                           for i, g in enumerate(groups)]
            with ctx.Pool(len(groups)) as pool:
                results = pool.map(_worker, worker_args)
            all_rows = [r for chunk in results for r in chunk]
            for flight in flights:
                summarize_rows([r for r in all_rows if r.get("flight") == flight],
                               f"flight {flight}")

        out = pd.DataFrame(all_rows)
        out.to_csv(OUT_CSV, index=False)
        if len(flights) > 1:
            print(f"\n=== Overall ({len(flights)} flights) ===")
            summarize_rows(all_rows, OUT_CSV)


if __name__ == "__main__":
    main()
