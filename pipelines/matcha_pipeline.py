import argparse
import contextlib
import multiprocessing
import os
import cv2
import numpy as np
import pandas as pd
import torch

# MATCHA hardcodes many CUDA calls. Shim them to no-ops on CPU-only machines.
if not torch.cuda.is_available():
    torch.cuda.synchronize      = lambda *a, **k: None
    torch.cuda.empty_cache      = lambda *a, **k: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.max_memory_allocated    = lambda *a, **k: 0
    torch.Tensor.cuda = lambda self, *a, **k: self                   # tensor.cuda() -> self
    torch.nn.Module.cuda = lambda self, *a, **k: self                # module.cuda() -> self

from pipelines.visloc_utils import (
    MIN_INL, SZ_W, SZ_H, RANSAC_THRESH,
    FLIGHTS_AVAILABLE, load_flight, collect_pipeline_rows_multitile,
    print_summary, save_dense_viz, TeeLogger,
)
from matcha.feature.matcha_feature import MatchaFeature
from matcha.matcher.base_matcher import BaseMatcher
from matcha.utils.device import to_numpy

OUT_CSV = "visloc_matcha_results.csv"
VIZ_DIR = "visloc_matcha_visualizations"


def bgr_to_tensor(bgr, img_w, img_h, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rs  = cv2.resize(rgb, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(rs).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)


def cuda_cleanup(device):
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def amp_context(device, enabled):
    if enabled and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def oom_hint(img_w, img_h):
    return (f"MATCHA ran out of CUDA memory at {img_w}x{img_h}. "
            "Restart the Kaggle session to clear old models, then retry with "
            "--img-w 512 --img-h 352 --amp. Raise the size only if it fits.")


def extract_matcha_features(bgr, matcher, img_w, img_h, device, use_amp):
    t = None
    try:
        t = bgr_to_tensor(bgr, img_w, img_h, device)
        with torch.inference_mode(), amp_context(device, use_amp):
            kpts, desc = matcher.model.detect_and_describe(img=t)
    except torch.cuda.OutOfMemoryError as exc:
        cuda_cleanup(device)
        raise RuntimeError(oom_hint(img_w, img_h)) from exc
    finally:
        if t is not None:
            del t
    return kpts, desc


def match_matcha_features(kpts0, desc0, kpts1, desc1, img_w, img_h, conf_thresh, ransac_thresh, device):
    try:
        with torch.inference_mode():
            matches, scores = matcher_matches(desc0, desc1)
    except torch.cuda.OutOfMemoryError as exc:
        cuda_cleanup(device)
        raise RuntimeError(oom_hint(img_w, img_h)) from exc

    kp0 = np.asarray(to_numpy(kpts0[0])).reshape(-1, 2)
    kp1 = np.asarray(to_numpy(kpts1[0])).reshape(-1, 2)
    m   = np.asarray(to_numpy(matches[0])).reshape(-1).astype(np.int64)
    scr = np.asarray(to_numpy(scores[0])).reshape(-1).astype(np.float32)

    valid = m >= 0
    mid0, mid1 = np.where(valid)[0], m[valid]
    conf = scr[mid0]

    scale = np.array([SZ_W / img_w, SZ_H / img_h], dtype=np.float32)
    kp0_f = (kp0[mid0] * scale).astype(np.float32)
    kp1_f = (kp1[mid1] * scale).astype(np.float32)

    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=int(valid.sum()), good=0, inliers=0,
             H=None, _kp0=kp0_f, _kp1=kp1_f, _conf=conf, _mask=None)
    mask = conf >= conf_thresh
    r["good"], r["_mask"] = int(mask.sum()), mask
    if r["good"] < 4:
        return r
    H, mh = cv2.findHomography(kp0_f[mask].reshape(-1, 1, 2),
                               kp1_f[mask].reshape(-1, 1, 2),
                               cv2.USAC_MAGSAC, ransac_thresh,
                               maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


def matcher_matches(desc0, desc1):
    if desc0.shape[-1] == 0 or desc1.shape[-1] == 0:
        shape = desc0.shape[0], desc0.shape[-1]
        return (
            torch.full(shape, -1, dtype=torch.long, device=desc0.device),
            torch.zeros(shape, dtype=desc0.dtype, device=desc0.device),
        )

    sim = torch.einsum("bdn,bdm->bnm", desc0, desc1)
    scores, matches = sim.max(dim=-1)

    reverse = sim.transpose(1, 2).argmax(dim=-1)
    src_ids = torch.arange(matches.shape[-1], device=matches.device).expand_as(matches)
    mutual = src_ids == torch.gather(reverse, -1, matches)
    matches = torch.where(mutual, matches, matches.new_tensor(-1))

    # Descriptors are L2-normalized, so cosine similarity maps cleanly to [0, 1].
    scores = ((scores + 1.0) * 0.5).clamp(0.0, 1.0)
    scores = torch.where(mutual, scores, scores.new_zeros(()))
    return matches, scores


def _load_model(device, weights, img_w, img_h):
    model = MatchaFeature(config={"keypoint_method": "disk",
                                   "image_size": (img_w, img_h)})
    incompatible = model.load_state_dict(
        torch.load(weights, map_location="cpu"), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"  WARNING — MATCHA weights mismatch: "
              f"missing={incompatible.missing_keys}, "
              f"unexpected={incompatible.unexpected_keys}")
    return BaseMatcher(model, device)


def _make_match_factory(matcher, img_w, img_h, device, use_amp):
    def match_factory(drone):
        kpd, descd = extract_matcha_features(drone, matcher, img_w, img_h, device, use_amp)
        def match_fn(p):
            kps, descs = extract_matcha_features(p, matcher, img_w, img_h, device, use_amp)
            return match_matcha_features(kpd, descd, kps, descs,
                                         img_w, img_h, 0.0, RANSAC_THRESH, device)
        return match_fn
    return match_factory


def collect_flight_rows(flight, match_factory, dist, viz_dir, progress=True):
    tiles, drone_dir, drone_csv, _ = load_flight(flight)
    df = pd.read_csv(drone_csv)
    if progress: print(f"\n=== Flight {flight}: {len(df)} images ===")
    return collect_pipeline_rows_multitile(
        tiles, df, match_factory, dist, min_inl=MIN_INL,
        drone_dir=drone_dir, flight=flight,
        viz_fn=save_dense_viz if viz_dir else None, viz_dir=viz_dir, progress=progress)


def summarize_rows(rows, label):
    if not rows: return
    df = pd.DataFrame(rows)
    if "skipped" not in df: return
    valid = df[~df["skipped"].fillna(False)]
    if not valid.empty: print_summary(valid, label, min_inl=MIN_INL)


def _worker(args):
    flight_group, gpu_id, weights, img_w, img_h, dist, amp, viz_dir = args
    device = torch.device(f"cuda:{gpu_id}")
    matcher = _load_model(device, weights, img_w, img_h)
    match_factory = _make_match_factory(matcher, img_w, img_h, device, amp)
    return [r for f in flight_group for r in collect_flight_rows(f, match_factory, dist, viz_dir, False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist",      type=float, default=25.0)
    ap.add_argument("--weights",   type=str,   default="weights/matcha_pretrained.pth")
    ap.add_argument("--img-w",     type=int,   default=512, help="must be divisible by 32")
    ap.add_argument("--img-h",     type=int,   default=352, help="must be divisible by 32")
    ap.add_argument("--amp",       action="store_true",
                    help="use CUDA fp16 autocast during MATCHA feature extraction")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--flights",   nargs="+", default=["all"],
                    help="Flight IDs to evaluate, e.g. 01 03 05, or 'all' (default)")
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights

    if args.img_w % 32 or args.img_h % 32:
        raise ValueError(f"--img-w/--img-h must be divisible by 32 (got {args.img_w}x{args.img_h})")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"MATCHA weights not found at {args.weights}. "
            "Download matcha_pretrained.pth from https://github.com/nv-dvl/matcha "
            "and place it in ./weights/."
        )

    n_gpus  = max(1, torch.cuda.device_count())
    viz_dir = VIZ_DIR if args.visualize else None
    print(f"  Method: MATCHA | Size: {args.img_w}x{args.img_h} | AMP: {args.amp} | "
          f"RANSAC: {RANSAC_THRESH}px | MinInl: {MIN_INL} | "
          f"Dist: {args.dist}m | Flights: {' '.join(flights)} | GPUs: {n_gpus}")

    groups = [g for g in [flights[i::n_gpus] for i in range(n_gpus)] if g]

    log_path = OUT_CSV.replace(".csv", ".log")
    with TeeLogger(log_path):
        if len(groups) == 1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  Device: {device}")
            print(f"  Loading MATCHA (disk, {args.img_w}x{args.img_h}) ... ", end="", flush=True)
            matcher = _load_model(device, args.weights, args.img_w, args.img_h)
            print("done")

            match_factory = _make_match_factory(matcher, args.img_w, args.img_h, device, args.amp)
            all_rows = []
            for flight in flights:
                rows = collect_flight_rows(flight, match_factory, args.dist, viz_dir)
                all_rows.extend(rows)
                summarize_rows(rows, f"flight {flight}")
        else:
            ctx = multiprocessing.get_context("spawn")
            worker_args = [(g, i, args.weights, args.img_w, args.img_h,
                            args.dist, args.amp, viz_dir)
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
