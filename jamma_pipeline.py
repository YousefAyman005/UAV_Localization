import argparse
import multiprocessing
import sys
import time
import types
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from visloc_utils import (
    MIN_INL, RANSAC_THRESH,
    FLIGHTS_AVAILABLE, load_flight, collect_pipeline_rows_multitile,
    print_summary, save_dense_viz, TeeLogger,
)

OUT_CSV = "visloc_jamma_results.csv"
VIZ_DIR = "visloc_jamma_visualizations"

JAMMA_URL      = "https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt"
JAMMA_CKPT_DEFAULT = "weights/jamma.ckpt"

# Verbatim from upstream demo/utlis.py — do not add or remove keys.
JAMMA_CFG = {
    "coarse": {"d_model": 256},
    "fine": {
        "d_model": 64,
        "dsmax_temperature": 0.1,
        "thr": 0.1,
        "inference": True,
    },
    "match_coarse": {
        "thr": 0.2,
        "use_sm": True,
        "border_rm": 2,
        "dsmax_temperature": 0.1,
        "inference": True,
    },
    "fine_window_size": 5,
    "resolution": [8, 2],
}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def jamma_preprocess(bgr, resize, df, device):
    """BGR ndarray -> (ImageNet-normalised RGB tensor [1,3,S,S], coarse mask [1,S/8,S/8],
    (sx, sy)) where S = max(h_new, w_new) — square pad matches upstream training."""
    h0, w0 = bgr.shape[:2]
    scale = resize / max(h0, w0)
    # Floor to df multiple — matches upstream get_divisible_wh (int(x // df * df)).
    w_new = int(w0 * scale) // df * df
    h_new = int(h0 * scale) // df * df
    rgb = cv2.cvtColor(cv2.resize(bgr, (w_new, h_new)), cv2.COLOR_BGR2RGB)
    # Square pad — matches upstream pad_bottom_right(pad_to=max(h_new, w_new)).
    pad_to = max(h_new, w_new)
    padded = np.zeros((pad_to, pad_to, 3), dtype=np.uint8)
    padded[:h_new, :w_new] = rgb
    t = torch.from_numpy(padded).float().div(255.).permute(2, 0, 1).unsqueeze(0)
    t = ((t - _IMAGENET_MEAN) / _IMAGENET_STD).to(device)
    # Mask marks the valid (non-padded) region at coarse (stride-8) resolution.
    m = torch.zeros((1, pad_to // 8, pad_to // 8), dtype=torch.bool, device=device)
    m[0, :h_new // 8, :w_new // 8] = True
    return t, m, (w0 / w_new, h0 / h_new)


def match_jamma(t0, m0, s0, t1, m1, s1, backbone, matcher, conf_thresh, ransac_thresh):
    data = {"imagec_0": t0, "imagec_1": t1, "mask0": m0, "mask1": m1}
    try:
        with torch.inference_mode():
            backbone(data)
            matcher(data)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"JamMa ran out of CUDA memory. Try reducing --resize (current: "
                f"{t0.shape[-1]}px). Run with a smaller value, e.g. --resize 640."
            ) from exc
        raise

    kp0  = data["mkpts0_f"].cpu().numpy().astype(np.float32).reshape(-1, 2)
    kp1  = data["mkpts1_f"].cpu().numpy().astype(np.float32).reshape(-1, 2)
    conf = data["mconf_f"].cpu().numpy()
    kp0[:, 0] *= s0[0]; kp0[:, 1] *= s0[1]
    kp1[:, 0] *= s1[0]; kp1[:, 1] *= s1[1]

    r = dict(sat_kp=len(kp1), drone_kp=len(kp0), raw=len(kp0), good=0, inliers=0,
             H=None, _kp0=kp0, _kp1=kp1, _conf=conf, _mask=None)
    mask = conf >= conf_thresh
    r["good"], r["_mask"] = int(mask.sum()), mask
    if r["good"] < 4:
        return r
    H, mh = cv2.findHomography(kp0[mask].reshape(-1, 1, 2),
                               kp1[mask].reshape(-1, 1, 2),
                               cv2.USAC_MAGSAC, ransac_thresh,
                               maxIters=5000, confidence=0.9999)
    if H is not None and mh is not None:
        r["inliers"], r["H"] = int(mh.sum()), H
    return r


def _ts(label):
    print(f"  [{time.strftime('%H:%M:%S')}] {label}", flush=True)


def _load_jamma(device, weights_path=None):
    # pytorch_lightning ≥ 2.0 removed the profiler submodule; shim it so
    # JamMa's top-level import doesn't crash before we can pass profiler=None.
    if "pytorch_lightning" not in sys.modules:
        import pytorch_lightning  # noqa: F401 — ensure base package is loaded first
    import pytorch_lightning as _pl
    if not hasattr(_pl, "profiler") or not hasattr(_pl.profiler, "PassThroughProfiler"):
        _stub = types.ModuleType("pytorch_lightning.profiler")
        _stub.PassThroughProfiler = type("PassThroughProfiler", (), {})
        sys.modules["pytorch_lightning.profiler"] = _stub
        _pl.profiler = _stub

    _ts("importing JamMa modules (triggers mamba-ssm CUDA build if first run)...")
    from src.jamma.jamma import JamMa as JamMaMatcher
    from src.jamma.backbone import CovNextV2_nano
    _ts("imports done")

    _ts("building model skeleton...")
    class _Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = CovNextV2_nano()
            self.matcher  = JamMaMatcher(JAMMA_CFG, profiler=None)

    wrapper = _Wrapper()
    _ts("skeleton built")

    import os
    ckpt = weights_path or JAMMA_CKPT_DEFAULT
    if os.path.isfile(ckpt):
        _ts(f"loading weights from {ckpt} ...")
        state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
    else:
        _ts(f"weights not found at {ckpt}, downloading from {JAMMA_URL} ...")
        state_dict = torch.hub.load_state_dict_from_url(
            JAMMA_URL, map_location="cpu", file_name="jamma.ckpt")["state_dict"]
    _ts("weights loaded, applying state_dict...")
    wrapper.load_state_dict(state_dict, strict=True)
    _ts(f"state_dict applied — moving to {device}...")

    backbone = wrapper.backbone.eval().to(device)
    matcher  = wrapper.matcher.eval().to(device)
    _ts("model on device, ready")
    return backbone, matcher


JAMMA_CONF = 0.2


def _worker(args):
    flight_group, gpu_id, resize, df_val, dist, viz_dir, weights_path = args
    device = torch.device(f"cuda:{gpu_id}")
    backbone, matcher = _load_jamma(device, weights_path)

    def match_factory(drone):
        t0, m0, s0 = jamma_preprocess(drone, resize, df_val, device)
        return lambda p: match_jamma(
            t0, m0, s0,
            *jamma_preprocess(p, resize, df_val, device),
            backbone, matcher, JAMMA_CONF, RANSAC_THRESH,
        )

    rows = []
    for flight in flight_group:
        tiles, drone_dir, drone_csv, _ = load_flight(flight)
        df = pd.read_csv(drone_csv)
        rows.extend(collect_pipeline_rows_multitile(
            tiles, df, match_factory, dist, min_inl=MIN_INL,
            drone_dir=drone_dir, flight=flight,
            viz_fn=save_dense_viz if viz_dir else None,
            viz_dir=viz_dir,
            progress=False))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist",      type=float, default=25.0)
    ap.add_argument("--resize",    type=int,   default=832)
    ap.add_argument("--df",        type=int,   default=8)
    ap.add_argument("--weights",   type=str,   default=None,
                    help=f"Path to jamma.ckpt (default: {JAMMA_CKPT_DEFAULT}, "
                         "falls back to GitHub download if missing)")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--flights",   nargs="+", default=["all"],
                    help="Flight IDs to evaluate, e.g. 01 03 05, or 'all' (default)")
    args = ap.parse_args()

    if args.df <= 0 or args.df % 8 != 0:
        raise ValueError(f"--df must be a positive multiple of 8 (got {args.df}); "
                         "JamMa's coarse stride is 8.")
    if args.resize <= 0:
        raise ValueError(f"--resize must be positive (got {args.resize}).")
    if not torch.cuda.is_available():
        raise RuntimeError("JamMa requires CUDA (mamba-ssm has no CPU/MPS kernels).")

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    n_gpus  = max(1, torch.cuda.device_count())
    viz_dir = VIZ_DIR if args.visualize else None
    print(f"  Method: JamMa | Resize: {args.resize} | DF: {args.df} | Conf: {JAMMA_CONF} | "
          f"RANSAC: {RANSAC_THRESH}px | MinInl: {MIN_INL} | "
          f"Dist: {args.dist}m | Flights: {' '.join(flights)} | GPUs: {n_gpus}")

    groups = [g for g in [flights[i::n_gpus] for i in range(n_gpus)] if g]

    log_path = OUT_CSV.replace(".csv", ".log")
    with TeeLogger(log_path):
        if len(groups) == 1:
            device = torch.device("cuda")
            print(f"  Device: {device}")
            print(f"  Loading JamMa ({args.resize}px) ... ", end="", flush=True)
            backbone, matcher = _load_jamma(device, args.weights)
            print("done")

            def match_factory(drone):
                t0, m0, s0 = jamma_preprocess(drone, args.resize, args.df, device)
                return lambda p: match_jamma(
                    t0, m0, s0,
                    *jamma_preprocess(p, args.resize, args.df, device),
                    backbone, matcher, JAMMA_CONF, RANSAC_THRESH,
                )

            all_rows = []
            for flight in flights:
                tiles, drone_dir, drone_csv, _ = load_flight(flight)
                df = pd.read_csv(drone_csv)
                print(f"\n=== Flight {flight}: {len(df)} images ===")
                rows = collect_pipeline_rows_multitile(
                    tiles, df, match_factory, args.dist, min_inl=MIN_INL,
                    drone_dir=drone_dir, flight=flight,
                    viz_fn=save_dense_viz if args.visualize else None,
                    viz_dir=viz_dir)
                all_rows.extend(rows)
                flight_df = pd.DataFrame(rows)
                valid = flight_df[~flight_df["skipped"].fillna(False)]
                if not valid.empty:
                    print_summary(valid, f"flight {flight}", min_inl=MIN_INL)
        else:
            ctx = multiprocessing.get_context("spawn")
            worker_args = [(g, i, args.resize, args.df, args.dist, viz_dir, args.weights)
                           for i, g in enumerate(groups)]
            with ctx.Pool(len(groups)) as pool:
                results = pool.map(_worker, worker_args)
            all_rows = [r for chunk in results for r in chunk]
            for flight in flights:
                flight_rows = [r for r in all_rows if r.get("flight") == flight]
                if flight_rows:
                    fdf = pd.DataFrame(flight_rows)
                    valid = fdf[~fdf["skipped"].fillna(False)]
                    if not valid.empty:
                        print_summary(valid, f"flight {flight}", min_inl=MIN_INL)

        out = pd.DataFrame(all_rows)
        out.to_csv(OUT_CSV, index=False)
        if len(flights) > 1:
            print(f"\n=== Overall ({len(flights)} flights) ===")
            valid_all = out[~out["skipped"].fillna(False)]
            if not valid_all.empty:
                print_summary(valid_all, OUT_CSV, min_inl=MIN_INL)


if __name__ == "__main__":
    main()
