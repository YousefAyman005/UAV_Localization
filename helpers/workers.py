"""Parallel orchestration for the feature-matching pipelines.

`run_pipeline` is the entry point each matcher script calls. Two modes:
  - 'gpu_flights' — one worker per CUDA device, flights split across them.
  - 'cpu_chunks'  — per-flight, rows split across forked CPU workers.

Worker functions live at module scope so they pickle cleanly under
multiprocessing's spawn/fork contexts.
"""

import argparse
import multiprocessing
import os
import random

import numpy as np
import pandas as pd

from helpers.results import summarize_rows
from helpers.utils import (
    FLIGHTS_AVAILABLE, MIN_INL, RANSAC_THRESH, TeeLogger,
    collect_pipeline_rows_multitile, load_flight,
)
from helpers.visualization import save_dense_viz, setup_viz_dir


def _sample_df(df, limit):
    """Deterministic random sample of `limit` rows (no-op if limit covers df)."""
    if limit is None or limit >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=limit, random_state=0).sort_index().reset_index(drop=True)


def _collect_flight(flight, match_factory, viz_fn, viz_dir, clahe, limit,
                    min_inl, progress, yaw_cal=True):
    tiles, drone_dir, drone_csv, _ = load_flight(flight)
    df = _sample_df(pd.read_csv(drone_csv), limit)
    if progress:
        print(f"\n=== Flight {flight}: {len(df)} images ===")
    return collect_pipeline_rows_multitile(
        tiles, df, match_factory,
        drone_dir=drone_dir, flight=flight, clahe=clahe, min_inl=min_inl,
        viz_fn=viz_fn if viz_dir else None, viz_dir=viz_dir, progress=progress,
        yaw_cal=yaw_cal)


def _gpu_worker(args):
    flight_group, gpu_id, spec, run = args
    import torch  # imported lazily so CPU-only specs don't need torch
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    device        = torch.device(f"cuda:{gpu_id}")
    model         = spec["load_model"](device, run["args"])
    match_factory = spec["make_match_factory"](model, device, run["args"])
    return [r for f in flight_group
            for r in _collect_flight(
                f, match_factory, spec["viz_fn"], run["viz_dir"],
                run["clahe"], run["limit"], run["min_inl"], progress=False,
                yaw_cal=run["yaw_cal"])]


def _cpu_worker(args):
    chunk_df, tiles, drone_dir, flight, spec, run = args
    import cv2
    random.seed(0); np.random.seed(0); cv2.setRNGSeed(0)
    match_factory = spec["make_match_factory"](None, None, run["args"])
    return collect_pipeline_rows_multitile(
        tiles, chunk_df, match_factory,
        drone_dir=drone_dir, flight=flight, clahe=run["clahe"],
        min_inl=run["min_inl"],
        viz_fn=spec["viz_fn"] if run["viz_dir"] else None,
        viz_dir=run["viz_dir"], progress=False, yaw_cal=run["yaw_cal"])


def _run_gpu_flights(flights, spec, run):
    import torch
    n_gpus = max(1, torch.cuda.device_count())
    groups = [g for g in [flights[i::n_gpus] for i in range(n_gpus)] if g]

    if len(groups) <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")
        model         = spec["load_model"](device, run["args"])
        match_factory = spec["make_match_factory"](model, device, run["args"])
        return [r for f in flights
                for r in _collect_flight(
                    f, match_factory, spec["viz_fn"], run["viz_dir"],
                    run["clahe"], run["limit"], run["min_inl"], progress=True,
                    yaw_cal=run["yaw_cal"])]

    ctx = multiprocessing.get_context("spawn")
    worker_args = [(g, i, spec, run) for i, g in enumerate(groups)]
    with ctx.Pool(len(groups)) as pool:
        chunks = pool.map(_gpu_worker, worker_args)
    return [r for chunk in chunks for r in chunk]


def _run_cpu_chunks(flights, workers, spec, run):
    n_workers = workers or os.cpu_count() or 1
    rows = []
    for flight in flights:
        tiles, drone_dir, drone_csv, _ = load_flight(flight)
        df = _sample_df(pd.read_csv(drone_csv), run["limit"])
        print(f"\n=== Flight {flight}: {len(df)} images ===")
        n_chunks = min(n_workers, max(len(df), 1))
        edges = np.linspace(0, len(df), n_chunks + 1, dtype=int)
        chunks = [df.iloc[edges[i]:edges[i + 1]].reset_index(drop=True)
                  for i in range(n_chunks)]

        if len(chunks) == 1:
            match_factory = spec["make_match_factory"](None, None, run["args"])
            rows.extend(collect_pipeline_rows_multitile(
                tiles, df, match_factory,
                drone_dir=drone_dir, flight=flight, clahe=run["clahe"],
                min_inl=run["min_inl"],
                viz_fn=spec["viz_fn"] if run["viz_dir"] else None,
                viz_dir=run["viz_dir"], yaw_cal=run["yaw_cal"]))
        else:
            chunk_args = [(c, tiles, drone_dir, flight, spec, run) for c in chunks]
            with multiprocessing.Pool(len(chunks)) as pool:
                results = pool.map(_cpu_worker, chunk_args)
            rows.extend(r for chunk in results for r in chunk)
    return rows


def _add_common_args(parser):
    parser.add_argument("--dist",        type=float, default=25.0,
                        help="Display-only top-k distance threshold.")
    parser.add_argument("--visualize",   action="store_true")
    parser.add_argument("--flights",     nargs="+", default=["all"],
                        help="Flight IDs (e.g. 01 03 05) or 'all'.")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Cap drone images per flight (quick tests).")
    parser.add_argument("--min-inliers", type=int, default=MIN_INL,
                        help=f"Minimum RANSAC inliers to accept H (default: {MIN_INL}).")
    parser.add_argument("--no-clahe",    action="store_true",
                        help="Disable CLAHE preprocessing (on by default).")
    parser.add_argument("--no-yaw-cal",  action="store_true",
                        help="Disable the per-leg YAW_OFFSET correction of "
                             "Phi1 (on by default; for ablation).")


def _banner(label, args, flights, parallelism):
    parts = [f"Method: {label}",
             f"RANSAC(sim-4dof): {RANSAC_THRESH}px",
             f"MinInl: {args.min_inliers}",
             f"Dist: {args.dist}m",
             f"CLAHE: {'off' if args.no_clahe else 'on'}",
             f"YawCal: {'off' if args.no_yaw_cal else 'on'}"]
    if parallelism == "cpu_chunks":
        parts.append(f"Workers: {args.workers or 'auto'}")
    parts.append(f"Flights: {' '.join(flights)}")
    return "  " + " | ".join(parts)


def run_pipeline(*, name, load_model, make_match_factory, label=None,
                 add_args=None, viz_fn=save_dense_viz, parallelism="gpu_flights"):
    """Run a feature-matching benchmark.

    name / label: output-file stem and human-readable method name; either a
                  string or a callable(args) -> str.
    parallelism:
      'gpu_flights' — one worker per CUDA device, each handling a flight subset.
                      Single-GPU / CPU-only runs in-process.
      'cpu_chunks'  — for each flight, split rows across forked CPU workers.
                      Adds `--workers` to the CLI.
    """
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    if parallelism == "cpu_chunks":
        parser.add_argument("--workers", type=int, default=None,
                            help="Parallel workers (default: cpu_count).")
    if add_args:
        add_args(parser)
    args = parser.parse_args()

    flights  = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    name_str = name(args) if callable(name) else name
    label    = label(args) if callable(label) else (label or name_str)
    out_csv  = f"visloc_{name_str}_results.csv"
    viz_dir  = f"visloc_{name_str}_visualizations" if args.visualize else None
    setup_viz_dir(viz_dir)

    spec = dict(load_model=load_model, make_match_factory=make_match_factory,
                viz_fn=viz_fn)
    run  = dict(args=args, viz_dir=viz_dir, clahe=not args.no_clahe,
                limit=args.limit, min_inl=args.min_inliers,
                yaw_cal=not args.no_yaw_cal)

    with TeeLogger(out_csv.replace(".csv", ".log")):
        print(_banner(label, args, flights, parallelism))
        if parallelism == "gpu_flights":
            all_rows = _run_gpu_flights(flights, spec, run)
        elif parallelism == "cpu_chunks":
            all_rows = _run_cpu_chunks(flights, args.workers, spec, run)
        else:
            raise ValueError(f"unknown parallelism={parallelism!r}")

        for flight in flights:
            summarize_rows([r for r in all_rows if r.get("flight") == flight],
                           f"flight {flight}", run["min_inl"])
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        if len(flights) > 1:
            print(f"\n=== Overall ({len(flights)} flights) ===")
            summarize_rows(all_rows, out_csv, run["min_inl"])
