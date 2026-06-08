"""Per-flight K calibration with the roma_extre teacher.

K (drone-footprint GSD factor) sets the satellite-patch scale
(`patch_span_m = SEARCH_FACTOR * K * altitude_m`). It is correct when
`K * altitude == true camera footprint`.

Two modes:

  --mode hscale  (default, recommended)
    Measure K directly from the geometry. RoMa fits a *similarity* homography
    H (drone→patch); its intrinsic scale s = sqrt(det(H[:2,:2])) is exactly the
    drone↔patch scale ratio, so
        s = true_footprint / (SEARCH_FACTOR * K_used * altitude)
    ⇒  K_true = s * SEARCH_FACTOR * K_used   (per image; median over a flight).
    This is INVARIANT to the K_used the patch was built with, so one pass at the
    current K recovers the true K with a sharp signal — unlike a median-error
    sweep, which is flat because RoMa matches well at almost any scale.

  --mode sweep
    The original grid sweep: for each K, median offset_m / inliers / A@25, pick
    K* = argmin median offset_m (inlier tiebreak). Kept for reference; it could
    not pin K (the error curve is too flat — RoMa is too scale-robust).

Anchors 01/02/03/08 have known K (~1.0/1.0/0.95/1.0); the recovered values for
them validate the method before trusting the rest. Reuses roma_pipeline's
load_model / make_match_factory / add_args and the shared per-image loop
(`collect_pipeline_rows_multitile`), which now records `h_scale` per row.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

torch.manual_seed(0)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # so we can import roma_pipeline
sys.path.insert(0, os.path.dirname(_HERE))      # repo root, for helpers.*

import roma_pipeline as roma  # noqa: E402  (pulls in romatch; container-only)

from helpers.utils import (  # noqa: E402
    load_flight, collect_pipeline_rows_multitile,
    FLIGHTS_AVAILABLE, MIN_INL, K_PER_FLIGHT, K_DEFAULT, SEARCH_FACTOR,
)

ANCHORS = {"01": 1.00, "02": 1.00, "03": 0.95, "08": 1.00}


def _sample_df(df, n):
    """Evenly-spaced subset spanning the whole flight trajectory (not first-N —
    consecutive drone frames overlap one location and would not be representative)."""
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(df) - 1, n).round().astype(int))
    return df.iloc[idx].reset_index(drop=True)


# ── mode: hscale ────────────────────────────────────────────────────────────

def _hscale_flight(flight, tiles, drone_dir, sdf, match_factory, min_inl):
    """Run roma once at the current K_PER_FLIGHT[flight]; recover K from H scale."""
    k_used = K_PER_FLIGHT.get(flight, K_DEFAULT)
    rows = collect_pipeline_rows_multitile(
        tiles, sdf, match_factory, drone_dir=drone_dir, flight=flight,
        min_inl=min_inl, clahe=True, viz_fn=None, viz_dir=None,
        progress=False, k_override=None)          # k_override None → uses k_used
    rdf = pd.DataFrame(rows)
    if "h_scale" not in rdf or "inliers" not in rdf:
        return dict(flight=flight, n=0, k_used=k_used, k_median=None)
    acc = rdf[(rdf["inliers"].fillna(0) >= min_inl) & rdf["h_scale"].notna()]
    if acc.empty:
        return dict(flight=flight, n=0, k_used=k_used, k_median=None)
    k_imp = acc["h_scale"].astype(float) * SEARCH_FACTOR * k_used
    return dict(flight=flight, n=int(len(acc)), k_used=round(k_used, 3),
                k_median=round(float(k_imp.median()), 3),
                k_p25=round(float(k_imp.quantile(0.25)), 3),
                k_p75=round(float(k_imp.quantile(0.75)), 3),
                med_inliers=round(float(acc["inliers"].median()), 0))


def _run_hscale(flights, match_factory, args):
    results = []
    for flight in flights:
        tiles, drone_dir, drone_csv, _ = load_flight(flight)
        sdf = _sample_df(pd.read_csv(drone_csv), args.sample)
        r = _hscale_flight(flight, tiles, drone_dir, sdf, match_factory,
                           args.min_inliers)
        results.append(r)
        print(f"  flight {flight}: n={r['n']:>3} K_used={r['k_used']} "
              f"-> K*={r.get('k_median')} "
              f"[{r.get('k_p25')}, {r.get('k_p75')}] "
              f"med_inl={r.get('med_inliers')}")
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\n  Wrote {args.out}")

    print("\n  ===== K from homography scale  (K* = median over flight) =====")
    print("  flight   K*     [p25,  p75 ]   current   anchor")
    for r in results:
        f = r["flight"]
        anc = f"{ANCHORS[f]:.2f}" if f in ANCHORS else "—"
        print(f"  {f:>4}  {str(r.get('k_median')):>5}   "
              f"[{r.get('k_p25')}, {r.get('k_p75')}]   "
              f"{K_PER_FLIGHT.get(f, 'unset'):>6}   {anc:>5}")
    print("\n  Validation: anchors 01/02/03/08 should land near 1.00/1.00/0.95/1.00.")
    print("  K* is invariant to K_used — tight [p25,p75] ⇒ trustworthy; wide ⇒ "
          "oblique/few matches (e.g. 05/11).")
    print("\n  suggested K_PER_FLIGHT:")
    body = ", ".join(f'"{r["flight"]}": {r["k_median"]}'
                     for r in results if r.get("k_median") is not None)
    print(f"  K_PER_FLIGHT = {{{body}}}")


# ── mode: sweep (reference) ───────────────────────────────────────────────────

def _frange(lo, hi, step):
    out, v = [], lo
    while v <= hi + 1e-9:
        out.append(round(v, 3)); v += step
    return out


def _aggregate(rows, n):
    rdf = pd.DataFrame(rows)
    off = rdf["offset_m"].dropna() if "offset_m" in rdf else pd.Series(dtype=float)
    inl = rdf["inliers"].dropna() if "inliers" in rdf else pd.Series(dtype=float)
    a25 = int(rdf["success_25"].fillna(False).sum()) if "success_25" in rdf else 0
    return dict(n=n, valid=int(len(off)),
                valid_frac=round(len(off) / n, 3) if n else 0.0,
                median_off_m=round(float(off.median()), 2) if len(off) else None,
                mean_inliers=round(float(inl.mean()), 1) if len(inl) else 0.0,
                a25=round(a25 / n, 3) if n else 0.0)


def _pick_best(aggs, min_valid_frac=0.2, err_eps=1.0):
    cand = [a for a in aggs
            if a["median_off_m"] is not None and a["valid_frac"] >= min_valid_frac]
    if not cand:
        return None
    best_err = min(a["median_off_m"] for a in cand)
    near = [a for a in cand if a["median_off_m"] <= best_err + err_eps]
    return max(near, key=lambda a: a["mean_inliers"])


def _run_sweep(flights, match_factory, args):
    all_rows, best_per_flight = [], {}
    coarse = _frange(args.k_min, args.k_max, args.k_step)
    for flight in flights:
        tiles, drone_dir, drone_csv, _ = load_flight(flight)
        sdf = _sample_df(pd.read_csv(drone_csv), args.sample)
        n = len(sdf)
        print(f"=== flight {flight} ({n} frames) ===")

        def eval_k(k):
            rows = collect_pipeline_rows_multitile(
                tiles, sdf, match_factory, drone_dir=drone_dir, flight=flight,
                min_inl=args.min_inliers, clahe=not args.no_clahe,
                viz_fn=None, viz_dir=None, progress=False, k_override=k)
            agg = _aggregate(rows, n); agg.update(flight=flight, k=k)
            all_rows.append(agg)
            return agg

        aggs = [eval_k(k) for k in coarse]
        best = _pick_best(aggs)
        if best is not None:
            done = {a["k"] for a in aggs}
            refine = [k for k in _frange(best["k"] - args.refine_half,
                                         best["k"] + args.refine_half, args.refine_step)
                      if k not in done]
            aggs += [eval_k(k) for k in refine]
            best = _pick_best(aggs)
        best_per_flight[flight] = best
        print(f"  -> best K = {best['k'] if best else None} "
              f"(current={K_PER_FLIGHT.get(flight, 'unset')})")
    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"\n  Wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Per-flight K calibration (roma).")
    roma.add_args(ap)                       # --pretrained / --extre-weights / --num-matches
    ap.set_defaults(pretrained="extre")
    ap.add_argument("--mode", choices=["hscale", "sweep"], default="hscale")
    ap.add_argument("--flights", nargs="+", default=["all"])
    ap.add_argument("--sample", type=int, default=40,
                    help="Drone frames sampled per flight (evenly spaced).")
    ap.add_argument("--min-inliers", type=int, default=MIN_INL)
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--out", default="visloc_kcalib_results.csv")
    # sweep-only grid params
    ap.add_argument("--k-min", type=float, default=0.4)
    ap.add_argument("--k-max", type=float, default=2.4)
    ap.add_argument("--k-step", type=float, default=0.2)
    ap.add_argument("--refine-half", type=float, default=0.15)
    ap.add_argument("--refine-step", type=float, default=0.05)
    args = ap.parse_args()

    flights = FLIGHTS_AVAILABLE if args.flights == ["all"] else args.flights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  K-calibration | mode: {args.mode} | matcher: roma_{args.pretrained} | "
          f"sample: {args.sample}/flight | min-inl: {args.min_inliers}")
    print(f"  flights: {' '.join(flights)}\n")

    matcher = roma.load_model(device, args)
    match_factory = roma.make_match_factory(matcher, device, args)

    if args.mode == "hscale":
        _run_hscale(flights, match_factory, args)
    else:
        _run_sweep(flights, match_factory, args)


if __name__ == "__main__":
    main()
