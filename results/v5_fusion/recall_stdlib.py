#!/usr/bin/env python3
"""Replicate analyze/retrieval_recall.py exactly, stdlib-only (no pandas).
R@k per flight = mean over non-skipped rows with valid (>=0) rank of (rank < k),
then averaged across flights (macro-average), per the original groupby().mean()."""
import csv, glob, os, re

RECALL_KS = (1, 5, 10)
MODE_COLS = [("denied", "gt_tile_rank")]  # + gt_rank_r* discovered per file


def truthy(v):
    return str(v).strip().lower() in ("true", "1")


def recall(vals, k):
    valid = [x for x in vals if x is not None and x >= 0]
    if not valid:
        return float("nan")
    return sum(1 for x in valid if x < k) / len(valid)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def process(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if "skipped" in rows[0]:
        rows = [r for r in rows if not truthy(r.get("skipped"))]
    modes = list(MODE_COLS) + [
        (c.replace("gt_rank_", ""), c)
        for c in sorted(rows[0]) if re.fullmatch(r"gt_rank_r\d+", c)
    ]
    flights = sorted({r["flight"] for r in rows})
    out = {}
    for label, col in modes:
        per_flight = {k: [] for k in RECALL_KS}
        for fl in flights:
            sub = [num(r[col]) for r in rows if r["flight"] == fl]
            for k in RECALL_KS:
                rk = recall(sub, k)
                if rk == rk:  # not nan
                    per_flight[k].append(rk)
        out[label] = {k: (sum(v) / len(v) if v else float("nan"))
                      for k, v in per_flight.items()}
    return out, len(rows), len(flights)


csvs = sorted(glob.glob(os.path.join(os.path.dirname(__file__),
              "job_results", "visloc_cliphf_clip_lora_v5_largep14_a*_results.csv")))
print(f"{'alpha':>6} {'N':>5} {'flts':>4} | "
      + "  ".join(f"{m}_R@{k}" for m in ("denied", "r1000", "r5000") for k in RECALL_KS))
for path in csvs:
    alpha = re.search(r"_a([\d.]+)_results", path).group(1)
    out, n, nf = process(path)
    cells = []
    for m in ("denied", "r1000", "r5000"):
        for k in RECALL_KS:
            v = out.get(m, {}).get(k, float("nan"))
            cells.append(f"{v:.3f}" if v == v else "  -  ")
    print(f"{alpha:>6} {n:>5} {nf:>4} | " + "  ".join(f"{c:>9}" for c in cells))
