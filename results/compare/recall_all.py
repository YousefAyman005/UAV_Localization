#!/usr/bin/env python3
"""Consistent denied-mode (gt_tile_rank) Recall@k across rounds, stdlib-only.
R@k = per-flight mean of (valid rank < k), then macro-averaged across flights
(exactly analyze/retrieval_recall.py). Pass one or more dirs of CSVs as argv;
each row is tagged with its source dir so same-named encoders don't collide."""
import csv, glob, os, re, sys

KS = (1, 5, 10)


def truthy(v): return str(v).strip().lower() in ("true", "1")
def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None
def recall(vals, k):
    valid = [x for x in vals if x is not None and x >= 0]
    return (sum(1 for x in valid if x < k) / len(valid)) if valid else float("nan")


def denied_rk(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and "skipped" in rows[0]:
        rows = [r for r in rows if not truthy(r.get("skipped"))]
    flights = sorted({r["flight"] for r in rows})
    out = {}
    for k in KS:
        per = []
        for fl in flights:
            rk = recall([num(r["gt_tile_rank"]) for r in rows if r["flight"] == fl], k)
            if rk == rk:
                per.append(rk)
        out[k] = sum(per) / len(per) if per else float("nan")
    return out, len(rows)


rows = []
for d in sys.argv[1:]:
    tag = os.path.basename(os.path.normpath(d.rstrip("/job_results")))
    for path in glob.glob(os.path.join(d, "*.csv")):
        m = re.search(r"visloc_(.+?)_results\.csv$", os.path.basename(path))
        name = m.group(1)
        am = re.search(r"_a([\d.]+)", name)
        gm = re.search(r"_g([\d.]+)", name)
        enc = name[:am.start()] if am else name
        alpha = am.group(1) if am else "-"
        gal = gm.group(1) if gm else ""
        rk, n = denied_rk(path)
        rows.append((tag, enc, alpha, gal, n, rk))

rows.sort(key=lambda r: (r[0], r[1], float(r[2]) if r[2] != "-" else -1))
print(f"{'src':>10} {'encoder':>34} {'a':>4} {'gal':>4} {'N':>5} | {'R@1':>6} {'R@5':>6} {'R@10':>6}")
for tag, enc, alpha, gal, n, rk in rows:
    print(f"{tag:>10} {enc:>34} {alpha:>4} {gal:>4} {n:>5} | "
          + " ".join(f"{rk[k]:6.3f}" for k in KS))
