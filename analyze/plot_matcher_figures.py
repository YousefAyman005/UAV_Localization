"""Thesis figures for the geometric-matcher benchmark (Results ch.).

Consumes the per-image visloc_*_results.csv files (staged from the SLURM
tars, see --from-tars) and renders publication figures (PDF+PNG) plus a
LaTeX summary table into --out. Pure pandas/matplotlib — runs anywhere,
no container needed. Only --update-meta (one-off) touches the dataset.

Usage:
    # one-off staging after a re-run sweep:
    python analyze/plot_matcher_figures.py --from-tars tar/zz_378*.tar
    # figures:
    python analyze/plot_matcher_figures.py --fig all
    python analyze/plot_matcher_figures.py --fig curves perflight
    # spatial extras:
    python analyze/plot_matcher_figures.py --fig spatial --matcher roma_extre
    # one-off, wherever dataset access is allowed:
    python analyze/plot_matcher_figures.py --update-meta --dataset-root <UAV_VisLoc_dataset>
"""
import argparse
import os
import re
import shutil
import sys
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Duplicated from helpers.utils (importing helpers pulls cv2/torch, which
# this script must not depend on). Keep in sync.
MIN_INL = 7                          # helpers/utils.py:31
FLIGHTS = ["01", "02", "03", "04", "05", "06", "08", "10", "11"]
TEST_FRAC = 0.25                     # split_flight_rows default

OVERVIEW_DIR = "thesis/figures/sat_overviews"

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4,
})
SINGLE_W, DOUBLE_W = 3.4, 7.0        # inches (thesis column widths)
LINESTYLES = ["-", "--", ":", "-."]  # variant differentiation within a family

# Okabe–Ito colorblind-safe palette, one color per method family.
FAMILY_COLORS = {
    "baseline":  "#888888",
    "lightglue": "#E69F00",
    "loftr":     "#CC79A7",
    "eloftr":    "#0072B2",
    "xoftr":     "#009E73",
    "roma":      "#D55E00",
    "matcha":    "#56B4E9",
}

# Single source of truth: csv stem -> metadata. `best` marks the variant
# shown in cross-method figures (pinned from empirical A@25, see plan Task 3).
# `band="test"` = fine-tuned on the spatial train band, so evaluate on the
# held-out test band only (and label accordingly).
MATCHERS = {
    "baseline_sift":      dict(family="baseline",  variant="SIFT",    label="SIFT (baseline)",     best=True),
    "baseline_orb":       dict(family="baseline",  variant="ORB",     label="ORB (baseline)"),
    "baseline_brisk":     dict(family="baseline",  variant="BRISK",   label="BRISK (baseline)"),
    "lightglue_sift":     dict(family="lightglue", variant="SIFT",    label="LightGlue–SIFT"),
    "lightglue_disk":     dict(family="lightglue", variant="DISK",    label="LightGlue–DISK"),
    "lightglue_dedodeb":  dict(family="lightglue", variant="DeDoDe",  label="LightGlue–DeDoDe", best=True),
    "loftr_outdoor":      dict(family="loftr",     variant="outdoor", label="LoFTR",               best=True),
    "eloftr":             dict(family="eloftr",    variant="stock",   label="ELoFTR",              best=True),
    "eloftr_lora":        dict(family="eloftr",    variant="LoRA",    label="ELoFTR–LoRA (test band)",  band="test"),
    "eloftr_calib_track": dict(family="eloftr",    variant="calib",   label="ELoFTR+calib (test band)", band="test"),
    "eloftr_calib_en":    dict(family="eloftr",    variant="calibEN", label="ELoFTR+calib-EN (test band)", band="test"),
    "xoftr":              dict(family="xoftr",     variant="-",       label="XoFTR",               best=True),
    "roma_outdoor":       dict(family="roma",      variant="outdoor", label="RoMa"),
    "roma_extre":         dict(family="roma",      variant="AEM",     label="RoMa–AEM",            best=True),
    "matcha":             dict(family="matcha",    variant="-",       label="MATCHA",              best=True),
}
for _m in MATCHERS.values():
    _m["color"] = FAMILY_COLORS[_m["family"]]
    _m.setdefault("best", False)
    _m.setdefault("band", "all")


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  [fig] {out_dir}/{name}.pdf|png")


def stage_from_tars(tar_paths, results_dir):
    """Extract each tar's visloc_*_results.csv as visloc_<variant>_results.csv.

    The variant starts as the tar-name suffix (zz_<job>_<variant>.tar) and is
    upgraded to the inner CSV stem when that is a strict extension of it —
    covers both naming conventions in tar/: lightglue tars encode the variant
    only in the tar name (generic inner CSV), the eloftr tars share one tar
    name and encode the variant only in the inner CSV. Same variant in
    several tars: the highest job id wins. Never clobbers a destination
    newer than its tar."""
    os.makedirs(results_dir, exist_ok=True)
    chosen = {}                            # variant -> (job, tar_path, member)
    for p in tar_paths:
        m = re.match(r"zz_(\d+)_(.+)\.tar$", os.path.basename(p))
        if not m:
            print(f"  [skip] unparseable tar name: {p}"); continue
        job, variant = int(m.group(1)), m.group(2)
        with tarfile.open(p) as tf:
            members = [n for n in tf.getnames()
                       if re.search(r"visloc_.+_results\.csv$", n)]
        if len(members) != 1:
            print(f"  [skip] {os.path.basename(p)}: found {len(members)} result CSVs")
            continue
        inner = re.search(r"visloc_(.+)_results\.csv$", members[0]).group(1)
        if inner != variant and inner.startswith(variant):
            variant = inner                # inner CSV stem is more specific
        if variant in chosen and chosen[variant][0] >= job:
            print(f"  [skip] {os.path.basename(p)} ({variant}): "
                  f"job {chosen[variant][0]} is newer")
            continue
        chosen[variant] = (job, p, members[0])
    for variant, (job, p, member) in sorted(chosen.items()):
        dest = os.path.join(results_dir, f"visloc_{variant}_results.csv")
        if os.path.exists(dest) and os.path.getmtime(dest) > os.path.getmtime(p):
            print(f"  [keep] {dest}: newer than {os.path.basename(p)}"); continue
        with tarfile.open(p) as tf, open(dest, "wb") as out:
            shutil.copyfileobj(tf.extractfile(member), out)
        print(f"  [ok]   job {job}: {variant:24s} -> {dest}")


# ── data layer / metrics ────────────────────────────────────────────────────

def test_band_mask(df, test_frac=TEST_FRAC):
    """Held-out spatial band per flight. Mirrors helpers.utils.split_flight_rows:
    sort by the wider-spread geographic axis, top round(n*test_frac) rows are
    the test band (which is buffer/val-invariant by construction)."""
    mask = pd.Series(False, index=df.index)
    for _, g in df.groupby("flight"):
        lat = g["lat"].to_numpy(dtype=float); lon = g["lon"].to_numpy(dtype=float)
        axis = "lat" if (lat.max() - lat.min()) >= (lon.max() - lon.min()) else "lon"
        order = np.argsort(g[axis].to_numpy(dtype=float))
        n_test = int(round(len(g) * test_frac))
        mask.loc[g.index[order[len(g) - n_test:]]] = True
    return mask


def gated_err(df):
    """Per-image localization error with gate failures at +inf.
    offset_m is NaN exactly when inliers < MIN_INL (helpers/utils.py:481),
    so NaN -> inf makes every accuracy computation share one denominator."""
    e = pd.to_numeric(df["offset_m"], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isnan(e), np.inf, e)


def acc_curve(df, xs):
    """A@X (%) for an array of thresholds xs (meters)."""
    e = np.sort(gated_err(df))
    return np.searchsorted(e, np.asarray(xs, dtype=float), side="right") / len(e) * 100.0


def _crosscheck(ev, label, tol_pp=0.5):
    """Recompute A@t and compare to the stored gated success_t columns.
    Disagreement means a stale/pre-yawfix CSV or changed gate semantics."""
    e = gated_err(ev)
    for t in (5, 10, 15, 20, 25, 30):
        col = f"success_{t}"
        if col not in ev.columns:
            return
        stored = (ev[col] == True).mean() * 100  # noqa: E712 (col may hold NaN)
        recomp = (e <= t).mean() * 100
        if abs(stored - recomp) > tol_pp:
            print(f"  [WARN] {label}: A@{t} recomputed {recomp:.1f}% != stored "
                  f"{stored:.1f}% — stale or pre-yawfix CSV?")


def load_results(results_dir):
    """{stem: {meta, df, df_full, n_total, n_skipped, gt_rate}} for every
    registry entry whose CSV exists. df = evaluation rows (non-skipped,
    gt_in_patch, band filter applied); df_full = same without band filter."""
    out = {}
    for stem, meta in MATCHERS.items():
        path = os.path.join(results_dir, f"visloc_{stem}_results.csv")
        if not os.path.exists(path):
            print(f"  [warn] {meta['label']}: no {path} — omitted")
            continue
        df = pd.read_csv(path)
        df["flight"] = df["flight"].astype(str).str.zfill(2)
        skipped = df["skipped"] == True   # noqa: E712 (NaN-tolerant)
        valid = df[~skipped]
        gt_ok = valid["gt_in_patch"] == True  # noqa: E712
        ev_full = valid[gt_ok].copy()
        ev = ev_full[test_band_mask(ev_full)] if meta["band"] == "test" else ev_full
        _crosscheck(ev, meta["label"])
        out[stem] = dict(meta=meta, df=ev, df_full=ev_full,
                         n_total=len(df), n_skipped=int(skipped.sum()),
                         gt_rate=100.0 * gt_ok.mean() if len(valid) else float("nan"))
    return out


def print_acc(data):
    hdr = (f"{'matcher':22s} {'N':>5} {'A@5':>6} {'A@10':>6} {'A@25':>6} "
           f"{'A@30':>6} {'med_m':>7} {'gt%':>6} {'skip':>5}")
    print(hdr); print("-" * len(hdr))
    for stem, d in data.items():
        e = gated_err(d["df"]); ok = e[np.isfinite(e)]
        med = float(np.median(ok)) if len(ok) else float("nan")
        print(f"{stem:22s} {len(d['df']):5d} "
              f"{(e <= 5).mean()*100:6.1f} {(e <= 10).mean()*100:6.1f} "
              f"{(e <= 25).mean()*100:6.1f} {(e <= 30).mean()*100:6.1f} "
              f"{med:7.1f} {d['gt_rate']:6.1f} {d['n_skipped']:5d}")


def _best(data):
    return [(s, d) for s, d in data.items() if d["meta"]["best"]]


def _acc_at(df, t=25.0):
    return (gated_err(df) <= t).mean() * 100.0


# ── figures ─────────────────────────────────────────────────────────────────

def fig_curves(data, out):
    xs = np.linspace(0.5, 30.0, 60)
    # cross-method: one line per family-best
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 2.8))
    for stem, d in data.items():
        if not d["meta"]["best"]:
            continue
        ax.plot(xs, acc_curve(d["df"], xs), color=d["meta"]["color"],
                label=d["meta"]["label"])
    ax.set_xlabel("error threshold $X$ (m)"); ax.set_ylabel("A@$X$ m (%)")
    ax.set_xlim(0, 30); ax.set_ylim(0, 100)
    ax.legend(ncols=2, loc="upper left", framealpha=0.9)
    _save(fig, out, "curves_main")

    # per-family: variants against each other
    fams = {}
    for stem, d in data.items():
        fams.setdefault(d["meta"]["family"], []).append(d)
    for fam, members in fams.items():
        if len(members) < 2:
            continue
        # if any member is test-band-only, put ALL members on the test band
        # so the family comparison is apples-to-apples (spec: Band handling)
        test_band = any(d["meta"]["band"] == "test" for d in members)
        fig, ax = plt.subplots(figsize=(SINGLE_W, 2.6))
        for i, d in enumerate(members):
            df = (d["df_full"][test_band_mask(d["df_full"])]
                  if test_band else d["df"])
            label = d["meta"]["label"].replace(" (test band)", "")
            ax.plot(xs, acc_curve(df, xs), color=d["meta"]["color"],
                    linestyle=LINESTYLES[i % len(LINESTYLES)], label=label)
        ax.set_xlabel("error threshold $X$ (m)")
        ax.set_ylabel("A@$X$ m (%)" + (" — test band" if test_band else ""))
        ax.set_xlim(0, 30); ax.set_ylim(0, 100)
        ax.legend(loc="upper left", framealpha=0.9)
        _save(fig, out, f"curves_family_{fam}")


def fig_perflight(data, out):
    best = _best(data)
    M = np.full((len(best), len(FLIGHTS)), np.nan)
    for i, (s, d) in enumerate(best):
        for j, fl in enumerate(FLIGHTS):
            g = d["df"][d["df"]["flight"] == fl]
            if len(g):
                M[i, j] = _acc_at(g)
    labels = [d["meta"]["label"] for _, d in best]

    # heatmap
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 0.42 * len(best) + 1.1))
    ax.grid(False)
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if M[i, j] < 55 else "black")
    ax.set_xticks(range(len(FLIGHTS)), FLIGHTS)
    ax.set_yticks(range(len(best)), labels)
    ax.set_xlabel("flight")
    fig.colorbar(im, ax=ax, label="A@25 m (%)", fraction=0.025, pad=0.02)
    _save(fig, out, "perflight_heatmap")

    # grouped bars (alternative view of the same matrix)
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 2.6))
    x = np.arange(len(FLIGHTS)); w = 0.85 / len(best)
    for i, (s, d) in enumerate(best):
        ax.bar(x + (i - (len(best) - 1) / 2) * w, M[i], width=w,
               color=d["meta"]["color"], label=d["meta"]["label"])
    ax.set_xticks(x, FLIGHTS); ax.set_xlabel("flight"); ax.set_ylabel("A@25 m (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), framealpha=0.9)
    _save(fig, out, "perflight_bars")


def fig_dist(data, out):
    best = _best(data)
    # ECDF of gated error, log-x to 100 m (gate failures at inf => curves
    # plateau below 100%, which is the honest visual)
    xs = np.logspace(0, 2, 150)
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.7))
    for s, d in best:
        ax.plot(xs, acc_curve(d["df"], xs), color=d["meta"]["color"],
                label=d["meta"]["label"])
    ax.set_xscale("log"); ax.set_xlim(1, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("localization error (m)"); ax.set_ylabel("images within (%)")
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, out, "dist_ecdf")

    # two-panel box: (a) per family-best, pooled flights; (b) per flight for
    # the strongest matcher — the per-flight bias floors show as medians.
    ref_stem, ref = max(best, key=lambda sd: _acc_at(sd[1]["df"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    vals, labs, cols = [], [], []
    for s, d in best:
        e = gated_err(d["df"]); ok = e[np.isfinite(e)]
        if len(ok):
            vals.append(ok); labs.append(d["meta"]["label"]); cols.append(d["meta"]["color"])
    bp = ax1.boxplot(vals, showfliers=False, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    ax1.set_xticks(range(1, len(labs) + 1), labs, rotation=40, ha="right", fontsize=7)
    ax1.set_ylabel("gated error (m)"); ax1.set_title("(a) accepted images, all flights")

    fvals = []
    for fl in FLIGHTS:
        e = gated_err(ref["df"][ref["df"]["flight"] == fl])
        fvals.append(e[np.isfinite(e)])
    bp2 = ax2.boxplot(fvals, showfliers=False, patch_artist=True, widths=0.6)
    for patch in bp2["boxes"]:
        patch.set_facecolor(ref["meta"]["color"]); patch.set_alpha(0.6)
    for med in bp2["medians"]:
        med.set_color("black")
    ax2.set_xticks(range(1, len(FLIGHTS) + 1), FLIGHTS)
    ax2.set_xlabel("flight")
    ax2.set_title(f"(b) {ref['meta']['label']} per flight")
    _save(fig, out, "dist_box")


def make_table(data, out):
    os.makedirs(out, exist_ok=True)
    rows = []
    for stem, d in data.items():
        e = gated_err(d["df"]); ok = e[np.isfinite(e)]
        tm = pd.to_numeric(d["df"].get("t_match_ms"), errors="coerce").dropna()
        rows.append(dict(
            family=d["meta"]["family"], matcher=d["meta"]["label"],
            best=d["meta"]["best"], N=len(d["df"]),
            **{f"A{t}": round((e <= t).mean() * 100, 1) for t in (5, 10, 25, 30)},
            med_err_m=round(float(np.median(ok)), 1) if len(ok) else None,
            gt_in_patch_pct=round(d["gt_rate"], 1),
            skipped=d["n_skipped"],
            med_t_ms=round(float(tm.median()), 0) if len(tm) else None))
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out, "summary_matchers.csv")
    df.to_csv(csv_path, index=False)

    def fmt(r):
        cells = [r["matcher"], str(r["N"]),
                 *(f"{r[f'A{t}']:.1f}" for t in (5, 10, 25, 30)),
                 "--" if r["med_err_m"] is None else f"{r['med_err_m']:.1f}",
                 f"{r['gt_in_patch_pct']:.1f}", str(r["skipped"]),
                 "--" if r["med_t_ms"] is None else f"{r['med_t_ms']:.0f}"]
        if r["best"]:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        return " & ".join(cells) + r" \\"

    lines = [r"% auto-generated by analyze/plot_matcher_figures.py -- do not edit",
             r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r"matcher & $N$ & A@5 & A@10 & A@25 & A@30 & med.\,err & GT-in-patch & skip & $t_\mathrm{match}$ \\",
             r" & & (\%) & (\%) & (\%) & (\%) & (m) & (\%) & & (ms) \\",
             r"\midrule"]
    for fam in dict.fromkeys(df["family"]):           # keep registry order
        for _, r in df[df["family"] == fam].iterrows():
            lines.append(fmt(r))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    tex_path = os.path.join(out, "summary_matchers.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [tab] {csv_path}\n  [tab] {tex_path}")


def fig_acc2030(data, out):
    """Grouped bars of A@20/25/30 for every loaded matcher variant."""
    items = list(data.items())
    x = np.arange(len(items))
    ts = (20, 25, 30)
    shades = [plt.cm.viridis(v) for v in (0.25, 0.55, 0.85)]
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 3.0))
    w = 0.8 / len(ts)
    for j, t in enumerate(ts):
        vals = [_acc_at(d["df"], t) for _, d in items]
        ax.bar(x + (j - 1) * w, vals, width=w, color=shades[j], label=f"A@{t} m")
    ax.set_xticks(x, [d["meta"]["label"] for _, d in items],
                  rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, out, "acc2030_all")


# ── retrieval (CLIP-family) figures ─────────────────────────────────────────
# Per-image retrieval CSVs (clip_pipeline schema): gt_tile_rank is the
# 0-based full-gallery rank of the GT tile (R@k = rank < k); gt_rank_r1000/
# r5000 are ranks within a GPS-prior-restricted gallery. All retrieval
# figures evaluate on the spatial TEST band so zero-shot and LoRA-fine-tuned
# models (trained on the train band) share one protocol.

RETRIEVAL_MODELS = [
    ("cliphf_base_largep14_a1.0",                 "CLIP ViT-L/14",          "#888888"),
    ("cliphf_base_siglip2-basep16-384_a1.0",      "SigLIP2 B/16-384",       "#E69F00"),
    ("camp",                                      "CAMP (U-1652)",          "#56B4E9"),
    ("sample4geo",                                "Sample4Geo (U-1652)",    "#009E73"),
    ("cliphf_clip_lora_v2_imgonly_largep14_a1.0", "CLIP-L/14+LoRA img-only", "#0072B2"),
    ("cliphf_clip_lora_v5_largep14_a1.0",         "CLIP-L/14+LoRA (v5)",     "#D55E00"),
    ("cliphf_siglip2_lora_siglip2-basep16-384_a1.0", "SigLIP2+LoRA",        "#CC79A7"),
]
RETRIEVAL_SWEEPS = [
    ("cliphf_base_largep14_a{a}",                 "CLIP-L/14 base",        "#888888", "-"),
    ("cliphf_base_siglip2-basep16-384_a{a}",      "SigLIP2 base",          "#E69F00", "-"),
    ("cliphf_clip_lora_v5_largep14_a{a}",         "CLIP-L/14+LoRA (v5)",   "#D55E00", "-."),
    ("cliphf_siglip2_lora_siglip2-basep16-384_a{a}", "SigLIP2 LoRA",       "#CC79A7", "--"),
]
ALPHAS = ["0.0", "0.5", "0.7", "0.8", "1.0"]


def _load_retrieval(results_dir, stem, quiet=False):
    path = os.path.join(results_dir, f"visloc_{stem}_results.csv")
    if not os.path.exists(path):
        if not quiet:
            print(f"  [warn] retrieval: no {path} — omitted")
        return None
    df = pd.read_csv(path)
    df["flight"] = df["flight"].astype(str).str.zfill(2)
    df = df[df["skipped"] != True]  # noqa: E712
    return df[test_band_mask(df)]


def _recall(df, k=1, col="gt_tile_rank"):
    r = pd.to_numeric(df[col], errors="coerce")
    return (r < k).mean() * 100.0   # NaN (GT outside restricted gallery) = miss


def fig_retrieval(results_dir, out):
    loaded = [(label, color, df) for stem, label, color in RETRIEVAL_MODELS
              if (df := _load_retrieval(results_dir, stem)) is not None]
    if not loaded:
        print("  [warn] retrieval: no CSVs found — all retrieval figures skipped")
        return

    # (1) R@1/5/10 per model, full gallery (GPS-denied)
    x = np.arange(len(loaded))
    shades = [plt.cm.viridis(v) for v in (0.25, 0.55, 0.85)]
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 2.9))
    w = 0.8 / 3
    for j, k in enumerate((1, 5, 10)):
        ax.bar(x + (j - 1) * w, [_recall(df, k) for _, _, df in loaded],
               width=w, color=shades[j], label=f"R@{k}")
    ax.set_xticks(x, [label for label, _, _ in loaded],
                  rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("recall (%) — test band, GPS-denied"); ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, out, "retrieval_recall_bars")

    # (2) text-fusion alpha sweep: R@1 vs image weight
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.7))
    for tmpl, label, color, ls in RETRIEVAL_SWEEPS:
        pts = [(float(a), _recall(df, 1)) for a in ALPHAS
               if (df := _load_retrieval(results_dir, tmpl.format(a=a),
                                         quiet=True)) is not None]
        if len(pts) >= 2:
            ax.plot(*zip(*pts), color=color, linestyle=ls, marker="o",
                    markersize=3, label=label)
    ax.set_xlabel(r"fusion weight $\alpha$ (image share)")
    ax.set_ylabel("R@1 (%) — test band")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(framealpha=0.9)
    _save(fig, out, "retrieval_alpha_sweep")

    # (3) effect of a GPS prior restricting the gallery
    modes = [("gt_tile_rank", "GPS-denied"),
             ("gt_rank_r5000", "prior $\\leq$ 5 km"),
             ("gt_rank_r1000", "prior $\\leq$ 1 km")]
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 2.9))
    for j, (col, mlabel) in enumerate(modes):
        vals = [_recall(df, 1, col) if col in df.columns else np.nan
                for _, _, df in loaded]
        ax.bar(x + (j - 1) * w, vals, width=w, color=shades[j], label=mlabel)
    ax.set_xticks(x, [label for label, _, _ in loaded],
                  rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("R@1 (%) — test band"); ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, out, "retrieval_gps_prior")


def make_retrieval_table(results_dir, out):
    """summary_retrieval.{tex,csv}: R@1/5/10 (full gallery, GPS-denied) plus
    R@1 under the 5 km / 1 km GPS-prior galleries, per model, on the test
    band. Mirrors make_table; per-column best R@1/5/10 cells are bolded."""
    os.makedirs(out, exist_ok=True)
    rows = []
    for stem, label, _ in RETRIEVAL_MODELS:
        df = _load_retrieval(results_dir, stem)
        if df is None:
            continue
        rows.append(dict(
            model=label, N=len(df),
            R1=round(_recall(df, 1), 1), R5=round(_recall(df, 5), 1),
            R10=round(_recall(df, 10), 1),
            R1_5km=round(_recall(df, 1, "gt_rank_r5000"), 1)
                   if "gt_rank_r5000" in df.columns else None,
            R1_1km=round(_recall(df, 1, "gt_rank_r1000"), 1)
                   if "gt_rank_r1000" in df.columns else None))
    if not rows:
        print("  [warn] retrieval table: no CSVs found"); return
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out, "summary_retrieval.csv")
    df.to_csv(csv_path, index=False)

    best = {c: df[c].max() for c in ("R1", "R5", "R10")}

    def cell(v, col=None):
        if v is None:
            return "--"
        s = f"{v:.1f}"
        return rf"\textbf{{{s}}}" if col and v == best[col] else s

    lines = [r"% auto-generated by analyze/plot_matcher_figures.py -- do not edit",
             r"% R@k on the spatial test band, pooled over flights",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"model & $N$ & R@1 & R@5 & R@10 & "
             r"R@1$_{\leq5\,\mathrm{km}}$ & R@1$_{\leq1\,\mathrm{km}}$ \\",
             r" & & (\%) & (\%) & (\%) & (\%) & (\%) \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join([
            r["model"], str(r["N"]),
            cell(r["R1"], "R1"), cell(r["R5"], "R5"), cell(r["R10"], "R10"),
            cell(r["R1_5km"]), cell(r["R1_1km"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(out, "summary_retrieval.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [tab] {csv_path}\n  [tab] {tex_path}")


# ── spatial figure + its one-off meta update ───────────────────────────────

def update_meta(dataset_root):
    """One-off: append satellite corner coordinates to sat_overviews_meta.csv
    so the spatial figure can map GPS -> overview px without the dataset.
    The dataset CSV is literally named with a space (helpers/utils.py:173)."""
    meta_path = os.path.join(OVERVIEW_DIR, "sat_overviews_meta.csv")
    meta = pd.read_csv(meta_path, dtype={"flight": str})
    sat = pd.read_csv(os.path.join(dataset_root, "satellite_ coordinates_range.csv"))
    for col in ("LT_lat", "LT_lon", "RB_lat", "RB_lon"):
        meta[col] = np.nan
    for i, r in meta.iterrows():
        rows = sat[sat["mapname"] == f"satellite{r['flight']}.tif"]
        if rows.empty:
            print(f"  [warn] no corner entry for flight {r['flight']}"); continue
        s = rows.iloc[0]
        meta.loc[i, ["LT_lat", "LT_lon", "RB_lat", "RB_lon"]] = [
            s["LT_lat_map"], s["LT_lon_map"], s["RB_lat_map"], s["RB_lon_map"]]
    meta.to_csv(meta_path, index=False)
    print(f"  [ok] corner coords written to {meta_path}")


def fig_spatial(data, out, matcher, flights, vec_scale):
    from matplotlib.collections import LineCollection
    meta_path = os.path.join(OVERVIEW_DIR, "sat_overviews_meta.csv")
    meta = pd.read_csv(meta_path, dtype={"flight": str})
    if "LT_lat" not in meta.columns or meta["LT_lat"].isna().any():
        sys.exit(f"{meta_path} lacks corner coordinates — run once:\n"
                 f"  python analyze/plot_matcher_figures.py --update-meta "
                 f"--dataset-root <UAV_VisLoc_dataset>")
    if matcher not in data:
        sys.exit(f"--matcher {matcher}: no loaded results "
                 f"(have: {', '.join(sorted(data))})")
    d = data[matcher]
    meta = meta.set_index("flight")
    for fl in flights:
        g = d["df"][d["df"]["flight"] == fl]
        if g.empty or fl not in meta.index:
            print(f"  [warn] spatial: no rows/meta for flight {fl}"); continue
        m = meta.loc[fl]
        pplon = m["width_px"] / (m["RB_lon"] - m["LT_lon"])
        pplat = m["height_px"] / (m["LT_lat"] - m["RB_lat"])
        s = m["overview_scale"]

        def to_px(lat, lon):
            return ((lon - m["LT_lon"]) * pplon * s,
                    (m["LT_lat"] - lat) * pplat * s)

        gx, gy = to_px(g["lat"].to_numpy(), g["lon"].to_numpy())
        acc = np.isfinite(gated_err(g))
        px, py = to_px(g["pred_lat"].to_numpy(), g["pred_lon"].to_numpy())
        err = pd.to_numeric(g["offset_m"], errors="coerce").to_numpy()

        img = plt.imread(os.path.join(OVERVIEW_DIR, f"satellite{fl}_overview.png"))
        # crop view to the GT extent + margin
        mx = 0.15 * max(gx.max() - gx.min(), gy.max() - gy.min(), 1.0)
        x0, x1 = max(gx.min() - mx, 0), min(gx.max() + mx, img.shape[1])
        y0, y1 = max(gy.min() - mx, 0), min(gy.max() + mx, img.shape[0])

        fig_h = SINGLE_W * (y1 - y0) / max(x1 - x0, 1.0)
        fig, ax = plt.subplots(figsize=(SINGLE_W, min(fig_h, 8.0)))
        ax.grid(False)
        ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)         # y down = image coords
        ax.set_xticks([]); ax.set_yticks([])

        # exaggerated error vectors GT -> prediction, colored by error (m).
        # Errors beyond the color scale (gross failures, e.g. over water)
        # would paint map-length lines at ×vec_scale — mark them as × instead.
        vmax = 50.0
        ok = acc & (err <= vmax)
        gross = acc & (err > vmax)
        ex = gx[ok] + (px[ok] - gx[ok]) * vec_scale
        ey = gy[ok] + (py[ok] - gy[ok]) * vec_scale
        segs = np.stack([np.column_stack([gx[ok], gy[ok]]),
                         np.column_stack([ex, ey])], axis=1)
        lc = LineCollection(segs, cmap="viridis", linewidths=0.9,
                            norm=plt.Normalize(0, vmax))
        lc.set_array(err[ok])
        ax.add_collection(lc)
        ax.scatter(gx[ok], gy[ok], s=2, c="white", linewidths=0)
        ax.scatter(gx[gross], gy[gross], s=10, c="red", marker="x",
                   linewidths=0.6, label=f"error > {vmax:g} m")
        ax.scatter(gx[~acc], gy[~acc], s=8, facecolors="none",
                   edgecolors="red", linewidths=0.6, label="gate failed")
        fig.colorbar(lc, ax=ax, label="error (m)", fraction=0.04, pad=0.02)
        n_fail = int((~acc).sum())
        ax.set_title(f"flight {fl} — {d['meta']['label']} "
                     f"(vectors ×{vec_scale:g}, {int(gross.sum())} > {vmax:g} m, "
                     f"{n_fail} gate-failed)")
        if n_fail or gross.any():
            ax.legend(loc="lower right", framealpha=0.9)
        _save(fig, out, f"spatial_errors_{fl}_{matcher}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig", nargs="+", default=[],
                    choices=["all", "curves", "perflight", "dist", "spatial",
                             "table", "acc2030", "retrieval"])
    ap.add_argument("--results-dir", default="tar/figures_input")
    ap.add_argument("--out", default="thesis/figures/matchers")
    ap.add_argument("--from-tars", nargs="+", metavar="TAR",
                    help="stage visloc CSVs out of SLURM result tars, then exit")
    ap.add_argument("--update-meta", action="store_true",
                    help="bake satellite corner coords into sat_overviews_meta.csv, then exit")
    ap.add_argument("--dataset-root", help="UAV_VisLoc dataset root (only for --update-meta)")
    ap.add_argument("--print-acc", action="store_true",
                    help="print per-matcher accuracy summary, no figures")
    ap.add_argument("--matcher", default="roma_extre", help="matcher for --fig spatial")
    ap.add_argument("--flights", nargs="+", default=["all"], help="flights for --fig spatial")
    ap.add_argument("--vec-scale", type=float, default=5.0,
                    help="error-vector exaggeration in the spatial figure")
    args = ap.parse_args()

    if args.from_tars:
        stage_from_tars(args.from_tars, args.results_dir); return
    if args.update_meta:
        if not args.dataset_root:
            ap.error("--update-meta requires --dataset-root")
        update_meta(args.dataset_root); return

    data = load_results(args.results_dir)
    if not data:
        sys.exit(f"No registered visloc_*_results.csv found in {args.results_dir} "
                 f"— run --from-tars first.")
    if args.print_acc:
        print_acc(data); return

    if not args.fig:
        ap.error("nothing to do: pass --fig, --print-acc, --from-tars or --update-meta")
    figs = set(args.fig)
    if figs & {"all", "curves"}:    fig_curves(data, args.out)
    if figs & {"all", "perflight"}: fig_perflight(data, args.out)
    if figs & {"all", "dist"}:      fig_dist(data, args.out)
    if figs & {"all", "acc2030"}:   fig_acc2030(data, args.out)
    if figs & {"all", "retrieval"}:
        fig_retrieval(args.results_dir, args.out)
        make_retrieval_table(args.results_dir, args.out)
    if figs & {"all", "table"}:     make_table(data, args.out)
    if figs & {"all", "spatial"}:
        flights = FLIGHTS if args.flights == ["all"] else args.flights
        fig_spatial(data, args.out, args.matcher, flights, args.vec_scale)


if __name__ == "__main__":
    main()
