# UAV → Satellite Visual Localization — Repo Summary (portable handoff)

> A self-contained brief of everything in this repo, written so another Claude session
> (or person) in a different context can pick it up. Captures *what it is, how it's built,
> what's been run, and what the results are*.

---

## 1. What the project is

A **bachelor-thesis benchmark for UAV (drone) → satellite visual localization**.

**Task:** given a drone image plus a *noisy GPS prior*, find where the drone actually is on a
georeferenced satellite map, and report the localization error in **meters**.

It compares **three families of methods** on the public **UAV-VisLoc** dataset:

1. **Feature-matching geometric localization** — match drone↔satellite, estimate a homography
   `H`, project the patch centre through `H`, convert to GPS, measure error.
   (SIFT/ORB baseline, LightGlue, LoFTR, RoMa, MATCHA.)
2. **Embedding retrieval** — tile the satellite into a gallery, embed tiles + drone image,
   rank tiles by cosine similarity (no geometry).
   (CLIP, GeoCLIP, SatCLIP, MobileCLIP, DINOv2.)
3. **Text-conditioned CLIP experiment (in progress)** — use VLM-generated captions as a
   *view-invariant bridge* across the drone↔satellite domain gap; LoRA-fine-tune CLIP so
   drone view, satellite view, and text align, then do image+text *fusion* retrieval.

---

## 2. Execution model (critical context)

Real runs happen on a **SLURM cluster** inside an **Apptainer/Singularity container**, *not*
on a dev machine.

- **`uav_localization.def`** builds the image: `apptainer build uav_localization.sif uav_localization.def`.
  Base = `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, Miniconda py310, **torch 2.5.1 / cu124**
  (RoMa forces ≥2.5.1). Bakes in all heavy matcher deps:
  `open_clip_torch, geoclip, kornia (0.7.x), pycolmap, poselib, timm, lightning, torchgeo, rasterio`,
  plus external repos installed/cloned: **DeDoDe, LightGlue, RoMa (pinned commit `edd1b8b`),
  MATCHA** (cloned to `/opt/matcha`, exposed via `PYTHONPATH`; DIFT redirected from the deleted
  `stabilityai/stable-diffusion-2-1` HF model to a local `/data/sd21`), and **SatCLIP**
  (cloned to `/opt/satclip`, on `PYTHONPATH`). HF stack pinned: `transformers==4.49.0`,
  `diffusers==0.33.1`, `huggingface-hub==0.30.2`.
- The **repo is bind-mounted read-only** at `/opt/uav_localization` at runtime — *no rebuild
  needed to change code*, just re-run.
- Each `slurm/run_*.sh` `sbatch`es one pipeline: binds dataset + pre-staged weights + an
  HF/torch cache from `$DATAPOOL3/datasets/Visloc/...`, runs in a node-local `job_results/`,
  then **tars results back** to `tar/`.
- **Compute nodes are offline** — every model/weight must be pre-downloaded into the bound
  cache. Text-CLIP SLURM scripts set `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`.
- A local dev machine generally **cannot import** the matcher deps. Validate edits with
  `python -m py_compile <file>` (there is **no test suite, lint config, or build step**).
  Smoke-test logic with `--limit N` on a single `--flights`.

---

## 3. Repo layout

```
pipelines/            # one thin script per method
  Baseline_pipeline.py    # SIFT/ORB baseline
  surf_pipeline.py        # SIFT/SURF-style baseline variant
  lightglue_pipeline.py   # LightGlue (superpoint/disk/aliked extractors)
  loftr_pipeline.py       # LoFTR (kornia)
  roma_pipeline.py        # RoMa (outdoor/indoor; + "extreme"/aerialextrematch weights)
  matcha_pipeline.py      # MATCHA (DIFT/diffusion-feature matcher)
  clip_pipeline.py        # embedding retrieval: clip/geoclip/satclip/mobileclip/dinov2
  caption_crops.py        # [text-CLIP] VLM captioner
  clip_lora_train.py      # [text-CLIP] tri-modal InfoNCE LoRA training
  clip_fusion_pipeline.py # [text-CLIP] image+text fusion retrieval
  debug_compare.py        # ad-hoc matcher comparison helper

helpers/              # shared core — matchers are thin wrappers over this
  utils.py                # THE geo core (crops, GSD calibration, GPS<->px, prior noise)
  workers.py              # run_pipeline(): CLI, flight iteration, GPU/CPU parallelism, output
  results.py              # row building + A@Xm summary printing
  visualization.py        # best/worst match visualizations

analyze/              # post-processing / diagnostics
  retrieval_recall.py     # retrieval CSVs -> Recall@1/5/10 table
  matchability_gap.py     # does CLIP retrieval rank correlate with RoMa matchability?
  crossview_cosine.py     # [text-CLIP] did LoRA close the drone<->sat cosine gap?
  caption_qa.py           # [text-CLIP] quality-gate VLM captions before training

slurm/                # one run_*.sh per pipeline (sbatch wrappers around apptainer)
uav_localization.def  # container recipe
uav_localization.sif  # built container image (~8 GB, git-ignored)
weights/              # pre-staged model weights (git-ignored)
logs/  stats/  tar/   # SLURM stdout, sacct stats, tarred results (git-ignored)
job_results/          # unpacked result CSVs/logs + cached galleries (git-ignored)
CLAUDE.md             # in-repo guidance for Claude Code
```

~3.7k lines of Python total. Largest files: `clip_pipeline.py` (462), `analyze/matchability_gap.py`
(488), `helpers/utils.py` (379), `clip_fusion_pipeline.py` (312), `clip_lora_train.py` (293).

---

## 4. The geo core (`helpers/utils.py`) — most important file

Per-image localization flow (`collect_pipeline_rows_multitile`):

> load drone img → `tile_for_gps` picks the satellite tile → **add a simulated GPS-prior offset**
> (`PRIOR_OFFSET_STD_M = 80 m`, seeded per-row by `crc32(flight/filename)` so it's reproducible
> across processes) → `metric_crop` a **metric-isotropic, heading-rotated** satellite patch at a
> target ground-sampling distance → `match_factory` returns homography `H` → project patch centre
> through `H` → `patch_px_to_gps` → `haversine_m` error → row.

Key pieces:
- **`metric_crop` / `_metric_affine`** build the 2×3 affine mapping *output-patch px → satellite px*
  so the patch is metric-isotropic at GSD `m_per_px = SEARCH_FACTOR * K * height_m / SZ_W`.
  `SEARCH_FACTOR = 1.5` (patch larger than drone view, so there's room to localize),
  `SZ_W,SZ_H = 1024,680`. Yaw (`Phi1`) rotates the patch to drone heading.
- **`K_PER_FLIGHT`** calibrates per-flight drone-footprint GSD: `{"01":1.00,"02":1.00,"03":0.95,"08":1.00}`;
  `K_DEFAULT = 1.75*2*tan(35°)`. This calibration (commit "calibrating K") matters a lot for accuracy.
- **`gps_to_px` / `sat_px_to_gps` / `patch_px_to_gps` / `haversine_m`** — geo conversions (float64).
- **`load_flight`** → `(tiles, drone_dir, drone_csv, sat_csv)`, `tiles = [(bgr, geo), ...]`.
- **`split_flight_rows`** — deterministic **SPATIAL** train/test split (sort by wider geo axis,
  contiguous band as test, optional guard buffer). Random splits *leak* because consecutive drone
  frames overlap heavily — this is essential for the text-CLIP experiment.
- **`crop_gt_patch`** — satellite patch on the *true* GPS (no prior noise); shared by captioner + LoRA trainer.
- Determinism set at import: `random/np/cv2` seeded 0.

Acceptance + metrics:
- Accept `H` iff `inliers >= MIN_INL` (default **7**). RANSAC threshold 5.0px, top-50 matches.
- Per image: `offset_m`, `success_{5,10,15,20,25}` (A@Xm = within X meters), inlier counts.
- `helpers/results.py` prints the **A@Xm summary** (the headline metric) and accepted/raw error stats.

**Dataset quirks:** `FLIGHTS_AVAILABLE = ["01","02","03","08"]` (current 4-flight subset; this set
was changed over time — see §8). Drone CSV columns: `filename, lat, lon, height, Phi1 (yaw), ...`.
Satellite georeferencing comes from `satellite_ coordinates_range.csv` (note the space in the name).

---

## 5. How a matcher pipeline is structured (the pattern)

Each `pipelines/*_pipeline.py` is **thin**. It defines:
- `load_model(device, args)`
- `make_match_factory(model, device, args)` → `match_factory(drone_bgr)` → `fn(patch_bgr)` →
  dict with at least `sat_kp, drone_kp, raw, good, inliers, H`
- `add_args(parser)` (optional, method-specific flags)
- a viz function

…then calls **`helpers.workers.run_pipeline(...)`**, which owns CLI parsing, flight iteration,
parallelism, visualization, and writes `visloc_<name>_results.csv` + `.log`.

**To add a matcher: copy this shape — do not re-implement the loop.**

Parallelism modes in `run_pipeline`:
- `gpu_flights` — one worker per CUDA device, flights split across them (default for NN matchers).
- `cpu_chunks` — per flight, rows split across forked CPU workers (adds `--workers`; used by SIFT).

Common CLI flags: `--flights 01 03 | all`, `--limit N`, `--min-inliers`, `--no-clahe`, `--visualize`,
`--dist`. **CLAHE preprocessing is ON by default.**

---

## 6. Embedding retrieval (`pipelines/clip_pipeline.py`)

Tiles the satellite into a gallery (`iter_tiles`, cached as `.npz` per tile-size/stride/mtime in
`cache/clip_gallery`), embeds every tile + the drone image, ranks tiles by cosine similarity (no
homography). Tile size 1024, stride 512 in cached runs.

Models: `clip, geoclip, satclip, mobileclip, dinov2`. SatCLIP ckpt = `weights/satclip-vit16-l40.ckpt`.

Output `visloc_<model>_results.csv` adds `gt_tile_rank`, `top{k}_hit`, and **GPS-degraded**
`gt_rank_r<R>` columns (rank when the search is restricted to a radius R around the noisy prior).
`analyze/retrieval_recall.py` turns the rank columns into **Recall@1/5/10** per
`(model, flight, mode)` where mode ∈ {`denied` = GPS-denied/global, `r1000`, `r5000`}.

---

## 7. Text-conditioned CLIP experiment (in progress, no final results yet)

**Hypothesis:** VLM captions describing *permanent ground features* (road shape, building density,
water, land cover, cardinal directions) are view-invariant, so aligning drone/satellite/text in one
CLIP space via LoRA should bridge the domain gap and improve retrieval.

Differs from the rest of the repo:
- Uses **HF `transformers` CLIP (`openai/clip-vit-base-patch32`) + `peft` LoRA**, *not* the
  container's `open_clip` (peft can't target open_clip's packed attention; HF CLIP exposes separate
  q/k/v/out_proj + fc1/fc2 Linears). Only the adapter is saved to `weights/clip_lora/`.
- **Within-flight SPATIAL split** (`split_flight_rows`) — every flight contributes a train band and
  a held-out test band.

Flow:
1. **`caption_crops.py`** — VLM captioner. Backends: `ollama` (default), `qwen2vl`, `moondream`,
   `anthropic`. Targets: `sat` (GT crops, for training), `drone` (query images), `tile` (gallery grid).
   Resumable JSONL in `cache/captions/`. Strong system prompt bans color/brightness/view words so
   captions survive the drone↔sat gap.
2. **`clip_lora_train.py`** — tri-modal symmetric **InfoNCE** over drone↔text, sat↔text, drone↔sat.
3. **`clip_fusion_pipeline.py`** — fusion retrieval: `q = normalize(alpha*image_emb + (1-alpha)*text_emb)`.
   `--fuse-alpha` sweeps image-vs-text weight (0 = text↔text, 1 = image-only); `--no-sat-text`
   ablates to query-only. Reuses `clip_pipeline.py` gallery/retrieval/CSV machinery so
   `analyze/retrieval_recall.py` still works.

Diagnostics:
- **`analyze/crossview_cosine.py`** — measures drone↔GT-sat cosine before vs after LoRA (the honest
  "did text close the gap" metric, independent of retrieval accuracy).
- **`analyze/caption_qa.py`** — quality-gates captions (flags leaked color/brightness/banned words,
  word-length distribution, duplicate rate) before training.

---

## 8. What has actually been run + results so far

> Caveat: the flight set used was changed mid-project (git: "changing used flights"), so the
> latest per-method logs don't all cover the *same* flights — RoMa's latest runs use 5 flights
> (01,02,03,06,08), LoFTR/LightGlue logs use 10, and the older retrieval recall sweep used 11
> (01–06, 08–11). `FLIGHTS_AVAILABLE` in code is now the 4-flight subset `01,02,03,08`. Treat
> cross-method numbers below as indicative, not a locked apples-to-apples table.

### Feature-matching (headline metric = A@25m, accepted-error median; latest logs Jun 1, 2026)

| Method | Coverage | A@25m | A@10m | Median accepted err | Median inliers |
|---|---|---|---|---|---|
| **RoMa (outdoor)** | 5 flights, 3129 imgs | **64.5%** | 17.7% | **19.7 m** | 3095 |
| RoMa "extreme" (aerialextrematch wts) | 5 flights | 64.0% | 17.6% | 19.7 m | 2565 |
| LoFTR | 10 flights, 5840 imgs | 44.6% | 11.0% | 26.9 m | 122 |
| LightGlue (variant A) | 10 flights | 43.1% | 10.8% | 28.8 m | 200 |
| LightGlue (variant B) | 10 flights | 37.8% | 9.4% | 32.9 m | 278 |
| LightGlue (variant C) | 10 flights | 28.3% | 6.6% | 31.8 m | 10 |
| SIFT baseline | (weakest) | — | — | — | — |

**RoMa is by far the strongest matcher** (dense matching → far more inliers → much lower error).
The "extreme"/aerialextrematch RoMa weights (`weights/roma_extre.pth`) perform ~on par with stock
outdoor RoMa here. Per-flight error varies wildly (e.g. one flight's accepted median was ~640 m vs
another's ~20 m), which is what the matchability-gap analysis investigates.

### Embedding retrieval (older 11-flight sweep, `job_results/recall_summary.csv`)

Recall@1 / @5 / @10, per flight, in three modes (`denied` = global; `r1000`/`r5000` = restrict
search to that-meter radius around the noisy prior):
- **DINOv2** and **classic CLIP** are the strongest backbones; **GeoCLIP is weakest**.
- GPS gating helps enormously: e.g. flight 01 CLIP R@1 goes 0.098 (denied) → 0.140 (r1000);
  R@10 0.42 → 0.59. Retrieval alone is weak (single-digit to mid-teens R@1) — motivating the
  geometric matchers and the text-CLIP fusion idea.
- `recall_summary.csv` holds clip/geoclip/dinov2; satclip/mobileclip were also run (logs present).

### Diagnostics run
- **`matchability_gap.py`** collected, for 300 queries × top-50 CLIP candidates, the
  (clip_rank, clip_sim, RoMa inliers, candidate_dist_m) — to see whether retrieval rank predicts
  RoMa matchability. Raw output in `job_results/matchability_gap_results/raw_results.csv`.

### Text-CLIP experiment
Pipelines exist and compile; captioning/training/fusion have been iterated on (recent commits:
captioning fixes, "generate text to use with clip approach", "using qwen to generate captions").
**No final fusion-retrieval result CSV is present yet** — this is the active frontier.

---

## 9. How to run things

```bash
# Local logic check (no heavy deps)
python -m py_compile pipelines/<x>_pipeline.py helpers/utils.py

# Matchers on the cluster (arg = method/variant)
sbatch slurm/run_roma.sh outdoor          # or: extre (uses weights/roma_extre.pth)
sbatch slurm/run_lightglue.sh disk        # superpoint|disk|aliked
sbatch slurm/run_loftr.sh
sbatch slurm/run_surf.sh
sbatch slurm/run_matcha.sh
sbatch slurm/run_baseline.sh
sbatch slurm/run_clip.sh all              # embedding retrieval, all models

# Single-flight smoke test (in-container / wherever deps exist)
python pipelines/roma_pipeline.py --flights 03 --limit 20

# Retrieval CSVs -> Recall@k
python analyze/retrieval_recall.py --csvs visloc_clip_results.csv ... --out recall_summary.csv

# Matchability gap diagnostic
sbatch slurm/run_matchability_gap.sh      # or: python analyze/matchability_gap.py --collect/--plot

# Text-CLIP flow (captioning is offline-able via Ollama)
python pipelines/caption_crops.py --target sat   --flights all   # then drone, then tile
python pipelines/clip_lora_train.py --flights all
python pipelines/clip_fusion_pipeline.py --flights all --lora-ckpt weights/clip_lora \
       --fuse-alpha 0.0 0.5 0.7 1.0
# cluster: sbatch slurm/run_caption.sh {sat|drone|tile}; run_clip_lora.sh; run_clip_fusion.sh [base]
```

SLURM script anatomy (e.g. `run_roma.sh`): sets a stats-file location, makes node-local
`job_results/` + `torch_home/`, `apptainer run --nv` with binds (repo `:ro`, dataset `:ro`,
torch-hub + HF caches `:ro`, writable job_results), runs the pipeline, captures `APPTAINER_EXIT`,
**tars `job_results` into `tar/zz_<jobid>_<name>.tar`**, then `exit $APPTAINER_EXIT`.

---

## 10. Conventions / contracts (don't break these)

- Outputs are **flat in the repo root**: `visloc_<name>_results.csv` + `.log`
  (+ optional `visloc_<name>_visualizations/`). The **CSV column schema is the contract**
  consumed by `analyze/` and the SLURM tar step — keep it stable.
- Datasets, weights, caches, `*.csv/*.log/*.pth/*.sif/*.tar/*.out`, and all generated outputs are
  **git-ignored**. Never commit them. `cache/`, `UAV_VisLoc_dataset/`, `job_results/`, `logs/`,
  `stats/`, `tar/`, `weights/` are all ignored.
- Determinism is load-bearing (seeds at import; per-row GPS-prior noise seeded by crc32). Keep it.
- `K_PER_FLIGHT` GSD calibration directly drives accuracy — changing it changes all results.

---

## 11. State & open threads (as of 2026-06-03)

- **Done & working:** all geometric matchers (RoMa best), embedding retrieval, recall analysis,
  matchability-gap diagnostic, the whole `run_pipeline` framework + container.
- **Active frontier:** the text-conditioned CLIP experiment — caption quality, LoRA training, and
  fusion-retrieval sweeps. Diagnostics (`crossview_cosine.py`, `caption_qa.py`) exist to judge it
  honestly; the headline fusion result isn't finalized.
- **Working-tree changes** at snapshot: edits to several `slurm/run_*.sh` (lightglue, loftr, roma,
  roma_extre, surf) — uncommitted.
- **Note:** the repo's `~/.claude/plans/` directory currently contains plans from an *unrelated*
  SAE/interpretability project and the project memory dir is empty — ignore those; the text-CLIP
  design rationale lives in `CLAUDE.md` itself.

*This summary was generated by reading the source, SLURM scripts, container recipe, result logs,
and recall CSVs — not just CLAUDE.md. Numbers are from the latest run logs in `logs/` (Jun 2026).*
