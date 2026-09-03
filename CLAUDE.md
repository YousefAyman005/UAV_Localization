# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A bachelor-thesis benchmark for **UAV (drone) → satellite visual localization**: given a drone
image with a noisy GPS prior, find where it is on a georeferenced satellite map and report the
error in meters. Two families of methods are compared, plus an in-progress text-conditioned CLIP
experiment.

## Execution model (read this first)

Real runs happen on a **SLURM cluster** inside an **Apptainer** container, not on a dev machine.

- `uav_localization.def` builds the image (`apptainer build uav_localization.sif uav_localization.def`).
  It bakes in all the heavy matcher deps (torch 2.5.1/cu124, open_clip, kornia, LightGlue, RoMa,
  MATCHA via PYTHONPATH, SatCLIP via PYTHONPATH, pycolmap, …). The repo itself is **bind-mounted
  read-only** at `/opt/uav_localization` at runtime — no rebuild needed to change code.
- Each `slurm/run_*.sh` script `sbatch`es one pipeline: it binds the dataset, the pre-staged
  weights, and an HF/torch cache from `$DATAPOOL3/datasets/Visloc/...`, runs the pipeline, then
  **tars results back** to `tar/`. Compute nodes are **offline** — every model/weight must be
  pre-downloaded into the bound cache, and the text-CLIP SLURM scripts set
  `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`.
- A local machine generally **cannot import** the matcher deps. Validate edits with
  `python -m py_compile <file>` (there is no test suite, lint config, or build step). Smoke-test
  logic with `--limit N` on a single `--flights`.

## Pipeline families

### 1. Feature-matching matchers (`pipelines/*_pipeline.py`)
SIFT/ORB/BRISK baseline, LightGlue, LoFTR, EfficientLoFTR, XoFTR, RoMa, MATCHA. Each script is **thin**: it defines
`load_model`, `make_match_factory`, `add_args`, and a viz fn, then calls
`helpers.workers.run_pipeline(...)`. To add a matcher, copy this shape — do not re-implement the
loop.

`run_pipeline` (`helpers/workers.py`) owns CLI parsing, flight iteration, parallelism
(`gpu_flights`: one worker per CUDA device; `cpu_chunks`: fork rows per flight), visualization,
and `visloc_<name>_results.csv` + `.log` output. The per-image work is
`helpers/utils.py::collect_pipeline_rows_multitile`:

> load drone img → `tile_for_gps` picks the satellite tile → add a simulated GPS-prior offset
> (`PRIOR_OFFSET_STD_M`, seeded per-row by crc32 so it's reproducible) → `metric_crop` a
> metric-isotropic, heading-rotated satellite patch → `match_factory` returns `H` →
> project the drone-image centre through `H` → `patch_px_to_gps` → `haversine_m` error → row.

A match factory returns a dict with at least `sat_kp, drone_kp, raw, good, inliers, H`. `H` must
come from the shared robust-fit stage `helpers.utils.fit_similarity` (4-DOF RANSAC similarity,
drone→patch) so that methods differ only in their matches, never in the estimator. Dense /
semi-dense matchers (LoFTR, RoMa, MATCHA, …) get this dict for free from
`helpers.utils.dense_match_result(kp0, kp1, conf)`; sparse matchers build it themselves.
`run_pipeline` prints a common banner (`label=` gives the method name) — pipelines do not. Acceptance
gate is `inliers >= MIN_INL`. Metrics per image: gated `offset_m` / `success_{5..30}`, ungated
`raw_err_m` (centre projection vs TRUE GT — the patch centre is the noisy prior, not GT), and
the oracle `gt_in_patch` flag (GT outside the searched patch ⇒ unsolvable by construction).
`helpers/results.py` builds rows and prints the summary: gated + ungated A@Xm, GT-in-patch
rate, and skip counts.

### 2. Embedding retrieval (`pipelines/clip_pipeline.py`)
Tiles the satellite into a gallery (`iter_tiles`, cached `.npz` per tile-size/stride/mtime),
embeds every tile + the drone image, ranks tiles by cosine similarity (no homography). Models:
`clip, geoclip, satclip, mobileclip, dinov2`, plus the zero-shot University-1652 cross-view
geo-localization models `camp` and `sample4geo` (run via their own `slurm/run_camp.sh` /
`slurm/run_sample4geo.sh`, since they need pre-staged checkpoints). Output
`visloc_<model>_results.csv` adds
`gt_tile_rank`, `top{k}_hit`, and GPS-degraded `gt_rank_r<R>` columns. `analyze/retrieval_recall.py`
turns those rank columns into Recall@1/5/10 tables.

## The geo core (`helpers/utils.py`)

The single most important file. `metric_crop` / `_metric_affine` build the affine mapping
output-patch px → satellite px at a target ground-sampling distance; `K_PER_FLIGHT` calibrates the
per-flight GSD (`metric_m_per_px`). `gps_to_px` / `sat_px_to_gps` / `patch_px_to_gps` / `haversine_m`
do the geo conversions. `load_flight` returns `(tiles, drone_dir, drone_csv, sat_csv)` where
`tiles = [(bgr, geo), ...]`. Determinism is set at import (`random/np/cv2` seeded 0).

**Dataset quirks:** `FLIGHTS_AVAILABLE` = `["01","02","03","04","05","06","08","10","11"]` —
the nine-flight benchmark (07/09 excluded). All flights use a single satellite image. Drone CSV
columns: `filename, lat, lon, height, Phi1 (yaw), ...`.

## Text-conditioned CLIP experiment (in progress)

Goal: use VLM captions as a view-invariant bridge across the drone↔satellite gap; LoRA-fine-tune
CLIP so drone view, satellite view, and text align, then do **image+text fusion** retrieval. See
`~/.claude/plans/` and the `project-text-clip` memory for design rationale. Key points that differ
from the rest of the repo:

- Uses **HF `transformers` CLIP-family models + `peft` LoRA**, *not* the container's `open_clip`
  (peft can't target open_clip's packed attention). `--backbone` accepts CLIP
  (`openai/clip-vit-base-patch32 | -large-patch14`) and SigLIP/SigLIP2 ids
  (e.g. `google/siglip2-base-patch16-384`); input res / normalization / text padding are derived
  from the model config (weights must be pre-staged into the HF cache for offline nodes).
- **Within-flight SPATIAL split** via `helpers.utils.split_flight_rows` (random splits leak because
  consecutive drone frames overlap): every flight contributes a train band and a held-out test band.
  The trainer drops a guard band at the seam (`--split-buffer`, default 0.05; test band unaffected).
- Files: `caption_crops.py` (VLM captioner via a local Ollama server; targets `sat` = GT crops for
  training, `drone` = query images, `tile` = gallery grid; `--band` picks the spatial band —
  `--target drone --band train` feeds the trainer's drone↔own-caption term; resumable JSONL in
  `cache/captions/`), `clip_lora_train.py` (tri-modal InfoNCE with per-pairing weights
  `--w-dt/--w-st/--w-ds/--w-ddt` — `--w-dt 0 --w-st 0` is the image-only attribution control;
  in-batch negatives whose GT positions are closer than `--neg-mask-m` meters are masked as false
  negatives; refuses to overwrite an existing adapter without `--overwrite`), and
  `clip_fusion_pipeline.py` (fusion retrieval; reuses the `clip_pipeline.py` gallery/retrieve/CSV
  machinery so `analyze/retrieval_recall.py` still works). `--fuse-alpha` is the IMAGE weight on
  query and gallery (0 = text↔text, 1 = true image-only); `--gallery-alpha` decouples the gallery
  blend (`--fuse-alpha 1.0 --gallery-alpha 0.7` = image-only query vs text-fused gallery → no VLM
  at query time).

## Common commands

```bash
# Quick local logic check (no heavy deps needed)
python -m py_compile pipelines/<x>_pipeline.py helpers/utils.py

# Run a matcher on the cluster (each script wraps apptainer; arg = method/variant)
sbatch slurm/run_roma.sh outdoor          # or: extre (uses weights/roma_extre.pth)
sbatch slurm/run_lightglue.sh disk
sbatch slurm/run_clip.sh all              # embedding retrieval, all models

# Quick single-flight smoke test (in-container or wherever deps exist)
python pipelines/roma_pipeline.py --flights 03 --limit 20

# Post-process retrieval CSVs into Recall@k
python analyze/retrieval_recall.py --csvs visloc_clip_results.csv ... --out recall_summary.csv

# Text-CLIP flow (captioning is free + offline-able via Ollama on a Mac)
python pipelines/caption_crops.py --target sat   --flights all   # then drone, then tile
python pipelines/clip_lora_train.py --flights all                # spatial train band
python pipelines/clip_fusion_pipeline.py --flights all --lora-ckpt weights/clip_lora \
       --fuse-alpha 0.0 0.5 0.7 1.0
# cluster equivalents: sbatch slurm/run_caption.sh {sat|drone|tile}; run_clip_lora.sh; run_clip_fusion.sh [base]
```

## Conventions

- Outputs are flat in the repo root: `visloc_<name>_results.csv` + `.log` (+ optional
  `visloc_<name>_visualizations/` with `--visualize`). The CSV column schema is the contract
  consumed by `analyze/` and the SLURM tar step — keep it stable.
- Common matcher CLI flags (from `run_pipeline`): `--flights 01 03 | all`, `--limit N`,
  `--min-inliers`, `--no-clahe`, `--visualize`. CLAHE preprocessing is **on by default**.
- Datasets, weights, caches, and all generated outputs are git-ignored; never commit `*.csv/*.pth/
  *.sif/UAV_VisLoc_dataset/cache/`.
