# UAV → Satellite Visual Localization

A reproducible benchmark for **UAV (drone) → satellite visual localization**, built for a
bachelor thesis. Given a drone image and a *noisy GPS prior*, each method must find where the
drone actually is on a georeferenced satellite map. Errors are reported in **meters**, so
results are comparable across flights with different satellite resolutions.

Two families of methods are compared on nine cleaned flights of the public
[UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) dataset:

| Family | Idea | Methods |
|---|---|---|
| **Geometric feature matching** | Match drone ↔ satellite patch, fit a shared 4-DOF similarity, project the image centre to a metric position | SIFT / ORB / BRISK baselines, LightGlue (DISK / DeDoDe / SIFT), LoFTR, EfficientLoFTR (+ LoRA), XoFTR, RoMa (+ AerialExtreMatch weights), MATCHA |
| **Embedding retrieval** | Tile the satellite into a gallery and rank tiles by cosine similarity, no geometry | CLIP, SigLIP2, GeoCLIP, SatCLIP, MobileCLIP, DINOv2, CAMP, Sample4Geo, LoRA-fine-tuned CLIP / SigLIP2, image+text fusion with VLM captions |

## Headline results

Full benchmark, nine flights, N = 5058 images, GPS prior σ = 80 m. A@Xm is the fraction of
images localized within X meters (gated on `inliers >= 7`).

| Matcher | A@25m | A@30m | Median err (m) | t_match (ms) |
|---|---|---|---|---|
| SIFT (baseline) | 17.7 | 21.7 | 28.2 | 730 |
| LightGlue–DeDoDe | 56.1 | 66.6 | 22.2 | 235 |
| LoFTR | 56.6 | 66.6 | 22.0 | 127 |
| XoFTR | 57.8 | 67.6 | 21.7 | 53 |
| RoMa | 59.1 | 69.6 | 21.5 | 492 |
| **EfficientLoFTR** | **59.9** | **70.1** | 21.1 | 73 |

Retrieval on the held-out spatial test band (N = 1270), Recall@k in %:

| Model | R@1 | R@5 | R@10 |
|---|---|---|---|
| CLIP ViT-L/14 (zero-shot) | 6.4 | 20.3 | 29.5 |
| Sample4Geo (University-1652) | 19.8 | 42.0 | 53.1 |
| **CLIP ViT-L/14 + LoRA (v5)** | **37.1** | **66.9** | **76.5** |

Three findings: learned matchers beat classical ones roughly threefold but have plateaued
(the top three are within 0.8 pp); the dominant remaining error is a matcher-independent
per-flight bias that can be calibrated out with no training (`analyze/bias_calib.py`);
for retrieval, LoRA task adaptation matters far more than the backbone, while caption-text
fusion gives no net gain over image-only queries. See `thesis/` for the full write-up.

## Repository layout

```
pipelines/                 one thin script per method
  baseline_pipeline.py       SIFT / ORB / BRISK
  lightglue_pipeline.py      LightGlue (disk | dedodeb | sift)
  loftr_pipeline.py          LoFTR (kornia)
  eloftr_pipeline.py         EfficientLoFTR (optional LoRA adapter)
  xoftr_pipeline.py          XoFTR
  roma_pipeline.py           RoMa (outdoor | indoor | extre)
  matcha_pipeline.py         MATCHA
  clip_pipeline.py           embedding retrieval (clip/geoclip/satclip/mobileclip/dinov2/camp/sample4geo)
  caption_crops.py           VLM captioner (Ollama) for the text-CLIP experiment
  clip_lora_train.py         tri-modal InfoNCE LoRA fine-tuning of HF CLIP / SigLIP2
  clip_fusion_pipeline.py    image+text fusion retrieval
  gen_eloftr_pairs.py        teacher-distilled (RoMa-AEM) training pairs for EfficientLoFTR
  eloftr_lora_train.py       LoRA fine-tuning of EfficientLoFTR on those pairs
  calibrate_k.py             per-flight ground-sampling-distance (K) calibration
  debug_compare.py           crop-geometry diagnostic panels

helpers/                   shared core (matchers are thin wrappers over this)
  utils.py                   geo core: metric crops, K calibration, GPS<->px, prior noise, spatial splits
  workers.py                 run_pipeline(): CLI, flight iteration, GPU/CPU parallelism, output
  results.py                 row building and A@Xm summaries
  visualization.py           best/worst match figures

analyze/                   post-processing, diagnostics, thesis figures
  retrieval_recall.py        retrieval CSVs -> Recall@1/5/10
  bias_calib.py              train-band per-flight bias calibration
  band_metrics.py            restrict any results CSV to a spatial band
  matchability_gap.py        does CLIP rank predict RoMa matchability?
  crossview_cosine.py        drone<->sat cosine, stock vs LoRA
  caption_qa.py              quality gate for VLM captions
  plot_*.py                  thesis figures

slurm/                     one run_*.sh per pipeline (sbatch wrappers around apptainer)
third_party/CAMP/          vendored CAMP retrieval model
thesis/                    abstract, methodology, results, conclusion (markdown) + figures
uav_localization.def       Apptainer container recipe
CLAUDE.md                  detailed in-repo developer guidance
```

Datasets, weights, caches, `*.sif`, and every generated output (`*.csv`, `*.log`, `*.tar`,
`*.out`, `job_results/`, `tar/`, `logs/`, `stats/`) are git-ignored.

## How the benchmark works

Per drone image (`helpers/utils.py::collect_pipeline_rows_multitile`):

1. Load the drone image and pick the satellite tile that contains its GPS.
2. Add a simulated GPS-prior offset (σ = 80 m, seeded per image via crc32 so runs are reproducible).
3. Cut a **metric-isotropic, heading-rotated** satellite patch around the noisy prior
   (`metric_crop`, search factor 1.75, per-flight calibrated GSD from `K_PER_FLIGHT`).
4. The matcher returns correspondences. All matchers share one robust estimator,
   `fit_similarity` (4-DOF RANSAC similarity), so methods differ only in their matches.
5. Project the drone image centre through the fit, convert to GPS, measure the haversine
   error against the true position.

Accepted if `inliers >= MIN_INL` (7). Each row records gated `offset_m` and `success_{5..30}`,
ungated `raw_err_m`, inlier counts, match time, and a `gt_in_patch` flag.

Retrieval instead tiles the satellite (1024 px, stride 512), embeds tiles and drone image,
and records the rank of the ground-truth tile, both GPS-denied and restricted to a radius
around the prior. Training for the LoRA variants uses a **within-flight spatial split**
(`split_flight_rows`: train 60 % | buffer 5 % | val 10 % | test 25 %), because random splits
leak through overlapping consecutive frames.

## Running

Everything runs on a **SLURM cluster inside an Apptainer container**. Compute nodes are
offline, so every checkpoint and HF model must be pre-staged under
`$DATAPOOL3/datasets/Visloc/weights/`. The repo is bind-mounted read-only at
`/opt/uav_localization`, so code changes need no rebuild.

```bash
# Build the container once (~8 GB)
apptainer build uav_localization.sif uav_localization.def

# Geometric matchers (arg = variant; extra args are forwarded to the pipeline)
sbatch slurm/run_baseline.sh sift            # sift | orb | brisk
sbatch slurm/run_lightglue.sh dedodeb        # disk | dedodeb | sift
sbatch slurm/run_loftr.sh outdoor
sbatch slurm/run_eloftr.sh
sbatch slurm/run_xoftr.sh
sbatch slurm/run_roma.sh outdoor             # outdoor | indoor | extre
sbatch slurm/run_matcha.sh

# Embedding retrieval
sbatch slurm/run_clip.sh all                 # clip | geoclip | satclip | mobileclip | dinov2 | all
sbatch slurm/run_camp.sh
sbatch slurm/run_sample4geo.sh

# Text-CLIP experiment
sbatch slurm/run_caption_ollama.sh sat       # sat | drone | tile  (Ollama server runs in-container)
sbatch slurm/run_clip_lora.sh
sbatch slurm/run_clip_fusion.sh

# EfficientLoFTR LoRA
sbatch slurm/run_gen_pairs.sh                # RoMa-AEM teacher labels on the train band
sbatch slurm/run_eloftr_lora.sh

# Diagnostics and figures
sbatch slurm/run_calibrate_k.sh
sbatch slurm/run_crossview.sh
sbatch slurm/run_plot_fig.sh analyze/plot_metric_crop_fig.py --flight 08
```

Each SLURM script runs the pipeline in a node-local `job_results/`, then tars the results
back to `tar/zz_<jobid>_<name>.tar`. Stdout goes to `logs/<jobid>_<jobname>.out`.

Common pipeline flags (from `run_pipeline`): `--flights 01 03 | all`, `--limit N`,
`--min-inliers`, `--dist`, `--no-clahe`, `--no-yaw-cal`, `--visualize`. CLAHE preprocessing
is on by default. A quick smoke test inside the container:

```bash
python pipelines/roma_pipeline.py --flights 03 --limit 20
```

There is no test suite. Validate edits without the heavy deps via
`python -m py_compile pipelines/<x>_pipeline.py helpers/utils.py`.

## Outputs

Each run writes `visloc_<name>_results.csv` and `visloc_<name>_results.log` (plus
`visloc_<name>_visualizations/` with `--visualize`). The CSV column schema is the contract
consumed by `analyze/` and the thesis figure scripts. Turn retrieval CSVs into Recall@k with:

```bash
python analyze/retrieval_recall.py --csvs visloc_clip_results.csv ... --out recall_summary.csv
```

## Adding a matcher

Copy any `pipelines/*_pipeline.py`. Define `load_model`, `make_match_factory` (returns a dict
with at least `sat_kp, drone_kp, raw, good, inliers, H`, where `H` comes from
`helpers.utils.fit_similarity`; dense matchers can return
`helpers.utils.dense_match_result(kp0, kp1, conf)` directly), optional `add_args`, and a viz
function, then call `helpers.workers.run_pipeline(...)`. Do not re-implement the loop. Add a matching
`slurm/run_<name>.sh` that binds the pre-staged weights.

## Dataset notes

- Flights used: `01, 02, 03, 04, 05, 06, 08, 10, 11` (07 and 09 excluded). Frames with no
  ground structure (e.g. open water) were removed before benchmarking.
- Drone CSV columns: `filename, lat, lon, height, Phi1 (yaw), ...`. Satellite georeferencing
  comes from `satellite_ coordinates_range.csv` (the space in the name is in the dataset).
- The per-flight GSD calibration (`K_PER_FLIGHT`) and a per-leg yaw correction directly
  drive accuracy. Changing them changes every result.
