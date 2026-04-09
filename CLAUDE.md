# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Bachelor thesis project: UAV localization via aerial-to-satellite image matching. Generates a synthetic dataset of UAV/satellite image pairs over Berlin using Google Maps Static API.

## Key Commands

```bash
# Activate venv (required before running scripts)
source .venv/bin/activate
# OR use the venv python directly:
.venv/bin/python3 <script>

# Generate dataset (dry-run first to verify CSV)
.venv/bin/python3 berlin_dataset.py --dry-run
.venv/bin/python3 berlin_dataset.py

# Install dependencies
.venv/bin/pip install requests Pillow pandas tqdm numpy
```

## Architecture

**`berlin_dataset.py`** — Main script. Downloads 1,000 satellite images (600m coverage, 1024x1024px) from Google Maps Static API, then 3 UAV images per satellite at random altitudes/offsets/rotations with simulated camera effects. Outputs PNG images and `berlin_pairs.csv` with ground truth (offset, rotation, altitude). Supports `--dry-run` and resumable downloads.

**`generate_uav_crops.py`** — Abandoned alternative approach (crops UAV from satellite images). Not used.

## Dataset Layout

```
berlin_uav_dataset/
  satellite/   # berlin_sat_0001.png ... (1024x1024, 600m coverage)
  uav/         # berlin_uav_0001_1_alt80.png ... (1024x1024, variable coverage)
berlin_pairs.csv   # ground truth: offsets, rotation, altitude per pair
```

## Important Details

- Google Maps API key is in `berlin_dataset.py` — do not commit to public repos
- Uses Homebrew Python 3.13 venv (system Python 3.9 lacks installed packages)
- UAV images have random offset from satellite center (not centered — critical for training)
- Deterministic RNG seeds ensure reproducibility; changing `RANDOM_SEED` changes the entire dataset
- Script is resumable: re-running skips already-downloaded satellites based on CSV
- Preview HTML auto-opens after first 50 satellites for visual QA
