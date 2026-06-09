#!/bin/bash
# Local captioning of ALL flights, ALL targets (sat/drone/tile) via Ollama.
# Resumable: caption_crops.py skips IDs already present in each cache/captions/*.jsonl.
# Meant to run inside a detached tmux session so it survives a closed laptop.
set -u
cd /home/elsayed/workspace/UAV_Localization
PY=/home/elsayed/miniforge3/envs/uav_localization/bin/python
LOG=logs/caption_all.log
mkdir -p logs

{
  echo "=== captioning ALL targets/flights | started $(date) ==="
  for t in sat drone tile; do
    echo
    echo "########## TARGET=$t  ($(date +%H:%M:%S)) ##########"
    "$PY" pipelines/caption_crops.py --target "$t" --flights all
  done
  echo
  echo "=== ALL DONE $(date) ==="
} 2>&1 | tee "$LOG"

echo
echo ">>> captioning finished. Press Ctrl-C or close this pane. <<<"
# keep the pane alive so an attaching user can read the summary
sleep infinity
