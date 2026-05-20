#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-clip-recall
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# Week-1 retrieval-recall gate: clip_pipeline.py over 5 embedding models
# (CLIP, GeoCLIP, SatCLIP, MobileCLIP-S2, DINOv2) × all flights, computing
# both GPS-denied and GPS-degraded (1km, 5km) GT-tile ranks. Aggregates with
# analyze/retrieval_recall.py into recall_summary.csv.
#
# Cache: --rebuild-cache forces fresh embeddings under the job's ephemeral
# clip_gallery. None of the staged cluster caches are reused (MobileCLIP &
# DINOv2 don't have pre-staged galleries anyway).

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"

run_in_container() {
  # HF_HOME points at the read-only staged HF cache on $DATAPOOL3. All models
  # used here (CLIP-ViT-B-32-laion2b, openai/clip-vit-large-patch14 for
  # GeoCLIP) must be pre-downloaded on the login node into that cache.
  # HF_HUB_OFFLINE=1 guarantees no download is attempted (which would crash
  # on the read-only mount); a missing model errors loudly instead.
  apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights:/opt/uav_localization/weights:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/torch_hub/checkpoints:/data/torch_home/hub/checkpoints:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/torch_hub/facebookresearch_dinov2_main:/data/torch_home/hub/facebookresearch_dinov2_main:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    "$@"
}

EXIT=0
for MODEL in clip geoclip satclip mobileclip dinov2; do
  echo "=== model=${MODEL} ==="
  run_in_container \
    /opt/uav_localization/pipelines/clip_pipeline.py \
      --model "${MODEL}" \
      --flights all \
      --rebuild-cache \
      --gps-radii 1000 5000 \
      --satclip-ckpt /opt/uav_localization/weights/satclip-vit16-l40.ckpt \
      --cache-dir /data/job_results/clip_gallery \
    || EXIT=$?
done

echo "=== aggregating recall_summary.csv ==="
run_in_container \
  /opt/uav_localization/analyze/retrieval_recall.py \
    --csvs visloc_clip_results.csv visloc_geoclip_results.csv \
           visloc_satclip_results.csv visloc_mobileclip_results.csv \
           visloc_dinov2_results.csv \
    --out  recall_summary.csv \
  || EXIT=$?

cp "${LOCAL_JOB_DIR}/job_results/recall_summary.csv" \
   "${SLURM_SUBMIT_DIR}/tar/recall_summary_${SLURM_JOB_ID}.csv" 2>/dev/null || true

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_clip_recall.tar" job_results
cp "zz_${SLURM_JOB_ID}_clip_recall.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $EXIT
