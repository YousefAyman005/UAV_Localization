#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-clip-fusion
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Image+text fusion retrieval with the LoRA-fine-tuned CLIP.
# Prereqs:
#   - LoRA adapter:   $DATAPOOL3/datasets/Visloc/weights/clip_lora  (from run_clip_lora.sh)
#   - Drone captions: $DATAPOOL3/datasets/Visloc/cache/captions/{flight}_drone.jsonl
#   - HF model cached under $DATAPOOL3/datasets/Visloc/weights/huggingface
# First arg = "base" runs the stock-CLIP baseline (no adapter); anything else
# (default) uses the fine-tuned adapter. Extra args after it are forwarded.
#   sbatch slurm/run_clip_fusion.sh             # fine-tuned, default alpha sweep
#   sbatch slurm/run_clip_fusion.sh base        # stock-CLIP image-only baseline

MODE=${1:-lora}; shift || true
if [ "$MODE" = "base" ]; then
    LORA_ARG=""
else
    LORA_ARG="--lora-ckpt /opt/uav_localization/weights/clip_lora"
fi

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "$DATAPOOL3/datasets/Visloc/weights/clip_gallery"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/captions:/opt/uav_localization/cache/captions:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/clip_lora:/opt/uav_localization/weights/clip_lora:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/clip_gallery:/opt/uav_localization/cache/clip_gallery" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/clip_fusion_pipeline.py \
        --flights all \
        ${LORA_ARG} \
        --fuse-alpha 0.0 0.3 0.5 0.7 1.0 \
        --caption-dir /opt/uav_localization/cache/captions \
        --cache-dir   /opt/uav_localization/cache/clip_gallery \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_clip_fusion_${MODE}.tar" job_results
cp "zz_${SLURM_JOB_ID}_clip_fusion_${MODE}.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
