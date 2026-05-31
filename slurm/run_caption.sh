#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-caption
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# FREE captioning with a local VLM on the GPU (no API cost, fully offline).
# Prereqs: the VLM weights must be cached under
#   $DATAPOOL3/datasets/Visloc/weights/huggingface  (pre-download Qwen2-VL once).
# Args: TARGET (sat|drone|tile) then extra flags. Captions persist to DATAPOOL3.
# With a within-flight spatial split, caption ALL flights for every target:
#   sbatch slurm/run_caption.sh sat   --flights all   # GT crops (training band)
#   sbatch slurm/run_caption.sh drone --flights all   # query images (test band)
#   sbatch slurm/run_caption.sh tile  --flights all   # gallery grid (satellite db)

TARGET=${1:-sat}; shift || true

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "$DATAPOOL3/datasets/Visloc/cache/captions"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/captions:/opt/uav_localization/cache/captions" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --pwd /opt/uav_localization \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/caption_crops.py \
        --backend qwen2vl \
        --target "${TARGET}" \
        --out-dir /opt/uav_localization/cache/captions \
        "$@"

exit $?
