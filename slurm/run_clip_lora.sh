#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-clip-lora
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Tri-modal LoRA fine-tuning of CLIP (drone <-> satellite <-> text).
# Prereqs on the cluster (offline nodes):
#   - Captions:  $DATAPOOL3/datasets/Visloc/cache/captions/{flight}_sat.jsonl
#                (produce with caption_crops.py on an ONLINE node).
#   - HF model:  openai/clip-vit-base-patch32 cached under
#                $DATAPOOL3/datasets/Visloc/weights/huggingface  (pre-download once).
# Extra CLI args are forwarded, e.g.:  sbatch slurm/run_clip_lora.sh --epochs 15

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "$DATAPOOL3/datasets/Visloc/cache/pairs"
mkdir -p "$DATAPOOL3/datasets/Visloc/weights/clip_lora"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/captions:/opt/uav_localization/cache/captions:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/pairs:/opt/uav_localization/cache/pairs" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/clip_lora:/opt/uav_localization/weights/clip_lora" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/clip_lora_train.py \
        --flights 01 02 03 04 05 06 08 09 \
        --caption-dir /opt/uav_localization/cache/captions \
        --pairs-dir   /opt/uav_localization/cache/pairs \
        --out-dir     /opt/uav_localization/weights/clip_lora \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_clip_lora.tar" job_results
cp "zz_${SLURM_JOB_ID}_clip_lora.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
