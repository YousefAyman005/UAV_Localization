#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-crossview
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=1:00:00

# Cross-view cosine diagnostic: stock CLIP vs LoRA-fine-tuned CLIP, measuring
# drone<->GT-satellite cosine similarity on the held-out spatial TEST band.
# Output is printed to this job's .out log (no result files), so there is no
# copy-back step.
# Prereq: LoRA adapter at $DATAPOOL3/datasets/Visloc/weights/clip_lora
#         (produced by run_clip_lora.sh) and openai/clip-vit-base-patch32 cached
#         under $DATAPOOL3/datasets/Visloc/weights/huggingface.

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "${LOCAL_JOB_DIR}/job_results"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/clip_lora:/opt/uav_localization/weights/clip_lora:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/analyze/crossview_cosine.py \
        --lora-ckpt /opt/uav_localization/weights/clip_lora \
        --flights all
APPTAINER_EXIT=$?

exit $APPTAINER_EXIT
