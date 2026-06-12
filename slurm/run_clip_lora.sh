#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-clip-lora
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH -p gpu3,gpu5
#SBATCH --mem=32G
#SBATCH --time=05:00:00

# Tri-modal LoRA fine-tuning of CLIP (drone <-> satellite <-> text).
# Backbone defaults to ViT-L/14 (matches the canonical adapter + the fusion eval).
# Sizing: ~2.5-3h / <30GB RAM for L/14 b64 w/ --grad-ckpt on an A100-40GB; keep
# requests tight — oversized walltime/mem kills backfill chances on full partitions.
# Pass --grad-ckpt when the job may land on a 40GB card (gpu3).
# Prereqs on the cluster (offline nodes):
#   - Captions:  $DATAPOOL3/datasets/Visloc/cache/captions/{flight}_sat.jsonl
#                (produce with caption_crops.py on an ONLINE node).
#   - HF model:  openai/clip-vit-large-patch14 cached under
#                $DATAPOOL3/datasets/Visloc/weights/huggingface  (pre-download once).
# Arg 1 = output adapter dir name under weights/ (default clip_lora); extra CLI args
# after it are forwarded, e.g.:
#   sbatch slurm/run_clip_lora.sh clip_lora_all9            # all 9 flights -> new dir
#   sbatch slurm/run_clip_lora.sh clip_lora_all9 --epochs 30
#   sbatch slurm/run_clip_lora.sh clip_lora_imgonly --w-dt 0 --w-st 0   # image-only control
# The trainer refuses to overwrite an adapter already present in the output dir
# (pass --overwrite to allow); use a fresh dir name per experiment.

OUT_NAME=${1:-clip_lora}; shift || true

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "$DATAPOOL3/datasets/Visloc/cache/pairs"
mkdir -p "$DATAPOOL3/datasets/Visloc/weights/${OUT_NAME}"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/captions:/opt/uav_localization/cache/captions:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/pairs:/opt/uav_localization/cache/pairs" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/${OUT_NAME}:/opt/uav_localization/weights/${OUT_NAME}" \
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
        --flights all \
        --backbone openai/clip-vit-large-patch14 \
        --rank 32 --alpha 64 --epochs 60 \
        --caption-dir /opt/uav_localization/cache/captions \
        --pairs-dir   /opt/uav_localization/cache/pairs \
        --out-dir     /opt/uav_localization/weights/${OUT_NAME} \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_clip_lora.tar" job_results
mkdir -p "${SLURM_SUBMIT_DIR}/tar"
cp "zz_${SLURM_JOB_ID}_clip_lora.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
