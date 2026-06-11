#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-eloftr
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=2:00:00

# Efficient LoFTR (zju3dv/EfficientLoFTR), full model / FP32. Same dense-matcher
# pipeline shape as LoFTR. The pretrained checkpoint is NOT auto-downloaded (it
# lives on Google Drive) and compute nodes are offline, so it must be pre-staged
# on the cluster and is bound in below.

ELOFTR_WEIGHTS_HOST="$DATAPOOL3/datasets/Visloc/weights/eloftr_outdoor.ckpt"
if [ ! -f "$ELOFTR_WEIGHTS_HOST" ]; then
  echo "ERROR: EfficientLoFTR weights not found at $ELOFTR_WEIGHTS_HOST" >&2
  echo "Download eloftr_outdoor.ckpt from the project's Google Drive and place it there:" >&2
  echo "  pip install gdown && gdown --folder \\" >&2
  echo "  https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q" >&2
  exit 1
fi

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"

# Optionally mount a trained LoRA adapter (ELOFTR_LORA_NAME picks the datapool
# weights dir, default eloftr_lora) and evaluate it by forwarding
# --lora-ckpt /opt/uav_localization/weights/<name> (writes the separate
# visloc_eloftr_lora_results.csv). Baseline runs are unaffected. The repo must
# contain a matching git-ignored mount point weights/<name> (repo mounted ro).
LORA_NAME="${ELOFTR_LORA_NAME:-eloftr_lora}"
LORA_HOST="$DATAPOOL3/datasets/Visloc/weights/${LORA_NAME}"
LORA_BIND=()
if [ -d "$LORA_HOST" ]; then
  LORA_BIND=(--bind "${LORA_HOST}:/opt/uav_localization/weights/${LORA_NAME}:ro")
fi

echo "SEARCH_FACTOR override: UAV_SEARCH_FACTOR=${UAV_SEARCH_FACTOR:-1.75}"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/torch_hub/checkpoints:/data/torch_home/hub/checkpoints:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${ELOFTR_WEIGHTS_HOST}:/data/weights/eloftr_outdoor.ckpt:ro" \
    "${LORA_BIND[@]}" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --env UAV_SEARCH_FACTOR=${UAV_SEARCH_FACTOR:-1.75} \
    --env UAV_PRIOR_STD_M=${UAV_PRIOR_STD_M:-80.0} \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/eloftr_pipeline.py \
        --weights /data/weights/eloftr_outdoor.ckpt \
        --flights 01 02 03 06 08 \
        --visualize \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_eloftr.tar" job_results
cp "zz_${SLURM_JOB_ID}_eloftr.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
