#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-eloftr-lora
#SBATCH --output=logs/%j_%x.out
#SBATCH --partition=gpu5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# LoRA-finetune Efficient-LoFTR on the teacher-distilled drone<->satellite pairs.
# Prereqs (offline nodes):
#   - Pairs:  cache/eloftr_pairs2/      (train band, with teacher) } produced by
#             cache/eloftr_pairs2_val/  (val band, geo only)       } run_gen_pairs.sh
#   - Base weights: $DATAPOOL3/datasets/Visloc/weights/eloftr_outdoor.ckpt
#   - ELOFTR_LORA_OUT names the adapter out-dir (default eloftr_lora2_r8aug). The
#     dir AND its `_last` sibling are mkdir'd on datapool and bound below; the repo
#     working tree must contain matching git-ignored mount points
#     weights/<name> and weights/<name>_last (the repo itself is mounted ro).
# Trains at native 1024x680 -> request an A100-80GB (partition gpu5; the default
# gpu* pool can land on a 16GB V100 and OOM). If memory is tight, forward extra
# args, e.g.:  sbatch slurm/run_eloftr_lora.sh --long-side 832 --grad-ckpt
# Standard 3-config round:
#   ELOFTR_LORA_OUT=eloftr_lora2_r8aug sbatch slurm/run_eloftr_lora.sh
#   ELOFTR_LORA_OUT=eloftr_lora2_r16   sbatch slurm/run_eloftr_lora.sh --rank 16 --alpha 32
#   ELOFTR_LORA_OUT=eloftr_lora2_noaug sbatch slurm/run_eloftr_lora.sh --no-aug

ELOFTR_WEIGHTS_HOST="$DATAPOOL3/datasets/Visloc/weights/eloftr_outdoor.ckpt"
if [ ! -f "$ELOFTR_WEIGHTS_HOST" ]; then
  echo "ERROR: EfficientLoFTR weights not found at $ELOFTR_WEIGHTS_HOST" >&2
  exit 1
fi

OUT_NAME="${ELOFTR_LORA_OUT:-eloftr_lora2_r8aug}"
echo "Adapter out-dir: weights/${OUT_NAME} (+ _last)"

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "$DATAPOOL3/datasets/Visloc/weights/${OUT_NAME}"
mkdir -p "$DATAPOOL3/datasets/Visloc/weights/${OUT_NAME}_last"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/eloftr_pairs2:/opt/uav_localization/cache/eloftr_pairs2:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/eloftr_pairs2_val:/opt/uav_localization/cache/eloftr_pairs2_val:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/${OUT_NAME}:/opt/uav_localization/weights/${OUT_NAME}" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/${OUT_NAME}_last:/opt/uav_localization/weights/${OUT_NAME}_last" \
    --bind "${ELOFTR_WEIGHTS_HOST}:/data/weights/eloftr_outdoor.ckpt:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/eloftr_lora_train.py \
        --weights /data/weights/eloftr_outdoor.ckpt \
        --pairs-dir /opt/uav_localization/cache/eloftr_pairs2 \
        --val-pairs-dir /opt/uav_localization/cache/eloftr_pairs2_val \
        --out-dir "/opt/uav_localization/weights/${OUT_NAME}" \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_eloftr_lora.tar" job_results
cp "zz_${SLURM_JOB_ID}_eloftr_lora.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
