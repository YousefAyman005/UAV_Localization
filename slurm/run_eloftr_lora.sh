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
#   - Pairs:  cache/eloftr_pairs/      (train, with teacher)  } produced by
#             cache/eloftr_pairs_val/  (test band, geo only)  } run_gen_pairs.sh
#   - Base weights: $DATAPOOL3/datasets/Visloc/weights/eloftr_outdoor.ckpt
# Trains at native 1024x680 -> request an A100-80GB (partition gpu5; the default
# gpu* pool can land on a 16GB V100 and OOM). If memory is tight, forward extra
# args, e.g.:  sbatch slurm/run_eloftr_lora.sh --long-side 832 --batch-size 2 --grad-ckpt
# Other useful overrides: --epochs 15 --gt-mode teacher --lora-mlp

ELOFTR_WEIGHTS_HOST="$DATAPOOL3/datasets/Visloc/weights/eloftr_outdoor.ckpt"
if [ ! -f "$ELOFTR_WEIGHTS_HOST" ]; then
  echo "ERROR: EfficientLoFTR weights not found at $ELOFTR_WEIGHTS_HOST" >&2
  exit 1
fi

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "$DATAPOOL3/datasets/Visloc/weights/eloftr_lora"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/eloftr_pairs:/opt/uav_localization/cache/eloftr_pairs:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/eloftr_pairs_val:/opt/uav_localization/cache/eloftr_pairs_val:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/eloftr_lora:/opt/uav_localization/weights/eloftr_lora" \
    --bind "${ELOFTR_WEIGHTS_HOST}:/data/weights/eloftr_outdoor.ckpt:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/eloftr_lora_train.py \
        --weights /data/weights/eloftr_outdoor.ckpt \
        --pairs-dir /opt/uav_localization/cache/eloftr_pairs \
        --val-pairs-dir /opt/uav_localization/cache/eloftr_pairs_val \
        --out-dir /opt/uav_localization/weights/eloftr_lora \
        --flights 01 02 03 08 \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_eloftr_lora.tar" job_results
cp "zz_${SLURM_JOB_ID}_eloftr_lora.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
