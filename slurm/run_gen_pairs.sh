#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-gen-pairs
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=8:00:00

# Generate teacher-distilled training pairs for the Efficient-LoFTR LoRA finetune.
# Runs the roma_extre teacher (AerialExtreMatch RoMa) over the TRAIN spatial band
# of each flight, writing satellite crops + filtered correspondences + a
# geo-homography prior to cache/eloftr_pairs/. The container already ships romatch.
#
# Default = train labels. For the held-out validation set (crops + geo-homography,
# no teacher), submit:
#   sbatch slurm/run_gen_pairs.sh --split test --no-teacher --offset-mode jitter \
#          --out-dir /opt/uav_localization/cache/eloftr_pairs_val
# Extra args are forwarded, e.g.: sbatch slurm/run_gen_pairs.sh --num-samples 4000

EXTRE_WEIGHTS_HOST="$DATAPOOL3/datasets/Visloc/weights/roma_extre.pth"
if [ ! -f "$EXTRE_WEIGHTS_HOST" ]; then
  echo "ERROR: fine-tuned weights not found at $EXTRE_WEIGHTS_HOST" >&2
  echo "Download with: curl -L -o '$EXTRE_WEIGHTS_HOST' \\" >&2
  echo "  https://github.com/Xecades/AerialExtreMatch/releases/download/v1.0.0/roma_extre.pth" >&2
  exit 1
fi

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "$DATAPOOL3/datasets/Visloc/cache/eloftr_pairs"
mkdir -p "$DATAPOOL3/datasets/Visloc/cache/eloftr_pairs_val"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/eloftr_pairs:/opt/uav_localization/cache/eloftr_pairs" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/eloftr_pairs_val:/opt/uav_localization/cache/eloftr_pairs_val" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/torch_hub/checkpoints:/data/torch_home/hub/checkpoints:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${EXTRE_WEIGHTS_HOST}:/data/weights/roma_extre.pth:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/gen_eloftr_pairs.py \
        --teacher extre \
        --extre-weights /data/weights/roma_extre.pth \
        --flights 01 02 03 08 \
        --out-dir /opt/uav_localization/cache/eloftr_pairs \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_gen_pairs.tar" job_results
cp "zz_${SLURM_JOB_ID}_gen_pairs.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
