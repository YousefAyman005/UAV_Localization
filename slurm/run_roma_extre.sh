#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-roma-extre
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# RoMa fine-tuned on the AerialExtreMatch dataset (Xecades/AerialExtreMatch).
# Same architecture as roma_outdoor; only the checkpoint differs. The weights
# file must live on the cluster (see README at bottom) and is bound in below.

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

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/torch_hub/checkpoints:/data/torch_home/hub/checkpoints:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${EXTRE_WEIGHTS_HOST}:/data/weights/roma_extre.pth:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/roma_pipeline.py \
        --pretrained extre \
        --extre-weights /data/weights/roma_extre.pth \
        --flights all \
        --visualize
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_roma_extre.tar" job_results
cp "zz_${SLURM_JOB_ID}_roma_extre.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
