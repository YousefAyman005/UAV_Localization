#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-kcalib
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
# RoMa uses a DINOv2 ViT-L backbone, which OOMs on the 16GB V100s the default
# gpu* partition can land on — restrict to the A100 partitions (gpu3/gpu4=40GB,
# gpu5=80GB) for headroom and best availability.
#SBATCH -p gpu3,gpu4,gpu5

# Per-flight K calibration sweep with the roma_extre teacher. Re-derives
# K_PER_FLIGHT from scratch (argmin median offset_m, inlier tiebreak) to verify the
# earlier LightGlue-based values. Extra args after the script are forwarded, e.g.
#   sbatch slurm/run_calibrate_k.sh --flights 04 05 06 10 11 --sample 30
ROMA_EXTRE_HOST="$DATAPOOL3/datasets/Visloc/weights/roma_extre.pth"
if [ ! -f "$ROMA_EXTRE_HOST" ]; then
  echo "ERROR: roma_extre.pth not found at $ROMA_EXTRE_HOST" >&2
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
    --bind "${ROMA_EXTRE_HOST}:/data/weights/roma_extre.pth:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/calibrate_k.py \
        --pretrained extre \
        --extre-weights /data/weights/roma_extre.pth \
        --flights all \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_kcalib.tar" job_results
cp "zz_${SLURM_JOB_ID}_kcalib.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
