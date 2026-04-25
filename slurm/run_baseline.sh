#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-baseline
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00

METHOD=${1:-sift}
case "$METHOD" in
  sift|orb|brisk) ;;
  *) echo "Usage: sbatch $0 sift|orb|brisk" >&2; exit 1 ;;
esac

source "/etc/slurm/local_job_dir.sh"
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${SLURM_SUBMIT_DIR}/UAV_VisLoc_dataset"
mkdir -p "${SLURM_SUBMIT_DIR}/weights"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights:/opt/uav_localization/weights:ro" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/opt/uav_localization/weights/torch_hub \
    --env HF_HOME=/opt/uav_localization/weights/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/Baseline_pipeline.py \
        --method "${METHOD}" \
        --flights all
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_baseline_${METHOD}.tar" job_results
cp "zz_${SLURM_JOB_ID}_baseline_${METHOD}.tar" "${SLURM_SUBMIT_DIR}/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
