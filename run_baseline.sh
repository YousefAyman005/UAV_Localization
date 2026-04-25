#!/bin/bash
set -euo pipefail
#SBATCH --job-name=uav-baseline
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G

# Activate $LOCAL_JOB_DIR (fast local NVMe scratch) — must come before any
# reference to that variable.
source "/etc/slurm/local_job_dir.sh"

# Output directory inside local NVMe scratch.
mkdir -p "${LOCAL_JOB_DIR}/job_results"

# Run the container.
# Dataset bind: UAV_VisLoc_dataset is the name visloc_utils.py looks for
# relative to the code root (/opt/uav_localization), so we mount it there.
# Working directory is set to /data/job_results so the output CSV lands there.
apptainer exec --nv \
    --bind "${DATAPOOL1}/datasets/UAV_Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "${HOME}/UAV_Localization:/opt/uav_localization:ro" \
    --bind "${HOME}/UAV_Localization/weights:/opt/uav_localization/weights:ro" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --pwd /data/job_results \
    "${HOME}/UAV_Localization/uav_localization.sif" \
    /opt/conda/bin/python /opt/uav_localization/Baseline_pipeline.py \
        --flights 03 \
        --method sift

# Copy results back to submit directory AFTER the container exits.
# Using tar first reduces network traffic to shared storage.
cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}.tar" job_results
cp "zz_${SLURM_JOB_ID}.tar" "${SLURM_SUBMIT_DIR}/"

# Remove job_results so the Slurm epilog autocopy does not run a redundant
# second copy.
rm -rf "${LOCAL_JOB_DIR}/job_results"
