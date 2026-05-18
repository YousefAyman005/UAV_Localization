#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-calibrate-k-baseline
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# Same K sweep as run_calibrate_k.sh but using the SIFT/ORB/BRISK baseline
# pipeline. Lets us compare whether the optimal K is matcher-specific or a
# pure data/camera property.
#
# Same K grid, same flights — only the matcher and CPU parallelism change.

METHOD=${1:-sift}
LIMIT=${2:-0}
FLIGHTS=("01" "02" "03" "04" "05" "06" "08" "09" "10" "11")
KS=(0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.90 0.95 1.00 1.10 1.20 1.30 1.40 1.55 1.70 1.90 2.15 2.45 2.75 3.10)

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"

run_in_container() {
  apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/torch_hub/checkpoints:/data/torch_home/hub/checkpoints:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    "$@"
}

LIMIT_ARG=()
if [ -n "$LIMIT" ] && [ "$LIMIT" != "0" ]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

EXIT=0
for FLIGHT in "${FLIGHTS[@]}"; do
  for K in "${KS[@]}"; do
    echo "=== flight=${FLIGHT}  K=${K} ==="
    run_in_container \
      /opt/uav_localization/pipelines/Baseline_pipeline.py \
        --method "${METHOD}" \
        --workers 10 \
        --flights "${FLIGHT}" \
        "${LIMIT_ARG[@]}" \
        --k-override "${K}" \
        --results-suffix "k${K}_f${FLIGHT}" \
      || EXIT=$?
  done
done

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_calibrate_k_baseline_${METHOD}.tar" job_results
cp "zz_${SLURM_JOB_ID}_calibrate_k_baseline_${METHOD}.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $EXIT
