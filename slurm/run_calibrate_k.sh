#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-calibrate-k
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00

# Sweep K (patch_span_m = K * altitude) across all flights, grouped by native
# drone resolution. Edit KS below and submit twice (once per half) to fit
# the time budget; after both finish, merge the tarballs and run
# pipelines/calibrate_k.py over the combined results dir.
#
#   3976x2652 group: 01 02 03 04 06 08 09 11   (8 flights)
#   3000x2000 group: 05 10                     (2 flights)

METHOD=${1:-disk}
LIMIT=${2:-80}
FLIGHTS=("01" "02" "03" "04" "05" "06" "08" "09" "10" "11")
# Half 1: KS=(1.4 1.55 1.7 1.85 2.0 2.15 2.3)
# Half 2: KS=(2.45 2.6 2.75 2.9 3.05 3.2)
KS=(1.4 1.55 1.7 1.85 2.0 2.15 2.3)

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

EXIT=0
for FLIGHT in "${FLIGHTS[@]}"; do
  for K in "${KS[@]}"; do
    echo "=== flight=${FLIGHT}  K=${K} ==="
    run_in_container \
      /opt/uav_localization/pipelines/lightglue_pipeline.py \
        --method "${METHOD}" \
        --flights "${FLIGHT}" \
        --limit "${LIMIT}" \
        --k-override "${K}" \
        --results-suffix "k${K}_f${FLIGHT}" \
      || EXIT=$?
  done
done

# Aggregate sweep CSVs to a fresh k_calibration.json in the job dir, then
# copy it back to the repo (under tar/ for inspection and out of source tree).
echo "=== aggregating to k_calibration.json ==="
run_in_container \
  /opt/uav_localization/pipelines/calibrate_k.py \
    --results-dir /data/job_results \
    --out /data/job_results/k_calibration.json \
  || EXIT=$?

cp "${LOCAL_JOB_DIR}/job_results/k_calibration.json" \
   "${SLURM_SUBMIT_DIR}/tar/k_calibration_${SLURM_JOB_ID}.json" 2>/dev/null || true

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_calibrate_k.tar" job_results
cp "zz_${SLURM_JOB_ID}_calibrate_k.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $EXIT
