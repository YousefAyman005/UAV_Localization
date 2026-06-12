#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-plot-fig
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# CPU-only figure generator: runs one analyze/plot_*.py inside the container
# with the dataset bound. Usage:
#   sbatch slurm/run_plot_fig.sh analyze/plot_metric_crop_fig.py --flight 08
# The figure script's --out is forced to the node-local job dir and the
# resulting files are tarred back to tar/ (extract into thesis/figures/).
#SBATCH -p pool2
#SBATCH --mem=16G
#SBATCH --time=0:30:00

PLOT_SCRIPT="$1"; shift
[[ "$PLOT_SCRIPT" =~ ^analyze/plot_[a-zA-Z0-9_]+\.py$ ]] || {
    echo "ERROR: PLOT_SCRIPT must match analyze/plot_*.py, got: $PLOT_SCRIPT" >&2
    exit 1
}

source "/etc/slurm/local_job_dir.sh"
mkdir -p "${SLURM_SUBMIT_DIR}/tar"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"

apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/captions:/opt/uav_localization/cache/captions:ro" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    python3 "/opt/uav_localization/${PLOT_SCRIPT}" \
        --out /data/job_results \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_plot_fig.tar" job_results
cp "zz_${SLURM_JOB_ID}_plot_fig.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
