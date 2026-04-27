#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-clip
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00

MODEL=${1:-all}
case "$MODEL" in
  clip|geoclip|satclip|all) ;;
  *) echo "Usage: sbatch $0 clip|geoclip|satclip|all" >&2; exit 1 ;;
esac

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${SLURM_SUBMIT_DIR}/UAV_VisLoc_dataset"
mkdir -p "${SLURM_SUBMIT_DIR}/weights"
mkdir -p "${SLURM_SUBMIT_DIR}/cache/clip_gallery"
mkdir -p "$DATAPOOL3/datasets/Visloc/weights/clip_gallery"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights:/opt/uav_localization/weights:ro" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/clip_gallery:/opt/uav_localization/cache/clip_gallery" \
    --env TORCH_HOME=/opt/uav_localization/weights/torch_hub \
    --env HF_HOME=/opt/uav_localization/weights/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/clip_pipeline.py \
        --model "${MODEL}" \
        --satclip-ckpt /opt/uav_localization/weights/satclip-vit16-l40.ckpt \
        --cache-dir /opt/uav_localization/cache/clip_gallery \
        --flights all
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_clip_${MODEL}.tar" job_results
cp "zz_${SLURM_JOB_ID}_clip_${MODEL}.tar" "${SLURM_SUBMIT_DIR}/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
