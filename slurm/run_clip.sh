#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-clip
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00

MODEL=${1:-all}
case "$MODEL" in
  clip|geoclip|satclip|mobileclip|dinov2|all) ;;
  *) echo "Usage: sbatch $0 clip|geoclip|satclip|mobileclip|dinov2|all" >&2; exit 1 ;;
esac

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights:/opt/uav_localization/weights:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/torch_hub/checkpoints:/data/torch_home/hub/checkpoints:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/clip_gallery:/opt/uav_localization/cache/clip_gallery" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/clip_pipeline.py \
        --model "${MODEL}" \
        --satclip-ckpt    /opt/uav_localization/weights/satclip-vit16-l40.ckpt \
        --mobileclip-ckpt /opt/uav_localization/weights/mobileclip-s2.bin \
        --cache-dir /opt/uav_localization/cache/clip_gallery \
        --flights all
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_clip_${MODEL}.tar" job_results
cp "zz_${SLURM_JOB_ID}_clip_${MODEL}.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
