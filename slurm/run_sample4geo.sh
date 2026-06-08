#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-sample4geo
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=46G
#SBATCH --time=3:00:00

# Zero-shot Sample4Geo (ICCV'24 cross-view geo-localization) as a retrieval model on
# UAV-VisLoc, using the pretrained University-1652 weights. Sample4Geo's TimmModel is just
# a timm ConvNeXt-Base, rebuilt inline in clip_pipeline.py (timm is in the container), so
# no vendoring/rebuild is needed. Its checkpoint is pre-staged at
# $DATAPOOL3/datasets/Visloc/weights/sample4geo_u1652.pth (offline nodes), bound read-only
# via the weights dir. ConvNeXt-base inference is light (fits a 40GB A100 easily).
# Extra args are forwarded, e.g. a smoke test:  sbatch slurm/run_sample4geo.sh --limit 20

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "${SLURM_SUBMIT_DIR}/tar"

run_in_container() {
  apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights:/opt/uav_localization/weights:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    "$@"
}

EXIT=0
run_in_container \
  /opt/uav_localization/pipelines/clip_pipeline.py \
    --model sample4geo \
    --flights 01 02 03 08 \
    --test-split \
    --rebuild-cache \
    --gps-radii 1000 5000 \
    --sample4geo-ckpt /opt/uav_localization/weights/sample4geo_u1652.pth \
    --cache-dir /data/job_results/clip_gallery \
    "$@" \
  || EXIT=$?

run_in_container \
  /opt/uav_localization/analyze/retrieval_recall.py \
    --csvs visloc_sample4geo_results.csv \
    --out  recall_sample4geo.csv \
  || EXIT=$?

cp "${LOCAL_JOB_DIR}/job_results/recall_sample4geo.csv" \
   "${SLURM_SUBMIT_DIR}/tar/recall_sample4geo_${SLURM_JOB_ID}.csv" 2>/dev/null || true

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_sample4geo.tar" job_results
cp "zz_${SLURM_JOB_ID}_sample4geo.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $EXIT
