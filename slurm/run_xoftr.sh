#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-xoftr
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
# Pinned to gpu3 (A100-40GB) so t_match_ms is comparable across matchers.
#SBATCH -p gpu3
#SBATCH --mem=48G
#SBATCH --time=2:00:00

# XoFTR (OnderT/XoFTR), cross-modal LoFTR-family matcher. NO container rebuild:
# the repo is pure-python with deps already in the image, so it's cloned on
# shared storage and bind-mounted at /opt/XoFTR (the pipeline adds it to sys.path).
# Pretrained weights are not auto-downloaded (Google Drive) and compute nodes are
# offline, so both the clone and the checkpoint must be pre-staged on $DATAPOOL3.
# Runs single-GPU (DataIOWrapper hardcodes cuda:0). A100 optional:
#   sbatch -p gpu3,gpu4 slurm/run_xoftr.sh        # 40GB A100 (overkill but fine)
#   sbatch slurm/run_xoftr.sh                      # default gpu partition (V100 16GB)

XOFTR_REPO_HOST="$DATAPOOL3/datasets/Visloc/XoFTR"
XOFTR_WEIGHTS_HOST="$DATAPOOL3/datasets/Visloc/weights/weights_xoftr_640.ckpt"
if [ ! -d "$XOFTR_REPO_HOST" ]; then
  echo "ERROR: XoFTR repo not found at $XOFTR_REPO_HOST" >&2
  echo "  git clone --depth 1 https://github.com/OnderT/XoFTR.git '$XOFTR_REPO_HOST'" >&2
  exit 1
fi
if [ ! -f "$XOFTR_WEIGHTS_HOST" ]; then
  echo "ERROR: XoFTR weights not found at $XOFTR_WEIGHTS_HOST" >&2
  echo "  Download weights_xoftr_640.ckpt from the project's Google Drive:" >&2
  echo "  gdown --folder https://drive.google.com/drive/folders/1RAI243OHuyZ4Weo1NiTy280bCE_82s4q" >&2
  exit 1
fi

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "${XOFTR_REPO_HOST}:/opt/XoFTR:ro" \
    --bind "${XOFTR_WEIGHTS_HOST}:/data/weights/weights_xoftr_640.ckpt:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/torch_hub/checkpoints:/data/torch_home/hub/checkpoints:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/xoftr_pipeline.py \
        --weights /data/weights/weights_xoftr_640.ckpt \
        --flights 01 02 03 06 08 \
        --visualize \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_xoftr.tar" job_results
cp "zz_${SLURM_JOB_ID}_xoftr.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
