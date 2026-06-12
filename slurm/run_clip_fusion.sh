#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-clip-fusion
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
# Any A100 partition works (ViT-L/14 OOMs on the 16GB V100s of gpu1/gpu2/gpu*).
#SBATCH -p gpu3,gpu4,gpu5
#SBATCH --mem=32G
# A 5-alpha sweep costs ~28 min/alpha on 9 flights (satellite TIFs are reloaded
# per flight per alpha) -> ~2h20m total; 2h was too tight and a timeout loses
# the un-tarred results. Single-cell runs finish well under 1h.
#SBATCH --time=3:30:00

# Image+text fusion retrieval with the LoRA-fine-tuned CLIP.
# Prereqs:
#   - LoRA adapter:   $DATAPOOL3/datasets/Visloc/weights/clip_lora  (from run_clip_lora.sh)
#   - Drone captions: $DATAPOOL3/datasets/Visloc/cache/captions/{flight}_drone.jsonl
#   - HF model cached under $DATAPOOL3/datasets/Visloc/weights/huggingface
# First arg selects the adapter: "base" = stock-CLIP baseline (no adapter);
# "lora" (default) = weights/clip_lora; any other value = that dir name under
# weights/ (e.g. clip_lora_all9). Extra args after it are forwarded.
#   sbatch slurm/run_clip_fusion.sh                  # weights/clip_lora (4-flight)
#   sbatch slurm/run_clip_fusion.sh base             # stock-CLIP baseline
#   sbatch slurm/run_clip_fusion.sh clip_lora_all9   # the 9-flight adapter
# alpha is the IMAGE weight (1.0 = true image-only endpoint). --gallery-alpha
# decouples the gallery blend from the query, e.g. the VLM-free-query cell:
#   sbatch slurm/run_clip_fusion.sh clip_lora_all9 --fuse-alpha 1.0 --gallery-alpha 0.7

MODE=${1:-lora}; shift || true
ADAPTER="clip_lora"
if [ "$MODE" = "base" ]; then
    LORA_ARG=""
else
    [ "$MODE" != "lora" ] && ADAPTER="$MODE"
    LORA_ARG="--lora-ckpt /opt/uav_localization/weights/${ADAPTER}"
fi

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"
mkdir -p "$DATAPOOL3/datasets/Visloc/weights/clip_gallery"
# Adapter dir must exist for the ro bind even in "base" mode (nothing read from it).
mkdir -p "$DATAPOOL3/datasets/Visloc/weights/${ADAPTER}"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/cache/captions:/opt/uav_localization/cache/captions:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/${ADAPTER}:/opt/uav_localization/weights/${ADAPTER}:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/clip_gallery:/opt/uav_localization/cache/clip_gallery" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/clip_fusion_pipeline.py \
        --flights all \
        --backbone openai/clip-vit-large-patch14 \
        ${LORA_ARG} \
        --fuse-alpha 0.0 0.5 0.7 0.8 1.0 \
        --caption-dir /opt/uav_localization/cache/captions \
        --cache-dir   /opt/uav_localization/cache/clip_gallery \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_clip_fusion_${MODE}.tar" job_results
mkdir -p "${SLURM_SUBMIT_DIR}/tar"
cp "zz_${SLURM_JOB_ID}_clip_fusion_${MODE}.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
