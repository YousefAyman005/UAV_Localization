#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-roma
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
# Pinned to gpu3 (A100-40GB) so t_match_ms is comparable across matchers;
# also avoids the 16GB V100s of the default gpu* partition (RoMa's ViT-L OOMs).
#SBATCH -p gpu3

# RoMa (zju3dv/RoMa). Variant = first arg: outdoor | indoor | extre.
#   extre = AerialExtreMatch fine-tune; needs the pre-staged roma_extre.pth bound.
# Extra args after the variant are forwarded to the pipeline, e.g.:
#   sbatch slurm/run_roma.sh extre --flights all
PRETRAINED=${1:-outdoor}
case "$PRETRAINED" in
  outdoor|indoor|extre) ;;
  *) echo "Usage: sbatch $0 outdoor|indoor|extre [pipeline args...]" >&2; exit 1 ;;
esac
shift || true   # drop the variant arg so "$@" holds only extra pipeline args

# extre loads a .pth checkpoint into the stock outdoor model — bind it in.
EXTRE_BIND=()
EXTRE_ARGS=()
if [ "$PRETRAINED" = "extre" ]; then
  ROMA_EXTRE_HOST="$DATAPOOL3/datasets/Visloc/weights/roma_extre.pth"
  if [ ! -f "$ROMA_EXTRE_HOST" ]; then
    echo "ERROR: roma_extre.pth not found at $ROMA_EXTRE_HOST" >&2; exit 1
  fi
  EXTRE_BIND=(--bind "${ROMA_EXTRE_HOST}:/data/weights/roma_extre.pth:ro")
  EXTRE_ARGS=(--extre-weights /data/weights/roma_extre.pth)
fi

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/torch_home/hub"

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/torch_hub/checkpoints:/data/torch_home/hub/checkpoints:ro" \
    --bind "$DATAPOOL3/datasets/Visloc/weights/huggingface:/data/torch_home/huggingface:ro" \
    "${EXTRE_BIND[@]}" \
    --bind "${LOCAL_JOB_DIR}/torch_home:/data/torch_home" \
    --bind "${LOCAL_JOB_DIR}/job_results:/data/job_results" \
    --env TORCH_HOME=/data/torch_home \
    --env HF_HOME=/data/torch_home/huggingface \
    --pwd /data/job_results \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/roma_pipeline.py \
        --pretrained "${PRETRAINED}" \
        "${EXTRE_ARGS[@]}" \
        --flights all \
        "$@"
APPTAINER_EXIT=$?

cd "${LOCAL_JOB_DIR}"
tar -cf "zz_${SLURM_JOB_ID}_roma_${PRETRAINED}.tar" job_results
cp "zz_${SLURM_JOB_ID}_roma_${PRETRAINED}.tar" "${SLURM_SUBMIT_DIR}/tar/"
rm -rf "${LOCAL_JOB_DIR}/job_results"

exit $APPTAINER_EXIT
