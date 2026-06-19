#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-caption-probe
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=32G

# Cheap prompt probe: caption a few crops with each candidate prompt to pick a
# prompt BEFORE the full re-caption. Same Ollama-in-container setup as
# run_caption_ollama.sh, but runs analyze/caption_prompt_probe.py (which prints
# captions to the slurm log). Extra args pass through, e.g.:
#   sbatch slurm/run_caption_probe.sh --flight 08

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg

OLLAMA_DIR="$DATAPOOL3/datasets/Visloc/weights/ollama"
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p "${LOCAL_JOB_DIR}/ollama_home"

# Ollama server inside the container (--nv exposes the job GPU). HOME points at
# node-local scratch so serve only writes its keypair/temp there.
apptainer exec --nv \
    --bind "$OLLAMA_DIR/runtime:/opt/ollama:ro" \
    --bind "$OLLAMA_DIR/models:/ollama_models:ro" \
    --bind "${LOCAL_JOB_DIR}/ollama_home:/ollama_home" \
    --env HOME=/ollama_home \
    --env OLLAMA_MODELS=/ollama_models \
    --env OLLAMA_HOST=$OLLAMA_HOST \
    --env OLLAMA_KEEP_ALIVE=-1 \
    --env OLLAMA_NUM_PARALLEL=1 \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/ollama/bin/ollama serve > "${LOCAL_JOB_DIR}/ollama_serve.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT

for i in $(seq 1 60); do
    curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null && break
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "ollama serve died during startup:"
        cat "${LOCAL_JOB_DIR}/ollama_serve.log"
        exit 1
    fi
    sleep 2
done

apptainer run --nv \
    --bind "${SLURM_SUBMIT_DIR}:/opt/uav_localization:ro" \
    --bind "$DATAPOOL3/datasets/Visloc:/opt/uav_localization/UAV_VisLoc_dataset:ro" \
    --env OLLAMA_HOST=$OLLAMA_HOST \
    --pwd /opt/uav_localization \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/analyze/caption_prompt_probe.py \
        "$@"
RC=$?

cp "${LOCAL_JOB_DIR}/ollama_serve.log" \
   "${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_ID}_ollama_serve.log" 2>/dev/null

exit $RC
