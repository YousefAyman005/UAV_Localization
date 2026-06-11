#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youssef.elsayed@hhi.fraunhofer.de
#SBATCH --job-name=uav-caption-ollama
#SBATCH --output=logs/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G

# Captioning with the SAME model as the Mac workflow (qwen3.5:9b via Ollama),
# fully offline: an Ollama server runs inside the container on the job GPU and
# caption_crops.py talks to it over localhost (its requests-only REST fallback,
# so the container needs no `ollama` package).
# Prereqs (staged once from the headnode, no rebuild of the .sif):
#   $DATAPOOL3/datasets/Visloc/weights/ollama/runtime  (bin/ + lib/, v0.30.7)
#   $DATAPOOL3/datasets/Visloc/weights/ollama/models   (qwen3.5:9b blobs)
# Args: TARGET (sat|drone|tile) then extra flags, e.g. the missing train band:
#   sbatch slurm/run_caption_ollama.sh drone --band train --flights all
# Captions append to the resumable per-flight JSONLs on the captions datapool
# bind, so a requeued/repeated job continues where the last one stopped.

TARGET=${1:-drone}; shift || true

source "/etc/slurm/local_job_dir.sh"
echo "$PWD/stats/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg

OLLAMA_DIR="$DATAPOOL3/datasets/Visloc/weights/ollama"
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p "${LOCAL_JOB_DIR}/ollama_home"

# Ollama server inside the container (Ubuntu 22.04 userland; --nv exposes the
# job GPU). HOME points at node-local scratch: serve only writes its keypair
# and temp files there, never to /home or the read-only model bind.
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
    --bind "$DATAPOOL3/datasets/Visloc/cache/captions:/opt/uav_localization/cache/captions" \
    --env OLLAMA_HOST=$OLLAMA_HOST \
    --pwd /opt/uav_localization \
    "${SLURM_SUBMIT_DIR}/uav_localization.sif" \
    /opt/uav_localization/pipelines/caption_crops.py \
        --target "${TARGET}" \
        "$@"
RC=$?

# keep the server log for debugging; captions already live on the datapool
# bind, so there is no job_results copy-back step for this job
cp "${LOCAL_JOB_DIR}/ollama_serve.log" \
   "${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_ID}_ollama_serve.log" 2>/dev/null

exit $RC
