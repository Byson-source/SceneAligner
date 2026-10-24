#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# LaMAR HGE training data: data check, MapAnything on the frames of the train and val sessions (cached), then the
# density maps of random windows assembled from the cached points: other windows at every epoch of train.py
# ($DATA/epoch_<e>), and one set of val windows ($DATA/val/epoch_000, validation of train.py).
# Called by prepare_hf.sh / prepare_hg.sh with DATA and RTYPE. NUM_GPUS: MapAnything processes (default: the GPUs
# of the node); NUM_WORKERS: CPU processes of the windows. e.g. qsub -v EPOCHS=40,WINDOWS=100 shells/lamar/prepare_hf.sh

source /etc/profile.d/modules.sh
export PATH=${HOME}/miniconda3/bin:$PATH && source activate scenealigner
export PATH="${CONDA_PREFIX}/bin:$PATH"  # activate is a no-op when the caller already activated the env: keep it before base
export OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
REPO="${REPO:-${HOME}/oss/SceneAligner}"  # repository (PBS runs a copy of this script); override with qsub -v REPO=<dir>
cd "$REPO"
set -euo pipefail

LAMAR_ROOT="${LAMAR_ROOT:-/groups/gai51740/RBT3/umemura/zfloc/lamar}"
MAP_DIR="${MAP_DIR:-$LAMAR_ROOT/HGE/sessions/map}"
FLOORPLAN="${FLOORPLAN:-$LAMAR_ROOT/hge.png}"
DATA="${DATA:-/groups/gai51740/RBT3/umemura/zfloc/dataset/lamar}"
PLAN_SIZE="${PLAN_SIZE:-1024}"


python -c "import sys; print('python:', sys.executable, sys.version.split()[0])"
(cd / && python -c "import mapanything.models") || { echo "MapAnything is not installed in $(which python): see train/lamar/README.md"; exit 1; }

python train/lamar/lamar.py --check --map_dir "$MAP_DIR" --raw_dir "$LAMAR_ROOT" --floorplan "$FLOORPLAN" --plan_size "$PLAN_SIZE"

SPLITS="${SPLITS:-train val}"  # train -> $DATA, val -> $DATA/val (validation of train.py); e.g. qsub -v SPLITS=val
EPOCHS="${EPOCHS:-20}"         # sets of train windows, one per epoch of train.py ($DATA/epoch_<e>); val: one set
WINDOWS="${WINDOWS:-50}"       # random windows per session and epoch, every one a different combination of frames
CACHE="${CACHE:-$DATA/session_cache}"  # MapAnything on chunks of the sessions, shared by train and val
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MEM_WORKERS=$(( $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) / 8 ))  # up to 8 GB per window
NUM_WORKERS="${NUM_WORKERS:-$(( $(nproc) < MEM_WORKERS ? $(nproc) : MEM_WORKERS > 1 ? MEM_WORKERS : 1 ))}"  # CPU processes of the samples
export PYTHONUNBUFFERED=1  # the messages of the shards reach the terminal (job output) at once
echo "$NUM_GPUS GPUs, $NUM_WORKERS workers"
common=(--map_dir "$MAP_DIR" --raw_dir "$LAMAR_ROOT" --floorplan "$FLOORPLAN" --plan_size "$PLAN_SIZE" --session_cache "$CACHE")
run_shards() {  # run_shards <processes> <gpu: 1|0 (no GPU)> <args...>: one process per shard (progress bar of shard 0), then wait
    local n="$1" gpu="$2" pids=()
    shift 2
    for ((i = 0; i < n; i++)); do
        CUDA_VISIBLE_DEVICES=$( ((gpu)) && echo "$i") python train/lamar/prepare_lamar.py "$@" --shard "$i" --num_shards "$n" &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "shard $i failed (its traceback is above)"; exit 1; }
    done
}

# 1. MapAnything on chunks of consecutive frames of the sessions, one process per GPU (cached chunks are skipped)
run_shards "$NUM_GPUS" 1 --stage recon "${common[@]}" --sessions $SPLITS
# 2. The windows of every epoch from the cached points (CPU). The epochs finished by a shard are skipped on a rerun:
#    delete $DATA/epoch_* and $DATA/val/epoch_* after changing EPOCHS, WINDOWS or NUM_WORKERS
for split in $SPLITS; do
    output="$DATA$([[ $split == train ]] || echo "/$split")"
    epochs=$([[ $split == train ]] && echo "$EPOCHS" || echo 1)
    run_shards "$NUM_WORKERS" 0 --stage samples "${common[@]}" --sessions "$split" \
        --output "$output" --epochs "$epochs" --windows_per_session "$WINDOWS"
    python train/lamar/prepare_lamar.py --merge --output "$output"
done
