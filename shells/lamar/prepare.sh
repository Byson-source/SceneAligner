#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# LaMAR HGE training data: data check, MapAnything on the frames of the train and test sessions (cached), then the
# density maps of random windows assembled from the cached points: other windows at every epoch of train.py
# ($DATA/epoch_<e>), and one set of test windows ($DATA/test/epoch_000, for now the validation of train.py).
# Called by prepare_hf.sh / prepare_hg.sh with DATA and RTYPE. NUM_GPUS: MapAnything processes (default: the GPUs
# of the node); NUM_WORKERS: CPU processes of the windows. e.g. qsub -v RTYPE=rt_HF,EPOCHS=40,WINDOWS=10000 shells/lamar/prepare_hf.sh

source /etc/profile.d/modules.sh
export PATH=${HOME}/miniconda3/bin:$PATH && source activate scenealigner
export PATH="${CONDA_PREFIX}/bin:$PATH"  # activate is a no-op when the caller already activated the env: keep it before base
export OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
REPO="${REPO:-${HOME}/oss/SceneAligner}"  # repository (PBS runs a copy of this script); override with qsub -v RTYPE=rt_HG,REPO=<dir>
cd "$REPO"
set -euo pipefail

LAMAR_ROOT="${LAMAR_ROOT:-/groups/gai51740/RBT3/umemura/zfloc/lamar}"
MAP_DIR="${MAP_DIR:-$LAMAR_ROOT/HGE/sessions/map}"
FLOORPLAN="${FLOORPLAN:-$LAMAR_ROOT/hge.png}"
DATA="${DATA:-/groups/gai51740/RBT3/umemura/zfloc/dataset/lamar}"
PLAN_SIZE="${PLAN_SIZE:-1024}"
PLAN_SOURCE="${PLAN_SOURCE:-lidar}"  # floorplan positions of the correspondences: lidar (as prepare_s3d.py) or recon


python -c "import sys; print('python:', sys.executable, sys.version.split()[0])"
(cd / && python -c "import mapanything.models") || { echo "MapAnything is not installed in $(which python): see train/lamar/README.md"; exit 1; }

python train/lamar/lamar.py --check --map_dir "$MAP_DIR" --raw_dir "$LAMAR_ROOT" --floorplan "$FLOORPLAN" --plan_size "$PLAN_SIZE"

SPLITS="${SPLITS:-train test}"  # train -> $DATA, <split> -> $DATA/<split>; test for now is the validation of train.py (SPLITS="train val" for the UnLoc val)
EPOCHS="${EPOCHS:-20}"         # sets of train windows, one per epoch of train.py ($DATA/epoch_<e>); val: one set
WINDOWS="${WINDOWS:-8864}"     # random windows per epoch (x 2 samples = 554 steps of 32, half the 1108 before), all different combinations of frames
VAL_WINDOWS="${VAL_WINDOWS:-600}"  # val windows (one set)
CACHE="${CACHE:-$DATA/session_cache}"  # MapAnything on chunks of the sessions, shared by the splits
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MEM_WORKERS=$(( $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) / 8 ))  # up to 8 GB per window
NUM_CPUS=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)  # nproc alone gives OMP_NUM_THREADS (1, above)
NUM_WORKERS="${NUM_WORKERS:-$(( NUM_CPUS < MEM_WORKERS ? NUM_CPUS : MEM_WORKERS > 1 ? MEM_WORKERS : 1 ))}"  # CPU processes of the samples
export PYTHONUNBUFFERED=1  # the messages and the progress bars reach the terminal (job output) at once
echo "$NUM_GPUS GPUs, $NUM_CPUS CPUs, $(( MEM_WORKERS * 8 )) GB free: $NUM_WORKERS workers"
common=(--map_dir "$MAP_DIR" --raw_dir "$LAMAR_ROOT" --floorplan "$FLOORPLAN" --plan_size "$PLAN_SIZE" --session_cache "$CACHE")
# 1. MapAnything on chunks of consecutive frames of the sessions, one process per GPU, one progress bar of all the
#    chunks (cached chunks are skipped)
STAGE="${STAGE:-all}"  # all, recon or samples (prepare_train_hf.sh: one recon job, then NUM_PARTS samples jobs)
PART="${PART:-0}" NUM_PARTS="${NUM_PARTS:-1}"  # samples: this job makes the epochs PART, PART + NUM_PARTS, ...
if [[ $STAGE != samples ]]; then
    python train/lamar/prepare_lamar.py --stage recon "${common[@]}" --sessions $SPLITS --workers "$NUM_GPUS"
fi
if [[ $STAGE == recon ]]; then exit 0; fi
# 2. The windows of every epoch from the cached points (CPU processes, one progress bar of all the windows). The
#    finished epochs are skipped on a rerun:
#    delete $DATA/epoch_* and $DATA/<split>/epoch_* after changing EPOCHS, WINDOWS or VAL_WINDOWS
for split in $SPLITS; do
    output="$DATA$([[ $split == train ]] || echo "/$split")"
    epochs=$([[ $split == train ]] && echo "$EPOCHS" || echo 1)
    windows=$([[ $split == train ]] && echo "$WINDOWS" || echo "$VAL_WINDOWS")
    spread=$([[ $split == test ]] && echo 1 || echo 8)  # test: consecutive frames (interval 1); others: n of n to 8 n frames
    python train/lamar/prepare_lamar.py --stage samples "${common[@]}" --sessions "$split" --workers "$NUM_WORKERS" \
        --output "$output" --epochs "$epochs" --windows_per_epoch "$windows" --max_spread "$spread" --plan_source "$PLAN_SOURCE" \
        --part "$PART" --num_parts "$NUM_PARTS"
done
