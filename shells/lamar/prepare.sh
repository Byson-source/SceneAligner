#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# LaMAR HGE training data: data check, then MapAnything reconstructions and density maps of the train windows.
# Called by prepare_hf.sh / prepare_hg.sh with DATA and RTYPE. NUM_GPUS: processes (default: the GPUs of the node).

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
DATA="${DATA:-data/lamar_hge}"
PLAN_SIZE="${PLAN_SIZE:-1024}"


python -c "import sys; print('python:', sys.executable, sys.version.split()[0])"
(cd / && python -c "import mapanything.models") || { echo "MapAnything is not installed in $(which python): see train/lamar/README.md"; exit 1; }

python train/lamar/lamar.py --check --map_dir "$MAP_DIR" --raw_dir "$LAMAR_ROOT" --floorplan "$FLOORPLAN" --plan_size "$PLAN_SIZE"

# One process per GPU (rt_HF: all GPUs of the node, rt_HG: one), each with a share of the windows; logs in $DATA/shards
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
mkdir -p "$DATA/shards" && rm -f "$DATA"/shards/lamar_*.csv
echo "$NUM_GPUS GPUs"
pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python train/lamar/prepare_lamar.py --map_dir "$MAP_DIR" --raw_dir "$LAMAR_ROOT" \
        --floorplan "$FLOORPLAN" --output "$DATA" --plan_size "$PLAN_SIZE" --recon_cache "$DATA/recon_cache" \
        --shard "$i" --num_shards "$NUM_GPUS" > "$DATA/shards/prepare_$i.log" 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "shard $i failed: see $DATA/shards/prepare_$i.log"; exit 1; }
done
python train/lamar/prepare_lamar.py --merge --output "$DATA"
