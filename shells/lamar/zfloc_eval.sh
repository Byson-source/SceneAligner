#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# SceneAligner on the inputs of the Z-FLoc LaMAR HGE experiment, from the point clouds to the poses:
#   1. Z-FLoc tools/lamar_setup/create_lamar_hge.py: kept frames, poses_world.npy, poses_map.npy ($ZFLOC_DATA)
#   2. Z-FLoc tools/metadata_generation/generate_pcd.py: MapAnything point clouds of the windows of every length
#      ($PCD_ROOT/hge_<L>; one GPU per length, existing windows are skipped)
#   3. train/lamar/evaluate_zfloc.py: SceneAligner + RANSAC restricted to the scale band of Z-FLoc, metric of Z-FLoc
# Called by zfloc_eval_hf.sh / zfloc_eval_hg.sh with DATA, RTYPE and CHECKPOINT (a run directory trained on HGE,
# "pre-trained" or jhcho99/SceneAligner). SCALE_TOL: half width of the scale band (default 0.35 of Z-FLoc, -1: unrestricted).

source /etc/profile.d/modules.sh
export PATH=${HOME}/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
REPO="${REPO:-${HOME}/oss/SceneAligner}"  # repository (PBS runs a copy of this script); override with qsub -v RTYPE=rt_HG,REPO=<dir>
ZFLOC_REPO="${ZFLOC_REPO:-${HOME}/oss/z-floc}"
ZFLOC_ENV="${ZFLOC_ENV:-mapanything}"  # conda environment of Z-FLoc (MapAnything, Mask2Former)
set -euo pipefail

LAMAR_ROOT="${LAMAR_ROOT:-/groups/gai51740/RBT3/umemura/zfloc/lamar}"
MAP_DIR="${MAP_DIR:-$LAMAR_ROOT/HGE/sessions/map}"
FLOORPLAN="${FLOORPLAN:-$LAMAR_ROOT/hge.png}"
DATA="${DATA:-/groups/gai51740/RBT3/umemura/zfloc/dataset/lamar}"
RTYPE="${RTYPE:-rt_HG}"
CHECKPOINT="${CHECKPOINT:-outputs/lamar_$(basename "$DATA")_${RTYPE}}"
PLAN_SIZE="${PLAN_SIZE:-1024}"
ZFLOC_DATA="${ZFLOC_DATA:-$DATA/zfloc_hge}"  # output of create_lamar_hge.py
PCD_ROOT="${PCD_ROOT:-$DATA/zfloc_pcd}"      # point clouds of generate_pcd.py, hge_<L>/<scene>/pcd/<stem>.npz
LENGTHS="${LENGTHS:-15 20 35 50 100}"
SEEDS="${SEEDS:-42 43 44 45 46}"
SCALE_TOL="${SCALE_TOL:-0.35}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
OUT="${OUT:-$REPO/outputs/zfloc_eval_$(basename "$CHECKPOINT")_tol${SCALE_TOL}}"
mkdir -p "$OUT" "$PCD_ROOT"
echo "$NUM_GPUS GPUs"

# 1-2. Point clouds of Z-FLoc
source activate "$ZFLOC_ENV"
export PATH="${CONDA_PREFIX}/bin:$PATH"
cd "$ZFLOC_REPO"
if [ ! -f "$ZFLOC_DATA/split.yaml" ]; then
    python tools/lamar_setup/create_lamar_hge.py --session-dir "$MAP_DIR" --target-dir "$ZFLOC_DATA"
fi
read -ra lengths <<< "$LENGTHS"
pids=()
for ((i = 0; i < NUM_GPUS && i < ${#lengths[@]}; i++)); do
    (  # lengths i, i + NUM_GPUS, ... on GPU i
        for ((j = i; j < ${#lengths[@]}; j += NUM_GPUS)); do
            L=${lengths[$j]}
            CUDA_VISIBLE_DEVICES=$i python tools/metadata_generation/generate_pcd.py --dataset hge --mode hge_complete \
                --dataset_path "$ZFLOC_DATA" --evol_path "$PCD_ROOT/hge_$L" --traj_len "$L" > "$PCD_ROOT/hge_$L.log" 2>&1
        done
    ) &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "generate_pcd on GPU $i failed: see $PCD_ROOT/hge_<L>.log"; exit 1; }
done

# 3. SceneAligner: one process per GPU with a share of the windows, then the merge
source activate scenealigner
export PATH="${CONDA_PREFIX}/bin:$PATH"
cd "$REPO"
python -c "import sys; print('python:', sys.executable, sys.version.split()[0])"
output="$OUT/test_results.json"
args=(--seq_length $LENGTHS --seeds $SEEDS)
pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python train/lamar/evaluate_zfloc.py --pcd_root "$PCD_ROOT" --zfloc_data "$ZFLOC_DATA" \
        --map_dir "$MAP_DIR" --floorplan "$FLOORPLAN" --checkpoint_dir "$CHECKPOINT" --plan_size "$PLAN_SIZE" \
        --scale_tol "$SCALE_TOL" --save_vis --output "$output" --shard "$i" --num_shards "$NUM_GPUS" "${args[@]}" \
        > "${output%.json}.shard$i.log" 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "shard $i failed: see ${output%.json}.shard$i.log"; exit 1; }
done
if ((NUM_GPUS > 1)); then  # one process writes the results itself
    python train/lamar/evaluate_zfloc.py --merge --output "$output" --num_shards "$NUM_GPUS" "${args[@]}"
fi
