#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# Evaluation on LaMAR HGE (Z-FLoc protocol): the val session (L = 20, seed 42), then the test sessions (all L, seeds 42-46).
# Called by evaluate_hf.sh / evaluate_hg.sh with DATA, RTYPE and CHECKPOINT (a run directory, "pre-trained" or
# jhcho99/SceneAligner). ZFLOC_NPZ (optional): Z-FLoc assets/metadata/hge.npz, to evaluate exactly its test frames.

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
RTYPE="${RTYPE:-rt_HG}"
CHECKPOINT="${CHECKPOINT:-outputs/lamar_$(basename "$DATA")_${RTYPE}}"
PLAN_SIZE="${PLAN_SIZE:-1024}"
OUT="${OUT:-outputs/eval_$(basename "$CHECKPOINT")}"
ZFLOC_NPZ_ARG=()
if [ -n "${ZFLOC_NPZ:-}" ]; then
    ZFLOC_NPZ_ARG=(--zfloc_npz "$ZFLOC_NPZ")
fi

python -c "import sys; print('python:', sys.executable, sys.version.split()[0])"
(cd / && python -c "import mapanything.models") || { echo "MapAnything is not installed in $(which python): see train/lamar/README.md"; exit 1; }

python train/lamar/evaluate_lamar.py --map_dir "$MAP_DIR" --floorplan "$FLOORPLAN" --checkpoint_dir "$CHECKPOINT" \
    --plan_size "$PLAN_SIZE" --sessions val --seq_length 20 --seeds 42 --recon_cache "$DATA/recon_cache_eval" \
    --output "$OUT/val_results.json"
python train/lamar/evaluate_lamar.py --map_dir "$MAP_DIR" --floorplan "$FLOORPLAN" --checkpoint_dir "$CHECKPOINT" \
    --plan_size "$PLAN_SIZE" --recon_cache "$DATA/recon_cache_eval" "${ZFLOC_NPZ_ARG[@]}" --save_vis \
    --output "$OUT/test_results.json"
