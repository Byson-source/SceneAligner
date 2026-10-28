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
#      ($PCD_ROOT/hge_<L>; one length after the other, existing windows are skipped)
#   3. Z-FLoc tools/metadata_generation/extract_primitives.py: wall BEV, wall lines and gravity of every window
#      (SOLVER=zfloc only; existing windows are skipped)
#   4. train/lamar/evaluate_zfloc.py, metric of Z-FLoc. SOLVER=zfloc (default): only the correspondences of SceneAligner,
#      all their pairs as hypotheses for the scoring and LM of Z-FLoc on the BEV of 3 (the density map with its gravity);
#      SOLVER=ransac: RANSAC restricted to the scale band of Z-FLoc, ground-truth gravity
# On one GPU. The progress goes to the job output and to $PCD_ROOT/hge_<L>.log and $OUT/test_results.log.
# Called by zfloc_eval_hg.sh with DATA, RTYPE and CHECKPOINT (a run directory trained on HGE,
# "pre-trained" or jhcho99/SceneAligner). SCALE_TOL (SOLVER=ransac): half width of the scale band (default 0.35 of
# Z-FLoc, -1: unrestricted). SOLVER=zfloc imports Z-FLoc ($ZFLOC_REPO) in the scenealigner environment: numba, scipy, pandas.

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
SOLVER="${SOLVER:-zfloc}"
if [ "$SOLVER" = zfloc ]; then suffix=zfloc; else suffix="ransac_tol${SCALE_TOL}"; fi
OUT="${OUT:-$REPO/outputs/zfloc_eval_$(basename "$CHECKPOINT")_${suffix}}"
mkdir -p "$OUT" "$PCD_ROOT"
export PYTHONUNBUFFERED=1  # prints of python right away in the job output
log() { echo "[$(date '+%F %T')] $*"; }
num_pcds() { find "$PCD_ROOT/hge_$1" -name '*.npz' 2>/dev/null | wc -l; }
log "checkpoint $CHECKPOINT, solver $SOLVER, scale_tol $SCALE_TOL, lengths $LENGTHS, seeds $SEEDS"
log "data $ZFLOC_DATA, point clouds $PCD_ROOT, results $OUT"

# 1-2. Point clouds of Z-FLoc
source activate "$ZFLOC_ENV"
export PATH="${CONDA_PREFIX}/bin:$PATH"
cd "$ZFLOC_REPO"
if [ -f "$ZFLOC_DATA/split.yaml" ]; then
    log "1/4 create_lamar_hge.py: skipped ($ZFLOC_DATA/split.yaml exists)"
else
    log "1/4 create_lamar_hge.py -> $ZFLOC_DATA"
    python tools/lamar_setup/create_lamar_hge.py --session-dir "$MAP_DIR" --target-dir "$ZFLOC_DATA"
fi
for L in $LENGTHS; do
    log "2/4 generate_pcd.py L=$L: $(num_pcds "$L") point clouds already, log $PCD_ROOT/hge_$L.log"
    python tools/metadata_generation/generate_pcd.py --dataset hge --mode hge_complete --dataset_path "$ZFLOC_DATA" \
        --evol_path "$PCD_ROOT/hge_$L" --traj_len "$L" 2>&1 | tee "$PCD_ROOT/hge_$L.log"
    log "2/4 generate_pcd.py L=$L: done, $(num_pcds "$L") point clouds"
    if [ "$SOLVER" = zfloc ]; then
        log "3/4 extract_primitives.py L=$L, log $PCD_ROOT/hge_${L}_primitives.log"
        python tools/metadata_generation/extract_primitives.py --dataset hge --save_dir "$PCD_ROOT/hge_$L" \
            --seq_length "$L" --output_npz "$PCD_ROOT/hge_$L.npz" 2>&1 | tee "$PCD_ROOT/hge_${L}_primitives.log"
        log "3/4 extract_primitives.py L=$L: done, $(find "$PCD_ROOT/hge_$L" -path '*/bev_img/*.png' | wc -l) BEVs"
    fi
done

# 3. SceneAligner
source activate scenealigner
export PATH="${CONDA_PREFIX}/bin:$PATH"
cd "$REPO"
if [ "$SOLVER" = zfloc ]; then
    python -c "import numba, scipy, pandas" || { echo "SOLVER=zfloc: pip install numba scipy pandas in scenealigner" >&2; exit 1; }
fi
output="$OUT/test_results.json"
log "4/4 evaluate_zfloc.py, log ${output%.json}.log"
python train/lamar/evaluate_zfloc.py --pcd_root "$PCD_ROOT" --zfloc_data "$ZFLOC_DATA" --map_dir "$MAP_DIR" \
    --floorplan "$FLOORPLAN" --checkpoint_dir "$CHECKPOINT" --plan_size "$PLAN_SIZE" --scale_tol "$SCALE_TOL" --save_vis \
    --solver "$SOLVER" --zfloc_repo "$ZFLOC_REPO" --output "$output" --seq_length $LENGTHS --seeds $SEEDS 2>&1 | tee "${output%.json}.log"
log "done: $output, density maps and alignments (SOLVER=zfloc: overlays on hge.png) in ${output%.json}_vis"
