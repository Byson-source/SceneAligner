#!/usr/bin/env bash
# LaMAR HGE: data check, training data, training with wandb, model selection on the val session, test evaluation.
#
#   LAMAR_ROOT=<lamar> bash train/lamar/run.sh [DATA] [RUN]
#
# LAMAR_ROOT holds HGE/sessions/map (benchmark release), the raw sessions ios_<date>_<time>/ (download_lamar.sh)
# and hge.png. ZFLOC_NPZ (optional) is Z-FLoc's assets/metadata/hge.npz, which fixes the evaluated frames of
# ios_2022-01-18_17.10.39_000 to the ones Z-FLoc evaluates.
set -euo pipefail

cd "$(dirname "$0")/../.."
LAMAR_ROOT="${LAMAR_ROOT:-/groups/gai51740/RBT3/umemura/zfloc/lamar}"
MAP_DIR="${MAP_DIR:-$LAMAR_ROOT/HGE/sessions/map}"
FLOORPLAN="${FLOORPLAN:-$LAMAR_ROOT/hge.png}"
DATA="${1:-${DATA:-data/lamar_hge}}"
RUN="${2:-${RUN:-outputs/lamar_run}}"
PLAN_SIZE="${PLAN_SIZE:-1024}"
EPOCHS="${EPOCHS:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-scenealigner-lamar}"
ZFLOC_NPZ_ARG=()
if [ -n "${ZFLOC_NPZ:-}" ]; then
    ZFLOC_NPZ_ARG=(--zfloc_npz "$ZFLOC_NPZ")
fi

python train/lamar/lamar.py --check --map_dir "$MAP_DIR" --raw_dir "$LAMAR_ROOT" --floorplan "$FLOORPLAN" --plan_size "$PLAN_SIZE" "${ZFLOC_NPZ_ARG[@]}"

python train/lamar/prepare_lamar.py --map_dir "$MAP_DIR" --raw_dir "$LAMAR_ROOT" --floorplan "$FLOORPLAN" --output "$DATA" \
    --plan_size "$PLAN_SIZE" --recon_cache "$DATA/recon_cache"

python train.py --data "$DATA" --output "$RUN" --epochs "$EPOCHS" --resolutions 1024 1280 --min_points 1000 \
    --wandb_project "$WANDB_PROJECT" --wandb_name "$(basename "$RUN")"

python train/lamar/evaluate_lamar.py --map_dir "$MAP_DIR" --floorplan "$FLOORPLAN" --checkpoint_dir "$RUN" \
    --plan_size "$PLAN_SIZE" --sessions val --seq_length 20 --seeds 42 --recon_cache "$DATA/recon_cache_eval" \
    --output "$RUN/val_results.json"

python train/lamar/evaluate_lamar.py --map_dir "$MAP_DIR" --floorplan "$FLOORPLAN" --checkpoint_dir "$RUN" \
    --plan_size "$PLAN_SIZE" --recon_cache "$DATA/recon_cache_eval" "${ZFLOC_NPZ_ARG[@]}" --save_vis \
    --output "$RUN/test_results.json"
