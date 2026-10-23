#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# Training of the LoRA layers on the LaMAR HGE data of prepare_lamar.py, logged to wandb.
# Called by lamar_hf.sh / lamar_hg.sh with DATA and RTYPE; can be submitted alone: qsub -v DATA=<data> shells/train/train.sh
# Hyperparameters of the Structured3D training of SceneAligner (--epochs 20 --min_points 1000, one resolution: that of
# the floorplan, 1024 here). Resubmit the same command to resume after the last completed epoch (RUN/state.pt).
# One process per GPU of the node (torchrun, DistributedDataParallel; rt_HF: 8, rt_HG: 1), a batch of 4 per GPU.

source /etc/profile.d/modules.sh
export PATH=${HOME}/miniconda3/bin:$PATH && source activate scenealigner
export PATH="${CONDA_PREFIX}/bin:$PATH"  # activate is a no-op when the caller already activated the env: keep it before base
export OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
REPO="${REPO:-${HOME}/oss/SceneAligner}"  # repository (PBS runs a copy of this script); override with qsub -v REPO=<dir>
cd "$REPO"
set -euo pipefail

DATA="${DATA:-data/lamar_hge}"
RTYPE="${RTYPE:-rt_HG}"
RUN="${RUN:-outputs/lamar_$(basename "$DATA")_${RTYPE}}"
EPOCHS="${EPOCHS:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-scenealigner-lamar}"
NEGATIVES="${NEGATIVES:-batch}"  # the other windows of the batch are negatives (all windows share the floorplan)
PLAN_GRID="${PLAN_GRID:-8}"      # one correspondence per 8 px (1.4 m) of the 1024 px floorplan
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
VAL_DATA="${VAL_DATA:-$DATA/val}"  # val windows (prepare.sh), evaluated after every epoch if they exist
val_args=()
if [[ -f "$VAL_DATA/lamar_mapanything.csv" ]]; then val_args=(--val_data "$VAL_DATA"); fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py --data "$DATA" --output "$RUN" --epochs "$EPOCHS" \
    --resolutions 1024 --min_points 1000 \
    --negatives "$NEGATIVES" --plan_grid "$PLAN_GRID" "${val_args[@]}" \
    --wandb_project "$WANDB_PROJECT" --wandb_name "$(basename "$RUN")"
