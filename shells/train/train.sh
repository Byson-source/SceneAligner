#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# Training of the LoRA layers on the LaMAR HGE data of prepare_lamar.py, logged to wandb. Epoch e takes the windows
# of $DATA/epoch_<e> (other windows at every epoch; in turn if prepare.sh made fewer sets than EPOCHS).
# Called by lamar_hf.sh / lamar_hg.sh with DATA and RTYPE; can be submitted alone: qsub -v RTYPE=rt_HG,DATA=<data> shells/train/train.sh
# Hyperparameters of the Structured3D training of SceneAligner (--epochs 20 --min_points 1000, one resolution: that of
# the floorplan, 1024 here). Every job writes a new run (outputs/<...>_<date>_<time>);
# qsub -v RTYPE=rt_HG,RESUME=<run> continues <run> after its last completed epoch (<run>/state.pt).
# One process per GPU of the node (torchrun, DistributedDataParallel; rt_HF: 8, rt_HG: 1), a batch of 4 per GPU.
# Fine-tuning of trained LoRA layers instead of new ones: qsub -v RTYPE=rt_HF,INIT=jhcho99/SceneAligner,LR=1e-5,EPOCHS=5 shells/train/lamar_hf.sh

source /etc/profile.d/modules.sh
export PATH=${HOME}/miniconda3/bin:$PATH && source activate scenealigner
export PATH="${CONDA_PREFIX}/bin:$PATH"  # activate is a no-op when the caller already activated the env: keep it before base
export OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1
REPO="${REPO:-${HOME}/oss/SceneAligner}"  # repository (PBS runs a copy of this script); override with qsub -v RTYPE=rt_HG,REPO=<dir>
cd "$REPO"
set -euo pipefail

DATA="${DATA:-/groups/gai51740/RBT3/umemura/zfloc/dataset/lamar}"
RTYPE="${RTYPE:-rt_HG}"
RUN="${RUN:-${RESUME:-outputs/lamar_$(basename "$DATA")_${RTYPE}_$(date +%Y%m%d_%H%M%S)}}"  # a new run per job
resume_args=()
if [[ -n "${RESUME:-}" ]]; then resume_args=(--resume); fi  # qsub -v RTYPE=rt_HG,RESUME=<run>: continues that run
EPOCHS="${EPOCHS:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-scenealigner-lamar}"
NEGATIVES="${NEGATIVES:-batch}"  # the other windows of the batch are negatives (all windows share the floorplan)
PLAN_GRID="${PLAN_GRID:-8}"      # one correspondence per 8 px (1.4 m) of the 1024 px floorplan
FALSE_NEGATIVE_PX="${FALSE_NEGATIVE_PX:-8}"  # points within 8 px (1.4 m) on the floorplan are not negatives (0: off)
FALSE_NEGATIVES="${FALSE_NEGATIVES:-soft}"   # but soft positives (Gaussian, sigma 4 px), or "ignore"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
VAL_DATA="${VAL_DATA:-$DATA/test}"  # validation windows (prepare.sh; for now the test split, $DATA/val for the UnLoc val), evaluated after every epoch if they exist
LR="${LR:-1e-4}"
init_args=()
if [[ -n "${INIT:-}" ]]; then init_args=(--init "$INIT"); fi  # LoRA layers to start from (default: new ones)
val_args=()
if compgen -G "$VAL_DATA/epoch_*/lamar_mapanything.csv" > /dev/null; then val_args=(--val_data "$VAL_DATA"); fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py --data "$DATA" --output "$RUN" --epochs "$EPOCHS" --lr "$LR" \
    --resolutions 1024 --min_points 1000 \
    --negatives "$NEGATIVES" --plan_grid "$PLAN_GRID" --false_negative_px "$FALSE_NEGATIVE_PX" --false_negatives "$FALSE_NEGATIVES" "${init_args[@]}" "${val_args[@]}" "${resume_args[@]}" \
    --wandb_project "$WANDB_PROJECT" --wandb_name "$(basename "$RUN")"
