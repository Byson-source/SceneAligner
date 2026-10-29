#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -v RTYPE=rt_HF
#PBS -m abe
#PBS -k oed
# Submits the whole pipeline on rt_HF, each job after the previous ones succeeded (run on the login node with bash,
# not qsub): MapAnything (1 node, prepare_hf.sh STAGE=recon) -> the windows in NUM_PARTS jobs of 1 node sharing the
# epochs (STAGE=samples) -> training on TRAIN_NODES nodes (a batch of 4 x 8 x nodes).
#   bash shells/lamar/prepare_train_hf.sh
#   NUM_PARTS=4 TRAIN_NODES=4 TRAIN_EPOCHS=100 bash shells/lamar/prepare_train_hf.sh
# Delete $DATA/epoch_* and $DATA/<split>/epoch_* first when the windows change (the finished epochs are skipped).
set -euo pipefail
REPO="${REPO:-${HOME}/oss/SceneAligner}"
cd "$REPO"
NUM_PARTS="${NUM_PARTS:-4}"
TRAIN_NODES="${TRAIN_NODES:-4}"
train_vars="RTYPE=rt_HF${TRAIN_EPOCHS:+,EPOCHS=$TRAIN_EPOCHS}"

recon=$(qsub -v RTYPE=rt_HF,STAGE=recon shells/lamar/prepare_hf.sh)
echo "recon: $recon"
samples=""
for (( part = 0; part < NUM_PARTS; part++ )); do
    job=$(qsub -W depend=afterok:"$recon" -v RTYPE=rt_HF,STAGE=samples,PART=$part,NUM_PARTS=$NUM_PARTS shells/lamar/prepare_hf.sh)
    echo "samples $part/$NUM_PARTS: $job"
    samples+=":$job"
done
train=$(qsub -W depend=afterok"$samples" -l select="$TRAIN_NODES" -v "$train_vars" shells/train/lamar_hf.sh)
echo "train ($TRAIN_NODES nodes): $train"
