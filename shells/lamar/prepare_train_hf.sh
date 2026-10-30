#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -v RTYPE=rt_HF
#PBS -k oed
# Submits the whole pipeline on rt_HF, each job after the previous ones succeeded (run on the login node with bash,
# not qsub): MapAnything (1 node, prepare_hf.sh STAGE=recon) -> the windows in NUM_PARTS jobs of 1 node sharing the
# epochs (STAGE=samples) -> one training job on TRAIN_NODES nodes (a batch of 4 x 8 x nodes). After its walltime, run
# this again with PREPARE=0 RESUME=<run> (it continues from <run>/state.pt).
#   bash shells/lamar/prepare_train_hf.sh
#   NUM_PARTS=4 TRAIN_NODES=4 TRAIN_EPOCHS=500 bash shells/lamar/prepare_train_hf.sh
#   PREPARE=0 RESUME=outputs/<run> TRAIN_EPOCHS=500 bash shells/lamar/prepare_train_hf.sh  # an existing run
#   (TRAIN_LR=3e-4: the learning rate of the training job, also of a resumed run)
# Delete $DATA/epoch_* and $DATA/<split>/epoch_* first when the windows change (the finished epochs are skipped).
set -euo pipefail
if [[ -n ${PBS_JOBID:-} ]]; then echo "run on the login node with bash (it submits the jobs), not with qsub"; exit 1; fi
REPO="${REPO:-${HOME}/oss/SceneAligner}"
cd "$REPO"
PREPARE="${PREPARE:-1}"  # 0: only the training jobs
NUM_PARTS="${NUM_PARTS:-4}"
TRAIN_NODES="${TRAIN_NODES:-4}"
RUN="${RESUME:-outputs/lamar_$(basename "${DATA:-lamar}")_rt_HFx${TRAIN_NODES}_$(date +%Y%m%d_%H%M%S)}"
train_vars="RTYPE=rt_HF,RUN=$RUN${RESUME:+,RESUME=$RESUME}${TRAIN_EPOCHS:+,EPOCHS=$TRAIN_EPOCHS}${TRAIN_LR:+,LR=$TRAIN_LR}${DATA:+,DATA=$DATA}"

depend=()
if (( PREPARE )); then
    recon=$(qsub -v RTYPE=rt_HF,STAGE=recon shells/lamar/prepare_hf.sh)
    echo "recon: $recon"
    samples=""
    for (( part = 0; part < NUM_PARTS; part++ )); do
        job=$(qsub -W depend=afterok:"$recon" -v RTYPE=rt_HF,STAGE=samples,PART=$part,NUM_PARTS=$NUM_PARTS shells/lamar/prepare_hf.sh)
        echo "samples $part/$NUM_PARTS: $job"
        samples+=":$job"
    done
    depend=(-W depend=afterok"$samples")
fi
train=$(qsub "${depend[@]}" -l select="$TRAIN_NODES" -v "$train_vars" shells/train/lamar_hf.sh)
echo "train ($TRAIN_NODES nodes${RESUME:+, resume}): $train"
echo "run: $RUN"
