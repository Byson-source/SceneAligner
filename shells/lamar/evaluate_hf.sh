#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -v RTYPE=rt_HF
#PBS -m abe
#PBS -k oed
# Evaluation on LaMAR HGE (rt_HF). Submit from any directory: qsub shells/lamar/evaluate_hf.sh
REPO="${REPO:-${HOME}/oss/SceneAligner}"  # repository (PBS runs a copy of this script); override with qsub -v REPO=<dir>
REPO="$REPO" DATA="${DATA:-data/lamar_hge}" RTYPE=rt_HF bash "$REPO/shells/lamar/evaluate.sh"
