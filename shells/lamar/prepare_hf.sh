#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HF
#PBS -m abe
#PBS -k oed
# LaMAR HGE training data (rt_HF). Submit from the repository root: qsub shells/lamar/prepare_hf.sh
REPO="${REPO:-${PBS_O_WORKDIR:-$(pwd)}}"
DATA="${DATA:-data/lamar_hge}" RTYPE=rt_HF bash "$REPO/shells/lamar/prepare.sh"
