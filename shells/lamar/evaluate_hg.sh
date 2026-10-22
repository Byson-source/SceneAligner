#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# Evaluation on LaMAR HGE (rt_HG). Submit from the repository root: qsub shells/lamar/evaluate_hg.sh
REPO="${REPO:-${PBS_O_WORKDIR:-$(pwd)}}"
DATA="${DATA:-data/lamar_hge}" RTYPE=rt_HG bash "$REPO/shells/lamar/evaluate.sh"
