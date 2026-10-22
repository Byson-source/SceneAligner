#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# SceneAligner training on LaMAR HGE (rt_HG). Submit from the repository root: qsub shells/train/lamar_hg.sh
REPO="${REPO:-${PBS_O_WORKDIR:-$(pwd)}}"
DATA="${DATA:-data/lamar_hge}" RTYPE=rt_HG bash "$REPO/shells/train/train.sh"
