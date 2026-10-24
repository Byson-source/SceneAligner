#!/bin/bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe
#PBS -k oed
# SceneAligner training on LaMAR HGE (rt_HG). Submit from any directory: qsub shells/train/lamar_hg.sh
REPO="${REPO:-${HOME}/oss/SceneAligner}"  # repository (PBS runs a copy of this script); override with qsub -v REPO=<dir>
REPO="$REPO" DATA="${DATA:-/groups/gai51740/RBT3/umemura/zfloc/dataset/lamar}" RTYPE=rt_HG bash "$REPO/shells/train/train.sh"
