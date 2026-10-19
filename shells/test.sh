#!/bin/bash

#PBS -P gai51740
#PBS -q R9920261300
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -v RTYPE=rt_HG
#PBS -m abe



{
    source /etc/profile.d/modules.sh
    source ${HOME}/.bash_profile

    export PATH=${HOME}/miniconda3/bin:$PATH
    source activate scenealigner



    export OMP_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    cd ${HOME}/oss/SceneAligner

    python evaluate_s3d.py --s3d_root ${HOME}/oss/SceneAligner/data/Structured3D


    exit 0
}
