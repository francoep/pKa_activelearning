#!/bin/bash
#SBATCH -J greedy AL
#SBATCH -t 672:00:00
#SBATCH -N 1
#SBATCH -p <myGPU_partition>
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

echo Running on `hostname`
echo ld_library_path $LD_LIBRARY_PATH
echo pythonpath $PYTHONPATH
echo path $PATH
echo pwd `pwd`

#Be sure that the following are available on your system:
# Python3
# Cuda 10.1+
# PyTorch compiled with the corresponding Cuda version

#Running the jobs
a=`sed -n "${SLURM_ARRAY_TASK_ID}p" greedy_sel_al.cmds`

echo $a
eval $a

exit
