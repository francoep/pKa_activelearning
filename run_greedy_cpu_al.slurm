#!/bin/bash
#SBATCH -J Greedy_submission_monitor
#SBATCH -t 672:00:00
#SBATCH -N 1
#SBATCH -p <my_CPU partition>
#SBATCH --ntasks=1


echo Running on `hostname`
echo ld_library_path $LD_LIBRARY_PATH
echo pythonpath $PYTHONPATH
echo path $PATH
echo pwd `pwd`

#the series of job commands to execute
a=`sed -n "${SLURM_ARRAY_TASK_ID}p" generate_figure4_greedy_data.cmds`

#a=`sed -n "${SLURM_ARRAY_TASK_ID}p" generate_figure4a_greedy_data.cmds`

#a=`sed -n "${SLURM_ARRAY_TASK_ID}p" generate_figure5_greedy_data.cmds`
echo $a
eval $a

exit
