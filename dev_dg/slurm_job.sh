#!/usr/bin/env bash


#SBATCH -J 'slds_run'
#SBATCH -o slds_singlearea-%j.out
#SBATCH -p Brody
#SBATCH -time=5:00:00
#SBATCH --mail-user=dikshag@princeton.edu
#SBATCH --mail-type=all
#SBATCH --mem= 36000
#SBATCH -c 10 
#SBATCH --array=1-4

module load anacondapy/2020.11
conda activate ssmrun

python fit_slds_singlearea.py $SLURM_ARRAY_TASK_ID