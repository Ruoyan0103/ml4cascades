#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --partition=batch
##SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=thermo

module load gcc/11.4.0
module load openblas/0.3.24
module load openmpi/4.1.6
srun turbogap md

