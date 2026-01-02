#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=cas_{num}
#SBATCH --error=job.error
#SBATCH --output=job.output

module load triton/2025.1-gcc
module load openblas/0.3.28
module load openmpi/5.0.3
srun turbogap md

