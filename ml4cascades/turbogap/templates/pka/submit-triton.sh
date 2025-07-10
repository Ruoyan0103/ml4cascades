#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --partition=batch
##SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=50G
#SBATCH --job-name={job_name}

module load gcc/11.4.0
module load openblas/0.3.24
module load openmpi/4.1.6
srun turbogap md

