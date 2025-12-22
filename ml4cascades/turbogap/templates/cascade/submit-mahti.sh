#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=project_2012355
#SBATCH --time=24:00:00
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12

module load gcc/11.2.0
module load openblas/0.3.18-omp
module load openmpi/4.1.2
srun turbogap md
