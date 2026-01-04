#!/bin/bash

#SBATCH --time=100:00:00
#SBATCH --partition=batch
##SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=60G
#SBATCH --job-name={file}


module load openmpi fftw openblas eigen ffmpeg zstd
#export LD_LIBRARY_PATH=/scratch/work/jinr1/.conda_envs/torch-env/lib:$LD_LIBRARY_PATH
srun lmp -in {file}
