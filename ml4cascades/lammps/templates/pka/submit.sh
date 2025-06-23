#!/bin/bash

#SBATCH --time=00:03:00
#SBATCH --partition=batch
##SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=5G
#SBATCH --job-name={file}


module load gcc openmpi fftw openblas eigen ffmpeg zstd
srun lmp -in {file}
