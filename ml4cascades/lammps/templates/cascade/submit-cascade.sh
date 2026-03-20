#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --job-name=cas_{num}
#SBATCH --error=job.error
#SBATCH --output=job.output

module load triton/2025.1-gcc
module load openmpi fftw openblas eigen ffmpeg zstd
export LD_LIBRARY_PATH=/scratch/work/jinr1/.conda_envs/torch-env/lib:$LD_LIBRARY_PATH
srun /scratch/phys/t30429_nume-dft-ml/04-Ruoyan/CODE/lammps-eph/lammps-stable_23Jun2022_update4/src/lmp_mpi -in input.lmp

