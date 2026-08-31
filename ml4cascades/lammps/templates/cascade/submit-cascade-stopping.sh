#!/bin/bash

##SBATCH --time=36:00:00

##SBATCH --partition=sumo
##SBATCH --account=sumo
##SBATCH --nodes=1
##SBATCH --ntasks=40
##SBATCH --cpus-per-task=1

##SBATCH --partition=batch
##SBATCH --nodes=2
##SBATCH --ntasks=64
##SBATCH --cpus-per-task=1

##SBATCH --mem=300G
##SBATCH --job-name=cas_{num}
##SBATCH --error=job.error
##SBATCH --output=job.output

#SBATCH --time=05:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=10            # 40 per node, fills csl/skl nodes exactly
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G              # 3M atoms ~15-20GB total; 60G/node is safe
#SBATCH --job-name=srim_{num}
#SBATCH --error=job.error
#SBATCH --output=job.output
##SBATCH --constraint=avx512    # targets csl/skl nodes; best LAMMPS vectorization


module load triton/2025.1-gcc
module load openmpi fftw openblas eigen ffmpeg zstd
export LD_LIBRARY_PATH=/scratch/work/jinr1/.conda_envs/torch-env/lib:$LD_LIBRARY_PATH

LAMMPS=/scratch/phys/t30429_nume-dft-ml/01-codes-and-packages/Triton-v3/i-lammps-with-QUIP/install-lammps-stable_2Aug2023_update3/bin/lmp

# Part 1: ballistic phase (0 to 0.1 ps)
srun $LAMMPS -in input1.lmp
mv log.lammps log1.lammps

# Part 2: relaxation phase (0.1 to 1.0 ps)
read xcm_init ycm_init zcm_init < com_s1.txt
srun $LAMMPS -in input2.lmp -var xcm_init $xcm_init -var ycm_init $ycm_init -var zcm_init $zcm_init
mv log.lammps log2.lammps

# Part 3: stopping phase (1.0 to 200.0 ps)
read xcm_init ycm_init zcm_init < com_s2.txt
srun $LAMMPS -in input3.lmp -var xcm_init $xcm_init -var ycm_init $ycm_init -var zcm_init $zcm_init
mv log.lammps log3.lammps

# Part 4: stopping phase (200.0 to 400.0 ps)
read xcm_init ycm_init zcm_init < com_s3.txt
srun $LAMMPS -in input4.lmp -var xcm_init $xcm_init -var ycm_init $ycm_init -var zcm_init $zcm_init
mv log.lammps log4.lammps



