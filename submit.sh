#!/bin/bash


#SBATCH --time=24:00:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --job-name=login
#SBATCH --error=job.error
#SBATCH --output=job.output

python -m ml4cascades.potentials.sw --choice 7
