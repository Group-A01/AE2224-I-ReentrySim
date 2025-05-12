#!/bin/sh

#SBATCH --job-name=dkiryukhin-reentry-pq
#SBATCH --partition=compute
#SBATCH --account=education-ae-bsc-lr
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB

module load miniconda3
conda activate tudat-space
python routine-pq.py