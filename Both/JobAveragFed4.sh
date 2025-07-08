#!/bin/bash -l
#SBATCH --job-name=edgepy
#SBATCH --partition=gpu-a100-small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:59:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load 2024r1
module load py-pip/23.1.2

# If you use conda or venv, activate here
# source ~/your-env/bin/activate
# conda activate your-conda-env

python -u AverageFed4.py

