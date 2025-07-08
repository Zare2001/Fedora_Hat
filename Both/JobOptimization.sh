#!/bin/bash -l
#SBATCH --job-name=edgepyOpti
#SBATCH --partition=gpu-a100          # full 80 GB A100
#SBATCH --ntasks=1                    # one Python process
#SBATCH --gpus-per-task=1             # one full GPU
#SBATCH --cpus-per-task=8             # enough CPU to feed the GPU
#SBATCH --mem-per-cpu=8G              # 64 GB host RAM total
#SBATCH --time=24:00:00               # up to 48 h allowed in this queue
#SBATCH --signal=15@1800              # 30-min checkpoint warning
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load 2024r1
module load py-pip/23.1.2

# If you use conda or venv, activate here
# source ~/your-env/bin/activate
# conda activate your-conda-env

python OptimizationFed.py
