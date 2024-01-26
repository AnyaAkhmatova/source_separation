#!/bin/bash
#SBATCH --job-name="SpexPlus-train"
#SBATCH --output="train.out"
#SBATCH --time=20:0:0
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=10

# Executable
python3 train.py -cp ./configs/ -cn training
