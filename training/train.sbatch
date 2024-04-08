#!/bin/bash
#SBATCH --job-name="SpexPlus-train"
#SBATCH --output="train.out"
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4

# Executable
python3 train.py -cp ./configs/ -cn training
