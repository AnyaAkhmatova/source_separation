#!/bin/bash
#SBATCH --job-name="SpexPlusShort-causal-train"
#SBATCH --output="short-causal-train.out"
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2

# Executable
python3 short_causal_train.py -cp ./configs/ -cn short_causal_training
