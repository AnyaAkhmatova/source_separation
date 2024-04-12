#!/bin/bash
#SBATCH --job-name="SpexPlus-causal-train"
#SBATCH --output="causal-train.out"
#SBATCH --time=15:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2

# Executable
python3 causal_train.py -cp ./configs/ -cn causal_training
