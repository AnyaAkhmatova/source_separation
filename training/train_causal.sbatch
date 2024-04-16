#!/bin/bash
#SBATCH --job-name="SpexPlus-train-causal"
#SBATCH --output="train-causal.out"
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2

# Executable
python3 causal_train.py -cp ./configs/ -cn spexplus_causal
