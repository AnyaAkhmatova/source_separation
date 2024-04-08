#!/bin/bash
#SBATCH --job-name="SpexPlus-causal-train-obt"
#SBATCH --output="causal-train.out"
#SBATCH --time=0:30:0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

# Executable
python3 causal_train.py -cp ./configs/ -cn one_batch_test
