#!/bin/bash
#SBATCH --job-name="obt"
#SBATCH --output="obt.out"
#SBATCH --time=0:10:0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

# Executable
python3 short_causal_train.py -cp ./configs/ -cn one_batch_test
