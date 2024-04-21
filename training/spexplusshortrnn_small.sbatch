#!/bin/bash
#SBATCH --job-name="SpexPlusShortRNN-small-train"
#SBATCH --output="spexplusshortrnn-small-train.out"
#SBATCH --time=5:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2

# Executable
python3 short_causal_train.py -cp ./configs/ -cn spexplusshortrnn_small
