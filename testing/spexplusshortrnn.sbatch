#!/bin/bash
#SBATCH --job-name="SpexPlusShortRNN-test"
#SBATCH --output="spexplusshortrnn-test.out"
#SBATCH --time=1:0:0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

# Executable
python3 short_causal_test.py -cp ./configs/ -cn spexplusshortrnn
