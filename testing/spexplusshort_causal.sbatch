#!/bin/bash
#SBATCH --job-name="SpexPlusShort-test-causal"
#SBATCH --output="short-test-causal.out"
#SBATCH --time=3:0:0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

# Executable
python3 short_causal_test.py -cp ./configs/ -cn spexplusshort_causal
