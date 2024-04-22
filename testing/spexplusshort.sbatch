#!/bin/bash
#SBATCH --job-name="SpexPlusShort-test"
#SBATCH --output="spexplusshort-test.out"
#SBATCH --time=2:0:0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

# Executable
python3 short_causal_test.py -cp ./configs/ -cn spexplusshort
