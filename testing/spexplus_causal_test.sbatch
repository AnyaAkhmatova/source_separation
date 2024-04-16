#!/bin/bash
#SBATCH --job-name="SpexPlus-causal-test"
#SBATCH --output="spexplus-causal-test.out"
#SBATCH --time=1:0:0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

# Executable
python3 causal_test.py -cp ./configs/ -cn spexplus_causal_testing
