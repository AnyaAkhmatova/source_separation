#!/bin/bash
#SBATCH --job-name="SpexPlus-test"
#SBATCH --output="test.out"
#SBATCH --time=3:0:0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

# Executable
python3 test.py -cp ./configs/ -cn testing
