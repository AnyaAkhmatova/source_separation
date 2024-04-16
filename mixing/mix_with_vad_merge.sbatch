#!/bin/bash
#SBATCH --job-name="SpexPlus-mix-with-vad-merge"
#SBATCH --output="mix-with-vad-merge.out"
#SBATCH --time=3:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10

# Executable
python3 mix.py -cp ./configs/ -cn mixing_with_vad_merge
