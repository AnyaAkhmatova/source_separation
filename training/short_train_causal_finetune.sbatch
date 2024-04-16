#!/bin/bash
#SBATCH --job-name="SpexPlusShort-train-causal-finetune"
#SBATCH --output="short-train-causal-finetune.out"
#SBATCH --time=4:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2

# Executable
python3 short_causal_train.py -cp ./configs/ -cn spexplusshort_causal_finetune
