#!/bin/bash
#SBATCH --job-name="SpexPlusShortCacheModel-channels-train"
#SBATCH --output="spexplusshortcachemodel-channels-train.out"
#SBATCH --time=20:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2

# Executable
python3 short_causal_train.py -cp ./configs/ -cn spexplusshortcachemodel_channels
