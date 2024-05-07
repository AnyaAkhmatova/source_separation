#!/bin/bash
#SBATCH --job-name="SpexPlusShortGRUModel-channels-cont2-train"
#SBATCH --output="spexplusshortgrumodel-channels-cont2-train.out"
#SBATCH --time=10:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2

# Executable
python3 short_causal_train.py -cp ./configs/ -cn spexplusshortgrumodel_channels_cont2
