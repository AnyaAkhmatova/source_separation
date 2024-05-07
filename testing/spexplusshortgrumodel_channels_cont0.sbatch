#!/bin/bash
#SBATCH --job-name="SpexPlusShortGRUModel-channels-cont0-test"
#SBATCH --output="spexplusshortgrumodel-channels-cont0-test.out"
#SBATCH --time=3:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2

# Executable
python3 short_causal_test.py -cp ./configs/ -cn spexplusshortgrumodel_channels_cont0
