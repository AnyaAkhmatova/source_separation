#!/bin/bash
#SBATCH --job-name="SpexPlusShortRNNModel-rnn-train"
#SBATCH --output="spexplusshortrnnmodel-rnn-train.out"
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2

# Executable
python3 short_causal_train.py -cp ./configs/ -cn spexplusshortrnnmodel_rnn
