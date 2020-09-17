#!/bin/bash
#SBATCH --job-name=new_alg_gpu
#SBATCH --account=def-ehyangit
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=2G                # memory (per node)
#SBATCH --time=0-01:00            # time (DD-HH:MM)

module purge
module load cuda

nvcc new_alg.cu -o new_alg
./new_alg #name of your program

# salloc --time=1:0:0 --gres=gpu:1 --mem=2G  --account=def-ehyangit