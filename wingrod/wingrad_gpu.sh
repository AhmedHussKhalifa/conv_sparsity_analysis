#!/bin/bash
#SBATCH --job-name=wingrad_gpu
#SBATCH --account=def-ehyangit
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=2G                # memory (per node)
#SBATCH --time=0-01:00            # time (DD-HH:MM)

module purge
module load cuda

nvcc -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64 cudnn.cu -o cudnn
./cudnn #name of your program

# salloc --time=1:0:0 --gres=gpu:1 --mem=2G  --account=def-ehyangit