#!/bin/bash
#SBATCH --job-name=im2col_gpu
#SBATCH --account=def-ehyangit
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=2G                # memory (per node)
#SBATCH --time=0-01:00            # time (DD-HH:MM)

module purge
# module load nixpkgs/16.09 
# module load gcc/4.8.5
# module load cuda/7.5.18

module load nixpkgs/16.09
module load intel/2016.4
module load cuda

module list

nvcc -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64 im2col.cu -o im2col
./im2col #name of your program

# salloc --time=1:0:0 --gres=gpu:1 --mem=2G  --account=def-ehyangit