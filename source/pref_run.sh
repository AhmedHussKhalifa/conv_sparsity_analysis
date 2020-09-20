#!/bin/bash
#SBATCH --job-name=run_infer_gpu
#SBATCH --account=def-ehyangit
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G        # memory per node
#SBATCH --time=0-01:30      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
module purge
module load cuda cudnn 

source /home/ahamsala/scratch/tensorflow_gpu_1.12/bin/activate
time python ./run_inference.py --select Org --model_name IV3 --END 4

nvcc  -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64 ../new_alg/new_alg.cu -o ../new_alg/new_alg
# nvcc  -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64 new_alg.cu -o new_alg
time ../new_alg/new_alg > new_alg_runTime.txt #name of your program

module purge
module load nixpkgs/16.09
module load intel/2016.4
module load cuda

nvcc -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64 ../im2col/im2col.cu -o ../im2col/im2col
time ../im2col/im2col > im2col_runTime.txt #name of your program

g++ ../sparse/sparse.cpp -o ../sparse/sparse
time ../sparse/sparse > sparse.txt


# salloc --time=0:20:0 --gres=gpu:1 --mem=2G  --account=def-ehyangit