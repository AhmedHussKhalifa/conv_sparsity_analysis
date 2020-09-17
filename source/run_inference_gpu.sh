#!/bin/bash
#SBATCH --job-name=run_inference_gpu
#SBATCH --account=def-ehyangit
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G        # memory per node
#SBATCH --time=0-00:10      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 
source tensorflow_gpu_1.12/bin/activate
python ./run_inference.py