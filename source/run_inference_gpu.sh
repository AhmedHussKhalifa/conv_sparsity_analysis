#!/bin/bash
#SBATCH --job-name=run_infer_gpu
#SBATCH --account=def-ehyangit
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8G        # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 
source /home/ahamsala/scratch/tensorflow_gpu_1.12/bin/activate
time python ./run_inference.py --select Org --model_name IV3 --END 2