#!/bin/bash
#SBATCH --partition=camas           # Partition to submit to
#SBATCH --requeue
#SBATCH --job-name=res_app    # Name of your job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --gres=gpu:2              # Request 4 GPUs (specific to camas partition)
#SBATCH --mem=128G                  # Memory per node
#SBATCH --time=48:00:00             # Maximum run time (48 hours)

###SBATCH -k o
#SBATCH --output=/home/a.norouzikandelati/Projects/res_app/output/pipline.o
#SBATCH  --error=/home/a.norouzikandelati/Projects/res_app/error/pipeline.e

echo
echo "--- We are now in $PWD ..."
echo

## echo "I am Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, and array task ${SLURM_ARRAY_TASK_ID}."

# Activate Conda environment
module load anaconda3

source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
source activate resapp

# # Load CUDA module``
# module load cuda/12.2.0
# module load cudnn/8.9.7_cuda12.2

module load cuda/11.8.0
module load cudnn/8.9.4_cuda11.8


python /home/a.norouzikandelati/Projects/res_app/segmentation_pipeline.py
# python /home/a.norouzikandelati/Projects/res_app/test.py
