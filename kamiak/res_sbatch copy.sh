#!/bin/bash
#SBATCH --partition=camas           # Partition to submit to
#SBATCH --requeue
#SBATCH --job-name=res_app    # Name of your job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks per node
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --gres=gpu:2              # Request 2 GPUs (specific to camas partition)
#SBATCH --mem=128G                  # Memory per node
#SBATCH --time=48:00:00             # Maximum run time (48 hours)

#SBATCH --output=/home/a.norouzikandelati/Projects/residue_estimator_app/kamiak/output/pipeline.o
#SBATCH --error=/home/a.norouzikandelati/Projects/residue_estimator_app/kamiak/error/pipeline.e

echo
echo "--- We are now in $PWD ..."
echo

# Activate Conda environment
module load anaconda3

source /opt/apps/anaconda3/22.10.0/etc/profile.d/conda.sh
source activate resapp

# Load CUDA 12.2 and cuDNN 8.9.7 (compatible with TensorFlow setup)
module load cuda/12.2.0
module load cudnn/8.9.7_cuda12.2
nvidia-smi

# Run the Python script
# python /home/a.norouzikandelati/Projects/residue_estimator_app/kamiak/test.py
python /home/a.norouzikandelati/Projects/residue_estimator_app/kamiak/segmentation_pipeline.py