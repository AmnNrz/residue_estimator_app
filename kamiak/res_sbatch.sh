#!/bin/bash
#SBATCH --partition=camas           # Partition to submit to
#SBATCH --requeue
#SBATCH --job-name=res_app           # Name of your job
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks per node
#SBATCH --cpus-per-task=10           # Number of CPU cores per task
#SBATCH --gres=gpu:2                # Request GPUs 
#SBATCH --mem=128G                   # Memory per node
#SBATCH --time=48:00:00              # Maximum run time (48 hours)

#SBATCH --output=/home/a.norouzikandelati/Projects/residue_estimator_app/kamiak/output/pipeline.o
#SBATCH --error=/home/a.norouzikandelati/Projects/residue_estimator_app/kamiak/error/pipeline.e

echo
echo "--- We are now in $PWD ..."
echo

# Load Singularity module
module load singularity/3.0.0

# # Pull your custom TensorFlow GPU image from DockerHub
# # This step will download your image and convert it into a Singularity Image File (SIF)
# singularity pull docker://amnnrz/tensorflow-gpu:23.07-tf2-py3

# Run your TensorFlow script inside the Singularity container with GPU support
singularity exec --nv \
  /home/a.norouzikandelati/Projects/residue_estimator_app/kamiak/tensorflow-gpu_23.07-tf2-py3.sif \
  python /home/a.norouzikandelati/Projects/residue_estimator_app/kamiak/segmentation_pipeline.py