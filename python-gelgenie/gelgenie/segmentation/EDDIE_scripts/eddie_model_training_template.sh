#!/bin/bash
#$ -N segmentation_training
#$ -cwd
#$ -l h_rt=24:00:00
# Request 1 GPU:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=128G
#$ -M m.aquilina@ed.ac.uk
#$ -m beas
#$ -o /exports/csce/eddie/eng/groups/DunnGroup/matthew/qsub_environment/logs
#$ -e /exports/csce/eddie/eng/groups/DunnGroup/matthew/qsub_environment/logs

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load all modules/environment/python
module load anaconda
source activate gelgenie

# Set cuda in interactive session
source /exports/applications/support/set_cuda_visible_devices.sh

# Run training script
gelseg_train --parameter_config /exports/csce/eddie/eng/groups/DunnGroup/matthew/Automatic-Gel-Analysis/gelgenie/segmentation/training/config_files/training_configs/YYYYYY.toml

wandb sync --clean --clean-force  # cleans up wandb local log files
