#!/bin/bash

################################################################################
#                                                                              #
# UNet Training With Carvana Images                                            #
#                                                                              #
################################################################################

#  job name: -N
#$ -N UNet_Training_64G_1GPU_24hr_100_epochs
#
# Grid Engine options (lines prefixed with #$)
#$ -cwd
#
# Predicted Runtime Limit of 24 hours
#$ -l h_rt=24:00:00
#
# Request 4 GPU:
#$ -pe gpu-titanx 1
#
# Allocate Memory Limit of 64 GByte for each gpu
#$ -l h_vmem=64G
#
#$ -m beas
#$ -M s2137314@ed.ac.uk
#
# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load all modules/environment/python
module load anaconda
source activate gel_env

# Set cuda in interactive session
source /exports/applications/support/set_cuda_visible_devices.sh

# Run the program
python -u /exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/Automatic-Gel-Analysis/backend/segmentation/UNet_Training.py

