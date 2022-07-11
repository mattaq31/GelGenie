#!/bin/bash

################################################################################
#                                                                              #
# UNet Training With Corvana Images                                            #
#                                                                              #
################################################################################

#  job name: -N

# Grid Engine options (lines prefixed with #$)
#$ -cwd
#
# Predicted Runtime Limit of 1 hour
#$ -l h_rt=01:00:00
#
# Allocate Memory Limit of 6 GByte
#$ -l h_vmem=6G

# Load all modules/environment/python
module load anaconda/5.3.1
module load cuda/11.0.2
source activate gel_env

# Set cuda in interactive session
source /exports/applications/support/set_cuda_visible_devices.sh

# Run the program
python -u /exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/Automatic-Gel-Analysis/backend/segmentation/UNet_Training.py

