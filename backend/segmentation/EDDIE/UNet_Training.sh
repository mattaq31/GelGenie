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
/exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/Automatic-Gel-Analysis/backend/EDDIE/prerequisites.sh

# Run the program
/exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/Automatic-Gel-Analysis/backend/UNet_Training.py

