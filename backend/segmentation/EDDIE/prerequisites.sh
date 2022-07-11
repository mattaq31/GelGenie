#!/bin/bash

################################################################################
#                                                                              #
# Loading All Prerequisite Modules                                             #
#                                                                              #
################################################################################

#  job name: -N

# Predicted Runtime Limit of 5 minutes
#$ -l h_rt=00:05:00
#
# Allocate GPU Memory Limit of 1 Gbyte
#$ -pe gpu-titanx 1 -l h_vmem=1G



module load anaconda/5.3.1
module load cuda/11.0.2
source activate gel_env

# Set cuda in interactive session
source /exports/applications/support/set_cuda_visible_devices.sh 

# go to correct directory
cd /exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/Automatic-Gel-Analysis/backend/

# install packages from requirements.txt in backend folder
conda install --file /exports/csce/eddie/eng/groups/DunnGroup/kiros/2022_summer_intern/Automatic-Gel-Analysis/backend/requirements.txt --channel conda-forge
conda install -c conda-forge python-socketio
conda install -c conda-forge opencv
conda install -c conda-forge tqdm
conda install -c conda-forge wandb

# install codebase
pip install -e

#install torchshow
pip install torchshow



# Install Pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c conda-forge -c pytorch