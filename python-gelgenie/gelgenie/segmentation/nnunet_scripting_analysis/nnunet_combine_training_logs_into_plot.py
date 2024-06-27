"""
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

"""
This code combines multiple training logs from nnunet into a single file for later plotting (nnunet provides plots but no csv outputs).
"""

import os
import pandas as pd

input_folders = [
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/nnunet_final/fold_0',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/nnunet_final/fold_all']

training_log_groups = [['training_log_2023_12_22_10_03_45.txt',
                        'training_log_2023_12_24_08_59_56.txt',
                        'training_log_2023_12_26_07_39_47.txt'],
                       ['training_log_2023_12_22_10_16_45.txt',
                        'training_log_2023_12_24_08_59_18.txt',
                        'training_log_2023_12_26_12_20_33.txt']]

for input_folder, training_logs in zip(input_folders, training_log_groups):
    dice_log = []
    train_log = []
    epoch_log = []
    max_epoch = 600

    for log in training_logs:
        file = os.path.join(input_folder, log)
        with open(file, 'r') as f:
            lines = f.readlines()
        start_point_found = False
        for line in lines:
            if start_point_found:
                if 'train_loss' in line:
                    train_log.append(float(line.split('train_loss')[-1].replace('\n', '').replace(' ', '')))
                if 'Pseudo dice' in line:
                    dice_log.append(float(
                        line.split('Pseudo dice')[-1].replace('\n', '')
                        .replace(' ', '').replace('[', '')
                        .replace(']', '')))
                if 'Epoch' in line and 'Epoch time' not in line:
                    epoch_log.append(int(line.split('Epoch')[-1].replace('\n', '').replace(' ', ''))+1)
            elif 'Epoch' in line:
                new_epoch = int(line.split('Epoch')[-1].replace('\n', '').replace(' ', '')) + 1
                if new_epoch in epoch_log:
                    epoch_log = epoch_log[:new_epoch]
                    train_log = train_log[:new_epoch-1]
                    dice_log = dice_log[:new_epoch-1]
                else:
                    epoch_log.append(new_epoch)
                start_point_found = True
                continue

    dice_log = dice_log[:max_epoch]
    train_log = train_log[:max_epoch]
    epoch_log = epoch_log[:max_epoch]

    # Creating a DataFrame
    df = pd.DataFrame({
        'Pseudo-Dice Score': dice_log,
        'Train Loss': train_log,
        'Epoch': epoch_log
    })

    df.to_csv(os.path.join(input_folder, 'combined_training_stats.csv'), index=False)
