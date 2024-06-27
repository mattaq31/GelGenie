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

import os
import shutil


inference_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/nnunet_final/test_inference/epoch_best_fold_0'
# inference_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/nnunet_final/test_inference/epoch_600_fold_all'

conv_file = os.path.join(inference_folder, 'image_mapper.txt')

with open(conv_file, 'r') as f:
    lines = f.readlines()

lines = [(x.split(',')[0].split('.')[0] + '.tif', x.split(',')[1].replace('\n', '').replace('_0000.', '.')) for x in lines]

for i in lines:
    shutil.move(os.path.join(inference_folder, i[1]), os.path.join(inference_folder, i[0]))
