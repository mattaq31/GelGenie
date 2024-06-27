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
import imageio
import cv2
import json

from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, \
    extract_image_names_from_folder

dir_train_mask = [
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/quantitation_ladder_gels/masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/val_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/val_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/val_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/val_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/quantitation_ladder_gels/val_masks'
]

dir_train_img = [
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/quantitation_ladder_gels/images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/val_images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/val_images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/val_images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/val_images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/quantitation_ladder_gels/val_images'
]

validation_folders = [5, 6, 7, 8, 9]
out_folder = '/Users/matt/Desktop/nnunet_data'
labels_folder = '/Users/matt/Desktop/nnunet_data/Dataset088_GELSEG/labelsTr'
images_folder = '/Users/matt/Desktop/nnunet_data/Dataset088_GELSEG/imagesTr'
splits_file = '/Users/matt/Desktop/nnunet_data/Dataset088_GELSEG/splits_final.json'
dataset_name = 'GELSEG'

create_dir_if_empty(out_folder, labels_folder, images_folder)

global_index = 1
validation_images = []
train_images = []
for i, (dir_mask, dir_img) in enumerate(zip(dir_train_mask, dir_train_img)):
    for mfile, ifile in zip(extract_image_names_from_folder(dir_mask), extract_image_names_from_folder(dir_img)):
        print(ifile)
        image = imageio.v2.imread(ifile)
        mask = imageio.v2.imread(mfile)

        # pil_mask = np.array(Image.open(mfile))

        if image.shape[-1] == 3:  # Actual input: 3 channels
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[-1] == 4:  # Actual input: 4 channels
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        imageio.v2.imwrite(os.path.join(images_folder, f'{dataset_name}_{global_index:04d}_0000.tif'), image)
        imageio.v2.imwrite(os.path.join(labels_folder, f'{dataset_name}_{global_index:04d}.tif'), mask)

        # shutil.copy2(mfile, os.path.join(labels_folder, f'dataset_name_{global_index:04d}.tif'))

        if i in validation_folders:
            validation_images.append(f'{dataset_name}_{global_index:04d}')
        else:
            train_images.append(f'{dataset_name}_{global_index:04d}')
        global_index += 1

splits_dict = [{'train': train_images, 'val': validation_images}]
with open(splits_file, 'w') as f:
    json.dump(splits_dict, f, sort_keys=True, indent=4)
