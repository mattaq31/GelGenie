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
from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder
import shutil


# mask_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/masks'
# originals_paths = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/originals/matthew_gels_2']
# output_image_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/images'

# mask_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/masks'
# originals_paths = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/originals/LSDB_data/caps',
#                    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/originals/LSDB_data/est-pcr']
# output_image_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/images'

mask_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/masks'
originals_paths = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/originals/matthew_gels']
output_image_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/images'

mask_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/stella_gels/masks'
originals_paths = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/originals/stella_gels']
output_image_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/stella_gels/images'


original_images = []
for path in originals_paths:
    original_images.extend(extract_image_names_from_folder(path))

original_filenames = [file.split('/')[-1].split('.')[0] for file in original_images]

for index, file in enumerate(extract_image_names_from_folder(mask_path)):

    if 'Edited SegMaps' not in file:
        continue
    filename = file.split('/')[-1].split('.')[0]
    new_name = filename.replace(' Edited SegMaps', '')
    updated_mask = file.replace(filename, new_name)

    if new_name not in original_filenames:
        continue

    input_image_file = original_images[original_filenames.index(new_name)]

    # final operations
    if os.path.isfile(updated_mask):
        os.remove(file)
    else:
        os.rename(file, updated_mask)
        shutil.copy2(input_image_file, output_image_path)
