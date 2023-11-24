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


