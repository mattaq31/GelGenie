import os
import imageio
import cv2

from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, extract_image_names_from_folder


input_folders = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/test_images',
                 '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/test_images',
                 '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/quantitation_ladder_gels/test_images',
                 '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/test_images',
                 '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/test_images']


out_folder = '/Users/matt/Desktop/nnunet_data'
dataset_name = 'GELTEST'

create_dir_if_empty(out_folder)

index_out = 0
image_mapper = []
for input_folder in input_folders:
    for i, ifile in enumerate(extract_image_names_from_folder(input_folder)):
        image = imageio.v2.imread(ifile)
        if image.shape[-1] == 3:  # Actual input: 3 channels
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[-1] == 4:  # Actual input: 4 channels
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        new_name = f'{dataset_name}_{index_out:04d}_0000.tif'
        imageio.v2.imwrite(os.path.join(out_folder, new_name), image)
        index_out += 1
        image_mapper.append((os.path.basename(ifile), new_name))

with open(os.path.join(out_folder, 'image_mapper.txt'), 'w') as f:
    for item in image_mapper:
        f.write(f'{item[0]},{item[1]}\n')
