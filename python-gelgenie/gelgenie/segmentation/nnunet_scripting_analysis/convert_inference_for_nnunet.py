import os
import imageio
import cv2

from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, extract_image_names_from_folder


input_folders = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/val_images',
                 '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/val_images',
                 '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/val_images',
                 '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/val_images']

input_folders = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/base_data',]

out_folder = '/Users/matt/Desktop/nnunet_data'
dataset_name = 'GELTEST'

create_dir_if_empty(out_folder)

index_out = 0
for input_folder in input_folders:
    for i, ifile in enumerate(extract_image_names_from_folder(input_folder)):
        image = imageio.v2.imread(ifile)
        if image.shape[-1] == 3:  # Actual input: 3 channels
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[-1] == 4:  # Actual input: 4 channels
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        imageio.v2.imwrite(os.path.join(out_folder, f'{dataset_name}_{index_out:04d}_0000.tif'), image)
        index_out += 1


