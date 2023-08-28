import os
import imageio
import cv2
import shutil
from PIL import Image
import numpy as np

from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, extract_image_names_from_folder

dir_train_mask = [
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/masks']

dir_train_img = [
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/images']


out_folder = '/Users/matt/Desktop/nnunet_data'
labels_folder = '/Users/matt/Desktop/nnunet_data/Dataset088_GELSEG/labelsTr'
images_folder = '/Users/matt/Desktop/nnunet_data/Dataset088_GELSEG/imagesTr'
dataset_name = 'GELSEG'

create_dir_if_empty(out_folder, labels_folder, images_folder)

global_index = 1
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
        global_index += 1


