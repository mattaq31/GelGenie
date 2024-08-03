import os
from collections import defaultdict

import cv2
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder


def dilate_image(image, iterations=1):
    kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel for dilation
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image


def erode_image(image, iterations=1):
    kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel for dilation
    dilated_image = cv2.erode(image, kernel, iterations=iterations)
    return dilated_image


input_folder = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_evaluation/ladder_eval/base_data']
base_images = extract_image_names_from_folder(input_folder[0])
mask_folder = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_evaluation/ladder_eval/qupath_data/james_data_v3_fixed_global/indiv_label_segmaps']
mask_images = extract_image_names_from_folder(mask_folder[0])

reference_path = "/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_evaluation/ladder_eval/reference_ladder_masses.csv"

data_df = defaultdict(list)

thermo_ref = [20] * 9 + [70] + [30] * 7 + [40]
NEB_ref = [40, 40, 48, 40, 32, 120, 40, 57, 45, 122, 34, 31, 27, 23, 124, 49, 37, 32, 61]

check_type = 'erode'

for im_index, (base_image, mask_image) in enumerate(tqdm(zip(base_images, mask_images), total=len(base_images))):
    labels = imageio.v2.imread(mask_image)
    np_image = imageio.v2.imread(base_image)
    name = os.path.basename(base_image).split('.')[0]

    np_image = np_image[564:625, 383:451]
    labels = labels[564:625, 383:451]

    if check_type == 'erode':
        dil_image = erode_image(labels, iterations=1)
        dil_image = dil_image + labels
        dil_image_2 = erode_image(labels, iterations=2)
        dil_image_2 = dil_image_2 + labels
        dil_image_3 = erode_image(labels, iterations=3)
        dil_image_3 = dil_image_3 + labels
    else:
        dil_image = erode_image(labels, iterations=1)
        dil_image = dil_image + labels
        dil_image_2 = erode_image(labels, iterations=2)
        dil_image_2 = dil_image_2 + labels
        dil_image_3 = erode_image(labels, iterations=3)
        dil_image_3 = dil_image_3 + labels

    fig, ax = plt.subplots(1, 4, figsize=(10, 10))
    ax[0].imshow(labels, cmap='gray')
    ax[1].imshow(dil_image, cmap='gray')
    ax[2].imshow(dil_image_2, cmap='gray')
    ax[3].imshow(dil_image_3, cmap='gray')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    plt.tight_layout()
    plt.show()
    break
