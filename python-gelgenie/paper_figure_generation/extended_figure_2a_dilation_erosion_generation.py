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

import os.path
import matplotlib.pyplot as plt
from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder
from tqdm import tqdm
import numpy as np
from skimage.color import label2rgb
import cv2

from collections import defaultdict
import imageio


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

seed = 5

for im_index, (base_image, mask_image) in enumerate(tqdm(zip(base_images, mask_images), total=len(base_images))):

    labels = imageio.v2.imread(mask_image)
    np_image = imageio.v2.imread(base_image)
    name = os.path.basename(base_image).split('.')[0]

    if name != '18_NEB' and name != '4_Thermo':
        continue

    if 'Thermo' in name:
        ref_ladder = thermo_ref
    else:
        ref_ladder = NEB_ref

    labels = labels.astype(np.uint16)
    painted_base_image = label2rgb(labels, image=255-np_image)
    imageio.imwrite(f'/Users/matt/Desktop/dilation_erosion/{name}_base.png', 255-np_image)
    imageio.imwrite(f'/Users/matt/Desktop/dilation_erosion/{name}_base_labels.png', (painted_base_image*255).astype(np.uint8))
    for iter in range(4):  # for each iteration, dilates or erodes image to observe effect of errors in boundary estimation
        if iter < 2:
            dil_image = dilate_image(labels, iterations=iter+1)
            label = 'dilate %s' % (iter+1)
        elif iter < 4:
            dil_image = erode_image(labels, iterations=iter-1)
            label = 'erode %s' % (iter-1)
        painted_image = label2rgb(dil_image, image=255-np_image)
        imageio.imwrite(f'/Users/matt/Desktop/dilation_erosion/real_image_{name}_{label}.png', (painted_image*255).astype(np.uint8))


