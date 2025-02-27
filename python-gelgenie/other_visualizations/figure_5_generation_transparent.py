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
import copy

import cv2
import numpy as np
from skimage.measure import regionprops, label
import imageio
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from collections import OrderedDict
from gelgenie.segmentation.helper_functions.segmentation_postprocessing import (find_nearest_points, find_super_groups,
                                                                                fill_in_bands, separate_into_columns,
                                                                                merge_band_groups,
                                                                                first_round_band_analysis,
                                                                                second_round_band_analysis)


main_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/lane_finding_tests/paper_plots'
main_folder = '/Users/matt/Desktop/tst'

# example 1 - problem with conjoined bands
# input_image_map = os.path.join(main_folder, 'input_images', '25_NEB_unet_dec_21_map.png')
# input_base_image = os.path.join(main_folder, 'input_images', '25_NEB_adjusted.tif')
# base_map = (imageio.v2.imread(input_image_map))
# base_map = base_map[:, 285:1208, :]
# input_base_image = imageio.v2.imread(input_base_image)
# input_base_image = input_base_image[:, 285:1208]

# example 2
# input_image_map = os.path.join(main_folder, 'input_images', '49_unet_dec_21_map.png')
# input_base_image = os.path.join(main_folder, 'input_images', '49_adjusted.tif')
# base_map = (imageio.v2.imread(input_image_map))
# base_map = base_map[:438, 273:1212, :]
# input_base_image = imageio.v2.imread(input_base_image)
# input_base_image = input_base_image[:438, 273:1212]

# example 3
# input_image_map = os.path.join(main_folder, 'input_images', '81_unet_dec_21_map.png')
# input_base_image = os.path.join(main_folder, 'input_images', '81_adjusted.tiff')
# base_map = (imageio.v2.imread(input_image_map))
# base_map = base_map[:710, 242:1220, :]
# input_base_image = imageio.v2.imread(input_base_image)
# input_base_image = input_base_image[:710, 242:1220]

# example 4
input_image_map = os.path.join(main_folder, 'input_images', '143_unet_dec_21_map.png')
input_base_image = os.path.join(main_folder, 'input_images', '143.tif')
base_map = (imageio.v2.imread(input_image_map))
base_map = base_map[:534, 428:1225, :]
input_base_image = imageio.v2.imread(input_base_image)
input_base_image = input_base_image[:534, 428:1225]

base_map_custom_color = copy.deepcopy(base_map)
base_map_custom_color[base_map_custom_color == 163] = 254
base_map_custom_color[base_map_custom_color == 106] = 149
base_map_custom_color[base_map_custom_color == 255] = 150
base_map_custom_color[base_map_custom_color == 13] = 0

base_map[base_map == 163] = 0
base_map[base_map == 106] = 201
base_map[base_map == 255] = 100
base_map[base_map == 13] = 255

binary_image = base_map.argmax(axis=-1)

contour_width = 2.5
box_width = 6

overlay_color = '#FF9600'
rectangle_color = '#3142AC'

################ STEP 1 - Centroid and Image Prep ################
labeled_image = label(binary_image, connectivity=2)

# finds contours for visualization purposes
contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Compute region properties for labeled blobs
props = regionprops(labeled_image, binary_image)

band_reference_dict = OrderedDict()
for prop in props:
    band_reference_dict[prop.label] = {'centroid': prop.centroid,
                                       'width': prop.bbox[3] - prop.bbox[1],
                                       'height': prop.bbox[2] - prop.bbox[0],
                                       'bbox': prop.bbox}

median_length_blobs = np.median([prop['width'] for prop in band_reference_dict.values()])
median_height_blobs = np.median([prop['height'] for prop in band_reference_dict.values()])

# rule-of-thumb thresholds
horizontal_threshold = median_length_blobs / 3
vertical_threshold = median_height_blobs / 2

################ STEP 2 - Column finding ################
# Find nearest points and separate columns
nearest_centroid_indices = find_nearest_points([prop['centroid'] for prop in band_reference_dict.values()],
                                               median_length_blobs)
# Separate connected blobs into columns
merged_groups = separate_into_columns(nearest_centroid_indices)

for band, cent_ind in zip(band_reference_dict.values(), nearest_centroid_indices):
    band['nearest_centroid_id'] = list(band_reference_dict.keys())[cent_ind]

band_groups = merge_band_groups(merged_groups, band_reference_dict)
super_groups = find_super_groups(band_groups, horizontal_threshold)

################ STEP 3 - Outlier Approach ################
first_round_band_analysis(super_groups, band_reference_dict, vertical_threshold)
################ STEP 4 - Direct Approach ################
second_round_band_analysis(super_groups, band_reference_dict, vertical_threshold)

updated_image = fill_in_bands(super_groups, binary_image, labeled_image)

# base image only
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.imshow(1-input_base_image, cmap='gray')
plt.axis('off')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(main_folder, 'base_image.png'), dpi=300)
plt.show()
plt.close(fig)


# Part 1
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.imshow(1-input_base_image, cmap='gray')
plt.imshow(base_map_custom_color)
for contour in contours:
    ax.plot(contour[:, 0, 0], contour[:, 0, 1], '-', color=overlay_color, linewidth=contour_width, zorder=1)
all_blob_centroids = np.array([prop.centroid for prop in props])
# for i, centroid in enumerate(all_blob_centroids):
#     ax.scatter(centroid[1], centroid[0], c='b', marker='+', linewidth=1.5, zorder=3)
plt.axis('off')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(main_folder, 'P1_lane_finding.png'), dpi=300)
plt.show()
plt.close(fig)

# Part 2
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.imshow(1-input_base_image, cmap='gray')
plt.imshow(base_map_custom_color)
for contour in contours:
    ax.plot(contour[:, 0, 0], contour[:, 0, 1], '-', color=overlay_color, linewidth=contour_width, zorder=1)
for i, band_gp in enumerate(band_groups.values()):
    rect = patches.Rectangle((band_gp['bbox'][1], band_gp['bbox'][0]), band_gp['bbox'][3] - band_gp['bbox'][1],
                             band_gp['bbox'][2] - band_gp['bbox'][0], linewidth=box_width,
                             edgecolor=rectangle_color,
                             facecolor='none')
    ax.add_patch(rect)
plt.axis('off')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(main_folder, 'P2_lane_finding.png'), dpi=300)
plt.show()
plt.close(fig)

# Part 3 - lanes finished
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.imshow(1-input_base_image, cmap='gray')
plt.imshow(base_map_custom_color)
for contour in contours:
    ax.plot(contour[:, 0, 0], contour[:, 0, 1], '-', color=overlay_color, linewidth=contour_width, zorder=1)
for i, band_gp in enumerate(super_groups.values()):
    rect = patches.Rectangle((band_gp['bbox'][1], band_gp['bbox'][0]), band_gp['bbox'][3] - band_gp['bbox'][1],
                             band_gp['bbox'][2] - band_gp['bbox'][0], linewidth=box_width,
                             edgecolor=rectangle_color,
                             facecolor='none')
    ax.add_patch(rect)
plt.axis('off')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(main_folder, 'P3_lane_finding.png'), dpi=300)
plt.show()
plt.close(fig)

# Part 3.2 - no centroids
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.imshow(1-input_base_image, cmap='gray')
plt.imshow(base_map_custom_color)
for contour in contours:
    ax.plot(contour[:, 0, 0], contour[:, 0, 1], '-', color=overlay_color, linewidth=contour_width, zorder=1)
for i, band_gp in enumerate(super_groups.values()):
    rect = patches.Rectangle((band_gp['bbox'][1], band_gp['bbox'][0]), band_gp['bbox'][3] - band_gp['bbox'][1],
                             band_gp['bbox'][2] - band_gp['bbox'][0], linewidth=box_width,
                             edgecolor=rectangle_color,
                             facecolor='none')
    ax.add_patch(rect)
plt.axis('off')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(main_folder, 'P3-2_lane_finding.png'), dpi=300)
plt.show()
plt.close(fig)

# Part 4 - convex hull
# finds contours for visualization purposes
old_bands = np.copy(updated_image)
old_bands[old_bands == 1] = 0
new_bands = np.copy(updated_image)
new_bands[new_bands == 2] = 0

old_contours, _ = cv2.findContours(old_bands.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
new_contours, _ = cv2.findContours(new_bands.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

fig, ax = plt.subplots(1,1, figsize=(15, 10))
plt.imshow(1-input_base_image, cmap='gray')
plt.imshow(base_map_custom_color)

# sets colour for new bands
overlay_image = np.zeros((updated_image.shape[0], updated_image.shape[1], 4))
overlay_image[updated_image == 1, :] = 1.0
overlay_image[updated_image == 1, 0] = 49/255
overlay_image[updated_image == 1, 1] = 66/255
overlay_image[updated_image == 1, 2] = 172/255

plt.imshow(overlay_image)
for contour in old_contours:
    ax.plot(contour[:, 0, 0], contour[:, 0, 1], '-', color=overlay_color, linewidth=contour_width, zorder=1)
for contour in new_contours:
    ax.plot(contour[:, 0, 0], contour[:, 0, 1], '-', color=rectangle_color, linewidth=contour_width, zorder=1)

plt.axis('off')
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(main_folder, 'P4_lane_finding.png'), dpi=300)
plt.show()
plt.close(fig)
