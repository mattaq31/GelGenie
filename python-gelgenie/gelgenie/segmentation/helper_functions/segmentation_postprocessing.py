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
from collections import OrderedDict

import imageio
import numpy as np
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from skimage.morphology import convex_hull_image
from tqdm import tqdm

from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, \
    extract_image_names_from_folder


def find_nearest_points(all_blob_centroids, median_length_blobs):
    """
    Finds the nearest points to each blob centroid provided, using the median  blob length as a threshold.
    :param all_blob_centroids: List of blob centroids, where each row represents a single centroid
    :param median_length_blobs: Median blob length. Threshold is set to median_length_blobs/2
    :return: List containing the closest centroid for each centroid in the original list.
    """
    nearest_centroid_indices = []
    for self_index, sel_centroid in enumerate(all_blob_centroids):
        distance_y = np.array([a[0] - sel_centroid[0] for a in all_blob_centroids])
        distance_x = np.array([a[1] - sel_centroid[1] for a in all_blob_centroids])

        valid_indices = np.where(np.abs(distance_x) < (median_length_blobs / 2))[0]
        distances = np.sqrt(distance_y[valid_indices] ** 2 + distance_x[valid_indices] ** 2)

        if len(valid_indices) == 1:
            nearest_point = self_index
        else:
            nearest_point = valid_indices[np.argsort(distances)[1]]

        nearest_centroid_indices.append(nearest_point)

    return nearest_centroid_indices


def separate_into_columns(nearest_centroid_list):
    """
    Combines all centroids into separate columns by using the 'friend' system,
    where each centroid is connected to its nearest centroid and so on.
    :param nearest_centroid_list: List of indices of the nearest centroid to each particular band.
    :return: List of lists, where each list contains the indices of the centroids in a single group.
    """
    group_indices = []
    group_count = 0

    for i, cent_friend in enumerate(
            nearest_centroid_list):  # separates each centroid into groups of 'friends' where each new friend is the nearest centroid to the previous friend
        a_group = [i]
        # friend = indices[i]

        while cent_friend not in a_group:  # adds the latest friend's own friend to the group and so on
            a_group.append(cent_friend)
            cent_friend = nearest_centroid_list[cent_friend]

        group_count += 1
        group_indices.append(a_group)

    # runs through all the  groups identified and removes duplicates - each blob should only be in one group
    idx = 0
    while idx < len(group_indices):
        current_array = group_indices[idx]
        for j in range(idx + 1, len(group_indices)):
            if any(item in current_array for item in group_indices[j]):
                current_array = list(set(current_array) | set(group_indices[j]))
                group_indices[idx] = current_array
                group_indices.pop(j)
                idx -= 1
                break
        idx += 1
    return group_indices


def merge_band_groups(merged_groups, band_dict):
    """
    Combines the bounding boxes and centroids of blobs in the provided groups.
    :param merged_groups: List of centroid groups (indices) that need to be combined.
    :param band_dict: The actual properties of each band in the list.
    :return: Dictionary of groups, where each group contains the bounding box, centroid, and band ids of all blobs in the group.
    """
    band_posn_ids = list(band_dict.keys())
    band_groups = {}

    for group_index, group in enumerate(merged_groups):
        if not group:
            continue  # Skip empty groups
        # Extract bounding boxes for each object in the group
        group_boxes = [band_dict[band_posn_ids[i]]['bbox'] for i in group]
        # Combine bounding boxes
        min_row = min(box[0] for box in group_boxes)
        min_col = min(box[1] for box in group_boxes)
        max_row = max(box[2] for box in group_boxes)
        max_col = max(box[3] for box in group_boxes)
        merged_box = (min_row, min_col, max_row, max_col)
        band_groups[group_index] = {'bbox': merged_box, 'centroid': ((min_row + max_row) / 2, (min_col + max_col) / 2),
                                    'bands': [band_posn_ids[i] for i in group]}

    return band_groups


def find_super_groups(band_groups, horizontal_threshold):
    """
    Performs a second merge operation where previously found groups are combined again into a final super-group,
    each of which should encompass an entire lane.
    :param band_groups: Dict of dicts, containing the properties of each group to merge.
    :param horizontal_threshold: The vertical threshold used to combine boxes into super groups.
    :return: Dict of merged super groups, and their corresponding box coordinates.
    """
    super_merge_groups = {}
    sorted_centroids = sorted([[k, v['centroid'][1], v['bbox']] for k, v in band_groups.items()], key=lambda x: x[1])

    current_group_indices = [0]
    current_super_group = 1

    def combine_bounding_boxes(group_indices):
        group_boxes = [sorted_centroids[g][2] for g in group_indices]
        min_row = np.min([g[0] for g in group_boxes])
        min_col = np.min([g[1] for g in group_boxes])
        max_row = np.max([g[2] for g in group_boxes])
        max_col = np.max([g[3] for g in group_boxes])
        super_merge_groups[current_super_group] = {'bbox': [min_row, min_col, max_row, max_col],
                                                   'centroid': [(min_row + max_row) / 2, (min_col + max_col) / 2],
                                                   'bands': sum([band_groups[sorted_centroids[g][0]]['bands'] for g in
                                                                 group_indices], [])}

    for i in range(1, len(sorted_centroids)):
        if abs(sorted_centroids[i][1] - sorted_centroids[i - 1][1]) <= horizontal_threshold:
            current_group_indices.append(i)
        else:
            combine_bounding_boxes(current_group_indices)
            current_group_indices = [i]
            current_super_group += 1

    combine_bounding_boxes(current_group_indices)

    return super_merge_groups


def mad(data, axis=None):
    """
    median absolute deviation - computed as stated.  The result can be used as an estimate of the standard deviation
     of the data.
    :param data: List of data points
    :param axis: Axis over which to apply the operation, if None, will condense to one value.
    :return: The MAD value, which is an estimate of the standard deviation.
    """
    median = np.median(data, axis=axis)
    mad_value = np.median(np.abs(data - median), axis=axis)

    # 1.4826 is the value for normally distributed data - see https://en.wikipedia.org/wiki/Median_absolute_deviation
    return 1.4826 * mad_value


def is_outlier(data, threshold=3):
    """
    Determines which of the data points in a dataset could be considered an outlier, based on the MAD value.
    :param data: List of points to be analyzed.
    :param threshold: The threshold to use for determining if a point is an outlier.
    This will be multiplied with the MAD value.
    :return: List of booleans, where a True value indicates that the corresponding point is an outlier.
    """

    scaled_mad = mad(data)
    if scaled_mad == 0:
        return [False] * len(data)
    median = np.median(data)
    comparator_data = np.abs(data - median) > threshold * scaled_mad

    return comparator_data


def first_round_band_analysis(super_group_data, band_dictionary, vertical_threshold):
    """
    Analyzes each blob/band in a super-group to determine their status - normal, broken (split in half by mistake by
    the segmentation model) or 'small'.
    The process involves a horizontal position outlier separation using MAD then a simple threshold check to determine
    if outliers are simply 'small' bands or 'broken'.  The outliers found are added directly to the super group dictionaries.
    :param super_group_data: List of dictionaries for each super-group, where each dictionary contains the indices of
     all blobs within the super-group as well as its overall bounding box.
    :param band_dictionary: Dictionary of region properties for each blob in the image.
    :param vertical_threshold: The threshold to apply for checking if a band is broken.
    :return: N/A
    """

    for sk, sv in super_group_data.items():  # one check per super-group
        band_properties = [band_dictionary[j] for j in sv['bands']]
        band_h_lengths = [prop['width'] for prop in band_properties]

        outlier_indices = is_outlier(band_h_lengths)
        detected_outliers = [band_properties[i] for i in range(len(outlier_indices)) if outlier_indices[i]]
        detected_band_ids = [sv['bands'][i] for i in range(len(outlier_indices)) if outlier_indices[i]]

        if np.sum(outlier_indices) == 1:
            # only one abnormally small blob - count it as a normal blob, do
            # nothing but save the index
            outlier_indices = np.where(outlier_indices)[0]
            sv['SmallBands'] = [sv['bands'][i] for i in range(len(outlier_indices)) if outlier_indices[i]]
            sv['BrokenBands'] = []

        elif np.sum(outlier_indices) == 2:
            # in this simple case just check to see if the two outliers overlap and if not, simple classify as 'small'
            outlier_indices = np.where(outlier_indices)[0]

            p1 = np.array(detected_outliers[0]['centroid'])
            p2 = np.array(detected_outliers[1]['centroid'])

            # if the outliers do overlap, then combine and re-calculate the blob centroid
            if np.abs(p1[0] - p2[0]) <= vertical_threshold:
                sv['BrokenBands'] = [sv['bands'][i] for i in range(len(outlier_indices)) if outlier_indices[i]]
                sv['SmallBands'] = []
            else:
                sv['SmallBands'] = [sv['bands'][i] for i in range(len(outlier_indices)) if outlier_indices[i]]
                sv['BrokenBands'] = []

        elif np.sum(outlier_indices) > 2:
            sv['BrokenBands'] = []
            sv['SmallBands'] = []
            # iterates through all outliers and attempts to determine their nature
            while len(detected_band_ids) > 0:
                target_outlier = detected_outliers[0]['centroid']
                all_outliers_centroid_x = np.array([d['centroid'][1] for d in detected_outliers])
                all_outliers_centroid_y = np.array([d['centroid'][0] for d in detected_outliers])

                distance_x = all_outliers_centroid_x - target_outlier[1]
                distance_y = all_outliers_centroid_y - target_outlier[0]
                overlapping_outliers = np.where(np.abs(distance_y) < vertical_threshold)[0]

                if len(overlapping_outliers) == 1:
                    # if there is only one overlapping outlier, this means that the target outlier does not have a mate
                    # and is just a small band or some other mis-segmented blob
                    sv['SmallBands'].append(detected_band_ids[0])
                    detected_outliers.pop(0)
                    detected_band_ids.pop(0)
                else:
                    # if more than 1, then some blobs could potentially be overlapping

                    # First, find the closest outlier to the target outlier
                    distances = np.sqrt(distance_x[overlapping_outliers] ** 2 + distance_y[overlapping_outliers] ** 2)
                    potential_mate = np.argsort(distances)[1]

                    # p1 should be the actual target outlier, p2 should be the closest outlier to p1
                    p1 = np.array(detected_outliers[0]['centroid'])
                    p2 = np.array(detected_outliers[potential_mate]['centroid'])

                    if np.abs(p1[0] - p2[0]) <= vertical_threshold:
                        sv['BrokenBands'].extend([detected_band_ids[0], detected_band_ids[potential_mate]])
                    else:
                        sv['SmallBands'].extend([detected_band_ids[0], detected_band_ids[potential_mate]])

                    detected_outliers.pop(potential_mate)
                    detected_band_ids.pop(potential_mate)
                    detected_outliers.pop(0)
                    detected_band_ids.pop(0)
        else:
            sv['SmallBands'] = []
            sv['BrokenBands'] = []


def second_round_band_analysis(super_group_info, band_dict, threshold):
    for sg in super_group_info.values():
        # Check differences in vertical axes between centroids
        all_group_band_centroids = [band_dict[b]['centroid'] for b in sg['bands']]
        broken_band_set = []
        for band in sg['bands']:
            target_centroid = band_dict[band]['centroid']
            distance_y = np.abs([g[0] - target_centroid[0] for g in all_group_band_centroids])

            valid_indices = np.where((distance_y > 0) & (distance_y < threshold))[0]

            if len(valid_indices) > 0:
                band_set = [band] + [sg['bands'][v] for v in valid_indices]
                broken_band_set.append(band_set)
        sg['BrokenBandsv2'] = broken_band_set


def fill_in_bands(super_group_info, basic_image, label_image):
    image_copy = np.copy(basic_image)
    for sg in super_group_info.values():
        if len(sg['BrokenBandsv2']) > 0:
            for band_set in sg['BrokenBandsv2']:
                # Extract the binary blobs for the current indices_set
                blobs = [label_image == sel_band for sel_band in band_set]
                # Combine binary blobs to create a convex hull
                convex_hull = convex_hull_image(np.logical_or.reduce(blobs))
                # Paint the convex hull on the fixed_image
                image_copy[convex_hull] = 1  # Assuming binary values (0 and 255)

    return image_copy


def full_postprocessing_analysis(binary_image, image_name, original_image=None, debugging_plots=False, full_band_info_plots=False, save_to_folder=False, direct_vis=False):

    labeled_image = label(binary_image, connectivity=2)

    # finds contours just for visualization purposes
    if debugging_plots:
        contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    if debugging_plots:
        # Display boundaries on the original image and blob centroids with index
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        # Plot original image with boundaries and blob centroids
        ax.imshow(1 - binary_image, cmap='gray')
        for contour in contours:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=0.8, zorder=1)
        all_blob_centroids = np.array([prop.centroid for prop in props])
        for i, centroid in enumerate(all_blob_centroids):
            ax.scatter(centroid[1], centroid[0], c='b', marker='+', linewidth=1.5, zorder=3)
        ax.set_title(
            f'Original Image with Centroids, median length={median_length_blobs}, median height={median_height_blobs}')
        plt.suptitle(f'Image {image_name}')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    ################ STEP 2 - Column finding ################

    # Find nearest points and separate columns
    nearest_centroid_indices = find_nearest_points([prop['centroid'] for prop in band_reference_dict.values()],
                                                   median_length_blobs)

    # Separate connected blobs into columns
    merged_groups = separate_into_columns(nearest_centroid_indices)

    for band, cent_ind in zip(band_reference_dict.values(), nearest_centroid_indices):
        band['nearest_centroid_id'] = list(band_reference_dict.keys())[cent_ind]

    band_groups = merge_band_groups(merged_groups, band_reference_dict)

    if debugging_plots:
        # Display boundaries on the original image and blob centroids with index
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        # Plot original image with boundaries and blob centroids
        ax.imshow(1 - binary_image, cmap='gray')

        for i, band_gp in enumerate(band_groups.values()):
            ax.scatter(band_gp['centroid'][1], band_gp['centroid'][0], c='b', marker='+', linewidth=1.5, zorder=3)
            rect = patches.Rectangle((band_gp['bbox'][1], band_gp['bbox'][0]), band_gp['bbox'][3] - band_gp['bbox'][1],
                                     band_gp['bbox'][2] - band_gp['bbox'][0], linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

        ax.set_title(f'Stage 1 Column Finding')
        plt.suptitle(f'Image {image_name}')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # Perform the second merge operation
    super_groups = find_super_groups(band_groups, horizontal_threshold)

    if debugging_plots:
        # Display boundaries on the original image and blob centroids with index
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        # Plot original image with boundaries and blob centroids
        ax.imshow(1 - binary_image, cmap='gray')

        for i, band_gp in enumerate(super_groups.values()):
            ax.scatter(band_gp['centroid'][1], band_gp['centroid'][0], c='b', marker='+', linewidth=1.5, zorder=3)
            rect = patches.Rectangle((band_gp['bbox'][1], band_gp['bbox'][0]), band_gp['bbox'][3] - band_gp['bbox'][1],
                                     band_gp['bbox'][2] - band_gp['bbox'][0], linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

        ax.set_title(f'Stage 1 Column Finding')
        plt.suptitle(f'Image {image_name}')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    ################ STEP 3 - Outlier Approach ################
    first_round_band_analysis(super_groups, band_reference_dict, vertical_threshold)
    ################ STEP 4 - Direct Approach ################
    second_round_band_analysis(super_groups, band_reference_dict, vertical_threshold)

    if full_band_info_plots:
        # Display boundaries on the original image and blob centroids with index
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        ax1.imshow(original_image)
        # Plot original image with boundaries and blob centroids
        ax2.imshow(1 - binary_image, cmap='gray')

        for i, band_gp in enumerate(super_groups.values()):
            if len(band_gp['BrokenBandsv2']) > 0:
                flat_bb2_list = [inner_band for gp in band_gp['BrokenBandsv2'] for inner_band in gp]
            else:
                flat_bb2_list = []
            for band in band_gp['bands']:
                color = 'g' if band in band_gp['SmallBands'] else 'r' if band in band_gp['BrokenBands'] else 'purple' if band in flat_bb2_list else 'b'
                if band in flat_bb2_list and band in band_gp['BrokenBands']:
                    color = 'orange'
                elif band in flat_bb2_list and band in band_gp['SmallBands']:
                    color = 'black'
                elif band in flat_bb2_list and band in band_gp['SmallBands'] and band in band_gp['BrokenBands']:
                    color = 'yellow'
                ax2.scatter(band_reference_dict[band]['centroid'][1], band_reference_dict[band]['centroid'][0], c=color,
                            marker='+', linewidth=1.5, zorder=3)

        ax2.set_title(f'First Clean-Up Round Band Classifications')
        plt.suptitle(f'Image {image_name}')
        plt.tight_layout()
        if isinstance(save_to_folder, str):
            plt.savefig(os.path.join(save_to_folder, image_name), dpi=300)
        if direct_vis:
            plt.show()
        plt.close(fig)

    # for now, band-filling is only done using the v2 broken bands.  In the future, can consider improving the outlier approach too.
    updated_image = fill_in_bands(super_groups, binary_image, labeled_image)

    if debugging_plots:
        # Display boundaries on the original image and blob centroids with index
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        # Plot original image with boundaries and blob centroids
        ax1.imshow(1 - binary_image, cmap='gray')
        for contour in contours:
            ax1.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=1.2, zorder=1)
        all_blob_centroids = np.array([prop.centroid for prop in props])
        for i, centroid in enumerate(all_blob_centroids):
            ax1.scatter(centroid[1], centroid[0], c='b', marker='+', linewidth=1.5, zorder=3)
        ax1.set_title(f'Original Image with Centroids')

        ax2.imshow(1 - updated_image, cmap='gray')
        ax2.set_title(f'New image after second clean-up round')

        plt.suptitle(f'Image {image_name}')
        plt.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    folder_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/probability_map_samples/v1/direct_model_outputs'
    folder_type = 'probability_maps'

    folder_path = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_evaluation/full_test_set_eval/unet_dec_21'
    folder_type = 'direct_images'

    # uncomment here to output to file
    # output_path = '/Users/matt/Desktop/output_plots_v2'
    # debugging_plots = False
    # full_band_info_plots = True
    # direct_plot = False

    # comment here to remove debugging plots
    output_path = False
    debugging_plots = True
    full_band_info_plots = True
    direct_plot = True

    if folder_type == 'probability_maps':
        for image_folder in os.listdir(folder_path):
            if not os.path.isdir(os.path.join(folder_path, image_folder)):  # non-folder files
                continue
            create_dir_if_empty(os.path.join(output_path, image_folder))

            ################ STEP 1 - Image loading and centroid definition ################
            # Standard image load, label and process into an image with a unique integer for each object
            binary_image = np.load(os.path.join(folder_path, image_folder, 'seg_mask_one_hot.npy'))
            binary_image = binary_image.argmax(axis=0)
            binary_image = binary_image.astype(np.uint8)
            full_postprocessing_analysis(binary_image, debugging_plots=debugging_plots, image_name=image_folder,
                                         full_band_info_plots=full_band_info_plots, save_to_folder=output_path,
                                         direct_vis=direct_plot)
    else:
        all_images = extract_image_names_from_folder(folder_path)
        binary_outputs = [im for im in all_images if 'map_only' in im]
        for im_name in tqdm(binary_outputs):
            overlaid_image_name = im_name.replace('_map_only.png', '.png')
            binary_image = imageio.v2.imread(os.path.join(folder_path, im_name)).argmax(axis=-1)
            overlaid_image = imageio.v2.imread(os.path.join(folder_path, overlaid_image_name))
            full_postprocessing_analysis(binary_image, image_name=os.path.basename(im_name).replace('.png', ''),
                                         original_image=overlaid_image, debugging_plots=debugging_plots,
                                         full_band_info_plots=full_band_info_plots, save_to_folder=output_path,
                                         direct_vis=direct_plot)

