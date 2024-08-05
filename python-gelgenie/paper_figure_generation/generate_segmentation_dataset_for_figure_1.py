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
import pickle

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset, ImageMaskDataset
from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder, index_converter
from tqdm import tqdm
import numpy as np
from scipy import ndimage as ndi
from skimage.color import label2rgb
import cv2
import pandas as pd
import skimage
from collections import defaultdict
import seaborn as sns
import imageio
import re
from scipy.stats import linregress
import math


def band_label_and_sort(band_labels):
    # top-left non-zero pixel is always labelled with a 1

    props = skimage.measure.regionprops(band_labels)
    label_id_mapping = {}

    all_band_x_lhs = []
    all_band_x_rhs = []
    all_band_labels = []
    all_band_x_cent = []
    all_band_y_cent = []
    for band in props:
        all_band_x_lhs.append(band.bbox[1])
        all_band_x_rhs.append(band.bbox[3])
        all_band_labels.append(band.label)
        all_band_x_cent.append(band.centroid[1])
        all_band_y_cent.append(band.centroid[0])

    lane_counter = 1
    while len(all_band_y_cent) > 0:
        lhs_band_pos = np.argmin(all_band_x_lhs)
        lhs_low = all_band_x_lhs[lhs_band_pos]
        lhs_high = all_band_x_rhs[lhs_band_pos]

        curr_lane_bands = []
        for index, (l, r, c, label) in enumerate(zip(all_band_x_lhs, all_band_x_rhs, all_band_x_cent, all_band_labels)):
            if lhs_high > c > lhs_low:
                curr_lane_bands.append(index)
            elif l < lhs_high < r:
                curr_lane_bands.append(index)
            elif l < lhs_low < r:
                curr_lane_bands.append(index)
        curr_lane_bands = sorted(curr_lane_bands, key=lambda x: all_band_y_cent[x])

        bands_to_pop = []
        for band_index, l in enumerate(curr_lane_bands):
            label_id_mapping[all_band_labels[l]] = [lane_counter, band_index+1]
            bands_to_pop.append(l)
        all_band_labels = [all_band_labels[l] for l in range(len(all_band_labels)) if l not in bands_to_pop]
        all_band_x_lhs = [all_band_x_lhs[l] for l in range(len(all_band_x_lhs)) if l not in bands_to_pop]
        all_band_x_rhs = [all_band_x_rhs[l] for l in range(len(all_band_x_rhs)) if l not in bands_to_pop]
        all_band_x_cent = [all_band_x_cent[l] for l in range(len(all_band_x_cent)) if l not in bands_to_pop]
        all_band_y_cent = [all_band_y_cent[l] for l in range(len(all_band_y_cent)) if l not in bands_to_pop]

        lane_counter += 1
    return label_id_mapping


def dilate_image(image, iterations=1):
    kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel for dilation
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image


def erode_image(image, iterations=1):
    kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel for dilation
    dilated_image = cv2.erode(image, kernel, iterations=iterations)
    return dilated_image


def random_dilate_erode(image, max_iterations=3):

    new_label_image = np.copy(image)
    for band in np.unique(image):
        if band == 0:
            continue
        band_only_image = np.copy(image)
        band_only_image[band_only_image != band] = 0
        iterations = np.random.randint(1, max_iterations+1)

        degrade_path = np.random.choice([0, 1, 2])
        if degrade_path == 0:
            new_dilation = dilate_image(band_only_image, iterations=iterations)
        elif degrade_path == 1:
            new_dilation = erode_image(band_only_image, iterations=iterations)
        else:
            continue

        new_label_image[new_label_image == band] = 0
        new_label_image[new_dilation > 0] = band

    return new_label_image


def dilation_erosion_analysis(np_image, bg_image, labels, erosion_iters=2, dilation_iters=2, random_iters=2, request_bc=False):

    data_dict = defaultdict(list)
    for iter in range(erosion_iters):
        adjusted_labels = erode_image(labels, iterations=iter+1)
        for label in np.unique(labels):
            if label == 0:
                continue
            data_dict[f'erosion {iter+1}'].append(np.sum(np_image[np.where(adjusted_labels == label)]))
            if request_bc:
                data_dict[f'erosion {iter+1} BC'].append(np.sum(np_image[np.where(adjusted_labels == label)]) - np.sum(bg_image[np.where(adjusted_labels == label)]))

    for iter in range(dilation_iters):
        adjusted_labels = dilate_image(labels, iterations=iter + 1)
        for label in np.unique(labels):
            if label == 0:
                continue
            data_dict[f'dilation {iter + 1}'].append(np.sum(np_image[np.where(adjusted_labels == label)]))
            if request_bc:
                data_dict[f'dilation {iter+1} BC'].append(np.sum(np_image[np.where(adjusted_labels == label)]) - np.sum(bg_image[np.where(adjusted_labels == label)]))

    for iter in range(random_iters):
        adjusted_labels = random_dilate_erode(labels, max_iterations=iter + 1)
        for label in np.unique(labels):
            if label == 0:
                continue
            data_dict[f'random adjustment {iter + 1}'].append(np.sum(np_image[np.where(adjusted_labels == label)]))
            if request_bc:
                data_dict[f'random adjustment {iter+1} BC'].append(np.sum(np_image[np.where(adjusted_labels == label)]) - np.sum(bg_image[np.where(adjusted_labels == label)]))

    return data_dict


def load_ga_csv_files_from_folders(parent_folder, prefix="ga_"):
    """
    Load CSV files from folders and create a dictionary of DataFrames with prefixed names.

    Parameters:
    - parent_folder (str): The path to the parent folder containing numbered subfolders.
    - prefix (str): The prefix to be added to the names of the DataFrames (default is "ga_").

    Returns:
    - dataframes (dict): A dictionary where keys are prefixed folder names, and values are corresponding DataFrames.
    """
    dataframes = {}  # Dictionary to store DataFrames

    # Iterate through folders in the parent folder
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)

        # Check if the item in the parent folder is a directory
        if os.path.isdir(folder_path):
            csv_file_path = os.path.join(folder_path, "collated_data_with_band_quality.csv")

            # Check if "collated_data_with_band_quality.csv" exists in the folder
            if os.path.isfile(csv_file_path):
                # Read the CSV file into a DataFrame
                dataframe = pd.read_csv(csv_file_path)
                dataframe = dataframe.rename(columns={'Raw Volume':'GA-Raw-Vol', 'Background Corrected Volume':'GA-BC-Vol'})

                # Add the prefix to the folder name and use it as the key in the dictionary
                prefixed_folder_name = f"{prefix}{folder_name}"
                dataframes[prefixed_folder_name] = dataframe

    return dataframes


input_folder = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/base_data']
base_images = extract_image_names_from_folder(input_folder[0])
mask_folder = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/qupath_data/james_data_v3_fixed_global/indiv_label_segmaps']
mask_images = extract_image_names_from_folder(mask_folder[0])
ga_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/gelanalyzer'
all_ga_data = load_ga_csv_files_from_folders(ga_folder)

background_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/qupath_data/james_data_v3_fixed_global/background_images'
background_images = extract_image_names_from_folder(background_folder)

thermo_ref = [20] * 9 + [70] + [30] * 7 + [40]
NEB_ref = [40, 40, 48, 40, 32, 120, 40, 57, 45, 122, 34, 31, 27, 23, 124, 49, 37, 32, 61]

seed = 8
np.random.seed(seed)
data_df = defaultdict(list)

full_dataset = {}

for im_index, (base_image, mask_image, background_image) in enumerate(tqdm(zip(base_images, mask_images, background_images), total=len(base_images))):

    labels = imageio.v2.imread(mask_image)
    np_image = imageio.v2.imread(base_image)
    bg_image = imageio.v2.imread(background_image)
    name = os.path.basename(base_image).split('.')[0]

    if '25' in name or '26' in name or '27' in name or '28' in name or '30' in name:
        continue  # Problematic images; ignore

    if 'Thermo' in name:
        ref_ladder = thermo_ref
    else:
        ref_ladder = NEB_ref

    labels = labels.astype(np.uint16)
    band_mappers = band_label_and_sort(labels)  # implements the same algorithm as in GelGenie, so that results should match

    data_df = defaultdict(list)
    for label in np.unique(labels):
        if label == 0:
            continue
        intensity_sum = np.sum(np_image[np.where(labels == label)])
        intensity_average = np.average(np_image[np.where(labels == label)])
        intensity_std = np.std(np_image[np.where(labels == label)])

        rb_sum = np.sum(np_image[np.where(labels == label)]) - np.sum(bg_image[np.where(labels == label)])

        band_pos = np.where(labels == label)
        band_width = np.max(band_pos[1]) - np.min(band_pos[1]) + 1
        band_height = np.max(band_pos[0]) - np.min(band_pos[0]) + 1

        data_df['Raw'].append(intensity_sum)
        data_df['RB Corrected'].append(rb_sum)
        data_df['Lane ID'].append(band_mappers[label][0])
        data_df['Band ID'].append(band_mappers[label][1])
        data_df['Ref.'].append(ref_ladder[band_mappers[label][1] - 1])
        data_df['Pixel Count'].append(len(band_pos[0]))
        data_df['Band Width'].append(band_width)
        data_df['Band Height'].append(band_height)
        data_df['Pixel Average'].append(intensity_average)
        data_df['Pixel STD'].append(intensity_std)

    data_df = {**data_df, **dilation_erosion_analysis(np_image, bg_image, labels, erosion_iters=2, dilation_iters=2, random_iters=1, request_bc=True)}
    data_df = pd.DataFrame.from_dict(data_df)

    data_df = pd.merge(data_df, all_ga_data[f'ga_{im_index}'], on=['Lane ID', 'Band ID'])
    data_df.drop(columns=['Reliable Band'], inplace=True)
    figs_per_row = 3
    rows = math.ceil((len(np.unique(data_df['Lane ID']) + 1) / figs_per_row))
    if rows == 1:
        double_indexing = False
    else:
        double_indexing = True

    full_dataset[name] = data_df

    for col_index, column in enumerate(data_df.columns):
        if column == 'Lane ID' or column == 'Band ID' or column == 'Ref.':
            continue
        for lane in np.unique(data_df['Lane ID']):
            ref = data_df[data_df['Lane ID'] == lane]['Ref.']
            target = data_df[data_df['Lane ID'] == lane][column]


    # fig, ax = plt.subplots(rows, figs_per_row, figsize=(18, 15))
    #
    # all_corr_coeff = {}
    # color_wheel = ['b', 'g', 'r', 'k']
    # plot_col_index = 0
    # for col_index, column in enumerate(data_df.columns):
    #     if column == 'Lane ID' or column == 'Band ID' or column == 'Ref.':
    #         continue
    #     for lane in np.unique(data_df['Lane ID']):
    #         ref = data_df[data_df['Lane ID'] == lane]['Ref.']
    #         target = data_df[data_df['Lane ID'] == lane][column]
    #         slope, intercept, r_value, p_value, std_err = linregress(ref, target)
    #         ax[index_converter(lane-1, figs_per_row, double_indexing)].scatter(
    #                              data_df[data_df['Lane ID'] == lane]['Ref.'],
    #                              data_df[data_df['Lane ID'] == lane][column],
    #                              label=f'{column}, R2: {r_value**2:.3f}', c=color_wheel[plot_col_index])
    #
    #         ref_plot = np.linspace(np.min(ref), np.max(ref), num=10)
    #         ax[index_converter(lane-1, figs_per_row, double_indexing)].plot(ref_plot, slope * ref_plot + intercept, color=color_wheel[plot_col_index], linestyle='dotted')
    #         ax[index_converter(lane-1, figs_per_row, double_indexing)].legend()
    #         ax[index_converter(lane - 1, figs_per_row, double_indexing)].set_title(f'Lane {lane}')
    #         ax[index_converter(lane - 1, figs_per_row, double_indexing)].set_yscale('log')
    #
    #     plot_col_index += 1
    # plt.suptitle(name)
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # for norm_val in ['Orig Sum'] + [f'Dil {iter+1} Sum' for iter in range(9)]:  # normalizes by maximum value ( 0 < x <= 1)
    #     correlation_coefficients = data_df.groupby('Lane ID').apply(lambda x: x[norm_val].corr(x['Ref.']), include_groups=False)
    #     all_corr_coeff[norm_val] = np.average(correlation_coefficients)
    #
    #     if norm_val == 'Orig Sum':
    #         for lane in range(5):
    #             if lane >= len(correlation_coefficients):
    #                 continue
    #
    #             ref = data_df[data_df['Lane ID'] == lane + 1]['Ref.']
    #             target = data_df[data_df['Lane ID'] == lane + 1][norm_val]
    #
    #             slope, intercept, r_value, p_value, std_err = linregress(ref, target)
    #
    #             ax[lane].scatter(data_df[data_df['Lane ID'] == lane + 1]['Ref.'],
    #                              data_df[data_df['Lane ID'] == lane + 1][norm_val],
    #                              label=f'{norm_val}, Corr: {r_value**2:.3f}')
    #             ax[lane].plot(ref, slope * ref + intercept, color='red', linestyle='dotted')
    #             ax[lane].legend()
    #
pickle.dump(full_dataset, open('/Users/matt/Desktop/full_dataset.pkl', 'wb'))
