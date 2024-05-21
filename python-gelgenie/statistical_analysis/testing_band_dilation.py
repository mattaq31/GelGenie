import os.path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset, ImageMaskDataset
from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder
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


input_folder = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/base_data']
base_images = extract_image_names_from_folder(input_folder[0])
mask_folder = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/qupath_data/james_data_v3_fixed_global/indiv_label_segmaps']
mask_images = extract_image_names_from_folder(mask_folder[0])

reference_path = "/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/reference_ladder_masses.csv"

data_df = defaultdict(list)

thermo_ref = [20] * 9 + [70] + [30] * 7 + [40]
NEB_ref = [40, 40, 48, 40, 32, 120, 40, 57, 45, 122, 34, 31, 27, 23, 124, 49, 37, 32, 61]

seed = 5

for im_index, (base_image, mask_image) in enumerate(tqdm(zip(base_images, mask_images), total=len(base_images))):

    labels = imageio.v2.imread(mask_image)
    np_image = imageio.v2.imread(base_image)
    name = os.path.basename(base_image).split('.')[0]
    # if '10_Thermo' not in name:
    #     continue
    # if im_index < 13:
    #     continue

    if '25' in name or '26' in name or '27' in name or '28' in name or '30' in name:
        continue

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
        intensity_sum = np.sum(np_image[np.where(labels == label)])  # only raw data volume for now
        data_df['Orig Sum'].append(intensity_sum)
        data_df['Lane ID'].append(band_mappers[label][0])
        data_df['Band ID'].append(band_mappers[label][1])
        data_df['Ref.'].append(ref_ladder[band_mappers[label][1] - 1])

    for iter in range(9):  # for each iteration, dilates or erodes image to observe effect of errors in boundary estimation
        if iter < 4:
            dil_image = dilate_image(labels, iterations=iter+1)
        elif iter < 8:
            dil_image = erode_image(labels, iterations=iter-3)
            plt.imshow(label2rgb(dil_image, image=np_image), cmap='gray')
            plt.title(f'Iteration {iter+1}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'/Users/matt/Desktop/{name}_dil_{iter+1}.pdf', dpi=300)
            # plt.show()
            plt.close()

        if iter == 8:
            dil_image = random_dilate_erode(labels, max_iterations=2)

            # plt.imshow(label2rgb(dil_image, image=np_image), cmap='gray')
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(f'/Users/matt/Desktop/{name}_dil_{iter+1}.pdf', dpi=300)
            # plt.close()

        for label in np.unique(labels):
            if label == 0:
                continue

            dilated_sum = np.sum(np_image[np.where(dil_image == label)])
            data_df[f'Dil {iter+1} Sum'].append(dilated_sum)

    data_df = pd.DataFrame.from_dict(data_df)

    fig, ax = plt.subplots(1, 5, figsize=(18, 8))
    all_corr_coeff = {}
    for norm_val in ['Orig Sum'] + [f'Dil {iter+1} Sum' for iter in range(9)]:  # normalizes by maximum value ( 0 < x <= 1)
        correlation_coefficients = data_df.groupby('Lane ID').apply(lambda x: x[norm_val].corr(x['Ref.']), include_groups=False)
        all_corr_coeff[norm_val] = np.average(correlation_coefficients)

        if norm_val == 'Orig Sum':
            for lane in range(5):
                if lane >= len(correlation_coefficients):
                    continue

                ref = data_df[data_df['Lane ID'] == lane + 1]['Ref.']
                target = data_df[data_df['Lane ID'] == lane + 1][norm_val]

                slope, intercept, r_value, p_value, std_err = linregress(ref, target)

                ax[lane].scatter(data_df[data_df['Lane ID'] == lane + 1]['Ref.'],
                                 data_df[data_df['Lane ID'] == lane + 1][norm_val],
                                 label=f'{norm_val}, Corr: {r_value**2:.3f}')
                ax[lane].plot(ref, slope * ref + intercept, color='red', linestyle='dotted')
                ax[lane].legend()

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()
    plt.close()
    break
    # for norm_val in ['Orig Sum', 'Ref.'] + [f'Dil {iter+1} Sum' for iter in range(9)]:  # normalizes by maximum value ( 0 < x <= 1)
    #     min_values = data_df.groupby('Lane ID')[norm_val].transform('min')
    #     max_values = data_df.groupby('Lane ID')[norm_val].transform('max')
    #     data_df[norm_val] = data_df[norm_val] / max_values
    #
    # for norm_val in ['Orig Sum'] + [f'Dil {iter+1} Sum' for iter in range(9)]:  # normalizes by maximum value ( 0 < x <= 1)
    #     data_df[norm_val.replace('Sum', 'Error')] = np.abs(data_df[norm_val] - data_df['Ref.'])
    #     data_df[norm_val.replace('Sum', 'Per. Error')] = (np.abs(data_df[norm_val] - data_df['Ref.'])/data_df['Ref.'])*100
    #
    # df_melted = pd.melt(data_df,
    #                     value_vars=['Orig Error'] + [f'Dil {iter+1} Error' for iter in range(9)],
    #                     var_name='Values')
    #
    # ax = sns.boxplot(x='Values', y='value', hue='Values', data=df_melted, width=0.5)
    # ax.tick_params(axis='x', rotation=90)
    # plt.tight_layout()
    # plt.savefig(f'/Users/matt/Desktop/test_results_with_random/{name}_boxplot.pdf', dpi=300)
    # # plt.show()
    # plt.close()

    # print('Iter %s, Average original error: %s, Average dilated error: %s' % (iter, data_df['Orig. Per. Error'].mean(), data_df['Dil. Per. Error'].mean()))

    # data_df.hist(column=['Orig. Error', 'Dil. Error'], bins=100)
    # plt.show()
    # rgb_labels = label2rgb(labels, image=np_image)
    # plt.imshow(rgb_labels, cmap='gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    #
    #     plt.imshow(label2rgb(dil_image, image=np_image), cmap='gray')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f'/Users/matt/Desktop/{name}_dil_{iter+1}.pdf', dpi=300)
    #     plt.close()
    #
    # plt.imshow(label2rgb(labels, image=np_image), cmap='gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(f'/Users/matt/Desktop/{name}_orig.pdf', dpi=300)
    # plt.close()

    # if im_index == 2:
    #     break

# NEED TO THINK ABOUT WHAT THIS DATA MEANS AND HOW TO PRESENT IT!
