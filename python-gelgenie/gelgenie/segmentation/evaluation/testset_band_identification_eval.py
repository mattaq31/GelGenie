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
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import imageio
from scipy import ndimage as ndi

from gelgenie.segmentation.data_handling.dataloaders import ImageMaskDataset
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, index_converter

eval_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/quantitative_results/full_test_set_eval'
output_folder = '/Users/matt/Desktop/band_level_accuracy_analysis'
image_folders = [
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/test_images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/test_images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/test_images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/quantitation_ladder_gels/test_images',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/test_images']
gt_mask_folders = [
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/test_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/test_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/nathan_gels/test_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/quantitation_ladder_gels/test_masks',
    '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/lsdb_gels/test_masks']
create_dir_if_empty(output_folder)

eval_models = ['unet_dec_21', 'nnunet_final_fold_0', 'unet_dec_21_lsdb_only', 'watershed', 'multiotsu']

dataset = ImageMaskDataset(image_folders, gt_mask_folders,
                           1, padding=False, individual_padding=False, minmax_norm=False)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

plotting = True

gel_level_stats = {}

for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

    # ground truth image and mask setup
    image_name = batch['image_name'][0]
    gel_level_stats[image_name] = {'identified': [], 'error': [], 'multi_band': []}
    base_image = batch['image'].numpy().squeeze()
    gt_mask = batch['mask'].numpy().squeeze()
    gt_labels, num_bands = ndi.label(gt_mask)

    # figure prep
    if plotting:
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # running through all selected models
    for model_index, model in enumerate(eval_models):

        mask_path = os.path.join(eval_folder, model, '%s_map_only.png' % image_name)
        model_mask = imageio.v2.imread(mask_path)[:, :, 0]
        model_mask[model_mask > 0] = 1  # binarising everything (removing colours)
        model_labels, _ = ndi.label(model_mask)  # final unique ID labels

        # running through each individual band
        lband_identified = []
        lmulti_band = []
        lband_error = []
        lintensity_error = []
        for i in range(1, num_bands + 1):

            local_mask = gt_mask.copy()  # makes a copy of the local GT mask
            local_mask[gt_labels != i] = 0
            local_mask[local_mask > 0] = 1  # removes all other bands and retains only the targeted band info
            total_pixels = np.sum(local_mask)  # all +ve pixels are 1, so can just sum
            true_intensity = np.sum(base_image[local_mask > 0])

            query_band_mask = model_labels.copy()

            band_identified = False  # by default assume band not found
            multi_band = False
            band_error = 1
            intensity_error = 1
            predicted_intensity = 0
            predicted_pixels = 0
            predicted_band_ids = []

            # run through each band ID that is present in the GT mask area
            for unique_index, band_id in enumerate(np.unique(query_band_mask[gt_labels == i])):
                if band_id == 0:  # just background
                    continue
                band_identified = True  # if we've arrived here then a band has been identified in some form
                if unique_index > 1:  # this means the band was split into multiple parts (problematic)
                    multi_band = True
                predicted_intensity += np.sum(base_image[query_band_mask == band_id])  # summing all pixels in the band
                predicted_pixels += np.sum(query_band_mask == band_id)  # getting the positive pixel quantity
                predicted_band_ids.append(band_id)

            if band_identified:  # calculate segmentation errors
                band_error = np.abs(total_pixels - predicted_pixels) / total_pixels
                intensity_error = np.abs(true_intensity - predicted_intensity) / true_intensity
                # if intensity_error > 1.5:
                #
                #     plt.imshow(base_image, cmap='gray')
                #
                #     for band_id in predicted_band_ids:
                #         plt.imshow(query_band_mask == band_id, alpha=0.5)
                #     plt.show()
                #     plt.imshow(base_image, cmap='gray')
                #     plt.imshow(local_mask, alpha=0.5)
                #     plt.show()

            if band_error > 1:  # error too large - assumed not identified
                lband_identified.append(False)
            else:
                lband_identified.append(band_identified)
            lband_error.append(band_error)
            lintensity_error.append(intensity_error)
            lmulti_band.append(multi_band)

        gel_level_stats[image_name]['identified'].append(lband_identified)
        gel_level_stats[image_name]['error'].append(lintensity_error)
        gel_level_stats[image_name]['multi_band'].append(lmulti_band)

        if plotting:
            colours = []
            for i in range(num_bands):
                if lband_identified[i]:
                    if lmulti_band[i]:
                        colours.append('red')
                    else:
                        colours.append('green')
                else:
                    colours.append('black')

            plot_index = index_converter(model_index, 3)
            # plt.scatter(range(num_bands), lband_error, label='Pixel Error', c=colours)
            ax[plot_index].scatter(range(num_bands), lintensity_error, label='Intensity Error', c=colours)
            ax[plot_index].set_title(model)
            ax[plot_index].set_xlabel('Band Number')
            ax[plot_index].set_ylabel('Percentage Error')
            ax[plot_index].set_ylim([-0.05, 1.05])

    if plotting:
        ax[index_converter(5, 3)].axis('off')
        # plt.legend()
        plt.tight_layout()
        # plt.savefig(os.path.join(output_folder, '%s_error_plot.png' % image_name), dpi=300)
        plt.show()
# pickle.dump(gel_level_stats, open(os.path.join(output_folder, 'gel_level_stats_no_ceiling.pkl'), 'wb'))
