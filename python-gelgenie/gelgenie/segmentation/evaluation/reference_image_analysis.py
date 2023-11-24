import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.color import label2rgb
from torch.utils.data import DataLoader
from tqdm import tqdm

from gelgenie.segmentation.data_handling.dataloaders import ImageDataset
from gelgenie.segmentation.evaluation.core_functions import ref_data_folder, model_predict_and_process
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty

# TODO: document and split up functions, and add more analysis (e.g. correlation comparison)


def segmentation_accuracy_comparison(data_folder, datafile):
    analysis_folder = os.path.join(data_folder, datafile.split('.')[0] + '_analysis')
    input_file = os.path.join(data_folder, datafile)
    create_dir_if_empty(analysis_folder)
    data = pd.read_csv(input_file)
    band_ids = range(1, len(data['Reference Data']) + 1)
    fig, ax = plt.subplots(3, 1, figsize=(12, 15))
    colors = ['g', 'b', 'r']

    ax[0].plot(band_ids, data['Reference Data'], label='GT', c=colors[0])
    ax[0].plot(band_ids, data['Gel Analyzer Results'], label='Gelanalyzer', c=colors[1])
    ax[0].plot(band_ids, data['Segmentation Results'], label='Segmentation', c=colors[2])
    ax[0].set_title('Raw Intensities')
    ax[0].set_xlabel('Band ID')
    ax[0].set_ylabel('Intensity')
    ax[0].set_xticks(range(len(data['Reference Data'])))
    ax[0].legend()

    r_ratio = [r/data['Reference Data'][0] for r in data['Reference Data']]
    g_ratio = [r/data['Gel Analyzer Results'][0] for r in data['Gel Analyzer Results']]
    s_ratio = [r/data['Segmentation Results'][0] for r in data['Segmentation Results']]

    ax[1].plot(band_ids, r_ratio, label='GT', c=colors[0])
    ax[1].plot(band_ids, g_ratio, label='Gelanalyzer', c=colors[1])
    ax[1].plot(band_ids, s_ratio, label='Segmentation', c=colors[2])
    ax[1].set_title('Intensity ratios w.r.t. first band')
    ax[1].set_xlabel('Band ID')
    ax[1].set_ylabel('Ratio')
    ax[1].set_xticks(range(len(data['Reference Data'])))
    ax[1].legend()

    ax[2].plot(band_ids, [np.abs(g-r) for r, g in zip(r_ratio, g_ratio)], label='Gelanalyzer', c=colors[1])
    ax[2].plot(band_ids, [np.abs(s-r) for r, s in zip(r_ratio, s_ratio)], label='Segmentation', c=colors[2])
    ax[2].set_title('Errors w.r.t. GT (intensity ratios)')
    ax[2].set_xlabel('Band ID')
    ax[2].set_ylabel('Error')
    ax[2].set_xticks(range(len(data['Reference Data'])))
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(analysis_folder, 'segmentation_comparison.pdf'))
    plt.close(fig)

    gelanalyzer_masses = [g * data['Reference Data'][0] for g in g_ratio]
    seg_masses = [s * data['Reference Data'][0] for s in s_ratio]

    plt.figure()
    plt.plot(band_ids, data['Reference Data'], label='GT', c=colors[0])
    plt.plot(band_ids, gelanalyzer_masses, label='Gelanalyzer', c=colors[1])
    plt.plot(band_ids, seg_masses, label='Segmentation', c=colors[2])
    plt.title('Predicted DNA Masses')
    plt.xlabel('Band ID')
    plt.ylabel('DNA Mass (ng)')
    plt.xticks(range(len(data['Reference Data'])))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_folder, 'mass_comparison.pdf'))
    plt.close(fig)


def standard_ladder_analysis(model, output_folder):
    dataset = ImageDataset(ref_data_folder, 1, padding=True)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    easy_image = 'ladder_reference_gel'
    hard_image = 'overheated_ref'

    # preparing model outputs, including separation of different bands and labelling
    for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['image_name'][0] == easy_image:
            e_im = batch['image'].detach().squeeze().cpu().numpy()
            _, e_mask = model_predict_and_process(model, batch['image'])
            e_labels, _ = ndi.label(e_mask[:, 300:900, 0:500].argmax(axis=0))
            e_im_snapshot = e_im[300:900, 0:500]
            e_direct_labels = label2rgb(e_labels, image=e_im_snapshot)
        elif batch['image_name'][0] == hard_image:
            h_im = batch['image'].detach().squeeze().cpu().numpy()
            _, h_mask = model_predict_and_process(model, batch['image'])
            h_labels, _ = ndi.label(h_mask[:, 400:1200, 0:600].argmax(axis=0))
            h_im_snapshot = h_im[400:1200, 0:600]
            h_direct_labels = label2rgb(h_labels, image=h_im_snapshot)

    # results preview
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(e_im_snapshot, cmap='gray')
    ax[0, 1].imshow(e_direct_labels)
    ax[1, 0].imshow(h_im_snapshot, cmap='gray')
    ax[1, 1].imshow(h_direct_labels)

    ax[0, 0].set_title('Reference Image')
    ax[0, 1].set_title('Segmented Image')
    ax[0, 0].set_ylabel('Well-behaved ladder')
    ax[1, 0].set_ylabel('Overheated ladder')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.suptitle('Segmentation results on first ladder of reference images')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'standard_segmentation_results.pdf'))

    # individual label preview
    max_images_per_row = 10
    segmentation_results = []
    for im_labels, im_snapshot, filename in zip([e_labels, h_labels], [e_im_snapshot, h_im_snapshot],
                                                ['easy ladder', 'hard ladder']):
        total_labels = len(np.unique(im_labels))
        rsize = math.ceil(total_labels / max_images_per_row)
        csize = max_images_per_row
        fig, ax = plt.subplots(rsize, csize, figsize=(csize * 2, rsize * 2))

        for i in range(rsize):
            for j in range(csize):
                ax[i, j].axis('off')

        seg_intensity_sum = []

        for sel_val in np.unique(im_labels):
            if sel_val == 0:  # this is just background
                continue

            pixel_intensities = np.sum(im_snapshot[im_labels == sel_val])
            seg_intensity_sum.append(pixel_intensities)

            band_focus = np.zeros(im_snapshot.shape)
            band_focus[im_labels == sel_val] = sel_val

            row = math.floor((sel_val - 1) / max_images_per_row)
            col = (sel_val - 1) % max_images_per_row

            ax[row, col].imshow(band_focus)
            ax[row, col].set_title('Band %s,\n Intensity %s' % (sel_val, pixel_intensities))

        plt.suptitle('Ordering of each band label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, '%s band order.pdf' % filename))
        segmentation_results.append(seg_intensity_sum)

    # exporting of segmentation results for manual validation
    gel_analyzer_harcoded = [
        [507, 341, 524, 400, 337, 1732, 536, 814, 442, 1552, np.nan, np.nan, np.nan, np.nan, 1283, 256, 231, 125, 371],
        [834, 707, 946, 772, 575, 1847, 716, 1119, 743, 1805, 371, 344, 292, 314, 1840, 628, 406, np.nan, np.nan]]
    reference_values_harcoded = [40, 40, 48, 40, 32, 120, 40, 57, 45, 122, 34, 31, 27, 23, 124, 49, 37, 32, 61]

    combined_easy_data = {'Reference Data': reference_values_harcoded, 'Segmentation Results': segmentation_results[0],
                          'Gel Analyzer Results': gel_analyzer_harcoded[0]}
    combined_hard_data = {'Reference Data': reference_values_harcoded, 'Segmentation Results': segmentation_results[1],
                          'Gel Analyzer Results': gel_analyzer_harcoded[1]}
    easy_df = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in combined_easy_data.items()}) # dealing with mismatched list lengths
    hard_df = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in combined_hard_data.items()})

    easy_df.to_csv(os.path.join(output_folder, 'easy_ladder_segmentation_results.csv'), index=False)
    hard_df.to_csv(os.path.join(output_folder, 'hard_ladder_segmentation_results.csv'), index=False)
