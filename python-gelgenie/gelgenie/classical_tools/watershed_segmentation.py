from collections import defaultdict

import skimage

from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.filters import sobel

import numpy as np
import matplotlib.pyplot as plt

from gelgenie.segmentation.data_handling.dataloaders import ImageMaskDataset
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, index_converter
from tqdm import tqdm

import os
from torch.utils.data import DataLoader


def mask_expansion(original_image, labeled_image, next_bg):
    """
    Expands the areas of found bands to those adjacent pixels which
    would potentially make bands on the next pass.
    :param original_image: Raw gel image (ndarray, 16-bit).
    :param labeled_image: Image with segmented bands.
    :param next_bg: Next background threshold value that will be considered in the next pass.

    :return: Image mask containing areas where bands are already present (or have been expanded via flood algorithm)
    """

    # Get properties of the labeled bands (will use to find centroid of bands)
    props = skimage.measure.regionprops(labeled_image)

    # Initialize output image
    output_image = np.zeros_like(original_image)

    for band in props:
        # Define centroid
        centroid = band.centroid
        # Set seed point to closest integer coord pair to centroid
        seed_point = (round(centroid[0]), round(centroid[1]))

        # Initialize flooding image as full of zeros
        image_for_flood = np.zeros_like(original_image)

        # Set every pixel in flooding image that could possibly make a new band next pass to 1
        image_for_flood[original_image > next_bg] = 1

        # Use floodfill on the flooding image using the found bands
        # This will ideally exclude any areas that neighbour found bands from being considered as bands
        output_image_part = skimage.morphology.flood(image_for_flood, seed_point, tolerance=None)

        # Add the excluded areas of each iteration to final excluded area
        output_image += output_image_part

    return output_image


def watershed_seg_direct(image, sure_fg, sure_bg):
    """
    Apply watershed algorithm to find bands in a given image.
    :param image: numpy array containing image raw data (grayscale, 16-bit)
    :param sure_fg: Threshold value above which a band is definitely present.
    :param sure_bg: Threshold value under which a band is definitely not present.
    :param verbose:  Set to true to print out figures containing results immediately.
    :return: Input markers, elevation map and output segmentation (direct and overlayed)
    """

    # Use Sobel filter on original image to find elevation map
    elevation_map = sobel(image)

    # Define markers for the background and foreground of the image
    markers = np.zeros_like(image)
    markers[image < sure_bg] = 1
    markers[image > sure_fg] = 2

    markers_plot = np.ones_like(image) * 120  # visual mid-range pixel to aid viewing

    markers_plot[image < sure_bg] = 0  # will show up as purple
    markers_plot[image > sure_fg] = 255  # will show up as white
    markers_plot = 255 - markers_plot

    # Apply the watershed algorithm itself, using the elevation map and markers
    segmentation = skimage.segmentation.watershed(elevation_map, markers)

    # Fill holes and relabel bands, giving each a unique label
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_bands, _ = ndi.label(segmentation)

    # Overlay labels on original image
    image_label_overlay = label2rgb(labeled_bands, image=image)

    return markers_plot, elevation_map, labeled_bands, image_label_overlay


def multistep_watershed(img, sure_fg, sure_bg, repetitions, background_jump=0.005):
    """
    Applies the watershed algorithm to segment bands and optionally
    repeats the analysis multiple times to attempt to find more bands.
    :param img: Numpy array containing image data (2D, float)
    :param sure_fg: Foreground threshold value
    :param sure_bg: Background threshold value
    :param repetitions: Number of times to repeat analysis
    :param background_jump: Absolute value to shift background/foreground on each iteration
    :return: Dict containing intermediate images during analysis, the final segmentation map and an RGB image
    containing the original data and an overlay of the found bands.
    """

    # Create copy of loaded image to apply mask to
    working_img = img.copy()

    # Create template for labels (each integer label corresponds to band ID)
    final_segmentation = np.zeros_like(img, dtype="int32")

    working_image_dict = defaultdict(list)

    # Repeat the watershed algorithm "repetitions" times
    for i in range(0, repetitions):
        # Run watershed algorithm

        markers_plot, elevation_map, labeled_bands, image_label_overlay = watershed_seg_direct(working_img, sure_fg,
                                                                                               sure_bg)

        # Add found bands from this iteration to all found bands
        final_segmentation += labeled_bands

        # Create mask
        mask = mask_expansion(working_img, labeled_bands, next_bg=sure_bg - background_jump)

        # Apply mask (removing areas of image which should no longer be explored)
        working_img[mask > 0] = 0

        # Reduce fg and bg values on each pass
        sure_fg -= background_jump
        sure_bg -= background_jump

        sure_bg = max(sure_bg, 0.01) # if background dips below 0, then the algorithm will start to fail

        working_image_dict[i] = [markers_plot, elevation_map, labeled_bands, image_label_overlay, mask,
                                 working_img.copy()]

    # Relabel bands to ensure correct labelling
    labeled_fbands, _ = ndi.label(final_segmentation)

    # Overlay found bands on original image
    final_overlay = label2rgb(labeled_fbands, image=img, bg_label=0, bg_color=[0, 0, 0])

    return working_image_dict, final_overlay, final_segmentation


def watershed_analysis(input_image, image_name=None, intermediate_plot_folder=None, repetitions=1, background_jump=0.08,
                       foreground_jump=0.25, use_multiotsu=False):
    """
    Conducts watershed analysis on input image, attempting to do multiple runs if possible.
    :param input_image: 2D numpy array containing image data (ideally in float format)
    :param image_name: Name of image to use for saving intermediate output
    :param intermediate_plot_folder: Output folder to save working images as a diagnostic
    :param repetitions: Number of times to attempt to repeat watershed analysis
    :param background_jump: Proportion to reduce background threshold by on each iteration (max 1, min 0)
    :param foreground_jump: Absolute value to add to background threshold to set arbitrary foreground threshold
    :param use_multiotsu: Use a multi-otsu threshold to set the foreground/background values rather than an absolute jump
    :return: Fully segmented image and RGB array containing the original image with the segmented bands overlaid
    """
    if use_multiotsu:
        thresholds = skimage.filters.threshold_multiotsu(input_image)
        otsu_bg, otsu_fg = thresholds[0], thresholds[1]
    else:
        otsu_bg = skimage.filters.threshold_otsu(input_image)
        otsu_fg = otsu_bg + foreground_jump

    working_image_dict, final_overlay, segmentation_map = multistep_watershed(input_image, otsu_fg, otsu_bg, repetitions,
                                                                            background_jump=background_jump)

    if intermediate_plot_folder:
        fig, ax = plt.subplots(len(working_image_dict.keys()), 5, figsize=(15, 10))
        index_conversion = True if repetitions > 1 else False
        ipr = 5  # images per row
        for i in range(len(working_image_dict.keys())):
            ax[index_converter((i*ipr) + 0, ipr, index_conversion)].axis('off')
            ax[index_converter((i*ipr) + 1, ipr, index_conversion)].axis('off')
            ax[index_converter((i*ipr) + 2, ipr, index_conversion)].axis('off')
            ax[index_converter((i*ipr) + 3, ipr, index_conversion)].axis('off')
            ax[index_converter((i*ipr) + 4, ipr, index_conversion)].axis('off')

            ax[index_converter((i*ipr) + 0, ipr, index_conversion)].imshow(working_image_dict[i][0], cmap='Purples')
            ax[index_converter((i*ipr) + 1, ipr, index_conversion)].imshow(working_image_dict[i][1], cmap='Purples_r')
            ax[index_converter((i*ipr) + 2, ipr, index_conversion)].imshow(working_image_dict[i][3])
            ax[index_converter((i*ipr) + 3, ipr, index_conversion)].imshow(working_image_dict[i][4], cmap='Purples_r')
            ax[index_converter((i*ipr) + 4, ipr, index_conversion)].imshow(working_image_dict[i][5], cmap='Purples')

            ax[index_converter((i*ipr) + 0, ipr, index_conversion)].set_title('Markers')
            ax[index_converter((i*ipr) + 1, ipr, index_conversion)].set_title('Elevation Map')
            ax[index_converter((i*ipr) + 2, ipr, index_conversion)].set_title('Watershed Segmentation')
            ax[index_converter((i*ipr) + 3, ipr, index_conversion)].set_title('Mask Removal')
            ax[index_converter((i*ipr) + 4, ipr, index_conversion)].set_title('New Working Image')

        plt.tight_layout()
        plt.savefig(os.path.join(intermediate_plot_folder, f'{image_name}_watershed_intermediates.png'),
                    pad_inches=0, bbox_inches='tight', transparent=True, dpi=300)
        plt.close()

    return final_overlay, segmentation_map


def multiotsu_analysis(input_image, image_name=None, intermediate_plot_folder=None):
    """
    Direct segmentation using Otsu thresholding to separate various levels from an image,
    then extract the highest level as the foreground.
    :param input_image: 2D numpy array containing image data (ideally in float format)
    :param image_name: Name of image to use for saving intermediate output
    :param intermediate_plot_folder: Output folder to save fully thresholded image as a diagnostic
    :return: Final segmented image
    """
    thresholds = skimage.filters.threshold_multiotsu(input_image)  # extracts pixel thresholds from image (2 per image)
    # Generate regions
    regions = np.digitize(input_image, bins=thresholds)  # assigns a threshold to each pixel in the image

    if intermediate_plot_folder:  # plots fully classified image as a diagnostic
        plt.figure(figsize=(10, 10))
        plt.imshow(regions)
        plt.axis('off')
        plt.savefig(os.path.join(intermediate_plot_folder, f'{image_name}_multiotsu_intermediate.png'),pad_inches=0, bbox_inches='tight', transparent=True, dpi=300)
        plt.close()

    regions[regions < len(thresholds)] = 0  # retain only the highest threshold (foreground)

    return regions


if __name__ == '__main__':
    input_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/test_images'
    mask_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/test_masks'
    output_folder = '/Users/matt/Desktop/test_watershed'
    create_dir_if_empty(output_folder)

    dataset = ImageMaskDataset(input_folder, mask_folder, 1, padding=False, individual_padding=True,
                               minmax_norm=False)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        np_image = batch['image'].detach().squeeze().cpu().numpy()
        gt_mask = batch['mask']
        final_overlay, segmentation_map = watershed_analysis(np_image, batch["image_name"][0],
                                                             intermediate_plot_folder=output_folder,
                                                             repetitions=1, background_jump=0.08,
                                                             use_multiotsu=True)
        otsu_labelled = multiotsu_analysis(np_image, batch["image_name"][0], intermediate_plot_folder=output_folder)
        plt.figure(figsize=(10, 10))
        plt.imshow(final_overlay)
        plt.xticks([]), plt.yticks([])
        plt.title('Final Segmentation')
        plt.savefig(os.path.join(output_folder, f'{batch["image_name"][0]}_final.png'), dpi=300)
        plt.close()
