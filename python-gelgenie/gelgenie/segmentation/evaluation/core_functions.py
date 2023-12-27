from gelgenie.classical_tools.watershed_segmentation import watershed_analysis, multiotsu_analysis
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset, ImageMaskDataset
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, index_converter
from gelgenie.segmentation.helper_functions.dice_score import multiclass_dice_coeff

import os
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.color import label2rgb
from tqdm import tqdm
import math
import imageio
import numpy as np
import itertools
from collections import defaultdict
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image

# location of reference data, to be imported if required in other files
ref_data_folder = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)),
                               'data_analysis', 'ref_data')


def model_predict_and_process(model, image):
    """
    Runs the provided segmentation model and pre-processes it into an ordered mask for subsequent labelling.
    :param model: Pytorch segmentation model
    :param image: Input image (torch tensor)
    :return:
    """
    with torch.no_grad():
        mask = model(image)
        one_hot = F.one_hot(mask.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
        ordered_mask = one_hot.numpy().squeeze()
    return mask, ordered_mask


def model_multi_augment_predict_and_process(model, image):
    """
    This function runs the provided model multiple times on augmented versions of the input, then combines the
    outputs into one averaged estimate.
    :param model: Pytorch segmentation model
    :param image: gel image in N,C,H,W format
    :return: direct pytorch mask and ordered numpy mask for easy use
    """
    mirror_axes = [0, 1]

    with torch.no_grad():
        mask = model(image)
        axes_combinations = [c for i in range(len(mirror_axes)) for c in
                             itertools.combinations([m + 2 for m in mirror_axes], i + 1)]
        for axes in axes_combinations:
            mask += torch.flip(model(torch.flip(image, axes)), axes)
        mask /= (len(axes_combinations) + 1)

    one_hot = F.one_hot(mask.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
    ordered_mask = one_hot.numpy().squeeze()

    return mask, ordered_mask


def save_model_output(output_folder, model_name, image_name, labelled_image):
    """
    Saves labelled model output to file.
    :param output_folder: Output folder location
    :param model_name: Name of model
    :param image_name: Image name
    :param labelled_image: Image with RGB segmentation labels painted on top
    :return: N/A
    """
    imageio.v2.imwrite(os.path.join(output_folder, model_name, '%s.png' % image_name), (labelled_image * 255).astype(np.uint8))


def save_segmentation_map(output_folder, model_name, image_name, segmentation_map, positive_pixel_colour=(163, 106, 13)):

    if len(segmentation_map.shape) == 3:
        segmentation_map = segmentation_map.argmax(axis=0)

    rgba_array = np.ones((segmentation_map.shape[0], segmentation_map.shape[1], 4), dtype=np.uint8)*255

    # negative pixels should have no alpha and no colour
    alpha_channel = np.where(segmentation_map > 0, 255, 0)
    rgba_array[:, :, 3] = alpha_channel

    # Set RGB channels for positive data
    rgba_array[:, :, :3] = np.where(segmentation_map[..., None] > 0, positive_pixel_colour, 0)

    # Create PIL Image from RGBA array
    image = Image.fromarray(rgba_array, 'RGBA')

    image.save(os.path.join(output_folder, model_name, '%s_map_only.png' % image_name))

    return image


def plot_model_comparison(model_outputs, model_names, image_name, raw_image, output_folder, images_per_row,
                          double_indexing, comments=None, title_length_cutoff=20):
    """
    Plots a comparison of input images and their segmentation results.
    :param model_outputs: Segmentation images for each input model
    :param model_names: Name of each model (in order)
    :param image_name: Name of test image
    :param raw_image: Raw un-segmented numpy image (for plotting)
    :param output_folder: Output folder to save results
    :param images_per_row: Number of images to plot per row in the output figure
    :param double_indexing: Whether or not double indexing is required for the plot axes
    :param comments: A string to add to the title of each model (if required)
    :return: N/A
    """
    # results preview

    rows = math.ceil((len(model_outputs) + 1) / images_per_row)
    fig, ax = plt.subplots(rows, images_per_row, figsize=(15, 15))

    for i in range(len(model_outputs) + 1, rows * images_per_row):
        ax[index_converter(i, images_per_row, double_indexing)].axis('off')  # turns off borders for unused panels

    zero_ax_index = index_converter(0, images_per_row, double_indexing)

    ax[zero_ax_index].imshow(raw_image, cmap='gray')
    ax[zero_ax_index].set_title('Reference Image')

    for index, (mask, name) in enumerate(zip(model_outputs, model_names)):
        plot_index = index_converter(index + 1, images_per_row, double_indexing)

        ax[plot_index].imshow(mask)
        title = name
        if comments:
            title += ' ' + comments[index]
        if len(title) > title_length_cutoff:
            title = title[:int(len(title) / 3)] + '\n' + title[int(len(title) / 3):int((2 * len(title)) / 3)] + '\n' + title[int((2 * len(title)) / 3):]
        elif len(name) > title_length_cutoff * 2:
            title = title[:int(len(title) / 2)] + '\n' + title[int(len(title) / 2):]
        else:
            title = title

        ax[plot_index].set_title(title, fontsize=13)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.suptitle('Segmentation result for image %s' % image_name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'method_comparison', '%s method comparison.png' % image_name), dpi=300)
    plt.close(fig)


def run_watershed(image, image_name, output_watershed):
    """
    Runs the multiotsu thresholding system on the input image and converts to torch format for loss computation
    :param image: Input image (numpy array)
    :param image_name: Image name (string)
    :param output_watershed: Watershed folder for saving intermediates directly
    :return: Torch mask for dice computation and numpy array of watershed labels
    """
    _, watershed_labels = watershed_analysis(image, image_name,
                                             intermediate_plot_folder=output_watershed,
                                             repetitions=1, background_jump=0.08,
                                             use_multiotsu=True)

    temp_array = watershed_labels.copy()
    temp_array[temp_array > 0] = 1
    # converts back to required format for dice score calculation
    torch_mask = F.one_hot(torch.tensor(temp_array).long().unsqueeze(0), 2).permute(0, 3, 1, 2).float()

    return torch_mask, watershed_labels


def run_multiotsu(image, image_name, output_otsu):
    """
    Runs the multiotsu thresholding system on the input image and converts to torch format for loss computation
    :param image: Input image (numpy array)
    :param image_name: Image name (string)
    :param output_otsu: Otsu folder for saving intermediates directly
    :return: Torch mask for dice computation and numpy array of otsu labels
    """
    otsu_labels = multiotsu_analysis(image, image_name, intermediate_plot_folder=output_otsu)

    otsu_labels[otsu_labels > 0] = 1

    # converts back to required format for dice score calculation
    torch_mask = F.one_hot(torch.tensor(otsu_labels).long().unsqueeze(0), 2).permute(0, 3, 1, 2).float()

    return torch_mask, otsu_labels


def read_nnunet_inference_from_file(nfile):
    # get image name, then convert into nnunet-compatible name
    # read image from file, convert to 1 channel segmentation format
    # pass on for dice score calculation and RGB labelling
    n_im = imageio.v2.imread(nfile)
    torch_mask = F.one_hot(torch.tensor(n_im).long().unsqueeze(0), 2).permute(0, 3, 1, 2).float()

    return torch_mask, n_im


def segment_and_quantitate(models, model_names, input_folder, mask_folder, output_folder,
                           minmax_norm=False, multi_augment=False, images_per_row=3,
                           run_classical_techniques=False, nnunet_models_and_folders=None):
    """

    Segments images in input_folder using the selected models and computes their Dice score versus the ground truth labels.
    :param models: Pre-loaded pytorch segmentation models
    :param model_names: Name for each model (list)
    :param input_folder: Input folder containing gel images
    :param mask_folder: Corresponding folder containing ground truth mask labels for loss computation
    :param output_folder: Output folder to save results
    :param minmax_norm: Set to true to min-max normalise images before segmentation
    :param multi_augment: Set to true to perform test-time augmentation
    :param images_per_row: Number of images to plot per row in the output comparison figure
    :param run_classical_techniques: Set to true to also run watershed and multiotsu segmentation apart from selected models
    :param nnunet_models_and_folders: List of tuples containing (model name, folder location) for pre-computed nnunet results on the same dataset
    :return: N/A (all outputs saved to file)
    """
    dataset = ImageMaskDataset(input_folder, mask_folder, 1, padding=False, individual_padding=True,
                               minmax_norm=minmax_norm)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    if run_classical_techniques:
        models.extend(['watershed', 'multiotsu'])
        model_names.extend(['watershed', 'multiotsu'])

    if nnunet_models_and_folders:
        nnunet_model_names, nnunet_eval_outputs = zip(*nnunet_models_and_folders)
        models.extend(nnunet_eval_outputs)
        model_names.extend(nnunet_model_names)
    else:
        nnunet_model_names = []

    for mname in model_names:
        create_dir_if_empty(os.path.join(output_folder, mname))

    create_dir_if_empty(os.path.join(output_folder, 'method_comparison'))
    create_dir_if_empty(os.path.join(output_folder, 'metrics'))


    double_indexing = True  # axes will have two indices rather than one
    if math.ceil((len(model_names) + 2) / images_per_row) == 1:  # axes will only have one index rather than 2
        double_indexing = False

    metrics_dict = {}
    metrics = ['Dice Score', 'MultiClass Dice Score', 'True Negatives', 'False Positives', 'False Negatives', 'True Positives']

    for metric in metrics:
        metrics_dict[metric] = defaultdict(list)

    # preparing model outputs, including separation of different bands and labelling
    for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        orig_image_height, orig_image_width = int(batch['image_height'][0]), int(batch['image_width'][0])
        np_image = batch['image'].detach().squeeze().cpu().numpy()

        pad_h1 = (np_image.shape[0] - orig_image_height) // 2
        pad_w1 = (np_image.shape[1] - orig_image_width) // 2
        pad_h2 = (np_image.shape[0] - orig_image_height) - pad_h1
        pad_w2 = (np_image.shape[1] - orig_image_width) - pad_w1

        np_image = np_image[pad_h1:-pad_h2, pad_w1:-pad_w2]  # unpads the image for loss computation and display

        gt_mask = batch['mask'][:, pad_h1:-pad_h2, pad_w1:-pad_w2]
        image_name = batch['image_name'][0]

        all_model_outputs = []
        display_dice_scores = []

        gt_one_hot = F.one_hot(gt_mask.long(), 2).permute(0, 3, 1, 2).float()

        for model, mname in zip(models, model_names):

            # classical methods
            if mname == 'watershed':
                torch_one_hot, mask = run_watershed(np_image, image_name, os.path.join(output_folder, mname))
            elif mname == 'multiotsu':
                torch_one_hot, mask = run_multiotsu(np_image, image_name, os.path.join(output_folder, mname))
            elif mname in nnunet_model_names:  # nnunet results are pre-computed
                torch_one_hot, mask = read_nnunet_inference_from_file(os.path.join(model, image_name + '.tif'))
            else:  # standard ML models
                if multi_augment:
                    torch_mask, mask = model_multi_augment_predict_and_process(model, batch['image'])
                else:
                    torch_mask, mask = model_predict_and_process(model, batch['image'])
                torch_one_hot = F.one_hot(torch_mask.argmax(dim=1), 2).permute(0, 3, 1, 2).float()

                torch_one_hot = torch_one_hot[:, :, pad_h1:-pad_h2, pad_w1:-pad_w2]  # unpads model outputs
                mask = mask[:, pad_h1:-pad_h2, pad_w1:-pad_w2]

            # dice score calculation
            dice_score = multiclass_dice_coeff(torch_one_hot[:, 1:, ...],
                                               gt_one_hot[:, 1:, ...],
                                               reduce_batch_first=False).cpu().numpy()

            dice_score_multi = multiclass_dice_coeff(torch_one_hot,
                                                     gt_one_hot,
                                                     reduce_batch_first=False).cpu().numpy()
            display_dice_scores.append('Dice Score: %.3f' % dice_score)

            # confusion matrix calculation
            if mname == 'watershed':
                c_mask = mask.flatten()
                c_mask[c_mask > 0] = 1
            elif mname == 'multiotsu' or mname in nnunet_model_names:
                c_mask = mask.flatten()
            else:
                c_mask = mask.argmax(axis=0).flatten()
            tn, fp, fn, tp = confusion_matrix(c_mask, gt_mask.numpy().squeeze().flatten()).ravel()

            for metric, value in zip(metrics, [dice_score, dice_score_multi, tn, fp, fn, tp]):
                metrics_dict[metric][image_name].append(value)

            # direct model plotting
            if mname == 'watershed':
                labels = mask
            elif mname == 'multiotsu' or mname in nnunet_model_names:
                labels, _ = ndi.label(mask)
            else:
                labels, _ = ndi.label(mask.argmax(axis=0))

            rgb_labels = label2rgb(labels, image=np_image)

            all_model_outputs.append(rgb_labels)
            save_model_output(output_folder, mname, image_name, rgb_labels)
            save_segmentation_map(output_folder, mname, image_name, mask)

        gt_labels, _ = ndi.label(gt_one_hot.numpy().squeeze().argmax(axis=0))
        gt_rgb_labels = label2rgb(gt_labels, image=np_image)

        # comparison plotting
        plot_model_comparison([gt_rgb_labels] + all_model_outputs, ['Ground Truth'] + model_names,
                              image_name, np_image, output_folder,
                              images_per_row, double_indexing, comments=[''] + display_dice_scores)

    # combines and saves final dice score data into a table
    for key, value in metrics_dict.items():
        pd_data = pd.DataFrame.from_dict(value, orient='index')
        pd_data.columns = model_names
        pd_data.loc['mean'] = pd_data.mean()
        pd_data.to_csv(os.path.join(output_folder, 'metrics', '%s.csv' % key), mode='w', header=True, index=True, index_label='Image')


def segment_and_plot(models, model_names, input_folder, output_folder, minmax_norm=False, multi_augment=False,
                     images_per_row=2, run_classical_techniques=False, nnunet_models_and_folders=None):
    """
    Segments images in input_folder using models and saves the output image and a quick comparison to the output folder.
    :param models: Pre-loaded pytorch segmentation models
    :param model_names: Name for each model (list)
    :param input_folder: Input folder containing gel images
    :param output_folder: Output folder to save results
    :param minmax_norm: Set to true to min-max normalise images before segmentation
    :param multi_augment: Set to true to perform test-time augmentation
    :param images_per_row: Number of images to plot per row in the output comparison figure
    :param run_classical_techniques: Set to true to also run watershed and multiotsu segmentation apart from selected models
    :param nnunet_models_and_folders: List of tuples containing (model name, folder location) for pre-computed nnunet results on the same dataset
    :return: N/A (all outputs saved to file)
    """

    dataset = ImageDataset(input_folder, 1, padding=False, individual_padding=True, minmax_norm=minmax_norm)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    if run_classical_techniques:
        models.extend(['watershed', 'multiotsu'])
        model_names.extend(['watershed', 'multiotsu'])

    if nnunet_models_and_folders:
        nnunet_model_names, nnunet_eval_outputs = zip(*nnunet_models_and_folders)
        models.extend(nnunet_eval_outputs)
        model_names.extend(nnunet_model_names)
    else:
        nnunet_model_names = []

    double_indexing = True  # axes will have two indices rather than one
    if math.ceil((len(model_names) + 1) / images_per_row) == 1:  # axes will only have one index rather than 2
        double_indexing = False

    for mname in model_names:
        create_dir_if_empty(os.path.join(output_folder, mname))

    create_dir_if_empty(os.path.join(output_folder, 'method_comparison'))

    # preparing model outputs, including separation of different bands and labelling
    for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        orig_image_height, orig_image_width = int(batch['image_height'][0]), int(batch['image_width'][0])
        np_image = batch['image'].detach().squeeze().cpu().numpy()

        pad_h1 = (np_image.shape[0] - orig_image_height) // 2
        pad_w1 = (np_image.shape[1] - orig_image_width) // 2
        pad_h2 = (np_image.shape[0] - orig_image_height) - pad_h1
        pad_w2 = (np_image.shape[1] - orig_image_width) - pad_w1

        np_image = np_image[pad_h1:-pad_h2, pad_w1:-pad_w2] # unpads the image for loss computation and display

        image_name = batch['image_name'][0]
        all_model_outputs = []

        for model, mname in zip(models, model_names):
            # classical methods
            if mname == 'watershed':
                _, mask = run_watershed(np_image, image_name, None)
            elif mname == 'multiotsu':
                _, mask = run_multiotsu(np_image, image_name, None)
            elif mname in nnunet_model_names:  # nnunet results are pre-computed
                _, mask = read_nnunet_inference_from_file(os.path.join(model, image_name + '.tif'))
            else:
                if multi_augment:
                    _, mask = model_multi_augment_predict_and_process(model, batch['image'])
                else:
                    _, mask = model_predict_and_process(model, batch['image'])
                mask = mask[:, pad_h1:-pad_h2, pad_w1:-pad_w2]

            # direct model plotting
            if mname == 'watershed':
                labels = mask
            elif mname == 'multiotsu' or mname in nnunet_model_names:
                labels, _ = ndi.label(mask)
            else:
                labels, _ = ndi.label(mask.argmax(axis=0))

            rgb_labels = label2rgb(labels, image=np_image)
            all_model_outputs.append(rgb_labels)
            save_model_output(output_folder, mname, '%s.png' % image_name, rgb_labels)
            save_segmentation_map(output_folder, mname, image_name, mask)

        plot_model_comparison(all_model_outputs, model_names, image_name, np_image, output_folder,
                              images_per_row, double_indexing)
