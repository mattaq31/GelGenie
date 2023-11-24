import numpy as np
from matplotlib import pyplot as plt
import os


def plot_img_and_mask(img, mask):
    """
    :param img: image file
    :param mask: mask file
    :return:
    """
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_stats(stats_dict, keynames, experiment_log_dir, filename):

    plot_filename = os.path.join(experiment_log_dir, filename)

    num_plots = 0
    valid_keys = []
    for key in keynames:  # checks to see if any illegal keywords have been provided
        if all(metric in stats_dict for metric in key):
            num_plots += 1
            valid_keys.append(key)

    f, ax = plt.subplots(num_plots, 1, figsize=(10, 7))

    if not isinstance(ax, np.ndarray):  # hacky, better fix?
        ax = [ax]

    for ind, key in enumerate(valid_keys):
        for metric in key:
            ax[ind].plot(stats_dict['Epoch'], stats_dict[metric], label=metric, linestyle='--', marker='o')
        ax[ind].set_xlabel('Epoch')
        ax[ind].legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(f)


def visualise_segmentation(image, mask_pred, mask_true, epoch_number, dice_score, segmentation_path,
                           optional_name=None):
    """
    Prepares a matplotlib plot comparing a segmentation prediction with the true mask.
    :param image: image tensor
    :param mask_pred: mask prediction tensor
    :param mask_true: true mask tensor
    :param epoch_number: current epoch number
    :param dice_score: dice score for current epoch
    :param segmentation_path: directory path for output of segmentation images
    :param optional_name: Optional name to append to output file
    :return: Numpy arrays of image, mask prediction, super-imposed mask and true mask
    """
    n_channels = 1 if len(image.shape) == 2 else 3
    if n_channels == 1:
        height = image.size(dim=0)
        width = image.size(dim=1)
        image_array = image.detach().squeeze().cpu().numpy()
    elif n_channels == 3:
        height = image.size(dim=1)
        width = image.size(dim=2)
        image_array = np.transpose(image.detach().squeeze().cpu().numpy(), (1, 2, 0))  # tensor [C,H,W] to array [H,W,C]

    mask_pred_array_multichannel = mask_pred.detach().squeeze().cpu().numpy()  # Copies prediction into np array [C, H, W]
    mask_true_array = mask_true.detach().squeeze().cpu().numpy()[1]  # Copies true mask[1] into np array [H,W]

    mask_pred_array = np.zeros((height, width))
    for channel in range(1, mask_pred_array_multichannel.shape[0]):  # background class skipped (to set as 0)
        mask_pred_array += mask_pred_array_multichannel[channel]

    combi_mask_array = np.zeros((height, width, 3))  # np array [H, W, C]
    for i in range(height):
        for j in range(width):
            if mask_pred_array[i][j] == 1:  # It is classified as gel band
                combi_mask_array[i][j] = [1, 0, 0]  # Set the pixel to have red colour
            else:  # Background
                if n_channels == 1:  # image_array [H,W] / grayscale
                    combi_mask_array[i][j] = np.repeat(image_array[i][j], 3)  # Copies grayscale value to RGB channels
                elif n_channels == 3:  # image_array [H,W,C] / RGB
                    combi_mask_array[i][j] = image_array[i][j]  # Copies RGB channel values all at once

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))

    if n_channels == 1:
        axs[0].imshow(image_array, cmap='gray')
    else:
        axs[0].imshow(image_array)

    axs[1].imshow(mask_pred_array, cmap='gray')
    axs[1].set_title('Mask Prediction')

    axs[2].imshow(combi_mask_array)
    axs[2].set_title('Superimposed\nMask Prediction')

    axs[3].imshow(mask_true_array, cmap='gray')
    axs[3].set_title('True Mask')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])  # remove ticks
    fig.suptitle('Example Validation Image from Epoch %s (Dice Score %.3f)' % (epoch_number, dice_score), fontsize=16)
    plt.tight_layout()

    plt.savefig(os.path.join(segmentation_path, f'sample_epoch_{epoch_number}_{optional_name}.pdf'), dpi=300)
    plt.close(fig)

    # For saving un-thresholded mask predictions
    # mask_pred_array = np.transpose(mask_pred_array, (1, 2, 0))  # C, H, W to H, W, C
    # np.save(str(Path(segmentation_path + f'/epoch{epoch_number}_mask_pred')), mask_pred_array)

    return image_array, mask_pred_array, combi_mask_array, mask_true_array
