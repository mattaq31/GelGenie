from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from gel_tools.band_detection import watershed_seg


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


def plot_stats(base_dir, train_loss_log, val_loss_log):
    """
    :param train_loss_log: array holding loss values
    :param val_loss_log: array holding validation dice scores
    :param base_dir: path of model directory that stores outputs
    :return:
    """
    # TODO: customizable according to how many different types of metrics are provided
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    if len(train_loss_log) > 0:
        axs.plot([epoch + 1 for epoch in range(len(train_loss_log))], train_loss_log, label='training loss',
                 linestyle='--', color='b')
    if len(val_loss_log) > 0:
        axs.plot([epoch + 1 for epoch in range(len(val_loss_log))], val_loss_log, label='validation dice score',
                 linestyle='--', color='r')
    axs.set_xlabel('Epoch')
    axs.legend()
    axs.set_ylabel('Loss/ Dice score')
    plt.tight_layout()
    plt.savefig(Path(base_dir + '/loss_plots.png'))
    plt.close(fig)


def show_segmentation(image, mask_pred, mask_true, epoch_number, dice_score, segmentation_path, n_channels):
    """
    #TODO: fill in!
    :param image: image tensor
    :param mask_pred: mask prediction tensor
    :param mask_true: true mask tensor
    :param epoch_number: current epoch number
    :param dice_score: dice score for current epoch
    :param segmentation_path: directory path for output of segmentation images
    :param n_channels: number of input channels
    :return: arrays of image, mask prediction, super-imposed mask, separated objects, true mask
    """


    if n_channels == 1:
        height = image.size(dim=0)
        width = image.size(dim=1)
        image_array = image.detach().squeeze().cpu().numpy()
    elif n_channels == 3:
        height = image.size(dim=1)
        width = image.size(dim=2)
        image_array = np.transpose(image.detach().squeeze().cpu().numpy(), (1, 2, 0))  # tensor [C,H,W] to array [H,W,C]

    mask_pred_array = mask_pred.detach().squeeze().cpu().numpy()  # Copies prediction into np array [C, H, W]
    mask_true_array = mask_true.detach().squeeze().cpu().numpy()[1]  # Copies true mask[1] into np array [H,W]

    # Threshold mask prediction array into integer classes of 0 or 1
    threshold_mask_array = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            # The threshold is set to 0.8
            if mask_pred_array[0][h][w] < 0.2 and mask_pred_array[1][h][w] > 0.8:
                threshold_mask_array[h][w] = 1

    labelled_bands = watershed_seg(threshold_mask_array, 0.5, 0.5)

    combi_mask_array = np.zeros((height, width, 3))  # np array [H, W, C]
    for i in range(height):
        for j in range(width):
            if threshold_mask_array[i][j] == 1:  # It is classified as gel band
                combi_mask_array[i][j] = [1, 0, 0]  # Set the pixel to have red colour
            else:  # Background
                if n_channels == 1:  # image_array [H,W] / grayscale
                    combi_mask_array[i][j] = np.repeat(image_array[i][j], 3)  # Copies grayscale value to RGB channels
                elif n_channels == 3:  # image_array [H,W,C] / RGB
                    combi_mask_array[i][j] = image_array[i][j]  # Copies RGB channel values all at once

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10, 7))

    if n_channels == 1:
        axs[0].imshow(image_array, cmap='gray')
    else:
        axs[0].imshow(image_array)

    axs[1].imshow(threshold_mask_array, cmap='gray')
    axs[1].set_title('Mask Prediction')

    axs[2].imshow(labelled_bands, cmap='tab20')
    axs[2].set_title('Labelled Bands')

    axs[3].imshow(combi_mask_array)
    axs[3].set_title('Mask Prediction Superimposed')
    axs[3].set_xlabel(f'Dice Score: {dice_score}')

    axs[4].imshow(mask_true_array, cmap='gray')
    axs[4].set_title('True Mask')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])  # remove ticks
    plt.tight_layout()

    plt.savefig(Path(segmentation_path + f'/epoch{epoch_number}.pdf'))
    plt.close(fig)

    # For saving un-thresholded mask predictions
    # mask_pred_array = np.transpose(mask_pred_array, (1, 2, 0))  # C, H, W to H, W, C
    # np.save(str(Path(segmentation_path + f'/epoch{epoch_number}_mask_pred')), mask_pred_array)

    return image_array, threshold_mask_array, labelled_bands, combi_mask_array, mask_true_array
