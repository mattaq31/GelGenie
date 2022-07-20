# code here adapted from https://github.com/milesial/Pytorch-UNet/tree/e36c782fbfc976b7326182a47dd7213bd3360a7e

from pathlib import Path
import matplotlib.pyplot as plt
import torchshow as ts
import numpy as np
import pandas as pd


def plot_img_and_mask(img, mask):
    """
    TODO: fill
    :param img:
    :param mask:
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


def plot_stats(train_loss_log, val_loss_log, base_dir):
    """
    TODO: fill
    :param train_loss_log:
    :param val_loss_log:
    :param base_dir:
    :return:
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))  # TODO: make this customizable according to how many different types of metrics are provided
    axs.plot([epoch + 1 for epoch in range(len(train_loss_log))], train_loss_log, label='training loss', linestyle='--',
                color='b')
    axs.plot([epoch + 1 for epoch in range(len(val_loss_log))], val_loss_log, label='validation dice score', linestyle='--',
                color='r')
    # axs[1].plot([epoch + 1 for epoch in range(len(train_loss_log))], train_loss_log, label='training loss',
    #             linestyle='--',
    #             color='b')
    # axs[2].plot([epoch + 1 for epoch in range(len(val_loss_log))], val_loss_log, label='validation dice score',
    #             linestyle='--',
    #             color='r')
    axs.set_xlabel('Epoch')
    axs.legend()
    axs.set_ylabel('Loss/ Dice score')
    plt.tight_layout()
    plt.savefig(Path(base_dir + '/loss_plots.png'))
    plt.close(fig)


def show_segmentation(image, mask_pred, mask_true, epoch_number, dice_score, segmentation_path):
    combi_mask = image.detach().clone().squeeze()  # Copies the image into np array
    for i in range(image.size(dim=1)):  # Height
        for j in range(image.size(dim=2)):  # Width
            if mask_pred[0][i][j] < 0.5 and mask_pred[1][i][j] > 0.5:  # is labelled
                combi_mask[0][i][j] = 1
                combi_mask[1][i][j] = 0
                combi_mask[2][i][j] = 0

    def combine_channels(mask_array):
        height = mask_array.shape[1]
        width = mask_array.shape[2]
        combined_array = np.tile(0, (height, width))
        for h in range(height):
            for w in range(width):
                if mask_array[1][h][w] > 0.5:
                    combined_array[h][w] = 1  # is labelled
        return combined_array

    image_array = np.transpose(image.detach().clone().squeeze().numpy(), (1, 2, 0))  # Copies the image into np array
    mask_pred_array = combine_channels(mask_pred.detach().clone().squeeze().numpy())  # Copies prediction into np array
    mask_true_array = combine_channels(mask_true.detach().clone().squeeze().numpy())  # Copies true mask into np array
    combi_mask = np.transpose(combi_mask.detach().clone().squeeze().numpy(), (1, 2, 0))

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 7))
    axs[0].imshow(image_array)
    axs[0].set_title('Original Image')
    axs[1].imshow(mask_pred_array)
    axs[1].set_title('Mask Prediction')
    axs[2].imshow(combi_mask)
    axs[2].set_title('Mask Prediction Superimposed')
    axs[2].set_xlabel(f'Dice Score: {dice_score}')
    axs[3].imshow(mask_true_array)
    axs[3].set_title('True Mask')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])  # remove ticks
    plt.tight_layout()
    plt.savefig(Path(segmentation_path + f'/epoch{epoch_number}.pdf'))
    plt.close(fig)

    # add dice score

def excel_stats(train_loss_log, val_loss_log, base_dir):
    loss_array = np.array([train_loss_log, val_loss_log]).T
    loss_dataframe = pd.DataFrame(loss_array, columns=['Training Loss', 'Validation Dice Score'])
    loss_dataframe.index.names = ['Epoch']
    loss_dataframe.index += 1
    loss_dataframe.to_csv(Path(base_dir + '/loss.csv'))