# code here adapted from https://github.com/milesial/Pytorch-UNet/tree/e36c782fbfc976b7326182a47dd7213bd3360a7e

from pathlib import Path
import matplotlib.pyplot as plt


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
    axs[0].plot([epoch + 1 for epoch in range(len(train_loss_log))], train_loss_log, label='train set', linestyle='--',
                color='b')
    axs[0].plot([epoch + 1 for epoch in range(len(val_loss_log))], val_loss_log, label='validation set', linestyle='--',
                color='r')
    for ax in axs:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    plt.tight_layout()
    plt.savefig(Path(base_dir + '/loss_plots.png'))
    plt.close(fig)
