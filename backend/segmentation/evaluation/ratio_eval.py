import click
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import ndimage
from segmentation.helper_functions.dice_score import dice_coeff

from segmentation.unet import UNet, smp_UNet, smp_UNetPlusPlus
from segmentation.data_handling.dataloaders import BasicDataset, ImageDataset


@click.command()
@click.option('--images_path', default=None, help='[Path] location of image directory')
@click.option('--masks_path', default=None, help='[Path] location of mask directory')
@click.option('--checkpoint_path', default=None, help='[Path] path or checkpoint saved')
@click.option('--net_name', default=None, help='[Str] name of model [milesial-UNet/ smp-UNet/ UNetPlusPlus]')
def model_eval(images_path, masks_path, checkpoint_path, net_name):
    """
    Evaluate Model loaded from past checkpoint on a set of images (and masks)
    :param images_path: location of image directory
    :param masks_path: location of mask directory (can be none)
    :param checkpoint_path: path of checkpoint saved
    :param net_name: name of model (milesial-UNet/ smp-UNet/ UNetPlusPlus)
    """
    if net_name == 'smp-UNet':
        net = smp_UNet(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
        padding = True  # Forced padding
    elif net_name == 'UNetPlusPlus':
        net = smp_UNetPlusPlus(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
        padding = True  # Forced padding
    elif net_name == 'milesial-UNet':
        net = UNet(n_channels=1, n_classes=2, bilinear=False)
        padding = False
    else:
        raise RuntimeError(f'No net identified, selected net_name is {net_name}')

    if masks_path is None:
        test_set = ImageDataset(images_path, n_channels=1, padding=padding)  # Only images loaded
    else:
        test_set = BasicDataset(images_path, masks_path, n_channels=1, padding=padding)  # Both images and masks loaded

    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)
    net.train()

    # Load checkpoint and retrieve saved state dict of network
    saved_dict = torch.load(f=checkpoint_path, map_location=torch.device("cpu"))
    net.load_state_dict(saved_dict['network'])
    # net.load_state_dict(saved_dict) for older checkpoints which only saves network but not scheduler and optimizer
    print(f'Model loaded from {checkpoint_path}')

    for batch in test_loader:
        image = batch['image']
        if masks_path is not None:
            true_mask = batch['mask']
            true_mask_array = true_mask.detach().squeeze().cpu().numpy()  # Tensor:(1, H, W) -> ndarray: (H,W)

        with torch.no_grad():
            mask_pred = net(image)
        image = image.squeeze()

        mask_pred_array = np.transpose(mask_pred.detach().squeeze().cpu().numpy(), (1, 2, 0))  # CHW to HWC
        height, width = mask_pred_array.shape[0], mask_pred_array.shape[1]

        threshold = 0.8
        # Threshold mask prediction pixel values to either 0 or 1 according to set threshold
        thresholded = np.zeros((height, width))
        for row in range(height):
            for column in range(width):
                if mask_pred_array[row][column][0] < (1 - threshold) and mask_pred_array[row][column][1] > threshold:
                    thresholded[row][column] = 1

        # use a boolean condition to find where pixel values are 1
        blobs = thresholded == 1

        # label connected regions that satisfy this condition, connect regions that are only connected diagonally
        labels, nlabels = ndimage.label(blobs, structure=[[1, 1, 1],
                                                          [1, 1, 1],
                                                          [1, 1, 1]])


        # plot
        if masks_path is None:
            fig, ax = plt.subplots(1, 2, figsize=(60, 60))
        else:
            fig, ax = plt.subplots(1, 3, figsize=(60, 60))

        # Show original image
        original_image = image.detach().squeeze().cpu().numpy()

        volume_labels = np.zeros((nlabels + 1), float)
        for h in range(height):
            for w in range(width):
                volume_labels[labels[h][w]] += original_image[h][w]  # idx = label, value += intensity(between 0 and 1)

        # find their centres of mass. in this case I'm weighting by the pixel values in
        # `img`, but you could also pass the boolean values in `blobs` to compute the
        # unweighted centroids.
        r, c = np.vstack(ndimage.center_of_mass(thresholded, labels, np.arange(nlabels) + 1)).T

        # find their distances from the top-left corner
        d = np.sqrt(r*r + c*c)

        # Superimpose mask prediction on original image
        ax[0].imshow(original_image, cmap='gray', interpolation='none')
        ax[0].imshow(np.ma.masked_array(labels, ~blobs), cmap=plt.cm.rainbow)
        ax[0].set_title('Predicted Mask', fontsize=60)
        # for ri, ci, di, count in zip(r, c, d, range(nlabels)):
        #     ax[0].annotate(f'{count + 1}: V={round(volume_labels[count + 1], 1)}', xy=(ci, ri), xytext=(0, -5),
        #                    textcoords='offset points', ha='center', va='top',
        #                    fontsize=8, color='blue')

        if masks_path:
            # use a boolean condition to find where pixel values are 1
            true_gel_blobs = true_mask_array == 1

            # label connected regions that satisfy this condition, connect regions that are only connected diagonally
            true_gel_bands, nbands = ndimage.label(true_gel_blobs, structure=[[1, 1, 1],
                                                                              [1, 1, 1],
                                                                              [1, 1, 1]])
            ax[1].imshow(original_image, cmap='gray', interpolation='none')
            ax[1].imshow(np.ma.masked_array(true_gel_bands, ~true_gel_blobs), cmap=plt.cm.rainbow)
            ax[1].set_title('True Mask', fontsize=60)

            # Evaluation on predicted mask and compute the Dice score
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            dice_score = \
                dice_coeff(mask_pred,
                           F.one_hot(true_mask.to(dtype=torch.long), net.n_classes).permute(0, 3, 1, 2).float(),
                           reduce_batch_first=False).item()
            fig.supxlabel(f'Dice Score: {dice_score}', fontsize=60)  # Print dice score at bottom of plot

        for aa in ax.flat:
            aa.set_axis_off()
        fig.tight_layout()
        plt.savefig(f'C:/Users/s2137314/Downloads/{batch["image_name"]}.pdf')
        plt.close(fig)

        break


if __name__ == '__main__':
    model_eval(sys.argv[1:])  # for use when debugging with pycharm
