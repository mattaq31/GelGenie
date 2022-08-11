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
@click.option('--checkpoint_path', default=None, help='[Path] location of checkpoint directory')
@click.option('--net_name', default=None, help='[Str] name of model [milesial-UNet/ smp-UNet/ UNetPlusPlus]')
def model_eval(images_path, masks_path, checkpoint_path, net_name):
    if net_name == 'smp-UNet':
        net = smp_UNet(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
        padding = True
    elif net_name == 'UNetPlusPlus':
        net = smp_UNetPlusPlus(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
        padding = True
    elif net_name == 'milesial-UNet':
        net = UNet(n_channels=1, n_classes=2, bilinear=False)
        padding = False
    else:
        raise RuntimeError(f'No net identified, selected net_name is {net_name}')

    saved_dict = torch.load(f=checkpoint_path, map_location=torch.device("cpu"))
    net.load_state_dict(saved_dict['network'])

    if masks_path is None:
        test_set = ImageDataset(images_path, n_channels=1, padding=padding)
    else:
        test_set = BasicDataset(images_path, masks_path, n_channels=1, padding=padding)

    n_test = int(len(test_set))
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)
    net.train()
    saved_dict = torch.load(f=checkpoint_path, map_location=torch.device("cpu"))
    net.load_state_dict(saved_dict['network'])
    # net.load_state_dict(saved_dict)
    print(f'Model loaded from {checkpoint_path}')

    for batch in test_loader:
        image = batch['image']
        if masks_path is not None:
            true_mask = batch['mask']
            true_mask_array = true_mask.detach().squeeze().cpu().numpy()  # Tensor:(1, H, W) -> ndarray: (H,W)

        with torch.no_grad():
            mask_pred = net(image)
        image = image.squeeze()
        mask_pred.squeeze()

        mask_pred_array = np.transpose(mask_pred.detach().squeeze().cpu().numpy(), (1, 2, 0))  # CHW to HWC
        height, width = mask_pred_array.shape[0], mask_pred_array.shape[1]

        threshold = 0.8
        thresholded = np.zeros((height, width))
        for row in range(height):
            for column in range(width):
                if mask_pred_array[row][column][0] < (1 - threshold) and mask_pred_array[row][column][1] > threshold:
                    thresholded[row][column] = 1

        # use a boolean condition to find where pixel values are > 0.75
        blobs = thresholded == 1

        # label connected regions that satisfy this condition
        labels, nlabels = ndimage.label(blobs, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])


        # plot
        if masks_path is None:
            fig, ax = plt.subplots(1, 2, figsize=(60, 60))
        else:
            fig, ax = plt.subplots(1, 3, figsize=(60, 60))

        # ax[0].imshow(thresholded)
        original_image = image.detach().squeeze().cpu().numpy()
        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original Image', fontsize=60)

        ax[1].imshow(original_image, cmap='gray', interpolation='none')
        ax[1].imshow(np.ma.masked_array(labels, ~blobs), cmap=plt.cm.rainbow)
        ax[1].set_title('Predicted Mask', fontsize=60)

        if masks_path is not None:
            # use a boolean condition to find where pixel values are > 0.75
            true_gel_blobs = true_mask_array == 1

            # label connected regions that satisfy this condition
            true_gel_bands, nbands = ndimage.label(true_gel_blobs, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            ax[2].imshow(original_image, cmap='gray', interpolation='none')
            ax[2].imshow(np.ma.masked_array(true_gel_bands, ~true_gel_blobs), cmap=plt.cm.rainbow)
            ax[2].set_title('True Mask', fontsize=60)

            # Evaluation on predicted mask
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            dice_score = \
                dice_coeff(mask_pred,
                           F.one_hot(true_mask.to(dtype=torch.long), net.n_classes).permute(0, 3, 1, 2).float(),
                           reduce_batch_first=False).item()
            fig.supxlabel(f'Dice Score: {dice_score}', fontsize=60)

        for aa in ax.flat:
            aa.set_axis_off()
        fig.tight_layout()
        plt.savefig(f'C:/Users/s2137314/Downloads/{batch["image_name"]}.pdf')
        # plt.show()
        plt.close(fig)


if __name__ == '__main__':
    model_eval(sys.argv[1:])  # for use when debugging with pycharm
