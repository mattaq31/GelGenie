import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchshow as ts

from segmentation.unet_utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device):
    """
    TODO: fill in.
    TODO: Suggestion - I would convert this function to simply accept a data filepath rather than a complete dataloader.
    :param net:
    :param dataloader:
    :param device:
    :return:
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


def evaluate_epoch(net, dataloader, device, epoch, segmentation_path):
    # TODO: I would delete this and add all functionality to 'evaluate' function.  You could then break that into sub-tasks, or create a class.
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    show_image = None
    show_mask_pred = None
    show_mask_true = None

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # gets the image, prediction mask, and true mask
            show_image = image
            show_mask_pred = mask_pred
            show_mask_true = mask_true

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    show_combi_mask = torch.empty(1, 3, 640, 959)  # TODO: remove this hardcoding!  Can get image dimensions from variables
    for i in range(640):  # TODO: remove this hardcoding!
        for j in range(959):
            if (show_mask_pred[0][0][i][j] < 0.5 and show_mask_pred[0][1][i][j] > 0.5):
                show_combi_mask[0][0][i][j] = 1
                show_combi_mask[0][1][i][j] = 0
                show_combi_mask[0][2][i][j] = 0
            else:
                show_combi_mask[0][0][i][j] = image[0][0][i][j]
                show_combi_mask[0][1][i][j] = image[0][1][i][j]
                show_combi_mask[0][2][i][j] = image[0][2][i][j]

    ts.show([torch.squeeze(show_image), torch.squeeze(show_mask_pred),
             torch.squeeze(show_combi_mask), torch.squeeze(show_mask_true)])
    ts.save([torch.squeeze(show_image), torch.squeeze(show_mask_pred),
             torch.squeeze(show_combi_mask), torch.squeeze(show_mask_true)], segmentation_path + f'/epoch{epoch}.jpg')
    # TODO: add more info to plots e.g. legend and captions.

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
