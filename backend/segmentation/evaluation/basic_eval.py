import torch
import torch.nn.functional as F
from tqdm import tqdm

from segmentation.helper_functions.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device):
    """
    TODO: Suggestion - I would convert this function to simply accept a data filepath rather than a complete dataloader.
    :param net: model network
    :param dataloader: dataloader containing dataset to be evaluated
    :param device: device used
    :return:
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to device, set the type of pixel values
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # one_hot format has only a single 1 bit and the rest are 0 bits
        # i.e. if n_classes is 3 will transform [0] to [1,0,0], [1] to [0,1,0], [2] to [0,0,1]
        # The permute() function changes it from [N, H, W, C] to [N, C, H, W]
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format if 3 channels, else apply sigmoid function
            # Calculate dice score
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, image, mask_pred, mask_true
