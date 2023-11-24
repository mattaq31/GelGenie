from torch.utils.data import DataLoader
import numpy as np
from gelgenie.segmentation.data_handling.dataloaders import ImageMaskDataset
from .augmentations import get_training_augmentation, get_nondestructive_training_augmentation
from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder


def prep_train_val_dataloaders(dir_train_img, dir_train_mask, split_training_dataset, dir_val_img, dir_val_mask,
                               n_channels, val_percent, batch_size, num_workers,
                               apply_augmentations, weak_augmentations, padding, individual_padding):
    """
    Prepares a matched training and validation dataloader for training a segmentation model.
    :param dir_train_img: Path of directory of training set images
    :param dir_train_mask: Path of directory of training set masks
    :param split_training_dataset: (Bool) Whether to split training dataset into training/ validation datasets
    :param dir_val_img: Path of directory of validation set images
    :param dir_val_mask: Path of directory of validation set masks
    :param n_channels: (int) Number of colour channels for model input
    :param val_percent: (float) % of the data that is used as validation normalized between 0 and 1
    :param batch_size: (int) Number of images loaded per batch
    :param num_workers: (int) Number of workers for dataloader (parallel dataloader threads speed up data processing)
    :param apply_augmentations: (Bool) Whether to apply augmentations when loading training images
    :param weak_augmentations: (Bool) Set to true to only allow non-destructive augmentations
    :param padding: (Bool) Whether to apply padding to images and masks when loading training and validation images
    :param individual_padding (Bool) Whether to apply padding to images and masks individually (only batch size of 1 possible)
    :return: Training dataloader, Validation dataloader, number of training images, number of validation images
    """

    # Create datasets
    if split_training_dataset:  # Split into train / validation partitions TODO: extend this to the multiple folders case
        image_names = extract_image_names_from_folder(dir_train_img)

        # Calculate expected length of validation set
        n_val = int(len(image_names) * val_percent)

        rng = np.random.default_rng()
        rng.shuffle(image_names, axis=0)
        # Slice off the random images with expected length (without overlap between sets)
        val_image_names = image_names[:n_val]
        train_image_names = image_names[n_val:]
        if dir_val_img:
            raise RuntimeError('Did not expect to have a validation set directory when splitting training set.')
        dir_val_img = dir_train_img
        dir_val_mask = dir_train_mask
    else:
        train_image_names = None
        val_image_names = None

    if weak_augmentations:
        augmentations = get_nondestructive_training_augmentation()
    else:
        augmentations = get_training_augmentation()

    train_set = ImageMaskDataset(dir_train_img, dir_train_mask, n_channels,
                                 augmentations=augmentations if apply_augmentations else None,
                                 padding=padding, individual_padding=individual_padding, image_names=train_image_names)

    val_set = ImageMaskDataset(dir_val_img, dir_val_mask, n_channels,  # validation set enforced to not have extra padding (as will be the case at test time)
                               augmentations=None, padding=False, individual_padding=True,
                               image_names=val_image_names)

    # Confirm the length of training/validation sets
    n_train = len(train_set)
    n_val = len(val_set)

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, num_workers=1, pin_memory=True)
    # TODO: can validation settings be improved?

    return train_loader, val_loader, n_train, n_val
