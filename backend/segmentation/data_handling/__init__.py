from torch.utils.data import DataLoader
import numpy as np
from segmentation.data_handling.dataloaders import BasicDataset
from.augmentations import get_training_augmentation
from segmentation.helper_functions.general_functions import extract_image_names_from_folder


def prep_dataloader(dir_train_img, dir_train_mask, split_training_dataset, dir_val_img, dir_val_mask,
                    n_channels, img_scale, val_percent, batch_size, num_workers,
                    apply_augmentations, padding):
    """
    TODO: fill in documentation
    :param dir_train_img:
    :param dir_train_mask:
    :param split_training_dataset:
    :param dir_val_img:
    :param dir_val_mask:
    :param n_channels:
    :param img_scale:
    :param val_percent:
    :param batch_size:
    :param num_workers:
    :param apply_augmentations:
    :param padding:
    :return:
    """

    # 1. Create dataset
    if split_training_dataset:
        image_names = extract_image_names_from_folder(dir_train_img)

        n_val = int(len(image_names) * val_percent)
        n_train = len(image_names) - n_val

        rng = np.random.default_rng()
        rng.shuffle(image_names, axis=0)
        val_image_names = image_names[:n_val]
        train_image_names = image_names[n_val:]
    else:
        train_image_names = None
        val_image_names = None

    if apply_augmentations:
        train_set = BasicDataset(dir_train_img, dir_train_mask, n_channels, img_scale,
                                 augmentations=get_training_augmentation(), padding=padding,
                                 image_names=train_image_names)
    else:
        train_set = BasicDataset(dir_train_img, dir_train_mask, n_channels, img_scale,
                                 augmentations=None, padding=padding,
                                 image_names=train_image_names)
    val_set = BasicDataset(dir_val_img, dir_val_mask, n_channels, img_scale,
                           augmentations=None, padding=padding,
                           image_names=val_image_names)
    # 2. Split into train / validation partitions TODO: restore this functionality, make it user-controlled
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    n_train = int(len(train_set))
    n_val = int(len(val_set))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=1, num_workers=1, pin_memory=True)

    return train_loader, val_loader, n_train, n_val
