from torch.utils.data import DataLoader, random_split

from segmentation.unet_utils.augmentations import get_training_augmentation
from segmentation.unet_utils.data_loading import BasicDataset


def prep_dataloader(dir_train_img, dir_train_mask, dir_val_img, dir_val_mask,
                    n_channels, img_scale, val_percent, batch_size, num_workers,
                    apply_augmentations, padding):

    # 1. Create dataset
    if apply_augmentations is True:
        train_set = BasicDataset(dir_train_img, dir_train_mask, n_channels, img_scale,
                                 augmentations=get_training_augmentation(), padding=padding)
    elif apply_augmentations is False:
        train_set = BasicDataset(dir_train_img, dir_train_mask, n_channels, img_scale,
                                 augmentations=None, padding=padding)

    val_set = BasicDataset(dir_val_img, dir_val_mask, n_channels, img_scale,
                           augmentations=None, padding=padding)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    n_train = int(len(train_set))
    n_val = int(len(val_set))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, num_workers=1, pin_memory=True)

    return train_loader, val_loader, n_train, n_val


