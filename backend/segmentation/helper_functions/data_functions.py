from torch.utils.data import DataLoader, random_split
from segmentation.unet_utils.data_loading import BasicDataset
import torch


def prep_dataloader(dir_img, dir_mask, n_channels, img_scale, val_percent, batch_size, num_workers):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, n_channels, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1, num_workers=1, pin_memory=True)

    return train_loader, val_loader, n_train, n_val


