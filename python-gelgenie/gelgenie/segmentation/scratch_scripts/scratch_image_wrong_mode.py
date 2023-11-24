from gelgenie.segmentation.data_handling import prep_train_val_dataloaders
from tqdm import tqdm

from gelgenie.segmentation.unet import UNet

dir_img = 'C:/2022_Summer_Intern/Gel_Images_UNet_Test/Images_Q1+Q2+selected/'
dir_mask = 'C:/2022_Summer_Intern/Gel_Images_UNet_Test/Masks_Q1+Q2+selected/'
n_channels = 1
img_scale = 0.5
val_percent = 0.1
batch_size = 1
num_workers = 1
epochs = 10
classes = 1
bilinear = False

if __name__ == '__main__':
    net = UNet(n_channels=int(n_channels), n_classes=classes, bilinear=bilinear)

    train_loader, val_loader, n_train, n_val = \
            prep_train_val_dataloaders(dir_img, dir_mask, n_channels, img_scale, val_percent, batch_size, num_workers)

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
