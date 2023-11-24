from matplotlib import pyplot as plt
from tqdm import tqdm
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from gelgenie.segmentation.unet import UNet


def segmentationEval(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)

    images = []
    predicted_masks = []

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image = batch['image']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # gets the image, prediction mask, and true mask
        images.append(image)
        predicted_masks.append(mask_pred)

    net.train()

    return images, predicted_masks

class ImageDataset(Dataset):
    def __init__(self, images_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        pil_img = pil_img.resize((512, 680), resample=Image.BICUBIC)

        img_ndarray = np.asarray(pil_img)  # (H, W, C)
        image_transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = image_transform(img_ndarray) # (C, H, W)
        return image_tensor



    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):

        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = self.load(img_file[0])

        img_tensor = self.preprocess(img, self.scale)

        return {
            'image': img_tensor.float().contiguous(),
        }

def show_segmentation(images, predicted_masks, segmentation_path):
    combined_mask_array = []
    images_array = []
    masks_array = []

    for image, mask_pred in images, predicted_masks:
        combi_mask = image.detach().clone().squeeze()  # Copies the image into np array
        for i in range(image.size(dim=1)):  # Height
            for j in range(image.size(dim=2)):  # Width
                if mask_pred[0][i][j] < 0.5 and mask_pred[1][i][j] > 0.5:  # is labelled
                    combi_mask[0][i][j] = 1
                    combi_mask[1][i][j] = 0
                    combi_mask[2][i][j] = 0

        def combine_channels(mask_array):
            height = mask_array.shape[1]
            width = mask_array.shape[2]
            combined_array = np.tile(0, (height, width))
            for h in range(height):
                for w in range(width):
                    if mask_array[1][h][w] > 0.5:
                        combined_array[h][w] = 1  # is labelled
            return combined_array

        image_array = np.transpose(image.detach().squeeze().cpu().numpy(), (1, 2, 0))  # Copies the image into np array
        mask_pred_array = combine_channels(mask_pred.detach().squeeze().cpu().numpy())  # Copies prediction into np array
        combi_mask = np.transpose(combi_mask.detach().squeeze().cpu().numpy(), (1, 2, 0))

        combined_mask_array.append(combi_mask)
        images_array.append(image_array)
        masks_array.append(mask_pred_array)

    fig, axs = plt.subplots(nrows=len(images_array), ncols=3, figsize=(10, 7))
    for row in range(len(axs)):
        axs[row][0].imshow(images_array[row])
        axs[row][1].imshow(combi_mask[row])
        axs[row][2].imshow(mask_pred_array[row])

    axs[0][0].set_title('Original Image')
    axs[0][1].set_title('Mask Prediction Superimposed')
    axs[0][2].set_title('Mask Prediction')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])  # remove ticks
    plt.tight_layout()
    plt.savefig(Path(segmentation_path + 'segmentations.pdf'))
    plt.close(fig)

    return None



if __name__ == '__main__':
    dir_img = "C:/2022_Summer_Intern/QuPath_Cell_Detection_Test/Original Image"
    img_scale = 0.5
    segmentation_path = "C:/2022_Summer_Intern/Gel_Images_UNet_Test/Segmentation_Test/"
    checkpoint_path = "C:/Users/s2137314/Downloads/checkpoint_epoch206.pth"

    # 1. Create dataset
    segmentation_set = ImageDataset(dir_img, img_scale)

    # 2. Create data loaders
    batch_size = 1
    num_workers = 1
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    segmentation_loader = DataLoader(segmentation_set, shuffle=False, drop_last=True, batch_size=1, num_workers=1,
                                     pin_memory=True)

    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    net.train()
    modelweights = torch.load(f=checkpoint_path, map_location=torch.device("cpu"))
    net.load_state_dict(state_dict=modelweights)

    images, predicted_masks = segmentationEval(net, segmentation_loader, device=torch.device("cpu"))
    show_segmentation(images, predicted_masks, segmentation_path)
