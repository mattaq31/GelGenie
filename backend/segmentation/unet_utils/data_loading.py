# code here taken from https://github.com/milesial/Pytorch-UNet/tree/e36c782fbfc976b7326182a47dd7213bd3360a7e
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from segmentation.helper_functions.general_functions import extract_image_names_from_folder
import torchvision.transforms as transforms
import imageio
import cv2
import os


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, n_channels: int, scale: float = 1.0, mask_suffix: str = ''):
        """
        TODO: fill in!
        :param images_dir:
        :param masks_dir:
        :param n_channels:
        :param scale:
        :param mask_suffix:
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert (n_channels == 1 or n_channels == 3), 'Number of channels must be either 1 or 3'
        self.n_channels = n_channels
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.standard_image_transform = transforms.Compose([transforms.ToTensor()])

        self.image_names = extract_image_names_from_folder(images_dir)
        self.mask_names = extract_image_names_from_folder(masks_dir)
        self.masks_dict = {os.path.basename(mask).split('.')[0]: mask for mask in self.mask_names}

        if not self.image_names:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        if not self.mask_names:
            raise RuntimeError(f'No input file found in {masks_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.image_names)} examples')

    def __len__(self):
        return len(self.image_names)


    @staticmethod
    def load_image(self, filename, n_channels):
        image = imageio.imread(filename)

        # Converting to desired number of channels
        if n_channels == 1:  # Target input: 1 channel
            if image.shape[-1] == 3:  # Actual input: 3 channels
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[-1] == 4: # Actual input: 4 channels
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            # No change required for already grayscale images
        elif n_channels == 3:  # Target input: 3 channels
            if image.shaoe[-1] == 4:  # Actual input: 4 channels
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[-1] != 3:  # Actual input: 1 channels
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


        # Normalizing image
        if image.dtype == 'uint8':
            max_val = 255
        elif image.dtype == 'uint16':
            max_val = 65535
        image = image.astype(np.float32) / (max_val - 0)

        return self.standard_image_transform(image)

    @staticmethod
    def load_mask(filename):
        pil_mask = Image.open(filename)
        final_mask = np.array(pil_mask)
        unique = np.unique(final_mask)
        final_mask = np.array([[np.where(unique == i)[0][0] for i in j] for j in final_mask])
        return torch.from_numpy(final_mask)

# in your init function - run glob on the dataset folder, this gets all images and puts them in a list
    #2 when you get your id in __getitem__, just index the above list

    def __getitem__(self, idx):

        img_file = self.image_names[idx]
        mask_file = self.masks_dict[os.path.basename(img_file).split('.')[0]]

        if os.path.basename(img_file).split('.')[0] != os.path.basename(mask_file).split('.')[0]:
            raise RuntimeError('Gel and Mask images do not match')

        img_tensor = self.load_image(self, filename=img_file, n_channels=self.n_channels)
        mask_tensor = self.load_mask(mask_file)

        assert img_tensor.size(dim=-2) == mask_tensor.size(dim=-2) and \
               img_tensor.size(dim=-1) == mask_tensor.size(dim=-1), \
            f'Image and mask should be the same size, but are {img_tensor.size} and {mask_tensor.size}'

        return {
            'image': img_tensor,
            'mask': mask_tensor.int().contiguous()  # TODO: why do we need this .contiguous() call?
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
