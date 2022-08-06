# code here modified from https://github.com/milesial/Pytorch-UNet/tree/e36c782fbfc976b7326182a47dd7213bd3360a7e
import logging
from pathlib import Path
import imageio
import cv2
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from segmentation.helper_functions.general_functions import extract_image_names_from_folder


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, n_channels: int, scale: float = 1.0, mask_suffix: str = '',
                 augmentations=None, padding: bool = False):
        """
        TODO: fill in documentation
        :param images_dir:
        :param masks_dir:
        :param n_channels:
        :param scale:
        :param mask_suffix:
        :param augmentations:
        :param padding:
        """

        assert (n_channels == 1 or n_channels == 3), 'Dataset number of channels must be either 1 or 3'
        assert 0 < scale <= 1, 'Image scaling must be between 0 and 1'

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.n_channels = n_channels
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.standard_image_transform = transforms.Compose([transforms.ToTensor()])
        self.image_names = extract_image_names_from_folder(images_dir)
        self.mask_names = extract_image_names_from_folder(masks_dir)

        # this step allows the image and mask to have different file extensions
        self.masks_dict = {os.path.basename(mask).split('.')[0]: mask for mask in self.mask_names}
        self.augmentations = augmentations
        self.padding = padding

        if padding:
            max_dimension = 0
            # loops through provided images and extracts the largest image dimension, for use if padding is selected
            for root, dirs, files in os.walk(self.images_dir):
                for name in files:
                    image_file = os.path.join(root, name)
                    image = imageio.imread(image_file)  # TODO: investigate the warning here...
                    max_dimension = max(max_dimension, image.shape[0], image.shape[1])
            max_dimension = 32 * (max_dimension // 32 + 1)  # to be divisible by 32 TODO: why?

            self.max_dimension = max_dimension

        if not self.image_names:
            raise RuntimeError(f'No images found in {images_dir}, make sure you put your images there')
        if not self.mask_names:
            raise RuntimeError(f'No images found in {masks_dir}, make sure you put your masks there')
        logging.info(f'Creating dataset with {len(self.image_names)} examples')

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def load_image(self, filename, n_channels):
        image = imageio.imread(filename)

        # Converts to desired number of channels
        if n_channels == 1:  # Target input: 1 channel
            if image.shape[-1] == 3:  # Actual input: 3 channels
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[-1] == 4:  # Actual input: 4 channels
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            # No change required for already grayscale images
        elif n_channels == 3:  # Target input: 3 channels
            if image.shape[-1] == 4:  # Actual input: 4 channels
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[-1] != 3:  # Actual input: 1 channels
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Normalizing image
        if image.dtype == 'uint8':
            max_val = 255
        elif image.dtype == 'uint16':
            max_val = 65535
        else:
            raise RuntimeError('Image type not recognized.')

        image = image.astype(np.float32) / max_val

        return image

    @staticmethod
    def load_mask(filename):
        mask = np.array(Image.open(filename))
        unique = np.unique(mask)
        # TODO: add a description for the below
        final_mask = np.array([[np.where(unique == i)[0][0] for i in j] for j in mask])
        return final_mask

    def __getitem__(self, idx):

        img_file = self.image_names[idx]
        mask_file = self.masks_dict[os.path.basename(img_file).split('.')[0]]

        if os.path.basename(img_file).split('.')[0] != os.path.basename(mask_file).split('.')[0]:
            raise RuntimeError('Gel and mask images do not match - there is some mismatch in the data folders provided')

        img_array = self.load_image(self, filename=img_file, n_channels=self.n_channels)
        mask_array = self.load_mask(mask_file)

        assert img_array.shape == mask_array.shape, \
            f'Image and mask should be the same size, but are {img_array.shape} and {mask_array.shape}'

        if self.augmentations:
            sample = self.augmentations(image=img_array, mask=mask_array)
            img_array = sample['image']
            mask_array = sample['mask']

        if self.padding:
            top = (self.max_dimension - img_array.shape[0]) // 2
            bottom = self.max_dimension - img_array.shape[0] - top
            left = (self.max_dimension - img_array.shape[1]) // 2
            right = self.max_dimension - img_array.shape[1] - left

            img_array = np.pad(img_array, pad_width=((top, bottom), (left, right)), mode='constant')
            mask_array = np.pad(mask_array, pad_width=((top, bottom), (left, right)), mode='constant')

        img_tensor = self.standard_image_transform(img_array)
        mask_tensor = torch.from_numpy(mask_array)

        return {
            'image': img_tensor,
            'mask': mask_tensor.int().contiguous()  # TODO: why do we need this .contiguous() call?
        }
