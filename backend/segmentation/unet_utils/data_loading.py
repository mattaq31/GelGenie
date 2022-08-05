# code here taken from https://github.com/milesial/Pytorch-UNet/tree/e36c782fbfc976b7326182a47dd7213bd3360a7e
import logging
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
    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 n_channels: int,
                 scale: float = 1.0,
                 mask_suffix: str = '',
                 augmentations=None,
                 padding=False):
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

        max_dimension = 0
        for root, dirs, files in os.walk(self.images_dir):
            for name in files:
                image_file = os.path.join(root, name)
                image = imageio.imread(image_file)
                max_dimension = max(max_dimension, image.shape[0], image.shape[1])
        max_dimension = 32*(max_dimension//32+1)  # to be divisible by 32
        self.max_dimension = max_dimension
        self.augmentations = augmentations

        self.padding = padding

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
            elif image.shape[-1] == 4:  # Actual input: 4 channels
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

        return image

    @staticmethod
    def load_mask(filename):
        pil_mask = Image.open(filename)
        final_mask = np.array(pil_mask)
        unique = np.unique(final_mask)
        final_mask = np.array([[np.where(unique == i)[0][0] for i in j] for j in final_mask])
        return final_mask

    # in your init function - run glob on the dataset folder, this gets all images and puts them in a list
    # 2 when you get your id in __getitem__, just index the above list

    def __getitem__(self, idx):

        img_file = self.image_names[idx]
        mask_file = self.masks_dict[os.path.basename(img_file).split('.')[0]]

        if os.path.basename(img_file).split('.')[0] != os.path.basename(mask_file).split('.')[0]:
            raise RuntimeError('Gel and Mask images do not match')

        img_array = self.load_image(self, filename=img_file, n_channels=self.n_channels)
        mask_array = self.load_mask(mask_file)

        assert img_array.shape[0] == mask_array.shape[0] and \
               img_array.shape[1] == mask_array.shape[1], \
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
