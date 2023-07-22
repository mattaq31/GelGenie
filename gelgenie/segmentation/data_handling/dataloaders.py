# code here modified from https://github.com/milesial/Pytorch-UNet/tree/e36c782fbfc976b7326182a47dd7213bd3360a7e
from pathlib import Path
import imageio
import cv2
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from rich import print as rprint

from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder


class ImageMaskDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, n_channels: int, mask_suffix: str = '',
                 augmentations=None, padding: bool = False, image_names=None):
        """
        :param images_dir: Path of image directory
        :param masks_dir: Path of mask directory
        :param n_channels: (int) Number of colour channels for model input
        :param mask_suffix: (string) suffix added to mask files
        :param augmentations: getter function for augmentation function
        :param padding: (Bool) Whether to apply padding to images and masks
        :param image_names: ([String]) List of image names selected
        """

        assert (n_channels == 1 or n_channels == 3), 'Dataset number of channels must be either 1 or 3'
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.n_channels = n_channels
        self.mask_suffix = mask_suffix
        self.standard_image_transform = transforms.Compose([transforms.ToTensor()])  # Transforms image to tensor
        if image_names is not None:
            self.image_names = image_names  # Only include selected image names
        else:
            self.image_names = extract_image_names_from_folder(images_dir)  # Select all image names from directory

        self.mask_names = extract_image_names_from_folder(masks_dir)  # Select all mask names from directory

        # this step allows the image and mask to have different file extensions
        self.masks_dict = {os.path.basename(mask).split('.')[0]: mask for mask in self.mask_names}
        self.augmentations = augmentations
        self.padding = padding

        if padding:
            max_dimension = 0
            # loops through provided images and extracts the largest image dimension for use if padding is selected
            for root, dirs, files in os.walk(self.images_dir):
                for name in files:
                    image_file = os.path.join(root, name)
                    image = imageio.v2.imread(image_file)  # TODO: does this need updating?
                    max_dimension = max(max_dimension, image.shape[0], image.shape[1])
            max_dimension = 32 * (max_dimension // 32 + 1)  # to be divisible by 32 as required by smp-UNet/ UNet++

            self.max_dimension = max_dimension

        if not self.image_names:
            raise RuntimeError(f'No images found in {images_dir}, make sure you put your images there.')
        if not self.mask_names:
            raise RuntimeError(f'No images found in {masks_dir}, make sure you put your masks there.')
        rprint(f'[bold blue]Created dataset with {len(self.image_names)} images.[/bold blue]')

    def __len__(self):  # Gets length of dataset
        return len(self.image_names)

    @staticmethod
    def load_image(filename, n_channels):
        image = imageio.v2.imread(filename)

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
            max_val = 255  # values are 0 - 256
        elif image.dtype == 'uint16':
            max_val = 65535  # values are 0 - 65536
        else:
            raise RuntimeError(f'Image type {image.dtype} not recognized, only accepts uint8 or uint16 for now.')

        image = image.astype(np.float32) / max_val

        return image

    @staticmethod
    def load_mask(filename):
        mask = np.array(Image.open(filename))
        # masks are specially prepared for easy reading, so there shouldn't the need for any further processing.
        # However, I have left a check here to indicate if something changes in the input data.
        unique = np.unique(mask)  # Acquire unique pixel values of the mask
        if not all(unique == [0, 1]):
            raise RuntimeError('Mask data does not match expected format.')

        # Previous processing:
        # Final pixel value = Index of original pixel value in the list of unique pixel values
        # np.where(unique==i) creates a tuple containing array of indexes, in this case only 1 index will be available
        # i.e. [5, 9, 7] --> [0, 2, 1]
        # final_mask = np.array([[np.where(unique == i)[0][0] for i in j] for j in mask])

        return mask

    def __getitem__(self, idx):
        img_file = self.image_names[idx]
        mask_file = self.masks_dict[os.path.basename(img_file).split('.')[0]]  # Get mask file with same name

        if os.path.basename(img_file).split('.')[0] != os.path.basename(mask_file).split('.')[0]:
            raise RuntimeError('Gel and mask images do not match - there is some mismatch in the data folders provided')

        img_array = self.load_image(filename=img_file, n_channels=self.n_channels)
        mask_array = self.load_mask(mask_file)

        assert img_array.shape == mask_array.shape, \
            f'Image and mask should be the same size, but are {img_array.shape} and {mask_array.shape}'

        if self.augmentations:
            sample = self.augmentations(image=img_array, mask=mask_array)  # Apply augmentations
            img_array = sample['image']
            mask_array = sample['mask']

        if self.padding:
            top = (self.max_dimension - img_array.shape[0]) // 2  # Get amount of pixels to pad at top of image
            bottom = self.max_dimension - img_array.shape[0] - top  # Get amount of pixels to pad at bottom of image
            left = (self.max_dimension - img_array.shape[1]) // 2  # Get amount of pixels to pad at left of image
            right = self.max_dimension - img_array.shape[1] - left  # Get amount of pixels to pad at right of image

            img_array = np.pad(img_array, pad_width=((top, bottom), (left, right)), mode='constant')
            mask_array = np.pad(mask_array, pad_width=((top, bottom), (left, right)), mode='constant')

        img_tensor = self.standard_image_transform(img_array)
        mask_tensor = torch.from_numpy(mask_array)
        return {
            'image': img_tensor,
            'image_name': os.path.basename(img_file).split('.')[0],
            'mask': mask_tensor.int()
        }


class ImageDataset(ImageMaskDataset):  # TODO: fix the inheritance here - should be the other way round!
    def __init__(self, images_dir: str, n_channels: int, padding: bool = False):
        """
        For datasets of images only, used in model_eval if no masks are provided
        :param images_dir: Path of image directory
        :param n_channels: (int) Number of colour channels for model input
        :param padding: (Bool) Whether to apply padding
        """
        super().__init__(images_dir, images_dir, n_channels=n_channels, padding=padding)
        self.images_dir = Path(images_dir)
        self.n_channels = n_channels
        self.standard_image_transform = transforms.Compose([transforms.ToTensor()])
        self.padding = padding

        if padding:
            max_dimension = 0
            # loops through provided images and extracts the largest image dimension, for use if padding is selected
            for root, dirs, files in os.walk(self.images_dir):
                for name in files:
                    image_file = os.path.join(root, name)
                    image = imageio.v2.imread(image_file)
                    max_dimension = max(max_dimension, image.shape[0], image.shape[1])
            max_dimension = 32 * (max_dimension // 32 + 1)  # to be divisible by 32 as required by smp-UNet/ UNet++

            self.max_dimension = max_dimension

    def __getitem__(self, idx):
        img_file = self.image_names[idx]

        img_array = self.load_image(filename=img_file, n_channels=self.n_channels)

        if self.augmentations:
            sample = self.augmentations(image=img_array)  # Apply augmentations
            img_array = sample['image']

        if self.padding:
            top = (self.max_dimension - img_array.shape[0]) // 2  # Get amount of pixels to pad at top of image
            bottom = self.max_dimension - img_array.shape[0] - top  # Get amount of pixels to pad at bottom of image
            left = (self.max_dimension - img_array.shape[1]) // 2  # Get amount of pixels to pad at left of image
            right = self.max_dimension - img_array.shape[1] - left  # Get amount of pixels to pad at right of image
            img_array = np.pad(img_array, pad_width=((top, bottom), (left, right)), mode='constant')  # Pad with 0 value

        img_tensor = self.standard_image_transform(img_array)

        return {
            'image': img_tensor,
            'image_name': os.path.basename(img_file).split('.')[0],
        }
