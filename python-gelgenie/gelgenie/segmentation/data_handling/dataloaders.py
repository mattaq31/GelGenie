"""
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

import imageio
import cv2
import os
from os.path import join
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from rich import print as rprint

from gelgenie.segmentation.helper_functions.general_functions import extract_image_names_from_folder


class ImageDataset(Dataset):
    def __init__(self, images_dir: str, n_channels: int, padding: bool = False,
                 individual_padding=False, image_names=None, augmentations=None,
                 minmax_norm=False, percentile_norm=False):
        """
        For datasets of images only, used in model_eval if no masks are provided
        :param images_dir: Path of image directory
        :param n_channels: (int) Number of colour channels for model input
        :param augmentations: getter function for augmentation function
        :param padding: (Bool) Whether to apply padding to images and masks to a constant value for the entire dataset
        :param individual_padding: (Bool) Whether to apply padding to images and masks individually (for UNet)
        :param image_names: ([String]) List of image names selected
        :param minmax_norm: (Bool) Whether to apply minmax normalization to images (unique normalisation for each image)
        :param percentile_norm: (Bool) Whether to apply percentile normalization to images (unique normalisation for each image)
        """

        if percentile_norm and minmax_norm:
            raise RuntimeError('Cannot have both percentile and minmax normalization.')

        self.n_channels = n_channels
        self.minmax_norm = minmax_norm
        self.percentile_norm = percentile_norm
        self.standard_image_transform = transforms.Compose([transforms.ToTensor()])  # Transforms image to tensor

        self.image_folders = images_dir

        self.image_names = []
        self.image_finding_logic(image_names)

        self.augmentations = augmentations
        self.padding = padding
        self.individual_padding = individual_padding
        self.data_metrics = self.extract_full_dataset_metrics()
        self.max_dimension = self.data_metrics['Max Dimension']

        if not self.image_names:
            raise RuntimeError(f'No images found in {images_dir}, make sure you put your images there.')

        if len(self.image_names) == 1:
            rprint(f'[bold blue]Created dataset with {len(self.image_names)} image.[/bold blue]')
        else:
            rprint(f'[bold blue]Created dataset with {len(self.image_names)} images.[/bold blue]')

    def image_finding_logic(self, image_names):
        if image_names is not None:
            self.image_names = image_names  # Only include selected image names
        else:
            if isinstance(self.image_folders, list):  # include images from multiple directories
                for im_dir in self.image_folders:
                    curr_folder_images = extract_image_names_from_folder(im_dir)
                    self.image_names.extend(curr_folder_images)
            else:
                self.image_names.extend(extract_image_names_from_folder(self.image_folders))

    def extract_full_dataset_metrics(self):
        max_dimension = 0
        # loops through provided images and extracts the largest image dimension for use if padding is selected
        class_counts = np.zeros((1, 2), dtype=int)
        for file in self.image_names:  # TODO: should this be changed to rectangular rather than square images?
            image = imageio.v2.imread(file)  # TODO: does this need updating?
            max_dimension = max(max_dimension, image.shape[0], image.shape[1])

        max_dimension = 32 * (max_dimension // 32 + 1)  # to be divisible by 32 as required by smp-UNet/ UNet++
        if self.padding:
            rprint(f'[bold blue]Padding images to {max_dimension}x{max_dimension}[/bold blue]')

        return {'Max Dimension': max_dimension}

    def __len__(self):  # Gets length of dataset
        return len(self.image_names)

    @staticmethod
    def channel_converter(image, n_channels):
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
        return image

    def load_image(self, filename):
        image = imageio.v2.imread(filename)

        image = ImageDataset.channel_converter(image, self.n_channels)

        # Normalizing image
        if self.minmax_norm:
            min_pixel = np.min(image)
            max_pixel = np.max(image)
            image = (image.astype(np.float32) - min_pixel) / (max_pixel-min_pixel)
        elif self.percentile_norm:
            # calculate top 0.1 and bottom 0.1% of pixels to discard
            min_pixel = np.percentile(image, 0.1)
            max_pixel = np.percentile(image, 99.9)
            # Clip the image to the percentile range and normalize
            image = (np.clip(image, min_pixel, max_pixel).astype(np.float32) - min_pixel) / (max_pixel - min_pixel)
        else:
            if image.dtype == 'uint8':
                max_val = 255  # values are 0 - 255
            elif image.dtype == 'uint16':
                max_val = 65535  # values are 0 - 65535
            else:
                raise RuntimeError(f'Image type {image.dtype} not recognized, only accepts uint8 or uint16 for now.')

            image = image.astype(np.float32) / max_val

        return image

    def get_padding(self, img_array):
        top = (self.max_dimension - img_array.shape[0]) // 2  # Get amount of pixels to pad at top of image
        bottom = self.max_dimension - img_array.shape[0] - top  # Get amount of pixels to pad at bottom of image
        left = (self.max_dimension - img_array.shape[1]) // 2  # Get amount of pixels to pad at left of image
        right = self.max_dimension - img_array.shape[1] - left  # Get amount of pixels to pad at right of image
        return (top, bottom), (left, right)

    @staticmethod
    def get_individual_padding(img_array):
        new_vert = 32 * (img_array.shape[0] // 32 + 1)
        new_horiz = 32 * (img_array.shape[1] // 32 + 1)

        top = (new_vert - img_array.shape[0]) // 2  # Get amount of pixels to pad at top of image
        bottom = new_vert - img_array.shape[0] - top  # Get amount of pixels to pad at bottom of image
        left = (new_horiz - img_array.shape[1]) // 2  # Get amount of pixels to pad at left of image
        right = new_horiz - img_array.shape[1] - left  # Get amount of pixels to pad at right of image
        return (top, bottom), (left, right)

    def __getitem__(self, idx):
        img_file = self.image_names[idx]

        img_array = self.load_image(filename=img_file)
        orig_height, orig_width = img_array.shape[0], img_array.shape[1]

        if self.augmentations:
            sample = self.augmentations(image=img_array)  # Apply augmentations
            img_array = sample['image']

        if self.padding:
            vertical, horizontal = self.get_padding(img_array)
            img_array = np.pad(img_array, pad_width=(vertical, horizontal), mode='constant')  # Pad with 0 value
        elif self.individual_padding:
            vertical, horizontal = self.get_individual_padding(img_array)
            img_array = np.pad(img_array, pad_width=(vertical, horizontal), mode='constant')  # Pad with 0 value

        img_tensor = self.standard_image_transform(img_array)

        return {
            'image': img_tensor,
            'image_name': os.path.basename(img_file).split('.')[0],
            'image_height': orig_height,
            'image_width': orig_width
        }


class ImageMaskDataset(ImageDataset):
    def __init__(self, images_dir, masks_dir, n_channels: int, mask_suffix: str = '.tif',
                 augmentations=None, padding: bool = False, individual_padding=False, image_names=None,
                 minmax_norm=False, percentile_norm=False):
        """
        :param images_dir: Path of image directory
        :param masks_dir: Path of mask directory
        :param n_channels: (int) Number of colour channels for model input
        :param mask_suffix: (string) suffix added to mask files
        :param augmentations: getter function for augmentation function
        :param padding: (Bool) Whether to apply padding to images and masks
        :param individual_padding: (Bool) Whether to apply padding to images and masks individually (for UNet)
        :param image_names: ([String]) List of image names selected
        :param minmax_norm: (Bool) Whether to apply minmax normalization to images (unique normalisation for each image)
        :param percentile_norm: (Bool) Whether to apply percentile normalization to images (unique normalisation for each image)
        """
        self.mask_suffix = mask_suffix
        self.mask_names = []
        self.masks_dirs = masks_dir
        super().__init__(images_dir, n_channels=n_channels, padding=padding, individual_padding=individual_padding,
                         image_names=image_names, augmentations=augmentations, minmax_norm=minmax_norm,
                         percentile_norm=percentile_norm)

        self.class_weighting = self.data_metrics['Class Weighting']

        if not self.mask_names:
            raise RuntimeError(f'No images found in {masks_dir}, make sure you put your masks there.')
        if len(self.mask_names) != len(self.image_names):
            raise RuntimeError(f'Number of images and masks do not match, there are {len(self.image_names)} images '
                               f'and {len(self.mask_names)} masks.')

    def image_finding_logic(self, image_names):
        if image_names is not None:
            self.image_names = image_names  # Only include selected image names
            for curr_im in image_names:
                corresponding_mask_im = join(self.masks_dirs, os.path.basename(curr_im).split('.')[0] + self.mask_suffix)
                if not os.path.isfile(corresponding_mask_im):
                    raise RuntimeError(f'Could not find corresponding mask image for {curr_im}.')
                self.mask_names.append(corresponding_mask_im)
        else:
            if isinstance(self.image_folders, list):  # include images from multiple directories
                if not isinstance(self.masks_dirs, list):
                    raise RuntimeError('If images_dir is a list, masks_dir must also be a list')
                for im_dir, mask_dir in zip(self.image_folders, self.masks_dirs):
                    curr_folder_images = extract_image_names_from_folder(im_dir)
                    for curr_im in curr_folder_images:
                        corresponding_mask_im = join(mask_dir, os.path.basename(curr_im).split('.')[0] + self.mask_suffix)
                        if not os.path.isfile(corresponding_mask_im):
                            raise RuntimeError(f'Could not find corresponding mask image for {curr_im} '
                                               f'or mask/image folder ordering does not match.')
                        self.mask_names.append(corresponding_mask_im)
                    self.image_names.extend(curr_folder_images)
            else:
                self.image_names.extend(extract_image_names_from_folder(self.image_folders))
                self.mask_names.extend(extract_image_names_from_folder(self.masks_dirs))

    def extract_full_dataset_metrics(self):
        max_dimension = 0
        # loops through provided images and extracts the largest image dimension for use if padding is selected
        class_counts = np.zeros((1, 2), dtype=int)
        for file in self.mask_names:  # TODO: should this be changed to rectangular rather than square images?
            image = imageio.v2.imread(file)  # TODO: does this need updating?
            max_dimension = max(max_dimension, image.shape[0], image.shape[1])
            unique, counts = np.unique(image, return_counts=True)
            class_counts[0, unique] += counts

        max_dimension = 32 * (max_dimension // 32 + 1)  # to be divisible by 32 as required by smp-UNet/ UNet++

        class_weighting = np.sum(class_counts) / (2 * class_counts)  # calculates class weighting

        if self.padding:
            rprint(f'[bold blue]Padding images to {max_dimension}x{max_dimension}[/bold blue]')

        rprint(f'[bold blue]Class weighting is {class_weighting[:, 0]} for background, '
               f'{class_weighting[:, 1]} for bands[/bold blue]')

        return {'Max Dimension': max_dimension, 'Class Weighting': class_weighting}

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
        mask_file = self.mask_names[idx]  # Get mask file with same name

        if os.path.basename(img_file).split('.')[0] != os.path.basename(mask_file).split('.')[0]:
            raise RuntimeError('Gel and mask images do not match - there is some mismatch in the data folders provided')

        img_array = self.load_image(filename=img_file)
        mask_array = self.load_mask(mask_file)
        orig_height, orig_width = img_array.shape[0], img_array.shape[1]

        assert img_array.shape == mask_array.shape, \
            f'Image and mask should be the same size, but are {img_array.shape} and {mask_array.shape}'

        if self.augmentations:
            sample = self.augmentations(image=img_array, mask=mask_array)  # Apply augmentations
            img_array = sample['image']
            mask_array = sample['mask']

        if self.padding:
            vertical, horizontal = self.get_padding(img_array)
            img_array = np.pad(img_array, pad_width=(vertical, horizontal), mode='constant')  # Pad with 0 value
            mask_array = np.pad(mask_array, pad_width=(vertical, horizontal), mode='constant')
        elif self.individual_padding:
            vertical, horizontal = self.get_individual_padding(img_array)
            img_array = np.pad(img_array, pad_width=(vertical, horizontal), mode='constant')  # Pad with 0 value
            mask_array = np.pad(mask_array, pad_width=(vertical, horizontal), mode='constant')

        img_tensor = self.standard_image_transform(img_array)
        mask_tensor = torch.from_numpy(mask_array)
        return {
            'image': img_tensor,
            'image_name': os.path.basename(img_file).split('.')[0],
            'mask': mask_tensor.int(),
            'image_height': orig_height,
            'image_width': orig_width
        }

