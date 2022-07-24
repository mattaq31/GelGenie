# code here taken from https://github.com/milesial/Pytorch-UNet/tree/e36c782fbfc976b7326182a47dd7213bd3360a7e
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img, n_channels, scale, is_mask):
        """
        TODO: fill in!
        :param pil_img:
        :param n_channels:
        :param scale:
        :param is_mask:
        :return:
        """
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixels'

        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        if not is_mask:
            if n_channels == 1:
                pil_img = pil_img.convert('L')
            else:
                pil_img = pil_img.convert('RGB')

            return self.standard_image_transform(pil_img)
        else:
            final_img = np.array(pil_img)  # TODO: what happens when we have multiple classes?  Need to search online for best implementation of this
            return torch.from_numpy(final_img)

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
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img_tensor = self.preprocess(img, self.n_channels, self.scale, is_mask=False)
        mask_tensor = self.preprocess(mask, self.n_channels, self.scale, is_mask=True)

        return {
            'image': img_tensor,
            'mask': mask_tensor.int().contiguous()  # TODO: why do we need this .contiguous() call?
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
