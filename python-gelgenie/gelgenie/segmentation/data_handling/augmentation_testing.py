import imageio
import os
from torch.utils.data import DataLoader
import numpy as np

from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset
from gelgenie.segmentation.data_handling import get_training_augmentation
import albumentations as albu


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


images = '/Users/matt/Desktop/inputs'
images = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels_2/images'

output = '/Users/matt/Desktop/deter_test_images_3'

create_dir_if_empty(output)

val_set = ImageDataset(images, 1, padding=False, augmentations=get_training_augmentation())

dataloader = DataLoader(val_set, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

configs = [
    ('Brightness', albu.RandomBrightness(limit=0.2, p=1)),
    ('Gamma', albu.RandomGamma(p=1)),
    ('Hue', albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=1.0)),  # Saturation
    ('Blur', albu.AdvancedBlur(p=1)),
    # ('BlurGlass', albu.GlassBlur(p=1)),
    ('MotionBlur', albu.MotionBlur(p=1)),
    ('Noise', albu.GaussNoise(var_limit=0.2, p=1)),  # Noise
    ('Downscale', albu.Downscale(p=1)),
    ('Compression', albu.ImageCompression(quality_lower=70, p=1)),
]

configs = [
    # ('BlurGlass', albu.GlassBlur(sigma=5, p=1)),
    # ('Noise', albu.GaussNoise(var_limit=0.005, p=1)),  # Noise
    # ('Downscale', albu.Downscale(scale_min=0.6, scale_max=0.9, p=1)),
    ('Crop', albu.RandomSizedCrop(p=1.0, min_max_height=(50, 500), height=320, width=320))
]

configs = [('Full', get_training_augmentation())]

for name, aug in configs:
    val_set = ImageDataset(images, 1, padding=False, augmentations=aug)
    dataloader = DataLoader(val_set, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)
    create_dir_if_empty(os.path.join(output, name))
    for im_index, batch in enumerate(dataloader):
            image_out = to_numpy(batch['image']).squeeze()
            imageio.v2.imwrite(os.path.join(output, name, '%s.png' % (batch['image_name'][0])),
                               (image_out * 255).astype(np.uint8))
