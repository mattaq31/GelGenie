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

import albumentations as albu


def get_training_augmentation():
    """
    Returns a set of augmentations to be applied to training images which feature everything from semi-destructive
    operations like compression and weaker augmentations like flips/rotations.
    :return: Albumentations transform pipeline.
    """
    transform = [

        # Spatial-level transforms (Both images and masks augmented)
        albu.Flip(p=0.5),  # Flip (horizontally, vertically or both)
        albu.RandomRotate90(p=0.5),  # Randomly rotate by 90 degrees
        albu.SafeRotate(limit=30, border_mode=0, value=0, p=0.5),  # Rotate randomly within 30 degrees without cropping

        # Pixel-level transforms (Only image is augmented)
        albu.OneOf(  # Brightness augmentations
            [
                albu.RandomBrightness(limit=0.1, p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5
        ),

        albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=0.5),  # Saturation

        albu.OneOf(  # Blur
            [
                albu.AdvancedBlur(p=1),
                albu.MotionBlur(p=1),
            ],
            p=0.5,
        ),

        albu.GaussNoise(var_limit=0.001, p=0.5),  # Noise

        albu.OneOf(  # Image Quality
            [
                albu.Downscale(scale_min=0.6, scale_max=0.9, p=1),
                albu.ImageCompression(quality_lower=70, p=1),
            ],
            p=0.5
        ),
    ]
    return albu.Compose(transform)


def get_nondestructive_training_augmentation():
    """
    Returns a set of augmentations to be applied to training images which only feature non-destructive operations like
    flips and rotations.
    :return: Albumentations transform pipeline.
    """
    transform = [
        # Spatial-level transforms (Both images and masks augmented)
        albu.Flip(p=0.5),  # Flip (horizontally, vertically or both)
        albu.RandomRotate90(p=0.5),  # Randomly rotate by 90 degrees
        albu.SafeRotate(limit=30, border_mode=0, value=0, p=0.5),  # Rotate randomly within 30 degrees without cropping

    ]
    return albu.Compose(transform)
