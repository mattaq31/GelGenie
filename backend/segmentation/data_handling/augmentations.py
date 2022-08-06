import albumentations as albu


# For base set of training set / all of validation set
# def get_augmentation_pad_only(image_array, mask_array):
#     transform = albu.Compose[
#         albu.PadIfNeeded(min_height=2048, min_width=2048),
#         # border_mode=0
#     ]
#     sample = transform(image=image_array, mask=mask_array)
#     return sample['image'], sample['mask']

def get_training_augmentation():
    transform = [
        # Padding
        # albu.PadIfNeeded(min_height=1536, min_width=1536, border_mode=0, value=0),


        # Spatial-level transforms

        # Flip (horizontally, vertically or both)
        albu.Flip(p=0.5),

        albu.RandomRotate90(p=0.5),

        albu.SafeRotate(limit=30, border_mode=0, value=0, p=0.5),

        # Pixel-level transforms

        # Brightness
        albu.OneOf(
            [
                albu.RandomBrightness(limit=0.1, p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5
        ),

        # Saturation
        albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=0.5),

        # Blur
        albu.OneOf(
            [
                albu.AdvancedBlur(p=1),
                albu.GlassBlur(p=1),
                albu.MotionBlur(p=1),
            ],
            p=0.5,
        ),

        # Noise
        albu.GaussNoise(var_limit=0.2, p=0.5),

        # Quality
        albu.OneOf(
            [
                albu.Downscale(p=1),
                albu.ImageCompression(quality_lower=50, p=1),
            ],
            p=0.5
        ),
    ]
    return albu.Compose(transform)
