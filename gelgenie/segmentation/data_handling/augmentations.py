import albumentations as albu

# TODO: re-evaluate these
def get_training_augmentation():
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
                albu.GlassBlur(p=1),
                albu.MotionBlur(p=1),
            ],
            p=0.5,
        ),

        albu.GaussNoise(var_limit=0.2, p=0.5),  # Noise

        albu.OneOf(  # Image Quality
            [
                albu.Downscale(p=1),
                albu.ImageCompression(quality_lower=50, p=1),
            ],
            p=0.5
        ),
    ]
    return albu.Compose(transform)
