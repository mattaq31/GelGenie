import imageio
import numpy as np
from PIL import Image
import os
from gelgenie.segmentation.evaluation import model_eval_load
import torch

def get_individual_padding(img_array):
    new_vert = 32 * (img_array.shape[0] // 32 + 1)
    new_horiz = 32 * (img_array.shape[1] // 32 + 1)

    top = (new_vert - img_array.shape[0]) // 2  # Get amount of pixels to pad at top of image
    bottom = new_vert - img_array.shape[0] - top  # Get amount of pixels to pad at bottom of image
    left = (new_horiz - img_array.shape[1]) // 2  # Get amount of pixels to pad at left of image
    right = new_horiz - img_array.shape[1] - left  # Get amount of pixels to pad at right of image
    return (top, bottom), (left, right)

model_selected = 'finetuned'

if model_selected == 'universal':
    test_in = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/bioimage_io_models/universal_model/test_data/input_134.tif'
    test_out = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/bioimage_io_models/universal_model/test_data/output_134.png'
    output_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/bioimage_io_models/universal_model/test_data'
    exp_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/unet_dec_21'
    epoch = 579
else:
    test_in = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/bioimage_io_models/finetuned_model/test_data/input_134.tif'
    test_out = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/bioimage_io_models/finetuned_model/test_data/output_134.tif'
    output_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/bioimage_io_models/finetuned_model/test_data'
    exp_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/unet_dec_21_finetune'
    epoch = 590

image = imageio.v2.imread(test_in)

if image.dtype == 'uint8':
    max_val = 255  # values are 0 - 255
elif image.dtype == 'uint16':
    max_val = 65535  # values are 0 - 65535
else:
    raise RuntimeError(f'Image type {image.dtype} not recognized, only accepts uint8 or uint16 for now.')

image = image.astype(np.float32) / max_val # converts image to float32 and normalizes to 0-1
mask = np.array(Image.open(test_out).convert('L'))
mask[mask > 0] = 1

# pads to a multiple of 32
vertical, horizontal = get_individual_padding(image)
image = np.pad(image, pad_width=(vertical, horizontal), mode='constant')
mask = np.pad(mask, pad_width=(vertical, horizontal), mode='constant')

image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
mask = np.expand_dims(mask, axis=0)

model = model_eval_load(exp_folder, epoch)

with torch.no_grad():
    model_raw_mask = model(torch.tensor(image))

# saves files to numpy format
np.save(os.path.join(output_folder, 'test_input.npy'), image)
np.save(os.path.join(output_folder, 'test_output.npy'), model_raw_mask.numpy())
