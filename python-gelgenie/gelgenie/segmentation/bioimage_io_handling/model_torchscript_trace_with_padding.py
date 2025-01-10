# DISCLAIMER:  In pytorch this model appeared to work well but in deepimagej it warped all input images.  Not sure why this would happen.
import os
import torch
import torch.nn as nn

from gelgenie.segmentation.evaluation import model_eval_load
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty


def symmetric_pad(image):
    _, _, h, w = image.shape

    new_h = 32 * ((h // 32) + 1)  # new target dimensions
    new_w = 32 * ((w // 32) + 1)

    # Calculate symmetric padding values
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left

    # Apply padding using pytorch function
    padded_image = nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_image, (h, w, pad_top, pad_bottom, pad_left, pad_right)

def symmetric_unpad(image, original_dimensions):

    h, w, pad_top, pad_bottom, pad_left, pad_right = original_dimensions

    # Removes symmetric padding as normal
    return image[:, :, pad_top:h + pad_top, pad_left:w + pad_left]

# Model with pre/post-padding processing for easy use in deepimagej
class ModelWithSymmetricPadding(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        padded_image, original_dimensions = symmetric_pad(image)
        model_output = self.model(padded_image)
        formatted_mask = symmetric_unpad(model_output, original_dimensions)

        return formatted_mask

model_paths = ['/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/unet_dec_21',
               '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/unet_dec_21_finetune']
epochs = [579, 590]

for model_path, epoch in zip(model_paths, epochs):
    output_folder = os.path.join(model_path, 'torchscript_checkpoints')
    create_dir_if_empty(output_folder)
    output_file_name = os.path.basename(model_path) + '_epoch_' + str(epoch) + '_inbuilt_padding'
    net = model_eval_load(model_path, epoch)

    dummy_input = torch.zeros((1, 1, 126, 124), dtype=torch.float32)
    model_with_padding = ModelWithSymmetricPadding(net)

    traced_script_module = torch.jit.trace(model_with_padding, dummy_input)
    traced_script_module.save(os.path.join(output_folder, output_file_name + '.pt'))
