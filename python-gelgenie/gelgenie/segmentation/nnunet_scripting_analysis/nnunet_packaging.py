# this code was translated into a simpler format from the original code in nnunet (https://github.com/MathijsdeBoer/nnUNet), which is licensed under Apache 2.0

from scipy.ndimage import gaussian_filter
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.color import label2rgb
import imageio
import itertools
from typing import Tuple, Union, List
from skimage import io
import math
import numpy as np


class PackagedTTA(nn.Module):
    def __init__(self, nnunet_model):
        super(PackagedTTA, self).__init__()
        self.main_net = nnunet_model

    def forward(self, x):
        with torch.no_grad():
            mask = self.main_net(x)
            mask += torch.flip(self.main_net(torch.flip(x, (2,))), (2,))
            mask += torch.flip(self.main_net(torch.flip(x, (3,))), (3,))
            mask += torch.flip(self.main_net(torch.flip(x, (2, 3))), (2, 3))
            mask /= 4
        return mask


class PackagedNnUNet(nn.Module):
    """
    This module is a translation of the standard nnunet inference module, but with various changes made to ensure it
    is compatible with torchscripting.  Many values are harcoded so if any changes were to be made to the model,
    these would need to be fixed/adjusted.
    """
    def __init__(self, nnunet_model, gaussian_map, tile_size=(448, 576)):
        super(PackagedNnUNet, self).__init__()
        self.main_net = PackagedTTA(nnunet_model)
        self.register_buffer('gaussian_map', gaussian_map, persistent=True)
        self.tile_size = tile_size

    def compute_steps_for_sliding_window(self, image_size: Tuple[int, int], tile_size: Tuple[int, int]):

        target_step_sizes_in_voxels = (224, 288)

        num_steps = [int(math.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

        # the highest step value for this dimension is
        max_step_value = image_size[0] - tile_size[0]
        if num_steps[0] > 1:
            actual_step_size = max_step_value / (num_steps[0] - 1)
        else:
            actual_step_size = 1.0  # does not matter because there is only one step at 0

        steps_here_0 = [int(round(actual_step_size * i)) for i in range(num_steps[0])]

        # the highest step value for this dimension is
        max_step_value = image_size[1] - tile_size[1]
        if num_steps[1] > 1:
            actual_step_size = max_step_value / (num_steps[1] - 1)
        else:
            actual_step_size = 1.0  # does not matter because there is only one step at 0

        steps_here_1 = [int(round(actual_step_size * i)) for i in range(num_steps[1])]

        return steps_here_0, steps_here_1

    def run_sliding(self, x):
        image_size = (1, 1, x.size()[2], x.size()[3])
        # will need to compute image patching sliding window before moving forward
        steps = self.compute_steps_for_sliding_window((image_size[2], image_size[3]), self.tile_size)

        slicers = torch.jit.annotate(List[Tuple[Tuple[int, int], Tuple[int, int]]], [])
        for sx in steps[0]:
            for sy in steps[1]:
                slicers.append(((sx, sx+self.tile_size[0]), (sy, sy+self.tile_size[1])))

        # run network on each individual slice, one by one
        # after TTA, multiply with pre-made gaussian map.
        # Then keep a tally of the gaussian additions for each pixel.
        # Divide by this tally after all slices done.
        n_predictions = torch.zeros(image_size[1:], dtype=torch.half, device=self.gaussian_map.device)
        predicted_logits = torch.zeros((2, *image_size[1:]), dtype=torch.half, device=self.gaussian_map.device)

        for sl in slicers:
            workon = x[:, :, sl[0][0]:sl[0][1], sl[1][0]:sl[1][1]]
            prediction = self.main_net(workon)[0]
            predicted_logits[:, 0, sl[0][0]:sl[0][1], sl[1][0]:sl[1][1]] += prediction * self.gaussian_map
            n_predictions[0, sl[0][0]:sl[0][1], sl[1][0]:sl[1][1]] += self.gaussian_map

        predicted_logits /= n_predictions
        return predicted_logits

    def pad_if_necessary(self, x):

        if x.size()[2] >= 448 and x.size()[3] >= 576:
            return x, ((0, 0), (0, 0))

        difference = [0, 0, 448-x.size()[2], 576-x.size()[3]]

        pad_list = [[0, 0], [0, 0], [difference[2] // 2, difference[2] // 2 + (difference[2] % 2)], [difference[3] // 2, difference[3] // 2 + (difference[3] % 2)]]

        torch_pad_list = pad_list[3] + pad_list[2] + [0, 0, 0, 0]
        res = F.pad(x, torch_pad_list, 'constant', value=0.0)

        slicer = ((pad_list[2][0], 448 - pad_list[2][1]), (pad_list[3][0], 576 - pad_list[3][1]))
        return res, slicer

    def forward(self, x):
        x = (x - x.mean()) / (max(x.std(), 1e-8))

        x, slicer = self.pad_if_necessary(x)

        predicted_logits = self.run_sliding(x)
        # convert predicted logits into a definite segmentation
        probabilities = torch.softmax(predicted_logits.float(), 0)
        segmentation = probabilities.argmax(0)

        if slicer[0][1] == 0 and slicer[1][1] == 0:
            return segmentation
        else:
            return segmentation[:, slicer[0][0]:slicer[0][1], slicer[1][0]:slicer[1][1]]


# one-time use function - would be best to save to file and read into java directly
def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.type(dtype).to(device)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


if __name__ == '__main__':

    input_image = '/Users/matt/Desktop/nnunet_comp/input_data/GELTEST_0040_0000.tif'  # replace with your own images
    output_image = '/Users/matt/Desktop/nn_trial_package_all_combined_large.png'
    output_model = '/Users/matt/Desktop/gaussian_tta_packaged_nnunet.pt'

    ######################### - requires model to be pre-traced, can download from huggingface
    nnunet_trace = torch.jit.load('/Users/matt/Desktop/torchscript_checkpoints/nnunet_model.pt')
    #########################

    image_float = io.imread(input_image).astype(np.float32)
    image_compute = io.imread(input_image)[None, None].astype(np.float32)
    tdata = torch.from_numpy(image_compute).contiguous().float()  # convert to tensor and prep for running

    # gaussian map required, which is always the same size and with the same parameters, so can be pre-computed
    gauss_map = compute_gaussian((448, 576), sigma_scale=1. / 8, value_scaling_factor=10, device=torch.device('cpu'))

    module = PackagedNnUNet(nnunet_trace, gauss_map)
    module.eval()

    nnunet = torch.jit.script(module)  # full trace occurs here
    nnunet.save(output_model)

    print('started inferring...')
    segmentation_model = nnunet(tdata)

    # label segmentations on original image for easy visualisation
    labels, _ = ndi.label(segmentation_model.numpy().squeeze())
    if image_float.max() > 256:  # 16-bit input
        image_float = image_float / 65535
    else:  # 8-bit input
        image_float = image_float / 255

    rgb_labels = label2rgb(labels, image=image_float)
    # output always 8-bit
    imageio.v2.imwrite(output_image, (rgb_labels * 255).astype(np.uint8))
