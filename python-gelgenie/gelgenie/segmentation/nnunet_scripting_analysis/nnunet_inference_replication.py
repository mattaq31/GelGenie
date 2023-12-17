# this code was translated into a simpler format from the original code in nnunet (https://github.com/MathijsdeBoer/nnUNet), which is licensed under Apache 2.0

from scipy.ndimage import gaussian_filter
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.color import label2rgb
import imageio
from tqdm import tqdm
import itertools
from typing import Tuple, Union, List
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from skimage import io


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


# - this will need to be coded into java somehow as it changes for each image
def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


# should be ok to code in java (I hope)
def run_model_with_tta(model, image):
    mirror_axes = [0, 1]
    with torch.no_grad():
        mask = model(image)
        axes_combinations = [c for i in range(len(mirror_axes)) for c in
                             itertools.combinations([m + 2 for m in mirror_axes], i + 1)]
        for axes in axes_combinations:
            mask += torch.flip(model(torch.flip(image, (*axes,))), (*axes,))
        mask /= (len(axes_combinations) + 1)
    return mask


if __name__ == '__main__':

    input_image = '/Users/matt/Desktop/nnunet_comp/input_data/GELTEST_0017_0000.tif'  # can prepare any image here
    output_image = '/Users/matt/Desktop/nn_trial_large_2.png'
    network_file = '/Users/matt/Desktop/torchscript_checkpoints/nnunet_model.pt' # need to download from huggingface or otherwise

    tile_size = (448, 576)
    network = torch.jit.load(network_file)

    image_float = io.imread(input_image).astype(np.float32)
    image_compute = io.imread(input_image)[None, None]

    image_compute = (image_compute - image_compute.mean()) / (max(image_compute.std(), 1e-8))  # this is the only image preprocessing that seems to be necessary.

    tdata = torch.from_numpy(image_compute).contiguous().float()  # convert to tensor and prep for running

    # at this point, image will need to padded to minimum size if it is too small.  Can this be simplified in Java?
    t_nn, padding_slicer_revert = pad_nd_image(tdata, tile_size, 'constant', {'value': 0}, True, None)

    image_size = t_nn.size()

    # gaussian map required, which is always the same size and with the same parameters, so can be pre-computed
    gauss_map = compute_gaussian((448, 576), sigma_scale=1. / 8, value_scaling_factor=10, device=torch.device('cpu'))

    # will need to compute image patching sliding window before moving forward
    steps = compute_steps_for_sliding_window(image_size[2:], tile_size, 0.5)
    slicers = []
    for d in range(image_size[1]):
        for sx in steps[0]:
            for sy in steps[1]:
                slicers.append(
                    tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                             zip((sx, sy), tile_size)]]))

    # run network on each individual slice, one by one
    # after TTA, multiply with pre-made gaussian map.
    # Then keep a tally of the gaussian additions for each pixel.
    # Divide by this tally after all slices done.
    n_predictions = torch.zeros(image_size[1:], dtype=torch.half)
    predicted_logits = torch.zeros((2, *image_size[1:]), dtype=torch.half)

    for sl in tqdm(slicers):
        workon = t_nn[sl][None]
        prediction = run_model_with_tta(network, workon)[0]

        predicted_logits[sl] += prediction * gauss_map
        n_predictions[sl[1:]] += gauss_map

    predicted_logits /= n_predictions

    # revert padding if image was too small originally.
    predicted_logits_postpad = predicted_logits[tuple([slice(None), *padding_slicer_revert[1:]])]

    # convert predicted logits into a definite segmentation
    probabilities = torch.softmax(predicted_logits_postpad.float(), 0)
    segmentation = probabilities.argmax(0)

    # label segmentations on original image for easy visualisation
    labels, _ = ndi.label(segmentation.numpy().squeeze())
    if image_float.max() > 256:  # 16-bit input
        image_float = image_float / 65535
    else:  # 8-bit input
        image_float = image_float / 255

    rgb_labels = label2rgb(labels, image=image_float)

    # output always 8-bit
    imageio.v2.imwrite(output_image, (rgb_labels * 255).astype(np.uint8))
