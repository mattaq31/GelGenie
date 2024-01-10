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

from scipy import ndimage as ndi
import numpy as np
from scipy.ndimage import distance_transform_edt


def unet_weight_map(mask, wc=None, w0=10, sigma=5):
    """
    Obtained from: https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras
    """
    labels, _ = ndi.label(mask)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((mask.shape[0], mask.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            # finds distance between all pixels that are not part of the selected object and the
            # closest pixel of the selected object
            distances[:, :, i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2) # sorts to get the two smallest distances at the top
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        w = w0 * np.exp(-1 / 2 * ((d1 + d2) / sigma) ** 2) * no_labels  # applies unet loss function (equation 2)
    else:
        w = np.zeros_like(mask)
    if wc is not None:
        class_weights = np.zeros_like(w)
        for ind, v in enumerate(wc[0, :]): # applies the class weighting (to balance out backgrounds and non-backgrounds) to each mask pixel
            class_weights[mask == ind] = v
        w = w + class_weights
    return w
