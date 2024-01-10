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

import base64
import io
from collections import defaultdict

from PIL import Image
import numpy as np

import skimage
from skimage.color import rgb2gray, label2rgb
import pandas as pd
import skimage.util as util

from gelgenie.classical_tools.band_detection import find_bands
from gelgenie.classical_tools.utils import convert_pil_image_base_64, convert_numpy_image_base_64


class GelAnalysis:
    def __init__(self, image=None, image_type='file'):

        self.np_image = None
        self.gray_image = None

        if image is None:
            self.base_image = None
        else:
            self.set_image(image, image_type)

    def prepare_im(self):
        self.np_image = np.array(self.base_image)
        if len(self.np_image.shape) == 2:
            gray_im = self.np_image
        else:
            gray_im = rgb2gray(self.np_image)
        self.gray_image = skimage.util.img_as_uint(gray_im)

    def set_image(self, image, image_type='file'):
        if image_type == 'file':
            self.base_image = Image.open(image)
        elif image_type == 'b64':
            base64_decoded = base64.b64decode(image)
            self.base_image = Image.open(io.BytesIO(base64_decoded))
        else:
            raise RuntimeError('Incorrect image type supplied.')
        self.prepare_im()

    def get_image_dim(self):
        if self.base_image is None:
            raise RuntimeError('No image supplied.')
        return self.base_image.height, self.base_image.width

    def get_b64_image(self, im_format='png'):
        if self.base_image is None:
            raise RuntimeError('No image supplied.')
        return convert_pil_image_base_64(self.base_image, im_format=im_format)[0]

    def get_otsu_threshold(self):
        if self.base_image is None:
            raise RuntimeError('No image supplied.')
        otsu_th = skimage.filters.threshold_otsu(self.gray_image)
        otsu_percent = otsu_th / 65535 * 100
        return otsu_percent

    def get_bands_overlay_b64(self):
        if self.base_image is None:
            raise RuntimeError('No image supplied.')

        return convert_numpy_image_base_64(self.overlayed_image_bands, self.overlay_inverted)

    def find_bands(self, fg, bg, reps):
        if self.base_image is None:
            raise RuntimeError('No image supplied.')

        self.overlayed_image_bands, self.band_regions, self.overlay_inverted, self.band_table, self.band_mask = \
            find_bands(self.gray_image, int(fg) / 100 * 65535, int(bg) / 100 * 65535, int(reps))

        # Create band dictionary
        self.band_dict = defaultdict(list)

        # Filter bands with area less than 50, and find props of those larger than 50
        for band_no, region_object in enumerate(self.band_regions, 1):

            if region_object.area >= 50:  # area threshold for a band
                # Calculate and append weighted area
                weighted_area = round(region_object.mean_intensity.item() * region_object.area.item() / (255 * 255))
                # Add relevant props to dictionary
                self.band_dict["id"].append(band_no)  # cannot be changed by user
                self.band_dict["label"].append(band_no)  # default label is band number, but can be changed
                self.band_dict["center_x"].append(region_object.centroid[1])
                self.band_dict["center_y"].append(region_object.centroid[0])
                self.band_dict["center"].append(region_object.centroid)
                self.band_dict["area"].append(region_object.area.item())
                self.band_dict["w_area"].append(weighted_area)
                self.band_dict["c_area"].append(weighted_area)
                self.band_dict["bbox"].append(region_object.bbox)
                # TODO: band dict shouldn't be saved as an attribute

        self.band_dataframe = pd.DataFrame.from_dict(self.band_dict).set_index(['id'])

    def remove_band(self, band_no):
        self.band_dataframe.drop(band_no, inplace=True)

        # Remove band from image
        self.band_mask = np.where(self.band_mask == band_no, 0, self.band_mask)
        updated_image = label2rgb(self.band_mask, image=self.gray_image, bg_label=0, bg_color=[0, 0, 0])

        # Invert updated image
        updated_image_inv = util.invert(updated_image)

        # update images
        self.overlayed_image_bands = updated_image
        self.overlay_inverted = updated_image_inv

    def calibrate_band_area(self, factor):
        self.band_dataframe["c_area"] = round(factor * self.band_dataframe["w_area"], 2)

    def extract_lane_profile(self, x_pos, y_end):
        # TODO: this currently just plots a vertical line from a selected band - needs to be updated.
        self.current_profile = skimage.measure.profile_line(self.gray_image, [0, x_pos], [y_end, x_pos], linewidth=7,
                                                            reduce_func=np.mean)
        return self.current_profile
