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

from monai.networks.nets import UNet, AttentionUnet


class monai_resunet(UNet):
    def __init__(self, in_channels=1, classes=2, **kwargs):
        super().__init__(spatial_dims=2, in_channels=in_channels, out_channels=classes,
                         channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                         num_res_units=2, **kwargs)
        self.n_channels = in_channels
        self.n_classes = classes


class monai_attunet(AttentionUnet):
    def __init__(self, in_channels=1, classes=2, **kwargs):
        super().__init__(spatial_dims=2, in_channels=in_channels, out_channels=classes,
                         channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), **kwargs)
        self.n_channels = in_channels
        self.n_classes = classes

