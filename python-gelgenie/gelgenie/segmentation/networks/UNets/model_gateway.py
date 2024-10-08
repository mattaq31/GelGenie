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


from gelgenie.segmentation.networks.UNets.unet_parts import DoubleConv, Down, Up, OutConv
from segmentation_models_pytorch import UnetPlusPlus, Unet
import torch.nn as nn


class milesial_UNet(nn.Module):
    # Implementation here taken from https://github.com/milesial/Pytorch-UNet/tree/e36c782fbfc976b7326182a47dd7213bd3360a7e
    def __init__(self, n_classes, in_channels=1, bilinear=False):
        super(milesial_UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class smp_UNetPlusPlus(UnetPlusPlus):
    def __init__(self, in_channels=1, classes=2, encoder_weights=None, **kwargs):
        super().__init__(in_channels=in_channels, classes=classes, encoder_weights=encoder_weights, **kwargs)
        self.n_channels = in_channels
        self.n_classes = classes


class smp_UNet(Unet):
    def __init__(self, in_channels=1, classes=2, encoder_weights=None, **kwargs):
        super().__init__(in_channels=in_channels, classes=classes, encoder_weights=encoder_weights, **kwargs)
        self.n_channels = in_channels
        self.n_classes = classes

