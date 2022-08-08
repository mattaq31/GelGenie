# code here taken from https://github.com/milesial/Pytorch-UNet/tree/e36c782fbfc976b7326182a47dd7213bd3360a7e
""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from segmentation_models_pytorch import UnetPlusPlus


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
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
    def __init__(self,
                 encoder_name="resnet34",
                 encoder_depth=5,
                 encoder_weights="imagenet",
                 decoder_use_batchnorm=True,
                 decoder_channels=(256, 128, 64, 32, 16),
                 decoder_attention_type=None,
                 in_channels=3,
                 classes=1,
                 activation=None,
                 aux_params=None):
        super().__init__(encoder_name,
                         encoder_depth,
                         encoder_weights,
                         decoder_use_batchnorm,
                         decoder_channels,
                         decoder_attention_type,
                         in_channels,
                         classes,
                         activation,
                         aux_params)
        self.n_channels = in_channels
        self.n_classes = classes
        self.bilinear = False
