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

