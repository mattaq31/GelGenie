from torchinfo import summary
from gelgenie.segmentation.helper_functions.general_functions import create_summary_table
import torch.nn as nn


def model_configure(model_name='milesial-UNet',
                    n_channels=1, classes=2, bilinear=True,
                    pretrained='imagenet', device='cpu'):
    from gelgenie.segmentation.networks.UNets.unet_model import UNet, smp_UNetPlusPlus, smp_UNet

    # n_classes is the number of probabilities you want to get per pixel
    if model_name == 'milesial-UNet':
        net = UNet(n_channels=int(n_channels), n_classes=classes, bilinear=bilinear)  # initializing random weights
    elif model_name == 'UNetPlusPlus':
        net = smp_UNetPlusPlus(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=pretrained,  # choose which pretrained weights to use (or none)
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
    elif model_name == 'smp-UNet':
        net = smp_UNet(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=pretrained,  # choose which pretrained weights to use (or none)
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
    elif model_name == 'dummy':
        net = DummyModel(int(n_channels), classes, bilinear)

    else:
        raise RuntimeError(f'Model {model_name} unidentified, must be milesial-UNet or UNetPlusPlus')

    net.to(device=device)

    # prints out model summary to output directory
    model_structure = summary(net, mode='train', depth=5, device=device, verbose=0)
    model_info = [['Network', model_name],
                  ['Trainable Parameters', model_structure.trainable_params],
                  ['Input channels', n_channels],
                  ['Output channels', classes],
                  ['Downscaling operation', 'Bilinear' if bilinear else 'Transposed conv'],
                  ]
    model_docstring = create_summary_table("Model Summary", ['Parameter', 'Value'], ['cyan', 'green'], model_info)

    return net, model_structure, model_docstring


class DummyModel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DummyModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.main_module = nn.Conv2d(n_channels, n_classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.main_module(x)

