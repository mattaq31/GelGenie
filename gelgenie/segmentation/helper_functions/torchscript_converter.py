from torch.utils.data import DataLoader

from gelgenie.segmentation.networks.UNets.model_gateway import smp_UNet, smp_UNetPlusPlus
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset
import torch


def load_model(checkpoint):
    # net = smp_UNet(
    #             encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #             in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #             classes=2,  # model output channels (number of classes in your dataset)
    #         )
    net = smp_UNetPlusPlus(
                encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=2,  # model output channels (number of classes in your dataset)
            )
    net.eval()
    saved_dict = torch.load(f=checkpoint, map_location=torch.device("cpu"))
    net.load_state_dict(saved_dict['network'])
    print(f'Model loaded from {checkpoint}')
    return net


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# prepping model
checkpoint_file_path = "/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/smp_unet++_july28_1/checkpoints/checkpoint_epoch_306.pth"

n_channels = 1

model_name = 'u++_306_full'

net = load_model(checkpoint_file_path)

image_dir = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/dummy_set/images_train'
val_set = ImageDataset(image_dir, n_channels, padding=True)

dataloader = DataLoader(val_set, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

# ONNX MODEL CREATION
net.eval()

for im_index, batch in enumerate(dataloader):
    with torch.no_grad():
        traced_script_module = torch.jit.trace(net, batch['image'])
        mask_pred = net(batch['image'])
        break

# Save the TorchScript model
traced_script_module.save("/Users/matt/Desktop/torchscript_model.pt")
