from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
import torch.nn.functional as F

from gelgenie.segmentation.networks.UNets.model_gateway import smp_UNet, smp_UNetPlusPlus
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset

import numpy as np
import torch
from skimage.color import label2rgb

import torch.onnx
import onnx
import onnxruntime
import cv2
from onnxsim import simplify


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
checkpoint_file_path = "/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/base_smp_unet_small_data/checkpoints/checkpoint_epoch_400.pth"
checkpoint_file_path = "/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/smp_unet++_small_data/checkpoints/checkpoint_epoch_260.pth"
checkpoint_file_path = "/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/smp_unet++_july28_james/checkpoints/checkpoint_epoch_600.pth"
checkpoint_file_path = "/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/smp_unet++_july28_nathan/checkpoints/checkpoint_epoch_504.pth"
# checkpoint_file_path = "/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/smp_unet++_july28_1/checkpoints/checkpoint_epoch_306.pth"

n_channels = 1
model_name = 'base_smp_unet_chkpt_400'
model_name = 'smp_unetplusplus_chkpt_260'
model_name = 'u++_600_james'
model_name = 'u++_504_nathan'
# model_name = 'u++_306_full'

net = load_model(checkpoint_file_path)

image_dir = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/data/processed_gels/dummy_set/images_train'
val_set = ImageDataset(image_dir, n_channels, padding=True)

dataloader = DataLoader(val_set, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

# ONNX MODEL CREATION
net.eval()

for im_index, batch in enumerate(dataloader):
    with torch.no_grad():
        mask_pred = net(batch['image'])
        break

# Export the model
torch.onnx.export(net,  # model being run
                  batch['image'],  # model input (or a tuple for multiple inputs)
                  "/Users/matt/Desktop/%s.onnx" % model_name,
                  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=12,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},  # variable length axes
                                'output': {0: 'batch_size', 2: 'height', 3: 'width'}})

onnx_model = onnx.load("/Users/matt/Desktop/%s.onnx" % model_name)
onnx.checker.check_model(onnx_model)

model_simp, check = simplify(onnx_model)

assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, "/Users/matt/Desktop/%s-sim.onnx" % model_name)
ort_session = onnxruntime.InferenceSession("/Users/matt/Desktop/%s-sim.onnx" % model_name)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(batch['image'])}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(mask_pred), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
print('--------------------------')

cv_net = cv2.dnn.readNet("/Users/matt/Desktop/%s-sim.onnx" % model_name)


fig, ax = plt.subplots(1, 4, figsize=(20,10))

original_image = batch['image'].squeeze().detach().squeeze().cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(batch['image'])}
ort_outs = ort_session.run(None, ort_inputs)

# compute CV2 image
cv_net.setInput(to_numpy(batch['image']))
cv_out = cv_net.forward()

one_hot = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
onn = one_hot.numpy().squeeze()

onnx_out = F.one_hot(torch.from_numpy(ort_outs[0]).argmax(dim=1), 2).permute(0, 3, 1, 2).float()
onnx_out = onnx_out.numpy().squeeze()

cv_out = F.one_hot(torch.from_numpy(cv_out).argmax(dim=1), 2).permute(0, 3, 1, 2).float()
cv_out = cv_out.numpy().squeeze()

model_onnx_labels = label2rgb(onnx_out.argmax(axis=0), image=original_image)

model_direct_labels = label2rgb(onn.argmax(axis=0), image=original_image)

cv_labels = label2rgb(cv_out.argmax(axis=0), image=original_image)

ax[0].imshow(original_image, cmap='gray')
ax[1].imshow(model_direct_labels)
ax[2].imshow(model_onnx_labels)
ax[3].imshow(cv_labels)
plt.tight_layout()
plt.show()

