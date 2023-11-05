import torch.onnx
import onnx
from onnxsim import simplify
import os

from gelgenie.segmentation.evaluation import model_eval_load
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def simple_onnx_export(model_folder, epoch):
    """
    Exports a pytorch model to onnx format, then simplifies it for use with OpenCV (4.6).
    :param model_folder: Main model folder containing epoch checkpoints.
    :param epoch: Specific epoch to export.
    :return: Pytorch model and filepath of exported onnx model.
    """
    output_folder = os.path.join(model_folder, 'onnx_checkpoints')
    create_dir_if_empty(output_folder)

    descriptive_name = os.path.basename(model_folder) + '_epoch_' + str(epoch)
    dummy_input = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
    temp_file = os.path.join(output_folder, 'temp_checkpoint.onnx')
    final_file = os.path.join(output_folder, descriptive_name + '.onnx')

    net = model_eval_load(model_folder, epoch)

    torch.onnx.export(net,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      temp_file,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},  # variable length axes
                                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}})

    onnx_model = onnx.load(temp_file)
    onnx.checker.check_model(onnx_model)
    model_simp, check = simplify(onnx_model)

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, final_file)
    os.remove(temp_file)  # deletes temporary onnx model, only simplified version needed

    return net, final_file


def visual_onnx_export(model_folder, epoch, image_folder):
    """
    Runs an onnx export, and generates a sample image using onnx and opencv for comparison purposes.
    :param model_folder: Main model folder containing epoch checkpoints.
    :param epoch: Specific epoch to export.
    :param image_folder: Folder containing images that can be tested on using the selected model.
    :return: N/A
    """
    from matplotlib import pyplot as plt
    from gelgenie.segmentation.data_handling.dataloaders import ImageDataset
    import cv2
    from skimage.color import label2rgb
    import onnxruntime
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import numpy as np
    import traceback

    net, final_file = simple_onnx_export(model_folder, epoch)  # runs normal export

    val_set = ImageDataset(image_folder, 1, padding=True)
    dataloader = DataLoader(val_set, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    for im_index, batch in enumerate(dataloader):  # extracts the first image from the folder
        with torch.no_grad():
            mask_pred = net(batch['image'])
            break

    ort_session = onnxruntime.InferenceSession(final_file, providers=['CPUExecutionProvider'])

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(batch['image'])}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    try:
        np.testing.assert_allclose(to_numpy(mask_pred), ort_outs[0], rtol=1e-03, atol=1e-05)
    except AssertionError:
        print('--------')
        traceback.print_exc()
        print('--------')
        print("The original predicted and onnx predicted values don't match perfectly, "
              "but differences can be small enough to ignore.")
        print('--------')

    cv_net = cv2.dnn.readNet(final_file)

    original_image = batch['image'].squeeze().detach().squeeze().cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(batch['image'])}
    ort_outs = ort_session.run(None, ort_inputs)

    # compute CV2 image
    cv_net.setInput(to_numpy(batch['image']))
    cv_out = cv_net.forward()

    # computes one hot prediction for all methods
    one_hot = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
    onn = one_hot.numpy().squeeze()
    onnx_out = F.one_hot(torch.from_numpy(ort_outs[0]).argmax(dim=1), 2).permute(0, 3, 1, 2).float()
    onnx_out = onnx_out.numpy().squeeze()
    cv_out = F.one_hot(torch.from_numpy(cv_out).argmax(dim=1), 2).permute(0, 3, 1, 2).float()
    cv_out = cv_out.numpy().squeeze()

    # labels image directly with model output
    model_onnx_labels = label2rgb(onnx_out.argmax(axis=0), image=original_image)
    model_direct_labels = label2rgb(onn.argmax(axis=0), image=original_image)
    cv_labels = label2rgb(cv_out.argmax(axis=0), image=original_image)

    # generates comparison plot
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(original_image, cmap='gray')
    ax[1].imshow(model_direct_labels)
    ax[2].imshow(model_onnx_labels)
    ax[3].imshow(cv_labels)

    ax[0].set_title('Original Image')
    ax[1].set_title('PyTorch Model Output')
    ax[2].set_title('ONNX Model Output')
    ax[3].set_title('OpenCV Model Output')

    for i in range(4):
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, 'onnx_checkpoints',
                             'visual_comparison_%s.png' % (os.path.basename(model_folder) + '_epoch_' + str(epoch))),
                dpi=300)
