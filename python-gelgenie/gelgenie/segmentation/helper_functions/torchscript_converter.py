import os.path
import torch

from gelgenie.segmentation.evaluation import model_eval_load
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty


def torchscript_export(model_folder, epoch):
    """
    Exports a pytorch model to torchscript format.
    :param model_folder: Main model folder containing epoch checkpoints.
    :param epoch: Specific epoch to export.
    :return: N/A
    """
    output_folder = os.path.join(model_folder, 'torchscript_checkpoints')
    create_dir_if_empty(output_folder)
    descriptive_name = os.path.basename(model_folder) + '_epoch_' + str(epoch)

    net = model_eval_load(model_folder, epoch)

    dummy_input = torch.zeros((1, 1, 128, 128), dtype=torch.float32)

    traced_script_module = torch.jit.trace(net, dummy_input)
    traced_script_module.save(os.path.join(output_folder, descriptive_name + '.pt'))
