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
