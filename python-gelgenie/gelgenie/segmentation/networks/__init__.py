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

from torchinfo import summary
from gelgenie.segmentation.helper_functions.general_functions import create_summary_table
import torch.nn as nn
import os
import ast
from pydoc import locate
import torch


available_models = {}
all_models_folder = os.path.abspath(os.path.join(__file__, os.path.pardir))

# quick logic searching for all folders in models directory
model_categories = [f.name for f in os.scandir(all_models_folder) if (f.is_dir() and '__' not in f.name)]
# Main logic for searching for handler files and registering model architectures in system.
for category in model_categories:
    handler_file = os.path.join(all_models_folder, category, 'model_gateway.py')
    if not os.path.isfile(handler_file):
        continue
    p = ast.parse(open(handler_file, 'r').read())
    classes = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
    for _class in classes:
        available_models[_class.lower()] = ('gelgenie.segmentation.networks.'+category+'.model_gateway.'+_class)


def model_configure(model_name='dummy', device='cpu', pytorch_2_compile=False, **kwargs):

    if model_name == 'dummy':
        net = DummyModel(1, kwargs['classes'])
    else:
        if model_name not in available_models:
            raise RuntimeError(f'Model {model_name} unidentified, available models include: {available_models}')
        else:
            net = locate(available_models[model_name])(**kwargs)

    if int(torch.__version__[0]) > 1 and pytorch_2_compile:  # TODO: can this be tested at some point?
        print('Compiling model using PyTorch 2.0 compile')
        net = torch.compile(net)

    net.to(device=device)

    # prints out model summary to output directory
    model_structure = summary(net, mode='train', depth=5, device=device, verbose=0)
    model_info = [['Network', model_name],
                  ['Trainable Parameters', model_structure.trainable_params],
                  ['Output channels', kwargs['classes']],
                  ]  # TODO: anything else to add here?
    model_docstring = create_summary_table("Model Summary", ['Parameter', 'Value'], ['cyan', 'green'], model_info)

    return net, model_structure, model_docstring


class DummyModel(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(DummyModel, self).__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.main_module = nn.Conv2d(in_channels, n_classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.main_module(x)

