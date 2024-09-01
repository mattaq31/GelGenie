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

# this code was translated into a simpler format from the original code in nnunet (https://github.com/MathijsdeBoer/nnUNet), which is licensed under Apache 2.0

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from os.path import join
from torch._dynamo import OptimizedModule
import torch
from torch import nn

"""
To run this code, you will need to install nnunet into a separate environment, following nnunet's installation instructions:
https://github.com/MIC-DKFZ/nnUNet

Next, you will be able to run the code below to trace a pre-trained model and export it to a torchscript .pt file.  
The nnunet models trained for GelGenie are available on HuggingFace e.g.: https://huggingface.co/mattaq/GelGenie-nnUNet-Dec-2023

When the above is done, move on to nnunet_packaging.py.
"""
if __name__ == '__main__':

    # MODEL PREP
    #############################
    predictor_net = PlainConvUNet(input_channels=1,
                                  n_stages=7,
                                  features_per_stage=[32, 64, 128, 256, 512, 512, 512],
                                  conv_op=nn.Conv2d,
                                  kernel_sizes=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                                  strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
                                  num_classes=2,
                                  deep_supervision=False,
                                  n_conv_per_stage=[2, 2, 2, 2, 2, 2, 2],
                                  n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],
                                  conv_bias=True,
                                  norm_op=nn.modules.instancenorm.InstanceNorm2d,
                                  norm_op_kwargs={'affine': True, 'eps': 1e-05},
                                  dropout_op=None,
                                  dropout_op_kwargs=None,
                                  nonlin=nn.modules.activation.LeakyReLU,
                                  nonlin_kwargs={'inplace': True}
                                  )
    use_folds = ['0']
    model_training_output_dir = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/nnunet_final'
    checkpoint_name = 'checkpoint_best.pth'

    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                map_location=torch.device('cpu'))
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

    for params in parameters:
        if not isinstance(predictor_net, OptimizedModule):
            predictor_net.load_state_dict(params)
        else:
            predictor_net._orig_mod.load_state_dict(params)

    predictor_net.eval()
    #############################
    # MODEL EXPORT
    # internal model can be fully traced here.  This does not need to be initialised anymore.
    scripted_net = torch.jit.trace(predictor_net, torch.rand(1, 1, 448, 576))
    scripted_net.save('/Users/matt/Desktop/nnunet_model.pt')
    print('model exported')
