import torch

from bioimageio.spec.model.v0_5 import (
    Author,
    AxisId,
    BatchAxis,
    ChannelAxis,
    CiteEntry,
    Doi,
    FileDescr,
    HttpUrl,
    Identifier,
    InputTensorDescr,
    IntervalOrRatioDataDescr,
    LicenseId,
    ModelDescr,
    OrcidId,
    ParameterizedSize,
    SpaceInputAxis,
    TensorId,
    TorchscriptWeightsDescr,
    WeightsDescr,
    OutputTensorDescr,
    SizeReference,
    SpaceOutputAxis,
    ArchitectureFromFileDescr,
    Version,
)

from bioimageio.core import test_model
from bioimageio.spec import save_bioimageio_package


# this script prepares and packages gelgenie models into bioimage.io format (also compaible with deepimagej).  You need to run the prepare_sample_data.py file first to be able to run this code.

model_selected = 'universal'

if model_selected == 'finetuned':
    root = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/bioimage_io_models/finetuned_model'
    model_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/unet_dec_21_finetune'
    test_input = '/test_data/test_input.npy'
    sample_input = '/test_data/input_134.tif'
    test_output = '/test_data/test_output.npy'
    sample_output = '/test_data/output_134.tif'
    model_name = 'GelGenie-Finetuned-V1'
    description = 'U-Net trained to segment and extract gel bands from gel electrophoresis images. This is the finetuned version (V1) of the model.'
    model_weights = '/torchscript_checkpoints/unet_dec_21_finetune_epoch_590.pt'
    output_filename = '/gelgenie_finetuned_model_bioimageio.zip'
else:
    root = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/bioimage_io_models/universal_model'
    model_folder = '/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/segmentation_models/December 2023/unet_dec_21'
    test_input = '/test_data/test_input.npy'
    sample_input = '/test_data/input_134.tif'
    test_output = '/test_data/test_output.npy'
    sample_output = '/test_data/output_134.png'
    model_name = 'GelGenie-Universal-V1'
    description = 'U-Net trained to segment and extract gel bands from gel electrophoresis images. This is the universal version (V1) of the model.'
    model_weights = "/torchscript_checkpoints/unet_dec_21_epoch_579.pt"
    output_filename = '/gelgenie_universal_model_bioimageio.zip'

pytorch_version = Version(torch.__version__)

# this reads in the model directly from the file and prepares the architecture - you also need to provide any keyword arguments here for it to work.
# This is not actually required to export the model - can just use the torchscript weights instead.
pytorch_architecture = ArchitectureFromFileDescr(
    source="/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/main_code/python-gelgenie/gelgenie/segmentation/networks/UNets/model_gateway.py",
    callable=Identifier("smp_UNet"),
    kwargs=dict(
        in_channels=1,
        classes=2,
        encoder_name='resnet18',
    )
)

# prepares expected input formatting here - not sure of the syntax for this, mainly copied from tutorials
# it seems that the ParameterizedSize class helps deepimagej automatically pad/unpad input images, which is great.
input_descr = InputTensorDescr(
    id=TensorId("input"),
    axes=[BatchAxis(),
          ChannelAxis(channel_names=[Identifier("input")]),
          SpaceInputAxis(
              id=AxisId('y'),
              size=ParameterizedSize(min=32, step=32),
              scale=1,
              concatenable=False),
          SpaceInputAxis(
              id=AxisId('x'),
              size=ParameterizedSize(min=32, step=32),
              scale=1,
              concatenable=False),
    ],
    test_tensor=FileDescr(source=root + test_input),
    sample_tensor=FileDescr(source=root + sample_input),
    data=IntervalOrRatioDataDescr(type="float32"),
)

# prepares expected output formatting here - not sure of the syntax for this, mainly copied from tutorials
output_descr = OutputTensorDescr(
    id=TensorId("prediction"),
    axes=[BatchAxis(),
          ChannelAxis(channel_names=[Identifier("prediction_1"), Identifier("prediction_2")]),
          SpaceOutputAxis(id=AxisId('y'),
                          scale=1,
                          size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("y"))),
          SpaceOutputAxis(id=AxisId('x'),
                          scale=1,
                          size=SizeReference(tensor_id=TensorId("input"), axis_id=AxisId("x"))),
    ],
    test_tensor=FileDescr(source=root + test_output),
    sample_tensor=FileDescr(source=root + sample_output),
    data=IntervalOrRatioDataDescr(type="float32"),
)


# puts all metadata together and generates model here
my_model_descr = ModelDescr(
  name=model_name,
  description=description,
  covers=[root + "/cover_images/cover_1.png",
          root + "/cover_images/cover_2.png",
          root + "/cover_images/cover_3.png"],
  authors=[
      Author(
          name="Matthew Aquilina",
          affiliation="Dana-Farber Cancer Institute & Wyss Institute",
          github_user="mattaq31",
          orcid=OrcidId("0000-0002-4039-1398"))
  ],
  cite=[
    CiteEntry(text=("Aquilina M, Wu NJW, Kwan K, Busic F, Dodd J, Nicolas-Saenz L, et al. "
"GelGenie: an AI-powered framework for gel electrophoresis image analysis. "
"bioRxiv. 2024 Sep 6;2024.09.06.611479."),
      doi=Doi("10.1101/2024.09.06.611479"))
  ],
  license=LicenseId("Apache-2.0"),
  documentation=root + "/README.md",
  git_repo=HttpUrl("https://github.com/mattaq31/GelGenie"),
  tags= ['Gel Electrophoresis', 'Gel Quantitation', 'Machine Learning',
         'Image Segmentation', 'Pytorch', 'QuPath', 'DeepImageJ'],
  inputs=[input_descr],
  outputs=[output_descr],
  weights=WeightsDescr(
      torchscript=TorchscriptWeightsDescr(
          source=model_folder + model_weights,
          pytorch_version=pytorch_version,
      ),
  ),
  attachments=[FileDescr(source=model_folder + "/config.toml")],
)

# tests model (output cannot be visualized in pycharm - use a notebook)
validation_summary = test_model(my_model_descr)
validation_summary.display()

# saves to file, ready for use or upload to bioimage.io
save_bioimageio_package(my_model_descr, output_path= root + output_filename)

