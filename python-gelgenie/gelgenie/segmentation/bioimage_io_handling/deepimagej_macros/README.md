Running BioImage.IO models in Fiji using DeepImageJ
===================================================

- To run GelGenie models released in BioImage.IO format, first install Fiji and then install the DeepImageJ plugin (instructions [here](https://deepimagej.github.io)).
- Download a GelGenie model through the BioImage.IO interface within the plugin (more details TBC).
- Load in your gel image.  Make sure it is a single-channel image (if it's RGB, combine it using ImageJ's standard tools - more details TBC).
- Select your image, then run the preprocess.ijm macro to pad and normalize your image.
- Run the GelGenie model through the DeepImageJ interface (more details on scripting TBC).
- Select the model output, then run the postprocess.ijm macro to combine the output into a binary mask.
- Run the measure_areas.ijm macro to measure the area of all bands in your image and output to a csv file.

The above is the current working method, will polish scripts and automated things further when more info available.