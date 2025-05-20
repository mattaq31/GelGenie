<p align="center">
    <picture>
        <img alt="GelGenie logo" src="./logo/full_logo.png" width="50%" height="auto">
    </picture>
</p>
<p align="center">
    <em>One-click gel electrophoresis analysis.</em>
</p>
<div align="center">

![Platform Support](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/mattaq31/GelGenie/blob/main/LICENSE)
![GitHub all releases](https://img.shields.io/github/downloads/mattaq31/GelGenie/total)
[![QuPath Extension](https://img.shields.io/badge/QuPath-Extension-blue?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAABGdBTUEAALGPC%2FxhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAApdQTFRFAAAAH0qCa673ETJe%2F%2F%2F%2Fb7T7EDFbZKn2BBAff%2F%2F%2FPXzMaKz6GEBzUpTnVcP%2FQ4PTD06aPH%2FUJV%2BoLmmyBA8cWn2pFTlnN5f%2FAggRFz1uMWqxOnjFQoHRTo%2Fge7%2F%2FAAAAJ12gMW24NXPBNnbFOHfHPX3NRYXWUpTlAAwhJ1yfMW66P3%2FOTo%2Ffm%2Bf%2FIlOQMG24PX3MT5DgEzZiLGWsQYHRVZjrH0yFMW66N3XDR4fYAAAAJFaUP3%2FPAAAEJlmZO3rKS4bSAAAAJFWSOHfFMFiNAAAAHUd8NHG9ES5SLWasMnG%2FLGWtAAACHkd7Pm%2BtdImjCRovIU6GM3C8MjY8kpGPCBcrG0JzKV6hMWy3MW25LGauIVCLAgYMAwMDMzMziIiIAAACDCE8FzpnHUd7HkqAG0R3EjNcAAgXAAAAAAAABQUFNTU1AAAAAgICCAgINHC7PG6tQWaYSHKlOWupOm2vd4qpnpadq6KiwcjOlpKZW2qFNmqsPnW8fH6QrpuUuKWk0NPYwb2%2Bl4B2noyHZWx9Mmeqb4auwbq7mIJ4s6SjvrK0noaAk393kXtzeWNXQFZ1OnK7lZCfwrKwxMDDv7u9nH94lHpyqp%2BflHl0hm5jZWRuNG20QXa6kH1%2Fm4B1xL7Ay8%2FToIaBnH57l398nIWCuLS2gnmANGepQXe%2BjXuAkXlroYyFxcTGwsLFvLm8ubq%2Bvb%2FDs6Sid2VsNWuvOHTAfoOdjndpiXNowLq7zdTYztTXyNHWvcHFkXZtXF1yVIC7f3FtjnpwysrNv7%2FCtbO2v8TJubi7fG9uPGedNXK9WHmplpOcwcXJp52bo5SQqJmXmpmkU3iqM3C8UIPGbYaviJu4iZu4Zn6mQnKxHUFvNnK%2BNXC8%2F%2F%2F%2FYgFKdAAAAG90Uk5TAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEPIB8OAQEqg8Pb28F%2FJgNc3dlSAVjt6Eoi08cWb%2F77WQeqkhLBqgMQvaEBBJl3VfXoMxGtoQUuw%2F7WQSia6P383oms%2FtlSCTZrh4RgJwIaovJ%2FHV0R1jkCdQAAAAFiS0dEBI9o2VEAAAAJcEhZcwAACxMAAAsTAQCanBgAAAELSURBVBjTY2AAAkZJKWkZWTkmBghglldQVFJWUVVTZ2EF8dk0NLXyCwqLivO1dXTZGRg4OPX0S0rLyisqq6oNDLm4GXiMjGtq6%2BobGpuaW1pNTHkZ%2BMzM29o7Oru6e3r7%2Bi0s%2BRmsrCdMnDR5ytRp02fMnDXbRoDB1m7O3HnzFyxctHjJ0mXL7R0YHJ1WrFy1es3ades3bNy02dmFwdVty9Zt23fs3LV7z959%2Be6CDEIenvsPHDx0%2BMjRY8dPeHkLM4j4%2BJ48dfrM2XPnL1y85OcvysAgFhCYf%2FnK1WvXbwTdDA5hAbpdPDQsPOLW7cio6JjYuHiQbyQSEpOSU1LT0jMys7IZUEBObh4AnN1fbGkkYr8AAAAldEVYdGRhdGU6Y3JlYXRlADIwMTYtMDktMTFUMTk6MDY6MzUrMDE6MDCMiha4AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE2LTA5LTExVDE5OjA2OjI3KzAxOjAwpuK%2FswAAAABJRU5ErkJggg%3D%3D)](https://qupath.github.io)
</div>

---
Main code repository for GelGenie, an app that uses deep learning models to segment gel electrophoresis images and completely automate the gel analysis process.  The repo is split into two:
- `python-gelgenie` contains a python package for preparing gel image datasets, creating segmentation architectures and training/evaluating deep learning models with PyTorch.  More details on usage and installation in the python package [README](./python-gelgenie/README.md).
- `qupath-gelgenie` contains a QuPath extension that provides an easy-to-access interface for GelGenie models as well as a rich set of tools for analysing and exporting segmentation results.

GelGenie Feature Highlights:
- One-click segmentation of gel electrophoresis images
- General-purpose models that work with a wide range of gel conditions, imagers and stains
- Customisable band measurement, bar chart generation and normalisation
- Easy export to csv for further processing
- Scriptable interface for model inference and labelling
- Python environment for training new models and advanced features
- Open-source and configurable by everyone!

<div style="display: flex; flex-wrap: wrap; justify-content: space-around;" align="center">
  <img src="/qupath-gelgenie/screenshots/example_inference_1.gif" alt="GelGenie in action" style="width: 60%; margin: 0.5%;">
  <img src="./qupath-gelgenie/screenshots/s1.png" alt="GelGenie Screenshot 1" style="width: 47%; margin: 0.5%;">
  <img src="./qupath-gelgenie/screenshots/s2.png" alt="GelGenie Screenshot 2" style="width: 47%; margin: 0.5%;">
  <img src="./qupath-gelgenie/screenshots/s3.png" alt="GelGenie Screenshot 3" style="width: 47%; margin: 0.5%;">
  <img src="./qupath-gelgenie/screenshots/s4.png" alt="GelGenie Screenshot 4" style="width: 47%; margin: 0.5%;">
</div>

## Installing the QuPath GelGenie Extension

- First, install the latest version of QuPath (currently 0.5.1), following the instructions for your operating system [here](https://qupath.github.io).
- Next, download the latest version of the GelGenie extension from the [releases](https://github.com/mattaq31/GelGenie/releases) page (make sure to only download the .jar file):

<p align="center">
<img src="./qupath-gelgenie/screenshots/release_download.png" alt="How to download GelGenie" width="500">
</p>

- Then drag & drop the downloaded .jar file onto the main QuPath window to install it:
<p align="center">
<img src="./qupath-gelgenie/screenshots/drag_n_drop.png" alt="Drag-n-drop installation" width="500">
</p>

- Finally, run GelGenie from the extensions menu:

<p align="center">
<img src="./qupath-gelgenie/screenshots/open_extension.png" alt="How to activate GelGenie" width="500">
</p>

## QuPath GelGenie Extension Tutorial/Manual

- For a comprehensive manual describing the installation and use of GelGenie, jump to the readme page [here](/qupath-gelgenie/README.md).
- A quick-start video showing how to install the app and the main features is available here: [https://youtu.be/_HZyM8Shgc0](https://youtu.be/_HZyM8Shgc0).

## Quick Access for Fiji/ImageJ Users (BioImage.IO)
- We've also provided our core models in [BioImage.IO](https://bioimage.io/#/) format for those more familiar with ImageJ.  Check the instructions [here](/python-gelgenie/gelgenie/segmentation/bioimage_io_handling/deepimagej_macros/README.md) for installing our models within ImageJ, and for a macro to help run and analyze images directly.

##  GelGenie's Python Environment and Training New Models (for developers)

Full documentation, Python installation instructions and more details are provided here [Python Quick Start](./python-gelgenie/README.md).
##  Acessing Pre-Trained Models (for developers)

All pre-trained models are available on HuggingFace.  The main registry can be found at the link [here](https://huggingface.co/datasets/mattaq/GelGenie-Model-Zoo/blob/main/registry.json).  The QuPath GelGenie extension directly downloads the models from HuggingFace, with no user tweaking required.

## Accessing Labelled Gel Datasets 

All data can be downloaded in the original format used for training, validation and testing from our Zenodo upload with DOI [10.5281/zenodo.14641949](https://doi.org/10.5281/zenodo.14641949).

## Development & Support
The principal GelGenie researcher and developer is [Matthew Aquilina](https://www.linkedin.com/in/matthewaq/), who built up the concept of gel electrophoresis segmentation and oversaw the project together with [Katherine Dunn](https://www.katherinedunnresearch.eng.ed.ac.uk) at the University of Edinburgh.  The other developers/scientists who contributed to the project are:

- [Nathan Wu](https://nathanw23.github.io) - Gel labelling, lab data generation, data analysis and statistical pipeline development
- [Kiros Kwan](https://www.linkedin.com/in/kiros-kwan/) - Gel labelling, deep learning framework development and model training
- [Filip Buŝić](https://www.linkedin.com/in/filipbusic/) - Image analysis, classical segmentation algorithms and prototype GUI development
- [James Dodd](https://www.linkedin.com/in/james-dodd-b636041ab/) - Gel labelling, lab data generation, prototype testing and feedback
- [Peter Bankhead](https://github.com/petebankhead) - QuPath extension development, deep learning algorithms and java development guidance
- [Alan O'Callaghan](https://github.com/alanocallaghan) - QuPath extension development and java development guiidance
- [Laura Nicolás-Sáenz](https://github.com/lauranicolass) - Clustering algorithm development
- We also received advice, gel data (thanks Arun, Stella, Yichen, Huangchen, Thomas, Joana and Ricarda!) and alpha testing feedback from other colleagues in both the University of Edinburgh and the Dana-Farber/Wyss Institute for which we are incredibly grateful! 

The project was directly supported by both the [School of Engineering](https://www.eng.ed.ac.uk) (who funded Kiros and Filip) and the [Precision Medicine Doctoral Training Programme](https://www.ed.ac.uk/usher/precision-medicine) at the University of Edinburgh (via Medical Research Council (MRC) grant number MR/N013166/1).  The EDDIE compute cluster (from the Edinburgh Compute and Data Facility ([ECDF](http://www.ecdf.ed.ac.uk/))) was used to train the baseline machine learning models.

For more details of everyone's coding contributions, please check the graphs [here](https://github.com/mattaq31/GelGenie/graphs/contributors).

Contributions from the open-source community are welcome!  In particular, we are looking for help with introducing unit tests to both the Python and Java GelGenie packages!

## Literature Citation

Our main manuscript has been published in Nature Communications, with DOI: [https://doi.org/10.1038/s41467-025-59189-0](https://doi.org/10.1038/s41467-025-59189-0)

Bibtex citation details available below:

```bibtex
@article{aquilinaGelGenieAIpoweredFramework2025,
	title = {{GelGenie}: an {AI}-powered framework for gel electrophoresis image analysis},
	volume = {16},
	copyright = {All rights reserved},
	issn = {2041-1723},
	url = {https://doi.org/10.1038/s41467-025-59189-0},
	doi = {10.1038/s41467-025-59189-0},
	abstract = {Gel electrophoresis is a ubiquitous laboratory method for the separation and semi-quantitative analysis of biomolecules. However, gel image analysis principles have barely advanced for decades, in stark contrast to other fields where AI has revolutionised data processing. Here, we show that an AI-based system can automatically identify gel bands in seconds for a wide range of experimental conditions, surpassing the capabilities of current software in both ease-of-use and versatility. We use a dataset containing 500+ images of manually-labelled gels to train various U-Nets to accurately identify bands through segmentation, i.e. classifying pixels as ‘band’ or ‘background’. When applied to gel electrophoresis data from other laboratories, our system generates results that quantitatively match those of the original authors. We have publicly released our models through GelGenie, an open-source application that allows users to extract bands from gel images on their own devices, with no expert knowledge or experience required.},
	number = {1},
	journal = {Nature Communications},
	author = {Aquilina, Matthew and Wu, Nathan J. W. and Kwan, Kiros and Bušić, Filip and Dodd, James and Nicolás-Sáenz, Laura and O’Callaghan, Alan and Bankhead, Peter and Dunn, Katherine E.},
	month = may,
	year = {2025},
	pages = {4087},
}
```
Tools for recreating our analyses in the above manuscript are provided [here](https://github.com/mattaq31/GelGenie/tree/main/python-gelgenie).  The manual analyses data generated using GelAnalyzer and ImageStudio can be accessed from our Zenodo deposition [here](https://doi.org/10.5281/zenodo.14827469).


