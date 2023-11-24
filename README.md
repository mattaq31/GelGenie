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
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)
![GitHub all releases](https://img.shields.io/github/downloads/mattaq31/GelGenie/total)
</div>

---
Main repository for GelGenie, an app that uses deep learning models to automatically segment gel electrophoresis images.  The repo is split into two:
- `python-gelgenie` contains a python package for preparing gel image datasets, creating segmentation architectures and training/evaluating deep learning models with PyTorch.  More details on usage and installation in the python package [README](./python-gelgenie/README.md).
- `qupath-gelgenie` contains a QuPath extension that provides an easy-to-access interface for GelGenie models as well as a rich set of tools for analysing and exporting segmentation results.

Repo containing 1) a python package for training and evaluating gel image segmentation models, 2) a QuPath plugin to use these models directly in QuPath and 3) a prototype Electron GUI for segmenting gel images using classical watershed segmentation.

Full Description, feature list and installation instructions TBD

ALSO ADD SCREENSHOTS OF APP USAGE

Installing the QuPath GelGenie Extension
==============================

Download the latest version of the extension from the XXXX page.

Then drag & drop the downloaded .jar file onto the main QuPath window to install it. 

ADD SCREENSHOTS HERE

Installing GelGenie's Python Environment
==============================
To install the gelgenie python package and its dependencies simply run the following command from the `./python-gelgenie` directory:

`pip install -e .`

Package requirements and further installation details coming soon.

ADD DETAILS OF ELECTRON SERVER TOO
