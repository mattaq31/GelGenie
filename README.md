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

Repo containing 1) a python package for training and evaluating gel image segmentation models, 2) a QuPath plugin to use these models directly in QuPath and 3) a prototype Electron GUI for segmenting gel images using classical watershed segmentation.

Full Description TBD

- Python model training and analysis located in `./gelgenie`
- Qupath Extension code located in `./qupath-gelgenie`
- Some initial tests for using QuPath to label gel images are located in `./Semi-Auto Labelling`
- Electron GUI located in `./prototype_frontend`
- Rough scripts and data located in `./scratch`


Setting up GelGenie Environment
==============================
To install Python package run the following command from the home directory:

`pip install -e .`

Package requirements and installation details coming soon.
