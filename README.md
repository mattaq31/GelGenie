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
Main coding repository for GelGenie, an app that uses deep learning models to automatically segment gel electrophoresis images.  The repo is split into two:
- `python-gelgenie` contains a python package for preparing gel image datasets, creating segmentation architectures and training/evaluating deep learning models with PyTorch.  More details on usage and installation in the python package [README](./python-gelgenie/README.md).
- `qupath-gelgenie` contains a QuPath extension that provides an easy-to-access interface for GelGenie models as well as a rich set of tools for analysing and exporting segmentation results.

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

Accessing Labelled Gel Datasets
===============================

TBC

Training New Segmentation Models
================================

TBC

Development & Support
==============================

The principal GelGenie researcher and developer is [Matthew Aquilina](https://www.linkedin.com/in/matthewaq/), who built up the concept of gel electrophoresis segmentation and oversaw the project together with [Katherine Dunn](https://www.katherinedunnresearch.eng.ed.ac.uk) at the University of Edinburgh.  Many others have also contributed to the project:

- [Nathan Wu](https://nathanw23.github.io) - Gel labelling, lab data generation, data analysis and statistical pipeline development
- [Kiros Kwan](https://www.linkedin.com/in/kiros-kwan/) - Gel labelling, deep learning framework development and model training
- [Filip Buŝić](https://www.linkedin.com/in/filipbusic/) - Image analysis, classical segmentation algorithms and prototype GUI development
- [James Dodd](https://www.linkedin.com/in/james-dodd-b636041ab/) - Gel labelling, lab data generation, prototype testing and feedback
- [Peter Bankhead](https://github.com/petebankhead) - QuPath extension development, deep learning algorithms and java development guidance
- Details of other members of QuPath team TBC

The project was supported by both the [School of Engineering](https://www.eng.ed.ac.uk) (who funded Kiros and Filip) and the [Precision Medicine Doctoral Training Programme](https://www.ed.ac.uk/usher/precision-medicine) at the University of Edinburgh (via Medical Research Council (MRC) grant number MR/N013166/1).  The EDDIE compute cluster (from the Edinburgh Compute and Data Facility ([ECDF](http://www.ecdf.ed.ac.uk/))) was used to train the baseline machine learning models.

For more details of everyone's coding contributions, please check the graphs [here](https://github.com/mattaq31/GelGenie/graphs/contributors).

Citation details TBC
