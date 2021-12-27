Gel Electrophoresis Automatic Analyzer Backend
==============================
Full Description TBD

Setting up Environment
==============================
To install package run the following command from the home directory:

`pip install -e .`


Packaging Gel Server
==============================
The following command packages the gel server for use as an executable:

`pyinstaller server.py -n gel_server --onedir --hidden-import skimage.filters.edges --hidden-import engineio.async_drivers.aiohttp --hidden-import engineio.async_aiohttp --hidden-import skimage.filters.thresholding --hidden-import skimage.segmentation`

Full details on packaging etc TBD