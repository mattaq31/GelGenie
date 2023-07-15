GelGenie Backend
==============================
Full Description TBD

Packaging Gel Server
==============================
The following command packages the gel server for use as an executable:

`pyinstaller server.py -n gel_server --onedir --hidden-import skimage.filters.edges --hidden-import engineio.async_drivers.aiohttp --hidden-import engineio.async_aiohttp --hidden-import skimage.filters.thresholding --hidden-import skimage.segmentation`

Full details on packaging etc TBD