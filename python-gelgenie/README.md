GelGenie Python
==============================

- Data analysis scripts located in `data_analysis`
- Old gel functions used for Electron Python server located in `gel_tools`
- All ML code located in `segmentation`

Full Description TBD

Packaging Gel Server (Electron only)
==============================
The following command packages the gel server for use as an executable:

`pyinstaller ../prototype_frontend/server.py -n gel_server --onedir --hidden-import skimage.filters.edges --hidden-import engineio.async_drivers.aiohttp --hidden-import engineio.async_aiohttp --hidden-import skimage.filters.thresholding --hidden-import skimage.segmentation`

Full details on packaging etc TBD
