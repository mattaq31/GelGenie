Gel Electrophoresis Automatic Analyzer Prototype Frontend
==============================
This app was an early prototype for what would eventually become GelGenie (developed by Filip and Matthew).  It is still functional but has limited features.

## Development

To install necessary packages for running electron app, run:

`npm install`

To start system in development mode, run:

`npm start`

The python server will need to be initialized separately and also requires additional python packages.

## Build/Distribution

To package the app for distribution, run one of the build commands in package.json.

The python server needs to be bundled separately using `pyinstaller` and then placed in a new directory 'build' under prototype_frontend (not synced on git):

`pyinstaller ../prototype_frontend/server.py -n gel_server --onedir --hidden-import skimage.filters.edges --hidden-import engineio.async_drivers.aiohttp --hidden-import engineio.async_aiohttp --hidden-import skimage.filters.thresholding --hidden-import skimage.segmentation`
