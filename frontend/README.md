Gel Electrophoresis Automatic Analyzer Frontend
==============================
##Development

To install necessary packages for running electron app, run:

`npm install`

To start system in development mode, run:

`npm start`

The python server will need to be initialized separately.
##Build/Distribution

To package the app for distribution, run one of the build commands in package.json.

The python server needs to be bundled separately using `pyinstaller` and then placed in a new directory 'build' under frontend (not synced on git).

More details TBD.