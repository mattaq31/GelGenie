// TODO: update appID and other build options including app signing

// Modules to control application life and create native browser window
const {app, BrowserWindow, dialog, ipcMain} = require('electron')
const path = require('path')
const io = require('socket.io-client');
const {PythonShell} = require('python-shell');

const socket = io("http://localhost:9111");

app.commandLine.appendSwitch('remote-debugging-port', '9222')
function createWindow () {
    // Create the browser window.
    const mainWindow = new BrowserWindow({
        width: 1200,
        height: 1000,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true
        }
    })

    // and load the index.html of the app.
    mainWindow.loadFile('html/index.html')
    // mainWindow.setResizable(false)

    // Open the DevTools.
    // mainWindow.webContents.openDevTools()
}

ipcMain.on( 'export', (e, args) => {

        if (args === 'main_data_csv'){
            dialog.showSaveDialog({
                title: 'Define output name and location',
                filters: [{ name: 'CSV file', extensions: ['csv'] }],
                properties: ['createDirectory']
            }).then( result => {
                socket.emit("exportToCsv", result.filePath);
            })
        }
        else if (args === 'profile_csv'){
            dialog.showSaveDialog({
                title: 'Define output name and location',
                filters: [{ name: 'CSV file', extensions: ['csv'] }],
                properties: ['createDirectory']
            }).then( result => {
                socket.emit("exportProfileCsv", result.filePath);
            })
        }
        else if (args === 'profile_image'){
            dialog.showSaveDialog({
                title: 'Define output name and location',
                filters: [{ name: 'Image file', extensions: ['tif', 'png'] }],
                properties: ['createDirectory']
            }).then( result => {
                socket.emit("exportProfileGraph", result.filePath);
            })
        }
        else if (args.includes('main_image')){
            dialog.showSaveDialog({
                title: 'Define output name and location',
                filters: [{ name: 'Image file', extensions: ['tif', 'png'] }],
                properties: ['createDirectory']
            }).then( result => {
                socket.emit("exportToBandImage", result.filePath, args === 'main_image_light');
            })
        }

})

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {

    createWindow()

    app.on('activate', function () {
        // On macOS it's common to re-create a window in the app when the
        // dock icon is clicked and there are no other windows open.
        if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit()
})

// Initialize tooltips
// $(document).ready(function() {
//     $('.open').tooltip({trigger: 'hover', title: 'Open image', placement: 'bottom'});
//     $('.find').tooltip({trigger: 'hover', title: 'Find bands', placement: 'bottom'});
//     $('#export-btn').tooltip({trigger: 'hover', title: 'Export', placement: 'bottom'});
//     $('#white_bg').tooltip({trigger: 'hover', title: 'Light mode', placement: 'bottom'});
//     $('#black_bg').tooltip({trigger: 'hover', title: 'Dark mode', placement: 'bottom'});
//     // TO DO: Add the rest of the tooltips
// });