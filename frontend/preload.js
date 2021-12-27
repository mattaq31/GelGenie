const {contextBridge, ipcRenderer} = require('electron')

contextBridge.exposeInMainWorld(  // bridge between main and renderer processes
    'electron',
    {
        export: (request) => ipcRenderer.send('export', request)
    }
)