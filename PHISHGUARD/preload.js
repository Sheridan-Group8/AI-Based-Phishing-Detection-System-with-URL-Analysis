const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('phishguard', {
    isElectron: true,
    platform: process.platform,
});
