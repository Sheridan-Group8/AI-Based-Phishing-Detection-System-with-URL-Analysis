// Preload script — runs with context isolation enabled.
//
// Exposes a minimal surface:
//   - platform / isElectron — used by the renderer to tweak UX
//   - launchSecret — the per-launch token the renderer sends as
//     X-Launch-Secret so the Flask backend can confirm that the request
//     came from the process Electron spawned.
const { contextBridge, ipcRenderer } = require('electron');

// additionalArguments carries the secret from main.js:
//   --phishguard-launch-secret=<hex>
function readLaunchSecret() {
    const flag = '--phishguard-launch-secret=';
    for (const arg of process.argv) {
        if (typeof arg === 'string' && arg.startsWith(flag)) {
            return arg.slice(flag.length);
        }
    }
    return '';
}

contextBridge.exposeInMainWorld('electron', {
    platform: process.platform,
    isElectron: true,
    launchSecret: readLaunchSecret(),
    // Renderer asks the main process to open a URL in the user's default
    // browser. Main validates the URL is on the OAuth whitelist before
    // calling shell.openExternal — the renderer can't open arbitrary
    // links this way.
    openExternal: (url) => ipcRenderer.invoke('pg:open-external', url),
});
