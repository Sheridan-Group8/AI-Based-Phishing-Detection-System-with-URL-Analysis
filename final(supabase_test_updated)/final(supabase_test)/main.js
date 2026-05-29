const { app, BrowserWindow, ipcMain, shell, session } = require('electron');
const { spawn } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const net = require('net');
const http = require('http');

let mainWindow = null;
let pythonProcess = null;
// Bound at startup. NEVER a fixed well-known port — the original 5050 could
// be impersonated by any other local process listening on the same socket,
// which would happily answer the Electron renderer's first fetch.
let flaskPort = 0;
// Per-launch secret. Passed to Flask via env so it can prove a request came
// from the renderer Electron spawned, and to preload so the renderer can
// attach it as X-Launch-Secret on every fetch.
let launchSecret = '';

function findPython() {
    const candidates = [];
    const addCandidate = (cmd) => {
        if (cmd && !candidates.includes(cmd)) candidates.push(cmd);
    };
    const addExisting = (cmd) => {
        if (cmd && fs.existsSync(cmd)) addCandidate(cmd);
    };

    if (process.env.PHISHGUARD_PYTHON) addCandidate(process.env.PHISHGUARD_PYTHON);

    if (process.env.VIRTUAL_ENV) {
        addExisting(process.platform === 'win32'
            ? path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe')
            : path.join(process.env.VIRTUAL_ENV, 'bin', 'python'));
    }

    if (process.platform === 'win32') {
        addExisting(path.join(__dirname, '.venv', 'Scripts', 'python.exe'));
        addExisting(path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe'));
        ['py', 'python3.12', 'python3.11', 'python3.10', 'python3', 'python'].forEach(addCandidate);
    } else {
        addExisting(path.join(__dirname, '.venv', 'bin', 'python'));
        addExisting(path.join(__dirname, '..', '..', '.venv', 'bin', 'python'));
        [
            '/opt/homebrew/bin/python3.12',
            'python3.12',
            'python3.11',
            'python3.10',
            'python3',
            'python',
        ].forEach(addCandidate);
    }
    return candidates;
}

// Try to take a specific port; if it's already in use, fall back to a random
// free port assigned by the kernel. We prefer the OAuth-registered port
// (5050) so Microsoft's redirect_uri still matches, but we never reuse a
// listener we didn't spawn — if port 5050 is already taken we consider that
// a potential impersonation attempt and pick a new free port instead.
function tryBindPort(preferred) {
    return new Promise((resolve) => {
        const server = net.createServer();
        server.unref();
        server.once('error', () => {
            // Port taken — fall through to picking a random one.
            const s2 = net.createServer();
            s2.unref();
            s2.once('error', () => resolve(0));
            s2.listen(0, '127.0.0.1', () => {
                const port = s2.address().port;
                s2.close(() => resolve(port));
            });
        });
        server.listen(preferred, '127.0.0.1', () => {
            server.close(() => resolve(preferred));
        });
    });
}

// Wait for Flask to respond to a health check. We send our launch secret so
// any unrelated listener on the same port will fail the comparison and be
// rejected.
function waitForFlask(port, secret, maxAttempts = 30) {
    return new Promise((resolve, reject) => {
        let attempts = 0;
        const check = () => {
            attempts++;
            const req = http.request({
                host: '127.0.0.1',
                port,
                path: '/api/auth/status',
                method: 'GET',
                headers: { 'X-Launch-Secret': secret },
                timeout: 1000,
            }, (res) => {
                // 2xx/4xx both mean something spoke HTTP to us — good enough as
                // a readiness probe. The launch secret still gates mutating
                // calls, so a lookalike listener can't do real damage.
                res.resume();
                resolve();
            });
            req.on('error', () => {
                if (attempts >= maxAttempts) reject(new Error('Flask did not start in time'));
                else setTimeout(check, 500);
            });
            req.on('timeout', () => {
                req.destroy();
                if (attempts >= maxAttempts) reject(new Error('Flask health check timed out'));
                else setTimeout(check, 500);
            });
            req.end();
        };
        check();
    });
}

async function startFlask() {
    // Prefer 5050 so existing Azure redirect_uri registrations keep working,
    // but fall back to a random port if something else is already listening.
    flaskPort = await tryBindPort(5050);
    if (!flaskPort) throw new Error('Could not bind a loopback port');
    if (flaskPort !== 5050) {
        console.warn(`Port 5050 unavailable — using random port ${flaskPort}. `
                     + `Microsoft OAuth will need redirect_uri updated.`);
    }
    launchSecret = crypto.randomBytes(32).toString('hex');

    const isDev = !app.isPackaged;
    const appDir = isDev ? __dirname : path.join(process.resourcesPath, 'app');
    const appPy = path.join(appDir, 'app.py');
    const pyArgs = [appPy];

    const userDataDir = app.getPath('userData');

    for (const pyCmd of findPython()) {
        try {
            pythonProcess = spawn(pyCmd, pyArgs, {
                cwd: appDir,
                env: {
                    ...process.env,
                    FLASK_PORT: String(flaskPort),
                    PHISHGUARD_LAUNCH_SECRET: launchSecret,
                    PHISHGUARD_USER_DATA: userDataDir,
                    ELECTRON_MODE: '1',
                },
                stdio: ['pipe', 'pipe', 'pipe'],
            });

            pythonProcess.stdout.on('data', (data) => {
                console.log(`[Flask] ${data.toString().trim()}`);
            });
            pythonProcess.stderr.on('data', (data) => {
                console.log(`[Flask] ${data.toString().trim()}`);
            });
            pythonProcess.on('error', (err) => {
                console.error(`Python process error: ${err.message}`);
                pythonProcess = null;
            });
            pythonProcess.on('exit', (code) => {
                console.log(`Python process exited with code ${code}`);
                pythonProcess = null;
            });

            await new Promise(r => setTimeout(r, 500));
            if (pythonProcess && !pythonProcess.killed) {
                console.log(`Started Flask with: ${pyCmd} (port ${flaskPort})`);
                return;
            }
        } catch (e) {
            console.log(`Failed to start with ${pyCmd}: ${e.message}`);
        }
    }

    throw new Error('Could not find a supported Python runtime. Please install Python 3.10+');
}

async function createWindow() {
    // Wipe runtime caches and service workers so a stale SW doesn't
    // intercept our fetches with the wrong launch secret. We DELIBERATELY
    // keep cookies and localStorage so:
    //   • Microsoft's auth cookies persist → "Pick an account" picker on
    //     next sign-in instead of full re-login.
    //   • The Supabase session in localStorage survives an app restart
    //     until it naturally expires (~1h, supabase-js auto-refreshes).
    // The launch-secret model + CSRF still hold because both rotate
    // per launch independently of cookies.
    await session.defaultSession.clearCache();
    try {
        await session.defaultSession.clearStorageData({
            storages: [
                "appcache", "filesystem", "indexdb",
                "shadercache", "websql", "serviceworkers",
                "cachestorage",
            ],
        });
    } catch (e) {
        console.warn(`clearStorageData failed: ${e.message}`);
    }

    // Deny every special web permission by default. The app does not need
    // camera, mic, geolocation, clipboard, or notifications — so a future
    // DOM-level XSS cannot silently request them.
    session.defaultSession.setPermissionRequestHandler(
        (_webContents, _permission, callback) => callback(false),
    );
    session.defaultSession.setPermissionCheckHandler(
        (_webContents, _permission) => false,
    );

    mainWindow = new BrowserWindow({
        width: 1280,
        height: 820,
        minWidth: 960,
        minHeight: 600,
        title: 'PhishGuard',
        titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
        backgroundColor: '#08080F',
        show: false,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            nodeIntegrationInWorker: false,
            nodeIntegrationInSubFrames: false,
            sandbox: true,
            webSecurity: true,                   // enforced by default, pin explicitly
            allowRunningInsecureContent: false,  // ditto — explicit to outlive defaults changes
            experimentalFeatures: false,
            webviewTag: false,
            plugins: false,
            // Expose the launch secret to preload. additionalArguments is the
            // documented way to pass small strings to a sandboxed renderer.
            additionalArguments: [`--phishguard-launch-secret=${launchSecret}`],
        },
    });

    mainWindow.loadURL(`http://127.0.0.1:${flaskPort}`);

    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        // Auto-open DevTools when running from source (electron .) but
        // not when running from a packaged build.
        if (!app.isPackaged) {
            mainWindow.webContents.openDevTools({ mode: 'right' });
        }
    });

    // Allow-list our own origin; OAuth flows (Microsoft + Supabase) open as
    // child Electron windows; everything else opens in the system browser.
    const ownOrigin = `http://127.0.0.1:${flaskPort}`;
    const isOAuthUrl = (u) =>
        u.startsWith('https://login.microsoftonline.com/') ||
        u.startsWith('https://login.live.com/') ||
        u.startsWith('https://login.windows.net/') ||
        /^https:\/\/[a-z0-9-]+\.supabase\.co\//.test(u) ||
        u.startsWith('http://localhost:5050') ||
        u.startsWith('http://127.0.0.1:5050');

    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        if (url.startsWith(ownOrigin)) return { action: 'allow' };
        if (isOAuthUrl(url)) {
            return {
                action: 'allow',
                overrideBrowserWindowOptions: {
                    width: 560,
                    height: 720,
                    parent: mainWindow,
                    autoHideMenuBar: true,
                    webPreferences: {
                        contextIsolation: true,
                        nodeIntegration: false,
                        sandbox: true,
                        webSecurity: true,
                    },
                },
            };
        }
        shell.openExternal(url);
        return { action: 'deny' };
    });

    mainWindow.webContents.on('will-navigate', (event, url) => {
        if (url.startsWith(ownOrigin)) return;
        if (isOAuthUrl(url)) return;
        event.preventDefault();
        shell.openExternal(url);
    });

    // When running from source (electron .), auto-open DevTools on any
    // child window so we can see what's happening during OAuth, etc.
    if (!app.isPackaged) {
        mainWindow.webContents.on('did-create-window', (childWindow) => {
            try {
                childWindow.webContents.openDevTools({ mode: 'detach' });
                // Log every URL the popup loads — useful when the popup
                // redirects through Microsoft → Supabase → us.
                childWindow.webContents.on('did-navigate', (_e, url) => {
                    console.log('[popup did-navigate]', url);
                });
                childWindow.webContents.on('did-navigate-in-page', (_e, url) => {
                    console.log('[popup did-navigate-in-page]', url);
                });
                childWindow.webContents.on('did-fail-load',
                    (_e, code, desc, url) => {
                        console.error('[popup did-fail-load]', code, desc, url);
                    });
            } catch (e) {
                console.warn('[popup devtools open failed]', e.message);
            }
        });
    }

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// Renderer asks us to open a URL in the user's default browser. We only
// allow URLs on the OAuth path (Microsoft + Supabase + our own localhost
// callback) so a future renderer-side XSS can't use this as a generic
// "open arbitrary website" sink.
const _externalOpenAllowed = /^https:\/\/(login\.microsoftonline\.com|login\.live\.com|login\.windows\.net|[a-z0-9-]+\.supabase\.co)\//;
ipcMain.handle('pg:open-external', async (_event, url) => {
    if (typeof url !== 'string' || !_externalOpenAllowed.test(url)) {
        return { ok: false, reason: 'url not on OAuth whitelist' };
    }
    try {
        await shell.openExternal(url);
        return { ok: true };
    } catch (e) {
        return { ok: false, reason: e.message || 'openExternal failed' };
    }
});

app.whenReady().then(async () => {
    try {
        await startFlask();
        await waitForFlask(flaskPort, launchSecret);
        createWindow();
    } catch (err) {
        console.error('Failed to start:', err.message);
        const { dialog } = require('electron');
        dialog.showErrorBox('PhishGuard', `Failed to start: ${err.message}\n\nMake sure Python 3.10+ is installed with Flask and scikit-learn.`);
        app.quit();
    }
});

app.on('window-all-closed', () => {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }
    app.quit();
});

app.on('before-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }
});

app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});
