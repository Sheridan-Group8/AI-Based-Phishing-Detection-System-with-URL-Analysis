const { app, BrowserWindow, dialog } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');

let mainWindow = null;
let flaskProcess = null;

const FLASK_PORT = process.env.FLASK_PORT || '5050';
const APP_URL = `http://127.0.0.1:${FLASK_PORT}`;

function pythonCandidates() {
    const candidates = [];
    const add = (cmd) => {
        if (cmd && !candidates.includes(cmd)) candidates.push(cmd);
    };
    const addIfExists = (cmd) => {
        if (fs.existsSync(cmd)) add(cmd);
    };

    if (process.env.PHISHGUARD_PYTHON) add(process.env.PHISHGUARD_PYTHON);

    if (process.env.VIRTUAL_ENV) {
        addIfExists(process.platform === 'win32'
            ? path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe')
            : path.join(process.env.VIRTUAL_ENV, 'bin', 'python'));
    }

    addIfExists(process.platform === 'win32'
        ? path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe')
        : path.join(__dirname, '..', '.venv', 'bin', 'python'));

    if (process.platform === 'win32') {
        ['py', 'python', 'python3'].forEach(add);
    } else {
        ['python3', 'python'].forEach(add);
    }

    return candidates;
}

function waitForFlask(attemptsLeft = 40) {
    return new Promise((resolve, reject) => {
        const check = () => {
            const req = http.get(APP_URL, (res) => {
                res.resume();
                resolve();
            });

            req.on('error', () => {
                if (attemptsLeft <= 1) {
                    reject(new Error(`Flask did not start at ${APP_URL}`));
                    return;
                }
                setTimeout(() => {
                    attemptsLeft -= 1;
                    check();
                }, 250);
            });

            req.setTimeout(1000, () => {
                req.destroy();
            });
        };

        check();
    });
}

async function startFlask() {
    const appPy = path.join(__dirname, 'app.py');
    let lastError = null;

    for (const command of pythonCandidates()) {
        try {
            flaskProcess = spawn(command, [appPy], {
                cwd: __dirname,
                env: {
                    ...process.env,
                    FLASK_PORT,
                    PHISHGUARD_DEBUG: process.env.PHISHGUARD_DEBUG || '',
                },
                stdio: ['ignore', 'pipe', 'pipe'],
            });

            flaskProcess.stdout.on('data', (data) => {
                console.log(`[Flask] ${data.toString().trim()}`);
            });
            flaskProcess.stderr.on('data', (data) => {
                console.error(`[Flask] ${data.toString().trim()}`);
            });
            flaskProcess.on('exit', (code) => {
                console.log(`Flask exited with code ${code}`);
                flaskProcess = null;
            });

            await waitForFlask();
            return;
        } catch (error) {
            lastError = error;
            if (flaskProcess) {
                flaskProcess.kill();
                flaskProcess = null;
            }
        }
    }

    throw lastError || new Error('No Python runtime found');
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1280,
        height: 820,
        minWidth: 960,
        minHeight: 600,
        title: 'PhishGuard',
        backgroundColor: '#08080f',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    mainWindow.loadURL(APP_URL);
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

function stopFlask() {
    if (flaskProcess) {
        flaskProcess.kill();
        flaskProcess = null;
    }
}

app.whenReady().then(async () => {
    try {
        await startFlask();
        createWindow();
    } catch (error) {
        dialog.showErrorBox(
            'PhishGuard',
            `Could not start the local dashboard.\n\n${error.message}`
        );
        app.quit();
    }
});

app.on('window-all-closed', () => {
    stopFlask();
    app.quit();
});

app.on('before-quit', stopFlask);
