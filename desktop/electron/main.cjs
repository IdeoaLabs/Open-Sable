'use strict';

const { app, BrowserWindow, ipcMain, shell, nativeTheme } = require('electron');
const path = require('path');
const fs = require('fs');
const dotenv = require('dotenv');

// ─── Load SableCore .env ─────────────────────────────────────────────────────
const envPath = path.join(__dirname, '../../.env');
if (fs.existsSync(envPath)) dotenv.config({ path: envPath });

const WEBCHAT_PORT = process.env.WEBCHAT_PORT || '8789';
const WEBCHAT_HOST = process.env.WEBCHAT_HOST || 'localhost';
const WEBCHAT_TOKEN = process.env.WEBCHAT_TOKEN || '';
const IS_DEV = process.env.VITE_DEV === 'true';

// ─── Main window ─────────────────────────────────────────────────────────────
let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 780,
    minHeight: 560,
    frame: false,
    // titleBarStyle must NOT be set when frame:false on macOS,
    // otherwise native traffic-light buttons appear duplicated
    // alongside the custom ones rendered in App.jsx.
    ...(process.platform !== 'darwin' && { titleBarStyle: 'hidden' }),
    transparent: true,
    backgroundColor: '#00000000',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      webviewTag: true,
    },
    icon: process.platform === 'darwin'
      ? path.join(__dirname, '../public/icon.icns')
      : path.join(__dirname, '../public/icon.png'),
  });

  if (IS_DEV) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// ─── IPC Handlers ─────────────────────────────────────────────────────────────

// Return gateway config to renderer
ipcMain.handle('get-config', () => ({
  wsUrl: `ws://${WEBCHAT_HOST}:${WEBCHAT_PORT}`,
  token: WEBCHAT_TOKEN,
}));

// Window controls
ipcMain.on('window-minimize', () => mainWindow?.minimize());
ipcMain.on('window-maximize', () => {
  if (mainWindow?.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow?.maximize();
  }
});
ipcMain.on('window-close', () => mainWindow?.close());

// Open external URL in browser
ipcMain.on('open-external', (_e, url) => shell.openExternal(url));

// Theme
ipcMain.handle('get-theme', () => nativeTheme.shouldUseDarkColors ? 'dark' : 'light');
ipcMain.on('set-theme', (_e, theme) => {
  nativeTheme.themeSource = theme;
});
