#!/usr/bin/env python3
"""Open-Sable — Professional GUI Installer / Manager.

4-page wizard: Welcome → Config → Progress → Done
Management hub: Update / Reinstall / Launch / Uninstall
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

try:
    import urllib.request
except ImportError:
    pass

# ════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════

APP_NAME = "Open-Sable"
APP_VERSION = "1.7.0"
APP_TAGLINE = "Your Autonomous AI Agent — Think, Learn, Act"
REPO_URL = "https://github.com/IdeoaLabs/Open-Sable.git"
REPO_BRANCH = "master"
OLLAMA_WIN_URL = "https://ollama.com/download/OllamaSetup.exe"

IS_WIN = sys.platform == "win32"
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

# On macOS / Linux, bundled .app doesn't inherit the user's shell PATH.
# Ensure common tool directories are on PATH so git, node, brew, etc. are found.
if not IS_WIN:
    _extra_paths = [
        "/opt/homebrew/bin", "/opt/homebrew/sbin",          # Homebrew (Apple Silicon)
        "/usr/local/bin", "/usr/local/sbin",                # Homebrew (Intel) & system
        os.path.expanduser("~/.local/bin"),                 # pip --user, pipx
        "/usr/bin", "/bin", "/usr/sbin", "/sbin",
    ]
    _current = os.environ.get("PATH", "")
    _missing = [p for p in _extra_paths if p not in _current.split(os.pathsep)]
    if _missing:
        os.environ["PATH"] = os.pathsep.join(_missing) + os.pathsep + _current

# Theme
BG_DARK = "#0d1117"
BG_CARD = "#161b22"
BG_INPUT = "#21262d"
FG_TEXT = "#e6edf3"
FG_DIM = "#7d8590"
ACCENT = "#00d4aa"
ACCENT_HOVER = "#00f0c0"
ERROR_C = "#f85149"
WARNING_C = "#d29922"


# ════════════════════════════════════════════════════════════════════
# Asset paths (PyInstaller-aware)
# ════════════════════════════════════════════════════════════════════

def resource_path(relative: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


ASSETS_DIR = resource_path("assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
ICON_ICO = os.path.join(ASSETS_DIR, "icon.ico")
ICON_ICNS = os.path.join(ASSETS_DIR, "icon.icns")
ICON_PNG = LOGO_PATH

# JavaScript bootstrap for the macOS .app bundle
ELECTRON_BOOTSTRAP_JS = r"""'use strict';

const { app, nativeImage } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const DIR = '__INSTALL_DIR__';

process.env.PATH = [
  '/opt/homebrew/bin', '/opt/homebrew/sbin', '/usr/local/bin',
  process.env.PATH,
].join(':');

const envPath = path.join(DIR, '.env');
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, 'utf8').split('\n')) {
    const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
  }
}
process.env.WEBCHAT_PORT = process.env.WEBCHAT_PORT || '8789';
process.env.WEBCHAT_HOST = process.env.WEBCHAT_HOST || 'localhost';
// Prevent start.sh from launching a second Electron window.
// start.sh reads DESKTOP_ENABLED from the .env *file*, not from env vars,
// so we must patch the file itself.
process.env.DESKTOP_ENABLED = 'false';
process.env._SABLE_ELECTRON_APP = '1';
if (fs.existsSync(envPath)) {
  let envText = fs.readFileSync(envPath, 'utf8');
  if (/^DESKTOP_ENABLED=true/m.test(envText)) {
    envText = envText.replace(/^DESKTOP_ENABLED=true/m, 'DESKTOP_ENABLED=false');
    try { fs.writeFileSync(envPath, envText); } catch (_) {}
  }
} 

// Set dock icon to Sable logo
const iconPath = path.join(__dirname, '..', 'opensable.icns');
if (process.platform === 'darwin' && fs.existsSync(iconPath)) {
  app.whenReady().then(() => {
    try { app.dock.setIcon(nativeImage.createFromPath(iconPath)); } catch (_) {}
  });
}

function isBackendRunning() {
  try {
    const pidFiles = fs.readdirSync(DIR).filter(f => /^\.sable-.*\.pid$/.test(f));
    for (const pf of pidFiles) {
      const pid = parseInt(fs.readFileSync(path.join(DIR, pf), 'utf8').trim(), 10);
      if (pid) { try { process.kill(pid, 0); return true; } catch (_) {} }
    }
  } catch (_) {}
  return false;
}

function startBackend() {
  try {
    spawn('bash', ['start.sh', 'start'], {
      cwd: DIR, detached: true, stdio: 'ignore', env: { ...process.env },
    }).unref();
  } catch (_) {}
}

if (!isBackendRunning()) { startBackend(); }

// Watchdog: restart backend if it dies while the app is open
const watchdog = setInterval(() => { if (!isBackendRunning()) startBackend(); }, 15000);

function stopBackend() {
  try {
    const pidFiles = fs.readdirSync(DIR).filter(f => /^\.sable-.*\.pid$/.test(f));
    for (const pf of pidFiles) {
      const pid = parseInt(fs.readFileSync(path.join(DIR, pf), 'utf8').trim(), 10);
      if (pid) {
        try { process.kill(-pid, 'SIGTERM'); } catch (_) {}   // kill process group
        try { process.kill(pid, 'SIGTERM'); } catch (_) {}     // kill process directly
      }
      try { fs.unlinkSync(path.join(DIR, pf)); } catch (_) {}
    }
  } catch (_) {}
  // Also run start.sh stop as a fallback
  try { spawn('bash', ['start.sh', 'stop'], { cwd: DIR, stdio: 'ignore' }); } catch (_) {}
}

// Kill backend and clear watchdog when the app is quitting
app.on('before-quit', () => {
  clearInterval(watchdog);
  stopBackend();
});

// On macOS, quit the app when all windows are closed (not a document editor)
app.on('window-all-closed', () => { app.quit(); });

const desktopDir = path.join(DIR, 'desktop');
const realMain = path.join(desktopDir, 'electron', 'main.cjs');
module.paths.unshift(path.join(desktopDir, 'node_modules'));

if (fs.existsSync(realMain)) {
  process.chdir(DIR);
  require(realMain);
} else {
  const { dialog } = require('electron');
  app.whenReady().then(() => {
    dialog.showErrorBox('Open-Sable Not Found',
      'Could not find the desktop app at:\n' + desktopDir + '\n\nPlease run the installer first.');
    app.quit();
  });
}
"""


# ════════════════════════════════════════════════════════════════════
# Models — only <thinking>-capable
# ════════════════════════════════════════════════════════════════════

MODELS = [
    ("qwen3.5:0.8b", "Qwen 3.5 0.8B  —  fastest, 500 MB"),
    ("qwen3.5:1.5b", "Qwen 3.5 1.5B  —  fast, 1 GB"),
    ("qwen3.5:4b", "Qwen 3.5 4B  —  balanced, 2.5 GB"),
    ("qwen3.5:8b", "Qwen 3.5 8B  —  recommended, 5 GB"),
    ("qwen3.5:14b", "Qwen 3.5 14B  —  powerful, 9 GB"),
    ("qwen3.5:35b", "Qwen 3.5 35B  —  very powerful, 22 GB"),
    ("deepseek-r1:1.5b", "DeepSeek-R1 1.5B  —  fast reasoning, 1 GB"),
    ("deepseek-r1:7b", "DeepSeek-R1 7B  —  strong reasoning, 4.5 GB"),
    ("deepseek-r1:8b", "DeepSeek-R1 8B  —  strong reasoning, 5 GB"),
    ("deepseek-r1:14b", "DeepSeek-R1 14B  —  deep reasoning, 9 GB"),
    ("deepseek-r1:32b", "DeepSeek-R1 32B  —  expert reasoning, 20 GB"),
    ("deepseek-r1:70b", "DeepSeek-R1 70B  —  maximum reasoning, 43 GB"),
    ("qwen3:0.6b", "Qwen 3 0.6B  —  ultra-light, 400 MB"),
    ("qwen3:1.7b", "Qwen 3 1.7B  —  light, 1 GB"),
    ("qwen3:4b", "Qwen 3 4B  —  balanced, 2.5 GB"),
    ("qwen3:8b", "Qwen 3 8B  —  recommended, 5 GB"),
    ("qwen3:14b", "Qwen 3 14B  —  powerful, 9 GB"),
    ("qwen3:32b", "Qwen 3 32B  —  very powerful, 20 GB"),
]

INSTALL_SLIDES = [
    ("What is Open-Sable?",
     "An autonomous AI agent that can think, learn, and act.\n"
     "It runs locally on your machine — your data stays private."),
    ("Powered by <thinking>",
     "Open-Sable uses models with native reasoning:\n"
     "Qwen 3.5, DeepSeek-R1, and Qwen 3.\n"
     "The AI thinks step-by-step before responding."),
    ("Modular & Extensible",
     "Add skills, connect to APIs, build agents.\n"
     "Dashboard for monitoring, WebChat for interaction."),
    ("100% Local & Private",
     "No cloud needed. Ollama runs models on your hardware.\n"
     "Your conversations and data never leave your machine."),
]


# ════════════════════════════════════════════════════════════════════
# macOS-safe button (tk.Label-based, colors always work)
# ════════════════════════════════════════════════════════════════════

def make_button(parent, text, command, bg=BG_INPUT, fg=FG_TEXT,
                hover_bg=BG_CARD, hover_fg=None, font=("Segoe UI", 10),
                padx=16, pady=8, cursor="hand2", **kw):
    """Create a label that looks and acts like a button — colors work on macOS."""
    if hover_fg is None:
        hover_fg = fg
    lbl = tk.Label(parent, text=text, font=font, bg=bg, fg=fg,
                   padx=padx, pady=pady, cursor=cursor, **kw)
    lbl.bind("<Enter>", lambda e: lbl.configure(bg=hover_bg, fg=hover_fg))
    lbl.bind("<Leave>", lambda e: lbl.configure(bg=bg, fg=fg))
    lbl.bind("<Button-1>", lambda e: command())
    return lbl


# ════════════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════════════

def _safe_cwd(path: str) -> str:
    """Return *path* if it exists, otherwise fall back to the user's home dir.
    Prevents WinError 267 / FileNotFoundError when cwd doesn't exist yet."""
    if path and os.path.isdir(path):
        return path
    return os.path.expanduser("~")


def _refresh_windows_path():
    """Re-read PATH from the Windows registry so newly-installed tools are found.
    No-op on non-Windows platforms."""
    if not IS_WIN:
        return
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Environment]::GetEnvironmentVariable('Path','Machine')+';'+"
             "[Environment]::GetEnvironmentVariable('Path','User')"],
            capture_output=True, text=True, timeout=5, creationflags=_NO_WINDOW)
        if r.returncode == 0 and r.stdout.strip():
            os.environ["PATH"] = r.stdout.strip()
    except Exception:
        pass


def _download_with_timeout(url: str, dest: str, timeout: int = 120):
    """Download *url* to *dest* with a connection + read timeout."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "OpenSable-Installer"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
    except Exception as e:
        # Retry once
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(resp, f)
        except Exception:
            raise Exception(f"Download failed: {url}\n{e}")


def find_python() -> Tuple[Optional[List[str]], Optional[str]]:
    for cmd in [["python3.13"], ["python3.12"], ["python3"], ["python"]]:
        try:
            r = subprocess.run(cmd + ["--version"], capture_output=True, text=True, timeout=5,
                               creationflags=_NO_WINDOW)
            if r.returncode == 0:
                ver = r.stdout.strip().split()[-1]
                major, minor = map(int, ver.split(".")[:2])
                if major >= 3 and minor >= 11:
                    return cmd, ver
        except Exception:
            continue
    return None, None


def find_git() -> Optional[str]:
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5,
                           creationflags=_NO_WINDOW)
        return r.stdout.strip().split()[-1] if r.returncode == 0 else None
    except Exception:
        return None


def find_node() -> Optional[str]:
    try:
        r = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5,
                           creationflags=_NO_WINDOW)
        if r.returncode == 0:
            ver = r.stdout.strip().lstrip("v")
            if int(ver.split(".")[0]) >= 20:
                return ver
    except Exception:
        pass
    return None


def find_npm() -> Optional[str]:
    """Find npm, including common Windows paths when PATH hasn't refreshed."""
    try:
        r = subprocess.run(["npm", "--version"], capture_output=True, text=True, timeout=5,
                           creationflags=_NO_WINDOW)
        if r.returncode == 0:
            return shutil.which("npm") or "npm"
    except Exception:
        pass
    if IS_WIN:
        candidates = [
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "nodejs", "npm.cmd"),
            os.path.join(os.environ.get("APPDATA", ""), "npm", "npm.cmd"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                try:
                    r = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=5,
                                       creationflags=_NO_WINDOW)
                    if r.returncode == 0:
                        return p
                except Exception:
                    pass
        # Refresh PATH from registry
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[Environment]::GetEnvironmentVariable('Path','Machine')+';'+[Environment]::GetEnvironmentVariable('Path','User')"],
                capture_output=True, text=True, timeout=5, creationflags=_NO_WINDOW)
            if r.returncode == 0 and r.stdout.strip():
                os.environ["PATH"] = r.stdout.strip()
                npm_path = shutil.which("npm")
                if npm_path:
                    return npm_path
        except Exception:
            pass
    return None


def find_ollama() -> Optional[str]:
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5,
                           creationflags=_NO_WINDOW)
        if r.returncode == 0:
            return r.stdout.strip().split()[-1]
    except Exception:
        pass
    return None


def ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def get_local_version(install_dir: str) -> str:
    pyproject = os.path.join(install_dir, "pyproject.toml")
    if os.path.isfile(pyproject):
        with open(pyproject) as f:
            for line in f:
                if line.strip().startswith("version"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    init = os.path.join(install_dir, "opensable", "__init__.py")
    if os.path.isfile(init):
        with open(init) as f:
            for line in f:
                if "__version__" in line:
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "unknown"


def get_remote_version() -> Optional[str]:
    try:
        url = f"https://raw.githubusercontent.com/IdeoaLabs/Open-Sable/{REPO_BRANCH}/pyproject.toml"
        r = urllib.request.urlopen(url, timeout=5)
        for line in r.read().decode().splitlines():
            if line.strip().startswith("version"):
                return line.split("=")[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def default_install_dir() -> str:
    """Find existing install or return default."""
    # Check if we're running from within a repo (only when not frozen/bundled)
    if not getattr(sys, 'frozen', False):
        check = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            if (os.path.isfile(os.path.join(check, "pyproject.toml"))
                    and os.path.isdir(os.path.join(check, "opensable"))
                    and os.path.isfile(os.path.join(check, ".installed"))):
                return check
            parent = os.path.dirname(check)
            if parent == check:
                break
            check = parent
    # Check common locations
    for candidate in [
        os.path.expanduser("~/opensable"),
        os.path.expanduser("~/OpenSable"),
        os.path.expanduser("~/Open-Sable"),
        os.path.expanduser("~/SableCore_"),
    ]:
        if (os.path.isdir(candidate)
                and os.path.isfile(os.path.join(candidate, "pyproject.toml"))
                and os.path.isdir(os.path.join(candidate, "opensable"))
                and os.path.isfile(os.path.join(candidate, ".installed"))):
            return candidate
    return os.path.join(os.path.expanduser("~"), "opensable")


# ════════════════════════════════════════════════════════════════════
# Update Engine
# ════════════════════════════════════════════════════════════════════

class UpdateEngine:
    PROTECTED_PATHS = [".env", "agents", "data", "episodes", "logs", "models"]

    def __init__(self, install_dir, log_cb, progress_cb, done_cb):
        self.install_dir = install_dir
        self.log = log_cb
        self.progress = progress_cb
        self.done = done_cb
        self._cancelled = False

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self):
        self._cancelled = True

    def _exec(self, cmd, cwd=None, check=True, shell=False):
        self.log(f"  $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", "dim")
        effective_cwd = _safe_cwd(cwd or self.install_dir)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=effective_cwd, encoding='utf-8', errors='replace', shell=shell,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                creationflags=_NO_WINDOW,
            )
            output = []
            for line in iter(proc.stdout.readline, ""):
                if self._cancelled:
                    proc.kill()
                    raise Exception("Cancelled")
                s = line.rstrip()
                if s:
                    self.log(f"    {s}", "dim")
                    output.append(s)
            proc.wait()
            if check and proc.returncode != 0:
                raise Exception(f"Command failed (exit {proc.returncode})")
            return "\n".join(output)
        except FileNotFoundError:
            raise Exception(f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}")

    def _run(self):
        try:
            steps = [
                ("Backing up config", self._backup),
                ("Pulling latest code", self._git_pull),
                ("Updating dependencies", self._update_deps),
                ("Rebuilding dashboard", self._rebuild_dashboard),
                ("Verifying", self._verify),
            ]
            total = len(steps)
            for i, (name, func) in enumerate(steps):
                if self._cancelled:
                    self.log("\n⚠ Update cancelled.", "warning")
                    self.done(False, "Cancelled")
                    return
                self.log(f"\n━━━ Step {i+1}/{total}: {name}", "step")
                self.progress((i / total) * 100, name)
                func()

            self.progress(100, "Update complete!")
            meta = os.path.join(self.install_dir, ".sable-update.json")
            with open(meta, "w") as f:
                json.dump({
                    "last_update": datetime.now().isoformat(),
                    "version": get_local_version(self.install_dir),
                    "method": "gui-update",
                }, f, indent=2)

            self.log("\n✔ Update complete!", "success")
            self.done(True, None)
        except Exception as e:
            self.log(f"\n✘ Update error: {e}", "error")
            self.done(False, str(e))

    def _backup(self):
        for name in self.PROTECTED_PATHS:
            src = os.path.join(self.install_dir, name)
            if os.path.exists(src):
                self.log(f"  Protected: {name}", "dim")
        self.log("  ✔ Config backed up", "ok")

    def _git_pull(self):
        git_dir = os.path.join(self.install_dir, ".git")
        if not os.path.isdir(git_dir):
            self.log("  Not a git repo — skipping", "warning")
            return
        self._exec(["git", "stash", "--include-untracked"], check=False)
        self._exec(["git", "pull", "--rebase", "origin", REPO_BRANCH], check=False)
        self._exec(["git", "stash", "pop"], check=False)
        self.log("  ✔ Code updated", "ok")

    def _update_deps(self):
        venv_pip = os.path.join(self.install_dir, "venv", "Scripts" if IS_WIN else "bin",
                                "pip.exe" if IS_WIN else "pip")
        if not os.path.isfile(venv_pip):
            venv_pip = os.path.join(self.install_dir, ".venv", "Scripts" if IS_WIN else "bin",
                                    "pip.exe" if IS_WIN else "pip")
        if os.path.isfile(venv_pip):
            self._exec([venv_pip, "install", "-e", ".[core]", "-q"], check=False)
            req = os.path.join(self.install_dir, "requirements.txt")
            if os.path.isfile(req):
                self._exec([venv_pip, "install", "-r", req, "-q"], check=False)
            self.log("  ✔ Dependencies updated", "ok")
        else:
            self.log("  ⚠ No venv found — skip deps", "warning")

    def _rebuild_dashboard(self):
        dash = os.path.join(self.install_dir, "dashboard")
        if not os.path.isdir(dash) or not find_node():
            self.log("  ⚠ Skipping dashboard rebuild", "warning")
            return
        npm = find_npm() or "npm"
        self._exec([npm, "install", "--legacy-peer-deps"], cwd=dash, check=False)
        self._exec([npm, "run", "build"], cwd=dash, check=False)
        self.log("  ✔ Dashboard rebuilt", "ok")

    def _verify(self):
        ver = get_local_version(self.install_dir)
        self.log(f"  ✔ Version: {ver}", "ok")


# ════════════════════════════════════════════════════════════════════
# Reinstall Engine
# ════════════════════════════════════════════════════════════════════

class ReinstallEngine:
    """Full rebuild: venv, deps, npm, dashboard — preserves user data."""

    PROTECTED_PATHS = UpdateEngine.PROTECTED_PATHS

    def __init__(self, config, log_cb, progress_cb, done_cb):
        self.config = config
        self.log = log_cb
        self.progress = progress_cb
        self.done = done_cb
        self._cancelled = False

    @property
    def install_dir(self):
        return self.config["install_dir"]

    @property
    def venv_python(self):
        if IS_WIN:
            return os.path.join(self.install_dir, "venv", "Scripts", "python.exe")
        return os.path.join(self.install_dir, "venv", "bin", "python")

    @property
    def venv_pip(self):
        if IS_WIN:
            return os.path.join(self.install_dir, "venv", "Scripts", "pip.exe")
        return os.path.join(self.install_dir, "venv", "bin", "pip")

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self):
        self._cancelled = True

    def _exec(self, cmd, cwd=None, check=True, shell=False):
        self.log(f"  $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", "dim")
        effective_cwd = _safe_cwd(cwd or self.install_dir)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=effective_cwd, encoding='utf-8', errors='replace', shell=shell,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                creationflags=_NO_WINDOW,
            )
            output = []
            for line in iter(proc.stdout.readline, ""):
                if self._cancelled:
                    proc.kill()
                    raise Exception("Cancelled")
                s = line.rstrip()
                if s:
                    self.log(f"    {s}", "dim")
                    output.append(s)
            proc.wait()
            if check and proc.returncode != 0:
                raise Exception(f"Command failed (exit {proc.returncode})")
            return "\n".join(output)
        except FileNotFoundError:
            raise Exception(f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}")

    def _run(self):
        try:
            steps = [
                ("Backing up user data", self._backup),
                ("Pulling latest code", self._git_pull),
                ("Rebuilding Python environment", self._rebuild_venv),
                ("Installing Python dependencies", self._install_deps),
                ("Installing Ollama", self._check_ollama),
                ("Installing Node.js", self._check_node),
                ("Rebuilding frontends", self._rebuild_node),
                ("Configuring environment", self._init_environment),
                ("Clearing old sessions", self._clear_sessions),
                ("Checking AI model", self._check_model),
                ("Verifying installation", self._verify),
            ]
            total = len(steps)
            for i, (name, func) in enumerate(steps):
                if self._cancelled:
                    self.log("\n⚠ Reinstall cancelled.", "warning")
                    self.done(False, "Cancelled")
                    return
                self.log(f"\n━━━ Step {i+1}/{total}: {name}", "step")
                self.progress((i / total) * 100, name)
                func()

            self.progress(100, "Reinstall complete!")
            meta = os.path.join(self.install_dir, ".sable-update.json")
            with open(meta, "w") as f:
                json.dump({
                    "last_update": datetime.now().isoformat(),
                    "version": get_local_version(self.install_dir),
                    "method": "gui-reinstall",
                }, f, indent=2)
            self.log("\n✔ Reinstall complete! Everything rebuilt.", "success")
            # Refresh install marker
            try:
                Path(os.path.join(self.install_dir, ".installed")).write_text(
                    f"installed={datetime.now().isoformat()}\n")
            except OSError:
                pass
            self.done(True, None)
        except Exception as e:
            self.log(f"\n✘ Reinstall error: {e}", "error")
            self.log("  Your data and config are safe.", "dim")
            self.done(False, str(e))

    def _backup(self):
        backup_dir = os.path.join(self.install_dir, ".reinstall-backup")
        os.makedirs(backup_dir, exist_ok=True)
        for name in self.PROTECTED_PATHS:
            src = os.path.join(self.install_dir, name)
            dst = os.path.join(backup_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                self.log(f"  Backed up: {name}", "dim")
            elif os.path.isdir(src):
                self.log(f"  Protected: {name}/ (will not be touched)", "dim")
        self.log("  ✔ User data backed up", "ok")

    def _git_pull(self):
        git_dir = os.path.join(self.install_dir, ".git")
        if not os.path.isdir(git_dir):
            self.log("  Not a git repo — skipping code update", "warning")
            return
        self._exec(["git", "stash", "--include-untracked"], check=False)
        self._exec(["git", "pull", "--rebase", "origin", REPO_BRANCH], check=False)
        self._exec(["git", "stash", "pop"], check=False)
        backup_dir = os.path.join(self.install_dir, ".reinstall-backup")
        if os.path.isdir(backup_dir):
            for name in self.PROTECTED_PATHS:
                bf = os.path.join(backup_dir, name)
                if os.path.isfile(bf):
                    shutil.copy2(bf, os.path.join(self.install_dir, name))
        self.log("  ✔ Code updated to latest", "ok")

    def _rebuild_venv(self):
        venv_dir = os.path.join(self.install_dir, "venv")
        if os.path.isdir(venv_dir):
            self.log("  Removing old Python environment...", "dim")
            shutil.rmtree(venv_dir, ignore_errors=True)
        py_cmd, py_ver = find_python()
        if not py_cmd:
            raise Exception("Python 3.11+ required. Install from python.org")
        self.log(f"  Creating new venv with Python {py_ver}...", "dim")
        self._exec(py_cmd + ["-m", "venv", venv_dir])
        self.log("  ✔ Python environment rebuilt", "ok")

    def _install_deps(self):
        pip = [self.venv_python, "-m", "pip"]
        self._exec(pip + ["install", "--upgrade", "pip", "setuptools", "wheel", "-q"], check=False)
        self._exec(pip + ["install", "-e", ".[core]", "-q"],
                    cwd=self.install_dir, check=False)
        req = os.path.join(self.install_dir, "requirements.txt")
        if os.path.isfile(req):
            self._exec(pip + ["install", "-r", req, "-q"], cwd=self.install_dir, check=False)
        self.log("  ✔ All Python dependencies installed", "ok")

    def _check_ollama(self):
        if find_ollama():
            self.log("  ✔ Ollama installed", "ok")
            return
        self.log("  Installing Ollama...", "dim")
        if IS_WIN:
            dl = os.path.join(tempfile.gettempdir(), "OllamaSetup.exe")
            _download_with_timeout(OLLAMA_WIN_URL, dl)
            self._exec([dl, "/VERYSILENT", "/NORESTART"], check=False)
        else:
            self._exec("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
        if find_ollama():
            self.log("  ✔ Ollama installed", "ok")
        else:
            self.log("  ⚠ Install Ollama manually: ollama.com", "warning")

    def _check_node(self):
        if find_node():
            self.log("  ✔ Node.js 20+ found", "ok")
            return
        self.log("  Installing Node.js...", "dim")
        if IS_LINUX:
            if shutil.which("apt-get"):
                self._exec("curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -",
                            shell=True, check=False)
                self._exec(["sudo", "apt-get", "install", "-y", "nodejs"])
            elif shutil.which("dnf"):
                self._exec(["sudo", "dnf", "install", "-y", "nodejs", "npm"])
        elif IS_MAC and shutil.which("brew"):
            self._exec(["brew", "install", "node"])
        elif IS_WIN and shutil.which("winget"):
            self._exec(["winget", "upgrade", "--id", "OpenJS.NodeJS.LTS",
                        "--accept-source-agreements", "--accept-package-agreements", "-e"])
        if find_node():
            self.log("  ✔ Node.js installed", "ok")
        else:
            self.log("  ⚠ Install Node.js 20+ manually: nodejs.org", "warning")

    def _rebuild_node(self):
        if not find_node():
            self.log("  ⚠ Node.js not available — skipping", "warning")
            return
        npm = find_npm() or "npm"
        for project in ["dashboard", "desktop", "aggr"]:
            pkg = os.path.join(self.install_dir, project, "package.json")
            if not os.path.isfile(pkg):
                continue
            self.log(f"  Rebuilding {project}...", "dim")
            proj_dir = os.path.join(self.install_dir, project)
            nm = os.path.join(proj_dir, "node_modules")
            if os.path.isdir(nm):
                shutil.rmtree(nm, ignore_errors=True)
            self._exec([npm, "install", "--legacy-peer-deps"], cwd=proj_dir, check=False)
            if project == "desktop":
                # Desktop needs both electron AND a vite build (dist/index.html)
                electron_bin = os.path.join(nm, ".bin", "electron")
                if not os.path.isfile(electron_bin):
                    self.log(f"  ⚠ {project} — electron binary missing", "warning")
            result = self._exec([npm, "run", "build"], cwd=proj_dir, check=False)
            # Retry on build failure: wipe node_modules and reinstall
            dist_check = os.path.join(proj_dir, "dist", "index.html")
            if not os.path.isfile(dist_check):
                self.log(f"  ⚠ {project} build failed — retrying...", "warning")
                if os.path.isdir(nm):
                    shutil.rmtree(nm, ignore_errors=True)
                self._exec([npm, "install", "--legacy-peer-deps"], cwd=proj_dir, check=False)
                self._exec([npm, "run", "build"], cwd=proj_dir, check=False)
            if os.path.isfile(dist_check):
                self.log(f"  ✔ {project} rebuilt", "ok")
            else:
                self.log(f"  ⚠ {project} build incomplete — dist/index.html missing", "warning")

    def _init_environment(self):
        """Ensure .env, agent profile, and directories exist (preserve existing)."""
        # Required directories
        for dirname in ["data", "logs", "config", "models", "episodes"]:
            os.makedirs(os.path.join(self.install_dir, dirname), exist_ok=True)
        self.log("  ✔ Directories verified", "ok")

        # .env
        env_file = os.path.join(self.install_dir, ".env")
        if not os.path.isfile(env_file):
            env_example = os.path.join(self.install_dir, ".env.example")
            model = self.config.get("model", "qwen3.5:0.8b")
            if os.path.isfile(env_example):
                with open(env_example, "r") as f:
                    content = f.read()
                content = content.replace("DEFAULT_MODEL=llama3.1:8b", f"DEFAULT_MODEL={model}")
                content = content.replace("GATEWAY_ENABLED=false", "GATEWAY_ENABLED=true")
                # Disable auto-select so the user's chosen model sticks
                content = content.replace("AUTO_SELECT_MODEL=true", "AUTO_SELECT_MODEL=false")
                with open(env_file, "w") as f:
                    f.write(content)
            else:
                with open(env_file, "w") as f:
                    f.write(f"DEFAULT_MODEL={model}\nAUTO_SELECT_MODEL=false\nGATEWAY_ENABLED=true\nWEBCHAT_PORT=8789\n")
            self.log("  ✔ .env created", "ok")
        else:
            self.log("  ✔ .env preserved", "ok")

        # Agent profile
        agents_dir = os.path.join(self.install_dir, "agents", "sable")
        template_dir = os.path.join(self.install_dir, "agents", "_template")
        os.makedirs(os.path.join(agents_dir, "data"), exist_ok=True)
        for fname in ["profile.env", "soul.md", "tools.json"]:
            dest = os.path.join(agents_dir, fname)
            if not os.path.isfile(dest):
                src = os.path.join(template_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dest)
                    if fname == "profile.env":
                        model = self.config.get("model", "qwen3.5:0.8b")
                        with open(dest, "r") as f:
                            c = f.read()
                        c = c.replace("DEFAULT_MODEL=qwen3.5:0.8b", f"DEFAULT_MODEL={model}")
                        c = c.replace("AGENT_NAME=MyAgent", "AGENT_NAME=Sable")
                        c = c.replace("WEBCHAT_PORT=8792", "WEBCHAT_PORT=8789")
                        with open(dest, "w") as f:
                            f.write(c)
        self.log("  ✔ Agent profile verified (agents/sable/)", "ok")

    def _check_model(self):
        model = self.config.get("model", "qwen3.5:0.8b")
        if not ollama_running():
            if find_ollama():
                self.log("  Starting Ollama...", "dim")
                if IS_WIN:
                    subprocess.Popen(["ollama", "serve"], creationflags=0x08000000)
                else:
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                for _ in range(10):
                    time.sleep(1)
                    if ollama_running():
                        break
        if ollama_running():
            try:
                r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10,
                                   creationflags=_NO_WINDOW)
                if model.split(":")[0] in r.stdout:
                    self.log(f"  ✔ Model {model} available", "ok")
                    return
            except Exception:
                pass
            self.log(f"  Model {model} not found — pulling...", "dim")
            self._exec(["ollama", "pull", model], check=False)
            self.log(f"  ✔ Model {model} ready", "ok")
        else:
            self.log(f"  ⚠ Ollama not running — run: ollama pull {model}", "warning")

    def _clear_sessions(self):
        """Remove old chat sessions from ~/.opensable/ so reinstall feels fresh."""
        dot_opensable = os.path.join(os.path.expanduser("~"), ".opensable")
        if not os.path.isdir(dot_opensable):
            self.log("  No previous sessions found", "dim")
            return
        cleared = 0
        for root, dirs, files in os.walk(dot_opensable):
            if os.path.basename(root) == "sessions":
                for f in files:
                    if f.endswith(".json"):
                        os.remove(os.path.join(root, f))
                        cleared += 1
        if cleared:
            self.log(f"  ✔ Cleared {cleared} old chat sessions", "ok")
        else:
            self.log("  No previous sessions found", "dim")

    def _verify(self):
        if os.path.isfile(self.venv_python):
            self.log("  ✔ Python environment OK", "ok")
        else:
            self.log("  ✘ Python environment missing", "error")
            return
        try:
            r = subprocess.run(
                [self.venv_python, "-c", "import opensable; print(opensable.__version__)"],
                capture_output=True, text=True, timeout=15, cwd=self.install_dir
            )
            if r.returncode == 0:
                self.log(f"  ✔ opensable v{r.stdout.strip()} verified", "ok")
        except Exception:
            self.log("  ⚠ Could not verify import", "warning")
        dist_index = os.path.join(self.install_dir, "dashboard", "dist", "index.html")
        if os.path.isfile(dist_index):
            self.log("  ✔ Dashboard built", "ok")
        else:
            self.log("  ⚠ Dashboard not built — attempting repair...", "warning")
            dash = os.path.join(self.install_dir, "dashboard")
            if os.path.isfile(os.path.join(dash, "package.json")) and find_node():
                self._exec(["npm", "install", "--legacy-peer-deps"], cwd=dash, check=False)
                self._exec(["npm", "run", "build"], cwd=dash, check=False)
                if os.path.isfile(dist_index):
                    self.log("  ✔ Dashboard repaired", "ok")
                else:
                    self.log("  ⚠ Dashboard repair failed — build manually", "warning")
        # Desktop (Electron)
        desktop_dir = os.path.join(self.install_dir, "desktop")
        electron_bin = os.path.join(desktop_dir, "node_modules", ".bin", "electron")
        desktop_dist = os.path.join(desktop_dir, "dist", "index.html")
        if os.path.isfile(electron_bin):
            self.log("  ✔ Desktop app ready (Electron)", "ok")
        else:
            self.log("  ⚠ Desktop app not installed", "warning")
        if os.path.isfile(desktop_dist):
            self.log("  ✔ Desktop UI built", "ok")
        else:
            self.log("  ⚠ Desktop UI not built — attempting repair...", "warning")
            if os.path.isfile(os.path.join(desktop_dir, "package.json")) and find_node():
                self._exec(["npm", "install", "--legacy-peer-deps"], cwd=desktop_dir, check=False)
                self._exec(["npm", "run", "build"], cwd=desktop_dir, check=False)
                if os.path.isfile(desktop_dist):
                    self.log("  ✔ Desktop UI repaired", "ok")
                else:
                    self.log("  ⚠ Desktop UI repair failed — build manually", "warning")
        # .env & agent profile
        if os.path.isfile(os.path.join(self.install_dir, ".env")):
            self.log("  ✔ Configuration (.env)", "ok")
        if os.path.isfile(os.path.join(self.install_dir, "agents", "sable", "profile.env")):
            self.log("  ✔ Agent profile (agents/sable/)", "ok")
        # Sandbox
        if os.path.isfile(os.path.join(self.install_dir, "opensable", "core", "sandbox_runner.py")):
            self.log("  ✔ Sandbox engine present", "ok")
        if ollama_running():
            self.log("  ✔ Ollama API OK", "ok")
        else:
            self.log("  ⚠ Ollama not running", "warning")
        self.log("  ✔ All systems checked", "ok")


# ════════════════════════════════════════════════════════════════════
# Uninstall Engine
# ════════════════════════════════════════════════════════════════════

class UninstallEngine:
    """Remove Open-Sable components based on user selection.
    Accepts options dict: venv, node_modules, generated, shortcuts, services, ollama, user_data."""

    USER_DATA = ["agents", "data", ".env", "logs", "episodes", "models"]

    def __init__(self, install_dir, options, log_cb, progress_cb, done_cb):
        self.install_dir = install_dir
        self.options = options if isinstance(options, dict) else {"venv": True, "node_modules": True, "generated": True, "shortcuts": True, "services": True, "user_data": not options}
        self.log = log_cb
        self.progress = progress_cb
        self.done = done_cb
        self._cancelled = False

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self):
        self._cancelled = True

    def _run(self):
        try:
            steps = []
            if self.options.get("services"):
                steps.append(("Removing auto-update services", self._remove_services))
            if self.options.get("shortcuts"):
                steps.append(("Removing desktop shortcuts", self._remove_shortcuts))
            if self.options.get("venv"):
                steps.append(("Removing Python environment", self._remove_venv))
            if self.options.get("node_modules"):
                steps.append(("Removing Node modules", self._remove_node_modules))
            if self.options.get("generated"):
                steps.append(("Cleaning generated files", self._remove_generated))
            if self.options.get("ollama"):
                steps.append(("Uninstalling Ollama", self._remove_ollama))
            if self.options.get("user_data"):
                steps.append(("Removing user data", self._remove_user_data))

            total = len(steps)
            for i, (name, func) in enumerate(steps):
                if self._cancelled:
                    self.log("\n⚠ Uninstall cancelled.", "warning")
                    self.done(False, "Cancelled")
                    return
                self.log(f"\n━━━ Step {i+1}/{total}: {name}", "step")
                self.progress((i / total) * 100, name)
                func()

            self.progress(100, "Uninstall complete!")
            self.log("\n✔ Selected components have been removed.", "success")
            kept = [k.replace("_", " ") for k, v in self.options.items() if not v]
            if kept:
                self.log(f"  Kept: {', '.join(kept)}", "dim")
            self.log(f"  Location: {self.install_dir}", "dim")
            # Remove install marker so GUI shows fresh-install page next time
            marker = os.path.join(self.install_dir, ".installed")
            if os.path.isfile(marker):
                try:
                    os.remove(marker)
                except OSError:
                    pass
            self.done(True, None)
        except Exception as e:
            self.log(f"\n✘ Uninstall error: {e}", "error")
            self.done(False, str(e))

    def _remove_services(self):
        if IS_LINUX:
            # Stop agent service + update timer
            for svc in ["opensable", "opensable-update.timer"]:
                try:
                    subprocess.run(["systemctl", "--user", "stop", svc],
                                   capture_output=True, timeout=10)
                    subprocess.run(["systemctl", "--user", "disable", svc],
                                   capture_output=True, timeout=10)
                except Exception:
                    pass
            self.log("  ✔ Stopped systemd services", "ok")
            user_dir = os.path.expanduser("~/.config/systemd/user")
            for name in ["opensable.service", "opensable-update.service", "opensable-update.timer"]:
                p = os.path.join(user_dir, name)
                if os.path.isfile(p):
                    os.remove(p)
                    self.log(f"  ✔ Removed {name}", "ok")
            try:
                subprocess.run(["systemctl", "--user", "daemon-reload"],
                               capture_output=True, timeout=10)
            except Exception:
                pass
        elif IS_MAC:
            for plist_name in ["com.ideoalabs.opensable-update.plist",
                               "com.ideoalabs.opensable.plist"]:
                plist = os.path.expanduser(f"~/Library/LaunchAgents/{plist_name}")
                if os.path.isfile(plist):
                    try:
                        subprocess.run(["launchctl", "unload", plist], capture_output=True, timeout=10)
                    except Exception:
                        pass
                    os.remove(plist)
                    self.log(f"  ✔ Removed {plist_name}", "ok")
        elif IS_WIN:
            try:
                subprocess.run(["schtasks", "/delete", "/tn", "OpenSable-Update", "/f"],
                               capture_output=True, timeout=15, creationflags=_NO_WINDOW)
                self.log("  ✔ Removed scheduled task", "ok")
            except Exception:
                pass

    def _remove_shortcuts(self):
        removed = 0
        if IS_WIN:
            for loc in [
                os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows",
                             "Start Menu", "Programs", "Open-Sable.lnk"),
                os.path.join(os.path.expanduser("~"), "Desktop", "Open-Sable.lnk"),
            ]:
                if os.path.isfile(loc):
                    os.remove(loc)
                    self.log(f"  ✔ Removed {os.path.basename(loc)}", "ok")
                    removed += 1
        elif IS_MAC:
            app = os.path.expanduser("~/Applications/Open-Sable.app")
            if os.path.isdir(app):
                shutil.rmtree(app, ignore_errors=True)
                self.log("  ✔ Removed Open-Sable.app", "ok")
                removed += 1
        else:
            desktop_file = os.path.expanduser("~/.local/share/applications/opensable.desktop")
            if os.path.isfile(desktop_file):
                os.remove(desktop_file)
                self.log("  ✔ Removed .desktop entry", "ok")
                removed += 1
            cli_link = os.path.expanduser("~/.local/bin/opensable")
            if os.path.isfile(cli_link):
                os.remove(cli_link)
                self.log("  ✔ Removed CLI link", "ok")
                removed += 1
        if removed == 0:
            self.log("  No shortcuts found", "dim")

    def _remove_venv(self):
        for name in ["venv", ".venv"]:
            venv_dir = os.path.join(self.install_dir, name)
            if os.path.isdir(venv_dir):
                self.log(f"  Removing {name}/...", "dim")
                shutil.rmtree(venv_dir, ignore_errors=True)
                self.log(f"  ✔ Removed {name}/", "ok")
                return
        self.log("  No Python venv found", "dim")

    def _remove_node_modules(self):
        for project in ["dashboard", "sable_dev", "aggr", "desktop", "mobile", "website"]:
            nm = os.path.join(self.install_dir, project, "node_modules")
            if os.path.isdir(nm):
                self.log(f"  Removing {project}/node_modules/...", "dim")
                shutil.rmtree(nm, ignore_errors=True)
                self.log(f"  ✔ Removed {project}/node_modules/", "ok")

    def _remove_generated(self):
        for ext in ["", ".bat"]:
            p = os.path.join(self.install_dir, f"opensable-update{ext}")
            if os.path.isfile(p):
                os.remove(p)
                self.log(f"  ✔ Removed opensable-update{ext}", "ok")
        for name in ["opensable.bat", "opensable.ico", "opensable.png"]:
            p = os.path.join(self.install_dir, name)
            if os.path.isfile(p):
                os.remove(p)
        dist = os.path.join(self.install_dir, "dashboard", "dist")
        if os.path.isdir(dist):
            shutil.rmtree(dist, ignore_errors=True)
            self.log("  ✔ Removed dashboard/dist/", "ok")
        for name in [".sable-update.json", ".reinstall-backup"]:
            p = os.path.join(self.install_dir, name)
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        for root, dirs, _ in os.walk(self.install_dir):
            for d in dirs:
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        self.log("  ✔ Cleaned generated files", "ok")

    def _remove_install_dir(self):
        if os.path.isdir(self.install_dir):
            self.log(f"  Removing {self.install_dir}...", "dim")
            shutil.rmtree(self.install_dir, ignore_errors=True)
            self.log("  ✔ Install directory removed", "ok")

    def _remove_ollama(self):
        if not find_ollama():
            self.log("  Ollama not found — skipping", "dim")
            return
        self.log("  Removing Ollama...", "dim")
        if IS_LINUX:
            try:
                subprocess.run(["sudo", "rm", "-f", "/usr/local/bin/ollama"],
                               capture_output=True, timeout=10)
                subprocess.run(["sudo", "systemctl", "stop", "ollama"],
                               capture_output=True, timeout=10)
                subprocess.run(["sudo", "systemctl", "disable", "ollama"],
                               capture_output=True, timeout=10)
                subprocess.run(["sudo", "rm", "-f", "/etc/systemd/system/ollama.service"],
                               capture_output=True, timeout=10)
                self.log("  ✔ Ollama removed", "ok")
            except Exception as e:
                self.log(f"  ⚠ Could not fully remove Ollama: {e}", "warning")
        elif IS_MAC:
            try:
                subprocess.run(["brew", "uninstall", "ollama"], capture_output=True, timeout=30)
                self.log("  ✔ Ollama removed via brew", "ok")
            except Exception:
                self.log("  ⚠ Remove Ollama manually from Applications", "warning")
        elif IS_WIN:
            try:
                subprocess.run(["winget", "uninstall", "Ollama.Ollama"],
                               capture_output=True, timeout=30)
                self.log("  ✔ Ollama uninstalled", "ok")
            except Exception:
                self.log("  ⚠ Remove Ollama from Add/Remove Programs", "warning")

    def _remove_user_data(self):
        removed = 0
        for name in self.USER_DATA:
            p = os.path.join(self.install_dir, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
                self.log(f"  ✔ Removed {name}/", "ok")
                removed += 1
            elif os.path.isfile(p):
                os.remove(p)
                self.log(f"  ✔ Removed {name}", "ok")
                removed += 1
        # Also remove ~/.opensable/ (sessions, caches, runtime state)
        dot_opensable = os.path.join(os.path.expanduser("~"), ".opensable")
        if os.path.isdir(dot_opensable):
            shutil.rmtree(dot_opensable, ignore_errors=True)
            self.log("  ✔ Removed ~/.opensable/ (sessions & runtime state)", "ok")
            removed += 1
        if removed == 0:
            self.log("  No user data found", "dim")


# ════════════════════════════════════════════════════════════════════
# Installer Engine
# ════════════════════════════════════════════════════════════════════

class InstallerEngine:
    def __init__(self, config, log_cb, progress_cb, done_cb):
        self.config = config
        self.log = log_cb
        self.progress = progress_cb
        self.done = done_cb
        self._cancelled = False

    @property
    def install_dir(self):
        return self.config["install_dir"]

    @property
    def venv_python(self):
        if IS_WIN:
            return os.path.join(self.install_dir, "venv", "Scripts", "python.exe")
        return os.path.join(self.install_dir, "venv", "bin", "python")

    @property
    def venv_pip(self):
        if IS_WIN:
            return os.path.join(self.install_dir, "venv", "Scripts", "pip.exe")
        return os.path.join(self.install_dir, "venv", "bin", "pip")

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self):
        self._cancelled = True

    def _exec(self, cmd, cwd=None, check=True, shell=False):
        self.log(f"  $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", "dim")
        effective_cwd = _safe_cwd(cwd or self.install_dir)
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=effective_cwd, encoding='utf-8', errors='replace', shell=shell,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                creationflags=_NO_WINDOW,
            )
            output = []
            for line in iter(proc.stdout.readline, ""):
                if self._cancelled:
                    proc.kill()
                    raise Exception("Cancelled")
                s = line.rstrip()
                if s:
                    self.log(f"    {s}", "dim")
                    output.append(s)
            proc.wait()
            if check and proc.returncode != 0:
                raise Exception(f"Command failed (exit {proc.returncode})")
            return "\n".join(output)
        except FileNotFoundError:
            raise Exception(f"Command not found: {cmd[0] if isinstance(cmd, list) else cmd}")

    def _run(self):
        try:
            steps = [
                ("Checking system prerequisites", self._bootstrap_system),
                ("Downloading Open-Sable", self._clone),
                ("Creating Python environment", self._create_venv),
                ("Installing dependencies", self._install_deps),
                ("Installing Ollama", self._install_ollama),
                ("Installing Node.js", self._install_node),
                ("Building Dashboard & Frontends", self._setup_node),
                ("Configuring environment", self._init_environment),
                ("Clearing old sessions", self._clear_sessions),
                ("Pulling AI model", self._pull_model),
                ("Creating shortcuts", self._create_shortcuts),
                ("Setting up services", self._install_services),
                ("Verifying installation", self._verify),
            ]
            total = len(steps)
            for i, (name, func) in enumerate(steps):
                if self._cancelled:
                    self.log("\n⚠ Installation cancelled.", "warning")
                    self.done(False, "Cancelled")
                    return
                self.log(f"\n━━━ Step {i+1}/{total}: {name}", "step")
                self.progress((i / total) * 100, name)
                func()

            self.progress(100, "Installation complete!")
            self.log("\n✔ Open-Sable installed successfully!", "success")
            self.log(f"  Location: {self.install_dir}", "dim")
            # Write install marker
            try:
                Path(os.path.join(self.install_dir, ".installed")).write_text(
                    f"installed={datetime.now().isoformat()}\n")
            except OSError:
                pass
            self.done(True, None)
        except Exception as e:
            self.log(f"\n✘ Installation error: {e}", "error")
            self.done(False, str(e))

    # ── System prerequisite bootstrap ────────────────────────────────

    def _bootstrap_system(self):
        """Check and install system-level prerequisites."""
        if IS_WIN:
            self._bootstrap_windows()
        elif IS_MAC:
            self._bootstrap_macos()
        elif IS_LINUX:
            self._bootstrap_linux()

    def _bootstrap_macos(self):
        """macOS: ensure Homebrew, Python 3.11+, Git, and Node.js are available."""
        has_brew = shutil.which("brew")
        py_cmd, _ = find_python()
        has_python = py_cmd is not None
        has_git = find_git() is not None
        has_node = find_node() is not None
        want_node = self.config.get("install_node", True)

        if has_brew and has_python and has_git and (has_node or not want_node):
            self.log("  ✔ All prerequisites available", "ok")
            return

        if not has_brew:
            # Homebrew installation requires sudo — run in Terminal.app
            self.log("  Homebrew not found — opening Terminal to install...", "dim")
            self.log("  ╔═══════════════════════════════════════════════════════╗", "step")
            self.log("  ║  A Terminal window will open.                        ║", "step")
            self.log("  ║  Enter your password when prompted, then wait.       ║", "step")
            self.log("  ║  The installer will continue automatically.          ║", "step")
            self.log("  ╚═══════════════════════════════════════════════════════╝", "step")

            marker = os.path.join(tempfile.gettempdir(), ".opensable-bootstrap-done")
            if os.path.exists(marker):
                os.unlink(marker)

            script_path = os.path.join(tempfile.gettempdir(), "opensable-bootstrap.sh")
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write("set -e\n")
                f.write('clear\n')
                f.write('echo ""\n')
                f.write('echo "╔═══════════════════════════════════════════════════════╗"\n')
                f.write('echo "║        Open-Sable — Installing Prerequisites         ║"\n')
                f.write('echo "╚═══════════════════════════════════════════════════════╝"\n')
                f.write('echo ""\n')
                # Homebrew
                f.write('if ! command -v brew &>/dev/null; then\n')
                f.write('  echo "▸ Installing Homebrew (macOS package manager)..."\n')
                f.write('  echo "  You may be asked for your password."\n')
                f.write('  echo ""\n')
                f.write('  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n')
                f.write('  # Add brew to current shell\n')
                f.write('  if [ -f /opt/homebrew/bin/brew ]; then\n')
                f.write('    eval "$(/opt/homebrew/bin/brew shellenv)"\n')
                f.write('  elif [ -f /usr/local/bin/brew ]; then\n')
                f.write('    eval "$(/usr/local/bin/brew shellenv)"\n')
                f.write('  fi\n')
                f.write('  # Add to shell profile for future sessions\n')
                f.write('  if [ -f /opt/homebrew/bin/brew ] && ! grep -q "brew shellenv" ~/.zprofile 2>/dev/null; then\n')
                f.write("    echo 'eval \"$(/opt/homebrew/bin/brew shellenv)\"' >> ~/.zprofile\n")
                f.write('  fi\n')
                f.write('  echo "✔ Homebrew installed"\n')
                f.write('else\n')
                f.write('  echo "✔ Homebrew already installed"\n')
                f.write('fi\n')
                f.write('echo ""\n')
                # Python
                f.write('if ! python3 --version 2>/dev/null | grep -qE "3\\.(1[1-9]|[2-9][0-9])"; then\n')
                f.write('  echo "▸ Installing Python 3.13..."\n')
                f.write('  brew install python@3.13\n')
                f.write('  echo "✔ Python installed"\n')
                f.write('else\n')
                f.write('  echo "✔ Python already installed"\n')
                f.write('fi\n')
                f.write('echo ""\n')
                # Node.js
                if want_node:
                    f.write('if ! command -v node &>/dev/null || [ "$(node -e "process.stdout.write(String(+process.version.slice(1).split(\\".\\")[0]>=18))")" != "1" ]; then\n')
                    f.write('  echo "▸ Installing Node.js..."\n')
                    f.write('  brew install node\n')
                    f.write('  echo "✔ Node.js installed"\n')
                    f.write('else\n')
                    f.write('  echo "✔ Node.js already installed"\n')
                    f.write('fi\n')
                    f.write('echo ""\n')
                # Git (should come with Xcode CLT installed by Homebrew, but just in case)
                f.write('if ! command -v git &>/dev/null; then\n')
                f.write('  echo "▸ Installing Git..."\n')
                f.write('  brew install git\n')
                f.write('fi\n')
                # Done
                f.write('echo ""\n')
                f.write('echo "✔ All prerequisites installed!"\n')
                f.write('echo "  You can close this Terminal window."\n')
                f.write('echo "  The installer will continue automatically."\n')
                f.write(f'echo "ok" > "{marker}"\n')
            os.chmod(script_path, 0o755)

            subprocess.Popen(["open", "-a", "Terminal", script_path])

            # Poll for completion
            self.log("  Waiting for Terminal to finish...", "dim")
            timeout, elapsed = 900, 0
            while not os.path.exists(marker):
                time.sleep(3)
                elapsed += 3
                if elapsed % 60 == 0:
                    self.log(f"  Still waiting... ({elapsed // 60}m)", "dim")
                if elapsed > timeout:
                    raise Exception(
                        "Prerequisites install timed out.\n"
                        "Open Terminal and run:\n"
                        '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n'
                        "  brew install python@3.13 node"
                    )
                if self._cancelled:
                    raise Exception("Cancelled")

            try:
                os.unlink(marker)
                os.unlink(script_path)
            except OSError:
                pass

            # Refresh PATH so new tools are found
            for p in ["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin"]:
                if p not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

            self.log("  ✔ Prerequisites installed via Terminal", "ok")
        else:
            # Homebrew is present — install missing tools directly (no sudo needed)
            if not has_python:
                self.log("  Installing Python 3.13 via Homebrew...", "dim")
                self._exec(["brew", "install", "python@3.13"], check=False)
                py_cmd, py_ver = find_python()
                if py_cmd:
                    self.log(f"  ✔ Python {py_ver} installed", "ok")
                else:
                    raise Exception("Failed to install Python.\nRun: brew install python@3.13")

            if not has_git:
                self.log("  Installing Git via Homebrew...", "dim")
                self._exec(["brew", "install", "git"], check=False)
                if find_git():
                    self.log("  ✔ Git installed", "ok")
                else:
                    self.log("  ⚠ Git install failed — install manually", "warning")

            if not has_node and want_node:
                self.log("  Installing Node.js via Homebrew...", "dim")
                self._exec(["brew", "install", "node"], check=False)
                if find_node():
                    self.log("  ✔ Node.js installed", "ok")
                else:
                    self.log("  ⚠ Node.js install failed — install manually", "warning")

            self.log("  ✔ System prerequisites OK", "ok")

    def _bootstrap_linux(self):
        """Linux: ensure Python 3.11+, Git, and Node.js are available."""
        py_cmd, _ = find_python()
        if not py_cmd:
            self.log("  Python 3.11+ not found — installing...", "dim")
            if shutil.which("apt-get"):
                self._exec(["sudo", "apt-get", "update", "-qq"], check=False)
                self._exec(["sudo", "apt-get", "install", "-y",
                            "python3", "python3-venv", "python3-pip"], check=False)
            elif shutil.which("dnf"):
                self._exec(["sudo", "dnf", "install", "-y",
                            "python3", "python3-pip"], check=False)
            elif shutil.which("pacman"):
                self._exec(["sudo", "pacman", "-S", "--noconfirm",
                            "python", "python-pip"], check=False)
            py_cmd, py_ver = find_python()
            if py_cmd:
                self.log(f"  ✔ Python {py_ver} installed", "ok")
            else:
                raise Exception("Python 3.11+ required. Install via your package manager.")

        if not find_git():
            self.log("  Git not found — installing...", "dim")
            if shutil.which("apt-get"):
                self._exec(["sudo", "apt-get", "install", "-y", "git"], check=False)
            elif shutil.which("dnf"):
                self._exec(["sudo", "dnf", "install", "-y", "git"], check=False)
            elif shutil.which("pacman"):
                self._exec(["sudo", "pacman", "-S", "--noconfirm", "git"], check=False)
            if find_git():
                self.log("  ✔ Git installed", "ok")
            else:
                self.log("  ⚠ Install git manually", "warning")

        self.log("  ✔ System prerequisites OK", "ok")

    def _bootstrap_windows(self):
        """Windows: check for Python, Git — attempt install via winget."""
        py_cmd, _ = find_python()
        if not py_cmd:
            if shutil.which("winget"):
                self.log("  Installing Python via winget...", "dim")
                self._exec(["winget", "install", "--id", "Python.Python.3.13",
                            "--accept-source-agreements", "--accept-package-agreements",
                            "-e", "--silent"], check=False)
                # Refresh PATH so the newly-installed Python is discoverable
                _refresh_windows_path()
                py_cmd, py_ver = find_python()
                if py_cmd:
                    self.log(f"  ✔ Python {py_ver} installed", "ok")
                else:
                    raise Exception("Python 3.11+ required.\nDownload from python.org")
            else:
                raise Exception("Python 3.11+ required.\nDownload from python.org")

        if not find_git():
            if shutil.which("winget"):
                self.log("  Installing Git via winget...", "dim")
                self._exec(["winget", "install", "--id", "Git.Git",
                            "--accept-source-agreements", "--accept-package-agreements",
                            "-e", "--silent"], check=False)
                # Refresh PATH so the newly-installed Git is discoverable
                _refresh_windows_path()
                if find_git():
                    self.log("  ✔ Git installed", "ok")
                else:
                    self.log("  ⚠ Install git from git-scm.com", "warning")
            else:
                self.log("  ⚠ Install git from git-scm.com", "warning")

        self.log("  ✔ System prerequisites OK", "ok")

    # ── Clone / Download ─────────────────────────────────────────────

    def _clone(self):
        if os.path.isdir(os.path.join(self.install_dir, ".git")):
            self.log("  Already a git repo — pulling latest...", "dim")
            self._exec(["git", "fetch", "origin", REPO_BRANCH], check=False)
            self._exec(["git", "reset", "--hard", f"origin/{REPO_BRANCH}"], check=False)
            self.log("  ✔ Repository updated", "ok")
            return

        cloned = False
        if find_git():
            os.makedirs(os.path.dirname(self.install_dir), exist_ok=True)
            try:
                self._exec(["git", "clone", "--branch", REPO_BRANCH, "--depth", "1",
                             REPO_URL, self.install_dir],
                            cwd=os.path.dirname(self.install_dir))
                self.log("  ✔ Repository cloned", "ok")
                return
            except Exception as e:
                self.log(f"  ⚠ git clone failed: {e}", "warning")
                self.log("  Falling back to archive download...", "dim")

        # Fallback: download tarball
        if not find_git():
            self.log("  Git not available - downloading archive...", "dim")
        url = (f"https://github.com/IdeoaLabs/Open-Sable/archive/"
               f"refs/heads/{REPO_BRANCH}.tar.gz")
        tarball = os.path.join(tempfile.gettempdir(), "opensable.tar.gz")
        _download_with_timeout(url, tarball)

        tmp_extract = tempfile.mkdtemp(prefix="opensable-")
        try:
            with tarfile.open(tarball) as tf:
                try:
                    tf.extractall(tmp_extract, filter="data")
                except TypeError:
                    tf.extractall(tmp_extract)

            # GitHub tarballs have a single top-level dir (e.g., Open-Sable-master/)
            contents = os.listdir(tmp_extract)
            src = (os.path.join(tmp_extract, contents[0])
                   if len(contents) == 1 and os.path.isdir(os.path.join(tmp_extract, contents[0]))
                   else tmp_extract)

            os.makedirs(self.install_dir, exist_ok=True)
            for item in os.listdir(src):
                s, d = os.path.join(src, item), os.path.join(self.install_dir, item)
                if os.path.exists(d):
                    if os.path.isdir(d):
                        shutil.rmtree(d)
                    else:
                        os.remove(d)
                shutil.move(s, d)
        finally:
            shutil.rmtree(tmp_extract, ignore_errors=True)
            try:
                os.unlink(tarball)
            except OSError:
                pass

        self.log("  ✔ Source code downloaded", "ok")

        # Initialize git for future updates
        if find_git():
            try:
                self._exec(["git", "init"], check=False)
                self._exec(["git", "remote", "add", "origin", REPO_URL], check=False)
                self._exec(["git", "fetch", "--depth", "1", "origin", REPO_BRANCH],
                           check=False)
                self._exec(["git", "reset", "--soft", f"origin/{REPO_BRANCH}"],
                           check=False)
                self.log("  ✔ Git initialized for future updates", "ok")
            except Exception:
                self.log("  ⚠ Git init skipped - updates will need manual git setup", "warning")

    def _create_venv(self):
        py_cmd, py_ver = find_python()
        if not py_cmd:
            raise Exception("Python 3.11+ required. Visit python.org")
        venv_dir = os.path.join(self.install_dir, "venv")
        self.log(f"  Creating venv with Python {py_ver}...", "dim")
        self._exec(py_cmd + ["-m", "venv", venv_dir])
        self.log("  ✔ Python environment created", "ok")

    def _install_deps(self):
        pip = [self.venv_python, "-m", "pip"]
        self._exec(pip + ["install", "--upgrade", "pip", "setuptools", "wheel", "-q"], check=False)
        self._exec(pip + ["install", "-e", ".[core]", "-q"],
                    cwd=self.install_dir, check=False)
        req = os.path.join(self.install_dir, "requirements.txt")
        if os.path.isfile(req):
            self._exec(pip + ["install", "-r", req, "-q"], cwd=self.install_dir, check=False)
        self.log("  ✔ Dependencies installed", "ok")

    def _install_ollama(self):
        if not self.config.get("install_ollama", True):
            self.log("  Skipped (user choice)", "dim")
            return
        if find_ollama():
            self.log(f"  ✔ Ollama already installed", "ok")
            return
        self.log("  Downloading Ollama...", "dim")
        if IS_WIN:
            dl = os.path.join(tempfile.gettempdir(), "OllamaSetup.exe")
            _download_with_timeout(OLLAMA_WIN_URL, dl)
            self._exec([dl, "/VERYSILENT", "/NORESTART"], check=False)
        else:
            self._exec("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
        if find_ollama():
            self.log("  ✔ Ollama installed", "ok")
        else:
            self.log("  ⚠ Install manually: ollama.com", "warning")

    def _install_node(self):
        if not self.config.get("install_node", True):
            self.log("  Skipped (user choice)", "dim")
            return
        if find_node():
            self.log(f"  ✔ Node.js already installed", "ok")
            return
        # Check if an older version exists (needs upgrade, not fresh install)
        _old_node = None
        try:
            r = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5,
                               creationflags=_NO_WINDOW)
            if r.returncode == 0:
                _old_node = r.stdout.strip().lstrip("v")
        except Exception:
            pass
        if _old_node:
            self.log(f"  Node.js {_old_node} is too old (need 20+) — upgrading...", "dim")
        else:
            self.log("  Installing Node.js 20 LTS...", "dim")
        if IS_LINUX:
            if shutil.which("apt-get"):
                self._exec("curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -",
                            shell=True, check=False)
                self._exec(["sudo", "apt-get", "install", "-y", "nodejs"])
            elif shutil.which("dnf"):
                self._exec(["sudo", "dnf", "install", "-y", "nodejs", "npm"])
            elif shutil.which("pacman"):
                self._exec(["sudo", "pacman", "-S", "--noconfirm", "nodejs", "npm"])
        elif IS_MAC and shutil.which("brew"):
            self._exec(["brew", "install", "node"])
        elif IS_WIN:
            _win_ok = False
            if shutil.which("winget"):
                if _old_node:
                    # Try upgrade first; may fail if not originally installed via winget
                    rc = self._exec(["winget", "upgrade", "--id", "OpenJS.NodeJS.LTS",
                                     "--accept-source-agreements", "--accept-package-agreements", "-e"],
                                    check=False)
                    if not find_node():
                        self.log("  winget upgrade not applicable — trying fresh install...", "dim")
                        self._exec(["winget", "install", "--id", "OpenJS.NodeJS.LTS",
                                    "--accept-source-agreements", "--accept-package-agreements", "-e"],
                                   check=False)
                else:
                    self._exec(["winget", "install", "--id", "OpenJS.NodeJS.LTS",
                                "--accept-source-agreements", "--accept-package-agreements", "-e"],
                               check=False)
                # Refresh PATH so the newly-installed Node is discoverable
                _refresh_windows_path()
                _win_ok = bool(find_node())
            if not _win_ok:
                # Fallback: download MSI directly (detect x86 vs x64)
                import struct
                bits = struct.calcsize("P") * 8
                arch = "x64" if bits == 64 else "x86"
                node_msi_url = f"https://nodejs.org/dist/v20.19.2/node-v20.19.2-{arch}.msi"
                self.log(f"  Downloading Node.js 20 LTS installer ({arch})...", "dim")
                msi = os.path.join(tempfile.gettempdir(), "node-v20-setup.msi")
                _download_with_timeout(node_msi_url, msi)
                self._exec(["msiexec", "/i", msi, "/qn", "/norestart"], check=False)
                _refresh_windows_path()
        if find_node():
            self.log("  ✔ Node.js installed", "ok")
        else:
            self.log("  ⚠ Install Node.js 20+ manually: nodejs.org", "warning")

    def _setup_node(self):
        if not find_node():
            self.log("  ⚠ Node.js not available — skipping frontends", "warning")
            return
        npm = find_npm()
        if not npm:
            self.log("  ⚠ npm not found — skipping frontends. Install Node.js from nodejs.org", "warning")
            return
        for project in ["dashboard", "desktop", "aggr"]:
            proj_dir = os.path.join(self.install_dir, project)
            pkg = os.path.join(proj_dir, "package.json")
            if not os.path.isfile(pkg):
                continue
            self.log(f"  Building {project}...", "dim")
            self._exec([npm, "install", "--legacy-peer-deps"], cwd=proj_dir, check=False)
            if project == "desktop":
                # Desktop needs both electron AND a vite build (dist/index.html)
                electron_bin = os.path.join(proj_dir, "node_modules", ".bin", "electron")
                if not os.path.isfile(electron_bin):
                    self.log(f"  ⚠ {project} — electron binary not found", "warning")
                self._exec([npm, "run", "build"], cwd=proj_dir, check=False)
                dist_check = os.path.join(proj_dir, "dist", "index.html")
                if os.path.isfile(dist_check):
                    self.log(f"  ✔ {project} ready", "ok")
                else:
                    self.log(f"  ⚠ {project} — dist/index.html missing, UI may not load", "warning")
            else:
                self._exec([npm, "run", "build"], cwd=proj_dir, check=False)
                self.log(f"  ✔ {project} ready", "ok")

    def _init_environment(self):
        """Create .env, agent profile, required directories, and verify sandbox."""
        # ── Required directories ──
        for dirname in ["data", "logs", "config", "models", "episodes"]:
            dirpath = os.path.join(self.install_dir, dirname)
            os.makedirs(dirpath, exist_ok=True)
        self.log("  ✔ Directories created (data, logs, config, models, episodes)", "ok")

        # ── .env file ──
        env_file = os.path.join(self.install_dir, ".env")
        if not os.path.isfile(env_file):
            env_example = os.path.join(self.install_dir, ".env.example")
            model = self.config.get("model", "qwen3.5:0.8b")
            if os.path.isfile(env_example):
                with open(env_example, "r") as f:
                    content = f.read()
                # Apply user's chosen model
                content = content.replace("DEFAULT_MODEL=llama3.1:8b", f"DEFAULT_MODEL={model}")
                # Disable auto-select so the user's chosen model sticks
                content = content.replace("AUTO_SELECT_MODEL=true", "AUTO_SELECT_MODEL=false")
                # Enable gateway by default
                content = content.replace("GATEWAY_ENABLED=false", "GATEWAY_ENABLED=true")
                with open(env_file, "w") as f:
                    f.write(content)
            else:
                # Minimal .env if no example found
                with open(env_file, "w") as f:
                    f.write(f"# Open-Sable Configuration\n")
                    f.write(f"DEFAULT_MODEL={model}\n")
                    f.write(f"AUTO_SELECT_MODEL=false\n")
                    f.write(f"OLLAMA_BASE_URL=http://localhost:11434\n")
                    f.write(f"GATEWAY_ENABLED=true\n")
                    f.write(f"WEBCHAT_HOST=127.0.0.1\n")
                    f.write(f"WEBCHAT_PORT=8789\n")
                    f.write(f"ENABLE_SANDBOX=true\n")
                    f.write(f"CLI_ENABLED=false\n")
            self.log("  ✔ .env created with your settings", "ok")
        else:
            self.log("  ✔ .env already exists (preserved)", "ok")

        # ── Agent profile (agents/sable/) ──
        agents_dir = os.path.join(self.install_dir, "agents", "sable")
        template_dir = os.path.join(self.install_dir, "agents", "_template")
        os.makedirs(agents_dir, exist_ok=True)
        os.makedirs(os.path.join(agents_dir, "data"), exist_ok=True)

        profile_files = {
            "profile.env": None,
            "soul.md": None,
            "tools.json": None,
        }
        for fname in profile_files:
            dest = os.path.join(agents_dir, fname)
            if not os.path.isfile(dest):
                template_src = os.path.join(template_dir, fname)
                if os.path.isfile(template_src):
                    shutil.copy2(template_src, dest)
                    # Patch profile.env with user's model
                    if fname == "profile.env":
                        model = self.config.get("model", "qwen3.5:0.8b")
                        with open(dest, "r") as f:
                            content = f.read()
                        content = content.replace("DEFAULT_MODEL=qwen3.5:0.8b",
                                                  f"DEFAULT_MODEL={model}")
                        content = content.replace("AGENT_NAME=MyAgent",
                                                  "AGENT_NAME=Sable")
                        content = content.replace("WEBCHAT_PORT=8792",
                                                  "WEBCHAT_PORT=8789")
                        with open(dest, "w") as f:
                            f.write(content)
        self.log("  ✔ Agent profile initialized (agents/sable/)", "ok")

        # ── Sandbox verification ──
        sandbox_file = os.path.join(self.install_dir, "opensable", "core", "sandbox_runner.py")
        if os.path.isfile(sandbox_file):
            try:
                result = subprocess.run(
                    [self.venv_python, "-c",
                     "from opensable.core.sandbox_runner import run_sandboxed_python; "
                     "print(run_sandboxed_python('print(42)', cpu_seconds=2))"],
                    capture_output=True, text=True, timeout=10,
                    cwd=self.install_dir, creationflags=_NO_WINDOW
                )
                if result.returncode == 0 and "42" in result.stdout:
                    self.log("  ✔ Sandbox engine verified (process isolation)", "ok")
                else:
                    self.log("  ⚠ Sandbox test returned unexpected result", "warning")
            except Exception:
                self.log("  ⚠ Sandbox test skipped (will work at runtime)", "warning")
        else:
            self.log("  ⚠ Sandbox runner not found", "warning")

        # ── Config/permissions ──
        perms_file = os.path.join(self.install_dir, "config", "permissions.json")
        if not os.path.isfile(perms_file):
            perms_src = os.path.join(self.install_dir, "config", "permissions.json")
            if not os.path.isfile(perms_src):
                # Create default permissions
                perms = {
                    "default": {
                        "browser_navigate": "always_allow",
                        "file_read": "always_allow",
                        "file_write": "ask",
                        "system_command": "always_allow",
                    }
                }
                with open(perms_file, "w") as f:
                    json.dump(perms, f, indent=2)
                self.log("  ✔ Default permissions created", "ok")
        else:
            self.log("  ✔ Permissions config exists", "ok")

    def _pull_model(self):
        model = self.config.get("model", "qwen3.5:0.8b")
        if not ollama_running():
            if find_ollama():
                self.log("  Starting Ollama...", "dim")
                if IS_WIN:
                    subprocess.Popen(["ollama", "serve"], creationflags=0x08000000)
                else:
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                for _ in range(10):
                    time.sleep(1)
                    if ollama_running():
                        break
        if not ollama_running():
            self.log(f"  ⚠ Ollama not running. Run: ollama pull {model}", "warning")
            return
        try:
            r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10,
                               creationflags=_NO_WINDOW)
            if model.split(":")[0] in r.stdout:
                self.log(f"  ✔ Model {model} already available", "ok")
                return
        except Exception:
            pass
        self.log(f"  Pulling {model}...", "dim")
        self._exec(["ollama", "pull", model], check=False)
        self.log(f"  ✔ Model {model} ready", "ok")

    def _install_services(self):
        """Install updater script + agent systemd/launchd service."""
        self._install_updater()
        self._install_agent_service()

    def _install_agent_service(self):
        """Create a systemd user service / launchd plist for opensable itself."""
        if IS_LINUX:
            user_dir = os.path.expanduser("~/.config/systemd/user")
            os.makedirs(user_dir, exist_ok=True)
            svc_path = os.path.join(user_dir, "opensable.service")
            with open(svc_path, "w") as f:
                f.write(f"[Unit]\nDescription=Open-Sable AI Agent\n")
                f.write(f"After=network.target ollama.service\n\n")
                f.write(f"[Service]\nType=simple\n")
                f.write(f"WorkingDirectory={self.install_dir}\n")
                f.write(f"ExecStart={self.install_dir}/venv/bin/python -m opensable\n")
                f.write(f"Restart=on-failure\nRestartSec=10\n")
                f.write(f"Environment=PYTHONUNBUFFERED=1\n\n")
                f.write(f"[Install]\nWantedBy=default.target\n")
            try:
                subprocess.run(["systemctl", "--user", "daemon-reload"],
                               capture_output=True, timeout=10)
                self.log("  ✔ Agent service created (opensable.service)", "ok")
                self.log("  ℹ  Start with: systemctl --user start opensable", "dim")
                self.log("  ℹ  Auto-start: systemctl --user enable opensable", "dim")
            except Exception:
                self.log("  ✔ Agent service file created", "ok")
        elif IS_MAC:
            plist_dir = os.path.expanduser("~/Library/LaunchAgents")
            os.makedirs(plist_dir, exist_ok=True)
            plist = os.path.join(plist_dir, "com.ideoalabs.opensable.plist")
            with open(plist, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n')
                f.write('<plist version="1.0"><dict>\n')
                f.write('<key>Label</key><string>com.ideoalabs.opensable</string>\n')
                f.write('<key>ProgramArguments</key><array>\n')
                f.write(f'  <string>{self.install_dir}/venv/bin/python</string>\n')
                f.write(f'  <string>-m</string>\n')
                f.write(f'  <string>opensable</string>\n')
                f.write('</array>\n')
                f.write(f'<key>WorkingDirectory</key><string>{self.install_dir}</string>\n')
                f.write('<key>RunAtLoad</key><false/>\n')
                f.write('<key>KeepAlive</key><false/>\n')
                f.write('</dict></plist>\n')
            self.log("  ✔ Agent plist created (com.ideoalabs.opensable)", "ok")
            self.log("  ℹ  Start with: launchctl load ~/Library/LaunchAgents/com.ideoalabs.opensable.plist", "dim")
        elif IS_WIN:
            # Create a start script for Windows
            start_bat = os.path.join(self.install_dir, "start-opensable.bat")
            with open(start_bat, "w", encoding="mbcs" if IS_WIN else "utf-8") as f:
                f.write(f'@echo off\ncd /d "{self.install_dir}"\n')
                f.write(f'call venv\\Scripts\\activate.bat\n')
                f.write(f'python -m opensable %*\n')
            self.log("  ✔ Start script created (start-opensable.bat)", "ok")

    def _install_updater(self):
        updater_path = os.path.join(self.install_dir, "opensable-update")
        ext = ".bat" if IS_WIN else ""
        if IS_WIN:
            with open(updater_path + ext, "w", encoding="mbcs") as f:
                f.write(f'@echo off\necho Updating Open-Sable...\n')
                f.write(f'cd /d "{self.install_dir}"\n')
                f.write(f'git fetch origin {REPO_BRANCH}\n')
                f.write(f'git stash --include-untracked 2>nul\n')
                f.write(f'git pull --rebase origin {REPO_BRANCH}\n')
                f.write(f'git stash pop 2>nul\n')
                f.write(f'call venv\\Scripts\\activate.bat\n')
                f.write(f'pip install -e ".[core]" -q\n')
                f.write(f'if exist requirements.txt pip install -r requirements.txt -q\n')
                f.write(f'if exist dashboard\\package.json (\n  cd dashboard && npm install --legacy-peer-deps -q && npm run build && cd ..\n)\n')
                f.write(f'if exist aggr\\package.json (\n  cd aggr && npm install --legacy-peer-deps -q && npm run build && cd ..\n)\n')
                f.write(f'echo Update complete!\npause\n')
        else:
            with open(updater_path, "w") as f:
                f.write(f'#!/bin/bash\nset -e\necho "Updating Open-Sable..."\n')
                f.write(f'cd "{self.install_dir}"\n')
                f.write(f'git fetch origin {REPO_BRANCH}\n')
                f.write(f'git stash --include-untracked 2>/dev/null || true\n')
                f.write(f'git pull --rebase origin {REPO_BRANCH} || true\n')
                f.write(f'git stash pop 2>/dev/null || true\n')
                f.write(f'source venv/bin/activate\n')
                f.write(f'pip install -e ".[core]" -q\n')
                f.write(f'[ -f requirements.txt ] && pip install -r requirements.txt -q\n')
                f.write(f'[ -f dashboard/package.json ] && (cd dashboard && npm install --legacy-peer-deps -q && npm run build; cd ..)\n')
                f.write(f'[ -f aggr/package.json ] && (cd aggr && npm install --legacy-peer-deps -q && npm run build; cd ..)\n')
                f.write(f'echo "Update complete!"\n')
            os.chmod(updater_path, 0o755)
        self.log("  ✔ Update script created", "ok")

        # Background service
        if IS_LINUX:
            user_dir = os.path.expanduser("~/.config/systemd/user")
            os.makedirs(user_dir, exist_ok=True)
            with open(os.path.join(user_dir, "opensable-update.service"), "w") as f:
                f.write(f"[Unit]\nDescription=Open-Sable Update Check\n\n")
                f.write(f"[Service]\nType=oneshot\n")
                f.write(f"ExecStart={self.install_dir}/opensable-update\n")
                f.write(f"WorkingDirectory={self.install_dir}\n")
            with open(os.path.join(user_dir, "opensable-update.timer"), "w") as f:
                f.write("[Unit]\nDescription=Open-Sable Daily Update\n\n")
                f.write("[Timer]\nOnCalendar=daily\nPersistent=true\n")
                f.write("RandomizedDelaySec=3600\n\n[Install]\nWantedBy=timers.target\n")
            try:
                subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, timeout=10)
                subprocess.run(["systemctl", "--user", "enable", "opensable-update.timer"], capture_output=True, timeout=10)
                subprocess.run(["systemctl", "--user", "start", "opensable-update.timer"], capture_output=True, timeout=10)
                self.log("  ✔ Auto-update timer enabled (daily)", "ok")
            except Exception:
                self.log("  ⚠ Could not enable timer", "warning")
        elif IS_MAC:
            plist_dir = os.path.expanduser("~/Library/LaunchAgents")
            os.makedirs(plist_dir, exist_ok=True)
            plist = os.path.join(plist_dir, "com.ideoalabs.opensable-update.plist")
            with open(plist, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n')
                f.write('<plist version="1.0"><dict>\n')
                f.write(f'<key>Label</key><string>com.ideoalabs.opensable-update</string>\n')
                f.write(f'<key>ProgramArguments</key><array><string>{self.install_dir}/opensable-update</string></array>\n')
                f.write('<key>StartCalendarInterval</key><dict><key>Hour</key><integer>10</integer></dict>\n')
                f.write('</dict></plist>\n')
            try:
                subprocess.run(["launchctl", "load", plist], capture_output=True, timeout=10)
                self.log("  ✔ Auto-update scheduled (daily 10 AM)", "ok")
            except Exception:
                self.log("  ⚠ Could not enable plist", "warning")
        elif IS_WIN:
            try:
                updater_bat = os.path.join(self.install_dir, "opensable-update.bat")
                subprocess.run(["schtasks", "/create", "/tn", "OpenSable-Update",
                                "/tr", updater_bat, "/sc", "daily", "/st", "10:00", "/f"],
                               capture_output=True, timeout=15, creationflags=_NO_WINDOW)
                self.log("  ✔ Auto-update scheduled (daily 10:00)", "ok")
            except Exception:
                self.log("  ⚠ Could not create scheduled task", "warning")

    def _create_shortcuts(self):
        if IS_WIN:
            self._shortcuts_windows()
        elif IS_MAC:
            self._shortcuts_macos()
        else:
            self._shortcuts_linux()

    def _shortcuts_windows(self):
        bat = os.path.join(self.install_dir, "opensable.bat")
        with open(bat, "w", encoding="mbcs" if IS_WIN else "utf-8") as f:
            f.write(f'@echo off\ncd /d "{self.install_dir}"\ncall venv\\Scripts\\activate.bat\npython -m opensable %*\n')
        self.log("  ✔ opensable.bat created", "ok")
        icon_dest = os.path.join(self.install_dir, "opensable.ico")
        try:
            if os.path.isfile(ICON_ICO):
                shutil.copy2(ICON_ICO, icon_dest)
        except Exception:
            icon_dest = ""
        try:
            for location, name in [
                (os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows",
                              "Start Menu", "Programs"), "Start Menu"),
                (os.path.join(os.path.expanduser("~"), "Desktop"), "Desktop"),
            ]:
                if os.path.isdir(location):
                    lnk = os.path.join(location, "Open-Sable.lnk")
                    icon_line = f'$s.IconLocation="{icon_dest},0";' if icon_dest else ''
                    ps = (f'$ws=New-Object -COM WScript.Shell;$s=$ws.CreateShortcut("{lnk}");'
                          f'$s.TargetPath="cmd.exe";'
                          f'$s.Arguments="/k cd /d `"{self.install_dir}`" && venv\\Scripts\\activate.bat && python -m opensable";'
                          f'$s.WorkingDirectory="{self.install_dir}";'
                          f'{icon_line}$s.Save()')
                    subprocess.run(["powershell", "-Command", ps], capture_output=True, timeout=10,
                                   creationflags=_NO_WINDOW)
                    self.log(f"  ✔ {name} shortcut created", "ok")
        except Exception:
            pass

    def _shortcuts_macos(self):
        app_bundle = os.path.expanduser("~/Applications/Open-Sable.app")
        electron_app = os.path.join(
            self.install_dir, "desktop", "node_modules", "electron", "dist", "Electron.app"
        )

        # Build a proper Electron .app bundle (not a shell wrapper)
        if os.path.isdir(electron_app):
            if os.path.isdir(app_bundle):
                shutil.rmtree(app_bundle)
            shutil.copytree(electron_app, app_bundle, symlinks=True)
        else:
            # Fallback: build minimal .app structure
            os.makedirs(os.path.join(app_bundle, "Contents", "MacOS"), exist_ok=True)
            os.makedirs(os.path.join(app_bundle, "Contents", "Resources"), exist_ok=True)

        contents = os.path.join(app_bundle, "Contents")
        res_dir = os.path.join(contents, "Resources")

        # Copy our icon
        icon_src = os.path.join(self.install_dir, "installers", "assets", "icon.icns")
        if not os.path.isfile(icon_src):
            icon_src = ICON_ICNS
        try:
            if os.path.isfile(icon_src):
                shutil.copy2(icon_src, os.path.join(res_dir, "opensable.icns"))
        except Exception:
            pass

        # Customize Info.plist
        plist_path = os.path.join(contents, "Info.plist")
        try:
            subprocess.run(["plutil", "-replace", "CFBundleName", "-string", "Open-Sable", plist_path], check=False)
            subprocess.run(["plutil", "-replace", "CFBundleDisplayName", "-string", "Open-Sable", plist_path], check=False)
            subprocess.run(["plutil", "-replace", "CFBundleIdentifier", "-string", "com.ideoalabs.opensable", plist_path], check=False)
            subprocess.run(["plutil", "-replace", "CFBundleVersion", "-string", APP_VERSION, plist_path], check=False)
            subprocess.run(["plutil", "-replace", "CFBundleShortVersionString", "-string", APP_VERSION, plist_path], check=False)
            subprocess.run(["plutil", "-replace", "CFBundleIconFile", "-string", "opensable", plist_path], check=False)
        except Exception:
            pass

        # Remove default_app.asar and create our bootstrap app
        default_asar = os.path.join(res_dir, "default_app.asar")
        if os.path.isfile(default_asar):
            os.remove(default_asar)

        app_code_dir = os.path.join(res_dir, "app")
        os.makedirs(app_code_dir, exist_ok=True)

        with open(os.path.join(app_code_dir, "package.json"), "w") as f:
            f.write(f'{{"name":"open-sable","version":"{APP_VERSION}","main":"main.js"}}\n')

        with open(os.path.join(app_code_dir, "main.js"), "w") as f:
            f.write(ELECTRON_BOOTSTRAP_JS.replace("__INSTALL_DIR__", self.install_dir))

        # Ad-hoc codesign
        try:
            codesign = shutil.which("codesign")
            if codesign:
                subprocess.run([codesign, "--force", "--deep", "--sign", "-", app_bundle],
                               capture_output=True, check=False)
        except Exception:
            pass

        # Refresh icon cache so macOS picks up our icon
        try:
            lsregister = ("/System/Library/Frameworks/CoreServices.framework"
                          "/Frameworks/LaunchServices.framework/Support/lsregister")
            if os.path.isfile(lsregister):
                subprocess.run([lsregister, "-f", app_bundle], capture_output=True, check=False)
            subprocess.run(["touch", app_bundle], capture_output=True, check=False)
        except Exception:
            pass

        self.log("  ✔ Open-Sable.app created in ~/Applications", "ok")

    def _shortcuts_linux(self):
        apps = os.path.expanduser("~/.local/share/applications")
        os.makedirs(apps, exist_ok=True)
        icon_dest = os.path.join(self.install_dir, "opensable.png")
        icon_copied = False
        # Try multiple icon sources: bundled assets, then repo's copy
        icon_candidates = [
            ICON_PNG,
            os.path.join(ASSETS_DIR, "icon_source.png"),
            os.path.join(self.install_dir, "installers", "assets", "icon_source.png"),
            os.path.join(self.install_dir, "installers", "assets", "logo.png"),
        ]
        for src in icon_candidates:
            try:
                if os.path.isfile(src):
                    shutil.copy2(src, icon_dest)
                    icon_copied = True
                    break
            except Exception:
                continue
        if not icon_copied:
            icon_dest = ""
        with open(os.path.join(apps, "opensable.desktop"), "w") as f:
            f.write(f"[Desktop Entry]\nName=Open-Sable\nComment={APP_TAGLINE}\n")
            f.write(f"Exec=bash -c 'cd {self.install_dir} && source venv/bin/activate && python -m opensable'\n")
            if icon_dest:
                f.write(f"Icon={icon_dest}\n")
            f.write(f"Terminal=true\nType=Application\nCategories=Development;Utility;\n")
        local_bin = os.path.expanduser("~/.local/bin")
        os.makedirs(local_bin, exist_ok=True)
        cli = os.path.join(local_bin, "opensable")
        with open(cli, "w") as f:
            f.write(f'#!/bin/bash\ncd "{self.install_dir}" && source venv/bin/activate && python -m opensable "$@"\n')
        os.chmod(cli, 0o755)
        self.log("  ✔ Desktop entry + CLI link created", "ok")

    def _clear_sessions(self):
        """Remove old chat sessions from ~/.opensable/ so install starts fresh."""
        dot_opensable = os.path.join(os.path.expanduser("~"), ".opensable")
        if not os.path.isdir(dot_opensable):
            self.log("  No previous sessions found", "dim")
            return
        cleared = 0
        for root, dirs, files in os.walk(dot_opensable):
            if os.path.basename(root) == "sessions":
                for f in files:
                    if f.endswith(".json"):
                        os.remove(os.path.join(root, f))
                        cleared += 1
        if cleared:
            self.log(f"  ✔ Cleared {cleared} old chat sessions", "ok")
        else:
            self.log("  No previous sessions found", "dim")

    def _verify(self):
        errors = 0
        checks = 0
        if os.path.isfile(self.venv_python):
            self.log("  ✔ Python environment OK", "ok")
            checks += 1
        else:
            self.log("  ✘ Python environment missing", "error")
            errors += 1
        try:
            r = subprocess.run([self.venv_python, "-c", "import opensable; print(opensable.__version__)"],
                               capture_output=True, text=True, timeout=15, cwd=self.install_dir)
            if r.returncode == 0:
                self.log(f"  ✔ opensable v{r.stdout.strip()}", "ok")
                checks += 1
            else:
                self.log("  ✘ opensable import failed", "error")
                errors += 1
        except Exception:
            self.log("  ⚠ Could not verify import", "warning")
        # Dashboard
        dist_index = os.path.join(self.install_dir, "dashboard", "dist", "index.html")
        if os.path.isfile(dist_index):
            self.log("  ✔ Dashboard built", "ok")
            checks += 1
        else:
            self.log("  ⚠ Dashboard not built", "warning")
        # Desktop (Electron)
        electron_bin = os.path.join(self.install_dir, "desktop", "node_modules", ".bin", "electron")
        if os.path.isfile(electron_bin):
            self.log("  ✔ Desktop app ready (Electron)", "ok")
            checks += 1
        else:
            self.log("  ⚠ Desktop app not installed", "warning")
        # .env
        env_file = os.path.join(self.install_dir, ".env")
        if os.path.isfile(env_file):
            self.log("  ✔ Configuration (.env)", "ok")
            checks += 1
        else:
            self.log("  ⚠ No .env file", "warning")
        # Agent profile
        profile_env = os.path.join(self.install_dir, "agents", "sable", "profile.env")
        if os.path.isfile(profile_env):
            self.log("  ✔ Agent profile (agents/sable/)", "ok")
            checks += 1
        else:
            self.log("  ⚠ Agent profile not initialized", "warning")
        # Sandbox
        sandbox = os.path.join(self.install_dir, "opensable", "core", "sandbox_runner.py")
        if os.path.isfile(sandbox):
            self.log("  ✔ Sandbox engine present", "ok")
            checks += 1
        # Ollama
        if ollama_running():
            self.log("  ✔ Ollama API OK", "ok")
            checks += 1
        else:
            self.log("  ⚠ Ollama not running", "warning")
        self.log(f"\n  {checks} components verified, {errors} error(s)", "ok" if errors == 0 else "warning")
        if errors:
            raise Exception(f"{errors} verification error(s)")


# ════════════════════════════════════════════════════════════════════
# GUI Wizard
# ════════════════════════════════════════════════════════════════════

class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} — Setup Wizard")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)

        # Center window
        w, h = 720, 620
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        # Try to set icon
        try:
            if IS_WIN and os.path.isfile(ICON_ICO):
                self.iconbitmap(ICON_ICO)
            elif os.path.isfile(ICON_PNG):
                img = tk.PhotoImage(file=ICON_PNG)
                self.iconphoto(True, img)
        except Exception:
            pass

        # Load logo
        self._logo_img = None
        try:
            if os.path.isfile(LOGO_PATH):
                self._logo_img = tk.PhotoImage(file=LOGO_PATH)
                # Scale down if needed
                pw, ph = self._logo_img.width(), self._logo_img.height()
                if pw > 120:
                    factor = max(1, pw // 120)
                    self._logo_img = self._logo_img.subsample(factor, factor)
        except Exception:
            pass

        # Styles
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background=BG_DARK, foreground=FG_TEXT,
                        font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=BG_DARK, foreground=FG_TEXT,
                        font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel", background=BG_DARK, foreground=FG_DIM,
                        font=("Segoe UI", 12))
        style.configure("Small.TLabel", background=BG_DARK, foreground=FG_DIM,
                        font=("Segoe UI", 9))
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"))
        style.configure("TRadiobutton", background=BG_DARK, foreground=FG_TEXT,
                        font=("Segoe UI", 10), focuscolor=BG_DARK)
        style.map("TRadiobutton",
                  background=[("active", BG_CARD)],
                  foreground=[("active", ACCENT)])
        style.configure("Horizontal.TProgressbar", troughcolor=BG_INPUT,
                        background=ACCENT, thickness=8)

        # Variables
        self.install_dir_var = tk.StringVar(value=default_install_dir())
        self.model_var = tk.StringVar(value="qwen3.5:0.8b")
        self.install_ollama_var = tk.BooleanVar(value=True)
        self.install_node_var = tk.BooleanVar(value=True)

        self.engine = None
        self._pages = []
        self._current_page = 0
        self._slide_idx = 0
        self._slide_timer = None

        # Container
        self._container = tk.Frame(self, bg=BG_DARK)
        self._container.pack(fill="both", expand=True)

        # Build all pages
        self._build_pages()
        self._show_page(0)

    def _build_pages(self):
        for p in self._pages:
            p.destroy()
        self._pages = []
        self._pages.append(self._build_welcome_page())
        self._pages.append(self._build_config_page())
        self._pages.append(self._build_progress_page())
        self._pages.append(self._build_done_page())

    def _show_page(self, idx):
        for p in self._pages:
            p.pack_forget()
        self._current_page = idx
        self._pages[idx].pack(fill="both", expand=True)

    # ── Page 0: Welcome ─────────────────────────────────────────────

    def _build_welcome_page(self):
        page = tk.Frame(self._container, bg=BG_DARK)
        content = tk.Frame(page, bg=BG_DARK)
        content.place(relx=0.5, rely=0.5, anchor="center")

        # Logo
        if self._logo_img:
            tk.Label(content, image=self._logo_img, bg=BG_DARK).pack(pady=(0, 10))

        ttk.Label(content, text=APP_NAME, style="Title.TLabel").pack()
        ttk.Label(content, text=APP_TAGLINE, style="Subtitle.TLabel").pack(pady=(2, 15))

        # Detect existing install
        install_dir = self.install_dir_var.get()
        is_installed = (os.path.isdir(install_dir)
                        and os.path.isfile(os.path.join(install_dir, "pyproject.toml"))
                        and os.path.isdir(os.path.join(install_dir, "opensable"))
                        and os.path.isfile(os.path.join(install_dir, ".installed")))

        if is_installed:
            local_ver = get_local_version(install_dir)

            # ── Header card: version + path ──
            header_card = tk.Frame(content, bg=BG_CARD, padx=24, pady=14)
            header_card.pack(fill="x", padx=20, pady=(0, 8))

            hdr_top = tk.Frame(header_card, bg=BG_CARD)
            hdr_top.pack(fill="x")
            tk.Label(hdr_top, text="✔", fg=ACCENT, bg=BG_CARD,
                     font=("Segoe UI", 16, "bold")).pack(side="left", padx=(0, 8))
            tk.Label(hdr_top, text=APP_NAME, fg=FG_TEXT, bg=BG_CARD,
                     font=("Segoe UI", 15, "bold")).pack(side="left")
            tk.Label(hdr_top, text=f"  v{local_ver}", fg=ACCENT, bg=BG_CARD,
                     font=("Segoe UI", 15, "bold")).pack(side="left")

            ttk.Label(header_card, text=install_dir,
                      foreground=FG_DIM, background=BG_CARD,
                      font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))

            # Update check label (inside header card)
            update_label = ttk.Label(header_card, text="  Checking for updates...",
                                     foreground=FG_DIM, background=BG_CARD,
                                     font=("Segoe UI", 9))
            update_label.pack(anchor="w", pady=(2, 0))

            def _check_remote():
                rv = get_remote_version()
                if rv and rv != local_ver:
                    self.after(0, lambda: update_label.configure(
                        text=f"  ⬆ Update available: v{rv}", foreground=WARNING_C))
                elif rv:
                    self.after(0, lambda: update_label.configure(
                        text="  ✔ You're on the latest version", foreground=ACCENT))
                else:
                    self.after(0, lambda: update_label.configure(
                        text="  ⚠ Could not check for updates", foreground=FG_DIM))
            threading.Thread(target=_check_remote, daemon=True).start()

            # ── System status row ──
            status_row = tk.Frame(content, bg=BG_DARK)
            status_row.pack(fill="x", padx=20, pady=(4, 10))

            def _status_pill(parent, label, value, ok):
                pill = tk.Frame(parent, bg=BG_CARD, padx=8, pady=4)
                pill.pack(side="left", padx=3, expand=True, fill="x")
                color = ACCENT if ok else FG_DIM
                icon = "●" if ok else "○"
                tk.Label(pill, text=f"{icon} {label}", fg=color, bg=BG_CARD,
                         font=("Segoe UI", 8, "bold")).pack(anchor="w")
                tk.Label(pill, text=value or "not found", fg=FG_DIM if not ok else FG_TEXT,
                         bg=BG_CARD, font=("Segoe UI", 8)).pack(anchor="w")

            _, py_ver = find_python()
            git_ver = find_git()
            node_ver = find_node()
            ollama_ver = find_ollama()

            _status_pill(status_row, "Python", py_ver, py_ver is not None)
            _status_pill(status_row, "Git", git_ver, git_ver is not None)
            _status_pill(status_row, "Node", node_ver, node_ver is not None)
            _status_pill(status_row, "Ollama", ollama_ver, ollama_ver is not None)

            # ── Primary action: Launch ──
            make_button(content, text="  🚀  Launch Open-Sable  ",
                        command=self._launch,
                        bg=ACCENT, fg=BG_DARK, hover_bg=ACCENT_HOVER, hover_fg=BG_DARK,
                        font=("Segoe UI", 13, "bold"), padx=28, pady=10
                        ).pack(pady=(2, 8))

            # ── Management grid ──
            grid = tk.Frame(content, bg=BG_DARK)
            grid.pack(pady=(0, 6))

            def _mgmt_btn(parent, text, fg, bg_c, hover_bg, cmd, row, col):
                btn = make_button(parent, text=text, command=cmd,
                                  bg=bg_c, fg=fg, hover_bg=hover_bg, hover_fg=fg,
                                  font=("Segoe UI", 10, "bold"), padx=16, pady=7)
                btn.configure(width=16, anchor="center")
                btn.grid(row=row, column=col, padx=4, pady=3)

            _mgmt_btn(grid, "⬆  Update", ACCENT, "#0d3320", "#145530",
                      self._start_update, 0, 0)
            _mgmt_btn(grid, "↻  Reinstall", WARNING_C, BG_INPUT, BG_CARD,
                      self._start_reinstall, 0, 1)
            _mgmt_btn(grid, "📂  Open Folder", FG_TEXT, BG_INPUT, BG_CARD,
                      self._open_folder, 1, 0)
            _mgmt_btn(grid, "🔧  Fresh Install", FG_TEXT, BG_INPUT, BG_CARD,
                      lambda: self._show_page(1), 1, 1)

            # Uninstall — separate, less prominent
            make_button(content, text="🗑  Uninstall...",
                        command=self._show_uninstall_page,
                        bg="#2d1518", fg=ERROR_C, hover_bg="#3b1a1a", hover_fg=ERROR_C,
                        font=("Segoe UI", 9), padx=12, pady=4
                        ).pack(pady=(6, 0))

        else:
            # Fresh install welcome
            ttk.Label(content, text="v" + APP_VERSION, style="Small.TLabel").pack(pady=(0, 20))

            btn = make_button(content, text="  Install Open-Sable  ",
                             command=lambda: self._show_page(1),
                             bg=ACCENT, fg=BG_DARK, hover_bg=ACCENT_HOVER, hover_fg=BG_DARK,
                             font=("Segoe UI", 14, "bold"), padx=30, pady=12)
            btn.pack(pady=10)

            ttk.Label(content, text=(
                "Free • Open Source • 100% Local\n"
                "No cloud, no subscriptions, no data collection"
            ), style="Small.TLabel", justify="center").pack(pady=(10, 0))

        return page

    # ── Page 1: Config ───────────────────────────────────────────────

    def _build_config_page(self):
        page = tk.Frame(self._container, bg=BG_DARK)

        # Header
        header = tk.Frame(page, bg=BG_DARK)
        header.pack(fill="x", padx=30, pady=(20, 10))
        ttk.Label(header, text="Configuration", style="Title.TLabel").pack(anchor="w")

        # Scrollable area using Text widget (Tk 9.0 macOS native scroll)
        scroll_text = tk.Text(page, bg=BG_DARK, highlightthickness=0, bd=0,
                              cursor="arrow", wrap="none", padx=30)
        scroll_sb = ttk.Scrollbar(page, orient="vertical", command=scroll_text.yview)
        scroll_text.configure(yscrollcommand=scroll_sb.set)
        scroll_sb.pack(side="right", fill="y")
        scroll_text.pack(side="left", fill="both", expand=True)

        def _embed(widget):
            """Embed a frame into the scrollable Text widget, full width."""
            scroll_text.window_create("end", window=widget, stretch=True)
            scroll_text.insert("end", "\n")

        # Auto-resize embedded frames to fill the Text widget width
        def _on_configure(event):
            w = event.width - 60  # account for padx on both sides
            if w < 100:
                return
            for name in scroll_text.window_names():
                scroll_text.window_configure(name, minwidth=w)
        scroll_text.bind("<Configure>", _on_configure)

        # ── Install location ──
        loc_frame = tk.Frame(scroll_text, bg=BG_DARK)
        ttk.Label(loc_frame, text="Install Location").pack(anchor="w")
        row = tk.Frame(loc_frame, bg=BG_DARK)
        row.pack(fill="x", pady=5)
        tk.Entry(row, textvariable=self.install_dir_var, font=("Consolas", 10),
                 bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_TEXT,
                 relief="flat", bd=5).pack(side="left", fill="x", expand=True)
        make_button(row, text="Browse", command=self._browse_dir,
                   bg=BG_INPUT, fg=FG_TEXT, hover_bg=BG_CARD,
                   font=("Segoe UI", 9), padx=10, pady=5
                   ).pack(side="left", padx=(5, 0))
        _embed(loc_frame)

        # ── Dependencies status ──
        dep_frame = tk.Frame(scroll_text, bg=BG_DARK)
        ttk.Label(dep_frame, text="Dependencies").pack(anchor="w", pady=(0, 5))

        deps = [
            ("Python 3.11+", find_python()[1]),
            ("Git", find_git()),
            ("Node.js 18+", find_node()),
            ("Ollama", find_ollama()),
        ]
        for name, ver in deps:
            r = tk.Frame(dep_frame, bg=BG_DARK)
            r.pack(fill="x", pady=1)
            if ver:
                ttk.Label(r, text=f"  ✔ {name} {ver}", foreground=ACCENT,
                          font=("Segoe UI", 10)).pack(anchor="w")
            else:
                ttk.Label(r, text=f"  ○ {name} (will be installed)", foreground=WARNING_C,
                          font=("Segoe UI", 10)).pack(anchor="w")
        _embed(dep_frame)

        # ── Model selection ──
        model_frame = tk.Frame(scroll_text, bg=BG_DARK)
        ttk.Label(model_frame, text="AI Model (all support <thinking>)").pack(anchor="w", pady=(0, 5))

        for value, label in MODELS:
            ttk.Radiobutton(model_frame, text=label, variable=self.model_var,
                           value=value).pack(anchor="w", padx=10, pady=1)
        _embed(model_frame)

        # ── Options ──
        opt_frame = tk.Frame(scroll_text, bg=BG_DARK)
        ttk.Checkbutton(opt_frame, text="Install Ollama (if not present)",
                        variable=self.install_ollama_var).pack(anchor="w")
        ttk.Checkbutton(opt_frame, text="Install Node.js (if not present)",
                        variable=self.install_node_var).pack(anchor="w")
        _embed(opt_frame)

        # ── Buttons ──
        btn_frame = tk.Frame(scroll_text, bg=BG_DARK)
        make_button(btn_frame, text="← Back",
                   command=lambda: self._show_page(0),
                   bg=BG_INPUT, fg=FG_TEXT, hover_bg=BG_CARD,
                   padx=15, pady=8).pack(side="left")

        make_button(btn_frame, text="  Install  ",
                   command=self._start_install,
                   bg=ACCENT, fg=BG_DARK, hover_bg=ACCENT_HOVER, hover_fg=BG_DARK,
                   font=("Segoe UI", 12, "bold"), padx=25, pady=8
                   ).pack(side="right")
        _embed(btn_frame)

        # Block direct text editing but keep embedded widgets interactive
        scroll_text.bind("<Key>", lambda e: "break")
        scroll_text.configure(insertwidth=0)
        return page

    # ── Page 2: Progress ─────────────────────────────────────────────

    def _build_progress_page(self):
        page = tk.Frame(self._container, bg=BG_DARK)

        # Header area (title + info slides)
        top = tk.Frame(page, bg=BG_DARK)
        top.pack(fill="x", padx=30, pady=(20, 5))

        ttk.Label(top, text="Installing...", style="Title.TLabel").pack(anchor="w")
        self._progress_subtitle = ttk.Label(top, text="Preparing...",
                                            style="Subtitle.TLabel")
        self._progress_subtitle.pack(anchor="w")

        # Progress bar
        bar_frame = tk.Frame(page, bg=BG_DARK)
        bar_frame.pack(fill="x", padx=30, pady=(10, 5))
        self._progress_bar = ttk.Progressbar(bar_frame, style="Horizontal.TProgressbar",
                                              mode="determinate", maximum=100)
        self._progress_bar.pack(fill="x", side="left", expand=True)
        self._progress_pct = ttk.Label(bar_frame, text="0%", style="Small.TLabel")
        self._progress_pct.pack(side="right", padx=(8, 0))

        # Info slides card
        slide_card = tk.Frame(page, bg=BG_CARD, padx=20, pady=15)
        slide_card.pack(fill="x", padx=30, pady=(10, 5))

        self._slide_title = ttk.Label(slide_card, text=INSTALL_SLIDES[0][0],
                                       foreground=ACCENT, background=BG_CARD,
                                       font=("Segoe UI", 13, "bold"))
        self._slide_title.pack(anchor="w")
        self._slide_body = ttk.Label(slide_card, text=INSTALL_SLIDES[0][1],
                                      foreground=FG_TEXT, background=BG_CARD,
                                      font=("Segoe UI", 10), wraplength=600,
                                      justify="left")
        self._slide_body.pack(anchor="w", pady=(5, 5))

        self._slide_dots_frame = tk.Frame(slide_card, bg=BG_CARD)
        self._slide_dots_frame.pack(pady=(5, 0))
        self._update_slide_dots(0)

        # Start slide rotation
        self._rotate_slides()

        # Log output
        self._log_text = tk.Text(page, bg=BG_INPUT, fg=FG_TEXT, font=("Consolas", 9),
                                  relief="flat", bd=5, wrap="word", state="disabled",
                                  height=10, insertbackground=FG_TEXT)
        self._log_text.pack(fill="both", expand=True, padx=30, pady=(5, 10))

        # Text tags for coloring
        self._log_text.tag_configure("dim", foreground=FG_DIM)
        self._log_text.tag_configure("ok", foreground=ACCENT)
        self._log_text.tag_configure("step", foreground=ACCENT, font=("Consolas", 9, "bold"))
        self._log_text.tag_configure("error", foreground=ERROR_C)
        self._log_text.tag_configure("warning", foreground=WARNING_C)
        self._log_text.tag_configure("success", foreground=ACCENT, font=("Consolas", 10, "bold"))

        # Cancel button
        self._cancel_btn = make_button(page, text="Cancel",
                                       command=self._cancel_install,
                                       bg=BG_INPUT, fg=ERROR_C, hover_bg=BG_CARD, hover_fg=ERROR_C,
                                       padx=15, pady=5)
        self._cancel_btn.pack(pady=(0, 15))

        return page

    # ── Page 3: Done ─────────────────────────────────────────────────

    def _build_done_page(self):
        page = tk.Frame(self._container, bg=BG_DARK)
        content = tk.Frame(page, bg=BG_DARK)
        content.place(relx=0.5, rely=0.5, anchor="center")

        self._done_icon = ttk.Label(content, text="✔", foreground=ACCENT,
                                     font=("Segoe UI", 48))
        self._done_icon.pack()
        self._done_title = ttk.Label(content, text="Installation Complete!",
                                      style="Title.TLabel")
        self._done_title.pack(pady=(5, 5))
        self._done_subtitle = ttk.Label(content, text="Open-Sable is ready to use.",
                                         style="Subtitle.TLabel")
        self._done_subtitle.pack(pady=(0, 20))

        btn_row = tk.Frame(content, bg=BG_DARK)
        btn_row.pack(pady=10)

        self._done_launch_btn = make_button(btn_row, text="  🚀 Launch Open-Sable  ",
                   command=self._launch,
                   bg=ACCENT, fg=BG_DARK, hover_bg=ACCENT_HOVER, hover_fg=BG_DARK,
                   font=("Segoe UI", 13, "bold"), padx=25, pady=10)
        self._done_launch_btn.pack(side="left", padx=5)

        self._done_folder_btn = make_button(btn_row, text=" 📂 Open Folder ",
                   command=self._open_folder,
                   bg=BG_INPUT, fg=FG_TEXT, hover_bg=BG_CARD,
                   padx=15, pady=8)
        self._done_folder_btn.pack(side="left", padx=5)

        # Start Over button
        make_button(content, text="← Start Over",
                   command=self._start_over,
                   bg=BG_INPUT, fg=FG_DIM, hover_bg=BG_CARD,
                   padx=12, pady=5).pack(pady=(15, 0))

        return page

    # ── Slide rotation ───────────────────────────────────────────────

    def _rotate_slides(self):
        self._slide_idx = (self._slide_idx + 1) % len(INSTALL_SLIDES)
        title, body = INSTALL_SLIDES[self._slide_idx]
        self._slide_title.configure(text=title)
        self._slide_body.configure(text=body)
        self._update_slide_dots(self._slide_idx)
        self._slide_timer = self.after(6000, self._rotate_slides)

    def _update_slide_dots(self, idx):
        for w in self._slide_dots_frame.winfo_children():
            w.destroy()
        for i in range(len(INSTALL_SLIDES)):
            color = ACCENT if i == idx else FG_DIM
            tk.Label(self._slide_dots_frame, text="●", fg=color, bg=BG_CARD,
                     font=("Segoe UI", 6), cursor="hand2").pack(side="left", padx=2)

    # ── Actions ──────────────────────────────────────────────────────

    def _browse_dir(self):
        d = filedialog.askdirectory(title="Select Install Location")
        if d:
            self.install_dir_var.set(d)

    def _start_install(self):
        self._show_page(2)
        self._clear_log()
        self._progress_subtitle.configure(text="Installing...")

        config = {
            "install_dir": self.install_dir_var.get(),
            "model": self.model_var.get(),
            "install_ollama": self.install_ollama_var.get(),
            "install_node": self.install_node_var.get(),
        }
        self.engine = InstallerEngine(config, self._write_log, self._update_progress,
                                       self._install_done)
        self.engine.start()

    def _start_update(self):
        install_dir = self.install_dir_var.get()
        if not os.path.isdir(install_dir):
            messagebox.showerror("Error", f"Install directory not found:\n{install_dir}")
            return
        self._show_page(2)
        self._clear_log()
        self._progress_subtitle.configure(text="Updating...")
        self.engine = UpdateEngine(install_dir, self._write_log, self._update_progress,
                                    self._install_done)
        self.engine.start()

    def _start_reinstall(self):
        install_dir = self.install_dir_var.get()
        if not os.path.isdir(install_dir):
            messagebox.showerror("Error", f"Install directory not found:\n{install_dir}")
            return
        self._show_page(2)
        self._clear_log()
        self._progress_subtitle.configure(text="Reinstalling...")
        config = {
            "install_dir": install_dir,
            "model": self.model_var.get(),
            "install_ollama": True,
            "install_node": True,
            "reinstall": True,
        }
        self.engine = ReinstallEngine(config, self._write_log, self._update_progress,
                                       self._install_done)
        self.engine.start()

    def _show_uninstall_page(self):
        """Build and show the granular uninstall selection page."""
        install_dir = self.install_dir_var.get()
        if not os.path.isdir(install_dir):
            messagebox.showerror("Error", f"Install directory not found:\n{install_dir}")
            return

        # Hide current page
        for p in self._pages:
            p.pack_forget()

        # Build uninstall page dynamically
        page = tk.Frame(self._container, bg=BG_DARK)
        page.pack(fill="both", expand=True)

        # Scrollable area using Text widget (Tk 9.0 macOS native scroll)
        scroll_text = tk.Text(page, bg=BG_DARK, highlightthickness=0, bd=0,
                              cursor="arrow", wrap="none", padx=30)
        scroll_sb = ttk.Scrollbar(page, orient="vertical", command=scroll_text.yview)
        scroll_text.configure(yscrollcommand=scroll_sb.set)
        scroll_sb.pack(side="right", fill="y")
        scroll_text.pack(side="left", fill="both", expand=True)

        def _embed(widget):
            scroll_text.window_create("end", window=widget, stretch=True)
            scroll_text.insert("end", "\n")

        def _on_configure(event):
            w = event.width - 60
            if w < 100:
                return
            for name in scroll_text.window_names():
                scroll_text.window_configure(name, minwidth=w)
        scroll_text.bind("<Configure>", _on_configure)

        # Header
        header = tk.Frame(scroll_text, bg=BG_DARK)
        ttk.Label(header, text="🗑  Uninstall Open-Sable",
                  font=("Segoe UI", 18, "bold"),
                  foreground=ERROR_C, background=BG_DARK).pack(anchor="w")
        ttk.Label(header, text="Select what you want to remove:",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        _embed(header)

        # Checkboxes with descriptions
        checks_frame = tk.Frame(scroll_text, bg=BG_DARK)

        vars_dict = {}

        def _add_check(key, label, desc, default=True, has_it=True):
            var = tk.BooleanVar(value=default and has_it)
            vars_dict[key] = var
            row = tk.Frame(checks_frame, bg=BG_CARD, padx=14, pady=8)
            row.pack(fill="x", pady=3)
            cb = tk.Checkbutton(row, variable=var, bg=BG_CARD, fg=FG_TEXT,
                                activebackground=BG_CARD, activeforeground=ACCENT,
                                selectcolor=BG_INPUT, bd=0, highlightthickness=0,
                                state="normal" if has_it else "disabled")
            cb.pack(side="left", padx=(0, 8))
            txt = tk.Frame(row, bg=BG_CARD)
            txt.pack(side="left", fill="x")
            color = FG_TEXT if has_it else FG_DIM
            tk.Label(txt, text=label, fg=color, bg=BG_CARD,
                     font=("Segoe UI", 11, "bold"), anchor="w").pack(anchor="w")
            tk.Label(txt, text=desc, fg=FG_DIM, bg=BG_CARD,
                     font=("Segoe UI", 9), anchor="w").pack(anchor="w")

        # Detect what exists
        has_venv = any(os.path.isdir(os.path.join(install_dir, n))
                       for n in ["venv", ".venv"])
        has_node = any(os.path.isdir(os.path.join(install_dir, p, "node_modules"))
                       for p in ["dashboard", "sable_dev", "aggr", "desktop", "mobile", "website"])
        has_ollama = find_ollama() is not None
        has_shortcuts = True  # always offer
        has_services = True   # always offer
        has_data = any(os.path.exists(os.path.join(install_dir, d))
                       for d in ["agents", "data", "episodes", ".env", "logs", "models"])

        venv_size = ""
        for n in ["venv", ".venv"]:
            vp = os.path.join(install_dir, n)
            if os.path.isdir(vp):
                try:
                    total = sum(f.stat().st_size for f in Path(vp).rglob("*") if f.is_file())
                    venv_size = f" (~{total // (1024*1024)} MB)"
                except Exception:
                    pass
                break

        _add_check("venv", f"Python Virtual Environment{venv_size}",
                   "venv/ — Python packages and interpreter",
                   default=True, has_it=has_venv)
        _add_check("node_modules", "Node Modules",
                   "node_modules/ — dashboard, aggr, desktop, mobile, website",
                   default=True, has_it=has_node)
        _add_check("generated", "Generated & Build Files",
                   "dashboard/dist/, __pycache__/, .bat scripts, update scripts",
                   default=True)
        _add_check("shortcuts", "Desktop Shortcuts & CLI Links",
                   ".desktop entry, Start Menu shortcut, ~/.local/bin/opensable",
                   default=True, has_it=has_shortcuts)
        _add_check("services", "Auto-Update Services",
                   "systemd timers, launchd plists, scheduled tasks",
                   default=True, has_it=has_services)
        _add_check("ollama", "Uninstall Ollama",
                   f"Remove the Ollama binary and program{' (detected)' if has_ollama else ' (not found)'}",
                   default=False, has_it=has_ollama)
        _add_check("user_data", "User Data (CAUTION)",
                   "agents/, data/, episodes/, logs/, models/, .env — YOUR personal data",
                   default=False, has_it=has_data)
        _embed(checks_frame)

        # Warning label
        warn_frame = tk.Frame(scroll_text, bg=BG_DARK)
        tk.Label(warn_frame, text="⚠  Source code (git repo) will NOT be removed.",
                 fg=FG_DIM, bg=BG_DARK, font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(warn_frame, text="    You can always reinstall from the same folder.",
                 fg=FG_DIM, bg=BG_DARK, font=("Segoe UI", 9)).pack(anchor="w")
        _embed(warn_frame)

        # Buttons
        btn_frame = tk.Frame(scroll_text, bg=BG_DARK)

        def _go_back():
            page.destroy()
            self._show_page(0)

        def _do_uninstall():
            # Check at least one thing selected
            selected = {k: v.get() for k, v in vars_dict.items()}
            if not any(selected.values()):
                messagebox.showinfo("Nothing selected", "Select at least one component to remove.")
                return
            # Confirm
            items = [k.replace("_", " ").title() for k, v in selected.items() if v]
            ok = messagebox.askyesno(
                "Confirm Uninstall",
                f"Remove the following?\n\n  • " + "\n  • ".join(items) +
                "\n\nThis cannot be undone.",
                icon="warning")
            if not ok:
                return
            page.destroy()
            self._show_page(2)
            self._clear_log()
            self._progress_subtitle.configure(text="Uninstalling...")
            self.engine = UninstallEngine(install_dir, selected,
                                           self._write_log, self._update_progress,
                                           self._install_done)
            self.engine.start()

        make_button(btn_frame, text="← Back",
                   command=_go_back,
                   bg=BG_INPUT, fg=FG_TEXT, hover_bg=BG_CARD,
                   padx=15, pady=8).pack(side="left")

        make_button(btn_frame, text="  🗑  Uninstall Selected  ",
                   command=_do_uninstall,
                   bg="#5c1a1a", fg="#ff6b6b", hover_bg="#7a2020", hover_fg="#ff6b6b",
                   font=("Segoe UI", 12, "bold"), padx=20, pady=8
                   ).pack(side="right")
        _embed(btn_frame)

        # Block direct text editing but keep embedded widgets interactive
        scroll_text.bind("<Key>", lambda e: "break")
        scroll_text.configure(insertwidth=0)

    def _cancel_install(self):
        if self.engine:
            self.engine.cancel()
        self._cancel_btn.configure(text="Cancelling...", fg=FG_DIM, cursor="")
        self._cancel_btn.unbind("<Button-1>")

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _write_log(self, text, tag="dim"):
        self.after(0, self._do_log, text, tag)

    def _do_log(self, text, tag):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", text + "\n", tag)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _update_progress(self, pct, label):
        self.after(0, self._do_progress, pct, label)

    def _do_progress(self, pct, label):
        self._progress_bar["value"] = pct
        self._progress_pct.configure(text=f"{int(pct)}%")
        self._progress_subtitle.configure(text=label)

    def _install_done(self, success, error):
        self.after(0, self._do_done, success, error)

    def _do_done(self, success, error):
        self._cancel_btn.configure(text="Cancel", fg=ERROR_C, cursor="hand2")
        self._cancel_btn.bind("<Button-1>", lambda e: self._cancel_install())
        is_uninstall = isinstance(self.engine, UninstallEngine)
        if success:
            self._done_icon.configure(text="✔", foreground=ACCENT)
            if is_uninstall:
                self._done_title.configure(text="Uninstall Complete!")
                self._done_subtitle.configure(text="Selected components have been removed.")
                self._done_launch_btn.pack_forget()
                self._done_folder_btn.pack_forget()
            else:
                self._done_title.configure(text="Complete!")
                self._done_subtitle.configure(text="Open-Sable is ready to use.")
                self._done_launch_btn.pack(side="left", padx=5)
                self._done_folder_btn.pack(side="left", padx=5)
            self._show_page(3)
        else:
            self._done_icon.configure(text="✘", foreground=ERROR_C)
            self._done_title.configure(text="Something went wrong")
            self._done_subtitle.configure(text=str(error)[:120] if error else "Unknown error")
            self._show_page(3)

    def _launch(self):
        install_dir = self.install_dir_var.get()
        start_sh = os.path.join(install_dir, "start.sh")

        if IS_MAC:
            # On macOS, the .app bundle handles both backend start and Electron launch
            app_bundle = os.path.expanduser("~/Applications/Open-Sable.app")
            if os.path.isdir(app_bundle):
                subprocess.Popen(["open", app_bundle])
                self.after(1000, self.destroy)
                return

        # Start backend (with DESKTOP_ENABLED=false to avoid double Electron)
        launch_env = {**os.environ, "DESKTOP_ENABLED": "false"}
        if IS_WIN:
            bat = os.path.join(install_dir, "opensable.bat")
            if os.path.isfile(bat):
                subprocess.Popen(["cmd", "/c", "start", bat], cwd=install_dir)
            else:
                subprocess.Popen(["cmd", "/c", "start", "cmd", "/k",
                                  f'cd /d "{install_dir}" && venv\\Scripts\\activate.bat && python -m opensable'],
                                 cwd=install_dir)
        elif os.path.isfile(start_sh):
            subprocess.Popen(["bash", start_sh, "start"], cwd=install_dir, env=launch_env)
        else:
            subprocess.Popen(["bash", "-c",
                              f"cd '{install_dir}' && source venv/bin/activate && python -m opensable"],
                             cwd=install_dir, env=launch_env)

        # Launch desktop app (Electron)
        def _open_desktop():
            time.sleep(4)  # wait for backend to start
            electron = os.path.join(install_dir, "desktop", "node_modules", ".bin", "electron")
            if os.path.isfile(electron):
                subprocess.Popen([electron, os.path.join(install_dir, "desktop")],
                                 cwd=install_dir,
                                 env={**os.environ,
                                      "WEBCHAT_PORT": "8789",
                                      "WEBCHAT_HOST": "localhost"})
            else:
                import webbrowser
                webbrowser.open("http://127.0.0.1:8789")
        threading.Thread(target=_open_desktop, daemon=True).start()
        self.after(1000, self.destroy)

    def _open_folder(self):
        d = self.install_dir_var.get()
        if not os.path.isdir(d):
            messagebox.showwarning("Folder not found", f"The folder does not exist:\n{d}")
            return
        if IS_WIN:
            os.startfile(d)
        elif IS_MAC:
            subprocess.Popen(["open", d])
        else:
            subprocess.Popen(["xdg-open", d])

    def _start_over(self):
        self._build_pages()
        self._show_page(0)


# ════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = InstallerApp()
    app.mainloop()
