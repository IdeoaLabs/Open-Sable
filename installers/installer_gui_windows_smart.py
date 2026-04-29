#!/usr/bin/env python3
"""Open-Sable Smart Windows Installer.

Windows-only setup wizard focused on x86/x64 machines.
- Installs required dependencies (Python, Git, Node.js, Ollama optional)
- Clones/updates Open-Sable
- Builds Python environment and optional web assets
- Uses safe auto-fix heuristics and optional LLM guidance (Ollama or OpenAI-compatible API)
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "Open-Sable"
APP_TAGLINE = "Smart Windows Setup Wizard"
APP_VERSION = "2.0.0-win-smart"
REPO_URL = "https://github.com/IdeoaLabs/Open-Sable.git"
REPO_BRANCH = "master"
OLLAMA_WIN_URL = "https://ollama.com/download/OllamaSetup.exe"

IS_WIN = sys.platform == "win32"
_NO_WINDOW = 0x08000000 if IS_WIN else 0

BG_DARK = "#0d1117"
BG_CARD = "#161b22"
BG_INPUT = "#21262d"
FG_TEXT = "#e6edf3"
FG_DIM = "#7d8590"
ACCENT = "#00d4aa"
ACCENT_HOVER = "#00f0c0"
ERROR_C = "#f85149"
WARNING_C = "#d29922"

MODELS = [
    ("qwen3.5:0.8b", "Qwen 3.5 0.8B - fastest, 500 MB"),
    ("qwen3.5:1.5b", "Qwen 3.5 1.5B - fast, 1 GB"),
    ("qwen3.5:4b", "Qwen 3.5 4B - balanced, 2.5 GB"),
    ("qwen3.5:8b", "Qwen 3.5 8B - recommended, 5 GB"),
    ("deepseek-r1:8b", "DeepSeek-R1 8B - strong reasoning, 5 GB"),
]

WINGET_INSTALLS = {
    "python": ["winget", "install", "--id", "Python.Python.3.12", "-e",
               "--accept-source-agreements", "--accept-package-agreements"],
    "git": ["winget", "install", "--id", "Git.Git", "-e",
            "--accept-source-agreements", "--accept-package-agreements"],
    "node": ["winget", "install", "--id", "OpenJS.NodeJS.LTS", "-e",
             "--accept-source-agreements", "--accept-package-agreements"],
}

REQUIRED_WEB_PROJECTS = {
    "dashboard": {
        "build_markers": ["dist/index.html"],
    },
    "desktop": {
        "build_markers": ["dist/index.html"],
    },
    "aggr": {
        "build_markers": ["dist/index.html"],
    },
    "sable_dev": {
        "build_markers": [".next/BUILD_ID"],
    },
}


# ---------------------------------------------------------------------------
# Assets and utilities
# ---------------------------------------------------------------------------


def resource_path(relative: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


ASSETS_DIR = resource_path("assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
ICON_ICO = os.path.join(ASSETS_DIR, "icon.ico")
ICON_PNG = LOGO_PATH


def make_button(parent, text, command, bg=BG_INPUT, fg=FG_TEXT,
                hover_bg=BG_CARD, hover_fg=None, font=("Segoe UI", 10),
                padx=16, pady=8, cursor="hand2", **kw):
    if hover_fg is None:
        hover_fg = fg
    lbl = tk.Label(parent, text=text, font=font, bg=bg, fg=fg,
                   padx=padx, pady=pady, cursor=cursor, **kw)
    lbl.bind("<Enter>", lambda e: lbl.configure(bg=hover_bg, fg=hover_fg))
    lbl.bind("<Leave>", lambda e: lbl.configure(bg=bg, fg=fg))
    lbl.bind("<Button-1>", lambda e: command())
    return lbl


def machine_arch() -> str:
    m = platform.machine().lower()
    if m in ("amd64", "x86_64"):
        return "x64"
    if m in ("x86", "i386", "i686"):
        return "x86"
    # Most Windows ARM systems still run x64 emulation for this stack.
    if "arm" in m:
        return "x64"
    return "x64"


def default_install_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "OpenSable")


def refresh_windows_path() -> None:
    if not IS_WIN:
        return
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "[Environment]::GetEnvironmentVariable('Path','Machine')+';'+[Environment]::GetEnvironmentVariable('Path','User')",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8, creationflags=_NO_WINDOW)
        if r.returncode == 0 and r.stdout.strip():
            os.environ["PATH"] = r.stdout.strip()
    except Exception:
        pass


def run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 1800) -> Tuple[int, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
        env=env,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        out = (out or "") + "\n[timeout]"
        return 124, out
    return proc.returncode, out or ""


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def find_python() -> Optional[List[str]]:
    candidates = [["python"], ["py", "-3.12"], ["py", "-3.11"], ["py", "-3"]]
    for c in candidates:
        try:
            code, out = run_cmd(c + ["--version"], timeout=8)
            if code == 0:
                m = re.search(r"Python\s+(\d+)\.(\d+)", out)
                if m and int(m.group(1)) >= 3 and int(m.group(2)) >= 11:
                    return c
        except Exception:
            continue
    return None


def find_git() -> bool:
    try:
        return run_cmd(["git", "--version"], timeout=8)[0] == 0
    except Exception:
        return False


def find_node() -> bool:
    try:
        code, out = run_cmd(["node", "--version"], timeout=8)
        if code != 0:
            return False
        major = int(out.strip().lstrip("v").split(".")[0])
        return major >= 20
    except Exception:
        return False


def find_ollama() -> bool:
    try:
        return run_cmd(["ollama", "--version"], timeout=8)[0] == 0
    except Exception:
        return False


def ollama_api_ready() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

class LLMFixAdvisor:
    def __init__(self, config: Dict[str, Any], log_cb: Callable[[str, str], None]):
        self.config = config
        self.log = log_cb

    def suggest_actions(self, step: str, error_text: str) -> List[Dict[str, str]]:
        provider = self.config.get("llm_provider", "none")
        if provider == "none":
            return []
        if provider == "ollama" and not (find_ollama() and ollama_api_ready()):
            self.log("  LLM advisor unavailable: Ollama is not ready.", "warning")
            return []
        try:
            prompt = self._build_prompt(step, error_text)
            raw = self._request(provider, prompt)
            return self._parse_actions(raw)
        except Exception as e:
            self.log(f"  LLM advisor failed: {e}", "warning")
            return []

    def _build_prompt(self, step: str, error_text: str) -> str:
        return (
            "You are an installer auto-fix planner for Windows.\n"
            "Return JSON only: {\"actions\":[...]} with max 3 actions.\n"
            "Allowed action types: retry, skip_optional, refresh_path, run.\n"
            "For run action, command MUST start with one of: winget, git, python, py, npm, ollama.\n"
            "Each action shape: {\"type\":\"...\",\"reason\":\"...\",\"command\":\"...\"}.\n"
            f"Failed step: {step}\n"
            f"Error output:\n{error_text[:3500]}\n"
        )

    def _request(self, provider: str, prompt: str) -> str:
        if provider == "ollama":
            model = self.config.get("llm_model") or self.config.get("model") or "qwen3.5:0.8b"
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return payload.get("response", "")

        base_url = (self.config.get("api_base_url") or "").rstrip("/")
        api_key = self.config.get("api_key") or ""
        model = self.config.get("api_model") or "gpt-4.1-mini"
        if not base_url or not api_key:
            raise RuntimeError("Missing API base URL or API key")

        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]

    def _parse_actions(self, text: str) -> List[Dict[str, str]]:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return []
        data = json.loads(m.group(0))
        actions = data.get("actions") or []
        if not isinstance(actions, list):
            return []
        out: List[Dict[str, str]] = []
        for a in actions[:3]:
            if not isinstance(a, dict):
                continue
            typ = str(a.get("type", "")).strip().lower()
            reason = str(a.get("reason", "")).strip()
            cmd = str(a.get("command", "")).strip()
            if typ in {"retry", "skip_optional", "refresh_path", "run"}:
                out.append({"type": typ, "reason": reason, "command": cmd})
        return out


# ---------------------------------------------------------------------------
# Installer engine
# ---------------------------------------------------------------------------

class SmartWindowsInstallerEngine:
    def __init__(self, config: Dict[str, Any], log_cb, progress_cb, done_cb):
        self.config = config
        self.log = log_cb
        self.progress = progress_cb
        self.done = done_cb
        self.cancelled = False
        self.llm = LLMFixAdvisor(config, log_cb)

    @property
    def install_dir(self) -> str:
        return self.config["install_dir"]

    @property
    def venv_python(self) -> str:
        return os.path.join(self.install_dir, ".venv", "Scripts", "python.exe")

    @property
    def venv_pip(self) -> str:
        return os.path.join(self.install_dir, ".venv", "Scripts", "pip.exe")

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self):
        self.cancelled = True

    def _run(self):
        if not IS_WIN:
            self.done(False, "This installer only supports Windows.")
            return

        steps: List[Tuple[str, Callable[[], None], bool]] = [
            ("Preparing install folders", self._prepare_dirs, False),
            ("Checking architecture and permissions", self._check_windows_context, False),
            ("Installing base dependencies", self._install_base_dependencies, False),
            ("Cloning or updating Open-Sable", self._fetch_repo, False),
            ("Creating Python virtual environment", self._create_venv, False),
            ("Installing Python libraries", self._install_python_libs, False),
            ("Installing required web and desktop assets", self._install_node_assets, False),
            ("Configuring LLM provider", self._configure_llm, True),
            ("Installing and preparing Ollama", self._prepare_ollama, True),
            ("Final verification", self._verify, False),
        ]

        try:
            total = len(steps)
            for idx, (name, func, optional) in enumerate(steps, start=1):
                if self.cancelled:
                    self.done(False, "Cancelled")
                    return
                self.log(f"\n━━━ Step {idx}/{total}: {name}", "step")
                self.progress((idx - 1) / total * 100, name)
                self._execute_with_auto_fix(name, func, optional=optional)

            self.progress(100, "Completed")
            self.log("\n✔ Smart installation complete.", "success")
            self.done(True, None)
        except Exception as e:
            self.log(f"\n✘ Installation failed: {e}", "error")
            self.done(False, str(e))

    def _execute_with_auto_fix(self, step_name: str, func: Callable[[], None], optional: bool):
        try:
            func()
            return
        except Exception as first_err:
            self.log(f"  Initial failure: {first_err}", "warning")
            if not self.config.get("auto_fix", True):
                if optional:
                    self.log("  Optional step skipped (auto-fix disabled).", "warning")
                    return
                raise

            if self._heuristic_fix(step_name, str(first_err)):
                try:
                    func()
                    self.log("  ✔ Auto-fix succeeded on retry.", "ok")
                    return
                except Exception as retry_err:
                    self.log(f"  Retry still failed: {retry_err}", "warning")

            if self.config.get("llm_provider", "none") != "none":
                if self._llm_fix(step_name, str(first_err)):
                    try:
                        func()
                        self.log("  ✔ LLM-guided fix succeeded on retry.", "ok")
                        return
                    except Exception as retry_err:
                        self.log(f"  Retry after LLM fix failed: {retry_err}", "warning")

            if optional:
                self.log("  Optional step skipped after auto-fix attempts.", "warning")
                return
            raise first_err

    def _heuristic_fix(self, step: str, error: str) -> bool:
        fixed = False
        low = error.lower()

        if "not found" in low or "no such file" in low or "is not recognized" in low:
            refresh_windows_path()
            self.log("  Heuristic fix: refreshed Windows PATH.", "dim")
            fixed = True

        if "pip" in low and "ssl" in low and os.path.isfile(self.venv_python):
            self._run_logged([self.venv_python, "-m", "pip", "install", "--upgrade", "certifi"], check=False)
            fixed = True

        if "git" in low and command_exists("winget"):
            self._run_logged(WINGET_INSTALLS["git"], check=False)
            refresh_windows_path()
            fixed = True

        if ("node" in low or "npm" in low) and command_exists("winget"):
            self._run_logged(WINGET_INSTALLS["node"], check=False)
            refresh_windows_path()
            fixed = True

        if "access is denied" in low:
            self.log("  Permission warning detected. Switching to user-writable paths only.", "warning")
            user_dir = default_install_dir()
            self.config["install_dir"] = user_dir
            fixed = True

        return fixed

    def _llm_fix(self, step: str, error: str) -> bool:
        actions = self.llm.suggest_actions(step, error)
        if not actions:
            return False
        self.log("  Applying LLM auto-fix suggestions...", "dim")
        changed = False
        for action in actions:
            typ = action.get("type", "")
            reason = action.get("reason", "")
            cmd = action.get("command", "")
            if reason:
                self.log(f"    - {reason}", "dim")
            if typ == "refresh_path":
                refresh_windows_path()
                changed = True
            elif typ == "retry":
                changed = True
            elif typ == "skip_optional":
                changed = True
            elif typ == "run" and cmd:
                if self._is_safe_llm_command(cmd):
                    self._run_logged(cmd.split(), check=False)
                    changed = True
                else:
                    self.log(f"    Unsafe LLM command ignored: {cmd}", "warning")
        return changed

    def _is_safe_llm_command(self, command: str) -> bool:
        allowed = ("winget ", "git ", "python ", "py ", "npm ", "ollama ")
        cmd = command.strip().lower()
        return any(cmd.startswith(a) for a in allowed)

    def _run_logged(self, cmd: List[str], cwd: Optional[str] = None, check: bool = True,
                    timeout: int = 3600) -> str:
        self.log("  $ " + " ".join(cmd), "dim")
        code, out = run_cmd(cmd, cwd=cwd, timeout=timeout)
        for line in out.splitlines():
            if line.strip():
                self.log("    " + line, "dim")
        if check and code != 0:
            raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")
        return out

    def _prepare_dirs(self):
        safe_mkdir(self.install_dir)
        for d in ("logs", "data", "models", "episodes", "config"):
            safe_mkdir(os.path.join(self.install_dir, d))
        self.log("  ✔ Directories ready.", "ok")

    def _check_windows_context(self):
        arch = machine_arch()
        self.log(f"  Windows architecture: {arch}", "ok")
        self.log(f"  Python process architecture: {platform.architecture()[0]}", "dim")

        if not os.access(self.install_dir, os.W_OK):
            raise RuntimeError("Install directory is not writable")

        if not command_exists("winget"):
            self.log("  ⚠ winget not found. Automatic dependency installation may be limited.", "warning")
        else:
            self.log("  ✔ winget available.", "ok")

    def _install_base_dependencies(self):
        py = find_python()
        if not py:
            if command_exists("winget"):
                self.log("  Installing Python 3.12 via winget...", "dim")
                self._run_logged(WINGET_INSTALLS["python"], check=False)
                refresh_windows_path()
                py = find_python()
        if not py:
            raise RuntimeError("Python 3.11+ is required")
        self.log("  ✔ Python 3.11+ found.", "ok")

        if not find_git():
            if command_exists("winget"):
                self.log("  Installing Git via winget...", "dim")
                self._run_logged(WINGET_INSTALLS["git"], check=False)
                refresh_windows_path()
        if not find_git():
            raise RuntimeError("Git is required")
        self.log("  ✔ Git found.", "ok")

        if not find_node():
            if command_exists("winget"):
                self.log("  Installing Node.js LTS via winget...", "dim")
                self._run_logged(WINGET_INSTALLS["node"], check=False)
                refresh_windows_path()

        if find_node():
            self.log("  ✔ Node.js 20+ found.", "ok")
        else:
            raise RuntimeError("Node.js 20+ is required for dashboard/desktop/sable_dev builds")

    def _fetch_repo(self):
        git_dir = os.path.join(self.install_dir, ".git")
        pyproject = os.path.join(self.install_dir, "pyproject.toml")

        if os.path.isdir(git_dir):
            self.log("  Existing repository found. Pulling latest changes...", "dim")
            self._run_logged(["git", "fetch", "origin"], cwd=self.install_dir, check=False)
            self._run_logged(["git", "checkout", REPO_BRANCH], cwd=self.install_dir, check=False)
            self._run_logged(["git", "pull", "--rebase", "origin", REPO_BRANCH], cwd=self.install_dir, check=False)
            self.log("  ✔ Repository updated.", "ok")
            return

        if os.path.isfile(pyproject):
            self.log("  Existing source folder detected (without .git).", "ok")
            return

        parent = os.path.dirname(self.install_dir)
        safe_mkdir(parent)
        tmp_clone = self.install_dir + ".tmp_clone"
        if os.path.isdir(tmp_clone):
            shutil.rmtree(tmp_clone, ignore_errors=True)

        self.log("  Cloning Open-Sable repository...", "dim")
        self._run_logged(["git", "clone", "--branch", REPO_BRANCH, REPO_URL, tmp_clone], cwd=parent)

        for item in os.listdir(tmp_clone):
            src = os.path.join(tmp_clone, item)
            dst = os.path.join(self.install_dir, item)
            if os.path.exists(dst):
                continue
            shutil.move(src, dst)
        shutil.rmtree(tmp_clone, ignore_errors=True)
        self.log("  ✔ Repository cloned.", "ok")

    def _create_venv(self):
        py = find_python()
        if not py:
            raise RuntimeError("Python 3.11+ unavailable")

        venv_dir = os.path.join(self.install_dir, ".venv")
        if not os.path.isdir(venv_dir):
            self.log("  Creating virtual environment...", "dim")
            self._run_logged(py + ["-m", "venv", ".venv"], cwd=self.install_dir)

        if not os.path.isfile(self.venv_python):
            raise RuntimeError("Virtual environment creation failed")
        self.log("  ✔ Virtual environment ready.", "ok")

    def _install_python_libs(self):
        self._run_logged([self.venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=self.install_dir, check=False)
        self._run_logged([self.venv_python, "-m", "pip", "install", "-e", ".[core]"], cwd=self.install_dir)

        req = os.path.join(self.install_dir, "requirements.txt")
        if os.path.isfile(req):
            self._run_logged([self.venv_python, "-m", "pip", "install", "-r", "requirements.txt"], cwd=self.install_dir, check=False)

        extra_req = os.path.join(self.install_dir, "requirements-extras.txt")
        if os.path.isfile(extra_req):
            self._run_logged([self.venv_python, "-m", "pip", "install", "-r", "requirements-extras.txt"], cwd=self.install_dir, check=False)

        lock_req = os.path.join(self.install_dir, "requirements-lock.txt")
        if os.path.isfile(lock_req):
            self._run_logged([self.venv_python, "-m", "pip", "install", "-r", "requirements-lock.txt"], cwd=self.install_dir, check=False)

        trading_req = os.path.join(self.install_dir, "requirements-trading.txt")
        if os.path.isfile(trading_req):
            self._run_logged([self.venv_python, "-m", "pip", "install", "-r", "requirements-trading.txt"], cwd=self.install_dir, check=False)

        self.log("  ✔ Python libraries installed.", "ok")

    def _install_node_assets(self):
        if not find_node():
            raise RuntimeError("Node.js 20+ not available for required builds")

        for project in REQUIRED_WEB_PROJECTS.keys():
            pkg = os.path.join(self.install_dir, project, "package.json")
            if not os.path.isfile(pkg):
                raise RuntimeError(f"Required project missing: {project}")

            self.log(f"  Building {project}...", "dim")
            project_dir = os.path.join(self.install_dir, project)
            self._run_logged(["npm", "install", "--legacy-peer-deps"], cwd=project_dir)
            self._run_logged(["npm", "run", "build"], cwd=project_dir)
            self._verify_project_build_output(project, project_dir)

        self.log("  ✔ Required web and desktop assets built successfully.", "ok")

    def _verify_project_build_output(self, project: str, project_dir: str):
        markers = REQUIRED_WEB_PROJECTS.get(project, {}).get("build_markers", [])
        for marker in markers:
            if not os.path.exists(os.path.join(project_dir, marker)):
                raise RuntimeError(f"Build output missing for {project}: {marker}")

    def _configure_llm(self):
        provider = self.config.get("llm_provider", "none")
        env_path = os.path.join(self.install_dir, ".env")

        env_lines = []
        if os.path.isfile(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                env_lines = [line.rstrip("\n") for line in f]

        values: Dict[str, str] = {}
        for line in env_lines:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()

        if provider == "ollama":
            values["LLM_PROVIDER"] = "ollama"
            values["OLLAMA_BASE_URL"] = "http://localhost:11434"
            values["OLLAMA_MODEL"] = self.config.get("model", "qwen3.5:0.8b")
        elif provider == "api":
            values["LLM_PROVIDER"] = "api"
            values["API_BASE_URL"] = self.config.get("api_base_url", "")
            values["API_KEY"] = self.config.get("api_key", "")
            values["API_MODEL"] = self.config.get("api_model", "")
        else:
            values["LLM_PROVIDER"] = "none"

        values.setdefault("WEBCHAT_PORT", "8789")
        values.setdefault("WEBCHAT_HOST", "localhost")
        values.setdefault("DESKTOP_ENABLED", "true")

        with open(env_path, "w", encoding="utf-8") as f:
            for k in sorted(values.keys()):
                f.write(f"{k}={values[k]}\n")

        self.log(f"  ✔ LLM provider configured: {provider}", "ok")

    def _prepare_ollama(self):
        if self.config.get("llm_provider") != "ollama" and not self.config.get("install_ollama", True):
            self.log("  Ollama setup skipped.", "warning")
            return

        if not find_ollama() and self.config.get("install_ollama", True):
            self.log("  Downloading Ollama installer...", "dim")
            dl = os.path.join(tempfile.gettempdir(), "OllamaSetup.exe")
            urllib.request.urlretrieve(OLLAMA_WIN_URL, dl)
            self._run_logged([dl, "/VERYSILENT", "/NORESTART"], check=False)
            refresh_windows_path()

        if not find_ollama():
            self.log("  ⚠ Ollama not installed. Continuing without local LLM.", "warning")
            return

        if not ollama_api_ready():
            self.log("  Starting Ollama service...", "dim")
            subprocess.Popen(["ollama", "serve"], creationflags=_NO_WINDOW,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(12):
                if ollama_api_ready():
                    break
                time.sleep(1)

        if self.config.get("llm_provider") == "ollama" and ollama_api_ready() and self.config.get("pull_model", True):
            model = self.config.get("model", "qwen3.5:0.8b")
            self.log(f"  Pulling Ollama model: {model}", "dim")
            self._run_logged(["ollama", "pull", model], check=False, timeout=7200)

        if ollama_api_ready():
            self.log("  ✔ Ollama API ready.", "ok")
        else:
            self.log("  ⚠ Ollama installed but API not reachable yet.", "warning")

    def _verify(self):
        checks = []
        checks.append(("Python venv", os.path.isfile(self.venv_python)))
        checks.append(("Git", find_git()))
        checks.append(("Node.js 20+", find_node()))
        checks.append(("PyProject", os.path.isfile(os.path.join(self.install_dir, "pyproject.toml"))))

        for project in REQUIRED_WEB_PROJECTS.keys():
            project_dir = os.path.join(self.install_dir, project)
            markers = REQUIRED_WEB_PROJECTS[project]["build_markers"]
            ok = all(os.path.exists(os.path.join(project_dir, m)) for m in markers)
            checks.append((f"Build artifact: {project}", ok))

        failed = 0
        for label, ok in checks:
            if ok:
                self.log(f"  ✔ {label}", "ok")
            else:
                self.log(f"  ✘ {label}", "error")
                failed += 1

        if failed:
            raise RuntimeError(f"Verification failed with {failed} issue(s)")

        marker = os.path.join(self.install_dir, ".installed")
        with open(marker, "w", encoding="utf-8") as f:
            f.write("smart-windows-installer=true\n")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SmartWindowsInstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} - Smart Windows Installer")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        w, h = 760, 680
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(760, 680)

        try:
            if IS_WIN and os.path.isfile(ICON_ICO):
                self.iconbitmap(ICON_ICO)
            elif os.path.isfile(ICON_PNG):
                img = tk.PhotoImage(file=ICON_PNG)
                self.iconphoto(True, img)
        except Exception:
            pass

        self._logo_img = None
        try:
            if os.path.isfile(LOGO_PATH):
                self._logo_img = tk.PhotoImage(file=LOGO_PATH)
                if self._logo_img.width() > 130:
                    factor = max(1, self._logo_img.width() // 130)
                    self._logo_img = self._logo_img.subsample(factor, factor)
        except Exception:
            pass

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background=BG_DARK, foreground=FG_TEXT, font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=BG_DARK, foreground=FG_TEXT, font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel", background=BG_DARK, foreground=FG_DIM, font=("Segoe UI", 12))
        style.configure("Small.TLabel", background=BG_DARK, foreground=FG_DIM, font=("Segoe UI", 9))
        style.configure("TRadiobutton", background=BG_DARK, foreground=FG_TEXT)
        style.configure("TCheckbutton", background=BG_DARK, foreground=FG_TEXT)
        style.configure("Horizontal.TProgressbar", troughcolor=BG_INPUT, background=ACCENT, thickness=8)

        self.install_dir_var = tk.StringVar(value=default_install_dir())
        self.model_var = tk.StringVar(value="qwen3.5:0.8b")
        self.arch_var = tk.StringVar(value=machine_arch())
        self.install_ollama_var = tk.BooleanVar(value=True)
        self.install_node_var = tk.BooleanVar(value=True)
        self.pull_model_var = tk.BooleanVar(value=True)
        self.auto_fix_var = tk.BooleanVar(value=True)

        self.llm_provider_var = tk.StringVar(value="ollama")
        self.api_base_url_var = tk.StringVar(value="")
        self.api_key_var = tk.StringVar(value="")
        self.api_model_var = tk.StringVar(value="gpt-4.1-mini")

        self.engine: Optional[SmartWindowsInstallerEngine] = None
        self._pages: List[tk.Frame] = []
        self._current_page = 0
        self._review_text: Optional[tk.Text] = None

        self._container = tk.Frame(self, bg=BG_DARK)
        self._container.pack(fill="both", expand=True)

        self._build_pages()
        self._show_page(0)

        # Keyboard fallback for high-DPI layouts where action buttons may be clipped.
        self.bind_all("<Alt-n>", lambda e: self._hotkey_next())
        self.bind_all("<Alt-N>", lambda e: self._hotkey_next())
        self.bind_all("<Alt-i>", lambda e: self._hotkey_install())
        self.bind_all("<Alt-I>", lambda e: self._hotkey_install())

    def _build_pages(self):
        for p in self._pages:
            p.destroy()
        self._pages = [
            self._build_welcome_page(),
            self._build_config_page(),
            self._build_review_page(),
            self._build_progress_page(),
            self._build_done_page(),
        ]

    def _show_page(self, idx: int):
        for p in self._pages:
            p.pack_forget()
        self._current_page = idx
        self._pages[idx].pack(fill="both", expand=True)

    def _build_welcome_page(self) -> tk.Frame:
        page = tk.Frame(self._container, bg=BG_DARK)
        content = tk.Frame(page, bg=BG_DARK)
        content.place(relx=0.5, rely=0.5, anchor="center")

        if self._logo_img:
            tk.Label(content, image=self._logo_img, bg=BG_DARK).pack(pady=(0, 10))

        ttk.Label(content, text=APP_NAME, style="Title.TLabel").pack()
        ttk.Label(content, text=APP_TAGLINE, style="Subtitle.TLabel").pack(pady=(2, 10))
        ttk.Label(content, text=f"Version {APP_VERSION}", style="Small.TLabel").pack(pady=(0, 14))

        card = tk.Frame(content, bg=BG_CARD, padx=20, pady=14)
        card.pack(fill="x", padx=20, pady=(0, 12))

        lines = [
            f"Architecture target: {self.arch_var.get()}",
            "Supported modes: x86 and x64 Windows",
            "Auto installs dependencies with silent commands",
            "Mandatory builds: desktop, dashboard, and sable_dev",
            "Smart auto-fix can use Ollama or external API when errors appear",
            "Can continue without LLM when no issues are detected",
        ]
        for line in lines:
            tk.Label(card, text="- " + line, fg=FG_TEXT, bg=BG_CARD, font=("Segoe UI", 10), anchor="w").pack(fill="x", pady=1)

        make_button(content, text="  Start Smart Setup  ",
                    command=lambda: self._show_page(1),
                    bg=ACCENT, fg=BG_DARK, hover_bg=ACCENT_HOVER, hover_fg=BG_DARK,
                    font=("Segoe UI", 13, "bold"), padx=28, pady=10).pack(pady=(0, 8))

        make_button(content, text="Exit",
                    command=self.destroy,
                    bg=BG_INPUT, fg=FG_TEXT, hover_bg=BG_CARD,
                    font=("Segoe UI", 10), padx=14, pady=6).pack()

        return page

    def _build_config_page(self) -> tk.Frame:
        page = tk.Frame(self._container, bg=BG_DARK)

        header = tk.Frame(page, bg=BG_DARK)
        header.pack(fill="x", padx=30, pady=(20, 10))
        ttk.Label(header, text="Configuration", style="Title.TLabel").pack(anchor="w")

        btn = tk.Frame(page, bg=BG_DARK)
        btn.pack(side="bottom", fill="x", padx=30, pady=(5, 18))
        ttk.Label(btn, text="Tip: Alt+N = Next", style="Small.TLabel").pack(side="left", padx=(8, 0))
        make_button(btn, text="<- Back", command=lambda: self._show_page(0),
                    bg=BG_INPUT, fg=FG_TEXT, hover_bg=BG_CARD,
                    padx=14, pady=8).pack(side="left")
        make_button(btn, text="  Next -> Review  ", command=self._go_review,
                    bg=ACCENT, fg=BG_DARK, hover_bg=ACCENT_HOVER, hover_fg=BG_DARK,
                    font=("Segoe UI", 12, "bold"), padx=28, pady=8).pack(side="right")

        body = tk.Frame(page, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=30)

        # Install location
        ttk.Label(body, text="Install location").pack(anchor="w")
        loc_row = tk.Frame(body, bg=BG_DARK)
        loc_row.pack(fill="x", pady=(4, 12))
        tk.Entry(loc_row, textvariable=self.install_dir_var, font=("Consolas", 10),
                 bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_TEXT,
                 relief="flat", bd=6).pack(side="left", fill="x", expand=True)
        make_button(loc_row, text="Browse", command=self._browse_dir,
                    bg=BG_INPUT, fg=FG_TEXT, hover_bg=BG_CARD,
                    font=("Segoe UI", 9), padx=10, pady=5).pack(side="left", padx=(6, 0))

        # Status
        status = tk.Frame(body, bg=BG_CARD, padx=12, pady=10)
        status.pack(fill="x", pady=(0, 12))
        deps = [
            ("Python 3.11+", find_python() is not None),
            ("Git", find_git()),
            ("Node.js 20+", find_node()),
            ("Ollama", find_ollama()),
            ("winget", command_exists("winget")),
        ]
        for name, ok in deps:
            color = ACCENT if ok else WARNING_C
            icon = "✔" if ok else "○"
            tk.Label(status, text=f"{icon} {name}", fg=color, bg=BG_CARD,
                     font=("Segoe UI", 10)).pack(anchor="w")

        # Model
        ttk.Label(body, text="Preferred local model (used when Ollama LLM mode is selected)").pack(anchor="w")
        model_menu = ttk.Combobox(body, textvariable=self.model_var, values=[m[0] for m in MODELS], state="readonly")
        model_menu.pack(fill="x", pady=(4, 12))

        # LLM provider section
        llm_card = tk.Frame(body, bg=BG_CARD, padx=12, pady=10)
        llm_card.pack(fill="x", pady=(0, 12))
        tk.Label(llm_card, text="Auto-fix intelligence", fg=FG_TEXT, bg=BG_CARD,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))

        ttk.Radiobutton(llm_card, text="Use local Ollama for auto-fix guidance",
                        variable=self.llm_provider_var, value="ollama").pack(anchor="w")
        ttk.Radiobutton(llm_card, text="Use external OpenAI-compatible API",
                        variable=self.llm_provider_var, value="api").pack(anchor="w")
        ttk.Radiobutton(llm_card, text="No LLM auto-fix (heuristics only)",
                        variable=self.llm_provider_var, value="none").pack(anchor="w")

        form = tk.Frame(llm_card, bg=BG_CARD)
        form.pack(fill="x", pady=(8, 0))
        tk.Label(form, text="API base URL", fg=FG_TEXT, bg=BG_CARD, font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        tk.Entry(form, textvariable=self.api_base_url_var, font=("Consolas", 9),
                 bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_TEXT,
                 relief="flat", bd=4).grid(row=0, column=1, sticky="ew", pady=2)
        tk.Label(form, text="API key", fg=FG_TEXT, bg=BG_CARD, font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=2)
        tk.Entry(form, textvariable=self.api_key_var, show="*", font=("Consolas", 9),
                 bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_TEXT,
                 relief="flat", bd=4).grid(row=1, column=1, sticky="ew", pady=2)
        tk.Label(form, text="API model", fg=FG_TEXT, bg=BG_CARD, font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=2)
        tk.Entry(form, textvariable=self.api_model_var, font=("Consolas", 9),
                 bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_TEXT,
                 relief="flat", bd=4).grid(row=2, column=1, sticky="ew", pady=2)
        form.grid_columnconfigure(1, weight=1)

        opts = tk.Frame(body, bg=BG_DARK)
        opts.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(opts, text="Install Ollama if missing", variable=self.install_ollama_var).pack(anchor="w")
        ttk.Checkbutton(opts, text="Pull selected local model", variable=self.pull_model_var).pack(anchor="w")
        ttk.Checkbutton(opts, text="Install Node.js for required builds (desktop/dashboard/sable_dev)", variable=self.install_node_var).pack(anchor="w")
        ttk.Checkbutton(opts, text="Enable automatic self-healing", variable=self.auto_fix_var).pack(anchor="w")

        return page

    def _build_review_page(self) -> tk.Frame:
        page = tk.Frame(self._container, bg=BG_DARK)

        header = tk.Frame(page, bg=BG_DARK)
        header.pack(fill="x", padx=30, pady=(20, 10))
        ttk.Label(header, text="Review", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Confirm settings before starting installation.", style="Subtitle.TLabel").pack(anchor="w")

        btn = tk.Frame(page, bg=BG_DARK)
        btn.pack(side="bottom", fill="x", padx=30, pady=(0, 18))
        ttk.Label(btn, text="Tip: Alt+I = Install", style="Small.TLabel").pack(side="left", padx=(8, 0))
        make_button(btn, text="<- Back", command=lambda: self._show_page(1),
                    bg=BG_INPUT, fg=FG_TEXT, hover_bg=BG_CARD,
                    padx=14, pady=8).pack(side="left")
        make_button(btn, text="  Install  ", command=self._start_install,
                    bg=ACCENT, fg=BG_DARK, hover_bg=ACCENT_HOVER, hover_fg=BG_DARK,
                    font=("Segoe UI", 12, "bold"), padx=28, pady=8).pack(side="right")

        card = tk.Frame(page, bg=BG_CARD, padx=14, pady=12)
        card.pack(fill="both", expand=True, padx=30, pady=(0, 10))

        self._review_text = tk.Text(card, bg=BG_CARD, fg=FG_TEXT, font=("Consolas", 10),
                                    relief="flat", bd=0, wrap="word", state="disabled",
                                    insertbackground=FG_TEXT)
        self._review_text.pack(fill="both", expand=True)

        return page

    def _build_progress_page(self) -> tk.Frame:
        page = tk.Frame(self._container, bg=BG_DARK)

        top = tk.Frame(page, bg=BG_DARK)
        top.pack(fill="x", padx=30, pady=(20, 8))
        ttk.Label(top, text="Installing", style="Title.TLabel").pack(anchor="w")
        self._progress_subtitle = ttk.Label(top, text="Preparing...", style="Subtitle.TLabel")
        self._progress_subtitle.pack(anchor="w")

        row = tk.Frame(page, bg=BG_DARK)
        row.pack(fill="x", padx=30, pady=(0, 8))
        self._progress_bar = ttk.Progressbar(row, style="Horizontal.TProgressbar", mode="determinate", maximum=100)
        self._progress_bar.pack(side="left", fill="x", expand=True)
        self._progress_pct = ttk.Label(row, text="0%", style="Small.TLabel")
        self._progress_pct.pack(side="right", padx=(8, 0))

        self._log_text = tk.Text(page, bg=BG_INPUT, fg=FG_TEXT, font=("Consolas", 9),
                                 relief="flat", bd=6, wrap="word", state="disabled",
                                 insertbackground=FG_TEXT)
        self._log_text.pack(fill="both", expand=True, padx=30, pady=(0, 10))
        self._log_text.tag_configure("dim", foreground=FG_DIM)
        self._log_text.tag_configure("ok", foreground=ACCENT)
        self._log_text.tag_configure("warning", foreground=WARNING_C)
        self._log_text.tag_configure("error", foreground=ERROR_C)
        self._log_text.tag_configure("success", foreground=ACCENT, font=("Consolas", 10, "bold"))
        self._log_text.tag_configure("step", foreground=ACCENT, font=("Consolas", 10, "bold"))

        self._cancel_btn = make_button(page, text="Cancel", command=self._cancel_install,
                                       bg=BG_INPUT, fg=ERROR_C, hover_bg=BG_CARD, hover_fg=ERROR_C,
                                       padx=12, pady=6)
        self._cancel_btn.pack(pady=(0, 14))

        return page

    def _build_done_page(self) -> tk.Frame:
        page = tk.Frame(self._container, bg=BG_DARK)
        content = tk.Frame(page, bg=BG_DARK)
        content.place(relx=0.5, rely=0.5, anchor="center")

        self._done_icon = ttk.Label(content, text="✔", foreground=ACCENT, font=("Segoe UI", 48))
        self._done_icon.pack()
        self._done_title = ttk.Label(content, text="Installation Complete", style="Title.TLabel")
        self._done_title.pack(pady=(4, 6))
        self._done_subtitle = ttk.Label(content, text="Open-Sable is ready.", style="Subtitle.TLabel")
        self._done_subtitle.pack(pady=(0, 18))

        actions = tk.Frame(content, bg=BG_DARK)
        actions.pack()
        self._open_btn = make_button(actions, text="Open Install Folder", command=self._open_folder,
                                     bg=BG_INPUT, fg=FG_TEXT, hover_bg=BG_CARD,
                                     padx=16, pady=8)
        self._open_btn.pack(side="left", padx=5)
        make_button(actions, text="Close", command=self.destroy,
                    bg=ACCENT, fg=BG_DARK, hover_bg=ACCENT_HOVER, hover_fg=BG_DARK,
                    font=("Segoe UI", 12, "bold"), padx=20, pady=8).pack(side="left", padx=5)

        return page

    def _browse_dir(self):
        d = filedialog.askdirectory(title="Select install location")
        if d:
            self.install_dir_var.set(d)

    def _start_install(self):
        install_dir = self.install_dir_var.get().strip()
        if not install_dir:
            messagebox.showerror("Missing path", "Please select an install location.")
            return

        if not IS_WIN:
            messagebox.showerror("Windows only", "This installer only supports Windows.")
            return

        config = {
            "install_dir": install_dir,
            "model": self.model_var.get(),
            "install_ollama": self.install_ollama_var.get(),
            "install_node": self.install_node_var.get(),
            "pull_model": self.pull_model_var.get(),
            "auto_fix": self.auto_fix_var.get(),
            "llm_provider": self.llm_provider_var.get(),
            "api_base_url": self.api_base_url_var.get().strip(),
            "api_key": self.api_key_var.get().strip(),
            "api_model": self.api_model_var.get().strip(),
        }

        if config["llm_provider"] == "api" and (not config["api_base_url"] or not config["api_key"]):
            ok = messagebox.askyesno(
                "External API not configured",
                "API provider is selected but base URL or API key is empty. Continue with heuristics-only fixes?",
            )
            if not ok:
                return
            config["llm_provider"] = "none"

        self._show_page(3)
        self._clear_log()
        self._progress_subtitle.configure(text="Starting...")

        self.engine = SmartWindowsInstallerEngine(config, self._write_log, self._update_progress, self._install_done)
        self.engine.start()

    def _cancel_install(self):
        if self.engine:
            self.engine.cancel()
            self._cancel_btn.configure(text="Cancelling...", fg=FG_DIM)

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _write_log(self, text: str, tag: str = "dim"):
        self.after(0, self._do_log, text, tag)

    def _do_log(self, text: str, tag: str):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", text + "\n", tag)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _update_progress(self, pct: float, label: str):
        self.after(0, self._do_progress, pct, label)

    def _do_progress(self, pct: float, label: str):
        self._progress_bar["value"] = pct
        self._progress_pct.configure(text=f"{int(pct)}%")
        self._progress_subtitle.configure(text=label)

    def _install_done(self, success: bool, error: Optional[str]):
        self.after(0, self._do_done, success, error)

    def _do_done(self, success: bool, error: Optional[str]):
        self._cancel_btn.configure(text="Cancel", fg=ERROR_C)
        if success:
            self._done_icon.configure(text="✔", foreground=ACCENT)
            self._done_title.configure(text="Installation Complete")
            self._done_subtitle.configure(text="Open-Sable is ready on this Windows system.")
        else:
            self._done_icon.configure(text="✘", foreground=ERROR_C)
            self._done_title.configure(text="Installation Failed")
            self._done_subtitle.configure(text=(error or "Unknown error")[:180])
        self._show_page(4)

    def _go_review(self):
        self._refresh_review()
        self._show_page(2)

    def _refresh_review(self):
        if not self._review_text:
            return
        lines = [
            f"Install dir      : {self.install_dir_var.get().strip()}",
            f"Architecture     : {self.arch_var.get()}",
            f"Local model      : {self.model_var.get()}",
            f"LLM provider     : {self.llm_provider_var.get()}",
            f"Install Ollama   : {self.install_ollama_var.get()}",
            f"Pull model       : {self.pull_model_var.get()}",
            f"Install Node.js  : {self.install_node_var.get()}",
            f"Auto-fix enabled : {self.auto_fix_var.get()}",
            "",
            "Required builds:",
            "  - desktop",
            "  - dashboard",
            "  - aggr",
            "  - sable_dev",
        ]
        if self.llm_provider_var.get() == "api":
            lines.extend([
                "",
                f"API base URL     : {self.api_base_url_var.get().strip() or '(empty)'}",
                f"API model        : {self.api_model_var.get().strip() or '(empty)'}",
            ])

        self._review_text.configure(state="normal")
        self._review_text.delete("1.0", "end")
        self._review_text.insert("end", "\n".join(lines) + "\n")
        self._review_text.configure(state="disabled")

    def _hotkey_next(self):
        if self._current_page == 1:
            self._go_review()

    def _hotkey_install(self):
        if self._current_page == 2:
            self._start_install()

    def _open_folder(self):
        try:
            os.startfile(self.install_dir_var.get())
        except Exception as e:
            messagebox.showerror("Open folder", str(e))


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not IS_WIN:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Windows only", "installer_gui_windows_smart.py only supports Windows.")
        raise SystemExit(1)

    app = SmartWindowsInstallerApp()
    app.mainloop()
