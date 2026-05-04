#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  SableCore,  Universal Installer                                     ║
║  One script to rule them all: Python, Node, dependencies, builds.    ║
║  Works on Linux, macOS, and Windows.                                 ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    python install.py               # Full interactive install
    python install.py --full        # Install everything (no prompts)
    python install.py --core        # Core Python only (minimal)
    python install.py --fix         # Re-check and fix broken installs
    python install.py --status      # Show what's installed / missing
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────
# ANSI Colors (auto-disabled on dumb terminals / Windows without VT)
# ─────────────────────────────────────────────────────────────────────

_COLOR_SUPPORT = (
    hasattr(sys.stdout, "isatty")
    and sys.stdout.isatty()
    and os.environ.get("TERM") != "dumb"
)

if platform.system() == "Windows":
    try:
        import ctypes
        k = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k.SetConsoleMode(k.GetStdHandle(-11), 7)
        _COLOR_SUPPORT = True
    except Exception:
        _COLOR_SUPPORT = False


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR_SUPPORT else text


# Semantic helpers
def bold(t: str) -> str: return _c("1", t)
def dim(t: str) -> str: return _c("2", t)
def green(t: str) -> str: return _c("32", t)
def red(t: str) -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def cyan(t: str) -> str: return _c("36", t)
def magenta(t: str) -> str: return _c("35", t)
def blue(t: str) -> str: return _c("34", t)
def bg_green(t: str) -> str: return _c("42;30", t)
def bg_red(t: str) -> str: return _c("41;97", t)
def bg_yellow(t: str) -> str: return _c("43;30", t)
def bg_cyan(t: str) -> str: return _c("46;30", t)


def ok(msg: str) -> None: print(f"  {green('✔')} {msg}")
def warn(msg: str) -> None: print(f"  {yellow('⚠')} {msg}")
def fail(msg: str) -> None: print(f"  {red('✘')} {msg}")
def info(msg: str) -> None: print(f"  {cyan('→')} {msg}")
def step(msg: str) -> None: print(f"\n{bold(cyan('━━━'))} {bold(msg)}")
def substep(msg: str) -> None: print(f"  {dim('·')} {msg}")


# ─────────────────────────────────────────────────────────────────────
# Platform Detection
# ─────────────────────────────────────────────────────────────────────

OS = platform.system()  # Linux | Darwin | Windows
ARCH = platform.machine()  # x86_64 | arm64 | aarch64
IS_MAC = OS == "Darwin"
IS_LINUX = OS == "Linux"
IS_WIN = OS == "Windows"
ROOT = Path(__file__).resolve().parent

VENV_DIR = ROOT / "venv"
if IS_WIN:
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
    VENV_ACTIVATE = str(VENV_DIR / "Scripts" / "activate")
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP = VENV_DIR / "bin" / "pip"
    VENV_ACTIVATE = f"source {VENV_DIR / 'bin' / 'activate'}"

# Sub-projects that need `npm install`
NPM_PROJECTS: list[dict] = [
    {"name": "Dev Studio",        "dir": "sable_dev",          "build": False, "optional": False},
    {"name": "Dashboard",         "dir": "dashboard",          "build": True,  "optional": False},
    {"name": "Desktop App",       "dir": "desktop",            "build": True,  "optional": True},
    {"name": "Aggr Charts",       "dir": "aggr",               "build": True,  "optional": True},
    {"name": "WhatsApp Bridge",   "dir": "whatsapp-bridge",    "build": False, "optional": True},
]


# ─────────────────────────────────────────────────────────────────────
# Utility: run commands
# ─────────────────────────────────────────────────────────────────────

def run(
    cmd: list[str] | str,
    cwd: Optional[Path] = None,
    capture: bool = False,
    env: Optional[dict] = None,
    timeout: int = 600,
    shell: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command, return CompletedProcess. Raises on failure unless capture=True."""
    merged_env = {**os.environ, **(env or {})}
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=capture,
            text=True,
            timeout=timeout,
            env=merged_env,
            shell=shell,
        )
    except FileNotFoundError:
        if capture:
            return subprocess.CompletedProcess(cmd, 127, "", f"Command not found: {cmd}")
        raise
    except subprocess.TimeoutExpired:
        if capture:
            return subprocess.CompletedProcess(cmd, 124, "", "Timeout")
        raise


def cmd_exists(name: str) -> bool:
    """Check if command is on PATH."""
    return shutil.which(name) is not None


def cmd_version(name: str) -> Optional[str]:
    """Get version string from `name --version`."""
    r = run([name, "--version"], capture=True)
    if r.returncode == 0:
        out = (r.stdout or "").strip()
        m = re.search(r"(\d+\.\d+[\.\d]*)", out)
        return m.group(1) if m else out
    return None


def npm_cmd() -> str:
    """Return npm command name (npm or npm.cmd on Windows)."""
    if IS_WIN and cmd_exists("npm.cmd"):
        return "npm.cmd"
    return "npm"


def npx_cmd() -> str:
    if IS_WIN and cmd_exists("npx.cmd"):
        return "npx.cmd"
    return "npx"


# ─────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────

def print_banner():
    banner = f"""
{bold(cyan('╔══════════════════════════════════════════════════════════════╗'))}
{bold(cyan('║'))}                                                            {bold(cyan('║'))}
{bold(cyan('║'))}   {bold('SableCore,  Universal Installer')}                          {bold(cyan('║'))}
{bold(cyan('║'))}   {dim('Your personal AI that actually does things')}               {bold(cyan('║'))}
{bold(cyan('║'))}                                                            {bold(cyan('║'))}
{bold(cyan('║'))}   {dim(f'Platform: {OS} {ARCH}')}                                   {bold(cyan('║'))}
{bold(cyan('╚══════════════════════════════════════════════════════════════╝'))}
"""
    print(banner)


# ─────────────────────────────────────────────────────────────────────
# 1. Python Checks
# ─────────────────────────────────────────────────────────────────────

def check_python() -> bool:
    step("Checking Python")
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"

    if v.major < 3 or (v.major == 3 and v.minor < 11):
        fail(f"Python {ver} detected,  Python 3.11+ required")
        if IS_MAC:
            info("Fix: brew install python@3.12")
        elif IS_LINUX:
            info("Fix: sudo apt install python3.12 python3.12-venv  (Ubuntu/Debian)")
            info("  or: sudo dnf install python3.12  (Fedora/RHEL)")
        elif IS_WIN:
            info("Fix: Download from https://www.python.org/downloads/")
        return False

    ok(f"Python {ver}")
    return True


# ─────────────────────────────────────────────────────────────────────
# 2. System Dependencies (git, curl, build tools)
# ─────────────────────────────────────────────────────────────────────

def check_system_deps() -> bool:
    step("Checking system dependencies")
    all_ok = True

    # git
    if cmd_exists("git"):
        ok(f"git {cmd_version('git') or '(found)'}")
    else:
        fail("git not found")
        if IS_MAC:
            info("Fix: xcode-select --install")
        elif IS_LINUX:
            info("Fix: sudo apt install git")
        elif IS_WIN:
            info("Fix: https://git-scm.com/download/win")
        all_ok = False

    # curl (not critical on Windows)
    if cmd_exists("curl"):
        ok(f"curl {cmd_version('curl') or '(found)'}")
    elif not IS_WIN:
        warn("curl not found,  some auto-installs may fail")
        if IS_LINUX:
            info("Fix: sudo apt install curl")

    # build tools (for native Python packages like chromadb)
    if IS_LINUX:
        if cmd_exists("gcc"):
            ok("Build tools (gcc) available")
        else:
            warn("gcc not found,  some Python packages may fail to compile")
            info("Fix: sudo apt install build-essential python3-dev")
            _try_install_build_tools()

    if IS_MAC:
        r = run(["xcode-select", "-p"], capture=True)
        if r.returncode == 0:
            ok("Xcode Command Line Tools installed")
        else:
            warn("Xcode CLT not found,  installing...")
            run(["xcode-select", "--install"])

    # WiFi hunting tools (Linux only,  used by wifi_survival_skill)
    if IS_LINUX:
        wifi_tools = {
            "aircrack-ng": "aircrack-ng",
            "airodump-ng": "aircrack-ng",
            "aireplay-ng": "aircrack-ng",
            "airmon-ng":   "aircrack-ng",
            "iw":          "iw",
            "nmcli":       "network-manager",
        }
        missing_pkgs: set[str] = set()
        for binary, pkg in wifi_tools.items():
            if cmd_exists(binary):
                ok(f"{binary} (WiFi tools)")
            else:
                warn(f"{binary} not found")
                missing_pkgs.add(pkg)
        if missing_pkgs:
            info(f"Fix: sudo apt install {' '.join(sorted(missing_pkgs))}")
            _try_install_wifi_tools(missing_pkgs)

    return all_ok


def _try_install_build_tools():
    """Attempt to install build-essential on Linux."""
    if not IS_LINUX:
        return
    if os.geteuid() == 0:
        run(["apt-get", "install", "-y", "build-essential", "python3-dev"], capture=True)
    else:
        info("Run: sudo apt install build-essential python3-dev")


def _try_install_wifi_tools(pkgs: set[str]) -> None:
    """Attempt to install WiFi hunting tools on Linux."""
    if not IS_LINUX or not pkgs:
        return
    if os.geteuid() == 0:
        result = run(["apt-get", "install", "-y", *sorted(pkgs)], capture=True)
        if result.returncode == 0:
            ok("WiFi tools installed")
        else:
            warn("WiFi tools install failed,  run manually")
    else:
        info(f"Run: sudo apt install {' '.join(sorted(pkgs))}")


# ─────────────────────────────────────────────────────────────────────
# 3. Node.js
# ─────────────────────────────────────────────────────────────────────

def check_node() -> bool:
    step("Checking Node.js")

    if cmd_exists("node"):
        ver = cmd_version("node")
        major = 0
        if ver:
            m = re.match(r"(\d+)", ver)
            if m:
                major = int(m.group(1))

        if major >= 18:
            ok(f"Node.js v{ver}")
            npm_ver = cmd_version(npm_cmd())
            if npm_ver:
                ok(f"npm v{npm_ver}")
            return True
        else:
            warn(f"Node.js v{ver} found,  v18+ recommended")
            info("Attempting upgrade...")
            return _install_node()
    else:
        warn("Node.js not found,  installing...")
        return _install_node()


def _install_node() -> bool:
    """Install Node.js automatically."""
    try:
        if IS_MAC:
            if cmd_exists("brew"):
                info("Installing via Homebrew...")
                r = run(["brew", "install", "node"], capture=True)
                if r.returncode == 0:
                    ok("Node.js installed via Homebrew")
                    return True
            return _install_node_nvm()

        elif IS_LINUX:
            info("Installing Node.js 20 LTS...")
            r = run(
                ["bash", "-c",
                 "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - "
                 "&& sudo apt-get install -y nodejs"],
                capture=True, timeout=120,
            )
            if r.returncode == 0 and cmd_exists("node"):
                ok("Node.js installed via NodeSource")
                return True
            return _install_node_nvm()

        elif IS_WIN:
            fail("Cannot auto-install Node.js on Windows")
            info("Download from: https://nodejs.org/en/download/")
            info("Then re-run this installer")
            return False

    except Exception as e:
        fail(f"Node.js installation failed: {e}")
        info("Install manually: https://nodejs.org/")
        return False

    return False


def _install_node_nvm() -> bool:
    """Fallback: install Node via nvm."""
    info("Trying nvm fallback...")
    try:
        r = run(
            ["bash", "-c",
             'curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash '
             '&& export NVM_DIR="$HOME/.nvm" '
             '&& [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" '
             '&& nvm install 20'],
            capture=True, timeout=180,
        )
        if r.returncode == 0:
            ok("Node.js installed via nvm")
            info("You may need to restart your terminal for nvm to take effect")
            return True
    except Exception:
        pass
    fail("Could not install Node.js automatically")
    info("Install manually: https://nodejs.org/")
    return False


# ─────────────────────────────────────────────────────────────────────
# 4. Python Virtual Environment
# ─────────────────────────────────────────────────────────────────────

def setup_venv() -> bool:
    step("Python virtual environment")

    if VENV_DIR.exists() and VENV_PYTHON.exists():
        ok(f"venv exists at {VENV_DIR.name}/")
        # Verify it's not broken
        r = run([str(VENV_PYTHON), "-c", "import sys; print(sys.version)"], capture=True)
        if r.returncode != 0:
            warn("venv appears broken,  recreating...")
            shutil.rmtree(VENV_DIR)
        else:
            return True

    info("Creating virtual environment...")
    r = run([sys.executable, "-m", "venv", str(VENV_DIR)], capture=True)
    if r.returncode != 0:
        fail(f"Failed to create venv: {r.stderr}")
        if IS_LINUX:
            info("Fix: sudo apt install python3-venv")
            run(["sudo", "apt", "install", "-y", "python3-venv"], capture=True)
            r = run([sys.executable, "-m", "venv", str(VENV_DIR)], capture=True)
            if r.returncode == 0:
                ok("venv created (after installing python3-venv)")
                return True
        return False

    ok("Virtual environment created")
    return True


# ─────────────────────────────────────────────────────────────────────
# 5. Python Dependencies (pip)
# ─────────────────────────────────────────────────────────────────────

def install_python_deps() -> bool:
    step("Python dependencies")

    # Upgrade pip/setuptools/wheel
    substep("Upgrading pip, setuptools, wheel...")
    r = run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        capture=True,
    )
    if r.returncode != 0:
        warn(f"pip upgrade had issues: {(r.stderr or '')[:200]}")

    # Install from requirements.txt
    req_file = ROOT / "requirements.txt"
    if req_file.exists():
        substep("Installing from requirements.txt...")
        r = run(
            [str(VENV_PIP), "install", "-r", str(req_file)],
            cwd=ROOT,
        )
        if r.returncode != 0:
            fail("Some packages from requirements.txt failed")
            info("Trying packages one by one to identify the problem...")
            _install_requirements_fallback(req_file)

    # Install the package itself in editable mode
    substep("Installing opensable package (editable)...")
    r = run(
        [str(VENV_PIP), "install", "-e", ".[core]"],
        cwd=ROOT,
    )
    if r.returncode != 0:
        warn("Editable install had issues,  trying without extras...")
        r = run([str(VENV_PIP), "install", "-e", "."], cwd=ROOT)
        if r.returncode != 0:
            fail("Could not install opensable package")
            return False

    ok("Python dependencies installed")
    return True


def _install_requirements_fallback(req_file: Path):
    """Install requirements one by one, skipping broken ones."""
    lines = req_file.read_text().splitlines()
    failed = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        pkg = line.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip()
        r = run([str(VENV_PIP), "install", line], capture=True)
        if r.returncode != 0:
            failed.append(pkg)

    if failed:
        warn(f"Failed packages: {', '.join(failed)}")
        info("These may need system libraries. Check errors above.")
    else:
        ok("All packages installed (fallback mode)")


# ─────────────────────────────────────────────────────────────────────
# 6. Playwright Browsers
# ─────────────────────────────────────────────────────────────────────

def install_playwright() -> bool:
    step("Playwright browsers (for web scraping)")

    # Check if already installed
    r = run(
        [str(VENV_PYTHON), "-c",
         "from playwright.sync_api import sync_playwright; "
         "p = sync_playwright().start(); b = p.chromium; p.stop(); print('ok')"],
        capture=True, timeout=30,
    )
    if r.returncode == 0 and "ok" in (r.stdout or ""):
        ok("Playwright browsers already installed")
        return True

    substep("Installing Chromium browser for Playwright...")

    # Install system deps on Linux
    if IS_LINUX:
        r = run(
            [str(VENV_PYTHON), "-m", "playwright", "install-deps", "chromium"],
            capture=True, timeout=120,
        )
        if r.returncode != 0:
            warn("Could not install Playwright system deps,  trying anyway...")
            info("If it fails: sudo npx playwright install-deps")

    r = run(
        [str(VENV_PYTHON), "-m", "playwright", "install", "chromium"],
        capture=True, timeout=180,
    )
    if r.returncode == 0:
        ok("Playwright Chromium installed")
        return True
    else:
        warn("Playwright browser install failed (optional,  web scraping may not work)")
        info(f"Manual fix: {VENV_ACTIVATE} && python -m playwright install chromium")
        return False


# ─────────────────────────────────────────────────────────────────────
# 6b. PinchTab (browser control for AI agents)
# ─────────────────────────────────────────────────────────────────────

def _pinchtab_binary() -> Optional[Path]:
    """Return path to pinchtab binary if installed inside project, else None."""
    local = ROOT / "bin" / "pinchtab"
    if local.exists() and os.access(str(local), os.X_OK):
        return local
    if cmd_exists("pinchtab"):
        return Path(shutil.which("pinchtab"))  # type: ignore[arg-type]
    return None


def install_pinchtab() -> bool:
    """Install PinchTab binary into project bin/ directory."""
    step("PinchTab (token-efficient browser control)")

    # Already installed?
    existing = _pinchtab_binary()
    if existing:
        ver = cmd_version(str(existing))
        ok(f"PinchTab {ver or '(found)'} at {existing}")
        return True

    substep("Installing PinchTab via official installer...")
    bin_dir = ROOT / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Determine platform and arch for direct download
    os_name = "linux" if IS_LINUX else ("darwin" if IS_MAC else "windows")
    arch_name = "arm64" if ARCH in ("arm64", "aarch64") else "amd64"
    ext = ".exe" if IS_WIN else ""
    binary_name = f"pinchtab{ext}"
    download_url = (
        f"https://github.com/pinchtab/pinchtab/releases/latest/download/"
        f"pinchtab_{os_name}_{arch_name}{ext}"
    )

    # Try direct download first (no shell pipe)
    try:
        import urllib.request
        dest = bin_dir / binary_name
        substep(f"Downloading {download_url}...")
        urllib.request.urlretrieve(download_url, str(dest))
        dest.chmod(0o755)
        ok(f"PinchTab installed to {dest}")

        # Verify it runs
        r = run([str(dest), "--version"], capture=True, timeout=10)
        if r.returncode == 0:
            ver = re.search(r"(\d+\.\d+[\.\d]*)", r.stdout or "")
            ok(f"PinchTab {ver.group(1) if ver else 'OK'} verified")
        return True
    except Exception as e1:
        substep(f"Direct download failed ({e1}), trying install script...")

    # Fallback: official install script into project bin
    try:
        env = {**os.environ, "INSTALL_DIR": str(bin_dir)}
        r = run(
            ["bash", "-c", "curl -fsSL https://pinchtab.com/install.sh | bash"],
            capture=True, timeout=120, env=env,
        )
        if r.returncode == 0 and _pinchtab_binary():
            ok("PinchTab installed via install script")
            return True
    except Exception:
        pass

    warn("PinchTab install failed (optional,  Playwright fallback will be used)")
    info("Manual install: curl -fsSL https://pinchtab.com/install.sh | bash")
    return False


# ─────────────────────────────────────────────────────────────────────
# 7. Ollama (Local LLM)
# ─────────────────────────────────────────────────────────────────────

def setup_ollama() -> bool:
    step("Ollama (local LLM engine)")

    if cmd_exists("ollama"):
        ver = cmd_version("ollama")
        ok(f"Ollama {ver or '(found)'}")
        _pull_optimal_model()
        return True

    info("Ollama not found,  installing...")

    try:
        if IS_MAC:
            if cmd_exists("brew"):
                r = run(["brew", "install", "ollama"], capture=True, timeout=120)
                if r.returncode == 0:
                    ok("Ollama installed via Homebrew")
                    _pull_optimal_model()
                    return True
            r = run(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                     capture=True, timeout=120)
            if r.returncode == 0:
                ok("Ollama installed")
                _pull_optimal_model()
                return True

        elif IS_LINUX:
            r = run(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                     capture=True, timeout=120)
            if r.returncode == 0:
                ok("Ollama installed")
                _pull_optimal_model()
                return True

        elif IS_WIN:
            warn("Cannot auto-install Ollama on Windows")
            info("Download from: https://ollama.com/download")
            return False

    except Exception as e:
        warn(f"Ollama installation failed: {e}")

    warn("Could not install Ollama automatically")
    info("Download manually: https://ollama.com")
    info("SableCore works without Ollama if you set API keys (OpenAI, etc.)")
    return False


def _detect_hardware() -> dict:
    """Detect RAM, CPU, GPU info."""
    hw: dict = {"ram_gb": 0, "cpu_cores": 0, "gpu_mem_gb": 0, "has_gpu": False}

    try:
        import psutil
        hw["ram_gb"] = psutil.virtual_memory().total / (1024**3)
        hw["cpu_cores"] = psutil.cpu_count(logical=True) or 1
    except ImportError:
        if IS_MAC:
            r = run(["sysctl", "-n", "hw.memsize"], capture=True)
            if r.returncode == 0:
                try:
                    hw["ram_gb"] = int(r.stdout.strip()) / (1024**3)
                except ValueError:
                    pass
            r = run(["sysctl", "-n", "hw.ncpu"], capture=True)
            if r.returncode == 0:
                try:
                    hw["cpu_cores"] = int(r.stdout.strip())
                except ValueError:
                    pass
        elif IS_LINUX:
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            hw["ram_gb"] = kb / (1024**2)
                            break
            except Exception:
                pass
            r = run(["nproc"], capture=True)
            if r.returncode == 0:
                try:
                    hw["cpu_cores"] = int(r.stdout.strip())
                except ValueError:
                    pass

    # GPU
    if cmd_exists("nvidia-smi"):
        r = run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture=True,
        )
        if r.returncode == 0:
            try:
                hw["gpu_mem_gb"] = int(r.stdout.strip().split("\n")[0]) / 1024
                hw["has_gpu"] = True
            except ValueError:
                pass

    return hw


def _pull_optimal_model():
    """Auto-select and pull optimal Ollama model based on hardware.
    
    Skips entirely if the user already has a custom model configured in .env
    and that model exists in Ollama.
    """
    # Check if user already has a model set
    env_file = ROOT / ".env"
    if env_file.exists():
        content = env_file.read_text()
        m = re.search(r'^DEFAULT_MODEL=(.+)$', content, re.MULTILINE)
        if m:
            existing = m.group(1).strip()
            if existing and existing != "llama3.2:3b":  # not the placeholder default
                # Verify it actually exists in Ollama (match full name or base)
                r = run(["ollama", "list"], capture=True)
                ollama_out = r.stdout or ""
                base_name = existing.split(":")[0]
                if r.returncode == 0 and (existing in ollama_out or base_name in ollama_out):
                    ok(f"Model already configured: {existing} (keeping)")
                    return

    hw = _detect_hardware()

    substep(f"Hardware: {hw['ram_gb']:.0f}GB RAM, {hw['cpu_cores']} cores"
            + (f", NVIDIA {hw['gpu_mem_gb']:.0f}GB VRAM" if hw["has_gpu"] else ", no GPU"))

    if hw["has_gpu"] and hw["gpu_mem_gb"] >= 20:
        model = "llama3.1:70b"
    elif hw["has_gpu"] and hw["gpu_mem_gb"] >= 8:
        model = "llama3.1:8b"
    elif hw["ram_gb"] >= 32:
        model = "llama3.1:8b"
    elif hw["ram_gb"] >= 8:
        model = "llama3.2:3b"
    else:
        model = "llama3.2:1b"

    substep(f"Auto-selected model: {bold(model)}")

    r = run(["ollama", "list"], capture=True)
    if r.returncode == 0 and model in (r.stdout or ""):
        ok(f"{model} already downloaded")
    else:
        info(f"Downloading {model} (this may take a while)...")
        r = run(["ollama", "pull", model], timeout=900)
        if r.returncode == 0:
            ok(f"{model} downloaded")
        else:
            warn(f"Could not download {model},  you can pull it later: ollama pull {model}")

    _set_env_var("DEFAULT_MODEL", model)

    # Pull the embedding model required by the RAG/codebase search system
    embed_model = "nomic-embed-text"
    r2 = run(["ollama", "list"], capture=True)
    if r2.returncode == 0 and embed_model in (r2.stdout or ""):
        ok(f"{embed_model} (embeddings) already downloaded")
    else:
        info(f"Downloading {embed_model} for vector embeddings...")
        r2 = run(["ollama", "pull", embed_model], timeout=300)
        if r2.returncode == 0:
            ok(f"{embed_model} downloaded")
        else:
            warn(f"Could not download {embed_model},  run: ollama pull {embed_model}")

    substep("Agent auto-downloads additional models at runtime as needed")


# ─────────────────────────────────────────────────────────────────────
# 8. NPM Sub-projects
# ─────────────────────────────────────────────────────────────────────

def install_npm_projects(full_mode: bool = False) -> int:
    step("JavaScript sub-projects (npm)")

    if not cmd_exists("node"):
        warn("Node.js not available,  skipping all JS sub-projects")
        return 0

    installed = 0
    npm = npm_cmd()

    for proj in NPM_PROJECTS:
        proj_dir = ROOT / proj["dir"]
        pkg_json = proj_dir / "package.json"
        name = proj["name"]

        if not pkg_json.exists():
            substep(f"{name}: {dim('not found,  skipping')}")
            continue

        # Ask for optional projects if not in full mode
        if proj["optional"] and not full_mode:
            print(f"\n  {cyan('?')} Install {bold(name)}? {dim('[y/N]')} ", end="", flush=True)
            try:
                ans = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans not in ("y", "yes"):
                substep(f"{name}: skipped")
                continue

        if _install_npm_project(proj, npm):
            installed += 1

    return installed


def _install_npm_project(proj: dict, npm: str) -> bool:
    """Install a single npm sub-project."""
    proj_dir = ROOT / proj["dir"]
    name = proj["name"]
    node_modules = proj_dir / "node_modules"

    # Check if already installed
    if node_modules.exists() and any(node_modules.iterdir()):
        pkg_lock = proj_dir / "package-lock.json"
        if pkg_lock.exists() or (node_modules / ".package-lock.json").exists():
            ok(f"{name}: dependencies already installed")
            if proj["build"]:
                return _build_npm_project(proj, npm)
            return True
        else:
            warn(f"{name}: node_modules exists but may be incomplete,  reinstalling")

    substep(f"{name}: installing dependencies...")

    # Special env for aggr
    if proj["dir"] == "aggr":
        env_local = proj_dir / ".env.local"
        if not env_local.exists():
            env_local.write_text(
                "VITE_APP_PROXY_URL=https://cors.aggr.trade/\n"
                "VITE_APP_API_URL=https://api.aggr.trade/\n"
                "VITE_APP_LIB_URL=https://lib.aggr.trade/\n"
                "VITE_APP_LIB_REPO_URL=https://github.com/Tucsky/aggr-lib\n"
                "VITE_APP_BASE_PATH=/aggr/\n"
                "VITE_APP_API_SUPPORTED_TIMEFRAMES=5,10,15,30,60,180,300,900,1260,1800,3600,7200,14400,21600,28800,43200,86400\n"
            )

    r = run([npm, "install"], cwd=proj_dir, timeout=300)
    if r.returncode != 0:
        fail(f"{name}: npm install failed")
        info(f"Manual fix: cd {proj['dir']} && npm install")
        # Auto-fix: clean and retry
        substep(f"{name}: attempting auto-fix (clean install)...")
        if node_modules.exists():
            shutil.rmtree(node_modules, ignore_errors=True)
        pkg_lock = proj_dir / "package-lock.json"
        if pkg_lock.exists():
            pkg_lock.unlink()
        r = run([npm, "install"], cwd=proj_dir, timeout=300)
        if r.returncode != 0:
            fail(f"{name}: auto-fix failed too")
            return False

    ok(f"{name}: dependencies installed")

    if proj["build"]:
        return _build_npm_project(proj, npm)

    return True


def _build_npm_project(proj: dict, npm: str) -> bool:
    """Build an npm project if it has a dist/build folder."""
    proj_dir = ROOT / proj["dir"]
    name = proj["name"]

    # Check if already built
    for dist_name in ("dist", "build", ".next"):
        dist_dir = proj_dir / dist_name
        if dist_dir.exists():
            index = dist_dir / "index.html"
            if index.exists() or dist_name == ".next":
                ok(f"{name}: already built")
                if proj["dir"] == "aggr":
                    _patch_aggr_tracking(proj_dir)
                return True

    substep(f"{name}: building...")

    build_cmd = [npm, "run", "build"]

    r = run(build_cmd, cwd=proj_dir, timeout=300)
    if r.returncode == 0:
        ok(f"{name}: build complete")
        if proj["dir"] == "aggr":
            _patch_aggr_tracking(proj_dir)
        return True
    else:
        warn(f"{name}: build failed (non-critical)")
        info(f"Manual fix: cd {proj['dir']} && npm run build")
        return False


def _patch_aggr_tracking(aggr_dir: Path):
    """Remove tracking/analytics from aggr (GTM, etc.)."""
    targets = [aggr_dir / "index.html", aggr_dir / "dist" / "index.html"]
    patterns = [
        (r'\s*<!-- Google Tag Manager -->.*?<!-- End Google Tag Manager -->\s*', '\n'),
        (r'\s*<!-- Google Tag Manager \(noscript\) -->.*?<!-- End Google Tag Manager \(noscript\) -->\s*', '\n'),
        (r'<script[^>]*>[^<]*googletagmanager[^<]*</script>', ''),
        (r'<noscript[^>]*>[^<]*googletagmanager[^<]*</noscript>', ''),
        (r'<iframe[^>]*googletagmanager[^>]*>[^<]*</iframe[^>]*>', ''),
        (r'<script[^>]*>[^<]*google-analytics[^<]*</script>', ''),
        (r'<script[^>]*>[^<]*(?:hotjar|fbq|mixpanel|amplitude|segment\.io|fullstory|clarity\.ms|zunvra)[^<]*</script>', ''),
        (r'<base\s+href="/"\s*/>', '<base href="/aggr/" />'),
    ]
    patched = 0
    for fp in targets:
        if not fp.exists():
            continue
        content = fp.read_text(encoding="utf-8")
        original = content
        for pat, repl in patterns:
            content = re.sub(pat, repl, content, flags=re.DOTALL | re.IGNORECASE)
        if content != original:
            fp.write_text(content, encoding="utf-8")
            patched += 1
    if patched:
        substep(f"Stripped tracking from {patched} aggr file(s)")


# ─────────────────────────────────────────────────────────────────────
# 9. Directories & Config
# ─────────────────────────────────────────────────────────────────────

def setup_directories():
    step("Project directories")
    dirs = ["data", "logs", "config", "data/checkpoints", "data/vectordb"]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    ok("Directories verified")


def setup_env_file():
    step("Environment configuration (.env)")

    env_file = ROOT / ".env"
    env_example = ROOT / ".env.example"

    if env_file.exists():
        ok(".env file already exists")
        return

    if env_example.exists():
        shutil.copy(env_example, env_file)
        ok("Created .env from .env.example")
        info("Edit .env to add your bot tokens and API keys")
    else:
        warn(".env.example not found,  creating minimal .env")
        env_file.write_text(textwrap.dedent("""\
            # SableCore Configuration
            # See .env.example for all options

            # LLM (local,  requires Ollama running)
            OLLAMA_BASE_URL=http://localhost:11434
            DEFAULT_MODEL=llama3.2:3b
            AUTO_SELECT_MODEL=true

            # Telegram Bot (get token from @BotFather)
            TELEGRAM_BOT_TOKEN=
            TELEGRAM_ALLOWED_USERS=

            # Dev Studio
            DEV_STUDIO_ENABLED=true

            # PinchTab (browser control,  auto-detected, no config needed)
            # PINCHTAB_URL=http://127.0.0.1:9867
            # PINCHTAB_DISABLED=false
        """))
        ok("Created minimal .env")

    info("Get Telegram bot token from @BotFather")
    info("Set API keys for external LLM providers (OpenAI, etc.) in .env")


def _set_env_var(key: str, value: str):
    """Set a variable in .env (create or update)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    content = env_file.read_text()
    pattern = rf"^{re.escape(key)}=.*$"
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, f"{key}={value}", content, flags=re.MULTILINE)
    else:
        content += f"\n{key}={value}\n"
    env_file.write_text(content)


# ─────────────────────────────────────────────────────────────────────
# 10. Extras (optional heavy deps)
# ─────────────────────────────────────────────────────────────────────

def install_extras_interactive():
    step("Optional features")

    print(f"""
  Choose additional capabilities to install:

    {bold('1')} Core only {dim('(minimal, ~100MB)')}
    {bold('2')} + Trading {dim('(ccxt, alpaca, technical analysis)')}
    {bold('3')} + Voice   {dim('(speech-to-text, text-to-speech)')}
    {bold('4')} + Vision  {dim('(image recognition, OCR,  5GB+)')}
    {bold('5')} All extras {dim('(everything,  5GB+)')}
""")

    print(f"  {cyan('?')} Select option {dim('[1-5, default: 1]')}: ", end="", flush=True)
    try:
        choice = input().strip() or "1"
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    extras_map = {
        "1": [],
        "2": ["requirements-trading.txt"],
        "3": [".[voice]"],
        "4": [".[vision]"],
        "5": ["requirements-trading.txt", "requirements-extras.txt"],
    }

    extras = extras_map.get(choice, [])
    if not extras:
        substep("Core only selected,  skipping extras")
        return

    for extra in extras:
        substep(f"Installing {extra}...")
        if extra.startswith("."):
            r = run([str(VENV_PIP), "install", "-e", extra], cwd=ROOT)
        else:
            req_path = ROOT / extra
            if req_path.exists():
                r = run([str(VENV_PIP), "install", "-r", str(req_path)], cwd=ROOT)
            else:
                warn(f"{extra} not found,  skipping")
                continue

        if r.returncode == 0:
            ok(f"{extra} installed")
        else:
            warn(f"Some packages from {extra} failed (non-critical)")


# ─────────────────────────────────────────────────────────────────────
# 11. Status / Health Check
# ─────────────────────────────────────────────────────────────────────

def show_status():
    """Show installation status of all components."""
    print_banner()
    step("Installation Status")

    print()
    # Python
    v = sys.version_info
    _status_line("Python", f"{v.major}.{v.minor}.{v.micro}", v.major >= 3 and v.minor >= 11)

    # Venv
    _status_line("Virtual env", str(VENV_DIR.name) + "/", VENV_DIR.exists() and VENV_PYTHON.exists())

    # Node
    node_ver = cmd_version("node") if cmd_exists("node") else None
    _status_line("Node.js", node_ver or "not found", node_ver is not None)

    # npm
    npm_ver = cmd_version(npm_cmd()) if cmd_exists(npm_cmd()) else None
    _status_line("npm", npm_ver or "not found", npm_ver is not None)

    # git
    git_ver = cmd_version("git") if cmd_exists("git") else None
    _status_line("git", git_ver or "not found", git_ver is not None)

    # Ollama
    ollama_ver = cmd_version("ollama") if cmd_exists("ollama") else None
    _status_line("Ollama", ollama_ver or "not found", ollama_ver is not None)

    # Playwright
    pw_ok = False
    if VENV_PYTHON.exists():
        r = run([str(VENV_PYTHON), "-c", "import playwright; print('ok')"], capture=True)
        pw_ok = r.returncode == 0
    _status_line("Playwright", "installed" if pw_ok else "not installed", pw_ok)

    # PinchTab
    pt_bin = _pinchtab_binary()
    if pt_bin:
        pt_ver = cmd_version(str(pt_bin))
        _status_line("PinchTab", f"{pt_ver or '(found)'} at {pt_bin}", True)
    else:
        _status_line("PinchTab", "not installed (optional)", None)

    # Python packages
    pkg_ok = False
    if VENV_PYTHON.exists():
        r = run([str(VENV_PYTHON), "-c", "import opensable; print('ok')"], capture=True)
        pkg_ok = r.returncode == 0
    _status_line("opensable pkg", "installed" if pkg_ok else "not installed", pkg_ok)

    print()
    step("JavaScript Sub-projects")
    print()

    for proj in NPM_PROJECTS:
        proj_dir = ROOT / proj["dir"]
        pkg_json = proj_dir / "package.json"
        node_modules = proj_dir / "node_modules"

        if not pkg_json.exists():
            _status_line(proj["name"], "not found", None)
            continue

        has_deps = node_modules.exists() and any(node_modules.iterdir())

        if proj["build"]:
            has_build = False
            for dist_name in ("dist", "build", ".next"):
                dist_dir = proj_dir / dist_name
                if dist_dir.exists() and (
                    (dist_dir / "index.html").exists() or dist_name == ".next"
                ):
                    has_build = True
                    break

            if has_deps and has_build:
                _status_line(proj["name"], "installed + built", True)
            elif has_deps:
                _status_line(proj["name"], "deps ok, not built", None)
            else:
                _status_line(proj["name"], "not installed", False)
        else:
            _status_line(proj["name"], "installed" if has_deps else "not installed", has_deps)

    # .env
    print()
    env_file = ROOT / ".env"
    _status_line(".env file", "exists" if env_file.exists() else "missing", env_file.exists())

    if env_file.exists():
        content = env_file.read_text()
        has_telegram = bool(re.search(r"^TELEGRAM_BOT_TOKEN=.+", content, re.MULTILINE))
        _status_line("  Telegram token", "configured" if has_telegram else "not set",
                      has_telegram if has_telegram else None)

        has_any_llm = any(
            re.search(rf"^{k}=.+", content, re.MULTILINE)
            for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                       "DEEPSEEK_API_KEY", "GROQ_API_KEY", "XAI_API_KEY"]
        )
        _status_line("  LLM API keys", "at least one set" if has_any_llm else "none set",
                      has_any_llm if has_any_llm else None)

    print()


def _status_line(name: str, value: str, status: Optional[bool]):
    """Print a status line with colored indicator."""
    if status is True:
        icon = green("✔")
    elif status is False:
        icon = red("✘")
    else:
        icon = yellow("~")
    print(f"  {icon} {name:.<28s} {value}")


# ─────────────────────────────────────────────────────────────────────
# 12. Fix Mode
# ─────────────────────────────────────────────────────────────────────

def fix_install():
    """Re-check and fix broken installations."""
    print_banner()
    step("Repair Mode,  checking and fixing issues")

    issues = 0
    fixed = 0

    # Check venv
    if not VENV_PYTHON.exists():
        warn("Virtual environment missing,  recreating")
        if setup_venv():
            fixed += 1
        else:
            issues += 1
    else:
        # Verify venv is not broken
        r = run([str(VENV_PYTHON), "-c", "import sys; print(sys.version)"], capture=True)
        if r.returncode != 0:
            warn("Virtual environment broken,  recreating")
            shutil.rmtree(VENV_DIR, ignore_errors=True)
            if setup_venv():
                fixed += 1
            else:
                issues += 1
        else:
            ok("Virtual environment OK")

    # Check Python deps
    if VENV_PYTHON.exists():
        r = run([str(VENV_PYTHON), "-c", "import opensable"], capture=True)
        if r.returncode != 0:
            warn("opensable package not installed,  fixing")
            if install_python_deps():
                fixed += 1
            else:
                issues += 1
        else:
            ok("Python packages OK")

    # Check npm projects
    npm = npm_cmd()
    if cmd_exists("node"):
        for proj in NPM_PROJECTS:
            proj_dir = ROOT / proj["dir"]
            if not (proj_dir / "package.json").exists():
                continue

            node_modules = proj_dir / "node_modules"
            if not node_modules.exists() or not any(node_modules.iterdir()):
                warn(f"{proj['name']}: missing node_modules,  reinstalling")
                r = run([npm, "install"], cwd=proj_dir, timeout=300)
                if r.returncode == 0:
                    ok(f"{proj['name']}: fixed")
                    fixed += 1
                else:
                    # Try clean install
                    substep(f"{proj['name']}: trying clean install...")
                    if node_modules.exists():
                        shutil.rmtree(node_modules, ignore_errors=True)
                    pkg_lock = proj_dir / "package-lock.json"
                    if pkg_lock.exists():
                        pkg_lock.unlink()
                    r = run([npm, "install"], cwd=proj_dir, timeout=300)
                    if r.returncode == 0:
                        ok(f"{proj['name']}: fixed (clean install)")
                        fixed += 1
                    else:
                        fail(f"{proj['name']}: could not fix")
                        issues += 1
            else:
                ok(f"{proj['name']}: OK")

    # Check .env
    env_file = ROOT / ".env"
    if not env_file.exists():
        warn(".env missing,  creating")
        setup_env_file()
        fixed += 1
    else:
        ok(".env file present")

    # Check PinchTab
    if not _pinchtab_binary():
        substep("PinchTab not found, attempting install...")
        if install_pinchtab():
            fixed += 1
    else:
        ok("PinchTab OK")

    # Check directories
    for d in ["data", "logs", "config"]:
        p = ROOT / d
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            warn(f"Created missing directory: {d}/")
            fixed += 1

    # Summary
    print()
    if issues == 0 and fixed == 0:
        ok(bold("Everything looks good! No issues found."))
    elif issues == 0:
        ok(f"Repair complete,  {fixed} issue(s) fixed, everything is good now!")
    else:
        warn(f"Repair done,  {fixed} fixed, {issues} remaining issue(s)")
        info("Check the errors above for manual fix instructions")


# ─────────────────────────────────────────────────────────────────────
# 13. Completion Summary
# ─────────────────────────────────────────────────────────────────────

def print_completion():
    if IS_WIN:
        activate = "venv\\Scripts\\activate"
        run_cmd = "python -m opensable"
    else:
        activate = "source venv/bin/activate"
        run_cmd = "python -m opensable"

    print(f"""
{bold(green('╔══════════════════════════════════════════════════════════════╗'))}
{bold(green('║'))}                                                            {bold(green('║'))}
{bold(green('║'))}   {bold(green('Installation Complete!'))}                                  {bold(green('║'))}
{bold(green('║'))}                                                            {bold(green('║'))}
{bold(green('╚══════════════════════════════════════════════════════════════╝'))}

  {bold('Quick Start:')}

    {cyan('1.')} Activate the virtual environment:
       {bold(activate)}

    {cyan('2.')} Edit {bold('.env')} with your bot token:
       {dim('Get one from @BotFather on Telegram')}

    {cyan('3.')} Start SableCore:
       {bold('./start.sh start')}         {dim('(background, recommended)')}
       {bold(run_cmd)}       {dim('(foreground)')}

  {bold('Useful Commands:')}

    ./start.sh status             {dim('Check if running')}
    ./start.sh logs               {dim('Follow live logs')}
    ./start.sh stop               {dim('Stop the agent')}
    ./start.sh restart            {dim('Restart the agent')}
    ./start.sh start --all        {dim('Start all agent profiles')}
    cd sable_dev && npm run dev   {dim('Start Dev Studio')}
    cd desktop && npm run dev     {dim('Start Desktop App')}

  {bold('Maintenance:')}

    python install.py --status    {dim('Show what is installed')}
    python install.py --fix       {dim('Auto-fix broken installs')}

  {dim('Docs: README.md | INSTALL.md | docs/')}
""")


# ─────────────────────────────────────────────────────────────────────
# Main Flow
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SableCore Universal Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python install.py               Interactive install (recommended)
              python install.py --full         Install everything, no prompts
              python install.py --core         Python core only (minimal)
              python install.py --fix          Repair broken installations
              python install.py --status       Show installation health
        """),
    )
    parser.add_argument("--full", action="store_true",
                        help="Install everything (no prompts)")
    parser.add_argument("--core", action="store_true",
                        help="Core Python only (minimal)")
    parser.add_argument("--fix", action="store_true",
                        help="Re-check and fix broken installs")
    parser.add_argument("--status", action="store_true",
                        help="Show installation status")
    args = parser.parse_args()

    # Special modes
    if args.status:
        show_status()
        return

    if args.fix:
        fix_install()
        return

    # ── Full install flow ──────────────────────────────────────────
    print_banner()

    tracker: dict[str, bool] = {}
    start_time = time.time()

    # 1. Python version
    if not check_python():
        fail("Python 3.11+ is required. Cannot continue.")
        sys.exit(1)

    # 2. System deps (git, curl, build tools)
    tracker["system_deps"] = check_system_deps()

    # 3. Node.js (needed for sub-projects)
    if not args.core:
        tracker["nodejs"] = check_node()

    # 4. Virtual environment
    if not setup_venv():
        fail("Cannot create virtual environment. Cannot continue.")
        sys.exit(1)
    tracker["venv"] = True

    # 5. Python dependencies
    tracker["python_deps"] = install_python_deps()
    if not tracker["python_deps"]:
        fail("Python dependency installation failed critically.")
        info("Try: python install.py --fix")
        sys.exit(1)

    # 6. Playwright browsers (optional)
    tracker["playwright"] = install_playwright()

    # 6b. PinchTab (token-efficient browser,  optional, enhances browsing)
    tracker["pinchtab"] = install_pinchtab()

    # 7. Directories & config
    setup_directories()
    setup_env_file()

    # 8. Ollama
    tracker["ollama"] = setup_ollama()

    # 9. npm sub-projects (skip in core mode)
    if not args.core:
        install_npm_projects(full_mode=args.full)

    # 10. Extras (interactive,  skip in full/core mode)
    if not args.core and not args.full:
        install_extras_interactive()
    elif args.full:
        trading_req = ROOT / "requirements-trading.txt"
        if trading_req.exists():
            step("Trading dependencies")
            r = run([str(VENV_PIP), "install", "-r", str(trading_req)], cwd=ROOT)
            if r.returncode == 0:
                ok("Trading dependencies installed")

    # Done!
    elapsed = time.time() - start_time
    print(f"\n  {dim(f'Total time: {elapsed:.0f}s')}")
    print_completion()

    # Show any warnings
    warnings = [k for k, v in tracker.items() if not v]
    if warnings:
        warn(f"Some optional components had issues: {', '.join(warnings)}")
        info("Run 'python install.py --status' to see details")
        info("Run 'python install.py --fix' to attempt auto-repair")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{yellow('Installation cancelled by user.')}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{red('Unexpected error:')} {e}")
        import traceback
        traceback.print_exc()
        print(f"\n{yellow('Try:')} python install.py --fix")
        sys.exit(1)
