#!/usr/bin/env bash
# =============================================================================
#  OpenSable — Raspberry Pi Installation Script
#  Supports: Raspberry Pi 3B+ / 4B / 5 (64-bit Raspberry Pi OS, aarch64)
#  LLM Backend: OpenWebUI API (remote — no local model required)
#  Author: OpenSable Project
#
#  Usage:
#    ./install-pi.sh            — Full install + post-install verification
#    ./install-pi.sh --repair   — Re-verify all deps, auto-fix what's broken
#    ./install-pi.sh --verify   — Check only (read-only, no changes)
# =============================================================================
set -euo pipefail

# ── Mode ──────────────────────────────────────────────────────────────────────
MODE="install"
SETUP_DISPLAY=false
for arg in "$@"; do
    case "$arg" in
        --repair)  MODE="repair" ;;
        --verify)  MODE="verify" ;;
        --display) SETUP_DISPLAY=true ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}${BOLD}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}${BOLD}[ OK ]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}${BOLD}[ERR ]${RESET}  $*" >&2; }
step()    { echo -e "\n${MAGENTA}${BOLD}══▶ $*${RESET}"; }
fixed()   { echo -e "${GREEN}${BOLD}[FIX]${RESET}  $*"; }
ask()     { echo -e "${YELLOW}${BOLD}[ ? ]${RESET}  $*"; }
sep()     { echo -e "${DIM}────────────────────────────────────────────────────${RESET}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PROFILE_DIR="$SCRIPT_DIR/agents/pi"
PROFILE_ENV="$PROFILE_DIR/profile.env"
REPAIR_LOG="$SCRIPT_DIR/logs/pi-repair.log"

# ── pip package → python import name map ──────────────────────────────────────
declare -A PKG_IMPORTS=(
    [aiohttp]="aiohttp"
    [aiofiles]="aiofiles"
    [fastapi]="fastapi"
    [uvicorn]="uvicorn"
    [websockets]="websockets"
    [httpx]="httpx"
    [python-telegram-bot]="telegram"
    [pydantic]="pydantic"
    [python-dotenv]="dotenv"
    [rich]="rich"
    [typer]="typer"
    [pillow]="PIL"
    [cryptography]="cryptography"
    [SQLAlchemy]="sqlalchemy"
    [aiosqlite]="aiosqlite"
    [openai]="openai"
    [tenacity]="tenacity"
    [structlog]="structlog"
    [psutil]="psutil"
    [schedule]="schedule"
    [chromadb]="chromadb"
)

# Required runtime directories
REQUIRED_DIRS=(
    "$SCRIPT_DIR/data"
    "$SCRIPT_DIR/data/vectordb"
    "$SCRIPT_DIR/logs"
    "$SCRIPT_DIR/config"
    "$PROFILE_DIR"
)

# ── try_import: 0 if importable, 1 if missing ─────────────────────────────────
try_import() {
    "$VENV_DIR/bin/python3" -c "import ${1}" 2>/dev/null
}

# ── install_package: pip install with up to 3 attempts & fallbacks ─────────────
install_package() {
    local pkg="$1" attempt=0 max=3
    while (( attempt < max )); do
        attempt=$(( attempt + 1 ))
        info "  Installing ${BOLD}${pkg}${RESET} (attempt ${attempt}/${max})..."
        local pip_flags="-q"
        (( attempt >= 2 )) && pip_flags="-q --no-cache-dir"
        (( attempt >= 3 )) && pip_flags="-q --no-cache-dir --no-build-isolation"
        "$VENV_DIR/bin/pip" install $pip_flags "$pkg" 2>>"$REPAIR_LOG" && return 0
        warn "  Attempt ${attempt} failed. Retrying..."
        sleep 2
    done
    error "  Could not install ${BOLD}${pkg}${RESET} after ${max} attempts."
    return 1
}

# ── verify_python_packages: check all imports, auto-repair if not --verify ─────
verify_python_packages() {
    step "Verifying Python packages"
    mkdir -p "$(dirname "$REPAIR_LOG")"
    echo "--- verify run $(date) ---" >> "$REPAIR_LOG"

    if [[ ! -d "$VENV_DIR" ]]; then
        error "Virtual environment not found. Run ${BOLD}./install-pi.sh${RESET} first."
        return 1
    fi

    local pass=0 repaired=0
    local -a failed_pkgs=()

    for pkg in "${!PKG_IMPORTS[@]}"; do
        local import="${PKG_IMPORTS[$pkg]}"
        if try_import "$import"; then
            ok "  ${BOLD}${pkg}${RESET}"
            (( pass++ )) || true
        else
            warn "  ${BOLD}${pkg}${RESET} — missing (import '${import}' failed)"
            if [[ "$MODE" != "verify" ]]; then
                if install_package "$pkg" && try_import "$import"; then
                    fixed "  ${BOLD}${pkg}${RESET} — repaired"
                    (( repaired++ )) || true
                    (( pass++ )) || true
                else
                    error "  ${BOLD}${pkg}${RESET} — could not repair"
                    failed_pkgs+=("$pkg")
                fi
            else
                failed_pkgs+=("$pkg")
            fi
        fi
    done

    sep
    echo -e "  ${GREEN}${BOLD}${pass} OK${RESET}  ${GREEN}${repaired} repaired${RESET}  ${RED}${#failed_pkgs[@]} failed${RESET}"

    if (( ${#failed_pkgs[@]} > 0 )); then
        error "Still broken: ${failed_pkgs[*]}"
        error "See ${BOLD}${REPAIR_LOG}${RESET} for details."
        return 1
    fi
    ok "All Python packages verified."
}

# ── verify_directories: create missing dirs unless --verify ────────────────────
verify_directories() {
    step "Verifying runtime directories"
    local repaired=0
    for dir in "${REQUIRED_DIRS[@]}"; do
        if [[ -d "$dir" ]]; then
            ok "  ${dir#$SCRIPT_DIR/}"
        else
            warn "  Missing: ${dir#$SCRIPT_DIR/}"
            if [[ "$MODE" != "verify" ]]; then
                mkdir -p "$dir"
                fixed "  Created: ${dir#$SCRIPT_DIR/}"
                (( repaired++ )) || true
            fi
        fi
    done
    [[ $repaired -gt 0 ]] && ok "Directories: ${repaired} created." || ok "All directories present."
}

# ── verify_opensable_package ───────────────────────────────────────────────────
verify_opensable_package() {
    step "Verifying OpenSable package"
    if "$VENV_DIR/bin/python3" -c "import opensable" 2>/dev/null; then
        ok "opensable importable."
    elif [[ "$MODE" != "verify" && -f "$SCRIPT_DIR/pyproject.toml" ]]; then
        info "Re-installing opensable (editable)..."
        "$VENV_DIR/bin/pip" install --no-deps -e "$SCRIPT_DIR" -q 2>>"$REPAIR_LOG"
        "$VENV_DIR/bin/python3" -c "import opensable" 2>/dev/null \
            && fixed "opensable repaired." \
            || error "opensable still broken — check ${REPAIR_LOG}"
    else
        error "opensable not importable. Run ${BOLD}./install-pi.sh --repair${RESET}."
        return 1
    fi
}

# ── verify_profile_config ──────────────────────────────────────────────────────
verify_profile_config() {
    step "Verifying agent profile"
    local REQUIRED_KEYS=(OPENWEBUI_API_URL OPENWEBUI_MODEL TELEGRAM_BOT_TOKEN AGENT_NAME)

    if [[ ! -f "$PROFILE_ENV" ]]; then
        warn "Profile not found: ${BOLD}${PROFILE_ENV}${RESET}"
        if [[ "$MODE" != "verify" ]]; then
            mkdir -p "$PROFILE_DIR"
            cat > "$PROFILE_ENV" << 'MINENV'
# Auto-generated minimal profile — fill in your values
AGENT_NAME=Sable-Pi
OPENWEBUI_API_URL=https://sofia.zunvra.com
OPENWEBUI_API_KEY=
OPENWEBUI_MODEL=llama3.2:latest
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USERS=
LOG_LEVEL=INFO
VISION_ENABLED=false
DESKTOP_ENABLED=false
WHATSAPP_ENABLED=false
VOICE_ENABLED=false
AUTONOMOUS_ENABLED=false
DATA_DIR=./data
LOG_FILE=./logs/opensable.log
LOW_VRAM_MODE=false
MAX_CONTEXT_LENGTH=4096
MINENV
            fixed "Minimal profile created — fill in TELEGRAM_BOT_TOKEN and OPENWEBUI_API_KEY."
        fi
        return 0
    fi

    ok "Profile: ${BOLD}${PROFILE_ENV}${RESET}"
    local missing=()
    for key in "${REQUIRED_KEYS[@]}"; do
        local val; val=$(grep -E "^${key}=" "$PROFILE_ENV" 2>/dev/null | cut -d= -f2- | tr -d ' ')
        if [[ -z "$val" ]]; then
            warn "  ${BOLD}${key}${RESET} is blank"
            missing+=("$key")
        else
            ok "  ${BOLD}${key}${RESET} = ${DIM}${val:0:50}${RESET}"
        fi
    done
    (( ${#missing[@]} > 0 )) && warn "Blank keys: ${missing[*]} — edit ${PROFILE_ENV}" || ok "All profile keys set."
}

# ── verify_connectivity ────────────────────────────────────────────────────────
verify_connectivity() {
    step "Verifying OpenWebUI connectivity"
    [[ ! -f "$PROFILE_ENV" ]] && warn "No profile — skipping." && return 0

    local url; url=$(grep -E '^OPENWEBUI_API_URL=' "$PROFILE_ENV" 2>/dev/null | cut -d= -f2- | tr -d ' ')
    local key; key=$(grep -E '^OPENWEBUI_API_KEY=' "$PROFILE_ENV" 2>/dev/null | cut -d= -f2- | tr -d ' ')
    [[ -z "$url" ]] && warn "OPENWEBUI_API_URL blank — skipping." && return 0

    local code; code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
        -H "Authorization: Bearer ${key}" "${url}/api/models" 2>/dev/null || echo "000")
    case "$code" in
        200) ok "Reachable: ${BOLD}${url}${RESET} — HTTP ${GREEN}200${RESET}" ;;
        401) warn "HTTP 401 — check OPENWEBUI_API_KEY in ${PROFILE_ENV}" ;;
        000) warn "Cannot reach ${url} — check network/DNS" ;;
        *)   warn "HTTP ${code} from ${url}/api/models" ;;
    esac
}

# ── setup_display: configure 3.5" XPT2046 / ILI9486 display (MPI3501) ───────
# Uses the tft35a driver from github.com/goodtft/LCD-show (the official driver
# for generic Chinese 3.5" 480×320 SPI TFT displays with XPT2046 touch).
# Pin mapping (confirmed via lcdwiki.com MPI3501 datasheet):
#   LCD_CS  = GPIO8  (SPI CE0, pin 24)
#   TP_CS   = GPIO7  (SPI CE1, pin 26)
#   TP_IRQ  = GPIO17 (pin 11)
#   LCD_RS  = GPIO24 (pin 18)
#   RST     = GPIO25 (pin 22)
setup_display() {
    step "Setting up 3.5\" XPT2046/ILI9486 display (MPI3501, 480×320 RGB565)"

    # ─ Detect boot config path (Bookworm = /boot/firmware, older = /boot) ──
    local boot_cfg="/boot/config.txt"
    [[ -f "/boot/firmware/config.txt" ]] && boot_cfg="/boot/firmware/config.txt"
    ok "Boot config: ${BOLD}${boot_cfg}${RESET}"

    # ─ Detect overlays dir (Bookworm keeps dtbo here too) ──────────────────
    local overlays_dir="/boot/overlays"
    [[ -d "/boot/firmware/overlays" ]] && overlays_dir="/boot/firmware/overlays"
    ok "Overlays dir: ${BOLD}${overlays_dir}${RESET}"

    # ─ Rotation ─────────────────────────────────────────────────────────────
    # tft35a overlay rotation values:
    #   90  = landscape, normal  (default, USB ports on right — most common)
    #   270 = landscape, flipped (USB ports on left)
    #   0   = portrait,  normal
    #   180 = portrait,  flipped
    ask "Display rotation / orientation:"
    echo -e "    ${DIM}90=landscape normal (default)  |  270=landscape flipped  |  0=portrait  |  180=portrait flipped${RESET}"
    read -r -p "    > [90] " DISP_ROTATE
    DISP_ROTATE="${DISP_ROTATE:-90}"
    # Validate
    case "$DISP_ROTATE" in
        0|90|180|270) ;;
        *) warn "Invalid rotation '${DISP_ROTATE}', defaulting to 90."; DISP_ROTATE=90 ;;
    esac
    ok "Rotation: ${BOLD}${DISP_ROTATE}°${RESET}"

    # ─ Install fonts if missing ─────────────────────────────────────────────
    info "Installing fonts (DejaVu Mono)..."
    sudo apt-get install -y -qq fonts-dejavu-core 2>/dev/null || true
    ok "Fonts ready."

    # ─ Download tft35a.dtbo from goodtft/LCD-show ──────────────────────────
    # This is the custom Device Tree overlay for the MPI3501 / generic XPT2046
    # display. It is NOT included in the standard Raspberry Pi kernel packages.
    local dtbo_url="https://raw.githubusercontent.com/goodtft/LCD-show/master/usr/tft35a-overlay.dtb"
    local dtbo_dst="${overlays_dir}/tft35a.dtbo"

    if [[ ! -f "$dtbo_dst" ]]; then
        info "Downloading tft35a.dtbo from goodtft/LCD-show..."
        if sudo wget -q --tries=3 --timeout=30 -O "$dtbo_dst" "$dtbo_url"; then
            ok "Downloaded: ${BOLD}${dtbo_dst}${RESET}"
        else
            warn "Download failed. Trying git clone fallback..."
            local tmp_lcd="/tmp/LCD-show-$$"
            if git clone --depth=1 https://github.com/goodtft/LCD-show.git "$tmp_lcd" 2>/dev/null; then
                sudo cp "$tmp_lcd/usr/tft35a-overlay.dtb" "$dtbo_dst"
                rm -rf "$tmp_lcd"
                ok "Installed via git clone: ${BOLD}${dtbo_dst}${RESET}"
            else
                err "Could not obtain tft35a.dtbo. Check internet connection."
                err "Manual install: copy tft35a.dtbo to ${overlays_dir}/ and re-run with --display"
                return 1
            fi
        fi
    else
        ok "tft35a.dtbo already present: ${BOLD}${dtbo_dst}${RESET}"
    fi

    # ─ Patch boot config (idempotent) ──────────────────────────────────────
    info "Patching ${BOLD}${boot_cfg}${RESET} for XPT2046/ILI9486 display..."

    # Remove any previous opensable display block (idempotent re-run)
    sudo sed -i '/# BEGIN opensable-display/,/# END opensable-display/d' "$boot_cfg" 2>/dev/null || true

    sudo tee -a "$boot_cfg" > /dev/null << BOOTCFG

# BEGIN opensable-display
# 3.5" XPT2046 / ILI9486 SPI display (MPI3501 / goodtft tft35a driver)
# Source: https://github.com/goodtft/LCD-show
dtparam=spi=on
# LCD panel via tft35a overlay (custom dtbo — ILI9486 + XPT2046)
# rotate: 90=landscape-normal, 270=landscape-flipped, 0/180=portrait
dtoverlay=tft35a:rotate=${DISP_ROTATE}
# Touch controller: XPT2046 is ADS7846-compatible
# cs=1  → TP_CS = SPI CE1 (GPIO7,  pin 26)
# penirq=17 → TP_IRQ = GPIO17 (pin 11) — confirmed MPI3501 datasheet
dtoverlay=ads7846,cs=1,penirq=17,speed=50000,keep_vref_on=0,swapxy=0,pmax=255,xohms=150
# END opensable-display
BOOTCFG
    ok "Boot config patched."

    # ─ Add current user to 'video' group ───────────────────────────────
    local cur_user; cur_user=$(who am i | awk '{print $1}' 2>/dev/null || echo "${SUDO_USER:-$USER}")
    if ! groups "$cur_user" 2>/dev/null | grep -q '\bvideo\b'; then
        info "Adding ${BOLD}${cur_user}${RESET} to 'video' group (needed for /dev/fb1)..."
        sudo usermod -aG video "$cur_user"
        ok "User added to video group (takes effect after re-login or reboot)."
    else
        ok "User already in 'video' group."
    fi

    # ─ Install systemd service ───────────────────────────────────────
    info "Installing sable-display systemd service..."
    local svc_src="$SCRIPT_DIR/scripts/sable-display.service"
    local svc_dst="/etc/systemd/system/sable-display.service"

    if [[ -f "$svc_src" ]]; then
        sed \
            -e "s|PLACEHOLDER_USER|${cur_user}|g" \
            -e "s|PLACEHOLDER_DIR|${SCRIPT_DIR}|g" \
            -e "s|PLACEHOLDER_VENV|${VENV_DIR}|g" \
            "$svc_src" | sudo tee "$svc_dst" > /dev/null
        sudo systemctl daemon-reload
        sudo systemctl enable sable-display.service 2>/dev/null && \
            ok "Service enabled: ${BOLD}sable-display.service${RESET} (starts on boot after reboot)" || \
            warn "Could not enable service — start manually after reboot."
    else
        warn "Service template not found: ${svc_src}"
    fi

    # ─ Update PROFILE_ENV with display rotate ─────────────────────────
    if [[ -f "$PROFILE_ENV" ]]; then
        grep -q '^DISPLAY_ROTATE=' "$PROFILE_ENV" \
            && sed -i "s|^DISPLAY_ROTATE=.*|DISPLAY_ROTATE=${DISP_ROTATE}|" "$PROFILE_ENV" \
            || echo "DISPLAY_ROTATE=${DISP_ROTATE}" >> "$PROFILE_ENV"
        ok "DISPLAY_ROTATE=${DISP_ROTATE} written to profile."
    fi

    # ─ Quick-test script ──────────────────────────────────────────────
    cat > "$SCRIPT_DIR/test-display.sh" << TESTSH
#!/usr/bin/env bash
# Quick test for the 3.5" display — runs display_logs.py in foreground
cd "$(dirname "\$0")"
source venv/bin/activate
FB_DEV=\${1:-/dev/fb1} DISPLAY_INTERVAL=0.5 DISPLAY_ROTATE=${DISP_ROTATE} \
    python3 scripts/display_logs.py
TESTSH
    chmod +x "$SCRIPT_DIR/test-display.sh"
    ok "Test script created: ${BOLD}./test-display.sh${RESET}"

    sep
    echo -e "${YELLOW}${BOLD}  ⚠  REBOOT REQUIRED${RESET}"
    echo -e "  Driver: ${BOLD}tft35a${RESET} (goodtft/LCD-show — ILI9486 + XPT2046 / MPI3501)"
    echo -e "  Overlay added to : ${BOLD}${boot_cfg}${RESET}"
    echo -e "  dtbo installed at: ${BOLD}${dtbo_dst}${RESET}"
    echo -e "  After reboot the display will appear on ${BOLD}/dev/fb1${RESET}."
    echo -e ""
    echo -e "  After reboot:"
    echo -e "   ${CYAN}sudo systemctl start sable-display${RESET}   — start log viewer now"
    echo -e "   ${CYAN}./test-display.sh${RESET}                   — test display in foreground"
    echo -e "   ${CYAN}sudo systemctl status sable-display${RESET}  — check service status"
    echo -e "   ${CYAN}ls -la /dev/fb*${RESET}                     — confirm /dev/fb1 exists"
    echo -e ""
    ask "Reboot now? [y/N]"
    read -r -p "    > " DO_REBOOT
    if [[ "${DO_REBOOT,,}" == "y" ]]; then
        info "Rebooting..."
        sudo reboot
    else
        warn "Remember to reboot before expecting the display to work."
    fi
}

# ============================================================================
#  --repair / --verify: run checks then exit (skip full install)
# ============================================================================
if [[ "$MODE" == "repair" || "$MODE" == "verify" ]]; then
    clear
    if [[ "$MODE" == "repair" ]]; then
        echo -e "${YELLOW}${BOLD}  OpenSable Pi — Self-Repair Mode${RESET}"
        echo -e "${DIM}  Re-verifying all components, auto-fixing what's broken...${RESET}"
        echo -e "${DIM}  Tip: use ${BOLD}./install-pi.sh --display${DIM} to (re)configure the display.${RESET}"
    else
        echo -e "${CYAN}${BOLD}  OpenSable Pi — Verify Mode (read-only)${RESET}"
        echo -e "${DIM}  Checking installation state without making any changes...${RESET}"
    fi
    sep
    EXIT_CODE=0
    verify_directories        || EXIT_CODE=1
    verify_python_packages    || EXIT_CODE=1
    verify_opensable_package  || EXIT_CODE=1
    verify_profile_config     || EXIT_CODE=1
    verify_connectivity
    sep
    if (( EXIT_CODE == 0 )); then
        echo -e "\n${GREEN}${BOLD}  ✓  All checks passed.${RESET}\n"
    else
        echo -e "\n${RED}${BOLD}  ✗  Some checks failed — see above.${RESET}"
        echo -e "${DIM}  Log: ${REPAIR_LOG}${RESET}\n"
    fi
    exit $EXIT_CODE
fi

# ── Banner ────────────────────────────────────────────────────────────────────
clear
echo -e "${CYAN}${BOLD}"
cat << 'BANNER'
   ____                 _____       _     _
  / __ \               / ____|     | |   | |
 | |  | |_ __   ___ _ | (___   __ _| |__ | | ___
 | |  | | '_ \ / _ \ '_ \___ \ / _` | '_ \| |/ _ \
 | |__| | |_) |  __/ | | |___) | (_| | |_) | |  __/
  \____/| .__/ \___|_| |_|____/ \__,_|_.__/|_|\___|
        | |       Raspberry Pi Installer
        |_|       OpenWebUI Edition
BANNER
echo -e "${RESET}"
echo -e "${DIM}  No local GPU required — uses your OpenWebUI API as the LLM backend${RESET}"
sep

# ── Architecture check ────────────────────────────────────────────────────────
step "Checking system compatibility"

ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    if [[ "$ARCH" == "armv7l" ]]; then
        error "32-bit ARM (armv7l) detected — likely a Raspberry Pi 2B or older 32-bit OS image."
        error "This installer requires ${BOLD}64-bit Raspberry Pi OS${RESET}${RED} (aarch64)."
        error "Please flash ${BOLD}Raspberry Pi OS (64-bit)${RESET}${RED} to your SD card and retry."
        error "Minimum hardware: ${BOLD}Raspberry Pi 3B+${RESET}${RED} with 64-bit OS."
        exit 1
    else
        warn "Architecture is '$ARCH', not aarch64. Proceeding anyway — unsupported platform."
    fi
else
    ok "Architecture: ${BOLD}${ARCH}${RESET} (64-bit ARM — compatible)"
fi

# Pi model detection
PI_MODEL="Unknown"
if [[ -f /proc/device-tree/model ]]; then
    PI_MODEL=$(tr -d '\0' < /proc/device-tree/model)
    ok "Detected: ${BOLD}${PI_MODEL}${RESET}"
    if echo "$PI_MODEL" | grep -qiE "Pi 2|Pi Zero"; then
        error "Raspberry Pi 2 / Zero is not supported."
        error "Minimum requirement: ${BOLD}Raspberry Pi 3B+${RESET}${RED} (1 GB RAM, aarch64 OS)"
        exit 1
    fi
else
    warn "Could not read /proc/device-tree/model — skipping Pi model check."
fi

# RAM check
TOTAL_RAM_MB=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)
ok "Available RAM: ${BOLD}${TOTAL_RAM_MB} MB${RESET}"
if (( TOTAL_RAM_MB < 900 )); then
    warn "Less than 1 GB RAM detected. OpenSable needs ~512 MB to run; performance may be limited."
fi

# ── Dependency check ──────────────────────────────────────────────────────────
step "Installing system dependencies"

info "Updating package list (this may take a moment)..."
sudo apt-get update -qq

PACKAGES=(
    python3
    python3-pip
    python3-venv
    python3-dev
    build-essential
    libffi-dev
    libssl-dev
    git
    curl
    sqlite3
    libsqlite3-dev
    libjpeg-dev
    zlib1g-dev
    libopenblas-dev     # speeds up numpy on ARM
)

MISSING=()
for pkg in "${PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING+=("$pkg")
    fi
done

if (( ${#MISSING[@]} > 0 )); then
    info "Installing: ${MISSING[*]}"
    sudo apt-get install -y --no-install-recommends "${MISSING[@]}" 2>&1 | tail -5
    ok "System packages installed."
else
    ok "All system packages already present."
fi

# ── Python venv ───────────────────────────────────────────────────────────────
step "Setting up Python virtual environment"

PYTHON_VERSION=$(python3 --version 2>&1)
ok "Python: ${BOLD}${PYTHON_VERSION}${RESET}"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at ${BOLD}$VENV_DIR${RESET} ..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created."
else
    ok "Virtual environment already exists — reusing."
fi

source "$VENV_DIR/bin/activate"
"$VENV_DIR/bin/pip" install --upgrade pip wheel setuptools -q

# ── Python packages ───────────────────────────────────────────────────────────
step "Installing Python packages (Pi-optimized, skipping heavy ML deps)"

# Core packages that work on ARM aarch64
CORE_PACKAGES=(
    # Web / async
    aiohttp
    aiofiles
    fastapi
    uvicorn[standard]
    websockets
    httpx
    # Telegram
    python-telegram-bot
    # Data / utils
    pydantic
    python-dotenv
    rich
    typer
    pillow              # image handling (no GPU)
    cryptography
    # Storage
    SQLAlchemy
    aiosqlite
    # Embeddings (light — no torch required)
    sentence-transformers
    # OpenAI-compatible client (for OpenWebUI API calls)
    openai
    # Misc
    tenacity
    structlog
    psutil
    schedule
)

# Packages to explicitly SKIP on Pi (too heavy or require GPU/X11)
# - torch / torchvision      (10+ GB, no CUDA on Pi)
# - ultralytics (YOLOv8)     (requires torch)
# - pyautogui                (requires X11 display)
# - playwright               (chromium headful — huge)
# - whisper (openai-whisper) (CPU inference is too slow)

info "Installing core packages (may take a few minutes on Pi)..."
mkdir -p "$(dirname "$REPAIR_LOG")"
"$VENV_DIR/bin/pip" install -q "${CORE_PACKAGES[@]}" 2>&1 \
    | tee -a "$REPAIR_LOG" \
    | grep -E "Successfully installed|ERROR|error" || true

if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    info "Installing OpenSable package (editable)..."
    "$VENV_DIR/bin/pip" install --no-deps -e "$SCRIPT_DIR" -q
fi

ok "Initial package installation done."

# ── POST-INSTALL VERIFICATION & AUTO-REPAIR ───────────────────────────────────
step "Post-install verification & auto-repair"
info "Testing every import — will auto-fix anything broken..."
echo ""

FAILED_TOTAL=0
REPAIRED_TOTAL=0

for pkg in "${!PKG_IMPORTS[@]}"; do
    import="${PKG_IMPORTS[$pkg]}"
    if try_import "$import"; then
        ok "  ${BOLD}${pkg}${RESET}"
    else
        warn "  ${BOLD}${pkg}${RESET} — broken, repairing..."
        if install_package "$pkg" && try_import "$import"; then
            fixed "  ${BOLD}${pkg}${RESET} — repaired"
            (( REPAIRED_TOTAL++ )) || true
        else
            error "  ${BOLD}${pkg}${RESET} — could not repair"
            (( FAILED_TOTAL++ )) || true
        fi
    fi
done

echo ""
if (( FAILED_TOTAL > 0 )); then
    warn "${FAILED_TOTAL} package(s) could not be installed. Agent may start but some features unavailable."
    warn "Re-run: ${BOLD}./install-pi.sh --repair${RESET}"
elif (( REPAIRED_TOTAL > 0 )); then
    ok "All packages OK — ${REPAIRED_TOTAL} auto-repaired during install."
else
    ok "All packages verified clean — nothing needed repair."
fi

verify_opensable_package
verify_directories

# ── Configuration ─────────────────────────────────────────────────────────────
step "Configuring Pi agent profile"

mkdir -p "$PROFILE_DIR"

echo ""
sep
echo -e "${YELLOW}${BOLD}  Let's configure your OpenSable Pi agent.${RESET}"
echo -e "${DIM}  Press Enter to accept defaults shown in [brackets].${RESET}"
sep
echo ""

# OpenWebUI API
ask "Your OpenWebUI API base URL:"
echo -e "    ${DIM}Example: https://sofia.zunvra.com${RESET}"
read -r -p "    > " OWUI_URL
OWUI_URL="${OWUI_URL:-https://sofia.zunvra.com}"

ask "OpenWebUI API Key (leave blank if your instance has no auth):"
read -r -s -p "    > " OWUI_KEY
echo ""

ask "OpenWebUI model name to use (must support vision for image analysis):"
echo -e "    ${DIM}Example: llama3.2:latest  |  llava:13b  |  gemma3:12b${RESET}"
echo -e "    ${DIM}Tip: any vision-capable model loaded in your OpenWebUI works.${RESET}"
read -r -p "    > [llama3.2:latest] " OWUI_MODEL
OWUI_MODEL="${OWUI_MODEL:-llama3.2:latest}"

# Telegram
ask "Telegram Bot Token (from @BotFather):"
read -r -p "    > " TG_TOKEN
TG_TOKEN="${TG_TOKEN:-}"

ask "Your Telegram user ID (numeric, e.g. 828351902):"
read -r -p "    > " TG_USER_ID
TG_USER_ID="${TG_USER_ID:-}"

# Agent name
ask "Agent name [Sable-Pi]:"
read -r -p "    > " AGENT_NAME
AGENT_NAME="${AGENT_NAME:-Sable-Pi}"

# ── Write profile.env ─────────────────────────────────────────────────────────
step "Writing ${PROFILE_ENV}"

cat > "$PROFILE_ENV" << EOF
# ============================================================================
#  OpenSable — Raspberry Pi Profile
#  Generated by install-pi.sh on $(date)
#  Device: ${PI_MODEL}
# ============================================================================

# ── Interface ────────────────────────────────────────────────────────────────
CLI_ENABLED=false
PIXEL_BRIDGE_ENABLED=false     # no Electron on Pi

# ── LLM: use OpenWebUI API (no local Ollama) ─────────────────────────────────
# Ollama is disabled — all inference goes through OpenWebUI
OLLAMA_BASE_URL=                # intentionally blank
DEFAULT_MODEL=${OWUI_MODEL}
AUTO_SELECT_MODEL=false
LOW_VRAM_MODE=false             # not relevant — no local model

# OpenWebUI (sofia.zunvra.com or your instance)
OPENWEBUI_API_URL=${OWUI_URL}
OPENWEBUI_API_KEY=${OWUI_KEY}
OPENWEBUI_MODEL=${OWUI_MODEL}

# ── Telegram (primary interface) ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN=${TG_TOKEN}
TELEGRAM_ALLOWED_USERS=${TG_USER_ID}

# ── Agent identity ────────────────────────────────────────────────────────────
AGENT_NAME=${AGENT_NAME}
AGENT_PERSONALITY=helpful

# ── Disabled on Pi (no GPU / no display / no Node.js bridge needed) ──────────
VISION_ENABLED=false            # local YOLOv8 requires torch
DESKTOP_ENABLED=false           # requires Electron GUI
WHATSAPP_ENABLED=false          # requires Node.js bridge
VOICE_ENABLED=false             # Whisper CPU is too slow on Pi
AUTONOMOUS_ENABLED=false        # optional — enable once stable

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR=./data
LOG_FILE=./logs/opensable.log
LOG_LEVEL=INFO
VECTOR_DB_PATH=./data/vectordb

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_RETENTION_DAYS=30
MAX_CONTEXT_LENGTH=4096         # keep low for 1 GB RAM

# ── Networking ────────────────────────────────────────────────────────────────
MOBILE_API_ENABLED=false
GATEWAY_ENABLED=false
WEBCHAT_HOST=0.0.0.0
WEBCHAT_PORT=8789

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=30
RATE_LIMIT_WINDOW=60
EOF

ok "Profile written to ${BOLD}${PROFILE_ENV}${RESET}"

# ── Start script shortcut ─────────────────────────────────────────────────────
step "Creating Pi start shortcut"

START_SCRIPT="$SCRIPT_DIR/start-pi.sh"
cat > "$START_SCRIPT" << 'STARTSH'
#!/usr/bin/env bash
# Start OpenSable with the Pi profile
# Usage: ./start-pi.sh [--repair]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--repair" ]]; then
    echo "Running self-repair before starting..."
    bash "$SCRIPT_DIR/install-pi.sh" --repair || true
fi
source "$SCRIPT_DIR/venv/bin/activate"
cd "$SCRIPT_DIR"
exec python3 -m opensable --profile pi "$@"
STARTSH
chmod +x "$START_SCRIPT"
ok "Created ${BOLD}./start-pi.sh${RESET}  (use ${BOLD}./start-pi.sh --repair${RESET} to self-fix on start)"

# ── Final verification pass ───────────────────────────────────────────────────
step "Final verification pass"
verify_profile_config
verify_connectivity

# ── Display setup (optional) ─────────────────────────────────────────────
if [[ "$SETUP_DISPLAY" == "true" ]]; then
    setup_display
else
    sep
    ask "Do you have a 3.5\" XPT2046 touchscreen attached? Set it up now? [y/N]"
    read -r -p "    > " _DISP_ANSWER
    if [[ "${_DISP_ANSWER,,}" == "y" ]]; then
        setup_display
    else
        info "Skipping display setup. Run ${BOLD}./install-pi.sh --display${RESET} anytime to add it."
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
sep
echo ""
echo -e "${GREEN}${BOLD}  ✓  Installation complete!${RESET}"
echo ""
echo -e "  ${BOLD}Self-repair commands:${RESET}"
echo -e "    ${CYAN}./install-pi.sh --repair${RESET}   — auto-fix any missing dependency"
echo -e "    ${CYAN}./install-pi.sh --verify${RESET}   — check only, no changes"
echo -e "    ${CYAN}./install-pi.sh --display${RESET}  — configure 3.5\" XPT2046 display"
echo -e "    ${CYAN}./start-pi.sh   --repair${RESET}   — repair then start immediately"
echo ""
echo -e "  ${BOLD}Display:${RESET}"
echo -e "    ${CYAN}./test-display.sh${RESET}                     — test log viewer in foreground"
echo -e "    ${DIM}sudo systemctl start sable-display${RESET}    — start via systemd"
echo -e "    ${DIM}sudo systemctl status sable-display${RESET}   — check status"
echo ""
echo -e "  ${BOLD}Edit config:${RESET}  ${BOLD}${PROFILE_ENV}${RESET}"
echo -e "  ${BOLD}Start:${RESET}        ${GREEN}${BOLD}./start-pi.sh${RESET}"
echo -e "  ${BOLD}Repair log:${RESET}   ${DIM}${REPAIR_LOG}${RESET}"
echo ""
sep
