#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Build OpenSable Installer for the current platform
#  Run from the installers/ directory
# ═══════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"

# Ensure PyInstaller is installed
if ! command -v pyinstaller &>/dev/null; then
    echo "[!] PyInstaller not found. Installing..."
    pip install pyinstaller
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        SPEC="opensable-installer-linux.spec"
        OUTPUT="OpenSable-Installer-Linux-x86_64"
        ;;
    Darwin)
        SPEC="opensable-installer.spec"
        OUTPUT="OpenSable-Installer.app"
        ;;
    *)
        echo "[ERROR] Unsupported OS: $OS"
        echo "On Windows, use: build-windows.bat"
        exit 1
        ;;
esac

echo "===== OpenSable Installer Build ====="
echo "  OS:   $OS"
echo "  Arch: $ARCH"
echo "  Spec: $SPEC"
echo ""

pyinstaller "$SPEC" --clean --noconfirm

echo ""
echo "===== Build complete! ====="
echo "Output: dist/$OUTPUT"
ls -lh "dist/$OUTPUT" 2>/dev/null || ls -lhd "dist/$OUTPUT" 2>/dev/null
