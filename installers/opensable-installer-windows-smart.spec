# -*- mode: python ; coding: utf-8 -*-
# Windows onefile build for Smart Installer
# Build on Windows with:
#   pyinstaller opensable-installer-windows-smart.spec --clean --noconfirm
# For x86: use 32-bit Python interpreter
# For x64: use 64-bit Python interpreter

import struct
from pathlib import Path

block_cipher = None
HERE = Path(SPECPATH)

bits = struct.calcsize("P") * 8
arch_tag = "x64" if bits == 64 else "x86"


a = Analysis(
    [str(HERE / "installer_gui_windows_smart.py")],
    pathex=[str(HERE)],
    binaries=[],
    datas=[(str(HERE / "assets"), "assets")],
    hiddenimports=[
        "tkinter",
        "tkinter.ttk",
        "tkinter.messagebox",
        "tkinter.filedialog",
        "urllib.request",
        "urllib.error",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "pytest",
        "PIL",
        "cv2",
        "torch",
        "tensorflow",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=f"OpenSable-SmartInstaller-Windows-{arch_tag}",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # Keep resources untouched; some environments can strip icon metadata when UPX is enabled.
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(HERE / "assets" / "icon.ico"),
)
