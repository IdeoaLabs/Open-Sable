# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['installers/installer_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('installers/assets', 'assets')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Open-Sable Installer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['installers/assets/icon.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Open-Sable Installer',
)
app = BUNDLE(
    coll,
    name='Open-Sable Installer.app',
    icon='installers/assets/icon.icns',
    bundle_identifier='com.ideoalabs.opensable.installer',
)
