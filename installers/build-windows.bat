@echo off
REM ═══════════════════════════════════════════════════════════
REM  Build OpenSable Installer for Windows (x86 + x64)
REM  Run from the installers/ directory
REM ═══════════════════════════════════════════════════════════
setlocal

echo ===== OpenSable Installer,  Windows Build =====
echo.

where pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] PyInstaller not found. Installing...
    pip install pyinstaller
)

echo.
echo [1/2] Building for current architecture...
pyinstaller opensable-installer-windows.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo ===== Build complete! =====
echo Output: dist\OpenSable-Installer-Windows-*.exe
echo.
echo NOTE: This builds for your current Python architecture.
echo   - 64-bit Python  →  OpenSable-Installer-Windows-x64.exe
echo   - 32-bit Python  →  OpenSable-Installer-Windows-x86.exe
echo   To build both, run this script with both 32-bit and 64-bit Python.
echo.
dir /b dist\OpenSable-Installer-Windows-*.exe 2>nul
pause
