@echo off
setlocal enabledelayedexpansion

REM Open-Sable Smart Installer builder (Windows only)
REM Usage:
REM   build-windows-smart.bat
REM     -> builds with current python architecture
REM
REM   build-windows-smart.bat --all
REM     -> builds x86 and x64 using env vars:
REM        PYTHON_X86=C:\Path\To\Python32\python.exe
REM        PYTHON_X64=C:\Path\To\Python64\python.exe

set SCRIPT_DIR=%~dp0
set SPEC=%SCRIPT_DIR%opensable-installer-windows-smart.spec
set ENTRY=%SCRIPT_DIR%installer_gui_windows_smart.py

if not exist "%SPEC%" (
  echo [ERROR] Spec file not found: %SPEC%
  exit /b 1
)

if not exist "%ENTRY%" (
  echo [ERROR] Entry file not found: %ENTRY%
  exit /b 1
)

if "%~1"=="--all" goto BUILD_ALL

call :BUILD_ONE python "current"
exit /b %errorlevel%

:BUILD_ALL
if "%PYTHON_X86%"=="" (
  echo [ERROR] PYTHON_X86 is not defined.
  exit /b 1
)
if "%PYTHON_X64%"=="" (
  echo [ERROR] PYTHON_X64 is not defined.
  exit /b 1
)

call :BUILD_ONE "%PYTHON_X86%" "x86"
if errorlevel 1 exit /b 1

call :BUILD_ONE "%PYTHON_X64%" "x64"
if errorlevel 1 exit /b 1

echo [OK] Both x86 and x64 builds completed.
exit /b 0

:BUILD_ONE
set PY_EXE=%~1
set BUILD_LABEL=%~2

echo.
echo ============================================================
echo Building Smart Installer (%BUILD_LABEL%)
echo Python: %PY_EXE%
echo ============================================================

set PY_EXE_CHECK=%PY_EXE%
echo %PY_EXE_CHECK% | findstr /C:"\" >nul
if not errorlevel 1 (
  if not exist "%PY_EXE_CHECK%" (
    echo [ERROR] Python executable not found: %PY_EXE_CHECK%
    exit /b 1
  )
) else (
  where "%PY_EXE_CHECK%" >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python executable not found in PATH: %PY_EXE_CHECK%
    exit /b 1
  )
)

"%PY_EXE%" -m py_compile "%ENTRY%"
if errorlevel 1 (
  echo [ERROR] Syntax check failed for installer_gui_windows_smart.py
  exit /b 1
)

"%PY_EXE%" -m pip show pyinstaller >nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing PyInstaller...
  "%PY_EXE%" -m pip install pyinstaller
  if errorlevel 1 (
    echo [ERROR] Failed to install PyInstaller
    exit /b 1
  )
)

pushd "%SCRIPT_DIR%"
"%PY_EXE%" -m PyInstaller "%SPEC%" --clean --noconfirm
set BUILD_RC=%errorlevel%
popd

if not "%BUILD_RC%"=="0" (
  echo [ERROR] PyInstaller build failed (%BUILD_LABEL%)
  exit /b 1
)

echo [OK] Build completed (%BUILD_LABEL%).
exit /b 0
