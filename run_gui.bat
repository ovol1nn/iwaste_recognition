@echo off
setlocal
cd /d "%~dp0"

set "PYTHONW="
call :try_env "%~dp0.venv"
call :try_env "%~dp0.venv312"
call :try_env "%~dp0..\.venv312"

if not defined PYTHONW (
  where python.exe >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Python 3.12 was not found.
    echo Install Python 3.12 and create .venv, then install requirements-runtime-dinov2.txt.
    pause
    exit /b 1
  )
  python.exe -c "import torch, tkinter" >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] No usable Python environment was found.
    echo The GUI requires PyTorch, TorchVision and Tkinter.
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\python.exe -m pip install -r requirements-runtime-dinov2.txt
    pause
    exit /b 1
  )
  set "PYTHONW=pythonw.exe"
)

if not exist "output" mkdir "output"
echo Starting DINOv2 GUI with: %PYTHONW%
start "DINOv2 GUI" "%PYTHONW%" -m solid_waste_model.gui
exit /b 0

:try_env
if defined PYTHONW exit /b 0
if not exist "%~1\Scripts\python.exe" exit /b 0
"%~1\Scripts\python.exe" -c "import torch, tkinter" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYTHONW=%~1\Scripts\pythonw.exe"
exit /b 0
