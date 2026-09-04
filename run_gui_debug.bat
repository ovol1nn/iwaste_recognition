@echo off
setlocal
cd /d "%~dp0"
set "PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON if exist "%~dp0.venv312\Scripts\python.exe" set "PYTHON=%~dp0.venv312\Scripts\python.exe"
if not defined PYTHON if exist "%~dp0..\.venv312\Scripts\python.exe" set "PYTHON=%~dp0..\.venv312\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python.exe"
echo Running GUI in console mode. Close the GUI window to return here.
"%PYTHON%" -m solid_waste_model.gui
echo.
echo GUI exited with code %ERRORLEVEL%.
pause
