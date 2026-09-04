@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
if exist ".venv312\Scripts\python.exe" set "PYTHON=.venv312\Scripts\python.exe"
if "%~1"=="" (
  echo Usage: run_predict.bat ^<image-or-folder^>
  exit /b 2
)
"%PYTHON%" -m solid_waste_model.dinov2_proto predict "%~1" --recursive --prototype "artifacts\solid_waste_dinov2_proto_v1\solid_waste_dinov2_proto_v1.prototypes.npz" --meta "artifacts\solid_waste_dinov2_proto_v1\solid_waste_dinov2_proto_v1.meta.json" --local-repository "runtime\torch_hub\facebookresearch_dinov2_main" --cache-dir "runtime\torch_hub" --device cuda --enable-rejection --save-json "output\prediction_results.json"
