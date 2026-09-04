@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
if exist ".venv312\Scripts\python.exe" set "PYTHON=.venv312\Scripts\python.exe"
if not exist "data\manifests\manifest.csv" "%PYTHON%" -m solid_waste_model.manifest
"%PYTHON%" -m solid_waste_model.dinov2_proto train --model-version solid_waste_dinov2_proto_v1 --manifest data\manifests\manifest.csv --output-dir artifacts\solid_waste_dinov2_proto_v1 --local-repository runtime\torch_hub\facebookresearch_dinov2_main --cache-dir runtime\torch_hub --device cuda --batch-size 8 --image-size 448 --prototypes-per-class 3 --crop-ratio 0.78
