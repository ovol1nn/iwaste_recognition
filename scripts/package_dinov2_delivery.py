from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "solid_waste_dinov2_proto_v1"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / MODEL_VERSION
HUB_CACHE_DIR = PROJECT_ROOT / "runtime" / "torch_hub"
DINO_SOURCE_DIR = HUB_CACHE_DIR / "facebookresearch_dinov2_main"
DINO_CHECKPOINT = HUB_CACHE_DIR / "checkpoints" / "dinov2_vits14_pretrain.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "delivery" / MODEL_VERSION


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))


def write_launcher(output_dir: Path) -> None:
    script = f'''@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\\Scripts\\python.exe" set "PYTHON=.venv\\Scripts\\python.exe"
if "%~1"=="" (
  echo Usage: run_predict.bat ^<image-or-folder^>
  exit /b 2
)
"%PYTHON%" -m solid_waste_model.dinov2_proto predict "%~1" ^
  --recursive ^
  --prototype "artifacts\\{MODEL_VERSION}.prototypes.npz" ^
  --meta "artifacts\\{MODEL_VERSION}.meta.json" ^
  --local-repository "runtime\\torch_hub\\facebookresearch_dinov2_main" ^
  --cache-dir "runtime\\torch_hub" ^
  --device cuda ^
  --enable-rejection ^
  --save-json "output\\prediction_results.json"
'''
    (output_dir / "run_predict.bat").write_text(script, encoding="utf-8")


def write_checksums(output_dir: Path) -> None:
    lines: list[str] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_zip(output_dir: Path) -> Path:
    archive_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in output_dir.rglob("*"):
            if path.is_file():
                archive.write(path, Path(output_dir.name) / path.relative_to(output_dir))
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an offline-capable DINOv2 solid-waste model delivery package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zip", action="store_true", help="Also create a ZIP archive next to the delivery directory.")
    args = parser.parse_args()

    required_files = [
        ARTIFACT_DIR / f"{MODEL_VERSION}.prototypes.npz",
        ARTIFACT_DIR / f"{MODEL_VERSION}.meta.json",
        ARTIFACT_DIR / f"{MODEL_VERSION}.report.json",
        DINO_CHECKPOINT,
        PROJECT_ROOT / "solid_waste_model" / "core" / "dinov2_proto.py",
        PROJECT_ROOT / "solid_waste_model" / "core" / "image_io.py",
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing or not DINO_SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"delivery inputs missing: {missing or [DINO_SOURCE_DIR]}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    artifact_output = args.output_dir / "artifacts"
    for name in ["prototypes.npz", "meta.json", "report.json"]:
        copy_file(ARTIFACT_DIR / f"{MODEL_VERSION}.{name}", artifact_output / f"{MODEL_VERSION}.{name}")
    copy_tree(DINO_SOURCE_DIR, args.output_dir / "runtime" / "torch_hub" / DINO_SOURCE_DIR.name)
    copy_file(DINO_CHECKPOINT, args.output_dir / "runtime" / "torch_hub" / "checkpoints" / DINO_CHECKPOINT.name)

    package_output = args.output_dir / "solid_waste_model"
    package_output.mkdir(parents=True, exist_ok=True)
    (package_output / "__init__.py").write_text(
        '"""DINOv2 solid-waste runtime package."""\n',
        encoding="utf-8",
    )
    copy_file(PROJECT_ROOT / "solid_waste_model" / "config.py", package_output / "config.py")
    copy_tree(PROJECT_ROOT / "solid_waste_model" / "configs", package_output / "configs")
    copy_file(PROJECT_ROOT / "solid_waste_model" / "dinov2_proto.py", package_output / "dinov2_proto.py")
    copy_file(PROJECT_ROOT / "solid_waste_model" / "dinov2_inference.py", package_output / "dinov2_inference.py")
    copy_file(PROJECT_ROOT / "solid_waste_model" / "core" / "__init__.py", package_output / "core" / "__init__.py")
    copy_file(PROJECT_ROOT / "solid_waste_model" / "core" / "config.py", package_output / "core" / "config.py")
    copy_file(PROJECT_ROOT / "solid_waste_model" / "core" / "dinov2_proto.py", package_output / "core" / "dinov2_proto.py")
    copy_file(PROJECT_ROOT / "solid_waste_model" / "core" / "dinov2_inference.py", package_output / "core" / "dinov2_inference.py")
    copy_file(PROJECT_ROOT / "solid_waste_model" / "core" / "image_io.py", package_output / "core" / "image_io.py")

    copy_file(PROJECT_ROOT / "requirements-runtime-dinov2.txt", args.output_dir / "requirements-runtime-dinov2.txt")
    copy_file(PROJECT_ROOT / "docs" / "DINOV2_DELIVERY_GUIDE.md", args.output_dir / "docs" / "DINOV2_DELIVERY_GUIDE.md")
    copy_file(PROJECT_ROOT / "docs" / "DINOV2_INPUT_OUTPUT.md", args.output_dir / "docs" / "DINOV2_INPUT_OUTPUT.md")
    write_launcher(args.output_dir)
    write_checksums(args.output_dir)
    print(f"delivery_dir={args.output_dir}")
    if args.zip:
        print(f"delivery_zip={create_zip(args.output_dir)}")


if __name__ == "__main__":
    main()
