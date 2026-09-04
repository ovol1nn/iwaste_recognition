from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CORE_DIR = PACKAGE_ROOT / "core"
CONFIGS_DIR = PACKAGE_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CURRENT_ARTIFACTS_DIR = ARTIFACTS_DIR / "current"
HISTORY_ARTIFACTS_DIR = ARTIFACTS_DIR / "history"

WORKSPACE_DIR = PROJECT_ROOT / "workspace"
DOCS_DIR = PROJECT_ROOT / "docs"
ML_DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = ML_DATA_DIR / "raw"
CURATED_DATA_DIR = ML_DATA_DIR / "curated"
MANIFESTS_DIR = ML_DATA_DIR / "manifests"
GENERATED_DATA_DIR = ML_DATA_DIR / "generated"
OFFLINE_AUG_DIR = GENERATED_DATA_DIR / "offline_aug"
TEST_FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

DEFAULT_MODEL_VERSION = "solid_waste_dinov2_proto_v1"
DEFAULT_PROB_THRESHOLD = 0.80
DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_IMAGE_SIZE = 448

CURRENT_MODEL_FILENAMES = {
    "pth": "model.pth",
    "onnx": "model.onnx",
    "meta": "model.meta.json",
    "prototype": "prototype_features.npz",
    "calibration": "calibration_report.json",
    "best_known": "model.best_known.pth",
    "best_reject": "model.best_reject.pth",
}

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SCENE_NAMES = ("实验室", "料棚")

DEFAULT_LABEL_SCHEMA_PATH = CONFIGS_DIR / "label_schema.json"
DEFAULT_PROFILE_PATH = CONFIGS_DIR / "material_profiles.json"
DEFAULT_META_TEMPLATE_PATH = CONFIGS_DIR / "model_meta_template.json"
DEFAULT_TRAINING_DEFAULTS_PATH = CONFIGS_DIR / "training_defaults.json"
DEFAULT_BAD_IMAGES_PATH = CONFIGS_DIR / "bad_images.txt"
DEFAULT_MANIFEST_CSV = MANIFESTS_DIR / "manifest.csv"
DEFAULT_MANIFEST_JSONL = MANIFESTS_DIR / "manifest.jsonl"
DEFAULT_REGRESSION_CASES = TEST_FIXTURES_DIR / "regression_cases.json"

DEFAULT_CURRENT_PTH_PATH = CURRENT_ARTIFACTS_DIR / CURRENT_MODEL_FILENAMES["pth"]
DEFAULT_CURRENT_ONNX_PATH = CURRENT_ARTIFACTS_DIR / CURRENT_MODEL_FILENAMES["onnx"]
DEFAULT_CURRENT_META_PATH = CURRENT_ARTIFACTS_DIR / CURRENT_MODEL_FILENAMES["meta"]
DEFAULT_CURRENT_PROTOTYPE_PATH = CURRENT_ARTIFACTS_DIR / CURRENT_MODEL_FILENAMES["prototype"]
DEFAULT_CURRENT_CALIBRATION_REPORT_PATH = CURRENT_ARTIFACTS_DIR / CURRENT_MODEL_FILENAMES["calibration"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


LABEL_SCHEMA = load_json(DEFAULT_LABEL_SCHEMA_PATH)
MATERIAL_SPECS = LABEL_SCHEMA["material_type"]
PARTICLE_SIZE_LABELS = LABEL_SCHEMA["particle_size"]
SHAPE_LABELS = LABEL_SCHEMA["shape"]
COLOR_LABELS = LABEL_SCHEMA["color"]

KNOWN_MATERIAL_SPECS = [spec for spec in MATERIAL_SPECS if spec.get("known_class", False)]
REJECT_MATERIAL_SPECS = [spec for spec in MATERIAL_SPECS if not spec.get("known_class", False)]

MATERIAL_TYPE_LABELS = [spec["display_name"] for spec in KNOWN_MATERIAL_SPECS]
KNOWN_CLASS_NAMES = set(MATERIAL_TYPE_LABELS)
DEFAULT_REJECT_CLASS_NAMES = {spec["display_name"] for spec in REJECT_MATERIAL_SPECS}

DIRECTORY_TO_DISPLAY_NAME = {spec["directory_name"]: spec["display_name"] for spec in MATERIAL_SPECS}
DISPLAY_NAME_TO_DIRECTORY = {spec["display_name"]: spec["directory_name"] for spec in MATERIAL_SPECS}
DISPLAY_NAME_TO_KEY = {spec["display_name"]: spec["key"] for spec in MATERIAL_SPECS}
MATERIAL_SPECS_BY_DISPLAY = {spec["display_name"]: spec for spec in MATERIAL_SPECS}


def build_history_filenames(model_version: str) -> dict[str, str]:
    return {
        "pth": f"{model_version}.pth",
        "onnx": f"{model_version}.onnx",
        "meta": f"{model_version}.meta.json",
        "prototype": f"{model_version}.prototype_features.npz",
        "calibration": f"{model_version}.calibration_report.json",
        "best_known": f"{model_version}.best_known.pth",
        "best_reject": f"{model_version}.best_reject.pth",
    }


def build_history_artifact_dir(model_version: str) -> Path:
    return HISTORY_ARTIFACTS_DIR / model_version


MODEL_FILENAMES = build_history_filenames(DEFAULT_MODEL_VERSION)
DEFAULT_META_PATH = DEFAULT_META_TEMPLATE_PATH
DEFAULT_ARTIFACT_META_PATH = DEFAULT_CURRENT_META_PATH
DEFAULT_PROTOTYPE_PATH = DEFAULT_CURRENT_PROTOTYPE_PATH
DEFAULT_CALIBRATION_REPORT_PATH = DEFAULT_CURRENT_CALIBRATION_REPORT_PATH
DATASET_ROOT = RAW_DATA_DIR
