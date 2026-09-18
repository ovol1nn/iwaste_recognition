"""Validate and prepare the manual round-two view/crop ablation plan.

The script writes a plan and prints manual commands only.  It never trains a
model, and it keeps the K=3 seed-42 augmented dataset and held-out batches
identical to the current control.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "IWaste_rec_model" / "data" / "manifests" / "round2_augmented_manifest.csv"
CONTROL_META = PROJECT_ROOT / "IWaste_rec_model" / "artifacts" / "solid_waste_dinov2_proto_round2_augmented_aug1" / "solid_waste_dinov2_proto_round2_augmented_aug1.meta.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "IWaste_rec_model" / "data" / "manifests" / "round2_view_crop_sweep_plan.json"
CONTROL_CROPS = ["full", "center", "top_left", "top_right", "bottom_left", "bottom_right"]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_data(rows: list[dict[str, str]]) -> dict[str, int]:
    if not rows:
        raise ValueError("augmentation manifest is empty")
    if any(row["augmentation_source"] != "original" for row in rows if row["split"] in {"val", "test"}):
        raise ValueError("validation and test rows must remain original images")
    counts = Counter(row["usage_role"] for row in rows)
    return {
        "train_rows": counts["known_train"],
        "validation_rows": counts["known_val"],
        "test_rows": counts["known_test"],
        "augmented_train_rows": sum(
            row["usage_role"] == "known_train" and row["augmentation_source"] == "offline_aug" for row in rows
        ),
    }


def validate_control(manifest: Path) -> dict[str, object]:
    if not CONTROL_META.is_file():
        raise FileNotFoundError(f"K=3 control artifact missing: {CONTROL_META}")
    control = json.loads(CONTROL_META.read_text(encoding="utf-8"))
    if int(control["prototypes_per_class_max"]) != 3:
        raise ValueError("control must use K=3")
    if Path(str(control["manifest"])).name != manifest.name:
        raise ValueError("control does not use the selected augmentation manifest")
    if list(control["crop_names"]) != CONTROL_CROPS or float(control["crop_ratio"]) != 0.78:
        raise ValueError("control does not match the expected six-view, crop-ratio=0.78 setting")
    return {
        "model_version": control["model_version"],
        "artifact_dir": str(CONTROL_META.parent),
        "prototypes_per_class": 3,
        "crops": control["crop_names"],
        "crop_ratio": control["crop_ratio"],
        "test_raw_accuracy": control["metrics"]["test"]["raw_accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the manual view/crop ablation; does not train.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_counts = validate_data(read_manifest(args.manifest))
    control = validate_control(args.manifest)
    common = {
        "manifest": str(args.manifest.relative_to(PROJECT_ROOT).as_posix()),
        "include_augmented_train": True,
        "prototypes_per_class": 3,
        "image_size": 448,
        "batch_size": 8,
        "class_temperature": 0.07,
        "vote_temperature": 0.03,
        "rejection_quantile": 0.10,
        "kmeans_iterations": 50,
        "device": "cuda",
    }
    experiments = [
        {
            "name": "V0_control_six_crop078",
            "purpose": "Existing K=3 control; six views at crop ratio 0.78.",
            "crops": CONTROL_CROPS,
            "crop_ratio": 0.78,
            "model_version": control["model_version"],
            "output_dir": str(Path(str(control["artifact_dir"])).relative_to(PROJECT_ROOT).as_posix()),
            "manual_training_required": False,
        },
        {
            "name": "V1_full_only",
            "purpose": "Measure recognition based on global material appearance without local crops.",
            "crops": ["full"],
            "crop_ratio": 0.78,
            "model_version": "solid_waste_dinov2_proto_round2_view_full_k3_augseed42",
            "output_dir": "IWaste_rec_model/artifacts/solid_waste_dinov2_proto_round2_view_full_k3_augseed42",
            "manual_training_required": True,
        },
        {
            "name": "V2_full_center",
            "purpose": "Keep global and central material detail while removing four corner crops.",
            "crops": ["full", "center"],
            "crop_ratio": 0.78,
            "model_version": "solid_waste_dinov2_proto_round2_view_full_center_k3_augseed42",
            "output_dir": "IWaste_rec_model/artifacts/solid_waste_dinov2_proto_round2_view_full_center_k3_augseed42",
            "manual_training_required": True,
        },
        {
            "name": "V3_six_crop090",
            "purpose": "Retain six views but enlarge local crops to preserve more context and reduce corner-detail dominance.",
            "crops": CONTROL_CROPS,
            "crop_ratio": 0.90,
            "model_version": "solid_waste_dinov2_proto_round2_view_six_crop090_k3_augseed42",
            "output_dir": "IWaste_rec_model/artifacts/solid_waste_dinov2_proto_round2_view_six_crop090_k3_augseed42",
            "manual_training_required": True,
        },
    ]
    plan = {
        "purpose": "round-two view and crop ablation; preparation only, no model training",
        "comparison_rule": "All experiments use the same K=3 prototype number, seed-42 augmented training manifest, validation/test batches and score calibration. Only crop list or crop_ratio changes.",
        "data_counts": data_counts,
        "control": control,
        "common_arguments": common,
        "experiments": experiments,
        "selection_rule": "Select a view strategy only if it improves or matches independent-test raw accuracy and macro-average material accuracy while reducing coal-gangue-to-steel-slag errors. Rejection rate alone is not sufficient.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("\nRun these commands manually from the project root:")
    for item in experiments:
        if not item["manual_training_required"]:
            continue
        crops = ",".join(item["crops"])
        print(
            "python IWaste_rec_model/scripts/train_round2_baseline.py "
            f"--manifest {common['manifest']} --include-augmented-train --prototypes-per-class 3 "
            f"--crops {crops} --crop-ratio {item['crop_ratio']} "
            f"--model-version {item['model_version']} --output-dir {item['output_dir']}"
        )
    print(f"\nplan_file={args.output}")


if __name__ == "__main__":
    main()
