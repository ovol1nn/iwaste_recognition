"""Validate and prepare the manual K=1/3/5 prototype comparison plan.

This script never starts training.  It fixes the data and inference settings used
for the second-round prototype-number experiment and prints the two manual
training commands still required.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "IWaste_rec_model" / "data" / "manifests" / "round2_augmented_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "IWaste_rec_model" / "data" / "manifests" / "round2_prototype_sweep_plan.json"
K3_ARTIFACT = PROJECT_ROOT / "IWaste_rec_model" / "artifacts" / "solid_waste_dinov2_proto_round2_augmented_aug1" / "solid_waste_dinov2_proto_round2_augmented_aug1.meta.json"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_manifest(rows: list[dict[str, str]]) -> dict[str, int]:
    if not rows:
        raise ValueError("augmentation manifest is empty")
    required = {"image_path", "material_type", "split", "augmentation_source", "usage_role", "sample_group"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"manifest misses required columns: {sorted(missing)}")
    if any(not row["sample_group"] for row in rows):
        raise ValueError("every row must retain a capture-batch identifier")
    if any(row["augmentation_source"] != "original" for row in rows if row["split"] in {"val", "test"}):
        raise ValueError("validation and test sets must not contain augmented images")
    counts = Counter(row["usage_role"] for row in rows)
    if not {"known_train", "known_val", "known_test"} <= set(counts):
        raise ValueError("manifest must contain known_train, known_val and known_test rows")
    return {
        "train_rows": counts["known_train"],
        "validation_rows": counts["known_val"],
        "test_rows": counts["known_test"],
        "offline_augmented_train_rows": sum(
            row["usage_role"] == "known_train" and row["augmentation_source"] == "offline_aug" for row in rows
        ),
        "original_train_rows": sum(
            row["usage_role"] == "known_train" and row["augmentation_source"] == "original" for row in rows
        ),
    }


def validate_existing_k3(manifest: Path) -> dict[str, object]:
    if not K3_ARTIFACT.exists():
        raise FileNotFoundError(f"existing K=3 artifact is missing: {K3_ARTIFACT}")
    meta = json.loads(K3_ARTIFACT.read_text(encoding="utf-8"))
    if int(meta["prototypes_per_class_max"]) != 3:
        raise ValueError("existing control artifact is not K=3")
    if Path(str(meta["manifest"])).name != manifest.name:
        raise ValueError("existing K=3 control does not use the selected augmented manifest")
    metrics = meta["metrics"]
    return {
        "artifact_dir": str(K3_ARTIFACT.parent),
        "model_version": meta["model_version"],
        "prototypes_per_class": 3,
        "train_count": metrics["train"]["count"],
        "validation_count": metrics["validation"]["count"],
        "test_count": metrics["test"]["count"],
        "test_raw_accuracy": metrics["test"]["raw_accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the manual K=1/3/5 prototype comparison; does not train.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    data_counts = validate_manifest(rows)
    control = validate_existing_k3(args.manifest)
    common_args = {
        "manifest": str(args.manifest.relative_to(PROJECT_ROOT).as_posix()),
        "include_augmented_train": True,
        "labels": ["煤矸石", "矿渣", "钢渣"],
        "image_size": 448,
        "batch_size": 8,
        "crops": ["full", "center", "top_left", "top_right", "bottom_left", "bottom_right"],
        "crop_ratio": 0.78,
        "class_temperature": 0.07,
        "vote_temperature": 0.03,
        "rejection_quantile": 0.10,
        "kmeans_iterations": 50,
        "device": "cuda",
    }
    experiments = [
        {
            "name": "K=1",
            "prototypes_per_class": 1,
            "model_version": "solid_waste_dinov2_proto_round2_proto_k1_augseed42",
            "output_dir": "IWaste_rec_model/artifacts/solid_waste_dinov2_proto_round2_proto_k1_augseed42",
            "manual_training_required": True,
        },
        {
            "name": "K=3",
            "prototypes_per_class": 3,
            "model_version": control["model_version"],
            "output_dir": str(Path(str(control["artifact_dir"])).relative_to(PROJECT_ROOT).as_posix()),
            "manual_training_required": False,
            "reused_control": True,
        },
        {
            "name": "K=5",
            "prototypes_per_class": 5,
            "model_version": "solid_waste_dinov2_proto_round2_proto_k5_augseed42",
            "output_dir": "IWaste_rec_model/artifacts/solid_waste_dinov2_proto_round2_proto_k5_augseed42",
            "manual_training_required": True,
        },
    ]
    plan = {
        "purpose": "round-two prototype-number comparison; preparation only, no model training",
        "comparison_rule": "K=1, K=3 and K=5 use the same seed-42 augmented training manifest and unchanged validation/test batches; only prototypes_per_class changes.",
        "data_counts": data_counts,
        "reused_k3_control": control,
        "common_arguments": common_args,
        "experiments": experiments,
        "selection_rule": "Prefer the model with higher independent-test raw accuracy and macro-average material accuracy. Coal-gangue-to-steel-slag errors must not increase; compare rejection rate and correct-accept rate as secondary metrics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("\nRun these commands manually from the project root:")
    for experiment in experiments:
        if not experiment["manual_training_required"]:
            continue
        print(
            "python IWaste_rec_model/scripts/train_round2_baseline.py "
            f"--manifest {common_args['manifest']} --include-augmented-train "
            f"--prototypes-per-class {experiment['prototypes_per_class']} "
            f"--model-version {experiment['model_version']} --output-dir {experiment['output_dir']}"
        )
    print(f"\nplan_file={args.output}")


if __name__ == "__main__":
    main()
