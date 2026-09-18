"""Summarize completed round-two view/crop ablation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS = ("煤矸石", "矿渣", "钢渣")
DEFAULT_PLAN = PROJECT_ROOT / "IWaste_rec_model" / "data" / "manifests" / "round2_view_crop_sweep_plan.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "IWaste_rec_model" / "data" / "manifests" / "round2_augmented_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "IWaste_rec_model" / "output" / "round2_view_crop_sweep"


def norm_path(value: str) -> str:
    return os.path.normcase(os.path.normpath(value))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize finished view/crop ablations; does not train.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = read_csv(args.manifest)
    test_rows = [row for row in manifest if row["usage_role"] == "known_test"]
    if not test_rows:
        raise ValueError("manifest has no known_test rows")
    test_by_path = {norm_path(row["image_path"]): row for row in test_rows}
    material_totals = Counter(row["material_type"] for row in test_rows)
    batch_totals = Counter((row["material_type"], row["sample_group"]) for row in test_rows)

    overall_rows: list[dict[str, object]] = []
    material_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"test_count": len(test_rows), "experiments": {}}
    for experiment in plan["experiments"]:
        name = experiment["name"]
        version = experiment["model_version"]
        artifact_dir = PROJECT_ROOT / experiment["output_dir"]
        report_path = artifact_dir / f"{version}.report.json"
        errors_path = artifact_dir / f"{version}.test_errors.csv"
        if not report_path.is_file() or not errors_path.is_file():
            raise FileNotFoundError(f"experiment not complete: {name} ({artifact_dir})")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        errors = read_csv(errors_path)
        if int(report["prototypes_per_class_max"]) != 3:
            raise ValueError(f"{name} is not K=3")
        if Path(str(report["manifest"])).name != args.manifest.name:
            raise ValueError(f"{name} uses another manifest: {report['manifest']}")
        if list(report["crop_names"]) != experiment["crops"] or float(report["crop_ratio"]) != float(experiment["crop_ratio"]):
            raise ValueError(f"{name} crop setting does not match its plan")

        wrong = Counter()
        rejected = Counter()
        wrong_rejected = Counter()
        confusion = Counter()
        batch_wrong = Counter()
        batch_rejected = Counter()
        batch_wrong_rejected = Counter()
        for row in errors:
            manifest_row = test_by_path.get(norm_path(row["image_path"]))
            if manifest_row is None:
                raise ValueError(f"test error path absent from manifest: {row['image_path']}")
            label = row["true_label"]
            batch = manifest_row["sample_group"]
            is_correct = row["is_correct"].lower() == "true"
            is_rejected = row["rejected"].lower() == "true"
            if not is_correct:
                wrong[label] += 1
                batch_wrong[(label, batch)] += 1
                confusion[(label, row["predicted_label"])] += 1
            if is_rejected:
                rejected[label] += 1
                batch_rejected[(label, batch)] += 1
            if not is_correct and is_rejected:
                wrong_rejected[label] += 1
                batch_wrong_rejected[(label, batch)] += 1

        metrics = report["metrics"]["test"]
        computed_accuracy = (len(test_rows) - sum(wrong.values())) / len(test_rows)
        if abs(float(metrics["raw_accuracy"]) - computed_accuracy) > 1e-9:
            raise ValueError(f"{name} raw accuracy does not match its test error file")
        overall = {
            "experiment": name,
            "model_version": version,
            "crops": ",".join(experiment["crops"]),
            "crop_ratio": experiment["crop_ratio"],
            "test_count": metrics["count"],
            "raw_accuracy": round(metrics["raw_accuracy"], 6),
            "rejection_rate": round(metrics["rejection_rate"], 6),
            "accepted_correct_rate": round(metrics["accepted_correct_rate"], 6),
        }
        overall_rows.append(overall)
        summary["experiments"][name] = overall
        for label in LABELS:
            total = material_totals[label]
            material_rows.append(
                {
                    "experiment": name,
                    "material": label,
                    "test_count": total,
                    "misclassified_count": wrong[label],
                    "raw_accuracy": round((total - wrong[label]) / total, 6),
                    "rejected_count": rejected[label],
                    "rejection_rate": round(rejected[label] / total, 6),
                    "accepted_correct_rate": round((total - wrong[label] - rejected[label] + wrong_rejected[label]) / total, 6),
                }
            )
        for (true_label, predicted_label), count in sorted(confusion.items()):
            confusion_rows.append({"experiment": name, "true_material": true_label, "predicted_material": predicted_label, "count": count})
        for (label, batch), total in sorted(batch_totals.items()):
            incorrect = batch_wrong[(label, batch)]
            rejected_count = batch_rejected[(label, batch)]
            both = batch_wrong_rejected[(label, batch)]
            batch_rows.append(
                {
                    "experiment": name,
                    "material": label,
                    "batch_id": batch,
                    "test_count": total,
                    "raw_accuracy": round((total - incorrect) / total, 6),
                    "rejection_rate": round(rejected_count / total, 6),
                    "accepted_correct_rate": round((total - incorrect - rejected_count + both) / total, 6),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "overall_summary.csv", overall_rows, list(overall_rows[0]))
    write_csv(args.output_dir / "material_summary.csv", material_rows, list(material_rows[0]))
    write_csv(args.output_dir / "confusion_summary.csv", confusion_rows, ["experiment", "true_material", "predicted_material", "count"])
    write_csv(args.output_dir / "batch_summary.csv", batch_rows, list(batch_rows[0]))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
