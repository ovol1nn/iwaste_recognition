"""Summarize existing round-two material errors without training or inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path


MODELS = {
    "baseline": "solid_waste_dinov2_proto_round2_baseline_grouped_v2",
    "aug_seed42": "solid_waste_dinov2_proto_round2_augmented_aug1",
    "aug_seed2026": "solid_waste_dinov2_proto_round2_augmented_seed2026",
    "aug_seed2027": "solid_waste_dinov2_proto_round2_augmented_seed2027",
}
LABELS = ("煤矸石", "矿渣", "钢渣")


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


def metric(value: str) -> float:
    return float(value)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Analyze existing round-two error CSVs; no model training or inference.")
    parser.add_argument("--manifest", type=Path, default=project_root / "IWaste_rec_model" / "data" / "manifests" / "round2_baseline_manifest.csv")
    parser.add_argument("--artifact-root", type=Path, default=project_root / "IWaste_rec_model" / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=project_root / "IWaste_rec_model" / "output" / "round1_material_error_analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_csv(args.manifest)
    test_rows = [row for row in manifest if row.get("split") == "test"]
    manifest_by_path = {norm_path(row["image_path"]): row for row in test_rows}
    totals = Counter(row["material_type"] for row in test_rows)
    batch_totals = Counter((row["material_type"], row["sample_group"]) for row in test_rows)

    material_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    error_detail_rows: list[dict[str, object]] = []
    overall: dict[str, object] = {"test_count": len(test_rows), "models": {}}

    for model_key, model_name in MODELS.items():
        artifact_dir = args.artifact_root / model_name
        errors = read_csv(artifact_dir / f"{model_name}.test_errors.csv")
        report = json.loads((artifact_dir / f"{model_name}.report.json").read_text(encoding="utf-8"))
        error_by_material = Counter()
        reject_by_material = Counter()
        confusion = Counter()
        batch_error = Counter()
        batch_reject = Counter()
        batch_wrong_rejected = Counter()
        confidence_by_material: defaultdict[str, list[float]] = defaultdict(list)
        similarity_by_material: defaultdict[str, list[float]] = defaultdict(list)
        margin_by_material: defaultdict[str, list[float]] = defaultdict(list)

        for error in errors:
            true_label = error["true_label"]
            manifest_row = manifest_by_path.get(norm_path(error["image_path"]), {})
            batch_id = manifest_row.get("sample_group", "未匹配批次")
            is_correct = error["is_correct"].lower() == "true"
            rejected = error["rejected"].lower() == "true"
            if not is_correct:
                error_by_material[true_label] += 1
                confusion[(true_label, error["predicted_label"])] += 1
                batch_error[(true_label, batch_id)] += 1
            if rejected:
                reject_by_material[true_label] += 1
                batch_reject[(true_label, batch_id)] += 1
            if not is_correct and rejected:
                batch_wrong_rejected[(true_label, batch_id)] += 1
            confidence_by_material[true_label].append(metric(error["confidence"]))
            similarity_by_material[true_label].append(metric(error["similarity"]))
            margin_by_material[true_label].append(metric(error["margin"]))
            error_detail_rows.append(
                {
                    "model": model_key,
                    "true_material": true_label,
                    "predicted_material": error["predicted_label"],
                    "is_correct": error["is_correct"],
                    "rejected": error["rejected"],
                    "confidence": error["confidence"],
                    "similarity": error["similarity"],
                    "margin": error["margin"],
                    "batch_id": batch_id,
                    "scene": error["scene"],
                    "image_path": error["image_path"],
                }
            )

        test_metrics = report["metrics"]["test"]
        model_summary = {
            "test_count": test_metrics["count"],
            "raw_accuracy": test_metrics["raw_accuracy"],
            "rejection_rate": test_metrics["rejection_rate"],
            "accepted_correct_rate": test_metrics["accepted_correct_rate"],
            "error_csv_rows": len(errors),
            "misclassified_count": sum(error_by_material.values()),
            "rejected_count": sum(reject_by_material.values()),
            "note": "misclassification counts are verified against report raw_accuracy; test_errors.csv also contains rejected samples",
        }
        overall["models"][model_key] = model_summary

        for label in LABELS:
            count = totals[label]
            wrong = error_by_material[label]
            values_conf = confidence_by_material[label]
            values_sim = similarity_by_material[label]
            values_margin = margin_by_material[label]
            material_rows.append(
                {
                    "model": model_key,
                    "material": label,
                    "test_count": count,
                    "misclassified_count": wrong,
                    "raw_accuracy": round((count - wrong) / count, 6),
                    "rejected_count": reject_by_material[label],
                    "error_or_reject_rows": wrong + reject_by_material[label] - sum(
                        1 for row in errors if row["true_label"] == label and row["is_correct"].lower() == "false" and row["rejected"].lower() == "true"
                    ),
                    "error_row_confidence_mean": round(statistics.mean(values_conf), 6) if values_conf else "",
                    "error_row_similarity_mean": round(statistics.mean(values_sim), 6) if values_sim else "",
                    "error_row_margin_mean": round(statistics.mean(values_margin), 6) if values_margin else "",
                }
            )

        for (true_label, predicted_label), count in sorted(confusion.items()):
            confusion_rows.append({"model": model_key, "true_material": true_label, "predicted_material": predicted_label, "count": count})

        for (label, batch_id), total in sorted(batch_totals.items()):
            wrong = batch_error[(label, batch_id)]
            rejected = batch_reject[(label, batch_id)]
            wrong_rejected = batch_wrong_rejected[(label, batch_id)]
            batch_rows.append(
                {
                    "model": model_key,
                    "material": label,
                    "batch_id": batch_id,
                    "test_count": total,
                    "misclassified_count": wrong,
                    "raw_accuracy": round((total - wrong) / total, 6),
                    "rejected_count": rejected,
                    "rejection_rate": round(rejected / total, 6),
                    "misclassified_and_rejected_count": wrong_rejected,
                    "accepted_correct_rate": round((total - wrong - rejected + wrong_rejected) / total, 6),
                    "available_metric": "批次内原始准确率、拒识率和正确接受率均由错误明细与测试清单计算",
                }
            )

    fields = list(material_rows[0])
    write_csv(args.output_dir / "material_summary.csv", material_rows, fields)
    write_csv(args.output_dir / "confusion_summary.csv", confusion_rows, list(confusion_rows[0]) if confusion_rows else ["model", "true_material", "predicted_material", "count"])
    write_csv(args.output_dir / "batch_error_summary.csv", batch_rows, list(batch_rows[0]))
    write_csv(args.output_dir / "error_details.csv", error_detail_rows, list(error_detail_rows[0]))
    (args.output_dir / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
