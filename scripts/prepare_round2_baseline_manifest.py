"""Create a batch-independent, three-class manifest for round-two baseline training.

This script only prepares the train/validation/test manifest. It never loads DINOv2
or starts model training.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_LABELS = ("煤矸石", "矿渣", "钢渣")
MANIFEST_FIELDS = (
    "image_path",
    "material_type",
    "particle_size",
    "shape",
    "color",
    "scene",
    "split",
    "is_known_class",
    "sample_source",
    "sample_group",
    "augmentation_source",
    "usage_role",
)


def parse_labels(value: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(labels) < 2:
        raise ValueError("labels must contain at least two material names")
    return labels


def assign_batch_splits(
    rows: list[dict[str, object]],
    labels: tuple[str, ...],
    validation_batches_per_class: int,
    test_batches_per_class: int,
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["material"])].append(row)

    split_by_batch: dict[str, str] = {}
    holdout_count = validation_batches_per_class + test_batches_per_class
    for label in labels:
        batches = sorted(
            {str(row["batch_id"]): str(row["file_timestamp"]) for row in grouped[label]}.items(),
            key=lambda item: (item[1], item[0]),
        )
        if len(batches) <= holdout_count:
            raise ValueError(
                f"{label} has {len(batches)} batches, insufficient for {validation_batches_per_class} validation and "
                f"{test_batches_per_class} test batches while retaining training batches"
            )
        test_ids = {batch_id for batch_id, _ in batches[-test_batches_per_class:]}
        val_start = len(batches) - holdout_count
        val_ids = {batch_id for batch_id, _ in batches[val_start : len(batches) - test_batches_per_class]}
        for batch_id, _ in batches:
            split_by_batch[batch_id] = "test" if batch_id in test_ids else "val" if batch_id in val_ids else "train"
    return split_by_batch


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Prepare the round-two three-class batch-independent manifest; does not train a model.")
    parser.add_argument("--metadata", type=Path, default=project_root / "ml_data" / "manifests" / "stockyard_image_database.json")
    parser.add_argument("--image-root", type=Path, default=project_root / "ml_data" / "raw" / "料棚")
    parser.add_argument("--output-csv", type=Path, default=project_root / "IWaste_rec_model" / "data" / "manifests" / "round2_baseline_manifest.csv")
    parser.add_argument("--output-jsonl", type=Path, default=project_root / "IWaste_rec_model" / "data" / "manifests" / "round2_baseline_manifest.jsonl")
    parser.add_argument("--summary-json", type=Path, default=project_root / "IWaste_rec_model" / "data" / "manifests" / "round2_baseline_split_summary.json")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--validation-batches-per-class", type=int, default=1)
    parser.add_argument("--test-batches-per-class", type=int, default=2)
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    if args.validation_batches_per_class < 1 or args.test_batches_per_class < 1:
        raise ValueError("validation-batches-per-class and test-batches-per-class must both be positive")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    selected = [row for row in metadata if row["material"] in labels and row["parse_status"] == "文件名时间已解析"]
    counts = Counter(str(row["material"]) for row in selected)
    missing = set(labels) - set(counts)
    if missing:
        raise ValueError(f"metadata has no usable images for: {sorted(missing)}")

    split_by_batch = assign_batch_splits(
        selected,
        labels,
        args.validation_batches_per_class,
        args.test_batches_per_class,
    )
    records: list[dict[str, object]] = []
    for row in sorted(selected, key=lambda item: (str(item["material"]), str(item["file_timestamp"]), str(item["relative_path"]))):
        image_path = args.image_root / Path(str(row["relative_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"image listed in metadata does not exist: {image_path}")
        split = split_by_batch[str(row["batch_id"])]
        records.append(
            {
                "image_path": image_path.as_posix(),
                "material_type": row["material"],
                "particle_size": "unknown",
                "shape": "unknown",
                "color": "unknown",
                "scene": f"料棚_{row['phase']}",
                "split": split,
                "is_known_class": True,
                "sample_source": "stockyard_phase1_phase2",
                "sample_group": row["batch_id"],
                "augmentation_source": "original",
                "usage_role": f"known_{split}",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    split_counts = Counter((str(row["material_type"]), str(row["split"])) for row in records)
    batch_counts = Counter((str(row["material_type"]), str(row["split"]), str(row["sample_group"])) for row in records)
    summary = {
        "purpose": "round-two unified three-class baseline; metadata preparation only, no model training",
        "metadata": args.metadata.as_posix(),
        "image_root": args.image_root.as_posix(),
        "labels": list(labels),
        "split_policy": "Within each material, chronological capture batches: latest batches for test, preceding batches for validation, all earlier batches for training.",
        "validation_batches_per_class": args.validation_batches_per_class,
        "test_batches_per_class": args.test_batches_per_class,
        "image_counts": [
            {"material": label, "split": split, "image_count": split_counts[(label, split)]}
            for label in labels
            for split in ("train", "val", "test")
        ],
        "batch_counts": [
            {"material": label, "split": split, "batch_count": len({group for material, role, group in batch_counts if material == label and role == split})}
            for label in labels
            for split in ("train", "val", "test")
        ],
        "total_images": len(records),
        "total_batches": len(split_by_batch),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"manifest_csv={args.output_csv}")
    print(f"manifest_jsonl={args.output_jsonl}")
    print(f"summary_json={args.summary_json}")


if __name__ == "__main__":
    main()
