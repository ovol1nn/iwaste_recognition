"""Generate a controlled augmented training set and manifest for round-two comparison.

Only rows marked known_train in the supplied baseline manifest are augmented.
Validation and test images are copied into the new manifest unchanged and are
never augmented. This script does not load DINOv2 and does not train a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
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


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.92, 1.08)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.01),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
        ]
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Prepare round-two augmented training manifest; does not train a model.")
    parser.add_argument("--input-manifest", type=Path, default=project_root / "IWaste_rec_model" / "data" / "manifests" / "round2_baseline_manifest.csv")
    parser.add_argument("--output-manifest", type=Path, default=project_root / "IWaste_rec_model" / "data" / "manifests" / "round2_augmented_manifest.csv")
    parser.add_argument("--output-jsonl", type=Path, default=project_root / "IWaste_rec_model" / "data" / "manifests" / "round2_augmented_manifest.jsonl")
    parser.add_argument("--summary-json", type=Path, default=project_root / "IWaste_rec_model" / "data" / "manifests" / "round2_augmented_summary.json")
    parser.add_argument("--output-root", type=Path, default=project_root / "IWaste_rec_model" / "data" / "generated" / "round2_augmented_images")
    parser.add_argument("--aug-per-image", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--clean-output", action="store_true", help="Delete only the specified augmentation output directory before generation.")
    args = parser.parse_args()

    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
    if set(labels) != set(DEFAULT_LABELS):
        raise ValueError(f"This comparison is fixed to three classes: {DEFAULT_LABELS}")
    if args.aug_per_image < 1 or args.image_size < 1:
        raise ValueError("aug-per-image and image-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with args.input_manifest.open("r", encoding="utf-8", newline="") as handle:
        original_rows = list(csv.DictReader(handle))
    if not original_rows:
        raise ValueError(f"empty input manifest: {args.input_manifest}")
    if any(row.get("augmentation_source") != "original" for row in original_rows):
        raise ValueError("input manifest must contain original rows only")
    if any(row.get("sample_group") == "" for row in original_rows):
        raise ValueError("every row must have a sample_group")

    train_rows = [row for row in original_rows if row["usage_role"] == "known_train" and row["material_type"] in labels]
    holdout_rows = [row for row in original_rows if row["usage_role"] in {"known_val", "known_test"}]
    if not train_rows or not holdout_rows:
        raise ValueError("manifest must contain known_train, known_val and known_test rows")
    if args.clean_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    augmenter = build_transform(args.image_size)
    augmented_rows: list[dict[str, object]] = []
    skipped = 0
    for index, row in enumerate(train_rows, start=1):
        source = Path(row["image_path"])
        if not is_image_file(source):
            skipped += args.aug_per_image
            continue
        try:
            with Image.open(source) as image:
                source_image = image.convert("RGB")
                for aug_index in range(args.aug_per_image):
                    augmented = augmenter(source_image)
                    output_dir = args.output_root / row["material_type"] / row["sample_group"]
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{source.stem}__aug_{aug_index:02d}.jpg"
                    augmented.save(output_path, quality=95)
                    augmented_row = {field: row[field] for field in MANIFEST_FIELDS}
                    augmented_row["image_path"] = output_path.as_posix()
                    augmented_row["augmentation_source"] = "offline_aug"
                    augmented_rows.append(augmented_row)
        except (OSError, ValueError):
            skipped += args.aug_per_image
        if index % 100 == 0 or index == len(train_rows):
            print(f"[augment] source_train={index}/{len(train_rows)} generated={len(augmented_rows)} skipped={skipped}")

    output_rows = [dict(row) for row in original_rows] + augmented_rows
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter((row["material_type"], row["split"], row["augmentation_source"]) for row in output_rows)
    summary = {
        "purpose": "round-two augmentation comparison; preparation only, no model training",
        "input_manifest": args.input_manifest.as_posix(),
        "output_manifest": args.output_manifest.as_posix(),
        "output_root": args.output_root.as_posix(),
        "labels": list(labels),
        "augmentation_policy": "Only known_train originals receive light crop, horizontal flip, small rotation, color jitter and blur. Validation/test remain original.",
        "aug_per_image": args.aug_per_image,
        "image_size": args.image_size,
        "seed": args.seed,
        "generated_augmented_images": len(augmented_rows),
        "skipped_images": skipped,
        "counts": [
            {"material": material, "split": split, "augmentation_source": source, "count": count}
            for (material, split, source), count in sorted(counts.items())
        ],
        "total_rows": len(output_rows),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
