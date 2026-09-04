from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .config import DATASET_ROOT, KNOWN_CLASS_NAMES, OFFLINE_AUG_DIR, SUPPORTED_IMAGE_SUFFIXES
from .manifest import assign_split, build_sample_group, discover_material_dirs, order_source_files


def build_augment_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.08, hue=0.01),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controlled offline augmentation images for train split only.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OFFLINE_AUG_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--known-target-train-count", type=int, default=120)
    parser.add_argument("--reject-target-train-count", type=int, default=48)
    parser.add_argument("--known-max-augs-per-image", type=int, default=4)
    parser.add_argument("--reject-max-augs-per-image", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene", action="append", dest="scenes", help="Scene folder to include; defaults to 实验室 and 料棚")
    parser.add_argument("--clean", action="store_true", help="Delete existing augmentation output before regenerating")
    parser.add_argument("--max-input-pixels", type=int, default=25000000, help="Skip images larger than this many pixels before augmentation")
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def allocate_aug_counts(train_files: list[Path], target_train_count: int, max_augs_per_image: int) -> dict[Path, int]:
    if not train_files:
        return {}
    originals = len(train_files)
    extra_needed = max(0, target_train_count - originals)
    if extra_needed == 0:
        return {path: 0 for path in train_files}

    base = extra_needed // originals
    remainder = extra_needed % originals
    allocations: dict[Path, int] = {}
    for index, path in enumerate(train_files):
        requested = base + (1 if index < remainder else 0)
        allocations[path] = min(max_augs_per_image, requested)
    return allocations


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    augmenter = build_augment_transform(args.image_size)
    generated = 0
    skipped = 0

    for scene_name, material_dir in discover_material_dirs(args.dataset_root, scene_names=args.scenes):
        material_name = material_dir.name
        is_known_class = material_name in KNOWN_CLASS_NAMES
        source_files = order_source_files(path for path in material_dir.iterdir() if is_image_file(path))
        train_files = [
            path
            for index, path in enumerate(source_files)
            if assign_split(index=index, total=len(source_files), is_known_class=is_known_class) == "train"
        ]
        target_train_count = args.known_target_train_count if is_known_class else args.reject_target_train_count
        max_augs_per_image = args.known_max_augs_per_image if is_known_class else args.reject_max_augs_per_image
        allocations = allocate_aug_counts(
            train_files=train_files,
            target_train_count=target_train_count,
            max_augs_per_image=max_augs_per_image,
        )
        print(
            f"[augment] scene={scene_name} material={material_name} originals_train={len(train_files)} "
            f"target_train={target_train_count} planned_aug={sum(allocations.values())}",
            flush=True,
        )

        for index, image_path in enumerate(source_files):
            split = assign_split(index=index, total=len(source_files), is_known_class=is_known_class)
            if split != "train":
                continue

            aug_count = allocations.get(image_path, 0)
            if aug_count == 0:
                continue

            sample_group = build_sample_group(scene_name, material_name, image_path)
            output_dir = args.output_root / scene_name / material_name / sample_group
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                with Image.open(image_path) as source_image:
                    width, height = source_image.size
                    if width * height > args.max_input_pixels:
                        skipped += 1
                        print(f"[skip] too_large path={image_path} size={width}x{height}", flush=True)
                        continue
                    image = source_image.convert("RGB")
            except OSError as exc:
                skipped += 1
                print(f"[skip] cannot_open path={image_path} error={exc}", flush=True)
                continue

            for aug_index in range(aug_count):
                augmented = augmenter(image)
                output_path = output_dir / f"aug_{aug_index:02d}{image_path.suffix.lower()}"
                augmented.save(output_path, quality=95)
                generated += 1
                if generated % 50 == 0:
                    print(f"[progress] generated={generated} skipped={skipped}", flush=True)

    print(f"generated_offline_aug_images={generated}", flush=True)
    print(f"skipped_images={skipped}", flush=True)
    print(args.output_root, flush=True)


if __name__ == "__main__":
    main()