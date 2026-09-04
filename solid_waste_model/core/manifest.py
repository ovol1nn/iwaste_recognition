from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .config import (
    DATASET_ROOT,
    DEFAULT_MANIFEST_CSV,
    DEFAULT_MANIFEST_JSONL,
    DEFAULT_PROFILE_PATH,
    DEFAULT_BAD_IMAGES_PATH,
    DEFAULT_SCENE_NAMES,
    DIRECTORY_TO_DISPLAY_NAME,
    KNOWN_CLASS_NAMES,
    OFFLINE_AUG_DIR,
    SUPPORTED_IMAGE_SUFFIXES,
)


@dataclass
class ManifestRecord:
    image_path: str
    material_type: str
    particle_size: str
    shape: str
    color: str
    scene: str
    split: str
    is_known_class: bool
    sample_source: str
    sample_group: str
    augmentation_source: str
    usage_role: str



def load_bad_image_patterns(path: Path | None = None) -> list[str]:
    bad_path = path or DEFAULT_BAD_IMAGES_PATH
    if not bad_path.exists():
        return []
    patterns: list[str] = []
    for line in bad_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped.replace("\\", "/"))
    return patterns


def is_bad_image(path: Path, patterns: Iterable[str]) -> bool:
    normalized = path.as_posix()
    return any(pattern in normalized for pattern in patterns)

def load_material_profiles(profile_path: Path | None = None) -> dict[str, dict[str, str]]:
    path = profile_path or DEFAULT_PROFILE_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _usage_role(is_known_class: bool, split: str) -> str:
    prefix = "known" if is_known_class else "reject"
    return f"{prefix}_{split}"


def stable_order_key(path: Path) -> tuple[str, str]:
    digest = hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()
    return digest, path.name.lower()


def order_source_files(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=stable_order_key)


def compute_split_counts(total: int, val_ratio: float = 0.15, test_ratio: float = 0.15) -> tuple[int, int]:
    if total <= 2:
        return 0, 0
    if total <= 4:
        return 1, 1

    val_count = max(2, round(total * val_ratio))
    test_count = max(2, round(total * test_ratio))

    # Always keep at least one original image in the train split.
    max_holdout = total - 1
    while val_count + test_count > max_holdout:
        if test_count > 2:
            test_count -= 1
        elif val_count > 2:
            val_count -= 1
        else:
            break
    return val_count, test_count


def assign_split(index: int, total: int, is_known_class: bool) -> str:
    del is_known_class  # current split policy is ratio-based for both known and reject classes
    val_count, test_count = compute_split_counts(total)
    if index < val_count:
        return "val"
    if index < val_count + test_count:
        return "test"
    return "train"


def _profile_for(material_name: str, profiles: dict[str, dict[str, str]]) -> dict[str, str]:
    if material_name in profiles:
        return profiles[material_name]
    return profiles["其他"]


def _scene_names(scene_names: Iterable[str] | None) -> tuple[str, ...]:
    if scene_names is None:
        return tuple(DEFAULT_SCENE_NAMES)
    return tuple(scene_names)


def discover_material_dirs(dataset_root: Path, scene_names: Iterable[str] | None = None) -> list[tuple[str, Path]]:
    root = Path(dataset_root)
    allowed_directories = set(DIRECTORY_TO_DISPLAY_NAME)
    configured_scenes = _scene_names(scene_names)
    scene_dirs = [root / scene_name for scene_name in configured_scenes if (root / scene_name).is_dir()]

    discovered: list[tuple[str, Path]] = []
    if scene_dirs:
        for scene_dir in scene_dirs:
            for material_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
                if material_dir.name in allowed_directories:
                    discovered.append((scene_dir.name, material_dir))
        return discovered

    flat_scene = root.name if root.name in configured_scenes else "default"
    for material_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if material_dir.name in allowed_directories:
            discovered.append((flat_scene, material_dir))
    return discovered


def build_sample_group(scene_name: str, material_name: str, image_path: Path) -> str:
    digest = hashlib.sha1(image_path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{scene_name}__{material_name}__{digest}"


def build_manifest_records(
    dataset_root: Path | None = None,
    profile_path: Path | None = None,
    offline_aug_root: Path | None = None,
    scene_names: Iterable[str] | None = None,
) -> list[ManifestRecord]:
    root = Path(dataset_root or DATASET_ROOT)
    aug_root = Path(offline_aug_root or OFFLINE_AUG_DIR)
    profiles = load_material_profiles(profile_path)
    bad_patterns = load_bad_image_patterns()
    records: list[ManifestRecord] = []
    skipped_bad_images = 0

    for scene_name, material_dir in discover_material_dirs(root, scene_names=scene_names):
        material_name = DIRECTORY_TO_DISPLAY_NAME[material_dir.name]
        is_known_class = material_name in KNOWN_CLASS_NAMES
        profile = _profile_for(material_name, profiles)
        original_files = order_source_files(path for path in material_dir.iterdir() if _is_image_file(path))
        filtered_files = []
        for image_path in original_files:
            if is_bad_image(image_path, bad_patterns):
                skipped_bad_images += 1
                print(f"[manifest-skip] bad_image path={image_path}")
                continue
            filtered_files.append(image_path)
        original_files = filtered_files

        for index, image_path in enumerate(original_files):
            split = assign_split(index=index, total=len(original_files), is_known_class=is_known_class)
            usage_role = _usage_role(is_known_class, split)
            sample_group = build_sample_group(scene_name, material_name, image_path)

            records.append(
                ManifestRecord(
                    image_path=image_path.as_posix(),
                    material_type=material_name,
                    particle_size=profile["particle_size"],
                    shape=profile["shape"],
                    color=profile["color"],
                    scene=scene_name,
                    split=split,
                    is_known_class=is_known_class,
                    sample_source="ml_data_raw",
                    sample_group=sample_group,
                    augmentation_source="original",
                    usage_role=usage_role,
                )
            )

            if split != "train":
                continue

            aug_dir = aug_root / scene_name / material_name / sample_group
            if not aug_dir.exists():
                continue

            for aug_file in sorted(path for path in aug_dir.iterdir() if _is_image_file(path)):
                if is_bad_image(aug_file, bad_patterns):
                    skipped_bad_images += 1
                    print(f"[manifest-skip] bad_image path={aug_file}")
                    continue
                records.append(
                    ManifestRecord(
                        image_path=aug_file.as_posix(),
                        material_type=material_name,
                        particle_size=profile["particle_size"],
                        shape=profile["shape"],
                        color=profile["color"],
                        scene=scene_name,
                        split="train",
                        is_known_class=is_known_class,
                        sample_source="ml_data_raw",
                        sample_group=sample_group,
                        augmentation_source="offline_aug",
                        usage_role=_usage_role(is_known_class, "train"),
                    )
                )

    if skipped_bad_images:
        print(f"[manifest] skipped_bad_images={skipped_bad_images}")
    return records


def write_manifest(records: Iterable[ManifestRecord], csv_path: Path, jsonl_path: Path) -> None:
    rows = [asdict(record) for record in records]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ManifestRecord.__annotations__.keys())

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_manifest_summary(records: Iterable[ManifestRecord]) -> None:
    summary: dict[tuple[str, str, str, str], int] = {}
    for record in records:
        key = (record.scene, record.material_type, record.usage_role, record.augmentation_source)
        summary[key] = summary.get(key, 0) + 1
    for scene, material_type, usage_role, augmentation_source in sorted(summary):
        print(
            f"{scene} | {material_type} | usage_role={usage_role:<12} | "
            f"aug={augmentation_source:<11} | count={summary[(scene, material_type, usage_role, augmentation_source)]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the solid-waste image manifest from scene/material folders.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--offline-aug-root", type=Path, default=OFFLINE_AUG_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_MANIFEST_JSONL)
    parser.add_argument("--scene", action="append", dest="scenes", help="Scene folder to include; defaults to 实验室 and 料棚")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_manifest_records(
        dataset_root=args.dataset_root,
        profile_path=args.profile,
        offline_aug_root=args.offline_aug_root,
        scene_names=args.scenes,
    )
    write_manifest(records, args.output_csv, args.output_jsonl)
    print(f"wrote {len(records)} records")
    print_manifest_summary(records)
    print(args.output_csv)
    print(args.output_jsonl)


if __name__ == "__main__":
    main()