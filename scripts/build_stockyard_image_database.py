"""Build a reproducible metadata table for all stockyard images.

The source images are never modified.  Image capture time is parsed from the
timestamp embedded in the file name.  The displayed time overlay is retained
as a separate future verification field because this workstation currently has
no local OCR engine configured.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MATERIALS = ("钢渣", "矿渣", "煤矸石")
PHASES = ("第一阶段", "第二阶段")


@dataclass
class ImageRecord:
    image_id: str
    relative_path: str
    file_name: str
    material: str
    phase: str
    camera_id: str
    file_timestamp: str
    capture_date: str
    capture_time: str
    day_night: str
    focus_group: str
    capture_folder: str
    folder_declared_datetime: str
    batch_id: str
    batch_gap_minutes: str
    file_size_bytes: int
    parse_status: str
    notes: str
    _timestamp: datetime

    def as_row(self) -> dict[str, str | int]:
        row = self.__dict__.copy()
        row.pop("_timestamp")
        return row


def parse_file_timestamp(file_name: str) -> datetime | None:
    match = re.search(r"(?<!\d)(20\d{12})(?:\d{1,6})?(?!\d)", Path(file_name).stem)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")


def parse_focus(folder_name: str) -> str:
    if "大焦距" in folder_name or "-55P" in folder_name:
        return "大焦距"
    if "中等焦距" in folder_name or "中焦距" in folder_name or "-51P" in folder_name:
        return "中焦距"
    if "最小焦距" in folder_name or "小焦距" in folder_name or "-50P" in folder_name:
        return "小焦距"
    return "未标注"


def parse_folder_datetime(folder_name: str) -> str:
    normalized = folder_name.replace("：", ":")
    date_match = re.search(r"(?<!\d)(20\d{2})[-年]?(\d{2})[-月]?(\d{2})(?!\d)", normalized)
    time_match = re.search(r"(?<!\d)(\d{1,2})\s*:\s*(\d{2})(?!\d)", normalized)
    if not date_match:
        return ""
    date_text = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
    if not time_match:
        return date_text
    return f"{date_text} {int(time_match.group(1)):02d}:{int(time_match.group(2)):02d}:00"


def camera_id_from_name(file_name: str) -> str:
    prefix = Path(file_name).stem.split("_", 1)[0]
    return prefix if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", prefix) else "未识别摄像头"


def find_capture_folder(image_path: Path, phase_root: Path) -> Path | None:
    for parent in image_path.parents:
        if parent == phase_root:
            return None
        if parse_focus(parent.name) != "未标注":
            return parent
    return None


def iter_images(root: Path) -> Iterable[tuple[str, str, Path, Path]]:
    for material in MATERIALS:
        material_root = root / material
        if not material_root.is_dir():
            continue
        for phase in PHASES:
            phase_root = material_root / phase
            if not phase_root.is_dir():
                continue
            for image_path in sorted(phase_root.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    yield material, phase, phase_root, image_path


def build_records(root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for material, phase, phase_root, image_path in iter_images(root):
        timestamp = parse_file_timestamp(image_path.name)
        capture_folder = find_capture_folder(image_path, phase_root) if phase == "第二阶段" else None
        focus_group = parse_focus(capture_folder.name) if capture_folder else "未标注"
        folder_datetime = parse_folder_datetime(capture_folder.name) if capture_folder else ""

        if timestamp is None:
            file_timestamp = ""
            capture_date = ""
            capture_time = ""
            day_night = ""
            parse_status = "文件名未识别时间"
            timestamp_for_sort = datetime.min
        else:
            file_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            capture_date = timestamp.strftime("%Y-%m-%d")
            capture_time = timestamp.strftime("%H:%M:%S")
            day_night = "晚上" if timestamp.hour >= 19 else "白天"
            parse_status = "文件名时间已解析"
            timestamp_for_sort = timestamp

        records.append(
            ImageRecord(
                image_id=f"stockyard_{len(records) + 1:05d}",
                relative_path=image_path.relative_to(root).as_posix(),
                file_name=image_path.name,
                material=material,
                phase=phase,
                camera_id=camera_id_from_name(image_path.name),
                file_timestamp=file_timestamp,
                capture_date=capture_date,
                capture_time=capture_time,
                day_night=day_night,
                focus_group=focus_group,
                capture_folder=capture_folder.name if capture_folder else "",
                folder_declared_datetime=folder_datetime,
                batch_id="",
                batch_gap_minutes="",
                file_size_bytes=image_path.stat().st_size,
                parse_status=parse_status,
                notes="",
                _timestamp=timestamp_for_sort,
            )
        )
    return records


def assign_batches(records: list[ImageRecord], gap_minutes: int) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, str, str], list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.material, record.phase, record.camera_id)].append(record)

    batches: list[dict[str, str | int | float]] = []
    for key in sorted(grouped):
        material, phase, camera_id = key
        source_records = sorted(grouped[key], key=lambda record: record._timestamp)
        current: list[ImageRecord] = []
        previous: ImageRecord | None = None
        batch_index = 0

        def finish_batch(items: list[ImageRecord]) -> None:
            nonlocal batch_index
            if not items:
                return
            batch_index += 1
            start = items[0]._timestamp
            end = items[-1]._timestamp
            batch_id = f"{phase[:2]}_{material}_{camera_id}_{start:%Y%m%d_%H%M%S}_{batch_index:02d}"
            for item in items:
                item.batch_id = batch_id
            batches.append(
                {
                    "batch_id": batch_id,
                    "material": material,
                    "phase": phase,
                    "camera_id": camera_id,
                    "batch_start": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "batch_end": end.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_minutes": round((end - start).total_seconds() / 60, 1),
                    "image_count": len(items),
                    "day_night_values": "、".join(sorted({item.day_night for item in items if item.day_night})),
                    "focus_groups": "、".join(sorted({item.focus_group for item in items if item.focus_group != "未标注"})) or "未标注",
                    "capture_folders": " | ".join(sorted({item.capture_folder for item in items if item.capture_folder})),
                    "rule": f"同摄像头、同物料、同阶段，相邻图片间隔超过{gap_minutes}分钟即新批次",
                }
            )

        for record in source_records:
            if previous is None:
                current = [record]
            else:
                gap = round((record._timestamp - previous._timestamp).total_seconds() / 60, 3)
                record.batch_gap_minutes = f"{gap:.3f}"
                date_changed = record._timestamp.date() != previous._timestamp.date()
                if date_changed or gap > gap_minutes:
                    finish_batch(current)
                    current = [record]
                else:
                    current.append(record)
            previous = record
        finish_batch(current)
    return batches


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(records: list[ImageRecord], batches: list[dict[str, object]], gap_minutes: int) -> dict[str, object]:
    material_phase = Counter((record.material, record.phase) for record in records)
    return {
        "source_root": "ml_data/raw/料棚",
        "total_images": len(records),
        "parsed_filename_timestamps": sum(record.parse_status == "文件名时间已解析" for record in records),
        "batch_gap_minutes": gap_minutes,
        "batch_count": len(batches),
        "time_source": "文件名中的采集时间",
        "counts_by_material_phase": [
            {"material": material, "phase": phase, "image_count": count}
            for (material, phase), count in sorted(material_phase.items())
        ],
        "batch_counts_by_material": [
            {"material": material, "batch_count": sum(batch["material"] == material for batch in batches)}
            for material in MATERIALS
        ],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build stockyard image metadata and capture batches.")
    parser.add_argument("--root", type=Path, default=project_root / "ml_data" / "raw" / "料棚")
    parser.add_argument("--output-dir", type=Path, default=project_root / "ml_data" / "manifests")
    parser.add_argument("--batch-gap-minutes", type=int, default=30)
    args = parser.parse_args()

    if args.batch_gap_minutes <= 0:
        raise ValueError("batch-gap-minutes must be positive")
    if not args.root.is_dir():
        raise FileNotFoundError(f"stockyard root not found: {args.root}")

    records = build_records(args.root)
    batches = assign_batches(records, args.batch_gap_minutes)
    image_rows = [record.as_row() for record in records]
    summary = build_summary(records, batches, args.batch_gap_minutes)

    image_csv = args.output_dir / "stockyard_image_database.csv"
    batch_csv = args.output_dir / "stockyard_capture_batches.csv"
    image_json = args.output_dir / "stockyard_image_database.json"
    batch_json = args.output_dir / "stockyard_capture_batches.json"
    summary_json = args.output_dir / "stockyard_image_database_summary.json"
    write_csv(image_csv, image_rows)
    write_csv(batch_csv, batches)
    image_json.write_text(json.dumps(image_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    batch_json.write_text(json.dumps(batches, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"image_csv={image_csv}")
    print(f"batch_csv={batch_csv}")


if __name__ == "__main__":
    main()
