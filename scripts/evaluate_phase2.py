"""Evaluate the DINOv2 prototype model on the second-phase 料棚 images.

This script deliberately reuses the same WasteClassifier path as the GUI.  It
only adds directory-label parsing, batch iteration, and machine-readable
summary output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from solid_waste_model.config import SUPPORTED_IMAGE_SUFFIXES  # noqa: E402
from solid_waste_model.gui import (  # noqa: E402
    CACHE_DIR,
    LOCAL_REPOSITORY,
    META_PATH,
    PROFILE_PATH,
    PROTOTYPE_PATH,
)
from solid_waste_model.dinov2_inference import WasteClassifier  # noqa: E402


DEFAULT_DATES = ("2026-08-29", "2026-08-30", "2026-09-03", "2026-09-07", "2026-09-08")
KNOWN_LABELS = ("煤矸石", "矿渣", "钢渣", "石膏")


def parse_folder_labels(date_name: str, folder_name: str) -> tuple[str, str]:
    """Return (true material, focal-length group) from a capture folder name."""

    material = "未识别物料"
    for label in ("煤矸石", "矿渣", "钢渣"):
        if label in folder_name:
            material = label
            break
    if material == "未识别物料" and re.search(r"种类\s*gz\b", folder_name, re.IGNORECASE):
        material = "钢渣"

    if "大焦距" in folder_name or "-55P" in folder_name:
        focus = "大焦距"
    elif "中等焦距" in folder_name or "中焦距" in folder_name or "-51P" in folder_name:
        focus = "中焦距"
    elif "最小焦距" in folder_name or "小焦距" in folder_name or "-50P" in folder_name:
        focus = "小焦距"
    else:
        focus = "未识别焦段"
    return material, focus


def extract_capture_time(text: str) -> str | None:
    """Extract HH:MM from a folder name or timestamped image filename."""

    normalized = text.replace("：", ":").replace("；", ":")
    match = re.search(r"(?<!\d)(\d{1,2})\s*:\s*(\d{2})(?!\d)", normalized)
    if match:
        return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"

    timestamp = re.search(r"(?<!\d)(20\d{12,})(?!\d)", text)
    if timestamp:
        digits = timestamp.group(1)
        if len(digits) >= 12:
            return f"{digits[8:10]}:{digits[10:12]}"
    return None


def extract_capture_date(text: str) -> str | None:
    """Extract YYYY-MM-DD from a capture folder name or image filename."""

    match = re.search(r"(?<!\d)(20\d{2})[-年]?(\d{2})[-月]?(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    compact = re.search(r"(?<!\d)(20\d{6})(?!\d)", text)
    if compact:
        digits = compact.group(1)
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return None


def day_night_for_time(capture_time: str | None) -> str:
    if capture_time is None:
        return "时间未知"
    hour, minute = (int(part) for part in capture_time.split(":"))
    return "晚上" if hour * 60 + minute >= 19 * 60 else "白天"


def collect_records(data_root: Path, dates: tuple[str, ...]) -> list[dict[str, str]]:
    # New layout: 料棚/<物料>/第二阶段/<白天|夜间>/<采集文件夹>/<图片>.
    # Keep the old date-at-root layout below for backward compatibility.
    material_dirs = [data_root / label for label in ("钢渣", "矿渣", "煤矸石")]
    if any((material_dir / "第二阶段").is_dir() for material_dir in material_dirs):
        records: list[dict[str, str]] = []
        for material_dir in material_dirs:
            stage_dir = material_dir / "第二阶段"
            if not stage_dir.is_dir():
                continue
            for day_night_dir in sorted(item for item in stage_dir.iterdir() if item.is_dir()):
                day_night = "晚上" if day_night_dir.name in {"夜间", "晚上"} else day_night_dir.name
                for capture_dir in sorted(item for item in day_night_dir.iterdir() if item.is_dir()):
                    folder_time = extract_capture_time(capture_dir.name)
                    folder_date = extract_capture_date(capture_dir.name)
                    for image_path in sorted(
                        item
                        for item in capture_dir.rglob("*")
                        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                    ):
                        capture_time = folder_time or extract_capture_time(image_path.name)
                        records.append(
                            {
                                "image_path": str(image_path),
                                "date": folder_date or extract_capture_date(image_path.name) or "",
                                "capture_folder": capture_dir.name,
                                "true_material": material_dir.name,
                                "focus_group": parse_folder_labels("", capture_dir.name)[1],
                                "capture_time": capture_time or "",
                                "day_night": day_night if day_night in {"白天", "晚上"} else day_night_for_time(capture_time),
                            }
                        )
        return records

    records: list[dict[str, str]] = []
    for date_name in dates:
        date_dir = data_root / date_name
        if not date_dir.is_dir():
            raise FileNotFoundError(f"日期目录不存在：{date_dir}")
        for capture_dir in sorted(item for item in date_dir.iterdir() if item.is_dir()):
            material, focus = parse_folder_labels(date_name, capture_dir.name)
            folder_time = extract_capture_time(capture_dir.name)
            for image_path in sorted(
                item
                for item in capture_dir.rglob("*")
                if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            ):
                records.append(
                    {
                        "image_path": str(image_path),
                        "date": date_name,
                        "capture_folder": capture_dir.name,
                        "true_material": material,
                        "focus_group": focus,
                        "capture_time": folder_time or extract_capture_time(image_path.name) or "",
                        "day_night": day_night_for_time(folder_time or extract_capture_time(image_path.name)),
                    }
                )
    return records


def predict_record(classifier: WasteClassifier, record: dict[str, str]) -> dict[str, Any]:
    payload = classifier.safe_predict_payload(record["image_path"])
    details = payload.get("model_result") or {}
    scores = details.get("material_scores") or {}
    raw_prediction = max(scores, key=scores.get) if scores else None
    final_prediction = payload.get("waste_type")
    rejected = bool(details.get("rejected", False))
    rejection_reasons = []
    if details.get("probability_rejected"):
        rejection_reasons.append("置信度低于阈值")
    if details.get("similarity_rejected"):
        rejection_reasons.append("相似度低于阈值")
    if details.get("margin_rejected"):
        rejection_reasons.append("类别间隔低于阈值")

    row: dict[str, Any] = {
        **record,
        "raw_prediction": raw_prediction,
        "final_prediction": final_prediction,
        "confidence": float(payload.get("confidence") or 0.0),
        "rejected": rejected,
        "rejection_reason": "；".join(rejection_reasons),
        "similarity_score": float(details.get("similarity_score") or 0.0),
        "margin": float(details.get("margin") or 0.0),
        "threshold_confidence": float(details.get("threshold") or 0.0),
        "threshold_similarity": float(details.get("similarity_threshold") or 0.0),
        "threshold_margin": float(details.get("margin_threshold") or 0.0),
        "raw_correct": raw_prediction == record["true_material"],
        "final_correct": final_prediction == record["true_material"],
        "correct_but_rejected": raw_prediction == record["true_material"] and rejected,
        "model_error": payload.get("model_error"),
        "model_version": details.get("model_version"),
    }
    for label in KNOWN_LABELS:
        row[f"score_{label}"] = float(scores.get(label) or 0.0)
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["true_material"], row["focus_group"])].append(row)

    summary: list[dict[str, Any]] = []
    for (material, focus), group in sorted(grouped.items()):
        total = len(group)
        raw_correct = sum(bool(row["raw_correct"]) for row in group)
        final_correct = sum(bool(row["final_correct"]) for row in group)
        rejected = sum(bool(row["rejected"]) for row in group)
        correct_but_rejected = sum(bool(row["correct_but_rejected"]) for row in group)
        other_final = sum(row["final_prediction"] == "其他" for row in group)
        summary.append(
            {
                "material": material,
                "focus_group": focus,
                "total": total,
                "raw_correct": raw_correct,
                "raw_accuracy": raw_correct / total if total else 0.0,
                "final_correct": final_correct,
                "final_accuracy": final_correct / total if total else 0.0,
                "rejected": rejected,
                "reject_rate": rejected / total if total else 0.0,
                "correct_but_rejected": correct_but_rejected,
                "correct_but_rejected_rate": correct_but_rejected / total if total else 0.0,
                "final_other": other_final,
                "final_other_rate": other_final / total if total else 0.0,
            }
        )
    return summary


def build_analysis(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    raw_correct = sum(bool(row["raw_correct"]) for row in rows)
    final_correct = sum(bool(row["final_correct"]) for row in rows)
    rejected = sum(bool(row["rejected"]) for row in rows)
    correct_but_rejected = sum(bool(row["correct_but_rejected"]) for row in rows)
    raw_counts = Counter(row["raw_prediction"] for row in rows)
    final_counts = Counter(row["final_prediction"] for row in rows)
    return {
        "total": total,
        "raw_correct": raw_correct,
        "raw_accuracy": raw_correct / total if total else 0.0,
        "final_correct": final_correct,
        "final_accuracy": final_correct / total if total else 0.0,
        "rejected": rejected,
        "reject_rate": rejected / total if total else 0.0,
        "correct_but_rejected": correct_but_rejected,
        "correct_but_rejected_rate": correct_but_rejected / total if total else 0.0,
        "raw_prediction_counts": dict(sorted(raw_counts.items())),
        "final_prediction_counts": dict(sorted(final_counts.items())),
        "interpretation": (
            "重点比较 raw_accuracy 与 final_accuracy，以及 correct_but_rejected。"
            "如果 correct_but_rejected 较高，说明当前‘其他’主要来自拒识阈值，"
            "不是模型原始类别判断错误；本模型中的‘其他’不是一个独立学习类别。"
        ),
        "by_group": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phase-2 料棚 images with the existing DINOv2 model.")
    parser.add_argument("--data-root", type=Path, default=PACKAGE_ROOT.parent / "ml_data" / "raw" / "料棚")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "output" / "phase2_dinov2_evaluation.json")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images; 0 means all images.")
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_records(args.data_root, DEFAULT_DATES)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("没有找到可识别图片")

    classifier = WasteClassifier(
        meta_path=META_PATH,
        prototype_path=PROTOTYPE_PATH,
        profile_path=PROFILE_PATH,
        local_repository=LOCAL_REPOSITORY,
        cache_dir=CACHE_DIR,
        device=args.device,
    )
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        row = predict_record(classifier, record)
        rows.append(row)
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"[phase2] {index}/{len(records)} {record['image_path']}", flush=True)

    summary = summarize(rows)
    output_payload = {
        "model_version": classifier.model_version,
        "model_meta_path": str(META_PATH),
        "prototype_path": str(PROTOTYPE_PATH),
        "data_root": str(args.data_root),
        "dates": list(DEFAULT_DATES),
        "analysis": build_analysis(rows, summary),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase2] saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
