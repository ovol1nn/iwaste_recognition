from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import (
    DEFAULT_REJECT_CLASS_NAMES,
    DEFAULT_MANIFEST_CSV,
    MATERIAL_TYPE_LABELS,
    PROJECT_ROOT,
    SUPPORTED_IMAGE_SUFFIXES,
    build_history_artifact_dir,
)
from .image_io import open_rgb_image


DEFAULT_MODEL_VERSION = "solid_waste_dinov2_proto_v1"
DEFAULT_DINO_REPOSITORY = "facebookresearch/dinov2"
DEFAULT_TORCH_HUB_MODEL = "dinov2_vits14"
DEFAULT_HUGGINGFACE_MODEL = "facebook/dinov2-small"
DEFAULT_CROPS = ("full", "center", "top_left", "top_right", "bottom_left", "bottom_right")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class PrototypeBank:
    labels: list[str]
    prototypes: np.ndarray


@dataclass
class DinoEncoder:
    model: torch.nn.Module
    device: torch.device
    transform: transforms.Compose
    loader: str
    model_id: str


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using cpu", flush=True)
        return torch.device("cpu")
    return torch.device(requested_device)


def normalize_rows(
    manifest_path: Path,
    usage_roles: set[str],
    include_augmented: bool,
    class_labels: Sequence[str],
) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["usage_role"] in usage_roles
            and row["is_known_class"] == "True"
            and row["material_type"] in class_labels
            and (include_augmented or row.get("augmentation_source") == "original")
        ]
    if not rows:
        raise ValueError(f"no known rows for usage roles: {sorted(usage_roles)}")
    return rows


def build_transform(image_size: int) -> transforms.Compose:
    if image_size % 14 != 0:
        raise ValueError("image_size must be divisible by 14 for DINOv2 ViT-S/14")
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_dinov2(
    device: torch.device,
    image_size: int,
    loader: str,
    model_name: str | None,
    repository: str,
    cache_dir: Path | None,
    local_repository: Path | None,
) -> DinoEncoder:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))

    hub_error: Exception | None = None
    if loader in {"auto", "torchhub"}:
        source = "local" if local_repository is not None else "github"
        repo_or_dir = str(local_repository) if local_repository is not None else repository
        hub_model = model_name or DEFAULT_TORCH_HUB_MODEL
        print(f"[dinov2] loading loader=torchhub model={hub_model} source={source}", flush=True)
        try:
            model = torch.hub.load(repo_or_dir, hub_model, source=source, pretrained=True, trust_repo=True)
            model = model.to(device)
            model.eval()
            return DinoEncoder(
                model=model,
                device=device,
                transform=build_transform(image_size),
                loader="torchhub",
                model_id=hub_model,
            )
        except Exception as exc:
            hub_error = exc
            if loader == "torchhub":
                raise RuntimeError("unable to load pretrained DINOv2 from Torch Hub") from exc
            print(f"[dinov2] torchhub_failed={exc}; trying Hugging Face", flush=True)

    if loader not in {"auto", "huggingface"}:
        raise ValueError(f"unsupported DINOv2 loader: {loader}")
    huggingface_model = model_name or DEFAULT_HUGGINGFACE_MODEL
    print(f"[dinov2] loading loader=huggingface model={huggingface_model}", flush=True)
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(huggingface_model, cache_dir=str(cache_dir) if cache_dir else None)
    except ImportError as exc:
        raise RuntimeError("Hugging Face fallback requires transformers. Run: python -m pip install -r solid_waste_model/requirements-model.txt") from exc
    except Exception as exc:
        message = "unable to load pretrained DINOv2 from Hugging Face"
        if hub_error is not None:
            message += "; Torch Hub also failed"
        raise RuntimeError(message) from exc
    model = model.to(device)
    model.eval()
    return DinoEncoder(
        model=model,
        device=device,
        transform=build_transform(image_size),
        loader="huggingface",
        model_id=huggingface_model,
    )


def crop_image(image: Image.Image, crop_name: str, crop_ratio: float) -> Image.Image:
    if crop_name == "full":
        return image

    width, height = image.size
    side = max(1, int(min(width, height) * crop_ratio))
    max_left = max(width - side, 0)
    max_top = max(height - side, 0)
    positions = {
        "center": (max_left // 2, max_top // 2),
        "top_left": (0, 0),
        "top_right": (max_left, 0),
        "bottom_left": (0, max_top),
        "bottom_right": (max_left, max_top),
    }
    if crop_name not in positions:
        raise ValueError(f"unsupported crop: {crop_name}")
    left, top = positions[crop_name]
    return image.crop((left, top, left + side, top + side))


def _extract_embedding(output: object) -> torch.Tensor:
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state[:, 0]
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "x_norm_patchtokens", "features"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value.mean(dim=1) if value.ndim == 3 else value
        raise ValueError(f"DINOv2 forward_features output has no supported tensor key: {list(output)}")
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"unexpected DINOv2 output type: {type(output)!r}")
    return output.mean(dim=1) if output.ndim == 3 else output


def encode_images(
    encoder: DinoEncoder,
    images: Sequence[Image.Image],
    batch_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            tensors = torch.stack([encoder.transform(image) for image in images[start : start + batch_size]]).to(encoder.device)
            features = _extract_embedding(encoder.model(tensors))
            features = torch.nn.functional.normalize(features, dim=1)
            vectors.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 0), dtype=np.float32)


def encode_rows(
    encoder: DinoEncoder,
    rows: Sequence[dict[str, str]],
    crop_names: Sequence[str],
    crop_ratio: float,
    batch_size: int,
    stage_name: str,
) -> dict[str, dict[str, object]]:
    encoded: dict[str, dict[str, object]] = {}
    pending_images: list[Image.Image] = []
    pending_keys: list[tuple[str, str]] = []

    def flush() -> None:
        if not pending_images:
            return
        vectors = encode_images(encoder, pending_images, batch_size)
        for (image_path, crop_name), vector in zip(pending_keys, vectors, strict=True):
            encoded[image_path]["features"][crop_name] = vector  # type: ignore[index]
        pending_images.clear()
        pending_keys.clear()

    for index, row in enumerate(rows, start=1):
        image_path = row["image_path"]
        try:
            image = open_rgb_image(image_path)
        except OSError as exc:
            print(f"[feature-skip] cannot_open path={image_path} error={exc}", flush=True)
            continue

        encoded[image_path] = {
            "image_path": image_path,
            "material_type": row["material_type"],
            "scene": row.get("scene", "default"),
            "features": {},
        }
        for crop_name in crop_names:
            pending_images.append(crop_image(image, crop_name, crop_ratio))
            pending_keys.append((image_path, crop_name))
            if len(pending_images) >= batch_size:
                flush()
        if index % 25 == 0 or index == len(rows):
            print(f"[feature] stage={stage_name} images={index}/{len(rows)}", flush=True)
    flush()
    if not encoded:
        raise RuntimeError(f"no readable images in stage {stage_name}")
    return encoded


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def spherical_kmeans(vectors: np.ndarray, cluster_count: int, iterations: int) -> np.ndarray:
    vectors = l2_normalize(vectors.astype(np.float32))
    cluster_count = min(cluster_count, len(vectors))
    if cluster_count == 1:
        return l2_normalize(vectors.mean(axis=0, keepdims=True))

    seed_indices = np.linspace(0, len(vectors) - 1, cluster_count, dtype=int)
    centers = vectors[seed_indices].copy()
    previous_assignment: np.ndarray | None = None
    for _ in range(iterations):
        assignment = np.argmax(vectors @ centers.T, axis=1)
        if previous_assignment is not None and np.array_equal(assignment, previous_assignment):
            break
        previous_assignment = assignment
        updated: list[np.ndarray] = []
        for cluster_index in range(cluster_count):
            members = vectors[assignment == cluster_index]
            updated.append(centers[cluster_index] if len(members) == 0 else members.mean(axis=0))
        centers = l2_normalize(np.stack(updated, axis=0))
    return centers.astype(np.float32)


def build_prototype_bank(
    encoded_rows: dict[str, dict[str, object]],
    prototypes_per_class: int,
    kmeans_iterations: int,
    class_labels: Sequence[str] = MATERIAL_TYPE_LABELS,
) -> PrototypeBank:
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in encoded_rows.values():
        label = str(record["material_type"])
        features = record["features"]
        for vector in features.values():  # type: ignore[union-attr]
            by_label[label].append(vector)  # type: ignore[arg-type]

    labels: list[str] = []
    prototypes: list[np.ndarray] = []
    for label in class_labels:
        vectors = by_label.get(label, [])
        if not vectors:
            continue
        centers = spherical_kmeans(np.stack(vectors), prototypes_per_class, kmeans_iterations)
        labels.extend([label] * len(centers))
        prototypes.extend(centers)
        print(f"[prototype] label={label} feature_count={len(vectors)} prototype_count={len(centers)}", flush=True)
    if not prototypes:
        raise ValueError("cannot build prototypes without feature vectors")
    return PrototypeBank(labels=labels, prototypes=np.stack(prototypes).astype(np.float32))


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / np.maximum(exp.sum(), 1e-12)


def class_similarities(vector: np.ndarray, bank: PrototypeBank) -> dict[str, float]:
    similarities = l2_normalize(vector.reshape(1, -1))[0] @ bank.prototypes.T
    class_labels = list(dict.fromkeys(bank.labels))
    return {
        label: float(max(similarities[index] for index, prototype_label in enumerate(bank.labels) if prototype_label == label))
        for label in class_labels
    }


def classify_features(
    features: dict[str, np.ndarray],
    bank: PrototypeBank,
    class_temperature: float,
    vote_temperature: float,
) -> dict[str, object]:
    class_labels = list(dict.fromkeys(bank.labels))
    crop_results: list[dict[str, object]] = []
    crop_max_scores: list[float] = []
    for crop_name, vector in features.items():
        scores = class_similarities(vector, bank)
        score_vector = np.array([scores[label] for label in class_labels], dtype=np.float32)
        probabilities = softmax(score_vector, class_temperature)
        top_index = int(np.argmax(score_vector))
        crop_results.append(
            {
                "crop": crop_name,
                "scores": scores,
                "top_label": class_labels[top_index],
                "top_similarity": float(score_vector[top_index]),
                "probabilities": {label: float(probabilities[index]) for index, label in enumerate(class_labels)},
            }
        )
        crop_max_scores.append(float(score_vector[top_index]))

    if not crop_results:
        raise ValueError("cannot classify an image without crop features")
    weights = softmax(np.array(crop_max_scores, dtype=np.float32), vote_temperature)
    probabilities = np.zeros(len(class_labels), dtype=np.float32)
    winner_similarities = np.zeros(len(crop_results), dtype=np.float32)
    for index, crop_result in enumerate(crop_results):
        crop_probabilities = crop_result["probabilities"]
        probabilities += weights[index] * np.array([crop_probabilities[label] for label in class_labels])  # type: ignore[index]

    order = np.argsort(probabilities)[::-1]
    winner_index = int(order[0])
    winner_label = class_labels[winner_index]
    for index, crop_result in enumerate(crop_results):
        winner_similarities[index] = crop_result["scores"][winner_label]  # type: ignore[index]

    return {
        "predicted_label": winner_label,
        "confidence": float(probabilities[winner_index]),
        "margin": float(probabilities[winner_index] - probabilities[int(order[1])]) if len(order) > 1 else 1.0,
        "similarity": float(np.dot(weights, winner_similarities)),
        "class_probabilities": {label: float(probabilities[index]) for index, label in enumerate(class_labels)},
        "crop_results": crop_results,
        "crop_weights": {crop_results[index]["crop"]: float(weights[index]) for index in range(len(crop_results))},
    }


def classify_encoded_rows(
    encoded_rows: dict[str, dict[str, object]],
    bank: PrototypeBank,
    class_temperature: float,
    vote_temperature: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for record in encoded_rows.values():
        prediction = classify_features(record["features"], bank, class_temperature, vote_temperature)  # type: ignore[arg-type]
        prediction.update(
            {
                "image_path": record["image_path"],
                "true_label": record["material_type"],
                "scene": record["scene"],
            }
        )
        prediction["is_correct"] = prediction["predicted_label"] == prediction["true_label"]
        results.append(prediction)
    return results


def calibrate_thresholds(validation_results: Sequence[dict[str, object]], quantile: float) -> dict[str, float]:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("rejection_quantile must be between 0 and 1")
    correct = [result for result in validation_results if result["is_correct"]]
    if not correct:
        return {"confidence": 0.0, "similarity": -1.0, "margin": -1.0}
    return {
        "confidence": float(np.quantile([result["confidence"] for result in correct], quantile)),
        "similarity": float(np.quantile([result["similarity"] for result in correct], quantile)),
        "margin": float(np.quantile([result["margin"] for result in correct], quantile)),
    }


def apply_rejection(result: dict[str, object], thresholds: dict[str, float]) -> bool:
    return (
        float(result["confidence"]) < thresholds["confidence"]
        or float(result["similarity"]) < thresholds["similarity"]
        or float(result["margin"]) < thresholds["margin"]
    )


def summarize_results(results: Sequence[dict[str, object]], thresholds: dict[str, float] | None = None) -> dict[str, object]:
    total = len(results)
    correct = sum(bool(result["is_correct"]) for result in results)
    summary: dict[str, object] = {
        "count": total,
        "raw_accuracy": float(correct / total) if total else 0.0,
        "by_scene": {},
    }
    if thresholds is not None:
        rejected = sum(apply_rejection(result, thresholds) for result in results)
        accepted_correct = sum(bool(result["is_correct"]) and not apply_rejection(result, thresholds) for result in results)
        summary.update(
            {
                "rejection_rate": float(rejected / total) if total else 0.0,
                "accepted_correct_rate": float(accepted_correct / total) if total else 0.0,
            }
        )
    scene_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        scene_groups[str(result["scene"])].append(result)
    summary["by_scene"] = {
        scene: {
            "count": len(scene_results),
            "raw_accuracy": float(sum(bool(item["is_correct"]) for item in scene_results) / len(scene_results)),
        }
        for scene, scene_results in sorted(scene_groups.items())
    }
    return summary


def write_error_report(results: Sequence[dict[str, object]], thresholds: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path",
        "scene",
        "true_label",
        "predicted_label",
        "is_correct",
        "rejected",
        "confidence",
        "similarity",
        "margin",
        "class_probabilities",
        "crop_weights",
        "crop_results",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if bool(result["is_correct"]) and not apply_rejection(result, thresholds):
                continue
            writer.writerow(
                {
                    "image_path": result["image_path"],
                    "scene": result["scene"],
                    "true_label": result["true_label"],
                    "predicted_label": result["predicted_label"],
                    "is_correct": result["is_correct"],
                    "rejected": apply_rejection(result, thresholds),
                    "confidence": f"{float(result['confidence']):.6f}",
                    "similarity": f"{float(result['similarity']):.6f}",
                    "margin": f"{float(result['margin']):.6f}",
                    "class_probabilities": json.dumps(result["class_probabilities"], ensure_ascii=False),
                    "crop_weights": json.dumps(result["crop_weights"], ensure_ascii=False),
                    "crop_results": json.dumps(result["crop_results"], ensure_ascii=False),
                }
            )


def save_bank(bank: PrototypeBank, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, labels=np.array(bank.labels), prototypes=bank.prototypes)


def load_bank(path: Path) -> PrototypeBank:
    with np.load(path, allow_pickle=False) as data:
        return PrototypeBank(labels=[str(item) for item in data["labels"]], prototypes=data["prototypes"].astype(np.float32))


def artifact_paths(output_dir: Path, model_version: str) -> dict[str, Path]:
    return {
        "prototype": output_dir / f"{model_version}.prototypes.npz",
        "meta": output_dir / f"{model_version}.meta.json",
        "report": output_dir / f"{model_version}.report.json",
        "val_errors": output_dir / f"{model_version}.validation_errors.csv",
        "test_errors": output_dir / f"{model_version}.test_errors.csv",
    }


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    output_dir = args.output_dir or build_history_artifact_dir(args.model_version)
    paths = artifact_paths(output_dir, args.model_version)
    crop_names = tuple(name.strip() for name in args.crops.split(",") if name.strip())
    invalid_crops = set(crop_names) - set(DEFAULT_CROPS)
    if invalid_crops:
        raise ValueError(f"unsupported crops: {sorted(invalid_crops)}")
    if not 0.0 < args.crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in (0, 1]")
    if args.batch_size < 1 or args.prototypes_per_class < 1:
        raise ValueError("batch_size and prototypes_per_class must be positive")
    class_labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    unknown_labels = set(class_labels) - set(MATERIAL_TYPE_LABELS)
    if len(class_labels) < 2 or unknown_labels:
        raise ValueError(f"labels must contain at least two known classes; invalid={sorted(unknown_labels)}")

    train_rows = normalize_rows(args.manifest, {"known_train"}, args.include_augmented_train, class_labels)
    validation_rows = normalize_rows(args.manifest, {"known_val"}, False, class_labels)
    test_rows = normalize_rows(args.manifest, {"known_test"}, False, class_labels)
    print(
        f"[dataset] train={len(train_rows)} val={len(validation_rows)} test={len(test_rows)} "
        f"include_augmented_train={args.include_augmented_train}",
        flush=True,
    )
    encoder = load_dinov2(
        device=device,
        image_size=args.image_size,
        loader=args.loader,
        model_name=args.dino_model,
        repository=args.repository,
        cache_dir=args.cache_dir,
        local_repository=args.local_repository,
    )
    train_encoded = encode_rows(encoder, train_rows, crop_names, args.crop_ratio, args.batch_size, "train")
    bank = build_prototype_bank(train_encoded, args.prototypes_per_class, args.kmeans_iterations, class_labels)
    validation_encoded = encode_rows(encoder, validation_rows, crop_names, args.crop_ratio, args.batch_size, "validation")
    test_encoded = encode_rows(encoder, test_rows, crop_names, args.crop_ratio, args.batch_size, "test")

    train_results = classify_encoded_rows(train_encoded, bank, args.class_temperature, args.vote_temperature)
    validation_results = classify_encoded_rows(validation_encoded, bank, args.class_temperature, args.vote_temperature)
    thresholds = calibrate_thresholds(validation_results, args.rejection_quantile)
    test_results = classify_encoded_rows(test_encoded, bank, args.class_temperature, args.vote_temperature)

    save_bank(bank, paths["prototype"])
    write_error_report(validation_results, thresholds, paths["val_errors"])
    write_error_report(test_results, thresholds, paths["test_errors"])
    report = {
        "model_version": args.model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "dinov2_vits14_frozen_feature_multi_prototype",
        "class_labels": list(class_labels),
        "dino_loader": encoder.loader,
        "dino_model": encoder.model_id,
        "image_size": args.image_size,
        "class_temperature": args.class_temperature,
        "vote_temperature": args.vote_temperature,
        "manifest": args.manifest.as_posix(),
        "crop_names": list(crop_names),
        "crop_ratio": args.crop_ratio,
        "prototypes_per_class_max": args.prototypes_per_class,
        "prototype_count": len(bank.labels),
        "rejection_thresholds": thresholds,
        "threshold_note": "Thresholds are calibrated only on correctly classified known validation images. No unknown-material validation data is available.",
        "metrics": {
            "train": summarize_results(train_results, thresholds),
            "validation": summarize_results(validation_results, thresholds),
            "test": summarize_results(test_results, thresholds),
        },
        "artifacts": {key: value.as_posix() for key, value in paths.items()},
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["meta"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"[complete] artifacts={output_dir}", flush=True)


def collect_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES else []
    iterator: Iterable[Path] = path.rglob("*") if recursive else path.glob("*")
    return sorted(item for item in iterator if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def predict(args: argparse.Namespace) -> None:
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    bank = load_bank(args.prototype)
    device = resolve_device(args.device)
    encoder = load_dinov2(
        device=device,
        image_size=int(meta.get("image_size", args.image_size)),
        loader=str(meta.get("dino_loader", args.loader)),
        model_name=str(meta["dino_model"]),
        repository=args.repository,
        cache_dir=args.cache_dir,
        local_repository=args.local_repository,
    )
    crop_names = tuple(meta["crop_names"])
    crop_ratio = float(meta["crop_ratio"])
    thresholds = {key: float(value) for key, value in meta["rejection_thresholds"].items()}
    paths = collect_images(args.input, args.recursive)
    if not paths:
        raise ValueError(f"no supported images found: {args.input}")
    rows = [
        {"image_path": path.as_posix(), "material_type": "", "scene": "predict"}
        for path in paths
    ]
    encoded = encode_rows(encoder, rows, crop_names, crop_ratio, args.batch_size, "predict")
    results: list[dict[str, object]] = []
    for record in encoded.values():
        result = classify_features(
            record["features"],
            bank,
            float(meta.get("class_temperature", args.class_temperature)),
            float(meta.get("vote_temperature", args.vote_temperature)),
        )  # type: ignore[arg-type]
        rejected = apply_rejection(result, thresholds) if args.enable_rejection else False
        reject_label = next(iter(DEFAULT_REJECT_CLASS_NAMES), "其他")
        result.update(
            {
                "image_path": record["image_path"],
                "result_label": reject_label if rejected else result["predicted_label"],
                "rejected": rejected,
            }
        )
        results.append(result)
        print(
            f"[predict] image={record['image_path']} result={result['result_label']} "
            f"confidence={float(result['confidence']):.4f} similarity={float(result['similarity']):.4f}",
            flush=True,
        )
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[predict] saved={args.save_json}", flush=True)


def add_shared_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--loader", choices=["auto", "torchhub", "huggingface"], default="auto")
    parser.add_argument("--dino-model", help="Optional model id. Defaults depend on the selected loader.")
    parser.add_argument("--repository", default=DEFAULT_DINO_REPOSITORY)
    parser.add_argument("--local-repository", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "workspace" / "cache" / "torch_hub")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--class-temperature", type=float, default=0.07)
    parser.add_argument("--vote-temperature", type=float, default=0.03)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv2 frozen-feature multi-prototype classifier for solid-waste images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="extract DINOv2 features, build prototypes, and evaluate")
    train_parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    train_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_CSV)
    train_parser.add_argument("--output-dir", type=Path)
    train_parser.add_argument("--prototypes-per-class", type=int, default=3)
    train_parser.add_argument("--kmeans-iterations", type=int, default=50)
    train_parser.add_argument("--crops", default=",".join(DEFAULT_CROPS))
    train_parser.add_argument("--crop-ratio", type=float, default=0.78)
    train_parser.add_argument("--rejection-quantile", type=float, default=0.10)
    train_parser.add_argument("--include-augmented-train", action="store_true")
    train_parser.add_argument("--labels", default=",".join(MATERIAL_TYPE_LABELS), help="Comma-separated active class labels for this model version.")
    add_shared_model_arguments(train_parser)

    predict_parser = subparsers.add_parser("predict", help="predict one image or a folder with saved prototypes")
    predict_parser.add_argument("input", type=Path)
    predict_parser.add_argument("--prototype", type=Path, required=True)
    predict_parser.add_argument("--meta", type=Path, required=True)
    predict_parser.add_argument("--recursive", action="store_true")
    predict_parser.add_argument("--enable-rejection", action="store_true")
    predict_parser.add_argument("--save-json", type=Path)
    add_shared_model_arguments(predict_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
