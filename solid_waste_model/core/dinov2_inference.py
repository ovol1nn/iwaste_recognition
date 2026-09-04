from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_PROFILE_PATH, DEFAULT_REJECT_CLASS_NAMES, HISTORY_ARTIFACTS_DIR, PROJECT_ROOT
from .dinov2_proto import DEFAULT_DINO_REPOSITORY, PrototypeBank, classify_features, crop_image, encode_images, load_bank, load_dinov2
from .image_io import open_rgb_image


DEFAULT_MODEL_VERSION = "solid_waste_dinov2_proto_v1"
DEFAULT_ARTIFACT_DIR = HISTORY_ARTIFACTS_DIR / DEFAULT_MODEL_VERSION
DEFAULT_META_PATH = DEFAULT_ARTIFACT_DIR / f"{DEFAULT_MODEL_VERSION}.meta.json"
DEFAULT_PROTOTYPE_PATH = DEFAULT_ARTIFACT_DIR / f"{DEFAULT_MODEL_VERSION}.prototypes.npz"
DEFAULT_HUB_CACHE_DIR = PROJECT_ROOT / "runtime" / "torch_hub"
DEFAULT_LOCAL_REPOSITORY = DEFAULT_HUB_CACHE_DIR / "facebookresearch_dinov2_main"


class WasteClassifier:
    """DINOv2 adapter preserving the legacy WasteClassifier public methods."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        meta_path: str | Path | None = None,
        profile_path: str | Path | None = None,
        prototype_path: str | Path | None = None,
        threshold: float | None = None,
        similarity_threshold: float | None = None,
        margin_threshold: float | None = None,
        device: str | None = None,
        local_repository: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.meta_path = Path(meta_path) if meta_path else DEFAULT_META_PATH
        self.profile_path = Path(profile_path) if profile_path else DEFAULT_PROFILE_PATH
        self.prototype_path = Path(prototype_path) if prototype_path else DEFAULT_PROTOTYPE_PATH
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_HUB_CACHE_DIR
        configured_repository = Path(local_repository) if local_repository else DEFAULT_LOCAL_REPOSITORY
        self.local_repository = configured_repository if configured_repository.is_dir() else None

        self.meta = self._load_json(self.meta_path)
        self.material_profiles = self._load_json(self.profile_path)
        self.bank: PrototypeBank = load_bank(self.prototype_path)
        configured_reject_labels = set(DEFAULT_REJECT_CLASS_NAMES)
        profile_only_labels = set(self.material_profiles) - set(self.bank.labels)
        self.reject_label = next(iter(configured_reject_labels or profile_only_labels), "其他")
        thresholds = self.meta["rejection_thresholds"]
        self.threshold = float(threshold if threshold is not None else thresholds["confidence"])
        self.similarity_threshold = float(
            similarity_threshold if similarity_threshold is not None else thresholds["similarity"]
        )
        self.margin_threshold = float(margin_threshold if margin_threshold is not None else thresholds["margin"])
        self.model_version = str(self.meta["model_version"])
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested_device)
        self.encoder = load_dinov2(
            device=self.device,
            image_size=int(self.meta["image_size"]),
            loader="torchhub" if self.local_repository is not None else str(self.meta["dino_loader"]),
            model_name=str(self.meta["dino_model"]),
            repository=DEFAULT_DINO_REPOSITORY,
            cache_dir=self.cache_dir,
            local_repository=self.local_repository,
        )
        self.crop_names = tuple(self.meta["crop_names"])
        self.crop_ratio = float(self.meta["crop_ratio"])
        self.class_temperature = float(self.meta["class_temperature"])
        self.vote_temperature = float(self.meta["vote_temperature"])

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _to_pil_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return open_rgb_image(image)
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image.astype(np.uint8), mode="L").convert("RGB")
            if image.ndim != 3:
                raise ValueError("numpy image must have 2 or 3 dimensions")
            rgb = image[:, :, :3].astype(np.uint8)[:, :, ::-1]
            return Image.fromarray(rgb, mode="RGB")
        raise TypeError(f"unsupported image type: {type(image)!r}")

    def predict_details(self, image: str | Path | Image.Image | np.ndarray) -> dict[str, Any]:
        pil_image = self._to_pil_image(image)
        vectors = encode_images(
            self.encoder,
            [crop_image(pil_image, crop_name, self.crop_ratio) for crop_name in self.crop_names],
            batch_size=len(self.crop_names),
        )
        result = classify_features(
            {crop_name: vectors[index] for index, crop_name in enumerate(self.crop_names)},
            self.bank,
            self.class_temperature,
            self.vote_temperature,
        )
        confidence = float(result["confidence"])
        similarity = float(result["similarity"])
        margin = float(result["margin"])
        probability_rejected = confidence < self.threshold
        similarity_rejected = similarity < self.similarity_threshold
        margin_rejected = margin < self.margin_threshold
        rejected = probability_rejected or similarity_rejected or margin_rejected
        waste_type = self.reject_label if rejected else result["predicted_label"]
        profile = dict(self.material_profiles.get(waste_type, self.material_profiles[self.reject_label]))

        return {
            "waste_type": waste_type,
            "confidence": confidence,
            "features": {
                "particle_size": profile["particle_size"],
                "shape": profile["shape"],
                "color": profile["color"],
                "source": "config_profile",
            },
            "model_result": {
                "model_version": self.model_version,
                "model_family": "dinov2_vits14_frozen_feature_multi_prototype",
                "material_scores": result["class_probabilities"],
                "threshold": self.threshold,
                "rejected": rejected,
                "probability_rejected": probability_rejected,
                "similarity_score": similarity,
                "similarity_threshold": self.similarity_threshold,
                "similarity_rejected": similarity_rejected,
                "margin": margin,
                "margin_threshold": self.margin_threshold,
                "margin_rejected": margin_rejected,
                "crop_weights": result["crop_weights"],
                "crop_results": result["crop_results"],
                "attribute_outputs": {
                    "particle_size": None,
                    "shape": None,
                    "color": None,
                },
            },
            "model_error": None,
        }

    def predict(self, image: str | Path | Image.Image | np.ndarray) -> tuple[str, float]:
        details = self.predict_details(image)
        return str(details["waste_type"]), float(details["confidence"])

    def safe_predict_payload(self, image: str | Path | Image.Image | np.ndarray) -> dict[str, Any]:
        try:
            return self.predict_details(image)
        except Exception as exc:
            return {
                "waste_type": None,
                "confidence": 0.0,
                "features": None,
                "model_result": None,
                "model_error": str(exc),
            }
