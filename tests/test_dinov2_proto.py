from __future__ import annotations

import numpy as np
from PIL import Image

from solid_waste_model.config import MATERIAL_TYPE_LABELS
from solid_waste_model.core.dinov2_proto import (
    PrototypeBank,
    apply_rejection,
    build_prototype_bank,
    classify_features,
    crop_image,
    spherical_kmeans,
)
from solid_waste_model.gui import load_random_test_rows


def test_crop_image_keeps_requested_square_region() -> None:
    image = Image.new("RGB", (200, 100), color=(120, 120, 120))
    assert crop_image(image, "full", 0.78).size == (200, 100)
    assert crop_image(image, "center", 0.78).size == (78, 78)
    assert crop_image(image, "bottom_right", 0.78).size == (78, 78)


def test_spherical_kmeans_returns_normalized_centers() -> None:
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    centers = spherical_kmeans(vectors, cluster_count=2, iterations=10)
    assert centers.shape == (2, 2)
    assert np.allclose(np.linalg.norm(centers, axis=1), 1.0)


def test_multi_crop_classifier_prefers_nearest_label_and_rejects_low_margin() -> None:
    labels = list(MATERIAL_TYPE_LABELS)
    bank = PrototypeBank(
        labels=labels,
        prototypes=np.eye(len(labels), dtype=np.float32),
    )
    result = classify_features(
        {"full": np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32), "center": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)},
        bank,
        class_temperature=0.07,
        vote_temperature=0.03,
    )
    assert result["predicted_label"] == labels[0]
    assert not apply_rejection(result, {"confidence": 0.5, "similarity": 0.6, "margin": 0.2})


def test_build_prototype_bank_keeps_class_prototypes() -> None:
    first, second = MATERIAL_TYPE_LABELS[:2]
    encoded = {
        "a": {"material_type": first, "features": {"center": np.array([1.0, 0.0], dtype=np.float32)}},
        "b": {"material_type": second, "features": {"center": np.array([0.0, 1.0], dtype=np.float32)}},
    }
    bank = build_prototype_bank(encoded, prototypes_per_class=1, kmeans_iterations=2)
    assert bank.labels == [first, second]


def test_load_random_test_rows_uses_only_original_known_test_rows(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_path,material_type,is_known_class,usage_role,augmentation_source\n"
        "a.jpg,矿渣,True,known_test,original\n"
        "b.jpg,钢渣,True,known_test,original\n"
        "c.jpg,矿渣,True,known_train,original\n"
        "d.jpg,矿渣,True,known_test,offline_aug\n",
        encoding="utf-8",
    )
    rows = load_random_test_rows(manifest, 10)
    assert {row["image_path"] for row in rows} == {"a.jpg", "b.jpg"}
