from pathlib import Path

from PIL import Image

from solid_waste_model.manifest import assign_split, build_manifest_records, build_sample_group, order_source_files, write_manifest


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(127, 127, 127)).save(path)


def create_dataset_fixture(root: Path) -> tuple[Path, Path]:
    dataset_root = root / "dataset"
    offline_aug_root = root / "offline_aug"

    layout = {
        "实验室": {
            "煤矸石": 10,
            "矿渣": 8,
            "钢渣": 6,
            "石膏": 5,
        },
        "料棚": {
            "煤矸石": 7,
            "矿渣": 5,
            "钢渣": 5,
        },
    }
    for scene_name, scene_layout in layout.items():
        for class_name, count in scene_layout.items():
            for index in range(count):
                create_image(dataset_root / scene_name / class_name / f"{class_name}_{index:02d}.jpg")

    # Only train originals receive offline augmentation.
    for scene_name, scene_layout in layout.items():
        for class_name in scene_layout:
            class_dir = dataset_root / scene_name / class_name
            files = order_source_files(class_dir.iterdir())
            train_files = [
                path
                for index, path in enumerate(files)
                if assign_split(index=index, total=len(files), is_known_class=True) == "train"
            ]
            for image_path in train_files[:1]:
                sample_group = build_sample_group(scene_name, class_name, image_path)
                for aug_index in range(2):
                    create_image(offline_aug_root / scene_name / class_name / sample_group / f"{sample_group}__aug_{aug_index:02d}.jpg")

    return dataset_root, offline_aug_root


def test_build_manifest_records_counts_and_roles(tmp_path: Path) -> None:
    dataset_root, offline_aug_root = create_dataset_fixture(tmp_path)
    records = build_manifest_records(dataset_root=dataset_root, offline_aug_root=offline_aug_root)

    usage_role_counts: dict[str, int] = {}
    for record in records:
        usage_role_counts[record.usage_role] = usage_role_counts.get(record.usage_role, 0) + 1

    assert usage_role_counts["known_val"] == 14
    assert usage_role_counts["known_test"] == 14
    assert usage_role_counts["known_train"] == 32
    assert "reject_train" not in usage_role_counts

    offline_aug_count = sum(record.augmentation_source == "offline_aug" for record in records)
    assert offline_aug_count == 14
    assert {record.scene for record in records} == {"实验室", "料棚"}


def test_manifest_has_required_fields_and_no_aug_leakage(tmp_path: Path) -> None:
    dataset_root, offline_aug_root = create_dataset_fixture(tmp_path)
    records = build_manifest_records(dataset_root=dataset_root, offline_aug_root=offline_aug_root)

    assert all(record.sample_group for record in records)
    assert all(record.scene for record in records)

    group_roles: dict[str, set[str]] = {}
    for record in records:
        group_roles.setdefault(record.sample_group, set()).add(record.usage_role)
    assert all(len(roles) == 1 for roles in group_roles.values())
    assert all(
        not (record.augmentation_source == "offline_aug" and record.usage_role.endswith(("val", "test")))
        for record in records
    )


def test_write_manifest_outputs(tmp_path: Path) -> None:
    dataset_root, offline_aug_root = create_dataset_fixture(tmp_path)
    records = build_manifest_records(dataset_root=dataset_root, offline_aug_root=offline_aug_root)
    csv_path = tmp_path / "manifest.csv"
    jsonl_path = tmp_path / "manifest.jsonl"
    write_manifest(records, csv_path, jsonl_path)
    assert csv_path.exists()
    assert jsonl_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "sample_group" in csv_text
    assert "scene" in csv_text