from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .config import DEFAULT_MANIFEST_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect manifest split and augmentation distribution.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    sample_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        usage_role = row.get("usage_role")
        if usage_role is None:
            raise KeyError(f"manifest row missing usage_role: {row}")
        counts[(row.get("scene", "default"), row["material_type"], usage_role, row["augmentation_source"])] += 1
        sample_groups[row["sample_group"]].add(usage_role)

    print(f"total_rows={len(rows)}")
    for key in sorted(counts):
        scene, material_type, usage_role, augmentation_source = key
        print(f"{scene} | {material_type} | usage_role={usage_role:<12} | aug={augmentation_source:<11} | count={counts[key]}")

    leaked_groups = {group: roles for group, roles in sample_groups.items() if len({role.split('_')[-1] for role in roles}) > 1}
    print(f"cross_split_groups={len(leaked_groups)}")
    if leaked_groups:
        for group, roles in sorted(leaked_groups.items()):
            print(f"[leak] {group}: {sorted(roles)}")


if __name__ == "__main__":
    main()
