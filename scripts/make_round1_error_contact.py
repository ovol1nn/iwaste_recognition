"""Make a visual contact sheet of selected existing error images."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ERROR_CSV = PROJECT_ROOT / "IWaste_rec_model" / "artifacts" / "solid_waste_dinov2_proto_round2_baseline_grouped_v2" / "solid_waste_dinov2_proto_round2_baseline_grouped_v2.test_errors.csv"
OUTPUT = PROJECT_ROOT / "IWaste_rec_model" / "output" / "round1_material_error_analysis" / "coal_to_steel_errors_contact.png"


def main() -> None:
    rows = list(csv.DictReader(ERROR_CSV.open("r", encoding="utf-8-sig", newline="")))
    rows = [row for row in rows if row["true_label"] == "\u7164\u77f8\u77f3" and row["predicted_label"] == "\u94a2\u6e23"]
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18, index=0)
    tiles: list[Image.Image] = []
    for index, row in enumerate(rows, start=1):
        image = Image.open(row["image_path"]).convert("RGB")
        image.thumbnail((420, 300))
        tile = Image.new("RGB", (440, 350), "white")
        tile.paste(image, ((440 - image.width) // 2, 10))
        draw = ImageDraw.Draw(tile)
        draw.text((10, 315), f"{index}: 煤矸石→钢渣  conf={float(row['confidence']):.3f}", font=font, fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (880, 350 * ((len(tiles) + 1) // 2)), "#eeeeee")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 440, (index // 2) * 350))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT)
    print(f"selected_errors={len(rows)}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
