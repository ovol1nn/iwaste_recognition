from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image


def normalize_image_path(path: str | Path) -> Path:
    raw_path = Path(path)
    try:
        return raw_path.resolve(strict=False)
    except OSError:
        return raw_path


def open_rgb_image(path: str | Path) -> Image.Image:
    image_path = normalize_image_path(path)
    try:
        data = image_path.read_bytes()
    except OSError as exc:
        raise OSError(f"cannot read image bytes: {image_path}") from exc

    bgr = cv2.imdecode(__import__("numpy").frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
    if bgr is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB")

    try:
        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise OSError(f"cannot open image: {image_path}") from exc