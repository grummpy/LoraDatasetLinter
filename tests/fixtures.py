"""Synthetic dataset pictures: shapes and gradients only."""

from __future__ import annotations

import hashlib
from pathlib import Path

from loradatasetlinter.policy import Policy, load_policy
from PIL import Image, ImageDraw


def relax(policy: Policy | None = None) -> Policy:
    """Small fixtures should not trip the 512 px training minimum."""
    loaded = policy or load_policy(None)
    loaded.resolution.min_side = 8
    return loaded


def save_rgb(path: Path, size: tuple[int, int], paint) -> Path:
    image = Image.new("RGB", size, (0, 0, 0))
    pixels = image.load()
    width, height = size
    for y in range(height):
        for x in range(width):
            pixels[x, y] = paint(x, y, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return path


def gradient(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    def paint(x, y, width, height):
        value = int(255 * x / max(1, width - 1))
        return (value, 30, 255 - value)

    return save_rgb(path, size, paint)


def circle(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    image = Image.new("RGB", size, (12, 18, 36))
    draw = ImageDraw.Draw(image)
    draw.ellipse((6, 10, size[0] - 8, size[1] - 6), fill=(220, 64, 96))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return path


def checker(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    def paint(x, y, width, height):
        on = (x + y) % 2 == 0
        return (230, 230, 240) if on else (20, 24, 48)

    return save_rgb(path, size, paint)


def diagonal(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    image = Image.new("RGB", size, (16, 40, 48))
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, size[0] - 1, size[1] - 1), fill=(240, 200, 40), width=3)
    draw.rectangle((4, 40, 24, 60), fill=(40, 180, 160))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return path


def solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Path:
    image = Image.new("RGB", size, color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return path


def nearest_upscale(path: Path, factor: int = 4, base: int = 16) -> Path:
    small = Image.new("RGB", (base, base), (20, 20, 30))
    draw = ImageDraw.Draw(small)
    draw.rectangle((1, 1, base // 2, base - 2), fill=(200, 80, 40))
    draw.ellipse((base // 2, 2, base - 2, base // 2), fill=(40, 180, 200))
    big = small.resize((base * factor, base * factor), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    big.save(path, "PNG")
    return path


def caption(image_path: Path, text: str) -> Path:
    path = image_path.with_suffix(".txt")
    path.write_text(text, encoding="utf-8")
    return path


def clean_dataset(root: Path) -> Path:
    """Four different pictures that share one caption, so tag drift stays quiet."""
    root.mkdir(parents=True, exist_ok=True)
    text = "red square, studio lighting"
    for name, maker in (
        ("gradient.png", gradient),
        ("circle.png", circle),
        ("checker.png", checker),
        ("diagonal.png", diagonal),
    ):
        path = maker(root / name)
        caption(path, text)
    return root


def snapshot(root: Path) -> dict[str, tuple[str, int, int]]:
    found: dict[str, tuple[str, int, int]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stat = path.stat()
        found[path.relative_to(root).as_posix()] = (digest, stat.st_size, stat.st_mtime_ns)
    return found


def finding_codes(report) -> set[str]:
    return {finding.code for finding in report.findings}
