"""Read-only walk of a kohya/ComfyUI image folder."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError

from loradatasetlinter.models import Dataset, ImageRecord
from loradatasetlinter.pixels import upscale_factor
from loradatasetlinter.policy import Policy

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_REPEAT = re.compile(r"^(\d+)_(.+)$")
_ORIENTATION = 274
ORIENTATION_NAMES = {
    1: "normal",
    2: "flip horizontal",
    3: "rotate 180",
    4: "flip vertical",
    5: "transpose",
    6: "rotate 90 CW",
    7: "transverse",
    8: "rotate 90 CCW",
}


def scan_folder(root: Path, policy: Policy) -> Dataset:
    images: list[ImageRecord] = []
    caption_files: list[str] = []
    other: list[str] = []
    symlinks: list[str] = []
    for kind, path in _walk(root):
        rel = path.relative_to(root).as_posix()
        if kind == "symlink":
            symlinks.append(rel)
            continue
        suffix = path.suffix.lower()
        if suffix == ".txt":
            caption_files.append(rel)
            continue
        if suffix in IMAGE_EXTENSIONS:
            images.append(_read_image(root, path, rel, policy))
            continue
        other.append(rel)

    by_stem: dict[tuple[str, str], list[str]] = {}
    for rel in caption_files:
        by_stem.setdefault((_parent_key(rel), Path(rel).stem), []).append(rel)
    claimed: set[str] = set()
    for record in images:
        options = by_stem.get((_parent_key(record.rel), Path(record.rel).stem), [])
        if not options:
            continue
        exact = [item for item in options if item.endswith(".txt")]
        caption_rel = sorted(exact or options)[0]
        record.caption_rel = caption_rel
        record.caption_text = _read_text(root / caption_rel)
        claimed.add(caption_rel)
    orphans = sorted(rel for rel in caption_files if rel not in claimed)
    images.sort(key=lambda record: record.rel)
    return Dataset(
        root=root,
        images=images,
        orphan_captions=orphans,
        other_files=sorted(other),
        skipped_symlinks=sorted(symlinks),
    )


def shared_caption_groups(dataset: Dataset) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for record in dataset.images:
        if record.caption_rel:
            groups.setdefault(record.caption_rel, []).append(record.rel)
    shared = [(caption, sorted(files)) for caption, files in groups.items() if len(files) > 1]
    shared.sort()
    return shared


def _walk(root: Path):
    found: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if not name.startswith(".") and name != "_linter_review"
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                found.append(("symlink", path))
            elif path.is_file():
                found.append(("file", path))
    return found


def _read_image(root: Path, path: Path, rel: str, policy: Policy) -> ImageRecord:
    file_hash = _sha256(path)
    file_size = path.stat().st_size
    class_name, repeats, folder = _class_for(Path(rel), policy.stats.default_repeats)
    record = ImageRecord(
        rel=rel,
        path=path,
        file_hash=file_hash,
        file_size=file_size,
        repeats=repeats,
        class_name=class_name,
        folder=folder,
    )
    try:
        with Image.open(path) as image:
            image.load()
            record.width, record.height = image.size
            record.mode = image.mode
            record.format = _format_name(image.format)
            record.exif_orientation = _orientation(image)
            record.has_alpha, record.alpha_nonopaque = _alpha(image)
            oriented = ImageOps.exif_transpose(image)
            record.oriented_width, record.oriented_height = oriented.size
            rgb = oriented.convert("RGB")
            record.pixel_hash = _pixel_hash(rgb)
            if policy.duplicates.enabled and policy.duplicates.perceptual:
                record.perceptual_hash, record.perceptual_hex = _perceptual(rgb, policy)
            if policy.resolution.enabled and policy.resolution.upscale_artifacts:
                record.upscale_factor = upscale_factor(
                    rgb, policy.resolution.upscale_block_fraction
                )
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        record.error = str(exc) or exc.__class__.__name__
    return record


def _perceptual(image: Image.Image, policy: Policy) -> tuple[int, str]:
    method = imagehash.phash if policy.duplicates.hash_method == "phash" else imagehash.dhash
    digest = method(image, hash_size=policy.duplicates.hash_size)
    hex_digest = str(digest)
    return int(hex_digest, 16), hex_digest


def _pixel_hash(image: Image.Image) -> str:
    digest = hashlib.sha256()
    width, height = image.size
    digest.update(f"{width}x{height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _alpha(image: Image.Image) -> tuple[bool, bool]:
    if image.mode in {"RGBA", "LA"}:
        low, _high = image.getchannel("A").getextrema()
        return True, low < 255
    if image.mode in {"PA", "RGBa"}:
        return True, True
    if image.mode == "P" and "transparency" in image.info:
        return True, True
    return False, False


def _orientation(image: Image.Image) -> int | None:
    try:
        value = image.getexif().get(_ORIENTATION)
    except (AttributeError, OSError, ValueError):
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_name(name: str | None) -> str:
    label = (name or "UNKNOWN").upper()
    if label == "JPG":
        return "JPEG"
    return label


def _class_for(rel: Path, default_repeats: int) -> tuple[str, int, str]:
    parts = rel.parts[:-1]
    if not parts:
        return "root", default_repeats, ""
    folder = Path(*parts).as_posix()
    for part in reversed(parts):
        match = _REPEAT.match(part)
        if match:
            return match.group(2), int(match.group(1)), folder
    return parts[-1], default_repeats, folder


def _parent_key(rel: str) -> str:
    parent = Path(rel).parent.as_posix()
    return "." if parent == "." else parent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
