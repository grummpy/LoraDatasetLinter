"""Shared records for a lint run. Nothing here touches the filesystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SEVERITIES = ("fail", "warn", "info")


@dataclass
class Finding:
    check: str
    code: str
    severity: str
    reason: str
    files: list[str]
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "code": self.code,
            "severity": self.severity,
            "reason": self.reason,
            "files": list(self.files),
            "details": self.details,
        }


@dataclass
class ImageRecord:
    rel: str
    path: Path
    file_hash: str
    file_size: int
    error: str | None = None
    pixel_hash: str | None = None
    perceptual_hash: int | None = None
    perceptual_hex: str | None = None
    width: int | None = None
    height: int | None = None
    oriented_width: int | None = None
    oriented_height: int | None = None
    mode: str | None = None
    format: str | None = None
    has_alpha: bool = False
    alpha_nonopaque: bool = False
    exif_orientation: int | None = None
    upscale_factor: int | None = None
    caption_rel: str | None = None
    caption_text: str | None = None
    repeats: int = 1
    class_name: str = "root"
    folder: str = ""
    buckets: dict[str, list[int]] = field(default_factory=dict)
    token_estimate: int | None = None

    @property
    def readable(self) -> bool:
        return (
            self.error is None
            and self.oriented_width is not None
            and self.oriented_height is not None
        )

    @property
    def min_side(self) -> int | None:
        if not self.readable:
            return None
        return min(self.oriented_width or 0, self.oriented_height or 0)

    @property
    def aspect(self) -> float | None:
        if not self.readable or not self.oriented_height:
            return None
        return (self.oriented_width or 0) / self.oriented_height


@dataclass
class Dataset:
    root: Path
    images: list[ImageRecord]
    orphan_captions: list[str]
    other_files: list[str]
    skipped_symlinks: list[str]


@dataclass
class Score:
    value: int
    status: str
    counts: dict[str, int]

    def to_dict(self) -> dict:
        return {"value": self.value, "status": self.status, "counts": dict(self.counts)}


@dataclass
class Report:
    dataset: str
    policy: dict
    findings: list[Finding]
    stats: dict
    score: Score
    images: list[dict]

    def to_dict(self) -> dict:
        return {
            "tool": "lora-dataset-linter",
            "version": _version(),
            "dataset": self.dataset,
            "status": self.score.status,
            "score": self.score.value,
            "counts": dict(self.score.counts),
            "policy": self.policy,
            "stats": self.stats,
            "findings": [finding.to_dict() for finding in self.findings],
            "images": self.images,
        }


def _version() -> str:
    from loradatasetlinter import __version__

    return __version__
