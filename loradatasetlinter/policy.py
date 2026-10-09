"""YAML policy load, merge, and CLI overrides."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from loradatasetlinter.errors import PolicyError
from loradatasetlinter.models import SEVERITIES

_DEFAULT_PATH = Path(__file__).with_name("default_policy.yaml")


@dataclass
class ResolutionPolicy:
    enabled: bool = True
    min_side: int = 512
    min_side_severity: str = "fail"
    upscale_artifacts: bool = True
    upscale_block_fraction: float = 0.99
    upscale_severity: str = "warn"
    median_outlier_ratio: float = 2.0
    median_min_images: int = 4
    outlier_severity: str = "warn"


@dataclass
class AspectPolicy:
    enabled: bool = True
    outlier_ratio: float = 1.75
    outlier_min_images: int = 4
    outlier_severity: str = "warn"
    extreme_aspect: float = 3.0
    extreme_severity: str = "warn"
    base_resolutions: list[int] = field(default_factory=lambda: [512, 768, 1024])
    min_bucket_size: int = 256
    max_bucket_size: int = 2048
    bucket_step: int = 64


@dataclass
class ClipPolicy:
    enabled: bool = False
    cosine_threshold: float = 0.95
    checkpoint: str | None = None
    model: str = "ViT-B-32"
    severity: str = "warn"


@dataclass
class DuplicatePolicy:
    enabled: bool = True
    exact_severity: str = "fail"
    pixel_severity: str = "fail"
    perceptual: bool = True
    hash_method: str = "phash"
    hash_size: int = 8
    hamming_threshold: int = 8
    near_severity: str = "warn"
    clip: ClipPolicy = field(default_factory=ClipPolicy)


@dataclass
class FilePolicy:
    enabled: bool = True
    corrupt_severity: str = "fail"
    cmyk_severity: str = "warn"
    alpha_severity: str = "warn"
    opaque_alpha_severity: str = "info"
    exif_orientation_severity: str = "warn"
    format_mix_severity: str = "info"


@dataclass
class CaptionPolicy:
    enabled: bool = True
    missing_severity: str = "fail"
    empty_severity: str = "fail"
    orphan_severity: str = "warn"
    shared_caption_severity: str = "warn"
    trigger_words: list[str] = field(default_factory=list)
    trigger_severity: str = "warn"
    rare_max_images: int = 1
    rare_severity: str = "info"
    drift_max_images: int = 2
    drift_min_dataset: int = 6
    drift_severity: str = "info"
    synonym_severity: str = "warn"
    synonym_max_distance: int = 1
    order_min_images: int = 3
    order_minority_ratio: float = 0.2
    order_severity: str = "info"
    separator_severity: str = "warn"
    max_tokens: int = 75
    token_severity: str = "warn"


@dataclass
class StatsPolicy:
    enabled: bool = True
    epochs: int = 10
    batch_size: int = 1
    default_repeats: int = 1
    balance_ratio: float = 3.0
    balance_severity: str = "warn"
    zero_repeats_severity: str = "warn"


@dataclass
class ScorePolicy:
    weights: dict[str, int] = field(default_factory=lambda: {"fail": 20, "warn": 6, "info": 1})
    pass_min: int = 90
    warn_min: int = 70


@dataclass
class OutputPolicy:
    thumbnail_px: int = 160


@dataclass
class Policy:
    version: int = 1
    resolution: ResolutionPolicy = field(default_factory=ResolutionPolicy)
    aspect: AspectPolicy = field(default_factory=AspectPolicy)
    duplicates: DuplicatePolicy = field(default_factory=DuplicatePolicy)
    files: FilePolicy = field(default_factory=FilePolicy)
    captions: CaptionPolicy = field(default_factory=CaptionPolicy)
    stats: StatsPolicy = field(default_factory=StatsPolicy)
    score: ScorePolicy = field(default_factory=ScorePolicy)
    output: OutputPolicy = field(default_factory=OutputPolicy)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "resolution": _public(self.resolution),
            "aspect": _public(self.aspect),
            "duplicates": _public(self.duplicates),
            "files": _public(self.files),
            "captions": _public(self.captions),
            "stats": _public(self.stats),
            "score": _public(self.score),
            "output": _public(self.output),
        }


def _public(obj: object) -> dict:
    raw = dict(obj.__dict__)
    for key, value in list(raw.items()):
        if hasattr(value, "__dict__") and not isinstance(value, str | Path):
            raw[key] = _public(value)
    return raw


def load_policy(path: Path | None = None) -> Policy:
    """Load defaults, then deep-merge an optional user file."""
    base = _read_yaml(_DEFAULT_PATH)
    if path is not None:
        if not path.is_file():
            raise PolicyError(f"Policy file not found: {path}")
        override = _read_yaml(path)
        base = deep_merge(base, override)
    return policy_from_dict(base)


def deep_merge(base: dict, override: dict, prefix: str = "") -> dict:
    merged = deepcopy(base)
    if not isinstance(override, dict):
        raise PolicyError(f"{prefix or 'policy'} must be a mapping")
    for key, value in override.items():
        label = f"{prefix}{key}"
        if key not in merged:
            raise PolicyError(f"Unknown policy key: {label}")
        if isinstance(merged[key], dict):
            if not isinstance(value, dict):
                raise PolicyError(f"{label} must be a mapping")
            merged[key] = deep_merge(merged[key], value, prefix=f"{label}.")
        else:
            merged[key] = value
    return merged


def policy_from_dict(data: dict) -> Policy:
    if not isinstance(data, dict):
        raise PolicyError("policy must be a mapping")
    version = _int(data, "version", minimum=1)
    if version != 1:
        raise PolicyError(f"Unsupported policy version: {version}")
    resolution = _section(data, "resolution")
    aspect = _section(data, "aspect")
    duplicates = _section(data, "duplicates")
    files = _section(data, "files")
    captions = _section(data, "captions")
    stats = _section(data, "stats")
    score = _section(data, "score")
    output = _section(data, "output")
    clip = _section(duplicates, "clip", parent="duplicates")
    weights = _section(score, "weights", parent="score")
    policy = Policy(
        version=version,
        resolution=ResolutionPolicy(
            enabled=_bool(resolution, "enabled"),
            min_side=_int(resolution, "min_side", minimum=1),
            min_side_severity=_severity(resolution, "min_side_severity"),
            upscale_artifacts=_bool(resolution, "upscale_artifacts"),
            upscale_block_fraction=_float(resolution, "upscale_block_fraction", low=0.0, high=1.0),
            upscale_severity=_severity(resolution, "upscale_severity"),
            median_outlier_ratio=_float(resolution, "median_outlier_ratio", low=1.0),
            median_min_images=_int(resolution, "median_min_images", minimum=1),
            outlier_severity=_severity(resolution, "outlier_severity"),
        ),
        aspect=AspectPolicy(
            enabled=_bool(aspect, "enabled"),
            outlier_ratio=_float(aspect, "outlier_ratio", low=1.0),
            outlier_min_images=_int(aspect, "outlier_min_images", minimum=1),
            outlier_severity=_severity(aspect, "outlier_severity"),
            extreme_aspect=_float(aspect, "extreme_aspect", low=1.0),
            extreme_severity=_severity(aspect, "extreme_severity"),
            base_resolutions=_int_list(aspect, "base_resolutions", minimum=64),
            min_bucket_size=_int(aspect, "min_bucket_size", minimum=64),
            max_bucket_size=_int(aspect, "max_bucket_size", minimum=64),
            bucket_step=_int(aspect, "bucket_step", minimum=1),
        ),
        duplicates=DuplicatePolicy(
            enabled=_bool(duplicates, "enabled"),
            exact_severity=_severity(duplicates, "exact_severity"),
            pixel_severity=_severity(duplicates, "pixel_severity"),
            perceptual=_bool(duplicates, "perceptual"),
            hash_method=_choice(duplicates, "hash_method", {"phash", "dhash"}),
            hash_size=_int(duplicates, "hash_size", minimum=4),
            hamming_threshold=_int(duplicates, "hamming_threshold", minimum=0),
            near_severity=_severity(duplicates, "near_severity"),
            clip=ClipPolicy(
                enabled=_bool(clip, "enabled"),
                cosine_threshold=_float(clip, "cosine_threshold", low=0.0, high=1.0),
                checkpoint=_optional_str(clip, "checkpoint"),
                model=_str(clip, "model"),
                severity=_severity(clip, "severity"),
            ),
        ),
        files=FilePolicy(
            enabled=_bool(files, "enabled"),
            corrupt_severity=_severity(files, "corrupt_severity"),
            cmyk_severity=_severity(files, "cmyk_severity"),
            alpha_severity=_severity(files, "alpha_severity"),
            opaque_alpha_severity=_severity(files, "opaque_alpha_severity"),
            exif_orientation_severity=_severity(files, "exif_orientation_severity"),
            format_mix_severity=_severity(files, "format_mix_severity"),
        ),
        captions=CaptionPolicy(
            enabled=_bool(captions, "enabled"),
            missing_severity=_severity(captions, "missing_severity"),
            empty_severity=_severity(captions, "empty_severity"),
            orphan_severity=_severity(captions, "orphan_severity"),
            shared_caption_severity=_severity(captions, "shared_caption_severity"),
            trigger_words=_str_list(captions, "trigger_words"),
            trigger_severity=_severity(captions, "trigger_severity"),
            rare_max_images=_int(captions, "rare_max_images", minimum=0),
            rare_severity=_severity(captions, "rare_severity"),
            drift_max_images=_int(captions, "drift_max_images", minimum=0),
            drift_min_dataset=_int(captions, "drift_min_dataset", minimum=1),
            drift_severity=_severity(captions, "drift_severity"),
            synonym_severity=_severity(captions, "synonym_severity"),
            synonym_max_distance=_int(captions, "synonym_max_distance", minimum=0),
            order_min_images=_int(captions, "order_min_images", minimum=1),
            order_minority_ratio=_float(captions, "order_minority_ratio", low=0.0, high=1.0),
            order_severity=_severity(captions, "order_severity"),
            separator_severity=_severity(captions, "separator_severity"),
            max_tokens=_int(captions, "max_tokens", minimum=1),
            token_severity=_severity(captions, "token_severity"),
        ),
        stats=StatsPolicy(
            enabled=_bool(stats, "enabled"),
            epochs=_int(stats, "epochs", minimum=1),
            batch_size=_int(stats, "batch_size", minimum=1),
            default_repeats=_int(stats, "default_repeats", minimum=0),
            balance_ratio=_float(stats, "balance_ratio", low=1.0),
            balance_severity=_severity(stats, "balance_severity"),
            zero_repeats_severity=_severity(stats, "zero_repeats_severity"),
        ),
        score=ScorePolicy(
            weights={
                "fail": _int(weights, "fail", minimum=0),
                "warn": _int(weights, "warn", minimum=0),
                "info": _int(weights, "info", minimum=0),
            },
            pass_min=_int(score, "pass_min", minimum=0, maximum=100),
            warn_min=_int(score, "warn_min", minimum=0, maximum=100),
        ),
        output=OutputPolicy(thumbnail_px=_int(output, "thumbnail_px", minimum=0)),
    )
    _validate_cross(policy)
    _reject_unknown(
        score.get("weights", {}),
        {"fail", "warn", "info"},
        "score.weights",
    )
    return policy


def apply_overrides(
    policy: Policy,
    *,
    min_side: int | None = None,
    hamming: int | None = None,
    base_resolution: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    trigger: str | None = None,
    hash_method: str | None = None,
    thumbnail_px: int | None = None,
) -> Policy:
    if min_side is not None:
        if min_side < 1:
            raise PolicyError("--min-side must be >= 1")
        policy.resolution.min_side = min_side
    if hamming is not None:
        if hamming < 0:
            raise PolicyError("--hamming must be >= 0")
        policy.duplicates.hamming_threshold = hamming
    if base_resolution is not None:
        if base_resolution < 64:
            raise PolicyError("--base-resolution must be >= 64")
        policy.aspect.base_resolutions = [base_resolution]
    if epochs is not None:
        if epochs < 1:
            raise PolicyError("--epochs must be >= 1")
        policy.stats.epochs = epochs
    if batch_size is not None:
        if batch_size < 1:
            raise PolicyError("--batch-size must be >= 1")
        policy.stats.batch_size = batch_size
    if trigger is not None:
        policy.captions.trigger_words = [
            part.strip() for part in trigger.split(",") if part.strip()
        ]
    if hash_method is not None:
        if hash_method not in {"phash", "dhash"}:
            raise PolicyError("--hash-method must be phash or dhash")
        policy.duplicates.hash_method = hash_method
    if thumbnail_px is not None:
        if thumbnail_px < 0:
            raise PolicyError("--thumbnail-px must be >= 0")
        policy.output.thumbnail_px = thumbnail_px
    _validate_cross(policy)
    return policy


def _validate_cross(policy: Policy) -> None:
    aspect = policy.aspect
    if aspect.max_bucket_size < aspect.min_bucket_size:
        raise PolicyError("aspect.max_bucket_size must be >= aspect.min_bucket_size")
    if aspect.min_bucket_size % aspect.bucket_step or aspect.max_bucket_size % aspect.bucket_step:
        raise PolicyError("aspect bucket sizes must be multiples of aspect.bucket_step")
    for base in aspect.base_resolutions:
        if base < aspect.min_bucket_size:
            raise PolicyError(
                f"aspect.base_resolution {base} is smaller than aspect.min_bucket_size"
            )
        if aspect.max_bucket_size < base:
            raise PolicyError(
                f"aspect.max_bucket_size ({aspect.max_bucket_size}) is smaller than "
                f"base resolution {base}"
            )
    if policy.score.warn_min > policy.score.pass_min:
        raise PolicyError("score.warn_min must be <= score.pass_min")
    if policy.captions.drift_max_images < policy.captions.rare_max_images:
        raise PolicyError("captions.drift_max_images must be >= captions.rare_max_images")


def _read_yaml(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise PolicyError(f"{path} must contain a YAML mapping")
    return loaded


def _section(data: dict, key: str, parent: str = "") -> dict:
    label = f"{parent}.{key}" if parent else key
    if key not in data:
        raise PolicyError(f"Missing policy section: {label}")
    value = data[key]
    if not isinstance(value, dict):
        raise PolicyError(f"{label} must be a mapping")
    return value


def _reject_unknown(data: dict, allowed: set[str], label: str) -> None:
    extra = sorted(set(data) - allowed)
    if extra:
        raise PolicyError(f"Unknown policy key: {label}.{extra[0]}")


def _severity(data: dict, key: str) -> str:
    value = _str(data, key)
    if value not in SEVERITIES:
        raise PolicyError(f"{key} must be one of {', '.join(SEVERITIES)}")
    return value


def _choice(data: dict, key: str, allowed: set[str]) -> str:
    value = _str(data, key)
    if value not in allowed:
        raise PolicyError(f"{key} must be one of {', '.join(sorted(allowed))}")
    return value


def _str(data: dict, key: str) -> str:
    if key not in data or not isinstance(data[key], str) or not data[key].strip():
        raise PolicyError(f"{key} must be a non-empty string")
    return data[key]


def _optional_str(data: dict, key: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    if not isinstance(data[key], str) or not data[key].strip():
        raise PolicyError(f"{key} must be a path string or null")
    return data[key]


def _bool(data: dict, key: str) -> bool:
    if key not in data or not isinstance(data[key], bool):
        raise PolicyError(f"{key} must be true or false")
    return data[key]


def _int(data: dict, key: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if key not in data or isinstance(data[key], bool) or not isinstance(data[key], int):
        raise PolicyError(f"{key} must be an integer")
    value = data[key]
    if minimum is not None and value < minimum:
        raise PolicyError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise PolicyError(f"{key} must be <= {maximum}")
    return value


def _float(
    data: dict,
    key: str,
    *,
    low: float | None = None,
    high: float | None = None,
) -> float:
    if key not in data or isinstance(data[key], bool) or not isinstance(data[key], int | float):
        raise PolicyError(f"{key} must be a number")
    value = float(data[key])
    if not math.isfinite(value):
        raise PolicyError(f"{key} must be a finite number")
    if low is not None and value < low:
        raise PolicyError(f"{key} must be >= {low}")
    if high is not None and value > high:
        raise PolicyError(f"{key} must be <= {high}")
    return value


def _int_list(data: dict, key: str, *, minimum: int) -> list[int]:
    if key not in data or not isinstance(data[key], list) or not data[key]:
        raise PolicyError(f"{key} must be a non-empty list")
    values: list[int] = []
    for item in data[key]:
        if isinstance(item, bool) or not isinstance(item, int):
            raise PolicyError(f"{key} must contain integers")
        if item < minimum:
            raise PolicyError(f"{key} values must be >= {minimum}")
        values.append(item)
    return values


def _str_list(data: dict, key: str) -> list[str]:
    if key not in data or not isinstance(data[key], list):
        raise PolicyError(f"{key} must be a list")
    values: list[str] = []
    for item in data[key]:
        if not isinstance(item, str) or not item.strip():
            raise PolicyError(f"{key} must contain non-empty strings")
        values.append(item)
    return values
