"""Aspect outliers and kohya bucket assignment."""

from __future__ import annotations

import math
import statistics

from loradatasetlinter.buckets import make_bucket_resolutions, select_bucket
from loradatasetlinter.models import Dataset, Finding, ImageRecord
from loradatasetlinter.policy import Policy


def is_aspect_outlier(aspect: float, median_aspect: float, ratio: float) -> bool:
    """True when ``aspect`` is at least ``ratio`` times the median, in log space."""
    if aspect <= 0 or median_aspect <= 0 or ratio < 1:
        return False
    distance = abs(math.log(aspect) - math.log(median_aspect))
    return distance >= math.log(ratio) - 1e-12


def is_extreme_aspect(aspect: float, limit: float) -> bool:
    if aspect <= 0:
        return False
    return max(aspect, 1.0 / aspect) >= limit - 1e-12


def assign_buckets(image: ImageRecord, policy: Policy) -> dict[str, list[int]]:
    if not image.readable or not image.oriented_width or not image.oriented_height:
        return {}
    assigned: dict[str, list[int]] = {}
    for base in policy.aspect.base_resolutions:
        resolutions = make_bucket_resolutions(
            (base, base),
            policy.aspect.min_bucket_size,
            policy.aspect.max_bucket_size,
            policy.aspect.bucket_step,
        )
        bucket, _resized, _error = select_bucket(
            image.oriented_width,
            image.oriented_height,
            resolutions,
        )
        assigned[str(base)] = [bucket[0], bucket[1]]
    return assigned


def check_aspect(dataset: Dataset, policy: Policy) -> list[Finding]:
    cfg = policy.aspect
    if not cfg.enabled:
        return []
    readable = [image for image in dataset.images if image.aspect]
    findings: list[Finding] = []
    for image in readable:
        aspect = float(image.aspect or 0)
        if is_extreme_aspect(aspect, cfg.extreme_aspect):
            findings.append(
                Finding(
                    check="aspect",
                    code="extreme_aspect",
                    severity=cfg.extreme_severity,
                    reason=(
                        f"Aspect ratio {_num(aspect)} exceeds {cfg.extreme_aspect:.2f}. "
                        "Kohya will crop it hard into the nearest bucket."
                    ),
                    files=[image.rel],
                    details={"aspect": round(aspect, 6), "limit": cfg.extreme_aspect},
                )
            )
    if len(readable) >= cfg.outlier_min_images:
        aspects = [float(image.aspect or 1) for image in readable]
        median = math.exp(statistics.median([math.log(value) for value in aspects]))
        for image, aspect in zip(readable, aspects, strict=True):
            if not is_aspect_outlier(aspect, median, cfg.outlier_ratio):
                continue
            findings.append(
                Finding(
                    check="aspect",
                    code="aspect_outlier",
                    severity=cfg.outlier_severity,
                    reason=(
                        f"Aspect ratio {_num(aspect)} is far from the folder median "
                        f"{_num(median)} (limit {cfg.outlier_ratio:.2f}×). "
                        "It will land in a different kohya bucket than the rest."
                    ),
                    files=[image.rel],
                    details={
                        "aspect": round(aspect, 6),
                        "median_aspect": round(median, 6),
                        "ratio": cfg.outlier_ratio,
                    },
                )
            )
    return findings


def _num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"
