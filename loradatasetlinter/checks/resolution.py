"""Minimum side, nearest-neighbor upscale hints, and median outliers."""

from __future__ import annotations

import statistics

from loradatasetlinter.models import Dataset, Finding
from loradatasetlinter.policy import Policy


def is_low_resolution(min_side: int, limit: int) -> bool:
    """True when the shortest side is strictly below the minimum."""
    return min_side < limit


def is_median_outlier(value: float, median: float, ratio: float) -> bool:
    """True at or beyond ``median / ratio`` and ``median * ratio``."""
    if median <= 0 or ratio < 1:
        return False
    return value <= (median / ratio) or value >= (median * ratio)


def check_resolution(dataset: Dataset, policy: Policy) -> list[Finding]:
    cfg = policy.resolution
    if not cfg.enabled:
        return []
    readable = [image for image in dataset.images if image.readable and image.min_side is not None]
    findings: list[Finding] = []
    for image in readable:
        side = int(image.min_side or 0)
        if is_low_resolution(side, cfg.min_side):
            findings.append(
                Finding(
                    check="resolution",
                    code="low_resolution",
                    severity=cfg.min_side_severity,
                    reason=(
                        f"Shortest side is {side} px, below the minimum of {cfg.min_side} px. "
                        "The trainer will upscale it and spend steps on a soft result."
                    ),
                    files=[image.rel],
                    details={
                        "min_side": side,
                        "limit": cfg.min_side,
                        "width": image.oriented_width,
                        "height": image.oriented_height,
                    },
                )
            )
        if cfg.upscale_artifacts and image.upscale_factor:
            findings.append(
                Finding(
                    check="resolution",
                    code="upscale_artifact",
                    severity=cfg.upscale_severity,
                    reason=(
                        f"Nearest-neighbor upscale pattern at {image.upscale_factor}×. "
                        "Flat blocks like this usually mean a small image was enlarged "
                        "without new detail."
                    ),
                    files=[image.rel],
                    details={"factor": image.upscale_factor},
                )
            )
    if len(readable) >= cfg.median_min_images:
        median = float(statistics.median([int(image.min_side or 0) for image in readable]))
        for image in readable:
            side = float(image.min_side or 0)
            if not is_median_outlier(side, median, cfg.median_outlier_ratio):
                continue
            findings.append(
                Finding(
                    check="resolution",
                    code="resolution_outlier",
                    severity=cfg.outlier_severity,
                    reason=(
                        f"Shortest side is {int(side)} px against a folder median of "
                        f"{_num(median)} px (limit {cfg.median_outlier_ratio:.2f}×). "
                        "It will not share a scale with the rest of the set."
                    ),
                    files=[image.rel],
                    details={
                        "min_side": side,
                        "median": median,
                        "ratio": cfg.median_outlier_ratio,
                    },
                )
            )
    return findings


def _num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"
