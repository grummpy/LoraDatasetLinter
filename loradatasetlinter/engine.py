"""Run every check and assemble a report. Reads images; writes nothing."""

from __future__ import annotations

from pathlib import Path

from loradatasetlinter.checks import run_checks
from loradatasetlinter.checks.aspect import assign_buckets
from loradatasetlinter.checks.stats import build_stats
from loradatasetlinter.errors import LintError
from loradatasetlinter.inventory import scan_folder
from loradatasetlinter.models import Finding, Report
from loradatasetlinter.policy import Policy
from loradatasetlinter.score import decide

_RANK = {"fail": 0, "warn": 1, "info": 2}


def scan_dataset(dataset: Path, policy: Policy) -> Report:
    root = dataset.expanduser().resolve()
    if not root.exists():
        raise LintError(f"Dataset folder not found: {root}")
    if not root.is_dir():
        raise LintError(f"Dataset path is not a folder: {root}")
    loaded = scan_folder(root, policy)
    findings: list[Finding] = []
    if not loaded.images:
        findings.append(
            Finding(
                check="stats",
                code="empty_dataset",
                severity="fail",
                reason="No images found. There is nothing to train.",
                files=[],
                details={},
            )
        )
    for image in loaded.images:
        image.buckets = assign_buckets(image, policy)
    findings.extend(run_checks(loaded, policy))
    findings.sort(
        key=lambda finding: (
            _RANK.get(finding.severity, 9),
            finding.check,
            finding.code,
            tuple(finding.files),
        )
    )
    stats = build_stats(loaded, policy)
    score = decide(findings, policy)
    codes: dict[str, list[str]] = {}
    for finding in findings:
        for rel in finding.files:
            codes.setdefault(rel, [])
            if finding.code not in codes[rel]:
                codes[rel].append(finding.code)
    images = [_image_row(image, codes.get(image.rel, [])) for image in loaded.images]
    return Report(
        dataset=str(root),
        policy=policy.to_dict(),
        findings=findings,
        stats=stats,
        score=score,
        images=images,
    )


def _image_row(image, issues: list[str]) -> dict:
    return {
        "file": image.rel,
        "width": image.width,
        "height": image.height,
        "oriented_width": image.oriented_width,
        "oriented_height": image.oriented_height,
        "format": image.format,
        "mode": image.mode,
        "class_name": image.class_name,
        "folder": image.folder,
        "repeats": image.repeats,
        "caption": image.caption_rel,
        "tokens": image.token_estimate,
        "buckets": image.buckets,
        "error": image.error,
        "issues": issues,
        "file_size": image.file_size,
        "upscale_factor": image.upscale_factor,
        "exif_orientation": image.exif_orientation,
    }
