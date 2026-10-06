"""Corrupt files, colour mode, alpha, EXIF orientation, and format mix."""

from __future__ import annotations

from collections import Counter

from loradatasetlinter.inventory import ORIENTATION_NAMES
from loradatasetlinter.models import Dataset, Finding
from loradatasetlinter.policy import Policy


def check_files(dataset: Dataset, policy: Policy) -> list[Finding]:
    cfg = policy.files
    if not cfg.enabled:
        return []
    findings: list[Finding] = []
    formats: Counter[str] = Counter()
    for image in dataset.images:
        if image.error or not image.readable:
            findings.append(
                Finding(
                    check="files",
                    code="corrupt_file",
                    severity=cfg.corrupt_severity,
                    reason=f"File is not a readable image: {image.error or 'unknown error'}.",
                    files=[image.rel],
                    details={"error": image.error or "unknown error"},
                )
            )
            continue
        formats[image.format or "UNKNOWN"] += 1
        if image.mode == "CMYK":
            findings.append(
                Finding(
                    check="files",
                    code="cmyk",
                    severity=cfg.cmyk_severity,
                    reason=(
                        "Image mode is CMYK. Loaders expect RGB and may drop a plate "
                        "or refuse the file."
                    ),
                    files=[image.rel],
                    details={"mode": image.mode},
                )
            )
        if image.has_alpha and image.alpha_nonopaque:
            findings.append(
                Finding(
                    check="files",
                    code="alpha_channel",
                    severity=cfg.alpha_severity,
                    reason=(
                        "Alpha channel has transparent pixels. They become black or "
                        "garbage depending on the loader."
                    ),
                    files=[image.rel],
                    details={"mode": image.mode},
                )
            )
        elif image.has_alpha:
            findings.append(
                Finding(
                    check="files",
                    code="opaque_alpha",
                    severity=cfg.opaque_alpha_severity,
                    reason=(
                        "An alpha channel is present and fully opaque. It is harmless "
                        "only if every loader composites it."
                    ),
                    files=[image.rel],
                    details={"mode": image.mode},
                )
            )
        if image.exif_orientation not in (None, 1):
            name = ORIENTATION_NAMES.get(image.exif_orientation, "unknown")
            findings.append(
                Finding(
                    check="files",
                    code="exif_orientation",
                    severity=cfg.exif_orientation_severity,
                    reason=(
                        f"EXIF orientation is {image.exif_orientation} ({name}). "
                        "Many trainers ignore EXIF and learn the unrotated pixels."
                    ),
                    files=[image.rel],
                    details={"orientation": image.exif_orientation, "name": name},
                )
            )
    if len(formats) > 1:
        summary = ", ".join(f"{name} {count}" for name, count in sorted(formats.items()))
        findings.append(
            Finding(
                check="files",
                code="format_mix",
                severity=cfg.format_mix_severity,
                reason=f"Dataset mixes formats: {summary}.",
                files=[image.rel for image in dataset.images if image.readable],
                details={"formats": dict(sorted(formats.items()))},
            )
        )
    return findings
