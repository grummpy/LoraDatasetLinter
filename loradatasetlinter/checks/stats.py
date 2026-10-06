"""Counts, repeat-times-epoch steps, and class balance."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from loradatasetlinter.buckets import make_bucket_resolutions, select_bucket
from loradatasetlinter.checks.captions import detect_separator, split_tags
from loradatasetlinter.models import Dataset, Finding, ImageRecord
from loradatasetlinter.policy import Policy
from loradatasetlinter.tokens import estimate_clip_tokens


def class_weights(dataset: Dataset) -> list[dict]:
    buckets: dict[str, dict] = {}
    for image in dataset.images:
        if not image.readable:
            continue
        key = image.folder or "root"
        row = buckets.setdefault(
            key,
            {
                "folder": image.folder,
                "class_name": image.class_name,
                "repeats": image.repeats,
                "images": 0,
            },
        )
        row["images"] += 1
        row["repeats"] = image.repeats
    rows = []
    for row in buckets.values():
        weight = row["images"] * row["repeats"]
        rows.append({**row, "weight": weight})
    rows.sort(key=lambda item: (item["folder"], item["class_name"]))
    total = sum(row["weight"] for row in rows)
    for row in rows:
        row["share"] = (row["weight"] / total) if total else 0.0
    return rows


def step_estimate(dataset: Dataset, policy: Policy) -> dict:
    rows = class_weights(dataset)
    weighted = sum(row["weight"] for row in rows)
    batch = policy.stats.batch_size
    epochs = policy.stats.epochs
    steps_per_epoch = math.ceil(weighted / batch) if weighted else 0
    naive = (weighted * epochs / batch) if batch else 0.0
    return {
        "epochs": epochs,
        "batch_size": batch,
        "weighted_images": weighted,
        "steps_per_epoch": steps_per_epoch,
        "steps": steps_per_epoch * epochs,
        "naive_steps": naive,
        "formula": "ceil(sum(images * repeats) / batch_size) * epochs",
        "classes": rows,
    }


def is_imbalanced(max_weight: float, min_weight: float, ratio: float) -> bool:
    if max_weight <= 0:
        return False
    if min_weight <= 0:
        return True
    return (max_weight / min_weight) >= ratio - 1e-12


def check_stats(dataset: Dataset, policy: Policy) -> list[Finding]:
    if not policy.stats.enabled:
        return []
    findings: list[Finding] = []
    rows = class_weights(dataset)
    zero = [row for row in rows if row["repeats"] == 0 and row["images"] > 0]
    if zero:
        files = [image.rel for image in dataset.images if image.readable and image.repeats == 0]
        names = ", ".join(row["folder"] or "root" for row in zero)
        findings.append(
            Finding(
                check="stats",
                code="zero_repeats",
                severity=policy.stats.zero_repeats_severity,
                reason=f"Repeat count is 0 in {names}, so these images add no training steps.",
                files=sorted(files),
                details={"folders": [row["folder"] for row in zero]},
            )
        )
    positive = [row for row in rows if row["images"] > 0]
    if len(positive) >= 2:
        weights = [row["weight"] for row in positive]
        high = max(weights)
        low = min(weights)
        if is_imbalanced(high, low, policy.stats.balance_ratio):
            heavy = max(positive, key=lambda row: row["weight"])
            light = min(positive, key=lambda row: row["weight"])
            ratio = float("inf") if low == 0 else high / low
            files = [
                image.rel
                for image in dataset.images
                if image.readable and image.folder in {heavy["folder"], light["folder"]}
            ]
            findings.append(
                Finding(
                    check="stats",
                    code="class_imbalance",
                    severity=policy.stats.balance_severity,
                    reason=(
                        "Class weights differ by "
                        f"{_num(ratio)}× "
                        f"({_label(heavy)} {heavy['weight']}, {_label(light)} {light['weight']}), "
                        f"at or above the limit of {policy.stats.balance_ratio:.2f}×. "
                        "Weight is images times the folder repeat count, so the heavier "
                        "class takes more of the step budget."
                    ),
                    files=sorted(files),
                    details={
                        "max_weight": high,
                        "min_weight": low,
                        "ratio": None if low == 0 else round(high / low, 6),
                        "limit": policy.stats.balance_ratio,
                    },
                )
            )
    return findings


def build_stats(dataset: Dataset, policy: Policy) -> dict:
    readable = [image for image in dataset.images if image.readable]
    formats = Counter((image.format or "UNKNOWN") for image in readable)
    estimate = step_estimate(dataset, policy)
    buckets = _bucket_histogram(readable, policy)
    tag_counts: Counter[str] = Counter()
    token_values: list[int] = []
    for image in dataset.images:
        text = (image.caption_text or "").strip()
        if not text:
            continue
        tokens = image.token_estimate
        if tokens is None:
            tokens = estimate_clip_tokens(text)
            image.token_estimate = tokens
        token_values.append(tokens)
        tags = split_tags(text, detect_separator(text))
        tag_counts.update(list(dict.fromkeys(tag.casefold() for tag in tags)))
    return {
        "images": len(dataset.images),
        "readable_images": len(readable),
        "unreadable": sum(1 for image in dataset.images if not image.readable),
        "captions": sum(1 for image in dataset.images if image.caption_rel),
        "orphan_captions": len(dataset.orphan_captions),
        "other_files": len(dataset.other_files),
        "skipped_symlinks": len(dataset.skipped_symlinks),
        "formats": dict(sorted(formats.items())),
        "steps": estimate,
        "buckets": buckets,
        "tag_frequency": [
            {"tag": tag, "count": count}
            for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "token_lengths": _token_summary(token_values, policy.captions.max_tokens),
    }


def _bucket_histogram(images: list[ImageRecord], policy: Policy) -> dict:
    histograms: dict[str, list[dict]] = {}
    for base in policy.aspect.base_resolutions:
        resolutions = make_bucket_resolutions(
            (base, base),
            policy.aspect.min_bucket_size,
            policy.aspect.max_bucket_size,
            policy.aspect.bucket_step,
        )
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for image in images:
            if not image.oriented_width or not image.oriented_height:
                continue
            bucket, _resized, _error = select_bucket(
                image.oriented_width,
                image.oriented_height,
                resolutions,
            )
            counts[bucket] += 1
        rows = [
            {"resolution": [width, height], "count": count}
            for (width, height), count in sorted(counts.items())
        ]
        histograms[str(base)] = rows
    return histograms


def _token_summary(values: list[int], limit: int) -> dict:
    if not values:
        return {"min": 0, "median": 0, "max": 0, "over_limit": 0, "limit": limit}
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2]
    if len(ordered) % 2 == 0:
        mid = (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    return {
        "min": ordered[0],
        "median": mid,
        "max": ordered[-1],
        "over_limit": sum(1 for value in values if value > limit),
        "limit": limit,
    }


def _label(row: dict) -> str:
    folder = row["folder"] or "root"
    return f"{folder} ({row['class_name']})"


def _num(value: float) -> str:
    if math.isinf(value):
        return "infinite"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"
