"""Caption sidecars, triggers, tag drift, and a CLIP token estimate."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from loradatasetlinter.inventory import shared_caption_groups
from loradatasetlinter.models import Dataset, Finding, ImageRecord
from loradatasetlinter.policy import Policy
from loradatasetlinter.tokens import estimate_clip_tokens

_SPELL = {
    "grey": "gray",
    "colour": "color",
    "colours": "colors",
    "centre": "center",
    "centres": "centers",
    "artefact": "artifact",
    "artefacts": "artifacts",
    "favourite": "favorite",
    "favourites": "favorites",
    "honour": "honor",
    "theatre": "theater",
    "blonde": "blond",
    "whisky": "whiskey",
}
_MIN_EDIT_LENGTH = 5


def normalize_tag(tag: str) -> str:
    text = tag.casefold().replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(_SPELL.get(word, word) for word in text.split(" "))


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            cost = 0 if char_left == char_right else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def detect_separator(text: str) -> str | None:
    if text.count(";") >= 1 and text.count(";") >= text.count(","):
        return ";"
    if "," in text:
        return ","
    if "\n" in text.strip():
        return "newline"
    return None


def split_tags(text: str, separator: str | None) -> list[str]:
    if separator == ";":
        parts = text.split(";")
    elif separator == "newline":
        parts = text.splitlines()
    elif separator == ",":
        parts = text.split(",")
    else:
        stripped = text.strip()
        return [stripped] if stripped else []
    return [part.strip() for part in parts if part.strip()]


def trigger_present(text: str, tags: list[str], trigger: str) -> bool:
    folded = text.casefold()
    needle = trigger.casefold()
    if any(tag.casefold() == needle for tag in tags):
        return True
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", folded) is not None


def conflicting_orders(
    sequences: list[list[str]],
    min_images: int,
    minority_ratio: float,
) -> list[dict]:
    """Pairs that co-occur in both orders often enough to be drift."""
    counts: dict[tuple[str, str], int] = Counter()
    for tags in sequences:
        seen = []
        for tag in tags:
            key = tag.casefold()
            if key in seen:
                continue
            for earlier in seen:
                counts[(earlier, key)] += 1
            seen.append(key)
    conflicts = []
    visited: set[tuple[str, str]] = set()
    for left, right in list(counts):
        pair = tuple(sorted((left, right)))
        if pair in visited or left == right:
            continue
        visited.add(pair)
        a, b = pair
        ab = counts.get((a, b), 0)
        ba = counts.get((b, a), 0)
        total = ab + ba
        if ab == 0 or ba == 0 or total < min_images:
            continue
        minority = min(ab, ba) / total
        if minority + 1e-12 < minority_ratio:
            continue
        conflicts.append({"a": a, "b": b, "ab": ab, "ba": ba, "minority": round(minority, 4)})
    conflicts.sort(key=lambda item: (item["a"], item["b"]))
    return conflicts


def check_captions(dataset: Dataset, policy: Policy) -> list[Finding]:
    cfg = policy.captions
    if not cfg.enabled:
        return []
    findings: list[Finding] = []
    parsed: list[tuple[ImageRecord, list[str], str | None]] = []
    for image in dataset.images:
        if image.caption_rel is None:
            findings.append(
                Finding(
                    check="captions",
                    code="missing_caption",
                    severity=cfg.missing_severity,
                    reason="No .txt caption sits next to this image.",
                    files=[image.rel],
                    details={},
                )
            )
            continue
        text = image.caption_text or ""
        if not text.strip():
            findings.append(
                Finding(
                    check="captions",
                    code="empty_caption",
                    severity=cfg.empty_severity,
                    reason="Caption file is empty.",
                    files=[image.rel, image.caption_rel],
                    details={"caption": image.caption_rel},
                )
            )
            continue
        separator = detect_separator(text)
        tags = split_tags(text, separator)
        image.token_estimate = estimate_clip_tokens(text)
        parsed.append((image, tags, separator))
        if image.token_estimate > cfg.max_tokens:
            findings.append(
                Finding(
                    check="captions",
                    code="token_length",
                    severity=cfg.token_severity,
                    reason=(
                        f"Estimated CLIP token length is {image.token_estimate}, "
                        f"above the limit of {cfg.max_tokens}. "
                        "SD1.x clips the caption at 75 tokens. This count is an estimate, "
                        "not the official BPE tokenizer."
                    ),
                    files=[image.rel, image.caption_rel],
                    details={"tokens": image.token_estimate, "limit": cfg.max_tokens},
                )
            )
        for trigger in cfg.trigger_words:
            if trigger_present(text, tags, trigger):
                continue
            findings.append(
                Finding(
                    check="captions",
                    code="missing_trigger",
                    severity=cfg.trigger_severity,
                    reason=f"Caption is missing trigger word {trigger!r}.",
                    files=[image.rel, image.caption_rel],
                    details={"trigger": trigger},
                )
            )
    for rel in dataset.orphan_captions:
        findings.append(
            Finding(
                check="captions",
                code="orphan_caption",
                severity=cfg.orphan_severity,
                reason="Caption file has no matching image.",
                files=[rel],
                details={},
            )
        )
    for caption_rel, files in shared_caption_groups(dataset):
        findings.append(
            Finding(
                check="captions",
                code="shared_caption",
                severity=cfg.shared_caption_severity,
                reason="Several images share one caption file, so they cannot be edited apart.",
                files=[*files, caption_rel],
                details={"caption": caption_rel},
            )
        )
    findings.extend(_tag_findings(parsed, cfg))
    return findings


def _tag_findings(parsed, cfg) -> list[Finding]:
    if not parsed:
        return []
    tag_images: dict[str, set[str]] = defaultdict(set)
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    sequences: list[list[str]] = []
    separators: list[tuple[str, str]] = []
    for image, tags, separator in parsed:
        sequences.append(tags)
        if separator is not None and len(tags) > 1:
            separators.append((image.rel, separator))
        seen: set[str] = set()
        for tag in tags:
            key = normalize_tag(tag)
            if not key or key in seen:
                continue
            seen.add(key)
            tag_images[key].add(image.rel)
            surfaces[key][tag] += 1
    findings: list[Finding] = []
    rare = sorted(key for key, files in tag_images.items() if len(files) <= cfg.rare_max_images)
    if rare:
        files = sorted({rel for key in rare for rel in tag_images[key]})
        shown = ", ".join(rare[:20])
        extra = f" (+{len(rare) - 20} more)" if len(rare) > 20 else ""
        findings.append(
            Finding(
                check="captions",
                code="rare_tag",
                severity=cfg.rare_severity,
                reason=(
                    f"Tags that appear in at most {cfg.rare_max_images} image"
                    f"{'' if cfg.rare_max_images == 1 else 's'}: {shown}{extra}."
                ),
                files=files,
                details={"tags": [{"tag": key, "count": len(tag_images[key])} for key in rare]},
            )
        )
    if len(parsed) >= cfg.drift_min_dataset:
        low = sorted(
            key
            for key, files in tag_images.items()
            if cfg.rare_max_images < len(files) <= cfg.drift_max_images
        )
        if low:
            files = sorted({rel for key in low for rel in tag_images[key]})
            shown = ", ".join(low[:20])
            findings.append(
                Finding(
                    check="captions",
                    code="low_support_tag",
                    severity=cfg.drift_severity,
                    reason=(
                        "Caption-tag drift: these tags show up in only a few images "
                        f"({shown}). They may be one-off labels rather than a concept."
                    ),
                    files=files,
                    details={"tags": [{"tag": key, "count": len(tag_images[key])} for key in low]},
                )
            )
    synonym_groups = _synonym_groups(surfaces, cfg.synonym_max_distance)
    for group in synonym_groups:
        files = sorted({rel for key in group["keys"] for rel in tag_images[key]})
        shown = ", ".join(f"{name} ({count})" for name, count in group["surfaces"])
        findings.append(
            Finding(
                check="captions",
                code="near_synonym",
                severity=cfg.synonym_severity,
                reason=f"Near-synonym tags are both in use: {shown}.",
                files=files,
                details=group,
            )
        )
    conflicts = conflicting_orders(sequences, cfg.order_min_images, cfg.order_minority_ratio)
    if conflicts:
        findings.append(
            Finding(
                check="captions",
                code="tag_order",
                severity=cfg.order_severity,
                reason=(
                    "Tag order is inconsistent for "
                    + ", ".join(f"{item['a']!r} / {item['b']!r}" for item in conflicts[:12])
                    + "."
                ),
                files=sorted({image.rel for image, _tags, _sep in parsed}),
                details={"pairs": conflicts},
            )
        )
    if separators:
        counts = Counter(separator for _rel, separator in separators)
        dominant = sorted(counts, key=lambda name: (-counts[name], name))[0]
        odd = sorted(rel for rel, separator in separators if separator != dominant)
        if odd:
            findings.append(
                Finding(
                    check="captions",
                    code="separator_mix",
                    severity=cfg.separator_severity,
                    reason=(
                        f"Caption separators are mixed. The common separator is {dominant}; "
                        "these files use another one."
                    ),
                    files=odd,
                    details={"dominant": dominant, "counts": dict(counts)},
                )
            )
    return findings


def _synonym_groups(surfaces: dict[str, Counter[str]], max_distance: int) -> list[dict]:
    groups: list[dict] = []
    for key, forms in sorted(surfaces.items()):
        if len(forms) < 2:
            continue
        groups.append(
            {
                "keys": [key],
                "surfaces": sorted((name, count) for name, count in forms.items()),
                "kind": "spelling",
            }
        )
    keys = sorted(surfaces)
    used: set[tuple[str, str]] = set()
    for index, left in enumerate(keys):
        if len(left) < _MIN_EDIT_LENGTH:
            continue
        for right in keys[index + 1 :]:
            if len(right) < _MIN_EDIT_LENGTH or left[0] != right[0]:
                continue
            distance = levenshtein(left, right)
            if distance == 0 or distance > max_distance:
                continue
            pair = (left, right)
            if pair in used:
                continue
            used.add(pair)
            forms = []
            for key in (left, right):
                forms.extend((name, count) for name, count in surfaces[key].items())
            groups.append(
                {
                    "keys": [left, right],
                    "surfaces": sorted(forms),
                    "kind": "edit",
                    "distance": distance,
                }
            )
    return groups
