"""Exact file hashes, exact pixels, perceptual hashes, and optional CLIP."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from loradatasetlinter.cluster import cluster_by_cosine, cluster_by_distance, hamming
from loradatasetlinter.models import Dataset, Finding, ImageRecord
from loradatasetlinter.policy import Policy


class ClipUnavailable(Exception):
    """CLIP was requested and cannot run without a download or a GPU."""


def check_duplicates(dataset: Dataset, policy: Policy) -> list[Finding]:
    if not policy.duplicates.enabled:
        return []
    readable = [image for image in dataset.images if image.readable]
    findings: list[Finding] = []
    findings.extend(_exact_files(readable, policy))
    findings.extend(_exact_pixels(readable, policy))
    if policy.duplicates.perceptual:
        findings.extend(_perceptual(readable, policy))
    findings.extend(_clip(readable, policy))
    return findings


def _exact_files(images: list[ImageRecord], policy: Policy) -> list[Finding]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for image in images:
        groups[image.file_hash].append(image)
    findings = []
    for digest, group in sorted(groups.items(), key=lambda item: item[1][0].rel):
        if len(group) < 2:
            continue
        files = sorted(image.rel for image in group)
        findings.append(
            Finding(
                check="duplicates",
                code="exact_duplicate",
                severity=policy.duplicates.exact_severity,
                reason=(
                    f"These {len(files)} files are byte-for-byte identical "
                    f"(sha256 {digest[:12]}). Keeping one copy is enough."
                ),
                files=files,
                details={"sha256": digest, "keep": files[0]},
            )
        )
    return findings


def _exact_pixels(images: list[ImageRecord], policy: Policy) -> list[Finding]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for image in images:
        if image.pixel_hash:
            groups[image.pixel_hash].append(image)
    findings = []
    for digest, group in sorted(groups.items(), key=lambda item: item[1][0].rel):
        file_hashes = {image.file_hash for image in group}
        if len(group) < 2 or len(file_hashes) < 2:
            continue
        files = sorted(image.rel for image in group)
        findings.append(
            Finding(
                check="duplicates",
                code="exact_pixels",
                severity=policy.duplicates.pixel_severity,
                reason=(
                    "These files decode to the same pixels but are not the same bytes. "
                    "A re-save or a metadata change still trains the same image twice."
                ),
                files=files,
                details={"pixel_sha256": digest, "keep": files[0]},
            )
        )
    return findings


def _perceptual(images: list[ImageRecord], policy: Policy) -> list[Finding]:
    hashed = [image for image in images if image.perceptual_hash is not None]
    if len(hashed) < 2:
        return []
    ids = [image.rel for image in hashed]
    hashes = {image.rel: int(image.perceptual_hash) for image in hashed}
    pixel = {image.rel: image.pixel_hash for image in hashed}
    threshold = policy.duplicates.hamming_threshold
    clusters = cluster_by_distance(ids, hashes, threshold)
    findings = []
    method = policy.duplicates.hash_method
    for cluster in clusters:
        if _all_same_pixels(cluster, pixel):
            continue
        distances = [
            hamming(hashes[left], hashes[right])
            for index, left in enumerate(cluster)
            for right in cluster[index + 1 :]
        ]
        farthest = max(distances) if distances else 0
        nearest = min(distances) if distances else 0
        findings.append(
            Finding(
                check="duplicates",
                code="near_duplicate",
                severity=policy.duplicates.near_severity,
                reason=(
                    f"{method} similarity chain has Hamming distances from {nearest}"
                    f" to {farthest}; each link is at or below the threshold of {threshold}. "
                    "Pairs at opposite ends of the chain can exceed the threshold. "
                    "Near-copies spend steps on the same picture."
                ),
                files=cluster,
                details={
                    "grouping": "connected_similarity_chain",
                    "hash_method": method,
                    "hamming_threshold": threshold,
                    "min_distance": nearest,
                    "max_distance": farthest,
                    "hashes": {rel: f"{hashes[rel]:x}" for rel in cluster},
                },
            )
        )
    return findings


def _all_same_pixels(cluster: list[str], pixel: dict[str, str | None]) -> bool:
    values = {pixel[rel] for rel in cluster}
    return len(values) == 1 and None not in values


def _clip(images: list[ImageRecord], policy: Policy) -> list[Finding]:
    clip = policy.duplicates.clip
    if not clip.enabled:
        return []
    try:
        vectors = embed_clip(images, checkpoint=clip.checkpoint, model_name=clip.model)
    except ClipUnavailable as exc:
        return [
            Finding(
                check="duplicates",
                code="clip_unavailable",
                severity="info",
                reason=str(exc),
                files=[],
                details={"enabled": True},
            )
        ]
    ids = list(vectors)
    clusters = cluster_by_cosine(ids, vectors, clip.cosine_threshold)
    findings = []
    for cluster in clusters:
        findings.append(
            Finding(
                check="duplicates",
                code="clip_near_duplicate",
                severity=clip.severity,
                reason=(
                    "CLIP similarity chain has links at or above "
                    f"{clip.cosine_threshold:.3f}; pairs at opposite ends can be less similar. "
                    "The pictures are semantically close even when the perceptual hash stays apart."
                ),
                files=cluster,
                details={
                    "grouping": "connected_similarity_chain",
                    "cosine_threshold": clip.cosine_threshold,
                    "model": clip.model,
                },
            )
        )
    return findings


def embed_clip(
    images: list[ImageRecord],
    *,
    checkpoint: str | None,
    model_name: str,
) -> dict[str, np.ndarray]:
    """Embed images with open_clip on CPU.

    Weights are loaded only from ``checkpoint``. Nothing is downloaded, and the
    model is never moved onto a GPU.
    """
    if not checkpoint:
        raise ClipUnavailable(
            "CLIP is enabled but duplicates.clip.checkpoint is empty. "
            "Point it at a local open_clip weight file. Weights are not downloaded."
        )
    weight_path = Path(checkpoint)
    if not weight_path.is_file():
        raise ClipUnavailable(
            f"CLIP checkpoint not found: {weight_path}. The linter does not download weights."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise ClipUnavailable(
            "CLIP embeddings need an extra CPU install "
            "(torch from the CPU wheel index, plus open-clip-torch). "
            "They stay off unless duplicates.clip.enabled is true."
        ) from exc
    device = torch.device("cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=str(weight_path),
        device=device,
    )
    model.eval()
    vectors: dict[str, np.ndarray] = {}
    for image in images:
        if not image.readable:
            continue
        with Image.open(image.path) as handle:
            handle.load()
            frame = ImageOps.exif_transpose(handle).convert("RGB")
        tensor = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        vectors[image.rel] = features[0].detach().cpu().numpy().astype(np.float64)
    return vectors
