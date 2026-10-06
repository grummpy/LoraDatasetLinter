"""Threshold clustering for perceptual hashes and optional CLIP vectors."""

from __future__ import annotations

import numpy as np


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def cluster_by_distance(
    ids: list[str],
    hashes: dict[str, int],
    threshold: int,
) -> list[list[str]]:
    """Group ids whose hash distance is at or below ``threshold``."""
    return _components(ids, lambda a, b: hamming(hashes[a], hashes[b]) <= threshold)


def cluster_by_cosine(
    ids: list[str],
    vectors: dict[str, np.ndarray],
    threshold: float,
) -> list[list[str]]:
    """Group ids whose cosine similarity is at or above ``threshold``.

    Vectors must already be L2-normalized.
    """
    return _components(ids, lambda a, b: float(np.dot(vectors[a], vectors[b])) >= threshold)


def _components(ids: list[str], close) -> list[list[str]]:
    parent = {item: item for item in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            if close(left, right):
                union(left, right)
    groups: dict[str, list[str]] = {}
    for item in ids:
        groups.setdefault(find(item), []).append(item)
    clusters = [sorted(group) for group in groups.values() if len(group) >= 2]
    clusters.sort(key=lambda group: group[0])
    return clusters
