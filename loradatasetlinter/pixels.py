"""Pixel measurements used by the resolution check. CPU and NumPy only."""

from __future__ import annotations

import numpy as np
from PIL import Image


def upscale_factor(image: Image.Image, min_fraction: float) -> int | None:
    """Return 4 or 2 when the image is a nearest-neighbor enlargement.

    A block is constant when every pixel inside it is identical. Smooth
    gradients and drawn edges fail that test. A flat image is ignored because
    every block is constant for a boring reason.
    """
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    if array.size == 0 or float(array.std()) < 4.0:
        return None
    for factor in (4, 2):
        if constant_block_fraction(array, factor) >= min_fraction:
            return factor
    return None


def constant_block_fraction(array: np.ndarray, factor: int) -> float:
    if factor < 2:
        return 0.0
    height, width = array.shape[:2]
    height = height - (height % factor)
    width = width - (width % factor)
    if height < factor * 4 or width < factor * 4:
        return 0.0
    cropped = array[:height, :width]
    blocks = cropped.reshape(height // factor, factor, width, array.shape[-1])
    blocks = blocks.reshape(height // factor, factor, width // factor, factor, array.shape[-1])
    spread = blocks.max(axis=(1, 3)) - blocks.min(axis=(1, 3))
    constant = np.all(spread == 0, axis=-1)
    return float(constant.mean())
