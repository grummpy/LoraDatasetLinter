"""Kohya-style aspect buckets.

The resolution list follows kohya-ss sd-scripts ``make_bucket_resolutions``:
buckets keep about ``base * base`` pixels, both sides are multiples of the step,
and each image is assigned to the predefined bucket with the closest aspect
ratio (exact size wins when the image is already on that list). This is the
path used when training is allowed to scale, which is the usual kohya default.
"""

from __future__ import annotations

import math


def make_bucket_resolutions(
    max_reso: tuple[int, int],
    min_size: int = 256,
    max_size: int = 1024,
    divisible: int = 64,
) -> list[tuple[int, int]]:
    """Return sorted ``(width, height)`` buckets that preserve roughly max_reso area."""
    if divisible < 1:
        raise ValueError("bucket step must be >= 1")
    if min_size < divisible or max_size < min_size:
        raise ValueError("bucket size bounds are inconsistent")
    max_width, max_height = max_reso
    if max_size < max_width or max_size < max_height:
        raise ValueError("max bucket size must be at least the base resolution")
    max_area = max_width * max_height
    resos: set[tuple[int, int]] = set()
    square = int(math.sqrt(max_area) // divisible) * divisible
    if square >= min_size:
        resos.add((square, square))
    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))
        width += divisible
    return sorted(resos)


def select_bucket(
    image_width: int,
    image_height: int,
    resolutions: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], float]:
    """Pick a bucket. Returns ``(bucket, resized_size, aspect_error)``."""
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if not resolutions:
        raise ValueError("no bucket resolutions")
    aspect = image_width / image_height
    exact = (image_width, image_height)
    if exact in set(resolutions):
        reso = exact
    else:
        # First minimum in sorted order matches numpy.argmin on that same list.
        reso = min(resolutions, key=lambda wh: abs((wh[0] / wh[1]) - aspect))
    bucket_aspect = reso[0] / reso[1]
    if aspect > bucket_aspect:
        scale = reso[1] / image_height
    else:
        scale = reso[0] / image_width
    resized = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
    error = bucket_aspect - aspect
    return reso, resized, error
