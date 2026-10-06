import math

from loradatasetlinter.buckets import make_bucket_resolutions, select_bucket


def test_bucket_list_is_step_aligned_and_contains_square():
    resos = make_bucket_resolutions((512, 512), 256, 2048, 64)
    assert (512, 512) in resos
    assert (256, 1024) in resos
    assert (1024, 256) in resos
    assert resos == sorted(set(resos))
    assert all(width % 64 == 0 and height % 64 == 0 for width, height in resos)
    areas = [width * height for width, height in resos]
    assert max(areas) <= 512 * 2048
    assert min(areas) >= 256 * 256


def test_square_image_uses_the_square_bucket():
    resos = make_bucket_resolutions((512, 512), 256, 2048, 64)
    bucket, _resized, error = select_bucket(512, 512, resos)
    assert bucket == (512, 512)
    assert error == 0


def test_wide_image_picks_the_closest_aspect():
    resos = make_bucket_resolutions((512, 512), 256, 2048, 64)
    bucket, _resized, _error = select_bucket(1000, 500, resos)
    aspect = 2.0
    expected = min(resos, key=lambda wh: abs((wh[0] / wh[1]) - aspect))
    assert bucket == expected
    assert bucket != (512, 512)


def test_each_standard_base_has_its_own_square():
    for base in (512, 768, 1024):
        resos = make_bucket_resolutions((base, base), 256, 2048, 64)
        square = int(math.sqrt(base * base) // 64) * 64
        assert (square, square) in resos
        bucket, _, _ = select_bucket(base, base, resos)
        assert bucket == (square, square)
