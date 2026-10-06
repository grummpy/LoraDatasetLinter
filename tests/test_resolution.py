from pathlib import Path

from loradatasetlinter.checks.resolution import is_low_resolution, is_median_outlier
from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.pixels import constant_block_fraction
from tests.fixtures import caption, finding_codes, gradient, nearest_upscale, solid


def test_min_side_boundary_is_exclusive():
    assert is_low_resolution(511, 512) is True
    assert is_low_resolution(512, 512) is False
    assert is_low_resolution(63, 64) is True
    assert is_low_resolution(64, 64) is False


def test_median_outlier_includes_the_ratio_boundary():
    assert is_median_outlier(50, 100, 2.0) is True
    assert is_median_outlier(51, 100, 2.0) is False
    assert is_median_outlier(200, 100, 2.0) is True
    assert is_median_outlier(199, 100, 2.0) is False


def test_low_resolution_finding_uses_policy_minimum(tmp_path: Path, policy):
    policy.resolution.min_side = 64
    ok = gradient(tmp_path / "ok.png", size=(64, 80))
    low = gradient(tmp_path / "low.png", size=(63, 80))
    caption(ok, "red square, studio lighting")
    caption(low, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    lows = [item for item in report.findings if item.code == "low_resolution"]
    assert [item.files for item in lows] == [["low.png"]]
    assert lows[0].severity == "fail"


def test_default_minimum_flags_a_small_training_image(tmp_path: Path):
    from loradatasetlinter.policy import load_policy

    path = gradient(tmp_path / "tiny.png", size=(64, 64))
    caption(path, "red square, studio lighting")
    report = scan_dataset(tmp_path, load_policy(None))
    assert "low_resolution" in finding_codes(report)


def test_nearest_upscale_is_flagged_and_a_gradient_is_not(tmp_path: Path, policy):
    nearest_upscale(tmp_path / "enlarged.png", factor=4, base=16)
    gradient(tmp_path / "native.png", size=(64, 64))
    solid(tmp_path / "flat.png", (40, 40, 40), size=(64, 64))
    for name in ("enlarged.png", "native.png", "flat.png"):
        caption(tmp_path / name, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    flagged = [item.files[0] for item in report.findings if item.code == "upscale_artifact"]
    assert flagged == ["enlarged.png"]


def test_resolution_outlier_boundary(tmp_path: Path, policy):
    policy.resolution.min_side = 8
    policy.resolution.median_min_images = 4
    policy.resolution.median_outlier_ratio = 2.0
    # Median of 100, 100, 100, 100, 50 is 100. 50 is exactly median/2.
    for index in range(4):
        path = gradient(tmp_path / f"m{index}.png", size=(100, 100))
        caption(path, "red square, studio lighting")
    edge = gradient(tmp_path / "edge.png", size=(50, 50))
    caption(edge, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    outliers = [item.files[0] for item in report.findings if item.code == "resolution_outlier"]
    assert "edge.png" in outliers

    inside = gradient(tmp_path / "inside.png", size=(51, 51))
    caption(inside, "red square, studio lighting")
    # Adding a 51 px image changes the median. Check the pure boundary above,
    # and here confirm a set whose smallest side sits just inside the ratio.
    folder = tmp_path / "inside_set"
    folder.mkdir()
    for index in range(4):
        path = gradient(folder / f"m{index}.png", size=(100, 100))
        caption(path, "red square, studio lighting")
    path = gradient(folder / "inside.png", size=(51, 51))
    caption(path, "red square, studio lighting")
    report = scan_dataset(folder, policy)
    outliers = [item.files[0] for item in report.findings if item.code == "resolution_outlier"]
    assert "inside.png" not in outliers


def test_constant_block_fraction_of_a_nearest_upscale_is_one():
    path = Path("unused")
    # Measured through the public helper on a real enlargement in the test above.
    assert path.name == "unused"
    import numpy as np
    from PIL import Image

    image = Image.new("RGB", (8, 8), (10, 20, 30))
    image.putpixel((0, 0), (200, 10, 10))
    big = image.resize((32, 32), Image.Resampling.NEAREST)
    array = np.asarray(big)
    assert constant_block_fraction(array, 4) == 1.0
    native = np.zeros((32, 32, 3), dtype=np.uint8)
    native[:, :, 0] = np.linspace(0, 255, 32, dtype=np.uint8)
    assert constant_block_fraction(native, 2) < 0.99
