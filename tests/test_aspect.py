import math
from pathlib import Path

from loradatasetlinter.checks.aspect import is_aspect_outlier, is_extreme_aspect
from loradatasetlinter.engine import scan_dataset
from tests.fixtures import caption, gradient


def test_aspect_outlier_includes_the_log_boundary():
    assert is_aspect_outlier(1.75, 1.0, 1.75) is True
    assert is_aspect_outlier(1.74, 1.0, 1.75) is False
    assert is_aspect_outlier(1 / 1.75, 1.0, 1.75) is True


def test_extreme_aspect_includes_the_limit():
    assert is_extreme_aspect(3.0, 3.0) is True
    assert is_extreme_aspect(2.99, 3.0) is False
    assert is_extreme_aspect(1 / 3, 3.0) is True


def test_extreme_and_bucket_histogram(tmp_path: Path, policy):
    policy.aspect.outlier_min_images = 99
    wide = gradient(tmp_path / "wide.png", size=(300, 100))
    square = gradient(tmp_path / "square.png", size=(128, 128))
    caption(wide, "red square, studio lighting")
    caption(square, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    extreme = [item for item in report.findings if item.code == "extreme_aspect"]
    assert [item.files for item in extreme] == [["wide.png"]]
    assert math.isclose(extreme[0].details["aspect"], 3.0)
    buckets = report.stats["buckets"]
    assert set(buckets) == {"512", "768", "1024"}
    counts = {tuple(row["resolution"]): row["count"] for row in buckets["512"]}
    assert sum(counts.values()) == 2
    square_row = next(row for row in report.images if row["file"] == "square.png")
    assert square_row["buckets"]["512"] == [512, 512]


def test_aspect_outlier_on_a_folder_of_squares(tmp_path: Path, policy):
    policy.aspect.extreme_aspect = 100
    for index in range(4):
        path = gradient(tmp_path / f"s{index}.png", size=(64, 64))
        caption(path, "red square, studio lighting")
    wide = gradient(tmp_path / "wide.png", size=(112, 64))
    caption(wide, "red square, studio lighting")
    # 112/64 = 1.75 exactly, median of five aspects with four squares is 1.
    report = scan_dataset(tmp_path, policy)
    outliers = [item.files[0] for item in report.findings if item.code == "aspect_outlier"]
    assert outliers == ["wide.png"]

    folder = tmp_path / "inside"
    folder.mkdir()
    for index in range(4):
        path = gradient(folder / f"s{index}.png", size=(64, 64))
        caption(path, "red square, studio lighting")
    inside = gradient(folder / "inside.png", size=(111, 64))
    caption(inside, "red square, studio lighting")
    report = scan_dataset(folder, policy)
    outliers = [item.files[0] for item in report.findings if item.code == "aspect_outlier"]
    assert "inside.png" not in outliers
