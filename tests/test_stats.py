from pathlib import Path

from loradatasetlinter.checks.stats import is_imbalanced, step_estimate
from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.policy import load_policy
from tests.fixtures import caption, finding_codes, gradient, relax


def test_step_formula_and_repeats(tmp_path: Path):
    policy = relax(load_policy(None))
    policy.stats.epochs = 3
    policy.stats.batch_size = 2
    policy.stats.default_repeats = 1
    for index in range(5):
        path = gradient(tmp_path / "4_cat" / f"c{index}.png", size=(32, 32))
        caption(path, "red square, studio lighting")
    for index in range(2):
        path = gradient(tmp_path / f"r{index}.png", size=(32, 32))
        caption(path, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    steps = report.stats["steps"]
    # 5 images * 4 repeats + 2 images * 1 = 22. ceil(22/2)*3 = 33.
    assert steps["weighted_images"] == 22
    assert steps["steps_per_epoch"] == 11
    assert steps["steps"] == 33
    assert steps["naive_steps"] == 33
    names = {row["class_name"] for row in steps["classes"]}
    assert names == {"cat", "root"}


def test_class_balance_includes_the_ratio_boundary(tmp_path: Path):
    assert is_imbalanced(30, 10, 3.0) is True
    assert is_imbalanced(29, 10, 3.0) is False
    policy = relax()
    policy.stats.balance_ratio = 3.0
    policy.stats.default_repeats = 1
    for index in range(3):
        path = gradient(tmp_path / "heavy" / f"h{index}.png")
        caption(path, "red square, studio lighting")
    path = gradient(tmp_path / "light" / "l0.png")
    caption(path, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    imbalance = [item for item in report.findings if item.code == "class_imbalance"]
    assert len(imbalance) == 1
    assert imbalance[0].details["ratio"] == 3.0

    folder = tmp_path / "under"
    folder.mkdir()
    for index in range(3):
        path = gradient(folder / "heavy" / f"h{index}.png")
        caption(path, "red square, studio lighting")
    for index in range(2):
        path = gradient(folder / "light" / f"l{index}.png")
        caption(path, "red square, studio lighting")
    report = scan_dataset(folder, policy)
    assert "class_imbalance" not in finding_codes(report)


def test_zero_repeats_add_no_steps(tmp_path: Path, policy):
    path = gradient(tmp_path / "0_skip" / "a.png")
    caption(path, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    assert "zero_repeats" in finding_codes(report)
    assert report.stats["steps"]["weighted_images"] == 0
    assert report.stats["steps"]["steps"] == 0


def test_step_estimate_matches_the_report(tmp_path: Path, policy):
    path = gradient(tmp_path / "10_widget" / "a.png")
    caption(path, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    from loradatasetlinter.inventory import scan_folder

    loaded = scan_folder(tmp_path, policy)
    assert (
        step_estimate(loaded, policy)["weighted_images"] == report.stats["steps"]["weighted_images"]
    )
