import os
from pathlib import Path

from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.inventory import scan_folder
from loradatasetlinter.report.html_out import render_html
from loradatasetlinter.report.json_out import render_json
from PIL import Image
from tests.fixtures import caption, gradient, relax, snapshot


def test_scan_never_mutates_the_dataset(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    path = gradient(root / "nested" / "photo.png")
    caption(path, "red square, studio lighting")
    (root / "notes.txt").write_text("orphan caption", encoding="utf-8")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG-not-really")
    (root / "link.png").symlink_to(outside)
    before = snapshot(root)
    real_open = open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")) and not isinstance(file, int):
            candidate = Path(file)
            if (
                root.resolve() == candidate.resolve()
                or root.resolve() in candidate.resolve().parents
            ):
                raise AssertionError(f"write opened inside the dataset: {file} mode={mode}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    original_save = Image.Image.save

    def guarded_save(self, fp, *args, **kwargs):
        if isinstance(fp, str | Path):
            candidate = Path(fp)
            if (
                root.resolve() == candidate.resolve()
                or root.resolve() in candidate.resolve().parents
            ):
                raise AssertionError(f"image save inside the dataset: {fp}")
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", guarded_save)
    policy = relax()
    report = scan_dataset(root, policy)
    render_json(report)
    render_html(report, root, thumbnail_px=24)
    assert snapshot(root) == before
    assert "link.png" not in {row["file"] for row in report.images}
    assert report.stats["skipped_symlinks"] == 1
    assert report.stats["orphan_captions"] == 1
    # The outside target of the symlink was not required, and the dataset listing is unchanged.
    assert set(os.listdir(root / "nested")) == {"photo.png", "photo.txt"}


def test_scan_records_and_prunes_file_and_directory_symlinks(tmp_path: Path, policy):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    gradient(outside / "external.png")
    (root / "file-link.png").symlink_to(outside / "external.png")
    (root / "directory-link").symlink_to(outside, target_is_directory=True)
    (root / "broken-directory-link").symlink_to(tmp_path / "missing", target_is_directory=True)

    dataset = scan_folder(root, policy)

    assert dataset.skipped_symlinks == [
        "broken-directory-link",
        "directory-link",
        "file-link.png",
    ]
    assert not dataset.images
