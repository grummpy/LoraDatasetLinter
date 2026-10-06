from pathlib import Path

from loradatasetlinter.engine import scan_dataset
from PIL import Image
from tests.fixtures import caption, finding_codes, gradient


def test_corrupt_file_is_a_failure(tmp_path: Path, policy):
    (tmp_path / "broken.jpg").write_bytes(b"\xff\xd8\xff not-a-jpeg")
    report = scan_dataset(tmp_path, policy)
    corrupt = [item for item in report.findings if item.code == "corrupt_file"]
    assert len(corrupt) == 1
    assert corrupt[0].severity == "fail"
    assert corrupt[0].files == ["broken.jpg"]
    assert report.score.status == "fail"


def test_cmyk_alpha_and_opaque_alpha(tmp_path: Path, policy):
    cmyk = tmp_path / "plate.jpg"
    Image.new("CMYK", (48, 48), (10, 20, 30, 5)).save(cmyk, "JPEG", quality=95)
    caption(cmyk, "red square, studio lighting")
    clear = tmp_path / "clear.png"
    Image.new("RGBA", (48, 48), (200, 20, 20, 0)).save(clear, "PNG")
    caption(clear, "red square, studio lighting")
    opaque = tmp_path / "opaque.png"
    Image.new("RGBA", (48, 48), (200, 20, 20, 255)).save(opaque, "PNG")
    caption(opaque, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    assert any(item.code == "cmyk" and item.files == ["plate.jpg"] for item in report.findings)
    assert any(
        item.code == "alpha_channel" and item.files == ["clear.png"] for item in report.findings
    )
    assert any(
        item.code == "opaque_alpha" and item.files == ["opaque.png"] for item in report.findings
    )
    assert "format_mix" in finding_codes(report)


def test_exif_orientation_is_reported_from_the_tag(tmp_path: Path, policy):
    path = tmp_path / "turned.jpg"
    image = Image.new("RGB", (32, 64), (20, 80, 180))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, "JPEG", exif=exif, quality=95)
    caption(path, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    turned = next(item for item in report.findings if item.code == "exif_orientation")
    assert turned.files == ["turned.jpg"]
    assert turned.details["orientation"] == 6
    row = report.images[0]
    assert (row["width"], row["height"]) == (32, 64)
    assert (row["oriented_width"], row["oriented_height"]) == (64, 32)


def test_format_mix_lists_both_containers(tmp_path: Path, policy):
    gradient(tmp_path / "a.png")
    caption(tmp_path / "a.png", "red square, studio lighting")
    image = Image.new("RGB", (64, 64), (10, 20, 30))
    image.save(tmp_path / "b.jpg", "JPEG", quality=95)
    caption(tmp_path / "b.jpg", "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    mix = next(item for item in report.findings if item.code == "format_mix")
    assert mix.severity == "info"
    assert mix.details["formats"]["PNG"] == 1
    assert mix.details["formats"]["JPEG"] == 1
