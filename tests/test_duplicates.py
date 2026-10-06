import shutil
from pathlib import Path

import numpy as np
from loradatasetlinter.cluster import cluster_by_cosine, cluster_by_distance, hamming
from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.policy import load_policy
from PIL import Image
from tests.fixtures import caption, circle, finding_codes, gradient, relax


def test_hamming_threshold_includes_equal_distance_and_excludes_the_next():
    threshold = 8
    base = 0
    at_edge = _flip_bits(base, threshold)
    past_edge = _flip_bits(base, threshold + 1)
    assert hamming(base, at_edge) == threshold
    assert hamming(base, past_edge) == threshold + 1
    grouped = cluster_by_distance(["a", "b"], {"a": base, "b": at_edge}, threshold)
    apart = cluster_by_distance(["a", "b"], {"a": base, "b": past_edge}, threshold)
    assert grouped == [["a", "b"]]
    assert apart == []


def test_exact_and_pixel_duplicates(tmp_path: Path, policy):
    original = circle(tmp_path / "keep.png")
    caption(original, "red circle, studio lighting")
    shutil.copyfile(original, tmp_path / "copy.png")
    caption(tmp_path / "copy.png", "red circle, studio lighting")
    retagged = tmp_path / "pixels.png"
    with Image.open(original) as image:
        image.save(retagged, "PNG", pnginfo=_comment())
    caption(retagged, "red circle, studio lighting")
    assert (tmp_path / "keep.png").read_bytes() == (tmp_path / "copy.png").read_bytes()
    assert (tmp_path / "keep.png").read_bytes() != retagged.read_bytes()

    report = scan_dataset(tmp_path, policy)
    exact = [item for item in report.findings if item.code == "exact_duplicate"]
    pixels = [item for item in report.findings if item.code == "exact_pixels"]
    assert len(exact) == 1
    assert exact[0].severity == "fail"
    assert exact[0].files == ["copy.png", "keep.png"]
    assert exact[0].details["keep"] == "copy.png"
    assert len(pixels) == 1
    assert set(pixels[0].files) == {"copy.png", "keep.png", "pixels.png"}


def test_near_duplicate_hamming_edge_on_images(tmp_path: Path):
    source = gradient(tmp_path / "a.png")
    caption(source, "red square, studio lighting")
    shifted = tmp_path / "b.png"
    distance = _diverge(source, shifted, "phash")
    caption(shifted, "red square, studio lighting")
    assert distance > 0

    at_edge = relax()
    at_edge.duplicates.hamming_threshold = distance
    report = scan_dataset(tmp_path, at_edge)
    near = [item for item in report.findings if item.code == "near_duplicate"]
    assert len(near) == 1
    assert near[0].details["max_distance"] == distance
    assert set(near[0].files) == {"a.png", "b.png"}

    below = relax()
    below.duplicates.hamming_threshold = distance - 1
    report = scan_dataset(tmp_path, below)
    assert "near_duplicate" not in finding_codes(report)


def test_disabled_duplicate_check_is_silent(tmp_path: Path, policy):
    original = circle(tmp_path / "a.png")
    shutil.copyfile(original, tmp_path / "b.png")
    caption(original, "red square, studio lighting")
    caption(tmp_path / "b.png", "red square, studio lighting")
    policy.duplicates.enabled = False
    report = scan_dataset(tmp_path, policy)
    assert "exact_duplicate" not in finding_codes(report)


def test_cosine_threshold_includes_equality():
    left = np.array([1.0, 0.0])
    angle = np.arccos(0.95)
    right = np.array([np.cos(angle), np.sin(angle)])
    right = right / np.linalg.norm(right)
    cosine = float(np.dot(left, right))
    vectors = {"a": left, "b": right}
    assert cluster_by_cosine(["a", "b"], vectors, cosine) == [["a", "b"]]
    assert cluster_by_cosine(["a", "b"], vectors, cosine + 1e-6) == []


def test_clip_disabled_does_not_import_torch(tmp_path: Path, policy):
    import sys

    sys.modules.pop("torch", None)
    sys.modules.pop("open_clip", None)
    circle(tmp_path / "a.png")
    caption(tmp_path / "a.png", "red square, studio lighting")
    scan_dataset(tmp_path, policy)
    assert "torch" not in sys.modules
    assert "open_clip" not in sys.modules


def test_clip_enabled_without_checkpoint_is_info_and_stays_off_gpu(tmp_path: Path, policy):
    import sys

    sys.modules.pop("torch", None)
    circle(tmp_path / "a.png")
    caption(tmp_path / "a.png", "red square, studio lighting")
    policy.duplicates.clip.enabled = True
    policy.duplicates.clip.checkpoint = str(tmp_path / "missing.pt")
    report = scan_dataset(tmp_path, policy)
    unavailable = [item for item in report.findings if item.code == "clip_unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0].severity == "info"
    assert (
        "download" in unavailable[0].reason.lower() or "not found" in unavailable[0].reason.lower()
    )
    assert "torch" not in sys.modules


def test_source_never_mentions_cuda_calls():
    root = Path(__file__).parents[1] / "loradatasetlinter"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert ".cuda" not in text
        assert "torch.cuda" not in text


def test_dhash_method_is_recorded(tmp_path: Path):
    source = gradient(tmp_path / "a.png")
    shifted = tmp_path / "b.png"
    _save_brightness(source, shifted, delta=6)
    caption(source, "red square, studio lighting")
    caption(shifted, "red square, studio lighting")
    policy = relax(load_policy(None))
    policy.duplicates.hash_method = "dhash"
    policy.duplicates.hamming_threshold = 64
    report = scan_dataset(tmp_path, policy)
    near = [item for item in report.findings if item.code == "near_duplicate"]
    assert len(near) == 1
    assert near[0].details["hash_method"] == "dhash"


def _flip_bits(value: int, count: int) -> int:
    flipped = value
    for bit in range(count):
        flipped ^= 1 << bit
    return flipped


def _comment():
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("comment", "same pixels, different bytes")
    return info


def _save_brightness(source: Path, dest: Path, delta: int) -> None:
    with Image.open(source) as image:
        array = np.asarray(image.convert("RGB")).astype(np.int16)
    array = np.clip(array + delta, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(dest, "PNG")


def _diverge(source: Path, dest: Path, method: str) -> int:
    """Change a corner until the perceptual hash moves by at least one bit."""
    with Image.open(source) as image:
        base = np.asarray(image.convert("RGB")).copy()
    for size in range(1, 24):
        trial = base.copy()
        trial[:size, :size] = (255, 0, 0) if size % 2 else (0, 0, 255)
        Image.fromarray(trial).save(dest, "PNG")
        distance = _hash_distance(source, dest, method)
        if distance >= 1:
            return distance
    raise AssertionError("perceptual hash did not change")


def _phash_distance(left: Path, right: Path) -> int:
    return _hash_distance(left, right, "phash")


def _hash_distance(left: Path, right: Path, method: str) -> int:
    import imagehash

    hasher = imagehash.phash if method == "phash" else imagehash.dhash
    with Image.open(left) as image:
        a = hasher(image.convert("RGB"))
    with Image.open(right) as image:
        b = hasher(image.convert("RGB"))
    return int(a - b)
