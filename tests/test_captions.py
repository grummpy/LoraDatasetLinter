from pathlib import Path

from loradatasetlinter.checks.captions import conflicting_orders, levenshtein, normalize_tag
from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.tokens import estimate_clip_tokens
from tests.fixtures import caption, circle, finding_codes, gradient


def test_missing_empty_and_orphan_captions(tmp_path: Path, policy):
    gradient(tmp_path / "bare.png")
    empty = gradient(tmp_path / "empty.png")
    caption(empty, "   \n")
    (tmp_path / "lonely.txt").write_text("red square, studio lighting", encoding="utf-8")
    report = scan_dataset(tmp_path, policy)
    assert any(
        item.code == "missing_caption" and item.files == ["bare.png"] for item in report.findings
    )
    assert any(
        item.code == "empty_caption" and "empty.png" in item.files for item in report.findings
    )
    assert any(
        item.code == "orphan_caption" and item.files == ["lonely.txt"] for item in report.findings
    )


def test_trigger_word_and_shared_caption(tmp_path: Path, policy):
    policy.captions.trigger_words = ["ohwx"]
    path = gradient(tmp_path / "a.png")
    caption(path, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    assert any(
        item.code == "missing_trigger" and item.details["trigger"] == "ohwx"
        for item in report.findings
    )
    caption(path, "ohwx, red square, studio lighting")
    other = circle(tmp_path / "a.jpg")
    # Same stem, so both images share a.png's sibling a.txt.
    assert other.name == "a.jpg"
    report = scan_dataset(tmp_path, policy)
    assert "missing_trigger" not in finding_codes(report)
    assert "shared_caption" in finding_codes(report)


def test_rare_tag_boundary(tmp_path: Path, policy):
    policy.captions.rare_max_images = 1
    policy.captions.drift_min_dataset = 99
    once = gradient(tmp_path / "once.png")
    caption(once, "solo tag, red square")
    for index in range(2):
        path = gradient(tmp_path / f"many{index}.png")
        caption(path, "red square, studio lighting")
    report = scan_dataset(tmp_path, policy)
    rare = next(item for item in report.findings if item.code == "rare_tag")
    tags = {row["tag"] for row in rare.details["tags"]}
    assert "solo tag" in tags
    assert "red square" not in tags


def test_low_support_synonym_order_and_separator(tmp_path: Path, policy):
    policy.captions.rare_max_images = 1
    policy.captions.drift_max_images = 2
    policy.captions.drift_min_dataset = 6
    policy.captions.order_min_images = 3
    policy.captions.order_minority_ratio = 0.2
    texts = [
        "red, blue, grey hair",
        "red, blue, gray_hair",
        "blue, red, standing",
        "red, blue, standing",
        "red, blue, twice tag, twice tag",
        "red; blue; twice tag",
        "blue, red, extra label",
    ]
    for index, text in enumerate(texts):
        path = gradient(tmp_path / f"p{index}.png")
        caption(path, text)
    report = scan_dataset(tmp_path, policy)
    codes = finding_codes(report)
    assert "near_synonym" in codes
    assert "separator_mix" in codes
    assert "tag_order" in codes
    low = next(item for item in report.findings if item.code == "low_support_tag")
    assert any(row["tag"] == "twice tag" and row["count"] == 2 for row in low.details["tags"])
    synonyms = [item for item in report.findings if item.code == "near_synonym"]
    assert any("grey hair" in item.reason or "gray hair" in item.reason for item in synonyms)


def test_tag_order_minority_boundary():
    sequences = [
        ["red", "blue"],
        ["red", "blue"],
        ["red", "blue"],
        ["red", "blue"],
        ["blue", "red"],
    ]
    assert conflicting_orders(sequences, min_images=5, minority_ratio=0.2)
    quieter = conflicting_orders(sequences, min_images=5, minority_ratio=0.21)
    assert quieter == []


def test_token_limit_boundary(tmp_path: Path, policy):
    text = "red square, studio lighting, soft shadow"
    count = estimate_clip_tokens(text)
    path = gradient(tmp_path / "a.png")
    caption(path, text)
    policy.captions.max_tokens = count
    report = scan_dataset(tmp_path, policy)
    assert "token_length" not in finding_codes(report)
    policy.captions.max_tokens = count - 1
    report = scan_dataset(tmp_path, policy)
    hit = next(item for item in report.findings if item.code == "token_length")
    assert hit.details["tokens"] == count
    assert hit.details["limit"] == count - 1


def test_unicode_caption_token_estimate_is_nonzero():
    assert estimate_clip_tokens("猫 女孩") > 0
    assert estimate_clip_tokens("кириллица portrait") > 0


def test_underscore_is_a_token_piece_not_part_of_a_word():
    assert estimate_clip_tokens("red_hair") == 3
    assert estimate_clip_tokens("red_hair") > estimate_clip_tokens("redhair")


def test_normalize_and_edit_distance():
    assert normalize_tag("Grey_Hair") == normalize_tag("gray hair")
    assert levenshtein("standing", "standlng") == 1
    assert levenshtein("standing", "stxxding") == 2
