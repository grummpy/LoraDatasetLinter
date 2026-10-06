from pathlib import Path

import yaml
from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.errors import PolicyError
from loradatasetlinter.policy import load_policy
from tests.fixtures import caption, finding_codes, gradient


def test_packaged_default_loads_and_matches_the_example():
    policy = load_policy(None)
    assert policy.resolution.min_side == 512
    assert policy.duplicates.clip.enabled is False
    assert policy.duplicates.hamming_threshold == 8
    assert policy.aspect.base_resolutions == [512, 768, 1024]
    packaged = Path(__file__).parents[1] / "loradatasetlinter" / "default_policy.yaml"
    example = Path(__file__).parents[1] / "docs" / "policy.example.yaml"
    assert example.read_text(encoding="utf-8") == packaged.read_text(encoding="utf-8")


def test_partial_policy_overrides_only_the_given_keys(tmp_path: Path):
    path = tmp_path / "policy.yaml"
    path.write_text("resolution:\n  min_side: 32\n", encoding="utf-8")
    policy = load_policy(path)
    assert policy.resolution.min_side == 32
    assert policy.duplicates.hamming_threshold == 8
    image = gradient(tmp_path / "data" / "a.png", size=(40, 40))
    caption(image, "red square, studio lighting")
    report = scan_dataset(tmp_path / "data", policy)
    assert "low_resolution" not in finding_codes(report)


def test_unknown_key_and_bad_version_are_errors(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("resolution:\n  not_a_knob: 1\n", encoding="utf-8")
    try:
        load_policy(bad)
    except PolicyError as exc:
        assert "not_a_knob" in str(exc)
    else:
        raise AssertionError("unknown key should fail")
    version = tmp_path / "version.yaml"
    version.write_text("version: 2\n", encoding="utf-8")
    try:
        load_policy(version)
    except PolicyError as exc:
        assert "version" in str(exc).lower()
    else:
        raise AssertionError("version 2 should fail")


def test_disabling_a_check_drops_its_findings(tmp_path: Path):
    data = yaml.safe_load(
        (Path(__file__).parents[1] / "loradatasetlinter" / "default_policy.yaml").read_text(
            encoding="utf-8"
        )
    )
    data["resolution"]["enabled"] = False
    data["resolution"]["min_side"] = 8
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    policy = load_policy(path)
    image = gradient(tmp_path / "set" / "tiny.png", size=(16, 16))
    caption(image, "red square, studio lighting")
    report = scan_dataset(tmp_path / "set", policy)
    assert "low_resolution" not in finding_codes(report)


def test_score_gates_from_policy(tmp_path: Path):
    from loradatasetlinter.models import Finding
    from loradatasetlinter.score import decide

    policy = load_policy(None)
    info = [Finding("captions", "rare_tag", "info", "rare", [], {}) for _ in range(10)]
    # 10 info * weight 1 => score 90, which is the pass minimum.
    assert decide(info, policy).status == "pass"
    assert decide(info, policy).value == 90
    assert decide(info + info[:1], policy).status == "warn"
    assert decide(info + info[:1], policy).value == 89
    many = [Finding("captions", "rare_tag", "info", "rare", [], {}) for _ in range(31)]
    failed = decide(many, policy)
    assert failed.value == 69
    assert failed.status == "fail"
    warned = [Finding("files", "cmyk", "warn", "cmyk", ["a.png"], {})]
    assert decide(warned, policy).status == "warn"
    hard = [Finding("files", "corrupt_file", "fail", "bad", ["a.png"], {})]
    assert decide(hard, policy).status == "fail"
