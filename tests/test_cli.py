import json
import os
import subprocess
from pathlib import Path

import loradatasetlinter.safety as safety
import pytest
from loradatasetlinter.cli import app
from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.fixplan import render_fix_plan
from loradatasetlinter.models import Finding
from tests.fixtures import caption, clean_dataset, gradient, relax, snapshot
from typer.testing import CliRunner

runner = CliRunner()


def _duplicate_pair(root: Path, folder: str = "") -> tuple[Path, Path]:
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    original = gradient(directory / "a.png")
    caption(original, "red square, studio lighting")
    copy = directory / "b.png"
    copy.write_bytes(original.read_bytes())
    caption(copy, "red square, studio lighting")
    return original, copy


def _write_existing_outputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    out = tmp_path / "out"
    out.mkdir()
    paths = (out / "report.json", out / "report.html", out / "fix-plan.sh")
    for index, path in enumerate(paths):
        path.write_text(f"previous-{index}", encoding="utf-8")
    return paths


def _write_fix_plan(root: Path, plan: Path) -> None:
    result = runner.invoke(
        app,
        ["scan", str(root), "--min-side", "8", "--fix-plan", str(plan), "--quiet"],
    )
    assert result.exit_code == 2, result.output


def test_clean_dataset_exits_zero_and_writes_reports_outside(tmp_path: Path):
    root = clean_dataset(tmp_path / "data")
    before = snapshot(root)
    out = tmp_path / "out"
    plan = tmp_path / "fix-plan.sh"
    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            "--output-dir",
            str(out),
            "--fix-plan",
            str(plan),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status: PASS" in result.output
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in html
    assert plan.is_file()
    assert "DRY_RUN" in plan.read_text(encoding="utf-8")
    assert snapshot(root) == before


def test_warn_and_fail_exit_codes(tmp_path: Path):
    warn_root = tmp_path / "warn"
    warn_root.mkdir()
    from PIL import Image

    image = Image.new("RGBA", (48, 48), (20, 200, 20, 10))
    image.save(warn_root / "clear.png", "PNG")
    caption(warn_root / "clear.png", "red square, studio lighting")
    result = runner.invoke(app, ["scan", str(warn_root), "--min-side", "8", "--quiet"])
    assert result.exit_code == 1, result.output

    fail_root = tmp_path / "fail"
    fail_root.mkdir()
    gradient(fail_root / "bare.png")
    result = runner.invoke(app, ["scan", str(fail_root), "--min-side", "8", "--quiet"])
    assert result.exit_code == 2, result.output


def test_refuses_to_write_inside_the_dataset(tmp_path: Path):
    root = clean_dataset(tmp_path / "data")
    before = snapshot(root)
    result = runner.invoke(
        app,
        ["scan", str(root), "--min-side", "8", "--json", str(root / "report.json")],
    )
    assert result.exit_code == 3
    assert "never modifies" in result.output.lower() or "Refusing" in result.output
    assert snapshot(root) == before
    assert not (root / "report.json").exists()


@pytest.mark.parametrize(
    ("first_flag", "second_flag"),
    [("--json", "--html"), ("--json", "--fix-plan"), ("--html", "--fix-plan")],
)
def test_refuses_colliding_output_aliases_without_changing_previous_output(
    tmp_path: Path, first_flag: str, second_flag: str
):
    root = clean_dataset(tmp_path / "data")
    out = tmp_path / "out"
    out.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(out, target_is_directory=True)
    target = out / "report.json"
    target.write_text("previous report", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            first_flag,
            str(target),
            second_flag,
            str(alias / "report.json"),
            "--quiet",
        ],
    )

    assert result.exit_code == 3
    assert "same output" in result.output
    assert target.read_text(encoding="utf-8") == "previous report"


def test_refuses_output_directory_identity_collision_before_writing(tmp_path: Path):
    root = clean_dataset(tmp_path / "data")
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "previous.txt"
    marker.write_text("keep", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(out, target_is_directory=True)

    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            "--json",
            str(out),
            "--output-dir",
            str(alias),
            "--quiet",
        ],
    )

    assert result.exit_code == 3
    assert "not a regular file" in result.output
    assert marker.read_text(encoding="utf-8") == "keep"


def test_refuses_output_hard_linked_to_dataset_without_truncating_caption(tmp_path: Path):
    root = clean_dataset(tmp_path / "data")
    caption_path = root / "gradient.txt"
    report = tmp_path / "report.json"
    os.link(caption_path, report)
    before = caption_path.read_bytes()

    result = runner.invoke(
        app,
        ["scan", str(root), "--min-side", "8", "--json", str(report), "--quiet"],
    )

    assert result.exit_code == 3
    assert "hard-linked" in result.output
    assert caption_path.read_bytes() == before
    assert report.read_bytes() == before


def test_refuses_hard_linked_output_pair_without_publishing(tmp_path: Path):
    root = clean_dataset(tmp_path / "data")
    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    json_path.write_text("previous", encoding="utf-8")
    os.link(json_path, html_path)

    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
            "--quiet",
        ],
    )

    assert result.exit_code == 3
    assert "same existing file" in result.output
    assert json_path.read_text(encoding="utf-8") == "previous"
    assert html_path.read_text(encoding="utf-8") == "previous"
    assert json_path.stat().st_ino == html_path.stat().st_ino


def test_rejects_directory_output_before_replacing_existing_json(tmp_path: Path):
    root = clean_dataset(tmp_path / "data")
    json_path = tmp_path / "report.json"
    json_path.write_text("previous json", encoding="utf-8")
    html_directory = tmp_path / "report.html"
    html_directory.mkdir()

    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            "--json",
            str(json_path),
            "--html",
            str(html_directory),
            "--quiet",
        ],
    )

    assert result.exit_code == 3
    assert "not a regular file" in result.output
    assert json_path.read_text(encoding="utf-8") == "previous json"


def test_second_render_failure_preserves_all_previous_outputs(tmp_path: Path, monkeypatch):
    root = clean_dataset(tmp_path / "data")
    json_path, html_path, plan = _write_existing_outputs(tmp_path)

    def fail_html(*_args, **_kwargs):
        raise OSError("injected second render failure")

    monkeypatch.setattr("loradatasetlinter.cli.render_html", fail_html)
    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
            "--fix-plan",
            str(plan),
            "--quiet",
        ],
    )

    assert result.exit_code == 3
    assert "injected second render failure" in result.output
    assert [path.read_text(encoding="utf-8") for path in (json_path, html_path, plan)] == [
        "previous-0",
        "previous-1",
        "previous-2",
    ]


def test_third_render_failure_preserves_all_previous_outputs(tmp_path: Path, monkeypatch):
    root = clean_dataset(tmp_path / "data")
    json_path, html_path, plan = _write_existing_outputs(tmp_path)

    def fail_plan(*_args, **_kwargs):
        raise OSError("injected third render failure")

    monkeypatch.setattr("loradatasetlinter.cli.render_fix_plan", fail_plan)
    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
            "--fix-plan",
            str(plan),
            "--quiet",
        ],
    )

    assert result.exit_code == 3
    assert "injected third render failure" in result.output
    assert [path.read_text(encoding="utf-8") for path in (json_path, html_path, plan)] == [
        "previous-0",
        "previous-1",
        "previous-2",
    ]


def test_stage_failure_preserves_all_previous_outputs(tmp_path: Path, monkeypatch):
    root = clean_dataset(tmp_path / "data")
    json_path, html_path, plan = _write_existing_outputs(tmp_path)
    original_stage = safety._stage_one
    calls = 0

    def fail_second_stage(target: Path, content: str) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")
        return original_stage(target, content)

    monkeypatch.setattr(safety, "_stage_one", fail_second_stage)
    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
            "--fix-plan",
            str(plan),
            "--quiet",
        ],
    )

    assert result.exit_code == 3
    assert "injected staging failure" in result.output
    assert [path.read_text(encoding="utf-8") for path in (json_path, html_path, plan)] == [
        "previous-0",
        "previous-1",
        "previous-2",
    ]


def test_replacement_failure_rolls_back_all_previous_outputs(tmp_path: Path, monkeypatch):
    root = clean_dataset(tmp_path / "data")
    json_path, html_path, plan = _write_existing_outputs(tmp_path)
    original_replace = safety._replace
    calls = 0

    def fail_third_replacement(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise OSError("injected replacement failure")
        original_replace(source, destination)

    monkeypatch.setattr(safety, "_replace", fail_third_replacement)
    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
            "--fix-plan",
            str(plan),
            "--quiet",
        ],
    )

    assert result.exit_code == 3
    assert "injected replacement failure" in result.output
    assert [path.read_text(encoding="utf-8") for path in (json_path, html_path, plan)] == [
        "previous-0",
        "previous-1",
        "previous-2",
    ]


def test_keyboard_interrupt_during_replacement_rolls_back_all_outputs(tmp_path: Path, monkeypatch):
    json_path, html_path, plan = _write_existing_outputs(tmp_path)
    original_replace = safety._replace
    calls = 0

    def interrupt_third_replacement(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise KeyboardInterrupt
        original_replace(source, destination)

    monkeypatch.setattr(safety, "_replace", interrupt_third_replacement)

    with pytest.raises(KeyboardInterrupt):
        safety.publish_outputs(
            [
                ("JSON report", json_path, "new-json"),
                ("HTML report", html_path, "new-html"),
                ("fix plan", plan, "new-plan"),
            ]
        )

    assert [path.read_text(encoding="utf-8") for path in (json_path, html_path, plan)] == [
        "previous-0",
        "previous-1",
        "previous-2",
    ]


def test_output_rollback_failure_reports_retained_backup(tmp_path: Path, monkeypatch):
    json_path, html_path, plan = _write_existing_outputs(tmp_path)
    original_replace = safety._replace
    calls = 0

    def fail_publish_and_restore(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise OSError("injected publish failure")
        if calls == 7:
            raise OSError("injected restore failure")
        original_replace(source, destination)

    monkeypatch.setattr(safety, "_replace", fail_publish_and_restore)

    with pytest.raises(safety.OutputRollbackError) as exc_info:
        safety.publish_outputs(
            [
                ("JSON report", json_path, "new-json"),
                ("HTML report", html_path, "new-html"),
                ("fix plan", plan, "new-plan"),
            ]
        )

    message = str(exc_info.value)
    assert "Output rollback incomplete" in message
    assert "fix plan" in message
    assert "retained backup at" in message
    assert [path.read_text(encoding="utf-8") for path in (json_path, html_path)] == [
        "previous-0",
        "previous-1",
    ]
    assert not plan.exists()
    assert list(plan.parent.glob(".fix-plan.sh.linter-backup-*"))


def test_cli_reports_incomplete_rollback_after_interrupt(tmp_path: Path, monkeypatch):
    root = clean_dataset(tmp_path / "data")
    json_path, html_path, plan = _write_existing_outputs(tmp_path)
    original_replace = safety._replace
    calls = 0

    def interrupt_publish_then_fail_restore(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise KeyboardInterrupt
        if calls == 7:
            raise OSError("injected restore failure")
        original_replace(source, destination)

    monkeypatch.setattr(safety, "_replace", interrupt_publish_then_fail_restore)
    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--min-side",
            "8",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
            "--fix-plan",
            str(plan),
            "--quiet",
        ],
    )

    backups = list(plan.parent.glob(".fix-plan.sh.linter-backup-*"))
    assert result.exit_code == 130
    assert "Output rollback incomplete" in result.output
    assert "injected restore failure" in result.output
    assert str(plan) in result.output
    assert len(backups) == 1
    assert str(backups[0]) in result.output


def test_missing_dataset_exits_three(tmp_path: Path):
    result = runner.invoke(app, ["scan", str(tmp_path / "missing")])
    assert result.exit_code == 3


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip()


def test_fix_plan_dry_run_does_not_move_and_apply_does(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    original = gradient(root / "a.png")
    caption(original, "red square, studio lighting")
    copy = root / "b.png"
    copy.write_bytes(original.read_bytes())
    caption(copy, "red square, studio lighting")
    plan = tmp_path / "fix-plan.sh"
    result = runner.invoke(
        app,
        ["scan", str(root), "--min-side", "8", "--fix-plan", str(plan), "--quiet"],
    )
    assert result.exit_code == 2
    script = plan.read_text(encoding="utf-8")
    assert "move " in script
    assert "b.png" in script
    before = snapshot(root)
    dry = subprocess.run(["bash", str(plan)], check=False, capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert snapshot(root) == before
    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "0"},
    )
    assert applied.returncode == 0, applied.stderr
    assert (root / "a.png").is_file()
    assert not (root / "b.png").is_file()
    assert (root / "_linter_review" / "duplicates" / "b.png").is_file()
    assert not (root / "b.txt").is_file()
    assert (root / "_linter_review" / "duplicates" / "b.txt").is_file()
    receipt = root / "_linter_review" / "recovery-receipts.tsv"
    assert "b.png" in receipt.read_text(encoding="utf-8")
    assert "b.txt" in receipt.read_text(encoding="utf-8")


def test_fix_plan_refuses_existing_destination_without_overwriting(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    original = gradient(root / "a.png")
    caption(original, "red square, studio lighting")
    copy = root / "b.png"
    copy.write_bytes(original.read_bytes())
    caption(copy, "red square, studio lighting")
    plan = tmp_path / "fix-plan.sh"
    result = runner.invoke(
        app,
        ["scan", str(root), "--min-side", "8", "--fix-plan", str(plan), "--quiet"],
    )
    assert result.exit_code == 2
    destination = root / "_linter_review" / "duplicates" / "b.png"
    destination.parent.mkdir(parents=True)
    destination.write_text("keep this review file", encoding="utf-8")

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "0"},
    )

    assert applied.returncode != 0
    assert "refusing to overwrite" in applied.stderr
    assert copy.is_file()
    assert destination.read_text(encoding="utf-8") == "keep this review file"


def test_fix_plan_rejects_review_symlink_at_apply_time(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    original = gradient(root / "a.png")
    caption(original, "red square, studio lighting")
    copy = root / "b.png"
    copy.write_bytes(original.read_bytes())
    caption(copy, "red square, studio lighting")
    plan = tmp_path / "fix-plan.sh"
    result = runner.invoke(
        app,
        ["scan", str(root), "--min-side", "8", "--fix-plan", str(plan), "--quiet"],
    )
    assert result.exit_code == 2
    outside = tmp_path / "outside"
    outside.mkdir()
    review = root / "_linter_review"
    review.mkdir()
    (review / "duplicates").symlink_to(outside, target_is_directory=True)

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "0"},
    )

    assert applied.returncode != 0
    assert "symlink" in applied.stderr
    assert copy.is_file()
    assert not any(outside.iterdir())


def test_fix_plan_keeps_shared_caption_for_retained_image(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    retained = gradient(root / "a.jpg")
    moved = root / "a.png"
    moved.write_bytes(retained.read_bytes())
    shared_caption = root / "a.txt"
    shared_caption.write_text("red square, studio lighting", encoding="utf-8")
    plan = tmp_path / "fix-plan.sh"
    result = runner.invoke(
        app,
        ["scan", str(root), "--min-side", "8", "--fix-plan", str(plan), "--quiet"],
    )
    assert result.exit_code == 2
    script = plan.read_text(encoding="utf-8")
    assert "shared caption" in script

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "0"},
    )

    assert applied.returncode == 0, applied.stderr
    assert retained.is_file()
    assert not moved.exists()
    assert shared_caption.is_file()
    assert (root / "_linter_review" / "duplicates" / "a.png").is_file()
    assert not (root / "_linter_review" / "duplicates" / "a.txt").exists()


def test_fix_plan_escapes_newline_filenames_in_comments_and_commands(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    malicious = "\n: > generated_shell_sentinel\n#keep.png"
    original = gradient(root / malicious)
    caption(original, "red square, studio lighting")
    copy = root / "z.png"
    copy.write_bytes(original.read_bytes())
    caption(copy, "red square, studio lighting")
    report = scan_dataset(root, relax())
    report.findings.append(
        Finding(
            check="duplicates",
            code="near_duplicate",
            severity="warn",
            reason="synthetic comment-injection fixture",
            files=[malicious, "z.png"],
        )
    )
    plan = tmp_path / "fix-plan.sh"
    script = render_fix_plan(report, root)
    plan.write_text(script, encoding="utf-8")
    sentinel = tmp_path / "generated_shell_sentinel"

    assert "\n: > generated_shell_sentinel\n" not in script
    syntax = subprocess.run(["bash", "-n", str(plan)], check=False, capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(
        ["bash", str(plan)], check=False, capture_output=True, text=True, cwd=tmp_path
    )
    assert dry.returncode == 0, dry.stderr
    assert not sentinel.exists()


def test_fix_plan_handles_trailing_newline_dataset_directory(tmp_path: Path):
    root = tmp_path / "data\n"
    root.mkdir()
    _original, copy = _duplicate_pair(root)
    plan = tmp_path / "fix-plan.sh"
    _write_fix_plan(root, plan)
    assert 'current="$(' not in plan.read_text(encoding="utf-8")

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "0"},
    )

    assert applied.returncode == 0, applied.stderr
    assert not copy.exists()
    assert (root / "_linter_review" / "duplicates" / "b.png").is_file()
    assert (root / "_linter_review" / "duplicates" / "b.txt").is_file()


def test_fix_plan_rejects_nested_directory_replaced_with_external_symlink(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    nested = root / "nested\n"
    _original, copy = _duplicate_pair(root, "nested\n")
    plan = tmp_path / "fix-plan.sh"
    _write_fix_plan(root, plan)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.png").write_bytes(b"outside-a")
    (outside / "a.txt").write_text("outside-a", encoding="utf-8")
    (outside / "b.png").write_bytes(b"outside-b")
    (outside / "b.txt").write_text("outside-b", encoding="utf-8")
    before = snapshot(outside)
    (root / "_linter_review" / "duplicates" / "nested\n").mkdir(parents=True)
    saved_nested = tmp_path / "nested-before-swap"
    nested.rename(saved_nested)
    nested.symlink_to(outside, target_is_directory=True)

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "0"},
    )

    assert applied.returncode != 0
    assert "symlink" in applied.stderr
    assert snapshot(outside) == before
    assert (saved_nested / copy.name).is_file()


def test_fix_plan_rejects_dataset_ancestor_replaced_with_external_symlink(tmp_path: Path):
    holder = tmp_path / "holder"
    root = holder / "data"
    root.mkdir(parents=True)
    _original, copy = _duplicate_pair(root)
    plan = tmp_path / "fix-plan.sh"
    _write_fix_plan(root, plan)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.png").write_bytes(b"outside-a")
    (outside / "a.txt").write_text("outside-a", encoding="utf-8")
    (outside / "b.png").write_bytes(b"outside-b")
    (outside / "b.txt").write_text("outside-b", encoding="utf-8")
    before = snapshot(outside)
    saved_holder = tmp_path / "holder-before-swap"
    holder.rename(saved_holder)
    holder.mkdir()
    (holder / "data").symlink_to(outside, target_is_directory=True)

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "0"},
    )

    assert applied.returncode != 0
    assert "symlink" in applied.stderr
    assert snapshot(outside) == before
    assert (saved_holder / "data" / copy.name).is_file()


@pytest.mark.parametrize("kind", ["directory", "symlink", "read_only"])
def test_fix_plan_prepares_safe_receipts_before_any_move(tmp_path: Path, kind: str):
    root = tmp_path / "data"
    root.mkdir()
    _original, copy = _duplicate_pair(root)
    plan = tmp_path / "fix-plan.sh"
    _write_fix_plan(root, plan)
    receipt = root / "_linter_review" / "recovery-receipts.tsv"
    receipt.parent.mkdir()
    external = tmp_path / "external-receipt"
    if kind == "directory":
        receipt.mkdir()
    elif kind == "symlink":
        external.write_text("outside receipt", encoding="utf-8")
        receipt.symlink_to(external)
    else:
        receipt.write_text("read only receipt", encoding="utf-8")
        receipt.chmod(0o400)

    try:
        applied = subprocess.run(
            ["bash", str(plan)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "DRY_RUN": "0"},
        )
    finally:
        if kind == "read_only":
            receipt.chmod(0o600)

    assert applied.returncode != 0
    assert copy.is_file()
    assert (root / "b.txt").is_file()
    assert not (root / "_linter_review" / "duplicates" / "b.png").exists()
    if kind == "symlink":
        assert external.read_text(encoding="utf-8") == "outside receipt"


def test_fix_plan_preflights_caption_destination_before_moving_image(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    _original, copy = _duplicate_pair(root)
    plan = tmp_path / "fix-plan.sh"
    _write_fix_plan(root, plan)
    caption_destination = root / "_linter_review" / "duplicates" / "b.txt"
    caption_destination.parent.mkdir(parents=True)
    caption_destination.write_text("existing caption review", encoding="utf-8")

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "0"},
    )

    assert applied.returncode != 0
    assert copy.is_file()
    assert (root / "b.txt").is_file()
    assert not (root / "_linter_review" / "duplicates" / "b.png").exists()
    assert caption_destination.read_text(encoding="utf-8") == "existing caption review"


@pytest.mark.parametrize("protected_kind", ["retained_caption", "outside_sentinel"])
def test_fix_plan_refuses_hard_linked_recovery_receipt_without_changing_target(
    tmp_path: Path, protected_kind: str
):
    root = tmp_path / "data"
    root.mkdir()
    _original, copy = _duplicate_pair(root)
    plan = tmp_path / "fix-plan.sh"
    _write_fix_plan(root, plan)
    receipt = root / "_linter_review" / "recovery-receipts.tsv"
    receipt.parent.mkdir()
    if protected_kind == "retained_caption":
        protected = root / "a.txt"
    else:
        protected = tmp_path / "outside-sentinel"
        protected.write_text("outside sentinel bytes", encoding="utf-8")
    before = protected.read_bytes()
    os.link(protected, receipt)

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "0"},
    )

    assert applied.returncode != 0
    assert "multiple hard links" in applied.stderr
    assert protected.read_bytes() == before
    assert copy.is_file()
    assert (root / "b.txt").is_file()
    assert not (root / "_linter_review" / "duplicates" / "b.png").exists()


def test_fix_plan_keeps_move_intent_when_receipt_identity_changes(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    _original, copy = _duplicate_pair(root)
    plan = tmp_path / "fix-plan.sh"
    _write_fix_plan(root, plan)
    receipt = root / "_linter_review" / "recovery-receipts.tsv"
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    counter = tmp_path / "mv-count"
    wrapper = wrapper_dir / "mv"
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "count=0",
                'if [[ -f "$MV_COUNT" ]]; then count=$(<"$MV_COUNT"); fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$MV_COUNT"',
                '/bin/mv "$@"',
                "status=$?",
                'if [[ "$count" == "2" && "$status" == "0" ]]; then '
                '/bin/mv "$RECEIPTS_TARGET" "$RECEIPTS_TARGET.replaced"; '
                'printf "%s" "replacement" > "$RECEIPTS_TARGET"; fi',
                'exit "$status"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o700)

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "0",
            "MV_COUNT": str(counter),
            "RECEIPTS_TARGET": str(receipt),
            "PATH": f"{wrapper_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )

    assert applied.returncode != 0
    assert not copy.exists()
    assert not (root / "b.txt").exists()
    assert (root / "_linter_review" / "duplicates" / "b.png").is_file()
    assert (root / "_linter_review" / "duplicates" / "b.txt").is_file()
    assert receipt.read_text(encoding="utf-8") == "replacement"
    journal = receipt.with_name(f"{receipt.name}.replaced").read_text(encoding="utf-8")
    assert "intent" in journal
    assert "moved" not in journal


def test_fix_plan_rolls_back_image_when_caption_move_fails(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    _original, copy = _duplicate_pair(root)
    plan = tmp_path / "fix-plan.sh"
    _write_fix_plan(root, plan)
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    counter = tmp_path / "mv-count"
    wrapper = wrapper_dir / "mv"
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "count=0",
                'if [[ -f "$MV_COUNT" ]]; then count=$(<"$MV_COUNT"); fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$MV_COUNT"',
                'if [[ "$count" == "2" ]]; then exit 1; fi',
                'exec /bin/mv "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o700)

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "0",
            "MV_COUNT": str(counter),
            "PATH": f"{wrapper_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )

    assert applied.returncode != 0
    assert copy.is_file()
    assert (root / "b.txt").is_file()
    assert not (root / "_linter_review" / "duplicates" / "b.png").exists()
    assert not (root / "_linter_review" / "duplicates" / "b.txt").exists()
    journal = (root / "_linter_review" / "recovery-receipts.tsv").read_text(encoding="utf-8")
    assert "intent" in journal
    assert "rolled_back" in journal


def test_fix_plan_marks_recovery_required_when_image_rollback_fails(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    _original, copy = _duplicate_pair(root)
    plan = tmp_path / "fix-plan.sh"
    _write_fix_plan(root, plan)
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    counter = tmp_path / "mv-count"
    wrapper = wrapper_dir / "mv"
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "count=0",
                'if [[ -f "$MV_COUNT" ]]; then count=$(<"$MV_COUNT"); fi',
                "count=$((count + 1))",
                'printf "%s" "$count" > "$MV_COUNT"',
                'if [[ "$count" == "2" || "$count" == "3" ]]; then exit 1; fi',
                'exec /bin/mv "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o700)

    applied = subprocess.run(
        ["bash", str(plan)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "0",
            "MV_COUNT": str(counter),
            "PATH": f"{wrapper_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )

    assert applied.returncode != 0
    assert not copy.exists()
    assert (root / "b.txt").is_file()
    destination = root / "_linter_review" / "duplicates" / "b.png"
    assert destination.is_file()
    assert not (root / "_linter_review" / "duplicates" / "b.txt").exists()
    journal = (root / "_linter_review" / "recovery-receipts.tsv").read_text(encoding="utf-8")
    assert "intent" in journal
    assert "recovery_required" in journal
    assert "rolled_back" not in journal
    assert str(destination) in journal
    assert str(copy) in journal
