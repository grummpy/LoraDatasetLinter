import json
import os
import subprocess
from pathlib import Path

import pytest
from loradatasetlinter.cli import app
from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.fixplan import render_fix_plan
from loradatasetlinter.models import Finding
from tests.fixtures import caption, clean_dataset, gradient, relax, snapshot
from typer.testing import CliRunner

runner = CliRunner()


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
    assert "same output" in result.output
    assert marker.read_text(encoding="utf-8") == "keep"


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
