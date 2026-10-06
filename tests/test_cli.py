import json
import os
import subprocess
from pathlib import Path

from loradatasetlinter.cli import app
from tests.fixtures import caption, clean_dataset, gradient, snapshot
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
