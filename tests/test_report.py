import json
import re
from pathlib import Path

from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.report.html_out import render_html
from loradatasetlinter.report.json_out import render_json
from loradatasetlinter.report.terminal import render_terminal
from tests.fixtures import clean_dataset, relax


def test_terminal_json_and_html_agree(tmp_path: Path):
    root = clean_dataset(tmp_path / "data")
    policy = relax()
    report = scan_dataset(root, policy)
    text = render_terminal(report)
    assert f"status: {report.score.status.upper()}" in text
    assert f"score: {report.score.value}" in text
    payload = json.loads(render_json(report))
    assert payload["status"] == report.score.status
    assert payload["score"] == report.score.value
    assert payload["tool"] == "lora-dataset-linter"
    assert {row["file"] for row in payload["images"]} == {
        "checker.png",
        "circle.png",
        "diagonal.png",
        "gradient.png",
    }
    assert "thumbnail" not in payload["images"][0]
    html = render_html(report, root, thumbnail_px=32)
    assert 'id="status"' in html
    assert report.score.status.upper() in html
    assert html.count("data:image/jpeg;base64,") == 4
    stripped = re.sub(r"data:image/jpeg;base64,[A-Za-z0-9+/=]+", "", html)
    assert "http://" not in stripped
    assert "https://" not in stripped
    assert 'src="http' not in stripped
    assert "@import" not in stripped
    assert report.score.status == "pass"
    assert report.score.value == 100


def test_html_has_no_thumbnail_when_disabled(tmp_path: Path):
    root = clean_dataset(tmp_path / "data")
    report = scan_dataset(root, relax())
    html = render_html(report, root, thumbnail_px=0)
    assert "data:image" not in html
