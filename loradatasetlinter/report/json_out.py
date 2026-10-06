"""JSON report. Thumbnails stay in the HTML file so this stays small."""

from __future__ import annotations

import json
from pathlib import Path

from loradatasetlinter.models import Report


def render_json(report: Report) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n"


def write_json(report: Report, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_json(report), encoding="utf-8", newline="\n")
