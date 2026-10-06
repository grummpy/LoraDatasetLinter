"""Self-contained HTML report. Thumbnails are embedded; nothing is fetched."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from jinja2 import Environment
from PIL import Image, ImageOps

from loradatasetlinter import __version__
from loradatasetlinter.models import Report

_TEMPLATE = Path(__file__).with_name("templates") / "report.html.j2"


def write_html(report: Report, destination: Path, dataset: Path, thumbnail_px: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_html(report, dataset, thumbnail_px),
        encoding="utf-8",
        newline="\n",
    )


def render_html(report: Report, dataset: Path, thumbnail_px: int) -> str:
    images = []
    for row in report.images:
        images.append(
            {
                **row,
                "thumbnail": _thumbnail(dataset / row["file"], thumbnail_px)
                if row.get("error") is None
                else "",
                "bucket_text": ", ".join(
                    f"{base}:{wh[0]}x{wh[1]}" for base, wh in (row.get("buckets") or {}).items()
                ),
            }
        )
    env = Environment(autoescape=True)
    template = env.from_string(_TEMPLATE.read_text(encoding="utf-8"))
    return template.render(
        version=__version__,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        report=report,
        stats=report.stats,
        steps=report.stats.get("steps", {}),
        images=images,
        findings=report.findings,
        bucket_sections=_bucket_sections(report.stats.get("buckets", {})),
        tags=(report.stats.get("tag_frequency") or [])[:40],
        classes=(report.stats.get("steps") or {}).get("classes", []),
    )


def _bucket_sections(histograms: dict) -> list[dict]:
    sections = []
    for base, rows in histograms.items():
        peak = max((row["count"] for row in rows), default=0)
        bars = [
            {
                "label": f"{row['resolution'][0]}x{row['resolution'][1]}",
                "count": row["count"],
                "pct": int(round(100 * row["count"] / peak)) if peak else 0,
            }
            for row in rows
        ]
        sections.append({"base": base, "bars": bars})
    return sections


def _thumbnail(path: Path, pixels: int) -> str:
    if pixels <= 0 or not path.is_file():
        return ""
    try:
        with Image.open(path) as image:
            image.load()
            frame = ImageOps.exif_transpose(image)
            if frame.mode in {"RGBA", "LA", "PA"}:
                background = Image.new("RGB", frame.size, (7, 11, 22))
                rgba = frame.convert("RGBA")
                background.paste(rgba, mask=rgba.getchannel("A"))
                frame = background
            else:
                frame = frame.convert("RGB")
            frame.thumbnail((pixels, pixels), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            frame.save(buffer, format="JPEG", quality=70)
    except (OSError, ValueError):
        return ""
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
