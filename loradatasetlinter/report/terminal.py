"""Plain-text summary for a terminal. No colour codes, so logs stay readable."""

from __future__ import annotations

from loradatasetlinter.models import Report

_LIMIT = 80


def render_terminal(report: Report) -> str:
    stats = report.stats
    steps = stats.get("steps", {})
    lines = [
        "LoRA Dataset Linter",
        f"dataset: {report.dataset}",
        f"status: {report.score.status.upper()}",
        f"score: {report.score.value}",
        (
            f"images: {stats.get('readable_images', 0)} readable"
            f" / {stats.get('images', 0)} files"
            f"  captions: {stats.get('captions', 0)}"
            f"  orphans: {stats.get('orphan_captions', 0)}"
            f"  unreadable: {stats.get('unreadable', 0)}"
        ),
        (
            "findings: "
            f"{report.score.counts.get('fail', 0)} fail, "
            f"{report.score.counts.get('warn', 0)} warn, "
            f"{report.score.counts.get('info', 0)} info"
        ),
        (
            "steps: "
            f"{steps.get('steps', 0)} "
            f"({steps.get('formula', '')}; "
            f"weighted images {steps.get('weighted_images', 0)}, "
            f"epochs {steps.get('epochs', 0)}, "
            f"batch {steps.get('batch_size', 0)}, "
            f"naive product {steps.get('naive_steps', 0)})"
        ),
    ]
    buckets = stats.get("buckets", {})
    for base, rows in buckets.items():
        if not rows:
            lines.append(f"buckets @{base}: (none)")
            continue
        parts = [f"{row['resolution'][0]}x{row['resolution'][1]} x{row['count']}" for row in rows]
        lines.append(f"buckets @{base}: " + ", ".join(parts))
    lines.append("")
    shown = report.findings[:_LIMIT]
    if not shown:
        lines.append("No findings.")
    for finding in shown:
        files = ", ".join(finding.files) if finding.files else "(dataset)"
        lines.append(f"[{finding.severity}] {finding.code}  {files}")
        lines.append(f"    {finding.reason}")
    remaining = len(report.findings) - len(shown)
    if remaining > 0:
        lines.append(f"... {remaining} more findings are in the JSON and HTML reports.")
    lines.append("")
    return "\n".join(lines)
