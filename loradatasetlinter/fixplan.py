"""A reviewable move/rename script. The linter never executes it."""

from __future__ import annotations

import shlex
from pathlib import Path

from loradatasetlinter.models import Report


def render_fix_plan(report: Report, dataset: Path) -> str:
    root = dataset.resolve()
    lines = [
        "#!/usr/bin/env bash",
        "# Suggested by LoRA Dataset Linter. Review every line before you run it.",
        "# The linter did not move or rename anything.",
        "# Dry run is the default. Apply with: DRY_RUN=0 bash fix-plan.sh",
        "set -euo pipefail",
        f"DATASET={shlex.quote(str(root))}",
        'DRY_RUN="${DRY_RUN:-1}"',
        "move() {",
        '  if [[ "$DRY_RUN" == "1" ]]; then',
        '    printf "would move %s -> %s\\n" "$1" "$2"',
        "  else",
        '    mkdir -p "$(dirname -- "$2")"',
        '    mv -- "$1" "$2"',
        "  fi",
        "}",
        "",
    ]
    scheduled: set[str] = set()
    moves = 0

    def add_move(rel: str, bucket: str, why: str) -> None:
        nonlocal moves
        if rel in scheduled:
            return
        scheduled.add(rel)
        destination = Path("_linter_review") / bucket / rel
        src = shlex.quote(str(root / rel))
        dest = shlex.quote(str(root / destination))
        lines.append(f"# {why}")
        lines.append(f"move {src} {dest}")
        moves += 1

    for finding in report.findings:
        if finding.code in {"exact_duplicate", "exact_pixels"} and len(finding.files) >= 2:
            keep = finding.details.get("keep", finding.files[0])
            lines.append(f"# keep {keep}")
            for rel in finding.files:
                if rel == keep:
                    continue
                add_move(rel, "duplicates", f"{finding.code}: duplicate of {keep}")
        elif finding.code == "low_resolution":
            for rel in finding.files:
                add_move(rel, "low_resolution", "below the minimum side")
        elif finding.code == "corrupt_file":
            for rel in finding.files:
                add_move(rel, "unreadable", "unreadable file")
        elif finding.code == "orphan_caption":
            for rel in finding.files:
                add_move(rel, "orphans", "caption with no image")
        elif finding.code == "near_duplicate":
            lines.append(
                "# near-duplicate, left in place (confirm they are not intentional variations): "
                + ", ".join(finding.files)
            )
        elif finding.code == "missing_caption":
            for rel in finding.files:
                caption_path = shlex.quote(str(root / Path(rel).with_suffix(".txt")))
                lines.append(f"# missing caption: create {caption_path}")
    if moves == 0:
        lines.append('printf "No file moves suggested.\\n"')
    lines.append("")
    return "\n".join(lines)


def write_fix_plan(report: Report, dataset: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_fix_plan(report, dataset), encoding="utf-8", newline="\n")
