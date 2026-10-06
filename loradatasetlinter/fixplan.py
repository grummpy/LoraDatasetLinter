"""A reviewable move/rename script. The linter never executes it."""

from __future__ import annotations

import json
from pathlib import Path

from loradatasetlinter.models import Report


def _bash_quote(value: str) -> str:
    """Return one physical, Bash ANSI-C-quoted line for an arbitrary path."""
    escaped: list[str] = []
    for byte in value.encode("utf-8", "surrogateescape"):
        if 32 <= byte <= 126 and byte not in {ord("'"), ord("\\")}:
            escaped.append(chr(byte))
        elif byte == ord("'"):
            escaped.append("\\'")
        elif byte == ord("\\"):
            escaped.append("\\\\")
        else:
            escaped.append(f"\\x{byte:02x}")
    return "$'" + "".join(escaped) + "'"


def _comment(text: str) -> str:
    """Keep untrusted filenames in a single non-executable shell comment line."""
    return "# " + json.dumps(text, ensure_ascii=True)


def _caption_owners(report: Report) -> tuple[dict[str, str], dict[str, set[str]]]:
    image_captions: dict[str, str] = {}
    owners: dict[str, set[str]] = {}
    for image in report.images:
        rel = image.get("file")
        caption = image.get("caption")
        if not isinstance(rel, str) or not isinstance(caption, str):
            continue
        image_captions[rel] = caption
        owners.setdefault(caption, set()).add(rel)
    return image_captions, owners


def render_fix_plan(report: Report, dataset: Path) -> str:
    root = dataset.resolve()
    lines = [
        "#!/usr/bin/env bash",
        "# Suggested by LoRA Dataset Linter. Review every line before you run it.",
        "# The linter did not move or rename anything.",
        "# Dry run is the default. Apply with: DRY_RUN=0 bash fix-plan.sh",
        "# Applied moves append shell-escaped source and destination paths to",
        "# _linter_review/recovery-receipts.tsv for manual recovery.",
        "set -euo pipefail",
        f"DATASET={_bash_quote(str(root))}",
        'DRY_RUN="${DRY_RUN:-1}"',
        'RECEIPTS="$DATASET/_linter_review/recovery-receipts.tsv"',
        "fail() {",
        '  printf "%s\\n" "fix-plan: $1" >&2',
        "  exit 1",
        "}",
        "append_component() {",
        '  if [[ "$1" == "/" ]]; then',
        '    printf "/%s" "$2"',
        "  else",
        '    printf "%s/%s" "$1" "$2"',
        "  fi",
        "}",
        "assert_dataset_root() {",
        '  [[ -d "$DATASET" && ! -L "$DATASET" ]] || fail "dataset root is missing or a symlink"',
        '  local remainder="${DATASET#/}" component current=""',
        '  while [[ -n "$remainder" ]]; do',
        '    component="${remainder%%/*}"',
        '    if [[ "$remainder" == */* ]]; then remainder="${remainder#*/}"; else remainder=""; fi',
        '    [[ -n "$component" && "$component" != "." && "$component" != ".." ]]'
        ' || fail "unsafe dataset path"',
        '    current="$(append_component "$current" "$component")"',
        '    [[ ! -L "$current" ]] || fail "dataset path contains a symlink"',
        "  done",
        "}",
        "assert_under_dataset() {",
        '  local target="$1" remainder component current="$DATASET"',
        '  if [[ "$DATASET" == "/" ]]; then',
        '    [[ "$target" == /* ]] || fail "move path is outside the dataset"',
        '    remainder="${target#/}"',
        "  else",
        '    case "$target" in',
        '      "$DATASET"/*) remainder="${target#"$DATASET"/}" ;;',
        '      *) fail "move path is outside the dataset" ;;',
        "    esac",
        "  fi",
        '  [[ -n "$remainder" ]] || fail "move path is the dataset root"',
        '  while [[ -n "$remainder" ]]; do',
        '    component="${remainder%%/*}"',
        '    if [[ "$remainder" == */* ]]; then remainder="${remainder#*/}"; else remainder=""; fi',
        '    [[ -n "$component" && "$component" != "." && "$component" != ".." ]]'
        ' || fail "unsafe move path"',
        '    current="$(append_component "$current" "$component")"',
        '    [[ ! -L "$current" ]] || fail "move path contains a symlink"',
        "  done",
        "}",
        "ensure_destination_parent() {",
        '  local target="$1" remainder component current="$DATASET"',
        '  assert_under_dataset "$target"',
        '  if [[ "$DATASET" == "/" ]]; then remainder="${target#/}"; '
        'else remainder="${target#"$DATASET"/}"; fi',
        '  case "$remainder" in _linter_review/*) ;; *) '
        'fail "destination is outside _linter_review" ;; esac',
        '  remainder="${remainder%/*}"',
        '  while [[ -n "$remainder" ]]; do',
        '    component="${remainder%%/*}"',
        '    if [[ "$remainder" == */* ]]; then remainder="${remainder#*/}"; else remainder=""; fi',
        '    current="$(append_component "$current" "$component")"',
        '    if [[ -L "$current" ]]; then fail "destination parent contains a symlink"; fi',
        '    if [[ -e "$current" ]]; then',
        '      [[ -d "$current" ]] || fail "destination parent is not a directory"',
        "    else",
        '      mkdir "$current" || fail "could not create a destination directory"',
        "    fi",
        '    [[ -d "$current" && ! -L "$current" ]] '
        '|| fail "destination parent changed unexpectedly"',
        "  done",
        "}",
        "write_receipt() {",
        '  assert_under_dataset "$RECEIPTS"',
        '  ensure_destination_parent "$RECEIPTS"',
        '  if [[ -L "$RECEIPTS" || ( -e "$RECEIPTS" && ! -f "$RECEIPTS" ) ]]; then',
        '    fail "recovery receipt path is unsafe"',
        "  fi",
        '  printf "moved\\t%q\\t%q\\n" "$1" "$2" >> "$RECEIPTS" '
        '|| fail "could not write recovery receipt"',
        "}",
        "move() {",
        '  if [[ "$DRY_RUN" == "1" ]]; then',
        '    printf "would move %s -> %s\\n" "$1" "$2"',
        "    return",
        "  fi",
        "  assert_dataset_root",
        '  assert_under_dataset "$1"',
        '  assert_under_dataset "$2"',
        '  case "$2" in "$DATASET"/_linter_review/*) ;; *) '
        'fail "destination is outside _linter_review" ;; esac',
        '  [[ -f "$1" && ! -L "$1" ]] || fail "source is missing or unsafe"',
        '  [[ ! -e "$2" && ! -L "$2" ]] '
        '|| fail "destination already exists; refusing to overwrite it"',
        '  ensure_destination_parent "$2"',
        '  mv -n "$1" "$2"',
        '  [[ ! -e "$1" && ! -L "$1" ]] || fail "move did not complete; no receipt was written"',
        '  [[ -e "$2" && ! -L "$2" ]] || fail "move destination is missing or unsafe"',
        '  write_receipt "$1" "$2"',
        "}",
        "",
    ]
    scheduled: set[str] = set()
    scheduled_images: set[str] = set()
    image_moves: list[tuple[str, str, str]] = []
    orphan_moves: list[tuple[str, str, str]] = []
    image_captions, caption_owners = _caption_owners(report)
    moves = 0

    def add_move(rel: str, bucket: str, why: str) -> None:
        nonlocal moves
        if rel in scheduled:
            return
        scheduled.add(rel)
        destination = Path("_linter_review") / bucket / rel
        src = _bash_quote(str(root / rel))
        dest = _bash_quote(str(root / destination))
        lines.append(_comment(why))
        lines.append(f"move {src} {dest}")
        moves += 1

    def schedule_image(rel: str, bucket: str, why: str) -> None:
        if rel in scheduled_images:
            return
        scheduled_images.add(rel)
        image_moves.append((rel, bucket, why))

    for finding in report.findings:
        if finding.code in {"exact_duplicate", "exact_pixels"} and len(finding.files) >= 2:
            keep = finding.details.get("keep", finding.files[0])
            lines.append(_comment(f"keep {keep}"))
            for rel in finding.files:
                if rel != keep:
                    schedule_image(rel, "duplicates", f"{finding.code}: duplicate of {keep}")
        elif finding.code == "low_resolution":
            for rel in finding.files:
                schedule_image(rel, "low_resolution", "below the minimum side")
        elif finding.code == "corrupt_file":
            for rel in finding.files:
                schedule_image(rel, "unreadable", "unreadable file")
        elif finding.code == "orphan_caption":
            for rel in finding.files:
                orphan_moves.append((rel, "orphans", "caption with no image"))
        elif finding.code == "near_duplicate":
            lines.append(
                _comment(
                    "near-duplicate, left in place (confirm they are not intentional variations): "
                    + ", ".join(finding.files)
                )
            )
        elif finding.code == "missing_caption":
            for rel in finding.files:
                message = f"missing caption: create {root / Path(rel).with_suffix('.txt')}"
                lines.append(_comment(message))

    flagged_captions: set[str] = set()
    for rel, bucket, why in image_moves:
        add_move(rel, bucket, why)
        caption = image_captions.get(rel)
        if caption is None:
            continue
        owners = caption_owners.get(caption, set())
        if len(owners) == 1:
            add_move(caption, bucket, f"caption paired with {rel}")
        elif caption not in flagged_captions:
            owner_list = ", ".join(sorted(owners))
            lines.append(
                _comment(
                    f"shared caption {caption} is used by {owner_list}; "
                    "left in place for manual review"
                )
            )
            flagged_captions.add(caption)
    for rel, bucket, why in orphan_moves:
        add_move(rel, bucket, why)
    if moves == 0:
        lines.append('printf "No file moves suggested.\\n"')
    lines.append("")
    return "\n".join(lines)


def write_fix_plan(report: Report, dataset: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_fix_plan(report, dataset), encoding="utf-8", newline="\n")
