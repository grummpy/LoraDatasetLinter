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
        "# Each image/caption pair is preflighted together before either file moves.",
        "# Applied units append shell-escaped intent and completion records to",
        "# _linter_review/recovery-receipts.tsv for manual recovery.",
        "set -euo pipefail",
        f"DATASET={_bash_quote(str(root))}",
        'DRY_RUN="${DRY_RUN:-1}"',
        'RECEIPTS="$DATASET/_linter_review/recovery-receipts.tsv"',
        "JOURNAL_READY=0",
        'JOURNAL_INODE=""',
        "fail() {",
        '  printf "%s\\n" "fix-plan: $1" >&2',
        "  exit 1",
        "}",
        "journal_inode() {",
        '  local value',
        '  # macOS /dev/fd reports a different device for the same open file.',
        '  if value="$(stat -L -f "%i" "$1" 2>/dev/null)"; then',
        '    printf "%s" "$value"',
        '  elif value="$(stat -Lc "%i" "$1" 2>/dev/null)"; then',
        '    printf "%s" "$value"',
        "  else",
        "    return 1",
        "  fi",
        "}",
        "journal_link_count() {",
        '  local value',
        '  if value="$(stat -L -f "%l" "$1" 2>/dev/null)"; then',
        '    printf "%s" "$value"',
        '  elif value="$(stat -Lc "%h" "$1" 2>/dev/null)"; then',
        '    printf "%s" "$value"',
        "  else",
        "    return 1",
        "  fi",
        "}",
        "append_component() {",
        '  if [[ "$1" == "/" ]]; then',
        '    REBUILT_PATH="/$2"',
        "  else",
        '    REBUILT_PATH="$1/$2"',
        "  fi",
        "}",
        "assert_dataset_root() {",
        '  [[ -d "$DATASET" && ! -L "$DATASET" ]] || fail "dataset root is missing or a symlink"',
        '  local remainder="${DATASET#/}" component current=""',
        '  while [[ -n "$remainder" ]]; do',
        '    component="${remainder%%/*}"',
        '    if [[ "$remainder" == */* ]]; then remainder="${remainder#*/}"; else remainder=""; fi',
        '    [[ -n "$component" && "$component" != "." && "$component" != ".." ]] '
        '|| fail "unsafe dataset path"',
        '    append_component "$current" "$component"',
        '    current="$REBUILT_PATH"',
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
        '    [[ -n "$component" && "$component" != "." && "$component" != ".." ]] '
        '|| fail "unsafe move path"',
        '    append_component "$current" "$component"',
        '    current="$REBUILT_PATH"',
        '    [[ ! -L "$current" ]] || fail "move path contains a symlink"',
        "  done",
        "}",
        "ensure_destination_parent() {",
        '  local target="$1" remainder component current="$DATASET"',
        '  assert_under_dataset "$target"',
        '  if [[ "$DATASET" == "/" ]]; then',
        '    remainder="${target#/}"',
        "  else",
        '    remainder="${target#"$DATASET"/}"',
        "  fi",
        '  case "$remainder" in',
        '    _linter_review/*) ;;',
        '    *) fail "destination is outside _linter_review" ;;',
        "  esac",
        '  remainder="${remainder%/*}"',
        '  while [[ -n "$remainder" ]]; do',
        '    component="${remainder%%/*}"',
        '    if [[ "$remainder" == */* ]]; then remainder="${remainder#*/}"; else remainder=""; fi',
        '    append_component "$current" "$component"',
        '    current="$REBUILT_PATH"',
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
        "prepare_journal() {",
        '  [[ "$JOURNAL_READY" == "1" ]] && return',
        '  local links fd_identity',
        "  assert_dataset_root",
        '  assert_under_dataset "$RECEIPTS"',
        '  ensure_destination_parent "$RECEIPTS"',
        '  if [[ -L "$RECEIPTS" || ( -e "$RECEIPTS" && ! -f "$RECEIPTS" ) ]]; then',
        '    fail "recovery receipt path is unsafe"',
        "  fi",
        '  if [[ -e "$RECEIPTS" ]]; then',
        '    links="$(journal_link_count "$RECEIPTS")" '
        '|| fail "could not inspect recovery receipts"',
        '    [[ "$links" == "1" ]] || fail "recovery receipt path has multiple hard links"',
        "  fi",
        '  exec 9>> "$RECEIPTS" || fail "could not prepare recovery receipts"',
        '  JOURNAL_INODE="$(journal_inode "$RECEIPTS")" '
        '|| fail "could not inspect recovery receipts"',
        '  fd_identity="$(journal_inode /dev/fd/9)" || fail "could not inspect open receipts"',
        '  links="$(journal_link_count "$RECEIPTS")" || fail "could not inspect recovery receipts"',
        '  [[ "$JOURNAL_INODE" == "$fd_identity" && "$links" == "1" ]] '
        '|| fail "recovery receipt identity changed or has multiple hard links"',
        '  JOURNAL_READY=1',
        '  write_journal "plan" "$DATASET" "" "" "" '
        '|| fail "could not write recovery receipt"',
        "}",
        "verify_journal() {",
        '  local path_identity fd_identity links',
        '  [[ "$JOURNAL_READY" == "1" && -f "$RECEIPTS" && ! -L "$RECEIPTS" ]] || return 1',
        '  path_identity="$(journal_inode "$RECEIPTS")" || return 1',
        '  fd_identity="$(journal_inode /dev/fd/9)" || return 1',
        '  links="$(journal_link_count "$RECEIPTS")" || return 1',
        '  [[ "$path_identity" == "$JOURNAL_INODE" && "$fd_identity" == "$JOURNAL_INODE" ]] '
        '&& [[ "$links" == "1" ]]',
        "}",
        "write_journal() {",
        '  verify_journal || return 1',
        '  printf "%s\\t%q\\t%q\\t%q\\t%q\\n" "$1" "$2" "$3" "$4" "$5" >&9',
        "}",
        "preflight_move() {",
        '  assert_under_dataset "$1"',
        '  assert_under_dataset "$2"',
        '  case "$2" in',
        '    "$DATASET"/_linter_review/*) ;;',
        '    *) fail "destination is outside _linter_review" ;;',
        "  esac",
        '  [[ -f "$1" && ! -L "$1" ]] || fail "source is missing or unsafe"',
        '  [[ ! -e "$2" && ! -L "$2" ]] '
        '|| fail "destination already exists; refusing to overwrite it"',
        '  ensure_destination_parent "$2"',
        "}",
        "move_one() {",
        '  mv -n "$1" "$2" || return 1',
        '  [[ ! -e "$1" && ! -L "$1" && -f "$2" && ! -L "$2" ]]',
        "}",
        "rollback_one() {",
        '  assert_under_dataset "$1"',
        '  assert_under_dataset "$2"',
        '  if [[ ! -e "$1" && ! -L "$1" ]]; then return 0; fi',
        '  [[ -f "$1" && ! -e "$2" && ! -L "$2" ]] || return 1',
        '  mv -n "$1" "$2" || return 1',
        '  [[ -f "$2" && ! -e "$1" && ! -L "$1" ]]',
        "}",
        "move_unit() {",
        '  if [[ "$DRY_RUN" == "1" ]]; then',
        '    printf "would move %s -> %s\\n" "$1" "$2"',
        '    if [[ "$#" == "4" ]]; then printf "would move %s -> %s\\n" "$3" "$4"; fi',
        "    return",
        "  fi",
        '  [[ "$#" == "2" || "$#" == "4" ]] || fail "invalid move unit"',
        "  prepare_journal",
        '  preflight_move "$1" "$2"',
        '  if [[ "$#" == "4" ]]; then preflight_move "$3" "$4"; fi',
        '  write_journal "intent" "$1" "$2" "${3:-}" "${4:-}" '
        '|| fail "could not record move intent"',
        '  if ! move_one "$1" "$2"; then',
        '    fail "image move failed; inspect the recovery receipt"',
        "  fi",
        '  if [[ "$#" == "4" ]] && ! move_one "$3" "$4"; then',
        '    local caption_restored=1 image_restored=1',
        '    rollback_one "$4" "$3" || caption_restored=0',
        '    rollback_one "$2" "$1" || image_restored=0',
        '    if [[ "$caption_restored" == "1" && "$image_restored" == "1" ]]; then',
        '      write_journal "rolled_back" "$1" "$2" "$3" "$4" '
        '|| fail "pair restoration verified but receipt finalization failed"',
        '      fail "caption move failed; pair restoration was verified"',
        "    fi",
        '    write_journal "recovery_required" "$2" "$1" "$4" "$3" '
        '|| fail "caption move and rollback failed; journal intent remains for recovery"',
        '    fail "caption move and rollback failed; inspect recovery_required receipt"',
        "  fi",
        '  write_journal "moved" "$1" "$2" "${3:-}" "${4:-}" '
        '|| fail "move completed but receipt finalization failed; inspect move intent"',
        "}",
        "",
    ]
    scheduled: set[str] = set()
    scheduled_images: set[str] = set()
    image_moves: list[tuple[str, str, str]] = []
    orphan_moves: list[tuple[str, str, str]] = []
    image_captions, caption_owners = _caption_owners(report)
    moves = 0

    def add_unit(rel: str, bucket: str, why: str, caption: str | None = None) -> None:
        nonlocal moves
        if rel in scheduled:
            return
        scheduled.add(rel)
        destination = Path("_linter_review") / bucket / rel
        command = f"move_unit {_bash_quote(str(root / rel))} {_bash_quote(str(root / destination))}"
        if caption is not None:
            if caption in scheduled:
                lines.append(_comment(f"caption {caption} was already scheduled; left in place"))
            else:
                scheduled.add(caption)
                caption_destination = Path("_linter_review") / bucket / caption
                command += (
                    f" {_bash_quote(str(root / caption))}"
                    f" {_bash_quote(str(root / caption_destination))}"
                )
        lines.append(_comment(why))
        lines.append(command)
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
        caption = image_captions.get(rel)
        owners = caption_owners.get(caption, set()) if caption is not None else set()
        if caption is None or len(owners) == 1:
            add_unit(rel, bucket, why, caption)
        else:
            add_unit(rel, bucket, why)
            if caption not in flagged_captions:
                owner_list = ", ".join(sorted(owners))
                lines.append(
                    _comment(
                        f"shared caption {caption} is used by {owner_list}; "
                        "left in place for manual review"
                    )
                )
                flagged_captions.add(caption)
    for rel, bucket, why in orphan_moves:
        add_unit(rel, bucket, why)
    if moves == 0:
        lines.append('printf "No file moves suggested.\\n"')
    lines.append("")
    return "\n".join(lines)


def write_fix_plan(report: Report, dataset: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_fix_plan(report, dataset), encoding="utf-8", newline="\n")
