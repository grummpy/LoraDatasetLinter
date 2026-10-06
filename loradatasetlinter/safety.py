"""Preflight and publish report outputs without mutating a dataset."""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loradatasetlinter.errors import LintError


@dataclass
class _StagedOutput:
    label: str
    target: Path
    staged: Path
    existed: bool
    backup: Path | None = None
    published: bool = False


def is_inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    parent = root.resolve()
    return resolved == parent or parent in resolved.parents


def preflight_outputs(
    dataset: Path,
    outputs: list[tuple[str, Path | None]],
    output_dir: Path | None,
) -> list[tuple[str, Path]]:
    """Resolve, type-check, and deconflict every requested file output."""
    dataset_ids = _dataset_file_ids(dataset)
    resolved_outputs: list[tuple[str, Path]] = []
    output_ids: dict[tuple[int, int], str] = {}
    seen_paths: dict[Path, str] = {}

    for label, output in outputs:
        if output is None:
            continue
        if output.is_symlink():
            raise LintError(f"Refusing to write {label} through a symlink: {output}.")
        target = output.resolve()
        _ensure_outside_dataset(dataset, output, target)
        _ensure_file_target(label, target)
        previous = seen_paths.get(target)
        if previous is not None:
            raise LintError(
                f"Refusing to use both {previous} and {label}: they resolve to the same output. "
                "Choose separate report and fix-plan paths."
            )
        identity = _file_identity(target)
        if identity is not None:
            if identity in dataset_ids:
                raise LintError(
                    f"Refusing to write {label}: {output} is hard-linked to a dataset file."
                )
            previous = output_ids.get(identity)
            if previous is not None:
                raise LintError(
                    f"Refusing to use both {previous} and {label}: "
                    "they name the same existing file."
                )
            output_ids[identity] = label
        seen_paths[target] = label
        resolved_outputs.append((label, target))

    _ensure_output_directory(dataset, output_dir, resolved_outputs)
    for index, (left_label, left) in enumerate(resolved_outputs):
        for right_label, right in resolved_outputs[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise LintError(
                    f"Refusing {left_label} and {right_label}: one output path contains the other."
                )
    return resolved_outputs


def publish_outputs(outputs: list[tuple[str, Path, str]]) -> None:
    """Stage all content first, then atomically replace all targets or restore them."""
    staged: list[_StagedOutput] = []
    try:
        for label, target, content in outputs:
            _ensure_file_target(label, target)
            _prepare_parent(target.parent)
            staged.append(
                _StagedOutput(
                    label=label,
                    target=target,
                    staged=_stage_one(target, content),
                    existed=target.exists(),
                )
            )
        for item in staged:
            if item.existed:
                item.backup = _reserve_backup(item.target)
                _replace(item.target, item.backup)
            _replace(item.staged, item.target)
            item.published = True
    except OSError:
        _rollback_outputs(staged)
        raise
    else:
        for item in staged:
            if item.backup is not None:
                _unlink_if_exists(item.backup)
    finally:
        for item in staged:
            _unlink_if_exists(item.staged)


def _ensure_outside_dataset(dataset: Path, original: Path, target: Path) -> None:
    if is_inside(target, dataset):
        raise LintError(
            f"Refusing to write {original} inside the dataset at {dataset}. "
            "The linter never modifies the dataset. Choose a path outside it."
        )


def _ensure_file_target(label: str, target: Path) -> None:
    if target.exists() and not target.is_file():
        raise LintError(f"Refusing to write {label}: {target} is not a regular file.")
    _ensure_parent(target.parent)


def _ensure_parent(parent: Path) -> None:
    for candidate in reversed((parent, *parent.parents)):
        if candidate.exists() and not candidate.is_dir():
            raise LintError(f"Output parent is not a directory: {candidate}")


def _prepare_parent(parent: Path) -> None:
    _ensure_parent(parent)
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise LintError(f"Output parent is not a directory: {parent}")


def _ensure_output_directory(
    dataset: Path,
    output_dir: Path | None,
    outputs: list[tuple[str, Path]],
) -> None:
    if output_dir is None:
        return
    directory = output_dir.resolve()
    _ensure_outside_dataset(dataset, output_dir, directory)
    if directory.exists() and not directory.is_dir():
        raise LintError(f"Refusing output directory {output_dir}: it is not a directory.")
    _ensure_parent(directory.parent)
    for label, target in outputs:
        if target == directory or target in directory.parents:
            raise LintError(
                f"Refusing output directory {output_dir}: it conflicts with {label} at {target}."
            )


def _dataset_file_ids(dataset: Path) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    for directory, _dirs, names in os.walk(dataset, followlinks=False):
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            identity = _file_identity(path)
            if identity is not None:
                identities.add(identity)
    return identities


def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        return None
    return metadata.st_dev, metadata.st_ino


def _stage_one(target: Path, content: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.linter-stage-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _reserve_backup(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.linter-backup-", dir=target.parent)
    os.close(descriptor)
    backup = Path(name)
    backup.unlink()
    return backup


def _replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _rollback_outputs(staged: list[_StagedOutput]) -> None:
    for item in reversed(staged):
        try:
            if item.backup is not None and item.backup.exists():
                _replace(item.backup, item.target)
            elif item.published:
                item.target.unlink(missing_ok=True)
        except OSError:
            # Preserve the original publish error; any surviving backup is evidence for recovery.
            continue


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
