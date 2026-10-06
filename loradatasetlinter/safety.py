"""Output paths stay outside the dataset so a lint run cannot rewrite it."""

from __future__ import annotations

from pathlib import Path

from loradatasetlinter.errors import LintError


def is_inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    parent = root.resolve()
    return resolved == parent or parent in resolved.parents


def ensure_outputs_outside(dataset: Path, outputs: list[Path | None]) -> None:
    for output in outputs:
        if output is None:
            continue
        if is_inside(output, dataset):
            raise LintError(
                f"Refusing to write {output} inside the dataset at {dataset}. "
                "The linter never modifies the dataset. Choose a path outside it."
            )
