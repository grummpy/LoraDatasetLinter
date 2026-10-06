"""Command line entry. Scanning is read-only; reports are written outside the dataset."""

from __future__ import annotations

from pathlib import Path

import typer

from loradatasetlinter import __version__
from loradatasetlinter.engine import scan_dataset
from loradatasetlinter.errors import LintError
from loradatasetlinter.fixplan import render_fix_plan
from loradatasetlinter.policy import apply_overrides, load_policy
from loradatasetlinter.report.html_out import render_html
from loradatasetlinter.report.json_out import render_json
from loradatasetlinter.report.terminal import render_terminal
from loradatasetlinter.safety import preflight_outputs, publish_outputs

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Check a LoRA image folder before a ComfyUI or kohya training run. CPU only.",
)


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the version and exit.",
        callback=_version,
        is_eager=True,
    ),
) -> None:
    """Local dataset linter. It does not train, and it does not contact a GPU or ComfyUI."""


@app.command()
def scan(
    dataset: Path = typer.Argument(..., help="Folder of images and same-name .txt captions."),
    policy: Path | None = typer.Option(None, "--policy", "-p", help="YAML policy file."),
    json_out: Path | None = typer.Option(None, "--json", help="Write the JSON report here."),
    html_out: Path | None = typer.Option(None, "--html", help="Write the HTML report here."),
    fix_plan: Path | None = typer.Option(
        None,
        "--fix-plan",
        help="Write a suggested move/rename script. It is not executed.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Write report.json and report.html into this directory, outside the dataset.",
    ),
    min_side: int | None = typer.Option(None, "--min-side", help="Override resolution.min_side."),
    hamming: int | None = typer.Option(
        None,
        "--hamming",
        help="Override the perceptual-hash Hamming threshold.",
    ),
    base_resolution: int | None = typer.Option(
        None,
        "--base-resolution",
        help="Use this single kohya bucket base instead of the policy list.",
    ),
    epochs: int | None = typer.Option(None, "--epochs", help="Epochs in the step estimate."),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Batch size in the step estimate."
    ),
    trigger: str | None = typer.Option(
        None,
        "--trigger",
        help="Comma-separated trigger words that should appear in every caption.",
    ),
    hash_method: str | None = typer.Option(None, "--hash-method", help="phash or dhash."),
    thumbnail_px: int | None = typer.Option(
        None,
        "--thumbnail-px",
        help="HTML thumbnail max side. 0 skips thumbnails.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Skip the terminal summary."),
) -> None:
    """Lint a dataset and print a summary. Exit 0 pass, 1 warn, 2 fail, 3 error."""
    try:
        loaded = load_policy(policy)
        loaded = apply_overrides(
            loaded,
            min_side=min_side,
            hamming=hamming,
            base_resolution=base_resolution,
            epochs=epochs,
            batch_size=batch_size,
            trigger=trigger,
            hash_method=hash_method,
            thumbnail_px=thumbnail_px,
        )
        json_path = json_out
        html_path = html_out
        if output_dir is not None:
            json_path = json_path or (output_dir / "report.json")
            html_path = html_path or (output_dir / "report.html")
        root = dataset.expanduser().resolve()
        targets = dict(
            preflight_outputs(
                root,
                [
                    ("JSON report", json_path),
                    ("HTML report", html_path),
                    ("fix plan", fix_plan),
                ],
                output_dir,
            )
        )
        report = scan_dataset(root, loaded)
        rendered: list[tuple[str, Path, str]] = []
        if json_path is not None:
            rendered.append(("JSON report", targets["JSON report"], render_json(report)))
        if html_path is not None:
            rendered.append(
                (
                    "HTML report",
                    targets["HTML report"],
                    render_html(report, root, loaded.output.thumbnail_px),
                )
            )
        if fix_plan is not None:
            rendered.append(("fix plan", targets["fix plan"], render_fix_plan(report, root)))
        if rendered:
            publish_outputs(rendered)
        if not quiet:
            typer.echo(render_terminal(report))
    except LintError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=3) from exc
    except OSError as exc:
        typer.echo(f"Could not write the report: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    codes = {"pass": 0, "warn": 1, "fail": 2}
    raise typer.Exit(code=codes[report.score.status])
