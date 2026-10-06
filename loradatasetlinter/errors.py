"""Errors the CLI turns into a non-zero exit without touching the dataset."""


class LintError(Exception):
    """The dataset could not be linted, or an output path was refused."""


class PolicyError(LintError):
    """The policy file is missing a required shape or names an unknown key."""
