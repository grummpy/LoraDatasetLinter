"""Readiness score. Failures dominate warnings, and the numeric gates can escalate."""

from __future__ import annotations

from loradatasetlinter.models import Finding, Score
from loradatasetlinter.policy import Policy


def decide(findings: list[Finding], policy: Policy) -> Score:
    counts = {"fail": 0, "warn": 0, "info": 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    penalty = sum(policy.score.weights[finding.severity] for finding in findings)
    value = max(0, min(100, 100 - penalty))
    status = "pass"
    if counts["warn"]:
        status = "warn"
    if counts["fail"]:
        status = "fail"
    if value < policy.score.warn_min:
        status = "fail"
    elif value < policy.score.pass_min and status == "pass":
        status = "warn"
    return Score(value=value, status=status, counts=counts)
