"""Dataset checks. Each one returns findings and does not write files."""

from loradatasetlinter.checks.aspect import check_aspect
from loradatasetlinter.checks.captions import check_captions
from loradatasetlinter.checks.duplicates import check_duplicates
from loradatasetlinter.checks.files import check_files
from loradatasetlinter.checks.resolution import check_resolution
from loradatasetlinter.checks.stats import check_stats
from loradatasetlinter.models import Dataset, Finding
from loradatasetlinter.policy import Policy


def run_checks(dataset: Dataset, policy: Policy) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(check_duplicates(dataset, policy))
    findings.extend(check_resolution(dataset, policy))
    findings.extend(check_aspect(dataset, policy))
    findings.extend(check_files(dataset, policy))
    findings.extend(check_captions(dataset, policy))
    findings.extend(check_stats(dataset, policy))
    return findings
