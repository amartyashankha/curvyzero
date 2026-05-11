"""Check environment docs for fidelity overclaims and count dashboards."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC_PATHS = (
    REPO_ROOT / "docs" / "working" / "environment" / "active_lanes.md",
    REPO_ROOT / "docs" / "working" / "environment" / "coverage_tracker.md",
    REPO_ROOT / "docs" / "working" / "environment" / "measurement_critique_2026-05-09.md",
    REPO_ROOT / "docs" / "working" / "environment" / "reorientation_packet.md",
    REPO_ROOT / "docs" / "working" / "environment" / "full_fidelity_execution_plan.md",
    REPO_ROOT / "docs" / "working" / "environment" / "full_fidelity_spec_matrix_2026-05-09.md",
    REPO_ROOT / "docs" / "working" / "environment" / "spec_backlog_2026-05-09.md",
)
TEXT_SUFFIXES = {".md", ".markdown", ".rst", ".txt"}

OVERCLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "full fidelity overclaim",
        re.compile(
            r"(?i)\b(?:full|complete)\s+(?:curvytron\s+)?(?:environment\s+)?"
            r"fidelity\s+(?:achieved|complete|done|ready)\b"
        ),
    ),
    (
        "training-ready overclaim",
        re.compile(
            r"(?i)\b(?:training[- ]ready|ready\s+for\s+(?:source[- ]faithful\s+)?training)\b"
        ),
    ),
    (
        "vector-ready overclaim",
        re.compile(r"(?i)\bvector[- ]ready\b"),
    ),
)
NEGATED_CLAIM = re.compile(r"(?i)\b(?:not|no|does\s+not|do\s+not|isn't|aren't)\b")
DASHBOARD_COUNT = re.compile(
    r"(?i)(?:"
    r"^\s*\|[^|]*(?:pytest|ruff|test|count|coverage|check|vector|source/env|speed)[^|]*\|"
    r"[^|]*(?:\d+\s+passed|all checks passed|passed:\d+|\d+\s+pass\b|\d+(?:\.\d+)?%)"
    r"|^\s*(?:[-*]\s*)?(?:full pytest|focused .*pytest|full repo ruff|"
    r"current counts?|hygiene counts?|latest known verification|test counts?|"
    r"standalone tests?|coverage)"
    r".*(?:\d+\s+passed|all checks passed|passed:\d+|\d+\s+pass\b|\d+(?:\.\d+)?%)"
    r"|^\s*(?:[-*]\s*)?(?:tests?|checks?)\s*:"
    r".*(?:\d+\s+passed|all checks passed|passed:\d+|\d+\s+pass\b)"
    r")"
)
ACCEPTANCE_COMMAND = re.compile(r"\b(?:uv run|python3?|tools/|scripts/)\b")
GENERIC_DASHBOARD_CONTEXT = re.compile(
    r"(?i)\b(?:full pytest|focused .*pytest|full repo ruff|current counts?|"
    r"hygiene counts?|latest known verification|test counts?|standalone tests?|"
    r"coverage)\b"
)


@dataclass(frozen=True, slots=True)
class ClaimIssue:
    path: Path
    line_number: int
    label: str
    line: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    files_scanned: int
    issues: tuple[ClaimIssue, ...]


def iter_doc_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path
            continue
        if not path.is_dir():
            continue
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES:
                yield candidate


def scan_paths(paths: Iterable[Path]) -> ScanResult:
    issues: list[ClaimIssue] = []
    files_scanned = 0
    for path in iter_doc_files(paths):
        files_scanned += 1
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        previous_line = ""
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            claim_context = f"{previous_line} {stripped}".strip()
            for label, pattern in OVERCLAIM_PATTERNS:
                if pattern.search(stripped) and not NEGATED_CLAIM.search(claim_context):
                    issues.append(ClaimIssue(path, line_number, label, stripped))
            if DASHBOARD_COUNT.search(stripped) and (
                GENERIC_DASHBOARD_CONTEXT.search(stripped)
                or not ACCEPTANCE_COMMAND.search(stripped)
            ):
                issues.append(
                    ClaimIssue(path, line_number, "standalone status-count dashboard", stripped)
                )
            previous_line = stripped

    return ScanResult(files_scanned=files_scanned, issues=tuple(issues))


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    for base in (Path.cwd().resolve(), REPO_ROOT):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            pass
    return str(resolved)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    noun = singular if count == 1 else plural or f"{singular}s"
    return f"{count} {noun}"


def _line_excerpt(line: str, max_length: int = 220) -> str:
    if len(line) <= max_length:
        return line
    return f"{line[: max_length - 3]}..."


def print_summary(result: ScanResult) -> None:
    issue_count = len(result.issues)
    if issue_count == 0:
        print(f"No environment doc claim guard issues found in {_plural(result.files_scanned, 'file')}.")
        return

    matched_files = {issue.path for issue in result.issues}
    print(
        "Found "
        f"{_plural(issue_count, 'environment doc claim guard issue')} "
        f"in {_plural(len(matched_files), 'file')}."
    )
    for issue in result.issues:
        print(
            f"{_display_path(issue.path)}:{issue.line_number}: "
            f"{issue.label}: {_line_excerpt(issue.line)}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check environment docs for overclaims and count dashboards."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Doc files or directories to scan. Defaults to front-door working docs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = tuple(args.paths) if args.paths else DEFAULT_DOC_PATHS
    result = scan_paths(paths)
    print_summary(result)
    return 1 if result.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
