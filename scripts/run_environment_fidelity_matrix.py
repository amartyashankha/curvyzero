"""List or run the current local environment-fidelity check matrix."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = Path("/private/tmp/curvy-environment-fidelity-matrix")
SCHEMA_VERSION = "curvyzero_environment_fidelity_matrix/v1"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    group: str
    description: str
    command_parts: tuple[str, ...]
    expected: str
    uses_artifact_root: bool = False
    slower: bool = False

    def command(self, artifact_root: Path) -> tuple[str, ...]:
        if not self.uses_artifact_root:
            return self.command_parts
        return (
            *self.command_parts,
            "--artifact-root",
            str((artifact_root / self.name).resolve()),
        )


@dataclass(frozen=True, slots=True)
class Suite:
    name: str
    description: str
    checks: tuple[str, ...]
    slower: bool = False


@dataclass(frozen=True, slots=True)
class CommandResult:
    check: str
    command: tuple[str, ...]
    returncode: int | None
    stdout_tail: str | None = None
    stderr_tail: str | None = None


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _source_check(
    name: str,
    manifest: str,
    runner: str,
    expected: str,
    description: str,
) -> Check:
    return Check(
        name=name,
        group="source",
        description=description,
        command_parts=(
            "uv",
            "run",
            "python",
            "tools/run_fidelity_batch.py",
            manifest,
            "--python-runner",
            runner,
            "--fail-on-mismatch",
        ),
        expected=expected,
        uses_artifact_root=True,
        slower=True,
    )


def _source_loop_check(
    name: str,
    scenario: str,
    runner: str,
    expected: str,
    description: str,
) -> Check:
    return Check(
        name=name,
        group="source",
        description=description,
        command_parts=(
            "uv",
            "run",
            "python",
            "tools/run_fidelity_loop.py",
            scenario,
            "--python-runner",
            runner,
            "--fail-on-mismatch",
        ),
        expected=expected,
        uses_artifact_root=True,
        slower=True,
    )


SOURCE_CHECKS: tuple[Check, ...] = (
    _source_check(
        "source-kinematics",
        "scenarios/environment/source_kinematics_batch.json",
        "source-kinematics",
        (
            "JS/Python common-trace parity for one-step movement, turn order, "
            "straight/turn multi-step movement, and varied elapsed-ms movement."
        ),
        "Source movement claim; bonuses and broad physics fidelity remain separate.",
    ),
    _source_check(
        "source-border",
        "scenarios/environment/source_border_batch.json",
        "source-border-rules",
        (
            "JS/Python common-trace parity for normal-wall basics, borderless "
            "wrap/body-skip behavior, and one exact edge/corner control."
        ),
        "Border basics claim; follow-up corner and world-island edges remain open.",
    ),
    _source_check(
        "source-normal-wall-multiplayer",
        "scenarios/environment/source_normal_wall_multiplayer_batch.json",
        "source-border-rules",
        (
            "JS/Python common-trace parity for narrow 3P/4P normal-wall death "
            "order, survivor scoring, and terminal draw."
        ),
        "Multiplayer wall claim; head-head and round lifecycle coverage are separate.",
    ),
    _source_check(
        "source-body",
        "scenarios/environment/source_body_canary_batch.json",
        "source-body-canary",
        (
            "JS/Python common-trace parity for seeded body collision, own-body "
            "latency, and same-frame point fixtures."
        ),
        "Body canary claim; emitted-body trails and broader gap cases remain open.",
    ),
    _source_check(
        "source-old-body-metadata",
        "scenarios/environment/source_body_old_metadata_batch.json",
        "source-body-canary",
        "JS/Python common-trace parity for one old-body death metadata fixture.",
        "Single old-body metadata claim; broad old-flag behavior remains open.",
    ),
    _source_check(
        "source-collision-order",
        "scenarios/environment/source_collision_order_batch.json",
        "source-body-canary",
        (
            "JS/Python common-trace parity for death-point-kills-later-player "
            "and head-head-looking same-endpoint order fixtures."
        ),
        "Narrow collision-order claim; broader reverse-order cases remain open.",
    ),
    _source_check(
        "source-print-manager",
        "scenarios/environment/source_print_manager_batch.json",
        "source-print-manager-canary",
        (
            "JS/Python common-trace parity for deterministic PrintManager "
            "toggle, timer, delayed start, and death-stop fixtures."
        ),
        "Deterministic PrintManager claim; random cadence and lifecycle are separate.",
    ),
    _source_check(
        "source-print-manager-random",
        "scenarios/environment/source_print_manager_random_batch.json",
        "source-print-manager-canary",
        (
            "JS/Python common-trace parity for same-frame PrintManager random "
            "call order and one taped cadence fixture."
        ),
        "Random-tape PrintManager claim; broad stochastic cadence remains open.",
    ),
    _source_check(
        "source-trail",
        "scenarios/environment/source_trail_batch.json",
        "source-trail-cadence-canary",
        (
            "JS/Python common-trace parity for normal trail point insertion "
            "and below-radius no-point behavior."
        ),
        "Normal trail cadence claim; hidden draw cursor and gap cases are separate.",
    ),
    _source_check(
        "source-trail-gap",
        "scenarios/environment/source_trail_gap_batch.json",
        "source-trail-gap-canary",
        (
            "JS/Python common-trace parity for forced trail-gap body absence, "
            "stored-body danger, and boundary transitions."
        ),
        "Forced trail-gap claim; natural gaps and vector speed defaults are separate.",
    ),
    _source_loop_check(
        "source-trail-gap-natural",
        "scenarios/environment/source_trail_gap_natural_multistep_hole_crossing.json",
        "source-trail-gap-canary",
        (
            "JS/Python common-trace parity for one natural taped multi-step "
            "trail-gap hole crossing."
        ),
        (
            "Separate natural trail-gap source fixture; outside the forced batch "
            "and vector speed defaults."
        ),
    ),
    Check(
        name="source-lifecycle",
        group="source",
        description=(
            "Narrow source-lifecycle claim; focused 3P all-dead, survivor, "
            "present/absent survivor scoring/warmdown/next-round, 3P match-end, 3P "
            "tie-at-max-score continuation, and 3P all-present multi-round "
            "match-end are now pinned, plus focused 4P all-dead and survivor "
            "warmdown/next-round fixtures; broader 4P match lifecycle, bonuses, "
            "production reset/autoreset, and vector "
            "lifecycle remain unsupported."
        ),
        command_parts=(
            "uv",
            "run",
            "pytest",
            "tests/test_source_lifecycle_runner.py",
            "tests/test_lifecycle_oracle.py",
            "-q",
        ),
        expected=(
            "Direct JS/Python parity for pinned lifecycle fixtures: 2P spawn "
            "RNG/warmup print-start, next-round spawn RNG, and heading retry; "
            "plus focused 3P spawn-order, 3P warmup/print-start, 3P "
            "present/absent first-round, 3P present/absent warmdown/next-round, "
            "3P all-dead warmdown/next-round, 4P first-round spawn, and 4P "
            "all-present all-dead warmdown/next-round fixtures; plus focused 3P survivor scoring "
            "through round:end and survivor warmdown/next-round fixtures; "
            "plus one present/absent 3P fixture where the absent avatar is in "
            "source deaths without a die event and survivor scoring selects "
            "avatar 1 at round:end; "
            "plus one 4P survivor fixture where death order gives avatars "
            "4, 3, 2 round scores 0, 1, 2, avatar 1 receives the survivor "
            "bonus, then game:stop and round:new emit at 8000 ms with "
            "reverse 4P spawn RNG; "
            "plus one max_score=1 2P fixture emits round:end winner 1 at "
            "3000 ms, then game:stop and end at 8000 ms, with no later "
            "round:new; plus one max_score=2 3P fixture emits round:end "
            "winner 1 after avatar deaths 3 then 2, then game:stop and end "
            "at 8000 ms, with no later round:new; plus one max_score=1 3P "
            "tie fixture emits round:end winner null after deaths 3, 2, 1, "
            "then game:stop and round:new at 8000 ms, with no end; plus one "
            "max_score=3 all-present 3P fixture carries avatar 1 score 2 "
            "through game:stop and round:new at 8000 ms, then reaches score 4 "
            "and emits game:stop and end at 19000 ms with no later round:new."
        ),
    ),
)

VECTOR_CHECKS: tuple[Check, ...] = (
    Check(
        name="vector-pytest",
        group="vector",
        description=(
            "Pytest guard for fixture-backed vector comparison; not full "
            "optimized environment fidelity."
        ),
        command_parts=(
            "uv",
            "run",
            "pytest",
            "tests/test_compare_vector_arrays_to_fidelity.py",
            "-q",
        ),
        expected="Focused comparator behavior holds for the supported vector fixture slice.",
    ),
    Check(
        name="vector-batch-actor-pytest",
        group="vector",
        description=(
            "Focused vector, batch-row, and actor-bridge pytest guard over "
            "the supported fixture slice."
        ),
        command_parts=(
            "uv",
            "run",
            "pytest",
            "tests/test_compare_vector_arrays_to_fidelity.py",
            "tests/test_benchmark_vector_batch_rows.py",
            "tests/test_benchmark_vector_actor_loop_bridge.py",
            "-q",
        ),
        expected=(
            "Comparator, batch-row harness, and actor-loop bridge behavior hold "
            "for the supported fixture slice."
        ),
    ),
    Check(
        name="vector-mixed-comparator",
        group="vector",
        description=(
            "Narrow fixture-backed vector comparator gate for body, borderless, "
            "wall, PrintManager, and forced trail-gap cases."
        ),
        command_parts=(
            "uv",
            "run",
            "python",
            "scripts/compare_vector_arrays_to_fidelity.py",
            "scenarios/environment/source_body_canary_batch.json",
            "scenarios/environment/source_borderless_wrap_step.json",
            "scenarios/environment/source_normal_wall_death_step.json",
            "scenarios/environment/source_print_manager_batch.json",
            "scenarios/environment/source_trail_gap_batch.json",
            "--body-capacity",
            "4",
            "--fail-on-unsupported",
            "--format",
            "plain",
        ),
        expected=(
            "Vector state/event comparison matches the supported mixed fixture "
            "set; unsupported fixtures should stay explicit."
        ),
    ),
)

SPEED_CHECKS: tuple[Check, ...] = (
    Check(
        name="batch-rows-quick",
        group="speed",
        description="Tiny B>1 vector batch-row timing smoke over supported fixtures.",
        command_parts=(
            "uv",
            "run",
            "python",
            "scripts/benchmark_vector_batch_rows.py",
            "--batch-sizes",
            "2",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--format",
            "plain",
        ),
        expected=(
            "Batch-row smoke runs over the default supported fixture slice; "
            "not a broad speed or fidelity proof."
        ),
    ),
    Check(
        name="actor-loop-quick",
        group="speed",
        description="Tiny vector actor-loop timing smoke over supported fixtures.",
        command_parts=(
            "uv",
            "run",
            "python",
            "scripts/benchmark_vector_actor_loop_bridge.py",
            "--batch-sizes",
            "2",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--rollout-steps",
            "1",
            "--hidden-dim",
            "4",
            "--simulations",
            "1",
            "--chunk-steps",
            "4",
            "--format",
            "plain",
        ),
        expected=(
            "Actor-loop smoke runs one tiny rollout over default supported "
            "fixtures; not production self-play evidence."
        ),
    ),
)

CHECKS: tuple[Check, ...] = (*SOURCE_CHECKS, *VECTOR_CHECKS, *SPEED_CHECKS)
CHECKS_BY_NAME = {check.name: check for check in CHECKS}

SUITES: tuple[Suite, ...] = (
    Suite(
        name="smoke",
        description=(
            "Quick local comparator plus tiny batch-row and actor-loop smokes "
            "over the supported fixture slice."
        ),
        checks=(
            "vector-pytest",
            "vector-mixed-comparator",
            "batch-rows-quick",
            "actor-loop-quick",
        ),
    ),
    Suite(
        name="source-core",
        description="Source claim regression batches; slower and not full fidelity.",
        checks=tuple(check.name for check in SOURCE_CHECKS),
        slower=True,
    ),
    Suite(
        name="vector-core",
        description=(
            "Fixture-backed vector, batch-row, and actor checks without source "
            "JS batches."
        ),
        checks=(
            "vector-batch-actor-pytest",
            "vector-mixed-comparator",
            "batch-rows-quick",
            "actor-loop-quick",
        ),
    ),
)
SUITES_BY_NAME = {suite.name: suite for suite in SUITES}


def resolve_artifact_root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def selected_checks(selectors: Sequence[str]) -> list[Check]:
    names: list[str] = []
    for selector in selectors:
        for raw_name in selector.split(","):
            name = raw_name.strip()
            if not name:
                continue
            if name in SUITES_BY_NAME:
                names.extend(SUITES_BY_NAME[name].checks)
            elif name in CHECKS_BY_NAME:
                names.append(name)
            else:
                valid = sorted((*SUITES_BY_NAME.keys(), *CHECKS_BY_NAME.keys()))
                raise ValueError(f"unknown check or suite {name!r}; valid names: {', '.join(valid)}")

    unique: list[Check] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(CHECKS_BY_NAME[name])
    return unique


def list_payload(artifact_root: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "repo_root": str(REPO_ROOT),
        "artifact_root": str(artifact_root),
        "default_artifact_root": str(DEFAULT_ARTIFACT_ROOT),
        "suites": [_suite_payload(suite) for suite in SUITES],
        "checks": [_check_payload(check, artifact_root) for check in CHECKS],
    }


def plan_payload(
    selectors: Sequence[str],
    artifact_root: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    checks = selected_checks(selectors)
    return {
        "schema": SCHEMA_VERSION,
        "repo_root": str(REPO_ROOT),
        "artifact_root": str(artifact_root),
        "selected": list(selectors),
        "dry_run": dry_run,
        "checks": [_check_payload(check, artifact_root) for check in checks],
    }


def run_checks(
    checks: Sequence[Check],
    artifact_root: Path,
    *,
    dry_run: bool,
    capture_output: bool = False,
    runner: Runner = subprocess.run,
) -> tuple[int, list[CommandResult]]:
    results: list[CommandResult] = []
    exit_code = 0
    for check in checks:
        command = check.command(artifact_root)
        if dry_run:
            results.append(CommandResult(check=check.name, command=command, returncode=None))
            continue

        completed = runner(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=capture_output,
            check=False,
        )
        returncode = int(completed.returncode)
        if returncode and exit_code == 0:
            exit_code = returncode
        results.append(
            CommandResult(
                check=check.name,
                command=command,
                returncode=returncode,
                stdout_tail=_tail(completed.stdout) if capture_output else None,
                stderr_tail=_tail(completed.stderr) if capture_output else None,
            )
        )
    return exit_code, results


def print_list(payload: dict[str, Any]) -> None:
    print(f"schema={payload['schema']}")
    print(f"artifact_root={payload['artifact_root']}")
    print("suites:")
    for suite in payload["suites"]:
        suffix = " slower=true" if suite["slower"] else ""
        print(f"  {suite['name']}: {', '.join(suite['checks'])}{suffix}")
        print(f"    {suite['description']}")
    print("checks:")
    for check in payload["checks"]:
        flags = []
        if check["slower"]:
            flags.append("slower")
        if check["uses_artifact_root"]:
            flags.append("artifact-root")
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        print(f"  {check['name']} ({check['group']}){flag_text}")
        print(f"    expected: {check['expected']}")
        print(f"    command: {shlex.join(check['command'])}")


def print_plan(payload: dict[str, Any]) -> None:
    mode = "dry-run" if payload["dry_run"] else "run"
    print(f"mode={mode}")
    print(f"artifact_root={payload['artifact_root']}")
    print(f"selected={', '.join(payload['selected'])}")
    for check in payload["checks"]:
        print(f"check={check['name']} expected={check['expected']}")
        print(f"command={shlex.join(check['command'])}")


def print_results(results: Sequence[CommandResult]) -> None:
    for result in results:
        if result.returncode is None:
            print(f"result={result.check} dry-run")
        else:
            print(f"result={result.check} returncode={result.returncode}")


def results_payload(
    plan: dict[str, Any],
    results: Sequence[CommandResult],
    *,
    exit_code: int,
) -> dict[str, Any]:
    return {
        **plan,
        "exit_code": exit_code,
        "results": [
            {
                "check": result.check,
                "command": list(result.command),
                "returncode": result.returncode,
                "stdout_tail": result.stdout_tail,
                "stderr_tail": result.stderr_tail,
            }
            for result in results
        ],
    }


def _check_payload(check: Check, artifact_root: Path) -> dict[str, Any]:
    command = check.command(artifact_root)
    return {
        "name": check.name,
        "group": check.group,
        "description": check.description,
        "expected": check.expected,
        "slower": check.slower,
        "uses_artifact_root": check.uses_artifact_root,
        "artifact_root": str((artifact_root / check.name).resolve())
        if check.uses_artifact_root
        else None,
        "command": list(command),
        "command_text": shlex.join(command),
    }


def _suite_payload(suite: Suite) -> dict[str, Any]:
    return {
        "name": suite.name,
        "description": suite.description,
        "checks": list(suite.checks),
        "slower": suite.slower,
    }


def _tail(value: str | None, limit: int = 8_000) -> str | None:
    if value is None:
        return None
    return value[-limit:]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List or run the current environment fidelity matrix."
    )
    parser.add_argument("--list", action="store_true", help="Print named checks and commands.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Run a suite or check name. Repeat or comma-separate. "
            "Common suites: smoke, source-core, vector-core."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected commands without executing them.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help=(
            "Root for source batch artifacts. Defaults under /private/tmp; each "
            "source check gets its own child directory."
        ),
    )
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    args = parser.parse_args(argv)

    artifact_root = resolve_artifact_root(args.artifact_root)
    if args.list:
        payload = list_payload(artifact_root)
        if args.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print_list(payload)
        return 0

    if not args.run:
        parser.error("pass --list or --run NAME")

    try:
        checks = selected_checks(args.run)
    except ValueError as error:
        parser.error(str(error))

    plan = plan_payload(args.run, artifact_root, dry_run=args.dry_run)
    if args.format == "json":
        exit_code, results = run_checks(
            checks,
            artifact_root,
            dry_run=args.dry_run,
            capture_output=not args.dry_run,
        )
        print(json.dumps(results_payload(plan, results, exit_code=exit_code), indent=2))
        return exit_code

    print_plan(plan)
    if not args.dry_run:
        print()
    exit_code, results = run_checks(checks, artifact_root, dry_run=args.dry_run)
    print_results(results)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
