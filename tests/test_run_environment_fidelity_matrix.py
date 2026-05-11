import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_environment_fidelity_matrix.py"
)
_SPEC = importlib.util.spec_from_file_location("run_environment_fidelity_matrix", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
fidelity_matrix = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fidelity_matrix
_SPEC.loader.exec_module(fidelity_matrix)

_COUNT_BRAG_RE = re.compile(
    r"\b\d+\s+(?:pass|passed|fail|failed|blocked|unsupported)\b"
    r"|\bdiff_status pass\b",
    re.IGNORECASE,
)


def test_source_core_expands_promoted_batches_with_isolated_artifact_roots(tmp_path):
    checks = fidelity_matrix.selected_checks(["source-core"])
    names = [check.name for check in checks]

    assert names == [
        "source-kinematics",
        "source-border",
        "source-normal-wall-multiplayer",
        "source-body",
        "source-old-body-metadata",
        "source-collision-order",
        "source-print-manager",
        "source-print-manager-random",
        "source-trail",
        "source-trail-gap",
        "source-trail-gap-natural",
        "source-lifecycle",
    ]

    commands = {check.name: check.command(tmp_path) for check in checks}

    assert commands["source-kinematics"] == (
        "uv",
        "run",
        "python",
        "tools/run_fidelity_batch.py",
        "scenarios/environment/source_kinematics_batch.json",
        "--python-runner",
        "source-kinematics",
        "--fail-on-mismatch",
        "--artifact-root",
        str(tmp_path / "source-kinematics"),
    )
    assert commands["source-old-body-metadata"][-2:] == (
        "--artifact-root",
        str(tmp_path / "source-old-body-metadata"),
    )
    assert commands["source-print-manager-random"][4:7] == (
        "scenarios/environment/source_print_manager_random_batch.json",
        "--python-runner",
        "source-print-manager-canary",
    )
    assert commands["source-trail-gap-natural"] == (
        "uv",
        "run",
        "python",
        "tools/run_fidelity_loop.py",
        "scenarios/environment/source_trail_gap_natural_multistep_hole_crossing.json",
        "--python-runner",
        "source-trail-gap-canary",
        "--fail-on-mismatch",
        "--artifact-root",
        str(tmp_path / "source-trail-gap-natural"),
    )
    assert commands["source-lifecycle"] == (
        "uv",
        "run",
        "pytest",
        "tests/test_source_lifecycle_runner.py",
        "tests/test_lifecycle_oracle.py",
        "-q",
    )


def test_list_payload_encodes_claims_and_vector_commands(tmp_path):
    payload = fidelity_matrix.list_payload(tmp_path)
    checks = {check["name"]: check for check in payload["checks"]}

    assert payload["artifact_root"] == str(tmp_path)
    for check in checks.values():
        assert not _COUNT_BRAG_RE.search(check["expected"])
        assert not _COUNT_BRAG_RE.search(check["description"])

    assert checks["vector-pytest"]["expected"] == (
        "Focused comparator behavior holds for the supported vector fixture slice."
    )
    assert checks["vector-batch-actor-pytest"]["expected"] == (
        "Comparator, batch-row harness, and actor-loop bridge behavior hold "
        "for the supported fixture slice."
    )
    assert checks["vector-mixed-comparator"]["expected"] == (
        "Vector state/event comparison matches the supported mixed fixture "
        "set; unsupported fixtures should stay explicit."
    )
    assert checks["source-trail-gap"]["expected"] == (
        "JS/Python common-trace parity for forced trail-gap body absence, "
        "stored-body danger, and boundary transitions."
    )
    assert checks["source-trail-gap-natural"]["expected"] == (
        "JS/Python common-trace parity for one natural taped multi-step "
        "trail-gap hole crossing."
    )
    assert checks["source-trail-gap-natural"]["description"] == (
        "Separate natural trail-gap source fixture; outside the forced batch "
        "and vector speed defaults."
    )
    assert checks["source-lifecycle"]["expected"] == (
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
    )
    assert checks["source-lifecycle"]["description"] == (
        "Narrow source-lifecycle claim; focused 3P all-dead, survivor, "
        "present/absent survivor scoring/warmdown/next-round, 3P match-end, 3P "
        "tie-at-max-score continuation, and 3P all-present multi-round "
        "match-end are now pinned, plus focused 4P all-dead and survivor "
        "warmdown/next-round fixtures; broader 4P match lifecycle, bonuses, "
        "production reset/autoreset, and vector "
        "lifecycle remain unsupported."
    )
    assert checks["source-lifecycle"]["uses_artifact_root"] is False
    assert checks["source-lifecycle"]["command"] == [
        "uv",
        "run",
        "pytest",
        "tests/test_source_lifecycle_runner.py",
        "tests/test_lifecycle_oracle.py",
        "-q",
    ]
    assert checks["batch-rows-quick"]["expected"] == (
        "Batch-row smoke runs over the default supported fixture slice; "
        "not a broad speed or fidelity proof."
    )
    assert checks["actor-loop-quick"]["expected"] == (
        "Actor-loop smoke runs one tiny rollout over default supported "
        "fixtures; not production self-play evidence."
    )
    assert checks["vector-mixed-comparator"]["command"] == [
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
    ]


def test_dry_run_json_prints_smoke_plan_without_executing(tmp_path, capsys):
    exit_code = fidelity_matrix.main(
        [
            "--run",
            "smoke",
            "--dry-run",
            "--format",
            "json",
            "--artifact-root",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["dry_run"] is True
    assert [check["name"] for check in payload["checks"]] == [
        "vector-pytest",
        "vector-mixed-comparator",
        "batch-rows-quick",
        "actor-loop-quick",
    ]
    assert [result["returncode"] for result in payload["results"]] == [
        None,
        None,
        None,
        None,
    ]


def test_count_brag_guard_rejects_status_counts():
    assert _COUNT_BRAG_RE.search("12 passed")
    assert _COUNT_BRAG_RE.search("1 fail")
    assert _COUNT_BRAG_RE.search("diff_status pass")


def test_selected_checks_accepts_comma_separated_selectors_without_duplicates():
    checks = fidelity_matrix.selected_checks(["vector-pytest,smoke"])

    assert [check.name for check in checks] == [
        "vector-pytest",
        "vector-mixed-comparator",
        "batch-rows-quick",
        "actor-loop-quick",
    ]


def test_vector_core_does_not_include_source_trail_gap_natural():
    checks = fidelity_matrix.selected_checks(["vector-core"])

    assert "source-trail-gap-natural" not in [check.name for check in checks]


def test_selected_checks_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown check or suite"):
        fidelity_matrix.selected_checks(["missing-suite"])
