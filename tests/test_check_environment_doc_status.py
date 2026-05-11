import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_environment_doc_status.py"
)
_SPEC = importlib.util.spec_from_file_location("check_environment_doc_status", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
doc_status = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = doc_status
_SPEC.loader.exec_module(doc_status)


def test_claim_guard_fails_on_overclaims_and_count_dashboards(tmp_path, capsys):
    docs = tmp_path / "docs" / "working" / "environment"
    docs.mkdir(parents=True)
    claim_doc = docs / "claims.md"
    clean_doc = docs / "clean.md"
    claim_doc.write_text(
        "\n".join(
            [
                "# Claims",
                "Full environment fidelity achieved.",
                "The current setup is training-ready.",
                "The vector path is vector-ready.",
                "| Full pytest | `303 passed` |",
                "Focused vector pytest: `66 passed`.",
                "Tests: `140 passed`.",
                "Coverage: `87%`.",
            ]
        ),
        encoding="utf-8",
    )
    clean_doc.write_text(
        "We are not at full fidelity, and training is not ready.\n",
        encoding="utf-8",
    )

    exit_code = doc_status.main([str(docs)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Found 7 environment doc claim guard issues in 1 file." in captured.out
    assert f"{claim_doc}:2: full fidelity overclaim" in captured.out
    assert f"{claim_doc}:3: training-ready overclaim" in captured.out
    assert f"{claim_doc}:4: vector-ready overclaim" in captured.out
    assert f"{claim_doc}:5: standalone status-count dashboard" in captured.out
    assert f"{claim_doc}:6: standalone status-count dashboard" in captured.out
    assert f"{claim_doc}:7: standalone status-count dashboard" in captured.out
    assert f"{claim_doc}:8: standalone status-count dashboard" in captured.out
    assert "clean.md" not in captured.out


def test_acceptance_commands_and_negated_claims_succeed(tmp_path, capsys):
    docs = tmp_path / "docs" / "working" / "environment"
    docs.mkdir(parents=True)
    clean_doc = docs / "clean.md"
    clean_doc.write_text(
        "\n".join(
            [
                "# Clean",
                "No full fidelity achieved yet; training-ready CurvyTron remains blocked.",
                "Acceptance for `source_kinematics_straight_multistep`: `uv run pytest tests/test_example.py -q` -> `3 passed`.",
                "Numbers like p50, 51,200 rows, and coordinate 43.2 are not status claims.",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = doc_status.main([str(docs)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No environment doc claim guard issues found in 1 file." in captured.out
