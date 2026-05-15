import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

from curvyzero.training.opponent_leaderboard import canonical_json_sha256


def _load_controller_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "promote_curvytron_rating_round.py"
    )
    spec = importlib.util.spec_from_file_location(
        "promote_curvytron_rating_round",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/attempts/attempt-a/train/lightzero_exp/ckpt/"
        f"iteration_{iteration}.pth.tar"
    )


def _rating_snapshot(round_id: str = "round-000000") -> dict:
    return {
        "schema_id": "curvyzero_curvytron_checkpoint_rating_snapshot/v0",
        "formula_version": "batch_elo_v1",
        "tournament_id": "arena-a",
        "rating_run_id": "elo-a",
        "ratings_ref": (
            "tournaments/curvytron/arena-a/ratings/elo-a/"
            f"rounds/{round_id}/ratings.json"
        ),
        "latest_ref": "tournaments/curvytron/arena-a/ratings/elo-a/latest.json",
        "context_hash": "ctx-a",
        "roster_hash": "roster-a",
        "round_id": round_id,
        "round_index": 0,
        "rating_spec": {
            "decision_source_frames": 1,
        },
        "pair_count": 1,
        "game_count": 21,
        "ratings": [
            {
                "checkpoint_id": "ckpt-a",
                "checkpoint_ref": _checkpoint_ref("run-a", 100000),
                "rank": 1,
                "rating": 1700.0,
                "games": 500,
                "distinct_opponents": 20,
                "failure_count": 0,
                "status": "active",
            }
        ],
    }


def _artifacts(round_id: str = "round-000000") -> dict:
    snapshot = _rating_snapshot(round_id)
    return {
        "input": {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-a",
            "round_id": round_id,
            "round_index": 0,
            "rating_spec": {
                "decision_source_frames": 1,
            },
        },
        "progress": {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-a",
            "round_id": round_id,
            "round_index": 0,
            "completed_pair_count": 1,
            "completed_game_count": 21,
        },
        "results": {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-a",
            "round_id": round_id,
            "round_index": 0,
            "pair_count": 1,
            "game_count": 21,
        },
        "ratings": snapshot,
        "latest": snapshot,
    }


def test_validate_round_artifacts_checks_source_hash_and_shape():
    controller = _load_controller_module()

    evidence = controller.validate_round_artifacts(
        _artifacts(),
        tournament_id="arena-a",
        rating_run_id="elo-a",
        round_id="round-000000",
    )

    assert evidence["round_artifacts_agree"] is True
    assert evidence["one_frame"] is True
    assert evidence["zero_failed_games"] is True
    assert evidence["rating_snapshot_sha256"] == canonical_json_sha256(
        _rating_snapshot()
    )
    assert evidence["rating_context_hash"] == "ctx-a"
    assert evidence["roster_hash"] == "roster-a"


def test_validate_round_artifacts_rejects_mutable_latest_drift():
    controller = _load_controller_module()
    artifacts = _artifacts()
    artifacts["latest"] = _rating_snapshot("round-000001")

    with pytest.raises(ValueError, match="latest.json round_id mismatch"):
        controller.validate_round_artifacts(
            artifacts,
            tournament_id="arena-a",
            rating_run_id="elo-a",
            round_id="round-000000",
        )


def test_build_publish_command_passes_expected_source_guards():
    controller = _load_controller_module()
    args = Namespace(
        tournament_id="arena-a",
        rating_run_id="elo-a",
        leaderboard_id="main",
        snapshot_id="snapshot-a",
        active_min_distinct_opponents=20,
        active_min_valid_games=300,
        max_failure_rate=0.02,
        max_active_rank=100,
        round_id="round-000000",
        allow_no_active_rows=False,
        diagnostic_only=False,
    )
    source = {
        "round_index": 0,
        "rating_context_hash": "ctx-a",
        "roster_hash": "roster-a",
        "rating_snapshot_sha256": "abc123",
    }

    command = controller.build_publish_command(args, source)

    assert "--leaderboard-expected-round-id" in command
    assert "round-000000" in command
    assert "--leaderboard-expected-round-index" in command
    assert "0" in command
    assert "--leaderboard-expected-rating-context-hash" in command
    assert "ctx-a" in command
    assert "--leaderboard-expected-roster-hash" in command
    assert "roster-a" in command
    assert "--leaderboard-expected-rating-snapshot-sha256" in command
    assert "abc123" in command


def test_build_write_assignment_command_can_target_control_volume(tmp_path):
    controller = _load_controller_module()
    args = Namespace(
        assignment_bank_run_id="assign-run",
        assignment_bank_attempt_id="try-assign-run",
        assignment_target_volume="control",
        mirror_assignment_checkpoints_to_control=True,
    )
    assignment_dir = tmp_path / "assignment"

    command = controller.build_write_assignment_command(
        args,
        assignment_dir=assignment_dir,
    )

    assert "--opponent-assignment-target-volume" in command
    assert "control" in command
    assert "--mirror-assignment-checkpoints-to-control" in command


def test_write_refresh_pointer_command_uses_control_volume(tmp_path):
    controller = _load_controller_module()
    command = controller.write_refresh_pointer_command(
        pointer_path=tmp_path / "pointer.json",
        pointer_ref="control:training/task/run/pointer.json",
        pointer_volume="control",
    )

    assert command[:5] == ["uv", "run", "--extra", "modal", "modal"]
    assert "volume" in command
    assert "put" in command
    assert controller.CONTROL_VOLUME_NAME in command
    assert "training/task/run/pointer.json" in command
    assert "control:training/task/run/pointer.json" not in command


def test_command_json_parser_prefers_schema_payload_over_nested_command():
    controller = _load_controller_module()
    text = """
log line
{
  "command": {
    "batch_size": 4
  },
  "ok": true,
  "called_train_muzero": true,
  "schema_id": "curvyzero_lightzero_curvytron_visual_survival_compact_output/v0"
}
"""

    payload = controller.best_json_object_from_command_output(text)

    assert payload["schema_id"].startswith("curvyzero_lightzero")
    assert payload["called_train_muzero"] is True


def test_verify_smoke_artifacts_requires_model_load_and_provider_rows(tmp_path):
    controller = _load_controller_module()
    assignment_ref = "training/task/run/attempts/try/opponents/assignments/a/assignment.json"
    assignment_sha = "a" * 64
    env_steps = tmp_path / "env_steps.jsonl"
    env_steps.write_text(
        (
            '{"opponent_assignment_ref":"'
            + assignment_ref
            + '","opponent_assignment_sha256":"'
            + assignment_sha
            + '","opponent_provider_load_ok":true}\n'
        ),
        encoding="utf-8",
    )
    summary = {
        "ok": True,
        "called_train_muzero": True,
        "initial_policy_checkpoint": {
            "load_result": {
                "loaded": True,
                "module_loads": [
                    {
                        "meaningful_model_load": True,
                        "fresh_optimizer_preserved": True,
                    }
                ],
            }
        },
        "auto_resume": {
            "found": False,
        },
    }

    result = controller.verify_smoke_artifacts(
        summary=summary,
        env_steps_path=env_steps,
        assignment_ref=assignment_ref,
        assignment_sha256=assignment_sha,
    )

    assert result["initial_checkpoint_loaded"] is True
    assert result["provider_ok_row_count"] == 1


def test_verify_smoke_artifacts_rejects_missing_provider_rows(tmp_path):
    controller = _load_controller_module()
    env_steps = tmp_path / "env_steps.jsonl"
    env_steps.write_text('{"opponent_provider_load_ok":false}\n', encoding="utf-8")
    summary = {
        "ok": True,
        "called_train_muzero": True,
        "initial_policy_checkpoint": {
            "load_result": {
                "loaded": True,
                "module_loads": [
                    {
                        "meaningful_model_load": True,
                        "fresh_optimizer_preserved": True,
                    }
                ],
            }
        },
        "auto_resume": {
            "found": False,
        },
    }

    with pytest.raises(ValueError, match="assignment/provider load"):
        controller.verify_smoke_artifacts(
            summary=summary,
            env_steps_path=env_steps,
            assignment_ref="assignment.json",
            assignment_sha256="a" * 64,
        )
