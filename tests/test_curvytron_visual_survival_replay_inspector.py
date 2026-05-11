import json

from curvyzero.training.curvytron_visual_survival_replay_inspector import (
    inspect_episode_artifact,
)


def test_replay_inspector_recovers_old_all_left_wall_death(tmp_path):
    artifact_path = tmp_path / "episode.json"
    sidecar_path = tmp_path / "episode.env_steps.jsonl"
    artifact_path.write_text(
        json.dumps(
            {
                "config": {
                    "seed": 1297473639,
                    "source_max_steps": 1024,
                    "opponent_policy_kind": "fixed_straight",
                },
                "episode": {
                    "seed": 1297473639,
                    "actions": [0] * 33,
                    "steps_run": 33,
                    "steps_survived": 32,
                    "terminal_reason": "survivor_win",
                },
            }
        ),
        encoding="utf-8",
    )
    sidecar_path.write_text(
        json.dumps({"step_index": 32, "trace_hash": "30c77a4dedae3f35"}) + "\n",
        encoding="utf-8",
    )

    result = inspect_episode_artifact(artifact_path)

    assert result["ok"] is True
    assert result["terminal_reason"] == "survivor_win"
    assert result["winner_ids"] == ["player_1"]
    assert result["loser_ids"] == ["player_0"]
    assert result["first_death"] == {
        "player": 0,
        "player_id": "player_0",
        "cause_name": "wall",
        "hit_owner": -1,
        "hit_owner_id": None,
    }
    assert result["trace_hash_match"] is True
    assert "player_0 hit the wall" in result["plain_read"]


def test_replay_inspector_rejects_missing_actions(tmp_path):
    artifact_path = tmp_path / "episode.json"
    artifact_path.write_text(json.dumps({"episode": {}, "config": {}}), encoding="utf-8")

    result = inspect_episode_artifact(artifact_path)

    assert result["ok"] is False
    assert result["reason"] == "missing_actions"


def test_replay_inspector_rejects_source_state_artifact(tmp_path):
    artifact_path = tmp_path / "episode.json"
    artifact_path.write_text(
        json.dumps(
            {
                "config": {
                    "env_variant": "source_state_fixed_opponent",
                    "opponent_policy_kind": "fixed_straight",
                },
                "episode": {"actions": [0, 0, 0]},
            }
        ),
        encoding="utf-8",
    )

    result = inspect_episode_artifact(artifact_path)

    assert result["ok"] is False
    assert result["reason"] == "unsupported_env_variant"
    assert result["env_variant"] == "source_state_fixed_opponent"


def test_replay_inspector_reports_invalid_json(tmp_path):
    artifact_path = tmp_path / "episode.json"
    artifact_path.write_text("", encoding="utf-8")

    result = inspect_episode_artifact(artifact_path)

    assert result["ok"] is False
    assert result["reason"] == "invalid_json"
