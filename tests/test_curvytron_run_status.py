import json

from curvyzero.infra.modal import lightzero_curvytron_run_status as status_mod
from curvyzero.infra.modal import run_management as runs


def _write_manifest(path, *, created_at, mean_steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "job_kind": "lightzero_curvytron_visual_survival_checkpoint_curve_eval",
                "eval_id": "matrix",
                "created_at": created_at,
                "survival_aggregate_table": [
                    {
                        "checkpoint": "iteration_1",
                        "seeds": 2,
                        "mean_steps": mean_steps,
                        "median_steps": mean_steps,
                        "min_steps": 5,
                        "max_steps": 40,
                        "ok_count": 2,
                        "capped_count": 1,
                        "failure_count": 0,
                    }
                ],
                "table": [
                    {
                        "checkpoint_label": "iteration_1",
                        "seed": 1,
                        "steps_survived": 40,
                        "terminal_reason": "cap",
                        "ok": True,
                        "action_histogram": {"0": 40},
                    },
                    {
                        "checkpoint_label": "iteration_1",
                        "seed": 2,
                        "steps_survived": 5,
                        "death_cause_name": "wall",
                        "ok": True,
                        "action_histogram": {"0": 4, "1": 1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_gif_summary(path, *, created_at):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_lightzero_curvytron_checkpoint_selfplay_gif_summary/v0",
                "created_at": created_at,
                "eval_id": "live_checkpoint_iteration_1",
                "checkpoint_label": "iteration_1",
                "ok": True,
                "gif_ref": "training/run/attempt/eval/live_checkpoint_iteration_1/selfplay/raw.gif",
                "frame_count": 4,
                "physical_steps": 12,
                "terminal_reason": "wall",
                "stop_reason": "environment_done",
                "opponent_mixture_enabled": True,
                "opponent_mixture_entry_name": "scripted",
                "opponent_mixture_age_label": "scripted_wall_avoidant",
                "opponent_mixture_entry_weight": 50.0,
                "greedy_action_collapse_warning": True,
                "greedy_action_summary": {
                    "decision_count": 12,
                    "action_counts_by_player": {
                        "player_0": {"0": 11, "1": 1, "2": 0},
                    },
                },
                "gif_variants": {
                    "collect_t1": {
                        "gif_ref": (
                            "training/run/attempt/eval/live_checkpoint_iteration_1/"
                            "selfplay/collect_t1.gif"
                        )
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_eval_manifest_rollup_uses_latest_manifest_by_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(status_mod, "RUNS_MOUNT", tmp_path)
    run_id = "run-a"
    attempt_id = "attempt-a"
    eval_root = tmp_path / runs.attempt_eval_ref(status_mod.TASK_ID, run_id, attempt_id, "matrix")

    _write_manifest(
        eval_root / "manifest_steps20_seeds_old.json",
        created_at="2026-05-12T01:00:00Z",
        mean_steps=10,
    )
    _write_manifest(
        eval_root / "manifest_steps20_seeds_new.json",
        created_at="2026-05-12T02:00:00Z",
        mean_steps=22.5,
    )

    rollup = status_mod._eval_manifest_rollup(
        run_id,
        attempt_id,
        collapse_threshold=0.8,
    )

    assert rollup["eval_manifest_count"] == 2
    assert rollup["latest_eval_manifest_ref"].endswith("manifest_steps20_seeds_new.json")
    assert rollup["latest_eval_checkpoint"] == "iteration_1"
    assert rollup["latest_eval_mean_steps"] == 22.5
    assert rollup["latest_eval_top_action"] == "0"
    assert rollup["latest_eval_action_fraction"] == 44 / 45
    assert rollup["latest_eval_collapsed"] is True
    assert len(rollup["eval_checkpoints"]) == 1
    checkpoint = rollup["eval_checkpoints"][0]
    assert checkpoint["checkpoint"] == "iteration_1"
    assert checkpoint["mean_steps"] == 22.5
    assert checkpoint["outcome_histogram"] == {"cap": 1, "wall": 1}
    assert checkpoint["action_summary"]["top_action"] == "0"
    assert checkpoint["action_summary"]["collapsed"] is True
    assert checkpoint["row_action_collapsed_count"] == 2


def test_eval_manifest_rollup_preserves_survivaldiag_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(status_mod, "RUNS_MOUNT", tmp_path)
    run_id = "run-survivaldiag"
    attempt_id = "attempt-survivaldiag"
    eval_root = tmp_path / runs.attempt_eval_ref(status_mod.TASK_ID, run_id, attempt_id, "matrix")
    manifest_path = eval_root / "manifest_survivaldiag.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "job_kind": "lightzero_curvytron_visual_survival_checkpoint_curve_eval",
                "eval_id": "matrix-survivaldiag",
                "created_at": "2026-05-13T01:00:00Z",
                "survival_aggregate_table": [
                    {
                        "checkpoint": "iteration_2",
                        "seeds": 2,
                        "mean_steps": 18,
                        "median_steps": 18,
                        "min_steps": 12,
                        "max_steps": 24,
                        "ok_count": 1,
                        "failure_count": 1,
                        "action_entropy": 0.42,
                    }
                ],
                "table": [
                    {
                        "checkpoint_label": "iteration_2",
                        "seed": 1,
                        "ok": True,
                        "training_reward": 1.0,
                        "bonus_pickup_count": 1,
                        "bonus_pickup_reward": 0.1,
                        "reward_components": {"survival": 0.9, "bonus": 0.1},
                        "terminal_cause": "normal_wall",
                        "action_histogram": {"0": 1, "1": 1},
                    },
                    {
                        "checkpoint_label": "iteration_2",
                        "seed": 2,
                        "ok": False,
                        "training_reward": 2.0,
                        "bonus_pickup_count": 3,
                        "bonus_pickup_reward": 0.4,
                        "reward_components": {"survival": 1.6, "bonus": 0.4},
                        "terminal_cause": "cap",
                        "action_histogram": {"1": 2},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rollup = status_mod._eval_manifest_rollup(
        run_id,
        attempt_id,
        collapse_threshold=0.8,
    )

    checkpoint = rollup["eval_checkpoints"][0]
    assert checkpoint["mean_training_reward"] == 1.5
    assert checkpoint["mean_bonus_pickup_count"] == 2.0
    assert checkpoint["mean_bonus_reward"] == 0.25
    assert checkpoint["mean_reward_components"] == {"bonus": 0.25, "survival": 1.25}
    assert checkpoint["terminal_cause_histogram"] == {"cap": 1, "normal_wall": 1}
    assert checkpoint["action_histogram"] == {"0": 1, "1": 3}
    assert checkpoint["action_entropy"] == 0.42
    assert checkpoint["failure_rate"] == 0.5
    assert checkpoint["eval_health"] == "has_failures"


def test_status_rolls_up_missing_reason_train_actions_poller_and_gifs(monkeypatch, tmp_path):
    monkeypatch.setattr(status_mod, "RUNS_MOUNT", tmp_path)
    run_id = "run-b"
    attempt_id = "attempt-b"
    train_ref = runs.attempt_train_ref(status_mod.TASK_ID, run_id, attempt_id)
    train_root = tmp_path / train_ref
    train_root.mkdir(parents=True)
    (train_root / "status_heartbeat.json").write_text(
        json.dumps({"status": "running", "stage": "stock_train_muzero"}),
        encoding="utf-8",
    )
    (train_root / "action_observability.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "row_count": 12,
                "ego_action_histogram": {"0": 11, "1": 1, "2": 0},
                "physical_action_histogram": {"0": 10, "1": 2, "2": 0},
                "terminal_reasons": {"wall": 1},
            }
        ),
        encoding="utf-8",
    )
    (train_root / "checkpoint_eval_poller.json").write_text(
        json.dumps(
            {
                "status": "running",
                "seen_count": 1,
                "scheduled_count": 1,
                "completed_count": 2,
                "eval_completed_count": 1,
                "gif_scheduled_count": 1,
                "gif_completed_count": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_gif_summary(
        tmp_path
        / runs.attempt_eval_ref(
            status_mod.TASK_ID, run_id, attempt_id, "live_checkpoint_iteration_1"
        )
        / "selfplay"
        / "summary.json",
        created_at="2026-05-12T03:00:00Z",
    )

    row = status_mod._run_status(
        run_id,
        attempt_id=attempt_id,
        collapse_threshold=0.9,
    )

    assert row["progress_exists"] is False
    assert row["progress_missing_reason"] == "progress_latest_absent_after_train_heartbeat"
    assert row["status_heartbeat_exists"] is True
    assert row["train_status"] == "running"
    assert row["action_observability_exists"] is True
    assert row["train_action_summary"]["top_action"] == "0"
    assert row["background_poller_status"] == "running"
    assert row["background_poller_gif_completed_count"] == 1
    assert row["gif_artifact_count"] == 1
    assert row["latest_gif_checkpoint"] == "iteration_1"
    assert row["latest_gif_action_summary"]["collapsed"] is True
    assert row["latest_gif_opponent_mixture_enabled"] is True
    assert row["latest_gif_opponent_mixture_entry_name"] == "scripted"
    assert row["latest_gif_opponent_mixture_age_label"] == "scripted_wall_avoidant"
    assert row["latest_gif_opponent_mixture_entry_weight"] == 50.0
    artifact = row["gif_artifacts"][0]
    assert artifact["opponent_mixture_enabled"] is True
    assert artifact["opponent_mixture_entry_name"] == "scripted"
    assert artifact["opponent_mixture_age_label"] == "scripted_wall_avoidant"
    assert artifact["opponent_mixture_entry_weight"] == 50.0


def test_status_treats_partial_progress_latest_as_unreadable(monkeypatch, tmp_path):
    monkeypatch.setattr(status_mod, "RUNS_MOUNT", tmp_path)
    run_id = "run-partial-progress"
    attempt_id = "attempt-partial-progress"
    train_ref = runs.attempt_train_ref(status_mod.TASK_ID, run_id, attempt_id)
    train_root = tmp_path / train_ref
    train_root.mkdir(parents=True)
    (train_root / "status_heartbeat.json").write_text(
        json.dumps({"status": "running", "stage": "stock_train_muzero"}),
        encoding="utf-8",
    )
    (train_root / "progress_latest.json").write_text("", encoding="utf-8")

    row = status_mod._run_status(
        run_id,
        attempt_id=attempt_id,
        collapse_threshold=0.9,
    )

    assert row["progress_exists"] is False
    assert row["progress_missing_reason"] == "progress_latest_unreadable"
    assert row["event"] == "unreadable"
    assert "invalid JSON" in row["progress_error"]


def test_checkpoint_summary_includes_checkpoint_mtimes(monkeypatch, tmp_path):
    monkeypatch.setattr(status_mod, "RUNS_MOUNT", tmp_path)
    run_id = "run-checkpoint-mtime"
    attempt_id = "attempt-checkpoint-mtime"
    ckpt_root = (
        tmp_path
        / runs.attempt_train_ref(status_mod.TASK_ID, run_id, attempt_id)
        / "lightzero_exp"
        / "ckpt"
    )
    ckpt_root.mkdir(parents=True)
    iteration_0 = ckpt_root / "iteration_0.pth.tar"
    iteration_10 = ckpt_root / "iteration_10.pth.tar"
    iteration_0.write_text("zero", encoding="utf-8")
    iteration_10.write_text("ten", encoding="utf-8")

    summary = status_mod._checkpoint_summary(run_id, attempt_id)

    assert summary["checkpoint_count"] == 2
    assert summary["latest_checkpoint"] == "iteration_10"
    assert summary["latest_checkpoint_mtime"] == summary["checkpoints"][1]["mtime"]
    assert [checkpoint["iteration"] for checkpoint in summary["checkpoints"]] == [0, 10]
    assert all(checkpoint["mtime"] is not None for checkpoint in summary["checkpoints"])


def test_stock_high_signal_preset_includes_attempt_ids():
    run_ids, attempt_ids = status_mod._preset_run_ids("stock-high-signal-v1")

    assert len(run_ids) == 10
    assert len(attempt_ids) == 10
    assert run_ids[0] == "curvytron-stock-stock-high-signal-v1-01-fixed-straight-sparse-b32-sim8"
    assert attempt_ids[0] == "stock-high-signal-v1-attempt-01-fixed-straight-sparse-b32-sim8"
