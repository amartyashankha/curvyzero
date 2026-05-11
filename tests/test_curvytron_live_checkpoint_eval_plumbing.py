import json

import pytest

from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod
from curvyzero.infra.modal import (
    lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod,
)


def _source_state_training_command():
    spec = train_mod._env_variant_spec(train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT)
    command = {
        "run_id": "run-source-state",
        "attempt_id": "attempt-source-state",
        "mode": "train",
        "learning_proof": False,
        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    }
    for key in train_mod._source_state_fixed_opponent_readiness_expected():
        if key in spec:
            command[key] = spec[key]
    return command


def test_source_state_variant_surface_identity_is_explicit():
    assert train_mod.DEFAULT_ENV_VARIANT == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT

    spec = train_mod._env_variant_spec(train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT)

    assert spec["env_type"] == train_mod.LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE
    assert spec["env_id"] == train_mod.LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID
    assert spec["observation_shape"] == list(train_mod.STACKED_SOURCE_STATE_GRAY64_SHAPE)
    assert spec["observation_schema_id"] == train_mod.STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID
    assert spec["debug_fidelity_only"] is False
    assert spec["source_fidelity_claim"] == "source_state_backed_non_browser_pixel"
    assert spec["single_product_runtime_path"] is True
    assert spec["legacy_debug_variant"] is False
    assert spec["underlying_env_class"] == "VectorMultiplayerEnv"
    assert spec["runtime_env_impl_id"] == train_mod.NATURAL_BONUS_ENV_IMPL_ID
    assert spec["runtime_topology"] == train_mod.SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY
    assert spec["two_seat_self_play"] is False
    assert spec["two_seat_self_play_status"] == train_mod.SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS
    assert spec["fixed_opponent_is_two_seat_self_play"] is False
    assert spec["browser_pixel_fidelity"] is False
    assert spec["uses_ale"] is False
    assert spec["visual_surface"] == "source_state_visual_tensor"
    assert spec["visual_truth_level"] == "source_state_backed_non_browser_pixel"
    assert spec["visual_source_state_backed"] is True
    assert spec["opponent_training_relation"] == train_mod.OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT
    assert spec["current_policy_self_play"] is False
    assert spec["trusted_current_policy_self_play"] is False
    assert spec["simultaneous_game_theory_claim"] is False


def test_source_state_fixed_opponent_readiness_gate_carries_to_live_summary():
    command = _source_state_training_command()
    action_observability = {
        "status": "ok",
        "row_count": 1,
        "done_count": 1,
        "observed_fields": {
            "requested_ego_action": True,
            "executed_ego_action": True,
            "fixed_opponent_action": True,
            "joint_action": True,
            "action_mask": True,
            "terminal_reason": True,
            "death_cause": True,
        },
    }

    gate = train_mod._source_state_fixed_opponent_training_readiness_gate(
        command=command,
        surface=dict(command),
        action_observability=action_observability,
    )
    summary = train_mod._live_train_summary_for_inspector(command)

    assert gate["ok"] is True
    assert gate["status"] == "ready"
    assert gate["problems"] == []
    for key, expected in train_mod._source_state_fixed_opponent_readiness_expected().items():
        assert command[key] == expected
        assert summary[key] == expected
        assert gate["checks"][f"command.{key}"]["ok"] is True
        assert gate["checks"][f"surface.{key}"]["ok"] is True
    assert gate["action_observability"]["observed_fields"] == (
        action_observability["observed_fields"]
    )
    assert summary["training_readiness_gate"]["ok"] is True
    assert summary["training_readiness_gate"]["expected"] == gate["expected"]


def test_source_state_fixed_opponent_readiness_gate_rejects_two_seat_or_browser_claims():
    command = _source_state_training_command()
    command["two_seat_self_play"] = True
    command["browser_pixel_fidelity"] = True

    gate = train_mod._source_state_fixed_opponent_training_readiness_gate(command=command)

    assert gate["ok"] is False
    assert gate["status"] == "blocked"
    assert "command.two_seat_self_play=True, expected False" in gate["problems"]
    assert "command.browser_pixel_fidelity=True, expected False" in gate["problems"]


def test_env_step_telemetry_summary_reports_action_and_death_observability(tmp_path):
    path = tmp_path / "env_steps.jsonl"
    row = {
        "schema_id": "curvyzero_source_state_visual_survival_env_step/v0",
        "telemetry_stride": 1,
        "telemetry_sampled": False,
        "scalar_action": 0,
        "ego_action": 0,
        "requested_ego_action": 0,
        "executed_ego_action": 0,
        "opponent_action_id": 1,
        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "joint_action": {"player_0": 0, "player_1": 1},
        "final_observation_action_mask": [0, 0, 0],
        "physical_env_advanced": True,
        "reward": 0.0,
        "done": True,
        "terminal_reason": "normal_wall",
        "death_cause": [[1, -1]],
        "death_cause_name": [["normal_wall", "none"]],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = train_mod._summarize_env_step_telemetry(path)

    assert summary["status"] == "ok"
    assert summary["row_count"] == 1
    assert summary["observed_fields"] == {
        "requested_ego_action": True,
        "executed_ego_action": True,
        "fixed_opponent_action": True,
        "joint_action": True,
        "action_mask": True,
        "terminal_reason": True,
        "death_cause": True,
    }


def test_legacy_debug_variants_are_not_the_default_product_runtime():
    for variant in (
        train_mod.ENV_VARIANT_FIXED_OPPONENT,
        train_mod.ENV_VARIANT_TURN_COMMIT,
    ):
        spec = train_mod._env_variant_spec(variant)

        assert spec["debug_fidelity_only"] is True
        assert spec["source_fidelity_claim"] == "none"
        assert spec["single_product_runtime_path"] is False
        assert spec["legacy_debug_variant"] is True


def test_eval_rejects_unknown_env_variant_before_checkpoint_read():
    with pytest.raises(ValueError, match="unknown env_variant"):
        eval_mod._run_eval(
            compute="cpu",
            checkpoint_ref="missing/checkpoint.pth.tar",
            output_ref="missing/eval.json",
            run_id="run",
            attempt_id="attempt",
            seed=0,
            max_eval_steps=1,
            step_detail_limit=1,
            source_max_steps=1,
            num_simulations=1,
            batch_size=1,
            emit_result_json=False,
            quiet_framework_logs=True,
            env_variant="wrong_env",
            opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
            opponent_checkpoint_ref=None,
            opponent_snapshot_ref=None,
            opponent_checkpoint_state_key=None,
        )


def test_live_checkpoint_trigger_spawns_without_volume_commit(tmp_path, monkeypatch):
    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

    class FakeFunction:
        def __init__(self) -> None:
            self.calls = []

        def spawn(self, **kwargs):
            self.calls.append(kwargs)

            class Call:
                object_id = "fc-live-eval-test"

            return Call()

    fake_volume = FakeVolume()
    fake_function = FakeFunction()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect",
        fake_function,
    )

    attempt_train_root = tmp_path / "training" / train_mod.TASK_ID / "run-a" / "attempts" / "attempt-a" / "train"
    publish = {
        "run_id": "run-a",
        "attempt_id": "attempt-a",
        "checkpoint_mirror": {
            "copied_checkpoints": [
                {
                    "ref": "training/lightzero-curvytron-visual-survival/run-a/checkpoints/lightzero/iteration_0.pth.tar",
                    "copied_now": False,
                },
                {
                    "ref": "training/lightzero-curvytron-visual-survival/run-a/checkpoints/lightzero/iteration_1.pth.tar",
                    "copied_now": True,
                },
            ]
        },
    }
    config = {
        "enabled": True,
        "compute": "cpu",
        "eval_id_prefix": "live_checkpoint",
        "seed": 7,
        "eval_seed_count": 2,
        "eval_seed_rng_seed": 11,
        "max_eval_steps": 32,
        "step_detail_limit": 2,
        "source_max_steps": 32,
        "num_simulations": 4,
        "batch_size": 8,
        "env_variant": train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
    }

    result = train_mod._schedule_live_checkpoint_background_eval(
        publish=publish,
        attempt_train_root=attempt_train_root,
        config=config,
    )

    assert result["scheduled"] is True
    assert len(fake_function.calls) == 1
    call = fake_function.calls[0]
    assert call["checkpoint_ref"].endswith("iteration_1.pth.tar")
    assert call["env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
    assert call["eval_seed_count"] == 2
    assert fake_volume.commit_count == 0
    assert not (attempt_train_root / "background_eval_requests").exists()
    request = result["requests"][0]
    assert request["status"] == "spawned"
    assert request["function_call_id"] == "fc-live-eval-test"
    assert request["config"]["env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT


def test_save_hook_trigger_uses_source_checkpoint_ref_not_future_mirror(tmp_path, monkeypatch):
    class FakeFunction:
        def __init__(self) -> None:
            self.calls = []

        def spawn(self, **kwargs):
            self.calls.append(kwargs)

            class Call:
                object_id = "fc-source-ref-test"

            return Call()

    fake_function = FakeFunction()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect",
        fake_function,
    )
    monkeypatch.chdir(tmp_path)

    exp_name = (
        tmp_path
        / "training"
        / train_mod.TASK_ID
        / "run-b"
        / "attempts"
        / "attempt-b"
        / "train"
        / "lightzero_exp"
    )
    checkpoint = exp_name / "ckpt" / "iteration_7.pth.tar"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake checkpoint")

    config = {
        "enabled": True,
        "compute": "cpu",
        "eval_id_prefix": "live_checkpoint",
        "seed": 7,
        "eval_seed_count": 2,
        "eval_seed_rng_seed": 11,
        "max_eval_steps": 32,
        "step_detail_limit": 2,
        "source_max_steps": 32,
        "num_simulations": 4,
        "batch_size": 8,
        "env_variant": train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
    }

    result = train_mod._spawn_checkpoint_eval_triggers(
        run_id="run-b",
        attempt_id="attempt-b",
        exp_name=exp_name,
        config=config,
        seen_checkpoint_refs=set(),
    )

    assert result[0]["status"] == "spawned"
    call = fake_function.calls[0]
    assert call["checkpoint_ref"].endswith(
        "training/lightzero-curvytron-visual-survival/run-b/attempts/attempt-b/train/lightzero_exp/ckpt/iteration_7.pth.tar"
    )
    assert "/checkpoints/lightzero/" not in call["checkpoint_ref"]
    assert result[0]["checkpoint"]["canonical_ref"].endswith(
        "training/lightzero-curvytron-visual-survival/run-b/checkpoints/lightzero/iteration_7.pth.tar"
    )


def test_checkpoint_eval_poller_spawns_eval_and_inspection_jobs(tmp_path, monkeypatch):
    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0
            self.reload_count = 0

        def commit(self) -> None:
            self.commit_count += 1

        def reload(self) -> None:
            self.reload_count += 1

    class FakeCall:
        object_id = "fc-poller-eval"

        def get(self):
            return {
                "ok": True,
                "inspection_report_ref": "eval/live_checkpoint_iteration_1/inspection/report.json",
                "inspection_report_markdown_ref": "eval/live_checkpoint_iteration_1/inspection/report.md",
            }

    class FakeEvalInspectFunction:
        def __init__(self) -> None:
            self.calls = []

        def spawn(self, **kwargs):
            self.calls.append(kwargs)
            return FakeCall()

    fake_volume = FakeVolume()
    fake_function = FakeEvalInspectFunction()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect",
        fake_function,
    )

    exp_name_ref = (
        train_mod.runs.attempt_train_ref(train_mod.TASK_ID, "run-c", "attempt-c")
        / "lightzero_exp"
    ).as_posix()
    exp_name = train_mod.runs.volume_path(tmp_path, exp_name_ref)
    ckpt_dir = exp_name / "ckpt"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "iteration_1.pth.tar").write_bytes(b"iteration checkpoint")
    (ckpt_dir / "ckpt_best.pth.tar").write_bytes(b"mutable best checkpoint")

    summary_path = train_mod.runs.volume_path(
        tmp_path,
        train_mod.runs.attempt_train_ref(train_mod.TASK_ID, "run-c", "attempt-c")
        / "summary.json",
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('{"ok": true}', encoding="utf-8")

    command = train_mod._checkpoint_eval_poller_command(
        seed=3,
        source_max_steps=32,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        background_eval_enabled=True,
        background_eval_compute="cpu",
        background_eval_id_prefix="live_checkpoint",
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=0,
        background_eval_max_steps=32,
        background_eval_step_detail_limit=2,
        background_eval_num_simulations=4,
        background_eval_batch_size=8,
    )

    result = train_mod._run_checkpoint_eval_poller(
        run_id="run-c",
        attempt_id="attempt-c",
        exp_name_ref=exp_name_ref,
        poll_interval_sec=0.01,
        stable_polls=0,
        max_runtime_sec=1.0,
        idle_after_train_done_sec=0.0,
        command=command,
    )

    assert result["scheduled_count"] == 1
    assert result["completed_count"] == 1
    assert result["completed"][0]["result"]["inspection_report_ref"].endswith(
        "inspection/report.json"
    )
    assert len(fake_function.calls) == 1
    call = fake_function.calls[0]
    assert call["checkpoint_ref"].endswith(
        "training/lightzero-curvytron-visual-survival/run-c/attempts/attempt-c/train/lightzero_exp/ckpt/iteration_1.pth.tar"
    )
    assert call["eval_id"] == "live_checkpoint_iteration_1"
    assert fake_volume.commit_count == 0
    status_path = train_mod._checkpoint_eval_poller_status_path(
        run_id="run-c",
        attempt_id="attempt-c",
    )
    assert status_path.exists()


def test_local_launcher_starts_trainer_and_eval_inspection_poller(capsys, monkeypatch):
    class FakeCall:
        def __init__(self, object_id: str) -> None:
            self.object_id = object_id

    class FakeFunction:
        def __init__(self, object_id: str) -> None:
            self.object_id = object_id
            self.calls = []

        def spawn(self, **kwargs):
            self.calls.append(kwargs)
            return FakeCall(self.object_id)

    fake_train = FakeFunction("fc-train")
    fake_poller = FakeFunction("fc-poller")
    monkeypatch.setattr(train_mod, "lightzero_curvytron_visual_survival_cpu", fake_train)
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
        fake_poller,
    )

    train_mod.main(
        mode="train",
        compute="cpu",
        run_id="run-d",
        attempt_id="attempt-d",
        wait_for_train=False,
        background_eval_enabled=True,
        background_eval_launch_kind=train_mod.BACKGROUND_EVAL_LAUNCH_POLLER,
        background_eval_seed_count=1,
        background_eval_max_steps=32,
        background_eval_num_simulations=4,
        background_eval_batch_size=8,
        background_eval_poll_interval_sec=0.01,
        background_eval_poller_max_runtime_sec=1.0,
        background_eval_poller_idle_after_done_sec=0.0,
    )

    assert len(fake_train.calls) == 1
    assert len(fake_poller.calls) == 1
    assert fake_train.calls[0]["background_eval_launch_kind"] == train_mod.BACKGROUND_EVAL_LAUNCH_POLLER
    assert fake_train.calls[0]["env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
    assert fake_train.calls[0]["opponent_policy_kind"] == train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT
    assert fake_poller.calls[0]["exp_name_ref"].endswith(
        "attempts/attempt-d/train/lightzero_exp"
    )
    assert fake_poller.calls[0]["env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
    assert fake_poller.calls[0]["opponent_policy_kind"] == train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT
    assert fake_poller.calls[0]["background_eval_seed_count"] == 1
    assert fake_poller.calls[0]["background_eval_max_steps"] == 32
    printed = capsys.readouterr().out
    assert '"function_call_id": "fc-train"' in printed
    assert '"poller_function_call_id": "fc-poller"' in printed
    assert '"launch_kind": "poller"' in printed
