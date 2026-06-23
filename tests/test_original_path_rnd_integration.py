import json
import sys
import types
from pathlib import Path

from curvyzero.infra.modal import (
    lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod,
)
from curvyzero.training import exploration_bonus as xb


class _FakeRunsVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def test_original_lightzero_path_rnd_meter_uses_reward_model_entrypoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_calls = []

    def fake_train_muzero(_configs, *, seed, max_train_iter, max_env_step):
        raise AssertionError("stock train_muzero should not run for rnd_meter_v0")

    def fake_train_muzero_with_reward_model(configs, *, seed, max_train_iter, max_env_step):
        train_calls.append(
            {
                "configs": configs,
                "seed": seed,
                "max_train_iter": max_train_iter,
                "max_env_step": max_env_step,
                "patched_reward_model": globals().get("RNDRewardModel"),
            }
        )
        return {"trained": True}

    fake_volume = _install_minimal_visual_train_mocks(
        monkeypatch,
        tmp_path,
        fake_train_muzero,
    )
    sys.modules["lzero.entry"].train_muzero_with_reward_model = (
        fake_train_muzero_with_reward_model
    )

    result = train_mod._run_visual_survival_train(
        **{
            **_minimal_visual_train_kwargs(
                run_id="rnd-meter-entrypoint-local-test",
                attempt_id="attempt-001",
            ),
            "exploration_bonus_mode": "rnd_meter_v0",
            "exploration_bonus_weight": 0.0,
            "exploration_bonus_feature_source": "policy_gray64_latest/v0",
            "exploration_bonus_rnd_batch_size": 32,
        }
    )

    assert result["ok"] is True
    assert result["trainer_entrypoint"] == "lzero.entry.train_muzero_with_reward_model"
    assert result["command"]["exploration_bonus"]["mode"] == "rnd_meter_v0"
    assert result["command"]["exploration_bonus"]["training_effect"] == (
        "reward_target_unchanged"
    )
    assert result["command"]["exploration_bonus_rnd_batch_size"] == 32
    assert fake_volume.commit_count == 1
    assert len(train_calls) == 1
    assert train_calls[0]["patched_reward_model"] is xb.CurvyRNDRewardModel
    reward_model_config = train_calls[0]["configs"][0]["reward_model"]
    assert reward_model_config["seed"] == 123
    assert reward_model_config["intrinsic_reward_weight"] == 0.0
    assert reward_model_config["curvyzero_metrics_latest_path"].endswith(
        "rnd_reward_model_metrics_latest.json"
    )
    assert reward_model_config["curvyzero_metrics_jsonl_path"].endswith(
        "rnd_reward_model_metrics.jsonl"
    )
    assert result["rnd_reward_model_metrics"]["enabled"] is True
    assert result["rnd_reward_model_metrics"]["latest_exists"] is False
    summary = json.loads((tmp_path / result["summary_ref"]).read_text(encoding="utf-8"))
    assert summary["trainer_entrypoint"] == "lzero.entry.train_muzero_with_reward_model"
    assert summary["command"]["exploration_bonus"]["input_spec"]["shape"] == [1, 64, 64]
    assert summary["command"]["rnd_reward_model_metrics"]["enabled"] is True


def test_original_lightzero_path_can_require_rnd_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_calls = []

    def fake_train_muzero(_configs, *, seed, max_train_iter, max_env_step):
        raise AssertionError("stock train_muzero should not run for rnd_meter_v0")

    def fake_train_muzero_with_reward_model(configs, *, seed, max_train_iter, max_env_step):
        del seed, max_train_iter, max_env_step
        train_calls.append({"patched_reward_model": globals().get("RNDRewardModel")})
        metrics_config = configs[0]["reward_model"]
        metrics = {
            "schema_id": "curvyzero_rnd_reward_model_metrics/v0",
            "collect_data_calls": 1,
            "train_with_data_calls": 1,
            "estimate_calls": 1,
            "train_cnt_rnd": 1,
            "last_predictor_hash_before_train": "predictor-before",
            "last_predictor_hash_after_train": "predictor-after",
            "last_target_hash_before_train": "target-stable",
            "last_target_hash_after_train": "target-stable",
            "last_target_reward_changed": False,
            "reason": "test_required_rnd_metrics",
        }
        latest_path = Path(metrics_config["curvyzero_metrics_latest_path"])
        jsonl_path = Path(metrics_config["curvyzero_metrics_jsonl_path"])
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(json.dumps(metrics), encoding="utf-8")
        jsonl_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        return {"trained": True}

    _install_minimal_visual_train_mocks(monkeypatch, tmp_path, fake_train_muzero)
    sys.modules["lzero.entry"].train_muzero_with_reward_model = (
        fake_train_muzero_with_reward_model
    )

    result = train_mod._run_visual_survival_train(
        **{
            **_minimal_visual_train_kwargs(
                run_id="rnd-required-metrics-local-test",
                attempt_id="attempt-001",
            ),
            "exploration_bonus_mode": "rnd_meter_v0",
            "exploration_bonus_weight": 0.0,
            "exploration_bonus_feature_source": "policy_gray64_latest/v0",
            "require_rnd_metrics": True,
        }
    )

    assert result["ok"] is True
    assert result["trainer_entrypoint"] == "lzero.entry.train_muzero_with_reward_model"
    assert train_calls[0]["patched_reward_model"] is xb.CurvyRNDRewardModel
    assert result["rnd_reward_model_metrics"]["latest_exists"] is True
    assert result["rnd_reward_model_metrics"]["event_count"] == 1
    assert result["rnd_reward_model_metrics"]["latest"]["last_target_reward_changed"] is False


def _install_minimal_visual_train_mocks(
    monkeypatch,
    tmp_path: Path,
    fake_train_muzero,
) -> _FakeRunsVolume:
    fake_volume = _FakeRunsVolume()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)
    monkeypatch.setattr(
        train_mod, "_version_or_missing", lambda *_args: train_mod.LIGHTZERO_VERSION
    )
    monkeypatch.setattr(train_mod.os, "chdir", lambda _path: None)

    def fake_build_visual_survival_configs(**kwargs):
        bonus_spec = xb.normalize_exploration_bonus_config(kwargs.get("exploration_bonus"))
        main_config = {"env": {"type": "fake-env"}, "policy": {"cuda": False}}
        if bonus_spec.enabled:
            main_config["reward_model"] = xb.lightzero_reward_model_config(
                bonus_spec,
                seed=kwargs["seed"],
            )
        return {
            "main_config": main_config,
            "create_config": {"env": {"type": "fake-env"}},
            "patches": [{"path": "fake", "new": True}],
            "surface": {
                "env_variant": kwargs["env_variant"],
                "env_type": "fake-env",
                "called_from": "minimal-visual-train-mock",
            },
        }

    monkeypatch.setattr(
        train_mod,
        "_build_visual_survival_configs",
        fake_build_visual_survival_configs,
    )
    monkeypatch.setattr(
        train_mod,
        "_compile_config_summary",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(train_mod, "_validate_visual_survival_surface", lambda **_kwargs: [])
    monkeypatch.setattr(
        train_mod,
        "_install_lightzero_full_resume_state_hooks",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(train_mod, "_install_checkpoint_progress_writer", lambda **_kwargs: None)
    monkeypatch.setattr(train_mod, "_install_live_checkpoint_publisher", lambda **_kwargs: None)
    monkeypatch.setattr(train_mod, "_install_lightzero_target_audit", lambda **_kwargs: None)
    monkeypatch.setattr(
        train_mod,
        "_scan_lightzero_artifacts",
        lambda _exp_name: {
            "checkpoint_files": [{"name": "iteration_0.pth.tar"}],
            "resume_state_files": [],
        },
    )
    monkeypatch.setattr(
        train_mod,
        "_mirror_lightzero_checkpoints",
        lambda **_kwargs: {"copied_checkpoints": [{"checkpoint": "iteration_0"}]},
    )
    monkeypatch.setattr(
        train_mod,
        "_summarize_env_step_telemetry",
        lambda _path: {"row_count": 1, "action_summary": {"collapsed": False}},
    )
    monkeypatch.setattr(
        train_mod,
        "_source_state_fixed_opponent_training_readiness_gate",
        lambda **_kwargs: {"ok": True},
    )
    fake_entry_module = types.ModuleType("lzero.entry")
    fake_entry_module.train_muzero = fake_train_muzero
    monkeypatch.setitem(sys.modules, "lzero.entry", fake_entry_module)
    return fake_volume


def _minimal_visual_train_kwargs(
    *,
    run_id: str,
    attempt_id: str,
) -> dict[str, object]:
    return {
        "mode": "train",
        "compute": "cpu",
        "seed": 123,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "max_env_step": 8,
        "max_train_iter": 1,
        "source_max_steps": 8,
        "decision_ms": train_mod.DEFAULT_DECISION_MS,
        "collector_env_num": 2,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": 1,
        "num_simulations": 1,
        "batch_size": 4,
        "lightzero_eval_freq": 0,
        "skip_lightzero_eval_in_profile": False,
        "profile_cuda_sync_enabled": False,
        "profile_allow_auto_resume": False,
        "profile_volume_commit": False,
        "lightzero_multi_gpu": False,
        "save_ckpt_after_iter": 1,
        "commit_on_checkpoint": False,
        "stop_after_learner_train_calls": 0,
        "env_variant": train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": train_mod.DEFAULT_REWARD_VARIANT,
        "source_state_trail_render_mode": train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
        "source_state_bonus_render_mode": train_mod.DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
        "policy_observation_backend": train_mod.DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
        "learner_seat_mode": train_mod.DEFAULT_LEARNER_SEAT_MODE,
        "ego_action_straight_override_probability": 0.0,
        "policy_action_repeat_min": 1,
        "policy_action_repeat_max": 1,
        "policy_action_repeat_extra_probability": 0.0,
        "control_noise_profile_id": train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        "disable_death_for_profile": False,
        "opponent_death_mode": train_mod.DEFAULT_OPPONENT_DEATH_MODE,
        "opponent_runtime_mode": train_mod.DEFAULT_OPPONENT_RUNTIME_MODE,
        "env_telemetry_stride": 1,
        "env_manager_type": "base",
        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_use_cuda": False,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_report_ref": None,
        "opponent_checkpoint_state_key": None,
        "opponent_mixture_spec": None,
        "opponent_assignment_ref": None,
        "opponent_assignment_refresh_interval_train_iter": 0,
        "opponent_assignment_refresh_ref": None,
        "background_eval_enabled": False,
        "background_eval_launch_kind": train_mod.BACKGROUND_EVAL_LAUNCH_HOOK,
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": train_mod.DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
        "background_eval_seed_count": 1,
        "background_eval_seed_rng_seed": None,
        "background_eval_max_steps": 8,
        "background_eval_step_detail_limit": None,
        "background_eval_num_simulations": 1,
        "background_eval_batch_size": 4,
        "background_gif_enabled": False,
        "background_gif_seed_offset": 1,
        "background_gif_max_steps": 8,
        "background_gif_frame_stride": 1,
        "background_gif_fps": 8.0,
        "background_gif_scale": 1,
        "background_gif_frame_size": train_mod.DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
        "background_gif_collect_temperature": train_mod.DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
        "background_gif_collect_epsilon": train_mod.DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
    }
