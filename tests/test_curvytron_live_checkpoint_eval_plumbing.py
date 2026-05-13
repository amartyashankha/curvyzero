import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod
from curvyzero.infra.modal import (
    lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod,
)


def _modal_image_copy_mount_entries(image):
    sync_original = next(
        value
        for key, value in image.__dict__.items()
        if key.startswith("_sync_original")
    )
    entries = []
    for cell in sync_original._load.__closure__ or ():
        value = cell.cell_contents
        if not callable(value):
            continue
        for nested_cell in value.__closure__ or ():
            mount = nested_cell.cell_contents
            entries.extend(getattr(mount, "_entries", ()))
    return entries


def _source_state_training_command():
    spec = train_mod._env_variant_spec(train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT)
    command = {
        "run_id": "run-source-state",
        "attempt_id": "attempt-source-state",
        "mode": "train",
        "learning_proof": False,
        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    }
    for key, expected in train_mod._source_state_fixed_opponent_readiness_expected().items():
        command[key] = spec.get(key, expected)
    return command


def test_checkpoint_progress_writer_updates_browser_speed_file(monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    exp_name = tmp_path / "training" / train_mod.TASK_ID / "run-a" / "attempts" / "attempt-a" / "train" / "lightzero_exp"
    attempt_train_root = exp_name.parent

    class FakeBaseLearner:
        train_iter = 17

        def save_checkpoint(self):
            checkpoint_path = exp_name / "ckpt" / "iteration_17.pth.tar"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"fake-checkpoint")
            return {"ok": True}

    def fake_train_muzero():
        return None

    monkeypatch.setitem(fake_train_muzero.__globals__, "BaseLearner", FakeBaseLearner)
    restore = train_mod._install_checkpoint_progress_writer(
        train_muzero=fake_train_muzero,
        run_id="run-a",
        attempt_id="attempt-a",
        exp_name=exp_name,
        attempt_train_root=attempt_train_root,
        started_monotonic=0.0,
    )

    try:
        assert restore is not None
        assert FakeBaseLearner().save_checkpoint() == {"ok": True}
    finally:
        if restore is not None:
            restore()

    progress_path = attempt_train_root / "progress_latest.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["iteration"] == 17
    assert progress["learner_train_iter"] == 17
    assert progress["event"] == "checkpoint"
    assert progress["elapsed_sec"] >= 0.0
    assert progress["checkpoint_name"] == "iteration_17.pth.tar"
    assert progress["checkpoint_ref"].endswith("lightzero_exp/ckpt/iteration_17.pth.tar")


def test_save_ckpt_hook_updates_browser_speed_file(monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    exp_name = tmp_path / "training" / train_mod.TASK_ID / "run-b" / "attempts" / "attempt-b" / "train" / "lightzero_exp"
    attempt_train_root = exp_name.parent

    class FakeBaseLearner:
        def call_hook(self, *args, **kwargs):
            return None

    class FakeSaveCkptHook:
        def __call__(self, engine):
            checkpoint_path = exp_name / "ckpt" / "iteration_15000.pth.tar"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"fake-checkpoint")
            return {"saved": True}

    class FakeEngine:
        train_iter = 15000

    def fake_train_muzero():
        return None

    fake_hook_module = types.ModuleType("ding.worker.learner.learner_hook")
    fake_hook_module.SaveCkptHook = FakeSaveCkptHook
    monkeypatch.setitem(sys.modules, "ding.worker.learner.learner_hook", fake_hook_module)
    monkeypatch.setitem(fake_train_muzero.__globals__, "BaseLearner", FakeBaseLearner)
    monkeypatch.setattr(
        train_mod,
        "_save_lightzero_resume_sidecar_state",
        lambda **kwargs: {"saved": True},
    )

    restore = train_mod._install_lightzero_full_resume_state_hooks(
        train_muzero=fake_train_muzero,
        run_id="run-b",
        attempt_id="attempt-b",
        exp_name=exp_name,
        auto_resume={},
        attempt_train_root=attempt_train_root,
        started_monotonic=0.0,
    )

    try:
        assert restore is not None
        assert FakeSaveCkptHook()(FakeEngine()) == {"saved": True}
    finally:
        if restore is not None:
            restore()

    progress_path = attempt_train_root / "progress_latest.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["iteration"] == 15000
    assert progress["learner_train_iter"] == 15000
    assert progress["event"] == "checkpoint"
    assert progress["source"] == "SaveCkptHook.__call__"
    assert progress["checkpoint_name"] == "iteration_15000.pth.tar"
    assert progress["checkpoint_ref"].endswith("lightzero_exp/ckpt/iteration_15000.pth.tar")


def test_eval_episode_and_tables_preserve_reward_components(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.step_index = 0

        def reset(self, seed=None):
            return {
                "observation": np.zeros((4, 64, 64), dtype=np.float32),
                "action_mask": np.ones(3, dtype=np.float32),
                "to_play": -1,
            }

        def step(self, action):
            self.step_index += 1
            done = self.step_index == 2
            info = {
                "terminal_reason": "round_survivor_win" if done else "none",
                "dense_survival_helper_for_ego": 1.0,
                "sparse_outcome_reward_for_ego": 0.0,
                "bonus_catch_count_step_for_ego": 1 if done else 0,
                "bonus_pickup_reward_for_ego": 5.0 if done else 0.0,
            }
            reward = 6.0 if done else 1.0
            observation = {
                "observation": np.zeros((4, 64, 64), dtype=np.float32),
                "action_mask": np.ones(3, dtype=np.float32),
                "to_play": -1,
            }
            return observation, reward, done, info

        def close(self):
            return None

    monkeypatch.setattr(
        eval_mod,
        "_policy_eval_action",
        lambda policy, observation: {"action": 1, "compact_output": {}},
    )

    episode = eval_mod._run_survival_episode(
        policy=object(),
        env=FakeEnv(),
        seed=123,
        max_eval_steps=8,
        step_detail_limit=None,
    )
    row = eval_mod._row_from_result(
        {
            "index": 0,
            "checkpoint_label": "iteration_7",
            "seed": 123,
            "checkpoint_ref": "ckpt/iteration_7.pth.tar",
        },
        {
            "ok": True,
            "episode": episode,
            "status": {"strict_policy_model_load_ok": True},
            "artifact": {"ref": "eval/row.json"},
        },
    )
    aggregate = eval_mod._survival_aggregate_table([row])[0]
    survival = eval_mod._survival_table([row])[0]

    assert episode["total_reward"] == 7.0
    assert episode["survival_reward"] == 2.0
    assert episode["bonus_pickup_count"] == 1
    assert episode["bonus_reward"] == 5.0
    assert episode["reward_components"] == {
        "bonus": 5.0,
        "survival": 2.0,
    }
    assert row["training_reward"] == 7.0
    assert row["reward_components"]["bonus"] == 5.0
    assert aggregate["mean_training_reward"] == 7.0
    assert aggregate["mean_survival_reward"] == 2.0
    assert aggregate["mean_bonus_pickup_count"] == 1.0
    assert aggregate["mean_bonus_reward"] == 5.0
    assert aggregate["mean_reward_components"] == {
        "bonus": 5.0,
        "survival": 2.0,
    }
    assert survival["training_reward"] == 7.0
    assert survival["bonus_pickup_count"] == 1
    assert survival["reward_components"]["survival"] == 2.0


def _source_state_fixed_opponent_wrapper_info():
    env = train_mod.CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {"source_max_steps": 1, "max_ticks": 1}
    )
    try:
        return env._base_info()
    finally:
        env.close()


def _install_fake_lightzero_atari_config(monkeypatch):
    class EasyDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = value

    main_config = {
        "exp_name": "fake-exp",
        "env": {"env_id": "PongNoFrameskip-v4"},
        "policy": {
            "cuda": False,
            "multi_gpu": False,
            "collector_env_num": 1,
            "evaluator_env_num": 1,
            "n_episode": 1,
            "num_simulations": 4,
            "batch_size": 16,
            "eval_freq": 1000,
            "discount_factor": 0.997,
            "td_steps": 5,
            "model": {
                "model_type": "conv",
                "image_channel": 4,
                "frame_stack_num": 1,
                "self_supervised_learning_loss": False,
                "observation_shape": [4, 64, 64],
                "action_space_size": 3,
                "support_scale": 10,
                "reward_support_size": 601,
                "value_support_size": 601,
            },
            "learn": {"learner": {"hook": {"save_ckpt_after_iter": 1000}}},
        },
    }
    monkeypatch.setitem(
        sys.modules,
        "easydict",
        types.SimpleNamespace(EasyDict=EasyDict),
    )
    monkeypatch.setitem(sys.modules, "zoo", types.ModuleType("zoo"))
    monkeypatch.setitem(sys.modules, "zoo.atari", types.ModuleType("zoo.atari"))
    monkeypatch.setitem(
        sys.modules,
        "zoo.atari.config",
        types.ModuleType("zoo.atari.config"),
    )
    monkeypatch.setitem(
        sys.modules,
        "zoo.atari.config.atari_muzero_config",
        types.SimpleNamespace(main_config=main_config),
    )


def test_default_env_variant_stays_fixed_opponent_while_turn_commit_is_profile_only():
    assert train_mod.DEFAULT_ENV_VARIANT == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT

    spec = train_mod._env_variant_spec(train_mod.ENV_VARIANT_SOURCE_STATE_TURN_COMMIT)

    assert spec["env_type"] == train_mod.LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE
    assert spec["env_id"] == train_mod.LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID
    assert spec["observation_shape"] == list(
        train_mod.TURN_COMMIT_STACKED_SOURCE_STATE_GRAY64_SHAPE
    )
    assert (
        spec["observation_schema_id"]
        == train_mod.TURN_COMMIT_STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID
    )
    assert spec["raw_observation_schema_id"] == train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
    assert spec["raw_frame_shape"] == list(
        train_mod.TURN_COMMIT_SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE
    )
    assert spec["debug_fidelity_only"] is False
    assert spec["source_fidelity_claim"] == "source_state_backed_non_browser_pixel"
    assert spec["single_product_runtime_path"] is True
    assert spec["legacy_debug_variant"] is False
    assert spec["turn_commit_adapter"] is True
    assert spec["two_seat_self_play"] is False
    assert spec["current_policy_two_seat_action_collection"] is True
    assert spec["trusted_current_policy_self_play"] is False
    assert spec["visual_source_state_backed"] is True


def test_stock_frozen_opponent_cuda_is_decoupled_from_gpu_learner(monkeypatch, tmp_path):
    _install_fake_lightzero_atari_config(monkeypatch)
    patched = train_mod._build_visual_survival_configs(
        seed=7,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=True,
        max_env_step=128,
        source_max_steps=64,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=2,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=8,
        batch_size=16,
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=8,
        save_ckpt_after_iter=100,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=64,
        env_manager_type="subprocess",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
        opponent_use_cuda=False,
        opponent_checkpoint={
            "resolved_checkpoint_path": "/tmp/frozen.pth.tar",
            "checkpoint_ref": "runs/checkpoints/iteration_7.pth.tar",
        },
        opponent_snapshot_ref="stage-007",
        opponent_checkpoint_state_key="model",
    )

    assert patched["main_config"]["policy"]["cuda"] is True
    assert patched["main_config"]["env"]["opponent_use_cuda"] is False
    assert patched["surface"]["cuda"] is True
    assert patched["surface"]["opponent_use_cuda"] is False


def test_stock_frozen_opponent_cuda_can_still_be_explicitly_enabled(monkeypatch, tmp_path):
    _install_fake_lightzero_atari_config(monkeypatch)
    patched = train_mod._build_visual_survival_configs(
        seed=7,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=True,
        max_env_step=128,
        source_max_steps=64,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=8,
        batch_size=16,
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=8,
        save_ckpt_after_iter=100,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=64,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
        opponent_use_cuda=True,
        opponent_checkpoint={
            "resolved_checkpoint_path": "/tmp/frozen.pth.tar",
            "checkpoint_ref": "runs/checkpoints/iteration_7.pth.tar",
        },
        opponent_snapshot_ref="stage-007",
        opponent_checkpoint_state_key="model",
    )

    assert patched["main_config"]["env"]["opponent_use_cuda"] is True
    assert patched["surface"]["opponent_use_cuda"] is True


def test_stock_train_accepts_proactive_wall_avoidant_source_state_opponent(
    monkeypatch,
    tmp_path,
):
    _install_fake_lightzero_atari_config(monkeypatch)

    patched = train_mod._build_visual_survival_configs(
        seed=7,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=False,
        max_env_step=128,
        source_max_steps=64,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=8,
        batch_size=16,
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=8,
        save_ckpt_after_iter=100,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=64,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
    )

    env_cfg = patched["main_config"]["env"]
    assert (
        train_mod.OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
        in train_mod.OPPONENT_POLICY_KIND_CHOICES
    )
    assert env_cfg["opponent_policy_kind"] == (
        train_mod.OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
    )
    assert env_cfg["opponent_training_relation"] == (
        train_mod.OPPONENT_TRAINING_RELATION_PROACTIVE_WALL_AVOIDANT
    )
    assert "opponent_checkpoint_path" not in env_cfg
    assert patched["surface"]["opponent_policy_kind"] == (
        train_mod.OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
    )


def test_source_state_opponent_mixture_uses_matching_surface_relation(
    monkeypatch,
    tmp_path,
):
    _install_fake_lightzero_atari_config(monkeypatch)
    mixture = train_mod.parse_opponent_mixture_spec(
        {
            "entries": [
                {
                    "name": "blank",
                    "weight": 1,
                    "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                    "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                }
            ]
        }
    )

    patched = train_mod._build_visual_survival_configs(
        seed=7,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=False,
        max_env_step=128,
        source_max_steps=64,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=8,
        batch_size=16,
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=8,
        save_ckpt_after_iter=100,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=64,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_mixture=mixture,
    )

    relation = train_mod.OPPONENT_TRAINING_RELATION_WEIGHTED_EPISODE_MIXTURE
    env_cfg = patched["main_config"]["env"]
    assert env_cfg["opponent_training_relation"] == relation
    assert patched["surface"]["opponent_training_relation"] == relation
    assert patched["surface"]["opponent_mixture"] == mixture


def test_survival_plus_bonus_no_outcome_uses_capped_separate_supports(
    monkeypatch,
    tmp_path,
):
    _install_fake_lightzero_atari_config(monkeypatch)

    patched = train_mod._build_visual_survival_configs(
        seed=7,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=False,
        max_env_step=65_536,
        source_max_steps=65_536,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=8,
        batch_size=16,
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=8,
        save_ckpt_after_iter=100,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=64,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
    )

    model_cfg = patched["main_config"]["policy"]["model"]
    env_cfg = patched["main_config"]["env"]
    target_config = env_cfg["lightzero_target_config"]
    reward_policy = env_cfg["reward_policy"]

    assert train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME in (
        train_mod.REWARD_VARIANT_CHOICES
    )
    assert env_cfg["reward_variant"] == (
        train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
    )
    assert reward_policy["sparse_outcome_reward"] is False
    assert reward_policy["sparse_outcome_telemetry_only"] is True
    assert reward_policy["same_step_bonus_pickup_reward"] is True
    assert target_config["uncapped_model_reward_support_scale"] == 2
    assert target_config["uncapped_model_value_support_scale"] == 131_072
    assert target_config["model_reward_support_capped"] is False
    assert target_config["model_value_support_capped"] is True
    assert target_config["model_support_cap"] == 300
    assert model_cfg["support_scale"] == 300
    assert model_cfg["reward_support_size"] == 601
    assert model_cfg["value_support_size"] == 601
    assert model_cfg["reward_support_range"] == (-300.0, 301.0, 1.0)
    assert model_cfg["value_support_range"] == (-300.0, 301.0, 1.0)
    assert target_config["model_reward_support_requested_scale"] == 2
    assert target_config["model_value_support_requested_scale"] == 131_072
    assert target_config["model_reward_support_effective_scale"] == 300
    assert target_config["model_value_support_effective_scale"] == 300
    assert patched["surface"]["model_reward_support_size"] == 601
    assert patched["surface"]["model_value_support_size"] == 601


def test_modal_config_passes_opponent_runtime_mode_through(monkeypatch, tmp_path):
    _install_fake_lightzero_atari_config(monkeypatch)

    patched = train_mod._build_visual_survival_configs(
        seed=11,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=False,
        max_env_step=128,
        source_max_steps=128,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=8,
        batch_size=16,
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=8,
        save_ckpt_after_iter=100,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=64,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_runtime_mode=train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    )

    env_cfg = patched["main_config"]["env"]
    surface = patched["surface"]
    assert env_cfg["opponent_runtime_mode"] == "blank_canvas_noop"
    assert env_cfg["opponent_collision_effect"] == (
        "disabled_no_player_1_movement_trail_collision_bonus_side_effects"
    )
    assert surface["opponent_runtime_mode"] == "blank_canvas_noop"
    assert surface["opponent_trail_mode"] == "none_blank_canvas_scrubbed"


def test_modal_config_passes_stock_policy_action_repeat_through(monkeypatch, tmp_path):
    _install_fake_lightzero_atari_config(monkeypatch)

    patched = train_mod._build_visual_survival_configs(
        seed=13,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=False,
        max_env_step=128,
        source_max_steps=128,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=8,
        batch_size=16,
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=8,
        save_ckpt_after_iter=100,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        ego_action_straight_override_probability=0.1,
        control_noise_profile_id="stock-repeat-smoke",
        policy_action_repeat_min=1,
        policy_action_repeat_max=3,
        policy_action_repeat_extra_probability=0.25,
        disable_death_for_profile=False,
        env_telemetry_stride=64,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
    )

    env_cfg = patched["main_config"]["env"]
    surface = patched["surface"]
    assert env_cfg["ego_action_straight_override_probability"] == 0.1
    assert env_cfg["policy_action_repeat_min"] == 1
    assert env_cfg["policy_action_repeat_max"] == 3
    assert env_cfg["policy_action_repeat_extra_probability"] == 0.25
    assert env_cfg["policy_action_repeat_semantics"] == (
        "repeat_selected_policy_action_inside_one_lightzero_env_step"
    )
    assert env_cfg["control_noise_profile_id"] == "stock-repeat-smoke"
    assert surface["policy_action_repeat_min"] == 1
    assert surface["policy_action_repeat_max"] == 3
    assert surface["policy_action_repeat_extra_probability"] == 0.25
    assert surface["policy_action_repeat_semantics"] == (
        "repeat_selected_policy_action_inside_one_lightzero_env_step"
    )


def test_stock_source_state_trail_render_mode_passes_to_env_config(monkeypatch, tmp_path):
    _install_fake_lightzero_atari_config(monkeypatch)

    patched = train_mod._build_visual_survival_configs(
        seed=7,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=False,
        max_env_step=128,
        source_max_steps=64,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=8,
        batch_size=16,
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=8,
        save_ckpt_after_iter=100,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=64,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        source_state_trail_render_mode=train_mod.TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    )

    env_cfg = patched["main_config"]["env"]
    assert (
        env_cfg["source_state_trail_render_mode"]
        == train_mod.TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    )
    assert (
        patched["surface"]["source_state_trail_render_mode"]
        == train_mod.TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    )
    assert env_cfg["default_trail_render_mode"] == (
        train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE
    )
    assert env_cfg["supported_trail_render_modes"] == list(
        train_mod.SOURCE_STATE_TRAIL_RENDER_MODE_CHOICES
    )


def test_stock_source_state_trail_render_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="source_state_trail_render_mode"):
        train_mod._validate_source_state_trail_render_mode("mystery")


def test_stock_opponent_death_mode_passes_to_env_config(monkeypatch, tmp_path):
    _install_fake_lightzero_atari_config(monkeypatch)

    patched = train_mod._build_visual_survival_configs(
        seed=7,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=False,
        max_env_step=128,
        source_max_steps=64,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=8,
        batch_size=16,
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=8,
        save_ckpt_after_iter=100,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=64,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_death_mode=train_mod.OPPONENT_DEATH_MODE_IMMORTAL,
    )

    env_cfg = patched["main_config"]["env"]
    assert env_cfg["death_mode"] == "normal"
    assert env_cfg["opponent_death_mode"] == train_mod.OPPONENT_DEATH_MODE_IMMORTAL
    assert env_cfg["opponent_death_mode_diagnostic"] is True
    assert patched["surface"]["opponent_death_mode"] == (
        train_mod.OPPONENT_DEATH_MODE_IMMORTAL
    )


def test_local_stock_launcher_passes_source_state_trail_render_mode(
    capsys, monkeypatch
):
    class FakeCall:
        object_id = "fc-stock"

    class FakeFunction:
        def __init__(self) -> None:
            self.kwargs = []

        def spawn(self, **kwargs):
            self.kwargs.append(kwargs)
            return FakeCall()

    fake_train = FakeFunction()
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_cpu",
        fake_train,
    )

    train_mod.main(
        mode="train",
        compute=train_mod.COMPUTE_CPU,
        run_id="stock-render-mode",
        attempt_id="attempt-render-mode",
        wait_for_train=False,
        background_eval_enabled=False,
        background_gif_enabled=False,
        source_state_trail_render_mode=train_mod.TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    )

    payload = fake_train.kwargs[0]
    printed = json.loads(capsys.readouterr().out)

    assert payload["source_state_trail_render_mode"] == (
        train_mod.TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    )
    assert printed["command"]["source_state_trail_render_mode"] == (
        train_mod.TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    )
    assert "trail_render_mode" not in payload


def test_background_eval_inspection_and_gif_can_be_explicitly_enabled():
    config = train_mod._background_eval_config_from_command(
        {"background_eval_enabled": True, "background_gif_enabled": True}
    )
    custom_config = train_mod._background_eval_config_from_command(
        {
            "background_eval_enabled": True,
            "background_gif_enabled": True,
            "background_gif_frame_size": 512,
            "natural_bonus_spawn": False,
            "opponent_death_mode": train_mod.OPPONENT_DEATH_MODE_IMMORTAL,
        }
    )
    capped_config = train_mod._background_eval_config_from_command(
        {
            "background_eval_enabled": True,
            "background_gif_enabled": True,
            "background_gif_max_steps": 17,
        }
    )
    reward_config = train_mod._background_eval_config_from_command(
        {
            "background_eval_enabled": True,
            "background_gif_enabled": True,
            "reward_variant": train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        }
    )
    reward_override_config = train_mod._background_eval_config_from_command(
        {
            "background_eval_enabled": True,
            "background_gif_enabled": True,
            "reward_variant": train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "eval_reward_variant": train_mod.REWARD_VARIANT_SPARSE_OUTCOME,
        }
    )

    assert config["enabled"] is True
    assert config["natural_bonus_spawn"] is True
    assert config["selfplay_gif"]["enabled"] is True
    assert config["selfplay_gif"]["natural_bonus_spawn"] is True
    assert config["selfplay_gif"]["max_steps"] is None
    assert config["selfplay_gif"]["step_limit_kind"] == "until_environment_done"
    assert config["selfplay_gif"]["collect_temperature"] == 1.0
    assert config["selfplay_gif"]["collect_epsilon"] == 0.25
    assert config["opponent_death_mode"] == train_mod.DEFAULT_OPPONENT_DEATH_MODE
    assert (
        config["selfplay_gif"]["opponent_death_mode"]
        == train_mod.DEFAULT_OPPONENT_DEATH_MODE
    )
    assert (
        train_mod._background_gif_max_steps_arg(config["selfplay_gif"]["max_steps"])
        == 0
    )
    assert config["env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
    assert config["selfplay_gif"]["training_reward_variant"] == train_mod.DEFAULT_REWARD_VARIANT
    assert (
        reward_config["eval_reward_variant"]
        == train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
    )
    assert (
        reward_config["reward_variant"]
        == train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
    )
    assert (
        reward_config["training_reward_variant"]
        == train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
    )
    assert (
        reward_config["model_reward_variant"]
        == train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
    )
    assert reward_override_config["eval_reward_variant"] == train_mod.REWARD_VARIANT_SPARSE_OUTCOME
    assert (
        reward_override_config["training_reward_variant"]
        == train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
    )
    assert custom_config["selfplay_gif"]["requested_frame_size"] == 512
    assert custom_config["selfplay_gif"]["frame_size"] == (
        train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    )
    assert custom_config["selfplay_gif"]["frame_size_policy"] == (
        "checkpoint_selfplay_gif_always_uses_full_source_state_rgb_canvas"
    )
    assert custom_config["natural_bonus_spawn"] is False
    assert custom_config["opponent_death_mode"] == train_mod.OPPONENT_DEATH_MODE_IMMORTAL
    assert (
        custom_config["selfplay_gif"]["opponent_death_mode"]
        == train_mod.OPPONENT_DEATH_MODE_IMMORTAL
    )
    assert custom_config["selfplay_gif"]["natural_bonus_spawn"] is False
    assert capped_config["selfplay_gif"]["max_steps"] == 17
    assert capped_config["selfplay_gif"]["step_limit_kind"] == "physical_step_cap"
    assert (
        train_mod._background_gif_max_steps_arg(
            capped_config["selfplay_gif"]["max_steps"]
        )
        == 17
    )


def test_modal_training_image_copies_curvytron_bonus_sprite_sheet():
    relative_path = Path(
        train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH
    )
    expected_local_path = Path.cwd() / relative_path
    expected_remote_path = train_mod.REMOTE_ROOT / relative_path
    entries = _modal_image_copy_mount_entries(train_mod.image)

    assert train_mod.CURVYTRON_BONUS_SPRITE_SHEET_LOCAL_PATH == expected_local_path
    assert train_mod.CURVYTRON_BONUS_SPRITE_SHEET_REMOTE_PATH == expected_remote_path
    assert expected_local_path.is_file()
    assert any(
        Path(entry.local_file) == expected_local_path
        and str(entry.remote_path) == expected_remote_path.as_posix()
        for entry in entries
    )


def test_two_seat_background_gif_defaults_to_training_collect_knobs():
    background = train_mod._two_seat_background_eval_config(
        payload={
            "run_id": "run-a",
            "attempt_id": "attempt-a",
            "seed": 0,
            "allow_optimizer_step": True,
            "max_ticks": 64,
            "natural_bonus_spawn": True,
            "collect_temperature": 0.2,
            "collect_epsilon": 0.0,
        },
        background_eval_enabled=True,
        background_eval_launch_kind=train_mod.BACKGROUND_EVAL_LAUNCH_POLLER,
        background_eval_compute="cpu",
        background_eval_id_prefix="live_checkpoint",
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=0,
        background_eval_max_steps=64,
        background_eval_step_detail_limit=2,
        background_eval_num_simulations=4,
        background_eval_batch_size=8,
        background_eval_poll_interval_sec=0.01,
        background_eval_poll_stable_polls=0,
        background_eval_poller_max_runtime_sec=1.0,
        background_eval_poller_idle_after_done_sec=0.0,
        background_gif_enabled=True,
        background_gif_seed_offset=10_000,
        background_gif_max_steps=0,
        background_gif_frame_stride=1,
        background_gif_fps=8.0,
        background_gif_scale=4,
        background_gif_frame_size=512,
    )

    assert background["selfplay_gif"]["collect_temperature"] == 0.2
    assert background["selfplay_gif"]["collect_epsilon"] == 0.0
    assert (
        background["selfplay_gif"]["collect_temperature_source"]
        == train_mod.BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_TRAINING
    )
    assert (
        background["selfplay_gif"]["collect_epsilon_source"]
        == train_mod.BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_TRAINING
    )


def test_two_seat_background_gif_collect_knobs_can_be_overridden():
    background = train_mod._two_seat_background_eval_config(
        payload={
            "run_id": "run-a",
            "attempt_id": "attempt-a",
            "seed": 0,
            "allow_optimizer_step": True,
            "max_ticks": 64,
            "natural_bonus_spawn": True,
            "collect_temperature": 0.2,
            "collect_epsilon": 0.0,
        },
        background_eval_enabled=True,
        background_eval_launch_kind=train_mod.BACKGROUND_EVAL_LAUNCH_POLLER,
        background_eval_compute="cpu",
        background_eval_id_prefix="live_checkpoint",
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=0,
        background_eval_max_steps=64,
        background_eval_step_detail_limit=2,
        background_eval_num_simulations=4,
        background_eval_batch_size=8,
        background_eval_poll_interval_sec=0.01,
        background_eval_poll_stable_polls=0,
        background_eval_poller_max_runtime_sec=1.0,
        background_eval_poller_idle_after_done_sec=0.0,
        background_gif_enabled=True,
        background_gif_seed_offset=10_000,
        background_gif_max_steps=0,
        background_gif_frame_stride=1,
        background_gif_fps=8.0,
        background_gif_scale=4,
        background_gif_frame_size=512,
        background_gif_collect_temperature=1.0,
        background_gif_collect_epsilon=0.25,
    )

    assert background["selfplay_gif"]["collect_temperature"] == 1.0
    assert background["selfplay_gif"]["collect_epsilon"] == 0.25
    assert (
        background["selfplay_gif"]["collect_temperature_source"]
        == train_mod.BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_OVERRIDE
    )
    assert (
        background["selfplay_gif"]["collect_epsilon_source"]
        == train_mod.BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_OVERRIDE
    )


def test_gif_browser_run_marker_is_written_under_run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)

    summary = train_mod._write_gif_browser_run_marker(
        run_id="run-marker",
        created_at="2026-05-11T00:00:00Z",
    )
    marker_path = train_mod.runs.volume_path(
        tmp_path,
        train_mod.runs.gif_browser_run_marker_ref(train_mod.TASK_ID, "run-marker"),
    )
    payload = json.loads(marker_path.read_text(encoding="utf-8"))

    assert marker_path.name == train_mod.runs.GIF_BROWSER_RUN_MARKER_FILENAME
    assert summary["ref"].endswith("/show_in_gif_browser.flag")
    assert payload["schema"] == train_mod.runs.GIF_BROWSER_RUN_MARKER_SCHEMA
    assert payload["task_id"] == train_mod.TASK_ID
    assert payload["run_id"] == "run-marker"
    assert payload["created_at"] == "2026-05-11T00:00:00Z"


def test_two_seat_payload_writes_gif_browser_run_marker(tmp_path, monkeypatch):
    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

    def fake_two_seat_train(**kwargs):
        return {
            "ok": True,
            "status": "completed",
            "received_checkpoint_dir": kwargs["checkpoint_dir"] is not None,
        }

    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", FakeVolume())
    monkeypatch.setattr(
        train_mod,
        "run_curvytron_two_seat_lightzero_train_smoke",
        fake_two_seat_train,
    )

    payload = {
        "seed": 0,
        "batch_size": 2,
        "steps": 4,
        "outer_iterations": 1,
        "collect_steps_per_iteration": 1,
        "updates_per_iteration": 1,
        "num_simulations": 1,
        "learner_updates": 1,
        "allow_optimizer_step": True,
        "replay_scope": "accumulated",
        "learner_sample_size": 2,
        "max_replay_rows": 16,
        "record_log_limit": 4,
        "replay_row_log_limit": 4,
        "max_ticks": 16,
        "death_mode": "normal",
        "decision_ms": 300.0,
        "alive_reward": 1.0,
        "dead_reward": 0.0,
        "terminal_outcome_reward_per_step": 1.0,
        "bonus_pickup_reward_per_catch": 0.0,
        "return_target_discount": 1.0,
        "action_selection_mode": "collect",
        "collect_temperature": 1.0,
        "collect_epsilon": 0.25,
        "action_noop_probability": 0.0,
        "action_noop_warmup_iterations": 0,
        "policy_action_repeat_min": 1,
        "policy_action_repeat_max": 1,
        "policy_action_repeat_extra_probability": 0.0,
        "policy_action_repeat_warmup_iterations": 0,
        "observation_noise_std": 0.0,
        "trail_render_mode": train_mod.TRAIL_RENDER_MODE_DEFAULT,
        "checkpoint_every_iterations": 1,
        "save_initial_checkpoint": True,
        "progress_every_iterations": 1,
        "progress_commit_every_iterations": 1,
        "run_id": "run-two-seat-marker",
        "attempt_id": "attempt-two-seat-marker",
    }

    result = train_mod._run_two_seat_selfplay_payload(
        payload,
        compute_label="cpu",
        use_cuda=False,
    )
    marker_path = train_mod.runs.volume_path(
        tmp_path,
        train_mod.runs.gif_browser_run_marker_ref(
            train_mod.TASK_ID,
            "run-two-seat-marker",
        ),
    )
    command_path = train_mod.runs.volume_path(
        tmp_path,
        train_mod.runs.attempt_root_ref(
            train_mod.TASK_ID,
            "run-two-seat-marker",
            "attempt-two-seat-marker",
        )
        / "command.json",
    )
    command = json.loads(command_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert marker_path.name == train_mod.runs.GIF_BROWSER_RUN_MARKER_FILENAME
    assert marker_path.exists()
    assert command["gif_browser_run_marker_ref"].endswith("/show_in_gif_browser.flag")


def test_two_seat_payload_can_skip_gif_browser_run_marker(tmp_path, monkeypatch):
    class FakeVolume:
        def commit(self) -> None:
            pass

    def fake_two_seat_train(**_kwargs):
        return {"ok": True, "status": "completed"}

    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", FakeVolume())
    monkeypatch.setattr(
        train_mod,
        "run_curvytron_two_seat_lightzero_train_smoke",
        fake_two_seat_train,
    )

    payload = {
        "seed": 0,
        "batch_size": 2,
        "steps": 4,
        "outer_iterations": 1,
        "collect_steps_per_iteration": 1,
        "updates_per_iteration": 1,
        "num_simulations": 1,
        "learner_updates": 1,
        "allow_optimizer_step": True,
        "replay_scope": "accumulated",
        "learner_sample_size": 2,
        "max_replay_rows": 16,
        "record_log_limit": 4,
        "replay_row_log_limit": 4,
        "max_ticks": 16,
        "death_mode": "normal",
        "decision_ms": 300.0,
        "alive_reward": 1.0,
        "dead_reward": 0.0,
        "terminal_outcome_reward_per_step": 1.0,
        "bonus_pickup_reward_per_catch": 0.0,
        "return_target_discount": 1.0,
        "action_selection_mode": "collect",
        "collect_temperature": 1.0,
        "collect_epsilon": 0.25,
        "action_noop_probability": 0.0,
        "action_noop_warmup_iterations": 0,
        "policy_action_repeat_min": 1,
        "policy_action_repeat_max": 1,
        "policy_action_repeat_extra_probability": 0.0,
        "policy_action_repeat_warmup_iterations": 0,
        "observation_noise_std": 0.0,
        "trail_render_mode": train_mod.TRAIL_RENDER_MODE_DEFAULT,
        "checkpoint_every_iterations": 1,
        "save_initial_checkpoint": True,
        "progress_every_iterations": 1,
        "progress_commit_every_iterations": 1,
        "run_id": "run-two-seat-no-marker",
        "attempt_id": "attempt-two-seat-no-marker",
        "gif_browser_run_marker_enabled": False,
    }

    result = train_mod._run_two_seat_selfplay_payload(
        payload,
        compute_label="cpu",
        use_cuda=False,
    )
    marker_path = train_mod.runs.volume_path(
        tmp_path,
        train_mod.runs.gif_browser_run_marker_ref(
            train_mod.TASK_ID,
            "run-two-seat-no-marker",
        ),
    )
    command_path = train_mod.runs.volume_path(
        tmp_path,
        train_mod.runs.attempt_root_ref(
            train_mod.TASK_ID,
            "run-two-seat-no-marker",
            "attempt-two-seat-no-marker",
        )
        / "command.json",
    )
    command = json.loads(command_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert not marker_path.exists()
    assert command["gif_browser_run_marker_enabled"] is False
    assert command["gif_browser_run_marker_ref"] is None


def test_two_seat_payload_forwards_frozen_opponent_knobs(tmp_path, monkeypatch):
    class FakeVolume:
        def commit(self) -> None:
            pass

    received = {}

    def fake_two_seat_train(**kwargs):
        received.update(kwargs)
        return {"ok": True, "status": "completed"}

    checkpoint = tmp_path / "frozen.pth.tar"
    checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", FakeVolume())
    monkeypatch.setattr(
        train_mod,
        "run_curvytron_two_seat_lightzero_train_smoke",
        fake_two_seat_train,
    )

    payload = {
        "seed": 0,
        "batch_size": 2,
        "steps": 4,
        "outer_iterations": 1,
        "collect_steps_per_iteration": 1,
        "updates_per_iteration": 1,
        "num_simulations": 1,
        "learner_updates": 1,
        "allow_optimizer_step": True,
        "replay_scope": "accumulated",
        "learner_sample_size": 2,
        "max_replay_rows": 16,
        "record_log_limit": 4,
        "replay_row_log_limit": 4,
        "max_ticks": 16,
        "death_mode": "normal",
        "decision_ms": 300.0,
        "alive_reward": 1.0,
        "dead_reward": 0.0,
        "terminal_outcome_reward_per_step": 1.0,
        "bonus_pickup_reward_per_catch": 0.0,
        "return_target_discount": 1.0,
        "action_selection_mode": "collect",
        "collect_temperature": 1.0,
        "collect_epsilon": 0.25,
        "action_noop_probability": 0.0,
        "action_noop_warmup_iterations": 0,
        "policy_action_repeat_min": 1,
        "policy_action_repeat_max": 1,
        "policy_action_repeat_extra_probability": 0.0,
        "policy_action_repeat_warmup_iterations": 0,
        "observation_noise_std": 0.0,
        "trail_render_mode": train_mod.TRAIL_RENDER_MODE_DEFAULT,
        "learning_rate": 0.0001,
        "frozen_opponent_probability": 0.25,
        "frozen_opponent_checkpoint_path": str(checkpoint),
        "frozen_opponent_checkpoint_ref": "training/example/iteration_50.pth.tar",
        "frozen_opponent_snapshot_ref": "snapshot-50",
        "frozen_opponent_checkpoint_state_key": "model",
        "frozen_opponent_player_id": 1,
        "frozen_opponent_num_simulations": 4,
        "frozen_opponent_use_cuda": False,
        "checkpoint_every_iterations": 1,
        "save_initial_checkpoint": True,
        "progress_every_iterations": 1,
        "progress_commit_every_iterations": 1,
        "run_id": "run-two-seat-frozen",
        "attempt_id": "attempt-two-seat-frozen",
        "gif_browser_run_marker_enabled": False,
    }

    result = train_mod._run_two_seat_selfplay_payload(
        payload,
        compute_label="cpu",
        use_cuda=False,
    )

    assert result["ok"] is True
    assert received["frozen_opponent_probability"] == 0.25
    assert received["frozen_opponent_checkpoint_path"] == str(checkpoint)
    assert (
        received["frozen_opponent_checkpoint_ref"]
        == "training/example/iteration_50.pth.tar"
    )
    assert received["frozen_opponent_snapshot_ref"] == "snapshot-50"
    assert received["frozen_opponent_checkpoint_state_key"] == "model"
    assert received["frozen_opponent_player_id"] == 1
    assert received["frozen_opponent_num_simulations"] == 4
    assert received["frozen_opponent_use_cuda"] is False


def test_source_state_fixed_opponent_surface_identity_is_explicit_control_lane():

    spec = train_mod._env_variant_spec(train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT)
    wrapper_info = _source_state_fixed_opponent_wrapper_info()

    assert spec["env_type"] == wrapper_info["lightzero_env_type"]
    assert spec["env_id"] == wrapper_info["env_id"]
    assert spec["observation_shape"] == wrapper_info["model_observation_shape"]
    assert spec["observation_shape"] == list(train_mod.STACKED_SOURCE_STATE_GRAY64_SHAPE)
    assert spec["observation_schema_id"] == wrapper_info["observation_schema_id"]
    assert spec["single_frame_schema_id"] == wrapper_info["single_frame_schema_id"]
    assert spec["raw_observation_schema_id"] == wrapper_info["raw_observation_schema_id"]
    assert spec["raw_observation_schema_id"] == train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
    assert spec["raw_frame_shape"] == [
        train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        3,
    ]
    assert spec["grayscale_frame_shape"] == [1, 64, 64]
    assert "bonus64" not in spec["observation_schema_id"]
    assert "bonus64" not in spec["frame_stack_proof"]
    assert "raw_canvas_to_downsampled_gray64" in spec["frame_stack_proof"]
    assert spec["debug_fidelity_only"] == wrapper_info["debug_fidelity_only"]
    assert spec["source_fidelity_claim"] == wrapper_info["source_fidelity_claim"]
    assert spec["single_product_runtime_path"] is True
    assert spec["legacy_debug_variant"] is False
    assert spec["underlying_env_class"] == wrapper_info["underlying_env_class"]
    assert spec["runtime_env_impl_id"] == wrapper_info["runtime_env_impl_id"]
    assert spec["runtime_topology"] == wrapper_info["runtime_topology"]
    assert spec["two_seat_self_play"] == wrapper_info["two_seat_self_play"]
    assert spec["two_seat_self_play_status"] == wrapper_info["two_seat_self_play_status"]
    assert spec["fixed_opponent_is_two_seat_self_play"] == (
        wrapper_info["fixed_opponent_is_two_seat_self_play"]
    )
    assert spec["browser_pixel_fidelity"] == wrapper_info["browser_pixel_fidelity"]
    assert spec["uses_ale"] == wrapper_info["uses_ale"]
    assert spec["visual_surface"] == wrapper_info["visual_surface"]
    assert spec["visual_truth_level"] == wrapper_info["visual_truth_level"]
    assert spec["visual_source_state_backed"] == wrapper_info["visual_source_state_backed"]
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


def test_source_state_readiness_gate_allows_frozen_checkpoint_opponent_metadata():
    command = _source_state_training_command()
    command["opponent_policy_kind"] = (
        train_mod.OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
    )
    command["opponent_training_relation"] = (
        train_mod.OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT
    )

    gate = train_mod._source_state_fixed_opponent_training_readiness_gate(
        command=command,
        surface=dict(command),
    )

    assert gate["ok"] is True
    assert gate["expected"]["opponent_policy_kind"] == (
        train_mod.OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
    )
    assert gate["expected"]["opponent_training_relation"] == (
        train_mod.OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT
    )


def test_source_state_readiness_gate_allows_episode_opponent_mixture():
    command = _source_state_training_command()
    mixture = train_mod.parse_opponent_mixture_spec(
        {
            "entries": [
                {
                    "name": "blank",
                    "weight": 1,
                    "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                    "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                }
            ]
        }
    )
    command["opponent_mixture_enabled"] = True
    command["opponent_mixture"] = mixture
    command["opponent_training_relation"] = (
        train_mod.OPPONENT_TRAINING_RELATION_WEIGHTED_EPISODE_MIXTURE
    )

    gate = train_mod._source_state_fixed_opponent_training_readiness_gate(
        command=command,
        surface=dict(command),
    )

    assert gate["ok"] is True
    assert gate["expected"]["opponent_training_relation"] == (
        train_mod.OPPONENT_TRAINING_RELATION_WEIGHTED_EPISODE_MIXTURE
    )


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
        "profile_env_timing_sec": {
            "opponent_action_sec": 0.1,
            "observation_sec": 0.2,
        },
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
    assert summary["profile_env_timing_sec"] == {
        "scope": "all_telemetry_rows",
        "sampled_sum": {"observation_sec": 0.2, "opponent_action_sec": 0.1},
        "sampled_count": {"observation_sec": 1, "opponent_action_sec": 1},
        "sampled_mean": {"observation_sec": 0.2, "opponent_action_sec": 0.1},
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


def test_eval_row_carries_winner_fields_and_derives_outcome():
    row = eval_mod._row_from_result(
        {
            "index": 0,
            "checkpoint_label": "iteration_7",
            "seed": 3,
            "checkpoint_ref": "training/run/iteration_7.pth.tar",
        },
        {
            "ok": True,
            "status": {
                "steps_survived": 33,
                "cap": 64,
                "strict_policy_model_load_ok": True,
            },
            "episode": {
                "terminal_reason": "survivor_win",
                "winner_ids": ["player_1"],
                "loser_ids": ["player_0"],
                "death_player": 0,
                "death_cause_name": "wall",
                "total_reward": -1.0,
                "action_histogram": {"0": 33},
            },
            "artifact": {"ref": "eval/iteration_7_seed3.json"},
            "remote_elapsed_sec": 1.25,
        },
    )

    assert row["winner_ids"] == ["player_1"]
    assert row["loser_ids"] == ["player_0"]
    assert row["outcome"] == "loss"
    assert row["training_reward"] == -1.0


def test_eval_outcome_histogram_rolls_up_win_loss_draw_cap_and_errors():
    rows = [
        {"checkpoint_label": "iteration_1", "ok": True, "winner_ids": ["player_0"]},
        {"checkpoint_label": "iteration_1", "ok": True, "death_player": 0},
        {
            "checkpoint_label": "iteration_1",
            "ok": True,
            "terminal_reason": "all_dead_draw",
        },
        {
            "checkpoint_label": "iteration_1",
            "ok": True,
            "terminal_reason": "cap",
            "steps_survived": 64,
            "cap": 64,
        },
        {"checkpoint_label": "iteration_1", "ok": False},
        {"checkpoint_label": "iteration_2", "ok": True, "death_player": 1},
    ]
    for row in rows:
        row["outcome"] = eval_mod._outcome_from_row(row)

    assert eval_mod._outcome_histogram(rows) == {
        "cap": 1,
        "draw": 1,
        "error": 1,
        "loss": 1,
        "win": 2,
    }
    assert eval_mod._survival_aggregate_table(rows) == [
        {
            "checkpoint": "iteration_1",
            "seeds": 5,
            "mean_steps": 64.0,
            "median_steps": 64.0,
            "min_steps": 64.0,
            "max_steps": 64.0,
            "ok_count": 4,
            "capped_count": 1,
            "failure_count": 1,
            "outcome_histogram": {
                "cap": 1,
                "draw": 1,
                "error": 1,
                "loss": 1,
                "win": 1,
            },
            "mean_training_reward": None,
            "mean_survival_reward": None,
            "mean_sparse_outcome_reward": None,
            "mean_bonus_pickup_count": None,
            "mean_bonus_reward": None,
            "mean_reward_components": {},
            "mean_elapsed_sec": None,
        },
        {
            "checkpoint": "iteration_2",
            "seeds": 1,
            "mean_steps": None,
            "median_steps": None,
            "min_steps": None,
            "max_steps": None,
            "ok_count": 1,
            "capped_count": 0,
            "failure_count": 0,
            "outcome_histogram": {"win": 1},
            "mean_training_reward": None,
            "mean_survival_reward": None,
            "mean_sparse_outcome_reward": None,
            "mean_bonus_pickup_count": None,
            "mean_bonus_reward": None,
            "mean_reward_components": {},
            "mean_elapsed_sec": None,
        },
    ]


def test_copy_source_state_raw_frame_prefers_rgb_raw_observation_and_returns_copy():
    import numpy as np

    raw = (np.arange(32 * 48 * 3) % 251).astype(np.uint8).reshape(32, 48, 3)

    class Env:
        raw_calls = 0
        render_calls = 0

        def raw_observation(self):
            self.raw_calls += 1
            return raw

        def render(self, mode):
            self.render_calls += 1
            raise AssertionError(f"render should not be used for {mode}")

    env = Env()

    frame = train_mod._copy_source_state_raw_frame(env)

    assert env.raw_calls == 1
    assert env.render_calls == 0
    assert frame.shape == (32, 48, 3)
    assert frame.dtype == np.uint8
    assert np.array_equal(frame, raw.astype(np.uint8))
    assert not np.shares_memory(frame, raw)
    frame[0, 0, 0] = 123
    assert raw[0, 0, 0] == 0


def test_checkpoint_gif_rgb_frame_helper_prefers_rgb_raw_observation():
    raw_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    raw_rgb[:, :] = np.asarray([3, 17, 241], dtype=np.uint8)

    class Env:
        raw_calls = 0
        human_calls = 0

        def raw_observation(self):
            self.raw_calls += 1
            return raw_rgb

        def human_rgb_observation(self, *, frame_size):
            self.human_calls += 1
            raise AssertionError("RGB raw_observation should win")

    env = Env()

    frame, source = train_mod._copy_source_state_human_rgb_frame_with_source(
        env,
        frame_size=128,
    )

    assert env.raw_calls == 1
    assert env.human_calls == 0
    assert source["source"] == "raw_observation"
    assert source["input_shape"] == [128, 128, 3]
    assert source["output_shape"] == [128, 128, 3]
    assert source["resize_method"] == "none"
    assert source["resized_nearest"] is False
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8
    assert np.array_equal(frame[0, 0], np.asarray([3, 17, 241], dtype=np.uint8))
    assert not np.shares_memory(frame, raw_rgb)


def test_checkpoint_gif_frame_helper_skips_legacy_gray_raw_for_rgb_canvas_like():
    old_gray_raw = np.full((1, 64, 64), 199, dtype=np.uint8)
    canvas_rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    canvas_rgb[:, :] = np.asarray([0, 255, 0], dtype=np.uint8)
    canvas_rgb[10, 20] = np.asarray([255, 0, 0], dtype=np.uint8)

    class Env:
        raw_calls = 0
        human_calls = 0
        render_calls = 0

        def raw_observation(self):
            self.raw_calls += 1
            return old_gray_raw

        def human_rgb_observation(self, *, frame_size):
            self.human_calls += 1
            assert frame_size == 96
            return canvas_rgb

        def render(self, mode):
            self.render_calls += 1
            raise AssertionError(f"human_rgb_observation should be enough for {mode}")

    env = Env()

    frame, source = train_mod._copy_source_state_human_rgb_frame_with_source(
        env,
        frame_size=96,
    )

    assert env.raw_calls == 1
    assert env.human_calls == 1
    assert env.render_calls == 0
    assert source["source"] == "human_rgb_observation"
    assert source["skipped_prior_sources"] == [
        {
            "source": "raw_observation",
            "status": "non_rgb",
            "shape": [1, 64, 64],
            "dtype": "uint8",
        }
    ]
    assert frame.shape == (96, 96, 3)
    assert np.array_equal(frame[10, 20], np.asarray([255, 0, 0], dtype=np.uint8))
    assert not np.all(frame[:, :, 0] == frame[:, :, 1])


def test_checkpoint_gif_turn_commit_capture_uses_rgb_raw_after_trail_points():
    from curvyzero.env.vector_visual_observation import (
        SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB,
    )
    from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
        CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv,
    )

    env = CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv(
        {
            "source_max_steps": 64,
            "max_ticks": 64,
            "disable_death_for_profile": True,
        }
    )
    try:
        env.reset(seed=0)
        for _step in range(12):
            env.step(1)
            timestep = env.step(1)
            assert timestep.info["physical_env_advanced"] is True

        frame, source = train_mod._copy_source_state_human_rgb_frame_with_source(
            env,
            frame_size=64,
        )
        raw_frame = env.raw_observation()
        state = env._env.state
        body_cursor = int(state["body_write_cursor"][0])
        map_size = float(state["map_size"][0])
        player_colors = np.asarray(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB, dtype=np.uint8)

        visible_trail_pixels = []
        for slot in range(body_cursor):
            if not bool(state["body_active"][0, slot]):
                continue
            position = state["body_pos"][0, slot]
            owner = int(state["body_owner"][0, slot])
            head_distance = np.linalg.norm(state["pos"][0] - position, axis=1).min()
            if head_distance <= 2.5:
                continue
            raw_max = train_mod.TURN_COMMIT_SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE[0] - 1
            px = int(np.clip(np.rint((float(position[0]) / map_size) * raw_max), 0, raw_max))
            py = int(np.clip(np.rint((float(position[1]) / map_size) * raw_max), 0, raw_max))
            if np.array_equal(raw_frame[py, px], player_colors[owner]):
                visible_trail_pixels.append((slot, owner))

        assert source["source"] == "raw_observation"
        assert source["skipped_prior_sources"] == []
        assert source["input_shape"] == list(
            train_mod.TURN_COMMIT_SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE
        )
        assert source["output_shape"] == [64, 64, 3]
        assert source["resize_method"] == "area_average"
        assert source["resized_nearest"] is False
        assert body_cursor > 2
        assert visible_trail_pixels
    finally:
        env.close()


def test_live_checkpoint_trigger_spawns_eval_and_selfplay_gif_without_volume_commit(
    tmp_path, monkeypatch
):
    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

    class FakeFunction:
        def __init__(self, object_id: str) -> None:
            self.object_id = object_id
            self.calls = []

        def spawn(self, **kwargs):
            self.calls.append(kwargs)
            object_id = self.object_id

            class Call:
                pass

            call = Call()
            call.object_id = object_id
            return call

    class FakeGifFunction:
        def __init__(self) -> None:
            self.calls = []

        def spawn(self, **kwargs):
            self.calls.append(kwargs)

            class Call:
                object_id = "fc-live-gif-test"

            return Call()

    fake_volume = FakeVolume()
    fake_eval_function = FakeFunction("fc-live-eval-test")
    fake_gif_function = FakeGifFunction()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect",
        fake_eval_function,
    )
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_selfplay_gif",
        fake_gif_function,
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
        "reward_variant": train_mod.REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
        "natural_bonus_spawn": False,
        "selfplay_gif": {
            "enabled": True,
            "seed": 10_007,
            "max_steps": 16,
            "source_max_steps": 32,
            "num_simulations": 4,
            "batch_size": 8,
            "frame_stride": 2,
            "fps": 12.0,
            "scale": 3,
            "frame_size": 320,
            "training_env_variant": train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
            "training_reward_variant": train_mod.REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
            "natural_bonus_spawn": False,
        },
    }

    result = train_mod._schedule_live_checkpoint_background_eval(
        publish=publish,
        attempt_train_root=attempt_train_root,
        config=config,
    )

    assert result["scheduled"] is True
    assert len(fake_eval_function.calls) == 1
    assert len(fake_gif_function.calls) == 1
    eval_call = fake_eval_function.calls[0]
    gif_call = fake_gif_function.calls[0]
    assert eval_call["checkpoint_ref"].endswith("iteration_1.pth.tar")
    assert gif_call["checkpoint_ref"].endswith("iteration_1.pth.tar")
    assert eval_call["env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
    assert (
        eval_call["reward_variant"]
        == train_mod.REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME
    )
    assert eval_call["eval_seed_count"] == 2
    assert eval_call["natural_bonus_spawn"] is False
    assert gif_call["seed"] != 10_007
    assert gif_call["max_steps"] == 16
    assert gif_call["frame_stride"] == 2
    assert gif_call["fps"] == 12.0
    assert gif_call["scale"] == 3
    assert gif_call["frame_size"] == train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    assert gif_call["training_env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
    assert (
        gif_call["training_reward_variant"]
        == train_mod.REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME
    )
    assert gif_call["natural_bonus_spawn"] is False
    assert fake_volume.commit_count == 0
    assert not (attempt_train_root / "background_eval_requests").exists()
    request = result["requests"][0]
    assert request["status"] == "spawned"
    assert request["function_call_id"] == "fc-live-eval-test"
    assert request["eval_inspection_scheduled"] is True
    assert request["selfplay_gif"]["scheduled"] is True
    assert request["selfplay_gif"]["function_call_id"] == "fc-live-gif-test"
    assert request["selfplay_gif"]["config"]["requested_frame_size"] == 320
    assert request["selfplay_gif"]["config"]["frame_size"] == (
        train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    )
    assert request["selfplay_gif"]["config"]["base_seed"] == 10_007
    assert request["selfplay_gif"]["config"]["effective_seed"] == gif_call["seed"]
    assert request["selfplay_gif"]["config"]["checkpoint_seed_mixing_enabled"] is True
    assert (
        request["selfplay_gif"]["config"]["training_reward_variant"]
        == train_mod.REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME
    )
    assert request["selfplay_gif"]["config"]["natural_bonus_spawn"] is False
    assert request["config"]["env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
    assert request["config"]["natural_bonus_spawn"] is False


def test_save_raw_frames_gif_preserves_rgb_dimensions_without_gray64_scaling(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    frames = np.zeros((2, 96, 128, 3), dtype=np.uint8)
    frames[0, :, :] = np.asarray([255, 0, 0], dtype=np.uint8)
    frames[1, :, :] = np.asarray([0, 255, 0], dtype=np.uint8)
    gif_path = tmp_path / "rgb.gif"

    artifact = train_mod._save_raw_frames_gif(
        frames=frames,
        gif_path=gif_path,
        fps=12.0,
        scale=4,
    )

    assert artifact["color_mode"] == "RGB"
    assert artifact["scale"] == 1
    assert artifact["pixel_size"] == [128, 96]
    with Image.open(gif_path) as image:
        assert image.size == (128, 96)
        assert image.n_frames == 2
        first_frame = image.convert("RGB")
        assert first_frame.getpixel((0, 0)) == (255, 0, 0)


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
        "selfplay_gif": {"enabled": False},
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


def test_hook_trigger_retries_after_spawn_failure_and_skips_mutable_best_checkpoint(
    tmp_path, monkeypatch
):
    class FlakyFunction:
        def __init__(self) -> None:
            self.calls = []
            self.fail_next = True

        def spawn(self, **kwargs):
            self.calls.append(kwargs)
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("transient modal spawn failure")

            class Call:
                object_id = "fc-retry-ok"

            return Call()

    fake_function = FlakyFunction()
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
        / "run-retry"
        / "attempts"
        / "attempt-retry"
        / "train"
        / "lightzero_exp"
    )
    ckpt_dir = exp_name / "ckpt"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "iteration_8.pth.tar").write_bytes(b"iteration checkpoint")
    (ckpt_dir / "ckpt_best.pth.tar").write_bytes(b"mutable best checkpoint")

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
        "selfplay_gif": {"enabled": False},
    }
    seen: set[str] = set()

    failed = train_mod._spawn_checkpoint_eval_triggers(
        run_id="run-retry",
        attempt_id="attempt-retry",
        exp_name=exp_name,
        config=config,
        seen_checkpoint_refs=seen,
    )
    retried = train_mod._spawn_checkpoint_eval_triggers(
        run_id="run-retry",
        attempt_id="attempt-retry",
        exp_name=exp_name,
        config=config,
        seen_checkpoint_refs=seen,
    )

    assert len(fake_function.calls) == 2
    assert all(call["checkpoint_ref"].endswith("iteration_8.pth.tar") for call in fake_function.calls)
    assert not any("ckpt_best" in call["checkpoint_ref"] for call in fake_function.calls)
    assert failed[0]["status"] == "spawn_failed"
    assert retried[0]["status"] == "spawned"
    assert len(seen) == 1


def test_selfplay_gif_spawn_is_guarded_for_joint_action_checkpoints():
    request, call = train_mod._spawn_one_checkpoint_background_gif(
        publish={"run_id": "run", "attempt_id": "attempt"},
        checkpoint_ref="training/lightzero-curvytron-visual-survival/run/checkpoints/lightzero/iteration_1.pth.tar",
        checkpoint_label="iteration_1",
        eval_id="live_checkpoint_iteration_1",
        config={
            "selfplay_gif": {
                "enabled": True,
                "training_env_variant": train_mod.ENV_VARIANT_SOURCE_STATE_JOINT_ACTION,
            }
        },
    )

    assert call is None
    assert request["scheduled"] is False
    assert request["reason"] == "background_gif_unsupported_for_source_state_joint_action"


def test_two_seat_rejects_mutable_frozen_opponent_checkpoint_ref():
    with pytest.raises(ValueError, match="immutable"):
        train_mod.main(
            mode=train_mod.TWO_SEAT_SELFPLAY_MODE,
            compute=train_mod.COMPUTE_CPU,
            run_id="mutable-frozen-ref",
            attempt_id="mutable-frozen-ref",
            two_seat_frozen_opponent_probability=0.25,
            two_seat_frozen_opponent_checkpoint_ref=(
                "training/lightzero-curvytron-visual-survival/example/"
                "checkpoints/lightzero/latest.pth.tar"
            ),
        )


def test_checkpoint_eval_poller_completes_eval_inspection_and_selfplay_gif_jobs(
    tmp_path, monkeypatch
):
    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0
            self.reload_count = 0

        def commit(self) -> None:
            self.commit_count += 1

        def reload(self) -> None:
            self.reload_count += 1

    class FakeCall:
        def __init__(self, object_id: str, result: dict) -> None:
            self.object_id = object_id
            self.result = result

        def get(self):
            return self.result

    class FakeFunction:
        def __init__(self, object_id: str, result: dict) -> None:
            self.object_id = object_id
            self.result = result
            self.calls = []

        def spawn(self, **kwargs):
            self.calls.append(kwargs)
            return FakeCall(self.object_id, self.result)

    fake_volume = FakeVolume()
    fake_eval_function = FakeFunction(
        "fc-poller-eval",
        {
            "ok": True,
            "inspection_report_ref": "eval/live_checkpoint_iteration_1/inspection/report.json",
            "inspection_report_markdown_ref": "eval/live_checkpoint_iteration_1/inspection/report.md",
        },
    )
    fake_gif_function = FakeFunction(
        "fc-poller-gif",
        {
            "ok": True,
            "gif_ref": "eval/live_checkpoint_iteration_1/selfplay/raw.gif",
            "raw_frames_ref": "eval/live_checkpoint_iteration_1/selfplay/raw_frames.npz",
            "frame_count": 5,
        },
    )
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect",
        fake_eval_function,
    )
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_selfplay_gif",
        fake_gif_function,
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
        reward_variant=train_mod.REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        natural_bonus_spawn=False,
        background_eval_enabled=True,
        background_eval_compute="cpu",
        background_eval_id_prefix="live_checkpoint",
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=0,
        background_eval_max_steps=32,
        background_eval_step_detail_limit=2,
        background_eval_num_simulations=4,
        background_eval_batch_size=8,
        background_gif_enabled=True,
        background_gif_seed_offset=10_000,
        background_gif_max_steps=24,
        background_gif_frame_stride=2,
        background_gif_fps=12.0,
        background_gif_scale=3,
        background_gif_frame_size=320,
        background_gif_collect_temperature=0.75,
        background_gif_collect_epsilon=0.125,
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
    assert result["completed_count"] == 2
    assert result["eval_completed_count"] == 1
    assert result["gif_scheduled_count"] == 1
    assert result["gif_completed_count"] == 1
    completed_by_kind = {item["kind"]: item for item in result["completed"]}
    assert completed_by_kind["eval_inspection"]["result"]["inspection_report_ref"].endswith(
        "inspection/report.json"
    )
    assert completed_by_kind["selfplay_gif"]["result"]["frame_count"] == 5
    assert len(fake_eval_function.calls) == 1
    assert len(fake_gif_function.calls) == 1
    scheduled = result["scheduled"][0]
    eval_call = fake_eval_function.calls[0]
    gif_call = fake_gif_function.calls[0]
    assert eval_call["checkpoint_ref"].endswith(
        "training/lightzero-curvytron-visual-survival/run-c/attempts/attempt-c/train/lightzero_exp/ckpt/iteration_1.pth.tar"
    )
    assert gif_call["checkpoint_ref"] == eval_call["checkpoint_ref"]
    assert gif_call["frame_size"] == train_mod.SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    assert eval_call["eval_id"] == "live_checkpoint_iteration_1"
    assert (
        eval_call["reward_variant"]
        == train_mod.REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME
    )
    assert gif_call["eval_id"] == "live_checkpoint_iteration_1"
    assert gif_call["seed"] != 10_003
    assert scheduled["selfplay_gif"]["config"]["base_seed"] == 10_003
    assert scheduled["selfplay_gif"]["config"]["effective_seed"] == gif_call["seed"]
    assert gif_call["max_steps"] == 24
    assert gif_call["frame_stride"] == 2
    assert gif_call["fps"] == 12.0
    assert gif_call["scale"] == 3
    assert gif_call["collect_temperature"] == 0.75
    assert gif_call["collect_epsilon"] == 0.125
    assert gif_call["natural_bonus_spawn"] is False
    assert (
        gif_call["training_reward_variant"]
        == train_mod.REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME
    )
    assert fake_volume.commit_count == 0
    status_path = train_mod._checkpoint_eval_poller_status_path(
        run_id="run-c",
        attempt_id="attempt-c",
    )
    assert status_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["gif_scheduled_count"] == 1
    assert status["gif_completed_count"] == 1


def test_local_launcher_passes_gif_config_to_poller_and_prints_enabled(
    capsys, monkeypatch
):
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
        background_gif_enabled=True,
        background_gif_seed_offset=1_234,
        background_gif_max_steps=48,
        background_gif_frame_stride=3,
        background_gif_fps=12.5,
        background_gif_scale=2,
        background_gif_collect_temperature=0.5,
        background_gif_collect_epsilon=0.125,
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
    assert fake_poller.calls[0]["background_gif_enabled"] is True
    assert fake_poller.calls[0]["background_gif_seed_offset"] == 1_234
    assert fake_poller.calls[0]["background_gif_max_steps"] == 48
    assert fake_poller.calls[0]["background_gif_frame_stride"] == 3
    assert fake_poller.calls[0]["background_gif_fps"] == 12.5
    assert fake_poller.calls[0]["background_gif_scale"] == 2
    assert fake_poller.calls[0]["background_gif_collect_temperature"] == 0.5
    assert fake_poller.calls[0]["background_gif_collect_epsilon"] == 0.125


def test_local_two_seat_launcher_passes_trail_render_mode(capsys, monkeypatch):
    class FakeCall:
        object_id = "fc-two-seat"

    class FakeFunction:
        def __init__(self) -> None:
            self.payloads = []

        def spawn(self, payload):
            self.payloads.append(payload)
            return FakeCall()

    fake_train = FakeFunction()
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_two_seat_selfplay_cpu",
        fake_train,
    )

    train_mod.main(
        mode=train_mod.TWO_SEAT_SELFPLAY_MODE,
        compute=train_mod.COMPUTE_CPU,
        run_id="two-seat-render-mode",
        attempt_id="attempt-render-mode",
        wait_for_train=False,
        background_eval_enabled=False,
        background_gif_enabled=False,
        two_seat_death_mode="profile_no_death",
        two_seat_trail_render_mode=train_mod.TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    )

    payload = fake_train.payloads[0]
    printed = json.loads(capsys.readouterr().out)

    assert payload["trail_render_mode"] == train_mod.TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    assert payload["death_mode"] == "profile_no_death"
    assert printed["command"]["trail_render_mode"] == (
        train_mod.TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    )
    assert printed["command"]["death_mode"] == "profile_no_death"


def test_local_two_seat_launcher_defaults_gif_browser_marker_enabled(
    capsys, monkeypatch
):
    class FakeCall:
        object_id = "fc-two-seat"

    class FakeFunction:
        def __init__(self) -> None:
            self.payloads = []

        def spawn(self, payload):
            self.payloads.append(payload)
            return FakeCall()

    fake_train = FakeFunction()
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_two_seat_selfplay_cpu",
        fake_train,
    )

    train_mod.main(
        mode=train_mod.TWO_SEAT_SELFPLAY_MODE,
        compute=train_mod.COMPUTE_CPU,
        run_id="two-seat-default-gif-marker",
        attempt_id="attempt-default-gif-marker",
        wait_for_train=False,
        background_eval_enabled=False,
        background_gif_collect_temperature=0.75,
        background_gif_collect_epsilon=0.125,
    )

    payload = fake_train.payloads[0]
    printed = json.loads(capsys.readouterr().out)

    assert payload["gif_browser_run_marker_enabled"] is True
    assert payload["natural_bonus_spawn"] is True
    assert printed["command"]["gif_browser_run_marker_enabled"] is True
    assert printed["background_eval"]["selfplay_gif"]["enabled"] is True
    assert printed["background_eval"]["selfplay_gif"]["natural_bonus_spawn"] is True
    assert printed["background_eval"]["selfplay_gif"]["collect_temperature"] == 0.75
    assert printed["background_eval"]["selfplay_gif"]["collect_epsilon"] == 0.125


def test_local_two_seat_launcher_rejects_unknown_trail_render_mode(monkeypatch):
    class FakeFunction:
        def spawn(self, payload):
            raise AssertionError("train should not launch")

    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_two_seat_selfplay_cpu",
        FakeFunction(),
    )

    with pytest.raises(ValueError, match="trail render mode"):
        train_mod.main(
            mode=train_mod.TWO_SEAT_SELFPLAY_MODE,
            compute=train_mod.COMPUTE_CPU,
            run_id="two-seat-render-mode-bad",
            attempt_id="attempt-render-mode-bad",
            wait_for_train=False,
            background_eval_enabled=False,
            background_gif_enabled=False,
            two_seat_trail_render_mode="mystery",
        )
