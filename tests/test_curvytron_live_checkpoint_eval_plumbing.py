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
from curvyzero.training.opponent_leaderboard import OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID
from curvyzero.training.opponent_registry import OPPONENT_ASSIGNMENT_SCHEMA_ID
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    CurvyZeroSourceStateVisualSurvivalLightZeroEnv,
    LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE,
)
from curvyzero.env.vector_multiplayer_env import SOURCE_PHYSICS_STEP_MS


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


def test_checkpoint_progress_writer_can_commit_after_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    exp_name = tmp_path / "training" / train_mod.TASK_ID / "run-live" / "attempts" / "attempt-live" / "train" / "lightzero_exp"
    attempt_train_root = exp_name.parent
    commit_labels = []

    class FakeBaseLearner:
        train_iter = 3

        def save_checkpoint(self):
            checkpoint_path = exp_name / "ckpt" / "iteration_3.pth.tar"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"fake-checkpoint")
            return {"ok": True}

    def fake_train_muzero():
        return None

    monkeypatch.setitem(fake_train_muzero.__globals__, "BaseLearner", FakeBaseLearner)
    monkeypatch.setattr(
        train_mod,
        "_commit_runs_volume_with_backoff",
        lambda *, label: commit_labels.append(label),
    )
    restore = train_mod._install_checkpoint_progress_writer(
        train_muzero=fake_train_muzero,
        run_id="run-live",
        attempt_id="attempt-live",
        exp_name=exp_name,
        attempt_train_root=attempt_train_root,
        started_monotonic=0.0,
        commit_on_checkpoint=True,
    )

    try:
        assert restore is not None
        assert FakeBaseLearner().save_checkpoint() == {"ok": True}
    finally:
        if restore is not None:
            restore()

    assert commit_labels == ["checkpoint_progress_commit"]


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


def test_stock_train_mode_calls_lightzero_train_muzero_entrypoint(monkeypatch, tmp_path):
    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

    fake_volume = FakeVolume()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)
    monkeypatch.setattr(train_mod, "_version_or_missing", lambda *_args: train_mod.LIGHTZERO_VERSION)
    monkeypatch.setattr(train_mod.os, "chdir", lambda _path: None)

    train_calls = []

    def fake_train_muzero(configs, *, seed, max_train_iter, max_env_step):
        train_calls.append(
            {
                "configs": configs,
                "seed": seed,
                "max_train_iter": max_train_iter,
                "max_env_step": max_env_step,
            }
        )
        return {"trained": True}

    fake_entry_module = types.ModuleType("lzero.entry")
    fake_entry_module.train_muzero = fake_train_muzero
    monkeypatch.setitem(sys.modules, "lzero.entry", fake_entry_module)

    def fake_build_visual_survival_configs(**kwargs):
        return {
            "main_config": {"env": {"type": "fake-env"}, "policy": {"cuda": False}},
            "create_config": {"env": {"type": "fake-env"}},
            "patches": [{"path": "fake", "new": True}],
            "surface": {
                "env_variant": kwargs["env_variant"],
                "env_type": "fake-env",
                "called_from": "stock-train-entrypoint-test",
            },
        }

    monkeypatch.setattr(
        train_mod,
        "_build_visual_survival_configs",
        fake_build_visual_survival_configs,
    )
    monkeypatch.setattr(train_mod, "_compile_config_summary", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(train_mod, "_validate_visual_survival_surface", lambda **_kwargs: [])
    monkeypatch.setattr(train_mod, "_install_lightzero_full_resume_state_hooks", lambda **_kwargs: None)
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

    result = train_mod._run_visual_survival_train(
        mode="train",
        compute="cpu",
        seed=123,
        run_id="stock-entrypoint-local-test",
        attempt_id="attempt-001",
        max_env_step=8,
        max_train_iter=1,
        source_max_steps=8,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=1,
        batch_size=4,
        lightzero_eval_freq=0,
        skip_lightzero_eval_in_profile=False,
        profile_cuda_sync_enabled=False,
        profile_allow_auto_resume=False,
        profile_volume_commit=False,
        lightzero_multi_gpu=False,
        save_ckpt_after_iter=1,
        commit_on_checkpoint=False,
        stop_after_learner_train_calls=0,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        source_state_trail_render_mode=train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
        source_state_bonus_render_mode=train_mod.DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
        policy_observation_backend=train_mod.DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
        learner_seat_mode=train_mod.DEFAULT_LEARNER_SEAT_MODE,
        ego_action_straight_override_probability=0.0,
        policy_action_repeat_min=1,
        policy_action_repeat_max=1,
        policy_action_repeat_extra_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        opponent_death_mode=train_mod.DEFAULT_OPPONENT_DEATH_MODE,
        opponent_runtime_mode=train_mod.DEFAULT_OPPONENT_RUNTIME_MODE,
        env_telemetry_stride=1,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_report_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_mixture_spec=None,
        opponent_assignment_ref=None,
        background_eval_enabled=False,
        background_eval_launch_kind=train_mod.BACKGROUND_EVAL_LAUNCH_HOOK,
        background_eval_compute="cpu",
        background_eval_id_prefix=train_mod.DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=None,
        background_eval_max_steps=8,
        background_eval_step_detail_limit=None,
        background_eval_num_simulations=1,
        background_eval_batch_size=4,
        background_gif_enabled=False,
        background_gif_seed_offset=1,
        background_gif_max_steps=8,
        background_gif_frame_stride=1,
        background_gif_fps=8.0,
        background_gif_scale=1,
        background_gif_frame_size=train_mod.DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
        background_gif_collect_temperature=train_mod.DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
        background_gif_collect_epsilon=train_mod.DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
    )

    assert result["ok"] is True
    assert result["called_train_muzero"] is True
    assert result["mode"] == "train"
    assert result["final_volume_commit"]["attempted"] is True
    assert result["final_volume_commit"]["ok"] is True
    assert fake_volume.commit_count == 1
    summary = json.loads((tmp_path / result["summary_ref"]).read_text(encoding="utf-8"))
    assert summary["trainer_entrypoint"] == "lzero.entry.train_muzero"
    assert summary["command"]["env_variant"] == train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
    assert summary["command"]["two_seat_self_play"] is False
    assert summary["command"]["current_policy_two_seat_action_collection"] is False
    assert summary["command"]["fixed_opponent_is_two_seat_self_play"] is False
    assert len(train_calls) == 1
    assert train_calls[0]["seed"] == 123
    assert train_calls[0]["max_train_iter"] == 1
    assert train_calls[0]["max_env_step"] == 8
    assert train_calls[0]["configs"] == [
        {"env": {"type": "fake-env"}, "policy": {"cuda": False}},
        {"env": {"type": "fake-env"}},
    ]


def test_initial_policy_checkpoint_matching_shape_prepares_model_only_seed(
    monkeypatch,
    tmp_path,
):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    checkpoint_ref = (
        "training/lightzero-curvytron-visual-survival/source-run/attempts/try-source/"
        "train/lightzero_exp/ckpt/iteration_10000.pth.tar"
    )
    checkpoint_path = tmp_path / checkpoint_ref
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    source_model = {
        "representation_network.weight": torch.ones((2, 2)),
        "prediction_network.bias": torch.zeros((2,)),
    }
    torch.save(
        {
            "model": source_model,
            "optimizer": {"state": {"should_not": "survive"}},
        },
        checkpoint_path,
    )

    report = train_mod._prepare_initial_policy_checkpoint_load(
        initial_policy_checkpoint_ref=checkpoint_ref,
        initial_policy_checkpoint_state_key=None,
        initial_policy_checkpoint_load_mode=(
            train_mod.INITIAL_POLICY_CHECKPOINT_LOAD_MODE_MATCHING_SHAPE
        ),
        attempt_train_root=tmp_path / "attempt-train",
    )

    assert report is not None
    assert report["checkpoint_ref"] == checkpoint_ref
    assert report["prepared"]["kind"] == "model_only_checkpoint"
    assert report["prepared"]["source_state_key"] == "model"
    assert report["prepared"]["optimizer_keys_removed"] == ["optimizer"]
    prepared = torch.load(report["load_path"], map_location="cpu")
    assert sorted(prepared) == ["model", "optimizer", "target_model"]
    assert prepared["model"].keys() == source_model.keys()
    assert prepared["target_model"].keys() == source_model.keys()
    assert (
        prepared["optimizer"]["curvyzero_marker"]
        == train_mod.INITIAL_POLICY_MODEL_ONLY_OPTIMIZER_MARKER
    )


def test_initial_policy_checkpoint_audit_reports_model_load_without_optimizer_load():
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.representation_network = torch.nn.Linear(2, 2)
            self.prediction_network = torch.nn.Linear(2, 2)

    model = TinyModel()
    state_dict = model.state_dict()
    audit = train_mod._InitialPolicyCheckpointLoadAudit(checkpoint={"checkpoint_ref": "x"})
    restore = train_mod._install_initial_policy_checkpoint_load_audit(audit)
    try:
        model.load_state_dict(state_dict, strict=True)
    finally:
        restore()

    summary = audit.summary()
    assert summary["loaded"] is True
    assert summary["fresh_optimizer_preserved"] is True
    assert summary["module_loads"][0]["meaningful_model_load"] is True
    assert summary["module_loads"][0]["fresh_optimizer_preserved"] is True


def test_initial_policy_checkpoint_audit_skips_marked_optimizer_load():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    audit = train_mod._InitialPolicyCheckpointLoadAudit(checkpoint={"checkpoint_ref": "x"})
    restore = train_mod._install_initial_policy_checkpoint_load_audit(audit)
    try:
        optimizer.load_state_dict(
            {"curvyzero_marker": train_mod.INITIAL_POLICY_MODEL_ONLY_OPTIMIZER_MARKER}
        )
    finally:
        restore()

    summary = audit.summary()
    assert summary["fresh_optimizer_preserved"] is True
    assert summary["optimizer_load_calls"][0]["skipped_to_preserve_fresh_optimizer"] is True


def test_resume_sidecar_save_failure_does_not_fail_stock_checkpoint_hook(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    exp_name = (
        tmp_path
        / "training"
        / train_mod.TASK_ID
        / "run-sidecar-failure"
        / "attempts"
        / "attempt-sidecar-failure"
        / "train"
        / "lightzero_exp"
    )
    attempt_train_root = exp_name.parent

    class FakeBaseLearner:
        def call_hook(self, *args, **kwargs):
            return None

    class FakeSaveCkptHook:
        def __call__(self, engine):
            checkpoint_path = exp_name / "ckpt" / "iteration_9.pth.tar"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"fake-checkpoint")
            return {"saved": True}

    class FakeEngine:
        train_iter = 9

    def fake_train_muzero():
        return None

    def boom(**_kwargs):
        raise RuntimeError("sidecar volume write failed")

    fake_hook_module = types.ModuleType("ding.worker.learner.learner_hook")
    fake_hook_module.SaveCkptHook = FakeSaveCkptHook
    monkeypatch.setitem(sys.modules, "ding.worker.learner.learner_hook", fake_hook_module)
    monkeypatch.setitem(fake_train_muzero.__globals__, "BaseLearner", FakeBaseLearner)
    monkeypatch.setattr(train_mod, "_save_lightzero_resume_sidecar_state", boom)

    restore = train_mod._install_lightzero_full_resume_state_hooks(
        train_muzero=fake_train_muzero,
        run_id="run-sidecar-failure",
        attempt_id="attempt-sidecar-failure",
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

    captured = capsys.readouterr()
    assert "curvyzero resume sidecar save failed" in captured.out
    progress = json.loads((attempt_train_root / "progress_latest.json").read_text())
    assert progress["iteration"] == 9


def test_target_audit_hooks_return_original_collect_and_replay_results(monkeypatch):
    class FakeSegment:
        action_space_size = 3
        action_segment = [1]
        reward_segment = [0.5]
        to_play_segment = [-1]
        child_visit_segment = [[1, 0, 0]]
        root_value_segment = [0.1]
        action_mask_segment = [[1, 1, 1]]
        obs_segment = [
            {
                "observation": np.zeros((4, 64, 64), dtype=np.float32),
                "action_mask": np.ones(3, dtype=np.float32),
                "to_play": -1,
            }
        ]

        def __len__(self):
            return 1

    collect_result = ([FakeSegment()], [{"episode": 1}])
    sample_result = (
        ["obs", "actions", "masks", "indices", "weights"],
        ["rewards", "values", "policies"],
    )

    class FakeCollector:
        def collect(self, marker=None):
            return collect_result

    class FakeGameBuffer:
        def push_game_segments(self, segments):
            return {"pushed": segments}

        def sample(self, batch_size):
            return sample_result

    def fake_train_muzero():
        return None

    monkeypatch.setitem(fake_train_muzero.__globals__, "Collector", FakeCollector)
    monkeypatch.setitem(fake_train_muzero.__globals__, "MuZeroGameBuffer", FakeGameBuffer)

    audit = train_mod._LightZeroTargetAudit(mode="train", env_variant="source_state_fixed_opponent")
    original_collect = FakeCollector.collect
    original_push = FakeGameBuffer.push_game_segments
    original_sample = FakeGameBuffer.sample
    restore = train_mod._install_lightzero_target_audit(
        train_muzero=fake_train_muzero,
        audit=audit,
    )

    try:
        collector = FakeCollector()
        buffer = FakeGameBuffer()
        assert collector.collect(marker="same-return") is collect_result
        assert buffer.push_game_segments(["segment-a"]) == {"pushed": ["segment-a"]}
        assert buffer.sample(4) is sample_result
    finally:
        restore()

    assert FakeCollector.collect is original_collect
    assert FakeGameBuffer.push_game_segments is original_push
    assert FakeGameBuffer.sample is original_sample
    summary = audit.summary()
    assert summary["training_behavior_changed"] is False
    assert summary["counts"]["collector_collect_calls"] == 1
    assert summary["counts"]["replay_push_calls"] == 1
    assert summary["counts"]["replay_sample_calls"] == 1
    assert summary["counts"]["game_segments_seen"] == 1


def _fake_resolved_opponent_assignment(
    *,
    assignment_id: str = "assignment-b",
    sha: str = "b" * 64,
) -> dict[str, object]:
    return {
        "assignment_id": assignment_id,
        "assignment_ref": f"training/{assignment_id}/assignment.json",
        "assignment_sha256": sha,
        "source_epoch": 3,
        "source_ref": "tournaments/main/snapshots/003.json",
        "opponent_mixture": {
            "schema_id": train_mod.OPPONENT_MIXTURE_SCHEMA_ID,
            "seed": 17,
            "entries": [
                {
                    "name": "blank",
                    "weight": 1.0,
                    "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                    "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                    "opponent_immortal": True,
                }
            ],
        },
    }


def _write_assignment_payload(
    path: Path,
    *,
    assignment_id: str,
    source_epoch: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": assignment_id,
                "source_epoch": source_epoch,
                "source_ref": f"tournaments/curvytron/leaderboards/main/snapshots/{source_epoch:03d}.json",
                "seed": source_epoch,
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                        "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                        "opponent_immortal": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_resolve_opponent_assignment_for_env_accepts_refresh_pointer(tmp_path):
    assignment_path = tmp_path / "assignment-live.json"
    pointer_path = tmp_path / "assignment-pointer.json"
    _write_assignment_payload(
        assignment_path,
        assignment_id="assignment-live",
        source_epoch=5,
    )
    assignment_payload = json.loads(assignment_path.read_text(encoding="utf-8"))
    assignment_sha256 = train_mod.canonical_assignment_json_sha256(assignment_payload)
    pointer_path.write_text(
        json.dumps(
            {
                "schema_id": train_mod.OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID,
                "assignment_ref": assignment_path.as_posix(),
                "assignment_sha256": assignment_sha256,
            }
        ),
        encoding="utf-8",
    )

    resolved = train_mod._resolve_opponent_assignment_for_env(
        opponent_assignment_ref=pointer_path.as_posix()
    )

    assert resolved is not None
    assert resolved["assignment_id"] == "assignment-live"
    assert resolved["assignment_sha256"] == assignment_sha256
    assert resolved["assignment_pointer"]["pointer_ref"] == pointer_path.as_posix()
    assert resolved["assignment_pointer"]["pointed_assignment_ref"] == assignment_path.as_posix()


def test_resolve_opponent_assignment_for_env_reloads_volume_before_pointer_read(
    monkeypatch,
    tmp_path,
):
    assignment_path = tmp_path / "assignment-live.json"
    pointer_path = tmp_path / "assignment-pointer.json"
    _write_assignment_payload(
        assignment_path,
        assignment_id="assignment-live",
        source_epoch=5,
    )
    assignment_payload = json.loads(assignment_path.read_text(encoding="utf-8"))
    assignment_sha256 = train_mod.canonical_assignment_json_sha256(assignment_payload)
    pointer_path.write_text(
        json.dumps(
            {
                "schema_id": train_mod.OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID,
                "assignment_ref": assignment_path.as_posix(),
                "assignment_sha256": assignment_sha256,
            }
        ),
        encoding="utf-8",
    )

    class FakeVolume:
        def __init__(self) -> None:
            self.reload_count = 0

        def reload(self) -> None:
            self.reload_count += 1

    fake_volume = FakeVolume()
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)

    resolved = train_mod._resolve_opponent_assignment_for_env(
        opponent_assignment_ref=pointer_path.as_posix(),
        reload_volume_before_read=True,
    )

    assert resolved is not None
    assert resolved["assignment_id"] == "assignment-live"
    assert fake_volume.reload_count == 1


def test_resolve_opponent_assignment_for_env_reloads_outside_runs_mount(
    monkeypatch,
    tmp_path,
):
    runs_mount = tmp_path / "runs"
    inside_cwd = runs_mount / "training"
    inside_cwd.mkdir(parents=True)
    assignment_path = runs_mount / "assignment-live.json"
    pointer_path = runs_mount / "assignment-pointer.json"
    _write_assignment_payload(
        assignment_path,
        assignment_id="assignment-live",
        source_epoch=5,
    )
    assignment_payload = json.loads(assignment_path.read_text(encoding="utf-8"))
    assignment_sha256 = train_mod.canonical_assignment_json_sha256(assignment_payload)
    pointer_path.write_text(
        json.dumps(
            {
                "schema_id": train_mod.OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID,
                "assignment_ref": assignment_path.as_posix(),
                "assignment_sha256": assignment_sha256,
            }
        ),
        encoding="utf-8",
    )

    class CwdSensitiveVolume:
        def __init__(self) -> None:
            self.reload_count = 0
            self.reload_cwd: Path | None = None

        def reload(self) -> None:
            self.reload_count += 1
            self.reload_cwd = Path.cwd()
            if train_mod._path_is_inside_or_equal(self.reload_cwd, runs_mount):
                raise RuntimeError("cwd is inside volume")

    fake_volume = CwdSensitiveVolume()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", runs_mount)
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)
    monkeypatch.chdir(inside_cwd)

    resolved = train_mod._resolve_opponent_assignment_for_env(
        opponent_assignment_ref=pointer_path.as_posix(),
        reload_volume_before_read=True,
    )

    assert resolved is not None
    assert resolved["assignment_id"] == "assignment-live"
    assert fake_volume.reload_count == 1
    assert fake_volume.reload_cwd is not None
    assert not train_mod._path_is_inside_or_equal(fake_volume.reload_cwd, runs_mount)
    assert Path.cwd() == inside_cwd


def test_resolve_opponent_assignment_for_env_restores_cwd_when_reload_fails(
    monkeypatch,
    tmp_path,
):
    runs_mount = tmp_path / "runs"
    inside_cwd = runs_mount / "training"
    inside_cwd.mkdir(parents=True)
    pointer_path = runs_mount / "assignment-pointer.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_id": train_mod.OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID,
                "assignment_ref": (runs_mount / "missing-assignment.json").as_posix(),
            }
        ),
        encoding="utf-8",
    )

    class FailingReloadVolume:
        def reload(self) -> None:
            raise RuntimeError("volume reload exploded")

    monkeypatch.setattr(train_mod, "RUNS_MOUNT", runs_mount)
    monkeypatch.setattr(train_mod, "runs_volume", FailingReloadVolume())
    monkeypatch.chdir(inside_cwd)

    with pytest.raises(RuntimeError, match="volume reload exploded"):
        train_mod._resolve_opponent_assignment_for_env(
            opponent_assignment_ref=pointer_path.as_posix(),
            reload_volume_before_read=True,
        )

    assert Path.cwd() == inside_cwd


def test_resolve_opponent_assignment_for_env_can_refresh_from_control_volume(
    monkeypatch,
    tmp_path,
):
    runs_mount = tmp_path / "runs"
    control_mount = tmp_path / "control"
    runs_mount.mkdir()
    control_mount.mkdir()
    assignment_path = control_mount / "assignment-live.json"
    pointer_path = control_mount / "assignment-pointer.json"
    _write_assignment_payload(
        assignment_path,
        assignment_id="assignment-live",
        source_epoch=5,
    )
    assignment_payload = json.loads(assignment_path.read_text(encoding="utf-8"))
    assignment_sha256 = train_mod.canonical_assignment_json_sha256(assignment_payload)
    pointer_path.write_text(
        json.dumps(
            {
                "schema_id": train_mod.OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID,
                "assignment_ref": "control:assignment-live.json",
                "assignment_sha256": assignment_sha256,
            }
        ),
        encoding="utf-8",
    )

    class FailingRunsVolume:
        def reload(self) -> None:
            raise AssertionError("runs volume must not reload for control pointer")

    class FakeControlVolume:
        def __init__(self) -> None:
            self.reload_count = 0

        def reload(self) -> None:
            self.reload_count += 1

    control_volume = FakeControlVolume()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", runs_mount)
    monkeypatch.setattr(train_mod, "CONTROL_MOUNT", control_mount)
    monkeypatch.setattr(train_mod, "runs_volume", FailingRunsVolume())
    monkeypatch.setattr(train_mod, "control_volume", control_volume)

    resolved = train_mod._resolve_opponent_assignment_for_env(
        opponent_assignment_ref="control:assignment-pointer.json",
        reload_volume_before_read=True,
    )

    assert resolved is not None
    assert resolved["assignment_id"] == "assignment-live"
    assert resolved["assignment_sha256"] == assignment_sha256
    assert resolved["assignment_ref"] == "assignment-live.json"
    assert resolved["assignment_resolution"]["mount"] == "control"
    assert resolved["assignment_pointer"]["pointer_ref"] == "assignment-pointer.json"
    assert control_volume.reload_count == 1


def test_resolve_opponent_assignment_for_env_rejects_bad_pointer_sha(tmp_path):
    assignment_path = tmp_path / "assignment-live.json"
    pointer_path = tmp_path / "assignment-pointer.json"
    _write_assignment_payload(
        assignment_path,
        assignment_id="assignment-live",
        source_epoch=5,
    )
    pointer_path.write_text(
        json.dumps(
            {
                "schema_id": train_mod.OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID,
                "assignment_ref": assignment_path.as_posix(),
                "assignment_sha256": "not-the-real-sha",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sha mismatch"):
        train_mod._resolve_opponent_assignment_for_env(
            opponent_assignment_ref=pointer_path.as_posix()
        )


class _FakeRunsVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def _install_minimal_visual_train_mocks(
    monkeypatch,
    tmp_path: Path,
    fake_train_muzero,
) -> _FakeRunsVolume:
    fake_volume = _FakeRunsVolume()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)
    monkeypatch.setattr(train_mod, "_version_or_missing", lambda *_args: train_mod.LIGHTZERO_VERSION)
    monkeypatch.setattr(train_mod.os, "chdir", lambda _path: None)
    monkeypatch.setattr(
        train_mod,
        "_build_visual_survival_configs",
        lambda **kwargs: {
            "main_config": {"env": {"type": "fake-env"}, "policy": {"cuda": False}},
            "create_config": {"env": {"type": "fake-env"}},
            "patches": [{"path": "fake", "new": True}],
            "surface": {
                "env_variant": kwargs["env_variant"],
                "env_type": "fake-env",
                "called_from": "minimal-visual-train-mock",
            },
        },
    )
    monkeypatch.setattr(train_mod, "_compile_config_summary", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(train_mod, "_validate_visual_survival_surface", lambda **_kwargs: [])
    monkeypatch.setattr(train_mod, "_install_lightzero_full_resume_state_hooks", lambda **_kwargs: None)
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
    opponent_assignment_ref: str | None = None,
    opponent_assignment_refresh_interval_train_iter: int = 0,
    opponent_assignment_refresh_ref: str | None = None,
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
        "opponent_assignment_ref": opponent_assignment_ref,
        "opponent_assignment_refresh_interval_train_iter": (
            opponent_assignment_refresh_interval_train_iter
        ),
        "opponent_assignment_refresh_ref": opponent_assignment_refresh_ref,
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


def test_opponent_assignment_refresh_pure_helpers_are_strict_and_copied():
    assignment = _fake_resolved_opponent_assignment()

    assert train_mod._lightzero_collect_train_iter_from_call((), {}) == 0
    assert train_mod._lightzero_collect_train_iter_from_call((), {"train_iter": 50}) == 50
    assert train_mod._lightzero_collect_train_iter_from_call((7,), {}) == 0
    assert train_mod._lightzero_collect_train_iter_from_call((None, 100), {}) == 100
    assert not train_mod._opponent_assignment_refresh_due(
        train_iter=49,
        interval_train_iter=50,
        last_checked_bucket=0,
    )
    assert train_mod._opponent_assignment_refresh_due(
        train_iter=50,
        interval_train_iter=50,
        last_checked_bucket=0,
    )
    assert not train_mod._opponent_assignment_refresh_due(
        train_iter=51,
        interval_train_iter=50,
        last_checked_bucket=1,
    )
    assert train_mod._opponent_assignment_refresh_due(
        train_iter=100,
        interval_train_iter=50,
        last_checked_bucket=1,
    )

    reset_param = train_mod._opponent_assignment_refresh_reset_param(
        env_num=2,
        opponent_assignment=assignment,
        refresh_index=4,
    )

    assert sorted(reset_param) == [0, 1]
    assert reset_param[0]["opponent_assignment_context"] == {
        "assignment_id": "assignment-b",
        "assignment_ref": "training/assignment-b/assignment.json",
        "assignment_sha256": "b" * 64,
        "source_epoch": 3,
        "source_ref": "tournaments/main/snapshots/003.json",
        "refresh_index": 4,
    }
    reset_param[0]["opponent_mixture"]["entries"][0]["name"] = "mutated"
    assert reset_param[1]["opponent_mixture"]["entries"][0]["name"] == "blank"
    assert assignment["opponent_mixture"]["entries"][0]["name"] == "blank"


def test_opponent_assignment_refresh_ready_report_requires_all_envs():
    assignment = _fake_resolved_opponent_assignment()

    class FakeEnvManager:
        env_num = 2
        ready_obs = {0: {"observation": "new"}, 1: {"observation": "new"}}
        last_reset_info = [
            {
                "opponent_assignment_id": "assignment-b",
                "opponent_assignment_ref": "training/assignment-b/assignment.json",
                "opponent_assignment_sha256": "b" * 64,
                "opponent_assignment_refresh_index": 2,
            },
            {
                "opponent_assignment_id": "assignment-b",
                "opponent_assignment_ref": "training/assignment-b/assignment.json",
                "opponent_assignment_sha256": "b" * 64,
                "opponent_assignment_refresh_index": 2,
            },
        ]

    report = train_mod._opponent_assignment_refresh_ready_report(
        env_manager=FakeEnvManager(),
        opponent_assignment=assignment,
        refresh_index=2,
    )
    assert report["ok"] is True

    class MissingReady(FakeEnvManager):
        ready_obs = {0: {"observation": "new"}}

    missing_report = train_mod._opponent_assignment_refresh_ready_report(
        env_manager=MissingReady(),
        opponent_assignment=assignment,
        refresh_index=2,
    )
    assert missing_report["ok"] is False
    assert missing_report["reason"] == "not all envs are ready after assignment refresh"

    class StaleInfo(FakeEnvManager):
        last_reset_info = [
            FakeEnvManager.last_reset_info[0],
            {
                **FakeEnvManager.last_reset_info[1],
                "opponent_assignment_sha256": "a" * 64,
            },
        ]

    stale_report = train_mod._opponent_assignment_refresh_ready_report(
        env_manager=StaleInfo(),
        opponent_assignment=assignment,
        refresh_index=2,
    )
    assert stale_report["ok"] is False
    assert stale_report["reason"] == "env assignment info mismatch after refresh"
    assert stale_report["mismatches"][0]["env_id"] == 1


def test_opponent_assignment_refresh_apply_resets_before_collect():
    assignment = _fake_resolved_opponent_assignment()

    class FakeEnvManager:
        env_num = 2

        def __init__(self):
            self.reset_calls = []
            self.ready_obs = {}
            self.last_reset_info = []

        def reset(self, reset_param):
            self.reset_calls.append(reset_param)
            self.ready_obs = {env_id: {"observation": "new"} for env_id in reset_param}
            self.last_reset_info = []
            for env_id in sorted(reset_param):
                context = reset_param[env_id]["opponent_assignment_context"]
                self.last_reset_info.append(
                    {
                        "opponent_assignment_id": context["assignment_id"],
                        "opponent_assignment_ref": context["assignment_ref"],
                        "opponent_assignment_sha256": context["assignment_sha256"],
                        "opponent_assignment_refresh_index": context["refresh_index"],
                    }
                )
            self.ready_obs = {env_id: {"observation": "new"} for env_id in reset_param}
            self.last_reset_info = []
            for env_id in sorted(reset_param):
                context = reset_param[env_id]["opponent_assignment_context"]
                self.last_reset_info.append(
                    {
                        "opponent_assignment_id": context["assignment_id"],
                        "opponent_assignment_ref": context["assignment_ref"],
                        "opponent_assignment_sha256": context["assignment_sha256"],
                        "opponent_assignment_refresh_index": context["refresh_index"],
                    }
                )

    class FakePolicy:
        def __init__(self):
            self.reset_calls = []

        def reset(self, env_ids):
            self.reset_calls.append(env_ids)

    class FakeCollector:
        def __init__(self):
            self._env = FakeEnvManager()
            self._policy = FakePolicy()
            self.reset_stats = []

        def _reset_stat(self, env_id):
            self.reset_stats.append(env_id)

    collector = FakeCollector()

    report = train_mod._apply_opponent_assignment_refresh_to_collector_env(
        collector=collector,
        opponent_assignment=assignment,
        refresh_index=3,
    )

    assert report["ok"] is True
    assert len(collector._env.reset_calls) == 1
    assert sorted(collector._env.reset_calls[0]) == [0, 1]
    assert collector._policy.reset_calls == [[0, 1]]
    assert collector.reset_stats == [0, 1]


def test_opponent_assignment_refresh_hook_handles_due_unchanged_and_failure(monkeypatch):
    assignment_a = _fake_resolved_opponent_assignment(
        assignment_id="assignment-a",
        sha="a" * 64,
    )
    assignment_b = _fake_resolved_opponent_assignment()
    events = []

    class FakeEnvManager:
        env_num = 2

        def __init__(self):
            self.reset_calls = []
            self.ready_obs = {0: {"observation": "old"}, 1: {"observation": "old"}}
            self.last_reset_info = [
                {
                    "opponent_assignment_id": "assignment-a",
                    "opponent_assignment_ref": "training/assignment-a/assignment.json",
                    "opponent_assignment_sha256": "a" * 64,
                    "opponent_assignment_refresh_index": 0,
                },
                {
                    "opponent_assignment_id": "assignment-a",
                    "opponent_assignment_ref": "training/assignment-a/assignment.json",
                    "opponent_assignment_sha256": "a" * 64,
                    "opponent_assignment_refresh_index": 0,
                },
            ]

        def reset(self, reset_param):
            self.reset_calls.append(reset_param)
            self.ready_obs = {env_id: {"observation": "new"} for env_id in reset_param}
            self.last_reset_info = []
            for env_id in sorted(reset_param):
                context = reset_param[env_id]["opponent_assignment_context"]
                self.last_reset_info.append(
                    {
                        "opponent_assignment_id": context["assignment_id"],
                        "opponent_assignment_ref": context["assignment_ref"],
                        "opponent_assignment_sha256": context["assignment_sha256"],
                        "opponent_assignment_refresh_index": context["refresh_index"],
                    }
                )

    class FakePolicy:
        def __init__(self):
            self.reset_calls = []

        def reset(self, env_ids):
            self.reset_calls.append(env_ids)

    class FakeCollector:
        def __init__(self):
            self._env = FakeEnvManager()
            self._policy = FakePolicy()
            self.reset_stats = []
            self.collect_calls = []

        def _reset_stat(self, env_id):
            self.reset_stats.append(env_id)

        def collect(self, marker=None, train_iter=0):
            self.collect_calls.append((marker, train_iter))
            return {"marker": marker, "train_iter": train_iter}

    def fake_train_muzero():
        return None

    pending_assignments = [assignment_a, assignment_b]

    def load_pending_assignment():
        return pending_assignments.pop(0)

    monkeypatch.setitem(fake_train_muzero.__globals__, "Collector", FakeCollector)
    original_collect = FakeCollector.collect
    restore = train_mod._install_lightzero_opponent_assignment_refresh_hook(
        train_muzero=fake_train_muzero,
        interval_train_iter=50,
        load_pending_assignment=load_pending_assignment,
        initial_assignment=assignment_a,
        event_sink=events.append,
    )

    try:
        collector = FakeCollector()
        assert collector.collect(marker="not-due", train_iter=49) == {
            "marker": "not-due",
            "train_iter": 49,
        }
        assert collector._env.reset_calls == []
        assert collector.collect(marker="unchanged", train_iter=50)["marker"] == "unchanged"
        assert collector._env.reset_calls == []
        assert events[-1]["decision"] == "unchanged"
        assert collector.collect(marker="changed", train_iter=100)["marker"] == "changed"
        assert len(collector._env.reset_calls) == 1
        assert events[-1]["decision"] == "applied"
        assert events[-1]["refresh_index"] == 1
        assert collector._policy.reset_calls == [[0, 1]]
        assert collector.reset_stats == [0, 1]
    finally:
        assert restore is not None
        restore()

    assert FakeCollector.collect is original_collect


def test_opponent_assignment_refresh_hook_can_apply_at_first_collect(monkeypatch):
    assignment_a = _fake_resolved_opponent_assignment(
        assignment_id="assignment-a",
        sha="a" * 64,
    )
    assignment_b = _fake_resolved_opponent_assignment()
    events = []

    class FakeEnvManager:
        env_num = 1

        def __init__(self):
            self.reset_calls = []
            self.ready_obs = {0: {"observation": "old"}}
            self.last_reset_info = []

        def reset(self, reset_param):
            self.reset_calls.append(reset_param)
            context = reset_param[0]["opponent_assignment_context"]
            self.ready_obs = {0: {"observation": "new"}}
            self.last_reset_info = [
                {
                    "opponent_assignment_id": context["assignment_id"],
                    "opponent_assignment_ref": context["assignment_ref"],
                    "opponent_assignment_sha256": context["assignment_sha256"],
                    "opponent_assignment_refresh_index": context["refresh_index"],
                }
            ]

    class FakeCollector:
        def __init__(self):
            self._env = FakeEnvManager()
            self.collect_calls = []

        def collect(self, train_iter=0):
            self.collect_calls.append(train_iter)
            return {"collected": True, "train_iter": train_iter}

    def fake_train_muzero():
        return None

    monkeypatch.setitem(fake_train_muzero.__globals__, "Collector", FakeCollector)
    restore = train_mod._install_lightzero_opponent_assignment_refresh_hook(
        train_muzero=fake_train_muzero,
        interval_train_iter=50,
        load_pending_assignment=lambda: assignment_b,
        initial_assignment=assignment_a,
        event_sink=events.append,
    )

    try:
        collector = FakeCollector()
        assert collector.collect(train_iter=0) == {"collected": True, "train_iter": 0}
    finally:
        assert restore is not None
        restore()

    assert collector.collect_calls == [0]
    assert len(collector._env.reset_calls) == 1
    assert events[-1]["decision"] == "applied"
    assert events[-1]["assignment_id"] == "assignment-b"


def test_opponent_assignment_refresh_hook_bad_pending_retries_without_reset(monkeypatch):
    assignment_a = _fake_resolved_opponent_assignment(
        assignment_id="assignment-a",
        sha="a" * 64,
    )
    assignment_b = _fake_resolved_opponent_assignment()
    events = []

    class FakeEnvManager:
        env_num = 2

        def __init__(self):
            self.reset_calls = []
            self.ready_obs = {0: {"observation": "old"}, 1: {"observation": "old"}}
            self.last_reset_info = []

        def reset(self, reset_param):
            self.reset_calls.append(reset_param)
            self.ready_obs = {env_id: {"observation": "new"} for env_id in reset_param}
            self.last_reset_info = []
            for env_id in sorted(reset_param):
                context = reset_param[env_id]["opponent_assignment_context"]
                self.last_reset_info.append(
                    {
                        "opponent_assignment_id": context["assignment_id"],
                        "opponent_assignment_ref": context["assignment_ref"],
                        "opponent_assignment_sha256": context["assignment_sha256"],
                        "opponent_assignment_refresh_index": context["refresh_index"],
                    }
                )

    class FakeCollector:
        def __init__(self):
            self._env = FakeEnvManager()
            self.collect_calls = 0

        def collect(self, train_iter=0):
            self.collect_calls += 1
            return {"collected": True, "train_iter": train_iter}

    def fake_train_muzero():
        return None

    load_calls = 0

    def load_pending_assignment():
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            raise FileNotFoundError("not yet")
        return assignment_b

    monkeypatch.setitem(fake_train_muzero.__globals__, "Collector", FakeCollector)
    restore = train_mod._install_lightzero_opponent_assignment_refresh_hook(
        train_muzero=fake_train_muzero,
        interval_train_iter=50,
        load_pending_assignment=load_pending_assignment,
        initial_assignment=assignment_a,
        event_sink=events.append,
    )

    try:
        collector = FakeCollector()
        assert collector.collect(train_iter=50) == {"collected": True, "train_iter": 50}
        assert collector._env.reset_calls == []
        assert collector.collect(train_iter=51) == {"collected": True, "train_iter": 51}
    finally:
        assert restore is not None
        restore()

    assert collector.collect_calls == 2
    assert len(collector._env.reset_calls) == 1
    assert events[0]["decision"] == "kept_previous"
    assert "pending assignment load failed" in events[0]["reason"]
    assert events[-1]["decision"] == "applied"


def test_opponent_assignment_refresh_hook_missing_sha_retries_without_reset(monkeypatch):
    assignment_a = _fake_resolved_opponent_assignment(
        assignment_id="assignment-a",
        sha="a" * 64,
    )
    assignment_b = _fake_resolved_opponent_assignment()
    events = []

    class FakeEnvManager:
        env_num = 1

        def __init__(self):
            self.reset_calls = []
            self.ready_obs = {0: {"observation": "old"}}
            self.last_reset_info = []

        def reset(self, reset_param):
            self.reset_calls.append(reset_param)
            context = reset_param[0]["opponent_assignment_context"]
            self.ready_obs = {0: {"observation": "new"}}
            self.last_reset_info = [
                {
                    "opponent_assignment_id": context["assignment_id"],
                    "opponent_assignment_ref": context["assignment_ref"],
                    "opponent_assignment_sha256": context["assignment_sha256"],
                    "opponent_assignment_refresh_index": context["refresh_index"],
                }
            ]

    class FakeCollector:
        def __init__(self):
            self._env = FakeEnvManager()
            self.collect_calls = 0

        def collect(self, train_iter=0):
            self.collect_calls += 1
            return {"collected": True, "train_iter": train_iter}

    def fake_train_muzero():
        return None

    pending = [dict(assignment_b, assignment_sha256=None), assignment_b]

    monkeypatch.setitem(fake_train_muzero.__globals__, "Collector", FakeCollector)
    restore = train_mod._install_lightzero_opponent_assignment_refresh_hook(
        train_muzero=fake_train_muzero,
        interval_train_iter=50,
        load_pending_assignment=lambda: pending.pop(0),
        initial_assignment=assignment_a,
        event_sink=events.append,
    )

    try:
        collector = FakeCollector()
        assert collector.collect(train_iter=50) == {"collected": True, "train_iter": 50}
        assert collector._env.reset_calls == []
        assert collector.collect(train_iter=51) == {"collected": True, "train_iter": 51}
    finally:
        assert restore is not None
        restore()

    assert collector.collect_calls == 2
    assert len(collector._env.reset_calls) == 1
    assert events[0]["decision"] == "kept_previous"
    assert "missing assignment_sha256" in events[0]["reason"]
    assert events[-1]["decision"] == "applied"


def test_opponent_assignment_refresh_hook_blocks_collect_on_reset_proof_failure(monkeypatch):
    assignment_a = _fake_resolved_opponent_assignment(
        assignment_id="assignment-a",
        sha="a" * 64,
    )
    assignment_b = _fake_resolved_opponent_assignment()
    events = []

    class FakeEnvManager:
        env_num = 2

        def __init__(self):
            self.reset_calls = []
            self.ready_obs = {0: {"observation": "old"}, 1: {"observation": "old"}}
            self.last_reset_info = []

        def reset(self, reset_param):
            self.reset_calls.append(reset_param)
            self.ready_obs = {env_id: {"observation": "new"} for env_id in reset_param}
            context = reset_param[0]["opponent_assignment_context"]
            self.last_reset_info = [
                {
                    "opponent_assignment_id": context["assignment_id"],
                    "opponent_assignment_ref": context["assignment_ref"],
                    "opponent_assignment_sha256": context["assignment_sha256"],
                    "opponent_assignment_refresh_index": context["refresh_index"],
                },
                {
                    "opponent_assignment_id": context["assignment_id"],
                    "opponent_assignment_ref": context["assignment_ref"],
                    "opponent_assignment_sha256": "a" * 64,
                    "opponent_assignment_refresh_index": context["refresh_index"],
                },
            ]

    class FakePolicy:
        def reset(self, env_ids):
            raise AssertionError("policy reset should not run after failed env proof")

    class FakeCollector:
        def __init__(self):
            self._env = FakeEnvManager()
            self._policy = FakePolicy()
            self.collect_called = False

        def collect(self, train_iter=0):
            self.collect_called = True
            return {"collected": True}

    def fake_train_muzero():
        return None

    monkeypatch.setitem(fake_train_muzero.__globals__, "Collector", FakeCollector)
    restore = train_mod._install_lightzero_opponent_assignment_refresh_hook(
        train_muzero=fake_train_muzero,
        interval_train_iter=50,
        load_pending_assignment=lambda: assignment_b,
        initial_assignment=assignment_a,
        event_sink=events.append,
    )

    collector = FakeCollector()
    try:
        with pytest.raises(RuntimeError, match="not proven"):
            collector.collect(train_iter=50)
    finally:
        assert restore is not None
        restore()

    assert collector.collect_called is False
    assert len(collector._env.reset_calls) == 1
    assert events[-1]["decision"] == "failed_after_reset_attempt"


def test_run_visual_survival_train_installs_refresh_hook_and_writes_events(
    monkeypatch,
    tmp_path,
):
    class FakeVolume:
        def __init__(self) -> None:
            self.commit_count = 0

        def commit(self) -> None:
            self.commit_count += 1

    fake_volume = FakeVolume()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", fake_volume)
    monkeypatch.setattr(train_mod, "_version_or_missing", lambda *_args: train_mod.LIGHTZERO_VERSION)
    monkeypatch.setattr(train_mod.os, "chdir", lambda _path: None)
    monkeypatch.setattr(
        train_mod,
        "_build_visual_survival_configs",
        lambda **kwargs: {
            "main_config": {"env": {"type": "fake-env"}, "policy": {"cuda": False}},
            "create_config": {"env": {"type": "fake-env"}},
            "patches": [{"path": "fake", "new": True}],
            "surface": {
                "env_variant": kwargs["env_variant"],
                "env_type": "fake-env",
                "called_from": "refresh-hook-install-test",
            },
        },
    )
    monkeypatch.setattr(train_mod, "_compile_config_summary", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(train_mod, "_validate_visual_survival_surface", lambda **_kwargs: [])
    monkeypatch.setattr(train_mod, "_install_lightzero_full_resume_state_hooks", lambda **_kwargs: None)
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

    assignment_a_path = tmp_path / "assignment-a.json"
    assignment_b_path = tmp_path / "assignment-b.json"
    assignment_pointer_path = tmp_path / "assignment-pointer.json"
    _write_assignment_payload(assignment_a_path, assignment_id="assignment-a", source_epoch=1)
    _write_assignment_payload(assignment_b_path, assignment_id="assignment-b", source_epoch=2)
    assignment_b_payload = json.loads(assignment_b_path.read_text(encoding="utf-8"))
    assignment_b_sha256 = train_mod.canonical_assignment_json_sha256(assignment_b_payload)
    assignment_pointer_path.write_text(
        json.dumps(
            {
                "schema_id": train_mod.OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID,
                "assignment_ref": assignment_b_path.as_posix(),
                "assignment_sha256": assignment_b_sha256,
            }
        ),
        encoding="utf-8",
    )

    class FakeEnvManager:
        env_num = 2

        def __init__(self) -> None:
            self.reset_calls = []
            self.ready_obs = {0: {"observation": "old"}, 1: {"observation": "old"}}
            self.last_reset_info = []

        def reset(self, reset_param):
            self.reset_calls.append(reset_param)
            self.ready_obs = {env_id: {"observation": "new"} for env_id in reset_param}
            self.last_reset_info = []
            for env_id in sorted(reset_param):
                context = reset_param[env_id]["opponent_assignment_context"]
                self.last_reset_info.append(
                    {
                        "opponent_assignment_id": context["assignment_id"],
                        "opponent_assignment_ref": context["assignment_ref"],
                        "opponent_assignment_sha256": context["assignment_sha256"],
                        "opponent_assignment_refresh_index": context["refresh_index"],
                    }
                )

    class FakePolicy:
        def __init__(self) -> None:
            self.reset_calls = []

        def reset(self, env_ids):
            self.reset_calls.append(env_ids)

    class FakeCollector:
        def __init__(self) -> None:
            self._env = FakeEnvManager()
            self._policy = FakePolicy()
            self.reset_stats = []
            self.collect_calls = []

        def _reset_stat(self, env_id):
            self.reset_stats.append(env_id)

        def collect(self, train_iter=0):
            self.collect_calls.append(train_iter)
            return {"collected": True, "train_iter": train_iter}

    collectors = []

    def fake_train_muzero(configs, *, seed, max_train_iter, max_env_step):
        collector = FakeCollector()
        collectors.append(collector)
        return collector.collect(train_iter=50)

    fake_entry_module = types.ModuleType("lzero.entry")
    fake_entry_module.train_muzero = fake_train_muzero
    monkeypatch.setitem(sys.modules, "lzero.entry", fake_entry_module)
    monkeypatch.setitem(fake_train_muzero.__globals__, "Collector", FakeCollector)

    result = train_mod._run_visual_survival_train(
        mode="train",
        compute="cpu",
        seed=123,
        run_id="refresh-hook-install-test",
        attempt_id="attempt-001",
        max_env_step=8,
        max_train_iter=1,
        source_max_steps=8,
        decision_ms=train_mod.DEFAULT_DECISION_MS,
        collector_env_num=2,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=1,
        batch_size=4,
        lightzero_eval_freq=0,
        skip_lightzero_eval_in_profile=False,
        profile_cuda_sync_enabled=False,
        profile_allow_auto_resume=False,
        profile_volume_commit=False,
        lightzero_multi_gpu=False,
        save_ckpt_after_iter=1,
        commit_on_checkpoint=False,
        stop_after_learner_train_calls=0,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        source_state_trail_render_mode=train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
        source_state_bonus_render_mode=train_mod.DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
        policy_observation_backend=train_mod.DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
        learner_seat_mode=train_mod.DEFAULT_LEARNER_SEAT_MODE,
        ego_action_straight_override_probability=0.0,
        policy_action_repeat_min=1,
        policy_action_repeat_max=1,
        policy_action_repeat_extra_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        opponent_death_mode=train_mod.DEFAULT_OPPONENT_DEATH_MODE,
        opponent_runtime_mode=train_mod.DEFAULT_OPPONENT_RUNTIME_MODE,
        env_telemetry_stride=1,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_report_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_mixture_spec=None,
        opponent_assignment_ref=str(assignment_a_path),
        opponent_assignment_refresh_interval_train_iter=50,
        opponent_assignment_refresh_ref=str(assignment_pointer_path),
        background_eval_enabled=False,
        background_eval_launch_kind=train_mod.BACKGROUND_EVAL_LAUNCH_HOOK,
        background_eval_compute="cpu",
        background_eval_id_prefix=train_mod.DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=None,
        background_eval_max_steps=8,
        background_eval_step_detail_limit=None,
        background_eval_num_simulations=1,
        background_eval_batch_size=4,
        background_gif_enabled=False,
        background_gif_seed_offset=1,
        background_gif_max_steps=8,
        background_gif_frame_stride=1,
        background_gif_fps=8.0,
        background_gif_scale=1,
        background_gif_frame_size=train_mod.DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
        background_gif_collect_temperature=train_mod.DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
        background_gif_collect_epsilon=train_mod.DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
    )

    assert result["ok"] is True
    assert len(collectors) == 1
    collector = collectors[0]
    assert collector.collect_calls == [50]
    assert len(collector._env.reset_calls) == 1
    assert collector._policy.reset_calls == [[0, 1]]
    assert collector.reset_stats == [0, 1]

    summary = json.loads((tmp_path / result["summary_ref"]).read_text(encoding="utf-8"))
    refresh_summary = summary["opponent_assignment_refresh"]
    assert refresh_summary["enabled"] is True
    assert refresh_summary["mode"] == "assignment_or_pointer_ref_refresh"
    assert refresh_summary["event_count"] == 1
    assert refresh_summary["events"][0]["decision"] == "applied"
    assert (
        summary["command"]["opponent_assignment_refresh"]["pending_assignment_ref"]
        == assignment_pointer_path.as_posix()
    )
    assert (
        "curvyzero_opponent_assignment_refresh_pointer/v0"
        in summary["command"]["opponent_assignment_refresh"]["control_plane_caveat"]
    )

    event_rows = [
        json.loads(line)
        for line in (tmp_path / refresh_summary["events_ref"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["decision"] for row in event_rows] == ["applied"]
    assert event_rows[0]["assignment_id"] == "assignment-b"


def test_run_visual_survival_train_fails_when_refresh_hook_is_not_installed(
    monkeypatch,
    tmp_path,
):
    def fake_train_muzero(configs, *, seed, max_train_iter, max_env_step):
        raise AssertionError("train_muzero should not run without the refresh hook")

    _install_minimal_visual_train_mocks(monkeypatch, tmp_path, fake_train_muzero)
    monkeypatch.setattr(
        train_mod,
        "_install_lightzero_opponent_assignment_refresh_hook",
        lambda **_kwargs: None,
    )
    assignment_a_path = tmp_path / "assignment-a.json"
    assignment_b_path = tmp_path / "assignment-b.json"
    _write_assignment_payload(assignment_a_path, assignment_id="assignment-a", source_epoch=1)
    _write_assignment_payload(assignment_b_path, assignment_id="assignment-b", source_epoch=2)

    result = train_mod._run_visual_survival_train(
        **_minimal_visual_train_kwargs(
            run_id="refresh-hook-missing-test",
            attempt_id="attempt-001",
            opponent_assignment_ref=str(assignment_a_path),
            opponent_assignment_refresh_interval_train_iter=50,
            opponent_assignment_refresh_ref=str(assignment_b_path),
        )
    )

    assert result["ok"] is False
    assert result["called_train_muzero"] is False
    assert any("refresh hook was not installed" in problem for problem in result["problems"])
    summary = json.loads((tmp_path / result["summary_ref"]).read_text(encoding="utf-8"))
    assert summary["opponent_assignment_refresh"]["enabled"] is True
    assert summary["train_result"]["ok"] is False


def test_live_checkpoint_publisher_calls_original_save_before_spawning(monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    events = []

    class FakeBaseLearner:
        train_iter = 5

        def save_checkpoint(self):
            events.append("original_save")
            return {"saved": True}

    def fake_train_muzero():
        return None

    def fake_spawn_checkpoint_eval_triggers(**kwargs):
        events.append("spawn_eval")
        assert kwargs["seen_checkpoint_refs"] == set()
        return [{"scheduled": True}]

    monkeypatch.setitem(fake_train_muzero.__globals__, "BaseLearner", FakeBaseLearner)
    monkeypatch.setattr(
        train_mod,
        "_spawn_checkpoint_eval_triggers",
        fake_spawn_checkpoint_eval_triggers,
    )

    restore = train_mod._install_live_checkpoint_publisher(
        train_muzero=fake_train_muzero,
        run_id="run-live-publisher",
        attempt_id="attempt-live-publisher",
        exp_name=tmp_path / "lightzero_exp",
        attempt_train_root=tmp_path,
        background_eval_config={"enabled": True},
    )

    try:
        assert restore is not None
        assert FakeBaseLearner().save_checkpoint() == {"saved": True}
    finally:
        restore()

    assert events == ["original_save", "spawn_eval"]


def test_fresh_resume_hooks_preserve_original_call_hook_eval_and_random_collect(monkeypatch):
    events = []

    class FakeCollector:
        pass

    class FakeEvaluator:
        def eval(self, tag=None):
            events.append(("eval", tag))
            return "original-eval-result"

    class FakeBaseLearner:
        def call_hook(self, name):
            events.append(("call_hook", name))
            return f"original-hook-{name}"

    def original_random_collect(*args, **kwargs):
        events.append(("random_collect", args, kwargs))
        return "original-random-collect"

    def fake_train_muzero():
        return None

    monkeypatch.setitem(fake_train_muzero.__globals__, "Collector", FakeCollector)
    monkeypatch.setitem(fake_train_muzero.__globals__, "Evaluator", FakeEvaluator)
    monkeypatch.setitem(fake_train_muzero.__globals__, "BaseLearner", FakeBaseLearner)
    monkeypatch.setitem(fake_train_muzero.__globals__, "random_collect", original_random_collect)

    original_call_hook = FakeBaseLearner.call_hook
    original_eval = FakeEvaluator.eval
    restore = train_mod._install_lightzero_full_resume_state_hooks(
        train_muzero=fake_train_muzero,
        run_id="fresh-resume-hook-passivity",
        attempt_id="attempt-001",
        exp_name=Path("lightzero_exp"),
        auto_resume={},
    )

    try:
        assert restore is not None
        assert FakeBaseLearner().call_hook("before_run") == "original-hook-before_run"
        assert FakeEvaluator().eval(tag="fresh") == "original-eval-result"
        assert fake_train_muzero.__globals__["random_collect"](1, mode="fresh") == (
            "original-random-collect"
        )
    finally:
        restore()

    assert FakeBaseLearner.call_hook is original_call_hook
    assert FakeEvaluator.eval is original_eval
    assert fake_train_muzero.__globals__["random_collect"] is original_random_collect
    assert events == [
        ("call_hook", "before_run"),
        ("eval", "fresh"),
        ("random_collect", (1,), {"mode": "fresh"}),
    ]


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
                    "opponent_immortal": True,
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


def test_opponent_assignment_ref_resolves_to_existing_mixture_contract(tmp_path):
    assignment_path = tmp_path / "assignment.json"
    assignment_path.write_text(
        json.dumps(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": "assignment-smoke",
                "source_epoch": 3,
                "source_ref": "tournaments/curvytron/leaderboards/main/snapshots/003.json",
                "seed": 17,
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                        "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                        "opponent_immortal": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = train_mod._resolve_opponent_assignment_for_env(
        opponent_assignment_ref=str(assignment_path),
    )

    assert resolved is not None
    assert resolved["assignment_id"] == "assignment-smoke"
    assert resolved["source_epoch"] == 3
    assert resolved["assignment_ref"] == str(assignment_path)
    assert resolved["assignment_sha256"]
    mixture = resolved["opponent_mixture"]
    assert mixture["schema_id"] == train_mod.OPPONENT_MIXTURE_SCHEMA_ID
    assert mixture["seed"] == 17
    assert mixture["entries"][0]["opponent_runtime_mode"] == (
        train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
    )


def test_opponent_assignment_context_is_passed_to_env_config(monkeypatch, tmp_path):
    _install_fake_lightzero_atari_config(monkeypatch)
    assignment_path = tmp_path / "assignment.json"
    assignment_path.write_text(
        json.dumps(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": "assignment-smoke",
                "source_epoch": 3,
                "source_ref": "tournaments/curvytron/leaderboards/main/snapshots/003.json",
                "seed": 17,
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                        "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                        "opponent_immortal": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    resolved = train_mod._resolve_opponent_assignment_for_env(
        opponent_assignment_ref=str(assignment_path),
    )
    context = train_mod._opponent_assignment_context_for_env(resolved)

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
        opponent_mixture=resolved["opponent_mixture"],
        opponent_assignment_context=context,
    )

    assert context == {
        "assignment_id": "assignment-smoke",
        "assignment_ref": str(assignment_path),
        "assignment_sha256": resolved["assignment_sha256"],
        "source_epoch": 3,
        "source_ref": "tournaments/curvytron/leaderboards/main/snapshots/003.json",
    }
    assert patched["main_config"]["env"]["opponent_assignment_context"] == context


def test_opponent_assignment_artifact_writer_stores_assignment_and_audit(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    commit_labels = []
    monkeypatch.setattr(
        train_mod,
        "_commit_runs_volume_with_backoff",
        lambda *, label: commit_labels.append(label),
    )
    assignment = {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": "assignment-smoke",
        "source_epoch": 3,
        "source_ref": "tournaments/curvytron/leaderboards/main/snapshots/003.json",
        "seed": 17,
        "entries": [
            {
                "name": "blank",
                "weight": 1,
                "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                "opponent_immortal": True,
            }
        ],
    }
    audit = {
        "schema_id": OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID,
        "assignment_id": "assignment-smoke",
        "assignment_sha256": train_mod.canonical_assignment_json_sha256(assignment),
    }

    result = train_mod._write_opponent_assignment_artifacts(
        run_id="assignment-run",
        attempt_id="attempt-a",
        assignment=assignment,
        audit=audit,
    )

    assert result["schema_id"] == "curvyzero_opponent_assignment_artifact_write/v0"
    assert result["assignment_id"] == "assignment-smoke"
    assert result["assignment_ref"].endswith(
        "assignment-run/attempts/attempt-a/opponents/assignments/"
        "assignment-smoke/assignment.json"
    )
    assert result["audit_ref"].endswith(
        "assignment-run/attempts/attempt-a/opponents/assignments/"
        "assignment-smoke/audit.json"
    )
    assert commit_labels == ["opponent_assignment_artifact_commit"]

    assignment_path = tmp_path / result["assignment_ref"]
    audit_path = tmp_path / result["audit_ref"]
    assert json.loads(assignment_path.read_text(encoding="utf-8")) == assignment
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit


def test_opponent_assignment_artifact_writer_can_store_in_control_volume(
    monkeypatch,
    tmp_path,
):
    runs_mount = tmp_path / "runs"
    control_mount = tmp_path / "control"
    runs_mount.mkdir()
    control_mount.mkdir()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", runs_mount)
    monkeypatch.setattr(train_mod, "CONTROL_MOUNT", control_mount)
    runs_commit_labels = []
    control_commit_labels = []
    monkeypatch.setattr(
        train_mod,
        "_commit_runs_volume_with_backoff",
        lambda *, label: runs_commit_labels.append(label),
    )
    monkeypatch.setattr(
        train_mod,
        "_commit_control_volume_with_backoff",
        lambda *, label: control_commit_labels.append(label),
    )
    assignment = {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": "assignment-control-smoke",
        "source_epoch": 3,
        "source_ref": "tournaments/curvytron/leaderboards/main/snapshots/003.json",
        "seed": 17,
        "entries": [
            {
                "name": "blank",
                "weight": 1,
                "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                "opponent_immortal": True,
            }
        ],
    }

    result = train_mod._write_opponent_assignment_artifacts(
        run_id="assignment-run",
        attempt_id="attempt-a",
        assignment=assignment,
        target_volume="control",
    )

    assert result["schema_id"] == "curvyzero_opponent_assignment_artifact_write/v0"
    assert result["target_volume"] == "control"
    assert result["assignment_ref"].startswith("control:")
    assert result["audit_ref"] is None
    assert runs_commit_labels == []
    assert control_commit_labels == ["opponent_assignment_artifact_commit"]

    assignment_path = control_mount / result["assignment_ref"].removeprefix("control:")
    assert json.loads(assignment_path.read_text(encoding="utf-8")) == assignment


def test_checkpoint_eval_poller_command_resolves_assignment_ref(tmp_path):
    assignment_path = tmp_path / "assignment.json"
    assignment_path.write_text(
        json.dumps(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": "poller-assignment",
                "seed": 19,
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                        "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                        "opponent_immortal": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    command = train_mod._checkpoint_eval_poller_command(
        seed=1,
        source_max_steps=32,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_assignment_ref=str(assignment_path),
        background_eval_enabled=True,
        background_eval_compute=train_mod.DEFAULT_BACKGROUND_EVAL_COMPUTE,
        background_eval_id_prefix=train_mod.DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=None,
        background_eval_max_steps=32,
        background_eval_step_detail_limit=None,
        background_eval_num_simulations=1,
        background_eval_batch_size=1,
        background_gif_enabled=True,
        background_gif_seed_offset=10,
        background_gif_max_steps=32,
        background_gif_frame_stride=1,
        background_gif_fps=8.0,
        background_gif_scale=1,
    )

    assert command["opponent_assignment"]["assignment_id"] == "poller-assignment"
    assert command["opponent_mixture"]["seed"] == 19
    assert command["opponent_mixture"]["entries"][0]["opponent_runtime_mode"] == (
        train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
    )


def test_checkpoint_eval_poller_command_reloads_control_volume_for_assignment_ref(
    monkeypatch,
    tmp_path,
):
    control_mount = tmp_path / "control"
    control_mount.mkdir()
    assignment_path = control_mount / "assignment.json"
    assignment_path.write_text(
        json.dumps(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": "poller-control-assignment",
                "seed": 29,
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                        "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                        "opponent_immortal": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeControlVolume:
        def __init__(self) -> None:
            self.reload_count = 0

        def reload(self) -> None:
            self.reload_count += 1

    fake_control = FakeControlVolume()
    monkeypatch.setattr(train_mod, "CONTROL_MOUNT", control_mount)
    monkeypatch.setattr(train_mod, "control_volume", fake_control)

    command = train_mod._checkpoint_eval_poller_command(
        seed=1,
        source_max_steps=32,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_assignment_ref="control:assignment.json",
        background_eval_enabled=True,
        background_eval_compute=train_mod.DEFAULT_BACKGROUND_EVAL_COMPUTE,
        background_eval_id_prefix=train_mod.DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=None,
        background_eval_max_steps=32,
        background_eval_step_detail_limit=None,
        background_eval_num_simulations=1,
        background_eval_batch_size=1,
        background_gif_enabled=False,
        background_gif_seed_offset=10,
        background_gif_max_steps=32,
        background_gif_frame_stride=1,
        background_gif_fps=8.0,
        background_gif_scale=1,
    )

    assert fake_control.reload_count == 1
    assert command["opponent_assignment"]["assignment_id"] == "poller-control-assignment"
    assert command["opponent_mixture"]["entries"][0]["opponent_immortal"] is True


def test_checkpoint_eval_poller_command_reloads_runs_for_assignment_checkpoint_ref(
    monkeypatch,
    tmp_path,
):
    control_mount = tmp_path / "control"
    runs_mount = tmp_path / "runs"
    control_mount.mkdir()
    checkpoint_ref = (
        "training/lightzero-curvytron-visual-survival/run-a/attempts/try-a/"
        "train/lightzero_exp/ckpt/iteration_0.pth.tar"
    )
    checkpoint_path = runs_mount / checkpoint_ref
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    assignment_path = control_mount / "assignment.json"
    assignment_path.write_text(
        json.dumps(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": "poller-control-frozen-assignment",
                "seed": 31,
                "entries": [
                    {
                        "name": "frozen",
                        "weight": 1,
                        "opponent_policy_kind": (
                            train_mod.OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
                        ),
                        "opponent_checkpoint_ref": checkpoint_ref,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeVolume:
        def __init__(self) -> None:
            self.reload_count = 0

        def reload(self) -> None:
            self.reload_count += 1

    fake_control = FakeVolume()
    fake_runs = FakeVolume()
    monkeypatch.setattr(train_mod, "CONTROL_MOUNT", control_mount)
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", runs_mount)
    monkeypatch.setattr(train_mod, "control_volume", fake_control)
    monkeypatch.setattr(train_mod, "runs_volume", fake_runs)

    command = train_mod._checkpoint_eval_poller_command(
        seed=1,
        source_max_steps=32,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_assignment_ref="control:assignment.json",
        background_eval_enabled=True,
        background_eval_compute=train_mod.DEFAULT_BACKGROUND_EVAL_COMPUTE,
        background_eval_id_prefix=train_mod.DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=None,
        background_eval_max_steps=32,
        background_eval_step_detail_limit=None,
        background_eval_num_simulations=1,
        background_eval_batch_size=1,
        background_gif_enabled=False,
        background_gif_seed_offset=10,
        background_gif_max_steps=32,
        background_gif_frame_stride=1,
        background_gif_fps=8.0,
        background_gif_scale=1,
    )

    assert fake_control.reload_count == 1
    assert fake_runs.reload_count == 1
    assert command["opponent_assignment"]["assignment_id"] == "poller-control-frozen-assignment"
    assert command["opponent_mixture"]["entries"][0]["opponent_checkpoint_path"] == str(
        checkpoint_path
    )


def test_checkpoint_eval_poller_function_accepts_assignment_ref_and_resolves_command(
    monkeypatch,
    tmp_path,
):
    captured = {}
    assignment_path = tmp_path / "assignment.json"
    assignment_path.write_text(
        json.dumps(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": "poller-local-assignment",
                "seed": 23,
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                        "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                        "opponent_immortal": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_checkpoint_eval_poller(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "command": kwargs["command"]}

    monkeypatch.setattr(
        train_mod,
        "_run_checkpoint_eval_poller",
        fake_run_checkpoint_eval_poller,
    )

    result = train_mod.lightzero_curvytron_visual_survival_checkpoint_eval_poller.local(
        run_id="poller-run",
        attempt_id="poller-attempt",
        exp_name_ref="training/lightzero-curvytron-visual-survival/poller-run/"
        "attempts/poller-attempt/train/lightzero_exp",
        seed=1,
        source_max_steps=32,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_assignment_ref=str(assignment_path),
        background_eval_max_steps=32,
        background_eval_num_simulations=1,
        background_eval_batch_size=1,
        background_gif_enabled=False,
    )

    command = result["command"]
    assert captured["run_id"] == "poller-run"
    assert command["opponent_assignment"]["assignment_id"] == "poller-local-assignment"
    assert command["opponent_mixture"]["seed"] == 23
    assert command["opponent_mixture"]["entries"][0]["opponent_runtime_mode"] == (
        train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
    )


def test_checkpoint_eval_poller_command_rejects_assignment_and_inline_mixture(tmp_path):
    assignment_path = tmp_path / "assignment.json"
    assignment_path.write_text(
        json.dumps(
            {
                "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
                "assignment_id": "poller-assignment",
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        train_mod._checkpoint_eval_poller_command(
            seed=1,
            source_max_steps=32,
            env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
            opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
            opponent_checkpoint_ref=None,
            opponent_snapshot_ref=None,
            opponent_checkpoint_state_key=None,
            opponent_mixture_spec={"entries": [{"name": "blank", "weight": 1, "opponent_policy_kind": "fixed_straight"}]},
            opponent_assignment_ref=str(assignment_path),
            background_eval_enabled=True,
            background_eval_compute=train_mod.DEFAULT_BACKGROUND_EVAL_COMPUTE,
            background_eval_id_prefix=train_mod.DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
            background_eval_seed_count=1,
            background_eval_seed_rng_seed=None,
            background_eval_max_steps=32,
            background_eval_step_detail_limit=None,
            background_eval_num_simulations=1,
            background_eval_batch_size=1,
            background_gif_enabled=False,
            background_gif_seed_offset=10,
            background_gif_max_steps=32,
            background_gif_frame_stride=1,
            background_gif_fps=8.0,
            background_gif_scale=1,
        )


def test_stock_source_state_mixture_config_instantiates_registered_env_and_steps_scalar_action(
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
                    "opponent_immortal": True,
                }
            ]
        }
    )

    patched = train_mod._build_visual_survival_configs(
        seed=19,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=False,
        max_env_step=128,
        source_max_steps=32,
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
        env_telemetry_stride=1,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_mixture=mixture,
    )

    env = CurvyZeroSourceStateVisualSurvivalLightZeroEnv(patched["main_config"]["env"])
    try:
        observation = env.reset(seed=19)
        timestep = env.step(1)
    finally:
        env.close()

    assert patched["create_config"]["env"]["type"] == (
        LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE
    )
    assert observation["observation"].shape == (4, 64, 64)
    np.testing.assert_array_equal(observation["action_mask"], np.array([1, 1, 1]))
    assert timestep.info["requested_ego_action"] == 1
    assert timestep.info["executed_ego_action"] == 1
    assert timestep.info["joint_action"]["player_0"] == 1
    assert timestep.info["opponent_action_id"] == 1
    assert timestep.info["opponent_mixture_enabled"] is True
    assert timestep.info["opponent_mixture_entry_name"] == "blank"
    assert timestep.info["opponent_runtime_mode"] == "blank_canvas_noop"
    assert timestep.info["blank_canvas_noop"] is True
    assert timestep.info["opponent_policy_sidecar"] == {
        "opponent_runtime_mode": train_mod.OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
        "action_ignored": True,
    }


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
    assert "td_steps" not in target_config
    assert patched["main_config"]["policy"]["td_steps"] == 5
    assert patched["surface"]["model_reward_support_size"] == 601
    assert patched["surface"]["model_value_support_size"] == 601


def test_survival_plus_bonus_plus_outcome_uses_scaled_terminal_supports(
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
        reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
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

    assert train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME in (
        train_mod.REWARD_VARIANT_CHOICES
    )
    assert env_cfg["reward_variant"] == (
        train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME
    )
    assert reward_policy["dense_survival_reward"] is True
    assert reward_policy["same_step_bonus_pickup_reward"] is True
    assert reward_policy["sparse_outcome_reward"] is True
    assert reward_policy["sparse_outcome_telemetry_only"] is False
    assert (
        reward_policy["terminal_outcome_scaled_by_episode_source_steps"]
        is True
    )
    assert reward_policy["terminal_outcome_scale"] == "episode_source_step_count"
    assert target_config["uncapped_model_reward_support_scale"] == 65_538
    assert target_config["uncapped_model_value_support_scale"] == 196_608
    assert target_config["model_reward_support_capped"] is True
    assert target_config["model_value_support_capped"] is True
    assert target_config["model_support_cap"] == 300
    assert model_cfg["support_scale"] == 300
    assert model_cfg["reward_support_size"] == 601
    assert model_cfg["value_support_size"] == 601
    assert "td_steps" not in target_config
    assert patched["main_config"]["policy"]["td_steps"] == 5


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


def test_modal_config_defaults_to_one_source_frame_per_policy_action(monkeypatch, tmp_path):
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
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        policy_action_repeat_min=train_mod.DEFAULT_POLICY_ACTION_REPEAT_MIN,
        policy_action_repeat_max=train_mod.DEFAULT_POLICY_ACTION_REPEAT_MAX,
        policy_action_repeat_extra_probability=(
            train_mod.DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
        ),
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
    assert train_mod.DEFAULT_DECISION_MS == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert env_cfg["decision_ms"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert env_cfg["decision_source_frames"] == 1
    assert env_cfg["source_physics_step_ms"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert env_cfg["max_ticks"] == env_cfg["source_max_steps"]
    assert env_cfg["source_max_steps_semantics"] == "source_physics_steps"
    assert env_cfg["policy_action_repeat_min"] == 1
    assert env_cfg["policy_action_repeat_max"] == 1
    assert env_cfg["policy_action_repeat_extra_probability"] == 0.0
    assert patched["surface"]["decision_source_frames"] == 1
    assert patched["surface"]["source_physics_step_ms"] == pytest.approx(
        SOURCE_PHYSICS_STEP_MS
    )


def test_modal_config_can_pass_explicit_cadence_to_non_trusted_eval_surface(
    monkeypatch,
    tmp_path,
):
    _install_fake_lightzero_atari_config(monkeypatch)

    patched = train_mod._build_visual_survival_configs(
        seed=13,
        exp_name=tmp_path / "exp",
        telemetry_path=tmp_path / "env_steps.jsonl",
        cuda=False,
        max_env_step=128,
        source_max_steps=128,
        decision_ms=10.0,
        decision_source_frames=2,
        source_physics_step_ms=5.0,
        source_max_steps_semantics="source_physics_steps",
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
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
        reward_variant=train_mod.DEFAULT_REWARD_VARIANT,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        policy_action_repeat_min=train_mod.DEFAULT_POLICY_ACTION_REPEAT_MIN,
        policy_action_repeat_max=train_mod.DEFAULT_POLICY_ACTION_REPEAT_MAX,
        policy_action_repeat_extra_probability=(
            train_mod.DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
        ),
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
    assert env_cfg["decision_ms"] == 10.0
    assert env_cfg["decision_source_frames"] == 2
    assert env_cfg["source_physics_step_ms"] == 5.0
    assert env_cfg["source_max_steps_semantics"] == "source_physics_steps"
    assert patched["surface"]["decision_ms"] == 10.0
    assert patched["surface"]["decision_source_frames"] == 2
    assert patched["surface"]["source_physics_step_ms"] == 5.0


def test_stock_source_state_train_rejects_bundled_decision_ms(monkeypatch, tmp_path):
    _install_fake_lightzero_atari_config(monkeypatch)

    with pytest.raises(ValueError, match="one CurvyTron source physics step"):
        train_mod._build_visual_survival_configs(
            seed=13,
            exp_name=tmp_path / "exp",
            telemetry_path=tmp_path / "env_steps.jsonl",
            cuda=False,
            max_env_step=128,
            source_max_steps=128,
            decision_ms=300.0,
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
            policy_action_repeat_min=train_mod.DEFAULT_POLICY_ACTION_REPEAT_MIN,
            policy_action_repeat_max=train_mod.DEFAULT_POLICY_ACTION_REPEAT_MAX,
            policy_action_repeat_extra_probability=(
                train_mod.DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
            ),
            disable_death_for_profile=False,
            env_telemetry_stride=64,
            env_manager_type="base",
            opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
            opponent_use_cuda=False,
            opponent_checkpoint=None,
            opponent_snapshot_ref=None,
            opponent_checkpoint_state_key=None,
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
        source_state_trail_render_mode=train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
        source_state_bonus_render_mode="simple_symbols",
    )

    env_cfg = patched["main_config"]["env"]
    assert (
        env_cfg["source_state_trail_render_mode"]
        == train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE
    )
    assert (
        patched["surface"]["source_state_trail_render_mode"]
        == train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE
    )
    assert env_cfg["source_state_bonus_render_mode"] == "simple_symbols"
    assert patched["surface"]["source_state_bonus_render_mode"] == "simple_symbols"
    assert env_cfg["default_trail_render_mode"] == (
        train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE
    )
    assert env_cfg["supported_trail_render_modes"] == list(
        train_mod.SOURCE_STATE_TRAIL_RENDER_MODE_CHOICES
    )
    assert (
        env_cfg["default_bonus_render_mode"]
        == train_mod.DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE
    )
    assert env_cfg["supported_bonus_render_modes"] == list(
        train_mod.SOURCE_STATE_BONUS_RENDER_MODE_CHOICES
    )


def test_stock_source_state_trail_render_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="source_state_trail_render_mode"):
        train_mod._validate_source_state_trail_render_mode("mystery")


def test_stock_source_state_bonus_render_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="source_state_bonus_render_mode"):
        train_mod._validate_source_state_bonus_render_mode("mystery")


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
        source_state_trail_render_mode=train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
        source_state_bonus_render_mode="simple_symbols",
    )

    payload = fake_train.kwargs[0]
    printed = json.loads(capsys.readouterr().out)

    assert payload["source_state_trail_render_mode"] == (
        train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE
    )
    assert printed["command"]["source_state_trail_render_mode"] == (
        train_mod.DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE
    )
    assert payload["source_state_bonus_render_mode"] == "simple_symbols"
    assert printed["command"]["source_state_bonus_render_mode"] == "simple_symbols"
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
    assert config["decision_ms"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert config["decision_source_frames"] == 1
    assert config["source_physics_step_ms"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert config["source_max_steps_semantics"] == "source_physics_steps"
    assert config["selfplay_gif"]["enabled"] is True
    assert config["selfplay_gif"]["natural_bonus_spawn"] is True
    assert config["selfplay_gif"]["decision_ms"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert config["selfplay_gif"]["decision_source_frames"] == 1
    assert config["selfplay_gif"]["source_physics_step_ms"] == pytest.approx(
        SOURCE_PHYSICS_STEP_MS
    )
    assert config["selfplay_gif"]["source_max_steps_semantics"] == "source_physics_steps"
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
                    "opponent_immortal": True,
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
        "decision_ms": 10.0,
        "decision_source_frames": 2,
        "source_physics_step_ms": 5.0,
        "source_max_steps_semantics": "source_physics_steps",
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
            "decision_ms": 10.0,
            "decision_source_frames": 2,
            "source_physics_step_ms": 5.0,
            "source_max_steps_semantics": "source_physics_steps",
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
    assert eval_call["decision_ms"] == 10.0
    assert eval_call["decision_source_frames"] == 2
    assert eval_call["source_physics_step_ms"] == 5.0
    assert eval_call["source_max_steps_semantics"] == "source_physics_steps"
    assert eval_call["natural_bonus_spawn"] is False
    assert gif_call["seed"] != 10_007
    assert gif_call["max_steps"] == 16
    assert gif_call["decision_ms"] == 10.0
    assert gif_call["decision_source_frames"] == 2
    assert gif_call["source_physics_step_ms"] == 5.0
    assert gif_call["source_max_steps_semantics"] == "source_physics_steps"
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
    assert request["selfplay_gif"]["config"]["decision_ms"] == 10.0
    assert request["selfplay_gif"]["config"]["decision_source_frames"] == 2
    assert request["selfplay_gif"]["config"]["source_physics_step_ms"] == 5.0
    assert (
        request["selfplay_gif"]["config"]["source_max_steps_semantics"]
        == "source_physics_steps"
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
    assert request["config"]["decision_ms"] == 10.0
    assert request["config"]["decision_source_frames"] == 2
    assert request["config"]["source_physics_step_ms"] == 5.0
    assert request["config"]["source_max_steps_semantics"] == "source_physics_steps"
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
        fps=train_mod.DEFAULT_BACKGROUND_GIF_FPS,
        scale=4,
    )

    assert train_mod.DEFAULT_BACKGROUND_GIF_FPS == 80.0
    assert artifact["duration_ms_per_frame"] == 12
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
        decision_ms=10.0,
        decision_source_frames=2,
        source_physics_step_ms=5.0,
        source_max_steps_semantics="source_physics_steps",
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
    assert eval_call["decision_ms"] == 10.0
    assert eval_call["decision_source_frames"] == 2
    assert eval_call["source_physics_step_ms"] == 5.0
    assert eval_call["source_max_steps_semantics"] == "source_physics_steps"
    assert gif_call["decision_ms"] == 10.0
    assert gif_call["decision_source_frames"] == 2
    assert gif_call["source_physics_step_ms"] == 5.0
    assert gif_call["source_max_steps_semantics"] == "source_physics_steps"
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
        opponent_assignment_ref=(
            "training/lightzero-curvytron-visual-survival/run-d/"
            "attempts/attempt-d/opponents/assignments/a/assignment.json"
        ),
        opponent_assignment_refresh_interval_train_iter=25,
        opponent_assignment_refresh_ref=(
            "training/lightzero-curvytron-visual-survival/run-d/"
            "attempts/attempt-d/opponents/assignments/b/assignment.json"
        ),
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
    assert fake_poller.calls[0]["decision_ms"] == train_mod.DEFAULT_DECISION_MS
    assert (
        fake_poller.calls[0]["decision_source_frames"]
        == train_mod.DEFAULT_DECISION_SOURCE_FRAMES
    )
    assert fake_poller.calls[0]["source_physics_step_ms"] == pytest.approx(
        train_mod.DEFAULT_SOURCE_PHYSICS_STEP_MS
    )
    assert fake_poller.calls[0]["source_max_steps_semantics"] == "source_physics_steps"
    assert fake_poller.calls[0]["opponent_policy_kind"] == train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT
    assert fake_train.calls[0]["opponent_assignment_ref"].endswith(
        "opponents/assignments/a/assignment.json"
    )
    assert fake_train.calls[0]["opponent_assignment_refresh_interval_train_iter"] == 25
    assert fake_train.calls[0]["opponent_assignment_refresh_ref"].endswith(
        "opponents/assignments/b/assignment.json"
    )
    assert fake_poller.calls[0]["opponent_assignment_ref"].endswith(
        "opponents/assignments/a/assignment.json"
    )
    assert "opponent_assignment_refresh_ref" not in fake_poller.calls[0]
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
        two_seat_trail_render_mode=train_mod.TRAIL_RENDER_MODE_DEFAULT,
    )

    payload = fake_train.payloads[0]
    printed = json.loads(capsys.readouterr().out)

    assert payload["trail_render_mode"] == train_mod.TRAIL_RENDER_MODE_DEFAULT
    assert payload["death_mode"] == "profile_no_death"
    assert printed["command"]["trail_render_mode"] == train_mod.TRAIL_RENDER_MODE_DEFAULT
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
