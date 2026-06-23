import json

import numpy as np
import pytest

from curvyzero.training import exploration_bonus as xb


def test_exploration_bonus_spec_default_is_disabled_and_stable():
    spec = xb.normalize_exploration_bonus_spec()

    assert spec.mode == "none"
    assert spec.enabled is False
    assert spec.training_effect == "disabled"
    assert spec.as_dict()["config_hash"] == spec.config_hash()
    assert spec.config_hash() == xb.normalize_exploration_bonus_spec().config_hash()


def test_exploration_bonus_spec_fails_closed():
    with pytest.raises(ValueError, match="unknown exploration_bonus_mode"):
        xb.normalize_exploration_bonus_spec(mode="rnd_surprise")
    with pytest.raises(ValueError, match="weight must be 0.0"):
        xb.normalize_exploration_bonus_spec(mode="none", weight=0.1)
    with pytest.raises(ValueError, match="metric-only"):
        xb.normalize_exploration_bonus_spec(mode="rnd_meter_v0", weight=0.1)
    with pytest.raises(ValueError, match="requires exploration_bonus_weight > 0.0"):
        xb.normalize_exploration_bonus_spec(mode="rnd_replay_target_v0", weight=0.0)
    with pytest.raises(ValueError, match="must be <= 1.0"):
        xb.normalize_exploration_bonus_spec(mode="rnd_replay_target_v0", weight=1.1)
    with pytest.raises(ValueError, match="feature_source"):
        xb.normalize_exploration_bonus_spec(
            mode="rnd_meter_v0",
            feature_source="policy_gray64_stack4/v0",
        )


def test_rnd_meter_spec_selects_reward_model_entrypoint_and_metadata():
    spec = xb.normalize_exploration_bonus_spec(mode="rnd_meter_v0")
    payload = spec.as_dict()

    round_trip = xb.normalize_exploration_bonus_config(payload)

    assert spec.enabled is True
    assert round_trip == spec
    assert xb.lightzero_entrypoint_name(spec) == "train_muzero_with_reward_model"
    assert xb.lightzero_trainer_entrypoint_ref(spec) == (
        "lzero.entry.train_muzero_with_reward_model"
    )
    assert payload["training_effect"] == "reward_target_unchanged"
    assert payload["target_reward_effect"] == "unchanged"
    assert payload["trainer_effect"] == "uses_reward_model_entrypoint_and_trains_rnd_meter"
    assert payload["input_spec"]["shape"] == [1, 64, 64]
    assert payload["input_spec"]["source_observation_shape"] == [4, 64, 64]


def test_rnd_replay_target_spec_is_objective_changing_and_bounded():
    spec = xb.normalize_exploration_bonus_spec(
        mode="rnd_replay_target_v0",
        weight=0.125,
        rnd_batch_size=8,
    )
    payload = spec.as_dict()

    assert spec.enabled is True
    assert spec.training_effect == "reward_target_augmented_by_intrinsic_rnd"
    assert spec.target_reward_effect == "intrinsic_weighted_addition"
    assert spec.trainer_effect == "uses_reward_model_entrypoint_and_trains_rnd_replay_target"
    assert payload["schema_id"] == "curvyzero_exploration_bonus/rnd_replay_target_v0/v0"
    assert payload["weight"] == 0.125
    assert payload["rnd_batch_size"] == 8
    assert xb.lightzero_reward_model_config(spec)["intrinsic_reward_weight"] == 0.125
    assert xb.normalize_exploration_bonus_config(payload) == spec


def test_exploration_bonus_config_rejects_stale_hash():
    payload = xb.normalize_exploration_bonus_spec(mode="rnd_meter_v0").as_dict()
    payload["rnd_batch_size"] = 32

    with pytest.raises(ValueError, match="config_hash"):
        xb.normalize_exploration_bonus_config(payload)


def test_exploration_bonus_config_rejects_unknown_fields():
    payload = xb.normalize_exploration_bonus_spec(mode="rnd_meter_v0").as_dict()
    payload.pop("config_hash")
    payload["surprise_training_knob"] = True

    with pytest.raises(ValueError, match="unknown.*surprise_training_knob"):
        xb.normalize_exploration_bonus_config(payload)


def test_exploration_bonus_config_rejects_stale_metadata_and_loose_scalars():
    payload = xb.normalize_exploration_bonus_spec(mode="rnd_meter_v0").as_dict()
    stale_schema = dict(payload)
    stale_schema["schema_id"] = "curvyzero_exploration_bonus/none/v0"
    with pytest.raises(ValueError, match="schema_id"):
        xb.normalize_exploration_bonus_config(stale_schema)

    stale_effect = dict(payload)
    stale_effect["training_effect"] = "disabled"
    with pytest.raises(ValueError, match="training_effect"):
        xb.normalize_exploration_bonus_config(stale_effect)

    stale_target_effect = dict(payload)
    stale_target_effect["target_reward_effect"] = "intrinsic_weighted_addition"
    with pytest.raises(ValueError, match="target_reward_effect"):
        xb.normalize_exploration_bonus_config(stale_target_effect)

    stale_trainer_effect = dict(payload)
    stale_trainer_effect["trainer_effect"] = "uses_stock_muzero_entrypoint"
    with pytest.raises(ValueError, match="trainer_effect"):
        xb.normalize_exploration_bonus_config(stale_trainer_effect)

    stale_training_only = dict(payload)
    stale_training_only["training_only"] = False
    with pytest.raises(ValueError, match="training_only"):
        xb.normalize_exploration_bonus_config(stale_training_only)

    with pytest.raises(ValueError, match="integer-valued"):
        xb.normalize_exploration_bonus_spec(mode="rnd_meter_v0", rnd_batch_size=1.9)
    with pytest.raises(ValueError, match="strict bool"):
        xb.normalize_exploration_bonus_spec(mode="rnd_meter_v0", rnd_input_norm="maybe")


def test_lightzero_rnd_meter_patch_is_atomic_and_weight_zero():
    spec = xb.normalize_exploration_bonus_spec(mode="rnd_meter_v0")
    main_config = {"env": {}, "policy": {"model": {"observation_shape": [4, 64, 64]}}}
    create_config = {"policy": {"type": "muzero"}}

    patches = xb.apply_lightzero_exploration_bonus_config(main_config, create_config, spec)

    assert [patch["path"] for patch in patches] == [
        "reward_model",
        "policy.use_rnd_model",
        "policy.use_momentum_representation_network",
        "policy.target_model_for_intrinsic_reward_update_type",
        "policy.target_update_freq_for_intrinsic_reward",
        "policy.target_update_theta_for_intrinsic_reward",
        "env.exploration_bonus",
    ]
    assert main_config["policy"]["use_rnd_model"] is True
    assert main_config["reward_model"]["type"] == "rnd_muzero"
    assert main_config["reward_model"]["intrinsic_reward_weight"] == 0.0
    assert main_config["reward_model"]["obs_shape"] == [1, 64, 64]
    assert main_config["reward_model"]["curvyzero_adapter"]["feature_source"] == (
        "policy_gray64_latest/v0"
    )
    assert main_config["env"]["exploration_bonus"]["training_effect"] == ("reward_target_unchanged")
    assert create_config == {"policy": {"type": "muzero"}}


def test_lightzero_rnd_replay_target_patch_marks_target_reward_change():
    spec = xb.normalize_exploration_bonus_spec(mode="rnd_replay_target_v0", weight=0.25)
    main_config = {"env": {}, "policy": {"model": {"observation_shape": [4, 64, 64]}}}

    patches = xb.apply_lightzero_exploration_bonus_config(main_config, {}, spec)

    assert [patch["path"] for patch in patches] == [
        "reward_model",
        "policy.use_rnd_model",
        "policy.use_momentum_representation_network",
        "policy.target_model_for_intrinsic_reward_update_type",
        "policy.target_update_freq_for_intrinsic_reward",
        "policy.target_update_theta_for_intrinsic_reward",
        "env.exploration_bonus",
    ]
    assert main_config["reward_model"]["intrinsic_reward_weight"] == 0.25
    assert main_config["env"]["exploration_bonus"]["training_effect"] == (
        "reward_target_augmented_by_intrinsic_rnd"
    )
    assert main_config["env"]["exploration_bonus"]["target_reward_effect"] == (
        "intrinsic_weighted_addition"
    )


def test_disabled_exploration_bonus_does_not_patch_lightzero_config():
    spec = xb.normalize_exploration_bonus_spec(mode="none")
    main_config = {"env": {}, "policy": {}}
    create_config = {"policy": {"type": "muzero"}}

    assert xb.apply_lightzero_exploration_bonus_config(main_config, create_config, spec) == []
    assert main_config == {"env": {}, "policy": {}}


def test_latest_gray64_adapter_extracts_from_batched_unroll():
    batch_size = 2
    unroll = 3
    obs = np.zeros((batch_size, unroll, 4, 64, 64), dtype=np.float32)
    for batch_idx in range(batch_size):
        for step_idx in range(unroll):
            obs[batch_idx, step_idx, 3, :, :] = (batch_idx * unroll + step_idx) / (
                batch_size * unroll
            )
    target_reward = np.zeros((batch_size, unroll, 1), dtype=np.float32)

    rnd_input = xb.extract_policy_gray64_latest_for_rnd(obs, target_reward)

    assert rnd_input.shape == (batch_size * unroll, 1, 64, 64)
    assert rnd_input.dtype == np.float32
    assert float(rnd_input[0, 0, 0, 0]) == 0.0
    assert float(rnd_input[-1, 0, 0, 0]) == pytest.approx(5 / 6)


@pytest.mark.parametrize(
    "obs_shape",
    [
        (1, 2, 3, 64, 64),
        (1, 2, 4, 63, 64),
        (1, 2, 4, 64, 63),
    ],
)
def test_latest_gray64_adapter_rejects_5d_obs_with_wrong_chw(obs_shape):
    obs = np.zeros(obs_shape, dtype=np.float32)
    target_reward = np.zeros((1, 2, 1), dtype=np.float32)

    with pytest.raises(ValueError):
        xb.extract_policy_gray64_latest_for_rnd(obs, target_reward)


def test_latest_gray64_adapter_extracts_from_flat_replay_batch():
    batch_size = 2
    unroll = 3
    obs = np.zeros((batch_size, unroll, 4, 64, 64), dtype=np.float32)
    obs[:, :, 3, :, :] = 0.75
    flat_obs = obs.reshape(batch_size, -1)
    target_reward = np.zeros((batch_size, unroll, 1), dtype=np.float32)

    rnd_input = xb.extract_policy_gray64_latest_for_rnd(flat_obs, target_reward)

    assert rnd_input.shape == (batch_size * unroll, 1, 64, 64)
    assert np.allclose(rnd_input, 0.75)


def test_latest_gray64_adapter_extracts_from_lightzero_channel_unroll_batch():
    batch_size = 1
    unroll = 6
    obs = np.zeros((batch_size, unroll, 4, 64, 64), dtype=np.float32)
    for step_idx in range(unroll):
        obs[0, step_idx, 3, :, :] = step_idx / unroll
    lightzero_obs = obs.reshape(batch_size, unroll * 4, 64, 64)
    target_reward = np.zeros((batch_size, unroll), dtype=np.float32)

    rnd_input = xb.extract_policy_gray64_latest_for_rnd(lightzero_obs, target_reward)

    assert rnd_input.shape == (batch_size * unroll, 1, 64, 64)
    assert float(rnd_input[0, 0, 0, 0]) == 0.0
    assert float(rnd_input[-1, 0, 0, 0]) == pytest.approx(5 / 6)


def test_latest_gray64_adapter_rejects_unnormalized_values():
    obs = np.zeros((1, 1, 4, 64, 64), dtype=np.float32)
    obs[:, :, 3, :, :] = 2.0
    target_reward = np.zeros((1, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="normalized"):
        xb.extract_policy_gray64_latest_for_rnd(obs, target_reward)


def test_compact_policy_gray64_adapter_normalizes_uint8_row_player_stacks():
    obs = np.zeros((2, 2, 4, 64, 64), dtype=np.uint8)
    obs[0, 0, 3, :, :] = 0
    obs[0, 1, 3, :, :] = 64
    obs[1, 0, 3, :, :] = 128
    obs[1, 1, 3, :, :] = 255
    target_reward = np.zeros((4, 1), dtype=np.float32)

    stack = xb.normalize_policy_gray64_stack_for_rnd(obs)
    rnd_input = xb.extract_policy_gray64_latest_for_rnd_from_compact_observation(
        obs,
        target_reward,
    )

    assert stack.shape == (4, 4, 64, 64)
    assert stack.dtype == np.float32
    assert stack.flags.c_contiguous
    assert rnd_input.shape == (4, 1, 64, 64)
    assert float(rnd_input[0, 0, 0, 0]) == 0.0
    assert float(rnd_input[1, 0, 0, 0]) == pytest.approx(64 / 255)
    assert float(rnd_input[2, 0, 0, 0]) == pytest.approx(128 / 255)
    assert float(rnd_input[3, 0, 0, 0]) == 1.0


def test_compact_policy_gray64_latest_adapter_ignores_stale_stack_channels():
    obs = np.zeros((2, 2, 4, 64, 64), dtype=np.float32)
    obs[:, :, 0, :, :] = 99.0
    obs[:, :, 1, :, :] = -10.0
    obs[:, :, 2, :, :] = np.nan
    obs[:, :, 3, :, :] = 0.5
    target_reward = np.zeros((4, 1), dtype=np.float32)

    rnd_input = xb.extract_policy_gray64_latest_for_rnd_from_compact_observation(
        obs,
        target_reward,
    )

    assert rnd_input.shape == (4, 1, 64, 64)
    assert np.allclose(rnd_input, 0.5)


def test_compact_policy_gray64_latest_adapter_accepts_flat_root_stacks():
    obs = np.zeros((3, 4, 64, 64), dtype=np.uint8)
    obs[0, 3, :, :] = 0
    obs[1, 3, :, :] = 127
    obs[2, 3, :, :] = 255
    target_reward = np.zeros((3, 1), dtype=np.float32)

    rnd_input = xb.extract_policy_gray64_latest_for_rnd_from_compact_observation(
        obs,
        target_reward,
    )

    assert rnd_input.shape == (3, 1, 64, 64)
    assert float(rnd_input[0, 0, 0, 0]) == 0.0
    assert float(rnd_input[1, 0, 0, 0]) == pytest.approx(127 / 255)
    assert float(rnd_input[2, 0, 0, 0]) == 1.0


def test_compact_policy_gray64_adapter_rejects_bad_target_shape_and_float_range():
    obs = np.zeros((2, 2, 4, 64, 64), dtype=np.float32)
    obs[0, 0, 3, :, :] = 2.0

    with pytest.raises(ValueError, match="normalized"):
        xb.normalize_policy_gray64_stack_for_rnd(obs)

    obs.fill(0.0)
    with pytest.raises(ValueError, match="target_reward must have shape"):
        xb.extract_policy_gray64_latest_for_rnd_from_compact_observation(
            obs,
            np.zeros((2, 2, 1), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="target_reward must have shape"):
        xb.extract_policy_gray64_latest_for_rnd_from_compact_observation(
            obs,
            np.zeros((4, 2), dtype=np.float32),
        )


def test_curvy_rnd_reward_model_trains_predictor_freezes_target_and_preserves_zero_weight_reward(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    config = {
        "input_type": "obs",
        "intrinsic_reward_type": "add",
        "intrinsic_reward_weight": 0.0,
        "curvyzero_adapter": {
            "shape": [1, 64, 64],
            "source_observation_shape": [4, 64, 64],
        },
        "hidden_size_list": [8],
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "batch_size": 4,
        "update_per_collect": 3,
        "rnd_buffer_size": 16,
        "curvyzero_metrics_latest_path": str(tmp_path / "rnd_latest.json"),
        "curvyzero_metrics_jsonl_path": str(tmp_path / "rnd_metrics.jsonl"),
    }
    model = xb.CurvyRNDRewardModel(config, device="cpu")

    predictor_before = {
        name: param.detach().clone()
        for name, param in model.reward_model.predictor.state_dict().items()
    }
    target_before = {
        name: param.detach().clone()
        for name, param in model.reward_model.target.state_dict().items()
    }

    class Segment:
        pass

    segment = Segment()
    segment.obs_segment = np.zeros((8, 4, 64, 64), dtype=np.float32)
    for idx in range(8):
        segment.obs_segment[idx, 3, :, :] = idx / 7.0
    model.collect_data(([segment], None))
    model.train_with_data()

    assert model.train_cnt_rnd == 3
    assert any(
        not torch.equal(param, predictor_before[name])
        for name, param in model.reward_model.predictor.state_dict().items()
    )
    assert all(
        torch.equal(param, target_before[name])
        for name, param in model.reward_model.target.state_dict().items()
    )
    assert all(param.requires_grad is False for param in model.reward_model.target.parameters())

    obs_batch = np.zeros((2, 3, 4, 64, 64), dtype=np.float32)
    obs_batch[:, :, 3, :, :] = np.linspace(0.0, 1.0, num=6, dtype=np.float32).reshape(2, 3, 1, 1)
    target_reward = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    output = model.estimate([[obs_batch], [target_reward]])

    assert np.array_equal(output[1][0], target_reward)
    assert output[1][0] is target_reward
    assert model.last_target_reward_changed is False
    assert model.last_target_reward_delta_abs_mean == 0.0
    assert model.last_target_reward_delta_abs_max == 0.0

    positive_config = dict(config)
    positive_config["intrinsic_reward_weight"] = 0.25
    positive_model = xb.CurvyRNDRewardModel(positive_config, device="cpu")
    positive_model.collect_data(([segment], None))
    positive_model.train_with_data()
    positive_output = positive_model.estimate([[obs_batch], [target_reward]])

    assert positive_model.last_target_reward_changed is True
    assert positive_model.last_target_reward_delta_abs_max > 0.0
    assert positive_model.last_target_reward_delta_abs_max <= 0.25 + 1e-6
    assert not np.array_equal(positive_output[1][0], target_reward)

    extra_top_level = {"sample_weight": np.ones((2, 3), dtype=np.float32)}
    target_mask = np.ones((2, 3, 1), dtype=np.float32)
    structured_output = positive_model.estimate(
        [[obs_batch, "obs_meta"], [target_reward, target_mask], extra_top_level]
    )

    assert len(structured_output) == 3
    assert structured_output[0][1] == "obs_meta"
    assert structured_output[1][1] is target_mask
    assert structured_output[2] is extra_top_level
    assert not np.array_equal(structured_output[1][0], target_reward)

    latest_text = (tmp_path / "rnd_latest.json").read_text(encoding="utf-8")
    assert "\x00" not in latest_text
    latest_metrics = json.loads(latest_text)
    assert latest_metrics["schema_id"] == "curvyzero_rnd_reward_model_metrics/v0"
    assert latest_metrics["estimate_calls"] >= 1
    assert latest_metrics["last_raw_mse_mean"] is not None
    assert latest_metrics["train_cnt_per_estimate"] is not None
    assert (tmp_path / "rnd_metrics.jsonl").read_text(encoding="utf-8").startswith("{")


def test_curvy_rnd_reward_model_seed_controls_init_and_sampling(tmp_path):
    pytest.importorskip("torch")
    config = {
        "seed": 1234,
        "input_type": "obs",
        "intrinsic_reward_type": "add",
        "intrinsic_reward_weight": 0.0,
        "curvyzero_adapter": {
            "shape": [1, 64, 64],
            "source_observation_shape": [4, 64, 64],
        },
        "hidden_size_list": [8],
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "batch_size": 4,
        "update_per_collect": 2,
        "rnd_buffer_size": 16,
    }
    first = xb.CurvyRNDRewardModel(config, device="cpu")
    second = xb.CurvyRNDRewardModel(config, device="cpu")

    assert first.model_state_hash("predictor") == second.model_state_hash("predictor")
    assert first.model_state_hash("target") == second.model_state_hash("target")

    class Segment:
        pass

    segment = Segment()
    segment.obs_segment = np.zeros((8, 4, 64, 64), dtype=np.float32)
    for idx in range(8):
        segment.obs_segment[idx, 3, :, :] = idx / 7.0

    first.collect_data(([segment], None))
    second.collect_data(([segment], None))
    first.train_with_data()
    second.train_with_data()

    assert first.last_train_loss == pytest.approx(second.last_train_loss)
    assert first.model_state_hash("predictor") == second.model_state_hash("predictor")
    assert first.metrics_snapshot(reason="test")["seed"] == 1234


def test_curvy_rnd_reward_model_update_cadence_and_small_buffer_metrics(monkeypatch):
    pytest.importorskip("torch")
    config = {
        "seed": 3,
        "input_type": "obs",
        "intrinsic_reward_type": "add",
        "intrinsic_reward_weight": 0.0,
        "curvyzero_adapter": {
            "shape": [1, 64, 64],
            "source_observation_shape": [4, 64, 64],
        },
        "hidden_size_list": [8],
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "batch_size": 4,
        "update_per_collect": 5,
        "rnd_buffer_size": 16,
    }
    model = xb.CurvyRNDRewardModel(config, device="cpu")

    model.train_with_data()
    assert model.train_with_data_calls == 1
    assert model.train_with_data_skipped_small_buffer_count == 1
    assert model.train_cnt_rnd == 0

    class Segment:
        pass

    segment = Segment()
    segment.obs_segment = np.zeros((8, 4, 64, 64), dtype=np.float32)
    for idx in range(8):
        segment.obs_segment[idx, 3, :, :] = idx / 7.0
    model.collect_data(([segment], None))
    state_hash_calls = []
    original_state_hash = model._state_hash

    def counted_state_hash(module_name):
        state_hash_calls.append(module_name)
        return original_state_hash(module_name)

    monkeypatch.setattr(model, "_state_hash", counted_state_hash)
    model.train_with_data()

    assert model.train_with_data_calls == 2
    assert model.train_with_data_skipped_small_buffer_count == 1
    assert model.train_cnt_rnd == 5
    assert state_hash_calls == ["predictor", "target", "predictor", "target"]
    assert model.last_predictor_hash_before_train != model.last_predictor_hash_after_train
    assert model.last_target_hash_before_train == model.last_target_hash_after_train

    obs_batch = np.zeros((2, 3, 4, 64, 64), dtype=np.float32)
    obs_batch[:, :, 3, :, :] = np.linspace(0.0, 1.0, num=6, dtype=np.float32).reshape(2, 3, 1, 1)
    target_reward = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    output = model.estimate([[obs_batch], [target_reward]])
    metrics = model.metrics_snapshot(reason="test")

    assert np.array_equal(output[1][0], target_reward)
    assert metrics["estimate_cnt_rnd"] == 1
    assert metrics["train_cnt_per_estimate"] == pytest.approx(5.0)
    assert metrics["train_with_data_calls_per_collect"] == pytest.approx(2.0)
    assert metrics["last_raw_mse_mean"] is not None
    assert metrics["last_raw_mse_p95"] is not None


def test_curvy_rnd_reward_model_reports_disable_cudnn_flag():
    pytest.importorskip("torch")
    config = {
        "seed": 5,
        "input_type": "obs",
        "intrinsic_reward_type": "add",
        "intrinsic_reward_weight": 0.0,
        "disable_cudnn": True,
        "curvyzero_adapter": {
            "shape": [1, 64, 64],
            "source_observation_shape": [4, 64, 64],
        },
        "hidden_size_list": [8],
        "batch_size": 4,
        "update_per_collect": 1,
    }
    model = xb.CurvyRNDRewardModel(config, device="cpu")

    assert model.metrics_snapshot(reason="test")["disable_cudnn"] is True
