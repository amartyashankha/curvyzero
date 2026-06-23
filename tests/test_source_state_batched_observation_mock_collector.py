import numpy as np
import pytest
from types import SimpleNamespace

from curvyzero.training import exploration_bonus as xb
from curvyzero.training.source_state_batched_observation_mock_collector import (
    BatchedLightZeroScalarActionBridge,
    BatchedLightZeroProfileEnvManager,
    BatchedLightZeroStockEnvManagerAdapter,
    BatchedSourceStateTrainerProfileLoop,
    materialize_lightzero_scalar_timestep,
    materialize_trainer_surface_policy_timestep,
    MockCollectorConfig,
)
from curvyzero.training.source_state_batched_observation_mock_collector import (
    run_mock_collector_profile,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
)
from curvyzero.training.source_state_batched_observation_profile import (
    SourceStateBatchedRenderRequest,
    SourceStateBatchedRenderResult,
)


class _ProfileLoopRenderer:
    backend_name = "test_profile_loop_renderer"

    def render(self, request: SourceStateBatchedRenderRequest) -> SourceStateBatchedRenderResult:
        rows = np.asarray(request.row_indices, dtype=np.int64)
        players = np.asarray(request.controlled_players, dtype=np.int64)
        for output_row, (row, player) in enumerate(zip(rows, players, strict=True)):
            request.out[output_row, 0].fill(int(row) * 32 + int(player) * 8)
        return SourceStateBatchedRenderResult(
            frames=request.out,
            telemetry={"render_sec": 0.0},
        )


def test_mock_collector_profile_materializes_lightzero_shaped_rows():
    result = run_mock_collector_profile(
        MockCollectorConfig(batch_size=2, steps=2, warmup_steps=1, seed=3)
    )

    assert result["profile_only"] is True
    assert result["stock_lightzero_integrated"] is False
    assert result["trainer_defaults_changed"] is False
    assert result["measured_rows"] == 8
    assert result["rows_per_step"] == 4
    assert result["rows_per_sec"] > 0.0
    assert result["pickle_bytes_per_row"] > 0.0
    assert result["flat_obs_nbytes"] == 8 * 4 * 64 * 64 * np.dtype(np.float32).itemsize
    assert result["action_mask_nbytes"] == 8 * 3 * np.dtype(np.bool_).itemsize
    assert result["reward_nbytes"] == 8 * np.dtype(np.float32).itemsize
    assert result["done_nbytes"] == 8 * np.dtype(np.bool_).itemsize
    assert result["final_observation_nbytes"] == 0
    assert result["info_count"] == 8
    assert result["materialized_timestep_count"] == 8
    assert result["semantic_contract"]["pixel_exact_required"] is False
    assert "row_player_order" in result["semantic_contract"]["required"]
    assert result["timings"]["facade_step_sec"] > 0.0
    assert result["timings"]["scalarize_sec"] >= 0.0


def test_mock_collector_profile_runs_rnd_latest_frame_meter():
    pytest.importorskip("torch")

    result = run_mock_collector_profile(
        MockCollectorConfig(
            batch_size=4,
            steps=2,
            warmup_steps=1,
            seed=5,
            include_rnd_meter=True,
            rnd_batch_size=2,
            rnd_update_per_collect=1,
        )
    )

    rnd_metrics = result["rnd_metrics"]
    assert rnd_metrics is not None
    assert rnd_metrics["collect_data_calls"] == 2
    assert rnd_metrics["train_with_data_calls"] == 2
    assert rnd_metrics["estimate_calls"] == 2
    assert rnd_metrics["train_cnt_rnd"] == 2
    assert rnd_metrics["source_observation_shape"] == list(xb.RND_SOURCE_OBSERVATION_SHAPE)
    assert rnd_metrics["input_shape"] == list(xb.RND_INPUT_SHAPE_POLICY_GRAY64_LATEST_V0)
    assert result["timings"]["rnd_collect_data_sec"] > 0.0
    assert result["timings"]["rnd_train_with_data_sec"] > 0.0
    assert result["timings"]["rnd_estimate_sec"] > 0.0


def test_mock_collector_profile_keeps_rnd_reward_meter_non_mutating():
    pytest.importorskip("torch")

    result = run_mock_collector_profile(
        MockCollectorConfig(
            batch_size=2,
            steps=1,
            warmup_steps=0,
            seed=7,
            include_rnd_meter=True,
            rnd_batch_size=2,
            rnd_update_per_collect=1,
        )
    )

    assert result["rnd_metrics"]["intrinsic_reward_weight"] == 0.0
    assert result["rnd_metrics"]["last_target_reward_changed"] is False
    assert np.isfinite(result["timings"]["rnd_estimate_sec"])


def test_batched_surface_profile_loop_materializes_policy_rows_in_surface_order():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=17,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    loop = BatchedSourceStateTrainerProfileLoop(surface)

    result = loop.reset(seed=17)

    assert loop.profile_only is True
    assert loop.stock_lightzero_integrated is False
    assert loop.touches_live_runs is False
    np.testing.assert_array_equal(
        result.surface_step.policy_env_row,
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.surface_step.policy_player,
        np.asarray([0, 1, 0, 1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        result.timestep.obs["observation"],
        result.surface_step.policy_observation,
    )
    np.testing.assert_array_equal(
        result.timestep.obs["action_mask"],
        result.surface_step.policy_action_mask,
    )
    assert [item["row"] for item in result.timestep.info] == [0, 0, 1, 1]
    assert [item["player"] for item in result.timestep.info] == [0, 1, 0, 1]
    np.testing.assert_allclose(
        result.flat_obs[:, -1, 0, 0],
        np.asarray([0, 8, 32, 40], dtype=np.float32) / np.float32(255.0),
    )
    np.testing.assert_array_equal(result.timestep.reward, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(result.timestep.done, np.zeros(4, dtype=bool))

    action = np.asarray([[0, 1], [2, 1]], dtype=np.int16)
    step_result = loop.step(action)
    np.testing.assert_array_equal(
        step_result.timestep.obs["observation"],
        step_result.surface_step.policy_observation,
    )
    assert step_result.target_reward.shape == (4, 1)


def test_batched_surface_profile_loop_rejects_missing_or_extra_actions():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=19,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    loop = BatchedSourceStateTrainerProfileLoop(surface)
    loop.reset(seed=19)

    with pytest.raises(ValueError, match="joint_action must have shape"):
        loop.step(np.asarray([[0, 1]], dtype=np.int16))

    with pytest.raises(ValueError, match="joint_action must have shape"):
        loop.step(np.asarray([[0, 1], [2, 1], [1, 0]], dtype=np.int16))


def test_batched_surface_profile_loop_partial_reset_keeps_neighboring_rows():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=23,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    reset_step = surface.reset(seed=23)
    original_row_1 = reset_step.observation[1].copy()

    partial_step = surface.reset(seed=np.asarray([29, 31]), row_mask=np.asarray([True, False]))

    np.testing.assert_array_equal(partial_step.observation[1], original_row_1)
    np.testing.assert_array_equal(
        partial_step.policy_env_row,
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        partial_step.policy_player,
        np.asarray([0, 1, 0, 1], dtype=np.int16),
    )


def test_scalar_action_bridge_exposes_lightzero_env_ids_after_reset():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=37,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    bridge = BatchedLightZeroScalarActionBridge(surface)

    output = bridge.reset(seed=37)

    np.testing.assert_array_equal(
        output.policy_env_id,
        np.asarray([0, 1, 2, 3], dtype=np.int32),
    )
    assert bridge.ready_env_ids == (0, 1, 2, 3)
    assert sorted(output.ready_obs) == [0, 1, 2, 3]
    for env_id, ready_obs in output.ready_obs.items():
        row, player = bridge.row_player_for_scalar_env_id(env_id)
        assert bridge.scalar_env_id(row=row, player=player) == env_id
        assert ready_obs["observation"].shape == (4, 64, 64)
        assert ready_obs["action_mask"].shape == (3,)
        assert ready_obs["to_play"] == -1


def test_scalar_action_bridge_profiles_materialized_timestep_bytes_and_counts():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=39,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    bridge = BatchedLightZeroScalarActionBridge(surface)

    reset_output = bridge.reset(seed=39)

    assert reset_output.profile_counts is not None
    reset_counts = reset_output.profile_counts
    assert reset_counts["flat_obs_nbytes"] == 4 * 4 * 64 * 64 * np.dtype(np.float32).itemsize
    assert reset_counts["action_mask_nbytes"] == 4 * 3 * np.dtype(np.bool_).itemsize
    assert reset_counts["reward_nbytes"] == 4 * np.dtype(np.float32).itemsize
    assert reset_counts["done_nbytes"] == 4 * np.dtype(np.bool_).itemsize
    assert reset_counts["final_observation_nbytes"] == 0
    assert reset_counts["info_count"] == 4
    assert reset_counts["materialized_timestep_count"] == 4

    step_output = bridge.step({0: 0, 1: 1})

    assert step_output.profile_counts is not None
    step_counts = step_output.profile_counts
    assert step_counts["flat_obs_nbytes"] == 2 * 4 * 64 * 64 * np.dtype(np.float32).itemsize
    assert step_counts["action_mask_nbytes"] == 2 * 3 * np.dtype(np.bool_).itemsize
    assert step_counts["reward_nbytes"] == 2 * np.dtype(np.float32).itemsize
    assert step_counts["done_nbytes"] == 2 * np.dtype(np.bool_).itemsize
    assert step_counts["final_observation_nbytes"] == 0
    assert step_counts["info_count"] == 2
    assert step_counts["materialized_timestep_count"] == 2


def test_scalar_action_bridge_commits_one_joint_action_from_scalar_actions():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=41,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    bridge = BatchedLightZeroScalarActionBridge(surface)
    bridge.reset(seed=41)

    output = bridge.step({0: 0, 1: 1, 2: 2, 3: 1})

    np.testing.assert_array_equal(
        output.surface_step.joint_action,
        np.asarray([[0, 1], [2, 1]], dtype=np.int16),
    )
    assert sorted(output.timestep_by_env_id) == [0, 1, 2, 3]
    assert sorted(output.ready_obs) == [0, 1, 2, 3]
    assert output.autoreset_row_mask.shape == (2,)
    for env_id, timestep in output.timestep_by_env_id.items():
        assert timestep.obs["observation"].shape == (4, 64, 64)
        assert timestep.obs["action_mask"].shape == (3,)
        assert timestep.info[0]["row"] == env_id // 2
        assert timestep.info[0]["player"] == env_id % 2


def test_scalar_action_bridge_allows_complete_row_omission_but_rejects_partial_missing_extra_and_invalid_actions():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=43,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    bridge = BatchedLightZeroScalarActionBridge(surface)
    bridge.reset(seed=43)

    output = bridge.step({0: 0, 1: 1})

    assert sorted(output.timestep_by_env_id) == [0, 1]

    with pytest.raises(ValueError, match="complete physical CurvyTron rows"):
        bridge.step({0: 0, 1: 1, 2: 2})

    with pytest.raises(ValueError, match="complete physical CurvyTron rows"):
        bridge.step({0: 0, 1: 1, 2: 2, 3: 1, 4: 0})

    with pytest.raises(ValueError, match="actions must be in"):
        bridge.step({0: 0, 1: 1, 2: 2, 3: 99})


def test_profile_env_manager_surface_tracks_ready_obs_and_step_results():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=47,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    bridge = BatchedLightZeroScalarActionBridge(surface)
    manager = BatchedLightZeroProfileEnvManager(bridge)

    manager.seed(47, dynamic_seed=False)
    manager.reset({env_id: {"seed": 47 + env_id} for env_id in range(manager.env_num)})

    assert manager.env_num == 4
    assert sorted(manager.ready_obs) == [0, 1, 2, 3]
    assert sorted(manager.last_reset_info) == [0, 1, 2, 3]
    assert manager.last_reset_info[2]["vector_surface_batch_size"] == 2
    assert manager.last_reset_info[2]["row"] == 1
    assert manager.last_reset_info[2]["player"] == 0

    result = manager.step({0: 0, 1: 1, 2: 2, 3: 1})

    assert sorted(result.timestep_by_env_id) == [0, 1, 2, 3]
    assert sorted(result.ready_obs) == [0, 1, 2, 3]
    np.testing.assert_array_equal(
        result.bridge_output.surface_step.joint_action,
        np.asarray([[0, 1], [2, 1]], dtype=np.int16),
    )


def test_profile_env_manager_rejects_partial_reset_and_close_step():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=53,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    manager = BatchedLightZeroProfileEnvManager(BatchedLightZeroScalarActionBridge(surface))

    with pytest.raises(ValueError, match="requires all scalar env ids"):
        manager.reset({0: {"seed": 53}})

    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.step({0: 0, 1: 1, 2: 2, 3: 1})


def test_profile_env_manager_keeps_terminal_timestep_before_autoreset():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=59,
        max_ticks=1,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    manager = BatchedLightZeroProfileEnvManager(
        BatchedLightZeroScalarActionBridge(surface, autoreset_terminal_rows=True)
    )
    manager.reset(59)

    result = manager.step({0: 0, 1: 1, 2: 2, 3: 1})

    assert sorted(result.timestep_by_env_id) == [0, 1, 2, 3]
    assert sorted(result.ready_obs) == [0, 1, 2, 3]
    assert result.bridge_output.autoreset_row_mask.any()
    assert result.bridge_output.profile_counts is not None
    assert result.bridge_output.profile_counts["final_observation_nbytes"] == (
        4 * 4 * 64 * 64 * np.dtype(np.float32).itemsize
    )
    assert result.bridge_output.profile_counts["materialized_timestep_count"] == 4
    for env_id, timestep in result.timestep_by_env_id.items():
        assert bool(np.asarray(timestep.done).item()) is True
        assert timestep.info[0]["row"] == env_id // 2
        assert timestep.info[0]["player"] == env_id % 2
        assert timestep.info[0]["final_observation_present"] is True
        assert timestep.info[0]["final_observation"].shape == (4, 64, 64)


def test_stock_env_manager_adapter_returns_env_id_timestep_mapping():
    class FakeBaseEnvTimestep:
        def __init__(self, obs, reward, done, info):
            self.obs = obs
            self.reward = reward
            self.done = done
            self.info = info

    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=2,
        seed=61,
        natural_bonus_spawn=False,
        observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        observation_renderer=_ProfileLoopRenderer(),
    )
    adapter = BatchedLightZeroStockEnvManagerAdapter(
        BatchedLightZeroScalarActionBridge(surface),
        base_env_timestep_cls=FakeBaseEnvTimestep,
    )

    adapter.seed(61, dynamic_seed=False)
    adapter.launch()
    assert sorted(adapter.ready_obs) == [0, 1, 2, 3]
    result = adapter.step(
        {
            0: {"action": np.asarray([0], dtype=np.int64)},
            1: np.asarray(1, dtype=np.int64),
            2: 2,
            3: {"action_id": 1},
        }
    )

    assert adapter.env_num == 4
    assert adapter.ready_obs_id == [0, 1, 2, 3]
    assert adapter.action_space.n == 3
    assert adapter.observation_space.shape == (4, 64, 64)
    assert adapter.env_ref.action_space.n == 3
    assert adapter.random_action().keys() == {0, 1, 2, 3}
    assert sorted(result) == [0, 1, 2, 3]
    assert sorted(adapter.ready_obs) == [0, 1, 2, 3]
    assert adapter.last_profile_step is not None
    for env_id, timestep in result.items():
        assert isinstance(timestep, FakeBaseEnvTimestep)
        assert timestep.obs["observation"].shape == (4, 64, 64)
        assert timestep.obs["action_mask"].shape == (3,)
        assert timestep.obs["to_play"] == -1
        assert isinstance(timestep.reward, float)
        assert isinstance(timestep.done, bool)
        assert timestep.info["row"] == env_id // 2
        assert timestep.info["player"] == env_id % 2


def test_stock_env_manager_adapter_preserves_mixed_terminal_live_rows_before_autoreset():
    class FakeBaseEnvTimestep:
        def __init__(self, obs, reward, done, info):
            self.obs = obs
            self.reward = reward
            self.done = done
            self.info = info

    class MixedTerminalSurface:
        batch_size = 2
        player_count = 2

        def __init__(self):
            self.reset_row_masks = []
            self.last_joint_action = None

        def reset(self, seed=None, row_mask=None):
            del seed
            mask = None if row_mask is None else np.asarray(row_mask, dtype=bool).copy()
            self.reset_row_masks.append(mask)
            row_values = np.asarray([0.91, 0.22], dtype=np.float32)
            if mask is None:
                row_values = np.asarray([0.11, 0.22], dtype=np.float32)
            return self._surface_step(
                row_values=row_values,
                done=np.asarray([False, False], dtype=bool),
                final_mask=np.asarray([False, False], dtype=bool),
            )

        def step(self, joint_action, timer_advance_ms=None):
            del timer_advance_ms
            self.last_joint_action = np.asarray(joint_action, dtype=np.int16).copy()
            return self._surface_step(
                row_values=np.asarray([0.51, 0.22], dtype=np.float32),
                done=np.asarray([True, False], dtype=bool),
                final_mask=np.asarray([True, False], dtype=bool),
                reward=np.asarray([[1.0, -1.0], [0.25, 0.5]], dtype=np.float32),
                final_row_value=np.float32(0.77),
            )

        def _surface_step(
            self,
            *,
            row_values,
            done,
            final_mask,
            reward=None,
            final_row_value=np.float32(0.0),
        ):
            rows = np.asarray([0, 0, 1, 1], dtype=np.int32)
            players = np.asarray([0, 1, 0, 1], dtype=np.int16)
            observation = np.zeros((2, 2, 4, 64, 64), dtype=np.float32)
            for row in range(2):
                for player in range(2):
                    observation[row, player, -1, 0, 0] = row_values[row] + (
                        np.float32(player) * np.float32(0.01)
                    )
            final_observation = np.zeros_like(observation)
            final_observation[0, :, -1, 0, 0] = final_row_value + np.asarray(
                [0.0, 0.01],
                dtype=np.float32,
            )
            legal_action_mask = np.ones((2, 2, 3), dtype=bool)
            reward_map = (
                np.zeros((2, 2), dtype=np.float32)
                if reward is None
                else np.asarray(reward, dtype=np.float32)
            )
            return SimpleNamespace(
                observation=observation,
                legal_action_mask=legal_action_mask,
                policy_observation=observation[rows, players].copy(),
                policy_action_mask=legal_action_mask[rows, players].copy(),
                policy_env_row=rows,
                policy_player=players,
                reward=reward_map,
                done=np.asarray(done, dtype=bool),
                final_observation_row_mask=np.asarray(final_mask, dtype=bool),
                final_observation=final_observation,
                final_reward_map=reward_map.copy(),
            )

    surface = MixedTerminalSurface()
    adapter = BatchedLightZeroStockEnvManagerAdapter(
        BatchedLightZeroScalarActionBridge(surface, autoreset_terminal_rows=True),
        base_env_timestep_cls=FakeBaseEnvTimestep,
    )

    adapter.seed(67, dynamic_seed=False)
    adapter.launch()
    result = adapter.step({0: 0, 1: 1, 2: 2, 3: 1})

    assert sorted(result) == [0, 1, 2, 3]
    assert adapter.ready_obs_id == [0, 1, 2, 3]
    np.testing.assert_array_equal(
        surface.last_joint_action,
        np.asarray([[0, 1], [2, 1]], dtype=np.int16),
    )
    profile_step = adapter.last_profile_step
    assert profile_step is not None
    np.testing.assert_array_equal(
        profile_step.bridge_output.autoreset_row_mask,
        np.asarray([True, False], dtype=bool),
    )
    assert profile_step.bridge_output.profile_counts["autoreset_row_count"] == 1
    assert len(surface.reset_row_masks) == 2
    np.testing.assert_array_equal(
        surface.reset_row_masks[-1],
        np.asarray([True, False], dtype=bool),
    )

    for env_id in (0, 1):
        timestep = result[env_id]
        assert timestep.done is True
        assert timestep.info["row"] == 0
        assert timestep.info["final_observation_present"] is True
        assert timestep.info["final_observation"].shape == (4, 64, 64)
        assert timestep.info["final_observation"][-1, 0, 0] == pytest.approx(
            0.77 + (env_id % 2) * 0.01
        )

    for env_id in (2, 3):
        timestep = result[env_id]
        assert timestep.done is False
        assert timestep.info["row"] == 1
        assert timestep.info["final_observation_present"] is False
        assert "final_observation" not in timestep.info

    assert adapter.ready_obs[0]["observation"][-1, 0, 0] == pytest.approx(0.91)
    assert adapter.ready_obs[2]["observation"][-1, 0, 0] == pytest.approx(0.22)
    assert adapter.ready_obs[3]["observation"][-1, 0, 0] == pytest.approx(0.23)


def test_surface_policy_materializer_attaches_terminal_final_observation():
    final_observation = np.zeros((2, 2, 4, 64, 64), dtype=np.float32)
    final_observation[1, 0, -1, 0, 0] = 0.75
    surface_step = SimpleNamespace(
        policy_observation=np.zeros((1, 4, 64, 64), dtype=np.float32),
        policy_action_mask=np.ones((1, 3), dtype=bool),
        policy_env_row=np.asarray([1], dtype=np.int32),
        policy_player=np.asarray([0], dtype=np.int16),
        reward=np.zeros((2, 2), dtype=np.float32),
        done=np.asarray([False, True], dtype=bool),
        final_observation_row_mask=np.asarray([False, True], dtype=bool),
        final_observation=final_observation,
        final_reward_map=np.asarray([[0.0, 0.0], [1.5, 0.0]], dtype=np.float32),
    )

    timestep, _flat_obs, _target_reward = materialize_trainer_surface_policy_timestep(
        surface_step=surface_step,
        batch_size=2,
        player_count=2,
    )

    assert timestep.info[0]["final_observation_present"] is True
    assert timestep.info[0]["final_reward"] == pytest.approx(1.5)
    assert timestep.info[0]["final_observation"][-1, 0, 0] == pytest.approx(0.75)
    np.testing.assert_array_equal(timestep.done, np.asarray([True], dtype=bool))


def test_surface_policy_materializer_requires_terminal_final_observation():
    surface_step = SimpleNamespace(
        policy_observation=np.zeros((1, 4, 64, 64), dtype=np.float32),
        policy_action_mask=np.ones((1, 3), dtype=bool),
        policy_env_row=np.asarray([1], dtype=np.int32),
        policy_player=np.asarray([0], dtype=np.int16),
        reward=np.zeros((2, 2), dtype=np.float32),
        done=np.asarray([False, True], dtype=bool),
        final_observation_row_mask=np.asarray([False, True], dtype=bool),
        final_reward_map=np.asarray([[0.0, 0.0], [1.5, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="final_observation is required"):
        materialize_trainer_surface_policy_timestep(
            surface_step=surface_step,
            batch_size=2,
            player_count=2,
        )


def test_surface_policy_materializer_rejects_malformed_terminal_final_observation():
    surface_step = SimpleNamespace(
        policy_observation=np.zeros((1, 4, 64, 64), dtype=np.float32),
        policy_action_mask=np.ones((1, 3), dtype=bool),
        policy_env_row=np.asarray([1], dtype=np.int32),
        policy_player=np.asarray([0], dtype=np.int16),
        reward=np.zeros((2, 2), dtype=np.float32),
        done=np.asarray([False, True], dtype=bool),
        final_observation_row_mask=np.asarray([False, True], dtype=bool),
        final_observation=np.zeros((2, 2, 4, 32, 32), dtype=np.float32),
        final_reward_map=np.asarray([[0.0, 0.0], [1.5, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="final_observation must have shape"):
        materialize_trainer_surface_policy_timestep(
            surface_step=surface_step,
            batch_size=2,
            player_count=2,
        )


def test_scalar_materializer_attaches_terminal_final_observation():
    final_observation = np.zeros((2, 2, 4, 64, 64), dtype=np.float32)
    final_observation[0, 1, -1, 0, 0] = 0.5

    timestep, _flat_obs, _target_reward = materialize_lightzero_scalar_timestep(
        step_observation=np.zeros((2, 2, 4, 64, 64), dtype=np.float32),
        step_reward=np.zeros((2, 2), dtype=np.float32),
        step_done=np.asarray([True, False], dtype=bool),
        final_observation=final_observation,
        batch_size=2,
        player_count=2,
    )

    assert timestep.info[1]["row"] == 0
    assert timestep.info[1]["player"] == 1
    assert timestep.info[1]["final_observation_present"] is True
    assert timestep.info[1]["final_observation"][-1, 0, 0] == pytest.approx(0.5)
    assert timestep.info[2]["final_observation_present"] is False


def test_scalar_materializer_preserves_batch_action_mask_order():
    action_mask = np.asarray(
        [
            [[True, False, True], [False, True, True]],
            [[True, True, False], [False, False, True]],
        ],
        dtype=bool,
    )

    timestep, _flat_obs, _target_reward = materialize_lightzero_scalar_timestep(
        step_observation=np.zeros((2, 2, 4, 64, 64), dtype=np.float32),
        step_reward=np.zeros((2, 2), dtype=np.float32),
        step_done=np.asarray([False, False], dtype=bool),
        final_observation=None,
        action_mask=action_mask,
        batch_size=2,
        player_count=2,
    )

    expected = action_mask.reshape(4, 3)
    np.testing.assert_array_equal(timestep.obs["action_mask"], expected)
