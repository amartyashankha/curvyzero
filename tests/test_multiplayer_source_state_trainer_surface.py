import math

import numpy as np
import pytest

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import DEBUG_METADATA_OBSERVATION_SCHEMA_ID
from curvyzero.env.vector_visual_observation import BONUS_RENDER_MODE_BROWSER_SPRITES
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BROWSER_LINES
from curvyzero.env.vector_visual_observation import normalize_source_state_gray64
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    STACK_RENDER_MODE_FAST_GRAY64_DIRECT,
)
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    player_perspective_rgb_palette,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    FINAL_VISUAL_OBSERVATION_POLICY_ID,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    FRAME_STACK_OWNER,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    JOINT_ACTION_LABEL,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    NATIVE_SOURCE_CONTROL_MODEL,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    TRAINER_OBSERVATION_SOURCE,
)


def _assert_surface_reset_contract(step, *, batch_size: int, player_count: int) -> None:
    assert step.observation.shape == (batch_size, player_count, 4, 64, 64)
    assert step.observation.dtype == np.float32
    assert float(step.observation.min()) >= 0.0
    assert float(step.observation.max()) <= 1.0
    assert step.legal_action_mask.shape == (batch_size, player_count, 3)
    assert step.legal_action_mask.dtype == bool
    np.testing.assert_array_equal(step.lightzero_action_mask, step.legal_action_mask)
    assert step.live_mask.shape == (batch_size, player_count)
    assert step.policy_observation.shape == (batch_size * player_count, 4, 64, 64)
    assert step.policy_observation.dtype == np.float32
    assert step.policy_action_mask.shape == (batch_size * player_count, 3)
    assert step.policy_action_mask.dtype == bool
    assert step.policy_env_row.dtype == np.int32
    assert step.policy_player.dtype == np.int16
    np.testing.assert_array_equal(
        step.reward,
        np.zeros((batch_size, player_count), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        step.joint_action,
        np.full((batch_size, player_count), -1, dtype=np.int16),
    )
    assert step.final_observation.shape == step.observation.shape
    np.testing.assert_array_equal(step.final_observation, np.zeros_like(step.observation))
    np.testing.assert_array_equal(
        step.final_observation_row_mask,
        np.zeros(batch_size, dtype=bool),
    )

    info = step.info
    assert info["trainer_surface_schema_id"] == MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID
    assert info["visual_source_state_backed"] is True
    assert info["source_state_backed"] is True
    assert info["rgb_to_gray64"] is True
    assert info["debug_fidelity_only"] is False
    assert info["metadata_only"] is False
    assert info["underlying_env_metadata_only"] is True
    assert info["underlying_env_observation_schema_id"] == DEBUG_METADATA_OBSERVATION_SCHEMA_ID
    assert info["underlying_env_observation_is_metadata_only"] is True
    assert info["underlying_env_observation_used_as_trainer_observation"] is False
    assert info["underlying_env_class"] == "VectorMultiplayerEnv"
    assert info["visual_stack_class"] == TRAINER_OBSERVATION_SOURCE
    assert info["trainer_observation_source"] == TRAINER_OBSERVATION_SOURCE
    assert info["trainer_observation_claim"] is True
    assert info["trainer_observation_claim_id"] == (
        "source_state_visual_stack_per_live_seat/v0"
    )
    assert info["trainer_replay_claim"] is False
    assert info["trainer_replay_claim_id"] is None
    assert info["uses_ale"] is False
    assert info["browser_pixel_fidelity"] is False
    assert info["trail_render_mode"] == TRAIL_RENDER_MODE_BROWSER_LINES
    assert info["default_trail_render_mode"] == TRAIL_RENDER_MODE_BROWSER_LINES
    assert info["trainer_supported_trail_render_modes"] == [
        TRAIL_RENDER_MODE_BROWSER_LINES,
        TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    ]
    assert info["bonus_render_mode"] == BONUS_RENDER_MODE_BROWSER_SPRITES
    assert info["default_bonus_render_mode"] == BONUS_RENDER_MODE_BROWSER_SPRITES
    assert info["browser_sprites_bonus_render_claim"] is True
    assert info["frame_stack_owner"] == FRAME_STACK_OWNER
    assert info["render_metadata"]["bonus_renderer_kind"] == "browser_sprites"
    assert info["render_metadata"]["bonus_renderer_is_approximation"] is False
    stats = info["visual_stack_dirty_render_stats"]
    assert stats["enabled"] is (player_count == 2)
    assert set(stats) == {
        "enabled",
        "rows",
        "attempts",
        "hits",
        "cold_starts",
        "fallbacks",
        "dirty_blocks_total",
        "hit_rate",
        "dirty_blocks_per_hit",
    }
    assert info["decision_cadence_is_wrapper_abstraction"] is True
    assert info["native_source_control_model"] == NATIVE_SOURCE_CONTROL_MODEL
    assert info["policy_row_mapping_schema_id"] == (
        "curvyzero_source_state_multiplayer_live_policy_rows/v0"
    )
    assert info["policy_row_count"] == batch_size * player_count
    np.testing.assert_array_equal(info["policy_env_row"], step.policy_env_row)
    np.testing.assert_array_equal(info["policy_player"], step.policy_player)
    np.testing.assert_array_equal(info["policy_action_mask"], step.policy_action_mask)
    assert info["render_metadata"]["trail_renderer_is_approximation"] is False
    assert info["trainer_observation_schema_id"] is not None
    assert info["single_frame_schema_id"].startswith("curvyzero_source_state_canvas_gray64")


def _assert_policy_rows_match_live_visual_observation(step) -> None:
    expected_env_row, expected_player = np.nonzero(step.live_mask)
    expected_env_row = expected_env_row.astype(np.int32, copy=False)
    expected_player = expected_player.astype(np.int16, copy=False)

    np.testing.assert_array_equal(step.policy_env_row, expected_env_row)
    np.testing.assert_array_equal(step.policy_player, expected_player)
    assert step.policy_observation.shape == (expected_env_row.size, 4, 64, 64)
    assert step.policy_action_mask.shape == (expected_env_row.size, 3)
    np.testing.assert_array_equal(
        step.policy_action_mask,
        step.legal_action_mask[expected_env_row, expected_player],
    )
    for policy_row, (row, player) in enumerate(
        zip(step.policy_env_row, step.policy_player, strict=True)
    ):
        np.testing.assert_array_equal(
            step.policy_observation[policy_row],
            step.observation[int(row), int(player)],
        )

    assert step.info["policy_row_count"] == int(np.count_nonzero(step.live_mask))
    np.testing.assert_array_equal(step.info["live_policy_row_mask"], step.live_mask)
    np.testing.assert_array_equal(step.info["policy_env_row"], step.policy_env_row)
    np.testing.assert_array_equal(step.info["policy_player"], step.policy_player)
    np.testing.assert_array_equal(step.info["policy_action_mask"], step.policy_action_mask)


def _assert_step_visual_contract(
    surface: SourceStateMultiplayerTrainerSurface,
    step,
    joint_action: np.ndarray,
) -> None:
    assert step.observation.shape == (surface.batch_size, surface.player_count, 4, 64, 64)
    assert step.observation.dtype == np.float32
    assert step.policy_observation.dtype == np.float32
    assert step.legal_action_mask.shape == (surface.batch_size, surface.player_count, 3)
    assert step.legal_action_mask.dtype == bool
    np.testing.assert_array_equal(step.lightzero_action_mask, step.legal_action_mask)
    np.testing.assert_array_equal(step.legal_action_mask, surface.env._action_mask())
    assert step.joint_action.shape == (surface.batch_size, surface.player_count)
    np.testing.assert_array_equal(step.joint_action, joint_action)

    assert step.info["trainer_surface_api"] == "step"
    assert step.info["trainer_surface_schema_id"] == MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID
    assert step.info["visual_source_state_backed"] is True
    assert step.info["source_state_backed"] is True
    assert step.info["metadata_only"] is False
    assert step.info["underlying_env_metadata_only"] is True
    assert step.info["underlying_env_observation_schema_id"] == (
        DEBUG_METADATA_OBSERVATION_SCHEMA_ID
    )
    assert step.info["underlying_env_observation_is_metadata_only"] is True
    assert step.info["underlying_env_observation_used_as_trainer_observation"] is False
    assert step.info["trainer_observation_source"] == TRAINER_OBSERVATION_SOURCE
    assert step.observation.shape != step.info["underlying_env_observation_shape"]
    np.testing.assert_array_equal(step.info["joint_action"], joint_action)
    assert step.info["joint_action_label"] == JOINT_ACTION_LABEL
    assert step.info["joint_action_player_major"] is True
    np.testing.assert_array_equal(
        step.info["underlying_env_action_mask"],
        step.legal_action_mask,
    )
    np.testing.assert_array_equal(step.info["legal_action_mask"], step.legal_action_mask)
    np.testing.assert_array_equal(step.info["lightzero_action_mask"], step.legal_action_mask)
    _assert_policy_rows_match_live_visual_observation(step)


def _assert_last_frame_matches_direct_source_state_render(
    surface: SourceStateMultiplayerTrainerSurface,
    step,
    *,
    row: int,
) -> None:
    for player in range(surface.player_count):
        expected = normalize_source_state_gray64(
            render_source_state_canvas_gray64(
                surface.env.state,
                row=row,
                player_rgb=player_perspective_rgb_palette(
                    surface.env.state,
                    row=row,
                    controlled_player=player,
                    player_count=surface.player_count,
                ),
                trail_render_mode=surface.trail_render_mode,
            )
        )[0]
        np.testing.assert_array_equal(
            step.observation[row, player, -1],
            expected,
        )


def _install_step_mapping_probe_state(surface: SourceStateMultiplayerTrainerSurface) -> None:
    env = surface.env
    player_count = surface.player_count
    safe_positions = np.asarray(
        [[25.0, 25.0], [25.0, 75.0], [75.0, 25.0], [75.0, 75.0]][:player_count],
        dtype=np.float64,
    )
    safe_headings = np.asarray(
        [0.0, math.pi / 2.0, math.pi, math.tau * 0.75][:player_count],
        dtype=np.float64,
    )
    env.state["pos"][:, :player_count] = safe_positions
    env.state["heading"][:, :player_count] = safe_headings
    env.state["prev_pos"][:, :player_count] = env.state["pos"][:, :player_count]
    env.state["speed"][:, :player_count] = 0.0
    env.state["print_manager_distance"][:, :player_count] = 999.0
    env.state["print_manager_last_pos"][:, :player_count] = env.state["pos"][
        :,
        :player_count,
    ]

    env.state["pos"][1, 0] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["prev_pos"][1, 0] = env.state["pos"][1, 0]
    env.state["heading"][1, 0] = math.pi
    env.state["speed"][1, 0] = 16.0
    env.state["print_manager_last_pos"][1, 0] = env.state["pos"][1, 0]
    surface.stack.reset_rows(env, np.ones(surface.batch_size, dtype=bool))


def _force_4p_player3_terminal_win(surface: SourceStateMultiplayerTrainerSurface) -> None:
    env = surface.env
    env.state["pos"][0] = np.asarray(
        [[1.0, 50.5], [50.5, 1.0], [99.0, 50.5], [10.0, 50.5]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray(
        [math.pi, math.tau * 0.75, 0.0, 0.0],
        dtype=np.float64,
    )
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0] = np.asarray([16.0, 16.0, 16.0, 8.0], dtype=np.float64)
    env.state["print_manager_distance"][0] = np.full(4, 999.0, dtype=np.float64)
    env.state["print_manager_last_pos"][0] = env.state["pos"][0]
    surface.stack.reset_rows(env, np.asarray([True], dtype=bool))


@pytest.mark.parametrize("player_count", [2, 4])
def test_reset_emits_source_state_visual_surface_for_supported_player_counts(player_count):
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=player_count,
        seed=123,
        natural_bonus_spawn=False,
    )

    step = surface.reset(seed=123)

    _assert_surface_reset_contract(step, batch_size=2, player_count=player_count)
    np.testing.assert_array_equal(step.observation, surface.stack.stack)
    for policy_row, (row, player) in enumerate(
        zip(step.policy_env_row, step.policy_player, strict=True)
    ):
        np.testing.assert_array_equal(
            step.policy_observation[policy_row, ...],
            step.observation[int(row), int(player), ...],
        )
    assert step.observation.shape != step.info["underlying_env_observation_shape"]


@pytest.mark.parametrize("player_count", [3, 4])
def test_step_maps_live_policy_rows_for_multiplayer_player_counts(player_count):
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=player_count,
        seed=101 + player_count,
        decision_ms=100.0,
        natural_bonus_spawn=False,
    )
    surface.reset(seed=101 + player_count)
    _install_step_mapping_probe_state(surface)
    joint_action = np.ones((2, player_count), dtype=np.int16)

    step = surface.step(joint_action)

    _assert_step_visual_contract(surface, step, joint_action)
    expected_live_mask = np.ones((2, player_count), dtype=bool)
    expected_live_mask[1, 0] = False
    np.testing.assert_array_equal(step.live_mask, expected_live_mask)
    np.testing.assert_array_equal(step.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(step.terminated, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(step.truncated, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(
        step.legal_action_mask,
        np.repeat(expected_live_mask[:, :, None], 3, axis=2),
    )
    expected_reward = expected_live_mask.astype(np.float32)
    np.testing.assert_array_equal(step.reward, expected_reward)
    np.testing.assert_array_equal(step.info["reward"], expected_reward)
    np.testing.assert_array_equal(step.info["alive"], expected_live_mask)
    np.testing.assert_array_equal(
        step.info["final_observation_row_mask"],
        np.zeros(2, dtype=bool),
    )
    np.testing.assert_array_equal(
        step.final_observation,
        np.zeros_like(step.observation),
    )


def test_step_preserves_player_major_joint_action_mask_and_survival_bonus_reward():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=7,
        decision_source_frames=1,
        natural_bonus_spawn=False,
    )
    surface.reset(seed=7)
    player_0_pos = surface.env.state["pos"][0, 0].copy()
    surface.env.seed_active_bonus(
        row=0,
        bonus_type="BonusGameClear",
        x=float(player_0_pos[0]),
        y=float(player_0_pos[1]),
        radius=3.0,
    )
    joint_action = np.asarray([[1, 1]], dtype=np.int16)

    step = surface.step(joint_action)

    np.testing.assert_array_equal(step.joint_action, joint_action)
    np.testing.assert_array_equal(step.info["joint_action"], joint_action)
    assert step.info["joint_action_label"] == JOINT_ACTION_LABEL
    assert step.info["joint_action_player_major"] is True
    assert step.info["underlying_env_observation_schema_id"] == (
        DEBUG_METADATA_OBSERVATION_SCHEMA_ID
    )
    assert step.info["underlying_env_observation_used_as_trainer_observation"] is False
    assert step.info["trainer_observation_source"] == TRAINER_OBSERVATION_SOURCE
    np.testing.assert_array_equal(step.legal_action_mask, surface.env._action_mask())
    np.testing.assert_array_equal(
        step.info["underlying_env_action_mask"],
        step.legal_action_mask,
    )
    np.testing.assert_array_equal(step.info["live_policy_row_mask"], step.live_mask)
    assert step.info["policy_row_count"] == int(np.count_nonzero(step.live_mask))
    for policy_row, (row, player) in enumerate(
        zip(step.policy_env_row, step.policy_player, strict=True)
    ):
        np.testing.assert_array_equal(
            step.policy_observation[policy_row, ...],
            step.observation[int(row), int(player), ...],
        )

    catch_count = np.asarray(step.info["bonus_catch_count_step"], dtype=np.float32)
    assert int(catch_count[0, 0]) == 1
    alive_after_step = step.info["present"] & step.info["alive"]
    expected_reward = alive_after_step.astype(np.float32) + (
        catch_count * np.float32(SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD)
    )
    np.testing.assert_array_equal(step.reward, expected_reward)
    np.testing.assert_array_equal(step.reward, np.asarray([[2.0, 1.0]], dtype=np.float32))


def test_terminal_final_observation_is_visual_stack_not_metadata_observation():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=11,
        decision_ms=100.0,
        natural_bonus_spawn=False,
    )
    surface.reset(seed=11)
    env = surface.env
    env.state["pos"][0] = np.asarray(
        [[5.0, 5.0], [87.0, 44.0]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray([math.pi / 4.0, 0.0], dtype=np.float64)
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0, 0] = 8.0
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]
    surface.stack.reset_rows(env, np.asarray([True], dtype=bool))

    step = surface.step(np.asarray([[1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(step.done, np.asarray([True], dtype=bool))
    assert step.final_observation.shape == (1, 2, 4, 64, 64)
    assert step.final_observation.dtype == np.float32
    np.testing.assert_array_equal(step.final_observation_row_mask, np.asarray([True]))
    np.testing.assert_array_equal(step.final_observation, step.info["final_observation"])
    np.testing.assert_array_equal(step.final_observation[0], step.observation[0])
    _assert_last_frame_matches_direct_source_state_render(surface, step, row=0)
    assert int(np.count_nonzero(step.final_observation)) > 0
    assert step.info["final_observation_policy"]["schema_id"] == (
        FINAL_VISUAL_OBSERVATION_POLICY_ID
    )
    assert step.info["final_observation_policy"]["metadata_only"] is False
    assert step.info["underlying_final_observation_shape"] == (1, 2, 6)


def test_terminal_final_observation_matches_direct_render_after_dirty_cache_warmup():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=17,
        decision_ms=100.0,
        natural_bonus_spawn=False,
    )
    surface.reset(seed=17)
    env = surface.env
    env.state["pos"][0] = np.asarray(
        [[5.0, 5.0], [87.0, 44.0]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray([math.pi / 4.0, 0.0], dtype=np.float64)
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0] = 0.0
    env.state["print_manager_distance"][0] = 999.0
    env.state["print_manager_last_pos"][0] = env.state["pos"][0]
    surface.stack.reset_rows(env, np.asarray([True], dtype=bool))
    surface.stack.update(env)
    warmed_stats = surface.stack.dirty_render_stats()
    assert warmed_stats["hits"] >= 1

    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0] = np.asarray([8.0, 16.0], dtype=np.float64)
    env.state["print_manager_distance"][0] = 999.0
    env.state["print_manager_last_pos"][0] = env.state["pos"][0]

    step = surface.step(np.asarray([[1, 1]], dtype=np.int16))
    terminal_stats = step.info["visual_stack_dirty_render_stats"]

    np.testing.assert_array_equal(step.done, np.asarray([True], dtype=bool))
    assert terminal_stats["attempts"] > warmed_stats["attempts"]
    assert (
        terminal_stats["hits"] > warmed_stats["hits"]
        or terminal_stats["fallbacks"] > warmed_stats["fallbacks"]
    )
    np.testing.assert_array_equal(step.final_observation_row_mask, np.asarray([True]))
    np.testing.assert_array_equal(step.final_observation[0], step.observation[0])
    _assert_last_frame_matches_direct_source_state_render(surface, step, row=0)


def test_p4_terminal_final_observation_is_visual_stack_after_three_wall_deaths():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=4,
        seed=41,
        decision_ms=100.0,
        natural_bonus_spawn=False,
    )
    surface.reset(seed=41)
    _force_4p_player3_terminal_win(surface)
    joint_action = np.ones((1, 4), dtype=np.int16)

    step = surface.step(joint_action)

    _assert_step_visual_contract(surface, step, joint_action)
    np.testing.assert_array_equal(step.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(step.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(step.truncated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(step.live_mask, np.zeros((1, 4), dtype=bool))
    np.testing.assert_array_equal(step.policy_env_row, np.asarray([], dtype=np.int32))
    np.testing.assert_array_equal(step.policy_player, np.asarray([], dtype=np.int16))
    np.testing.assert_array_equal(step.policy_action_mask, np.zeros((0, 3), dtype=bool))
    np.testing.assert_array_equal(step.legal_action_mask, np.zeros((1, 4, 3), dtype=bool))
    np.testing.assert_array_equal(
        step.info["alive"],
        np.asarray([[False, False, False, True]], dtype=bool),
    )
    assert step.info["round_winner_ids"] == [[3]]

    expected_reward = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(step.reward, expected_reward)
    np.testing.assert_array_equal(step.final_reward_map, expected_reward)
    np.testing.assert_array_equal(step.info["final_reward_map"], expected_reward)
    assert step.final_observation.shape == (1, 4, 4, 64, 64)
    assert step.final_observation.dtype == np.float32
    np.testing.assert_array_equal(step.final_observation_row_mask, np.asarray([True]))
    np.testing.assert_array_equal(step.final_observation, step.info["final_observation"])
    np.testing.assert_array_equal(step.final_observation[0], step.observation[0])
    _assert_last_frame_matches_direct_source_state_render(surface, step, row=0)
    assert int(np.count_nonzero(step.final_observation)) > 0
    assert step.info["final_observation_policy"]["schema_id"] == (
        FINAL_VISUAL_OBSERVATION_POLICY_ID
    )
    assert step.info["final_observation_policy"]["metadata_only"] is False
    assert step.info["underlying_final_observation_shape"] == (1, 4, 6)
    assert step.final_observation.shape != step.info["underlying_final_observation_shape"]


def test_fast_gray64_direct_is_rejected_for_trainer_surface():
    with pytest.raises(ValueError, match="profile-only"):
        SourceStateMultiplayerTrainerSurface(
            batch_size=1,
            player_count=2,
            trail_render_mode=STACK_RENDER_MODE_FAST_GRAY64_DIRECT,
        )


def test_body_circles_fast_is_explicit_approximate_mode():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        trail_render_mode=TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
        natural_bonus_spawn=False,
    )

    step = surface.reset(seed=33)

    assert step.info["trail_render_mode"] == TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    assert step.info["bonus_render_mode"] == BONUS_RENDER_MODE_BROWSER_SPRITES
    assert step.info["visual_observation_is_approximation"] is True
    assert step.info["approximate_trail_render_mode"] is True
    assert step.info["trail_renderer_is_approximation"] is True
    assert step.info["render_metadata"]["bonus_renderer_is_approximation"] is False
    assert step.info["visual_stack_dirty_render_stats"]["enabled"] is False
    assert "explicit approximate mode" in step.info["approximation_reason"]


def test_profile_no_death_mode_is_preserved_and_labeled_not_source_fidelity():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        death_mode=vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
        natural_bonus_spawn=False,
    )

    step = surface.reset(seed=77)

    assert step.info["death_mode"] == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
    assert step.info["death_suppression_for_profile"] is True
    assert step.info["death_suppression_claim"] == "profile_only_not_source_fidelity"
    assert step.info["source_state_backed"] is True
