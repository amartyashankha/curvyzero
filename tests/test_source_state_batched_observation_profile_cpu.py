import numpy as np
import pytest

from curvyzero.env.observation_surface_contract import POLICY_OBSERVATION_BACKEND_GPU
from curvyzero.env.observation_surface_contract import POLICY_OBSERVATION_BACKEND_CPU
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.training import exploration_bonus as xb
from curvyzero.training.source_state_batched_observation_profile import (
    SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
    SOURCE_STATE_BATCHED_OBSERVATION_OTHER_LUMA,
    SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
    SOURCE_STATE_BATCHED_OBSERVATION_SELF_LUMA,
    SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID,
    SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS,
    SourceStateBatchedObservationProfileFacade,
    SourceStateBatchedRenderRequest,
    SourceStateBatchedRenderResult,
    source_state_controlled_player_palette,
)


def _expected_last_slots(facade: SourceStateBatchedObservationProfileFacade) -> np.ndarray:
    frames = np.empty((facade.batch_size, 64, 64), dtype=np.float32)
    for row, controlled_player in enumerate(facade.controlled_players):
        raw = render_source_state_canvas_gray64(
            facade.state,
            row=row,
            player_rgb=source_state_controlled_player_palette(
                facade.state,
                row=row,
                controlled_player=int(controlled_player),
            ),
        )
        frames[row] = raw[0].astype(np.float32) / np.float32(255.0)
    return frames


def _expected_all_player_last_slots(facade: SourceStateBatchedObservationProfileFacade) -> np.ndarray:
    frames = np.empty((facade.batch_size, facade.player_count, 64, 64), dtype=np.float32)
    for row in range(facade.batch_size):
        for controlled_player in range(facade.player_count):
            raw = render_source_state_canvas_gray64(
                facade.state,
                row=row,
                player_rgb=source_state_controlled_player_palette(
                    facade.state,
                    row=row,
                    controlled_player=controlled_player,
                ),
            )
            frames[row, controlled_player] = raw[0].astype(np.float32) / np.float32(255.0)
    return frames


def _assert_telemetry_slots(info: dict) -> None:
    telemetry = info["telemetry"]
    assert set(SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS) <= set(telemetry)
    for field in SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS:
        assert isinstance(telemetry[field], float)
        assert telemetry[field] >= 0.0


class _RowMajorRecordingRenderer:
    backend_name = POLICY_OBSERVATION_BACKEND_CPU

    def __init__(self) -> None:
        self.rows: np.ndarray | None = None
        self.players: np.ndarray | None = None

    def render(self, request: SourceStateBatchedRenderRequest) -> SourceStateBatchedRenderResult:
        self.rows = request.row_indices.copy()
        self.players = request.controlled_players.copy()
        for output_row, (row, player) in enumerate(zip(self.rows, self.players, strict=True)):
            request.out[output_row, 0].fill(int(row) * 16 + int(player))
        telemetry = {field: 0.0 for field in SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS}
        return SourceStateBatchedRenderResult(frames=request.out, telemetry=telemetry)


class _ProfileGpuCandidateRenderer(_RowMajorRecordingRenderer):
    backend_name = SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND


def test_reset_owns_batched_rows_and_updates_stack_from_cpu_oracle():
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=3,
        controlled_players=[0, 1, 0],
        seed=123,
    )

    step = facade.reset()

    assert step.observation.shape == (3, 4, 64, 64)
    assert step.observation.dtype == np.float32
    np.testing.assert_array_equal(step.observation[:, :3], np.zeros((3, 3, 64, 64)))
    np.testing.assert_allclose(step.observation[:, -1], _expected_last_slots(facade))
    assert step.final_observation is None
    assert step.info["profile_only"] is True
    assert step.info["stock_lightzero_integrated"] is False
    assert step.info["trainer_defaults_changed"] is False
    assert step.info["contract"]["calls_train_muzero"] is False
    assert step.info["contract"]["future_gpu_render_boundary"]["not_implemented_here"] is True
    _assert_telemetry_slots(step.info)


def test_default_player_view_mode_keeps_controlled_row_shape():
    facade = SourceStateBatchedObservationProfileFacade(batch_size=2, seed=321)

    step = facade.reset()

    assert step.observation.shape == (2, 4, 64, 64)
    assert facade.observation.shape == (2, 4, 64, 64)
    assert step.info["contract"]["player_view_mode"] == "controlled_rows"
    assert step.info["contract"]["observation_shape"] == [2, 4, 64, 64]


def test_both_players_view_updates_stacks_in_row_major_render_order():
    renderer = _RowMajorRecordingRenderer()
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=2,
        player_count=3,
        controlled_players=[2, 1],
        player_view_mode=SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
        renderer=renderer,
        seed=11,
    )

    step = facade.reset()

    assert step.observation.shape == (2, 3, 4, 64, 64)
    np.testing.assert_array_equal(renderer.rows, np.asarray([0, 0, 0, 1, 1, 1]))
    np.testing.assert_array_equal(renderer.players, np.asarray([0, 1, 2, 0, 1, 2]))
    np.testing.assert_array_equal(step.observation[:, :, :3], np.zeros((2, 3, 3, 64, 64)))
    expected_last = np.asarray([[0, 1, 2], [16, 17, 18]], dtype=np.float32) / np.float32(255.0)
    np.testing.assert_allclose(step.observation[:, :, -1, 0, 0], expected_last)
    assert step.info["contract"]["observation_shape"] == [2, 3, 4, 64, 64]
    assert step.info["contract"]["action_controlled_players"] == [2, 1]
    assert (
        step.info["contract"]["future_gpu_render_boundary"]["input_control"]
        == "render_controlled_players per output row"
    )
    assert (
        step.info["contract"]["future_gpu_render_boundary"]["render_order"]
        == "row-major [(row0,p0), (row0,p1), ...] for both_players"
    )
    assert step.info["contract"]["future_gpu_render_boundary"]["output_shape"] == [6, 1, 64, 64]
    _assert_telemetry_slots(step.info)


def test_both_players_cpu_oracle_matches_direct_per_player_renders():
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=2,
        player_count=2,
        player_view_mode=SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
        seed=13,
    )

    step = facade.reset()

    np.testing.assert_allclose(step.observation[:, :, -1], _expected_all_player_last_slots(facade))


def test_step_shifts_fifo_stack_and_respects_per_row_controlled_player_actions():
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=2,
        controlled_players=[0, 1],
        seed=7,
    )
    reset = facade.reset()
    reset_last = reset.observation[:, -1].copy()

    step = facade.step(actions=np.asarray([0, 2]), other_actions=np.asarray([2, 0]))

    np.testing.assert_allclose(step.observation[:, -2], reset_last)
    np.testing.assert_allclose(step.observation[:, -1], _expected_last_slots(facade))
    np.testing.assert_array_equal(
        step.info["joint_actions"],
        np.asarray([[0, 2], [0, 2]], dtype=np.int16),
    )
    _assert_telemetry_slots(step.info)


def test_both_players_step_shifts_fifo_stack_for_all_player_views():
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=2,
        player_count=2,
        controlled_players=[0, 1],
        player_view_mode=SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
        seed=17,
    )
    reset = facade.reset()
    reset_last = reset.observation[:, :, -1].copy()

    step = facade.step(actions=np.asarray([0, 2]), other_actions=np.asarray([2, 0]))

    np.testing.assert_allclose(step.observation[:, :, -2], reset_last)
    np.testing.assert_allclose(step.observation[:, :, -1], _expected_all_player_last_slots(facade))
    np.testing.assert_array_equal(
        step.info["joint_actions"],
        np.asarray([[0, 2], [0, 2]], dtype=np.int16),
    )
    _assert_telemetry_slots(step.info)


def test_terminal_step_captures_final_stack_before_any_reset():
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=2,
        seed=5,
        max_ticks=1,
    )
    facade.reset()

    step = facade.step(SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID)

    np.testing.assert_array_equal(step.done, np.asarray([True, True]))
    assert step.final_observation is not None
    assert step.final_observation.shape == step.observation.shape
    np.testing.assert_array_equal(step.final_observation, step.observation)
    assert step.info["telemetry"]["final_obs_sec"] >= 0.0
    _assert_telemetry_slots(step.info)


def test_both_players_terminal_info_exposes_stacked_final_observation_contract():
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=2,
        player_count=2,
        player_view_mode=SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
        seed=19,
        max_ticks=1,
    )
    facade.reset()

    step = facade.step(SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID)

    assert step.final_observation is not None
    assert step.final_observation.shape == (2, 2, 4, 64, 64)
    np.testing.assert_array_equal(step.info["final_observation"], step.final_observation)
    np.testing.assert_array_equal(
        step.info["final_observation_row_mask"],
        np.asarray([True, True]),
    )
    np.testing.assert_array_equal(
        step.info["final_observation_rows"],
        np.asarray([0, 1], dtype=np.int32),
    )
    assert step.info["final_observation_policy"]["observation_shape"] == [2, 2, 4, 64, 64]
    assert step.info["final_observation_policy"]["player_view_mode"] == "both_players"
    assert step.info["final_observation_policy"]["autoreset"] == "not modeled"


def test_facade_rejects_scalar_lab_gpu_backend_until_a_real_batched_boundary_exists():
    with pytest.raises(ValueError, match="cpu_oracle"):
        SourceStateBatchedObservationProfileFacade(
            batch_size=1,
            observation_backend=POLICY_OBSERVATION_BACKEND_GPU,
        )


def test_facade_accepts_explicit_profile_gpu_candidate_renderer_without_cpu_fallback():
    renderer = _ProfileGpuCandidateRenderer()
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=2,
        player_count=2,
        player_view_mode=SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
        observation_backend=SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
        renderer=renderer,
        seed=23,
    )

    step = facade.reset()

    assert facade.renderer is renderer
    assert step.info["observation_backend"] == SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND
    assert (
        step.info["contract"]["profile_gpu_candidate_backend"]
        == SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND
    )
    np.testing.assert_array_equal(renderer.rows, np.asarray([0, 0, 1, 1]))
    np.testing.assert_array_equal(renderer.players, np.asarray([0, 1, 0, 1]))
    expected_last = np.asarray([[0, 1], [16, 17]], dtype=np.float32) / np.float32(255.0)
    np.testing.assert_allclose(step.observation[:, :, -1, 0, 0], expected_last)


def test_profile_gpu_candidate_requires_explicit_renderer_to_avoid_hidden_cpu_fallback():
    with pytest.raises(ValueError, match="no hidden CPU fallback"):
        SourceStateBatchedObservationProfileFacade(
            batch_size=1,
            observation_backend=SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
        )


def test_batched_controlled_rows_stack_feeds_rnd_latest_gray64_input():
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=3,
        controlled_players=[0, 1, 0],
        seed=29,
    )
    facade.reset()
    step = facade.step(actions=np.asarray([0, 1, 2]), other_actions=np.asarray([2, 1, 0]))

    target_reward = np.zeros((facade.batch_size, 1), dtype=np.float32)
    rnd_input = xb.extract_policy_gray64_latest_for_rnd(
        step.observation.reshape(facade.batch_size, 1, 4, 64, 64),
        target_reward,
    )

    assert rnd_input.shape == (facade.batch_size, 1, 64, 64)
    np.testing.assert_array_equal(rnd_input, step.observation[:, -1:, :, :])


def test_batched_both_player_stacks_can_materialize_rnd_inputs_per_player_view():
    facade = SourceStateBatchedObservationProfileFacade(
        batch_size=2,
        player_count=2,
        player_view_mode=SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
        seed=31,
    )
    facade.reset()
    step = facade.step(actions=np.asarray([0, 2]), other_actions=np.asarray([2, 0]))

    # This profile-only proof treats the player-view axis like LightZero's
    # unroll axis so the extractor sees the same stack layout without changing
    # production replay semantics.
    target_reward = np.zeros((facade.batch_size, facade.player_count, 1), dtype=np.float32)
    rnd_input = xb.extract_policy_gray64_latest_for_rnd(step.observation, target_reward)

    assert rnd_input.shape == (facade.batch_size * facade.player_count, 1, 64, 64)
    np.testing.assert_array_equal(
        rnd_input,
        step.observation.reshape(facade.batch_size * facade.player_count, 4, 64, 64)[
            :,
            -1:,
            :,
            :,
        ],
    )


def test_facade_rejects_invalid_player_view_mode():
    with pytest.raises(ValueError, match="player_view_mode"):
        SourceStateBatchedObservationProfileFacade(
            batch_size=1,
            player_view_mode="single_view",
        )


def test_palette_tracks_avatar_color_indices_not_player_indices():
    state = {
        "pos": np.zeros((1, 2, 2), dtype=np.float32),
        "avatar_color": np.asarray([[7, 3]], dtype=np.int16),
    }

    palette = source_state_controlled_player_palette(
        state,
        row=0,
        controlled_player=1,
    )

    assert palette[3] == (
        SOURCE_STATE_BATCHED_OBSERVATION_SELF_LUMA,
        SOURCE_STATE_BATCHED_OBSERVATION_SELF_LUMA,
        SOURCE_STATE_BATCHED_OBSERVATION_SELF_LUMA,
    )
    assert palette[7] == (
        SOURCE_STATE_BATCHED_OBSERVATION_OTHER_LUMA,
        SOURCE_STATE_BATCHED_OBSERVATION_OTHER_LUMA,
        SOURCE_STATE_BATCHED_OBSERVATION_OTHER_LUMA,
    )
