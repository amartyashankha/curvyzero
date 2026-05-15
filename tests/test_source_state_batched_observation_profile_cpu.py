import numpy as np
import pytest

from curvyzero.env.observation_surface_contract import POLICY_OBSERVATION_BACKEND_GPU
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.training.source_state_batched_observation_profile import (
    SOURCE_STATE_BATCHED_OBSERVATION_OTHER_LUMA,
    SOURCE_STATE_BATCHED_OBSERVATION_SELF_LUMA,
    SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID,
    SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS,
    SourceStateBatchedObservationProfileFacade,
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


def _assert_telemetry_slots(info: dict) -> None:
    telemetry = info["telemetry"]
    assert set(SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS) <= set(telemetry)
    for field in SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS:
        assert isinstance(telemetry[field], float)
        assert telemetry[field] >= 0.0


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


def test_facade_rejects_lab_gpu_backend_until_a_real_batched_boundary_exists():
    with pytest.raises(ValueError, match="cpu_oracle"):
        SourceStateBatchedObservationProfileFacade(
            batch_size=1,
            observation_backend=POLICY_OBSERVATION_BACKEND_GPU,
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
