import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import JOINT_ACTION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_SHAPE,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
)
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import render_source_state_rgb_canvas_like
from curvyzero.env.vector_visual_observation import rgb_canvas_like_to_gray64
from curvyzero.training.multiplayer_replay_v0 import MultiplayerMetadataReplayRecorder


SEED = 777
DECISION_SOURCE_FRAMES = 12
SOURCE_MAX_STEPS = 64
STRAIGHT_STRAIGHT = np.asarray([[1, 1]], dtype=np.int16)


def test_2p_vector_multiplayer_product_path_visual_bonus_terminal_and_replay():
    """End-to-end public 2P runtime proof.

    This stays on ``VectorMultiplayerEnv`` directly. The source-state LightZero
    wrapper stack/final visual frame remains a wrapper-surface follow-up; the
    direct public runtime exposes terminal debug metadata final observations.
    """

    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=SEED,
        decision_source_frames=DECISION_SOURCE_FRAMES,
        max_ticks=SOURCE_MAX_STEPS * DECISION_SOURCE_FRAMES,
        natural_bonus_spawn=False,
        body_capacity=2048,
        event_capacity=64,
        random_tape_capacity=32,
    )
    reset_batch = env.reset(seed=SEED)

    assert type(env) is VectorMultiplayerEnv
    assert env.source_frame_decision is True
    assert env.decision_source_frames == DECISION_SOURCE_FRAMES
    assert reset_batch.info["source_frame_decision"] is True
    assert reset_batch.info["decision_source_frames"] == DECISION_SOURCE_FRAMES
    assert reset_batch.info["bonus_support_mode"] == "disabled"
    assert reset_batch.info["bonus_support"]["natural_bonus_spawn"] is False

    reset_gray64 = _assert_raw_rgb_to_gray64_product_visual_path(env)

    player_0_pos = env.state["pos"][0, 0].copy()
    seed_info = env.seed_active_bonus(
        row=0,
        bonus_type="BonusGameClear",
        x=float(player_0_pos[0]),
        y=float(player_0_pos[1]),
        radius=3.0,
    )
    assert seed_info["bonus_type"] == "BonusGameClear"
    assert seed_info["natural_bonus_spawn"] is False

    stale_point = np.asarray([10.0, 10.0], dtype=np.float64)
    _seed_stale_body_and_visual_trail(env, stale_point=stale_point)
    seeded_gray64 = _assert_raw_rgb_to_gray64_product_visual_path(env)
    assert int(np.count_nonzero(seeded_gray64 != reset_gray64)) > 0

    recorder = MultiplayerMetadataReplayRecorder()
    previous_tick = int(env.state["tick"][0])
    first_batch = env.step(STRAIGHT_STRAIGHT)
    first_records = recorder.record_batch(
        first_batch,
        rng_history_ref="seed-777-product-path",
    )

    assert first_batch.done.tolist() == [False]
    assert int(env.state["tick"][0]) - previous_tick == DECISION_SOURCE_FRAMES
    assert first_batch.info["bonus_support_mode"] == "seeded"
    assert first_batch.info["bonus_support"]["natural_bonus_spawn"] is False
    assert first_batch.info["bonus_catch_count_step"].tolist() == [[1, 0]]
    assert first_batch.info["step_counters"]["bonus_game_clear_catches"] == 1
    assert first_batch.info["step_counters"]["bonus_stack_appends"] == 0
    assert not _has_active_body_point(env, stale_point=stale_point)
    assert not _has_active_visual_trail_point(env, stale_point=stale_point)
    np.testing.assert_array_equal(
        first_batch.info["action_sidecar"]["player_action"],
        STRAIGHT_STRAIGHT,
    )
    assert first_batch.info["action_sidecar"]["joint_action_schema_id"] == (
        JOINT_ACTION_SCHEMA_ID
    )
    assert first_records[0]["bonus_support_mode"] == "seeded"
    assert first_records[0]["natural_bonus_spawn"] is False

    step_count = 1
    terminal_batch = None
    terminal_delta = None
    while step_count < 80:
        previous_tick = int(env.state["tick"][0])
        batch = env.step(STRAIGHT_STRAIGHT)
        step_count += 1
        delta = int(env.state["tick"][0]) - previous_tick
        recorder.record_batch(batch, rng_history_ref="seed-777-product-path")
        if bool(batch.done[0]):
            terminal_batch = batch
            terminal_delta = delta
            break
        assert delta == DECISION_SOURCE_FRAMES

    assert terminal_batch is not None
    assert step_count == 15
    assert terminal_delta == 9
    assert terminal_delta < DECISION_SOURCE_FRAMES
    assert int(env.state["tick"][0]) == 177
    assert terminal_batch.done.tolist() == [True]
    assert terminal_batch.terminated.tolist() == [True]
    assert terminal_batch.truncated.tolist() == [False]
    assert terminal_batch.info["terminal_reason_name"].tolist() == ["round_survivor_win"]
    assert terminal_batch.info["winner_ids"] == [[1]]
    assert terminal_batch.info["loser_ids"] == [[0]]
    assert terminal_batch.info["death_count"].tolist() == [1]
    assert terminal_batch.info["death_player"][0, 0] == 0
    np.testing.assert_array_equal(
        terminal_batch.reward,
        np.asarray([[-1.0, 1.0]], dtype=np.float32),
    )
    assert env.state["alive"][0, :2].tolist() == [False, True]

    assert terminal_batch.final_observation is not None
    assert terminal_batch.final_reward is not None
    assert terminal_batch.final_observation.shape == terminal_batch.observation.shape
    np.testing.assert_array_equal(
        terminal_batch.info["final_observation_row_mask"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["final_reward_row_mask"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_rows"],
        np.asarray([0], dtype=np.int32),
    )
    assert terminal_batch.info["final_observation_row_policy"]["source_claim"] == (
        "debug_metadata_only_public_terminal_rows/v0"
    )
    _assert_raw_rgb_to_gray64_product_visual_path(env)

    terminal_record = recorder.records[-1]
    chunk = recorder.build_chunk()
    assert recorder.closed_by_terminal is True
    assert chunk.metadata["closed_by_terminal"] is True
    assert chunk.metadata["recorded_batch_count"] == step_count
    assert terminal_record["done"] is True
    assert terminal_record["terminated"] is True
    assert terminal_record["truncated"] is False
    assert terminal_record["terminal_rows"] == [0]
    assert terminal_record["final_observation_row_mask"] == [True]
    assert terminal_record["final_reward_row_mask"] == [True]
    assert terminal_record["final_observation_row_policy"]["source_claim"] == (
        "debug_metadata_only_public_terminal_rows/v0"
    )
    assert terminal_record["metadata_only"] is True
    assert terminal_record["trainer_replay_claim"] is False


def _assert_raw_rgb_to_gray64_product_visual_path(
    env: VectorMultiplayerEnv,
) -> np.ndarray:
    raw = render_source_state_rgb_canvas_like(env.state, row=0)
    assert raw.shape == (
        SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        3,
    )
    assert raw.dtype == np.uint8

    gray64_from_raw = rgb_canvas_like_to_gray64(raw)
    assert gray64_from_raw.shape == SOURCE_STATE_CANVAS_GRAY64_SHAPE
    assert gray64_from_raw.dtype == np.uint8
    np.testing.assert_array_equal(
        gray64_from_raw,
        render_source_state_canvas_gray64(env.state, row=0),
    )
    return gray64_from_raw.copy()


def _seed_stale_body_and_visual_trail(
    env: VectorMultiplayerEnv,
    *,
    stale_point: np.ndarray,
) -> None:
    state = env.state
    state["world_active"][0] = True
    state["body_active"][0, 0] = True
    state["body_pos"][0, 0] = stale_point
    state["body_radius"][0, 0] = 1.0
    state["body_owner"][0, 0] = 1
    state["body_num"][0, 0] = 0
    state["body_insert_tick"][0, 0] = int(state["tick"][0])
    state["body_insert_kind"][0, 0] = vector_runtime.BODY_KIND_NORMAL
    state["body_write_cursor"][0] = 1
    state["world_body_count"][0] = 1
    state["body_count"][0, 1] = 1

    state["visual_trail_active"][0, 0] = True
    state["visual_trail_pos"][0, 0] = stale_point
    state["visual_trail_radius"][0, 0] = 1.0
    state["visual_trail_owner"][0, 0] = 1


def _has_active_body_point(
    env: VectorMultiplayerEnv,
    *,
    stale_point: np.ndarray,
) -> bool:
    state = env.state
    stale_body = (
        state["body_active"][0]
        & (state["body_owner"][0] == 1)
        & np.isclose(state["body_pos"][0, :, 0], stale_point[0])
        & np.isclose(state["body_pos"][0, :, 1], stale_point[1])
    )
    return bool(stale_body.any())


def _has_active_visual_trail_point(
    env: VectorMultiplayerEnv,
    *,
    stale_point: np.ndarray,
) -> bool:
    state = env.state
    stale_visual = (
        state["visual_trail_active"][0]
        & (state["visual_trail_owner"][0] == 1)
        & np.isclose(state["visual_trail_pos"][0, :, 0], stale_point[0])
        & np.isclose(state["visual_trail_pos"][0, :, 1], stale_point[1])
    )
    return bool(stale_visual.any())
