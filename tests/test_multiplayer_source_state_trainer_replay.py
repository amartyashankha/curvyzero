import math

import numpy as np
import pytest

from curvyzero.env import vector_runtime
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    POLICY_ROW_STORAGE,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_ARRAY_KEYS,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_KIND,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_NON_CLAIMS,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)


def test_records_reset_step_terminal_arrays_and_preserves_final_visual_observation():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=11,
        decision_ms=100.0,
        natural_bonus_spawn=False,
    )
    reset_step = surface.reset(seed=11)
    nonterminal_step = surface.step(np.asarray([[1, 1]], dtype=np.int16))
    assert not bool(nonterminal_step.done.any())
    terminal_step = _terminal_step(surface)

    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    recorder.record(reset_step, rng_history_ref="rng-reset", source_ref="reset")
    recorder.record(nonterminal_step, rng_history_ref="rng-step", source_ref="step")
    recorder.record(terminal_step, rng_history_ref="rng-terminal", source_ref="terminal")
    chunk = recorder.build_chunk()

    assert set(SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_ARRAY_KEYS) <= set(
        chunk.arrays
    )
    assert chunk.arrays["observation"].shape == (3, 1, 2, 4, 64, 64)
    assert chunk.arrays["legal_action_mask"].shape == (3, 1, 2, 3)
    assert chunk.arrays["joint_action"].shape == (3, 1, 2)
    assert chunk.arrays["reward"].shape == (3, 1, 2)
    assert chunk.arrays["done"].shape == (3, 1)
    assert chunk.arrays["terminated"].shape == (3, 1)
    assert chunk.arrays["truncated"].shape == (3, 1)
    assert chunk.arrays["final_observation"].shape == (3, 1, 2, 4, 64, 64)
    assert chunk.arrays["final_observation_row_mask"].shape == (3, 1)
    assert chunk.arrays["final_reward_map"].shape == (3, 1, 2)
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([False, False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_observation"][2],
        terminal_step.final_observation,
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_reward_map"][2],
        terminal_step.final_reward_map,
    )
    assert int(np.count_nonzero(chunk.arrays["final_observation"][2])) > 0
    assert chunk.metadata["record_count"] == 3
    assert chunk.metadata["closed_by_terminal"] is True


def test_metadata_identifies_source_state_array_replay_without_lightzero_claims():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=23,
        natural_bonus_spawn=False,
    )
    step = surface.reset(seed=23)
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    recorder.record(step)

    chunk = recorder.build_chunk()
    metadata = chunk.metadata

    assert metadata["replay_schema_id"] == SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID
    assert metadata["replay_kind"] == SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_KIND
    assert metadata["surface_schema_id"] == MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID
    assert metadata["metadata_only"] is False
    assert metadata["metadata_only_replay_claim"] is False
    assert metadata["multiplayer_metadata_replay_claim"] is False
    assert metadata["trainer_replay_claim"] is True
    assert metadata["lightzero_training_claim"] is False
    assert metadata["lightzero_native_game_segment_claim"] is False
    assert metadata["native_game_segment_claim"] is False
    assert metadata["native_game_segment_integration_claim"] is False
    assert metadata["game_segment_integration_claim"] is False
    assert metadata["non_claims"] == SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_NON_CLAIMS
    assert "not_lightzero_training_integration" in metadata["non_claims"]
    assert metadata["policy_row_storage"] == POLICY_ROW_STORAGE


def test_recorded_arrays_are_copied_not_aliases_to_mutable_step_arrays():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=31,
        natural_bonus_spawn=False,
    )
    step = surface.reset(seed=31)
    expected_observation = step.observation.copy()
    expected_legal_action_mask = step.legal_action_mask.copy()
    expected_policy_observation = step.policy_observation.copy()
    expected_policy_env_row = step.policy_env_row.copy()

    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    recorder.record(step)
    step.observation[...] = -7.0
    step.legal_action_mask[...] = False
    step.policy_observation[...] = 42.0
    step.policy_env_row[...] = 99

    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(chunk.arrays["observation"][0], expected_observation)
    np.testing.assert_array_equal(
        chunk.arrays["legal_action_mask"][0],
        expected_legal_action_mask,
    )
    np.testing.assert_array_equal(
        chunk.policy_rows[0]["policy_observation"],
        expected_policy_observation,
    )
    np.testing.assert_array_equal(
        chunk.policy_rows[0]["policy_env_row"],
        expected_policy_env_row,
    )


def test_variable_policy_rows_are_stored_per_record_with_env_player_maps():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=41,
        decision_ms=100.0,
        natural_bonus_spawn=False,
    )
    reset_step = surface.reset(seed=41)
    terminal_step = _terminal_step(surface)
    recorder = SourceStateMultiplayerTrainerReplayRecorder()

    recorder.record(reset_step)
    recorder.record(terminal_step)
    chunk = recorder.build_chunk()

    assert chunk.metadata["policy_row_storage"] == POLICY_ROW_STORAGE
    assert len(chunk.policy_rows) == 2
    assert chunk.policy_rows[0]["policy_observation"].shape == (2, 4, 64, 64)
    assert chunk.policy_rows[0]["policy_action_mask"].shape == (2, 3)
    np.testing.assert_array_equal(
        chunk.policy_rows[0]["policy_env_row"],
        reset_step.policy_env_row,
    )
    np.testing.assert_array_equal(
        chunk.policy_rows[0]["policy_player"],
        reset_step.policy_player,
    )
    assert chunk.policy_rows[1]["policy_observation"].shape == (0, 4, 64, 64)
    assert chunk.policy_rows[1]["policy_action_mask"].shape == (0, 3)
    np.testing.assert_array_equal(
        chunk.policy_rows[1]["policy_env_row"],
        terminal_step.policy_env_row,
    )
    np.testing.assert_array_equal(
        chunk.policy_rows[1]["policy_player"],
        terminal_step.policy_player,
    )
    assert chunk.records[0]["policy_row_count"] == 2
    assert chunk.records[1]["policy_row_count"] == 0


def test_replay_rejects_policy_rows_that_do_not_match_live_mask():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=51,
        natural_bonus_spawn=False,
    )
    step = surface.reset(seed=51)
    step.policy_player[:] = step.policy_player[::-1]
    recorder = SourceStateMultiplayerTrainerReplayRecorder()

    with pytest.raises(ReplayCompatibilityError, match="policy_player"):
        recorder.record(step)


def test_replay_preserves_profile_no_death_metadata_without_source_claim():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=61,
        death_mode=vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
        natural_bonus_spawn=False,
    )
    step = surface.reset(seed=61)
    recorder = SourceStateMultiplayerTrainerReplayRecorder()

    record = recorder.record(step)
    chunk = recorder.build_chunk()

    assert record["death_mode"] == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
    assert record["death_suppression_for_profile"] is True
    assert record["death_suppression_claim"] == "profile_only_not_source_fidelity"
    assert chunk.metadata["death_mode"] == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
    assert chunk.metadata["death_suppression_for_profile"] is True
    assert chunk.metadata["death_suppression_claim"] == (
        "profile_only_not_source_fidelity"
    )
    assert record["original_curvytron_behavior_claim"] is False
    assert record["source_fidelity_claim"] == "restricted_by_project_training_helper"
    assert chunk.metadata["original_curvytron_behavior_claim"] is False
    assert chunk.metadata["source_fidelity_claim"] == (
        "restricted_by_project_training_helper"
    )
    assert chunk.metadata["lightzero_training_claim"] is False


def test_replay_preserves_death_immunity_metadata_without_source_claim():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=71,
        death_immunity_player_ids=(1,),
        natural_bonus_spawn=False,
    )
    step = surface.reset(seed=71)
    recorder = SourceStateMultiplayerTrainerReplayRecorder()

    record = recorder.record(step)
    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(
        step.info["death_immunity_player_ids"],
        np.asarray([1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        step.info["death_immunity_mask"],
        np.asarray([[False, True]], dtype=bool),
    )
    assert record["death_immunity_player_ids"] == [1]
    assert record["death_immunity_mask"] == [[False, True]]
    assert record["death_immunity_diagnostic"] is True
    assert record["death_immunity_claim"] == "diagnostic_not_source_faithful"
    assert record["original_curvytron_behavior_claim"] is False
    assert record["source_fidelity_claim"] == "restricted_by_project_training_helper"
    assert chunk.metadata["death_immunity_player_ids"] == [1]
    assert chunk.metadata["death_immunity_mask"] == [[False, True]]
    assert chunk.metadata["death_immunity_diagnostic"] is True
    assert chunk.metadata["death_immunity_claim"] == "diagnostic_not_source_faithful"
    assert chunk.metadata["original_curvytron_behavior_claim"] is False
    assert chunk.metadata["source_fidelity_claim"] == (
        "restricted_by_project_training_helper"
    )


def _terminal_step(surface):
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
    np.testing.assert_array_equal(
        step.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    return step
