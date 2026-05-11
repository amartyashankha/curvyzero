import json
from pathlib import Path
import sys

import numpy as np
import pytest

from curvyzero.env import vector_multiplayer_observation as obs
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerBatch
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_multiplayer_env import PUBLIC_JOINT_ACTION_SIDECAR_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import RESET_PROVENANCE_POLICY_ID
from curvyzero.env.vector_multiplayer_env import SEEDED_BONUS_TYPE_NAMES
from curvyzero.training import multiplayer_replay_contract as multiplayer_contract
from curvyzero.training import multiplayer_replay_v0
from curvyzero.training import replay_chunk_v0 as replay


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402

SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenarios" / "environment"

_REQUIRED_MULTIPLAYER_FIELDS = {
    "public_env_contract_id",
    "env_impl_id",
    "ruleset_id",
    "rules_hash",
    "native_control_model_id",
    "trainer_control_wrapper_id",
    "decision_ms",
    "lifecycle_policy_id",
    "episode_id",
    "reset_episode_id",
    "reset_episode_id_policy",
    "round_id",
    "source_round_id",
    "source_round_id_policy",
    "episode_end_mode",
    "step_index",
    "tick_index",
    "elapsed_ms",
    "player_count",
    "player_ids",
    "source_player_ids",
    "present",
    "alive",
    "action_mask",
    "joint_action",
    "reward",
    "done",
    "terminated",
    "truncated",
    "score",
    "round_score",
    "death_order",
    "death_player",
    "death_count",
    "round_done",
    "match_done",
    "needs_reset",
    "terminal_reason",
    "truncation_reason",
    "winner",
    "round_winner",
    "match_winner",
    "draw",
    "winner_ids",
    "round_winner_ids",
    "match_winner_ids",
    "reset_seed",
    "reset_source",
    "random_tape_cursor",
    "random_tape_draw_count",
    "rng_history_ref",
    "action_sidecar",
    "observation_schema_id",
    "observation_schema_hash",
    "action_space_id",
    "action_space_hash",
    "reward_schema_id",
    "reward_schema_hash",
    "final_observation_policy",
    "metadata_only",
    "trainer_observation_claim",
    "trainer_replay_claim",
    "learned_observation_claim",
}


@pytest.mark.parametrize("player_count", [3, 4])
def test_strict_replay_v0_rejects_multiplayer_player_dimension(player_count):
    arrays = _strict_replay_arrays(player_count=player_count)

    with pytest.raises(replay.ReplayCompatibilityError, match="player_count"):
        replay.validate_replay_chunk_v0(arrays=arrays, metadata={})


@pytest.mark.parametrize("player_count", [3, 4])
def test_strict_replay_v0_rejects_metadata_claiming_multiplayer_player_count(
    player_count,
):
    arrays = _strict_replay_arrays(player_count=2)
    metadata = _strict_replay_metadata(arrays)
    metadata["player_count"] = player_count

    with pytest.raises(replay.ReplayCompatibilityError, match="player_count"):
        replay.validate_replay_chunk_v0(arrays=arrays, metadata=metadata)


def test_multiplayer_replay_required_metadata_fields_are_explicit():
    assert (
        set(multiplayer_contract.MULTIPLAYER_REPLAY_REQUIRED_METADATA_FIELDS)
        == _REQUIRED_MULTIPLAYER_FIELDS
    )


@pytest.mark.parametrize(
    "field",
    multiplayer_contract.MULTIPLAYER_REPLAY_REQUIRED_METADATA_FIELDS,
)
def test_multiplayer_replay_metadata_guard_rejects_missing_required_field(field):
    metadata = _multiplayer_metadata(player_count=3)
    metadata.pop(field)

    with pytest.raises(replay.ReplayCompatibilityError, match=field):
        multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_multiplayer_replay_metadata_guard_rejects_strict_1v1_observation_schema(
    player_count,
):
    metadata = _multiplayer_metadata(
        player_count=player_count,
        observation_schema_id="curvyzero_egocentric_rays/v0",
    )

    with pytest.raises(replay.ReplayCompatibilityError, match="observation_schema_id"):
        multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_multiplayer_replay_metadata_guard_accepts_metadata_only_schema(player_count):
    metadata = _multiplayer_metadata(
        player_count=player_count,
        observation_schema_id="curvyzero_debug_metadata_only/v0",
    )

    multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("reset_episode_id", 2, "reset_episode_id"),
        ("source_round_id", 2, "source_round_id"),
    ],
)
def test_multiplayer_replay_metadata_guard_rejects_identity_alias_mismatch(
    field,
    value,
    match,
):
    metadata = _multiplayer_metadata(player_count=3)
    metadata[field] = value

    with pytest.raises(replay.ReplayCompatibilityError, match=match):
        multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_multiplayer_replay_v0_packages_public_env_row_metadata(player_count):
    output = _public_env_output(player_count=player_count)

    record = multiplayer_replay_v0.build_multiplayer_replay_record_from_public_env_output(
        info=output["info"],
        action_mask=output["action_mask"],
        reward=output["reward"],
        done=output["done"],
        terminated=output["terminated"],
        truncated=output["truncated"],
        rng_history_ref="debug/random-tape-row-0.json",
    )
    chunk = multiplayer_replay_v0.build_multiplayer_replay_chunk_v0([record])

    assert record["record_schema_id"] == (
        multiplayer_replay_v0.MULTIPLAYER_REPLAY_RECORD_SCHEMA_ID
    )
    assert record["player_count"] == player_count
    assert record["lifecycle_policy_id"] == output["info"]["lifecycle_policy_id"]
    assert record["reset_episode_id"] == int(output["info"]["reset_episode_id"][0])
    assert record["source_round_id"] == int(output["info"]["source_round_id"][0])
    assert record["metadata_only"] is True
    assert record["trainer_observation_claim"] is False
    assert record["observation_schema_id"] == "curvyzero_debug_metadata_only/v0"
    assert record["rng_history_ref"] == "debug/random-tape-row-0.json"
    assert record["death_order"] == [player_count - 1]
    assert record["joint_action"] == [1 for _ in range(player_count)]
    assert record["action_sidecar"]["player_action"] == [1 for _ in range(player_count)]
    assert chunk.metadata["chunk_schema_id"] == (
        multiplayer_replay_v0.MULTIPLAYER_REPLAY_CHUNK_SCHEMA_ID
    )
    assert chunk.metadata["record_count"] == 1
    assert chunk.records == (record,)


def test_multiplayer_replay_v0_rejects_chunk_with_mixed_player_counts():
    record_3p = _multiplayer_metadata(player_count=3)
    record_4p = _multiplayer_metadata(player_count=4)

    with pytest.raises(replay.ReplayCompatibilityError, match="player_count"):
        multiplayer_replay_v0.build_multiplayer_replay_chunk_v0([record_3p, record_4p])


def test_multiplayer_replay_metadata_guard_accepts_optional_opponent_policy_sidecar():
    metadata = _multiplayer_metadata(player_count=3)
    metadata["opponent_policy_sidecar"] = {
        "schema_id": multiplayer_contract.MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID,
        "policy_id": "random_uniform",
        "policy_version": "v0",
        "seed": 1001,
        "actions": [0, 1, 2],
    }

    multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


def test_multiplayer_replay_metadata_guard_rejects_bad_opponent_policy_sidecar():
    metadata = _multiplayer_metadata(player_count=3)
    metadata["opponent_policy_sidecar"] = {
        "schema_id": multiplayer_contract.MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID,
        "policy_id": "random_uniform",
        "seed": 1001,
        "actions": [0, 1, 2],
    }

    with pytest.raises(replay.ReplayCompatibilityError, match="policy_version"):
        multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("trainer_observation_claim", True, "trainer_observation_claim"),
        ("trainer_replay_claim", True, "trainer_replay_claim"),
        ("learned_observation_claim", True, "learned_observation_claim"),
        ("final_observation_policy", "trainer_ready_final_obs/v1", "final_observation_policy"),
    ],
)
def test_multiplayer_replay_metadata_guard_rejects_false_trainer_ready_claims(
    field,
    value,
    match,
):
    metadata = _multiplayer_metadata(player_count=3)
    metadata[field] = value

    with pytest.raises(replay.ReplayCompatibilityError, match=match):
        multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


def test_multiplayer_replay_metadata_guard_rejects_inactive_legal_mask():
    metadata = _multiplayer_metadata(player_count=3)
    metadata["alive"] = [True, False, True]
    metadata["action_mask"] = [[True, True, True], [True, True, True], [True, True, True]]
    metadata["action_sidecar"]["player_action_mask"] = metadata["action_mask"]
    metadata["action_sidecar"]["action_required"] = [True, False, True]

    with pytest.raises(replay.ReplayCompatibilityError, match="inactive player"):
        multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


def test_multiplayer_replay_metadata_guard_rejects_mismatched_action_sidecar():
    metadata = _multiplayer_metadata(player_count=3)
    metadata["action_sidecar"]["player_action"] = [1, 2, 1]

    with pytest.raises(replay.ReplayCompatibilityError, match="player_action"):
        multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


def test_multiplayer_replay_metadata_guard_rejects_unnamed_action_sidecar():
    metadata = _multiplayer_metadata(player_count=3)
    metadata["action_sidecar"].pop("schema_id")

    with pytest.raises(replay.ReplayCompatibilityError, match="schema_id"):
        multiplayer_contract.validate_multiplayer_replay_metadata_guard(metadata)


def test_multiplayer_replay_v0_accepts_current_public_env_step_batch():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        seed=101,
        random_tape_capacity=16,
    )
    env.reset(seed=np.asarray([101], dtype=np.uint64))
    batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    record = multiplayer_replay_v0.build_multiplayer_replay_record_from_public_env_output(
        info=batch.info,
        action_mask=batch.action_mask,
        reward=batch.reward,
        done=batch.done,
        terminated=batch.terminated,
        truncated=batch.truncated,
        rng_history_ref="debug/current-public-env-row-0.json",
    )

    assert record["public_env_contract_id"] == batch.info["public_env_contract_id"]
    assert record["env_impl_id"] == batch.info["env_impl_id"]
    assert record["ruleset_id"] == batch.info["ruleset_id"]
    assert record["rules_hash"] == batch.info["rules_hash"]
    assert record["native_control_model_id"] == batch.info["native_control_model_id"]
    assert record["trainer_control_wrapper_id"] == (
        batch.info["trainer_control_wrapper_id"]
    )
    assert record["decision_ms"] == batch.info["decision_ms"]
    assert record["lifecycle_policy_id"] == batch.info["lifecycle_policy_id"]
    assert record["episode_id"] == int(batch.info["episode_id"][0])
    assert record["reset_episode_id"] == int(batch.info["reset_episode_id"][0])
    assert record["reset_episode_id_policy"] == batch.info["reset_episode_id_policy"]
    assert record["round_id"] == int(batch.info["round_id"][0])
    assert record["source_round_id"] == int(batch.info["source_round_id"][0])
    assert record["source_round_id_policy"] == batch.info["source_round_id_policy"]
    assert record["step_index"] == int(batch.info["step_index"][0])
    assert record["tick_index"] == int(batch.info["tick_index"][0])
    assert record["elapsed_ms"] == float(batch.info["elapsed_ms"][0])
    assert record["needs_reset"] == bool(batch.info["needs_reset"][0])
    assert record["truncation_reason"] is None
    assert record["round_winner_ids"] == []
    assert record["match_winner_ids"] == []
    assert record["observation_schema_hash"] == batch.info["observation_schema_hash"]
    assert record["action_space_hash"] == batch.info["action_space_hash"]
    assert record["reward_schema_hash"] == batch.info["reward_schema_hash"]
    assert record["trainer_replay_claim"] is False
    assert record["learned_observation_claim"] is False


def test_multiplayer_replay_v0_carries_real_public_env_reset_provenance_fields():
    fixture_ref = "scenarios/environment/source_lifecycle_spawn_rng_order_4p.json"
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=16,
    )
    env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        source_fixture_random_tape_values=_lifecycle_random_tape(
            "source_lifecycle_spawn_rng_order_4p.json"
        ),
        source_fixture_ref=fixture_ref,
        source_fixture_warmup_advance_ms=0.0,
    )
    batch = env.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))

    record = multiplayer_replay_v0.build_multiplayer_replay_record_from_public_env_output(
        info=batch.info,
        action_mask=batch.action_mask,
        reward=batch.reward,
        done=batch.done,
        terminated=batch.terminated,
        truncated=batch.truncated,
    )

    assert record["random_tape_source"] == "source_fixture_random_tape_values"
    assert record["random_tape_length"] == 12
    assert record["rng_impl_id"] == "source_fixture_random_tape_values/v0"
    assert record["rng_history_ref"] == "source_fixture_random_tape_values/v0"
    assert record["source_fixture_ref"] == fixture_ref
    assert record["reset_provenance"]["schema_id"] == RESET_PROVENANCE_POLICY_ID
    assert record["reset_provenance"]["reset_seed"] == 555
    assert record["reset_provenance"]["random_tape_source"] == (
        "source_fixture_random_tape_values"
    )
    assert record["reset_provenance"]["random_tape_length"] == 12
    assert record["reset_provenance"]["rng_history_ref"] == (
        "source_fixture_random_tape_values/v0"
    )
    assert record["reset_provenance"]["source_fixture_ref"] == fixture_ref
    assert record["reset_provenance"]["seed_alone_replay_complete"] is False
    multiplayer_contract.validate_multiplayer_replay_metadata_guard(record)


def test_multiplayer_metadata_replay_recorder_preserves_seeded_public_bonus_metadata():
    scenario_name = "source_bonus_self_small_catch_step.json"
    batch, fixture_ref = _seeded_bonus_public_batch_from_fixture(
        scenario_name,
        body_capacity=8,
    )
    recorder = multiplayer_replay_v0.MultiplayerMetadataReplayRecorder()

    records = recorder.record_batch(batch, rng_history_ref=fixture_ref)
    chunk = recorder.build_chunk()
    record = records[0]

    assert batch.info["step_counters"]["bonus_self_small_catches"] == 1
    assert batch.info["step_counters"]["bonus_stack_appends"] == 1
    assert record["public_env_contract_id"] == (
        batch.info["seeded_bonus_public_env_contract_id"]
    )
    assert record["base_public_env_contract_id"] == (
        batch.info["base_public_env_contract_id"]
    )
    assert record["seeded_bonus_public_env_contract_id"] == (
        batch.info["seeded_bonus_public_env_contract_id"]
    )
    assert record["ruleset_id"] == "curvytron_seeded_bonus_subset/v0"
    assert record["bonus_support_mode"] == "seeded"
    assert record["bonus_support_mode_by_row"] == "seeded"
    assert record["bonus_support_enabled"] is True
    assert record["bonus_support_active_count"] == 0
    assert record["bonus_support_stack_count"] == [1, 0]
    assert record["bonus_support_supported_types"] == list(SEEDED_BONUS_TYPE_NAMES)
    assert record["bonus_support_supported_seeded_bonus_types"] == list(
        SEEDED_BONUS_TYPE_NAMES
    )
    assert record["natural_bonus_spawn"] is False
    assert record["bonus_support_natural_bonus_spawn"] is False
    assert record["borderless"] is False
    assert record["bonus_support_borderless"] is False
    assert record["bonus_metadata_audit_claim"] == (
        multiplayer_replay_v0.MULTIPLAYER_REPLAY_BONUS_METADATA_AUDIT_CLAIM
    )
    expected_bonus_support = {
        "policy_id": "curvyzero_public_seeded_bonus_fixture_support/v0",
        "claim": "seeded public runtime-backed bonus effects; no natural bonus spawn",
        "metadata_audit_claim": (
            multiplayer_replay_v0.MULTIPLAYER_REPLAY_BONUS_METADATA_AUDIT_CLAIM
        ),
        "mode": "seeded",
        "mode_by_row": "seeded",
        "enabled": True,
        "active_count": 0,
        "stack_count": [1, 0],
        "supported_types": list(SEEDED_BONUS_TYPE_NAMES),
        "supported_seeded_bonus_types": list(SEEDED_BONUS_TYPE_NAMES),
        "natural_bonus_spawn": False,
        "public_env_contract_id": record["public_env_contract_id"],
        "ruleset_id": record["ruleset_id"],
        "metadata_only": True,
        "trainer_replay_claim": False,
        "full_replay_arrays_claim": False,
        "borderless": False,
    }
    expected_bonus_support.update(record["bonus_support_unsupported_metadata"])
    expected_bonus_support["unsupported_metadata"] = (
        record["bonus_support_unsupported_metadata"]
    )
    assert record["bonus_support"] == expected_bonus_support
    assert record["trainer_observation_claim"] is False
    assert record["trainer_replay_claim"] is False
    assert record["learned_observation_claim"] is False
    assert "final_observation" not in record
    assert "final_reward_map" not in record
    assert "bonus_active" not in record["bonus_support"]
    assert "bonus_type" not in record["bonus_support"]
    assert chunk.metadata["metadata_only"] is True
    assert chunk.metadata["trainer_replay_claim"] is False
    assert chunk.metadata["learned_observation_claim"] is False
    assert chunk.records == records


def test_multiplayer_metadata_replay_recorder_preserves_seeded_borderless_bonus_metadata():
    scenario_name = "source_bonus_game_borderless_catch_step.json"
    batch, fixture_ref = _seeded_bonus_public_batch_from_fixture(
        scenario_name,
        body_capacity=4,
        bonus_type="BonusGameBorderless",
    )
    recorder = multiplayer_replay_v0.MultiplayerMetadataReplayRecorder()

    records = recorder.record_batch(batch, rng_history_ref=fixture_ref)
    record = records[0]

    assert batch.info["step_counters"]["bonus_game_borderless_catches"] == 1
    assert record["bonus_support_mode"] == "seeded"
    assert record["bonus_support_enabled"] is True
    assert record["bonus_support_active_count"] == 0
    assert record["bonus_support_stack_count"] == [0, 0]
    assert record["bonus_support_supported_types"] == list(SEEDED_BONUS_TYPE_NAMES)
    assert record["bonus_support_supported_seeded_bonus_types"] == list(
        SEEDED_BONUS_TYPE_NAMES
    )
    assert record["natural_bonus_spawn"] is False
    assert record["borderless"] is True
    assert record["bonus_support_borderless"] is True
    assert record["bonus_support"]["borderless"] is True
    assert record["bonus_support"]["metadata_only"] is True
    assert record["bonus_support"]["trainer_replay_claim"] is False
    assert record["bonus_support"]["full_replay_arrays_claim"] is False
    assert record["trainer_replay_claim"] is False
    assert "final_observation" not in record
    assert "final_reward_map" not in record
    assert "bonus_active" not in record["bonus_support"]
    assert "bonus_type" not in record["bonus_support"]


def test_multiplayer_replay_bonus_audit_does_not_mark_supported_effects_unsupported():
    batch, fixture_ref = _seeded_bonus_public_batch_from_fixture(
        "source_bonus_self_small_catch_step.json",
        body_capacity=8,
    )

    record = multiplayer_replay_v0.build_multiplayer_replay_record_from_public_env_output(
        info=batch.info,
        action_mask=batch.action_mask,
        reward=batch.reward,
        done=batch.done,
        terminated=batch.terminated,
        truncated=batch.truncated,
        rng_history_ref=fixture_ref,
    )

    unsupported_metadata = record["bonus_support_unsupported_metadata"]
    assert unsupported_metadata["unsupported_natural_bonus_types"] == []
    assert unsupported_metadata["unsupported_natural_bonus_effects"] == []
    assert set(record["bonus_support_supported_types"]).isdisjoint(
        unsupported_metadata["unsupported_natural_bonus_effects"],
    )
    assert "BonusEnemySlow" in record["bonus_support_supported_types"]
    assert "BonusEnemyStraightAngle" in record["bonus_support_supported_types"]
    assert record["bonus_support"]["metadata_only"] is True
    assert record["bonus_support"]["full_replay_arrays_claim"] is False


def test_multiplayer_replay_v0_preserves_future_unsupported_bonus_audit_flags():
    batch, fixture_ref = _seeded_bonus_public_batch_from_fixture(
        "source_bonus_self_small_catch_step.json",
        body_capacity=8,
    )
    info = dict(batch.info)
    bonus_support = dict(info["bonus_support"])
    bonus_support["unsupported_bonus_effect_present"] = np.asarray([True], dtype=bool)
    bonus_support["unsupported_bonus_types"] = ("BonusSelfMaster",)
    info["bonus_support"] = bonus_support

    record = multiplayer_replay_v0.build_multiplayer_replay_record_from_public_env_output(
        info=info,
        action_mask=batch.action_mask,
        reward=batch.reward,
        done=batch.done,
        terminated=batch.terminated,
        truncated=batch.truncated,
        rng_history_ref=fixture_ref,
    )

    unsupported_metadata = record["bonus_support_unsupported_metadata"]
    assert unsupported_metadata["unsupported_bonus_effect_present"] is True
    assert unsupported_metadata["unsupported_bonus_types"] == ["BonusSelfMaster"]
    assert unsupported_metadata["unsupported_natural_bonus_types"] == []
    assert unsupported_metadata["unsupported_natural_bonus_effects"] == []
    assert "BonusEnemySlow" not in unsupported_metadata["unsupported_natural_bonus_effects"]
    assert record["bonus_support"]["unsupported_bonus_effect_present"] is True
    assert record["bonus_support"]["unsupported_bonus_types"] == ["BonusSelfMaster"]
    assert record["bonus_support"]["unsupported_metadata"] == (
        unsupported_metadata
    )
    assert record["trainer_replay_claim"] is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("enabled_by_row", np.asarray([1], dtype=np.int16), "enabled_by_row"),
        ("active_count", np.asarray([-1], dtype=np.int32), "active_count"),
        ("stack_count", np.zeros((1, 1), dtype=np.int16), "stack_count"),
        (
            "supported_seeded_bonus_types",
            ("BonusSelfSmall", ""),
            "supported_seeded_bonus_types",
        ),
        ("natural_bonus_spawn", "false", "natural_bonus_spawn"),
        (
            "unsupported_bonus_effect_present",
            np.asarray(["true"], dtype=object),
            "unsupported_bonus_effect_present",
        ),
    ],
)
def test_multiplayer_replay_v0_rejects_malformed_bonus_support_metadata(
    field,
    value,
    match,
):
    batch, _fixture_ref = _seeded_bonus_public_batch_from_fixture(
        "source_bonus_self_small_catch_step.json",
        body_capacity=8,
    )
    info = dict(batch.info)
    bonus_support = dict(info["bonus_support"])
    bonus_support[field] = value
    info["bonus_support"] = bonus_support

    with pytest.raises(replay.ReplayCompatibilityError, match=match):
        multiplayer_replay_v0.build_multiplayer_replay_record_from_public_env_output(
            info=info,
            action_mask=batch.action_mask,
            reward=batch.reward,
            done=batch.done,
            terminated=batch.terminated,
            truncated=batch.truncated,
        )


def test_multiplayer_replay_v0_rejects_malformed_bonus_borderless_metadata():
    batch, _fixture_ref = _seeded_bonus_public_batch_from_fixture(
        "source_bonus_game_borderless_catch_step.json",
        body_capacity=4,
        bonus_type="BonusGameBorderless",
    )
    info = dict(batch.info)
    info["borderless"] = np.asarray([1], dtype=np.int16)

    with pytest.raises(replay.ReplayCompatibilityError, match="borderless"):
        multiplayer_replay_v0.build_multiplayer_replay_record_from_public_env_output(
            info=info,
            action_mask=batch.action_mask,
            reward=batch.reward,
            done=batch.done,
            terminated=batch.terminated,
            truncated=batch.truncated,
        )


def test_multiplayer_scalar_observation_replay_artifact_carries_trace_metadata_only():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=444,
        random_tape_capacity=24,
    )
    env.reset(seed=np.asarray([444], dtype=np.uint64))
    step_batch = env.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))
    packed = obs.pack_vector_multiplayer_observation_rows_v0(
        env.state,
        max_ticks=env.max_ticks,
        pad_to=6,
    )

    artifact = multiplayer_replay_v0.build_multiplayer_scalar_observation_replay_artifact_v0(
        observation_rows=packed,
        batch=step_batch,
    )

    assert artifact.metadata["replay_contract_id"] == (
        multiplayer_replay_v0.MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_CONTRACT_ID
    )
    assert artifact.metadata["replay_schema_hash"] == (
        multiplayer_replay_v0.MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_HASH
    )
    assert artifact.metadata["observation_schema_id"] == obs.MULTIPLAYER_OBSERVATION_SCHEMA_ID
    assert artifact.metadata["observation_schema_hash"] == obs.MULTIPLAYER_OBSERVATION_SCHEMA_HASH
    assert artifact.metadata["record_count"] == 4
    assert artifact.metadata["active_row_count"] == 4
    assert artifact.metadata["row_count"] == 6
    assert artifact.metadata["source_shape"] == [1, 4]
    assert artifact.metadata["public_env_metadata_only"] is True
    assert artifact.metadata["contains_scalar_observation_rows"] is True
    assert artifact.metadata["trainer_ready_env_claim"] is False
    assert artifact.metadata["trainer_replay_claim"] is False
    assert artifact.metadata["visual_replay_claim"] is False
    assert artifact.metadata["source_fidelity_completion_claim"] is False
    np.testing.assert_array_equal(artifact.arrays["observation"], packed.observation)
    np.testing.assert_array_equal(artifact.arrays["action_mask"], packed.action_mask)
    np.testing.assert_array_equal(
        artifact.arrays["lightzero_action_mask"],
        packed.lightzero_action_mask,
    )
    np.testing.assert_array_equal(artifact.arrays["row_mask"], packed.row_mask)
    assert len(artifact.records) == 4
    assert artifact.records[0]["env_row_id"] == 0
    assert artifact.records[0]["ego_player_id"] == 0
    assert artifact.records[0]["episode_id"] == int(step_batch.info["episode_id"][0])
    assert artifact.records[0]["reset_episode_id"] == int(
        step_batch.info["reset_episode_id"][0]
    )
    assert artifact.records[0]["round_id"] == 1
    assert artifact.records[0]["source_round_id"] == 1
    assert artifact.records[0]["reset_seed"] == 444
    assert artifact.records[0]["random_tape_cursor"] == int(
        step_batch.info["random_tape_cursor"][0]
    )
    assert artifact.records[0]["action_mask"] == [True, True, True]
    assert artifact.records[0]["lightzero_action_mask"] == [1, 1, 1]
    assert artifact.records[0]["public_env_record"]["round_id"] == 1
    assert artifact.records[0]["public_env_record"]["source_round_id"] == 1
    assert artifact.records[0]["public_env_record"]["reset_episode_id"] == (
        artifact.records[0]["episode_id"]
    )
    multiplayer_contract.validate_multiplayer_replay_metadata_guard(
        artifact.records[0]["public_env_record"]
    )
    assert "not_policy_targets" in artifact.records[0]["non_claims"]


def test_multiplayer_scalar_observation_replay_artifact_rejects_public_shape_mismatch():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        seed=445,
        random_tape_capacity=24,
    )
    env.reset(seed=np.asarray([445], dtype=np.uint64))
    step_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))
    packed = obs.pack_vector_multiplayer_observation_rows_v0(
        env.state,
        max_ticks=env.max_ticks,
    )
    bad_info = dict(step_batch.info)
    bad_info["player_count"] = 4

    with pytest.raises(replay.ReplayCompatibilityError, match="source_shape player count"):
        multiplayer_replay_v0.build_multiplayer_scalar_observation_replay_artifact_v0(
            observation_rows=packed,
            info=bad_info,
            public_action_mask=step_batch.action_mask,
            reward=step_batch.reward,
            done=step_batch.done,
            terminated=step_batch.terminated,
            truncated=step_batch.truncated,
        )


def test_multiplayer_metadata_replay_recorder_records_public_batch_rows_and_opponent_sidecar():
    output = _public_env_output(player_count=3, batch_size=2)
    batch = _public_batch_from_output(output)
    recorder = multiplayer_replay_v0.MultiplayerMetadataReplayRecorder()

    records = recorder.record_batch(
        batch,
        opponent_policy_sidecar={
            "policy_id": "random_uniform",
            "policy_version": "2026-05-10",
            "seed": np.asarray([11, 12], dtype=np.int64),
            "actions": np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int16),
        },
    )
    chunk = recorder.build_chunk()

    assert recorder.record_count == 2
    assert recorder.batch_count == 1
    assert recorder.closed_by_terminal is False
    assert len(records) == 2
    assert records[0]["sequence_index"] == 0
    assert records[1]["sequence_index"] == 1
    assert records[1]["batch_row"] == 1
    assert records[0]["metadata_only"] is True
    assert records[0]["trainer_observation_claim"] is False
    assert records[0]["opponent_policy_sidecar"] == {
        "schema_id": multiplayer_contract.MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID,
        "policy_id": "random_uniform",
        "policy_version": "2026-05-10",
        "seed": 11,
        "actions": [0, 1, 2],
        "metadata_only": True,
        "trainer_replay_claim": False,
        "learned_observation_claim": False,
    }
    assert records[1]["opponent_policy_sidecar"]["seed"] == 12
    assert records[1]["opponent_policy_sidecar"]["actions"] == [2, 1, 0]
    assert chunk.metadata["record_count"] == 2
    assert chunk.metadata["recorded_batch_count"] == 1
    assert chunk.metadata["metadata_only"] is True
    assert chunk.metadata["trainer_observation_claim"] is False
    assert chunk.metadata["trainer_replay_claim"] is False
    assert chunk.metadata["learned_observation_claim"] is False
    assert chunk.metadata["opponent_policy_sidecar_present"] is True
    assert chunk.records == records


def test_multiplayer_metadata_replay_recorder_closes_on_terminal_final_rows_until_reset():
    terminal_output = _public_env_output(
        player_count=2,
        done=True,
        terminated=True,
    )
    terminal_batch = _public_batch_from_output(terminal_output, include_final=True)
    open_batch = _public_batch_from_output(_public_env_output(player_count=2))
    recorder = multiplayer_replay_v0.MultiplayerMetadataReplayRecorder()

    recorder.record_batch(terminal_batch)

    assert recorder.closed_by_terminal is True
    assert recorder.build_chunk().metadata["closed_by_terminal"] is True
    with pytest.raises(replay.ReplayCompatibilityError, match="before reset"):
        recorder.record_batch(open_batch)

    recorder.reset()
    recorder.record_batch(open_batch)

    assert recorder.record_count == 1
    assert recorder.closed_by_terminal is False


def test_multiplayer_metadata_replay_recorder_packages_real_2p_terminal_public_batch():
    source_ref = "scenarios/environment/source_normal_wall_death_step.json"
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=101,
        decision_ms=100.0,
        map_size=88.0,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=32,
    )
    env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_random_tape_values=np.full((1, 32), 0.5, dtype=np.float64),
        source_fixture_ref=source_ref,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.state["pos"][0] = np.asarray([[87.35, 44.0], [44.0, 44.0]], dtype=np.float64)
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["heading"][0] = np.asarray([0.0, np.pi], dtype=np.float64)
    env.state["speed"][0] = np.asarray([env.speed, env.speed], dtype=np.float64)
    env.state["print_manager_distance"][0] = np.full(2, 999.0, dtype=np.float64)
    env.state["print_manager_last_pos"][0] = env.state["pos"][0]
    batch = env.step(np.asarray([[1, 1]], dtype=np.int16))
    recorder = multiplayer_replay_v0.MultiplayerMetadataReplayRecorder()

    records = recorder.record_batch(batch, rng_history_ref=source_ref)
    chunk = recorder.build_chunk()
    record = records[0]

    assert bool(batch.done[0]) is True
    assert recorder.closed_by_terminal is True
    assert record["env_impl_id"] == batch.info["env_impl_id"]
    assert record["player_count"] == 2
    assert record["metadata_only"] is True
    assert record["trainer_replay_claim"] is False
    assert record["learned_observation_claim"] is False
    assert record["rng_history_ref"] == source_ref
    assert record["source_fixture_ref"] == source_ref
    assert record["random_tape_source"] == "source_fixture_random_tape_values"
    assert record["done"] is True
    assert record["terminated"] is True
    assert record["truncated"] is False
    assert record["needs_reset"] is True
    assert record["score"] == [0, 1]
    assert record["alive"] == [False, True]
    assert record["death_order"] == [0]
    assert record["death_player"] == [0, -1]
    assert record["death_count"] == 1
    assert record["winner"] == 1
    assert record["round_winner"] == 1
    assert record["draw"] is False
    assert record["winner_ids"] == [1]
    assert record["joint_action"] == [1, 1]
    assert record["action_sidecar"]["player_action"] == [1, 1]
    assert record["action_sidecar"]["player_action_mask"] == [
        [False, False, False],
        [False, False, False],
    ]
    assert record["action_sidecar"]["action_required"] == [False, False]
    assert record["terminal_rows"] == [0]
    assert record["final_observation_rows"] == [0]
    assert record["final_observation_row_mask"] == [True]
    assert record["final_observation_row_policy"]["source_claim"] == (
        "debug_metadata_only_public_terminal_rows/v0"
    )
    assert record["final_observation_row_policy"]["row_mask"] == [True]
    assert record["final_reward_rows"] == [0]
    assert record["final_reward_row_mask"] == [True]
    assert record["final_reward_row_policy"]["source_claim"] == (
        "debug_metadata_only_public_terminal_rows/v0"
    )
    assert "final_observation" not in record
    assert "final_reward_map" not in record
    assert chunk.metadata["player_count"] == 2
    assert chunk.metadata["metadata_only"] is True
    assert chunk.metadata["trainer_replay_claim"] is False
    assert chunk.metadata["closed_by_terminal"] is True
    assert chunk.records == records


def _strict_replay_arrays(*, player_count: int) -> dict[str, np.ndarray]:
    chunk_steps = 1
    batch_size = 1
    obs_dim = 4
    action_count = 3
    return {
        "observation": np.zeros(
            (chunk_steps, batch_size, player_count, obs_dim),
            dtype=np.float32,
        ),
        "reward": np.zeros((chunk_steps, batch_size, player_count), dtype=np.float32),
        "action": np.zeros((chunk_steps, batch_size, player_count), dtype=np.int16),
        "action_weights": np.full(
            (chunk_steps, batch_size, player_count, action_count),
            np.float32(1.0 / action_count),
            dtype=np.float32,
        ),
        "root_value": np.zeros((chunk_steps, batch_size, player_count), dtype=np.float32),
        "done": np.zeros((chunk_steps, batch_size), dtype=bool),
        "terminated": np.zeros((chunk_steps, batch_size), dtype=bool),
        "truncated": np.zeros((chunk_steps, batch_size), dtype=bool),
        "episode_id": np.asarray(["episode-a"], dtype=np.str_),
        "reset_seed": np.asarray([1001], dtype=np.int64),
        "reset_source": np.asarray(["manual"], dtype=np.str_),
        "final_observation": np.zeros((batch_size, player_count, obs_dim), dtype=np.float32),
        "final_reward_map": np.zeros((batch_size, player_count), dtype=np.float32),
    }


def _strict_replay_metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return replay.build_replay_chunk_v0_metadata(
        arrays,
        rules_hash="rules-hash-001",
        observation_schema_hash="obs-hash-001",
        action_space_hash="action-hash-001",
        reward_schema_hash="reward-hash-001",
        ruleset_id="curvytron-v1-reference",
        observation_schema_id="curvyzero_egocentric_rays/v0",
        action_space_id="curvyzero_source_move_action_space/v0",
        reward_schema_id="curvyzero_1v1_no_bonus_reward/v0",
        producer="test_multiplayer_replay_contract",
    )


def _lifecycle_random_tape(scenario_name: str) -> np.ndarray:
    with (SCENARIO_ROOT / scenario_name).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sequence = payload["source_setup"]["random"]["math_random_sequence"]
    return np.asarray([sequence], dtype=np.float64)


def _seeded_bonus_fixture_state_and_action(
    scenario_name: str,
    *,
    body_capacity: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, float, dict[str, object]]:
    scenario_path = f"scenarios/environment/{scenario_name}"
    fixture = seed_bridge.seed_fixture(scenario_path, body_capacity=body_capacity)
    state = vector_compare.array_state_from_seed(fixture)
    prepared_step = vector_compare.prepare_fixture_array_step(fixture, step_index=0)
    source_moves = np.asarray(prepared_step["source_moves"], dtype=np.int8)
    actions = (source_moves.astype(np.int16) + 1).reshape(1, -1)
    with (SCENARIO_ROOT / scenario_name).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    initial_state = payload["initial_state"]
    assert isinstance(initial_state, dict)
    active_bonuses = initial_state["active_bonuses"]
    assert isinstance(active_bonuses, list)
    assert len(active_bonuses) == 1
    bonus = active_bonuses[0]
    assert isinstance(bonus, dict)
    return state, actions, float(prepared_step["step_ms"]), bonus


def _seeded_bonus_public_batch_from_fixture(
    scenario_name: str,
    *,
    body_capacity: int,
    bonus_type: str | None = None,
) -> tuple[VectorMultiplayerBatch, str]:
    fixture_ref = f"scenarios/environment/{scenario_name}"
    state, actions, step_ms, bonus = _seeded_bonus_fixture_state_and_action(
        scenario_name,
        body_capacity=body_capacity,
    )
    player_count = int(state["pos"].shape[1])
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        decision_ms=step_ms,
        body_capacity=body_capacity,
        event_capacity=16,
        timer_capacity=max(4, player_count),
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([101], dtype=np.uint64),
    )
    env.seed_active_bonus(
        row=0,
        bonus_type=bonus_type or str(bonus["type"]),
        x=float(bonus["x"]),
        y=float(bonus["y"]),
    )
    return env.step(actions), fixture_ref


def _multiplayer_metadata(
    *,
    player_count: int,
    observation_schema_id: str | None = None,
) -> dict[str, object]:
    player_ids = [f"p{index}" for index in range(player_count)]
    action_mask = [[True, True, True] for _ in range(player_count)]
    joint_action = [1 for _ in range(player_count)]
    metadata: dict[str, object] = {
        "public_env_contract_id": "curvyzero_public_multiplayer_env/v0",
        "env_impl_id": "curvyzero_vector_multiplayer_env/v0",
        "ruleset_id": "curvytron_no_bonus/v0",
        "rules_hash": "rules-hash-001",
        "native_control_model_id": "curvytron_realtime_controls_elapsed_frames/v0",
        "trainer_control_wrapper_id": "curvyzero_fixed_decision_wrapper/v0",
        "decision_ms": 300.0,
        "lifecycle_policy_id": "curvyzero_public_explicit_reset_warmdown_bridge/v0",
        "episode_id": 1,
        "reset_episode_id": 1,
        "reset_episode_id_policy": (
            "vector_reset_episode_id_increments_on_explicit_reset_only/v0"
        ),
        "round_id": 1,
        "source_round_id": 1,
        "source_round_id_policy": (
            "one_based_source_round_increments_on_next_round_spawn/v0"
        ),
        "episode_end_mode": "round",
        "step_index": 0,
        "tick_index": 0,
        "elapsed_ms": 0.0,
        "player_count": player_count,
        "player_ids": player_ids,
        "source_player_ids": [f"source-p{index}" for index in range(player_count)],
        "present": [True for _ in range(player_count)],
        "alive": [True for _ in range(player_count)],
        "action_mask": action_mask,
        "joint_action": joint_action,
        "reward": [0.0 for _ in range(player_count)],
        "done": False,
        "terminated": False,
        "truncated": False,
        "score": [0 for _ in range(player_count)],
        "round_score": [0 for _ in range(player_count)],
        "death_order": [],
        "death_player": [-1 for _ in range(player_count)],
        "death_count": 0,
        "round_done": False,
        "match_done": False,
        "needs_reset": False,
        "terminal_reason": "not_terminal",
        "truncation_reason": None,
        "winner": -1,
        "round_winner": -1,
        "match_winner": -1,
        "draw": False,
        "winner_ids": [],
        "round_winner_ids": [],
        "match_winner_ids": [],
        "reset_seed": 1001,
        "reset_source": "manual",
        "random_tape_cursor": 0,
        "random_tape_draw_count": 0,
        "rng_history_ref": None,
        "action_sidecar": {
            "schema_id": PUBLIC_JOINT_ACTION_SIDECAR_SCHEMA_ID,
            "player_action": joint_action,
            "player_action_mask": action_mask,
            "action_required": [True for _ in range(player_count)],
            "joint_action_schema_id": "curvyzero_external_joint_action_player_major/v0",
        },
        "observation_schema_id": observation_schema_id or "curvyzero_debug_metadata_only/v0",
        "observation_schema_hash": "observation-hash-001",
        "action_space_id": "curvyzero_turn3/v0",
        "action_space_hash": "action-hash-001",
        "reward_schema_id": "curvyzero_sparse_round_outcome/v0",
        "reward_schema_hash": "reward-hash-001",
        "final_observation_policy": "metadata_only_not_trainer_ready",
        "metadata_only": True,
        "trainer_observation_claim": False,
        "trainer_replay_claim": False,
        "learned_observation_claim": False,
    }
    return metadata


def _public_env_output(
    *,
    player_count: int,
    batch_size: int = 1,
    done: bool = False,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, object]:
    present = np.ones((batch_size, player_count), dtype=bool)
    alive = np.ones((batch_size, player_count), dtype=bool)
    alive[:, player_count - 1] = False
    action_mask = np.ones((batch_size, player_count, 3), dtype=bool)
    action_mask[:, player_count - 1, :] = False
    action = np.ones((batch_size, player_count), dtype=np.int16)
    death_player = np.tile(
        np.asarray(
            [player_count - 1, *[-1 for _ in range(player_count - 1)]],
            dtype=np.int16,
        ),
        (batch_size, 1),
    )
    done_array = np.full(batch_size, done, dtype=bool)
    terminated_array = np.full(batch_size, terminated, dtype=bool)
    truncated_array = np.full(batch_size, truncated, dtype=bool)
    action_mask[done_array, :, :] = False
    action_required = alive & ~done_array[:, None]
    return {
        "action_mask": action_mask,
        "reward": np.zeros((batch_size, player_count), dtype=np.float32),
        "done": done_array,
        "terminated": terminated_array,
        "truncated": truncated_array,
        "info": {
            "public_env_contract_id": "curvyzero_public_multiplayer_env/v0",
            "env_impl_id": "curvyzero_vector_multiplayer_env/v0",
            "ruleset_id": "curvytron_no_bonus/v0",
            "rules_hash": "rules-hash-001",
            "native_control_model_id": "curvytron_realtime_controls_elapsed_frames/v0",
            "trainer_control_wrapper_id": "curvyzero_fixed_decision_wrapper/v0",
            "decision_ms": 300.0,
            "lifecycle_policy_id": "curvyzero_public_explicit_reset_warmdown_bridge/v0",
            "episode_id": np.arange(1, 1 + batch_size, dtype=np.int64),
            "reset_episode_id": np.arange(1, 1 + batch_size, dtype=np.int64),
            "reset_episode_id_policy": (
                "vector_reset_episode_id_increments_on_explicit_reset_only/v0"
            ),
            "round_id": np.arange(1, 1 + batch_size, dtype=np.int64),
            "source_round_id": np.arange(1, 1 + batch_size, dtype=np.int64),
            "source_round_id_policy": (
                "one_based_source_round_increments_on_next_round_spawn/v0"
            ),
            "episode_end_mode": "round",
            "step_index": np.arange(batch_size, dtype=np.int32),
            "tick_index": np.arange(1, 1 + batch_size, dtype=np.int32),
            "elapsed_ms": np.asarray(
                [300.0 + index for index in range(batch_size)],
                dtype=np.float64,
            ),
            "player_count": player_count,
            "player_ids": tuple(f"player_{index}" for index in range(player_count)),
            "source_player_ids": np.arange(1, player_count + 1, dtype=np.int16),
            "present": present,
            "alive": alive,
            "score": np.zeros((batch_size, player_count), dtype=np.int32),
            "round_score": np.zeros((batch_size, player_count), dtype=np.int32),
            "death_player": death_player,
            "death_count": np.full(batch_size, 1, dtype=np.int32),
            "round_done": done_array.copy(),
            "match_done": np.zeros(batch_size, dtype=bool),
            "needs_reset": done_array.copy(),
            "terminal_reason": np.asarray(
                ["round_end" if done else "none" for _ in range(batch_size)],
                dtype=object,
            ),
            "truncation_reason": np.asarray(
                ["timeout" if truncated else None for _ in range(batch_size)],
                dtype=object,
            ),
            "winner": np.full(batch_size, 0 if done else -1, dtype=np.int16),
            "round_winner": np.full(batch_size, 0 if done else -1, dtype=np.int16),
            "match_winner": np.full(batch_size, -1, dtype=np.int16),
            "draw": np.zeros(batch_size, dtype=bool),
            "winner_ids": [[0] if done else [] for _ in range(batch_size)],
            "round_winner_ids": [[0] if done else [] for _ in range(batch_size)],
            "match_winner_ids": [[] for _ in range(batch_size)],
            "reset_seed": np.arange(1001, 1001 + batch_size, dtype=np.int64),
            "reset_source": np.asarray(["manual" for _ in range(batch_size)], dtype=object),
            "random_tape_cursor": np.arange(6, 6 + batch_size, dtype=np.int32),
            "random_tape_draw_count": np.arange(6, 6 + batch_size, dtype=np.int32),
            "action_sidecar": {
                "schema_id": PUBLIC_JOINT_ACTION_SIDECAR_SCHEMA_ID,
                "player_action": action,
                "player_action_mask": action_mask,
                "action_required": action_required,
                "joint_action_schema_id": "curvyzero_external_joint_action_player_major/v0",
            },
            "observation_schema_id": "curvyzero_debug_metadata_only/v0",
            "observation_schema_hash": "observation-hash-001",
            "action_space_id": "curvyzero_turn3/v0",
            "action_space_hash": "action-hash-001",
            "reward_schema_id": "curvyzero_sparse_round_outcome/v0",
            "reward_schema_hash": "reward-hash-001",
            "final_observation_policy": "terminal_public_observation_before_autoreset/v0",
            "metadata_only": True,
            "trainer_observation_claim": False,
        },
    }


def _public_batch_from_output(
    output: dict[str, object],
    *,
    include_final: bool = False,
) -> VectorMultiplayerBatch:
    action_mask = np.asarray(output["action_mask"])
    reward = np.asarray(output["reward"])
    done = np.asarray(output["done"])
    final_observation = None
    final_reward = None
    info = dict(output["info"])
    if include_final:
        final_observation = np.zeros(
            (action_mask.shape[0], action_mask.shape[1], 6),
            dtype=np.float32,
        )
        final_reward = reward.copy()
        info["final_observation"] = final_observation.copy()
        info["final_reward_map"] = final_reward.copy()
        info["final_observation_row_mask"] = done.copy()
        info["final_reward_row_mask"] = done.copy()
    return VectorMultiplayerBatch(
        observation=np.zeros(
            (action_mask.shape[0], action_mask.shape[1], 6),
            dtype=np.float32,
        ),
        action_mask=action_mask,
        reward=reward,
        done=done,
        terminated=np.asarray(output["terminated"]),
        truncated=np.asarray(output["truncated"]),
        final_observation=final_observation,
        final_reward=final_reward,
        info=info,
    )
