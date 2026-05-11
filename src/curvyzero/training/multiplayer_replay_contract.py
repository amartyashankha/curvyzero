"""Guardrails for public multiplayer replay metadata.

This is intentionally metadata-only. It does not define a trainer-ready
multiplayer observation payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


MULTIPLAYER_REPLAY_REQUIRED_METADATA_FIELDS = (
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
)

MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID = (
    "curvyzero_multiplayer_opponent_policy_sidecar/v0"
)

MULTIPLAYER_REPLAY_PLAYER_COUNTS = (2, 3, 4)

STRICT_1V1_REPLAY_OBSERVATION_SCHEMA_IDS = (
    "curvyzero_egocentric_rays/v0",
)

MULTIPLAYER_REPLAY_FINAL_OBSERVATION_POLICIES = (
    "terminal_public_observation_before_autoreset/v0",
    "metadata_only_not_trainer_ready",
)


def validate_multiplayer_replay_metadata_guard(metadata: Mapping[str, Any]) -> None:
    """Validate the narrow metadata-only guard for public 2P/3P/4P replay work."""

    missing = sorted(set(MULTIPLAYER_REPLAY_REQUIRED_METADATA_FIELDS) - set(metadata))
    if missing:
        raise ReplayCompatibilityError(
            "missing multiplayer replay metadata: " + ", ".join(missing)
        )

    for key in (
        "public_env_contract_id",
        "env_impl_id",
        "ruleset_id",
        "rules_hash",
        "native_control_model_id",
        "trainer_control_wrapper_id",
        "lifecycle_policy_id",
        "reset_episode_id_policy",
        "source_round_id_policy",
        "episode_end_mode",
        "observation_schema_hash",
        "action_space_id",
        "action_space_hash",
        "reward_schema_id",
        "reward_schema_hash",
    ):
        _metadata_string(metadata, key)
    decision_ms = _metadata_number(metadata, "decision_ms")
    if decision_ms <= 0.0:
        raise ReplayCompatibilityError("decision_ms must be positive")
    for key in ("step_index", "tick_index"):
        value = _metadata_int(metadata, key)
        if value < 0:
            raise ReplayCompatibilityError(f"{key} must be non-negative")
    elapsed_ms = _metadata_number(metadata, "elapsed_ms")
    if elapsed_ms < 0.0:
        raise ReplayCompatibilityError("elapsed_ms must be non-negative")
    episode_id = _metadata_id_scalar(metadata, "episode_id")
    reset_episode_id = _metadata_id_scalar(metadata, "reset_episode_id")
    if reset_episode_id != episode_id:
        raise ReplayCompatibilityError(
            "reset_episode_id must match episode_id for metadata replay v0"
        )
    round_id = _metadata_id_scalar(metadata, "round_id")
    source_round_id = _metadata_id_scalar(metadata, "source_round_id")
    if source_round_id != round_id:
        raise ReplayCompatibilityError(
            "source_round_id must match round_id for metadata replay v0"
        )

    player_count = metadata.get("player_count")
    if (
        isinstance(player_count, bool)
        or not isinstance(player_count, int)
        or player_count not in MULTIPLAYER_REPLAY_PLAYER_COUNTS
    ):
        raise ReplayCompatibilityError("player_count must be 2, 3, or 4 for multiplayer replay")

    _expect_id_sequence(metadata, "player_ids", player_count)
    _expect_id_sequence(metadata, "source_player_ids", player_count)
    present = _expect_bool_vector(metadata, "present", player_count)
    alive = _expect_bool_vector(metadata, "alive", player_count)
    if any(item_alive and not item_present for item_present, item_alive in zip(present, alive)):
        raise ReplayCompatibilityError("alive players must also be present")
    done = _metadata_bool(metadata, "done")
    terminated = _metadata_bool(metadata, "terminated")
    truncated = _metadata_bool(metadata, "truncated")
    active = [
        item_present and item_alive and not done
        for item_present, item_alive in zip(present, alive)
    ]
    action_mask = _expect_action_mask(
        metadata,
        player_count,
        active=active,
    )
    joint_action = _expect_int_vector(metadata, "joint_action", player_count)
    action_count = len(action_mask[0])
    _expect_joint_action(joint_action, action_count=action_count, active=active)
    _expect_number_vector(metadata, "reward", player_count)
    _expect_int_vector(metadata, "score", player_count)
    _expect_int_vector(metadata, "round_score", player_count)
    death_player = _expect_int_vector(metadata, "death_player", player_count)

    death_count = _metadata_int(metadata, "death_count")
    if death_count < 0 or death_count > player_count:
        raise ReplayCompatibilityError("death_count must be in [0, player_count]")
    death_order = _expect_int_sequence(metadata, "death_order")
    if len(death_order) != death_count:
        raise ReplayCompatibilityError("death_order length must equal death_count")
    if any(player < 0 or player >= player_count for player in death_order):
        raise ReplayCompatibilityError("death_order entries must be player indexes")
    if len(set(death_order)) != len(death_order):
        raise ReplayCompatibilityError("death_order entries must be unique")
    if death_order != death_player[:death_count]:
        raise ReplayCompatibilityError("death_order must match death_player prefix")

    if done != (terminated or truncated):
        raise ReplayCompatibilityError("done must equal terminated or truncated")
    _metadata_bool(metadata, "round_done")
    _metadata_bool(metadata, "match_done")
    needs_reset = _metadata_bool(metadata, "needs_reset")
    if done and not needs_reset:
        raise ReplayCompatibilityError("needs_reset must be true when done is true")
    _metadata_bool(metadata, "draw")

    winner = _metadata_int(metadata, "winner")
    if winner < -1 or winner >= player_count:
        raise ReplayCompatibilityError("winner must be -1 or a player index")
    round_winner = _metadata_int(metadata, "round_winner")
    if round_winner < -1 or round_winner >= player_count:
        raise ReplayCompatibilityError("round_winner must be -1 or a player index")
    match_winner = _metadata_int(metadata, "match_winner")
    if match_winner < -1 or match_winner >= player_count:
        raise ReplayCompatibilityError("match_winner must be -1 or a player index")
    for winner_id in _expect_int_sequence(metadata, "winner_ids"):
        if winner_id < 0 or winner_id >= player_count:
            raise ReplayCompatibilityError("winner_ids entries must be player indexes")
    for key in ("round_winner_ids", "match_winner_ids"):
        for winner_id in _expect_int_sequence(metadata, key):
            if winner_id < 0 or winner_id >= player_count:
                raise ReplayCompatibilityError(f"{key} entries must be player indexes")

    truncation_reason = metadata.get("truncation_reason")
    if truncated:
        if not isinstance(truncation_reason, str) or not truncation_reason:
            raise ReplayCompatibilityError(
                "truncation_reason must be non-empty when truncated is true"
            )
    elif truncation_reason is not None and not isinstance(truncation_reason, str):
        raise ReplayCompatibilityError("truncation_reason must be null or a string")

    reset_seed = _metadata_int(metadata, "reset_seed")
    if reset_seed < 0:
        raise ReplayCompatibilityError("reset_seed must be non-negative")
    reset_source = metadata.get("reset_source")
    if isinstance(reset_source, bool) or not isinstance(reset_source, int | str):
        raise ReplayCompatibilityError("reset_source must be an integer code or string")
    if isinstance(reset_source, str) and not reset_source:
        raise ReplayCompatibilityError("reset_source must be non-empty")

    for key in ("random_tape_cursor", "random_tape_draw_count"):
        value = _metadata_int(metadata, key)
        if value < 0:
            raise ReplayCompatibilityError(f"{key} must be non-negative")
    rng_history_ref = metadata.get("rng_history_ref")
    if rng_history_ref is not None and (
        not isinstance(rng_history_ref, str) or not rng_history_ref
    ):
        raise ReplayCompatibilityError("rng_history_ref must be null or a non-empty string")

    action_sidecar = metadata.get("action_sidecar")
    _expect_action_sidecar(
        action_sidecar,
        player_count=player_count,
        action_count=action_count,
        action_mask=action_mask,
        joint_action=joint_action,
        active=active,
    )
    if "opponent_policy_sidecar" in metadata:
        _expect_opponent_policy_sidecar(
            metadata["opponent_policy_sidecar"],
            player_count,
            action_count=action_count,
        )

    observation_schema_id = metadata.get("observation_schema_id")
    if not isinstance(observation_schema_id, str) or not observation_schema_id:
        raise ReplayCompatibilityError("observation_schema_id must be a non-empty string")
    if observation_schema_id in STRICT_1V1_REPLAY_OBSERVATION_SCHEMA_IDS:
        raise ReplayCompatibilityError(
            "observation_schema_id cannot claim strict 1v1 replay observation schema "
            "for public multiplayer metadata replay"
        )

    final_observation_policy = metadata.get("final_observation_policy")
    if not isinstance(final_observation_policy, str) or not final_observation_policy:
        raise ReplayCompatibilityError("final_observation_policy must be a non-empty string")
    if final_observation_policy not in MULTIPLAYER_REPLAY_FINAL_OBSERVATION_POLICIES:
        raise ReplayCompatibilityError(
            "final_observation_policy must be one of "
            f"{MULTIPLAYER_REPLAY_FINAL_OBSERVATION_POLICIES!r}"
        )

    if metadata.get("metadata_only") is not True:
        raise ReplayCompatibilityError("metadata_only must be true")
    if metadata.get("trainer_observation_claim") is not False:
        raise ReplayCompatibilityError("trainer_observation_claim must be false")
    if metadata.get("trainer_replay_claim") is not False:
        raise ReplayCompatibilityError("trainer_replay_claim must be false")
    if metadata.get("learned_observation_claim") is not False:
        raise ReplayCompatibilityError("learned_observation_claim must be false")
    _expect_optional_rng_metadata(metadata)


def _expect_sequence(metadata: Mapping[str, Any], key: str, length: int) -> list[Any]:
    value = metadata.get(key)
    if not isinstance(value, list | tuple):
        raise ReplayCompatibilityError(f"{key} must be a sequence")
    if len(value) != length:
        raise ReplayCompatibilityError(f"{key} length must match player_count")
    return list(value)


def _expect_id_sequence(metadata: Mapping[str, Any], key: str, length: int) -> list[Any]:
    values = _expect_sequence(metadata, key, length)
    normalized: list[Any] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise ReplayCompatibilityError(f"{key} must contain integer or string ids")
        if isinstance(value, str) and not value:
            raise ReplayCompatibilityError(f"{key} must contain non-empty ids")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ReplayCompatibilityError(f"{key} must contain unique ids")
    return normalized


def _expect_bool_vector(metadata: Mapping[str, Any], key: str, length: int) -> list[bool]:
    values = _expect_sequence(metadata, key, length)
    if any(not isinstance(value, bool) for value in values):
        raise ReplayCompatibilityError(f"{key} must contain booleans")
    return values


def _expect_int_vector(metadata: Mapping[str, Any], key: str, length: int) -> list[int]:
    values = _expect_sequence(metadata, key, length)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ReplayCompatibilityError(f"{key} must contain integers")
    return list(values)


def _expect_number_vector(metadata: Mapping[str, Any], key: str, length: int) -> list[float]:
    values = _expect_sequence(metadata, key, length)
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise ReplayCompatibilityError(f"{key} must contain numbers")
    return [float(value) for value in values]


def _expect_int_sequence(metadata: Mapping[str, Any], key: str) -> list[int]:
    value = metadata.get(key)
    if not isinstance(value, list | tuple):
        raise ReplayCompatibilityError(f"{key} must be a sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ReplayCompatibilityError(f"{key} must contain integers")
    return list(value)


def _expect_action_mask(
    metadata: Mapping[str, Any],
    player_count: int,
    *,
    active: list[bool],
) -> list[list[bool]]:
    value = metadata.get("action_mask")
    if not isinstance(value, list | tuple) or len(value) != player_count:
        raise ReplayCompatibilityError("action_mask must have shape [player_count, action_count]")
    action_count: int | None = None
    matrix: list[list[bool]] = []
    for row in value:
        if not isinstance(row, list | tuple) or not row:
            raise ReplayCompatibilityError(
                "action_mask must have shape [player_count, action_count]"
            )
        if action_count is None:
            action_count = len(row)
        if len(row) != action_count:
            raise ReplayCompatibilityError("action_mask rows must have equal length")
        if any(not isinstance(flag, bool) for flag in row):
            raise ReplayCompatibilityError("action_mask must contain booleans")
        matrix.append([bool(flag) for flag in row])
    for player, (row, is_active) in enumerate(zip(matrix, active)):
        if is_active and not any(row):
            raise ReplayCompatibilityError(
                f"action_mask active player {player} must have at least one legal action"
            )
        if not is_active and any(row):
            raise ReplayCompatibilityError(
                f"action_mask inactive player {player} must have no legal actions"
            )
    return matrix


def _expect_joint_action(
    value: list[int],
    *,
    action_count: int,
    active: list[bool],
) -> None:
    for player, (action, is_active) in enumerate(zip(value, active)):
        if action < -1 or action >= action_count:
            raise ReplayCompatibilityError("joint_action entries must be -1 or valid action ids")
        if is_active and action < 0:
            raise ReplayCompatibilityError(
                f"joint_action active player {player} must have a valid action id"
            )


def _expect_action_sidecar(
    value: Any,
    *,
    player_count: int,
    action_count: int,
    action_mask: list[list[bool]],
    joint_action: list[int],
    active: list[bool],
) -> None:
    if not isinstance(value, Mapping):
        raise ReplayCompatibilityError("action_sidecar must be a metadata object")
    missing = sorted(
        {
            "schema_id",
            "joint_action_schema_id",
            "player_action",
            "player_action_mask",
            "action_required",
        }
        - set(value)
    )
    if missing:
        raise ReplayCompatibilityError(
            "missing action_sidecar metadata: " + ", ".join(missing)
        )
    sidecar_schema_id = value.get("schema_id")
    if not isinstance(sidecar_schema_id, str) or not sidecar_schema_id:
        raise ReplayCompatibilityError("action_sidecar schema_id must be non-empty")
    schema_id = value.get("joint_action_schema_id")
    if not isinstance(schema_id, str) or not schema_id:
        raise ReplayCompatibilityError("action_sidecar joint_action_schema_id must be non-empty")

    player_action = _sidecar_int_vector(value.get("player_action"), "player_action", player_count)
    _expect_joint_action(player_action, action_count=action_count, active=active)
    if player_action != joint_action:
        raise ReplayCompatibilityError("action_sidecar player_action must equal joint_action")

    player_action_mask = _sidecar_bool_matrix(
        value.get("player_action_mask"),
        "player_action_mask",
        player_count=player_count,
        action_count=action_count,
    )
    if player_action_mask != action_mask:
        raise ReplayCompatibilityError("action_sidecar player_action_mask must equal action_mask")

    action_required = _sidecar_bool_vector(
        value.get("action_required"),
        "action_required",
        player_count,
    )
    if action_required != active:
        raise ReplayCompatibilityError(
            "action_sidecar action_required must match present/alive/done activity"
        )


def _expect_opponent_policy_sidecar(
    value: Any,
    player_count: int,
    *,
    action_count: int,
) -> None:
    if not isinstance(value, Mapping):
        raise ReplayCompatibilityError("opponent_policy_sidecar must be a metadata object")
    missing = sorted(
        {"schema_id", "policy_id", "policy_version", "seed", "actions"} - set(value)
    )
    if missing:
        raise ReplayCompatibilityError(
            "missing opponent_policy_sidecar metadata: " + ", ".join(missing)
        )
    schema_id = value.get("schema_id")
    if schema_id != MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID:
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar schema_id must be "
            f"{MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID!r}"
        )
    for key in ("policy_id", "policy_version"):
        item = value.get(key)
        if not isinstance(item, str) or not item:
            raise ReplayCompatibilityError(f"opponent_policy_sidecar {key} must be non-empty")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar seed must be a non-negative integer"
        )
    actions = value.get("actions")
    if not isinstance(actions, list | tuple) or len(actions) != player_count:
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar actions length must match player_count"
        )
    if any(isinstance(action, bool) or not isinstance(action, int) for action in actions):
        raise ReplayCompatibilityError("opponent_policy_sidecar actions must contain integers")
    if any(action < -1 or action >= action_count for action in actions):
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar actions must be -1 or valid action ids"
        )
    if value.get("trainer_replay_claim", False) is not False:
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar trainer_replay_claim must be false"
        )
    if value.get("learned_observation_claim", False) is not False:
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar learned_observation_claim must be false"
        )


def _sidecar_bool_vector(value: Any, key: str, length: int) -> list[bool]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ReplayCompatibilityError(f"action_sidecar {key} length must match player_count")
    if any(not isinstance(item, bool) for item in value):
        raise ReplayCompatibilityError(f"action_sidecar {key} must contain booleans")
    return [bool(item) for item in value]


def _sidecar_int_vector(value: Any, key: str, length: int) -> list[int]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ReplayCompatibilityError(f"action_sidecar {key} length must match player_count")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ReplayCompatibilityError(f"action_sidecar {key} must contain integers")
    return [int(item) for item in value]


def _sidecar_bool_matrix(
    value: Any,
    key: str,
    *,
    player_count: int,
    action_count: int,
) -> list[list[bool]]:
    if not isinstance(value, list | tuple) or len(value) != player_count:
        raise ReplayCompatibilityError(
            f"action_sidecar {key} must have shape [player_count, action_count]"
        )
    matrix: list[list[bool]] = []
    for row in value:
        if not isinstance(row, list | tuple) or len(row) != action_count:
            raise ReplayCompatibilityError(
                f"action_sidecar {key} must have shape [player_count, action_count]"
            )
        if any(not isinstance(item, bool) for item in row):
            raise ReplayCompatibilityError(f"action_sidecar {key} must contain booleans")
        matrix.append([bool(item) for item in row])
    return matrix


def _metadata_bool(metadata: Mapping[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise ReplayCompatibilityError(f"{key} must be a boolean")
    return value


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayCompatibilityError(f"{key} must be an integer")
    return value


def _metadata_string(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ReplayCompatibilityError(f"{key} must be a non-empty string")
    return value


def _metadata_number(metadata: Mapping[str, Any], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReplayCompatibilityError(f"{key} must be a number")
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ReplayCompatibilityError(f"{key} must be finite")
    return number


def _metadata_id_scalar(metadata: Mapping[str, Any], key: str) -> int | str:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ReplayCompatibilityError(f"{key} must be an integer or string id")
    if isinstance(value, str) and not value:
        raise ReplayCompatibilityError(f"{key} must be non-empty")
    return value


def _expect_optional_rng_metadata(metadata: Mapping[str, Any]) -> None:
    if "reset_provenance" in metadata:
        value = metadata["reset_provenance"]
        if not isinstance(value, Mapping):
            raise ReplayCompatibilityError("reset_provenance must be a metadata object")
        schema_id = value.get("schema_id")
        if not isinstance(schema_id, str) or not schema_id:
            raise ReplayCompatibilityError("reset_provenance schema_id must be non-empty")
        if value.get("seed_alone_replay_complete") is not False:
            raise ReplayCompatibilityError(
                "reset_provenance seed_alone_replay_complete must be false"
            )
    if "random_tape_source" in metadata:
        value = metadata["random_tape_source"]
        if value is not None and (not isinstance(value, str) or not value):
            raise ReplayCompatibilityError(
                "random_tape_source must be null or a non-empty string"
            )
    if "random_tape_length" in metadata:
        value = metadata["random_tape_length"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReplayCompatibilityError("random_tape_length must be a non-negative integer")
    if "rng_impl_id" in metadata:
        value = metadata["rng_impl_id"]
        if value is not None and (not isinstance(value, str) or not value):
            raise ReplayCompatibilityError("rng_impl_id must be null or a non-empty string")
    if "source_fixture_ref" in metadata:
        value = metadata["source_fixture_ref"]
        if value is not None and (not isinstance(value, str) or not value):
            raise ReplayCompatibilityError(
                "source_fixture_ref must be null or a non-empty string"
            )
