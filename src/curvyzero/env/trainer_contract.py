"""Trainer-facing observation, action, reward, and adapter contract constants.

This module pins the first non-debug CurvyZero trainer contract without
implementing observation generation. It is intentionally small: code that casts
rays, steps environments, writes replay, or adapts LightZero should import these
ids and hashes instead of retyping schema details.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

TRAINER_ADAPTER_CONTRACT_ID = "curvyzero_trainer_adapter_contract/v0"
OBSERVATION_SCHEMA_ID = "curvyzero_egocentric_rays/v0"
LEGACY_OBSERVATION_SCHEMA_IDS = ("curvyzero-observe-v0-rays",)
ACTION_SPACE_ID = "curvyzero_turn3/v0"
REWARD_SCHEMA_ID = "curvyzero_sparse_round_outcome/v0"
NATIVE_CONTROL_MODEL_ID = "curvytron_realtime_controls_elapsed_frames/v0"
TRAINER_CONTROL_WRAPPER_ID = "curvyzero_fixed_decision_wrapper/v0"

OBSERVATION_DTYPE = "float32"
ACTION_MASK_DTYPE = "bool"
LIGHTZERO_ACTION_MASK_DTYPE = "int8"
REWARD_DTYPE = "float32"

RAY_COUNT = 24
RAY_CHANNEL_NAMES = (
    "wall_or_out_of_bounds",
    "own_trail",
    "opponent_trail",
    "opponent_head",
)
RAY_ANGLES_DEGREES = tuple(range(0, 360, 15))

SCALAR_NAMES = (
    "ego_alive",
    "opponent_alive",
    "tick_fraction",
    "opponent_rel_x_clipped",
    "opponent_rel_y_clipped",
    "opponent_heading_sin_relative",
    "opponent_heading_cos_relative",
    "speed_norm",
    "turn_rate_norm",
    "trail_radius_norm",
)

STRUCTURED_OBSERVATION_SHAPE = {
    "rays": (RAY_COUNT, len(RAY_CHANNEL_NAMES)),
    "scalars": (len(SCALAR_NAMES),),
}
LIGHTZERO_FLAT_OBSERVATION_SHAPE = (
    RAY_COUNT * len(RAY_CHANNEL_NAMES) + len(SCALAR_NAMES),
)

ACTION_NAMES = ("left", "straight", "right")
ACTION_ID_TO_SOURCE_MOVE = (-1, 0, 1)
LIVE_ACTION_MASK = (True, True, True)
STRICT_LEFT_RIGHT_ACTION_MASK = (True, False, True)
INACTIVE_ACTION_MASK = (False, False, False)

TERMINAL_REASON_VALUES = (
    "none",
    "survivor_win",
    "all_dead_draw",
    "timeout",
    "horizon_truncated",
    "event_overflow_truncated",
    "infra_truncated",
)
TRUNCATION_REASON_VALUES = (
    None,
    "max_ticks",
    "horizon",
    "event_overflow",
    "infra",
)

RESET_INFO_KEYS = (
    "episode_id",
    "seed",
    "ruleset_id",
    "rules_hash",
    "observation_schema_id",
    "observation_schema_hash",
    "action_space_id",
    "action_space_hash",
    "reward_schema_id",
    "reward_schema_hash",
    "player_ids",
    "max_players",
    "env_impl_id",
)

STEP_INFO_KEYS = RESET_INFO_KEYS + (
    "ego_player_id",
    "step_index",
    "tick_index",
    "joint_action",
    "opponent_policy_id",
    "opponent_policy_version",
    "terminal_reason",
    "winner_ids",
    "loser_ids",
    "death_player_ids",
    "draw",
    "timeout",
    "truncation_reason",
    "done",
    "terminated",
    "truncated",
    "needs_reset",
    "final_observation",
    "final_reward_map",
    "event_ref",
    "event_range",
    "state_ref",
    "trace_ref",
    "trace_hash",
)

LIGHTZERO_TERMINAL_INFO_KEYS = STEP_INFO_KEYS + ("eval_episode_return",)

ACTION_SPACE_SCHEMA: dict[str, Any] = {
    "schema_id": ACTION_SPACE_ID,
    "native_control_model_id": NATIVE_CONTROL_MODEL_ID,
    "native_control_model": (
        "Source CurvyTron stores real-time player control state and advances "
        "server frames by elapsed milliseconds; it does not natively expose a "
        "trainer-style batched action step."
    ),
    "trainer_control_wrapper_id": TRAINER_CONTROL_WRAPPER_ID,
    "trainer_control_wrapper": (
        "Trainer wrappers may sample one action id per live player at a fixed "
        "decision cadence, convert those ids to source moves, hold that control "
        "state during the elapsed-ms frame window, and label replay rows with "
        "the wrapper cadence."
    ),
    "action_count": len(ACTION_NAMES),
    "action_order": ACTION_NAMES,
    "action_id_to_source_move": ACTION_ID_TO_SOURCE_MOVE,
    "mask_dtype": ACTION_MASK_DTYPE,
    "lightzero_mask_dtype": LIGHTZERO_ACTION_MASK_DTYPE,
    "live_mask": LIVE_ACTION_MASK,
    "strict_left_right_mask": STRICT_LEFT_RIGHT_ACTION_MASK,
    "inactive_mask": INACTIVE_ACTION_MASK,
}

OBSERVATION_SCHEMA: dict[str, Any] = {
    "schema_id": OBSERVATION_SCHEMA_ID,
    "legacy_schema_ids": LEGACY_OBSERVATION_SCHEMA_IDS,
    "dtype": OBSERVATION_DTYPE,
    "structured_shape": STRUCTURED_OBSERVATION_SHAPE,
    "lightzero_flat_observation_shape": LIGHTZERO_FLAT_OBSERVATION_SHAPE,
    "flat_pack_order": (
        "rays row-major by ray index then channel index",
        "scalars in listed order",
    ),
    "action_mask_is_separate": True,
    "ray_angles_degrees": RAY_ANGLES_DEGREES,
    "ray_angle_convention": (
        "degrees ego-left/counter-clockwise from ego heading in local coordinates; "
        "0 is straight ahead and angles wrap modulo 360"
    ),
    "ray_channels": RAY_CHANNEL_NAMES,
    "ray_value_policy": {
        "range": (0.0, 1.0),
        "normalization": "clip distance to arena diagonal and divide by arena diagonal",
        "no_hit": 1.0,
        "immediate_contact": 0.0,
        "borderless_boundary": (
            "normal wrap boundaries are not wall hits; return no_hit unless another "
            "channel is hit"
        ),
    },
    "scalar_names": SCALAR_NAMES,
    "scalar_policy": {
        "ego_alive": "1.0 if ego is alive else 0.0",
        "opponent_alive": "1.0 if the single v0 opponent is alive else 0.0",
        "tick_fraction": "decision tick divided by horizon, clipped to [0, 1]",
        "opponent_rel_x_clipped": (
            "opponent x in ego frame divided by arena diagonal and clipped to [-1, 1]"
        ),
        "opponent_rel_y_clipped": (
            "opponent y in ego-left frame divided by arena diagonal and clipped to [-1, 1]"
        ),
        "opponent_heading_sin_relative": "sin(opponent_heading - ego_heading)",
        "opponent_heading_cos_relative": "cos(opponent_heading - ego_heading)",
        "speed_norm": "ego speed per decision divided by arena diagonal",
        "turn_rate_norm": "absolute turn rate in radians divided by pi",
        "trail_radius_norm": "collision/trail radius divided by arena diagonal",
    },
    "hidden_state_exclusions": (
        "stable_player_index",
        "seat_id",
        "color",
        "absolute_x",
        "absolute_y",
        "source_trace_fields",
        "debug_event_rows",
    ),
}

REWARD_SCHEMA: dict[str, Any] = {
    "schema_id": REWARD_SCHEMA_ID,
    "dtype": REWARD_DTYPE,
    "episode_unit": "one_round",
    "perspective": "ego_player",
    "alignment": "reward_t_plus_1_after_wrapper_decision_t",
    "alignment_note": (
        "The reward is aligned to the trainer wrapper decision boundary. Native "
        "source CurvyTron advances elapsed-ms server frames under held control "
        "state, so fixed decision steps are wrapper/replay semantics."
    ),
    "nonterminal_reward": 0.0,
    "winner_reward": 1.0,
    "loser_reward": -1.0,
    "all_dead_draw_reward": 0.0,
    "pure_truncation_reward": 0.0,
    "terminated_and_truncated_policy": (
        "terminal outcome reward wins; truncated remains true in info"
    ),
    "shaping_terms": (),
}

TRAINER_ADAPTER_CONTRACT: dict[str, Any] = {
    "schema_id": TRAINER_ADAPTER_CONTRACT_ID,
    "observation_schema_id": OBSERVATION_SCHEMA_ID,
    "action_space_id": ACTION_SPACE_ID,
    "reward_schema_id": REWARD_SCHEMA_ID,
    "native_control_model_id": NATIVE_CONTROL_MODEL_ID,
    "trainer_control_wrapper_id": TRAINER_CONTROL_WRAPPER_ID,
    "reset_returns": {
        "observation": {
            "observation": {
                "shape": LIGHTZERO_FLAT_OBSERVATION_SHAPE,
                "dtype": OBSERVATION_DTYPE,
            },
            "action_mask": {
                "shape": (len(ACTION_NAMES),),
                "dtype": LIGHTZERO_ACTION_MASK_DTYPE,
            },
            "to_play": -1,
        },
        "info_keys": RESET_INFO_KEYS,
    },
    "step_returns": {
        "observation": {
            "observation": {
                "shape": LIGHTZERO_FLAT_OBSERVATION_SHAPE,
                "dtype": OBSERVATION_DTYPE,
            },
            "action_mask": {
                "shape": (len(ACTION_NAMES),),
                "dtype": LIGHTZERO_ACTION_MASK_DTYPE,
            },
            "to_play": -1,
        },
        "reward": {"shape": (), "dtype": REWARD_DTYPE},
        "done": "terminated OR truncated",
        "info_keys": STEP_INFO_KEYS,
        "terminal_info_keys": LIGHTZERO_TERMINAL_INFO_KEYS,
    },
    "autoreset": (
        "No hidden autoreset may discard the terminal transition; any public autoreset "
        "must return final observation/reward/info before resetting the row."
    ),
}


def stable_contract_hash(payload: Mapping[str, Any]) -> str:
    """Return the project short hash for canonical contract payloads."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


OBSERVATION_SCHEMA_HASH = stable_contract_hash(OBSERVATION_SCHEMA)
ACTION_SPACE_HASH = stable_contract_hash(ACTION_SPACE_SCHEMA)
REWARD_SCHEMA_HASH = stable_contract_hash(REWARD_SCHEMA)
TRAINER_ADAPTER_CONTRACT_HASH = stable_contract_hash(
    {
        "adapter": TRAINER_ADAPTER_CONTRACT,
        "action_space": ACTION_SPACE_SCHEMA,
        "observation": OBSERVATION_SCHEMA,
        "reward": REWARD_SCHEMA,
    }
)


def legal_action_mask(*, active: bool, allow_straight: bool = True) -> tuple[bool, bool, bool]:
    """Return the canonical trainer mask in left/straight/right order."""

    if not active:
        return INACTIVE_ACTION_MASK
    if not allow_straight:
        return STRICT_LEFT_RIGHT_ACTION_MASK
    return LIVE_ACTION_MASK


__all__ = [
    "ACTION_ID_TO_SOURCE_MOVE",
    "ACTION_MASK_DTYPE",
    "ACTION_NAMES",
    "ACTION_SPACE_HASH",
    "ACTION_SPACE_ID",
    "ACTION_SPACE_SCHEMA",
    "INACTIVE_ACTION_MASK",
    "LEGACY_OBSERVATION_SCHEMA_IDS",
    "LIGHTZERO_ACTION_MASK_DTYPE",
    "LIGHTZERO_FLAT_OBSERVATION_SHAPE",
    "LIGHTZERO_TERMINAL_INFO_KEYS",
    "LIVE_ACTION_MASK",
    "OBSERVATION_DTYPE",
    "OBSERVATION_SCHEMA",
    "OBSERVATION_SCHEMA_HASH",
    "OBSERVATION_SCHEMA_ID",
    "RAY_ANGLES_DEGREES",
    "RAY_CHANNEL_NAMES",
    "RAY_COUNT",
    "RESET_INFO_KEYS",
    "REWARD_DTYPE",
    "REWARD_SCHEMA",
    "REWARD_SCHEMA_HASH",
    "REWARD_SCHEMA_ID",
    "SCALAR_NAMES",
    "STEP_INFO_KEYS",
    "STRICT_LEFT_RIGHT_ACTION_MASK",
    "STRUCTURED_OBSERVATION_SHAPE",
    "TERMINAL_REASON_VALUES",
    "TRAINER_ADAPTER_CONTRACT",
    "TRAINER_ADAPTER_CONTRACT_HASH",
    "TRAINER_ADAPTER_CONTRACT_ID",
    "TRUNCATION_REASON_VALUES",
    "legal_action_mask",
    "stable_contract_hash",
]
