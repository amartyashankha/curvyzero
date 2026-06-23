"""Reward contract helpers shared by CurvyZero training surfaces."""

from __future__ import annotations

import math
from typing import Any

from curvyzero.contracts.curvytron import (
    REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
)
from curvyzero.env.trainer_contract import (
    REWARD_SCHEMA_HASH as SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_HASH,
)
from curvyzero.env.trainer_contract import (
    REWARD_SCHEMA_ID as SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
)
from curvyzero.env.trainer_contract import stable_contract_hash


ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT = "source_state_fixed_opponent"
ENV_VARIANT_SOURCE_STATE_JOINT_ACTION = "source_state_joint_action"

REWARD_VARIANT_AUTO = "auto"
DEFAULT_REWARD_VARIANT = REWARD_VARIANT_AUTO
DEFAULT_REWARD_OUTCOME_ALPHA = 1.0

SOURCE_STATE_FIXED_OPPONENT_MAX_MODEL_SUPPORT_SCALE = 300
DEFAULT_MODEL_SUPPORT_CAP: int | None = None
DEFAULT_TD_STEPS: int | None = None

SOURCE_STATE_FIXED_OPPONENT_REWARD_VARIANTS = (
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
)
SOURCE_STATE_JOINT_ACTION_REWARD_VARIANTS = (REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC,)

REWARD_VARIANT_CHOICES = (
    REWARD_VARIANT_AUTO,
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
    REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
)

DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_ID = "curvyzero_dense_survival_plus_sparse_outcome/v0"
DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA = {
    "schema_id": DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_ID,
    "dtype": "float32",
    "episode_unit": "one_round",
    "perspective": "ego_player",
    "alignment": "reward_t_plus_1_after_one_source_tick",
    "trainer_reward_terms": [
        "dense_alive_helper_for_ego_player",
        "sparse_round_outcome_for_ego_player",
    ],
    "dense_alive_helper": 1.0,
    "sparse_round_outcome_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
    "survival_length_metric_is_telemetry": True,
    "non_claims": [
        "not_zero_sum_after_dense_helper",
        "not_two_seat_self_play",
        "not_a_learning_claim",
    ],
}
DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_HASH = stable_contract_hash(
    DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA
)

SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD = 1.0
SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID = "curvyzero_survival_plus_bonus_no_outcome/v0"
SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA = {
    "schema_id": SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID,
    "dtype": "float32",
    "episode_unit": "one_round",
    "perspective": "ego_player",
    "alignment": "reward_t_plus_1_after_one_source_tick",
    "trainer_reward_terms": [
        "dense_alive_helper_for_ego_player",
        "same_step_bonus_pickup_helper_for_ego_player",
    ],
    "dense_alive_helper": 1.0,
    "bonus_pickup_reward_per_catch": SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD,
    "bonus_pickup_source": "bonus_catch_count_step[0, ego_player_index]",
    "sparse_round_outcome_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
    "sparse_round_outcome_is_telemetry_only": True,
    "terminal_outcome_bonus": 0.0,
    "loser_penalty": 0.0,
    "winner_bonus": 0.0,
    "draw_bonus": 0.0,
    "truncation_bonus": 0.0,
    "survival_length_metric_is_telemetry": True,
    "non_claims": [
        "not_zero_sum",
        "not_two_seat_self_play",
        "not_sparse_outcome_reward",
        "not_a_learning_claim",
    ],
}
SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_HASH = stable_contract_hash(
    SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA
)

SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_ID = (
    "curvyzero_survival_plus_bonus_plus_outcome/v0"
)
SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA = {
    "schema_id": SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_ID,
    "dtype": "float32",
    "episode_unit": "one_round",
    "perspective": "ego_player",
    "alignment": "reward_t_plus_1_after_one_source_tick",
    "trainer_reward_terms": [
        "dense_alive_helper_for_ego_player",
        "same_step_bonus_pickup_helper_for_ego_player",
        "terminal_sparse_outcome_scaled_by_accumulated_non_outcome_reward",
    ],
    "dense_alive_helper": 1.0,
    "bonus_pickup_reward_per_catch": SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD,
    "bonus_pickup_source": "bonus_catch_count_step[0, ego_player_index]",
    "sparse_round_outcome_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
    "sparse_round_outcome_is_telemetry_only": False,
    "terminal_outcome_alpha_default": DEFAULT_REWARD_OUTCOME_ALPHA,
    "terminal_outcome_scale": "accumulated_non_outcome_training_reward",
    "terminal_outcome_reward": (
        "sparse_round_outcome * reward_outcome_alpha * accumulated_non_outcome_reward"
    ),
    "terminal_outcome_bonus": 1.0,
    "loser_penalty": -1.0,
    "winner_bonus": 1.0,
    "draw_bonus": 0.0,
    "truncation_bonus": 0.0,
    "survival_length_metric_is_telemetry": True,
    "non_claims": [
        "not_zero_sum_after_dense_helper_and_bonus",
        "not_two_seat_self_play",
        "not_a_learning_claim",
    ],
}
SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_HASH = stable_contract_hash(
    SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA
)

ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID = "curvyzero_all_players_alive_diagnostic/v0"
ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA = {
    "schema_id": ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID,
    "dtype": "float32",
    "episode_unit": "one_round",
    "perspective": "centralized_joint_action_controller",
    "alignment": "reward_t_plus_1_after_one_source_tick",
    "reward_unit": "one_real_source_tick",
    "post_transition_all_players_alive_reward": 1.0,
    "post_transition_any_player_dead_reward": 0.0,
    "terminal_outcome_bonus": 0.0,
    "loser_penalty": 0.0,
    "winner_bonus": 0.0,
    "draw_bonus": 0.0,
    "truncation_bonus": 0.0,
    "episode_return": "sum of all-players-alive rewards for centralized control",
    "non_claims": [
        "not_per_player_reward",
        "not_zero_sum_reward",
        "not_true_competitive_self_play",
        "not_sparse_outcome_reward",
    ],
}
ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_HASH = stable_contract_hash(
    ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA
)


def normalize_reward_variant_for_env(*, env_variant: str, reward_variant: str) -> str:
    if reward_variant == REWARD_VARIANT_AUTO:
        if env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION:
            return REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC
        if env_variant == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
            return REWARD_VARIANT_SPARSE_OUTCOME
        return REWARD_VARIANT_AUTO
    return reward_variant


def normalize_reward_outcome_alpha(value: float) -> float:
    alpha = float(value)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("reward_outcome_alpha must be finite and in [0, 1]")
    return alpha


def reward_policy_for_variant(
    *,
    env_variant: str,
    reward_variant: str,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
) -> dict[str, Any]:
    reward_outcome_alpha = normalize_reward_outcome_alpha(reward_outcome_alpha)
    reward_variant = normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    if env_variant == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
        return _fixed_opponent_reward_policy(
            reward_variant=reward_variant,
            reward_outcome_alpha=reward_outcome_alpha,
        )
    if env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION:
        if reward_variant != REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            raise ValueError(
                "source_state_joint_action only supports reward_variant="
                f"{REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC!r}; got {reward_variant!r}"
            )
        return all_players_alive_diagnostic_reward_policy()
    raise ValueError(f"{env_variant} does not have a source-state reward contract")


def reward_schema_id_for_variant(*, env_variant: str, reward_variant: str) -> str:
    reward_policy = reward_policy_for_variant(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    return str(reward_policy["reward_schema_id"])


def reward_schema_hash_for_variant(reward_variant: str) -> str:
    if reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
        return SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_HASH
    if reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
        return DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_HASH
    if reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
        return SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_HASH
    if reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME:
        return SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_HASH
    if reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
        return ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_HASH
    raise RuntimeError(f"unsupported reward_variant {reward_variant!r}")


def reward_perspective_for_variant(reward_variant: str) -> str:
    if reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
        return "ego_player_sparse_round_outcome"
    if reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
        return "ego_player_dense_survival_helper_plus_sparse_outcome"
    if reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
        return "ego_player_dense_survival_plus_same_step_bonus_no_outcome"
    if reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME:
        return "ego_player_dense_survival_plus_same_step_bonus_plus_scaled_outcome"
    if reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
        return "diagnostic_all_players_alive_after_one_source_tick"
    raise RuntimeError(f"unsupported reward_variant {reward_variant!r}")


def reward_space_for_variant(
    *,
    reward_variant: str,
    max_source_ticks: int,
    policy_action_repeat_max: int,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
) -> dict[str, Any]:
    reward_outcome_alpha = normalize_reward_outcome_alpha(reward_outcome_alpha)
    if reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
        low = -1.0
        high = 1.0
    elif reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
        low = -1.0
        high = float(int(policy_action_repeat_max) + 1)
    elif reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
        low = 0.0
        high = float(
            int(policy_action_repeat_max) * (1.0 + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD)
        )
    elif reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME:
        low = -float(max_source_ticks) * float(reward_outcome_alpha)
        high = float(
            int(max_source_ticks) * float(reward_outcome_alpha)
            + int(policy_action_repeat_max) * (1.0 + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD)
        )
    elif reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
        low = 0.0
        high = 1.0
    else:
        raise RuntimeError(f"unsupported reward_variant {reward_variant!r}")
    return {
        "type": "Box",
        "shape": (),
        "dtype": "float32",
        "low": low,
        "high": high,
    }


def lightzero_target_config_for_reward(
    *,
    env_variant: str,
    reward_variant: str,
    source_max_steps: int,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP,
    td_steps: int | None = DEFAULT_TD_STEPS,
) -> dict[str, Any]:
    reward_outcome_alpha = normalize_reward_outcome_alpha(reward_outcome_alpha)
    reward_variant = normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    if env_variant == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
        return _fixed_opponent_lightzero_target_config(
            reward_variant=reward_variant,
            source_max_steps=source_max_steps,
            reward_outcome_alpha=reward_outcome_alpha,
            model_support_cap=model_support_cap,
            td_steps=td_steps,
        )
    if env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION:
        return _joint_action_lightzero_target_config(
            source_max_steps=source_max_steps,
            model_support_cap=model_support_cap,
            td_steps=td_steps,
        )
    return {}


def all_players_alive_diagnostic_reward_policy() -> dict[str, Any]:
    return {
        "reward_schema_id": ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID,
        "survival_only": False,
        "diagnostic_all_players_alive": True,
        "centralized_joint_action_control": True,
        "per_player_reward": False,
        "zero_sum_reward": False,
        "sparse_outcome_reward": False,
        "terminal_outcome_bonus": 0.0,
        "loser_penalty": 0.0,
        "winner_bonus": 0.0,
    }


def _fixed_opponent_reward_policy(
    *,
    reward_variant: str,
    reward_outcome_alpha: float,
) -> dict[str, Any]:
    if reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
        return {
            "reward_variant": reward_variant,
            "reward_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
            "survival_length_is_eval_metric": True,
            "dense_survival_reward": False,
            "sparse_outcome_reward": True,
            "survival_only": False,
            "diagnostic_all_players_alive": False,
            "centralized_joint_action_control": False,
            "per_player_reward": True,
            "zero_sum_reward": True,
            "nonterminal_reward": 0.0,
            "winner_bonus": 1.0,
            "loser_penalty": -1.0,
            "draw_bonus": 0.0,
            "truncation_bonus": 0.0,
        }
    if reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
        return {
            "reward_variant": reward_variant,
            "reward_schema_id": DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_ID,
            "survival_length_is_eval_metric": True,
            "dense_survival_reward": True,
            "dense_alive_helper": 1.0,
            "sparse_outcome_reward": True,
            "survival_only": False,
            "diagnostic_all_players_alive": False,
            "centralized_joint_action_control": False,
            "per_player_reward": True,
            "zero_sum_reward": False,
            "post_transition_alive_reward": 1.0,
            "post_transition_dead_reward": 0.0,
            "winner_bonus": 1.0,
            "loser_penalty": -1.0,
            "draw_bonus": 0.0,
            "truncation_bonus": 0.0,
        }
    if reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
        return {
            "reward_variant": reward_variant,
            "reward_schema_id": SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID,
            "survival_length_is_eval_metric": True,
            "dense_survival_reward": True,
            "dense_alive_helper": 1.0,
            "same_step_bonus_pickup_reward": True,
            "bonus_pickup_reward_per_catch": SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD,
            "bonus_pickup_source": "bonus_catch_count_step[0, ego_player_index]",
            "sparse_outcome_reward": False,
            "sparse_outcome_telemetry_only": True,
            "survival_only": False,
            "diagnostic_all_players_alive": False,
            "centralized_joint_action_control": False,
            "per_player_reward": True,
            "zero_sum_reward": False,
            "post_transition_alive_reward": 1.0,
            "post_transition_dead_reward": 0.0,
            "terminal_outcome_bonus": 0.0,
            "winner_bonus": 0.0,
            "loser_penalty": 0.0,
            "draw_bonus": 0.0,
            "truncation_bonus": 0.0,
        }
    if reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME:
        return {
            "reward_variant": reward_variant,
            "reward_schema_id": SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_ID,
            "survival_length_is_eval_metric": True,
            "dense_survival_reward": True,
            "dense_alive_helper": 1.0,
            "same_step_bonus_pickup_reward": True,
            "bonus_pickup_reward_per_catch": SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD,
            "bonus_pickup_source": "bonus_catch_count_step[0, ego_player_index]",
            "sparse_outcome_reward": True,
            "sparse_outcome_telemetry_only": False,
            "reward_outcome_alpha": float(reward_outcome_alpha),
            "terminal_outcome_scaled_by_episode_source_steps": False,
            "terminal_outcome_scaled_by_accumulated_non_outcome_reward": True,
            "terminal_outcome_scale": "accumulated_non_outcome_training_reward",
            "survival_only": False,
            "diagnostic_all_players_alive": False,
            "centralized_joint_action_control": False,
            "per_player_reward": True,
            "zero_sum_reward": False,
            "post_transition_alive_reward": 1.0,
            "post_transition_dead_reward": 0.0,
            "terminal_outcome_bonus": float(reward_outcome_alpha),
            "winner_bonus": float(reward_outcome_alpha),
            "loser_penalty": -float(reward_outcome_alpha),
            "draw_bonus": 0.0,
            "truncation_bonus": 0.0,
        }
    raise ValueError(
        "source_state_fixed_opponent reward_variant must be one of "
        f"{SOURCE_STATE_FIXED_OPPONENT_REWARD_VARIANTS!r}; got {reward_variant!r}"
    )


def _fixed_opponent_lightzero_target_config(
    *,
    reward_variant: str,
    source_max_steps: int,
    reward_outcome_alpha: float,
    model_support_cap: int | None,
    td_steps: int | None,
) -> dict[str, Any]:
    reward_support_scale = 1
    value_support_scale = 1
    if reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
        reward_support_scale = 2
        value_support_scale = int(source_max_steps) + 1
    if reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
        reward_support_scale = int(1.0 + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD)
        value_support_scale = int(
            source_max_steps * (1.0 + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD)
        )
    if reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME:
        reward_support_scale = int(
            source_max_steps * reward_outcome_alpha
            + 1.0
            + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
        )
        value_support_scale = int(
            source_max_steps
            * (1.0 + reward_outcome_alpha + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD)
        )
    max_support_scale = (
        SOURCE_STATE_FIXED_OPPONENT_MAX_MODEL_SUPPORT_SCALE
        if model_support_cap is None
        else int(model_support_cap)
    )
    if max_support_scale < 1:
        raise ValueError("model_support_cap must be >= 1 when provided")
    capped_reward_support_scale = min(int(reward_support_scale), max_support_scale)
    capped_value_support_scale = min(int(value_support_scale), max_support_scale)
    model_support_scale = max(capped_reward_support_scale, capped_value_support_scale)
    model_support_size = int(2 * model_support_scale + 1)
    model_support_range = (
        -float(model_support_scale),
        float(model_support_scale + 1),
        1.0,
    )
    target_config = {
        "discount_factor": 1.0,
        "model_support_scale": int(model_support_scale),
        "model_reward_support_size": model_support_size,
        "model_value_support_size": model_support_size,
        "model_reward_support_range": model_support_range,
        "model_value_support_range": model_support_range,
        "model_support_cap": int(max_support_scale),
        "model_reward_support_capped": int(reward_support_scale) > max_support_scale,
        "model_value_support_capped": int(value_support_scale) > max_support_scale,
        "model_reward_support_requested_scale": int(reward_support_scale),
        "model_value_support_requested_scale": int(value_support_scale),
        "model_reward_support_effective_scale": int(model_support_scale),
        "model_value_support_effective_scale": int(model_support_scale),
        "uncapped_model_reward_support_scale": int(reward_support_scale),
        "uncapped_model_value_support_scale": int(value_support_scale),
    }
    if td_steps is not None:
        if int(td_steps) < 1:
            raise ValueError("td_steps must be >= 1 when provided")
        target_config["td_steps"] = int(td_steps)
    return target_config


def _joint_action_lightzero_target_config(
    *,
    source_max_steps: int,
    model_support_cap: int | None,
    td_steps: int | None,
) -> dict[str, Any]:
    requested_support_scale = max(1, int(source_max_steps))
    if model_support_cap is None:
        support_scale = requested_support_scale
    else:
        if int(model_support_cap) < 1:
            raise ValueError("model_support_cap must be >= 1 when provided")
        support_scale = min(requested_support_scale, int(model_support_cap))
    target_config = {
        "discount_factor": 1.0,
        "td_steps": int(source_max_steps) if td_steps is None else int(td_steps),
        "model_support_scale": int(support_scale),
        "model_reward_support_size": int(2 * support_scale + 1),
        "model_value_support_size": int(2 * support_scale + 1),
        "model_support_cap": int(model_support_cap) if model_support_cap is not None else None,
        "model_reward_support_requested_scale": int(requested_support_scale),
        "model_value_support_requested_scale": int(requested_support_scale),
        "model_reward_support_effective_scale": int(support_scale),
        "model_value_support_effective_scale": int(support_scale),
        "uncapped_model_reward_support_scale": int(requested_support_scale),
        "uncapped_model_value_support_scale": int(requested_support_scale),
    }
    if int(target_config["td_steps"]) < 1:
        raise ValueError("td_steps must be >= 1 when provided")
    return target_config
