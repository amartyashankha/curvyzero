"""Reward/RND attestation for compact-owned trainer candidates."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from curvyzero.contracts.curvytron import (
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
)
from curvyzero.training.exploration_bonus import (
    EXPLORATION_BONUS_MODE_NONE,
    normalize_exploration_bonus_config,
)
from curvyzero.training.reward_contracts import DEFAULT_REWARD_OUTCOME_ALPHA
from curvyzero.training.reward_contracts import ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
from curvyzero.training.reward_contracts import normalize_reward_outcome_alpha
from curvyzero.training.reward_contracts import normalize_reward_variant_for_env
from curvyzero.training.reward_contracts import reward_perspective_for_variant
from curvyzero.training.reward_contracts import reward_policy_for_variant
from curvyzero.training.reward_contracts import reward_schema_hash_for_variant


COMPACT_REWARD_RND_CONTRACT_SCHEMA_ID = "curvyzero_compact_reward_rnd_contract/v1"
COMPACT_REWARD_RND_CONTRACT_MODE_EXTRINSIC_NO_RND = "extrinsic_reward_no_rnd_v1"
DEFAULT_COMPACT_REWARD_RND_ENV_VARIANT = ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
DEFAULT_COMPACT_REWARD_RND_REWARD_VARIANT = (
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
)


class CompactRewardRndContractError(ValueError):
    """Raised when a compact reward/RND contract overclaims support."""


def build_compact_reward_rnd_contract_v1(
    *,
    env_variant: str = DEFAULT_COMPACT_REWARD_RND_ENV_VARIANT,
    reward_variant: str = DEFAULT_COMPACT_REWARD_RND_REWARD_VARIANT,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    exploration_bonus_config: Any = None,
    rnd_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the current compact-owned reward/RND contract.

    The first compact-owned candidate is intentionally no-RND.  RND adapter
    probes exist elsewhere, but the compact trainer does not yet own RND
    updates, target-reward mutation, or checkpoint/resume lineage.
    """

    env_value = str(env_variant)
    reward_value = normalize_reward_variant_for_env(
        env_variant=env_value,
        reward_variant=str(reward_variant),
    )
    reward_alpha = normalize_reward_outcome_alpha(reward_outcome_alpha)
    reward_policy = reward_policy_for_variant(
        env_variant=env_value,
        reward_variant=reward_value,
        reward_outcome_alpha=reward_alpha,
    )
    exploration_bonus = normalize_exploration_bonus_config(exploration_bonus_config)
    if exploration_bonus.enabled:
        raise CompactRewardRndContractError(
            "compact-owned reward/RND contract does not support RND-enabled "
            "exploration_bonus yet"
        )
    if rnd_state_dict is not None:
        raise CompactRewardRndContractError(
            "rnd_state_dict requires an RND-enabled compact reward/RND contract"
        )

    exploration_payload = _compact_exploration_bonus_payload(exploration_bonus)
    contract = {
        "schema_id": COMPACT_REWARD_RND_CONTRACT_SCHEMA_ID,
        "compact_reward_rnd_contract_schema_id": (
            COMPACT_REWARD_RND_CONTRACT_SCHEMA_ID
        ),
        "compact_reward_rnd_contract_mode": (
            COMPACT_REWARD_RND_CONTRACT_MODE_EXTRINSIC_NO_RND
        ),
        "compact_reward_rnd_contract_verified": True,
        "env_variant": env_value,
        "reward_variant": reward_value,
        "reward_schema_id": str(reward_policy["reward_schema_id"]),
        "reward_schema_hash": reward_schema_hash_for_variant(reward_value),
        "reward_perspective": reward_perspective_for_variant(reward_value),
        "reward_outcome_alpha": float(reward_alpha),
        "reward_policy": _plain_mapping(reward_policy),
        "reward_target_source": "compact_replay_extrinsic_reward",
        "reward_target_effect": "extrinsic_reward_only",
        "reward_target_intrinsic_mutation": False,
        "exploration_bonus": exploration_payload,
        "exploration_bonus_mode": exploration_bonus.mode,
        "exploration_bonus_enabled": False,
        "exploration_bonus_config_hash": exploration_bonus.config_hash(),
        "exploration_bonus_target_reward_effect": exploration_bonus.target_reward_effect,
        "rnd_enabled": False,
        "rnd_state_required": False,
        "rnd_state_present": False,
        "rnd_update_supported": False,
        "rnd_reward_target_supported": False,
        "rnd_checkpoint_state_supported": False,
        "rnd_training_claim": False,
        "intrinsic_reward_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
        "non_claims": [
            "no_rnd_training_claim",
            "no_intrinsic_reward_claim",
            "no_reward_model_entrypoint_claim",
            "not_a_promotion_claim",
        ],
    }
    validate_compact_reward_rnd_contract_v1(contract, rnd_state_dict=rnd_state_dict)
    return contract


def validate_compact_reward_rnd_contract_v1(
    contract: Any,
    *,
    rnd_state_dict: Mapping[str, Any] | None = None,
) -> None:
    """Validate a compact reward/RND contract and fail on overclaims."""

    if not isinstance(contract, Mapping):
        raise CompactRewardRndContractError("compact reward/RND contract must be a mapping")
    schema_id = contract.get("schema_id") or contract.get(
        "compact_reward_rnd_contract_schema_id"
    )
    if schema_id != COMPACT_REWARD_RND_CONTRACT_SCHEMA_ID:
        raise CompactRewardRndContractError("compact reward/RND contract schema mismatch")
    if (
        contract.get("compact_reward_rnd_contract_mode")
        != COMPACT_REWARD_RND_CONTRACT_MODE_EXTRINSIC_NO_RND
    ):
        raise CompactRewardRndContractError("unsupported compact reward/RND contract mode")
    if contract.get("compact_reward_rnd_contract_verified") is not True:
        raise CompactRewardRndContractError("compact reward/RND contract is not verified")

    env_value = str(contract.get("env_variant") or "")
    reward_value = normalize_reward_variant_for_env(
        env_variant=env_value,
        reward_variant=str(contract.get("reward_variant") or ""),
    )
    reward_alpha = normalize_reward_outcome_alpha(
        contract.get("reward_outcome_alpha", DEFAULT_REWARD_OUTCOME_ALPHA)
    )
    reward_policy = reward_policy_for_variant(
        env_variant=env_value,
        reward_variant=reward_value,
        reward_outcome_alpha=reward_alpha,
    )
    if contract.get("reward_variant") != reward_value:
        raise CompactRewardRndContractError("compact reward/RND reward_variant mismatch")
    if contract.get("reward_schema_id") != reward_policy["reward_schema_id"]:
        raise CompactRewardRndContractError("compact reward/RND reward_schema_id mismatch")
    if contract.get("reward_schema_hash") != reward_schema_hash_for_variant(reward_value):
        raise CompactRewardRndContractError("compact reward/RND reward_schema_hash mismatch")
    if contract.get("reward_perspective") != reward_perspective_for_variant(reward_value):
        raise CompactRewardRndContractError("compact reward/RND reward_perspective mismatch")
    if _plain_mapping(contract.get("reward_policy", {})) != _plain_mapping(reward_policy):
        raise CompactRewardRndContractError("compact reward/RND reward_policy mismatch")

    exploration_bonus = normalize_exploration_bonus_config(
        contract.get("exploration_bonus")
    )
    if exploration_bonus.mode != EXPLORATION_BONUS_MODE_NONE:
        raise CompactRewardRndContractError(
            "compact reward/RND contract currently requires exploration_bonus mode='none'"
        )
    if contract.get("exploration_bonus_mode") != EXPLORATION_BONUS_MODE_NONE:
        raise CompactRewardRndContractError("compact reward/RND exploration mode mismatch")
    if contract.get("exploration_bonus_enabled") is not False:
        raise CompactRewardRndContractError("compact reward/RND exploration must be disabled")
    if contract.get("exploration_bonus_config_hash") != exploration_bonus.config_hash():
        raise CompactRewardRndContractError(
            "compact reward/RND exploration config hash mismatch"
        )
    if contract.get("reward_target_effect") != "extrinsic_reward_only":
        raise CompactRewardRndContractError("compact reward/RND target effect mismatch")
    if contract.get("reward_target_intrinsic_mutation") is not False:
        raise CompactRewardRndContractError(
            "compact reward/RND must not mutate rewards with intrinsic bonuses"
        )

    for field in (
        "rnd_enabled",
        "rnd_state_required",
        "rnd_state_present",
        "rnd_update_supported",
        "rnd_reward_target_supported",
        "rnd_checkpoint_state_supported",
        "rnd_training_claim",
        "intrinsic_reward_claim",
        "calls_train_muzero",
        "touches_live_runs",
        "promotion_claim",
    ):
        if contract.get(field) is not False:
            raise CompactRewardRndContractError(
                f"compact reward/RND contract overclaims {field}"
            )
    if rnd_state_dict is not None:
        raise CompactRewardRndContractError(
            "compact reward/RND contract is no-RND but rnd_state_dict is present"
        )


def compact_reward_rnd_contract_evidence_ref(contract: Mapping[str, Any]) -> str:
    """Return a compact evidence token for compatibility reports."""

    validate_compact_reward_rnd_contract_v1(contract)
    return ":".join(
        (
            COMPACT_REWARD_RND_CONTRACT_SCHEMA_ID,
            str(contract["env_variant"]),
            str(contract["reward_variant"]),
            str(contract["reward_schema_hash"])[:12],
            f"xb={contract['exploration_bonus_config_hash'][:12]}",
            "rnd=none",
        )
    )


def _plain_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(value) for key, value in dict(metadata).items()}


def _compact_exploration_bonus_payload(exploration_bonus: Any) -> dict[str, Any]:
    payload = exploration_bonus.as_dict()
    return {
        "schema_id": payload["schema_id"],
        "mode": payload["mode"],
        "weight": payload["weight"],
        "training_only": payload["training_only"],
        "training_effect": payload["training_effect"],
        "target_reward_effect": payload["target_reward_effect"],
        "config_hash": payload["config_hash"],
    }


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


__all__ = [
    "COMPACT_REWARD_RND_CONTRACT_MODE_EXTRINSIC_NO_RND",
    "COMPACT_REWARD_RND_CONTRACT_SCHEMA_ID",
    "CompactRewardRndContractError",
    "build_compact_reward_rnd_contract_v1",
    "compact_reward_rnd_contract_evidence_ref",
    "validate_compact_reward_rnd_contract_v1",
]
