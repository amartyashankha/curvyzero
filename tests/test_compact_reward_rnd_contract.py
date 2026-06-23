import pytest

from curvyzero.contracts.curvytron import (
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
)
from curvyzero.training.compact_reward_rnd_contract import (
    CompactRewardRndContractError,
)
from curvyzero.training.compact_reward_rnd_contract import (
    build_compact_reward_rnd_contract_v1,
)
from curvyzero.training.compact_reward_rnd_contract import (
    compact_reward_rnd_contract_evidence_ref,
)
from curvyzero.training.compact_reward_rnd_contract import (
    validate_compact_reward_rnd_contract_v1,
)
from curvyzero.training.exploration_bonus import normalize_exploration_bonus_spec
from curvyzero.training.reward_contracts import (
    SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID,
)


def test_default_compact_reward_rnd_contract_is_explicit_no_rnd():
    contract = build_compact_reward_rnd_contract_v1()

    assert contract["compact_reward_rnd_contract_verified"] is True
    assert contract["reward_variant"] == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
    assert (
        contract["reward_schema_id"]
        == SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID
    )
    assert contract["reward_target_effect"] == "extrinsic_reward_only"
    assert contract["reward_target_intrinsic_mutation"] is False
    assert contract["exploration_bonus_mode"] == "none"
    assert contract["exploration_bonus_enabled"] is False
    assert contract["rnd_enabled"] is False
    assert contract["rnd_state_required"] is False
    assert contract["rnd_state_present"] is False
    assert contract["rnd_update_supported"] is False
    assert contract["rnd_checkpoint_state_supported"] is False
    assert contract["rnd_training_claim"] is False
    assert contract["intrinsic_reward_claim"] is False
    assert "rnd=none" in compact_reward_rnd_contract_evidence_ref(contract)


def test_compact_reward_rnd_contract_rejects_rnd_enabled_config_and_state():
    rnd_meter = normalize_exploration_bonus_spec(mode="rnd_meter_v0").as_dict()

    with pytest.raises(CompactRewardRndContractError, match="does not support RND"):
        build_compact_reward_rnd_contract_v1(exploration_bonus_config=rnd_meter)
    with pytest.raises(CompactRewardRndContractError, match="rnd_state_dict"):
        build_compact_reward_rnd_contract_v1(rnd_state_dict={"predictor": {}})


def test_compact_reward_rnd_contract_validation_fails_on_overclaims():
    contract = build_compact_reward_rnd_contract_v1()
    rnd_overclaim = dict(contract)
    rnd_overclaim["rnd_enabled"] = True
    with pytest.raises(CompactRewardRndContractError, match="rnd_enabled"):
        validate_compact_reward_rnd_contract_v1(rnd_overclaim)

    stale_reward = dict(contract)
    stale_reward["reward_schema_hash"] = "stale"
    with pytest.raises(CompactRewardRndContractError, match="reward_schema_hash"):
        validate_compact_reward_rnd_contract_v1(stale_reward)

    stale_xb = dict(contract)
    stale_xb["exploration_bonus_mode"] = "rnd_meter_v0"
    with pytest.raises(CompactRewardRndContractError, match="exploration mode"):
        validate_compact_reward_rnd_contract_v1(stale_xb)
