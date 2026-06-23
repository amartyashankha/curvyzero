import pytest

from curvyzero.training import reward_contracts as rc


def test_auto_reward_variant_normalizes_per_source_state_env():
    assert (
        rc.normalize_reward_variant_for_env(
            env_variant=rc.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=rc.REWARD_VARIANT_AUTO,
        )
        == rc.REWARD_VARIANT_SPARSE_OUTCOME
    )
    assert (
        rc.normalize_reward_variant_for_env(
            env_variant=rc.ENV_VARIANT_SOURCE_STATE_JOINT_ACTION,
            reward_variant=rc.REWARD_VARIANT_AUTO,
        )
        == rc.REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC
    )


def test_reward_contract_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="reward_outcome_alpha"):
        rc.normalize_reward_outcome_alpha(-0.1)
    with pytest.raises(ValueError, match="reward_outcome_alpha"):
        rc.normalize_reward_outcome_alpha(1.1)


def test_survival_plus_bonus_no_outcome_support_matches_current_contract():
    target = rc.lightzero_target_config_for_reward(
        env_variant=rc.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=rc.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        source_max_steps=65_536,
    )

    assert target["uncapped_model_reward_support_scale"] == 2
    assert target["uncapped_model_value_support_scale"] == 131_072
    assert target["model_reward_support_capped"] is False
    assert target["model_value_support_capped"] is True
    assert target["model_support_cap"] == 300
    assert target["model_reward_support_size"] == 601
    assert target["model_value_support_size"] == 601
    assert "td_steps" not in target


def test_survival_plus_bonus_plus_outcome_alpha_affects_support_and_policy():
    policy = rc.reward_policy_for_variant(
        env_variant=rc.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=rc.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        reward_outcome_alpha=0.5,
    )
    target = rc.lightzero_target_config_for_reward(
        env_variant=rc.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=rc.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        source_max_steps=100,
        reward_outcome_alpha=0.5,
    )

    assert policy["reward_outcome_alpha"] == 0.5
    assert policy["terminal_outcome_bonus"] == 0.5
    assert policy["winner_bonus"] == 0.5
    assert policy["loser_penalty"] == -0.5
    assert target["uncapped_model_reward_support_scale"] == 52
    assert target["uncapped_model_value_support_scale"] == 250
    assert target["model_reward_support_capped"] is False
    assert target["model_value_support_capped"] is False


def test_fixed_opponent_support_cap_and_td_steps_validate():
    target = rc.lightzero_target_config_for_reward(
        env_variant=rc.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=rc.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        source_max_steps=65_536,
        model_support_cap=50,
        td_steps=7,
    )

    assert target["model_support_cap"] == 50
    assert target["model_support_scale"] == 50
    assert target["td_steps"] == 7

    with pytest.raises(ValueError, match="model_support_cap"):
        rc.lightzero_target_config_for_reward(
            env_variant=rc.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=rc.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            source_max_steps=65_536,
            model_support_cap=0,
        )
    with pytest.raises(ValueError, match="td_steps"):
        rc.lightzero_target_config_for_reward(
            env_variant=rc.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=rc.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            source_max_steps=65_536,
            td_steps=0,
        )


def test_reward_space_documents_repeat_max_bounds_without_changing_support():
    space = rc.reward_space_for_variant(
        reward_variant=rc.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        max_source_ticks=65_536,
        policy_action_repeat_max=3,
    )
    target = rc.lightzero_target_config_for_reward(
        env_variant=rc.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=rc.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        source_max_steps=65_536,
    )

    assert space["low"] == 0.0
    assert space["high"] == 6.0
    assert target["uncapped_model_reward_support_scale"] == 2
