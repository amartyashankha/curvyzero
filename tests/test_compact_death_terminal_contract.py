import pytest

from curvyzero.env.vector_runtime import DEATH_MODE_NORMAL
from curvyzero.env.vector_runtime import DEATH_MODE_PROFILE_NO_DEATH
from curvyzero.training.compact_death_terminal_contract import (
    COMPACT_DEATH_TERMINAL_EVIDENCE_SCHEMA_ID,
)
from curvyzero.training.compact_death_terminal_contract import (
    COMPACT_DEATH_TERMINAL_MODE_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP,
)
from curvyzero.training.compact_death_terminal_contract import (
    COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP,
)
from curvyzero.training.compact_death_terminal_contract import (
    CompactDeathTerminalContractError,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_compact_death_terminal_contract_v1,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_normal_collision_death_contract_from_profile_result_v1,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_normal_collision_death_evidence_from_profile_result_v1,
)
from curvyzero.training.compact_death_terminal_contract import (
    compact_death_terminal_contract_evidence_ref,
)
from curvyzero.training.compact_death_terminal_contract import (
    validate_compact_death_terminal_contract_v1,
)


def test_default_compact_death_terminal_contract_is_profile_no_death_terminal_nstep():
    contract = build_compact_death_terminal_contract_v1()

    assert contract["compact_death_terminal_contract_verified"] is True
    assert contract["compact_death_terminal_contract_promotion_gate_satisfied"] is False
    assert contract["compact_death_terminal_contract_blocker"] == (
        "normal_collision_death_not_proven"
    )
    assert contract["death_mode"] == DEATH_MODE_PROFILE_NO_DEATH
    assert contract["profile_only_terminal_contract"] is True
    assert contract["normal_collision_death_supported"] is False
    assert contract["profile_no_death_supported"] is True
    assert contract["max_ticks_terminal_supported"] is True
    assert contract["terminated_supported"] is True
    assert contract["truncated_supported"] is False
    assert contract["truncated_count_required_zero"] is True
    assert contract["terminal_final_observation_required"] is True
    assert contract["terminal_autoreset_observation_forbidden"] is True
    assert contract["terminal_final_reward_map_required"] is True
    assert (
        contract["terminal_unroll_value_target_mode"]
        == COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
    )
    assert contract["terminal_unroll_bootstrap_after_done"] is False
    assert contract["terminal_validity_masks_required"] is True
    assert contract["resident_terminal_final_observation_required"] is True
    assert "normal_death=false" in compact_death_terminal_contract_evidence_ref(contract)
    assert "promotion=false" in compact_death_terminal_contract_evidence_ref(contract)


def test_compact_death_terminal_contract_rejects_normal_death_until_proven():
    with pytest.raises(
        CompactDeathTerminalContractError,
        match="normal_collision_evidence",
    ):
        build_compact_death_terminal_contract_v1(death_mode=DEATH_MODE_NORMAL)


def test_compact_death_terminal_contract_accepts_normal_collision_evidence():
    contract = build_compact_death_terminal_contract_v1(
        death_mode=DEATH_MODE_NORMAL,
        normal_collision_evidence=_normal_collision_evidence(),
    )

    assert contract["compact_death_terminal_contract_mode"] == (
        COMPACT_DEATH_TERMINAL_MODE_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP
    )
    assert contract["compact_death_terminal_contract_promotion_gate_satisfied"] is True
    assert contract["compact_death_terminal_contract_blocker"] == ""
    assert contract["death_terminal_contract_status"] == (
        "normal_collision_death_terminal_nstep_v1"
    )
    assert contract["death_mode"] == DEATH_MODE_NORMAL
    assert contract["profile_only_terminal_contract"] is False
    assert contract["normal_collision_death_supported"] is True
    assert contract["terminated_supported"] is True
    assert contract["truncated_supported"] is False
    assert contract["truncated_row_count"] == 0
    assert contract["death_row_count"] == 1
    assert contract["normal_collision_death_causes"] == ["opponent_trail"]
    assert contract["normal_collision_death_hit_owner_present"] is True
    evidence_ref = compact_death_terminal_contract_evidence_ref(contract)
    assert "normal_death=true" in evidence_ref
    assert "terminated=true" in evidence_ref
    assert "truncated=false" in evidence_ref
    assert "promotion_gate=true" in evidence_ref


def test_compact_death_terminal_contract_accepts_host_terminal_final_observation():
    evidence = _normal_collision_evidence()
    evidence["resident_terminal_final_observation_used"] = False
    evidence["host_terminal_final_observation_used"] = True
    evidence["terminal_final_observation_used"] = True
    evidence["device_replay_terminal_rows_verified"] = False
    evidence["host_terminal_rows_verified"] = True
    evidence["terminal_rows_verified"] = True

    contract = build_compact_death_terminal_contract_v1(
        death_mode=DEATH_MODE_NORMAL,
        normal_collision_evidence=evidence,
    )

    assert contract["compact_death_terminal_contract_promotion_gate_satisfied"] is True
    assert contract["normal_collision_death_supported"] is True


def test_normal_collision_death_evidence_can_be_derived_from_profile_payload():
    evidence = build_normal_collision_death_evidence_from_profile_result_v1(
        _normal_collision_profile_payload(),
        evidence_id="unit-profile-normal-death-evidence",
        evidence_refs=["unit-profile-result.json"],
    )
    contract = build_normal_collision_death_contract_from_profile_result_v1(
        _normal_collision_profile_payload(),
        evidence_id="unit-profile-normal-death-evidence",
        evidence_refs=["unit-profile-result.json"],
    )

    assert evidence["death_mode"] == DEATH_MODE_NORMAL
    assert evidence["normal_collision_death_causes"] == ["opponent_trail"]
    assert evidence["normal_collision_death_evidence_rows"][0]["death_cause"] == ["opponent_trail"]
    assert evidence["terminal_sample_row_count"] == 4
    assert evidence["compact_muzero_learner_done_count"] == 4
    assert "promotion_claim" not in evidence
    assert "promotion_eligible" not in evidence
    assert "compact_coach_compatibility_gate_death_terminal_contract" not in evidence
    assert contract["compact_death_terminal_contract_promotion_gate_satisfied"] is True
    assert contract["normal_collision_death_evidence"] == evidence


def test_normal_collision_death_evidence_accepts_mixed_terminal_target_mode():
    payload = _normal_collision_profile_payload()
    sample_gate = payload["compact_rollout_slab_sample_gate_last_telemetry"]
    sample_gate["compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode"] = (
        "mixed:search_root_value_no_terminal,"
        f"{COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP}"
    )

    evidence = build_normal_collision_death_evidence_from_profile_result_v1(
        payload,
        evidence_id="unit-profile-mixed-normal-death-evidence",
        evidence_refs=["unit-profile-mixed-result.json"],
    )
    contract = build_normal_collision_death_contract_from_profile_result_v1(
        payload,
        evidence_id="unit-profile-mixed-normal-death-evidence",
        evidence_refs=["unit-profile-mixed-result.json"],
    )

    assert evidence["terminal_unroll_value_target_mode"] == (
        COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
    )
    assert evidence["sample_terminal_unroll_value_target_mode"].startswith("mixed:")
    assert evidence["post_terminal_masks_zero"] is True
    assert contract["normal_collision_death_evidence"] == evidence


def test_normal_collision_death_evidence_accepts_owner_only_telemetry():
    payload = _normal_collision_profile_payload()
    payload["compact_owner_search_owner_sample_telemetry"] = {
        "terminal_unroll_windows_supported": True,
        "terminal_sample_row_count": 4,
        "next_final_observation_row_count": 4,
        "resident_terminal_final_observation_used": True,
        "device_replay_index_rows_sample": True,
        "terminal_unroll_value_target_row_count": 4,
        "terminal_unroll_value_target_mode": (
            COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
        ),
    }
    payload["compact_owner_search_owner_learner_telemetry"] = {
        "compact_muzero_learner_done_count": 4,
        "compact_muzero_learner_truncated_count": 0,
        "compact_muzero_learner_value_valid_count": 8,
    }
    payload["compact_rollout_slab_sample_gate_last_telemetry"] = {}
    payload["compact_rollout_slab_learner_gate_last_telemetry"] = {
        "compact_rollout_slab_learner_gate_compact_muzero_telemetry": {}
    }

    evidence = build_normal_collision_death_evidence_from_profile_result_v1(
        payload,
        evidence_id="unit-owner-only-normal-death-evidence",
        evidence_refs=["unit-owner-search-result.json"],
    )

    assert evidence["terminal_sample_row_count"] == 4
    assert evidence["compact_muzero_learner_done_count"] == 4
    assert evidence["terminal_unroll_value_target_mode"] == (
        COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
    )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda payload: payload.pop("normal_collision_death_evidence_rows"),
            "normal_collision_death_evidence_rows",
        ),
        (
            lambda payload: payload.pop("compact_rollout_slab_sample_gate_last_telemetry"),
            "sample_gate_last_telemetry",
        ),
        (
            lambda payload: payload["compact_rollout_slab_sample_gate_last_telemetry"].update(
                {
                    "compact_rollout_slab_sample_gate_device_replay_index_rows": False,
                }
            ),
            "terminal_rows_verified",
        ),
        (
            lambda payload: payload.update({"normal_collision_death_causes": []}),
            "collision death causes",
        ),
        (
            lambda payload: payload.update({"terminal_final_reward_map_verified": False}),
            "terminal_final_reward_map_verified",
        ),
    ],
)
def test_normal_collision_profile_evidence_derivation_fails_closed(mutator, match):
    payload = _normal_collision_profile_payload()
    mutator(payload)

    with pytest.raises(CompactDeathTerminalContractError, match=match):
        build_normal_collision_death_evidence_from_profile_result_v1(
            payload,
            evidence_id="unit-profile-normal-death-evidence",
            evidence_refs=["unit-profile-result.json"],
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("truncated_row_count", 1, "truncated counts"),
        (
            "terminal_final_observation_before_autoreset",
            False,
            "terminal_final_observation_before_autoreset",
        ),
        (
            "terminal_unroll_value_target_mode",
            "bootstrap_after_done",
            "value target mode",
        ),
        ("evidence_refs", [], "evidence_refs"),
        ("normal_collision_death_causes", ["round_timeout"], "opponent_trail or wall"),
    ],
)
def test_compact_death_terminal_contract_rejects_weak_normal_evidence(
    field,
    value,
    match,
):
    evidence = _normal_collision_evidence()
    evidence[field] = value

    with pytest.raises(CompactDeathTerminalContractError, match=match):
        build_compact_death_terminal_contract_v1(
            death_mode=DEATH_MODE_NORMAL,
            normal_collision_evidence=evidence,
        )


def test_compact_death_terminal_contract_rejects_profile_evidence_mixup():
    with pytest.raises(CompactDeathTerminalContractError, match="only valid"):
        build_compact_death_terminal_contract_v1(
            death_mode=DEATH_MODE_PROFILE_NO_DEATH,
            normal_collision_evidence=_normal_collision_evidence(),
        )


def test_compact_death_terminal_contract_validation_rejects_overclaims():
    contract = build_compact_death_terminal_contract_v1()
    normal_death = dict(contract)
    normal_death["normal_collision_death_supported"] = True
    with pytest.raises(CompactDeathTerminalContractError, match="normal_collision"):
        validate_compact_death_terminal_contract_v1(normal_death)

    promotion = dict(contract)
    promotion["compact_death_terminal_contract_promotion_gate_satisfied"] = True
    with pytest.raises(CompactDeathTerminalContractError, match="promotion gate"):
        validate_compact_death_terminal_contract_v1(promotion)

    truncation = dict(contract)
    truncation["truncated_supported"] = True
    with pytest.raises(CompactDeathTerminalContractError, match="truncated_supported"):
        validate_compact_death_terminal_contract_v1(truncation)

    stale_mode = dict(contract)
    stale_mode["terminal_unroll_value_target_mode"] = "bootstrap_after_done"
    with pytest.raises(CompactDeathTerminalContractError, match="value target mode"):
        validate_compact_death_terminal_contract_v1(stale_mode)

    normal_contract = build_compact_death_terminal_contract_v1(
        death_mode=DEATH_MODE_NORMAL,
        normal_collision_evidence=_normal_collision_evidence(),
    )
    profile_only_normal = dict(normal_contract)
    profile_only_normal["profile_only_terminal_contract"] = True
    with pytest.raises(CompactDeathTerminalContractError, match="profile-only"):
        validate_compact_death_terminal_contract_v1(profile_only_normal)


def _normal_collision_evidence():
    return {
        "schema_id": COMPACT_DEATH_TERMINAL_EVIDENCE_SCHEMA_ID,
        "evidence_id": "unit-normal-opponent-trail-terminal",
        "death_mode": DEATH_MODE_NORMAL,
        "trainer_config_death_mode": DEATH_MODE_NORMAL,
        "normal_death_terminal_contract_owner": "compact_owned_trainer_config",
        "terminal_row_count": 1,
        "terminated_row_count": 1,
        "truncated_row_count": 0,
        "death_row_count": 1,
        "death_count_total": 1,
        "normal_collision_death_causes": ["opponent_trail"],
        "normal_collision_death_hit_owner_present": True,
        "done_semantics_verified": True,
        "terminal_final_observation_before_autoreset": True,
        "terminal_autoreset_observation_forbidden": True,
        "terminal_final_reward_map_verified": True,
        "terminal_unroll_value_target_mode": (
            COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
        ),
        "terminal_unroll_bootstrap_after_done": False,
        "terminal_validity_masks_verified": True,
        "post_terminal_masks_zero": True,
        "resident_terminal_final_observation_used": True,
        "device_replay_terminal_rows_verified": True,
        "terminal_sample_row_count": 1,
        "next_final_observation_row_count": 1,
        "terminal_unroll_value_target_row_count": 1,
        "compact_muzero_learner_done_count": 1,
        "compact_muzero_learner_truncated_count": 0,
        "evidence_refs": [
            "tests/test_source_state_hybrid_observation_profile.py::"
            "test_hybrid_compact_native_path_preserves_normal_collision_death_fixture",
            "tests/test_compact_death_terminal_contract.py::"
            "test_compact_death_terminal_contract_accepts_normal_collision_evidence",
        ],
    }


def _normal_collision_profile_payload():
    return {
        "death_mode": DEATH_MODE_NORMAL,
        "terminal_row_count": 4,
        "done_semantics_verified": True,
        "terminated_row_count": 4,
        "truncated_row_count": 0,
        "death_row_count": 4,
        "death_count_total": 4,
        "death_cause_count_by_name": {
            "wall": 0,
            "own_trail": 0,
            "opponent_trail": 4,
            "body_unknown": 0,
        },
        "normal_collision_death_causes": ["opponent_trail"],
        "normal_collision_death_hit_owner_present": True,
        "normal_collision_death_evidence_rows": [
            {
                "global_row": 3,
                "done": True,
                "terminated": True,
                "truncated": False,
                "terminal_reason": 1,
                "death_count": 1,
                "death_player": [0, -1],
                "death_cause": ["opponent_trail"],
                "death_hit_owner": [1, -1],
                "winner": 1,
                "draw": False,
                "reward": [-1.0, 1.0],
                "final_reward_map": [-1.0, 1.0],
                "final_reward_map_matches_reward": True,
                "final_observation_row": True,
            }
        ],
        "terminal_final_observation_row_count": 4,
        "terminal_final_observation_before_autoreset_verified": True,
        "terminal_final_reward_map_row_count": 4,
        "terminal_final_reward_map_matches_reward_row_count": 4,
        "terminal_final_reward_map_verified": True,
        "compact_rollout_slab_sample_gate_last_telemetry": {
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            "compact_rollout_slab_sample_gate_terminal_sample_row_count": 4,
            "compact_rollout_slab_sample_gate_next_final_observation_row_count": 4,
            "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used": True,
            "compact_rollout_slab_sample_gate_device_replay_index_rows": True,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count": 4,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode": (
                COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
            ),
        },
        "compact_rollout_slab_learner_gate_last_telemetry": {
            "compact_rollout_slab_learner_gate_compact_muzero_telemetry": {
                "compact_muzero_learner_done_count": 4,
                "compact_muzero_learner_truncated_count": 0,
                "compact_muzero_learner_value_valid_count": 8,
            },
        },
    }
