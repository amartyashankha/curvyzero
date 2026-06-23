"""Death/terminal attestation for compact-owned trainer candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from curvyzero.env.vector_runtime import DEATH_MODE_NORMAL
from curvyzero.env.vector_runtime import DEATH_MODE_PROFILE_NO_DEATH


COMPACT_DEATH_TERMINAL_CONTRACT_SCHEMA_ID = "curvyzero_compact_death_terminal_contract/v1"
COMPACT_DEATH_TERMINAL_MODE_PROFILE_NO_DEATH_TERMINAL_NSTEP = "profile_no_death_terminal_nstep_v1"
COMPACT_DEATH_TERMINAL_MODE_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP = (
    "normal_collision_death_terminal_nstep_v1"
)
COMPACT_DEATH_TERMINAL_STATUS_PROFILE_NO_DEATH_TERMINAL_NSTEP_ONLY = (
    "profile_no_death_terminal_nstep_only"
)
COMPACT_DEATH_TERMINAL_STATUS_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP = (
    COMPACT_DEATH_TERMINAL_MODE_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP
)
COMPACT_DEATH_TERMINAL_EVIDENCE_SCHEMA_ID = "curvyzero_compact_death_terminal_contract_evidence/v1"
COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP = (
    "stock_terminal_no_bootstrap_return_discount_1.0"
)

_NORMAL_COLLISION_CAUSES = frozenset(("opponent_trail", "wall"))


def _terminal_target_mode_contains_stock_no_bootstrap(value: Any) -> bool:
    mode = str(value)
    if mode == COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP:
        return True
    if not mode.startswith("mixed:"):
        return False
    parts = {part.strip() for part in mode.removeprefix("mixed:").split(",")}
    return COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP in parts


class CompactDeathTerminalContractError(ValueError):
    """Raised when a compact death/terminal contract overclaims support."""


def build_compact_death_terminal_contract_v1(
    *,
    death_mode: str = DEATH_MODE_PROFILE_NO_DEATH,
    normal_collision_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the current compact-owned death/terminal contract.

    The default compact-owned terminal contract remains scoped to the
    profile/no-death lane plus max-tick terminal rows.  Normal collision-death
    semantics are available only when the caller supplies structured evidence;
    truncation semantics remain unsupported.
    """

    death_mode_value = str(death_mode)
    if death_mode_value == DEATH_MODE_PROFILE_NO_DEATH:
        if normal_collision_evidence is not None:
            raise CompactDeathTerminalContractError(
                "normal_collision_evidence is only valid for death_mode='normal'"
            )
        contract = _profile_no_death_contract(death_mode_value)
    elif death_mode_value == DEATH_MODE_NORMAL:
        evidence = _validate_normal_collision_evidence(normal_collision_evidence)
        contract = _normal_collision_death_contract(
            death_mode=death_mode_value,
            evidence=evidence,
        )
    else:
        raise CompactDeathTerminalContractError(
            "compact death/terminal contract requires death_mode "
            f"{DEATH_MODE_PROFILE_NO_DEATH!r} or {DEATH_MODE_NORMAL!r}; "
            f"got {death_mode_value!r}"
        )
    validate_compact_death_terminal_contract_v1(contract)
    return contract


def build_normal_collision_death_evidence_from_profile_result_v1(
    profile_result: Mapping[str, Any],
    *,
    evidence_id: str,
    evidence_refs: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Derive normal-death evidence from a compact profile payload.

    This helper intentionally consumes measured payload fields and compact
    sample/learner telemetry.  It does not let callers pass free-form booleans
    for the proof-critical rows.
    """

    payload = _plain_evidence_value(profile_result)
    if not isinstance(payload, Mapping):
        raise CompactDeathTerminalContractError("profile_result must be a mapping")
    sample_gate_value = payload.get("compact_rollout_slab_sample_gate_last_telemetry")
    owner_sample_gate_value = payload.get("compact_owner_search_owner_sample_telemetry")
    sample_gate = (
        sample_gate_value if isinstance(sample_gate_value, Mapping) else {}
    )
    owner_sample_gate = (
        owner_sample_gate_value if isinstance(owner_sample_gate_value, Mapping) else {}
    )
    if owner_sample_gate and not _sample_gate_has_terminal_sample(sample_gate):
        sample_gate = owner_sample_gate
    if not sample_gate:
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence requires "
            "compact_rollout_slab_sample_gate_last_telemetry or "
            "compact_owner_search_owner_sample_telemetry"
        )
    learner_gate_value = payload.get("compact_rollout_slab_learner_gate_last_telemetry")
    muzero_telemetry_value = (
        learner_gate_value.get("compact_rollout_slab_learner_gate_compact_muzero_telemetry")
        if isinstance(learner_gate_value, Mapping)
        else None
    )
    if isinstance(muzero_telemetry_value, Mapping) and muzero_telemetry_value:
        muzero_telemetry = muzero_telemetry_value
    else:
        owner_muzero_telemetry_value = payload.get("compact_owner_search_owner_learner_telemetry")
        if isinstance(owner_muzero_telemetry_value, Mapping):
            muzero_telemetry = owner_muzero_telemetry_value
        else:
            raise CompactDeathTerminalContractError(
                "normal_collision_evidence requires "
                "compact_rollout_slab_learner_gate_compact_muzero_telemetry or "
                "compact_owner_search_owner_learner_telemetry"
            )
    causes = payload.get("normal_collision_death_causes")
    row_evidence = payload.get("normal_collision_death_evidence_rows")
    if not isinstance(row_evidence, list) or not row_evidence:
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence requires normal_collision_death_evidence_rows"
        )
    row_causes = sorted(
        {
            str(cause)
            for row in row_evidence
            if isinstance(row, Mapping)
            for cause in (
                row.get("death_cause") if isinstance(row.get("death_cause"), list) else []
            )
        }
    )
    if isinstance(causes, list):
        causes = [str(cause) for cause in causes]
    else:
        cause_counts = payload.get("death_cause_count_by_name")
        if isinstance(cause_counts, Mapping):
            causes = [
                cause
                for cause in ("opponent_trail", "wall")
                if int(cause_counts.get(cause, 0) or 0) > 0
            ]
        else:
            causes = [cause for cause in ("opponent_trail", "wall") if cause in row_causes]
    row_hit_owner_present = any(
        isinstance(row, Mapping)
        and any(int(owner) >= 0 for owner in row.get("death_hit_owner", []))
        for row in row_evidence
    )
    row_final_observation_present = any(
        isinstance(row, Mapping) and row.get("final_observation_row") is True
        for row in row_evidence
    )
    row_final_reward_verified = all(
        isinstance(row, Mapping) and row.get("final_reward_map_matches_reward") is True
        for row in row_evidence
    )
    target_mode = str(
        sample_gate.get(
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode",
            sample_gate.get("terminal_unroll_value_target_mode", ""),
        )
    )
    terminal_windows_supported = (
        sample_gate.get("compact_rollout_slab_sample_gate_terminal_unroll_windows_supported")
        is True
        or sample_gate.get("terminal_unroll_windows_supported") is True
    )
    next_final_observation_row_count = int(
        sample_gate.get(
            "compact_rollout_slab_sample_gate_next_final_observation_row_count",
            sample_gate.get("next_final_observation_row_count", 0),
        )
        or 0
    )
    terminal_sample_row_count = int(
        sample_gate.get(
            "compact_rollout_slab_sample_gate_terminal_sample_row_count",
            sample_gate.get("terminal_sample_row_count", 0),
        )
        or 0
    )
    terminal_unroll_value_target_row_count = int(
        sample_gate.get(
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count",
            sample_gate.get("terminal_unroll_value_target_row_count", 0),
        )
        or 0
    )
    terminal_target_mode_proves_no_bootstrap = bool(
        terminal_unroll_value_target_row_count > 0
        and _terminal_target_mode_contains_stock_no_bootstrap(target_mode)
    )
    evidence_target_mode = (
        COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
        if terminal_target_mode_proves_no_bootstrap
        else target_mode
    )
    resident_terminal_final_observation_used = sample_gate.get(
        "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used",
        sample_gate.get("resident_terminal_final_observation_used"),
    )
    host_terminal_final_observation_used = sample_gate.get(
        "compact_rollout_slab_sample_gate_host_terminal_final_observation_used",
        sample_gate.get("host_terminal_final_observation_used"),
    )
    device_replay_index_rows = sample_gate.get(
        "compact_rollout_slab_sample_gate_device_replay_index_rows",
        sample_gate.get("device_replay_index_rows_sample"),
    )
    evidence = {
        "schema_id": COMPACT_DEATH_TERMINAL_EVIDENCE_SCHEMA_ID,
        "evidence_id": evidence_id,
        "death_mode": payload.get("death_mode"),
        "trainer_config_death_mode": (
            payload.get("compact_owned_trainer_config_death_mode")
            or payload.get("trainer_config_death_mode")
            or payload.get("death_mode")
        ),
        "normal_death_terminal_contract_owner": (
            payload.get("normal_death_terminal_contract_owner") or "compact_owned_trainer_config"
        ),
        "terminal_row_count": payload.get("terminal_row_count"),
        "terminated_row_count": payload.get("terminated_row_count"),
        "truncated_row_count": payload.get("truncated_row_count"),
        "death_row_count": payload.get("death_row_count"),
        "death_count_total": payload.get("death_count_total"),
        "normal_collision_death_causes": causes,
        "normal_collision_death_hit_owner_present": bool(
            payload.get("normal_collision_death_hit_owner_present") or row_hit_owner_present
        ),
        "normal_collision_death_evidence_rows": row_evidence,
        "done_semantics_verified": payload.get("done_semantics_verified"),
        "terminal_final_observation_before_autoreset": (
            payload.get("terminal_final_observation_before_autoreset_verified") is True
            and row_final_observation_present
            and next_final_observation_row_count > 0
        ),
        "terminal_autoreset_observation_forbidden": True,
        "terminal_final_reward_map_verified": (
            payload.get("terminal_final_reward_map_verified") is True and row_final_reward_verified
        ),
        "terminal_unroll_value_target_mode": evidence_target_mode,
        "sample_terminal_unroll_value_target_mode": target_mode,
        "terminal_unroll_bootstrap_after_done": False,
        "terminal_validity_masks_verified": bool(
            terminal_windows_supported
            and int(muzero_telemetry.get("compact_muzero_learner_value_valid_count", 0) or 0) > 0
        ),
        "post_terminal_masks_zero": bool(
            terminal_windows_supported
            and int(muzero_telemetry.get("compact_muzero_learner_done_count", 0) or 0) > 0
            and terminal_target_mode_proves_no_bootstrap
        ),
        "resident_terminal_final_observation_used": resident_terminal_final_observation_used,
        "host_terminal_final_observation_used": host_terminal_final_observation_used,
        "terminal_final_observation_used": bool(
            resident_terminal_final_observation_used or host_terminal_final_observation_used
        ),
        "device_replay_terminal_rows_verified": device_replay_index_rows,
        "host_terminal_rows_verified": bool(
            host_terminal_final_observation_used
            and terminal_sample_row_count > 0
            and next_final_observation_row_count > 0
        ),
        "terminal_rows_verified": bool(
            device_replay_index_rows
            or (
                host_terminal_final_observation_used
                and terminal_sample_row_count > 0
                and next_final_observation_row_count > 0
            )
        ),
        "terminal_sample_row_count": terminal_sample_row_count,
        "next_final_observation_row_count": next_final_observation_row_count,
        "terminal_unroll_value_target_row_count": terminal_unroll_value_target_row_count,
        "compact_muzero_learner_done_count": muzero_telemetry.get(
            "compact_muzero_learner_done_count"
        ),
        "compact_muzero_learner_truncated_count": muzero_telemetry.get(
            "compact_muzero_learner_truncated_count",
            0,
        ),
        "evidence_refs": list(evidence_refs),
    }
    return _validate_normal_collision_evidence(evidence)


def build_normal_collision_death_contract_from_profile_result_v1(
    profile_result: Mapping[str, Any],
    *,
    evidence_id: str,
    evidence_refs: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Build the guarded normal-death contract from a compact profile payload."""

    evidence = build_normal_collision_death_evidence_from_profile_result_v1(
        profile_result,
        evidence_id=evidence_id,
        evidence_refs=evidence_refs,
    )
    return build_compact_death_terminal_contract_v1(
        death_mode=DEATH_MODE_NORMAL,
        normal_collision_evidence=evidence,
    )


def _base_contract(*, death_mode: str, mode: str, promotion_gate: bool) -> dict[str, Any]:
    return {
        "schema_id": COMPACT_DEATH_TERMINAL_CONTRACT_SCHEMA_ID,
        "compact_death_terminal_contract_schema_id": (COMPACT_DEATH_TERMINAL_CONTRACT_SCHEMA_ID),
        "compact_death_terminal_contract_mode": mode,
        "compact_death_terminal_contract_verified": True,
        "compact_death_terminal_contract_promotion_gate_satisfied": promotion_gate,
        "death_mode": death_mode,
        "profile_no_death_supported": True,
        "max_ticks_terminal_supported": True,
        "terminated_supported": True,
        "truncated_supported": False,
        "truncated_count_required_zero": True,
        "done_semantics": "done == terminated | truncated",
        "terminal_final_observation_required": True,
        "terminal_final_observation_source": (
            "next_final_observation_row selects final_observation before autoreset"
        ),
        "terminal_autoreset_observation_forbidden": True,
        "terminal_final_reward_map_required": True,
        "terminal_final_reward_source": (
            "final_reward uses next_final_reward_map for terminal rows and next_reward otherwise"
        ),
        "terminal_unroll_value_target_mode": (
            COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
        ),
        "terminal_unroll_discount": 1.0,
        "terminal_unroll_bootstrap_after_done": False,
        "terminal_validity_masks_required": True,
        "post_terminal_action_reward_masks_zero": True,
        "post_terminal_policy_value_masks_zero": True,
        "resident_terminal_final_observation_required": True,
        "device_replay_terminal_rows_supported": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
    }


def _profile_no_death_contract(death_mode: str) -> dict[str, Any]:
    contract = {
        **_base_contract(
            death_mode=death_mode,
            mode=COMPACT_DEATH_TERMINAL_MODE_PROFILE_NO_DEATH_TERMINAL_NSTEP,
            promotion_gate=False,
        ),
        "compact_death_terminal_contract_blocker": "normal_collision_death_not_proven",
        "death_terminal_contract_status": (
            COMPACT_DEATH_TERMINAL_STATUS_PROFILE_NO_DEATH_TERMINAL_NSTEP_ONLY
        ),
        "profile_only_terminal_contract": True,
        "normal_collision_death_supported": False,
        "non_claims": [
            "not_normal_collision_death_fidelity",
            "not_truncation_bootstrap_support",
            "not_death_terminal_promotion_gate_satisfied",
            "not_live_run_safety_claim",
            "not_a_promotion_claim",
        ],
    }
    return contract


def _normal_collision_death_contract(
    *,
    death_mode: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_base_contract(
            death_mode=death_mode,
            mode=COMPACT_DEATH_TERMINAL_MODE_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP,
            promotion_gate=True,
        ),
        "compact_death_terminal_contract_blocker": "",
        "death_terminal_contract_status": (
            COMPACT_DEATH_TERMINAL_STATUS_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP
        ),
        "profile_only_terminal_contract": False,
        "normal_collision_death_supported": True,
        "normal_collision_death_evidence": evidence,
        "trainer_config_death_mode": evidence["trainer_config_death_mode"],
        "normal_death_terminal_contract_owner": evidence["normal_death_terminal_contract_owner"],
        "compact_death_terminal_contract_evidence_schema_id": (
            COMPACT_DEATH_TERMINAL_EVIDENCE_SCHEMA_ID
        ),
        "normal_collision_death_evidence_id": evidence["evidence_id"],
        "normal_collision_death_evidence_refs": list(evidence["evidence_refs"]),
        "terminal_row_count": evidence["terminal_row_count"],
        "terminated_row_count": evidence["terminated_row_count"],
        "truncated_row_count": evidence["truncated_row_count"],
        "death_row_count": evidence["death_row_count"],
        "death_count_total": evidence["death_count_total"],
        "normal_collision_death_causes": list(evidence["normal_collision_death_causes"]),
        "normal_collision_death_hit_owner_present": evidence[
            "normal_collision_death_hit_owner_present"
        ],
        "non_claims": [
            "not_truncation_bootstrap_support",
            "not_live_run_safety_claim",
            "not_a_promotion_claim",
        ],
    }


def validate_compact_death_terminal_contract_v1(contract: Any) -> None:
    """Validate the compact-owned death/terminal contract."""

    if not isinstance(contract, Mapping):
        raise CompactDeathTerminalContractError("compact death/terminal contract must be a mapping")
    schema_id = contract.get("schema_id") or contract.get(
        "compact_death_terminal_contract_schema_id"
    )
    if schema_id != COMPACT_DEATH_TERMINAL_CONTRACT_SCHEMA_ID:
        raise CompactDeathTerminalContractError("compact death/terminal contract schema mismatch")
    mode = contract.get("compact_death_terminal_contract_mode")
    if mode not in (
        COMPACT_DEATH_TERMINAL_MODE_PROFILE_NO_DEATH_TERMINAL_NSTEP,
        COMPACT_DEATH_TERMINAL_MODE_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP,
    ):
        raise CompactDeathTerminalContractError("unsupported compact death/terminal contract mode")
    if contract.get("compact_death_terminal_contract_verified") is not True:
        raise CompactDeathTerminalContractError("compact death/terminal contract is not verified")
    if mode == COMPACT_DEATH_TERMINAL_MODE_PROFILE_NO_DEATH_TERMINAL_NSTEP:
        _validate_profile_no_death_contract(contract)
    else:
        _validate_normal_collision_contract(contract)
    _validate_common_terminal_contract(contract)


def _validate_profile_no_death_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("compact_death_terminal_contract_promotion_gate_satisfied") is not False:
        raise CompactDeathTerminalContractError(
            "profile_no_death terminal contract must not satisfy the promotion gate"
        )
    if (
        contract.get("compact_death_terminal_contract_blocker")
        != "normal_collision_death_not_proven"
    ):
        raise CompactDeathTerminalContractError("compact death/terminal blocker mismatch")
    if contract.get("death_mode") != DEATH_MODE_PROFILE_NO_DEATH:
        raise CompactDeathTerminalContractError(
            "compact death/terminal contract currently requires profile_no_death"
        )
    if contract.get("death_terminal_contract_status") != (
        COMPACT_DEATH_TERMINAL_STATUS_PROFILE_NO_DEATH_TERMINAL_NSTEP_ONLY
    ):
        raise CompactDeathTerminalContractError(
            "compact death/terminal profile_no_death status mismatch"
        )
    if contract.get("profile_only_terminal_contract") is not True:
        raise CompactDeathTerminalContractError(
            "compact death/terminal contract must be profile-only"
        )
    for field in ("normal_collision_death_supported",):
        if contract.get(field) is not False:
            raise CompactDeathTerminalContractError(
                f"compact death/terminal contract overclaims {field}"
            )


def _validate_normal_collision_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("compact_death_terminal_contract_promotion_gate_satisfied") is not True:
        raise CompactDeathTerminalContractError(
            "normal collision death contract must satisfy the promotion gate"
        )
    if str(contract.get("compact_death_terminal_contract_blocker") or ""):
        raise CompactDeathTerminalContractError(
            "normal collision death contract must not carry a blocker"
        )
    if contract.get("death_mode") != DEATH_MODE_NORMAL:
        raise CompactDeathTerminalContractError(
            "normal collision death contract requires death_mode='normal'"
        )
    if contract.get("death_terminal_contract_status") != (
        COMPACT_DEATH_TERMINAL_STATUS_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP
    ):
        raise CompactDeathTerminalContractError("normal collision death status mismatch")
    if contract.get("profile_only_terminal_contract") is not False:
        raise CompactDeathTerminalContractError(
            "normal collision death contract must not be profile-only"
        )
    if contract.get("normal_collision_death_supported") is not True:
        raise CompactDeathTerminalContractError("normal collision death support is missing")
    evidence = _validate_normal_collision_evidence(contract.get("normal_collision_death_evidence"))
    contract_trainer_config_death_mode = (
        contract.get("trainer_config_death_mode")
        if "trainer_config_death_mode" in contract
        else evidence["trainer_config_death_mode"]
    )
    if contract_trainer_config_death_mode != evidence["trainer_config_death_mode"]:
        raise CompactDeathTerminalContractError(
            "normal collision trainer config death mode mismatch"
        )
    contract_owner = (
        contract.get("normal_death_terminal_contract_owner")
        if "normal_death_terminal_contract_owner" in contract
        else evidence["normal_death_terminal_contract_owner"]
    )
    if contract_owner != evidence["normal_death_terminal_contract_owner"]:
        raise CompactDeathTerminalContractError("normal collision contract owner mismatch")
    if contract.get("compact_death_terminal_contract_evidence_schema_id") != (
        COMPACT_DEATH_TERMINAL_EVIDENCE_SCHEMA_ID
    ):
        raise CompactDeathTerminalContractError("normal collision evidence schema mismatch")
    if contract.get("normal_collision_death_evidence_id") != evidence["evidence_id"]:
        raise CompactDeathTerminalContractError("normal collision evidence id mismatch")
    for key in (
        "terminal_row_count",
        "terminated_row_count",
        "truncated_row_count",
        "death_row_count",
        "death_count_total",
    ):
        if contract.get(key) != evidence[key]:
            raise CompactDeathTerminalContractError(f"normal collision evidence {key} mismatch")
    if list(contract.get("normal_collision_death_evidence_refs") or []) != list(
        evidence["evidence_refs"]
    ):
        raise CompactDeathTerminalContractError("normal collision evidence refs mismatch")
    if list(contract.get("normal_collision_death_causes") or []) != list(
        evidence["normal_collision_death_causes"]
    ):
        raise CompactDeathTerminalContractError("normal collision evidence causes mismatch")
    if (
        contract.get("normal_collision_death_hit_owner_present")
        != evidence["normal_collision_death_hit_owner_present"]
    ):
        raise CompactDeathTerminalContractError(
            "normal collision death hit-owner evidence mismatch"
        )


def _validate_common_terminal_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("done_semantics") != "done == terminated | truncated":
        raise CompactDeathTerminalContractError("compact death/terminal done semantics mismatch")
    if (
        contract.get("terminal_unroll_value_target_mode")
        != COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
    ):
        raise CompactDeathTerminalContractError("compact death/terminal value target mode mismatch")
    for field in (
        "profile_no_death_supported",
        "max_ticks_terminal_supported",
        "terminated_supported",
        "truncated_count_required_zero",
        "terminal_final_observation_required",
        "terminal_autoreset_observation_forbidden",
        "terminal_final_reward_map_required",
        "terminal_validity_masks_required",
        "post_terminal_action_reward_masks_zero",
        "post_terminal_policy_value_masks_zero",
        "resident_terminal_final_observation_required",
        "device_replay_terminal_rows_supported",
    ):
        if contract.get(field) is not True:
            raise CompactDeathTerminalContractError(
                f"compact death/terminal contract missing {field}"
            )
    for field in (
        "truncated_supported",
        "terminal_unroll_bootstrap_after_done",
        "calls_train_muzero",
        "touches_live_runs",
        "promotion_claim",
    ):
        if contract.get(field) is not False:
            raise CompactDeathTerminalContractError(
                f"compact death/terminal contract overclaims {field}"
            )
    if contract.get("terminal_unroll_discount") != 1.0:
        raise CompactDeathTerminalContractError("compact death/terminal discount must be 1.0")


def _validate_normal_collision_evidence(
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence is required for death_mode='normal'"
        )
    value = {str(key): _plain_evidence_value(item) for key, item in dict(evidence).items()}
    if value.get("schema_id") != COMPACT_DEATH_TERMINAL_EVIDENCE_SCHEMA_ID:
        raise CompactDeathTerminalContractError("normal_collision_evidence schema mismatch")
    if value.get("death_mode") != DEATH_MODE_NORMAL:
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence requires death_mode='normal'"
        )
    trainer_config_death_mode = value.get("trainer_config_death_mode", DEATH_MODE_NORMAL)
    if trainer_config_death_mode != DEATH_MODE_NORMAL:
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence requires trainer_config_death_mode='normal'"
        )
    contract_owner = value.get(
        "normal_death_terminal_contract_owner",
        "compact_owned_trainer_config",
    )
    if contract_owner != "compact_owned_trainer_config":
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence requires compact-owned trainer contract owner"
        )
    evidence_id = str(value.get("evidence_id") or "").strip()
    if not evidence_id:
        raise CompactDeathTerminalContractError("normal_collision_evidence requires evidence_id")
    refs = value.get("evidence_refs")
    if not isinstance(refs, list) or not refs or not all(str(ref).strip() for ref in refs):
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence requires non-empty evidence_refs"
        )
    counts = {
        key: _positive_int(value.get(key), key)
        for key in (
            "terminal_row_count",
            "terminated_row_count",
            "death_row_count",
            "death_count_total",
            "terminal_sample_row_count",
            "next_final_observation_row_count",
            "terminal_unroll_value_target_row_count",
            "compact_muzero_learner_done_count",
        )
    }
    truncated_count = _non_negative_int(
        value.get("truncated_row_count"),
        "truncated_row_count",
    )
    learner_truncated_count = _non_negative_int(
        value.get("compact_muzero_learner_truncated_count"),
        "compact_muzero_learner_truncated_count",
    )
    if truncated_count != 0 or learner_truncated_count != 0:
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence requires truncated counts to be zero"
        )
    causes = value.get("normal_collision_death_causes")
    if not isinstance(causes, list) or not causes:
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence requires collision death causes"
        )
    cause_values = [str(cause) for cause in causes]
    if not any(cause in _NORMAL_COLLISION_CAUSES for cause in cause_values):
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence requires opponent_trail or wall death cause"
        )
    for field in (
        "normal_collision_death_hit_owner_present",
        "done_semantics_verified",
        "terminal_final_observation_before_autoreset",
        "terminal_autoreset_observation_forbidden",
        "terminal_final_reward_map_verified",
        "terminal_validity_masks_verified",
        "post_terminal_masks_zero",
    ):
        if value.get(field) is not True:
            raise CompactDeathTerminalContractError(f"normal_collision_evidence missing {field}")
    if not bool(
        value.get("terminal_rows_verified")
        or value.get("device_replay_terminal_rows_verified")
        or value.get("host_terminal_rows_verified")
    ):
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence missing terminal_rows_verified"
        )
    if not bool(
        value.get("terminal_final_observation_used")
        or value.get("resident_terminal_final_observation_used")
        or value.get("host_terminal_final_observation_used")
    ):
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence missing terminal_final_observation_used"
        )
    if (
        value.get("terminal_unroll_value_target_mode")
        != COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
    ):
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence value target mode mismatch"
        )
    if value.get("terminal_unroll_bootstrap_after_done") is not False:
        raise CompactDeathTerminalContractError(
            "normal_collision_evidence must prove no bootstrap after done"
        )
    value.update(counts)
    value["trainer_config_death_mode"] = trainer_config_death_mode
    value["normal_death_terminal_contract_owner"] = contract_owner
    value["truncated_row_count"] = truncated_count
    value["compact_muzero_learner_truncated_count"] = learner_truncated_count
    value["evidence_id"] = evidence_id
    value["evidence_refs"] = [str(ref) for ref in refs]
    value["normal_collision_death_causes"] = cause_values
    return value


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactDeathTerminalContractError(f"normal_collision_evidence requires {field}")
    return value


def _sample_gate_has_terminal_sample(value: Mapping[str, Any]) -> bool:
    for key in (
        "compact_rollout_slab_sample_gate_terminal_sample_row_count",
        "terminal_sample_row_count",
    ):
        try:
            if int(value.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            return False
    return False


def _positive_int(value: Any, field: str) -> int:
    result = _non_negative_int(value, field)
    if result <= 0:
        raise CompactDeathTerminalContractError(
            f"normal_collision_evidence requires positive {field}"
        )
    return result


def _non_negative_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CompactDeathTerminalContractError(
            f"normal_collision_evidence requires integer {field}"
        ) from exc
    if result < 0:
        raise CompactDeathTerminalContractError(
            f"normal_collision_evidence requires non-negative {field}"
        )
    return result


def _plain_evidence_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_evidence_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_evidence_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_evidence_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compact_death_terminal_contract_evidence_ref(contract: Mapping[str, Any]) -> str:
    """Return a compact evidence token for compatibility reports."""

    validate_compact_death_terminal_contract_v1(contract)
    mode = str(contract["compact_death_terminal_contract_mode"])
    if mode == COMPACT_DEATH_TERMINAL_MODE_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP:
        return ":".join(
            (
                COMPACT_DEATH_TERMINAL_CONTRACT_SCHEMA_ID,
                str(contract["death_mode"]),
                mode,
                str(contract["terminal_unroll_value_target_mode"]),
                str(contract["normal_collision_death_evidence_id"]),
                "normal_death=true",
                "terminated=true",
                "truncated=false",
                "promotion_gate=true",
            )
        )
    return ":".join(
        (
            COMPACT_DEATH_TERMINAL_CONTRACT_SCHEMA_ID,
            str(contract["death_mode"]),
            mode,
            str(contract["terminal_unroll_value_target_mode"]),
            "normal_death=false",
            "truncated=false",
            "promotion=false",
        )
    )


__all__ = [
    "COMPACT_DEATH_TERMINAL_EVIDENCE_SCHEMA_ID",
    "COMPACT_DEATH_TERMINAL_CONTRACT_SCHEMA_ID",
    "COMPACT_DEATH_TERMINAL_MODE_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP",
    "COMPACT_DEATH_TERMINAL_MODE_PROFILE_NO_DEATH_TERMINAL_NSTEP",
    "COMPACT_DEATH_TERMINAL_STATUS_NORMAL_COLLISION_DEATH_TERMINAL_NSTEP",
    "COMPACT_DEATH_TERMINAL_STATUS_PROFILE_NO_DEATH_TERMINAL_NSTEP_ONLY",
    "COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP",
    "CompactDeathTerminalContractError",
    "build_compact_death_terminal_contract_v1",
    "build_normal_collision_death_contract_from_profile_result_v1",
    "build_normal_collision_death_evidence_from_profile_result_v1",
    "compact_death_terminal_contract_evidence_ref",
    "validate_compact_death_terminal_contract_v1",
]
