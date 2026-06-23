import json

import pytest

from curvyzero.training.compact_coach_speed_row import (
    build_compact_coach_speed_row_evidence_v1,
)
from curvyzero.training.compact_coach_speed_row import (
    compact_coach_speed_row_evidence_ref,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_GATE_COACH_SPEED_ROW,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_LIFECYCLE_GATES,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_REQUIRED_PROMOTION_GATES,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_REF_PREFIX,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_SPEED_ROW_EVIDENCE_REF_PREFIX,
)
from curvyzero.training.compact_coach_compatibility import (
    CompactCoachCompatibilityError,
)
from curvyzero.training.compact_coach_compatibility import (
    CompactCoachCompatibilityReportV1,
)
from curvyzero.training.compact_coach_compatibility import (
    build_compact_coach_compatibility_report_v1,
)
from curvyzero.training.compact_coach_compatibility import (
    build_profile_only_compact_coach_report_v1,
)


def test_profile_only_compact_coach_report_names_missing_promotion_gates():
    report = build_profile_only_compact_coach_report_v1(
        speed_currency="compact_profile_active_roots_per_sec",
    )
    metadata = report.as_metadata()

    assert report.promotion_eligible is False
    assert report.promotion_blocker == "profile_only_candidate"
    assert "checkpoint_save_load" in report.missing_required_gates
    assert "eval_gif_tournament_load" in report.missing_required_gates
    assert "policy_refresh_handoff" in report.missing_required_gates
    assert metadata["compact_coach_compatibility_promotion_eligible"] is False
    assert (
        metadata["compact_coach_compatibility_speed_currency"]
        == "compact_profile_active_roots_per_sec"
    )
    assert metadata["compact_coach_compatibility_gate_matched_denominator"] is True
    assert metadata["compact_coach_compatibility_gate_public_stock_sample_diff"] is True
    assert metadata["compact_coach_compatibility_gate_trainer_entrypoint"] is False


def test_profile_only_report_rejects_promotion_claim():
    with pytest.raises(CompactCoachCompatibilityError, match="promotion_claim"):
        build_profile_only_compact_coach_report_v1(
            speed_currency="compact_profile_active_roots_per_sec",
            promotion_claim=True,
        )


def test_compact_owned_report_can_be_locally_eligible_without_promotion_claim(tmp_path):
    gates, evidence, speed_row = _complete_compact_owned_compatibility_inputs(tmp_path)
    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="compact_trainer_env_steps_per_sec",
        gates=gates,
        evidence=evidence,
        coach_speed_row_evidence=speed_row,
        promotion_claim=False,
    )

    assert report.promotion_eligible is True
    assert report.promotion_blocker == ""
    assert report.missing_required_gates == ()
    assert report.promotion_claim is False


def test_compact_owned_report_rejects_promotion_claim_without_readiness_contract(
    tmp_path,
):
    gates, evidence, speed_row = _complete_compact_owned_compatibility_inputs(tmp_path)

    with pytest.raises(CompactCoachCompatibilityError, match="post-compatibility"):
        build_compact_coach_compatibility_report_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=False,
            touches_live_runs=False,
            speed_currency="compact_trainer_env_steps_per_sec",
            gates=gates,
            evidence=evidence,
            coach_speed_row_evidence=speed_row,
            promotion_claim=True,
        )


def test_direct_report_construction_cannot_bypass_promotion_readiness_blocker():
    with pytest.raises(CompactCoachCompatibilityError, match="post-compatibility"):
        CompactCoachCompatibilityReportV1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=False,
            touches_live_runs=False,
            speed_currency="compact_trainer_env_steps_per_sec",
            gates={gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES},
            evidence={
                gate: f"unit_evidence_for_{gate}"
                for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
            },
            promotion_claim=True,
        )


def test_coach_speed_row_rejects_prefixed_string_without_structured_evidence():
    gates = {gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    evidence = {
        gate: f"unit_evidence_for_{gate}"
        for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
    }
    evidence["eval_gif_tournament_load"] = _eval_gif_evidence_ref()
    evidence[COMPACT_COACH_GATE_COACH_SPEED_ROW] = (
        COMPACT_COACH_SPEED_ROW_EVIDENCE_REF_PREFIX + "unit-fake"
    )

    with pytest.raises(CompactCoachCompatibilityError, match="structured"):
        build_compact_coach_compatibility_report_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=False,
            touches_live_runs=False,
            speed_currency="compact_trainer_env_steps_per_sec",
            gates=gates,
            evidence=evidence,
        )


def test_coach_speed_row_support_identity_cannot_close_gate(tmp_path):
    gates = {gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    evidence = {
        gate: f"unit_evidence_for_{gate}"
        for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
    }
    evidence["eval_gif_tournament_load"] = _eval_gif_evidence_ref()
    speed_row = _coach_speed_row_evidence(
        tmp_path,
        model_identity_scope="candidate_named_support_only",
    )
    evidence[COMPACT_COACH_GATE_COACH_SPEED_ROW] = (
        compact_coach_speed_row_evidence_ref(speed_row)
    )

    with pytest.raises(CompactCoachCompatibilityError, match="loaded checkpoint identity"):
        build_compact_coach_compatibility_report_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=False,
            touches_live_runs=False,
            speed_currency="compact_trainer_env_steps_per_sec",
            gates=gates,
            evidence=evidence,
            coach_speed_row_evidence=speed_row,
            promotion_claim=True,
        )


def test_lifecycle_complete_still_needs_coach_speed_row_for_promotion():
    gates = {gate: True for gate in COMPACT_COACH_LIFECYCLE_GATES}
    gates[COMPACT_COACH_GATE_COACH_SPEED_ROW] = False
    evidence = {gate: f"unit_evidence_for_{gate}" for gate in gates if gates[gate]}
    evidence["eval_gif_tournament_load"] = _eval_gif_evidence_ref()

    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="compact_trainer_lifecycle_evidence_no_speed",
        gates=gates,
        evidence=evidence,
    )

    assert report.promotion_eligible is False
    assert report.promotion_blocker == "missing_required_gates"
    assert report.missing_required_gates == (COMPACT_COACH_GATE_COACH_SPEED_ROW,)
    assert COMPACT_COACH_GATE_COACH_SPEED_ROW in report.missing_required_evidence


def test_eval_gif_tournament_load_gate_rejects_loader_only_evidence():
    gates = {gate: False for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    gates.update(
        {
            "trainer_entrypoint": True,
            "checkpoint_save_load": True,
            "resume_metadata": True,
            "eval_gif_tournament_load": True,
        }
    )

    with pytest.raises(CompactCoachCompatibilityError, match="current-chain"):
        build_compact_coach_compatibility_report_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=False,
            touches_live_runs=False,
            speed_currency="compact_trainer_checkpoint_no_speed",
            gates=gates,
            evidence={
                "trainer_entrypoint": "unit_trainer",
                "checkpoint_save_load": "unit_checkpoint",
                "resume_metadata": "unit_resume",
                "eval_gif_tournament_load": "unit_loader_smoke_no_game_or_gif",
            },
        )


def test_compact_owned_promotion_requires_evidence_for_every_required_gate():
    gates = {gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}

    with pytest.raises(CompactCoachCompatibilityError, match="coach_speed_row requires"):
        build_compact_coach_compatibility_report_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=False,
            touches_live_runs=False,
            speed_currency="compact_trainer_env_steps_per_sec",
            gates=gates,
            evidence={"trainer_entrypoint": "unit_compact_trainer"},
            promotion_claim=True,
        )


def test_reward_rnd_contract_gate_requires_its_own_evidence(tmp_path):
    gates = {gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    evidence = {
        gate: f"unit_evidence_for_{gate}"
        for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
        if gate != "reward_rnd_contract"
    }
    evidence["eval_gif_tournament_load"] = _eval_gif_evidence_ref()
    speed_row = _coach_speed_row_evidence(tmp_path)
    evidence[COMPACT_COACH_GATE_COACH_SPEED_ROW] = (
        compact_coach_speed_row_evidence_ref(speed_row)
    )

    with pytest.raises(CompactCoachCompatibilityError, match="reward_rnd_contract"):
        build_compact_coach_compatibility_report_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=False,
            touches_live_runs=False,
            speed_currency="compact_trainer_env_steps_per_sec",
            gates=gates,
            evidence=evidence,
            coach_speed_row_evidence=speed_row,
            promotion_claim=True,
        )


def test_reward_rnd_contract_evidence_removes_only_that_missing_gate():
    gates = {gate: False for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    gates.update(
        {
            "trainer_entrypoint": True,
            "checkpoint_save_load": True,
            "resume_metadata": True,
            "eval_gif_tournament_load": True,
            "reward_rnd_contract": True,
        }
    )
    evidence = {
        "trainer_entrypoint": "unit_trainer",
        "checkpoint_save_load": "unit_checkpoint",
        "resume_metadata": "unit_resume",
        "eval_gif_tournament_load": _eval_gif_evidence_ref(),
        "reward_rnd_contract": "unit_reward_rnd_contract",
    }

    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="compact_trainer_checkpoint_no_speed",
        gates=gates,
        evidence=evidence,
    )

    assert "reward_rnd_contract" not in report.missing_required_gates
    assert "reward_rnd_contract" not in report.missing_required_evidence
    assert report.promotion_eligible is False
    assert report.promotion_blocker == "missing_required_gates"
    assert "death_terminal_contract" in report.missing_required_gates
    assert "policy_refresh_handoff" in report.missing_required_gates


def test_death_terminal_contract_gate_requires_its_own_evidence(tmp_path):
    gates = {gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    evidence = {
        gate: f"unit_evidence_for_{gate}"
        for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
        if gate != "death_terminal_contract"
    }
    evidence["eval_gif_tournament_load"] = _eval_gif_evidence_ref()
    speed_row = _coach_speed_row_evidence(tmp_path)
    evidence[COMPACT_COACH_GATE_COACH_SPEED_ROW] = (
        compact_coach_speed_row_evidence_ref(speed_row)
    )

    with pytest.raises(CompactCoachCompatibilityError, match="death_terminal_contract"):
        build_compact_coach_compatibility_report_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=False,
            touches_live_runs=False,
            speed_currency="compact_trainer_env_steps_per_sec",
            gates=gates,
            evidence=evidence,
            coach_speed_row_evidence=speed_row,
            promotion_claim=True,
        )


def test_death_terminal_contract_evidence_removes_only_that_missing_gate():
    gates = {gate: False for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    gates.update(
        {
            "trainer_entrypoint": True,
            "checkpoint_save_load": True,
            "resume_metadata": True,
            "eval_gif_tournament_load": True,
            "reward_rnd_contract": True,
            "death_terminal_contract": True,
        }
    )
    evidence = {
        "trainer_entrypoint": "unit_trainer",
        "checkpoint_save_load": "unit_checkpoint",
        "resume_metadata": "unit_resume",
        "eval_gif_tournament_load": _eval_gif_evidence_ref(),
        "reward_rnd_contract": "unit_reward_rnd_contract",
        "death_terminal_contract": "unit_death_terminal_contract",
    }

    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="compact_trainer_checkpoint_no_speed",
        gates=gates,
        evidence=evidence,
    )

    assert "death_terminal_contract" not in report.missing_required_gates
    assert "death_terminal_contract" not in report.missing_required_evidence
    assert report.promotion_eligible is False
    assert report.promotion_blocker == "missing_required_gates"
    assert report.missing_required_gates == (
        "policy_refresh_handoff",
        "training_metrics_lineage",
        COMPACT_COACH_GATE_COACH_SPEED_ROW,
    )


def test_policy_refresh_handoff_gate_requires_its_own_evidence(tmp_path):
    gates = {gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    evidence = {
        gate: f"unit_evidence_for_{gate}"
        for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
        if gate != "policy_refresh_handoff"
    }
    evidence["eval_gif_tournament_load"] = _eval_gif_evidence_ref()
    speed_row = _coach_speed_row_evidence(tmp_path)
    evidence[COMPACT_COACH_GATE_COACH_SPEED_ROW] = (
        compact_coach_speed_row_evidence_ref(speed_row)
    )

    with pytest.raises(CompactCoachCompatibilityError, match="policy_refresh_handoff"):
        build_compact_coach_compatibility_report_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=False,
            touches_live_runs=False,
            speed_currency="compact_trainer_env_steps_per_sec",
            gates=gates,
            evidence=evidence,
            coach_speed_row_evidence=speed_row,
            promotion_claim=True,
        )


def test_policy_refresh_handoff_evidence_removes_only_that_missing_gate():
    gates = {gate: False for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    gates.update(
        {
            "trainer_entrypoint": True,
            "checkpoint_save_load": True,
            "resume_metadata": True,
            "eval_gif_tournament_load": True,
            "reward_rnd_contract": True,
            "death_terminal_contract": True,
            "policy_refresh_handoff": True,
        }
    )
    evidence = {
        "trainer_entrypoint": "unit_trainer",
        "checkpoint_save_load": "unit_checkpoint",
        "resume_metadata": "unit_resume",
        "eval_gif_tournament_load": _eval_gif_evidence_ref(),
        "reward_rnd_contract": "unit_reward_rnd_contract",
        "death_terminal_contract": "unit_death_terminal_contract",
        "policy_refresh_handoff": "unit_policy_refresh_handoff",
    }

    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="compact_trainer_checkpoint_no_speed",
        gates=gates,
        evidence=evidence,
    )

    assert "policy_refresh_handoff" not in report.missing_required_gates
    assert "policy_refresh_handoff" not in report.missing_required_evidence
    assert report.promotion_eligible is False
    assert report.promotion_blocker == "missing_required_gates"
    assert report.missing_required_gates == (
        "training_metrics_lineage",
        COMPACT_COACH_GATE_COACH_SPEED_ROW,
    )


def test_stock_bridge_route_requires_train_muzero_call():
    gates = {gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    gates[COMPACT_COACH_GATE_COACH_SPEED_ROW] = False
    evidence = {
        gate: f"unit_evidence_for_{gate}"
        for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
        if gate != COMPACT_COACH_GATE_COACH_SPEED_ROW
    }
    evidence["eval_gif_tournament_load"] = _eval_gif_evidence_ref()
    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="stock_train_muzero_profile_env_steps_per_sec",
        gates=gates,
        evidence=evidence,
    )

    assert report.promotion_eligible is False
    assert report.promotion_blocker == "missing_trainer_entrypoint"


def test_calls_train_muzero_is_rejected_on_compact_owned_route():
    with pytest.raises(CompactCoachCompatibilityError, match="requires"):
        build_compact_coach_compatibility_report_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            profile_only=False,
            calls_train_muzero=True,
            touches_live_runs=False,
            speed_currency="compact_trainer_env_steps_per_sec",
            gates={gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES},
        )


def _complete_compact_owned_compatibility_inputs(tmp_path):
    gates = {gate: True for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    evidence = {
        gate: f"unit_evidence_for_{gate}"
        for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
    }
    evidence["eval_gif_tournament_load"] = _eval_gif_evidence_ref()
    speed_row = _coach_speed_row_evidence(tmp_path)
    evidence[COMPACT_COACH_GATE_COACH_SPEED_ROW] = (
        compact_coach_speed_row_evidence_ref(speed_row)
    )
    return gates, evidence, speed_row


def _eval_gif_evidence_ref() -> str:
    return (
        COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_REF_PREFIX
        + "unit-compact-ckpt:unit-evidence"
    )


def _coach_speed_row_evidence_ref() -> str:
    return COMPACT_COACH_SPEED_ROW_EVIDENCE_REF_PREFIX + "unit-coach-row:evidence"


def _coach_speed_row_evidence(
    tmp_path,
    *,
    model_identity_scope="candidate_loaded_checkpoint",
):
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "row_001_result.json"
    lifecycle = {
        "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
        "ok": True,
        "checkpoint_id": "unit-compact-ckpt",
        "lifecycle_gates_complete": True,
        "missing_required_gates": ["coach_speed_row"],
        "promotion_eligible": False,
    }
    loaded_identity = {}
    if model_identity_scope == "candidate_loaded_checkpoint":
        loaded_identity = _loaded_identity()
        lifecycle["current_chain_identity"] = dict(loaded_identity)
    _write_json(
        lifecycle_path,
        lifecycle,
    )
    _write_json(
        manifest_path,
        {
            "schema_id": "curvyzero_compact_coach_speed_row_manifest/v1",
            "experiment_id": "unit-speed-row",
            "candidate_checkpoint_id": "unit-compact-ckpt",
            "route": "compact_owned_trainer",
            "profile_only": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "non_claims": _non_claims(),
            "rows": [
                {
                    "row_id": "001",
                    "candidate_checkpoint_id": "unit-compact-ckpt",
                    "route": "compact_owned_trainer",
                    "profile_only": False,
                    "calls_train_muzero": False,
                    "touches_live_runs": False,
                    "row_purpose": "coach_speed_row",
                    "speed_currency": "compact_trainer_env_steps_per_sec",
                    "promotion_claim": False,
                    "non_claims": _non_claims(),
                    "command": ["unit", "coach-speed-row"],
                }
            ],
        },
    )
    row = json.loads(manifest_path.read_text(encoding="utf-8"))["rows"][0]
    _write_json(
        result_path,
        {
            "schema_id": "curvyzero_compact_coach_speed_row_result/v1",
            "ok": True,
            "status": "complete",
            "problem": None,
            "returncode": 0,
            "run_invocation_id": "unit-run-invocation",
            "candidate_checkpoint_id": "unit-compact-ckpt",
            "row_id": "001",
            "row": row,
            "producer": {
                "schema_id": "curvyzero_compact_coach_speed_row_producer/v1",
                "producer_id": "unit-speed-row-producer",
                "run_id": "unit-speed-row",
                "produced_by": "tests/test_compact_coach_compatibility.py",
            },
            "summary": {
                "profile_only": False,
                "calls_train_muzero": False,
                "touches_live_runs": False,
                "status": "complete",
                "ok": True,
                "row_id": "001",
                "candidate_checkpoint_id": "unit-compact-ckpt",
                "route": "compact_owned_trainer",
                "row_purpose": "coach_speed_row",
                "promotion_claim": False,
                "speed_currency": "compact_trainer_env_steps_per_sec",
                "env_steps_collected": 120.0,
                "training_wall_sec": 2.0,
                "steps_per_sec": 60.0,
                "non_claims": _non_claims(),
            },
            "compact": {
                "ok": True,
                "candidate_checkpoint_id": "unit-compact-ckpt",
                "route": "compact_owned_trainer",
                "profile_only": False,
                "calls_train_muzero": False,
                "touches_live_runs": False,
                "real_compact_owned_training_work": True,
                "compact_owned_trainer_update_count": 1,
                "compact_owned_trainer_env_step_source": "unit_fixture",
                "model_identity_scope": model_identity_scope,
                "loaded_checkpoint_identity": loaded_identity,
                "non_claims": _non_claims(),
            },
            "non_claims": _non_claims(),
        },
    )
    return build_compact_coach_speed_row_evidence_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=lifecycle_path,
        manifest_path=manifest_path,
        row_id="001",
        result_json_path=result_path,
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _non_claims():
    return {
        "promotion_claim": False,
        "training_speedup_claim": False,
        "live_run_safety_claim": False,
        "stock_resume_claim": False,
        "rating_or_promotion_quality_claim": False,
    }


def _loaded_identity():
    return {
        "scope": "candidate_loaded_checkpoint",
        "identity_source": "unit_fixture",
        "candidate_loaded_checkpoint": True,
        "checkpoint_id": "unit-compact-ckpt",
        "trainer_id": "unit-compact-ckpt:trainer",
        "policy_version_ref": "unit-compact-ckpt:policy-update-1",
        "model_version_ref": "unit-compact-ckpt:model-update-1",
        "policy_source": "unit_policy_source",
        "learner_update_count": 1,
        "model_state_digest": "a" * 64,
        "compact_checkpoint_sha256": "b" * 64,
    }
