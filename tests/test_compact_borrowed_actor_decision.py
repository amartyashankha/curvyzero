from __future__ import annotations

import json

import pytest

from curvyzero.training.compact_borrowed_actor_decision import (
    DECISION_APPROVED_A1_BORROWED_OPERATIONAL_SPEED_CANDIDATE,
)
from curvyzero.training.compact_borrowed_actor_decision import (
    CompactBorrowedActorDecisionError,
)
from curvyzero.training.compact_borrowed_actor_decision import (
    build_compact_borrowed_actor_decision_v1,
)
from curvyzero.training.compact_borrowed_actor_decision import (
    validate_compact_borrowed_actor_decision_v1,
)


def test_borrowed_actor_decision_approves_two_lane_candidate(tmp_path):
    paths = _decision_fixture(tmp_path)

    payload = build_compact_borrowed_actor_decision_v1(
        run_id="unit-opt098",
        borrowed_speed_row_paths=paths["borrowed"],
        normal_death_profile_result_path=paths["normal_death"],
        same_shape_reference_speed_row_path=paths["reference"],
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["decision"] == DECISION_APPROVED_A1_BORROWED_OPERATIONAL_SPEED_CANDIDATE
    assert payload["attached_claims"]["borrowed_a1_operational_speed_candidate_allowed"] is True
    assert payload["attached_claims"]["same_shape_baseline_replaced"] is False
    assert payload["attached_claims"]["normal_death_speed_claim"] is False
    assert payload["speed_lane"]["mean_steps_per_sec"] == pytest.approx(32_500.0)
    validate_compact_borrowed_actor_decision_v1(payload)


def test_borrowed_actor_decision_rejects_copying_nodeath_speed_row(tmp_path):
    paths = _decision_fixture(tmp_path)
    row = _read_json(paths["borrowed"][0])
    row["compact"]["render_state_copy_steps"] = 1
    _write_json(paths["borrowed"][0], row)

    payload = build_compact_borrowed_actor_decision_v1(
        run_id="unit-opt098",
        borrowed_speed_row_paths=paths["borrowed"],
        normal_death_profile_result_path=paths["normal_death"],
        same_shape_reference_speed_row_path=paths["reference"],
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["decision"] == "not_approved_speed_rows_failed"
    assert payload["attached_claims"]["borrowed_a1_operational_speed_candidate_allowed"] is False


def test_borrowed_actor_decision_rejects_missing_normal_death_proof(tmp_path):
    paths = _decision_fixture(tmp_path)
    proof = _read_json(paths["normal_death"])
    proof["compact"]["terminal_row_count"] = 0
    _write_json(paths["normal_death"], proof)

    payload = build_compact_borrowed_actor_decision_v1(
        run_id="unit-opt098",
        borrowed_speed_row_paths=paths["borrowed"],
        normal_death_profile_result_path=paths["normal_death"],
        same_shape_reference_speed_row_path=paths["reference"],
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["decision"] == "not_approved_terminal_death_proof_failed"


def test_borrowed_actor_decision_validation_rejects_claim_drift(tmp_path):
    paths = _decision_fixture(tmp_path)
    payload = build_compact_borrowed_actor_decision_v1(
        run_id="unit-opt098",
        borrowed_speed_row_paths=paths["borrowed"],
        normal_death_profile_result_path=paths["normal_death"],
        same_shape_reference_speed_row_path=paths["reference"],
        created_at="2026-05-31T00:00:00+00:00",
    )
    payload["attached_claims"]["normal_death_speed_claim"] = True

    with pytest.raises(CompactBorrowedActorDecisionError, match="must be false"):
        validate_compact_borrowed_actor_decision_v1(payload)


def test_borrowed_actor_decision_validation_rejects_input_hash_drift(tmp_path):
    paths = _decision_fixture(tmp_path)
    payload = build_compact_borrowed_actor_decision_v1(
        run_id="unit-opt098",
        borrowed_speed_row_paths=paths["borrowed"],
        normal_death_profile_result_path=paths["normal_death"],
        same_shape_reference_speed_row_path=paths["reference"],
        created_at="2026-05-31T00:00:00+00:00",
    )
    row = _read_json(paths["borrowed"][0])
    row["compact"]["steps_per_sec"] = 1.0
    _write_json(paths["borrowed"][0], row)

    with pytest.raises(CompactBorrowedActorDecisionError, match="sha mismatch"):
        validate_compact_borrowed_actor_decision_v1(payload)


def _decision_fixture(tmp_path):
    borrowed_1 = tmp_path / "borrowed_r1.json"
    borrowed_2 = tmp_path / "borrowed_r2.json"
    reference = tmp_path / "reference.json"
    normal_death = tmp_path / "normal_death.json"
    _write_json(borrowed_1, _speed_row(actor_count=1, borrowed=True, speed=30_000.0, wall=6.1))
    _write_json(borrowed_2, _speed_row(actor_count=1, borrowed=True, speed=35_000.0, wall=5.2))
    _write_json(reference, _speed_row(actor_count=16, borrowed=False, speed=17_500.0, wall=10.5))
    _write_json(normal_death, _normal_death_row())
    return {
        "borrowed": [borrowed_1, borrowed_2],
        "reference": reference,
        "normal_death": normal_death,
    }


def _speed_row(*, actor_count: int, borrowed: bool, speed: float, wall: float):
    steps = 180
    warmup_steps = 45
    row = {
        "batch_size": 1024,
        "actor_count": actor_count,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "num_simulations": 1,
        "policy_refresh_interval": 4,
        "learner_device": "cuda",
        "hybrid_borrow_single_actor_render_state": borrowed,
    }
    handoff = "borrow_single_actor_env_state" if borrowed else "copy_actor_state_to_parent_buffers"
    return {
        "schema_id": "curvyzero_compact_coach_speed_row_result/v1",
        "ok": True,
        "status": "complete",
        "returncode": 0,
        "row": row,
        "summary": {
            "ok": True,
            "profile_only": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        },
        "compact": {
            "ok": True,
            "profile_only": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "real_compact_owned_training_work": True,
            "hybrid_borrow_single_actor_render_state": borrowed,
            "render_state_handoff_mode": handoff,
            "render_state_copy_steps": 0 if borrowed else steps + warmup_steps,
            "render_state_borrowed_steps": steps + warmup_steps if borrowed else 0,
            "compact_owned_loop_deferred_learner": False,
            "compact_rollout_slab_learner_gate_updates": 22,
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": 22,
            "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_count": steps,
            "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_count": steps,
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata": {
                "compact_policy_refresh_learner_update_count": 22,
            },
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata": {
                "compact_policy_refresh_learner_update_count": 22,
            },
            "steps_per_sec": speed,
            "training_wall_sec": wall,
            "env_steps_collected": 184_320.0,
            "source_profile_payload": {
                "death_mode": "profile_no_death",
                "terminal_row_count": 0,
                "death_row_count": 0,
                "terminated_row_count": 0,
                "truncated_row_count": 0,
                "resident_observation_host_fallback_count": 0.0,
            },
        },
    }


def _normal_death_row():
    return {
        "schema_id": "curvyzero_hybrid_observation_profile_collected_result/v0",
        "status": "complete",
        "returncode": 0,
        "row": {
            "batch_size": 1024,
            "actor_count": 1,
            "death_mode": "normal",
            "hybrid_borrow_single_actor_render_state": True,
        },
        "summary": {
            "profile_only": True,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "steps_per_sec": 11_000.0,
            "normal_death_terminal_contract_promotion_gate_satisfied": True,
            "render_state_handoff_mode": "borrow_single_actor_env_state",
            "render_state_copy_steps": 47,
            "render_state_borrowed_steps": 64,
            "borrow_single_actor_render_state": True,
            "resident_observation_used": True,
            "resident_observation_host_fallback_count": 0.0,
            "terminal_final_observation_before_autoreset_verified": True,
            "terminal_final_observation_row_count": 198,
            "normal_death_terminal_contract_evidence": {
                "terminal_sample_row_count": 15,
                "resident_terminal_final_observation_used": True,
                "terminal_unroll_value_target_row_count": 15,
                "terminal_unroll_value_target_mode": (
                    "stock_terminal_no_bootstrap_return_discount_1.0"
                ),
            },
        },
        "compact": {
            "death_mode": "normal",
            "terminal_row_count": 198,
            "terminated_row_count": 198,
            "truncated_row_count": 0,
        },
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
