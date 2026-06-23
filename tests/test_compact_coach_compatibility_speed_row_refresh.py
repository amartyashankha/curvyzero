from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

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
    COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_REF_PREFIX,
)
from curvyzero.training.compact_coach_compatibility import (
    build_compact_coach_compatibility_report_v1,
)
from curvyzero.training.compact_coach_speed_row import (
    build_compact_coach_speed_row_evidence_v1,
)
from curvyzero.training.compact_coach_speed_row import (
    compact_coach_speed_row_evidence_ref,
)


def test_speed_row_refresh_attaches_loaded_checkpoint_speed_evidence(tmp_path):
    module = _load_refresh_module()
    paths = _speed_row_fixture(tmp_path)

    payload = module.build_compact_coach_compatibility_speed_row_refresh_payload(
        run_id="unit-refresh",
        unified_lifecycle_report_path=paths["lifecycle"],
        speed_row_modal_report_path=paths["modal_report"],
        min_env_steps=1.0,
        min_training_wall_sec=1.0,
        min_steps_per_sec=1.0,
        created_at="2026-05-30T00:00:00+00:00",
    )

    metadata = payload["compatibility_metadata"]
    assert payload["ok"] is True
    assert payload["coach_speed_row_gate"] is True
    assert payload["promotion_eligible"] is True
    assert payload["promotion_claim"] is False
    assert payload["missing_required_gates"] == []
    assert payload["missing_required_evidence"] == []
    assert metadata["compact_coach_compatibility_gate_coach_speed_row"] is True
    assert (
        metadata["compact_coach_compatibility_speed_currency"]
        == "compact_trainer_env_steps_per_sec"
    )
    assert (
        metadata["compact_coach_compatibility_evidence"]["coach_speed_row"]
        == payload["speed_row_evidence_ref"]
    )
    assert payload["thresholds"]["passed"] is True
    assert payload["non_claims"]["training_speedup_claim"] is False
    assert payload["search_config"] == _search_config()
    assert payload["actor_handoff_config"] == _actor_handoff_config()
    assert payload["operational_surface"] == _operational_surface()
    assert payload["death_mode"] == "profile_no_death"
    assert payload["compact_torch_initial_inference_mode"] == "direct_core"
    assert payload["compact_torch_observation_memory_format"] == "channels_last"
    assert payload["compact_torch_model_memory_format"] == "contiguous"
    assert payload["compact_torch_defer_one_simulation_replay_payload_requested"] is False
    assert payload["compact_torch_memory_format_applies_to_search_service"] is True
    assert payload["hybrid_persistent_compact_render_state_buffer"] is True
    assert payload["render_state_handoff_mode"] == "persistent_compact_render_state_buffer"


def test_speed_row_refresh_writes_report_from_main(tmp_path):
    module = _load_refresh_module()
    paths = _speed_row_fixture(tmp_path)
    output_root = tmp_path / "compat"

    assert (
        module.main(
            [
                "--run-id",
                "unit-refresh-main",
                "--output-root",
                str(output_root),
                "--unified-lifecycle-report",
                str(paths["lifecycle"]),
                "--speed-row-modal-report",
                str(paths["modal_report"]),
                "--min-env-steps",
                "1",
                "--min-training-wall-sec",
                "1",
                "--min-steps-per-sec",
                "1",
            ]
        )
        == 0
    )

    report_path = output_root / "unit-refresh-main" / "compatibility_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert (
        payload["compatibility_metadata"]["compact_coach_compatibility_gate_coach_speed_row"]
        is True
    )


def test_speed_row_refresh_rejects_nodeath_speed_for_normal_death_lifecycle(tmp_path):
    module = _load_refresh_module()
    paths = _speed_row_fixture(tmp_path, normal_death_lifecycle=True)

    with pytest.raises(module.CompactCoachSpeedRowRefreshError, match="normal-death"):
        module.build_compact_coach_compatibility_speed_row_refresh_payload(
            run_id="unit-refresh",
            unified_lifecycle_report_path=paths["lifecycle"],
            speed_row_modal_report_path=paths["modal_report"],
            min_env_steps=1.0,
            min_training_wall_sec=1.0,
            min_steps_per_sec=1.0,
            created_at="2026-05-30T00:00:00+00:00",
        )


def test_speed_row_refresh_rejects_report_evidence_ref_mismatch(tmp_path):
    module = _load_refresh_module()
    paths = _speed_row_fixture(tmp_path)
    modal = _read_json(paths["modal_report"])
    modal["evidence_ref"] = "compact_coach_speed_row:wrong"
    _write_json(paths["modal_report"], modal)

    with pytest.raises(module.CompactCoachSpeedRowRefreshError, match="evidence_ref"):
        module.build_compact_coach_compatibility_speed_row_refresh_payload(
            run_id="unit-refresh",
            unified_lifecycle_report_path=paths["lifecycle"],
            speed_row_modal_report_path=paths["modal_report"],
            min_env_steps=1.0,
            min_training_wall_sec=1.0,
            min_steps_per_sec=1.0,
        )


def test_speed_row_refresh_rejects_modal_memory_format_drift(tmp_path):
    module = _load_refresh_module()
    paths = _speed_row_fixture(tmp_path)
    modal = _read_json(paths["modal_report"])
    modal["compact_torch_observation_memory_format"] = "contiguous"
    _write_json(paths["modal_report"], modal)

    with pytest.raises(module.CompactCoachSpeedRowRefreshError, match="memory_format"):
        module.build_compact_coach_compatibility_speed_row_refresh_payload(
            run_id="unit-refresh",
            unified_lifecycle_report_path=paths["lifecycle"],
            speed_row_modal_report_path=paths["modal_report"],
            min_env_steps=1.0,
            min_training_wall_sec=1.0,
            min_steps_per_sec=1.0,
        )


def test_speed_row_refresh_rejects_modal_actor_handoff_drift(tmp_path):
    module = _load_refresh_module()
    paths = _speed_row_fixture(tmp_path)
    modal = _read_json(paths["modal_report"])
    modal["render_state_handoff_mode"] = "copy_actor_state_to_parent_buffers"
    _write_json(paths["modal_report"], modal)

    with pytest.raises(module.CompactCoachSpeedRowRefreshError, match="render_state"):
        module.build_compact_coach_compatibility_speed_row_refresh_payload(
            run_id="unit-refresh",
            unified_lifecycle_report_path=paths["lifecycle"],
            speed_row_modal_report_path=paths["modal_report"],
            min_env_steps=1.0,
            min_training_wall_sec=1.0,
            min_steps_per_sec=1.0,
        )


def test_speed_row_refresh_rejects_support_only_identity(tmp_path):
    module = _load_refresh_module()
    paths = _speed_row_fixture(
        tmp_path,
        model_identity_scope="candidate_named_support_only",
    )

    with pytest.raises(
        module.CompactCoachSpeedRowRefreshError,
        match="loaded checkpoint identity",
    ):
        module.build_compact_coach_compatibility_speed_row_refresh_payload(
            run_id="unit-refresh",
            unified_lifecycle_report_path=paths["lifecycle"],
            speed_row_modal_report_path=paths["modal_report"],
            min_env_steps=1.0,
            min_training_wall_sec=1.0,
            min_steps_per_sec=1.0,
        )


def test_speed_row_refresh_rejects_too_short_threshold_row(tmp_path):
    module = _load_refresh_module()
    paths = _speed_row_fixture(tmp_path, env_steps=120.0, wall_sec=2.0)

    with pytest.raises(module.CompactCoachSpeedRowRefreshError, match="threshold"):
        module.build_compact_coach_compatibility_speed_row_refresh_payload(
            run_id="unit-refresh",
            unified_lifecycle_report_path=paths["lifecycle"],
            speed_row_modal_report_path=paths["modal_report"],
            min_env_steps=61_440.0,
            min_training_wall_sec=10.0,
            min_steps_per_sec=1.0,
        )


def test_speed_row_refresh_rejects_modal_report_not_ok(tmp_path):
    module = _load_refresh_module()
    paths = _speed_row_fixture(tmp_path)
    modal = _read_json(paths["modal_report"])
    modal["ok"] = False
    _write_json(paths["modal_report"], modal)

    with pytest.raises(module.CompactCoachSpeedRowRefreshError, match="ok=true"):
        module.build_compact_coach_compatibility_speed_row_refresh_payload(
            run_id="unit-refresh",
            unified_lifecycle_report_path=paths["lifecycle"],
            speed_row_modal_report_path=paths["modal_report"],
            min_env_steps=1.0,
            min_training_wall_sec=1.0,
            min_steps_per_sec=1.0,
        )


def _load_refresh_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / ("build_compact_coach_compatibility_speed_row_refresh.py")
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_coach_compatibility_speed_row_refresh_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _speed_row_fixture(
    tmp_path,
    *,
    env_steps: float = 122_880.0,
    wall_sec: float = 12.0,
    model_identity_scope: str = "candidate_loaded_checkpoint",
    normal_death_lifecycle: bool = False,
) -> dict[str, Path]:
    checkpoint_id = "unit-compact-ckpt"
    speed = env_steps / wall_sec
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "row_001_result.json"
    evidence_path = Path(f"{result_path}.compact_coach_speed_row.evidence.json")
    modal_report_path = tmp_path / "compact_coach_speed_row_modal_report.json"
    loaded_identity = (
        _loaded_identity(checkpoint_id)
        if model_identity_scope == "candidate_loaded_checkpoint"
        else {}
    )
    search_config = _search_config()
    actor_handoff_config = _actor_handoff_config()
    operational_surface = _operational_surface()
    lifecycle = {
        "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
        "ok": True,
        "run_id": checkpoint_id,
        "checkpoint_id": checkpoint_id,
        "lifecycle_gates_complete": True,
        "missing_required_gates": [COMPACT_COACH_GATE_COACH_SPEED_ROW],
        "missing_required_evidence": [COMPACT_COACH_GATE_COACH_SPEED_ROW],
        "promotion_eligible": False,
        "promotion_blocker": "missing_required_gates",
        "current_chain_identity": _loaded_identity(checkpoint_id),
        "compatibility_metadata": _base_compatibility_metadata(
            normal_death=normal_death_lifecycle
        ),
    }
    _write_json(lifecycle_path, lifecycle)
    manifest = {
        "schema_id": "curvyzero_compact_coach_speed_row_manifest/v1",
        "experiment_id": "unit-speed-row",
        "candidate_checkpoint_id": checkpoint_id,
        "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "death_mode": operational_surface["death_mode"],
        **search_config,
        "hybrid_persistent_compact_render_state_buffer": (
            actor_handoff_config["hybrid_persistent_compact_render_state_buffer"]
        ),
        "non_claims": _non_claims(),
        "rows": [
            {
                "row_id": "001",
                "candidate_checkpoint_id": checkpoint_id,
                "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
                "profile_only": False,
                "calls_train_muzero": False,
                "touches_live_runs": False,
                "row_purpose": "coach_speed_row",
                "speed_currency": "compact_trainer_env_steps_per_sec",
                "promotion_claim": False,
                "actor_count": operational_surface["actor_count"],
                "batch_size": operational_surface["batch_size"],
                "steps": operational_surface["steps"],
                "warmup_steps": operational_surface["warmup_steps"],
                "death_mode": operational_surface["death_mode"],
                **search_config,
                "hybrid_persistent_compact_render_state_buffer": (
                    actor_handoff_config["hybrid_persistent_compact_render_state_buffer"]
                ),
                "non_claims": _non_claims(),
                "command": ["unit", "speed-row"],
            }
        ],
    }
    _write_json(manifest_path, manifest)
    row = manifest["rows"][0]
    result = {
        "schema_id": "curvyzero_compact_coach_speed_row_result/v1",
        "ok": True,
        "status": "complete",
        "problem": None,
        "returncode": 0,
        "run_invocation_id": "unit-run-invocation",
        "candidate_checkpoint_id": checkpoint_id,
        "row_id": "001",
        "row": row,
        "producer": {
            "schema_id": "curvyzero_compact_coach_speed_row_producer/v1",
            "producer_id": "unit-producer",
            "run_id": "unit-speed-row",
            "produced_by": "unit-test",
        },
        "summary": {
            "profile_only": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "status": "complete",
            "ok": True,
            "row_id": "001",
            "candidate_checkpoint_id": checkpoint_id,
            "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            "row_purpose": "coach_speed_row",
            "promotion_claim": False,
            "speed_currency": "compact_trainer_env_steps_per_sec",
            **operational_surface,
            **search_config,
            **actor_handoff_config,
            "env_steps_collected": env_steps,
            "training_wall_sec": wall_sec,
            "steps_per_sec": speed,
            "non_claims": _non_claims(),
        },
        "compact": {
            "ok": True,
            "candidate_checkpoint_id": checkpoint_id,
            "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            "profile_only": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "real_compact_owned_training_work": True,
            "compact_owned_trainer_update_count": 7,
            "compact_owned_trainer_env_step_source": "unit_fixture",
            "model_identity_scope": model_identity_scope,
            "loaded_checkpoint_identity": loaded_identity,
            **operational_surface,
            **search_config,
            **actor_handoff_config,
            "non_claims": _non_claims(),
        },
        "non_claims": _non_claims(),
    }
    _write_json(result_path, result)
    evidence = build_compact_coach_speed_row_evidence_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        candidate_checkpoint_id=checkpoint_id,
        unified_lifecycle_report_path=lifecycle_path,
        manifest_path=manifest_path,
        row_id="001",
        result_json_path=result_path,
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    _write_json(evidence_path, evidence)
    modal_report = {
        "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
        "ok": True,
        "run_id": "unit-speed-row",
        "candidate_checkpoint_id": checkpoint_id,
        "manifest_path": str(manifest_path),
        "result_path": str(result_path),
        "evidence_path": str(evidence_path),
        "evidence_ref": compact_coach_speed_row_evidence_ref(evidence),
        "speed_currency": "compact_trainer_env_steps_per_sec",
        "env_steps_collected": env_steps,
        "training_wall_sec": wall_sec,
        "steps_per_sec": speed,
        **search_config,
        **actor_handoff_config,
        **operational_surface,
        "model_identity_scope": model_identity_scope,
        "profile_support_profile_only": True,
        "real_compact_owned_training_work": True,
        "promotion_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    _write_json(modal_report_path, modal_report)
    return {
        "lifecycle": lifecycle_path,
        "manifest": manifest_path,
        "result": result_path,
        "evidence": evidence_path,
        "modal_report": modal_report_path,
    }


def _base_compatibility_metadata(*, normal_death: bool = False) -> dict[str, Any]:
    gates = {gate: True for gate in COMPACT_COACH_LIFECYCLE_GATES}
    gates[COMPACT_COACH_GATE_COACH_SPEED_ROW] = False
    evidence = {
        gate: f"unit_evidence_for_{gate}"
        for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
        if gate != COMPACT_COACH_GATE_COACH_SPEED_ROW
    }
    evidence["eval_gif_tournament_load"] = (
        COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_REF_PREFIX
        + "unit-compact-ckpt:current-chain"
    )
    if normal_death:
        evidence["death_terminal_contract"] = (
            "curvyzero_compact_death_terminal_contract/v1:normal:"
            "normal_collision_death_terminal_nstep_v1:"
            "stock_terminal_no_bootstrap_return_discount_1.0:unit:"
            "normal_death=true:terminated=true:truncated=false:promotion_gate=true"
        )
    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="compact_trainer_lifecycle_evidence_no_speed",
        gates=gates,
        evidence=evidence,
        promotion_claim=False,
    )
    return report.as_metadata()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _search_config() -> dict[str, Any]:
    return {
        "search_service_kind": "compact_torch_search_service",
        "search_service_impl": "compact_torch_device_tree_fixed_shape_v0",
        "compact_torch_initial_inference_mode": "direct_core",
        "compact_torch_observation_memory_format": "channels_last",
        "compact_torch_model_memory_format": "contiguous",
        "compact_torch_defer_one_simulation_replay_payload_requested": False,
        "compact_torch_memory_format_applies_to_search_service": True,
    }


def _actor_handoff_config() -> dict[str, Any]:
    return {
        "hybrid_persistent_compact_render_state_buffer": True,
        "hybrid_borrow_single_actor_render_state": False,
        "render_state_handoff_mode": "persistent_compact_render_state_buffer",
        "render_state_copy_steps": 0,
        "render_state_borrowed_steps": 0,
        "render_state_row_overlay_steps": 0,
        "render_state_row_overlay_rows": 0,
        "render_state_row_overlay_bytes": 0,
    }


def _operational_surface() -> dict[str, Any]:
    return {
        "actor_count": 1,
        "batch_size": 1024,
        "steps": 120,
        "warmup_steps": 10,
        "death_mode": "profile_no_death",
        "compact_owned_training_loop_owner": "",
        "compact_owned_trainer_config_death_mode": "",
        "normal_death_terminal_contract_owner": "none",
        "terminal_row_count": 0,
        "death_row_count": 0,
        "terminated_row_count": 0,
        "truncated_row_count": 0,
        "terminal_final_observation_row_count": 0,
        "terminal_final_observation_before_autoreset_verified": False,
        "terminal_sample_row_count": 0,
        "terminal_unroll_value_target_mode": "none",
        "terminal_unroll_value_target_row_count": 0,
        "resident_observation_host_fallback_count": 0.0,
        "normal_death_terminal_contract_promotion_gate_satisfied": False,
        "source_profile_total_sec": 0.0,
        "source_profile_warmup_sec": 0.0,
        "source_profile_measured_sec": 0.0,
        "source_profile_timing_per_timestep_sec": 0.0,
        "speed_row_actor_step_wall_sec": 0.0,
        "speed_row_observation_sec": 0.0,
        "speed_row_renderer_stack_update_sec": 0.0,
        "speed_row_compact_rollout_slab_sec": 0.0,
        "speed_row_sample_gate_sec": 0.0,
        "speed_row_learner_gate_sec": 0.0,
        "speed_row_policy_refresh_sec": 0.0,
        "speed_row_primary_accounted_sec": 0.0,
        "speed_row_primary_residual_sec": 0.0,
    }


def _non_claims() -> dict[str, bool]:
    return {
        "promotion_claim": False,
        "training_speedup_claim": False,
        "live_run_safety_claim": False,
        "stock_resume_claim": False,
        "rating_or_promotion_quality_claim": False,
    }


def _loaded_identity(checkpoint_id: str) -> dict[str, Any]:
    return {
        "scope": "candidate_loaded_checkpoint",
        "identity_source": "unit_loaded_checkpoint",
        "candidate_loaded_checkpoint": True,
        "checkpoint_id": checkpoint_id,
        "trainer_id": f"{checkpoint_id}:trainer",
        "policy_version_ref": f"{checkpoint_id}:policy-update-1",
        "model_version_ref": f"{checkpoint_id}:model-update-1",
        "policy_source": "unit_policy_source",
        "learner_update_count": 1,
        "model_state_digest": "a" * 64,
        "compact_checkpoint_path": "/tmp/unit-compact-checkpoint.pt",
        "compact_checkpoint_sha256": "b" * 64,
        "support_scale": 300,
    }
