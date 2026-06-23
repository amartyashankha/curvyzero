from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def test_speed_row_smoke_loaded_checkpoint_mode_emits_loaded_identity(
    tmp_path,
    monkeypatch,
):
    module = _load_smoke_module()
    loaded_identity = _loaded_identity()
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    lifecycle_path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
                "ok": True,
                "checkpoint_id": "unit-compact-ckpt",
                "lifecycle_gates_complete": True,
                "missing_required_gates": ["coach_speed_row"],
                "promotion_eligible": False,
                "current_chain_identity": dict(loaded_identity),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "results"
    called: dict[str, Any] = {}
    loaded_model = SimpleNamespace(weight=1)

    def fake_load_lifecycle_checkpoint_model(**kwargs):
        called["load_kwargs"] = kwargs
        return {"model": loaded_model, "identity": dict(loaded_identity)}

    def fake_run_profile(
        *,
        args,
        learner_model: Any | None = None,
        search_model: Any | None = None,
        loaded_checkpoint_identity: Any | None = None,
    ):
        called["profile_args"] = args
        called["profile_learner_model"] = learner_model
        called["profile_search_model"] = search_model
        called["profile_loaded_identity"] = dict(loaded_checkpoint_identity or {})
        return _profile_payload(
            persistent_render_state=bool(args.hybrid_persistent_compact_render_state_buffer),
            borrow_render_state=bool(args.hybrid_borrow_single_actor_render_state),
        )

    monkeypatch.setattr(
        module,
        "_load_lifecycle_checkpoint_model",
        fake_load_lifecycle_checkpoint_model,
    )
    monkeypatch.setattr(module, "_run_local_compact_owned_profile", fake_run_profile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_compact_coach_speed_row_smoke.py",
            "--run-id",
            "unit-speed-row-loaded",
            "--output-root",
            str(output_root),
            "--unified-lifecycle-report",
            str(lifecycle_path),
            "--load-unified-lifecycle-checkpoint",
            "--search-service-kind",
            "compact_torch_search_service",
            "--compact-torch-initial-inference-mode",
            "direct_core",
            "--compact-torch-observation-memory-format",
            "channels_last",
            "--compact-torch-model-memory-format",
            "contiguous",
            "--hybrid-persistent-compact-render-state-buffer",
        ],
    )

    assert module.main() == 0

    result_path = output_root / "unit-speed-row-loaded" / "row_001_result.json"
    evidence_path = Path(f"{result_path}.compact_coach_speed_row.evidence.json")
    report_path = (
        output_root / "unit-speed-row-loaded" / "compact_coach_speed_row_smoke_report.json"
    )
    manifest = json.loads(
        (output_root / "unit-speed-row-loaded" / "manifest.json").read_text(encoding="utf-8")
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert called["load_kwargs"]["lifecycle_path"] == lifecycle_path
    assert called["profile_args"].hybrid_persistent_compact_render_state_buffer is True
    assert called["profile_learner_model"] is loaded_model
    assert called["profile_search_model"] is not loaded_model
    assert called["profile_loaded_identity"]["checkpoint_id"] == "unit-compact-ckpt"
    assert result["compact"]["model_identity_scope"] == "candidate_loaded_checkpoint"
    assert result["compact"]["loaded_checkpoint_identity"] == loaded_identity
    assert evidence["model_identity"]["scope"] == "candidate_loaded_checkpoint"
    assert (
        evidence["model_identity"]["result_loaded_checkpoint_identity"]["model_state_digest"]
        == loaded_identity["model_state_digest"]
    )
    assert report["model_identity_scope"] == "candidate_loaded_checkpoint"
    expected_search_config = {
        "search_service_kind": "compact_torch_search_service",
        "search_service_impl": "compact_torch_device_tree_fixed_shape_v0",
        "compact_torch_initial_inference_mode": "direct_core",
        "compact_torch_observation_memory_format": "channels_last",
        "compact_torch_model_memory_format": "contiguous",
        "compact_torch_defer_one_simulation_replay_payload_requested": False,
        "compact_torch_memory_format_applies_to_search_service": True,
    }
    assert {
        key: manifest[key]
        for key in (
            "compact_torch_observation_memory_format",
            "compact_torch_initial_inference_mode",
            "compact_torch_model_memory_format",
            "compact_torch_defer_one_simulation_replay_payload_requested",
            "compact_torch_memory_format_applies_to_search_service",
        )
    } == {
        key: expected_search_config[key]
        for key in (
            "compact_torch_observation_memory_format",
            "compact_torch_initial_inference_mode",
            "compact_torch_model_memory_format",
            "compact_torch_defer_one_simulation_replay_payload_requested",
            "compact_torch_memory_format_applies_to_search_service",
        )
    }
    assert evidence["search_config"] == expected_search_config
    for payload in (manifest["rows"][0], result["summary"], result["compact"], report):
        assert payload["compact_torch_observation_memory_format"] == "channels_last"
        assert payload["compact_torch_initial_inference_mode"] == "direct_core"
        assert payload["compact_torch_model_memory_format"] == "contiguous"
        assert payload["compact_torch_defer_one_simulation_replay_payload_requested"] is False
        assert payload["compact_torch_memory_format_applies_to_search_service"] is True
        assert payload["hybrid_persistent_compact_render_state_buffer"] is True
        if payload is not manifest["rows"][0]:
            assert payload["render_state_handoff_mode"] == (
                "persistent_compact_render_state_buffer"
            )


def test_speed_row_smoke_lean_trainer_step_routes_away_from_profile_runner(
    tmp_path,
    monkeypatch,
):
    module = _load_smoke_module()
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    lifecycle_path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
                "ok": True,
                "checkpoint_id": "unit-compact-ckpt",
                "lifecycle_gates_complete": True,
                "missing_required_gates": ["coach_speed_row"],
                "promotion_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "results"
    called: dict[str, int] = {"lean": 0, "profile": 0}

    def fake_profile_runner(**kwargs):
        called["profile"] += 1
        raise AssertionError("legacy profile runner should not be called")

    def fake_lean_runner(
        *,
        args,
        learner_model: Any | None = None,
        search_model: Any | None = None,
        loaded_checkpoint_identity: Any | None = None,
    ):
        del learner_model, search_model, loaded_checkpoint_identity
        called["lean"] += 1
        assert args.compact_owned_lean_trainer_step is True
        payload = _profile_payload(
            persistent_render_state=bool(args.hybrid_persistent_compact_render_state_buffer),
            borrow_render_state=bool(args.hybrid_borrow_single_actor_render_state),
        )
        payload["compact_owned_lean_trainer_step"] = True
        payload["compact_owned_training_loop_owner"] = "lean_compact_trainer_step"
        _attach_lean_trainer_counters(payload)
        return payload

    monkeypatch.setattr(module, "_run_local_compact_owned_profile", fake_profile_runner)
    monkeypatch.setattr(
        module,
        "_run_local_compact_owned_lean_trainer_profile",
        fake_lean_runner,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_compact_coach_speed_row_smoke.py",
            "--run-id",
            "unit-speed-row-lean",
            "--output-root",
            str(output_root),
            "--unified-lifecycle-report",
            str(lifecycle_path),
            "--compact-owned-lean-trainer-step",
        ],
    )

    assert module.main() == 0

    assert called == {"lean": 1, "profile": 0}
    run_dir = output_root / "unit-speed-row-lean"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "row_001_result.json").read_text(encoding="utf-8"))
    report = json.loads(
        (run_dir / "compact_coach_speed_row_smoke_report.json").read_text(encoding="utf-8")
    )
    for payload in (
        manifest,
        manifest["rows"][0],
        result["summary"],
        result["compact"],
        report,
    ):
        assert payload["compact_owned_lean_trainer_step"] is True
        assert payload["compact_owned_training_loop_owner"] == "lean_compact_trainer_step"
    for payload in (manifest, manifest["rows"][0], result["summary"], result["compact"]):
        assert payload["calls_train_muzero"] is False
        assert payload["touches_live_runs"] is False
        assert payload.get("promotion_claim", payload["non_claims"]["promotion_claim"]) is False


def test_speed_row_smoke_lean_trainer_step_runs_real_owned_loop():
    pytest.importorskip("torch")
    module = _load_smoke_module()
    args = _summary_args()
    args.search_service_kind = "device_target"
    args.compact_owned_lean_trainer_step = True
    args.compact_owned_loop_fused_learner_batch = True
    args.learner_num_unroll_steps = 2
    args.steps = 6
    args.sample_batch_size = 2
    args.sample_interval = 1
    args.replay_pair_capacity = 16
    args.learner_train_steps = 1
    args.learner_device = "cpu"
    args.num_simulations = 1
    args.seed = 20260530
    args.policy_refresh_interval = 1

    payload = module._run_local_compact_owned_lean_trainer_profile(
        args=args,
        loaded_checkpoint_identity={},
    )
    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity={},
    )

    assert payload["compact_owned_lean_trainer_step"] is True
    assert payload["compact_owned_training_loop_owner"] == "lean_compact_trainer_step"
    assert payload["compact_owned_trainer_record_step_calls"] > 0
    assert payload["compact_owned_trainer_loop_counter_source"] == (
        "run_hybrid_observation_profile"
    )
    assert payload["compact_rollout_slab_sample_gate_calls"] > 0
    assert payload["compact_rollout_slab_learner_gate_calls"] > 0
    assert payload["compact_rollout_slab_learner_gate_updates"] > 0
    assert (
        payload["compact_owned_trainer_sample_batch_count"]
        == (payload["compact_rollout_slab_sample_gate_calls"])
    )
    assert (
        payload["compact_owned_trainer_learner_update_count"]
        == (payload["compact_rollout_slab_learner_gate_updates"])
    )
    assert payload["compact_rollout_slab_learner_gate_real_muzero_update"] is True
    assert payload["compact_owned_loop_deferred_learner_pending"] is False
    assert (
        payload["compact_rollout_slab_sample_gate_last_telemetry"][
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"
        ]
        is True
    )
    assert (
        payload["compact_rollout_slab_learner_gate_last_telemetry"][
            "compact_rollout_slab_learner_gate_prebuilt_batch_used"
        ]
        is True
    )
    assert summary["compact_owned_training_loop_owner"] == "lean_compact_trainer_step"
    for row in (summary, compact):
        assert row["compact_owned_trainer_loop_counter_source"] == (
            "run_hybrid_observation_profile"
        )
        assert (
            row["compact_owned_trainer_record_step_calls"]
            == (payload["compact_owned_trainer_record_step_calls"])
        )
        assert (
            row["compact_owned_trainer_sample_batch_count"]
            == (row["compact_rollout_slab_sample_gate_calls"])
        )
        assert (
            row["compact_owned_trainer_learner_update_count"]
            == (row["compact_rollout_slab_learner_gate_updates"])
        )
    assert compact["real_compact_owned_training_work"] is True
    assert compact["compact_owned_lean_trainer_step"] is True


def test_speed_row_smoke_lean_trainer_step_supports_deferred_learner():
    pytest.importorskip("torch")
    module = _load_smoke_module()
    args = _summary_args()
    args.search_service_kind = "device_target"
    args.compact_owned_lean_trainer_step = True
    args.compact_owned_loop_deferred_learner = True
    args.compact_owned_loop_fused_learner_batch = True
    args.learner_num_unroll_steps = 2
    args.steps = 6
    args.sample_batch_size = 2
    args.sample_interval = 1
    args.replay_pair_capacity = 16
    args.learner_train_steps = 1
    args.learner_device = "cpu"
    args.num_simulations = 1
    args.seed = 20260601
    args.policy_refresh_interval = 1

    payload = module._run_local_compact_owned_lean_trainer_profile(
        args=args,
        loaded_checkpoint_identity={},
    )
    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity={},
    )

    assert payload["compact_owned_loop_defer_learner_gate"] is True
    assert payload["compact_owned_loop_deferred_learner_submit_count"] > 0
    assert (
        payload["compact_owned_loop_deferred_learner_completed_count"]
        == (payload["compact_owned_loop_deferred_learner_submit_count"])
    )
    assert payload["compact_owned_loop_deferred_learner_pending"] is False
    assert payload["compact_owned_loop_deferred_learner_pending_count"] == 0
    assert payload["compact_owned_loop_deferred_learner_max_pending_observed"] > 0
    assert (
        payload["compact_owned_trainer_learner_update_count"]
        == (payload["compact_rollout_slab_learner_gate_updates"])
    )
    assert payload["compact_owned_trainer_policy_version_ref"].endswith(
        f":update-{payload['compact_rollout_slab_learner_gate_updates']}"
    )
    for row in (summary, compact):
        assert row["compact_owned_loop_deferred_learner"] is True
        assert (
            row["compact_owned_loop_deferred_learner_submit_count"]
            == (payload["compact_owned_loop_deferred_learner_submit_count"])
        )
        assert (
            row["compact_owned_loop_deferred_learner_completed_count"]
            == (payload["compact_owned_loop_deferred_learner_completed_count"])
        )
        assert row["compact_owned_loop_deferred_learner_pending"] is False
        assert row["compact_owned_loop_deferred_learner_pending_count"] == 0
        assert (
            row["compact_owned_loop_deferred_learner_max_pending_observed"]
            == (payload["compact_owned_loop_deferred_learner_max_pending_observed"])
        )


def test_speed_row_smoke_lean_trainer_step_supports_deferred_sample_learner():
    pytest.importorskip("torch")
    module = _load_smoke_module()
    args = _summary_args()
    args.search_service_kind = "device_target"
    args.compact_owned_lean_trainer_step = True
    args.compact_owned_loop_deferred_learner = False
    args.compact_owned_loop_deferred_sample_learner = True
    args.compact_owned_loop_deferred_sample_learner_max_pending = 3
    args.compact_owned_loop_fused_learner_batch = True
    args.learner_num_unroll_steps = 2
    args.steps = 6
    args.sample_batch_size = 2
    args.sample_interval = 1
    args.replay_pair_capacity = 16
    args.learner_train_steps = 1
    args.learner_device = "cpu"
    args.num_simulations = 1
    args.seed = 20260601
    args.policy_refresh_interval = 1

    payload = module._run_local_compact_owned_lean_trainer_profile(
        args=args,
        loaded_checkpoint_identity={},
    )
    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity={},
    )

    assert payload["compact_owned_loop_defer_sample_learner_gate"] is True
    assert payload["compact_owned_loop_sample_learner_worker_kind"] == "in_process_thread"
    assert payload["compact_owned_loop_sample_learner_resource_distinct_from_actor_search"] is False
    assert payload["compact_owned_loop_deferred_sample_learner_submit_count"] > 0
    assert (
        payload["compact_owned_loop_deferred_sample_learner_completed_count"]
        == (payload["compact_owned_loop_deferred_sample_learner_submit_count"])
    )
    assert payload["compact_owned_loop_deferred_sample_learner_pending"] is False
    assert payload["compact_owned_loop_deferred_sample_learner_pending_count"] == 0
    assert payload["compact_owned_loop_deferred_sample_learner_max_pending"] == 3
    assert payload["compact_owned_loop_deferred_sample_learner_max_pending_observed"] > 0
    assert payload["compact_owned_loop_deferred_sample_learner_max_pending_observed"] <= 3
    assert payload["compact_owned_loop_deferred_sample_learner_drained"] is True
    assert payload["compact_owned_loop_final_deferred_drain_in_measured_sec"] is True
    assert payload["compact_owned_loop_final_deferred_drain_sec"] >= 0.0
    assert (
        payload["compact_rollout_slab_sample_gate_calls"]
        == (payload["compact_owned_loop_deferred_sample_learner_completed_count"])
    )
    assert payload["compact_rollout_slab_learner_gate_calls"] > 0
    assert (
        payload["compact_rollout_slab_learner_gate_calls"]
        <= (payload["compact_owned_loop_deferred_sample_learner_completed_count"])
    )
    assert (
        payload["compact_owned_trainer_sample_batch_count"]
        == (payload["compact_rollout_slab_sample_gate_calls"])
    )
    assert (
        payload["compact_owned_trainer_learner_update_count"]
        == (payload["compact_rollout_slab_learner_gate_updates"])
    )
    for row in (summary, compact):
        assert row["compact_owned_loop_deferred_learner"] is False
        assert row["compact_owned_loop_deferred_sample_learner"] is True
        assert row["compact_owned_loop_deferred_sample_learner_max_pending_requested"] == 3
        assert row["compact_owned_loop_sample_learner_worker_kind_requested"] == "in_process_thread"
        assert row["compact_owned_loop_sample_learner_worker_kind"] == "in_process_thread"
        assert row["compact_owned_loop_deferred_sample_learner_max_pending"] == 3
        assert (
            row["compact_owned_loop_deferred_sample_learner_submit_count"]
            == (payload["compact_owned_loop_deferred_sample_learner_submit_count"])
        )
        assert (
            row["compact_owned_loop_deferred_sample_learner_completed_count"]
            == (payload["compact_owned_loop_deferred_sample_learner_completed_count"])
        )
        assert row["compact_owned_loop_deferred_sample_learner_pending"] is False
        assert row["compact_owned_loop_deferred_sample_learner_pending_count"] == 0
        assert (
            row["compact_owned_loop_deferred_sample_learner_max_pending_observed"]
            == (payload["compact_owned_loop_deferred_sample_learner_max_pending_observed"])
        )
        assert row["compact_owned_loop_deferred_sample_learner_drained"] is True
        assert row["compact_owned_loop_final_deferred_drain_in_measured_sec"] is True


def test_speed_row_smoke_rejects_incomplete_deferred_sample_learner_proof():
    module = _load_smoke_module()
    args = _summary_args()
    args.compact_owned_loop_deferred_sample_learner = True
    payload = _profile_payload()
    payload.update(
        {
            "compact_owned_loop_defer_sample_learner_gate": True,
            "compact_owned_loop_sample_learner_worker_kind": "in_process_thread",
            "compact_owned_loop_deferred_sample_learner_submit_count": 2,
            "compact_owned_loop_deferred_sample_learner_completed_count": 1,
            "compact_owned_loop_deferred_sample_learner_pending": False,
            "compact_owned_loop_deferred_sample_learner_pending_count": 0,
            "compact_owned_loop_deferred_sample_learner_max_pending_observed": 1,
        }
    )

    with pytest.raises(ValueError, match="complete every submitted"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=payload,
            loaded_checkpoint_identity={},
        )


def test_speed_row_smoke_rejects_local_process_without_model_state_apply():
    module = _load_smoke_module()
    args = _summary_args()
    args.compact_owned_loop_deferred_sample_learner = True
    args.compact_owned_loop_deferred_sample_learner_max_pending = 2
    args.compact_owned_loop_sample_learner_worker_kind = "local_process"
    args.learner_device = "cuda"
    payload = _profile_payload()
    payload.update(
        {
            "compact_owned_loop_defer_sample_learner_gate": True,
            "compact_owned_loop_sample_learner_worker_kind": "local_process",
            "compact_owned_loop_sample_learner_worker_resource_scope": "process",
            "compact_owned_loop_sample_learner_worker_start_method": "spawn",
            "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings": (
                "expandable_segments:False"
            ),
            "compact_owned_loop_sample_learner_worker_bootstrap_source": "factory",
            "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": True,
            "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search": (
                False
            ),
            "compact_owned_loop_deferred_sample_learner_submit_count": 2,
            "compact_owned_loop_deferred_sample_learner_completed_count": 2,
            "compact_owned_loop_deferred_sample_learner_pending": False,
            "compact_owned_loop_deferred_sample_learner_pending_count": 0,
            "compact_owned_loop_deferred_sample_learner_max_pending": 2,
            "compact_owned_loop_deferred_sample_learner_max_pending_observed": 2,
            "compact_owned_loop_deferred_sample_learner_drained": True,
            "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending": 1,
            "compact_owned_loop_deferred_sample_learner_policy_lag_max": 2,
            "compact_owned_loop_deferred_sample_learner_last_submitted_request_id": 2,
            "compact_owned_loop_deferred_sample_learner_last_completed_request_id": 2,
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid": 1234,
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id": (
                "local_process:1234:sample-learner"
            ),
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device": (
                "cuda:0"
            ),
            (
                "compact_owned_loop_deferred_sample_learner_"
                "last_completed_worker_pid_distinct_from_actor_search"
            ): True,
            "compact_owned_loop_deferred_sample_learner_model_state_apply_count": 1,
            "compact_owned_loop_deferred_sample_learner_model_state_return_count": 2,
            "compact_owned_loop_deferred_sample_learner_model_state_omitted_count": 0,
            "compact_owned_loop_deferred_sample_learner_last_model_state_applied": True,
            "compact_owned_loop_final_deferred_drain_in_measured_sec": True,
            "compact_rollout_slab_sample_gate_calls": 2,
            "compact_rollout_slab_learner_gate_calls": 2,
            "compact_rollout_slab_learner_gate_updates": 2,
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": 2,
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_worker_state": {
                "schema_id": "curvyzero_compact_policy_refresh_search_worker_state/v1",
                "learner_update_count": 2,
                "refresh_count": 1,
                "model_state_digest": "c" * 64,
                "search_worker_model_object_id": 123,
            },
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata": {
                "compact_policy_refresh_search_worker_refreshed": True,
                "compact_policy_refresh_learner_update_count": 2,
                "compact_policy_refresh_model_state_digest": "c" * 64,
            },
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata": {
                "compact_policy_refresh_search_worker_refreshed": True,
                "compact_policy_refresh_learner_update_count": 2,
                "compact_policy_refresh_model_state_digest": "c" * 64,
            },
        }
    )
    _attach_lean_trainer_counters(payload)

    with pytest.raises(ValueError, match="apply every returned model state"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=payload,
            loaded_checkpoint_identity=_loaded_identity(),
        )


def test_speed_row_smoke_rejects_local_process_replay_observation_append_payload():
    module = _load_smoke_module()
    args = _summary_args()
    args.compact_owned_loop_deferred_sample_learner = True
    args.compact_owned_loop_deferred_sample_learner_max_pending = 2
    args.compact_owned_loop_sample_learner_worker_kind = "local_process"
    args.learner_device = "cuda"
    payload = _profile_payload()
    payload.update(
        {
            "compact_owned_loop_defer_sample_learner_gate": True,
            "compact_owned_loop_sample_learner_worker_kind": "local_process",
            "compact_owned_loop_sample_learner_worker_resource_scope": "process",
            "compact_owned_loop_sample_learner_worker_start_method": "spawn",
            "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings": (
                "expandable_segments:False"
            ),
            "compact_owned_loop_sample_learner_worker_bootstrap_source": "factory",
            "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": True,
            "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search": (
                False
            ),
            "compact_owned_loop_deferred_sample_learner_submit_count": 2,
            "compact_owned_loop_deferred_sample_learner_completed_count": 2,
            "compact_owned_loop_deferred_sample_learner_pending": False,
            "compact_owned_loop_deferred_sample_learner_pending_count": 0,
            "compact_owned_loop_deferred_sample_learner_max_pending": 2,
            "compact_owned_loop_deferred_sample_learner_max_pending_observed": 2,
            "compact_owned_loop_deferred_sample_learner_drained": True,
            "compact_owned_loop_final_deferred_drain_in_measured_sec": True,
            "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending": 1,
            "compact_owned_loop_deferred_sample_learner_policy_lag_max": 2,
            "compact_owned_loop_deferred_sample_learner_last_submitted_request_id": 2,
            "compact_owned_loop_deferred_sample_learner_last_completed_request_id": 2,
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid": 1234,
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device": (
                "cuda:0"
            ),
            (
                "compact_owned_loop_deferred_sample_learner_"
                "last_completed_worker_pid_distinct_from_actor_search"
            ): True,
            "compact_owned_loop_deferred_sample_learner_model_state_apply_count": 2,
            "compact_owned_loop_deferred_sample_learner_model_state_return_count": 2,
            "compact_owned_loop_deferred_sample_learner_model_state_omitted_count": 0,
            "compact_owned_loop_deferred_sample_learner_last_model_state_applied": True,
            "compact_owned_loop_deferred_sample_learner_request_host_only": True,
            "compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count": 0,
            "compact_owned_loop_deferred_sample_learner_result_host_only": True,
            "compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count": 0,
            "compact_owned_loop_deferred_sample_learner_request_bytes": 100,
            "compact_owned_loop_deferred_sample_learner_result_bytes": 100,
            "compact_owned_loop_deferred_sample_learner_worker_owns_model_state": True,
            "compact_owned_loop_deferred_sample_learner_worker_owns_replay_store": True,
            "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent": False,
            ("compact_owned_loop_deferred_sample_learner_full_replay_snapshot_submit_count"): 0,
            "compact_owned_loop_deferred_sample_learner_replay_append_entry_count": 2,
            "compact_owned_loop_deferred_sample_learner_replay_append_index_row_count": 4,
            "compact_owned_loop_deferred_sample_learner_replay_append_entry_bytes": 100,
            ("compact_owned_loop_deferred_sample_learner_replay_append_host_observation_bytes"): 16,
            ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_count"): 0,
            "compact_owned_loop_deferred_sample_learner_worker_replay_append_count": 2,
            "compact_owned_loop_deferred_sample_learner_worker_replay_index_row_count": 4,
            ("compact_owned_loop_deferred_sample_learner_worker_model_initialized_count"): 1,
            "compact_owned_loop_deferred_sample_learner_worker_completed_count": 2,
            "compact_rollout_slab_sample_gate_calls": 2,
            "compact_rollout_slab_learner_gate_calls": 2,
            "compact_rollout_slab_learner_gate_updates": 2,
        }
    )

    with pytest.raises(ValueError, match="still sends host replay observations"):
        module._require_deferred_sample_learner_proof(args, payload)


def test_speed_row_smoke_validates_local_process_owner_ref_model_transport():
    module = _load_smoke_module()
    args = _summary_args()
    args.compact_owned_loop_deferred_sample_learner = True
    args.compact_owned_loop_deferred_sample_learner_max_pending = 2
    args.compact_owned_loop_sample_learner_worker_kind = "local_process"
    args.learner_device = "cuda"
    payload = _valid_local_process_sample_learner_payload()
    payload.update(
        {
            (
                "compact_owned_loop_deferred_sample_learner_model_state_transport_kind"
            ): "owner_ref_v1",
            "compact_owned_loop_deferred_sample_learner_model_state_apply_count": 0,
            "compact_owned_loop_deferred_sample_learner_model_state_return_count": 0,
            ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_return_count"): 0,
            ("compact_owned_loop_deferred_sample_learner_model_owner_ref_return_count"): 2,
            ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_returned"): True,
            ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_digest"): "d" * 64,
        }
    )

    module._require_deferred_sample_learner_proof(args, payload)

    bad_payload = dict(payload)
    bad_payload["compact_owned_loop_deferred_sample_learner_model_state_return_count"] = 1
    with pytest.raises(ValueError, match="must not return or apply parent model state"):
        module._require_deferred_sample_learner_proof(args, bad_payload)

    bad_payload = dict(payload)
    bad_payload[("compact_owned_loop_deferred_sample_learner_model_owner_ref_return_count")] = 0
    with pytest.raises(ValueError, match="must return at least one owner ref"):
        module._require_deferred_sample_learner_proof(args, bad_payload)

    bad_payload = dict(payload)
    bad_payload[("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_digest")] = ""
    with pytest.raises(ValueError, match="must report final owner-ref digest"):
        module._require_deferred_sample_learner_proof(args, bad_payload)


def test_speed_row_smoke_projects_and_validates_owner_search_slab_proxy_fields():
    module = _load_smoke_module()
    args = _summary_args()
    args.search_service_kind = "owner_search_slab_proxy"
    payload = _profile_payload()
    payload.update(_owner_search_slab_proxy_fields())

    module._require_owner_search_slab_proxy_proof(payload)
    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity={},
    )

    for row in (summary, compact):
        assert row["compact_owner_search_slab_proxy"] is True
        assert row["compact_owner_search_boundary_kind"] == ("worker_search_parent_slab_commit")
        assert row["compact_owner_search_parent_slab_commits_replay"] is True
        assert row["compact_owner_search_worker_owns_search_state"] is True
        assert row["compact_owner_search_worker_owns_replay_state"] is True
        assert row["compact_owner_search_worker_owns_model_state"] is True
        assert row["compact_owner_search_owner_replay_append_enabled"] is True
        assert row["compact_owner_search_owner_loop_kind"] == ("persistent_priority_owner_loop_v1")
        assert row["compact_owner_search_owner_loop_persistent"] is True
        assert row["compact_owner_search_owner_action_priority_enabled"] is True
        assert row["compact_owner_search_replay_append_count"] == 2
        assert row["compact_owner_search_learner_update_count"] == 1
        assert row["compact_owner_search_request_bytes"] == 55
        assert row["compact_owner_search_result_bytes"] == 88
        assert row["compact_owner_search_fixed_action_result_buffer_requested"] is False
        assert row["compact_owner_search_fixed_action_result_buffer_used"] is False
        assert row["compact_owner_search_fixed_action_result_buffer_slot_count"] == 0
        assert row["compact_owner_search_fixed_action_result_buffer_wire_result_bytes"] == 0
        assert row["compact_owner_search_fixed_action_result_buffer_full_result_bytes"] == 0
        assert row["compact_owner_search_request_cuda_tensor_count"] == 0
        assert row["compact_owner_search_result_cuda_tensor_count"] == 0
        assert row["compact_owner_search_root_observation_bytes_sent"] == 0
        assert row["compact_owner_search_parent_reconstructed_search_result"] is True
        assert row["compact_owner_search_action_feedback_verified"] is True
        assert row["compact_owner_search_action_feedback_transition_count"] == 2
        assert row["compact_owner_search_action_feedback_action_count"] == 2
        assert row["compact_owner_search_action_feedback_mismatch_count"] == 0
        assert row["compact_owner_search_expected_joint_action_checksum"] == 11
        assert row["compact_owner_search_applied_joint_action_checksum"] == 11
        assert row["compact_owner_search_replay_action_checksum"] == 11
        assert row["compact_owner_search_model_state_bytes"] == 0
        assert row["compact_owner_search_model_state_return_count"] == 0
        assert row["compact_owner_search_model_state_snapshot_return_count"] == 0
        assert row["compact_owner_search_search_result_payload_bytes"] == 77
        assert (
            row["compact_owner_search_search_result_payload_transport_kind"]
            == "numpy_ndarray_ipc_v1"
        )
        assert row["compact_owner_search_search_result_payload_json_safe"] is False
        assert row["compact_owner_search_selected_action_bytes"] == 8
        assert row["compact_owner_search_visit_policy_bytes"] == 24
        assert row["compact_owner_search_root_value_bytes"] == 8
        assert row["compact_owner_search_parent_wait_sec"] == 0.01
        assert row["compact_owner_search_worker_wall_sec"] == 0.02
        assert row["compact_owner_search_owner_train_wall_sec"] == pytest.approx(0.011)
        assert row["compact_owner_search_owner_train_sample_sec"] == pytest.approx(0.001)
        assert row["compact_owner_search_owner_train_learner_update_sec"] == pytest.approx(0.002)
        assert row["compact_owner_search_owner_train_model_state_digest_sec"] == pytest.approx(
            0.003
        )
        assert row["compact_owner_search_owner_train_model_state_dict_sec"] == pytest.approx(0.004)
        assert row["compact_owner_search_owner_train_owner_ref_build_sec"] == pytest.approx(0.0005)
        assert row["compact_owner_search_owner_train_accounted_sec"] == pytest.approx(0.0105)
        assert row["compact_owner_search_owner_train_residual_sec"] == pytest.approx(0.0005)
        assert row["compact_owner_search_owner_train_timing_aggregate_count"] == 1
        assert row["compact_owner_search_root_slot_count"] == 2
        assert row["compact_owner_search_active_root_count"] == 2
        assert row["compact_owner_search_owner_maintenance_coalescing_kind"] == ""
        assert row["compact_owner_search_owner_maintenance_coalesced_skip_count"] == 0
        assert row["compact_owner_search_owner_async_learner_worker_enabled"] is False
        assert row["compact_owner_search_owner_async_learner_worker_kind"] == "none"
        assert row["compact_owner_search_owner_async_learner_submit_count"] == 0
        assert row["compact_whole_owner_buffer_replay_ceiling_enabled"] is True
        assert row["compact_whole_owner_buffer_replay_ceiling_projection_only"] is True
        assert row["compact_whole_owner_buffer_replay_ceiling_production_speed_claim"] is False
        assert row["compact_whole_owner_buffer_replay_ceiling_touches_live_training"] is False
        assert row["compact_whole_owner_buffer_replay_ceiling_requires_h100_validation"] is True
        assert row["compact_whole_owner_buffer_replay_ceiling_speed_currency"] == (
            "local_projection_no_speed"
        )
        assert row["compact_whole_owner_buffer_replay_ceiling_h100_validation_status"] == (
            "not_run"
        )
        assert (
            row["compact_whole_owner_buffer_replay_ceiling_variance_interpretation"]
            == "projection_not_measurement"
        )
        assert row["compact_whole_owner_buffer_replay_ceiling_promotion_eligible"] is False
        assert row["compact_whole_owner_buffer_replay_ceiling_observed_env_steps"] == pytest.approx(
            8.0
        )
        assert row["compact_whole_owner_buffer_replay_ceiling_observed_wall_sec"] == pytest.approx(
            0.5
        )
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_observed_env_steps_per_sec"
        ] == pytest.approx(16.0)
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_baseline_env_steps_per_sec"
        ] == pytest.approx(module.OPT104_BASELINE_ENV_STEPS_PER_SEC)
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_target_env_steps_per_sec"
        ] == pytest.approx(module.OPT104_BASELINE_ENV_STEPS_PER_SEC * 2.0)
        assert row["compact_whole_owner_buffer_replay_ceiling_target_wall_sec"] == pytest.approx(
            8.0 / (module.OPT104_BASELINE_ENV_STEPS_PER_SEC * 2.0)
        )
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_observed_replay_append_sec"
        ] == pytest.approx(0.004)
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_observed_owner_train_sample_sec"
        ] == pytest.approx(0.001)
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_observed_parent_wait_sec"
        ] == pytest.approx(0.01)
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_direct_replay_sample_surface_sec"
        ] == pytest.approx(0.005)
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_parent_wait_bounded_surface_sec"
        ] == pytest.approx(0.005)
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_preserved_search_update_floor_sec"
        ] == pytest.approx(0.017)
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_projected_removed_sec"
        ] == pytest.approx(0.005)
        assert row["compact_whole_owner_buffer_replay_ceiling_projected_wall_sec"] == pytest.approx(
            0.495
        )
        assert row[
            "compact_whole_owner_buffer_replay_ceiling_projected_env_steps_per_sec"
        ] == pytest.approx(8.0 / 0.495)
        assert row["compact_whole_owner_buffer_replay_ceiling_projected_reaches_2x"] is False
        assert row["compact_owner_search_owner_async_learner_completed_count"] == 0
        assert row["compact_owner_search_owner_async_learner_pending_count"] == 0
        assert row["compact_owner_search_resident_root_bridge_final_storage"] == "sparse_rows"
        assert row["compact_owner_search_resident_root_bridge_final_sparse_row_count"] == 1
        assert row["compact_rollout_slab_committed_index_row_count"] == 8
        assert row["compact_rollout_slab_stored_index_row_count"] == 8

    deferred_payload = dict(payload)
    deferred_payload.update(
        {
            "compact_owner_search_owner_defer_maintenance": True,
            "compact_owner_search_owner_maintenance_drain_request_count": 2,
            "compact_owner_search_owner_maintenance_request_count": 2,
            "compact_owner_search_owner_maintenance_staged_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_count": 2,
            "compact_owner_search_owner_maintenance_drained_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_count": 2,
            "compact_owner_search_owner_maintenance_pending_work_count": 0,
            "compact_owner_search_owner_maintenance_inflight": False,
            "compact_owner_search_owner_maintenance_final_drain_in_measured_sec": True,
            "compact_owner_search_owner_maintenance_coalescing_kind": (
                "eager_append_or_train_boundary_v1"
            ),
            "compact_owner_search_owner_maintenance_coalesced_skip_count": 1,
            "compact_owner_search_owner_maintenance_eager_append_drain_count": 1,
            "owner_search_async_learner_worker_requested": True,
            "owner_search_async_learner_worker_kind_requested": ("in_process_thread_v1"),
            "owner_search_async_learner_max_pending_requested": 1,
            "compact_owner_search_owner_async_learner_worker_enabled": True,
            "compact_owner_search_owner_async_learner_worker_kind": ("in_process_thread_v1"),
            "compact_owner_search_owner_async_learner_max_pending": 1,
            "compact_owner_search_owner_async_learner_submit_count": 1,
            "compact_owner_search_owner_async_learner_completed_count": 1,
            "compact_owner_search_owner_async_learner_pending_count": 0,
            "compact_owner_search_owner_async_learner_max_pending_observed": 1,
            "compact_owner_search_owner_action_while_async_learner_pending_count": 1,
            "compact_owner_search_owner_async_learner_failed": False,
            "compact_owner_search_owner_policy_lag_current": 0,
            "compact_owner_search_owner_policy_lag_max": 1,
            "compact_owner_search_owner_action_while_maintenance_pending_count": 1,
            "compact_owner_search_owner_action_while_policy_lagged_count": 1,
            "compact_owner_search_owner_action_served_before_maintenance_count": 1,
            "compact_owner_search_owner_maintenance_failed": False,
        }
    )
    module._require_owner_search_slab_proxy_proof(deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_maintenance_final_drain_in_measured_sec"] = (
        False
    )
    with pytest.raises(ValueError, match="final drain in wall time"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_maintenance_final_drain_sec"] = float("nan")
    with pytest.raises(ValueError, match="finite nonnegative final drain"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_maintenance_pending_work_count"] = 1
    with pytest.raises(ValueError, match="no pending work"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_loop_kind"] = "fifo_process_pool_v1"
    with pytest.raises(ValueError, match="persistent priority owner loop"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_action_while_maintenance_pending_count"] = 0
    with pytest.raises(ValueError, match="action while maintenance pending"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_action_served_before_maintenance_count"] = 0
    with pytest.raises(ValueError, match="action served before maintenance"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_policy_lag_max"] = 0
    with pytest.raises(ValueError, match="positive policy lag"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_action_while_policy_lagged_count"] = 0
    with pytest.raises(ValueError, match="action while policy lagged"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_submitted_learner_update_count"] = 0
    with pytest.raises(ValueError, match="submitted learner updates"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_async_learner_completed_count"] = 0
    with pytest.raises(ValueError, match="completed count must match submitted"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_async_learner_pending_count"] = 1
    with pytest.raises(ValueError, match="no pending learner jobs"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_deferred_payload = dict(deferred_payload)
    bad_deferred_payload["compact_owner_search_owner_action_while_async_learner_pending_count"] = 0
    with pytest.raises(ValueError, match="action while learner pending"):
        module._require_owner_search_slab_proxy_proof(bad_deferred_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_root_observation_bytes_sent"] = 1
    with pytest.raises(ValueError, match="zero root-observation bytes"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_request_cuda_tensor_count"] = 1
    with pytest.raises(ValueError, match="request must contain no CUDA tensors"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_selected_action_bytes"] = 0
    with pytest.raises(ValueError, match="selected_action_bytes"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload.pop("compact_owner_search_parent_wait_sec")
    with pytest.raises(ValueError, match="parent_wait_sec"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    action_only_payload = dict(deferred_payload)
    action_only_payload.update(
        {
            "compact_owner_search_action_only_result": True,
            "compact_owner_search_owner_materializes_replay": True,
            "compact_owner_search_parent_slab_commits_replay": False,
            "compact_owner_search_parent_reconstructed_search_result": False,
            "compact_owner_search_search_result_payload_bytes": 0,
            "compact_owner_search_search_result_payload_transport_kind": (
                "action_only_owner_cached_replay_v1"
            ),
            "compact_owner_search_search_result_payload_json_safe": True,
            "compact_owner_search_visit_policy_bytes": 0,
            "compact_owner_search_root_value_bytes": 0,
            "compact_owner_search_optional_array_bytes": 0,
            "compact_rollout_slab_committed_index_row_count": 0,
            "compact_rollout_slab_stored_index_row_count": 0,
        }
    )
    module._require_owner_search_slab_proxy_proof(action_only_payload)
    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=action_only_payload,
        loaded_checkpoint_identity={},
    )
    for row in (summary, compact):
        assert row["compact_owner_search_action_only_result"] is True
        assert row["compact_owner_search_owner_materializes_replay"] is True
        assert row["compact_owner_search_parent_slab_commits_replay"] is False
        assert row["compact_owner_search_parent_reconstructed_search_result"] is False
        assert row["compact_owner_search_search_result_payload_bytes"] == 0
        assert row["compact_owner_search_visit_policy_bytes"] == 0
        assert row["compact_owner_search_root_value_bytes"] == 0
        assert (
            row["compact_owner_search_search_result_payload_transport_kind"]
            == "action_only_owner_cached_replay_v1"
        )
        assert row["compact_owner_search_owner_maintenance_final_drain_in_measured_sec"] is True
        assert row["compact_owner_search_owner_maintenance_staged_work_item_count"] == 2
        assert row["compact_owner_search_owner_maintenance_drained_work_item_count"] == 2
        assert row["compact_owner_search_owner_maintenance_drained_replay_append_entry_count"] == 2
        assert row["compact_owner_search_owner_maintenance_drained_replay_append_count"] == 2
        assert row["compact_rollout_slab_committed_index_row_count"] == 0
        assert row["compact_rollout_slab_stored_index_row_count"] == 0

    bypass_args = SimpleNamespace(owner_search_slab_bypass=True)
    bypass_payload = dict(action_only_payload)
    bypass_payload.update(
        {
            "compact_owner_search_slab_bypass": True,
            "compact_owner_search_slab_bypass_kind": (
                module.COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
            ),
            "compact_rollout_slab_bypassed": True,
            "compact_rollout_slab_general_replay_row_builder_used": False,
            "compact_rollout_slab_retains_committed_index_rows": False,
            "compact_owner_search_slab_bypass_parent_committed_index_rows": 0,
            "compact_owner_search_slab_bypass_parent_stored_index_rows": 0,
        }
    )
    module._require_owner_search_slab_proxy_proof(
        bypass_payload,
        args=bypass_args,
    )

    batch_args = SimpleNamespace(
        owner_search_slab_bypass=True,
        owner_search_transition_batch_size=4,
    )
    batch_payload = dict(bypass_payload)
    batch_payload.update(
        {
            "owner_search_transition_batch_transport_requested": True,
            "compact_owner_search_transition_batch_transport_requested": True,
            "compact_owner_search_transition_batch_transport_enabled": True,
            "compact_owner_search_transition_batch_transport_kind": (
                module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
            ),
            "compact_owner_search_transition_batch_schema_id": (
                module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
            ),
            "compact_owner_search_transition_batch_count": 2,
            "compact_owner_search_transition_batch_entry_count": 8,
            "compact_owner_search_transition_batch_transport_entry_count": 2,
            "compact_owner_search_transition_batch_max_entries_per_batch": 4,
            "compact_owner_search_transition_batch_fixed_capacity": 4,
            "compact_owner_search_transition_batch_padding_count": 0,
            "compact_owner_search_transition_batch_overflow_count": 0,
            "compact_owner_search_transition_batch_fallback_count": 0,
            "compact_owner_search_transition_batch_fallback_reason": "none",
            "compact_owner_search_transition_batch_pending_count": 0,
            "compact_owner_search_transition_batch_transport_bytes": 128,
            "compact_owner_search_transition_batch_digest": "digest",
            "compact_owner_search_transition_batch_digest_verified": True,
            "compact_owner_search_transition_batch_build_sec": 0.001,
            "compact_owner_search_transition_batch_submit_sec": 0.001,
            "compact_owner_search_owner_replay_transition_legacy_entry_count": 0,
            "compact_owner_search_replay_append_transition_batch_count": 2,
            "compact_owner_search_replay_append_transition_batch_entry_count": 8,
            "compact_owner_search_replay_append_transport_entry_count": 2,
            "compact_owner_search_owner_replay_transition_batch_count": 2,
            "compact_owner_search_owner_replay_transition_batch_transition_count": 8,
            "compact_owner_search_owner_replay_append_request_count": 2,
            "compact_owner_search_owner_replay_append_staged_entry_count": 8,
            "compact_owner_search_owner_replay_append_submitted_entry_count": 8,
            "compact_owner_search_owner_replay_append_staged_transport_entry_count": 2,
            "compact_owner_search_owner_replay_append_submitted_transport_entry_count": 2,
            "compact_owner_search_replay_append_entry_count": 8,
            "compact_owner_search_replay_append_count": 8,
            "compact_owner_search_owner_replay_append_count": 8,
            "compact_owner_search_learner_update_count": 4,
            "compact_owner_search_owner_train_request_count": 2,
            "compact_owner_search_owner_expected_train_request_count": 2,
            "compact_owner_search_owner_model_refresh_request_count": 2,
            "compact_owner_search_owner_model_refresh_skipped_count": 0,
            "compact_owner_search_owner_submitted_learner_update_count": 4,
            "compact_owner_search_owner_learner_update_count": 4,
            "compact_owner_search_search_refresh_update_count": 4,
            "compact_owner_search_owner_train_timing_aggregate_count": 2,
            "compact_owner_search_owner_async_learner_submit_count": 2,
            "compact_owner_search_owner_async_learner_completed_count": 2,
            "compact_owner_search_action_feedback_transition_count": 8,
            "compact_owner_search_owner_maintenance_staged_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": 8,
            "compact_owner_search_owner_maintenance_drained_replay_append_count": 8,
        }
    )
    batch_payload["compact_owner_search_owner_learner_telemetry"] = {
        **dict(batch_payload["compact_owner_search_owner_learner_telemetry"]),
        "compact_owner_search_owner_train_timing_aggregate_count": 2,
    }
    module._require_owner_search_slab_proxy_proof(batch_payload, args=batch_args)

    direct_batch_args = SimpleNamespace(
        owner_search_slab_bypass=True,
        owner_search_transition_batch_size=4,
        owner_search_direct_transition_batch_replay=True,
        owner_search_require_resident_root_view=True,
        owner_search_resident_root_host_observation_stub=True,
        owner_search_direct_root_build_request=True,
        compact_owner_action_step_boundary=True,
        owner_search_fixed_action_result_buffer=True,
        owner_search_action_result_slot_capacity=8,
    )
    direct_batch_payload = dict(batch_payload)
    direct_batch_payload.update(
        {
            "compact_owner_search_direct_transition_batch_replay_requested": True,
            "compact_owner_search_direct_transition_batch_replay_used": True,
            "compact_owner_search_direct_transition_batch_replay_batch_count": 2,
            "compact_owner_search_direct_transition_batch_replay_transition_count": 8,
            "compact_owner_search_direct_transition_batch_replay_transport_entry_count": 2,
            "compact_owner_search_direct_transition_batch_replay_legacy_expanded_entry_count": 0,
            "compact_owner_search_direct_transition_batch_replay_index_entry_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_ring_entry_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_columnar_append_used": True,
            "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count": 8,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_requested": False,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_used": False,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_entry_view_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_step_view_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_learner_ready_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_entry_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_concat_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_reason": "none",
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_sec": 0.0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_successor_index_sec": 0.0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_total_sec": 0.0,
            "compact_owner_search_direct_transition_batch_replay_columnar_record_count": 8,
            "compact_owner_search_direct_transition_batch_replay_columnar_entry_view_object_count": 8,
            "compact_owner_search_direct_transition_batch_replay_columnar_step_view_object_count": 16,
            "compact_owner_search_direct_transition_batch_replay_fallback_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fallback_reason": "none",
            "compact_owner_search_direct_transition_batch_replay_last_append_sec": 0.002,
            "compact_owner_search_direct_transition_batch_replay_append_sec": 0.004,
            "compact_owner_search_direct_transition_batch_replay_accounted_sec": 0.0021,
            "compact_owner_search_direct_transition_batch_replay_array_extract_sec": 0.0001,
            "compact_owner_search_direct_transition_batch_replay_transition_validate_sec": 0.0002,
            "compact_owner_search_direct_transition_batch_replay_device_payload_sec": 0.0003,
            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flushed_count": 0,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_count": 0,
            "compact_owner_search_direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count": 0,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls": 0.0,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count": 0,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count": 0,
            "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_count_max": 0,
            "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_final_count": 0,
            "compact_owner_search_direct_transition_batch_replay_replay_payload_d2h_bytes": 0.0,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_flush_sec": 0.0,
            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flush_sec": 0.0,
            "compact_owner_search_direct_transition_batch_replay_index_rows_build_sec": 0.0004,
            "compact_owner_search_direct_transition_batch_replay_step_object_build_sec": 0.0005,
            "compact_owner_search_direct_transition_batch_replay_ring_append_sec": 0.0006,
            "compact_owner_search_direct_transition_batch_replay_columnar_prepare_sec": 0.00001,
            "compact_owner_search_direct_transition_batch_replay_columnar_register_sec": 0.00002,
            "compact_owner_search_direct_transition_batch_replay_columnar_append_store_sec": 0.00003,
            "compact_owner_search_direct_transition_batch_replay_columnar_retain_sec": 0.00004,
            "compact_owner_search_direct_transition_batch_replay_columnar_evict_sec": 0.0,
            "compact_owner_search_direct_transition_batch_replay_columnar_evict_release_sec": 0.0,
            "compact_owner_search_direct_transition_batch_replay_columnar_candidate_indices_sec": 0.00005,
            "compact_owner_search_direct_transition_batch_replay_columnar_cache_refresh_sec": 0.00006,
            "compact_owner_search_direct_transition_batch_replay_columnar_cache_rebuild_sec": 0.0,
            "compact_owner_search_direct_transition_batch_replay_columnar_total_sec": 0.00021,
        }
    )
    module._require_owner_search_slab_proxy_proof(
        direct_batch_payload,
        args=direct_batch_args,
    )
    derived_direct_args = SimpleNamespace(
        owner_search_slab_bypass=True,
        owner_search_transition_batch_size=4,
        owner_search_direct_transition_batch_replay=True,
        owner_search_owner_local_transition_derivation=True,
    )
    derived_direct_payload = dict(direct_batch_payload)
    derived_direct_payload.update(
        {
            # Remote owner-local rows can keep stale pre-drain generic transition-batch
            # fields at the top level; the owner-local proof fields are the authority.
            "compact_owner_search_transition_batch_transport_kind": (
                module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
            ),
            "compact_owner_search_transition_batch_schema_id": (
                module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
            ),
            "compact_owner_search_owner_local_transition_derivation_requested": True,
            "compact_owner_search_owner_local_transition_derivation_used": True,
            "compact_owner_search_owner_local_transition_derivation_schema_id": (
                module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
            ),
            "compact_owner_search_owner_local_transition_derivation_kind": (
                module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
            ),
            "compact_owner_search_owner_local_transition_derivation_batch_count": 2,
            "compact_owner_search_owner_local_transition_derivation_transition_count": 8,
            "compact_owner_search_owner_local_transition_derivation_transport_entry_count": 2,
            "compact_owner_search_owner_local_transition_derivation_pending_count": 0,
            "compact_owner_search_owner_local_transition_derivation_transport_bytes": 64,
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes": 0,
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count": 0,
            "compact_owner_search_owner_local_transition_derivation_digest": ("derived-digest"),
            "compact_owner_search_owner_local_transition_derivation_digest_verified": True,
            "compact_owner_search_owner_local_transition_derivation_build_sec": 0.0001,
            "compact_owner_search_owner_local_transition_derivation_submit_sec": 0.0001,
            "compact_owner_search_owner_local_transition_derivation_cache_hit_count": 8,
            "compact_owner_search_owner_local_transition_derivation_cache_miss_count": 0,
            "compact_owner_search_owner_local_transition_derivation_action_checksum_verified_count": 8,
            "compact_owner_search_owner_local_transition_derivation_action_checksum_mismatch_count": 0,
            "compact_owner_search_owner_local_transition_derivation_fallback_count": 0,
            "compact_owner_search_owner_local_transition_derivation_fallback_reason": "none",
            "compact_owner_search_owner_local_transition_derivation_dropped_pending_count": 0,
        }
    )
    module._require_owner_search_slab_proxy_proof(
        derived_direct_payload,
        args=derived_direct_args,
    )
    deferred_direct_args = SimpleNamespace(
        owner_search_slab_bypass=True,
        owner_search_transition_batch_size=4,
        owner_search_direct_transition_batch_replay=True,
        compact_torch_defer_one_simulation_replay_payload=True,
    )
    deferred_direct_payload = dict(direct_batch_payload)
    deferred_direct_payload.update(
        {
            "compact_torch_defer_one_simulation_replay_payload_requested": True,
            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flushed_count": 8,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_count": 8,
            "compact_owner_search_direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count": 8,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls": 8.0,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count": 8,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count": 0,
            "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_count_max": 8,
            "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_final_count": 4,
            "compact_owner_search_inner_pending_deferred_replay_payload_final_count": 0,
            "compact_owner_search_direct_transition_batch_replay_replay_payload_d2h_bytes": 0.0,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_flush_sec": 0.004,
            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flush_sec": 0.004,
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_action_model_state_digest": "digest-a",
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_model_state_digest": "digest-a",
        }
    )
    module._require_owner_search_slab_proxy_proof(
        deferred_direct_payload,
        args=deferred_direct_args,
    )
    bad_deferred_direct_payload = dict(deferred_direct_payload)
    bad_deferred_direct_payload[
        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count"
    ] = 1
    with pytest.raises(ValueError, match="model-refresh-crossed count zero"):
        module._require_owner_search_slab_proxy_proof(
            bad_deferred_direct_payload,
            args=deferred_direct_args,
        )
    bad_deferred_direct_payload = dict(deferred_direct_payload)
    bad_deferred_direct_payload[
        "compact_owner_search_inner_pending_deferred_replay_payload_final_count"
    ] = 1
    with pytest.raises(ValueError, match="final pending count must be zero"):
        module._require_owner_search_slab_proxy_proof(
            bad_deferred_direct_payload,
            args=deferred_direct_args,
        )
    fixed_soa_batch_args = SimpleNamespace(
        owner_search_slab_bypass=True,
        owner_search_transition_batch_size=4,
        owner_search_direct_transition_batch_replay=True,
        owner_search_fixed_soa_replay=True,
    )
    fixed_soa_payload = dict(direct_batch_payload)
    fixed_soa_payload.update(
        {
            "compact_owner_search_direct_transition_batch_replay_columnar_append_used": False,
            "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count": 0,
            "compact_owner_search_direct_transition_batch_replay_columnar_record_count": 0,
            "compact_owner_search_direct_transition_batch_replay_columnar_entry_view_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_columnar_step_view_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_requested": True,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_used": True,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_count": 8,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_entry_view_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_step_view_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_learner_ready_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_entry_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_concat_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_reason": "none",
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_sec": 0.00007,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_successor_index_sec": 0.00008,
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_total_sec": 0.00015,
        }
    )
    fixed_soa_payload["compact_owner_search_owner_learner_telemetry"] = {
        **dict(fixed_soa_payload["compact_owner_search_owner_learner_telemetry"]),
        "compact_replay_fixed_soa_learner_batch_handle_ring_schema_id": (
            "curvyzero_compact_replay_fixed_soa_learner_batch_handle/v1"
        ),
        "compact_replay_fixed_soa_learner_batch_handle_ring_requested": True,
        "compact_replay_fixed_soa_learner_batch_handle_ring_used": True,
        "compact_replay_fixed_soa_learner_batch_handle_ring_handle_id": 7,
        "compact_replay_fixed_soa_learner_batch_handle_ring_snapshot_version": 3,
        "compact_replay_fixed_soa_learner_batch_handle_ring_request_checksum": 12345,
        "compact_replay_fixed_soa_learner_batch_handle_ring_sample_row_count": 2,
        "compact_replay_fixed_soa_learner_batch_handle_ring_target_row_count": 2,
        "compact_replay_fixed_soa_learner_batch_handle_ring_fallback_count": 0,
        "compact_replay_fixed_soa_learner_batch_handle_ring_fallback_reason": "none",
    }
    module._require_owner_search_slab_proxy_proof(
        fixed_soa_payload,
        args=fixed_soa_batch_args,
    )
    fixed_soa_projected = module._owner_search_slab_proxy_proof_fields(fixed_soa_payload)
    assert (
        fixed_soa_projected["compact_owner_search_learner_resident_batch_handle_requested"]
        is True
    )
    assert (
        fixed_soa_projected["compact_owner_search_learner_resident_batch_handle_consumed"]
        is True
    )
    assert (
        fixed_soa_projected["compact_owned_loop_learner_resident_batch_handle_consumed"]
        is True
    )
    assert (
        fixed_soa_projected["compact_owner_search_learner_resident_batch_handle_fallback_count"]
        == 0
    )
    assert (
        fixed_soa_projected[
            "compact_owner_search_learner_resident_batch_handle_"
            "materialized_parent_fallback_count"
        ]
        == 0
    )
    bad_fixed_soa_payload = dict(fixed_soa_payload)
    bad_fixed_soa_payload["compact_owner_search_owner_learner_telemetry"] = {
        **dict(fixed_soa_payload["compact_owner_search_owner_learner_telemetry"]),
        "compact_replay_fixed_soa_learner_batch_handle_ring_used": False,
    }
    with pytest.raises(ValueError, match="consume resident learner-batch handle"):
        module._require_owner_search_slab_proxy_proof(
            bad_fixed_soa_payload,
            args=fixed_soa_batch_args,
        )
    bad_fixed_soa_payload = dict(fixed_soa_payload)
    bad_fixed_soa_payload["compact_owner_search_owner_learner_telemetry"] = {
        **dict(fixed_soa_payload["compact_owner_search_owner_learner_telemetry"]),
        "compact_replay_fixed_soa_learner_batch_handle_ring_fallback_count": 1,
        "compact_replay_fixed_soa_learner_batch_handle_ring_fallback_reason": (
            "handle_resolve_failed"
        ),
    }
    with pytest.raises(ValueError, match="fallback count must be zero"):
        module._require_owner_search_slab_proxy_proof(
            bad_fixed_soa_payload,
            args=fixed_soa_batch_args,
        )
    bad_fixed_soa_payload = dict(fixed_soa_payload)
    bad_fixed_soa_payload["compact_owner_search_direct_transition_batch_replay_fixed_soa_used"] = (
        False
    )
    with pytest.raises(ValueError, match="fixed SoA must be used"):
        module._require_owner_search_slab_proxy_proof(
            bad_fixed_soa_payload,
            args=fixed_soa_batch_args,
        )
    bad_fixed_soa_payload = dict(fixed_soa_payload)
    bad_fixed_soa_payload[
        "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_concat_count"
    ] = 1
    with pytest.raises(ValueError, match="fixed SoA .*table_concat_count zero"):
        module._require_owner_search_slab_proxy_proof(
            bad_fixed_soa_payload,
            args=fixed_soa_batch_args,
        )
    h100_like_direct_payload = dict(direct_batch_payload)
    h100_like_direct_payload.update(
        {
            "compact_owner_search_owner_sample_batch_size": 512,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": False,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": False,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "learner_ready_unroll2_cache_used"
            ): False,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "learner_ready_unroll2_cache_impl"
            ): "none",
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
            ): False,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"
            ): "none",
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_source"
            ): "none",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "none",
        }
    )
    h100_like_direct_payload["compact_owner_search_owner_sample_telemetry"] = {
        key: value
        for key, value in direct_batch_payload.items()
        if key.startswith("compact_owner_search_direct_transition_batch_replay_")
    }
    h100_like_direct_payload["compact_owner_search_owner_learner_telemetry"] = {
        **dict(h100_like_direct_payload["compact_owner_search_owner_learner_telemetry"]),
        "sample_row_count": 512,
        "compact_muzero_learner_sample_rows": 512,
        "terminal_sample_row_count": 512,
        "terminal_unroll_value_target_row_count": 512,
        "terminal_unroll_value_target_mode": ("stock_terminal_no_bootstrap_return_discount_1.0"),
        "next_final_observation_row_count": 512,
        "resident_terminal_final_observation_used": True,
        "terminal_final_observation_group_count": 1,
        "terminal_final_observation_fallback_count": 1,
        "terminal_final_observation_validate_only_count": 1,
        "terminal_final_observation_final_row_count_sum": 4,
        "terminal_final_observation_final_row_count_max": 4,
        "terminal_final_observation_sparse_storage_count": 1,
        "terminal_final_observation_missing_storage_count": 0,
        "terminal_final_observation_sparse_row_count_sum": 2,
        "terminal_final_observation_sparse_row_count_max": 2,
        "require_next_targets": True,
        "explicit_unroll_targets": True,
        "explicit_unroll_target_group_count": 1,
        "num_unroll_steps": 2,
        "terminal_unroll_windows_supported": True,
        "sample_schema_id": module.COMPACT_MUZERO_DIRECT_LEARNER_BATCH_SCHEMA_ID,
        "compact_muzero_learner_batch_schema_id": (module.COMPACT_MUZERO_LEARNER_BATCH_SCHEMA_ID),
        "compact_muzero_learner_batch_rows": 512,
        "compact_muzero_learner_batch_prevalidation_source": (
            "tensor_native_replay_unroll2_table_gather_v1"
        ),
        "compact_muzero_learner_batch_sample_order": "rng",
        "compact_muzero_learner_prebuilt_batch_used": True,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache_requested": True,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache_available_group_count": 352,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache_eligible_count": 1,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache_used": True,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache_call_count": 1,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache_fallback_count": 0,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache_fallback_reason": "none",
        "compact_muzero_learner_batch_learner_ready_unroll2_cache_impl": (
            "learner_ready_unroll2_cache_v1"
        ),
        "compact_muzero_learner_batch_tensor_native_replay_requested": True,
        "compact_muzero_learner_batch_tensor_native_replay_used": True,
        "compact_muzero_learner_batch_tensor_native_replay_call_count": 1,
        "compact_muzero_learner_batch_tensor_native_replay_fallback_count": 0,
        "compact_muzero_learner_batch_tensor_native_replay_fallback_reason": "none",
        "compact_muzero_learner_batch_tensor_native_replay_impl": (
            "maintained_unroll2_table_gather_v1"
        ),
        "compact_muzero_learner_batch_tensor_native_replay_table_source": (
            "maintained_record_table_v1"
        ),
        "compact_muzero_learner_batch_tensor_native_replay_table_reused_record_count": 719,
        "compact_muzero_learner_batch_tensor_native_replay_table_missing_record_count": 0,
        "compact_muzero_learner_batch_tensor_native_replay_table_rows": 9770,
        "compact_muzero_learner_batch_unroll_builder_path": ("learner_ready_unroll2_cache"),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested": True,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible": True,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used": True,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count": 0,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason": ("none"),
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count": 0,
    }
    module._require_owner_search_slab_proxy_proof(
        h100_like_direct_payload,
        args=direct_batch_args,
    )
    normalized = module._owner_search_slab_proxy_proof_fields(h100_like_direct_payload)[
        "compact_owner_search_owner_sample_telemetry"
    ]
    assert normalized["compact_rollout_slab_sample_gate_sample_row_count"] == 512
    assert normalized["compact_rollout_slab_sample_gate_target_row_count"] == 512
    assert normalized["compact_rollout_slab_sample_gate_requested_sample_row_count"] == 512
    assert normalized["compact_rollout_slab_sample_gate_require_next_targets"] is True
    assert normalized["compact_rollout_slab_sample_gate_terminal_sample_row_count"] == 512
    assert (
        normalized["compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count"] == 512
    )
    assert (
        normalized["compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode"]
        == "stock_terminal_no_bootstrap_return_discount_1.0"
    )
    assert (
        normalized["compact_rollout_slab_sample_gate_resident_terminal_final_observation_used"]
        is True
    )
    assert normalized["compact_rollout_slab_sample_gate_compact_muzero_learner_batch"] is True
    assert normalized["compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"] is True
    assert normalized["compact_owner_search_direct_transition_batch_replay_transition_count"] == 8
    fused_args = SimpleNamespace(
        compact_owned_loop_fused_learner_batch=True,
        compact_muzero_learner_batch_learner_ready_unroll2_cache=True,
        compact_muzero_learner_batch_tensor_native_replay=True,
        compact_muzero_learner_batch_unroll2_specialized_builder=True,
        learner_num_unroll_steps=2,
    )
    module._require_fused_learner_batch_proof(fused_args, h100_like_direct_payload)
    fusion_fields = module._sample_learner_fusion_fields(h100_like_direct_payload)
    assert (
        fusion_fields[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
        ]
        is True
    )
    assert (
        fusion_fields[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"
        ]
        == "maintained_unroll2_table_gather_v1"
    )
    assert (
        fusion_fields["compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path"]
        == "learner_ready_unroll2_cache"
    )
    h100_like_fixed_soa_payload = dict(h100_like_direct_payload)
    h100_like_fixed_soa_payload.update(
        {
            key: value
            for key, value in fixed_soa_payload.items()
            if key.startswith("compact_owner_search_direct_transition_batch_replay_")
        }
    )
    h100_like_fixed_soa_payload["compact_owner_search_owner_sample_telemetry"] = {
        key: value
        for key, value in fixed_soa_payload.items()
        if key.startswith("compact_owner_search_direct_transition_batch_replay_")
    }
    fixed_soa_learner_telemetry = dict(
        h100_like_fixed_soa_payload["compact_owner_search_owner_learner_telemetry"]
    )
    fixed_soa_learner_telemetry.update(
        {
            "compact_muzero_learner_batch_prevalidation_source": ("fixed_soa_direct_gather_v1"),
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_requested": True,
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_available_group_count": 0,
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_eligible_count": 0,
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_used": False,
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_call_count": 0,
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_fallback_count": 0,
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_fallback_reason": "none",
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_impl": (
                "fixed_soa_columns_v1"
            ),
            "compact_muzero_learner_batch_tensor_native_replay_requested": True,
            "compact_muzero_learner_batch_tensor_native_replay_used": True,
            "compact_muzero_learner_batch_tensor_native_replay_call_count": 1,
            "compact_muzero_learner_batch_tensor_native_replay_fallback_count": 0,
            "compact_muzero_learner_batch_tensor_native_replay_fallback_reason": "none",
            "compact_muzero_learner_batch_tensor_native_replay_impl": (
                "fixed_soa_direct_gather_v1"
            ),
            "compact_muzero_learner_batch_tensor_native_replay_table_source": (
                "fixed_soa_columns_v1"
            ),
            "compact_muzero_learner_batch_tensor_native_replay_table_reused_record_count": 352,
            "compact_muzero_learner_batch_tensor_native_replay_table_missing_record_count": 0,
            "compact_muzero_learner_batch_tensor_native_replay_table_rows": 9770,
            "compact_muzero_learner_batch_fixed_soa_record_count": 2155,
            "compact_muzero_learner_batch_fixed_soa_selected_record_count": 62,
            "fixed_soa_locality_sample_group_size": 8,
            "fixed_soa_locality_sample_used": True,
            "fixed_soa_locality_sample_semantic_drift": True,
            "fixed_soa_locality_selected_group_count": 64,
            "fixed_soa_locality_duplicate_group_count": 2,
            "fixed_soa_locality_local_replace_group_count": 0,
            "compact_muzero_learner_batch_unroll_builder_path": ("fixed_soa_direct_gather"),
        }
    )
    h100_like_fixed_soa_payload["compact_owner_search_owner_learner_telemetry"] = (
        fixed_soa_learner_telemetry
    )
    module._require_fused_learner_batch_proof(
        fused_args,
        h100_like_fixed_soa_payload,
    )
    fixed_fusion_fields = module._sample_learner_fusion_fields(h100_like_fixed_soa_payload)
    assert (
        fixed_fusion_fields[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"
        ]
        == "fixed_soa_direct_gather_v1"
    )
    assert (
        fixed_fusion_fields[
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl"
        ]
        == "fixed_soa_columns_v1"
    )
    assert (
        fixed_fusion_fields["compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path"]
        == "fixed_soa_direct_gather"
    )
    assert fixed_fusion_fields["compact_rollout_slab_sample_gate_fixed_soa_used"] is True
    assert (
        fixed_fusion_fields["compact_rollout_slab_sample_gate_fixed_soa_slot_write_count"]
        == fixed_soa_payload[
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_count"
        ]
    )
    assert (
        fixed_fusion_fields["compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count"]
        == 0
    )
    assert fixed_fusion_fields["compact_rollout_slab_sample_gate_fixed_soa_record_count"] == 2155
    assert (
        fixed_fusion_fields["compact_rollout_slab_sample_gate_fixed_soa_selected_record_count"]
        == 62
    )
    assert fixed_fusion_fields["compact_rollout_slab_sample_gate_fixed_soa_table_row_count"] == 9770
    assert (
        fixed_fusion_fields["compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size"]
        == 8
    )
    assert (
        fixed_fusion_fields["compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used"]
        is True
    )
    assert (
        fixed_fusion_fields[
            "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift"
        ]
        is True
    )
    assert (
        fixed_fusion_fields[
            "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count"
        ]
        == 64
    )
    assert (
        fixed_fusion_fields[
            "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count"
        ]
        == 2
    )

    bad_direct_batch_payload = dict(direct_batch_payload)
    bad_direct_batch_payload["compact_owner_search_direct_transition_batch_replay_used"] = False
    with pytest.raises(ValueError, match="direct transition-batch replay must be used"):
        module._require_owner_search_slab_proxy_proof(
            bad_direct_batch_payload,
            args=direct_batch_args,
        )

    bad_direct_batch_payload = dict(direct_batch_payload)
    bad_direct_batch_payload[
        "compact_owner_search_direct_transition_batch_replay_index_entry_object_count"
    ] = 1
    with pytest.raises(ValueError, match="must keep .*index_entry_object_count zero"):
        module._require_owner_search_slab_proxy_proof(
            bad_direct_batch_payload,
            args=direct_batch_args,
        )

    bad_direct_batch_payload = dict(direct_batch_payload)
    bad_direct_batch_payload[
        "compact_owner_search_direct_transition_batch_replay_fallback_count"
    ] = 1
    with pytest.raises(ValueError, match="must keep .*fallback_count zero"):
        module._require_owner_search_slab_proxy_proof(
            bad_direct_batch_payload,
            args=direct_batch_args,
        )

    bad_direct_batch_payload = dict(direct_batch_payload)
    bad_direct_batch_payload[
        "compact_owner_search_direct_transition_batch_replay_fallback_reason"
    ] = "legacy_expansion"
    with pytest.raises(ValueError, match="must not fallback"):
        module._require_owner_search_slab_proxy_proof(
            bad_direct_batch_payload,
            args=direct_batch_args,
        )

    bad_direct_batch_payload = dict(direct_batch_payload)
    bad_direct_batch_payload[
        "compact_owner_search_direct_transition_batch_replay_ring_entry_object_count"
    ] = 1
    with pytest.raises(ValueError, match="remove ring entry objects"):
        module._require_owner_search_slab_proxy_proof(
            bad_direct_batch_payload,
            args=direct_batch_args,
        )

    bad_direct_batch_payload = dict(direct_batch_payload)
    bad_direct_batch_payload[
        "compact_owner_search_direct_transition_batch_replay_columnar_append_used"
    ] = False
    with pytest.raises(ValueError, match="must use columnar append"):
        module._require_owner_search_slab_proxy_proof(
            bad_direct_batch_payload,
            args=direct_batch_args,
        )

    bad_direct_batch_payload = dict(direct_batch_payload)
    bad_direct_batch_payload[
        "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count"
    ] = 7
    with pytest.raises(ValueError, match="columnar slot count mismatch"):
        module._require_owner_search_slab_proxy_proof(
            bad_direct_batch_payload,
            args=direct_batch_args,
        )

    bad_direct_batch_payload = dict(direct_batch_payload)
    bad_direct_batch_payload[
        "compact_owner_search_direct_transition_batch_replay_last_append_sec"
    ] = float("nan")
    with pytest.raises(ValueError, match="nonnegative append sec"):
        module._require_owner_search_slab_proxy_proof(
            bad_direct_batch_payload,
            args=direct_batch_args,
        )

    bad_direct_batch_payload = dict(direct_batch_payload)
    bad_direct_batch_payload[
        "compact_owner_search_direct_transition_batch_replay_ring_append_sec"
    ] = float("nan")
    with pytest.raises(ValueError, match="nonnegative timing field"):
        module._require_owner_search_slab_proxy_proof(
            bad_direct_batch_payload,
            args=direct_batch_args,
        )

    bad_batch_payload = dict(batch_payload)
    bad_batch_payload.pop("compact_owner_search_transition_batch_schema_id")
    with pytest.raises(ValueError, match="must report compact_owner_search_transition"):
        module._require_owner_search_slab_proxy_proof(
            bad_batch_payload,
            args=batch_args,
        )

    bad_batch_payload = dict(batch_payload)
    bad_batch_payload["compact_owner_search_transition_batch_transport_entry_count"] = 8
    with pytest.raises(ValueError, match="transport count"):
        module._require_owner_search_slab_proxy_proof(
            bad_batch_payload,
            args=batch_args,
        )

    bad_batch_payload = dict(batch_payload)
    bad_batch_payload["compact_owner_search_owner_replay_transition_legacy_entry_count"] = 1
    with pytest.raises(ValueError, match="avoid legacy"):
        module._require_owner_search_slab_proxy_proof(
            bad_batch_payload,
            args=batch_args,
        )

    bad_batch_payload = dict(batch_payload)
    bad_batch_payload["compact_owner_search_transition_batch_transport_kind"] = "wrong"
    with pytest.raises(ValueError, match="expected kind"):
        module._require_owner_search_slab_proxy_proof(
            bad_batch_payload,
            args=batch_args,
        )

    bad_batch_payload = dict(batch_payload)
    bad_batch_payload["compact_owner_search_transition_batch_transport_requested"] = False
    with pytest.raises(ValueError, match="compact request bit"):
        module._require_owner_search_slab_proxy_proof(
            bad_batch_payload,
            args=batch_args,
        )

    stale_primary_drained_payload = dict(batch_payload)
    stale_primary_drained_payload.update(
        {
            "compact_owner_search_replay_append_transition_batch_count": 1,
            "compact_owner_search_replay_append_transition_batch_entry_count": 4,
            "compact_owner_search_replay_append_transport_entry_count": 1,
            "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_entry_count": 8,
            "compact_owner_search_owner_maintenance_drained_replay_append_transport_entry_count": 2,
        }
    )
    module._require_owner_search_slab_proxy_proof(
        stale_primary_drained_payload,
        args=batch_args,
    )

    bad_batch_payload = dict(batch_payload)
    bad_batch_payload["compact_owner_search_replay_append_transition_batch_count"] = 1
    with pytest.raises(ValueError, match="worker transition-batch count mismatch"):
        module._require_owner_search_slab_proxy_proof(
            bad_batch_payload,
            args=batch_args,
        )

    bad_batch_payload = dict(batch_payload)
    bad_batch_payload["compact_owner_search_replay_append_transport_entry_count"] = 8
    with pytest.raises(ValueError, match="worker transition-batch transport"):
        module._require_owner_search_slab_proxy_proof(
            bad_batch_payload,
            args=batch_args,
        )

    bad_bypass_payload = dict(bypass_payload)
    bad_bypass_payload.pop("compact_rollout_slab_general_replay_row_builder_used")
    with pytest.raises(ValueError, match="must report compact_rollout_slab_general"):
        module._require_owner_search_slab_proxy_proof(
            bad_bypass_payload,
            args=bypass_args,
        )

    bad_bypass_payload = dict(bypass_payload)
    bad_bypass_payload.pop("compact_owner_search_slab_bypass_parent_committed_index_rows")
    with pytest.raises(ValueError, match="must report compact_owner_search_slab_bypass"):
        module._require_owner_search_slab_proxy_proof(
            bad_bypass_payload,
            args=bypass_args,
        )

    bad_bypass_payload = dict(bypass_payload)
    bad_bypass_payload["compact_owner_search_slab_bypass"] = False
    with pytest.raises(ValueError, match="bypass mode"):
        module._require_owner_search_slab_proxy_proof(
            bad_bypass_payload,
            args=bypass_args,
        )

    bad_bypass_payload = dict(bypass_payload)
    bad_bypass_payload["compact_owner_search_slab_bypass_kind"] = "wrong"
    with pytest.raises(ValueError, match="direct-stepper kind"):
        module._require_owner_search_slab_proxy_proof(
            bad_bypass_payload,
            args=bypass_args,
        )

    bad_bypass_payload = dict(bypass_payload)
    bad_bypass_payload["compact_rollout_slab_general_replay_row_builder_used"] = True
    with pytest.raises(ValueError, match="general replay-row builder"):
        module._require_owner_search_slab_proxy_proof(
            bad_bypass_payload,
            args=bypass_args,
        )

    bad_bypass_payload = dict(bypass_payload)
    bad_bypass_payload["compact_owner_search_slab_bypass_parent_committed_index_rows"] = 1
    with pytest.raises(ValueError, match="parent committed proof"):
        module._require_owner_search_slab_proxy_proof(
            bad_bypass_payload,
            args=bypass_args,
        )

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload["compact_owner_search_parent_slab_commits_replay"] = True
    with pytest.raises(ValueError, match="parent slab does not commit replay"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload["compact_rollout_slab_committed_index_row_count"] = 1
    with pytest.raises(ValueError, match="zero parent slab rows"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload["compact_rollout_slab_stored_index_row_count"] = 1
    with pytest.raises(ValueError, match="zero parent slab rows"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload["compact_owner_search_replay_payload_handle_present"] = False
    with pytest.raises(ValueError, match="replay payload handle"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload[
        "compact_owner_search_owner_maintenance_drained_replay_append_entry_count"
    ] = 1
    with pytest.raises(ValueError, match="drain every submitted replay append entry"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload["compact_owner_search_action_feedback_verified"] = False
    with pytest.raises(ValueError, match="verify action feedback"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload["compact_owner_search_action_feedback_transition_count"] = 1
    with pytest.raises(ValueError, match="transitions must match submitted entries"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload["compact_owner_search_action_feedback_mismatch_count"] = 1
    with pytest.raises(ValueError, match="action feedback must not mismatch"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload["compact_owner_search_applied_joint_action_checksum"] = 12
    with pytest.raises(ValueError, match="checksums must agree"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)

    bad_action_only_payload = dict(action_only_payload)
    bad_action_only_payload.pop("compact_owner_search_inner_two_phase_action_step")
    with pytest.raises(ValueError, match="inner_two_phase_action_step"):
        module._require_owner_search_slab_proxy_proof(bad_action_only_payload)


def test_speed_row_smoke_owner_search_fused_tensor_native_uses_owner_sample_telemetry():
    module = _load_smoke_module()
    args = _summary_args()
    args.search_service_kind = "owner_search_slab_proxy"
    args.compact_owned_loop_fused_learner_batch = True
    args.compact_muzero_learner_batch_learner_ready_unroll2_cache = True
    args.compact_muzero_learner_batch_tensor_native_replay = True
    args.learner_num_unroll_steps = 2
    payload = _profile_payload()
    payload.update(_owner_search_slab_proxy_fields())
    assert payload["compact_owned_loop_entrypoint_enabled"] is False
    owner_sample_telemetry = payload["compact_owner_search_owner_sample_telemetry"]
    owner_sample_telemetry.update(
        {
            "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": True,
            ("compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"): True,
            "compact_rollout_slab_sample_gate_explicit_next_targets": True,
            "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
            "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_learner_ready_unroll2_cache_requested"
            ): True,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_learner_ready_unroll2_cache_available_group_count"
            ): 2,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_learner_ready_unroll2_cache_eligible_count"
            ): 2,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_learner_ready_unroll2_cache_used"
            ): True,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_learner_ready_unroll2_cache_call_count"
            ): 2,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_learner_ready_unroll2_cache_fallback_count"
            ): 0,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_learner_ready_unroll2_cache_fallback_reason"
            ): "none",
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_learner_ready_unroll2_cache_impl"
            ): "learner_ready_unroll2_cache_v1",
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_requested"
            ): True,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
            ): True,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_call_count"
            ): 2,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_fallback_count"
            ): 0,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_fallback_reason"
            ): "none",
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"
            ): "maintained_unroll2_table_gather_v1",
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_table_source"
            ): "maintained_record_table_v1",
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_table_reused_record_count"
            ): 2,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_table_missing_record_count"
            ): 0,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_table_rows"
            ): 128,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_table_concat_sec"
            ): 0.001,
            (
                "compact_rollout_slab_sample_gate_"
                "learner_batch_builder_tensor_native_replay_gather_sec"
            ): 0.002,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count": 0,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason": "none",
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count": 0,
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count": 2,
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count": 2,
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_missing_record_count": 0,
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_rows": 128,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": (
                "learner_ready_unroll2_cache"
            ),
        }
    )
    payload["compact_owner_search_owner_learner_telemetry"].update(
        {"compact_muzero_learner_prebuilt_batch_used": True}
    )

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity={},
    )

    for row in (summary, compact):
        assert row["compact_owned_loop_fused_learner_batch"] is True
        assert row["compact_rollout_slab_sample_gate_sec"] == pytest.approx(0.001)
        assert row["compact_rollout_slab_learner_gate_sec"] == pytest.approx(0.002)
        assert row["compact_rollout_slab_sample_gate_calls"] == 0
        assert row["compact_rollout_slab_learner_gate_updates"] == 0
        assert row["compact_rollout_slab_learner_gate_prebuilt_batch_used"] is True
        assert row["compact_rollout_slab_sample_gate_compact_muzero_learner_batch"] is True
        assert row["compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"] is True
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"]
            is True
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path"]
            == "learner_ready_unroll2_cache"
        )
        assert (
            row["compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used"] is True
        )
        assert row["compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count"] == 0
        assert (
            row[
                "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested"
            ]
            is True
        )
        assert (
            row["compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used"]
            is True
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count"
            ]
            == 2
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested"
            ]
            is True
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used"
            ]
            is True
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count"
            ]
            == 2
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_missing_record_count"
            ]
            == 0
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_rows"
            ]
            == 128
        )

    broken = dict(payload)
    broken["compact_owner_search_owner_sample_telemetry"] = dict(owner_sample_telemetry)
    broken["compact_owner_search_owner_sample_telemetry"][
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
    ] = False
    with pytest.raises(ValueError, match="tensor_native_replay_used"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=broken,
            loaded_checkpoint_identity={},
        )


def test_speed_row_smoke_owner_search_owned_fused_batch_does_not_need_parent_prebuilt():
    module = _load_smoke_module()
    args = _summary_args()
    args.search_service_kind = "owner_search_slab_proxy"
    args.compact_owned_loop_fused_learner_batch = True
    args.compact_muzero_learner_batch_learner_ready_unroll2_cache = True
    args.compact_muzero_learner_batch_tensor_native_replay = True
    args.learner_num_unroll_steps = 2
    payload = _profile_payload()
    payload.update(_owner_search_slab_proxy_fields())
    payload.update(
        {
            "compact_owned_training_loop_owner": "owner_search_worker",
            "compact_owner_search_action_only_result": True,
            "compact_owner_search_owner_materializes_replay": True,
            "compact_owner_search_parent_slab_commits_replay": False,
            "compact_owner_search_parent_reconstructed_search_result": False,
            "compact_owner_search_search_result_payload_bytes": 0,
            "compact_owner_search_search_result_payload_transport_kind": (
                "action_only_owner_cached_replay_v1"
            ),
            "compact_owner_search_visit_policy_bytes": 0,
            "compact_owner_search_root_value_bytes": 0,
            "compact_owner_search_optional_array_bytes": 0,
            "compact_rollout_slab_committed_index_row_count": 0,
            "compact_rollout_slab_stored_index_row_count": 0,
        }
    )
    payload["compact_owner_search_owner_sample_telemetry"].update(
        {
            "compact_rollout_slab_sample_gate_explicit_next_targets": True,
            "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
            "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count": 0,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason": "none",
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": "learner_ready_unroll2_cache_v1",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count": 0,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason": "none",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": "maintained_unroll2_table_gather_v1",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": "maintained_record_table_v1",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count": 0,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows": 128,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": (
                "learner_ready_unroll2_cache"
            ),
        }
    )

    module._require_owner_search_slab_proxy_proof(payload)
    module._require_fused_learner_batch_proof(args, payload)
    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity={},
    )

    from curvyzero.training import compact_coach_speed_row

    compact_coach_speed_row._validate_sample_learner_fusion_surface(summary)
    operational_surface = compact_coach_speed_row._speed_row_sample_learner_fusion_surface(
        [summary]
    )
    compact_coach_speed_row._validate_sample_learner_fusion_surface(operational_surface)
    broken_summary = dict(summary)
    broken_summary["compact_owner_search_owner_train_request_count"] = 0
    with pytest.raises(
        compact_coach_speed_row.CompactCoachSpeedRowEvidenceError,
        match="compact_rollout_slab_learner_gate_prebuilt_batch_used",
    ):
        compact_coach_speed_row._validate_sample_learner_fusion_surface(broken_summary)

    for row in (summary, compact):
        assert row["compact_rollout_slab_learner_gate_prebuilt_batch_used"] is False
        assert row["compact_owner_search_action_only_result"] is True
        assert row["compact_owner_search_owner_materializes_replay"] is True
        assert row["compact_owner_search_parent_slab_commits_replay"] is False
        assert row["compact_rollout_slab_sample_gate_compact_muzero_learner_batch"] is True
        assert row["compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"] is True
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"]
            is True
        )


def test_speed_row_smoke_preserves_profile_projection_when_fused_proof_fails(
    tmp_path,
    monkeypatch,
):
    module = _load_smoke_module()
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    lifecycle_path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
                "ok": True,
                "checkpoint_id": "unit-compact-ckpt",
                "lifecycle_gates_complete": True,
                "missing_required_gates": ["coach_speed_row"],
                "promotion_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "results"

    def fake_profile_runner(**kwargs):
        args = kwargs["args"]
        payload = _profile_payload(
            borrow_render_state=bool(args.hybrid_borrow_single_actor_render_state),
        )
        payload.update(_owner_search_slab_proxy_fields())
        payload.update(
            {
                "owner_search_inner_search_service_kind": "compact_torch_search_service",
                "owner_search_compact_torch_resident_root_bridge_ready": False,
                "resident_observation_host_fallback_count": 0.0,
                "compact_owner_search_inline_slab_proxy": True,
                "compact_owner_search_boundary_kind": ("inline_owner_search_parent_slab_commit"),
                "compact_owner_search_owner_loop_kind": "inline_priority_owner_loop_v1",
                "compact_owner_search_resident_root_bridge_ready": False,
                "compact_owner_search_resident_root_bridge_kind": "",
                "compact_owner_search_resident_root_bridge_device": "",
                "compact_owner_search_resident_root_bridge_h2d_bytes": 0.0,
                "compact_owner_search_resident_root_bridge_generation_id": 0,
                "compact_direct_root_store": True,
                "compact_direct_root_store_publish_count": 3,
                "compact_direct_root_store_resolve_count": 3,
                "compact_direct_root_store_last_root_slot_count": 2,
                "compact_owner_search_direct_root_handoff": True,
                "compact_owner_search_direct_root_rebuild_avoided": True,
                "compact_owner_search_direct_root_resolved": True,
                "compact_owner_search_direct_root_observation_bytes_sent": 0,
                "compact_owner_search_replay_append_entry_count": 3,
                "compact_owner_search_replay_append_count": 3,
                "compact_owner_search_owner_replay_append_staged_entry_count": 3,
                "compact_owner_search_owner_replay_append_submitted_entry_count": 3,
                "compact_owner_search_owner_replay_append_request_count": 3,
                "compact_owner_search_owner_replay_append_count": 3,
                "compact_owner_search_owner_train_interval": 3,
            }
        )
        owner_sample_telemetry = payload["compact_owner_search_owner_sample_telemetry"]
        owner_sample_telemetry.update(
            {
                "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": True,
                (
                    "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"
                ): True,
                "compact_rollout_slab_sample_gate_explicit_next_targets": True,
                "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
                "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
                "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_learner_ready_unroll2_cache_requested"
                ): True,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_learner_ready_unroll2_cache_available_group_count"
                ): 1,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_learner_ready_unroll2_cache_eligible_count"
                ): 1,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_learner_ready_unroll2_cache_used"
                ): False,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_learner_ready_unroll2_cache_call_count"
                ): 0,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_learner_ready_unroll2_cache_fallback_count"
                ): 1,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_learner_ready_unroll2_cache_fallback_reason"
                ): "unit missing learner-ready cache",
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_tensor_native_replay_requested"
                ): True,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_tensor_native_replay_used"
                ): False,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_tensor_native_replay_call_count"
                ): 0,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_tensor_native_replay_fallback_count"
                ): 1,
                (
                    "compact_rollout_slab_sample_gate_"
                    "learner_batch_builder_tensor_native_replay_fallback_reason"
                ): "unit missing maintained replay table",
            }
        )
        payload["compact_owner_search_owner_learner_telemetry"].update(
            {"compact_muzero_learner_prebuilt_batch_used": True}
        )
        return payload

    monkeypatch.setattr(module, "_run_local_compact_owned_profile", fake_profile_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_compact_coach_speed_row_smoke.py",
            "--run-id",
            "unit-post-profile-fused-proof-failure",
            "--output-root",
            str(output_root),
            "--unified-lifecycle-report",
            str(lifecycle_path),
            "--steps",
            "4",
            "--warmup-steps",
            "1",
            "--sample-interval",
            "3",
            "--sample-batch-size",
            "2",
            "--learner-num-unroll-steps",
            "2",
            "--search-service-kind",
            "owner_search_inline_proxy",
            "--owner-search-inner-search-service-kind",
            "compact_torch_search_service",
            "--compact-torch-initial-inference-mode",
            "direct_core",
            "--hybrid-borrow-single-actor-render-state",
            "--compact-owned-loop-fused-learner-batch",
            "--compact-muzero-learner-batch-learner-ready-unroll2-cache",
            "--compact-muzero-learner-batch-tensor-native-replay",
        ],
    )

    assert module.main() == 1

    run_dir = output_root / "unit-post-profile-fused-proof-failure"
    result = json.loads((run_dir / "row_001_result.json").read_text(encoding="utf-8"))
    report = json.loads(
        (run_dir / "compact_coach_speed_row_smoke_report.json").read_text(encoding="utf-8")
    )

    assert "fused sample/learner proof" in report["problem"]
    assert report["profile_failure_profile_payload_available"] is True
    assert result["summary"]["profile_failure_profile_payload_available"] is True
    assert (
        report[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested"
        ]
        is True
    )
    assert (
        report[
            "compact_rollout_slab_sample_gate_"
            "learner_batch_builder_tensor_native_replay_fallback_reason"
        ]
        == "unit missing maintained replay table"
    )
    assert (
        report["compact_owner_search_owner_sample_telemetry"][
            "compact_rollout_slab_sample_gate_"
            "learner_batch_builder_learner_ready_unroll2_cache_fallback_reason"
        ]
        == "unit missing learner-ready cache"
    )


def test_speed_row_smoke_preserves_profile_projection_when_evidence_save_fails(
    tmp_path,
    monkeypatch,
):
    module = _load_smoke_module()
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    lifecycle_path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
                "ok": True,
                "checkpoint_id": "unit-compact-ckpt",
                "lifecycle_gates_complete": True,
                "missing_required_gates": ["coach_speed_row"],
                "promotion_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "results"

    monkeypatch.setattr(
        module,
        "_run_local_compact_owned_profile",
        lambda **_kwargs: _profile_payload(),
    )

    def fake_save_evidence(**_kwargs):
        raise RuntimeError("unit evidence save failed")

    monkeypatch.setattr(
        module,
        "save_compact_coach_speed_row_evidence_v1",
        fake_save_evidence,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_compact_coach_speed_row_smoke.py",
            "--run-id",
            "unit-evidence-save-failure",
            "--output-root",
            str(output_root),
            "--unified-lifecycle-report",
            str(lifecycle_path),
        ],
    )

    assert module.main() == 1

    run_dir = output_root / "unit-evidence-save-failure"
    result = json.loads((run_dir / "row_001_result.json").read_text(encoding="utf-8"))
    report = json.loads(
        (run_dir / "compact_coach_speed_row_smoke_report.json").read_text(encoding="utf-8")
    )

    assert result["ok"] is False
    assert "unit evidence save failed" in report["problem"]
    assert report["profile_failure_profile_payload_available"] is True
    assert result["summary"]["profile_failure_profile_payload_available"] is True


def test_owner_search_fused_flags_are_not_sent_to_parent_profile(monkeypatch):
    module = _load_smoke_module()
    args = _modal_launcher_args(
        search_service_kind="owner_search_inline_proxy",
        compact_owned_loop_fused_learner_batch=True,
        compact_muzero_learner_batch_unroll2_specialized_builder=True,
        compact_muzero_learner_batch_learner_ready_unroll2_cache=True,
        compact_muzero_learner_batch_tensor_native_replay=True,
        learner_num_unroll_steps=2,
        learner_device="cpu",
    )
    captured: dict[str, Any] = {}

    class FakeSearchService:
        def close(self) -> None:
            captured["closed"] = True

    def fake_run_hybrid_observation_profile(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {
            "schema_id": "unit-profile",
            "profile_only": True,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }

    monkeypatch.setattr(module, "_build_search_service", lambda **_kwargs: FakeSearchService())
    monkeypatch.setattr(
        module,
        "run_hybrid_observation_profile",
        fake_run_hybrid_observation_profile,
    )

    module._run_local_compact_owned_profile(
        args=args,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    config = captured["config"]
    assert config.compact_owned_loop_entrypoint is False
    assert config.compact_rollout_slab_sample_gate is False
    assert config.compact_rollout_slab_learner_gate is False
    assert config.compact_owned_loop_fused_learner_batch is False
    assert config.compact_muzero_learner_batch_unroll2_specialized_builder is False
    assert config.compact_muzero_learner_batch_learner_ready_unroll2_cache is False
    assert config.compact_muzero_learner_batch_tensor_native_replay is False
    assert captured["closed"] is True


def test_owner_search_slab_bypass_uses_direct_transition_stepper(monkeypatch):
    module = _load_smoke_module()
    args = _modal_launcher_args(
        search_service_kind="owner_search_threaded_proxy",
        owner_search_defer_maintenance=True,
        owner_search_slab_bypass=True,
        owner_search_transition_batch_size=4,
        learner_device="cpu",
    )
    captured: dict[str, Any] = {}

    class FakeSearchService:
        def close(self) -> None:
            captured["closed"] = True

    class FakeDirectStepper:
        def __init__(self, **kwargs):
            captured["direct_stepper_kwargs"] = dict(kwargs)

    def fake_run_hybrid_observation_profile(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {
            "schema_id": "unit-profile",
            "profile_only": True,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }

    monkeypatch.setattr(module, "_build_search_service", lambda **_kwargs: FakeSearchService())
    monkeypatch.setattr(module, "CompactOwnerSearchDirectStepperV1", FakeDirectStepper)
    monkeypatch.setattr(
        module,
        "run_hybrid_observation_profile",
        fake_run_hybrid_observation_profile,
    )

    module._run_local_compact_owned_profile(
        args=args,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    kwargs = captured["kwargs"]
    stepper_kwargs = captured["direct_stepper_kwargs"]
    assert isinstance(kwargs["compact_rollout_slab"], FakeDirectStepper)
    assert stepper_kwargs["search_lane"] == (
        "compact_coach_speed_row_smoke:owner_search_slab_bypass"
    )
    assert stepper_kwargs["copy_root_observation"] is False
    assert stepper_kwargs["transition_batch_size"] == 4
    assert stepper_kwargs["resident_root_host_observation_stub"] is False
    assert stepper_kwargs["direct_root_build_request"] is False
    assert captured["config"].compact_owned_loop_entrypoint is False
    assert captured["config"].compact_rollout_slab_sample_gate is False
    assert captured["config"].compact_rollout_slab_learner_gate is False
    assert captured["closed"] is True


def test_owner_search_slab_bypass_threads_direct_root_build_request(monkeypatch):
    module = _load_smoke_module()
    args = _modal_launcher_args(
        search_service_kind="owner_search_threaded_proxy",
        owner_search_defer_maintenance=True,
        owner_search_slab_bypass=True,
        owner_search_require_resident_root_view=True,
        owner_search_resident_root_host_observation_stub=True,
        owner_search_direct_root_build_request=True,
        compact_owner_action_step_boundary=True,
        learner_device="cpu",
    )
    captured: dict[str, Any] = {}

    class FakeSearchService:
        def close(self) -> None:
            captured["closed"] = True

    class FakeDirectStepper:
        def __init__(self, **kwargs):
            captured["direct_stepper_kwargs"] = dict(kwargs)

    def fake_run_hybrid_observation_profile(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {
            "schema_id": "unit-profile",
            "profile_only": True,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }

    monkeypatch.setattr(module, "_build_search_service", lambda **_kwargs: FakeSearchService())
    monkeypatch.setattr(module, "CompactOwnerSearchDirectStepperV1", FakeDirectStepper)
    monkeypatch.setattr(
        module,
        "run_hybrid_observation_profile",
        fake_run_hybrid_observation_profile,
    )

    module._run_local_compact_owned_profile(
        args=args,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    stepper_kwargs = captured["direct_stepper_kwargs"]
    assert stepper_kwargs["resident_root_host_observation_stub"] is True
    assert stepper_kwargs["direct_root_build_request"] is True
    assert captured["config"].compact_owner_action_step_boundary is True
    assert captured["closed"] is True


def test_owner_search_replay_store_metadata_receives_fused_tensor_native_flags(monkeypatch):
    module = _load_smoke_module()
    args = _modal_launcher_args(
        search_service_kind="owner_search_inline_proxy",
        owner_search_require_resident_root_view=True,
        compact_owned_loop_fused_learner_batch=True,
        compact_muzero_learner_batch_unroll2_specialized_builder=True,
        compact_muzero_learner_batch_learner_ready_unroll2_cache=True,
        compact_muzero_learner_batch_tensor_native_replay=True,
        owner_search_slab_bypass=True,
        owner_search_direct_transition_batch_replay=True,
        owner_search_transition_batch_size=4,
        owner_search_fixed_soa_replay=True,
        learner_num_unroll_steps=2,
    )
    captured: dict[str, Any] = {}

    class FakeInlineProxy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "CompactLazyInlineOwnerSearchSlabProxyV1", FakeInlineProxy)

    module._build_owner_search_slab_proxy(
        args=args,
        model=module._TinyMuZero(),
        device="cpu",
        loaded_checkpoint_identity=_loaded_identity(),
        inline=True,
    )

    metadata = captured["replay_store_factory_kwargs"]["metadata"]
    assert captured["require_resident_root_view"] is True
    assert metadata["compact_owned_loop_fused_learner_batch"] is True
    assert metadata["compact_muzero_learner_batch_unroll2_specialized_builder"] is True
    assert metadata["compact_muzero_learner_batch_unroll2_specialized_builder_requested"] is True
    assert metadata["compact_muzero_learner_batch_learner_ready_unroll2_cache"] is True
    assert metadata["compact_muzero_learner_batch_learner_ready_unroll2_cache_requested"] is True
    assert metadata["compact_muzero_learner_batch_tensor_native_replay"] is True
    assert metadata["compact_muzero_learner_batch_tensor_native_replay_requested"] is True
    assert metadata["compact_replay_fixed_soa_unroll2_buffer_requested"] is True
    assert metadata["compact_replay_fixed_soa_learner_batch_handle_ring"] is True
    assert (
        metadata["compact_replay_fixed_soa_learner_batch_handle_ring_requested"]
        is True
    )


def test_owner_search_fixed_action_result_buffer_threads_proxy_kwargs(monkeypatch):
    module = _load_smoke_module()
    args = _modal_launcher_args(
        search_service_kind="owner_search_threaded_proxy",
        owner_search_slab_bypass=True,
        owner_search_require_resident_root_view=True,
        owner_search_resident_root_host_observation_stub=True,
        owner_search_direct_root_build_request=True,
        compact_owner_action_step_boundary=True,
        owner_search_fixed_action_result_buffer=True,
        owner_search_action_result_slot_capacity=8,
        learner_device="cpu",
    )
    captured: dict[str, Any] = {}

    class FakeThreadedProxy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "CompactLazyThreadedOwnerSearchSlabProxyV1", FakeThreadedProxy)

    module._build_owner_search_slab_proxy(
        args=args,
        model=module._TinyMuZero(),
        device="cpu",
        loaded_checkpoint_identity=_loaded_identity(),
        threaded=True,
    )

    assert captured["fixed_action_result_buffer"] is True
    assert captured["action_result_slot_capacity"] == 8
    assert captured["root_store_metadata"]["owner_search_fixed_action_result_buffer"] is True
    assert captured["root_store_metadata"]["owner_search_action_result_slot_capacity"] == 8
    config_fields = module._owner_search_config_fields(args)
    assert config_fields["owner_search_fixed_action_result_buffer_requested"] is True
    assert config_fields["owner_search_action_result_slot_capacity_requested"] == 8


def test_owner_search_mock_fast_learner_carries_terminal_proof_metadata():
    module = _load_smoke_module()
    metadata = {
        "compact_muzero_learner_batch_schema_id": "unit-schema",
        "compact_muzero_learner_value_valid_count": 8,
        "compact_muzero_learner_done_count": 4,
        "compact_muzero_learner_truncated_count": 0,
    }

    class FakeReplayStore:
        def __init__(self, result_metadata: dict[str, Any]) -> None:
            self.result_metadata = dict(result_metadata)

        def sample(self, **kwargs):
            assert kwargs["build_compact_muzero_learner_batch"] is True
            assert kwargs["compact_muzero_learner_batch_only"] is True
            return {
                "sample_row_count": 4,
                "learner_batch": SimpleNamespace(metadata=dict(self.result_metadata)),
                "sample_metadata": {},
                "telemetry": {"compact_rollout_slab_sample_gate_sample_row_count": 4},
            }

    factory = module._OwnerSearchFastMockLearnerFactorySidecarV1(
        model=SimpleNamespace(state_dict=lambda: {}),
        seed=7,
        device="cpu",
        support_scale=300,
        num_unroll_steps=2,
    )
    result = factory.train_owner_search_step(
        replay_store=FakeReplayStore(metadata),
        root_batch=None,
        search_result=None,
        sample_batch_size=4,
        train_steps=1,
        request=SimpleNamespace(
            request_id=3,
            policy_version_ref="unit-policy",
            model_version_ref="unit-model",
            policy_source="unit-source",
        ),
    )

    telemetry = result["learner_telemetry"]
    assert telemetry["compact_owner_search_mock_fast_learner"] is True
    assert telemetry["compact_muzero_learner_value_valid_count"] == 8
    assert telemetry["compact_muzero_learner_done_count"] == 4
    assert telemetry["compact_muzero_learner_truncated_count"] == 0
    assert telemetry["compact_muzero_learner_sec"] == 0.0
    assert result["learner_result"]["telemetry"] == telemetry

    bad_metadata = dict(metadata)
    bad_metadata.pop("compact_muzero_learner_done_count")
    with pytest.raises(RuntimeError, match="missing learner-batch proof keys"):
        factory.train_owner_search_step(
            replay_store=FakeReplayStore(bad_metadata),
            root_batch=None,
            search_result=None,
            sample_batch_size=4,
            train_steps=1,
            request=SimpleNamespace(
                request_id=4,
                policy_version_ref="unit-policy",
                model_version_ref="unit-model",
                policy_source="unit-source",
            ),
        )


def test_speed_row_smoke_validates_owner_search_replay_and_train_fields():
    module = _load_smoke_module()
    payload = _profile_payload()
    payload.update(_owner_search_slab_proxy_fields())
    payload.update(
        {
            "compact_owner_search_worker_owns_replay_state": True,
            "compact_owner_search_worker_owns_model_state": True,
            "compact_owner_search_replay_append_entry_count": 2,
            "compact_owner_search_replay_append_count": 2,
            "compact_owner_search_learner_update_count": 1,
            "compact_owner_search_model_owner_ref_returned": True,
            "compact_owner_search_model_owner_ref_digest": "owner-digest",
            "compact_owner_search_search_refresh_update_count": 1,
            "compact_owner_search_owner_replay_append_enabled": True,
            "compact_owner_search_owner_sample_batch_size": 2,
            "compact_owner_search_owner_train_steps": 1,
            "compact_owner_search_owner_train_interval": 2,
            "compact_owner_search_owner_model_refresh_interval": 1,
            "compact_owner_search_owner_expected_train_request_count": 1,
            "compact_owner_search_owner_replay_append_staged_entry_count": 2,
            "compact_owner_search_owner_replay_append_submitted_entry_count": 2,
            "compact_owner_search_owner_replay_append_request_count": 1,
            "compact_owner_search_owner_replay_append_count": 2,
            "compact_owner_search_owner_train_request_count": 1,
            "compact_owner_search_owner_model_refresh_request_count": 1,
            "compact_owner_search_owner_model_refresh_skipped_count": 0,
            "compact_owner_search_owner_learner_update_count": 1,
            "compact_rollout_slab_learner_gate_updates": 0,
        }
    )

    module._require_owner_search_slab_proxy_proof(payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_worker_owns_replay_state"] = False
    with pytest.raises(ValueError, match="worker-owned replay state"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_replay_append_count"] = 0
    with pytest.raises(ValueError, match="append replay in owner"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_replay_append_entry_count"] = 1
    with pytest.raises(ValueError, match="entry count must match submitted"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_replay_append_count"] = 1
    with pytest.raises(ValueError, match="append count must match owner append count"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_learner_update_count"] = 0
    with pytest.raises(ValueError, match="owner learner updates"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_model_owner_ref_digest"] = ""
    with pytest.raises(ValueError, match="owner-ref digest"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_consumed_learner_update"] = False
    with pytest.raises(ValueError, match="refresh search from owner learner"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_owner_sample_telemetry"] = {}
    with pytest.raises(ValueError, match="sample owner replay rows"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_owner_sample_telemetry"] = {
        **payload["compact_owner_search_owner_sample_telemetry"],
        "compact_rollout_slab_sample_gate_sample_row_count": 1,
    }
    with pytest.raises(ValueError, match="sample rows must match"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_owner_sample_batch_size"] = 0
    bad_payload["compact_owner_search_owner_sample_telemetry"] = {
        **payload["compact_owner_search_owner_sample_telemetry"],
        "compact_rollout_slab_sample_gate_requested_sample_row_count": 1,
    }
    with pytest.raises(ValueError, match="zero-batch samples must request zero rows"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_owner_sample_batch_size"] = 0
    bad_payload["compact_owner_search_owner_sample_telemetry"] = {
        **payload["compact_owner_search_owner_sample_telemetry"],
        "compact_rollout_slab_sample_gate_requested_sample_row_count": 0,
        "compact_rollout_slab_sample_gate_target_row_count": 3,
    }
    with pytest.raises(ValueError, match="zero-batch sample rows must match target rows"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_owner_train_timing_aggregate_count"] = 2
    with pytest.raises(ValueError, match="train timing count"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_owner_train_learner_update_sec"] = 0.0
    with pytest.raises(ValueError, match="positive learner update"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    args = _summary_args()
    args.search_service_kind = "owner_search_slab_proxy"
    args.sample_batch_size = 3
    args.sample_interval = 1
    args.learner_train_steps = 1
    args.learner_num_unroll_steps = 1
    cadence_payload = _profile_payload()
    cadence_payload.update(_owner_search_slab_proxy_fields())
    with pytest.raises(ValueError, match="sample batch size does not match"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=cadence_payload,
            loaded_checkpoint_identity={},
        )

    args = _summary_args()
    args.search_service_kind = "owner_search_slab_proxy"
    args.sample_batch_size = 2
    args.sample_interval = 1
    args.learner_train_steps = 2
    args.learner_num_unroll_steps = 1
    with pytest.raises(ValueError, match="train steps do not match"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=cadence_payload,
            loaded_checkpoint_identity={},
        )

    args = _summary_args()
    args.search_service_kind = "owner_search_slab_proxy"
    args.sample_batch_size = 2
    args.sample_interval = 3
    args.learner_train_steps = 1
    args.learner_num_unroll_steps = 1
    with pytest.raises(ValueError, match="train interval does not match"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=cadence_payload,
            loaded_checkpoint_identity={},
        )


def test_speed_row_smoke_requires_owner_search_metadata_when_requested():
    module = _load_smoke_module()
    args = _summary_args()
    args.search_service_kind = "owner_search_slab_proxy"
    payload = _profile_payload()
    payload["owner_search_slab_proxy_requested"] = True

    with pytest.raises(ValueError, match="requested but proof fields are missing"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=payload,
            loaded_checkpoint_identity={},
        )


def test_speed_row_smoke_requires_runtime_compact_torch_owner_search_bridge_fields():
    module = _load_smoke_module()
    payload = _profile_payload()
    payload.update(_owner_search_slab_proxy_fields())
    payload["owner_search_inner_search_service_kind"] = "compact_torch_search_service"
    payload["resident_observation_host_fallback_count"] = 0.0

    module._require_owner_search_slab_proxy_proof(payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_resident_root_bridge_ready"] = False
    with pytest.raises(ValueError, match="runtime resident-root bridge"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_resident_root_bridge_device"] = ""
    with pytest.raises(ValueError, match="bridge device"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_resident_root_bridge_h2d_bytes"] = 0.0
    with pytest.raises(ValueError, match="positive bridge bytes"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_resident_root_bridge_generation_id"] = 0
    with pytest.raises(ValueError, match="bridge generation id"):
        module._require_owner_search_slab_proxy_proof(bad_payload)

    bad_payload = dict(payload)
    bad_payload["resident_observation_host_fallback_count"] = 1.0
    with pytest.raises(ValueError, match="resident host fallback"):
        module._require_owner_search_slab_proxy_proof(bad_payload)


def test_inline_owner_search_direct_root_proof_skips_shared_bridge_requirement():
    module = _load_smoke_module()
    payload = _profile_payload()
    payload.update(_owner_search_slab_proxy_fields())
    payload.update(
        {
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
            "resident_observation_host_fallback_count": 0.0,
            "compact_owner_search_inline_slab_proxy": True,
            "compact_owner_search_boundary_kind": ("inline_owner_search_parent_slab_commit"),
            "compact_owner_search_owner_loop_kind": "inline_priority_owner_loop_v1",
            "compact_owner_search_resident_root_bridge_ready": False,
            "compact_owner_search_resident_root_bridge_kind": "",
            "compact_owner_search_resident_root_bridge_device": "",
            "compact_owner_search_resident_root_bridge_h2d_bytes": 0.0,
            "compact_owner_search_resident_root_bridge_generation_id": 0,
            "compact_direct_root_store": True,
            "compact_direct_root_store_publish_count": 2,
            "compact_direct_root_store_resolve_count": 2,
            "compact_direct_root_store_last_root_slot_count": 2,
            "compact_owner_search_direct_root_handoff": True,
            "compact_owner_search_direct_root_rebuild_avoided": True,
            "compact_owner_search_direct_root_resolved": True,
            "compact_owner_search_direct_root_observation_bytes_sent": 0,
        }
    )

    module._require_owner_search_slab_proxy_proof(payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_direct_root_rebuild_avoided"] = False
    with pytest.raises(ValueError, match="root rebuild is avoided"):
        module._require_owner_search_slab_proxy_proof(bad_payload)


def test_owner_search_direct_root_build_request_proof_fails_closed():
    module = _load_smoke_module()
    args = SimpleNamespace(
        owner_search_slab_bypass=True,
        owner_search_transition_batch_size=1,
        owner_search_direct_transition_batch_replay=False,
        owner_search_require_resident_root_view=True,
        owner_search_resident_root_host_observation_stub=True,
        owner_search_direct_root_build_request=True,
        compact_torch_defer_one_simulation_replay_payload=False,
    )
    payload = _profile_payload()
    payload.update(_owner_search_slab_proxy_fields())
    payload.update(
        {
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
            "resident_observation_host_fallback_count": 0.0,
            "compact_owner_search_inline_slab_proxy": True,
            "compact_owner_search_boundary_kind": ("inline_owner_search_parent_slab_commit"),
            "compact_owner_search_owner_loop_kind": "inline_priority_owner_loop_v1",
            "compact_owner_search_resident_root_bridge_ready": False,
            "compact_owner_search_resident_root_bridge_kind": "",
            "compact_owner_search_resident_root_bridge_device": "",
            "compact_owner_search_resident_root_bridge_h2d_bytes": 0.0,
            "compact_owner_search_resident_root_bridge_generation_id": 0,
            "compact_direct_root_store": True,
            "compact_direct_root_store_publish_count": 2,
            "compact_direct_root_store_resolve_count": 2,
            "compact_direct_root_store_last_root_slot_count": 2,
            "compact_owner_search_direct_root_handoff": True,
            "compact_owner_search_direct_root_rebuild_avoided": True,
            "compact_owner_search_direct_root_resolved": True,
            "compact_owner_search_direct_root_observation_bytes_sent": 0,
            "compact_owner_search_action_only_result": True,
            "compact_owner_search_owner_materializes_replay": True,
            "compact_owner_search_parent_slab_commits_replay": False,
            "compact_owner_search_parent_reconstructed_search_result": False,
            "compact_owner_search_search_result_payload_bytes": 0,
            "compact_owner_search_search_result_payload_transport_kind": (
                "action_only_owner_cached_replay_v1"
            ),
            "compact_owner_search_search_result_payload_json_safe": True,
            "compact_owner_search_visit_policy_bytes": 0,
            "compact_owner_search_root_value_bytes": 0,
            "compact_owner_search_optional_array_bytes": 0,
            "compact_owner_search_inner_two_phase_action_step": True,
            "compact_owner_search_inner_device_replay_payload_deferred": True,
            "compact_owner_search_use_inner_two_phase_device_replay": True,
            "compact_owner_search_inner_device_replay_payload_flushed_count": 0,
            "compact_owner_search_inner_deferred_one_simulation_replay_payload_flush_count": 0,
            "compact_owner_search_inner_deferred_one_simulation_replay_materialized_on_flush_count": 0,
            "compact_owner_search_inner_deferred_one_simulation_replay_recurrent_inference_calls": 0.0,
            "compact_owner_search_inner_deferred_one_simulation_model_identity_match_count": 0,
            "compact_owner_search_inner_deferred_one_simulation_model_refresh_crossed_count": 0,
            "compact_owner_search_inner_pending_deferred_replay_payload_final_count": 0,
            "compact_owner_search_inner_replay_payload_d2h_bytes": 0.0,
            "compact_owner_search_inner_deferred_one_simulation_replay_flush_sec": 0.0,
            "compact_owner_search_inner_device_replay_payload_flush_sec": 0.0,
            "compact_owner_search_inner_deferred_one_simulation_action_model_state_digest": "",
            "compact_owner_search_inner_deferred_one_simulation_flush_model_state_digest": "",
            "compact_owner_search_slab_bypass": True,
            "compact_owner_search_slab_bypass_kind": (
                module.COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
            ),
            "compact_rollout_slab_bypassed": True,
            "compact_rollout_slab_general_replay_row_builder_used": False,
            "compact_rollout_slab_committed_index_row_count": 0,
            "compact_rollout_slab_stored_index_row_count": 0,
            "compact_rollout_slab_retains_committed_index_rows": False,
            "compact_owner_search_slab_bypass_parent_committed_index_rows": 0,
            "compact_owner_search_slab_bypass_parent_stored_index_rows": 0,
            "compact_owner_search_resident_root_view_required": True,
            "compact_owner_search_resident_root_view_proved": True,
            "compact_owner_search_resident_root_view_kind": (
                "direct_root_batch_resident_handle_v1"
            ),
            "compact_owner_search_resident_root_view_generation_id": 3,
            "compact_owner_search_resident_root_view_fresh_for_step_index": 2,
            "compact_owner_search_resident_root_view_device": "cpu",
            "compact_owner_search_resident_root_view_source_backend": (
                "owner_search_shared_memory_root_to_resident_tensor_v1"
            ),
            "compact_owner_search_resident_root_view_root_shape": [4, 2, 4],
            "compact_owner_search_resident_root_view_stack_shape": [4, 2, 1, 8, 8],
            "compact_owner_search_resident_root_view_h2d_bytes": 0.0,
            "compact_owner_search_resident_root_view_d2h_bytes": 0.0,
            "compact_owner_search_resident_root_view_host_fallback_allowed": False,
            "compact_owner_search_resident_root_view_row_major_order": True,
            "compact_rollout_slab_resident_host_observation_stub_requested": True,
            "compact_rollout_slab_resident_host_observation_stubbed": True,
            "compact_rollout_slab_resident_host_observation_stub_kind": (
                "zero_stride_shape_only_v1"
            ),
            "compact_rollout_slab_resident_host_observation_stub_materialized_bytes": 0,
            "compact_rollout_slab_resident_host_observation_stub_logical_bytes": 512,
            "compact_owner_search_direct_root_build_request_requested": True,
            "compact_owner_search_direct_root_build_request_handoff": True,
            "compact_owner_search_direct_root_build_request_schema_id": (
                "curvyzero_compact_root_build_request/v1"
            ),
            "compact_owner_search_direct_root_build_request_kind": (
                "resident_root_view_build_request_v1"
            ),
            "compact_owner_search_direct_root_build_request_publish_count": 2,
            "compact_owner_search_direct_root_build_request_resolve_count": 2,
            "compact_owner_search_direct_root_build_request_root_count": 4,
            "compact_owner_search_direct_root_build_request_active_root_count": 4,
            "compact_owner_search_direct_root_build_request_observation_included": False,
            "compact_owner_search_direct_root_build_request_observation_bytes_sent": 0,
            "compact_owner_search_direct_root_build_request_resident_handle_present": True,
            "compact_owner_search_direct_root_parent_build_avoided": True,
            "compact_owner_search_direct_root_parent_build_call_count": 0,
            "compact_owner_search_direct_root_parent_build_sec": 0.0,
            "compact_owner_search_direct_root_build_request_sec": 0.001,
            "compact_owner_search_direct_root_owner_build_used": True,
            "compact_owner_search_direct_root_owner_build_count": 2,
            "compact_owner_search_direct_root_owner_build_sec": 0.002,
            "compact_owner_search_parent_compact_root_batch_objects_sent": 0,
            "compact_owner_search_root_build_request_host_observation_bytes_sent": 0,
            "compact_rollout_slab_parent_root_batch_build_avoided": True,
            "compact_rollout_slab_parent_root_batch_builder_used": False,
            "compact_rollout_slab_parent_root_batch_builder_call_count": 0,
            "compact_rollout_slab_root_batch_build_sec": 0.0,
            "compact_rollout_slab_root_build_request_sec": 0.001,
        }
    )

    module._require_owner_search_slab_proxy_proof(payload, args=args)
    projected = module._owner_search_slab_proxy_proof_fields(payload)
    assert projected["compact_owner_search_direct_root_build_request_handoff"] is True
    assert projected["compact_owner_search_direct_root_owner_build_count"] == 2

    mechanics_payload = dict(payload)
    mechanics_payload.update(
        {
            "compact_owner_mechanics_step_boundary_enabled": True,
            "compact_owner_mechanics_step_boundary": True,
            "compact_owner_mechanics_step_view_schema_id": "",
            "compact_owner_mechanics_step_frame_slot_schema_id": (
                "curvyzero_compact_owner_mechanics_step_frame_slot/v1"
            ),
            "compact_owner_mechanics_step_boundary_count": 2,
            "compact_owner_mechanics_parent_compact_batch_builder_call_count": 0,
            "compact_owner_mechanics_parent_compact_batch_object_count": 0,
            "compact_owner_mechanics_parent_compact_batch_builder_used": False,
            "compact_owner_mechanics_step_view_object_count": 0,
            "compact_owner_mechanics_host_observation_bytes_sent": 0,
            "compact_owner_mechanics_host_final_observation_bytes_sent": 0,
            "compact_owner_mechanics_resident_observation_handle_present": True,
            "compact_owner_mechanics_step_frame_handle_schema_id": (
                "curvyzero_compact_owner_mechanics_step_frame_handle/v1"
            ),
            "compact_owner_mechanics_step_frame_handle_ring_used": True,
            "compact_owner_mechanics_step_frame_handle_published": True,
            "compact_owner_mechanics_step_frame_handle_consumed": True,
            "compact_owner_mechanics_step_frame_handle_publish_count": 2,
            "compact_owner_mechanics_step_frame_handle_consume_count": 2,
            "compact_owner_mechanics_step_frame_handle_ring_slot_count": 4,
            "compact_owner_mechanics_step_frame_handle_slot_id": 2,
            "compact_owner_mechanics_step_frame_handle_generation": 2,
            "compact_owner_mechanics_step_frame_handle_digest": "abc123",
            "compact_owner_mechanics_step_frame_handle_digest_verified": True,
            "compact_owner_mechanics_step_frame_handle_owner_digest_verified": True,
            "compact_owner_mechanics_step_frame_handle_resident_observation_present": True,
            "compact_owner_mechanics_step_frame_slot_write_count": 2,
            "compact_owner_mechanics_parent_step_frame_build_count": 0,
            "compact_owner_step_frame_root_build_request_used": True,
            "compact_owner_step_frame_root_build_request_from_batch_helper_used": False,
            "compact_owner_step_frame_root_request_sidecar_array_bytes": 0,
            "compact_owner_step_frame_root_request_sidecar_field_count": 0,
            "compact_owner_root_action_context_handle_used": True,
            "compact_owner_root_action_context_handle_schema_id": (
                "curvyzero_compact_owner_root_action_context_handle/v1"
            ),
            "compact_owner_root_action_context_handle_id": 2,
            "compact_owner_root_action_context_transaction_id": 2,
            "compact_owner_root_action_context_dispatch_id": 2,
            "compact_owner_root_action_context_root_count": 4,
            "compact_owner_root_action_context_active_root_count": 4,
            "compact_owner_root_action_context_context_digest": "root-action-context-digest",
            "compact_owner_root_action_context_owner_store_count": 2,
            "compact_owner_root_action_context_owner_resolve_count": 2,
            "compact_owner_root_action_context_owner_release_count": 2,
            "compact_owner_root_action_context_owner_pending_count": 0,
            "compact_owner_root_action_context_owner_max_pending_count": 1,
            "compact_owner_root_action_context_owner_digest_verified": True,
            "compact_owner_search_pending_root_action_context_stored": False,
            "compact_owner_search_action_dispatch_pending_root_action_context_stored": False,
            "compact_owner_search_action_dispatch_pending_root_action_context_store_count": 0,
            "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count": 2,
            "compact_owner_search_parent_action_context_validation_count": 0,
            "compact_owner_search_owner_action_context_validation_count": 2,
            "compact_owner_root_search_transaction_boundary_supported": True,
            "compact_owner_root_search_transaction_requested": True,
            "compact_owner_root_search_transaction_used": True,
            "compact_owner_root_search_transaction_schema_id": (
                "curvyzero_compact_owner_root_search_transaction/v1"
            ),
            "compact_owner_root_search_transaction_id": 2,
            "compact_owner_root_search_transaction_begin_count": 2,
            "compact_owner_root_search_transaction_submit_count": 2,
            "compact_owner_root_search_transaction_resolve_count": 2,
            "compact_owner_root_search_transaction_pending_count": 0,
            "compact_owner_root_search_transaction_max_pending_count": 1,
            "compact_owner_root_search_transaction_parent_root_request_build_count": 0,
            "compact_owner_root_search_transaction_parent_root_request_stored": False,
            "compact_owner_root_search_transaction_parent_compact_batch_stored": False,
            "compact_owner_root_search_transaction_parent_rebuild_count": 0,
            "compact_owner_root_search_transaction_parent_root_action_context_stored": False,
            "compact_owner_root_search_transaction_parent_root_action_context_store_count": 0,
            "compact_owner_root_search_transaction_parent_root_action_context_array_bytes": 0,
            "compact_owner_root_search_transaction_parent_root_action_context_field_count": 0,
            "compact_owner_root_search_transaction_owner_root_request_build_count": 2,
            "compact_owner_root_search_transaction_owner_root_request_build_sec": 0.001,
            "compact_owner_root_search_transaction_owner_root_store_publish_count": 2,
            "compact_owner_root_search_transaction_frame_generation_verified": True,
            "compact_owner_root_search_transaction_frame_digest_verified": True,
            "compact_owner_root_search_transaction_action_identity_verified": True,
            "compact_owner_root_search_transaction_proxy_transition_closure_used": False,
            "compact_owner_root_search_transaction_applied_action_mismatch_count": 0,
        }
    )
    module._require_owner_search_slab_proxy_proof(mechanics_payload, args=args)
    mechanics_projected = module._owner_search_slab_proxy_proof_fields(mechanics_payload)
    assert mechanics_projected["compact_owner_mechanics_step_frame_handle_ring_used"] is True
    assert mechanics_projected["compact_owner_step_frame_root_build_request_used"] is True
    assert (
        mechanics_projected[
            "compact_owner_step_frame_root_build_request_from_batch_helper_used"
        ]
        is False
    )

    for key in (
        "compact_owner_mechanics_step_frame_slot_schema_id",
        "compact_owner_mechanics_step_frame_handle_ring_used",
        "compact_owner_mechanics_step_frame_slot_write_count",
        "compact_owner_step_frame_root_build_request_used",
        "compact_owner_step_frame_root_build_request_from_batch_helper_used",
        "compact_owner_step_frame_root_request_sidecar_array_bytes",
        "compact_owner_step_frame_root_request_sidecar_field_count",
        "compact_owner_root_action_context_handle_used",
        "compact_owner_root_action_context_handle_schema_id",
        "compact_owner_root_action_context_handle_id",
        "compact_owner_root_action_context_transaction_id",
        "compact_owner_root_action_context_dispatch_id",
        "compact_owner_root_action_context_root_count",
        "compact_owner_root_action_context_active_root_count",
        "compact_owner_root_action_context_context_digest",
        "compact_owner_root_action_context_owner_store_count",
        "compact_owner_root_action_context_owner_resolve_count",
        "compact_owner_root_action_context_owner_release_count",
        "compact_owner_root_action_context_owner_pending_count",
        "compact_owner_root_action_context_owner_max_pending_count",
        "compact_owner_root_action_context_owner_digest_verified",
        "compact_owner_search_pending_root_action_context_stored",
        "compact_owner_search_action_dispatch_pending_root_action_context_stored",
        "compact_owner_search_action_dispatch_pending_root_action_context_store_count",
        "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count",
        "compact_owner_search_parent_action_context_validation_count",
        "compact_owner_search_owner_action_context_validation_count",
        "compact_owner_root_search_transaction_used",
        "compact_owner_root_search_transaction_begin_count",
        "compact_owner_root_search_transaction_submit_count",
        "compact_owner_root_search_transaction_resolve_count",
        "compact_owner_root_search_transaction_pending_count",
        "compact_owner_root_search_transaction_parent_root_request_build_count",
        "compact_owner_root_search_transaction_parent_root_request_stored",
        "compact_owner_root_search_transaction_parent_compact_batch_stored",
        "compact_owner_root_search_transaction_parent_rebuild_count",
        "compact_owner_root_search_transaction_parent_root_action_context_stored",
        "compact_owner_root_search_transaction_parent_root_action_context_store_count",
        "compact_owner_root_search_transaction_parent_root_action_context_array_bytes",
        "compact_owner_root_search_transaction_parent_root_action_context_field_count",
        "compact_owner_root_search_transaction_owner_root_request_build_count",
        "compact_owner_root_search_transaction_owner_root_store_publish_count",
        "compact_owner_root_search_transaction_frame_generation_verified",
        "compact_owner_root_search_transaction_frame_digest_verified",
        "compact_owner_root_search_transaction_action_identity_verified",
        "compact_owner_root_search_transaction_applied_action_mismatch_count",
    ):
        missing_payload = dict(mechanics_payload)
        missing_payload.pop(key)
        with pytest.raises(ValueError, match="owner mechanics step-frame"):
            module._require_owner_search_slab_proxy_proof(missing_payload, args=args)

    for key, value, match in (
        (
            "compact_owner_mechanics_step_view_schema_id",
            "curvyzero_compact_owner_mechanics_step_view/v1",
            "legacy step-view",
        ),
        ("compact_owner_mechanics_step_frame_handle_ring_used", False, "ring_used"),
        ("compact_owner_mechanics_step_frame_slot_write_count", 0, "positive"),
        ("compact_owner_mechanics_step_frame_slot_write_count", 1, "slot write count"),
        ("compact_owner_mechanics_parent_step_frame_build_count", 1, "zero"),
        ("compact_owner_step_frame_root_build_request_used", False, "root_build_request_used"),
        (
            "compact_owner_step_frame_root_build_request_from_batch_helper_used",
            True,
            "from-batch helper",
        ),
        ("compact_owner_step_frame_root_request_sidecar_array_bytes", 8, "zero"),
        ("compact_owner_step_frame_root_request_sidecar_field_count", 1, "zero"),
        ("compact_owner_root_action_context_handle_used", False, "requires"),
        ("compact_owner_root_action_context_handle_schema_id", "bad-schema", "schema"),
        ("compact_owner_root_action_context_root_count", 0, "counts"),
        ("compact_owner_root_action_context_active_root_count", 0, "counts"),
        ("compact_owner_root_action_context_context_digest", "", "digest"),
        ("compact_owner_root_action_context_owner_store_count", 1, "transaction count"),
        ("compact_owner_root_action_context_owner_resolve_count", 1, "transaction count"),
        ("compact_owner_root_action_context_owner_release_count", 1, "transaction count"),
        ("compact_owner_root_action_context_owner_pending_count", 1, "zero"),
        ("compact_owner_root_action_context_owner_digest_verified", False, "requires"),
        ("compact_owner_search_pending_root_action_context_stored", True, "false"),
        (
            "compact_owner_search_action_dispatch_pending_root_action_context_stored",
            True,
            "false",
        ),
        (
            "compact_owner_search_action_dispatch_pending_root_action_context_store_count",
            1,
            "zero",
        ),
        (
            "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count",
            0,
            "positive",
        ),
        ("compact_owner_search_parent_action_context_validation_count", 1, "zero"),
        ("compact_owner_search_owner_action_context_validation_count", 0, "positive"),
        ("compact_owner_root_search_transaction_used", False, "requires"),
        ("compact_owner_root_search_transaction_begin_count", 1, "transaction count"),
        ("compact_owner_root_search_transaction_pending_count", 1, "zero"),
        (
            "compact_owner_root_search_transaction_parent_root_request_build_count",
            1,
            "zero",
        ),
        (
            "compact_owner_root_search_transaction_parent_root_request_stored",
            True,
            "false",
        ),
        (
            "compact_owner_root_search_transaction_parent_compact_batch_stored",
            True,
            "false",
        ),
        ("compact_owner_root_search_transaction_parent_rebuild_count", 1, "zero"),
        (
            "compact_owner_root_search_transaction_parent_root_action_context_stored",
            True,
            "false",
        ),
        (
            "compact_owner_root_search_transaction_parent_root_action_context_store_count",
            1,
            "zero",
        ),
        (
            "compact_owner_root_search_transaction_parent_root_action_context_array_bytes",
            1,
            "zero",
        ),
        (
            "compact_owner_root_search_transaction_parent_root_action_context_field_count",
            1,
            "zero",
        ),
        (
            "compact_owner_root_search_transaction_owner_root_request_build_count",
            1,
            "transaction count",
        ),
        (
            "compact_owner_root_search_transaction_owner_root_store_publish_count",
            1,
            "transaction count",
        ),
        (
            "compact_owner_root_search_transaction_frame_generation_verified",
            False,
            "requires",
        ),
        (
            "compact_owner_root_search_transaction_frame_digest_verified",
            False,
            "requires",
        ),
        (
            "compact_owner_root_search_transaction_action_identity_verified",
            False,
            "requires",
        ),
        (
            "compact_owner_root_search_transaction_applied_action_mismatch_count",
            1,
            "zero",
        ),
    ):
        bad_mechanics_payload = dict(mechanics_payload)
        bad_mechanics_payload[key] = value
        with pytest.raises(ValueError, match=match):
            module._require_owner_search_slab_proxy_proof(
                bad_mechanics_payload,
                args=args,
            )

    boundary_args = SimpleNamespace(**vars(args), compact_owner_action_step_boundary=True)
    with pytest.raises(ValueError, match="requested but not enabled"):
        module._require_owner_search_slab_proxy_proof(payload, args=boundary_args)

    boundary_payload = dict(payload)
    boundary_payload.update(_owner_action_step_boundary_fields())
    module._require_owner_search_slab_proxy_proof(boundary_payload, args=boundary_args)
    boundary_projected = module._owner_search_slab_proxy_proof_fields(boundary_payload)
    assert boundary_projected["compact_owner_action_step_boundary_enabled"] is True
    assert boundary_projected["compact_owner_action_step_boundary_proof_passed"] is True
    assert boundary_projected["compact_owner_action_step_boundary_feedback_action_count"] == 4

    overlap_args = SimpleNamespace(
        **vars(boundary_args),
        compact_owner_action_dispatch_step_overlap=True,
    )
    overlap_payload = dict(boundary_payload)
    overlap_payload.update(_owner_action_dispatch_step_overlap_fields())
    module._require_owner_search_slab_proxy_proof(overlap_payload, args=overlap_args)
    overlap_projected = module._owner_search_slab_proxy_proof_fields(overlap_payload)
    assert overlap_projected["compact_owner_action_dispatch_step_overlap_enabled"] is True
    assert (
        overlap_projected[
            "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count"
        ]
        == 0
    )
    assert (
        overlap_projected["compact_owner_search_action_dispatch_handle_sync_wrapper_count"]
        == 0
    )

    for key, value, match in (
        (
            "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper",
            True,
            "sync wrapper",
        ),
        (
            "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count",
            1,
            "sync_wrapper_count",
        ),
        (
            "compact_owner_search_action_dispatch_handle_sync_wrapper_count",
            1,
            "sync_wrapper_count",
        ),
        (
            "compact_owner_search_action_dispatch_handle_completed_at_submit_count",
            1,
            "completed_at_submit_count",
        ),
        (
            "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count",
            1,
            "waited during submit",
        ),
    ):
        bad_overlap_payload = dict(overlap_payload)
        bad_overlap_payload[key] = value
        with pytest.raises(ValueError, match=match):
            module._require_owner_search_slab_proxy_proof(
                bad_overlap_payload,
                args=overlap_args,
            )

    missing_overlap_payload = dict(overlap_payload)
    missing_overlap_payload.pop(
        "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count"
    )
    with pytest.raises(ValueError, match="wait-in-submit count"):
        module._require_owner_search_slab_proxy_proof(
            missing_overlap_payload,
            args=overlap_args,
        )

    for key, value, match in (
        (
            "compact_owner_action_step_boundary_proof_passed",
            False,
            "proof did not pass",
        ),
        (
            "compact_owner_action_step_boundary_feedback_action_count",
            3,
            "count mismatch",
        ),
        (
            "compact_owner_action_step_boundary_failure_reason",
            "payload_action_mismatch",
            "failure reason",
        ),
        (
            "compact_rollout_slab_committed_index_row_count",
            1,
            "commit zero parent rows",
        ),
    ):
        bad_boundary_payload = dict(boundary_payload)
        bad_boundary_payload[key] = value
        with pytest.raises(ValueError, match=match):
            module._require_owner_search_slab_proxy_proof(
                bad_boundary_payload,
                args=boundary_args,
            )

    fixed_args = SimpleNamespace(
        **vars(args),
        owner_search_defer_maintenance=True,
        owner_search_fixed_action_result_buffer=True,
    )
    fixed_payload = dict(payload)
    fixed_payload.update(
        {
            "compact_owner_search_owner_defer_maintenance": True,
            "compact_owner_search_owner_maintenance_drain_request_count": 2,
            "compact_owner_search_owner_maintenance_request_count": 2,
            "compact_owner_search_owner_maintenance_staged_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_count": 2,
            "compact_owner_search_owner_maintenance_drained_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_count": 2,
            "compact_owner_search_owner_maintenance_pending_work_count": 0,
            "compact_owner_search_owner_maintenance_inflight": False,
            "compact_owner_search_owner_maintenance_final_drain_in_measured_sec": True,
            "compact_owner_search_owner_maintenance_failed": False,
            "compact_owner_search_fixed_action_result_buffer_requested": True,
            "compact_owner_search_fixed_action_result_buffer_used": True,
            "compact_owner_search_fixed_action_result_buffer_slot_count": 4,
            "compact_owner_search_fixed_action_result_buffer_acquire_count": 2,
            "compact_owner_search_fixed_action_result_buffer_write_count": 2,
            "compact_owner_search_fixed_action_result_buffer_read_count": 2,
            "compact_owner_search_fixed_action_result_buffer_slot_id": 1,
            "compact_owner_search_fixed_action_result_buffer_last_slot_id": 1,
            "compact_owner_search_fixed_action_result_buffer_wire_result_bytes": 64,
            "compact_owner_search_fixed_action_result_buffer_full_result_bytes": 512,
            "compact_owner_search_fixed_action_result_buffer_pending_slot_count": 0,
        }
    )
    module._require_owner_search_slab_proxy_proof(fixed_payload, args=fixed_args)
    fixed_projected = module._owner_search_slab_proxy_proof_fields(fixed_payload)
    assert fixed_projected["compact_owner_search_fixed_action_result_buffer_used"] is True
    assert (
        fixed_projected["compact_owner_search_fixed_action_result_buffer_wire_result_bytes"] == 64
    )
    assert (
        fixed_projected["compact_owner_search_fixed_action_result_buffer_full_result_bytes"] == 512
    )

    for key, value, match in (
        (
            "compact_owner_search_fixed_action_result_buffer_used",
            False,
            "requested but not used",
        ),
        (
            "compact_owner_search_owner_defer_maintenance",
            False,
            "deferred owner maintenance",
        ),
        (
            "compact_owner_search_fixed_action_result_buffer_full_result_bytes",
            64,
            "exceed wire bytes",
        ),
        (
            "compact_owner_search_fixed_action_result_buffer_pending_slot_count",
            1,
            "drain all pending slots",
        ),
        (
            "compact_owner_search_fixed_action_result_buffer_read_count",
            1,
            "counts must agree",
        ),
    ):
        bad_fixed_payload = dict(fixed_payload)
        bad_fixed_payload[key] = value
        with pytest.raises(ValueError, match=match):
            module._require_owner_search_slab_proxy_proof(
                bad_fixed_payload,
                args=fixed_args,
            )

    for key, value, match in (
        (
            "compact_owner_search_direct_root_build_request_handoff",
            False,
            "handoff",
        ),
        (
            "compact_owner_search_direct_root_build_request_resolve_count",
            1,
            "publish/resolve",
        ),
        (
            "compact_owner_search_direct_root_owner_build_count",
            1,
            "owner build count",
        ),
        (
            "compact_owner_search_direct_root_build_request_observation_included",
            True,
            "observation bytes",
        ),
        (
            "compact_owner_search_direct_root_parent_build_call_count",
            1,
            "parent_build_call_count",
        ),
        (
            "compact_rollout_slab_parent_root_batch_builder_used",
            True,
            "parent root builder",
        ),
    ):
        bad_payload = dict(payload)
        bad_payload[key] = value
        with pytest.raises(ValueError, match=match):
            module._require_owner_search_slab_proxy_proof(bad_payload, args=args)


def test_threaded_owner_search_direct_root_proof_requires_background_overlap():
    module = _load_smoke_module()
    payload = _profile_payload()
    payload.update(_owner_search_slab_proxy_fields())
    payload.update(
        {
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
            "resident_observation_host_fallback_count": 0.0,
            "compact_owner_search_threaded_slab_proxy": True,
            "compact_owner_search_boundary_kind": ("threaded_owner_search_parent_slab_commit"),
            "compact_owner_search_worker_kind": "threaded_owner_search_v1",
            "compact_owner_search_worker_resource_scope": "colocated_thread",
            "compact_owner_search_owner_defer_maintenance": True,
            "compact_owner_search_owner_loop_kind": "threaded_priority_owner_loop_v1",
            "compact_owner_search_owner_background_maintenance_thread": True,
            "compact_owner_search_owner_background_overlap_enabled": True,
            "compact_owner_search_owner_maintenance_request_count": 2,
            "compact_owner_search_owner_maintenance_drain_request_count": 2,
            "compact_owner_search_owner_maintenance_staged_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_count": 2,
            "compact_owner_search_owner_maintenance_drained_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_count": 2,
            "compact_owner_search_owner_maintenance_final_drain_in_measured_sec": True,
            "compact_owner_search_owner_action_while_maintenance_pending_count": 1,
            "compact_owner_search_owner_action_served_before_maintenance_count": 1,
            "compact_owner_search_owner_policy_lag_max": 1,
            "compact_owner_search_owner_action_while_policy_lagged_count": 1,
            "compact_owner_search_resident_root_bridge_ready": False,
            "compact_owner_search_resident_root_bridge_kind": "",
            "compact_owner_search_resident_root_bridge_device": "",
            "compact_owner_search_resident_root_bridge_h2d_bytes": 0.0,
            "compact_owner_search_resident_root_bridge_generation_id": 0,
            "compact_direct_root_store": True,
            "compact_direct_root_store_publish_count": 2,
            "compact_direct_root_store_resolve_count": 2,
            "compact_direct_root_store_last_root_slot_count": 2,
            "compact_owner_search_direct_root_handoff": True,
            "compact_owner_search_direct_root_rebuild_avoided": True,
            "compact_owner_search_direct_root_resolved": True,
            "compact_owner_search_direct_root_observation_bytes_sent": 0,
        }
    )

    module._require_owner_search_slab_proxy_proof(payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_owner_background_overlap_enabled"] = False
    with pytest.raises(ValueError, match="background overlap"):
        module._require_owner_search_slab_proxy_proof(bad_payload)


def test_inline_background_owner_search_direct_root_proof_requires_background_overlap():
    module = _load_smoke_module()
    payload = _profile_payload()
    payload.update(_owner_search_slab_proxy_fields())
    payload.update(
        {
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
            "resident_observation_host_fallback_count": 0.0,
            "compact_owner_search_inline_background_slab_proxy": True,
            "compact_owner_search_boundary_kind": (
                "inline_background_owner_search_parent_slab_commit"
            ),
            "compact_owner_search_worker_kind": "inline_background_owner_search_v1",
            "compact_owner_search_worker_resource_scope": (
                "inline_process_background_maintenance_thread"
            ),
            "compact_owner_search_owner_defer_maintenance": True,
            "compact_owner_search_owner_loop_kind": ("inline_background_maintenance_owner_loop_v1"),
            "compact_owner_search_owner_background_maintenance_thread": True,
            "compact_owner_search_owner_background_overlap_enabled": True,
            "compact_owner_search_owner_maintenance_request_count": 2,
            "compact_owner_search_owner_maintenance_drain_request_count": 2,
            "compact_owner_search_owner_maintenance_staged_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_count": 2,
            "compact_owner_search_owner_maintenance_drained_work_item_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": 2,
            "compact_owner_search_owner_maintenance_drained_replay_append_count": 2,
            "compact_owner_search_owner_maintenance_final_drain_in_measured_sec": True,
            "compact_owner_search_owner_action_while_maintenance_pending_count": 1,
            "compact_owner_search_owner_action_served_before_maintenance_count": 1,
            "compact_owner_search_owner_policy_lag_max": 1,
            "compact_owner_search_owner_action_while_policy_lagged_count": 1,
            "compact_owner_search_resident_root_bridge_ready": False,
            "compact_owner_search_resident_root_bridge_kind": "",
            "compact_owner_search_resident_root_bridge_device": "",
            "compact_owner_search_resident_root_bridge_h2d_bytes": 0.0,
            "compact_owner_search_resident_root_bridge_generation_id": 0,
            "compact_direct_root_store": True,
            "compact_direct_root_store_publish_count": 2,
            "compact_direct_root_store_resolve_count": 2,
            "compact_direct_root_store_last_root_slot_count": 2,
            "compact_owner_search_direct_root_handoff": True,
            "compact_owner_search_direct_root_rebuild_avoided": True,
            "compact_owner_search_direct_root_resolved": True,
            "compact_owner_search_direct_root_observation_bytes_sent": 0,
        }
    )

    module._require_owner_search_slab_proxy_proof(payload)

    bad_payload = dict(payload)
    bad_payload["compact_owner_search_owner_background_overlap_enabled"] = False
    with pytest.raises(ValueError, match="background overlap"):
        module._require_owner_search_slab_proxy_proof(bad_payload)


def test_speed_row_smoke_requires_scalar_ref_provider_proof_fields():
    module = _load_smoke_module()
    args = _summary_args()
    args.compact_owned_loop_deferred_sample_learner = True
    args.compact_owned_loop_deferred_sample_learner_max_pending = 2
    args.compact_owned_loop_sample_learner_worker_kind = "local_process"
    args.learner_device = "cuda"
    payload = _valid_local_process_sample_learner_payload()
    payload[("compact_owned_loop_deferred_sample_learner_replay_append_transport_kind")] = (
        "scalar_ref_v1"
    )

    with pytest.raises(ValueError, match="must configure a provider"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[("compact_owned_loop_deferred_sample_learner_worker_observation_provider_present")] = (
        True
    )
    with pytest.raises(ValueError, match="must send provider bootstrap"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_count")] = 1
    with pytest.raises(ValueError, match="worker provider must apply bootstrap"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_bootstrap_step_count"
        )
    ] = 1
    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_host_observation_bytes")
    ] = 0
    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_count")
    ] = 1
    with pytest.raises(ValueError, match="must not send resident snapshots"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_count")
    ] = 0
    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_bytes")
    ] = 1
    with pytest.raises(ValueError, match="must not send resident snapshot bytes"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_bytes")
    ] = 0
    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_entry_count")
    ] = 1
    with pytest.raises(ValueError, match="must not send replay entries"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_entry_count")
    ] = 0
    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_index_row_count")
    ] = 1
    with pytest.raises(ValueError, match="must not send replay rows"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_index_row_count")
    ] = 0
    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_learner_call_count")
    ] = 1
    with pytest.raises(ValueError, match="must not send learner calls"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_learner_call_count")
    ] = 0
    with pytest.raises(ValueError, match="must send render-state facts"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_render_state_bytes")
    ] = 128
    with pytest.raises(ValueError, match="replay append must send render-state facts"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[("compact_owned_loop_deferred_sample_learner_replay_append_render_state_bytes")] = 256
    payload[
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_missing_stack_history_count"
        )
    ] = 1
    with pytest.raises(ValueError, match="missing stack history"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_missing_stack_history_count"
        )
    ] = 0
    payload[
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_materialized_entry_count"
        )
    ] = 1
    with pytest.raises(ValueError, match="materialize every append entry"):
        module._require_deferred_sample_learner_proof(args, payload)

    payload[
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_materialized_entry_count"
        )
    ] = 2
    module._require_deferred_sample_learner_proof(args, payload)


def test_speed_row_smoke_surfaces_nested_loop_telemetry_for_scalar_ref_guard():
    module = _load_smoke_module()
    args = _summary_args()
    args.compact_owned_loop_deferred_sample_learner = True
    args.compact_owned_loop_deferred_sample_learner_max_pending = 2
    args.compact_owned_loop_sample_learner_worker_kind = "local_process"
    args.learner_device = "cuda"
    payload = _valid_local_process_sample_learner_payload()
    transport_key = "compact_owned_loop_deferred_sample_learner_replay_append_transport_kind"
    payload[transport_key] = "scalar_ref_v1"
    scalar_ref_fields = {
        ("compact_owned_loop_deferred_sample_learner_worker_observation_provider_present"): True,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_count"): 1,
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_bootstrap_step_count"
        ): 1,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_host_observation_bytes"): 0,
        (
            "compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_count"
        ): 0,
        (
            "compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_bytes"
        ): 0,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_entry_count"): 0,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_index_row_count"): 0,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_learner_call_count"): 0,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_render_state_bytes"): 128,
        ("compact_owned_loop_deferred_sample_learner_replay_append_render_state_bytes"): 256,
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_missing_stack_history_count"
        ): 0,
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_materialized_entry_count"
        ): 2,
    }
    for key in scalar_ref_fields:
        payload.pop(key, None)
    payload["compact_owned_loop_telemetry"] = {
        **scalar_ref_fields,
        "policy_version_ref": "should_not_be_surfaced",
    }

    with pytest.raises(ValueError, match="must configure a provider"):
        module._require_deferred_sample_learner_proof(args, payload)

    module._surface_compact_owned_loop_telemetry(payload)

    assert "policy_version_ref" not in payload
    module._require_deferred_sample_learner_proof(args, payload)


def test_speed_row_smoke_lean_trainer_step_forces_final_refresh():
    pytest.importorskip("torch")
    module = _load_smoke_module()
    args = _summary_args()
    args.search_service_kind = "compact_torch_search_service"
    args.compact_torch_initial_inference_mode = "model_method"
    args.compact_torch_request_compile = False
    args.compact_torch_request_model_compile = False
    args.compact_torch_model_compile_mode = "reduce-overhead"
    args.compact_torch_timing_mode = "host_phase_sync"
    args.compact_owned_lean_trainer_step = True
    args.compact_owned_loop_fused_learner_batch = True
    args.learner_num_unroll_steps = 2
    args.batch_size = 8
    args.steps = 11
    args.warmup_steps = 3
    args.sample_batch_size = 4
    args.sample_interval = 3
    args.replay_pair_capacity = 64
    args.learner_train_steps = 1
    args.learner_device = "cpu"
    args.num_simulations = 1
    args.seed = 20260530
    args.policy_refresh_interval = 5

    payload = module._run_local_compact_owned_lean_trainer_profile(
        args=args,
        loaded_checkpoint_identity={},
    )
    summary, _compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity={},
    )

    assert summary["compact_owned_trainer_loop_counter_source"] == (
        "run_hybrid_observation_profile"
    )
    assert (
        summary["compact_owned_trainer_learner_update_count"]
        == (summary["compact_rollout_slab_learner_gate_updates"])
    )
    assert summary["compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count"] > 0
    assert summary["compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count"] == 1
    assert (
        summary["compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count"]
        == (summary["compact_rollout_slab_learner_gate_updates"])
    )
    assert (
        summary["compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata"][
            "compact_policy_refresh_learner_update_count"
        ]
        == summary["compact_rollout_slab_learner_gate_updates"]
    )
    assert (
        summary["compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata"][
            "compact_policy_refresh_learner_update_count"
        ]
        == summary["compact_rollout_slab_learner_gate_updates"]
    )


def test_speed_row_smoke_lean_profile_oracle_report_fails_closed():
    module = _load_smoke_module()
    lean = _lean_oracle_payload("lean_compact_trainer_step")
    profile = _lean_oracle_payload("hybrid_observation_profile_runner")
    profile["profile_only"] = True
    profile["calls_train_muzero"] = False
    profile["touches_live_runs"] = False

    report = module._compact_owned_lean_profile_oracle_report(
        lean_payload=lean,
        profile_payload=profile,
    )

    assert report["ok"] is True
    assert report["mismatch_count"] == 0
    assert "env_action_checksum_total" in report["compared_fields"]

    profile["terminal_row_count"] += 1
    with pytest.raises(ValueError, match="terminal_row_count"):
        module._compact_owned_lean_profile_oracle_report(
            lean_payload=lean,
            profile_payload=profile,
        )

    profile = _lean_oracle_payload("hybrid_observation_profile_runner")
    profile["profile_only"] = True
    profile["calls_train_muzero"] = False
    profile["touches_live_runs"] = False
    profile["compact_owned_training_loop_owner"] = "wrong_owner"
    with pytest.raises(ValueError, match="owner mismatch"):
        module._compact_owned_lean_profile_oracle_report(
            lean_payload=lean,
            profile_payload=profile,
        )


def test_speed_row_smoke_projects_lean_timing_budget():
    module = _load_smoke_module()
    args = _summary_args()
    args.compact_owned_lean_trainer_step = True
    payload = _profile_payload()
    payload["compact_owned_training_loop_owner"] = "lean_compact_trainer_step"
    payload["compact_owned_loop_sample_gate_last_telemetry"] = {
        "compact_rollout_slab_sample_gate_sample_seed": 20260542,
        "compact_rollout_slab_sample_gate_action_checksum": 111,
        "compact_rollout_slab_sample_gate_sample_row_checksum": 222,
        "compact_rollout_slab_sample_gate_sample_action_checksum": 333,
        "compact_rollout_slab_sample_gate_sampled_flat_row_checksum": 444,
        "compact_rollout_slab_sample_gate_sample_position_order_checksum": 555,
        "compact_rollout_slab_sample_gate_source_record_pair_checksum": 666,
        "compact_rollout_slab_sample_gate_source_record_window_checksum": 777,
    }
    payload["compact_rollout_slab_sample_gate_last_telemetry"] = dict(
        payload["compact_owned_loop_sample_gate_last_telemetry"]
    )
    payload["compact_owned_loop_learner_gate_last_telemetry"] = {
        "compact_rollout_slab_learner_gate_compact_muzero_telemetry": {
            "seed": 20260542,
        },
    }
    payload["compact_owned_loop_sample_gate_last_sample_metadata"] = {
        "seed": 20260542,
    }
    payload["compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_metadata"] = {
        "seed": 20260533,
    }
    _attach_lean_trainer_counters(payload)
    payload.update(
        {
            "env_action_checksum_total": 11,
            "env_done_checksum_total": 22,
            "env_reward_checksum_total": 33,
            "env_action_mask_checksum_total": 44,
            "env_trajectory_checksum_total": 55,
            "env_trajectory_ordered_checksum_total": 66,
            "env_terminal_row_checksum_total": 77,
            "env_autoreset_row_checksum_total": 88,
            "env_terminal_reason_checksum_total": 99,
            "env_death_count_checksum_total": 101,
            "env_death_cause_checksum_total": 202,
            "env_death_hit_owner_checksum_total": 303,
            "last_env_action_checksum": 404,
            "last_env_trajectory_checksum": 505,
            "last_env_terminal_row_checksum": 606,
            "last_env_autoreset_row_checksum": 707,
            "compact_rollout_slab_sample_gate_opportunities": 5,
            "compact_rollout_slab_sample_gate_skipped_count": 4,
        }
    )
    payload["total_sec"] = 0.75
    payload["warmup_sec"] = 0.25
    payload["measured_sec"] = 0.5
    payload["timings"].update(
        {
            "actor_step_wall_sec": 0.11,
            "actor_step_sec": 0.09,
            "actor_idle_wait_sec": 0.002,
            "actor_payload_copy_sec": 0.001,
            "actor_compact_write_sec": 0.004,
            "actor_render_state_write_sec": 0.003,
            "actor_autoreset_sec": 0.02,
            "actor_env_runtime_sec": 0.03,
            "actor_env_runtime_step_many_sec": 0.026,
            "actor_env_runtime_movement_sec": 0.01,
            "actor_env_runtime_collision_sec": 0.011,
            "actor_env_runtime_visual_trail_append_sec": 0.003,
            "actor_env_runtime_body_append_sec": 0.002,
            "actor_env_runtime_phase_accounted_sec": 0.028,
            "actor_env_runtime_phase_residual_sec": 0.002,
            "actor_env_public_prepare_sec": 0.005,
            "actor_env_public_info_sec": 0.004,
            "actor_env_compact_action_mask_sec": 0.003,
            "actor_env_reward_sec": 0.002,
            "actor_env_final_observation_sec": 0.001,
            "actor_env_batch_pack_sec": 0.006,
            "actor_env_post_runtime_bookkeeping_sec": 0.007,
            "observation_sec": 0.07,
            "renderer_stack_update_sec": 0.07,
            "renderer_render_sec": 0.02,
            "renderer_device_render_sec": 0.015,
            "renderer_host_to_device_sec": 0.001,
            "renderer_device_to_host_sec": 0.002,
            "renderer_production_to_compact_sec": 0.003,
            "renderer_persistent_compact_state_handoff_sec": 0.004,
            "renderer_persistent_delta_pack_sec": 0.005,
            "renderer_persistent_update_sec": 0.006,
            "stack_shift_sec": 0.007,
            "stack_latest_update_sec": 0.008,
            "resident_observation_stack_update_sec": 0.006,
            "resident_observation_frame_view_sec": 0.001,
            "resident_observation_stack_shift_sec": 0.002,
            "resident_observation_latest_write_sec": 0.003,
            "resident_observation_autoreset_sec": 0.004,
            "resident_observation_autoreset_frame_view_sec": 0.0005,
            "resident_observation_autoreset_index_build_sec": 0.0006,
            "resident_observation_autoreset_zero_sec": 0.0014,
            "resident_observation_autoreset_latest_write_sec": 0.0015,
            "scalar_materialization_sec": 0.009,
            "resident_observation_replay_snapshot_sec": 0.010,
            "compact_rollout_slab_sec": 0.13,
            "compact_rollout_slab_sample_gate_sec": 0.05,
            "compact_rollout_slab_learner_gate_sec": 0.03,
            "compact_rollout_slab_policy_refresh_after_learner_gate_sec": 0.01,
        }
    )
    payload["compact_rollout_slab_telemetry_totals"].update(
        {
            "compact_rollout_slab_search_dispatch_wall_sec": 0.12,
            "compact_rollout_slab_replay_index_rows_build_sec": 0.03,
            "compact_rollout_slab_replay_index_rows_store_sec": 0.04,
            "compact_rollout_slab_owner_replay_stage_sec": 0.05,
            "compact_rollout_slab_owner_search_parent_wait_sec": 0.21,
            "compact_rollout_slab_owner_search_worker_learner_train_sec": 0.31,
            "compact_rollout_slab_owner_search_worker_search_refresh_sec": 0.06,
        }
    )

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    for row in (summary, compact):
        assert row["source_profile_total_sec"] == 0.75
        assert row["source_profile_warmup_sec"] == 0.25
        assert row["source_profile_measured_sec"] == 0.5
        assert row["seed"] == 20260530
        assert row["sample_seed_base"] == 20260530
        assert row["sample_batch_size"] == 2
        assert row["sample_interval"] == 1
        assert row["replay_pair_capacity"] == 16
        assert row["learner_train_steps"] == 1
        assert row["policy_refresh_interval"] == 1
        assert row["num_simulations"] == 1
        assert row["compact_rollout_slab_sample_gate_last_seed"] == 20260542
        assert row["compact_rollout_slab_learner_gate_last_seed"] == 20260542
        assert row["compact_owned_loop_sample_gate_last_metadata_seed"] == 20260542
        assert (
            row["compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed"]
            == 20260533
        )
        assert row["env_trajectory_ordered_checksum_total"] == 66
        assert row["env_terminal_row_checksum_total"] == 77
        assert row["env_autoreset_row_checksum_total"] == 88
        assert row["env_terminal_reason_checksum_total"] == 99
        assert row["env_death_count_checksum_total"] == 101
        assert row["env_death_cause_checksum_total"] == 202
        assert row["env_death_hit_owner_checksum_total"] == 303
        assert row["last_env_terminal_row_checksum"] == 606
        assert row["last_env_autoreset_row_checksum"] == 707
        assert row["compact_rollout_slab_sample_gate_action_checksum"] == 111
        assert row["compact_rollout_slab_sample_gate_sample_row_checksum"] == 222
        assert row["compact_rollout_slab_sample_gate_sample_action_checksum"] == 333
        assert row["compact_rollout_slab_sample_gate_sampled_flat_row_checksum"] == 444
        assert row["compact_rollout_slab_sample_gate_sample_position_order_checksum"] == 555
        assert row["compact_rollout_slab_sample_gate_source_record_pair_checksum"] == 666
        assert row["compact_rollout_slab_sample_gate_source_record_window_checksum"] == 777
        assert row["compact_rollout_slab_sample_gate_sample_rows"] == 2
        assert row["compact_rollout_slab_learner_gate_sample_rows"] == 2
        assert row["compact_rollout_slab_sample_gate_opportunities"] == 5
        assert row["compact_rollout_slab_sample_gate_skipped_count"] == 4
        assert row["compact_rollout_slab_policy_refresh_after_learner_gate_calls"] == 1
        assert row["compact_rollout_slab_policy_refresh_after_learner_gate_interval"] == 1
        assert row["compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count"] == 0
        assert row["compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count"] == 0
        assert row["speed_row_actor_step_wall_sec"] == 0.11
        assert row["speed_row_observation_sec"] == 0.07
        assert row["speed_row_actor_step_sec"] == pytest.approx(0.09)
        assert row["speed_row_actor_idle_wait_sec"] == pytest.approx(0.002)
        assert row["speed_row_actor_payload_copy_sec"] == pytest.approx(0.001)
        assert row["speed_row_actor_compact_write_sec"] == pytest.approx(0.004)
        assert row["speed_row_actor_render_state_write_sec"] == pytest.approx(0.003)
        assert row["speed_row_actor_autoreset_sec"] == pytest.approx(0.02)
        assert row["speed_row_actor_env_runtime_sec"] == pytest.approx(0.03)
        assert row["speed_row_actor_env_runtime_collision_sec"] == pytest.approx(0.011)
        assert row["speed_row_actor_env_public_prepare_sec"] == pytest.approx(0.005)
        assert row["speed_row_actor_env_reward_sec"] == pytest.approx(0.002)
        assert row["speed_row_actor_step_other_sec"] == pytest.approx(0.012)
        assert row["speed_row_renderer_render_sec"] == pytest.approx(0.02)
        assert row["speed_row_renderer_device_render_sec"] == pytest.approx(0.015)
        assert row["speed_row_renderer_host_to_device_sec"] == pytest.approx(0.001)
        assert row["speed_row_renderer_device_to_host_sec"] == pytest.approx(0.002)
        assert row["speed_row_stack_shift_sec"] == pytest.approx(0.007)
        assert row["speed_row_stack_latest_update_sec"] == pytest.approx(0.008)
        assert row["speed_row_resident_observation_stack_update_sec"] == pytest.approx(0.006)
        assert row["speed_row_resident_observation_frame_view_sec"] == pytest.approx(0.001)
        assert row["speed_row_resident_observation_stack_shift_sec"] == pytest.approx(0.002)
        assert row["speed_row_resident_observation_latest_write_sec"] == pytest.approx(0.003)
        assert row["speed_row_resident_observation_autoreset_sec"] == pytest.approx(0.004)
        assert row["speed_row_resident_observation_autoreset_frame_view_sec"] == pytest.approx(
            0.0005
        )
        assert row["speed_row_resident_observation_autoreset_index_build_sec"] == pytest.approx(
            0.0006
        )
        assert row["speed_row_resident_observation_autoreset_zero_sec"] == pytest.approx(0.0014)
        assert row["speed_row_resident_observation_autoreset_latest_write_sec"] == pytest.approx(
            0.0015
        )
        assert row["speed_row_scalar_materialization_sec"] == pytest.approx(0.009)
        assert row["speed_row_resident_observation_replay_snapshot_sec"] == pytest.approx(0.010)
        assert row["speed_row_observation_other_sec"] == pytest.approx(0.006)
        assert row["speed_row_compact_rollout_slab_sec"] == 0.13
        assert row["speed_row_sample_gate_sec"] == 0.05
        assert row["speed_row_learner_gate_sec"] == 0.03
        assert row["speed_row_policy_refresh_sec"] == 0.01
        assert row["speed_row_primary_accounted_sec"] == pytest.approx(0.40)
        assert row["speed_row_primary_residual_sec"] == pytest.approx(0.10)
        assert row["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_owner_search_parent_wait_sec"
        ] == pytest.approx(0.21)
        assert row["speed_row_total_search_dispatch_wall_sec"] == pytest.approx(0.12)
        assert row["speed_row_total_replay_index_rows_build_sec"] == pytest.approx(0.03)
        assert row["speed_row_total_replay_index_rows_store_sec"] == pytest.approx(0.04)
        assert row["speed_row_total_owner_replay_stage_sec"] == pytest.approx(0.05)
        assert row["speed_row_total_owner_search_parent_wait_sec"] == pytest.approx(0.21)
        assert row["speed_row_total_owner_search_worker_learner_train_sec"] == pytest.approx(0.31)
        assert row["speed_row_total_owner_search_worker_search_refresh_sec"] == pytest.approx(0.06)


def test_speed_row_smoke_rejects_lean_normal_death_when_trainer_config_profile_no_death():
    module = _load_smoke_module()
    args = _summary_args()
    args.death_mode = "normal"
    args.hybrid_borrow_single_actor_render_state = True
    args.compact_owned_lean_trainer_step = True
    payload = _normal_death_profile_payload()
    payload["compact_owned_training_loop_owner"] = "lean_compact_trainer_step"
    payload["compact_owned_trainer_config_death_mode"] = "profile_no_death"

    with pytest.raises(ValueError, match="trainer config death_mode=normal"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=payload,
            loaded_checkpoint_identity=_loaded_identity(),
        )


def test_speed_row_smoke_lean_normal_death_emits_trainer_config_owner():
    module = _load_smoke_module()
    args = _summary_args()
    args.death_mode = "normal"
    args.hybrid_borrow_single_actor_render_state = True
    args.compact_owned_lean_trainer_step = True
    payload = _normal_death_profile_payload()
    payload["compact_owned_training_loop_owner"] = "lean_compact_trainer_step"
    payload["compact_owned_trainer_config_death_mode"] = "normal"
    _attach_lean_trainer_counters(payload)

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    for row in (summary, compact):
        assert row["death_mode"] == "normal"
        assert row["compact_owned_trainer_config_death_mode"] == "normal"
        assert row["compact_owned_trainer_loop_counter_source"] == (
            "run_hybrid_observation_profile"
        )
        assert row["compact_owned_trainer_record_step_calls"] == 4
        assert (
            row["compact_owned_trainer_sample_batch_count"]
            == (row["compact_rollout_slab_sample_gate_calls"])
        )
        assert (
            row["compact_owned_trainer_learner_update_count"]
            == (row["compact_rollout_slab_learner_gate_updates"])
        )
        assert row["normal_death_terminal_contract_owner"] == ("compact_owned_trainer_config")
        assert row["normal_death_terminal_contract_source"] == ("measured_speed_row_payload")
        assert row["normal_death_terminal_contract_trainer_config_matches_runtime"] is True


def test_modal_speed_row_launcher_threads_lean_trainer_flags(monkeypatch):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-unit",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(
            compact_owned_lean_trainer_step=True,
            compact_owned_lean_profile_oracle=True,
            compact_owned_loop_deferred_sample_learner=True,
            compact_owned_loop_deferred_sample_learner_max_pending=3,
            gpu_utilization_sampling=True,
            gpu_utilization_sample_interval_sec=0.25,
        ),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-unit"
    command = captured["command"]
    assert "--compact-owned-lean-trainer-step" in command
    assert "--compact-owned-lean-profile-oracle" in command
    assert "--compact-owned-loop-deferred-sample-learner" in command
    assert "--compact-owned-loop-deferred-sample-learner-max-pending" in command
    assert "--compact-owned-loop-sample-learner-worker-kind" in command
    assert "3" in command
    assert "--gpu-utilization-sampling" in command
    interval_index = command.index("--gpu-utilization-sample-interval-sec")
    assert command[interval_index + 1] == "0.25"
    owner_search_index = command.index("--owner-search-inner-search-service-kind")
    assert command[owner_search_index + 1] == "compact_torch_search_service"


def test_modal_speed_row_launcher_accepts_inline_background_owner_search(monkeypatch):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-inline-background",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = _modal_launcher_args(
        search_service_kind="owner_search_inline_background_proxy",
        owner_search_defer_maintenance=True,
    )
    payload = module._launch_remote(
        args,
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-inline-background"
    command = captured["command"]
    search_index = command.index("--search-service-kind")
    assert command[search_index + 1] == "owner_search_inline_background_proxy"
    assert "--owner-search-defer-maintenance" in command
    assert module._owner_search_config_fields(args) == {
        "owner_search_slab_proxy_requested": True,
        "owner_search_inline_proxy_requested": False,
        "owner_search_inline_background_proxy_requested": True,
        "owner_search_threaded_proxy_requested": False,
        "owner_search_inner_search_service_kind": "compact_torch_search_service",
        "owner_search_compact_torch_resident_root_bridge_ready": False,
        "owner_search_defer_maintenance_requested": True,
        "owner_search_slab_bypass_requested": False,
        "owner_search_transition_batch_size_requested": 1,
        "owner_search_transition_batch_transport_requested": False,
        "owner_search_direct_transition_batch_replay_requested": False,
        "owner_search_owner_local_transition_derivation_requested": False,
        "owner_search_owner_proxy_transition_closure_requested": False,
        "owner_search_require_resident_root_view_requested": False,
        "owner_search_resident_root_host_observation_stub_requested": False,
        "owner_search_direct_root_build_request_requested": False,
        "compact_owner_action_step_boundary_requested": False,
        "compact_owner_action_dispatch_step_overlap_requested": False,
        "owner_search_fixed_action_result_buffer_requested": False,
        "owner_search_action_result_slot_capacity_requested": 4,
        "owner_search_fixed_soa_replay_requested": False,
        "owner_search_defer_model_state_digest_to_refresh_requested": False,
        "owner_search_fixed_soa_locality_sample_group_size_requested": 1,
        "owner_search_async_learner_worker_requested": False,
        "owner_search_async_learner_worker_kind_requested": "in_process_thread_v1",
        "owner_search_async_learner_max_pending_requested": 1,
    }


def test_modal_speed_row_launcher_accepts_owner_search_slab_bypass(monkeypatch):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-slab-bypass",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = _modal_launcher_args(
        search_service_kind="owner_search_threaded_proxy",
        owner_search_defer_maintenance=True,
        owner_search_slab_bypass=True,
        owner_search_transition_batch_size=4,
        owner_search_direct_transition_batch_replay=True,
        owner_search_owner_local_transition_derivation=True,
        owner_search_owner_proxy_transition_closure=True,
        owner_search_require_resident_root_view=True,
        owner_search_resident_root_host_observation_stub=True,
        owner_search_direct_root_build_request=True,
        compact_owner_action_step_boundary=True,
        compact_owner_action_dispatch_step_overlap=True,
        owner_search_fixed_action_result_buffer=True,
        owner_search_action_result_slot_capacity=8,
    )
    payload = module._launch_remote(
        args,
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-slab-bypass"
    command = captured["command"]
    search_index = command.index("--search-service-kind")
    assert command[search_index + 1] == "owner_search_threaded_proxy"
    assert "--owner-search-defer-maintenance" in command
    assert "--owner-search-slab-bypass" in command
    batch_index = command.index("--owner-search-transition-batch-size")
    assert command[batch_index + 1] == "4"
    assert "--owner-search-direct-transition-batch-replay" in command
    assert "--owner-search-owner-local-transition-derivation" in command
    assert "--owner-search-owner-proxy-transition-closure" in command
    assert "--owner-search-require-resident-root-view" in command
    assert "--owner-search-resident-root-host-observation-stub" in command
    assert "--owner-search-direct-root-build-request" in command
    assert "--compact-owner-action-step-boundary" in command
    assert "--compact-owner-action-dispatch-step-overlap" in command
    assert "--owner-search-fixed-action-result-buffer" in command
    slot_capacity_index = command.index("--owner-search-action-result-slot-capacity")
    assert command[slot_capacity_index + 1] == "8"
    assert module._owner_search_config_fields(args)["owner_search_slab_bypass_requested"] is True
    assert (
        module._owner_search_config_fields(args)[
            "owner_search_transition_batch_transport_requested"
        ]
        is True
    )
    assert (
        module._owner_search_config_fields(args)[
            "owner_search_owner_local_transition_derivation_requested"
        ]
        is True
    )
    assert (
        module._owner_search_config_fields(args)[
            "owner_search_owner_proxy_transition_closure_requested"
        ]
        is True
    )
    assert (
        module._owner_search_config_fields(args)[
            "owner_search_direct_transition_batch_replay_requested"
        ]
        is True
    )
    assert (
        module._owner_search_config_fields(args)["owner_search_direct_root_build_request_requested"]
        is True
    )
    assert (
        module._owner_search_config_fields(args)["compact_owner_action_step_boundary_requested"]
        is True
    )
    assert (
        module._owner_search_config_fields(args)[
            "compact_owner_action_dispatch_step_overlap_requested"
        ]
        is True
    )
    assert (
        module._owner_search_config_fields(args)[
            "owner_search_fixed_action_result_buffer_requested"
        ]
        is True
    )
    assert (
        module._owner_search_config_fields(args)[
            "owner_search_action_result_slot_capacity_requested"
        ]
        == 8
    )


def test_modal_speed_row_launcher_accepted_fast_path_preset_threads_full_bundle(monkeypatch):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-fast-path",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(
            compact_owned_accepted_fast_path_preset=True,
            batch_size=2,
            actor_count=8,
            death_mode="profile_no_death",
            compact_owned_loop_fused_learner_batch=False,
            compact_owned_lean_trainer_step=False,
            hybrid_borrow_single_actor_render_state=False,
            compact_torch_initial_inference_mode="model_method",
        ),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-fast-path"
    command = captured["command"]
    expected_pairs = {
        "--batch-size": "1024",
        "--actor-count": "1",
        "--steps": "180",
        "--warmup-steps": "45",
        "--death-mode": "normal",
        "--sample-batch-size": "512",
        "--sample-interval": "8",
        "--replay-pair-capacity": "4096",
        "--learner-num-unroll-steps": "2",
        "--policy-refresh-interval": "4",
        "--search-service-kind": "compact_torch_search_service",
        "--compact-torch-initial-inference-mode": "direct_core",
    }
    for flag, value in expected_pairs.items():
        flag_index = command.index(flag)
        assert command[flag_index + 1] == value
    assert "--compact-owned-loop-fused-learner-batch" in command
    assert "--compact-owned-lean-trainer-step" in command
    assert "--hybrid-borrow-single-actor-render-state" in command
    assert "--compact-owned-lean-profile-oracle" not in command
    assert "--compact-owned-loop-deferred-sample-learner" not in command
    assert "--compact-profile-bounded-diagnostics" not in command
    assert "--compact-profile-cuda-sync-timing-diagnostics" not in command
    assert "--compact-profile-runtime-step-timing-diagnostics" not in command


def test_modal_speed_row_launcher_accepted_fast_path_preset_threads_long_stability_window(
    monkeypatch,
):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-long-fast-path",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(
            compact_owned_accepted_fast_path_preset=True,
            compact_owned_accepted_fast_path_step_window="stability_1084_270",
            steps=4,
            warmup_steps=1,
        ),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-long-fast-path"
    command = captured["command"]
    expected_pairs = {
        "--batch-size": "1024",
        "--actor-count": "1",
        "--steps": "1084",
        "--warmup-steps": "270",
        "--death-mode": "normal",
        "--sample-batch-size": "512",
        "--sample-interval": "8",
        "--replay-pair-capacity": "4096",
        "--learner-num-unroll-steps": "2",
        "--policy-refresh-interval": "4",
        "--search-service-kind": "compact_torch_search_service",
        "--compact-torch-initial-inference-mode": "direct_core",
    }
    for flag, value in expected_pairs.items():
        flag_index = command.index(flag)
        assert command[flag_index + 1] == value
    assert "--compact-owned-loop-fused-learner-batch" in command
    assert "--compact-owned-lean-trainer-step" in command
    assert "--hybrid-borrow-single-actor-render-state" in command
    assert "--compact-profile-bounded-diagnostics" in command
    assert "--compact-profile-cuda-sync-timing-diagnostics" not in command
    assert "--compact-profile-runtime-step-timing-diagnostics" not in command


def test_modal_speed_row_accepted_fast_path_preset_rejects_owner_search_overrides():
    module = _load_modal_runner_module()
    args = _modal_launcher_args(
        compact_owned_accepted_fast_path_preset=True,
        search_service_kind="owner_search_threaded_proxy",
        owner_search_defer_maintenance=True,
        owner_search_slab_bypass=True,
        owner_search_transition_batch_size=4,
        owner_search_direct_transition_batch_replay=True,
    )

    with pytest.raises(ValueError, match="owner-search override flags"):
        module._apply_accepted_fast_path_preset(args)


def test_modal_speed_row_launcher_threads_cuda_sync_timing_diagnostics(
    monkeypatch,
):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-sync-diagnostic",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(compact_profile_cuda_sync_timing_diagnostics=True),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-sync-diagnostic"
    assert "--compact-profile-cuda-sync-timing-diagnostics" in captured["command"]


def test_modal_speed_row_launcher_threads_runtime_step_timing_diagnostics(
    monkeypatch,
):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-runtime-diagnostic",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(compact_profile_runtime_step_timing_diagnostics=True),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-runtime-diagnostic"
    assert "--compact-profile-runtime-step-timing-diagnostics" in captured["command"]
    assert "--compact-profile-cuda-sync-timing-diagnostics" not in captured["command"]


def test_modal_speed_row_launcher_threads_cpu_perf_stat_diagnostics(monkeypatch):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-cpu-perf-stat",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(compact_profile_cpu_perf_stat_diagnostics=True),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-cpu-perf-stat"
    assert "--compact-profile-cpu-perf-stat-diagnostics" in captured["command"]


def test_modal_speed_row_launcher_threads_owner_search_fixed_soa_replay(monkeypatch):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-fixed-soa",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(
            search_service_kind="owner_search_threaded_proxy",
            owner_search_slab_bypass=True,
            owner_search_transition_batch_size=4,
            owner_search_direct_transition_batch_replay=True,
            owner_search_fixed_soa_replay=True,
            owner_search_fixed_soa_locality_sample_group_size=4,
            compact_owned_loop_fused_learner_batch=True,
            compact_muzero_learner_batch_learner_ready_unroll2_cache=True,
            compact_muzero_learner_batch_tensor_native_replay=True,
            learner_num_unroll_steps=2,
        ),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-fixed-soa"
    assert "--owner-search-fixed-soa-replay" in captured["command"]
    assert "--owner-search-fixed-soa-locality-sample-group-size" in captured["command"]
    locality_flag_index = captured["command"].index(
        "--owner-search-fixed-soa-locality-sample-group-size"
    )
    assert captured["command"][locality_flag_index + 1] == "4"
    assert "--owner-search-direct-transition-batch-replay" in captured["command"]


def test_modal_speed_row_launcher_threads_owner_search_digest_deferral(monkeypatch):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-digest-deferral",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(
            search_service_kind="owner_search_threaded_proxy",
            owner_search_defer_maintenance=True,
            owner_search_defer_model_state_digest_to_refresh=True,
        ),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-digest-deferral"
    assert "--owner-search-defer-model-state-digest-to-refresh" in captured["command"]


def test_modal_speed_row_launcher_threads_unroll2_specialized_builder(monkeypatch):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-unroll2-specialized",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(
            compact_owned_loop_fused_learner_batch=True,
            compact_muzero_learner_batch_unroll2_specialized_builder=True,
            learner_num_unroll_steps=2,
        ),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-unroll2-specialized"
    assert "--compact-muzero-learner-batch-unroll2-specialized-builder" in captured["command"]


def test_modal_speed_row_launcher_threads_tensor_native_replay(monkeypatch):
    module = _load_modal_runner_module()
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "function_call_id": "fc-tensor-native-replay",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    payload = module._launch_remote(
        _modal_launcher_args(
            compact_owned_loop_fused_learner_batch=True,
            compact_muzero_learner_batch_learner_ready_unroll2_cache=True,
            compact_muzero_learner_batch_tensor_native_replay=True,
            learner_num_unroll_steps=2,
        ),
        "unit/lifecycle.json",
        "unit/checkpoint.pt",
    )

    assert payload["function_call_id"] == "fc-tensor-native-replay"
    assert "--compact-muzero-learner-batch-learner-ready-unroll2-cache" in captured["command"]
    assert "--compact-muzero-learner-batch-tensor-native-replay" in captured["command"]


def test_modal_speed_row_launcher_classifies_resource_exhausted_launch(monkeypatch):
    module = _load_modal_runner_module()

    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="remote launch failed",
            stderr="StatusCode.RESOURCE_EXHAUSTED: server memory usage is too high",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(module.ModalRemoteLaunchError) as exc_info:
        module._launch_remote(
            _modal_launcher_args(compact_profile_cuda_sync_timing_diagnostics=True),
            "unit/lifecycle.json",
            "unit/checkpoint.pt",
        )

    payload = exc_info.value.payload
    assert payload["schema_id"] == module.SPAWN_SCHEMA_ID
    assert payload["status"] == "failed"
    assert payload["failure_stage"] == "launch"
    assert payload["problem"] == "remote launch resource exhausted"
    assert payload["modal_resource_exhausted"] is True
    assert payload["returncode"] == 1
    assert "--compact-profile-cuda-sync-timing-diagnostics" in payload["command"]
    assert "RESOURCE_EXHAUSTED" in payload["stderr_tail"]


def test_modal_speed_row_launcher_handles_structured_spawn_failure(monkeypatch):
    module = _load_modal_runner_module()

    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schema_id": module.SPAWN_SCHEMA_ID,
                    "ok": False,
                    "status": "spawn_failed",
                    "failure_stage": "launch",
                    "failure_phase": "modal_spawn",
                    "problem": "remote spawn resource exhausted",
                    "modal_error_code": "RESOURCE_EXHAUSTED",
                    "modal_resource_exhausted": True,
                    "function_call_id": "",
                    "stderr_tail": "server rejected the request because memory usage is too high",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(module.ModalRemoteLaunchError) as exc_info:
        module._launch_remote(
            _modal_launcher_args(compact_profile_cuda_sync_timing_diagnostics=True),
            "unit/lifecycle.json",
            "unit/checkpoint.pt",
        )

    payload = exc_info.value.payload
    assert payload["schema_id"] == module.SPAWN_SCHEMA_ID
    assert payload["status"] == "spawn_failed"
    assert payload["failure_stage"] == "launch"
    assert payload["problem"] == "remote spawn resource exhausted"
    assert payload["modal_error_code"] == "RESOURCE_EXHAUSTED"
    assert payload["modal_resource_exhausted"] is True
    assert payload["function_call_id"] == ""


def test_modal_speed_row_launcher_writes_structured_launch_failure_report(tmp_path):
    module = _load_modal_runner_module()
    args = _modal_launcher_args(
        compact_owned_accepted_fast_path_preset=True,
        compact_owned_accepted_fast_path_step_window="stability_1084_270",
        compact_profile_cuda_sync_timing_diagnostics=True,
    )
    module._apply_accepted_fast_path_preset(args)
    report_path = module._write_launch_failure_report(
        args=args,
        output_dir=tmp_path,
        launch_payload={
            "schema_id": module.SPAWN_SCHEMA_ID,
            "status": "failed",
            "failure_stage": "launch",
            "problem": "remote launch resource exhausted",
            "returncode": 1,
            "modal_error_code": "RESOURCE_EXHAUSTED",
            "stdout_tail": "remote launch failed",
            "stderr_tail": "RESOURCE_EXHAUSTED",
            "modal_resource_exhausted": True,
        },
    )

    report = json.loads(report_path.read_text())
    launch = json.loads((tmp_path / "launch.json").read_text())
    assert report["ok"] is False
    assert report["schema_id"] == "curvyzero_compact_coach_speed_row_modal_report/v1"
    assert report["failure_stage"] == "launch"
    assert report["problem"] == "remote launch resource exhausted"
    assert report["function_call_id"] == ""
    assert report["modal_launch_error_code"] == "RESOURCE_EXHAUSTED"
    assert report["modal_launch_resource_exhausted"] is True
    assert report["compact_owned_accepted_fast_path_preset"] is True
    assert report["compact_owned_accepted_fast_path_step_window"] == "stability_1084_270"
    assert report["compact_owned_accepted_fast_path_stability_diagnostic"] is True
    assert report["speed_row_comparison_role"] == "long_window_stability_diagnostic"
    assert report["compact_profile_bounded_diagnostics"] is True
    assert report["compact_profile_cuda_sync_timing_diagnostics"] is True
    assert report["promotion_claim"] is False
    assert report["calls_train_muzero"] is False
    assert report["touches_live_runs"] is False
    assert launch["status"] == "failed"
    assert launch["modal_resource_exhausted"] is True


def test_modal_speed_row_remote_failure_report_projects_cpu_perf_stat_fields(
    tmp_path,
    monkeypatch,
):
    module = _load_modal_runner_module()
    lifecycle_path = tmp_path / "lifecycle.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_root = tmp_path / "reports"
    lifecycle_path.write_text(json.dumps({}), encoding="utf-8")
    checkpoint_path.write_bytes(b"unit checkpoint")

    def fake_launch_remote(args, lifecycle_ref, checkpoint_ref):
        return {
            "schema_id": module.SPAWN_SCHEMA_ID,
            "status": "spawned",
            "function_call_id": "fc-perf-unavailable",
        }

    def fake_collect(function_call_id, timeout_sec):
        assert function_call_id == "fc-perf-unavailable"
        return {
            "schema_id": "curvyzero_compact_coach_speed_row_h100_bundle/v0",
            "ok": False,
            "status": "failed",
            "problem": "perf stat diagnostic requested but perf was not found",
            "compact_profile_cpu_perf_stat_diagnostics": True,
            "compact_profile_cpu_perf_stat_available": False,
            "compact_profile_cpu_perf_stat_returncode": 127,
        }

    monkeypatch.setattr(module, "_launch_remote", fake_launch_remote)
    monkeypatch.setattr(module, "_collect_modal_function_call", fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_compact_coach_speed_row_modal_smoke.py",
            "--run-id",
            "unit-perf-unavailable",
            "--unified-lifecycle-report",
            str(lifecycle_path),
            "--compact-checkpoint",
            str(checkpoint_path),
            "--output-root",
            str(output_root),
            "--skip-upload",
            "--compact-owned-accepted-fast-path-preset",
            "--compact-owned-accepted-fast-path-step-window",
            "stability_1084_270",
            "--compact-profile-runtime-step-timing-diagnostics",
            "--compact-profile-cpu-perf-stat-diagnostics",
        ],
    )

    assert module.main() == 1
    report = json.loads(
        (
            output_root / "unit-perf-unavailable" / "compact_coach_speed_row_modal_report.json"
        ).read_text(encoding="utf-8")
    )

    assert report["ok"] is False
    assert report["function_call_id"] == "fc-perf-unavailable"
    assert report["problem"] == "perf stat diagnostic requested but perf was not found"
    assert report["compact_profile_cpu_perf_stat_diagnostics"] is True
    assert report["compact_profile_cpu_perf_stat_available"] is False
    assert report["compact_profile_cpu_perf_stat_returncode"] == 127
    assert report["compact_owned_accepted_fast_path_step_window"] == "stability_1084_270"


def test_modal_speed_row_local_and_remote_per_call_prefixes_match():
    repo_root = Path(__file__).resolve().parents[1]
    local_prefixes = _literal_assignment_tuple(
        repo_root / "scripts" / "run_compact_coach_speed_row_modal_smoke.py",
        "_SAMPLE_GATE_PER_CALL_REPORT_PREFIXES",
    )
    remote_prefixes = _literal_assignment_tuple(
        repo_root / "src" / "curvyzero" / "infra" / "modal" / "compact_coach_speed_row.py",
        "_SAMPLE_GATE_PER_CALL_REPORT_PREFIXES",
    )

    assert local_prefixes == remote_prefixes
    assert (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "unroll_terminal_window_hint_per_call"
    ) in local_prefixes
    assert (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_tensor_fallback_per_call"
    ) in local_prefixes


def test_modal_speed_row_launcher_rejects_long_stability_window_without_preset():
    module = _load_modal_runner_module()

    with pytest.raises(ValueError, match="require --compact-owned-accepted-fast-path-preset"):
        module._launch_remote(
            _modal_launcher_args(
                compact_owned_accepted_fast_path_preset=False,
                compact_owned_accepted_fast_path_step_window="stability_724_180",
            ),
            "unit/lifecycle.json",
            "unit/checkpoint.pt",
        )


def test_modal_speed_row_launcher_rejects_direct_root_build_request_with_accepted_preset():
    module = _load_modal_runner_module()

    with pytest.raises(ValueError, match="--owner-search-direct-root-build-request"):
        module._launch_remote(
            _modal_launcher_args(
                compact_owned_accepted_fast_path_preset=True,
                owner_search_direct_root_build_request=True,
            ),
            "unit/lifecycle.json",
            "unit/checkpoint.pt",
        )


def test_modal_speed_row_launcher_rejects_action_step_boundary_with_accepted_preset():
    module = _load_modal_runner_module()

    with pytest.raises(ValueError, match="--compact-owner-action-step-boundary"):
        module._launch_remote(
            _modal_launcher_args(
                compact_owned_accepted_fast_path_preset=True,
                compact_owner_action_step_boundary=True,
            ),
            "unit/lifecycle.json",
            "unit/checkpoint.pt",
        )


def test_modal_speed_row_launcher_rejects_fixed_action_result_buffer_with_accepted_preset():
    module = _load_modal_runner_module()

    with pytest.raises(ValueError, match="--owner-search-fixed-action-result-buffer"):
        module._launch_remote(
            _modal_launcher_args(
                compact_owned_accepted_fast_path_preset=True,
                owner_search_fixed_action_result_buffer=True,
            ),
            "unit/lifecycle.json",
            "unit/checkpoint.pt",
        )


def test_modal_speed_row_launcher_rejects_fixed_action_result_buffer_without_deferred_maintenance():
    module = _load_modal_runner_module()

    with pytest.raises(
        ValueError,
        match="--owner-search-fixed-action-result-buffer requires --owner-search-defer-maintenance",
    ):
        module._launch_remote(
            _modal_launcher_args(
                search_service_kind="owner_search_threaded_proxy",
                owner_search_slab_bypass=True,
                owner_search_require_resident_root_view=True,
                owner_search_resident_root_host_observation_stub=True,
                owner_search_direct_root_build_request=True,
                owner_search_fixed_action_result_buffer=True,
            ),
            "unit/lifecycle.json",
            "unit/checkpoint.pt",
        )


def test_modal_speed_row_launcher_rejects_digest_deferral_with_accepted_preset():
    module = _load_modal_runner_module()

    with pytest.raises(
        ValueError,
        match="--owner-search-defer-model-state-digest-to-refresh",
    ):
        module._launch_remote(
            _modal_launcher_args(
                compact_owned_accepted_fast_path_preset=True,
                owner_search_defer_model_state_digest_to_refresh=True,
            ),
            "unit/lifecycle.json",
            "unit/checkpoint.pt",
        )


def test_modal_speed_row_accepted_fast_path_preset_rejects_remote_flag_drift():
    module = _load_modal_runner_module()
    args = _modal_launcher_args(compact_owned_accepted_fast_path_preset=True)
    summary = {
        "batch_size": 1024,
        "actor_count": 1,
        "steps": 180,
        "warmup_steps": 45,
        "env_steps_collected": 184320.0,
        "death_mode": "normal",
        "seed": 20260530,
        "sample_seed_base": 20260530,
        "sample_batch_size": 512,
        "sample_interval": 8,
        "replay_pair_capacity": 4096,
        "learner_train_steps": 1,
        "learner_num_unroll_steps": 2,
        "policy_refresh_interval": 4,
        "num_simulations": 1,
        "search_service_kind": "compact_torch_search_service",
        "compact_torch_initial_inference_mode": "direct_core",
        "compact_owned_loop_fused_learner_batch": True,
        "compact_owned_lean_trainer_step": True,
        "hybrid_persistent_compact_render_state_buffer": False,
        "hybrid_borrow_single_actor_render_state": True,
        "render_state_handoff_mode": "borrow_single_actor_env_state",
        "render_state_copy_steps": 0,
        "render_state_borrowed_steps": 225,
        "terminal_sample_row_count": 167,
        "terminal_unroll_value_target_row_count": 167,
        "normal_death_terminal_contract_promotion_gate_satisfied": True,
        "truncated_row_count": 0,
        "resident_observation_host_fallback_count": 0.0,
        "compact_profile_autoreset_direct_count": 0,
        "compact_rollout_slab_sample_gate_last_seed": 20260551,
        "compact_rollout_slab_learner_gate_last_seed": 20260551,
        "compact_owned_loop_sample_gate_last_metadata_seed": 20260551,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed": 20260533,
        "env_action_checksum_total": 1,
        "env_done_checksum_total": 2,
        "env_reward_checksum_total": 3,
        "env_action_mask_checksum_total": 4,
        "env_trajectory_checksum_total": 5,
        "env_trajectory_ordered_checksum_total": -6,
        "env_terminal_row_checksum_total": 7,
        "env_autoreset_row_checksum_total": 8,
        "env_terminal_reason_checksum_total": 9,
        "env_death_count_checksum_total": 10,
        "env_death_cause_checksum_total": 11,
        "env_death_hit_owner_checksum_total": 12,
        "last_env_action_checksum": 13,
        "last_env_trajectory_checksum": 14,
        "last_env_terminal_row_checksum": 15,
        "last_env_autoreset_row_checksum": 16,
        "compact_rollout_slab_sample_gate_action_checksum": 17,
        "compact_rollout_slab_sample_gate_sample_row_checksum": 18,
        "compact_rollout_slab_sample_gate_sample_action_checksum": 19,
        "compact_rollout_slab_sample_gate_sampled_flat_row_checksum": 20,
        "compact_rollout_slab_sample_gate_sample_position_order_checksum": 21,
        "compact_rollout_slab_sample_gate_source_record_pair_checksum": 22,
        "compact_rollout_slab_sample_gate_source_record_window_checksum": 23,
        "compact_owned_loop_record_step_calls": 225,
        "compact_owned_loop_appended_replay_entry_count": 180,
        "compact_rollout_slab_sample_gate_sample_rows": 11264,
        "compact_rollout_slab_learner_gate_sample_rows": 11264,
        "compact_rollout_slab_sample_gate_opportunities": 224,
        "compact_rollout_slab_sample_gate_skipped_count": 202,
        "compact_rollout_slab_sample_gate_calls": 22,
        "compact_rollout_slab_learner_gate_calls": 22,
        "compact_rollout_slab_learner_gate_updates": 22,
        "compact_owned_trainer_sample_batch_count": 22,
        "compact_owned_trainer_learner_update_count": 22,
        "compact_owned_trainer_policy_refresh_count": 6,
        "compact_rollout_slab_committed_index_row_count": 366548,
        "compact_rollout_slab_stored_index_row_count": 456602,
        "compact_rollout_slab_policy_refresh_after_learner_gate_calls": 6,
        "compact_rollout_slab_policy_refresh_after_learner_gate_interval": 4,
        "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count": 16,
        "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count": 1,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": 22,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest": "abc",
        "compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind": "result_v1",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind": "result_v1",
        "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count": 0,
        "compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count": 6,
        (
            "compact_rollout_slab_policy_refresh_after_learner_gate_"
            "parent_model_state_transport_avoided"
        ): False,
    }

    assert module._accepted_fast_path_preset_violations(args, {"summary": dict(summary)}) == []

    long_args = _modal_launcher_args(
        compact_owned_accepted_fast_path_preset=True,
        compact_owned_accepted_fast_path_step_window="stability_724_180",
    )
    long_summary = dict(summary)
    long_summary.update(
        {
            "steps": 724,
            "warmup_steps": 180,
            "env_steps_collected": 741376.0,
            "render_state_borrowed_steps": 904,
            "terminal_sample_row_count": 660,
            "terminal_unroll_value_target_row_count": 660,
        }
    )
    assert (
        module._accepted_fast_path_preset_violations(
            long_args,
            {"summary": long_summary},
        )
        == []
    )

    long_terminal_mismatch = dict(long_summary)
    long_terminal_mismatch["terminal_unroll_value_target_row_count"] = 659
    long_violations = module._accepted_fast_path_preset_violations(
        long_args,
        {"summary": long_terminal_mismatch},
    )
    assert any("terminal_unroll_value_target_row_count" in item for item in long_violations)

    long_bad_denominator = dict(long_summary)
    long_bad_denominator["env_steps_collected"] = 904 * 1024
    denominator_violations = module._accepted_fast_path_preset_violations(
        long_args,
        {"summary": long_bad_denominator},
    )
    assert any("env_steps_collected" in item for item in denominator_violations)

    missing_repeatability = dict(summary)
    missing_repeatability["env_trajectory_ordered_checksum_total"] = None
    missing_violations = module._accepted_fast_path_preset_violations(
        args,
        {"summary": missing_repeatability},
    )

    assert any("env_trajectory_ordered_checksum_total" in item for item in missing_violations)

    zero_repeatability = dict(summary)
    zero_repeatability["compact_rollout_slab_sample_gate_sample_position_order_checksum"] = 0
    zero_violations = module._accepted_fast_path_preset_violations(
        args,
        {"summary": zero_repeatability},
    )

    assert any("sample_position_order_checksum" in item for item in zero_violations)

    zero_counter = dict(summary)
    zero_counter["compact_rollout_slab_sample_gate_calls"] = 0
    zero_counter_violations = module._accepted_fast_path_preset_violations(
        args,
        {"summary": zero_counter},
    )

    assert any(
        "sample_gate_calls" in item and "positive" in item for item in zero_counter_violations
    )

    drifted_summary = dict(summary)
    drifted_summary["hybrid_borrow_single_actor_render_state"] = False
    drifted_summary["render_state_handoff_mode"] = "copy_actor_state_to_parent_buffers"
    violations = module._accepted_fast_path_preset_violations(
        args,
        {"summary": drifted_summary},
    )

    assert any("hybrid_borrow_single_actor_render_state" in item for item in violations)
    assert any("render_state_handoff_mode" in item for item in violations)


def test_modal_speed_row_cuda_sync_timing_diagnostic_violations():
    module = _load_modal_runner_module()
    args = _modal_launcher_args(compact_profile_cuda_sync_timing_diagnostics=True)
    summary = {
        "compact_profile_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled": True,
        "compact_rollout_slab_sample_gate_cuda_sync_count": 6,
        "compact_rollout_slab_sample_gate_cuda_sync_sec": 0.12,
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics"
        ): True,
        ("compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled"): True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count": 8,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec": 0.23,
        "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled": True,
        "compact_rollout_slab_learner_gate_cuda_sync_count": 10,
        "compact_rollout_slab_learner_gate_cuda_sync_sec": 0.34,
    }

    assert (
        module._cuda_sync_timing_diagnostic_violations(
            args,
            {"summary": dict(summary)},
        )
        == []
    )

    missing = dict(summary)
    missing.pop("compact_rollout_slab_sample_gate_cuda_sync_timing_enabled")
    assert any(
        "sample_gate_cuda_sync_timing_enabled" in item
        for item in module._cuda_sync_timing_diagnostic_violations(
            args,
            {"summary": missing},
        )
    )

    zero_count = dict(summary)
    zero_count["compact_rollout_slab_learner_gate_cuda_sync_count"] = 0
    assert any(
        "learner_gate_cuda_sync_count" in item and "positive" in item
        for item in module._cuda_sync_timing_diagnostic_violations(
            args,
            {"summary": zero_count},
        )
    )

    negative_sec = dict(summary)
    negative_sec["compact_rollout_slab_sample_gate_cuda_sync_sec"] = -0.01
    assert any(
        "sample_gate_cuda_sync_sec" in item and "nonnegative" in item
        for item in module._cuda_sync_timing_diagnostic_violations(
            args,
            {"summary": negative_sec},
        )
    )


def test_modal_speed_row_runtime_step_timing_diagnostic_violations():
    module = _load_modal_runner_module()
    args = _modal_launcher_args(compact_profile_runtime_step_timing_diagnostics=True)
    summary = {
        "compact_profile_runtime_step_timing_diagnostics": True,
        "compact_profile_runtime_step_count": 4,
        "compact_profile_runtime_step_sum_sec": 0.91,
        "compact_profile_runtime_step_min_sec": 0.21,
        "compact_profile_runtime_step_max_sec": 0.41,
        "compact_profile_runtime_step_p50_sec": 0.29,
        "compact_profile_runtime_step_p95_sec": 0.39,
    }

    assert (
        module._runtime_step_timing_diagnostic_violations(
            args,
            {"summary": dict(summary)},
        )
        == []
    )

    missing = dict(summary)
    missing["compact_profile_runtime_step_timing_diagnostics"] = False
    assert any(
        "runtime_step_timing_diagnostics" in item
        for item in module._runtime_step_timing_diagnostic_violations(
            args,
            {"summary": missing},
        )
    )

    zero_count = dict(summary)
    zero_count["compact_profile_runtime_step_count"] = 0
    assert any(
        "runtime_step_count" in item and "positive" in item
        for item in module._runtime_step_timing_diagnostic_violations(
            args,
            {"summary": zero_count},
        )
    )

    negative_sec = dict(summary)
    negative_sec["compact_profile_runtime_step_sum_sec"] = -0.01
    assert any(
        "runtime_step_sum_sec" in item and "nonnegative" in item
        for item in module._runtime_step_timing_diagnostic_violations(
            args,
            {"summary": negative_sec},
        )
    )


def test_modal_speed_row_cpu_perf_stat_parser_projects_numeric_fields():
    module = _load_remote_modal_producer_module()
    fields = module._parse_perf_stat_csv(
        "\n".join(
            [
                "1234.500000,msec,task-clock,1000,100.00,,",
                "2000000,,cycles,1000,100.00,,",
                "2500000,,ref-cycles,1000,100.00,,",
                "5000000,,instructions,1000,100.00,,",
                "1000000,,branches,1000,100.00,,",
                "25000,,branch-misses,1000,100.00,,",
                "400000,,cache-references,1000,100.00,,",
                "100000,,cache-misses,1000,100.00,,",
                "300000,,LLC-loads,1000,100.00,,",
                "<not supported>,,LLC-load-misses,1000,100.00,,",
                "200000,,dTLB-loads,1000,100.00,,",
                "3000,,dTLB-load-misses,1000,100.00,,",
                "0,,page-faults,1000,100.00,,",
                "4,,context-switches,1000,100.00,,",
                "1,,cpu-migrations,1000,100.00,,",
                "ignored,,unknown-event,1000,100.00,,",
            ]
        )
    )

    assert fields["compact_profile_cpu_perf_stat_diagnostics"] is True
    assert fields["compact_profile_cpu_perf_stat_parse_line_count"] == 15
    assert fields["compact_profile_cpu_perf_stat_parsed_event_count"] == 14
    assert fields["compact_profile_cpu_perf_stat_task_clock_sec"] == pytest.approx(1.2345)
    assert fields["compact_profile_cpu_perf_stat_cycles"] == 2_000_000.0
    assert fields["compact_profile_cpu_perf_stat_ref_cycles"] == 2_500_000.0
    assert fields["compact_profile_cpu_perf_stat_instructions"] == 5_000_000.0
    assert fields["compact_profile_cpu_perf_stat_instructions_per_cycle"] == (pytest.approx(2.5))
    assert fields["compact_profile_cpu_perf_stat_cache_miss_rate"] == pytest.approx(0.25)
    assert fields["compact_profile_cpu_perf_stat_llc_loads"] == 300_000.0
    assert fields["compact_profile_cpu_perf_stat_llc_load_misses_available"] is False
    assert "compact_profile_cpu_perf_stat_llc_load_misses" not in fields
    assert fields["compact_profile_cpu_perf_stat_dtlb_load_misses"] == 3_000.0
    assert fields["compact_profile_cpu_perf_stat_context_switches"] == 4.0
    assert fields["compact_profile_cpu_perf_stat_cpu_migrations"] == 1.0


def test_modal_speed_row_result_bundle_projects_cpu_perf_stat_config():
    module = _load_remote_modal_producer_module()

    assert (
        module._result_bundle_config_fields_from_config(
            {"compact_profile_cpu_perf_stat_diagnostics": True}
        )["compact_profile_cpu_perf_stat_diagnostics"]
        is True
    )
    assert (
        module._result_bundle_config_fields_from_config({})[
            "compact_profile_cpu_perf_stat_diagnostics"
        ]
        is False
    )


def test_remote_modal_owner_search_config_projects_digest_deferral():
    module = _load_remote_modal_producer_module()

    fields = module._owner_search_config_fields_from_config(
        {
            "search_service_kind": "owner_search_threaded_proxy",
            "owner_search_defer_model_state_digest_to_refresh": True,
            "owner_search_fixed_action_result_buffer": True,
            "owner_search_action_result_slot_capacity": 8,
            "owner_search_owner_local_transition_derivation": True,
            "owner_search_owner_proxy_transition_closure": True,
            "compact_owner_action_dispatch_step_overlap": True,
        }
    )

    assert fields["owner_search_threaded_proxy_requested"] is True
    assert fields["owner_search_defer_model_state_digest_to_refresh_requested"] is True
    assert fields["owner_search_fixed_action_result_buffer_requested"] is True
    assert fields["owner_search_action_result_slot_capacity_requested"] == 8
    assert fields["owner_search_owner_local_transition_derivation_requested"] is True
    assert fields["owner_search_owner_proxy_transition_closure_requested"] is True
    assert fields["compact_owner_action_dispatch_step_overlap_requested"] is True
    assert (
        module._result_bundle_config_fields_from_config(
            {
                "search_service_kind": "owner_search_threaded_proxy",
                "owner_search_defer_model_state_digest_to_refresh": True,
                "owner_search_fixed_action_result_buffer": True,
                "owner_search_action_result_slot_capacity": 8,
                "owner_search_owner_local_transition_derivation": True,
                "owner_search_owner_proxy_transition_closure": True,
                "compact_owner_action_dispatch_step_overlap": True,
            }
        )["owner_search_owner_proxy_transition_closure_requested"]
        is True
    )
    assert (
        module._result_bundle_config_fields_from_config(
            {
                "search_service_kind": "owner_search_threaded_proxy",
                "owner_search_defer_model_state_digest_to_refresh": True,
                "owner_search_fixed_action_result_buffer": True,
                "owner_search_action_result_slot_capacity": 8,
                "owner_search_owner_local_transition_derivation": True,
                "owner_search_owner_proxy_transition_closure": True,
                "compact_owner_action_dispatch_step_overlap": True,
            }
        )["compact_owner_action_dispatch_step_overlap_requested"]
        is True
    )
    assert (
        module._result_bundle_config_fields_from_config(
            {
                "search_service_kind": "owner_search_threaded_proxy",
                "owner_search_fixed_action_result_buffer": True,
                "owner_search_action_result_slot_capacity": 8,
            }
        )["owner_search_action_result_slot_capacity_requested"]
        == 8
    )
    assert (
        module._result_bundle_config_fields_from_config(
            {
                "search_service_kind": "owner_search_threaded_proxy",
                "owner_search_owner_local_transition_derivation": True,
            }
        )["owner_search_owner_local_transition_derivation_requested"]
        is True
    )


def test_remote_modal_producer_rejects_fixed_action_result_buffer_without_deferred_maintenance():
    module = _load_remote_modal_producer_module()

    with pytest.raises(
        ValueError,
        match="owner_search_fixed_action_result_buffer requires owner_search_defer_maintenance",
    ):
        module.main.info.raw_f(
            speed_row_spawn_result=True,
            unified_lifecycle_report_ref="unit/lifecycle.json",
            compact_checkpoint_ref="unit/checkpoint.pt",
            search_service_kind="owner_search_threaded_proxy",
            owner_search_slab_bypass=True,
            owner_search_require_resident_root_view=True,
            owner_search_resident_root_host_observation_stub=True,
            owner_search_direct_root_build_request=True,
            owner_search_fixed_action_result_buffer=True,
        )


def test_remote_modal_producer_rejects_owner_local_derivation_without_direct_replay():
    module = _load_remote_modal_producer_module()

    with pytest.raises(
        ValueError,
        match=(
            "owner_search_owner_local_transition_derivation requires "
            "owner_search_direct_transition_batch_replay"
        ),
    ):
        module.main.info.raw_f(
            speed_row_spawn_result=True,
            unified_lifecycle_report_ref="unit/lifecycle.json",
            compact_checkpoint_ref="unit/checkpoint.pt",
            search_service_kind="owner_search_threaded_proxy",
            owner_search_slab_bypass=True,
            owner_search_transition_batch_size=4,
            owner_search_owner_local_transition_derivation=True,
        )


def _fixed_soa_tensor_native_summary_fields() -> dict[str, Any]:
    return {
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": False,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": "fixed_soa_columns_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": 1,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": "fixed_soa_direct_gather_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": "fixed_soa_columns_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_impl": "fixed_soa_columns_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_direct_build_used": False,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec": 0.0,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested": False,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible": False,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used": False,
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count": 0,
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped": True,
        "compact_rollout_slab_sample_gate_fixed_soa_requested": True,
        "compact_rollout_slab_sample_gate_fixed_soa_used": True,
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_count": 5,
        "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_record_count": 2155,
        "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count": 62,
        "compact_rollout_slab_sample_gate_fixed_soa_table_row_count": 9770,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size": 8,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used": True,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift": True,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count": 64,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count": 2,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec": 0.001,
        "compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec": 0.001,
        "compact_rollout_slab_sample_gate_fixed_soa_total_sec": 0.002,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": (
            "fixed_soa_direct_gather"
        ),
    }


def test_modal_result_bundle_tensor_native_replay_violations():
    module = _load_remote_modal_producer_module()
    config = {
        "compact_owned_loop_fused_learner_batch": True,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
        "compact_muzero_learner_batch_tensor_native_replay": True,
    }
    summary = {
        "learner_num_unroll_steps": 2,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
        "compact_muzero_learner_batch_tensor_native_replay": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": "learner_ready_unroll2_cache_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": "maintained_unroll2_table_gather_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": "maintained_record_table_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows": 512,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec": 0.01,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec": 0.02,
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested": True,
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used": True,
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "learner_ready_unroll2_cache",
    }

    assert (
        module._tensor_native_replay_violations(
            config=config,
            summary=dict(summary),
        )
        == []
    )
    fixed_soa_summary = dict(summary)
    fixed_soa_summary.update(_fixed_soa_tensor_native_summary_fields())
    assert (
        module._tensor_native_replay_violations(
            config=config,
            summary=fixed_soa_summary,
        )
        == []
    )

    bad_config = dict(config)
    bad_config["compact_muzero_learner_batch_learner_ready_unroll2_cache"] = False
    assert any(
        "learner_ready_unroll2_cache" in item
        for item in module._tensor_native_replay_violations(
            config=bad_config,
            summary=dict(summary),
        )
    )

    bad_cases = (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used",
            False,
            "learner_ready_unroll2_cache_used",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
            1,
            "tensor_native_replay_fallback_count",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
            "generic",
            "tensor_native_replay_impl",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
            "rebuilt_on_sample_v0",
            "tensor_native_replay_table_source",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows",
            0,
            "tensor_native_replay_table_rows",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec",
            -0.01,
            "finite nonnegative seconds",
        ),
    )
    for field, value, expected in bad_cases:
        broken = dict(summary)
        broken[field] = value
        assert any(
            expected in item
            for item in module._tensor_native_replay_violations(
                config=config,
                summary=broken,
            )
        )


def test_modal_result_bundle_accepts_tensor_native_as_fused_source():
    module = _load_remote_modal_producer_module()
    config = {"compact_owned_loop_fused_learner_batch": True}
    summary = {
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
        "compact_rollout_slab_learner_gate_prebuilt_batch_used": True,
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": False,
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": False,
        "compact_rollout_slab_sample_gate_host_provider_learner_batch": False,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used": True,
    }

    assert (
        module._remote_fused_learner_batch_violations(
            config=config,
            summary=dict(summary),
        )
        == []
    )

    broken = dict(summary)
    broken["compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"] = (
        False
    )
    assert any(
        "tensor_native_replay" in item
        for item in module._remote_fused_learner_batch_violations(
            config=config,
            summary=broken,
        )
    )


def test_modal_speed_row_unroll2_specialized_builder_violations_and_report_fields():
    module = _load_modal_runner_module()
    args = _modal_launcher_args(compact_muzero_learner_batch_unroll2_specialized_builder=True)
    summary = {
        "compact_muzero_learner_batch_unroll2_specialized_builder": True,
        "learner_num_unroll_steps": 2,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl": "unroll2_specialized_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "unroll2_specialized",
    }

    assert (
        module._unroll2_specialized_builder_violations(
            args,
            {"summary": dict(summary)},
        )
        == []
    )
    assert module._unroll2_specialized_builder_proof_report_fields(summary) == {
        key: value
        for key, value in summary.items()
        if key != "compact_muzero_learner_batch_unroll2_specialized_builder"
        and key != "learner_num_unroll_steps"
    }

    learner_ready_args = _modal_launcher_args(
        compact_muzero_learner_batch_unroll2_specialized_builder=True,
        compact_muzero_learner_batch_learner_ready_unroll2_cache=True,
        compact_muzero_learner_batch_tensor_native_replay=True,
    )
    learner_ready_summary = dict(summary)
    learner_ready_summary.update(
        {
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
            "compact_muzero_learner_batch_tensor_native_replay": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used": False,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count": 0,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl": "none",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "learner_ready_unroll2_cache",
        }
    )
    assert (
        module._unroll2_specialized_builder_violations(
            learner_ready_args,
            {"summary": learner_ready_summary},
        )
        == []
    )

    stale_reason = dict(summary)
    stale_reason[
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason"
    ] = "guard_failed"
    assert any(
        "fallback_reason" in item
        for item in module._unroll2_specialized_builder_violations(
            args,
            {"summary": stale_reason},
        )
    )

    wrong_impl = dict(summary)
    wrong_impl[
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl"
    ] = "generic"
    assert any(
        "impl" in item
        for item in module._unroll2_specialized_builder_violations(
            args,
            {"summary": wrong_impl},
        )
    )


def test_modal_speed_row_tensor_native_replay_violations_and_report_fields():
    module = _load_modal_runner_module()
    args = _modal_launcher_args(
        compact_muzero_learner_batch_learner_ready_unroll2_cache=True,
        compact_muzero_learner_batch_tensor_native_replay=True,
    )
    summary = {
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
        "compact_muzero_learner_batch_tensor_native_replay": True,
        "learner_num_unroll_steps": 2,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": "learner_ready_unroll2_cache_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": "maintained_unroll2_table_gather_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": "maintained_record_table_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count": 5,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows": 512,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec": 0.01,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec": 0.02,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "learner_ready_unroll2_cache",
    }

    assert (
        module._learner_ready_unroll2_cache_violations(
            args,
            {"summary": dict(summary)},
        )
        == []
    )
    assert (
        module._tensor_native_replay_violations(
            args,
            {"summary": dict(summary)},
        )
        == []
    )
    assert module._tensor_native_replay_proof_report_fields(summary) == {
        field: summary[field]
        for field in module._TENSOR_NATIVE_REPLAY_PROOF_REPORT_FIELDS
        if field in summary
    }
    selected_maintained_summary = dict(summary)
    selected_maintained_summary.update(
        {
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": (
                "selected_maintained_record_table_gather_v1"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": (
                "selected_maintained_record_table_v1"
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count": 5,
        }
    )
    assert (
        module._tensor_native_replay_violations(
            args,
            {"summary": selected_maintained_summary},
        )
        == []
    )
    selected_maintained_missing_fast_metadata = dict(selected_maintained_summary)
    selected_maintained_missing_fast_metadata[
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used"
    ] = False
    assert any(
        "direct_fast_metadata_path_used" in item
        for item in module._tensor_native_replay_violations(
            args,
            {"summary": selected_maintained_missing_fast_metadata},
        )
    )
    fixed_soa_summary = dict(summary)
    fixed_soa_summary.update(_fixed_soa_tensor_native_summary_fields())
    assert (
        module._learner_ready_unroll2_cache_violations(
            args,
            {"summary": fixed_soa_summary},
        )
        == []
    )
    assert (
        module._tensor_native_replay_violations(
            args,
            {"summary": fixed_soa_summary},
        )
        == []
    )

    bad_cases = (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
            1,
            "fallback_count",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason",
            "guard_failed",
            "fallback_reason",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
            "generic",
            "impl",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
            "rebuilt_on_sample_v0",
            "table_source",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count",
            0,
            "table_reused_record_count",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count",
            1,
            "table_missing_record_count",
        ),
    )
    for field, value, expected in bad_cases:
        broken = dict(summary)
        broken[field] = value
        assert any(
            expected in item
            for item in module._tensor_native_replay_violations(
                args,
                {"summary": broken},
            )
        )


def test_speed_row_gpu_utilization_sampler_projects_fields():
    module = _load_smoke_module()

    sample = module._parse_nvidia_smi_utilization_row_v1(
        "2026/06/01 13:00:00.000, NVIDIA H100 80GB HBM3, 73, 42, 1234, 81559, 350.12"
    )
    assert sample == {
        "timestamp": "2026/06/01 13:00:00.000",
        "name": "NVIDIA H100 80GB HBM3",
        "gpu_util_percent": 73.0,
        "memory_util_percent": 42.0,
        "memory_used_mib": 1234.0,
        "memory_total_mib": 81559.0,
        "power_draw_w": 350.12,
    }

    fields = module._gpu_utilization_sampling_fields(
        {
            "speed_row_gpu_utilization_sampling": {
                "enabled": True,
                "interval_sec": 0.25,
                "sample_count": 2,
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "max_gpu_util_percent": 91.0,
                "mean_gpu_util_percent": 64.5,
                "gpu_util_nonzero_sample_count": 2,
                "gpu_util_over_50_sample_count": 1,
                "gpu_util_over_80_sample_count": 1,
                "max_memory_util_percent": 44.0,
                "max_memory_used_mib": 5678.0,
                "max_power_draw_w": 501.25,
                "errors": ["unit"],
            }
        }
    )

    assert fields == {
        "speed_row_gpu_utilization_sampling_enabled": True,
        "speed_row_gpu_utilization_sample_interval_sec": 0.25,
        "speed_row_gpu_utilization_sample_count": 2,
        "speed_row_gpu_name": "NVIDIA H100 80GB HBM3",
        "speed_row_gpu_utilization_max_percent": 91.0,
        "speed_row_gpu_utilization_mean_percent": 64.5,
        "speed_row_gpu_utilization_nonzero_sample_count": 2,
        "speed_row_gpu_utilization_over_50_sample_count": 1,
        "speed_row_gpu_utilization_over_80_sample_count": 1,
        "speed_row_gpu_memory_utilization_max_percent": 44.0,
        "speed_row_gpu_memory_used_max_mib": 5678.0,
        "speed_row_gpu_power_draw_max_w": 501.25,
        "speed_row_gpu_utilization_sampling_errors": ["unit"],
    }


def test_modal_speed_row_report_projects_sample_learner_child_timers():
    module = _load_modal_runner_module()
    runtime_step_fields = {
        "compact_profile_runtime_step_timing_diagnostics": True,
        "compact_profile_runtime_step_count": 3,
        "compact_profile_runtime_step_sum_sec": 0.91,
        "compact_profile_runtime_step_min_sec": 0.21,
        "compact_profile_runtime_step_max_sec": 0.41,
        "compact_profile_runtime_step_p50_sec": 0.29,
        "compact_profile_runtime_step_p95_sec": 0.39,
        "compact_profile_runtime_step_slowest_iteration": 4,
        "compact_profile_runtime_step_slowest_measured_iteration": 3,
        "compact_profile_runtime_step_slowest_actor_step_wall_sec": 0.13,
        "compact_profile_runtime_step_slowest_observation_sec": 0.07,
        "compact_profile_runtime_step_slowest_compact_rollout_slab_sec": 0.03,
        "compact_profile_runtime_step_slowest_sample_gate_sec": 0.11,
        "compact_profile_runtime_step_slowest_learner_gate_sec": 0.05,
        "compact_profile_runtime_step_slowest_policy_refresh_sec": 0.01,
        "compact_profile_runtime_step_slowest_primary_accounted_sec": 0.4,
        "compact_profile_runtime_step_slowest_primary_residual_sec": 0.01,
        "compact_profile_runtime_step_slowest_env_trajectory_checksum": 123456,
    }
    sample_gate_per_call_fields = {
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call_count": 2,
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call_sum_sec": 0.12,
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call_min_sec": 0.05,
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call_max_sec": 0.07,
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call_p50_sec": 0.06,
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call_p95_sec": 0.069,
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call_slowest_call_index": 2,
        "compact_rollout_slab_sample_gate_per_call_count": 2,
        "compact_rollout_slab_sample_gate_per_call_sum_sec": 0.22,
        "compact_rollout_slab_sample_gate_per_call_min_sec": 0.1,
        "compact_rollout_slab_sample_gate_per_call_max_sec": 0.12,
        "compact_rollout_slab_sample_gate_per_call_p50_sec": 0.11,
        "compact_rollout_slab_sample_gate_per_call_p95_sec": 0.119,
        "compact_rollout_slab_sample_gate_candidate_per_call_count": 2,
        "compact_rollout_slab_sample_gate_candidate_per_call_sum_sec": 0.07,
        "compact_rollout_slab_sample_gate_candidate_per_call_min_sec": 0.03,
        "compact_rollout_slab_sample_gate_candidate_per_call_max_sec": 0.04,
        "compact_rollout_slab_sample_gate_candidate_per_call_p50_sec": 0.035,
        "compact_rollout_slab_sample_gate_candidate_per_call_p95_sec": 0.0395,
        "compact_rollout_slab_sample_gate_rng_per_call_count": 2,
        "compact_rollout_slab_sample_gate_rng_per_call_sum_sec": 0.03,
        "compact_rollout_slab_sample_gate_rng_per_call_min_sec": 0.01,
        "compact_rollout_slab_sample_gate_rng_per_call_max_sec": 0.02,
        "compact_rollout_slab_sample_gate_rng_per_call_p50_sec": 0.015,
        "compact_rollout_slab_sample_gate_rng_per_call_p95_sec": 0.0195,
        "compact_rollout_slab_sample_gate_residual_per_call_count": 2,
        "compact_rollout_slab_sample_gate_residual_per_call_sum_sec": 0.05,
        "compact_rollout_slab_sample_gate_residual_per_call_min_sec": 0.02,
        "compact_rollout_slab_sample_gate_residual_per_call_max_sec": 0.03,
        "compact_rollout_slab_sample_gate_residual_per_call_p50_sec": 0.025,
        "compact_rollout_slab_sample_gate_residual_per_call_p95_sec": 0.0295,
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_count": 2,
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_sum_sec": 0.08,
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_p50_sec": 0.04,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_per_call_p95_sec": 0.031,
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_per_call_p95_sec": 0.021,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_per_call_p95_sec": 0.017,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call_p95_sec": 0.016,
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call_p95_sec": 0.013,
        "compact_rollout_slab_sample_gate_learner_batch_builder_write_output_per_call_max_sec": 0.012,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_per_call_sum_sec": 0.024,
    }
    fields = module._sample_learner_timer_report_fields(
        {
            "compact_rollout_slab_sample_gate_candidate_sec": 0.01,
            "compact_rollout_slab_sample_gate_learner_batch_build_sec": 0.02,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_sec": 0.0201,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "unroll_terminal_window_hint_sec"
            ): 0.02015,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_sec": 0.0202,
            "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_sec": 0.021,
            "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_sec": 0.022,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_process_cpu_time_delta_ns": 2500,
            "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics": True,
            "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled": True,
            "compact_rollout_slab_sample_gate_cuda_sync_count": 6,
            "compact_rollout_slab_sample_gate_cuda_sync_sec": 0.023,
            "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count": 8,
            "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec": 0.024,
            "compact_rollout_slab_learner_gate_validation_sec": 0.03,
            "compact_rollout_slab_learner_gate_backward_sec": 0.04,
            "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics": True,
            "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled": True,
            "compact_rollout_slab_learner_gate_cuda_sync_count": 10,
            "compact_rollout_slab_learner_gate_cuda_sync_sec": 0.05,
            **runtime_step_fields,
            **sample_gate_per_call_fields,
            "unrelated": 99,
        }
    )

    assert fields == {
        "compact_rollout_slab_sample_gate_candidate_sec": 0.01,
        "compact_rollout_slab_sample_gate_learner_batch_build_sec": 0.02,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_sec": 0.0201,
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_sec"
        ): 0.02015,
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_sec": 0.0202,
        "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_sec": 0.021,
        "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_sec": 0.022,
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_process_cpu_time_delta_ns": 2500,
        "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled": True,
        "compact_rollout_slab_sample_gate_cuda_sync_count": 6,
        "compact_rollout_slab_sample_gate_cuda_sync_sec": 0.023,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count": 8,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec": 0.024,
        "compact_rollout_slab_learner_gate_validation_sec": 0.03,
        "compact_rollout_slab_learner_gate_backward_sec": 0.04,
        "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled": True,
        "compact_rollout_slab_learner_gate_cuda_sync_count": 10,
        "compact_rollout_slab_learner_gate_cuda_sync_sec": 0.05,
        **runtime_step_fields,
        **sample_gate_per_call_fields,
    }


def test_modal_speed_row_report_projects_compact_rollout_slab_totals():
    module = _load_modal_runner_module()
    totals = {
        "compact_rollout_slab_owner_search_parent_wait_sec": 0.21,
        "compact_rollout_slab_owner_search_worker_learner_train_sec": 0.31,
    }
    fields = module._compact_rollout_slab_total_report_fields(
        {
            "compact_rollout_slab_telemetry_totals": totals,
            "speed_row_total_owner_search_parent_wait_sec": 0.21,
            "speed_row_total_owner_search_worker_learner_train_sec": 0.31,
            "unrelated": 99,
        }
    )

    assert fields == {
        "compact_rollout_slab_telemetry_totals": totals,
        "speed_row_total_owner_search_parent_wait_sec": 0.21,
        "speed_row_total_owner_search_worker_learner_train_sec": 0.31,
    }


def test_modal_speed_row_report_projects_sample_learner_transport_proof_fields():
    module = _load_modal_runner_module()
    fields = module._sample_learner_transport_proof_report_fields(
        {
            "compact_owned_loop_sample_learner_worker_bootstrap_source": "factory",
            "compact_owned_loop_deferred_sample_learner_request_host_only": True,
            "compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count": 0,
            "compact_owned_loop_deferred_sample_learner_result_host_only": True,
            "compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count": 0,
            "compact_owned_loop_deferred_sample_learner_snapshot_host_clone_used": True,
            "compact_owned_loop_deferred_sample_learner_request_bytes": 123,
            "compact_owned_loop_deferred_sample_learner_result_bytes": 456,
            "compact_owned_loop_deferred_sample_learner_worker_owns_model_state": True,
            "compact_owned_loop_deferred_sample_learner_worker_owns_replay_store": True,
            "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent": False,
            ("compact_owned_loop_deferred_sample_learner_full_replay_snapshot_submit_count"): 0,
            "compact_owned_loop_deferred_sample_learner_replay_append_entry_count": 7,
            "compact_owned_loop_deferred_sample_learner_replay_append_index_row_count": 14,
            "compact_owned_loop_deferred_sample_learner_replay_append_entry_bytes": 1000,
            ("compact_owned_loop_deferred_sample_learner_replay_append_host_observation_bytes"): 0,
            ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_count"): 0,
            ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_bytes"): 0,
            ("compact_owned_loop_deferred_sample_learner_replay_append_compact_batch_bytes"): 12,
            ("compact_owned_loop_deferred_sample_learner_replay_append_step_payload_bytes"): 34,
            ("compact_owned_loop_deferred_sample_learner_replay_append_render_state_bytes"): 56,
            ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_count"): 3,
            ("compact_owned_loop_deferred_sample_learner_last_provider_bootstrap_step_count"): 1,
            ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_bytes"): 78,
            (
                "compact_owned_loop_deferred_sample_learner_"
                "provider_bootstrap_host_observation_bytes"
            ): 0,
            (
                "compact_owned_loop_deferred_sample_learner_"
                "provider_bootstrap_resident_snapshot_count"
            ): 0,
            (
                "compact_owned_loop_deferred_sample_learner_"
                "provider_bootstrap_resident_snapshot_bytes"
            ): 0,
            (
                "compact_owned_loop_deferred_sample_learner_provider_bootstrap_render_state_bytes"
            ): 90,
            ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_entry_count"): 0,
            (
                "compact_owned_loop_deferred_sample_learner_"
                "provider_bootstrap_replay_index_row_count"
            ): 0,
            ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_learner_call_count"): 0,
            (
                "compact_owned_loop_deferred_sample_learner_worker_observation_provider_present"
            ): True,
            (
                "compact_owned_loop_deferred_sample_learner_"
                "worker_observation_provider_bootstrap_step_count"
            ): 3,
            (
                "compact_owned_loop_deferred_sample_learner_"
                "worker_last_observation_provider_bootstrap_step_count"
            ): 1,
            (
                "compact_owned_loop_deferred_sample_learner_"
                "worker_observation_provider_missing_stack_history_count"
            ): 0,
            (
                "compact_owned_loop_deferred_sample_learner_"
                "worker_observation_provider_materialized_entry_count"
            ): 7,
            (
                "compact_owned_loop_deferred_sample_learner_"
                "worker_last_observation_provider_materialized_entry_count"
            ): 2,
            ("compact_owned_loop_deferred_sample_learner_last_replay_append_entry_count"): 1,
            ("compact_owned_loop_deferred_sample_learner_last_replay_append_index_row_count"): 2,
            "compact_owned_loop_deferred_sample_learner_worker_model_initialized_count": 1,
            "compact_owned_loop_deferred_sample_learner_worker_completed_count": 4,
            "compact_owned_loop_deferred_sample_learner_worker_replay_append_count": 7,
            "compact_owned_loop_deferred_sample_learner_worker_replay_entry_count": 7,
            "compact_owned_loop_deferred_sample_learner_worker_replay_index_row_count": 14,
            ("compact_owned_loop_deferred_sample_learner_worker_replay_evicted_entry_count"): 0,
            ("compact_owned_loop_deferred_sample_learner_worker_replay_evicted_index_row_count"): 0,
            "unrelated": 99,
        }
    )

    assert fields == {
        "compact_owned_loop_sample_learner_worker_bootstrap_source": "factory",
        "compact_owned_loop_deferred_sample_learner_request_host_only": True,
        "compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count": 0,
        "compact_owned_loop_deferred_sample_learner_result_host_only": True,
        "compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count": 0,
        "compact_owned_loop_deferred_sample_learner_snapshot_host_clone_used": True,
        "compact_owned_loop_deferred_sample_learner_request_bytes": 123,
        "compact_owned_loop_deferred_sample_learner_result_bytes": 456,
        "compact_owned_loop_deferred_sample_learner_worker_owns_model_state": True,
        "compact_owned_loop_deferred_sample_learner_worker_owns_replay_store": True,
        "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent": False,
        "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_submit_count": 0,
        "compact_owned_loop_deferred_sample_learner_replay_append_entry_count": 7,
        "compact_owned_loop_deferred_sample_learner_replay_append_index_row_count": 14,
        "compact_owned_loop_deferred_sample_learner_replay_append_entry_bytes": 1000,
        ("compact_owned_loop_deferred_sample_learner_replay_append_host_observation_bytes"): 0,
        ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_count"): 0,
        ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_bytes"): 0,
        ("compact_owned_loop_deferred_sample_learner_replay_append_compact_batch_bytes"): 12,
        ("compact_owned_loop_deferred_sample_learner_replay_append_step_payload_bytes"): 34,
        ("compact_owned_loop_deferred_sample_learner_replay_append_render_state_bytes"): 56,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_count"): 3,
        ("compact_owned_loop_deferred_sample_learner_last_provider_bootstrap_step_count"): 1,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_bytes"): 78,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_host_observation_bytes"): 0,
        (
            "compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_count"
        ): 0,
        (
            "compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_bytes"
        ): 0,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_render_state_bytes"): 90,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_entry_count"): 0,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_index_row_count"): 0,
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_learner_call_count"): 0,
        ("compact_owned_loop_deferred_sample_learner_worker_observation_provider_present"): True,
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_bootstrap_step_count"
        ): 3,
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_last_observation_provider_bootstrap_step_count"
        ): 1,
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_missing_stack_history_count"
        ): 0,
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_materialized_entry_count"
        ): 7,
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_last_observation_provider_materialized_entry_count"
        ): 2,
        "compact_owned_loop_deferred_sample_learner_last_replay_append_entry_count": 1,
        "compact_owned_loop_deferred_sample_learner_last_replay_append_index_row_count": 2,
        "compact_owned_loop_deferred_sample_learner_worker_model_initialized_count": 1,
        "compact_owned_loop_deferred_sample_learner_worker_completed_count": 4,
        "compact_owned_loop_deferred_sample_learner_worker_replay_append_count": 7,
        "compact_owned_loop_deferred_sample_learner_worker_replay_entry_count": 7,
        "compact_owned_loop_deferred_sample_learner_worker_replay_index_row_count": 14,
        "compact_owned_loop_deferred_sample_learner_worker_replay_evicted_entry_count": 0,
        ("compact_owned_loop_deferred_sample_learner_worker_replay_evicted_index_row_count"): 0,
    }


def test_modal_speed_row_report_projects_owner_search_slab_proxy_fields():
    module = _load_modal_runner_module()
    source = {
        **_owner_search_slab_proxy_fields(),
        "compact_owner_mechanics_step_frame_slot_schema_id": (
            "curvyzero_compact_owner_mechanics_step_frame_slot/v1"
        ),
        "compact_owner_mechanics_step_frame_handle_ring_used": True,
        "compact_owner_mechanics_step_frame_slot_write_count": 2,
        "compact_owner_mechanics_parent_step_frame_build_count": 0,
        "compact_owner_step_frame_root_build_request_used": True,
        "compact_owner_step_frame_root_build_request_from_batch_helper_used": False,
        "compact_owner_step_frame_root_request_sidecar_array_bytes": 0,
        "compact_owner_step_frame_root_request_sidecar_field_count": 0,
        "compact_owner_root_action_context_handle_used": True,
        "compact_owner_root_action_context_handle_schema_id": (
            "curvyzero_compact_owner_root_action_context_handle/v1"
        ),
        "compact_owner_root_action_context_handle_id": 2,
        "compact_owner_root_action_context_transaction_id": 2,
        "compact_owner_root_action_context_dispatch_id": 2,
        "compact_owner_root_action_context_root_count": 4,
        "compact_owner_root_action_context_active_root_count": 4,
        "compact_owner_root_action_context_context_digest": "root-action-context-digest",
        "compact_owner_root_action_context_owner_store_count": 2,
        "compact_owner_root_action_context_owner_resolve_count": 2,
        "compact_owner_root_action_context_owner_release_count": 2,
        "compact_owner_root_action_context_owner_pending_count": 0,
        "compact_owner_root_action_context_owner_max_pending_count": 1,
        "compact_owner_root_action_context_owner_digest_verified": True,
        "compact_owner_search_pending_root_action_context_stored": False,
        "compact_owner_search_action_dispatch_pending_root_action_context_stored": False,
        "compact_owner_search_action_dispatch_pending_root_action_context_store_count": 0,
        "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count": 2,
        "compact_owner_search_parent_action_context_validation_count": 0,
        "compact_owner_search_owner_action_context_validation_count": 2,
        "compact_owner_root_search_transaction_used": True,
        "compact_owner_root_search_transaction_begin_count": 2,
        "compact_owner_root_search_transaction_submit_count": 2,
        "compact_owner_root_search_transaction_resolve_count": 2,
        "compact_owner_root_search_transaction_pending_count": 0,
        "compact_owner_root_search_transaction_parent_root_request_build_count": 0,
        "compact_owner_root_search_transaction_parent_root_request_stored": False,
        "compact_owner_root_search_transaction_parent_compact_batch_stored": False,
        "compact_owner_root_search_transaction_parent_rebuild_count": 0,
        "compact_owner_root_search_transaction_parent_root_action_context_stored": False,
        "compact_owner_root_search_transaction_parent_root_action_context_store_count": 0,
        "compact_owner_root_search_transaction_parent_root_action_context_array_bytes": 0,
        "compact_owner_root_search_transaction_parent_root_action_context_field_count": 0,
        "compact_owner_root_search_transaction_owner_root_request_build_count": 2,
        "compact_owner_root_search_transaction_owner_root_store_publish_count": 2,
        "compact_owner_root_search_transaction_frame_generation_verified": True,
        "compact_owner_root_search_transaction_frame_digest_verified": True,
        "compact_owner_root_search_transaction_action_identity_verified": True,
        "compact_owner_root_search_transaction_applied_action_mismatch_count": 0,
        "unrelated": 99,
    }
    fields = module._owner_search_slab_proxy_proof_report_fields(source)

    assert fields == {
        key: source[key]
        for key in module._OWNER_SEARCH_SLAB_PROXY_PROOF_REPORT_FIELDS
        if key in source
    }
    for key in (
        "compact_owner_mechanics_step_frame_slot_schema_id",
        "compact_owner_mechanics_step_frame_handle_ring_used",
        "compact_owner_mechanics_step_frame_slot_write_count",
        "compact_owner_mechanics_parent_step_frame_build_count",
        "compact_owner_step_frame_root_build_request_used",
        "compact_owner_step_frame_root_build_request_from_batch_helper_used",
        "compact_owner_step_frame_root_request_sidecar_array_bytes",
        "compact_owner_step_frame_root_request_sidecar_field_count",
        "compact_owner_root_action_context_handle_used",
        "compact_owner_root_action_context_handle_schema_id",
        "compact_owner_root_action_context_handle_id",
        "compact_owner_root_action_context_transaction_id",
        "compact_owner_root_action_context_dispatch_id",
        "compact_owner_root_action_context_root_count",
        "compact_owner_root_action_context_active_root_count",
        "compact_owner_root_action_context_context_digest",
        "compact_owner_root_action_context_owner_store_count",
        "compact_owner_root_action_context_owner_resolve_count",
        "compact_owner_root_action_context_owner_release_count",
        "compact_owner_root_action_context_owner_pending_count",
        "compact_owner_root_action_context_owner_max_pending_count",
        "compact_owner_root_action_context_owner_digest_verified",
        "compact_owner_search_pending_root_action_context_stored",
        "compact_owner_search_action_dispatch_pending_root_action_context_stored",
        "compact_owner_search_action_dispatch_pending_root_action_context_store_count",
        "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count",
        "compact_owner_search_parent_action_context_validation_count",
        "compact_owner_search_owner_action_context_validation_count",
        "compact_owner_root_search_transaction_used",
        "compact_owner_root_search_transaction_begin_count",
        "compact_owner_root_search_transaction_submit_count",
        "compact_owner_root_search_transaction_resolve_count",
        "compact_owner_root_search_transaction_pending_count",
        "compact_owner_root_search_transaction_parent_root_request_build_count",
        "compact_owner_root_search_transaction_parent_root_request_stored",
        "compact_owner_root_search_transaction_parent_compact_batch_stored",
        "compact_owner_root_search_transaction_parent_rebuild_count",
        "compact_owner_root_search_transaction_parent_root_action_context_stored",
        "compact_owner_root_search_transaction_parent_root_action_context_store_count",
        "compact_owner_root_search_transaction_parent_root_action_context_array_bytes",
        "compact_owner_root_search_transaction_parent_root_action_context_field_count",
        "compact_owner_root_search_transaction_owner_root_request_build_count",
        "compact_owner_root_search_transaction_owner_root_store_publish_count",
        "compact_owner_root_search_transaction_frame_generation_verified",
        "compact_owner_root_search_transaction_frame_digest_verified",
        "compact_owner_root_search_transaction_action_identity_verified",
        "compact_owner_root_search_transaction_applied_action_mismatch_count",
    ):
        assert fields[key] == source[key]
    assert "unrelated" not in fields


def test_speed_row_smoke_backend_factory_selects_floor_decomposition_services():
    module = _load_smoke_module()

    fixed = module._build_search_service(
        args=SimpleNamespace(
            search_service_kind="fixed_shape_search_owner",
            batch_size=2,
            num_simulations=3,
            seed=20260530,
        ),
        model=None,
        device="cpu",
        loaded_checkpoint_identity={},
    )
    assert fixed.search_impl == "fixed_shape_batched_search_owner_profile_only_v0"
    assert fixed.supports_two_phase_compact_search is True
    assert hasattr(fixed, "flush_device_replay_payload")

    compact_torch = module._build_search_service(
        args=SimpleNamespace(
            search_service_kind="compact_torch_search_service",
            batch_size=2,
            num_simulations=3,
            seed=20260530,
            compact_torch_request_compile=False,
            compact_torch_request_model_compile=False,
            compact_torch_model_compile_mode="default",
            compact_torch_timing_mode="host_phase_sync",
            compact_torch_initial_inference_mode="direct_core",
            compact_torch_observation_memory_format="channels_last",
            compact_torch_model_memory_format="contiguous",
            compact_torch_defer_one_simulation_replay_payload=True,
            run_id="unit-speed-row-backend",
        ),
        model=None,
        device="cpu",
        loaded_checkpoint_identity={},
    )
    assert compact_torch.search_impl == "compact_torch_device_tree_fixed_shape_v0"
    assert compact_torch.supports_two_phase_compact_search is True
    assert hasattr(compact_torch, "flush_device_replay_payload")
    assert compact_torch._compile_config.model_compile_mode == "default"
    assert compact_torch._compile_config.observation_memory_format == "channels_last"
    assert compact_torch._compile_config.model_memory_format == "contiguous"
    assert compact_torch._compile_config.request_model_compile is False
    assert compact_torch._compile_config.initial_inference_mode == "direct_core"
    assert compact_torch._compile_config.defer_one_simulation_replay_payload is True

    with pytest.raises(ValueError, match="requires compact_torch inner search"):
        module._build_search_service(
            args=SimpleNamespace(
                search_service_kind="owner_search_slab_proxy",
                owner_search_inner_search_service_kind="fixed_shape_search_owner",
                batch_size=2,
                num_simulations=3,
                seed=20260530,
                run_id="unit-owner-search-backend",
            ),
            model=None,
            device="cpu",
            loaded_checkpoint_identity={},
        )

    owner_proxy = module._build_search_service(
        args=SimpleNamespace(
            search_service_kind="owner_search_slab_proxy",
            owner_search_inner_search_service_kind="compact_torch_search_service",
            batch_size=2,
            num_simulations=3,
            seed=20260530,
            run_id="unit-owner-search-backend",
            compact_torch_request_compile=False,
            compact_torch_request_model_compile=False,
            compact_torch_model_compile_mode="default",
            compact_torch_timing_mode="host_phase_sync",
            compact_torch_initial_inference_mode="direct_core",
            compact_torch_observation_memory_format="channels_last",
            compact_torch_model_memory_format="contiguous",
            learner_device="cpu",
            sample_interval=1,
            replay_pair_capacity=16,
            learner_train_steps=1,
            learner_num_unroll_steps=1,
            death_mode="profile_no_death",
            sample_batch_size=2,
        ),
        model=None,
        device="cpu",
        loaded_checkpoint_identity={},
    )
    assert isinstance(owner_proxy, module.CompactLazyOwnerSearchSlabProxyV1)
    assert owner_proxy.metadata["compact_owner_search_slab_proxy"] is True
    assert owner_proxy.metadata["compact_owner_search_lazy_slab_proxy"] is True
    assert owner_proxy.metadata["compact_owner_search_slab_proxy_initialized"] is False
    assert owner_proxy.metadata["compact_owner_search_owner_replay_append_enabled"] is True
    assert owner_proxy.metadata["compact_owner_search_owner_train_interval"] == 2
    assert module._search_service_impl("owner_search_slab_proxy") == (
        "compact_owner_search_slab_proxy_v1"
    )
    assert module._owner_search_config_fields(
        SimpleNamespace(
            search_service_kind="owner_search_slab_proxy",
            owner_search_inner_search_service_kind="fixed_shape_search_owner",
        )
    ) == {
        "owner_search_slab_proxy_requested": True,
        "owner_search_inline_proxy_requested": False,
        "owner_search_inline_background_proxy_requested": False,
        "owner_search_threaded_proxy_requested": False,
        "owner_search_inner_search_service_kind": "fixed_shape_search_owner",
        "owner_search_inner_search_service_impl": (
            "fixed_shape_batched_search_owner_profile_only_v0"
        ),
        "owner_search_compact_torch_resident_root_bridge_ready": False,
        "owner_search_defer_maintenance_requested": False,
        "owner_search_slab_bypass_requested": False,
        "owner_search_transition_batch_size_requested": 1,
        "owner_search_transition_batch_transport_requested": False,
        "owner_search_direct_transition_batch_replay_requested": False,
        "owner_search_owner_local_transition_derivation_requested": False,
        "owner_search_owner_proxy_transition_closure_requested": False,
        "owner_search_require_resident_root_view_requested": False,
        "owner_search_resident_root_host_observation_stub_requested": False,
        "owner_search_direct_root_build_request_requested": False,
        "compact_owner_action_step_boundary_requested": False,
        "compact_owner_action_dispatch_step_overlap_requested": False,
        "owner_search_fixed_action_result_buffer_requested": False,
        "owner_search_action_result_slot_capacity_requested": 4,
        "owner_search_async_learner_worker_requested": False,
        "owner_search_async_learner_worker_kind_requested": "in_process_thread_v1",
        "owner_search_async_learner_max_pending_requested": 1,
    }

    owner_compact_torch_proxy = module._build_search_service(
        args=SimpleNamespace(
            search_service_kind="owner_search_slab_proxy",
            owner_search_inner_search_service_kind="compact_torch_search_service",
            batch_size=2,
            num_simulations=3,
            seed=20260530,
            run_id="unit-owner-search-compact-torch-backend",
            compact_torch_request_compile=False,
            compact_torch_request_model_compile=False,
            compact_torch_model_compile_mode="default",
            compact_torch_timing_mode="host_phase_sync",
            compact_torch_initial_inference_mode="direct_core",
            compact_torch_observation_memory_format="contiguous",
            compact_torch_model_memory_format="contiguous",
            learner_device="cpu",
            sample_interval=1,
            replay_pair_capacity=16,
            learner_train_steps=1,
            learner_num_unroll_steps=1,
            death_mode="profile_no_death",
            sample_batch_size=2,
        ),
        model=None,
        device="cpu",
        loaded_checkpoint_identity={},
    )
    assert isinstance(owner_compact_torch_proxy, module.CompactLazyOwnerSearchSlabProxyV1)
    assert (
        owner_compact_torch_proxy.metadata["compact_owner_search_use_inner_two_phase_device_replay"]
        is True
    )
    assert module._owner_search_config_fields(
        SimpleNamespace(
            search_service_kind="owner_search_slab_proxy",
            owner_search_inner_search_service_kind="compact_torch_search_service",
        )
    ) == {
        "owner_search_slab_proxy_requested": True,
        "owner_search_inline_proxy_requested": False,
        "owner_search_inline_background_proxy_requested": False,
        "owner_search_threaded_proxy_requested": False,
        "owner_search_inner_search_service_kind": "compact_torch_search_service",
        "owner_search_inner_search_service_impl": "compact_torch_device_tree_fixed_shape_v0",
        "owner_search_compact_torch_resident_root_bridge_ready": True,
        "owner_search_defer_maintenance_requested": False,
        "owner_search_slab_bypass_requested": False,
        "owner_search_transition_batch_size_requested": 1,
        "owner_search_transition_batch_transport_requested": False,
        "owner_search_direct_transition_batch_replay_requested": False,
        "owner_search_owner_local_transition_derivation_requested": False,
        "owner_search_owner_proxy_transition_closure_requested": False,
        "owner_search_require_resident_root_view_requested": False,
        "owner_search_resident_root_host_observation_stub_requested": False,
        "owner_search_direct_root_build_request_requested": False,
        "compact_owner_action_step_boundary_requested": False,
        "compact_owner_action_dispatch_step_overlap_requested": False,
        "owner_search_fixed_action_result_buffer_requested": False,
        "owner_search_action_result_slot_capacity_requested": 4,
        "owner_search_async_learner_worker_requested": False,
        "owner_search_async_learner_worker_kind_requested": "in_process_thread_v1",
        "owner_search_async_learner_max_pending_requested": 1,
    }
    threaded_owner_proxy = module._build_search_service(
        args=SimpleNamespace(
            search_service_kind="owner_search_threaded_proxy",
            owner_search_inner_search_service_kind="compact_torch_search_service",
            batch_size=2,
            num_simulations=3,
            seed=20260530,
            run_id="unit-owner-search-threaded-backend",
            compact_torch_request_compile=False,
            compact_torch_request_model_compile=False,
            compact_torch_model_compile_mode="default",
            compact_torch_timing_mode="host_phase_sync",
            compact_torch_initial_inference_mode="direct_core",
            compact_torch_observation_memory_format="contiguous",
            compact_torch_model_memory_format="contiguous",
            learner_device="cpu",
            sample_interval=1,
            replay_pair_capacity=16,
            learner_train_steps=1,
            learner_num_unroll_steps=1,
            death_mode="profile_no_death",
            sample_batch_size=2,
            owner_search_defer_maintenance=True,
        ),
        model=None,
        device="cpu",
        loaded_checkpoint_identity={},
    )
    assert isinstance(
        threaded_owner_proxy,
        module.CompactLazyThreadedOwnerSearchSlabProxyV1,
    )
    assert (
        threaded_owner_proxy.metadata["compact_owner_search_use_inner_two_phase_device_replay"]
        is True
    )
    assert threaded_owner_proxy.metadata["compact_owner_search_threaded_slab_proxy"] is True
    assert module._search_service_impl("owner_search_threaded_proxy") == (
        "compact_owner_search_threaded_proxy_v1"
    )
    assert module._owner_search_config_fields(
        SimpleNamespace(
            search_service_kind="owner_search_threaded_proxy",
            owner_search_inner_search_service_kind="compact_torch_search_service",
            owner_search_defer_maintenance=True,
        )
    ) == {
        "owner_search_slab_proxy_requested": True,
        "owner_search_inline_proxy_requested": False,
        "owner_search_inline_background_proxy_requested": False,
        "owner_search_threaded_proxy_requested": True,
        "owner_search_inner_search_service_kind": "compact_torch_search_service",
        "owner_search_inner_search_service_impl": "compact_torch_device_tree_fixed_shape_v0",
        "owner_search_compact_torch_resident_root_bridge_ready": False,
        "owner_search_defer_maintenance_requested": True,
        "owner_search_slab_bypass_requested": False,
        "owner_search_transition_batch_size_requested": 1,
        "owner_search_transition_batch_transport_requested": False,
        "owner_search_direct_transition_batch_replay_requested": False,
        "owner_search_owner_local_transition_derivation_requested": False,
        "owner_search_owner_proxy_transition_closure_requested": False,
        "owner_search_require_resident_root_view_requested": False,
        "owner_search_resident_root_host_observation_stub_requested": False,
        "owner_search_direct_root_build_request_requested": False,
        "compact_owner_action_step_boundary_requested": False,
        "compact_owner_action_dispatch_step_overlap_requested": False,
        "owner_search_fixed_action_result_buffer_requested": False,
        "owner_search_action_result_slot_capacity_requested": 4,
        "owner_search_async_learner_worker_requested": False,
        "owner_search_async_learner_worker_kind_requested": "in_process_thread_v1",
        "owner_search_async_learner_max_pending_requested": 1,
    }
    inline_background_owner_proxy = module._build_search_service(
        args=SimpleNamespace(
            search_service_kind="owner_search_inline_background_proxy",
            owner_search_inner_search_service_kind="compact_torch_search_service",
            batch_size=2,
            num_simulations=3,
            seed=20260530,
            run_id="unit-owner-search-inline-background-backend",
            compact_torch_request_compile=False,
            compact_torch_request_model_compile=False,
            compact_torch_model_compile_mode="default",
            compact_torch_timing_mode="host_phase_sync",
            compact_torch_initial_inference_mode="direct_core",
            compact_torch_observation_memory_format="contiguous",
            compact_torch_model_memory_format="contiguous",
            learner_device="cpu",
            sample_interval=1,
            replay_pair_capacity=16,
            learner_train_steps=1,
            learner_num_unroll_steps=1,
            death_mode="profile_no_death",
            sample_batch_size=2,
            owner_search_defer_maintenance=True,
        ),
        model=None,
        device="cpu",
        loaded_checkpoint_identity={},
    )
    assert isinstance(
        inline_background_owner_proxy,
        module.CompactLazyInlineBackgroundOwnerSearchSlabProxyV1,
    )
    assert (
        inline_background_owner_proxy.metadata[
            "compact_owner_search_use_inner_two_phase_device_replay"
        ]
        is True
    )
    assert (
        inline_background_owner_proxy.metadata["compact_owner_search_inline_background_slab_proxy"]
        is True
    )
    assert module._search_service_impl("owner_search_inline_background_proxy") == (
        "compact_owner_search_inline_background_proxy_v1"
    )
    assert module._owner_search_config_fields(
        SimpleNamespace(
            search_service_kind="owner_search_inline_background_proxy",
            owner_search_inner_search_service_kind="compact_torch_search_service",
            owner_search_defer_maintenance=True,
        )
    ) == {
        "owner_search_slab_proxy_requested": True,
        "owner_search_inline_proxy_requested": False,
        "owner_search_inline_background_proxy_requested": True,
        "owner_search_threaded_proxy_requested": False,
        "owner_search_inner_search_service_kind": "compact_torch_search_service",
        "owner_search_inner_search_service_impl": "compact_torch_device_tree_fixed_shape_v0",
        "owner_search_compact_torch_resident_root_bridge_ready": False,
        "owner_search_defer_maintenance_requested": True,
        "owner_search_slab_bypass_requested": False,
        "owner_search_transition_batch_size_requested": 1,
        "owner_search_transition_batch_transport_requested": False,
        "owner_search_direct_transition_batch_replay_requested": False,
        "owner_search_owner_local_transition_derivation_requested": False,
        "owner_search_owner_proxy_transition_closure_requested": False,
        "owner_search_require_resident_root_view_requested": False,
        "owner_search_resident_root_host_observation_stub_requested": False,
        "owner_search_direct_root_build_request_requested": False,
        "compact_owner_action_step_boundary_requested": False,
        "compact_owner_action_dispatch_step_overlap_requested": False,
        "owner_search_fixed_action_result_buffer_requested": False,
        "owner_search_action_result_slot_capacity_requested": 4,
        "owner_search_async_learner_worker_requested": False,
        "owner_search_async_learner_worker_kind_requested": "in_process_thread_v1",
        "owner_search_async_learner_max_pending_requested": 1,
    }
    projected = module._owner_search_slab_proxy_proof_fields(
        {
            "compact_owner_search_resident_root_bridge_ready": False,
            "compact_owner_search_resident_root_bridge_kind": "",
            "compact_owner_search_resident_root_bridge_device": "",
            "compact_owner_search_resident_root_bridge_h2d_bytes": 0.0,
            "compact_owner_search_resident_root_bridge_generation_id": 0,
            "compact_owner_search_inline_background_slab_proxy": False,
            "compact_rollout_slab_last_telemetry": {
                "compact_rollout_slab_resident_host_observation_stub_requested": True,
                "compact_rollout_slab_resident_host_observation_stubbed": True,
                "compact_rollout_slab_resident_host_observation_stub_kind": (
                    "zero_stride_shape_only_v1"
                ),
                "compact_rollout_slab_resident_host_observation_stub_materialized_bytes": 0,
                "compact_rollout_slab_resident_host_observation_stub_logical_bytes": 2048,
                "compact_rollout_slab_search_metadata": {
                    "compact_owner_search_inline_background_slab_proxy": True,
                    "compact_owner_search_resident_root_bridge_ready": True,
                    "compact_owner_search_resident_root_bridge_kind": (
                        "shared_memory_host_root_to_owner_resident_tensor_v1"
                    ),
                    "compact_owner_search_resident_root_bridge_device": "cpu",
                    "compact_owner_search_resident_root_bridge_h2d_bytes": 1024.0,
                    "compact_owner_search_resident_root_bridge_generation_id": 7,
                    "compact_owner_search_slab_bypass": True,
                    "compact_owner_search_slab_bypass_kind": (
                        module.COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
                    ),
                    "compact_rollout_slab_bypassed": True,
                    "compact_rollout_slab_general_replay_row_builder_used": False,
                    "compact_owner_search_slab_bypass_parent_committed_index_rows": 0,
                    "compact_owner_search_slab_bypass_parent_stored_index_rows": 0,
                },
            },
        }
    )
    assert projected["compact_owner_search_resident_root_bridge_ready"] is True
    assert projected["compact_owner_search_inline_background_slab_proxy"] is True
    assert projected["compact_owner_search_resident_root_bridge_kind"] == (
        "shared_memory_host_root_to_owner_resident_tensor_v1"
    )
    assert projected["compact_owner_search_resident_root_bridge_device"] == "cpu"
    assert projected["compact_owner_search_slab_bypass"] is True
    assert projected["compact_owner_search_slab_bypass_kind"] == (
        module.COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
    )
    assert projected["compact_rollout_slab_bypassed"] is True
    assert projected["compact_rollout_slab_general_replay_row_builder_used"] is False
    assert projected["compact_rollout_slab_resident_host_observation_stub_requested"] is True
    assert projected["compact_rollout_slab_resident_host_observation_stubbed"] is True
    assert projected["compact_rollout_slab_resident_host_observation_stub_kind"] == (
        "zero_stride_shape_only_v1"
    )
    assert projected["compact_rollout_slab_resident_host_observation_stub_materialized_bytes"] == 0
    assert projected["compact_rollout_slab_resident_host_observation_stub_logical_bytes"] == 2048
    assert projected["compact_owner_search_resident_root_bridge_h2d_bytes"] == 1024.0
    assert projected["compact_owner_search_resident_root_bridge_generation_id"] == 7


def test_owner_search_expected_train_count_tracks_replayable_transition_batches():
    module = _load_smoke_module()

    def expected_count(
        *,
        steps: int,
        warmup_steps: int,
        sample_interval: int,
        transition_batch_size: int,
        slab_bypass: bool = True,
    ) -> int:
        return module._owner_search_expected_train_request_count(
            SimpleNamespace(
                steps=steps,
                warmup_steps=warmup_steps,
                sample_interval=sample_interval,
                learner_num_unroll_steps=2,
                owner_search_slab_bypass=slab_bypass,
                owner_search_transition_batch_size=transition_batch_size,
            )
        )

    assert expected_count(steps=6, warmup_steps=1, sample_interval=1, transition_batch_size=4) == 1
    assert expected_count(steps=8, warmup_steps=1, sample_interval=1, transition_batch_size=4) == 2
    assert expected_count(steps=64, warmup_steps=0, sample_interval=4, transition_batch_size=4) == 15
    assert expected_count(steps=64, warmup_steps=16, sample_interval=8, transition_batch_size=4) == 8
    assert (
        expected_count(
            steps=64,
            warmup_steps=0,
            sample_interval=4,
            transition_batch_size=4,
            slab_bypass=False,
        )
        == 15
    )


@pytest.mark.parametrize("fixed_soa_replay", [False, True])
def test_direct_transition_batch_replay_store_strips_terminal_metadata_before_resident_append(
    monkeypatch,
    fixed_soa_replay,
):
    module = _load_smoke_module()
    np = module.np

    class FakeRing:
        def __init__(self) -> None:
            self.columnar_records = ()
            self.fixed_soa_records = ()
            self.legacy_append_called = False
            self._columnar_telemetry = {
                "record_count": 0.0,
                "entry_view_object_count": 0.0,
                "step_view_object_count": 0.0,
                "prepare_sec": 0.0,
                "register_sec": 0.0,
                "append_store_sec": 0.0,
                "retain_sec": 0.0,
                "evict_sec": 0.0,
                "evict_release_sec": 0.0,
                "candidate_indices_sec": 0.0,
                "cache_refresh_sec": 0.0,
                "cache_rebuild_sec": 0.0,
                "total_sec": 0.0,
            }
            self._fixed_soa_telemetry = {
                "record_count": 0.0,
                "slot_write_count": 0.0,
                "entry_view_object_count": 0.0,
                "step_view_object_count": 0.0,
                "learner_ready_object_count": 0.0,
                "table_entry_object_count": 0.0,
                "table_concat_count": 0.0,
                "fallback_count": 0.0,
                "slot_write_sec": 0.0,
                "successor_index_sec": 0.0,
                "total_sec": 0.0,
            }

        def append_columnar_entries(self, records):
            self.columnar_records = tuple(records)
            count = len(self.columnar_records)
            self._columnar_telemetry.update(
                {
                    "record_count": float(count),
                    "entry_view_object_count": float(count),
                    "step_view_object_count": float(count * 2),
                    "prepare_sec": 0.001,
                    "register_sec": 0.002,
                    "append_store_sec": 0.003,
                    "retain_sec": 0.004,
                    "evict_sec": 0.0,
                    "evict_release_sec": 0.0,
                    "candidate_indices_sec": 0.005,
                    "cache_refresh_sec": 0.006,
                    "cache_rebuild_sec": 0.0,
                    "total_sec": 0.021,
                }
            )
            return len(self.columnar_records)

        def append_fixed_soa_columnar_records(self, records):
            self.fixed_soa_records = tuple(records)
            count = len(self.fixed_soa_records)
            self._fixed_soa_telemetry.update(
                {
                    "record_count": float(count),
                    "slot_write_count": float(count),
                    "entry_view_object_count": 0.0,
                    "step_view_object_count": 0.0,
                    "learner_ready_object_count": 0.0,
                    "table_entry_object_count": 0.0,
                    "table_concat_count": 0.0,
                    "fallback_count": 0.0,
                    "slot_write_sec": 0.007,
                    "successor_index_sec": 0.008,
                    "total_sec": 0.015,
                }
            )
            return len(self.fixed_soa_records)

        def append_entries(self, entries):
            del entries
            self.legacy_append_called = True
            raise AssertionError("legacy ring append should not be used")

        def columnar_append_telemetry_snapshot(self):
            return dict(self._columnar_telemetry)

        def fixed_soa_append_telemetry_snapshot(self):
            return dict(self._fixed_soa_telemetry)

    fake_ring = FakeRing()
    store = module._OwnerSearchDirectTransitionBatchReplayStoreFactorySidecarV1(
        capacity=4,
        metadata={
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
            "compact_muzero_learner_batch_tensor_native_replay": True,
            module.FIXED_SOA_REPLAY_REQUESTED_KEY: bool(fixed_soa_replay),
        },
    )
    store._ring = fake_ring

    monkeypatch.setattr(
        module,
        "compact_search_deferred_replay_payload_digest_v1",
        lambda handle: f"digest:{handle}",
    )
    monkeypatch.setattr(
        module,
        "validate_compact_device_search_two_phase_payload_v1",
        lambda action_step, payload: None,
    )
    monkeypatch.setattr(
        module,
        "_compact_batch_from_root_batch",
        lambda root: SimpleNamespace(schema_id="unit_compact_batch"),
    )

    def fake_index_rows(
        previous_batch,
        previous_root,
        action_step,
        replay_payload,
        **kwargs,
    ):
        del previous_batch, previous_root, action_step, replay_payload
        return SimpleNamespace(
            metadata={
                "device_replay_index_rows": True,
                **dict(kwargs.get("metadata") or {}),
            },
            record_index=int(kwargs["record_index"]),
            next_record_index=int(kwargs["record_index"]) + 1,
            action=np.asarray([3], dtype=np.int16),
        )

    monkeypatch.setattr(
        module,
        "build_compact_device_replay_index_rows_v1_from_payload",
        fake_index_rows,
    )

    previous_resident = SimpleNamespace(host_fallback_allowed=False)
    current_resident = SimpleNamespace(host_fallback_allowed=False)
    previous_root = SimpleNamespace(
        observation_source=module.COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=previous_resident,
    )
    current_root = SimpleNamespace(
        observation_source=module.COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=current_resident,
    )
    search_result = SimpleNamespace(
        record_index=10,
        env_row=np.asarray([0], dtype=np.int64),
        player=np.asarray([0], dtype=np.int64),
        selected_action=np.asarray([3], dtype=np.int16),
        metadata={},
    )
    cached = SimpleNamespace(
        record_index=10,
        root_batch=previous_root,
        search_result=search_result,
        action_step=SimpleNamespace(replay_payload_handle="payload-10"),
        inner_replay_payload_handle="payload-10",
    )
    batch = SimpleNamespace(
        schema_id=module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID,
        transition_count=1,
        metadata={
            "compact_owner_search_replay_append_transition_batch_kind": (
                module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
            ),
            "compact_owner_search_replay_append_transition_batch_transition_count": 1,
            "compact_owner_search_transition_batch_fixed_capacity": 2,
            "compact_owner_search_transition_batch_padding_count": 1,
            "compact_owner_search_transition_batch_overflow_count": 0,
            "done_row_count": 999,
            "done_row_indices": [999],
            "next_final_observation_row_count": 999,
            "next_final_observation_row_indices": [999],
        },
        record_indices=np.asarray([10], dtype=np.int64),
        next_record_indices=np.asarray([11], dtype=np.int64),
        next_joint_action=np.asarray([[[3]]], dtype=np.int16),
        next_reward=np.asarray([[0.0]], dtype=np.float32),
        next_done=np.asarray([[False]], dtype=np.bool_),
        next_terminated=np.asarray([[False]], dtype=np.bool_),
        next_truncated=np.asarray([[False]], dtype=np.bool_),
        next_final_reward_map=np.asarray([[[0.0]]], dtype=np.float32),
        next_final_observation_row_mask=np.asarray([[False]], dtype=np.bool_),
        replay_payload_handles=("payload-10",),
        selected_action_digests=("",),
        search_replay_payload_digests=("digest:payload-10",),
        policy_source="unit_test",
    )
    flushed_handles: list[str] = []

    def fake_flush_device_replay_payload(handle: str):
        flushed_handles.append(str(handle))
        return SimpleNamespace(
            handle=handle,
            metadata={
                "device_replay_payload": True,
                "compact_torch_search_service_device_replay_payload_flushed": True,
                "compact_torch_search_one_simulation_replay_materialization_deferred": True,
                "compact_torch_search_one_simulation_replay_materialized_on_flush": True,
                "compact_torch_search_deferred_one_simulation_replay_recurrent_inference_calls": 1.0,
                "compact_torch_search_deferred_one_simulation_model_identity_match": True,
                "compact_torch_search_deferred_one_simulation_model_refresh_crossed_count": 0,
                "compact_torch_search_pending_deferred_replay_payload_count": 1.0,
                "compact_torch_search_pending_deferred_replay_payload_final_count": 0.0,
                "compact_torch_search_deferred_one_simulation_replay_flush_sec": 0.001,
                "compact_torch_search_service_device_replay_payload_flush_sec": 0.001,
                "compact_torch_search_service_replay_payload_d2h_bytes": 0.0,
                "compact_torch_search_deferred_one_simulation_action_model_state_digest": "digest-a",
                "compact_torch_search_deferred_one_simulation_flush_model_state_digest": "digest-a",
            },
        )

    result = store.append_owner_search_transition_batches(
        replay_append_transition_batches=(batch,),
        root_batch=current_root,
        search_result=search_result,
        request=SimpleNamespace(actor_step=11),
        root_batch_cache={10: previous_root},
        search_result_cache={"payload-10": cached},
        flush_device_replay_payload=fake_flush_device_replay_payload,
    )

    assert result["appended_count"] == 1
    assert flushed_handles == ["payload-10"]
    assert fake_ring.legacy_append_called is False
    records = fake_ring.fixed_soa_records if fixed_soa_replay else fake_ring.columnar_records
    assert len(records) == 1
    assert len(fake_ring.fixed_soa_records) == int(fixed_soa_replay)
    assert len(fake_ring.columnar_records) == int(not fixed_soa_replay)
    record = records[0]
    assert record.previous_resident_observation_replay_snapshot is previous_resident
    assert record.current_resident_observation_replay_snapshot is current_resident
    assert "done_row_count" not in record.index_rows.metadata
    assert "done_row_indices" not in record.index_rows.metadata
    assert "next_final_observation_row_count" not in record.index_rows.metadata
    assert "next_final_observation_row_indices" not in record.index_rows.metadata
    assert result["compact_owner_search_direct_transition_batch_replay_columnar_append_used"] is (
        not fixed_soa_replay
    )
    assert result[
        "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count"
    ] == int(not fixed_soa_replay)
    assert (
        result["compact_owner_search_direct_transition_batch_replay_fixed_soa_requested"]
        is fixed_soa_replay
    )
    assert (
        result["compact_owner_search_direct_transition_batch_replay_fixed_soa_used"]
        is fixed_soa_replay
    )
    assert result[
        "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_count"
    ] == int(fixed_soa_replay)
    assert (
        result["compact_owner_search_direct_transition_batch_replay_ring_entry_object_count"] == 0
    )
    assert result[
        "compact_owner_search_direct_transition_batch_replay_columnar_record_count"
    ] == int(not fixed_soa_replay)
    assert result[
        "compact_owner_search_direct_transition_batch_replay_columnar_entry_view_object_count"
    ] == int(not fixed_soa_replay)
    assert result[
        "compact_owner_search_direct_transition_batch_replay_columnar_step_view_object_count"
    ] == 2 * int(not fixed_soa_replay)
    assert (
        result[
            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flushed_count"
        ]
        == 1
    )
    assert (
        result[
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_count"
        ]
        == 1
    )
    assert (
        result[
            "compact_owner_search_direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count"
        ]
        == 1
    )
    assert (
        result[
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count"
        ]
        == 1
    )
    assert (
        result[
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count"
        ]
        == 0
    )
    assert (
        result[
            "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_final_count"
        ]
        == 0
    )
    if fixed_soa_replay:
        assert (
            result[
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_entry_view_object_count"
            ]
            == 0
        )
        assert (
            result[
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_step_view_object_count"
            ]
            == 0
        )
        assert (
            result[
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_entry_object_count"
            ]
            == 0
        )
        assert (
            result[
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_concat_count"
            ]
            == 0
        )
    else:
        assert (
            result["compact_owner_search_direct_transition_batch_replay_columnar_cache_refresh_sec"]
            == 0.006
        )


def test_direct_transition_batch_replay_store_derives_owner_local_outcome(
    monkeypatch,
):
    module = _load_smoke_module()
    np = module.np

    class FakeRing:
        def __init__(self) -> None:
            self.columnar_records = ()
            self._columnar_telemetry = {
                "record_count": 0.0,
                "entry_view_object_count": 0.0,
                "step_view_object_count": 0.0,
            }

        def append_columnar_entries(self, records):
            self.columnar_records = tuple(records)
            count = len(self.columnar_records)
            self._columnar_telemetry.update(
                {
                    "record_count": float(count),
                    "entry_view_object_count": float(count),
                    "step_view_object_count": float(count * 2),
                }
            )
            return count

        def columnar_append_telemetry_snapshot(self):
            return dict(self._columnar_telemetry)

    store = module._OwnerSearchDirectTransitionBatchReplayStoreFactorySidecarV1(
        capacity=4,
        metadata={
            "compact_owner_search_owner_local_transition_derivation_requested": True,
        },
    )
    store._ring = FakeRing()

    monkeypatch.setattr(
        module,
        "compact_search_deferred_replay_payload_digest_v1",
        lambda handle: f"digest:{handle}",
    )
    monkeypatch.setattr(
        module,
        "validate_compact_device_search_two_phase_payload_v1",
        lambda action_step, payload: None,
    )
    monkeypatch.setattr(
        module,
        "_compact_batch_from_root_batch",
        lambda root: SimpleNamespace(schema_id="unit_compact_batch"),
    )
    monkeypatch.setattr(
        module._OwnerSearchDirectTransitionBatchReplayStoreFactorySidecarV1,
        "_derived_joint_action_from_search",
        lambda self, search_result, current_root: np.asarray([[3]], dtype=np.int16),
    )
    monkeypatch.setattr(
        module,
        "compact_transition_outcome_v1_from_next_root_batch",
        lambda root: SimpleNamespace(
            next_reward=np.asarray([[1.5]], dtype=np.float32),
            next_done=np.asarray([False], dtype=np.bool_),
            next_terminated=np.asarray([False], dtype=np.bool_),
            next_truncated=np.asarray([False], dtype=np.bool_),
            next_final_reward_map=np.asarray([[1.5]], dtype=np.float32),
            next_final_observation_row_mask=np.asarray([False], dtype=np.bool_),
        ),
    )

    captured_kwargs: dict[str, Any] = {}

    def fake_index_rows(
        previous_batch,
        previous_root,
        action_step,
        replay_payload,
        **kwargs,
    ):
        del previous_batch, previous_root, action_step, replay_payload
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            metadata=dict(kwargs.get("metadata") or {}),
            record_index=int(kwargs["record_index"]),
        )

    monkeypatch.setattr(
        module,
        "build_compact_device_replay_index_rows_v1_from_payload",
        fake_index_rows,
    )

    previous_root = SimpleNamespace(
        observation_source=module.COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=SimpleNamespace(host_fallback_allowed=False),
    )
    current_root = SimpleNamespace(
        observation_source=module.COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=SimpleNamespace(host_fallback_allowed=False),
    )
    search_result = SimpleNamespace(
        record_index=10,
        env_row=np.asarray([0], dtype=np.int64),
        player=np.asarray([0], dtype=np.int64),
        selected_action=np.asarray([3], dtype=np.int16),
        metadata={},
    )
    cached = SimpleNamespace(
        record_index=10,
        root_batch=previous_root,
        search_result=search_result,
        action_step=SimpleNamespace(replay_payload_handle="payload-10"),
        inner_replay_payload_handle="payload-10",
    )
    batch = SimpleNamespace(
        schema_id=(module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID),
        transition_count=1,
        metadata={
            "compact_owner_search_replay_append_transition_batch_kind": (
                module.COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
            ),
            "compact_owner_search_transition_batch_fixed_capacity": 2,
            "compact_owner_search_transition_batch_padding_count": 1,
            "compact_owner_search_transition_batch_overflow_count": 0,
        },
        record_indices=np.asarray([10], dtype=np.int64),
        next_record_indices=np.asarray([11], dtype=np.int64),
        replay_payload_handles=("payload-10",),
        selected_action_digests=("",),
        search_replay_payload_digests=("digest:payload-10",),
        applied_action_counts=np.asarray([1], dtype=np.int64),
        applied_action_checksums=np.asarray(
            [store._action_checksum(np.asarray([3], dtype=np.int16))],
            dtype=np.int64,
        ),
        policy_source="unit_test",
    )

    result = store.append_owner_search_transition_batches(
        replay_append_transition_batches=(batch,),
        root_batch=current_root,
        search_result=search_result,
        request=SimpleNamespace(actor_step=11),
        root_batch_cache={10: previous_root},
        search_result_cache={"payload-10": cached},
        flush_device_replay_payload=lambda handle: SimpleNamespace(
            metadata={
                "compact_torch_search_service_device_replay_payload_flushed": True,
            },
        ),
    )

    assert result["appended_count"] == 1
    assert not hasattr(batch, "next_reward")
    assert result["compact_owner_search_owner_local_transition_derivation_requested"] is True
    assert result["compact_owner_search_owner_local_transition_derivation_used"] is True
    assert result["compact_owner_search_owner_local_transition_derivation_cache_hit_count"] == 1
    assert (
        result[
            "compact_owner_search_owner_local_transition_derivation_action_checksum_verified_count"
        ]
        == 1
    )
    assert np.array_equal(captured_kwargs["next_joint_action"], np.asarray([[3]]))
    assert np.array_equal(
        captured_kwargs["next_reward"],
        np.asarray([[1.5]], dtype=np.float32),
    )
    assert (
        captured_kwargs["metadata"]["compact_owner_search_owner_local_transition_derivation_used"]
        is True
    )


def test_speed_row_smoke_renderer_handles_autoreset_subsets():
    module = _load_smoke_module()
    renderer = module._PersistentDeviceRenderer(device="cpu")
    out = module.np.empty((4, 1, 64, 64), dtype=module.np.uint8)
    request = SimpleNamespace(
        out=out,
        row_indices=module.np.asarray([3, 3, 7, 7], dtype=module.np.int64),
        controlled_players=module.np.asarray([0, 1, 0, 1], dtype=module.np.int64),
    )

    result = renderer.render(request)

    assert result.frames.shape == (4, 1, 64, 64)
    assert tuple(result.device_frames.shape) == (2, 2, 1, 64, 64)


def test_speed_row_smoke_can_emit_borrowed_render_state_handoff_fields():
    module = _load_smoke_module()
    args = SimpleNamespace(
        search_service_kind="compact_torch_search_service",
        hybrid_persistent_compact_render_state_buffer=False,
        hybrid_borrow_single_actor_render_state=True,
        compact_torch_initial_inference_mode="direct_core",
        compact_torch_observation_memory_format="contiguous",
        compact_torch_model_memory_format="contiguous",
        compact_profile_bounded_diagnostics=False,
        actor_count=1,
        steps=4,
        warmup_steps=1,
    )

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=_profile_payload(borrow_render_state=True),
        loaded_checkpoint_identity=_loaded_identity(),
    )

    for payload in (summary, compact):
        assert payload["hybrid_persistent_compact_render_state_buffer"] is False
        assert payload["hybrid_borrow_single_actor_render_state"] is True
        assert payload["render_state_handoff_mode"] == "borrow_single_actor_env_state"
        assert payload["render_state_copy_steps"] == 0
        assert payload["render_state_borrowed_steps"] == 5


def test_speed_row_smoke_bounded_diagnostics_omit_nested_source_payload():
    module = _load_smoke_module()
    args = _summary_args()
    args.compact_profile_bounded_diagnostics = True
    profile_payload = _profile_payload()
    profile_payload.update(
        {
            "resident_replay_snapshot_mode": "latest_frame_history",
            "compact_owned_loop_replay_store_retained_resident_snapshot_count": 9,
            "compact_owned_loop_replay_store_retained_resident_snapshot_bytes": 12345,
            "compact_rollout_slab_sample_gate_replay_ring_entry_count": 7,
            "compact_rollout_slab_sample_gate_replay_ring_index_row_count": 14,
            "compact_rollout_slab_sample_gate_replay_ring_pair_capacity": 16,
            "compact_rollout_slab_sample_gate_replay_ring_evicted_pair_count": 2,
            "compact_rollout_slab_sample_gate_replay_ring_evicted_index_row_count": 4,
            "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_count": 5,
            "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_bytes": 6789,
        }
    )

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=profile_payload,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    for payload in (summary, compact):
        assert payload["compact_profile_bounded_diagnostics"] is True
        assert payload["source_profile_payload_embedded"] is False
        assert payload["resident_replay_snapshot_mode"] == "latest_frame_history"
        assert payload["compact_owned_loop_replay_store_retained_resident_snapshot_count"] == 9
        assert payload["compact_owned_loop_replay_store_retained_resident_snapshot_bytes"] == 12345
        assert payload["compact_rollout_slab_sample_gate_replay_ring_entry_count"] == 7
        assert payload["compact_rollout_slab_sample_gate_replay_ring_index_row_count"] == 14
        assert payload["compact_rollout_slab_sample_gate_replay_ring_pair_capacity"] == 16
        assert payload["compact_rollout_slab_sample_gate_replay_ring_evicted_pair_count"] == 2
        assert payload["compact_rollout_slab_sample_gate_replay_ring_evicted_index_row_count"] == 4
        assert (
            payload["compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_count"]
            == 5
        )
        assert (
            payload["compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_bytes"]
            == 6789
        )
    assert "source_profile_payload" not in compact
    assert compact["source_profile_payload_omitted_reason"] == (
        "compact_profile_bounded_diagnostics"
    )


def test_speed_row_smoke_projects_cuda_sync_diagnostics():
    module = _load_smoke_module()
    args = _summary_args()
    args.compact_profile_cuda_sync_timing_diagnostics = True
    builder_diag_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics"
    )
    builder_enabled_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled"
    )
    builder_count_key = "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count"
    builder_sec_key = "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec"
    runtime_step_stats = {
        "count": 3,
        "sum_sec": 0.91,
        "min_sec": 0.21,
        "max_sec": 0.41,
        "p50_sec": 0.29,
        "p95_sec": 0.39,
        "slowest_iteration": 4,
        "slowest_measured_iteration": 3,
        "slowest_actor_step_wall_sec": 0.13,
        "slowest_observation_sec": 0.07,
        "slowest_compact_rollout_slab_sec": 0.03,
        "slowest_sample_gate_sec": 0.11,
        "slowest_learner_gate_sec": 0.05,
        "slowest_policy_refresh_sec": 0.01,
        "slowest_primary_accounted_sec": 0.4,
        "slowest_primary_residual_sec": 0.01,
        "slowest_env_trajectory_checksum": 123456,
        "top_slowest_records": [
            {
                "iteration": 4,
                "measured_iteration": 3,
                "sample_gate_call_index": 2,
                "sample_gate_call_count_delta": 1,
                "sec": 0.41,
                "sample_gate_sec": 0.11,
                "sample_gate_builder_group_loop_sec": 0.08,
                "observation_sec": 0.07,
                "actor_step_wall_sec": 0.13,
                "primary_residual_sec": 0.01,
                "env_trajectory_checksum": 123456,
            }
        ],
        "sample_gate_active_count": 2,
        "sample_gate_active_sum_sec": 0.62,
        "sample_gate_active_min_sec": 0.21,
        "sample_gate_active_max_sec": 0.41,
        "sample_gate_active_p50_sec": 0.31,
        "sample_gate_active_p95_sec": 0.40,
        "sample_gate_active_sample_gate_count": 2,
        "sample_gate_active_sample_gate_sum_sec": 0.41,
        "sample_gate_active_sample_gate_p50_sec": 0.205,
        "sample_gate_active_sample_gate_p95_sec": 0.285,
        "sample_gate_active_sample_gate_residual_count": 2,
        "sample_gate_active_sample_gate_residual_sum_sec": 0.05,
        "sample_gate_active_sample_gate_residual_p50_sec": 0.025,
        "sample_gate_active_sample_gate_residual_p95_sec": 0.029,
        "sample_gate_active_sample_gate_builder_group_loop_count": 2,
        "sample_gate_active_sample_gate_builder_group_loop_sum_sec": 0.20,
        "sample_gate_active_sample_gate_builder_group_loop_p50_sec": 0.10,
        "sample_gate_active_sample_gate_builder_group_loop_p95_sec": 0.145,
        "sample_gate_inactive_count": 1,
        "sample_gate_inactive_sum_sec": 0.29,
        "sample_gate_inactive_min_sec": 0.29,
        "sample_gate_inactive_max_sec": 0.29,
        "sample_gate_inactive_p50_sec": 0.29,
        "sample_gate_inactive_p95_sec": 0.29,
        "sample_gate_inactive_sample_gate_sum_sec": 0.0,
        "early_count": 1,
        "early_sum_sec": 0.21,
        "early_sample_gate_active_count": 1,
        "early_sample_gate_sum_sec": 0.10,
        "early_sample_gate_builder_group_loop_sum_sec": 0.05,
        "early_sample_gate_active_sample_gate_count": 1,
        "early_sample_gate_active_sample_gate_sum_sec": 0.10,
        "early_sample_gate_active_sample_gate_p50_sec": 0.10,
        "early_sample_gate_active_sample_gate_p95_sec": 0.10,
        "early_sample_gate_active_sample_gate_builder_group_loop_count": 1,
        "early_sample_gate_active_sample_gate_builder_group_loop_sum_sec": 0.05,
        "early_sample_gate_active_sample_gate_builder_group_loop_p50_sec": 0.05,
        "early_sample_gate_active_sample_gate_builder_group_loop_p95_sec": 0.05,
        "mid_count": 1,
        "mid_sum_sec": 0.29,
        "mid_sample_gate_active_count": 0,
        "mid_sample_gate_sum_sec": 0.0,
        "mid_sample_gate_builder_group_loop_sum_sec": 0.0,
        "mid_sample_gate_active_sample_gate_count": 0,
        "mid_sample_gate_active_sample_gate_sum_sec": 0.0,
        "mid_sample_gate_active_sample_gate_p50_sec": 0.0,
        "mid_sample_gate_active_sample_gate_p95_sec": 0.0,
        "mid_sample_gate_active_sample_gate_builder_group_loop_count": 0,
        "mid_sample_gate_active_sample_gate_builder_group_loop_sum_sec": 0.0,
        "mid_sample_gate_active_sample_gate_builder_group_loop_p50_sec": 0.0,
        "mid_sample_gate_active_sample_gate_builder_group_loop_p95_sec": 0.0,
        "late_count": 1,
        "late_sum_sec": 0.41,
        "late_sample_gate_active_count": 1,
        "late_sample_gate_sum_sec": 0.31,
        "late_sample_gate_builder_group_loop_sum_sec": 0.15,
        "late_sample_gate_active_sample_gate_count": 1,
        "late_sample_gate_active_sample_gate_sum_sec": 0.31,
        "late_sample_gate_active_sample_gate_p50_sec": 0.31,
        "late_sample_gate_active_sample_gate_p95_sec": 0.31,
        "late_sample_gate_active_sample_gate_builder_group_loop_count": 1,
        "late_sample_gate_active_sample_gate_builder_group_loop_sum_sec": 0.15,
        "late_sample_gate_active_sample_gate_builder_group_loop_p50_sec": 0.15,
        "late_sample_gate_active_sample_gate_builder_group_loop_p95_sec": 0.15,
        "actor_step_wall_sum_sec": 0.31,
        "actor_step_wall_min_sec": 0.09,
        "actor_step_wall_max_sec": 0.13,
        "actor_step_wall_p50_sec": 0.10,
        "actor_step_wall_p95_sec": 0.12,
        "actor_env_runtime_sum_sec": 0.22,
        "actor_env_runtime_min_sec": 0.06,
        "actor_env_runtime_max_sec": 0.09,
        "actor_env_runtime_p50_sec": 0.07,
        "actor_env_runtime_p95_sec": 0.085,
        "actor_autoreset_sum_sec": 0.07,
        "actor_autoreset_min_sec": 0.01,
        "actor_autoreset_max_sec": 0.03,
        "actor_autoreset_p50_sec": 0.02,
        "actor_autoreset_p95_sec": 0.029,
        "sample_gate_sum_sec": 0.41,
        "sample_gate_min_sec": 0.0,
        "sample_gate_max_sec": 0.21,
        "sample_gate_p50_sec": 0.10,
        "sample_gate_p95_sec": 0.20,
        "sample_gate_residual_sum_sec": 0.05,
        "sample_gate_residual_min_sec": 0.0,
        "sample_gate_residual_max_sec": 0.03,
        "sample_gate_residual_p50_sec": 0.02,
        "sample_gate_residual_p95_sec": 0.029,
        "sample_gate_cuda_sync_sum_sec": 0.01,
        "sample_gate_cuda_sync_min_sec": 0.0,
        "sample_gate_cuda_sync_max_sec": 0.006,
        "sample_gate_cuda_sync_p50_sec": 0.003,
        "sample_gate_cuda_sync_p95_sec": 0.0058,
        "sample_gate_builder_group_loop_sum_sec": 0.20,
        "sample_gate_builder_group_loop_min_sec": 0.0,
        "sample_gate_builder_group_loop_max_sec": 0.09,
        "sample_gate_builder_group_loop_p50_sec": 0.07,
        "sample_gate_builder_group_loop_p95_sec": 0.088,
        "sample_gate_builder_cuda_sync_sum_sec": 0.04,
        "sample_gate_builder_cuda_sync_min_sec": 0.0,
        "sample_gate_builder_cuda_sync_max_sec": 0.02,
        "sample_gate_builder_cuda_sync_p50_sec": 0.01,
        "sample_gate_builder_cuda_sync_p95_sec": 0.019,
        "primary_residual_sum_sec": 0.03,
        "primary_residual_min_sec": 0.0,
        "primary_residual_max_sec": 0.02,
        "primary_residual_p50_sec": 0.01,
        "primary_residual_p95_sec": 0.019,
    }
    sample_gate_per_call_stats = {
        "count": 2,
        "sum_sec": 0.22,
        "min_sec": 0.1,
        "max_sec": 0.12,
        "p50_sec": 0.11,
        "p95_sec": 0.119,
        "slowest_call_index": 2,
        "slowest_iteration": 17,
        "slowest_measured_iteration": 9,
    }
    sample_gate_call_trace_records = [
        {
            "call_index": 2,
            "iteration": 17,
            "measured_iteration": 9,
            "measured_window_third": "late",
            "sample_seed": 123,
            "sample_row_count": 4,
            "sampled_pair_count": 8,
            "stored_pair_count": 2048,
            "stored_index_row_count": 512,
            "next_target_eligible_pair_count": 64,
            "next_target_eligible_index_row_count": 16,
            "runtime_snapshot_diagnostics": 1,
            "cuda_memory_snapshot_enabled": 1,
            "cuda_memory_allocated_before_bytes": 1000,
            "cuda_memory_allocated_after_bytes": 1200,
            "cuda_memory_allocated_delta_bytes": 200,
            "learner_batch_build_cuda_memory_allocated_before_bytes": 1000,
            "learner_batch_build_cuda_memory_allocated_after_bytes": 1150,
            "learner_batch_build_cuda_memory_allocated_delta_bytes": 150,
            "python_gc_gen0_before": 5,
            "python_gc_gen0_after": 6,
            "python_gc_gen0_delta": 1,
            "python_gc_gen0_collections_before": 10,
            "python_gc_gen0_collections_after": 12,
            "python_gc_gen0_collections_delta": 2,
            "python_gc_gen0_collected_before": 20,
            "python_gc_gen0_collected_after": 23,
            "python_gc_gen0_collected_delta": 3,
            "python_gc_gen0_uncollectable_before": 0,
            "python_gc_gen0_uncollectable_after": 0,
            "python_gc_gen0_uncollectable_delta": 0,
            "python_gc_gen1_collections_before": 3,
            "python_gc_gen1_collections_after": 4,
            "python_gc_gen1_collections_delta": 1,
            "python_gc_gen1_collected_before": 5,
            "python_gc_gen1_collected_after": 7,
            "python_gc_gen1_collected_delta": 2,
            "python_gc_gen1_uncollectable_before": 0,
            "python_gc_gen1_uncollectable_after": 0,
            "python_gc_gen1_uncollectable_delta": 0,
            "python_gc_gen2_collections_before": 1,
            "python_gc_gen2_collections_after": 1,
            "python_gc_gen2_collections_delta": 0,
            "python_gc_gen2_collected_before": 2,
            "python_gc_gen2_collected_after": 2,
            "python_gc_gen2_collected_delta": 0,
            "python_gc_gen2_uncollectable_before": 0,
            "python_gc_gen2_uncollectable_after": 0,
            "python_gc_gen2_uncollectable_delta": 0,
            "process_maxrss_before_raw": 100000,
            "process_maxrss_after_raw": 100100,
            "process_maxrss_delta_raw": 100,
            "process_cpu_time_before_ns": 1000,
            "process_cpu_time_after_ns": 1500,
            "process_cpu_time_delta_ns": 500,
            "thread_cpu_time_before_ns": 2000,
            "thread_cpu_time_after_ns": 2300,
            "thread_cpu_time_delta_ns": 300,
            "learner_batch_build_process_cpu_time_before_ns": 1100,
            "learner_batch_build_process_cpu_time_after_ns": 1400,
            "learner_batch_build_process_cpu_time_delta_ns": 300,
            "learner_batch_build_thread_cpu_time_before_ns": 2100,
            "learner_batch_build_thread_cpu_time_after_ns": 2250,
            "learner_batch_build_thread_cpu_time_delta_ns": 150,
            "terminal_final_observation_group_count": 4,
            "terminal_final_observation_index_fast_path_count": 0,
            "terminal_final_observation_fallback_count": 4,
            "terminal_final_observation_validate_only_count": 4,
            "terminal_final_observation_materialized_count": 0,
            "terminal_final_observation_final_row_count_sum": 5,
            "terminal_final_observation_final_row_count_max": 2,
            "terminal_final_observation_dense_storage_count": 0,
            "terminal_final_observation_sparse_storage_count": 4,
            "terminal_final_observation_missing_storage_count": 0,
            "terminal_final_observation_sparse_row_count_sum": 12,
            "terminal_final_observation_sparse_row_count_max": 3,
            "builder_group_loop_process_cpu_time_delta_ns": 170,
            "builder_group_loop_thread_cpu_time_delta_ns": 165,
            "builder_group_loop_accounted_process_cpu_time_delta_ns": 125,
            "builder_group_loop_accounted_thread_cpu_time_delta_ns": 120,
            "builder_group_loop_residual_process_cpu_time_delta_ns": 45,
            "builder_group_loop_residual_thread_cpu_time_delta_ns": 45,
            "builder_group_loop_prepare_process_cpu_time_delta_ns": 22,
            "builder_group_loop_prepare_thread_cpu_time_delta_ns": 21,
            "builder_group_loop_prepare_snapshot_process_cpu_time_delta_ns": 8,
            "builder_group_loop_prepare_snapshot_thread_cpu_time_delta_ns": 8,
            "builder_group_loop_prepare_index_process_cpu_time_delta_ns": 7,
            "builder_group_loop_prepare_index_thread_cpu_time_delta_ns": 6,
            "builder_group_loop_prepare_observation_process_cpu_time_delta_ns": 5,
            "builder_group_loop_prepare_observation_thread_cpu_time_delta_ns": 5,
            "builder_group_loop_prepare_accounted_process_cpu_time_delta_ns": 20,
            "builder_group_loop_prepare_accounted_thread_cpu_time_delta_ns": 19,
            "builder_group_loop_prepare_residual_process_cpu_time_delta_ns": 2,
            "builder_group_loop_prepare_residual_thread_cpu_time_delta_ns": 2,
            "builder_group_loop_terminal_value_bookkeeping_process_cpu_time_delta_ns": 3,
            "builder_group_loop_terminal_value_bookkeeping_thread_cpu_time_delta_ns": 3,
            "builder_unroll_fields_process_cpu_time_delta_ns": 70,
            "builder_unroll_fields_thread_cpu_time_delta_ns": 68,
            "builder_unroll_stack_fields_process_cpu_time_delta_ns": 31,
            "builder_unroll_stack_fields_thread_cpu_time_delta_ns": 30,
            "builder_unroll_fields_accounted_process_cpu_time_delta_ns": 61,
            "builder_unroll_fields_accounted_thread_cpu_time_delta_ns": 59,
            "builder_unroll_fields_residual_process_cpu_time_delta_ns": 9,
            "builder_unroll_fields_residual_thread_cpu_time_delta_ns": 9,
            "builder_terminal_metadata_accounted_process_cpu_time_delta_ns": 26,
            "builder_terminal_metadata_accounted_thread_cpu_time_delta_ns": 25,
            "builder_terminal_metadata_residual_process_cpu_time_delta_ns": 4,
            "builder_terminal_metadata_residual_thread_cpu_time_delta_ns": 4,
            "builder_terminal_metadata_final_observation_accounted_process_cpu_time_delta_ns": 18,
            "builder_terminal_metadata_final_observation_accounted_thread_cpu_time_delta_ns": 17,
            "builder_terminal_metadata_final_observation_residual_process_cpu_time_delta_ns": 3,
            "builder_terminal_metadata_final_observation_residual_thread_cpu_time_delta_ns": 3,
            "builder_terminal_metadata_final_observation_gather_process_cpu_time_delta_ns": 9,
            "builder_terminal_metadata_final_observation_gather_thread_cpu_time_delta_ns": 9,
            "builder_terminal_metadata_final_observation_storage_process_cpu_time_delta_ns": 3,
            "builder_terminal_metadata_final_observation_storage_thread_cpu_time_delta_ns": 2,
            "builder_terminal_metadata_final_observation_validate_process_cpu_time_delta_ns": 6,
            "builder_terminal_metadata_final_observation_validate_thread_cpu_time_delta_ns": 6,
            "builder_metadata_sync_process_cpu_time_delta_ns": 12,
            "builder_metadata_sync_thread_cpu_time_delta_ns": 11,
            "builder_metadata_build_process_cpu_time_delta_ns": 7,
            "builder_metadata_build_thread_cpu_time_delta_ns": 6,
            "sample_gate_sec": 0.12,
            "learner_batch_build_sec": 0.04,
            "builder_group_loop_sec": 0.02,
            "builder_group_loop_prepare_sec": 0.004,
            "builder_group_loop_prepare_accounted_sec": 0.0036,
            "builder_group_loop_prepare_residual_sec": 0.0004,
            "builder_group_loop_prepare_snapshot_sec": 0.0011,
            "builder_group_loop_prepare_index_sec": 0.0012,
            "builder_group_loop_prepare_observation_sec": 0.0013,
            "builder_group_loop_terminal_value_bookkeeping_sec": 0.001,
            "builder_terminal_metadata_accounted_sec": 0.009,
            "builder_terminal_metadata_residual_sec": 0.001,
            "builder_terminal_metadata_mask_sec": 0.002,
            "builder_terminal_metadata_final_observation_accounted_sec": 0.006,
            "builder_terminal_metadata_final_observation_residual_sec": 0.001,
            "builder_terminal_metadata_final_observation_gather_sec": 0.003,
            "builder_terminal_metadata_final_observation_storage_sec": 0.001,
            "builder_terminal_metadata_final_observation_validate_sec": 0.002,
            "builder_unroll_fields_accounted_sec": 0.0035,
            "builder_unroll_fields_residual_sec": 0.0005,
            "builder_unroll_stack_fields_sec": 0.004,
            "builder_finalize_outputs_sec": 0.003,
            "builder_group_loop_residual_sec": 0.005,
            "residual_sec": 0.01,
        }
    ]
    sample_gate_candidate_per_call_stats = {
        "count": 2,
        "sum_sec": 0.07,
        "min_sec": 0.03,
        "max_sec": 0.04,
        "p50_sec": 0.035,
        "p95_sec": 0.0395,
        "slowest_call_index": 2,
    }
    sample_gate_rng_per_call_stats = {
        "count": 2,
        "sum_sec": 0.03,
        "min_sec": 0.01,
        "max_sec": 0.02,
        "p50_sec": 0.015,
        "p95_sec": 0.0195,
        "slowest_call_index": 2,
    }
    sample_gate_residual_per_call_stats = {
        "count": 2,
        "sum_sec": 0.05,
        "min_sec": 0.02,
        "max_sec": 0.03,
        "p50_sec": 0.025,
        "p95_sec": 0.0295,
        "slowest_call_index": 2,
    }
    builder_group_loop_per_call_stats = {
        "count": 2,
        "sum_sec": 0.41,
        "min_sec": 0.19,
        "max_sec": 0.22,
        "p50_sec": 0.205,
        "p95_sec": 0.2185,
        "slowest_call_index": 2,
    }
    builder_group_loop_accounted_per_call_stats = {
        "count": 2,
        "sum_sec": 0.31,
        "min_sec": 0.14,
        "max_sec": 0.17,
        "p50_sec": 0.155,
        "p95_sec": 0.1685,
        "slowest_call_index": 2,
    }
    builder_group_loop_residual_per_call_stats = {
        "count": 2,
        "sum_sec": 0.10,
        "min_sec": 0.05,
        "max_sec": 0.05,
        "p50_sec": 0.05,
        "p95_sec": 0.05,
        "slowest_call_index": 1,
    }
    builder_cuda_sync_per_call_stats = {
        "count": 2,
        "sum_sec": 0.82,
        "min_sec": 0.32,
        "max_sec": 0.5,
        "p50_sec": 0.41,
        "p95_sec": 0.491,
        "slowest_call_index": 2,
    }
    profile_payload = _profile_payload()
    profile_payload.update(
        {
            "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics": True,
            "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled": True,
            "compact_rollout_slab_sample_gate_cuda_sync_count": 6,
            "compact_rollout_slab_sample_gate_cuda_sync_sec": 0.61,
            builder_diag_key: True,
            builder_enabled_key: True,
            builder_count_key: 8,
            builder_sec_key: 0.82,
            "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics": True,
            "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled": True,
            "compact_rollout_slab_learner_gate_cuda_sync_count": 10,
            "compact_rollout_slab_learner_gate_cuda_sync_sec": 1.03,
            "compact_profile_runtime_step_timing_diagnostics": True,
            "compact_profile_runtime_step_timing_stats": runtime_step_stats,
            "compact_rollout_slab_sample_gate_per_call_stats": sample_gate_per_call_stats,
            "compact_rollout_slab_sample_gate_call_trace_records": (sample_gate_call_trace_records),
            "compact_rollout_slab_sample_gate_candidate_per_call_stats": (
                sample_gate_candidate_per_call_stats
            ),
            "compact_rollout_slab_sample_gate_rng_per_call_stats": (sample_gate_rng_per_call_stats),
            "compact_rollout_slab_sample_gate_residual_per_call_stats": (
                sample_gate_residual_per_call_stats
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_stats": (
                builder_group_loop_per_call_stats
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec": 0.31,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec": 0.10,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_process_cpu_time_delta_ns": 9100,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_thread_cpu_time_delta_ns": 8900,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_process_cpu_time_delta_ns": 6500,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_process_cpu_time_delta_ns": 2600,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_process_cpu_time_delta_ns": 22,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_accounted_process_cpu_time_delta_ns": 20,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_residual_process_cpu_time_delta_ns": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_snapshot_process_cpu_time_delta_ns": 8,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_index_process_cpu_time_delta_ns": 7,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_observation_process_cpu_time_delta_ns": 5,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_terminal_value_bookkeeping_thread_cpu_time_delta_ns": 3,
            "compact_rollout_slab_sample_gate_terminal_final_observation_group_count": 4,
            "compact_rollout_slab_sample_gate_terminal_final_observation_fallback_count": 4,
            "compact_rollout_slab_sample_gate_terminal_final_observation_validate_only_count": 4,
            "compact_rollout_slab_sample_gate_terminal_final_observation_materialized_count": 0,
            "compact_rollout_slab_sample_gate_terminal_final_observation_final_row_count_sum": 5,
            "compact_rollout_slab_sample_gate_terminal_final_observation_sparse_row_count_max": 3,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_process_cpu_time_delta_ns": 4300,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_process_cpu_time_delta_ns": 3600,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_process_cpu_time_delta_ns": 700,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_process_cpu_time_delta_ns": 2100,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_process_cpu_time_delta_ns": 2600,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_process_cpu_time_delta_ns": 400,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_accounted_process_cpu_time_delta_ns": 1800,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_residual_process_cpu_time_delta_ns": 300,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_gather_process_cpu_time_delta_ns": 9,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_storage_process_cpu_time_delta_ns": 3,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_validate_process_cpu_time_delta_ns": 6,
            "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_thread_cpu_time_delta_ns": 1200,
            "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_process_cpu_time_delta_ns": 800,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_stats": (
                builder_group_loop_accounted_per_call_stats
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_stats": (
                builder_group_loop_residual_per_call_stats
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_per_call_stats": {
                "count": 2,
                "sum_sec": 0.04,
                "min_sec": 0.01,
                "max_sec": 0.03,
                "p50_sec": 0.02,
                "p95_sec": 0.029,
                "slowest_call_index": 2,
            },
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_gather_per_call_stats": {
                "count": 2,
                "sum_sec": 0.03,
                "min_sec": 0.01,
                "max_sec": 0.02,
                "p50_sec": 0.015,
                "p95_sec": 0.0195,
                "slowest_call_index": 2,
            },
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_validate_per_call_stats": {
                "count": 2,
                "sum_sec": 0.02,
                "min_sec": 0.005,
                "max_sec": 0.015,
                "p50_sec": 0.01,
                "p95_sec": 0.0145,
                "slowest_call_index": 2,
            },
            "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_per_call_stats": (
                builder_cuda_sync_per_call_stats
            ),
        }
    )

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=profile_payload,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    for payload in (summary, compact):
        assert payload["compact_profile_cuda_sync_timing_diagnostics"] is True
        assert payload["compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics"] is True
        assert payload["compact_rollout_slab_sample_gate_cuda_sync_timing_enabled"] is True
        assert payload["compact_rollout_slab_sample_gate_cuda_sync_count"] == 6
        assert payload["compact_rollout_slab_sample_gate_cuda_sync_sec"] == 0.61
        assert payload[builder_diag_key] is True
        assert payload[builder_enabled_key] is True
        assert payload[builder_count_key] == 8
        assert payload[builder_sec_key] == 0.82
        assert payload["compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics"] is True
        assert payload["compact_rollout_slab_learner_gate_cuda_sync_timing_enabled"] is True
        assert payload["compact_rollout_slab_learner_gate_cuda_sync_count"] == 10
        assert payload["compact_rollout_slab_learner_gate_cuda_sync_sec"] == 1.03
        assert payload["compact_profile_runtime_step_timing_diagnostics"] is True
        assert payload["compact_profile_runtime_step_timing_stats"] == runtime_step_stats
        assert payload["compact_profile_runtime_step_count"] == 3
        assert payload["compact_profile_runtime_step_sum_sec"] == 0.91
        assert payload["compact_profile_runtime_step_min_sec"] == 0.21
        assert payload["compact_profile_runtime_step_max_sec"] == 0.41
        assert payload["compact_profile_runtime_step_p50_sec"] == 0.29
        assert payload["compact_profile_runtime_step_p95_sec"] == 0.39
        assert payload["compact_profile_runtime_step_slowest_iteration"] == 4
        assert payload["compact_profile_runtime_step_slowest_measured_iteration"] == 3
        assert payload["compact_profile_runtime_step_slowest_actor_step_wall_sec"] == 0.13
        assert payload["compact_profile_runtime_step_slowest_observation_sec"] == 0.07
        assert payload["compact_profile_runtime_step_slowest_compact_rollout_slab_sec"] == 0.03
        assert payload["compact_profile_runtime_step_slowest_sample_gate_sec"] == 0.11
        assert payload["compact_profile_runtime_step_slowest_learner_gate_sec"] == 0.05
        assert payload["compact_profile_runtime_step_slowest_policy_refresh_sec"] == 0.01
        assert payload["compact_profile_runtime_step_slowest_primary_accounted_sec"] == 0.4
        assert payload["compact_profile_runtime_step_slowest_primary_residual_sec"] == 0.01
        assert payload["compact_profile_runtime_step_slowest_env_trajectory_checksum"] == 123456
        assert (
            payload["compact_profile_runtime_step_top_slowest_records"]
            == (runtime_step_stats["top_slowest_records"])
        )
        assert payload["compact_profile_runtime_step_sample_gate_active_count"] == 2
        assert payload["compact_profile_runtime_step_sample_gate_active_sum_sec"] == 0.62
        assert (
            payload["compact_profile_runtime_step_sample_gate_active_sample_gate_sum_sec"] == 0.41
        )
        assert payload["compact_profile_runtime_step_sample_gate_active_sample_gate_count"] == 2
        assert (
            payload["compact_profile_runtime_step_sample_gate_active_sample_gate_p95_sec"] == 0.285
        )
        assert (
            payload[
                "compact_profile_runtime_step_sample_gate_active_sample_gate_builder_group_loop_p50_sec"
            ]
            == 0.10
        )
        assert payload["compact_profile_runtime_step_sample_gate_inactive_count"] == 1
        assert payload["compact_profile_runtime_step_sample_gate_inactive_sum_sec"] == 0.29
        assert payload["compact_profile_runtime_step_early_count"] == 1
        assert payload["compact_profile_runtime_step_early_sample_gate_active_count"] == 1
        assert payload["compact_profile_runtime_step_early_sample_gate_sum_sec"] == 0.10
        assert (
            payload["compact_profile_runtime_step_early_sample_gate_active_sample_gate_p95_sec"]
            == 0.10
        )
        assert payload["compact_profile_runtime_step_mid_count"] == 1
        assert payload["compact_profile_runtime_step_mid_sample_gate_active_count"] == 0
        assert payload["compact_profile_runtime_step_mid_sample_gate_active_sample_gate_count"] == 0
        assert payload["compact_profile_runtime_step_late_count"] == 1
        assert payload["compact_profile_runtime_step_late_sample_gate_sum_sec"] == 0.31
        assert (
            payload["compact_profile_runtime_step_late_sample_gate_active_sample_gate_count"] == 1
        )
        assert (
            payload["compact_profile_runtime_step_late_sample_gate_active_sample_gate_p50_sec"]
            == 0.31
        )
        assert (
            payload[
                "compact_profile_runtime_step_late_sample_gate_active_sample_gate_builder_group_loop_p95_sec"
            ]
            == 0.15
        )
        assert payload["compact_profile_runtime_step_actor_step_wall_sum_sec"] == 0.31
        assert payload["compact_profile_runtime_step_actor_step_wall_p95_sec"] == 0.12
        assert payload["compact_profile_runtime_step_actor_env_runtime_sum_sec"] == 0.22
        assert payload["compact_profile_runtime_step_actor_env_runtime_p95_sec"] == 0.085
        assert payload["compact_profile_runtime_step_actor_autoreset_sum_sec"] == 0.07
        assert payload["compact_profile_runtime_step_actor_autoreset_p95_sec"] == 0.029
        assert payload["compact_profile_runtime_step_sample_gate_sum_sec"] == 0.41
        assert payload["compact_profile_runtime_step_sample_gate_p95_sec"] == 0.20
        assert payload["compact_profile_runtime_step_sample_gate_residual_sum_sec"] == 0.05
        assert payload["compact_profile_runtime_step_sample_gate_residual_p95_sec"] == 0.029
        assert payload["compact_profile_runtime_step_sample_gate_cuda_sync_sum_sec"] == 0.01
        assert payload["compact_profile_runtime_step_sample_gate_cuda_sync_p95_sec"] == 0.0058
        assert (
            payload["compact_profile_runtime_step_sample_gate_builder_group_loop_sum_sec"] == 0.20
        )
        assert (
            payload["compact_profile_runtime_step_sample_gate_builder_group_loop_p95_sec"] == 0.088
        )
        assert payload["compact_profile_runtime_step_sample_gate_builder_cuda_sync_sum_sec"] == 0.04
        assert (
            payload["compact_profile_runtime_step_sample_gate_builder_cuda_sync_p95_sec"] == 0.019
        )
        assert payload["compact_profile_runtime_step_primary_residual_sum_sec"] == 0.03
        assert payload["compact_profile_runtime_step_primary_residual_p95_sec"] == 0.019
        assert payload["compact_rollout_slab_sample_gate_per_call_stats"] == (
            sample_gate_per_call_stats
        )
        assert payload["compact_rollout_slab_sample_gate_call_trace_records"] == (
            sample_gate_call_trace_records
        )
        assert payload["compact_rollout_slab_sample_gate_per_call_count"] == 2
        assert payload["compact_rollout_slab_sample_gate_per_call_sum_sec"] == 0.22
        assert payload["compact_rollout_slab_sample_gate_per_call_p95_sec"] == 0.119
        assert payload["compact_rollout_slab_sample_gate_per_call_slowest_iteration"] == 17
        assert payload["compact_rollout_slab_sample_gate_per_call_slowest_measured_iteration"] == 9
        assert payload["compact_rollout_slab_sample_gate_candidate_per_call_stats"] == (
            sample_gate_candidate_per_call_stats
        )
        assert payload["compact_rollout_slab_sample_gate_candidate_per_call_p50_sec"] == 0.035
        assert payload["compact_rollout_slab_sample_gate_rng_per_call_stats"] == (
            sample_gate_rng_per_call_stats
        )
        assert payload["compact_rollout_slab_sample_gate_rng_per_call_max_sec"] == 0.02
        assert payload["compact_rollout_slab_sample_gate_residual_per_call_stats"] == (
            sample_gate_residual_per_call_stats
        )
        assert payload["compact_rollout_slab_sample_gate_residual_per_call_min_sec"] == 0.02
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_stats"
            ]
            == builder_group_loop_per_call_stats
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec"
            ]
            == 0.31
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec"
            ]
            == 0.10
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_process_cpu_time_delta_ns"
            ]
            == 9100
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_thread_cpu_time_delta_ns"
            ]
            == 8900
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_process_cpu_time_delta_ns"
            ]
            == 6500
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_process_cpu_time_delta_ns"
            ]
            == 2600
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_process_cpu_time_delta_ns"
            ]
            == 22
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_snapshot_process_cpu_time_delta_ns"
            ]
            == 8
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_process_cpu_time_delta_ns"
            ]
            == 700
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_process_cpu_time_delta_ns"
            ]
            == 400
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_gather_process_cpu_time_delta_ns"
            ]
            == 9
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_storage_process_cpu_time_delta_ns"
            ]
            == 3
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_validate_process_cpu_time_delta_ns"
            ]
            == 6
        )
        assert (
            payload["compact_rollout_slab_sample_gate_terminal_final_observation_group_count"] == 4
        )
        assert (
            payload["compact_rollout_slab_sample_gate_terminal_final_observation_fallback_count"]
            == 4
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_terminal_final_observation_validate_only_count"
            ]
            == 4
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_terminal_final_observation_materialized_count"
            ]
            == 0
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_terminal_final_observation_sparse_row_count_max"
            ]
            == 3
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_process_cpu_time_delta_ns"
            ]
            == 4300
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_process_cpu_time_delta_ns"
            ]
            == 2100
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_thread_cpu_time_delta_ns"
            ]
            == 1200
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_process_cpu_time_delta_ns"
            ]
            == 800
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_p95_sec"
            ]
            == 0.2185
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_stats"
            ]
            == builder_group_loop_accounted_per_call_stats
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_p95_sec"
            ]
            == 0.1685
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_stats"
            ]
            == builder_group_loop_residual_per_call_stats
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_sum_sec"
            ]
            == 0.10
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_per_call_p95_sec"
            ]
            == 0.029
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_gather_per_call_sum_sec"
            ]
            == 0.03
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_validate_per_call_sum_sec"
            ]
            == 0.02
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_per_call_stats"
            ]
            == builder_cuda_sync_per_call_stats
        )
        assert (
            payload[
                "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_per_call_sum_sec"
            ]
            == 0.82
        )


def test_speed_row_smoke_allows_borrowed_normal_death_terminal_snapshots():
    module = _load_smoke_module()
    args = SimpleNamespace(
        run_id="unit-normal-death-speed-row",
        search_service_kind="compact_torch_search_service",
        hybrid_persistent_compact_render_state_buffer=False,
        hybrid_borrow_single_actor_render_state=True,
        compact_owned_loop_deferred_learner=False,
        compact_torch_initial_inference_mode="direct_core",
        compact_torch_observation_memory_format="contiguous",
        compact_torch_model_memory_format="contiguous",
        compact_profile_bounded_diagnostics=False,
        actor_count=1,
        batch_size=2,
        steps=4,
        warmup_steps=1,
        death_mode="normal",
    )

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=_normal_death_profile_payload(),
        loaded_checkpoint_identity=_loaded_identity(),
    )

    for payload in (summary, compact):
        assert payload["death_mode"] == "normal"
        assert payload["render_state_handoff_mode"] == "borrow_single_actor_env_state"
        assert payload["render_state_copy_steps"] == 0
        assert payload["render_state_borrowed_steps"] == 5
        assert payload["render_state_row_overlay_steps"] == 1
        assert payload["render_state_row_overlay_rows"] == 3
        assert payload["render_state_row_overlay_bytes"] == 1024
        assert payload["terminal_row_count"] == 3
        assert payload["terminal_sample_row_count"] == 1
        assert payload["terminal_unroll_value_target_row_count"] == 1
        assert payload["normal_death_terminal_contract_promotion_gate_satisfied"] is True
        assert payload["normal_death_terminal_contract"]["death_mode"] == "normal"
        assert (
            payload["normal_death_terminal_contract_evidence_id"]
            == "unit-normal-death-speed-row:normal_death_speed_row"
        )


def test_speed_row_smoke_requires_and_emits_fused_learner_batch_proof():
    module = _load_smoke_module()
    args = _summary_args()
    args.death_mode = "normal"
    args.hybrid_borrow_single_actor_render_state = True
    args.compact_owned_loop_fused_learner_batch = True
    payload = _normal_death_profile_payload()
    payload["compact_rollout_slab_sample_gate_sec"] = 0.12
    payload["compact_rollout_slab_learner_gate_sec"] = 0.34
    payload["compact_rollout_slab_sample_gate_last_telemetry"].update(
        {
            "compact_rollout_slab_sample_gate_candidate_sec": 0.01,
            "compact_rollout_slab_sample_gate_candidate_universe_source": (
                "maintained_sample_universe_v1"
            ),
            "compact_rollout_slab_sample_gate_candidate_universe_cache_hit": True,
            "compact_rollout_slab_sample_gate_candidate_universe_snapshot_version": 32,
            "compact_rollout_slab_sample_gate_candidate_offset_checksum": 174362,
            "compact_rollout_slab_sample_gate_rng_sec": 0.02,
            "compact_rollout_slab_sample_gate_resident_check_sec": 0.03,
            "compact_rollout_slab_sample_gate_group_loop_sec": 0.04,
            "compact_rollout_slab_sample_gate_metadata_sec": 0.05,
            "compact_rollout_slab_sample_gate_learner_batch_build_sec": 0.06,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_sec": 0.061,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec": 0.055,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec": 0.006,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_sec": 0.0061,
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_terminal_value_bookkeeping_sec": 0.0062,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_sec": 0.062,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_sec": 0.0621,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_tensor_fallback_sec": 0.0622,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_validate_sec": 0.0623,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_sec": 0.0624,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_presence_sec": 0.0625,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_select_current_sec": 0.0626,
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_gather_sec": 0.0627,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_sec": 0.063,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_sec": 0.0630,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_identity_sec": 0.0631,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_sec": 0.0632,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_build_sec": 0.0633,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_value_sec": 0.0634,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_apply_sec": 0.0635,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_action_stack_sec": 0.0636,
            "compact_rollout_slab_sample_gate_learner_batch_builder_write_output_sec": 0.064,
            "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_sec": 0.065,
            "compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_sec": 0.066,
            "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_sec": 0.067,
            "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_sec": 0.068,
            "compact_rollout_slab_sample_gate_sample_batch_build_sec": 0.0,
            "compact_rollout_slab_sample_gate_accounted_sec": 0.21,
            "compact_rollout_slab_sample_gate_residual_sec": -0.09,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
            "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": True,
            "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": True,
            "compact_rollout_slab_sample_gate_explicit_next_targets": True,
            "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
            "compact_rollout_slab_sample_gate_explicit_unroll_target_group_count": 3,
            "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order": ("rng"),
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count": 13,
            "compact_rollout_slab_sample_gate_learner_batch_build_per_call_stats": {
                "count": 2,
                "sum_sec": 0.06,
                "max_sec": 0.04,
                "p95_sec": 0.039,
                "slowest_call_index": 2,
            },
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_stats": {
                "count": 2,
                "sum_sec": 0.061,
                "max_sec": 0.04,
                "p95_sec": 0.0395,
                "slowest_call_index": 2,
            },
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_stats": {
                "count": 2,
                "sum_sec": 0.055,
                "max_sec": 0.036,
                "p95_sec": 0.035,
                "slowest_call_index": 2,
            },
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_stats": {
                "count": 2,
                "sum_sec": 0.006,
                "max_sec": 0.004,
                "p95_sec": 0.0038,
                "slowest_call_index": 2,
            },
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call_stats": {
                "count": 2,
                "sum_sec": 0.0621,
                "max_sec": 0.0401,
                "p95_sec": 0.0396,
                "slowest_call_index": 2,
            },
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_per_call_stats": {
                "count": 2,
                "sum_sec": 0.0632,
                "max_sec": 0.0402,
                "p95_sec": 0.0397,
                "slowest_call_index": 2,
            },
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call_stats": {
                "count": 2,
                "sum_sec": 0.0630,
                "max_sec": 0.0400,
                "p95_sec": 0.0394,
                "slowest_call_index": 2,
            },
        }
    )
    payload["compact_rollout_slab_learner_gate_last_telemetry"].update(
        {
            "compact_rollout_slab_learner_gate_prebuilt_batch_used": True,
            "compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled": False,
            "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep": False,
            "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used": True,
            "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count": 1,
            "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count": 7,
            "compact_rollout_slab_learner_gate_validation_sec": 0.11,
            "compact_rollout_slab_learner_gate_zero_grad_sec": 0.12,
            "compact_rollout_slab_learner_gate_target_transform_sec": 0.13,
            "compact_rollout_slab_learner_gate_initial_inference_sec": 0.14,
            "compact_rollout_slab_learner_gate_recurrent_inference_sec": 0.15,
            "compact_rollout_slab_learner_gate_loss_build_sec": 0.16,
            "compact_rollout_slab_learner_gate_backward_sec": 0.17,
            "compact_rollout_slab_learner_gate_grad_clip_sec": 0.18,
            "compact_rollout_slab_learner_gate_optimizer_step_sec": 0.19,
            "compact_rollout_slab_learner_gate_loss_readback_sec": 0.20,
            "compact_rollout_slab_learner_gate_final_sync_sec": 0.21,
            "compact_rollout_slab_learner_gate_accounted_sec": 1.76,
            "compact_rollout_slab_learner_gate_residual_sec": -1.42,
        }
    )

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    for row in (summary, compact):
        assert row["compact_owned_loop_fused_learner_batch"] is True
        assert row["compact_rollout_slab_sample_gate_sec"] == 0.12
        assert row["compact_rollout_slab_learner_gate_sec"] == 0.34
        assert row["compact_rollout_slab_sample_gate_compact_muzero_learner_batch"] is True
        assert row["compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"] is True
        assert row["compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch"] is True
        assert (
            row["compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"]
            is True
        )
        assert row["compact_rollout_slab_sample_gate_explicit_next_targets"] is True
        assert row["compact_rollout_slab_sample_gate_explicit_unroll_targets"] is True
        assert row["compact_rollout_slab_sample_gate_explicit_unroll_target_group_count"] == 3
        assert row["compact_rollout_slab_sample_gate_num_unroll_steps"] == 2
        assert row["compact_rollout_slab_sample_gate_terminal_unroll_windows_supported"] is True
        assert row["compact_rollout_slab_learner_gate_prebuilt_batch_used"] is True
        assert (
            row["compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order"]
            == "rng"
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order"
            ]
            is True
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count"
            ]
            == 13
        )
        assert row["compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled"] is False
        assert row["compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used"] is True
        assert row["compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count"] == 1
        assert row["compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count"] == 7
        assert row["compact_rollout_slab_sample_gate_candidate_sec"] == 0.01
        assert row["compact_rollout_slab_sample_gate_candidate_universe_source"] == (
            "maintained_sample_universe_v1"
        )
        assert row["compact_rollout_slab_sample_gate_candidate_universe_cache_hit"] is True
        assert row["compact_rollout_slab_sample_gate_candidate_universe_snapshot_version"] == 32
        assert row["compact_rollout_slab_sample_gate_candidate_offset_checksum"] == 174362
        assert row["compact_rollout_slab_sample_gate_learner_batch_build_sec"] == 0.06
        assert row["compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_sec"] == 0.061
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec"]
            == 0.055
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec"]
            == 0.006
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_sec"]
            == 0.062
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_sec"]
            == 0.0621
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_sec"] == 0.063
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_sec"
            ]
            == 0.0630
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_sec"]
            == 0.0632
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_write_output_sec"] == 0.064
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_sec"] == 0.065
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_sec"]
            == 0.066
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_sec"] == 0.067
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_sec"]
            == 0.068
        )
        assert row["compact_rollout_slab_sample_gate_learner_batch_build_per_call_stats"] == {
            "count": 2,
            "sum_sec": 0.06,
            "max_sec": 0.04,
            "p95_sec": 0.039,
            "slowest_call_index": 2,
        }
        assert row["compact_rollout_slab_sample_gate_learner_batch_build_per_call_p95_sec"] == 0.039
        assert row[
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_stats"
        ] == {
            "count": 2,
            "sum_sec": 0.061,
            "max_sec": 0.04,
            "p95_sec": 0.0395,
            "slowest_call_index": 2,
        }
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_p95_sec"
            ]
            == 0.0395
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_p95_sec"
            ]
            == 0.035
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_sum_sec"
            ]
            == 0.006
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call_p95_sec"
            ]
            == 0.0396
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_per_call_p95_sec"
            ]
            == 0.0397
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call_p95_sec"
            ]
            == 0.0394
        )
        assert row["compact_rollout_slab_sample_gate_accounted_sec"] == 0.21
        assert row["compact_rollout_slab_learner_gate_validation_sec"] == 0.11
        assert row["compact_rollout_slab_learner_gate_backward_sec"] == 0.17
        assert row["compact_rollout_slab_learner_gate_accounted_sec"] == 1.76

    broken = _normal_death_profile_payload()
    with pytest.raises(ValueError, match="fused sample/learner proof"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=broken,
            loaded_checkpoint_identity=_loaded_identity(),
        )

    host_payload = _normal_death_profile_payload()
    host_payload["compact_rollout_slab_sample_gate_last_telemetry"].update(
        {
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
            "compact_rollout_slab_sample_gate_host_provider_learner_batch": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source": (
                "host_provider_sample_batch_builder_v1"
            ),
            "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes": 4096,
        }
    )
    host_payload["compact_rollout_slab_learner_gate_last_telemetry"].update(
        {"compact_rollout_slab_learner_gate_prebuilt_batch_used": True}
    )
    host_summary, host_compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=host_payload,
        loaded_checkpoint_identity=_loaded_identity(),
    )
    for row in (host_summary, host_compact):
        assert row["compact_rollout_slab_sample_gate_host_provider_learner_batch"] is True
        assert (
            row[
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source"
            ]
            == "host_provider_sample_batch_builder_v1"
        )
        assert (
            row["compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes"] == 4096
        )


def test_speed_row_smoke_requires_unroll2_specialized_builder_proof():
    module = _load_smoke_module()
    args = _summary_args()
    args.death_mode = "normal"
    args.hybrid_borrow_single_actor_render_state = True
    args.compact_owned_loop_fused_learner_batch = True
    args.compact_muzero_learner_batch_unroll2_specialized_builder = True
    args.learner_num_unroll_steps = 2
    payload = _normal_death_profile_payload()
    payload["compact_rollout_slab_sample_gate_sec"] = 0.12
    payload["compact_rollout_slab_learner_gate_sec"] = 0.34
    payload["compact_rollout_slab_sample_gate_last_telemetry"].update(
        {
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
            "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": True,
            "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": True,
            "compact_rollout_slab_sample_gate_explicit_next_targets": True,
            "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
            "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order": "rng",
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count": 1,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count": 0,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason": "none",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl": "unroll2_specialized_v1",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "unroll2_specialized",
        }
    )
    payload["compact_rollout_slab_learner_gate_last_telemetry"].update(
        {"compact_rollout_slab_learner_gate_prebuilt_batch_used": True}
    )

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    for row in (summary, compact):
        assert row["compact_muzero_learner_batch_unroll2_specialized_builder"] is True
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested"
            ]
            is True
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count"
            ]
            == 2
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used"
            ]
            is True
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count"
            ]
            == 2
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count"
            ]
            == 0
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason"
            ]
            == "none"
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl"
            ]
            == "unroll2_specialized_v1"
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path"]
            == "unroll2_specialized"
        )

    broken = _normal_death_profile_payload()
    broken["compact_rollout_slab_sample_gate_sec"] = 0.12
    broken["compact_rollout_slab_learner_gate_sec"] = 0.34
    broken["compact_rollout_slab_sample_gate_last_telemetry"].update(
        dict(payload["compact_rollout_slab_sample_gate_last_telemetry"])
    )
    broken["compact_rollout_slab_sample_gate_last_telemetry"][
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count"
    ] = 1
    broken["compact_rollout_slab_learner_gate_last_telemetry"].update(
        {"compact_rollout_slab_learner_gate_prebuilt_batch_used": True}
    )
    with pytest.raises(ValueError, match="unroll2_specialized_builder_fallback_count"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=broken,
            loaded_checkpoint_identity=_loaded_identity(),
        )
    for field, value, error in (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason",
            "guard_failed",
            "unroll2_specialized_builder_fallback_reason",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl",
            "generic",
            "unroll2_specialized_builder_impl",
        ),
    ):
        broken = _normal_death_profile_payload()
        broken["compact_rollout_slab_sample_gate_sec"] = 0.12
        broken["compact_rollout_slab_learner_gate_sec"] = 0.34
        broken["compact_rollout_slab_sample_gate_last_telemetry"].update(
            dict(payload["compact_rollout_slab_sample_gate_last_telemetry"])
        )
        broken["compact_rollout_slab_sample_gate_last_telemetry"][field] = value
        broken["compact_rollout_slab_learner_gate_last_telemetry"].update(
            {"compact_rollout_slab_learner_gate_prebuilt_batch_used": True}
        )
        with pytest.raises(ValueError, match=error):
            module._speed_summary_and_compact_payload(
                args=args,
                candidate_checkpoint_id="unit-compact-ckpt",
                profile_payload=broken,
                loaded_checkpoint_identity=_loaded_identity(),
            )


def test_speed_row_smoke_requires_tensor_native_replay_proof():
    module = _load_smoke_module()
    args = _summary_args()
    args.death_mode = "normal"
    args.hybrid_borrow_single_actor_render_state = True
    args.compact_owned_loop_fused_learner_batch = True
    args.compact_muzero_learner_batch_learner_ready_unroll2_cache = True
    args.compact_muzero_learner_batch_tensor_native_replay = True
    args.learner_num_unroll_steps = 2
    payload = _normal_death_profile_payload()
    payload["compact_rollout_slab_sample_gate_sec"] = 0.12
    payload["compact_rollout_slab_learner_gate_sec"] = 0.34
    payload["compact_rollout_slab_sample_gate_last_telemetry"].update(
        {
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
            "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": True,
            "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": True,
            "compact_rollout_slab_sample_gate_explicit_next_targets": True,
            "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
            "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order": "rng",
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count": 1,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count": 0,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason": "none",
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": "learner_ready_unroll2_cache_v1",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count": 0,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason": "none",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": "maintained_unroll2_table_gather_v1",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": "maintained_record_table_v1",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count": 2,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count": 0,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows": 128,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec": 0.001,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec": 0.002,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used": True,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count": 0,
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason": "none",
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count": 0,
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped": True,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "learner_ready_unroll2_cache",
        }
    )
    payload["compact_rollout_slab_learner_gate_last_telemetry"].update(
        {"compact_rollout_slab_learner_gate_prebuilt_batch_used": True}
    )

    summary, compact = module._speed_summary_and_compact_payload(
        args=args,
        candidate_checkpoint_id="unit-compact-ckpt",
        profile_payload=payload,
        loaded_checkpoint_identity=_loaded_identity(),
    )

    for row in (summary, compact):
        assert row["compact_muzero_learner_batch_tensor_native_replay"] is True
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested"
            ]
            is True
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"]
            is True
        )
        assert (
            row["compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"]
            == "maintained_unroll2_table_gather_v1"
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source"
            ]
            == "maintained_record_table_v1"
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count"
            ]
            == 2
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count"
            ]
            == 0
        )
        assert (
            row[
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows"
            ]
            == 128
        )
        assert (
            row["compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used"] is True
        )
        assert row["compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count"] == 0

    for field, value, error in (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used",
            False,
            "tensor_native_replay_used",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
            1,
            "tensor_native_replay_fallback_count",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
            "generic",
            "tensor_native_replay_impl",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
            "rebuilt_on_sample_v0",
            "tensor_native_replay_table_source",
        ),
        (
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used",
            False,
            "tensor_native_direct_prebuilt_path_used",
        ),
        (
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count",
            1,
            "tensor_native_direct_group_object_count",
        ),
    ):
        broken = _normal_death_profile_payload()
        broken["compact_rollout_slab_sample_gate_sec"] = 0.12
        broken["compact_rollout_slab_learner_gate_sec"] = 0.34
        broken["compact_rollout_slab_sample_gate_last_telemetry"].update(
            dict(payload["compact_rollout_slab_sample_gate_last_telemetry"])
        )
        broken["compact_rollout_slab_sample_gate_last_telemetry"][field] = value
        broken["compact_rollout_slab_learner_gate_last_telemetry"].update(
            {"compact_rollout_slab_learner_gate_prebuilt_batch_used": True}
        )
        with pytest.raises(ValueError, match=error):
            module._speed_summary_and_compact_payload(
                args=args,
                candidate_checkpoint_id="unit-compact-ckpt",
                profile_payload=broken,
                loaded_checkpoint_identity=_loaded_identity(),
            )


def test_speed_row_smoke_rejects_borrowed_normal_death_without_contract_proof():
    module = _load_smoke_module()
    args = SimpleNamespace(
        run_id="unit-normal-death-speed-row",
        search_service_kind="compact_torch_search_service",
        hybrid_persistent_compact_render_state_buffer=False,
        hybrid_borrow_single_actor_render_state=True,
        compact_owned_loop_deferred_learner=False,
        compact_torch_initial_inference_mode="direct_core",
        compact_torch_observation_memory_format="contiguous",
        compact_torch_model_memory_format="contiguous",
        compact_profile_bounded_diagnostics=False,
        actor_count=1,
        batch_size=2,
        steps=4,
        warmup_steps=1,
        death_mode="normal",
    )
    profile_payload = _profile_payload(borrow_render_state=True)
    profile_payload["death_mode"] = "normal"
    profile_payload["compact_rollout_slab_learner_gate_num_unroll_steps"] = 2
    profile_payload["contract"]["render_state_copy_steps"] = 1

    with pytest.raises(ValueError, match="normal-death speed row requires"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=profile_payload,
            loaded_checkpoint_identity=_loaded_identity(),
        )


def test_speed_row_smoke_rejects_profile_no_death_borrowed_copy_steps():
    module = _load_smoke_module()
    args = SimpleNamespace(
        run_id="unit-no-death-speed-row",
        search_service_kind="compact_torch_search_service",
        hybrid_persistent_compact_render_state_buffer=False,
        hybrid_borrow_single_actor_render_state=True,
        compact_owned_loop_deferred_learner=False,
        compact_torch_initial_inference_mode="direct_core",
        compact_torch_observation_memory_format="contiguous",
        compact_torch_model_memory_format="contiguous",
        actor_count=1,
        batch_size=2,
        steps=4,
        warmup_steps=1,
        death_mode="profile_no_death",
    )
    profile_payload = _profile_payload(borrow_render_state=True)
    profile_payload["contract"]["render_state_copy_steps"] = 1

    with pytest.raises(ValueError, match="profile_no_death borrowed"):
        module._speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=profile_payload,
            loaded_checkpoint_identity=_loaded_identity(),
        )


def test_speed_row_smoke_rejects_model_wide_channels_last_before_launch():
    module = _load_smoke_module()

    with pytest.raises(ValueError, match="model_memory_format=channels_last is parked"):
        module.main(
            [
                "--run-id",
                "unit-model-channels-last-rejected",
                "--unified-lifecycle-report",
                "unused.json",
                "--compact-torch-model-memory-format",
                "channels_last",
            ]
        )


def test_speed_row_smoke_rejects_deferred_compact_torch_shape_without_post_refresh_step():
    module = _load_smoke_module()

    with pytest.raises(ValueError, match="post-refresh actor/search step"):
        module.main(
            [
                "--run-id",
                "unit-no-post-refresh-step",
                "--unified-lifecycle-report",
                "unused.json",
                "--steps",
                "32",
                "--warmup-steps",
                "4",
                "--sample-interval",
                "4",
                "--compact-owned-loop-deferred-sample-learner",
                "--search-service-kind",
                "compact_torch_search_service",
            ]
        )


def test_speed_row_smoke_rejects_fixed_action_result_buffer_without_deferred_maintenance():
    module = _load_smoke_module()

    with pytest.raises(
        ValueError,
        match="--owner-search-fixed-action-result-buffer requires --owner-search-defer-maintenance",
    ):
        module.main(
            [
                "--run-id",
                "unit-fixed-action-result-no-defer",
                "--unified-lifecycle-report",
                "unused.json",
                "--search-service-kind",
                "owner_search_threaded_proxy",
                "--owner-search-slab-bypass",
                "--owner-search-require-resident-root-view",
                "--owner-search-resident-root-host-observation-stub",
                "--owner-search-direct-root-build-request",
                "--owner-search-fixed-action-result-buffer",
            ]
        )


def test_modal_speed_row_rejects_deferred_compact_torch_shape_without_post_refresh_step(
    monkeypatch,
):
    module = _load_modal_runner_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_compact_coach_speed_row_modal_smoke.py",
            "--run-id",
            "unit-modal-no-post-refresh-step",
            "--unified-lifecycle-report",
            "unused.json",
            "--steps",
            "32",
            "--warmup-steps",
            "4",
            "--sample-interval",
            "4",
            "--compact-owned-loop-deferred-sample-learner",
            "--search-service-kind",
            "compact_torch_search_service",
        ],
    )

    with pytest.raises(ValueError, match="post-refresh actor/search step"):
        module.main()


def test_speed_row_smoke_requires_final_post_refresh_search_and_replay_metadata():
    module = _load_smoke_module()
    profile_payload = _profile_payload()
    profile_payload[
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata"
    ] = {
        "compact_policy_refresh_search_worker_refreshed": True,
        "compact_policy_refresh_learner_update_count": 0,
        "compact_policy_refresh_model_state_digest": "c" * 64,
    }

    with pytest.raises(ValueError, match="final search metadata update count mismatch"):
        module._speed_summary_and_compact_payload(
            args=_summary_args(),
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=profile_payload,
            loaded_checkpoint_identity=_loaded_identity(),
        )

    profile_payload = _profile_payload()
    profile_payload[
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata"
    ] = {
        "compact_policy_refresh_search_worker_refreshed": True,
        "compact_policy_refresh_learner_update_count": 1,
        "compact_policy_refresh_model_state_digest": "d" * 64,
    }

    with pytest.raises(ValueError, match="final replay metadata digest mismatch"):
        module._speed_summary_and_compact_payload(
            args=_summary_args(),
            candidate_checkpoint_id="unit-compact-ckpt",
            profile_payload=profile_payload,
            loaded_checkpoint_identity=_loaded_identity(),
        )


def test_speed_row_smoke_allows_async_replay_metadata_lag_when_reported():
    module = _load_smoke_module()
    profile_payload = _profile_payload()
    profile_payload.update(
        {
            "compact_owned_loop_defer_sample_learner_gate": True,
            "compact_rollout_slab_learner_gate_calls": 2,
            "compact_rollout_slab_learner_gate_updates": 2,
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": 2,
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest": (
                "c" * 64
            ),
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata": {
                "compact_policy_refresh_search_worker_refreshed": True,
                "compact_policy_refresh_learner_update_count": 2,
                "compact_policy_refresh_model_state_digest": "c" * 64,
            },
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata": {
                "compact_policy_refresh_search_worker_refreshed": True,
                "compact_policy_refresh_learner_update_count": 1,
                "compact_policy_refresh_model_state_digest": "b" * 64,
            },
        }
    )

    module._require_compact_torch_policy_refresh_proof(profile_payload)
    fields = module._policy_refresh_proof_fields(profile_payload)
    assert (
        fields[
            "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_lag_to_final_update"
        ]
        == 0
    )
    assert (
        fields[
            "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_lag_to_final_update"
        ]
        == 1
    )


def test_speed_row_smoke_requires_owner_ref_policy_refresh_transport_proof():
    module = _load_smoke_module()
    profile_payload = _profile_payload()
    profile_payload.update(
        {
            "compact_owned_loop_defer_sample_learner_gate": True,
            "compact_owned_loop_deferred_sample_learner_model_state_transport_kind": (
                "owner_ref_v1"
            ),
            "compact_rollout_slab_learner_gate_calls": 2,
            "compact_rollout_slab_learner_gate_updates": 2,
            "compact_rollout_slab_policy_refresh_after_learner_gate_calls": 2,
            "compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind": (
                "owner_ref_v1"
            ),
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind": (
                "owner_ref_v1"
            ),
            "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count": 2,
            "compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count": 0,
            (
                "compact_rollout_slab_policy_refresh_after_learner_gate_"
                "parent_model_state_transport_avoided"
            ): True,
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": 2,
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest": (
                "c" * 64
            ),
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata": {
                "compact_policy_refresh_search_worker_refreshed": True,
                "compact_policy_refresh_learner_update_count": 2,
                "compact_policy_refresh_model_state_digest": "c" * 64,
            },
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata": {
                "compact_policy_refresh_search_worker_refreshed": True,
                "compact_policy_refresh_learner_update_count": 1,
                "compact_policy_refresh_model_state_digest": "b" * 64,
            },
        }
    )

    module._require_compact_torch_policy_refresh_proof(profile_payload)
    fields = module._policy_refresh_proof_fields(profile_payload)
    assert fields[
        "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count"
    ] == 2

    bad_payload = dict(profile_payload)
    bad_payload["compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count"] = 1
    with pytest.raises(ValueError, match="must not use parent model state"):
        module._require_compact_torch_policy_refresh_proof(bad_payload)

    bad_payload = dict(profile_payload)
    bad_payload[
        "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count"
    ] = 1
    with pytest.raises(ValueError, match="must use owner refs for every refresh"):
        module._require_compact_torch_policy_refresh_proof(bad_payload)


def test_owner_search_resident_replay_step_omits_host_observation_copy():
    import numpy as np

    from curvyzero.training.compact_observation_contract import (
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        ResidentObservationBatchV1,
    )
    from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
    from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT

    module = _load_smoke_module()
    batch_size = 2
    player_count = 1
    stack_shape = (1, 2, 2)
    root_count = batch_size * player_count
    resident_observation = np.arange(
        batch_size * player_count * np.prod(stack_shape),
        dtype=np.uint8,
    ).reshape(batch_size, player_count, *stack_shape)
    resident = ResidentObservationBatchV1(
        device_observation=resident_observation,
        root_device_observation=resident_observation.reshape(root_count, *stack_shape),
        generation_id=7,
        batch_size=batch_size,
        player_count=player_count,
        stack_shape=stack_shape,
        dtype=str(resident_observation.dtype),
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=11,
        source_backend="unit-resident-root",
        host_fallback_allowed=False,
    )
    root_batch = CompactRootBatchV1(
        observation=np.full((root_count, *stack_shape), 99.0, dtype=np.float32),
        legal_mask=np.ones((root_count, ACTION_COUNT), dtype=np.bool_),
        active_root_mask=np.ones((root_count,), dtype=np.bool_),
        to_play=np.zeros((root_count,), dtype=np.int64),
        env_row=np.arange(root_count, dtype=np.int32),
        player=np.zeros((root_count,), dtype=np.int16),
        policy_env_id=np.arange(root_count, dtype=np.int64),
        target_reward=np.zeros((root_count,), dtype=np.float32),
        done_root=np.zeros((root_count,), dtype=np.bool_),
        final_observation=None,
        final_observation_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        terminal_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        autoreset_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        metadata={"batch_size": batch_size, "player_count": player_count},
        resident_observation=resident,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    )
    replay_store = module._OwnerSearchReplayStoreFactorySidecarV1(capacity=4)

    step = replay_store._step_from_root_batch(
        root_batch,
        SimpleNamespace(),
        role="previous",
    )

    assert step.observation is None
    assert step.resident_observation_replay_snapshot is resident
    assert step.compact_batch.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    assert step.action_mask.shape == (batch_size, player_count, ACTION_COUNT)


def test_modal_h100_failure_bundle_preserves_inner_producer_report(tmp_path):
    from curvyzero.infra.modal import compact_coach_speed_row as module

    result_dir = tmp_path / "unit-failed-speed-row"
    result_dir.mkdir()
    result_path = result_dir / "row_001_result.json"
    report_path = result_dir / "compact_coach_speed_row_smoke_report.json"
    result_path.write_text(
        json.dumps(
            {
                "ok": False,
                "problem": "inner result problem",
                "summary": {"compact_profile_bounded_diagnostics": True},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "ok": False,
                "problem": "inner report problem",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = module._speed_row_failure_bundle(
        config={
            "compact_profile_bounded_diagnostics": True,
            "compact_profile_cuda_sync_timing_diagnostics": True,
            "compact_owned_loop_fused_learner_batch": True,
            "compact_owned_lean_trainer_step": True,
            "compact_torch_initial_inference_mode": "direct_core",
            "search_service_kind": "compact_torch_search_service",
        },
        run_id="unit-failed-speed-row",
        rc=1,
        result_dir=result_dir,
    )

    assert bundle["ok"] is False
    assert bundle["problem"] == "inner report problem"
    assert bundle["producer_return_code"] == 1
    assert bundle["producer_result_present"] is True
    assert bundle["producer_report_present"] is True
    assert bundle["producer_failure_result"]["problem"] == "inner result problem"
    assert bundle["producer_failure_report"]["problem"] == "inner report problem"
    assert bundle["compact_profile_bounded_diagnostics"] is True
    assert bundle["compact_profile_cuda_sync_timing_diagnostics"] is True
    assert bundle["compact_owned_loop_fused_learner_batch"] is True
    assert bundle["compact_owned_lean_trainer_step"] is True


def test_modal_h100_cuda_sync_timing_diagnostic_violations():
    from curvyzero.infra.modal import compact_coach_speed_row as module

    summary = {
        "compact_profile_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled": True,
        "compact_rollout_slab_sample_gate_cuda_sync_count": 12,
        "compact_rollout_slab_sample_gate_cuda_sync_sec": 0.03,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count": 11977,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec": 6.9,
        "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics": True,
        "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled": True,
        "compact_rollout_slab_learner_gate_cuda_sync_count": 28,
        "compact_rollout_slab_learner_gate_cuda_sync_sec": 0.84,
    }

    assert (
        module._cuda_sync_timing_diagnostic_violations(
            requested=True,
            summary=summary,
        )
        == []
    )
    assert (
        module._cuda_sync_timing_diagnostic_violations(
            requested=False,
            summary={},
        )
        == []
    )

    missing = dict(summary)
    missing.pop("compact_rollout_slab_sample_gate_cuda_sync_timing_enabled")
    assert any(
        "sample_gate_cuda_sync_timing_enabled" in item
        for item in module._cuda_sync_timing_diagnostic_violations(
            requested=True,
            summary=missing,
        )
    )

    zero_count = dict(summary)
    zero_count["compact_rollout_slab_learner_gate_cuda_sync_count"] = 0
    assert any(
        "learner_gate_cuda_sync_count" in item and "positive" in item
        for item in module._cuda_sync_timing_diagnostic_violations(
            requested=True,
            summary=zero_count,
        )
    )

    nan_sec = dict(summary)
    nan_sec["compact_rollout_slab_sample_gate_cuda_sync_sec"] = float("nan")
    assert any(
        "sample_gate_cuda_sync_sec" in item and "finite nonnegative" in item
        for item in module._cuda_sync_timing_diagnostic_violations(
            requested=True,
            summary=nan_sec,
        )
    )


def _load_smoke_module():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / ("build_compact_coach_speed_row_smoke.py")
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_coach_speed_row_smoke_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resident_renderer_device_preserves_explicit_mps() -> None:
    module = _load_smoke_module()

    assert module._resident_renderer_device("mps") == "mps"
    assert module._resident_renderer_device("cpu") == "cpu"


def _load_modal_runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_compact_coach_speed_row_modal_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_compact_coach_speed_row_modal_smoke_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_remote_modal_producer_module():
    from curvyzero.infra.modal import compact_coach_speed_row

    return compact_coach_speed_row


def _literal_assignment_tuple(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        assert isinstance(value, tuple)
        return tuple(str(item) for item in value)
    raise AssertionError(f"assignment not found: {name}")


def _modal_launcher_args(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "run_id": "unit-lean-modal",
        "batch_size": 2,
        "actor_count": 1,
        "steps": 4,
        "warmup_steps": 1,
        "death_mode": "profile_no_death",
        "sample_batch_size": 2,
        "sample_interval": 1,
        "replay_pair_capacity": 16,
        "learner_train_steps": 1,
        "learner_num_unroll_steps": 2,
        "policy_refresh_interval": 1,
        "learner_device": "cuda",
        "num_simulations": 1,
        "search_service_kind": "compact_torch_search_service",
        "owner_search_inner_search_service_kind": "compact_torch_search_service",
        "owner_search_defer_maintenance": False,
        "owner_search_slab_bypass": False,
        "owner_search_transition_batch_size": 1,
        "owner_search_direct_transition_batch_replay": False,
        "owner_search_owner_local_transition_derivation": False,
        "owner_search_owner_proxy_transition_closure": False,
        "owner_search_require_resident_root_view": False,
        "owner_search_resident_root_host_observation_stub": False,
        "owner_search_direct_root_build_request": False,
        "compact_owner_action_step_boundary": False,
        "compact_owner_action_dispatch_step_overlap": False,
        "owner_search_fixed_action_result_buffer": False,
        "owner_search_action_result_slot_capacity": 4,
        "owner_search_fixed_soa_replay": False,
        "owner_search_defer_model_state_digest_to_refresh": False,
        "seed": 20260530,
        "source_max_steps": 1048576,
        "decision_source_frames": 1,
        "source_physics_step_ms": 16.666666666666668,
        "source_max_steps_semantics": "source_physics_steps",
        "hybrid_persistent_compact_render_state_buffer": False,
        "hybrid_borrow_single_actor_render_state": False,
        "compact_owned_loop_deferred_learner": False,
        "compact_owned_loop_deferred_sample_learner": False,
        "compact_owned_loop_deferred_sample_learner_max_pending": 1,
        "compact_owned_loop_sample_learner_worker_kind": "in_process_thread",
        "compact_owned_loop_deferred_sample_learner_replay_append_transport_kind": (
            "durable_entry_v1"
        ),
        "compact_owned_loop_fused_learner_batch": True,
        "compact_muzero_learner_batch_unroll2_specialized_builder": False,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": False,
        "compact_muzero_learner_batch_tensor_native_replay": False,
        "compact_owned_accepted_fast_path_step_window": "accepted_180_45",
        "compact_profile_bounded_diagnostics": False,
        "compact_profile_cuda_sync_timing_diagnostics": False,
        "compact_profile_runtime_step_timing_diagnostics": False,
        "compact_profile_cpu_perf_stat_diagnostics": False,
        "compact_owned_lean_trainer_step": False,
        "compact_owned_lean_profile_oracle": False,
        "gpu_utilization_sampling": False,
        "gpu_utilization_sample_interval_sec": 1.0,
        "compact_torch_request_compile": False,
        "compact_torch_request_model_compile": False,
        "compact_torch_model_compile_mode": "reduce-overhead",
        "compact_torch_timing_mode": "host_phase_sync",
        "compact_torch_initial_inference_mode": "model_method",
        "compact_torch_observation_memory_format": "contiguous",
        "compact_torch_model_memory_format": "contiguous",
        "compact_torch_defer_one_simulation_replay_payload": False,
        "launch_timeout_sec": 30.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _attach_lean_trainer_counters(payload: dict[str, Any]) -> dict[str, Any]:
    payload.update(
        {
            "compact_owned_trainer_record_step_calls": int(payload.get("steps") or 0),
            "compact_owned_trainer_learner_update_count": int(
                payload.get("compact_rollout_slab_learner_gate_updates") or 0
            ),
            "compact_owned_trainer_sample_batch_count": int(
                payload.get("compact_rollout_slab_sample_gate_calls") or 0
            ),
            "compact_owned_trainer_policy_refresh_count": int(
                payload.get("compact_rollout_slab_policy_refresh_after_learner_gate_calls") or 0
            ),
            "compact_owned_trainer_policy_version_ref": "unit-policy:update-1",
            "compact_owned_trainer_model_version_ref": "unit-model:update-1",
            "compact_owned_trainer_loop_counter_source": "run_hybrid_observation_profile",
        }
    )
    return payload


def _profile_payload(
    *,
    persistent_render_state: bool = False,
    borrow_render_state: bool = False,
) -> dict[str, Any]:
    if borrow_render_state:
        render_state_handoff_mode = "borrow_single_actor_env_state"
    elif persistent_render_state:
        render_state_handoff_mode = "persistent_compact_render_state_buffer"
    else:
        render_state_handoff_mode = "copy_actor_state_to_parent_buffers"
    return {
        "schema_id": "curvyzero_source_state_hybrid_actor_zero_observation_profile/v0",
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "compact_owned_loop_entrypoint_enabled": True,
        "compact_owned_loop_profile_only": True,
        "compact_owned_loop_deferred_learner_pending": False,
        "compact_owned_loop_deferred_learner_submit_count": 0,
        "compact_owned_loop_deferred_learner_completed_count": 0,
        "compact_owned_loop_deferred_learner_pending_count": 0,
        "compact_owned_loop_deferred_learner_max_pending": 0,
        "compact_owned_loop_deferred_learner_max_pending_observed": 0,
        "compact_owned_loop_deferred_learner_wait_count": 0,
        "compact_owned_loop_deferred_learner_wait_sec": 0.0,
        "compact_owned_loop_deferred_learner_last_wait_sec": 0.0,
        "compact_owned_loop_deferred_sample_learner_pending": False,
        "compact_owned_loop_deferred_sample_learner_submit_count": 0,
        "compact_owned_loop_deferred_sample_learner_completed_count": 0,
        "compact_owned_loop_deferred_sample_learner_pending_count": 0,
        "compact_owned_loop_deferred_sample_learner_max_pending": 0,
        "compact_owned_loop_deferred_sample_learner_max_pending_observed": 0,
        "compact_owned_loop_deferred_sample_learner_wait_count": 0,
        "compact_owned_loop_deferred_sample_learner_wait_sec": 0.0,
        "compact_owned_loop_deferred_sample_learner_last_wait_sec": 0.0,
        "compact_owned_loop_deferred_sample_learner_drained": False,
        "compact_owned_loop_sample_learner_worker_start_method": "none",
        "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings": "none",
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid": 0,
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id": "none",
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device": "none",
        (
            "compact_owned_loop_deferred_sample_learner_"
            "last_completed_worker_pid_distinct_from_actor_search"
        ): False,
        "compact_owned_loop_deferred_sample_learner_model_state_apply_count": 0,
        "compact_owned_loop_deferred_sample_learner_last_model_state_applied": False,
        "compact_owned_loop_final_deferred_drain_sec": 0.0,
        "compact_owned_loop_final_deferred_sample_learner_drain_sec": 0.0,
        "compact_owned_loop_final_deferred_learner_drain_sec": 0.0,
        "compact_owned_loop_final_deferred_drain_in_measured_sec": False,
        "compact_rollout_slab_learner_gate_impl": "compact_muzero",
        "compact_rollout_slab_learner_gate_num_unroll_steps": 1,
        "compact_rollout_slab_learner_gate_real_muzero_update": True,
        "compact_rollout_slab_learner_gate_calls": 1,
        "compact_rollout_slab_learner_gate_updates": 1,
        "compact_rollout_slab_learner_gate_sample_row_count": 2,
        "compact_rollout_slab_sample_gate_calls": 1,
        "compact_rollout_slab_sample_gate_sample_row_count": 2,
        "compact_rollout_slab_policy_refresh_after_learner_gate_enabled": True,
        "compact_rollout_slab_policy_refresh_after_learner_gate_calls": 1,
        "compact_rollout_slab_policy_refresh_after_learner_gate_interval": 1,
        "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count": 0,
        "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count": 0,
        "compact_rollout_slab_policy_refresh_after_learner_gate_sec": 0.01,
        "compact_rollout_slab_policy_refresh_after_learner_gate_service_total_sec": 0.008,
        "compact_rollout_slab_policy_refresh_after_learner_gate_state_load_sec": 0.003,
        "compact_rollout_slab_policy_refresh_after_learner_gate_model_digest_sec": 0.005,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_service_total_sec": 0.008,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_state_load_sec": 0.003,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_digest_sec": 0.005,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": 1,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest": (
            "c" * 64
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_search_worker_distinct_from_learner": True,
        "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_count": 1,
        "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_count": 1,
        "compact_rollout_slab_policy_refresh_after_learner_gate_sample_metadata_count": 0,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_worker_state": {
            "schema_id": "curvyzero_compact_policy_refresh_search_worker_state/v1",
            "learner_update_count": 1,
            "refresh_count": 1,
            "model_state_digest": "c" * 64,
            "search_worker_model_object_id": 123,
        },
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata": {
            "compact_policy_refresh_search_worker_refreshed": True,
            "compact_policy_refresh_learner_update_count": 1,
            "compact_policy_refresh_model_state_digest": "c" * 64,
        },
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata": {
            "compact_policy_refresh_search_worker_refreshed": True,
            "compact_policy_refresh_learner_update_count": 1,
            "compact_policy_refresh_model_state_digest": "c" * 64,
        },
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_metadata": {},
        "contract": {
            "borrow_single_actor_render_state": borrow_render_state,
            "persistent_compact_render_state_buffer": persistent_render_state,
            "render_state_handoff_mode": render_state_handoff_mode,
            "render_state_copy_steps": 0 if persistent_render_state or borrow_render_state else 4,
            "render_state_borrowed_steps": 5 if borrow_render_state else 0,
            "render_state_row_overlay_steps": 0,
            "render_state_row_overlay_rows": 0,
            "render_state_row_overlay_bytes": 0,
        },
        "steps": 4,
        "warmup_steps": 1,
        "batch_size": 2,
        "death_mode": "profile_no_death",
        "terminal_row_count": 0,
        "timings": {"resident_observation_host_fallback_count": 0.0},
        "compact_rollout_slab_telemetry_totals": {
            "compact_rollout_slab_replay_payload_d2h_bytes": 0.0,
            "compact_rollout_slab_committed_replay_payload_d2h_bytes": 0.0,
        },
        "measured_sec": 0.5,
        "steps_per_sec": 16.0,
        "physical_rows_per_sec": 16.0,
    }


def _valid_local_process_sample_learner_payload() -> dict[str, Any]:
    payload = _profile_payload()
    payload.update(
        {
            "compact_owned_loop_defer_sample_learner_gate": True,
            "compact_owned_loop_sample_learner_worker_kind": "local_process",
            "compact_owned_loop_sample_learner_worker_resource_scope": "process",
            "compact_owned_loop_sample_learner_worker_start_method": "spawn",
            "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings": (
                "expandable_segments:False"
            ),
            "compact_owned_loop_sample_learner_worker_bootstrap_source": "factory",
            "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": True,
            "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search": (
                False
            ),
            "compact_owned_loop_deferred_sample_learner_submit_count": 2,
            "compact_owned_loop_deferred_sample_learner_completed_count": 2,
            "compact_owned_loop_deferred_sample_learner_pending": False,
            "compact_owned_loop_deferred_sample_learner_pending_count": 0,
            "compact_owned_loop_deferred_sample_learner_max_pending": 2,
            "compact_owned_loop_deferred_sample_learner_max_pending_observed": 2,
            "compact_owned_loop_deferred_sample_learner_drained": True,
            "compact_owned_loop_final_deferred_drain_in_measured_sec": True,
            "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending": 1,
            "compact_owned_loop_deferred_sample_learner_policy_lag_max": 2,
            "compact_owned_loop_deferred_sample_learner_last_submitted_request_id": 2,
            "compact_owned_loop_deferred_sample_learner_last_completed_request_id": 2,
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid": 1234,
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id": (
                "local_process_pool:unit"
            ),
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device": (
                "cuda:0"
            ),
            (
                "compact_owned_loop_deferred_sample_learner_"
                "last_completed_worker_pid_distinct_from_actor_search"
            ): True,
            "compact_owned_loop_deferred_sample_learner_model_state_apply_count": 2,
            "compact_owned_loop_deferred_sample_learner_model_state_return_count": 2,
            "compact_owned_loop_deferred_sample_learner_model_state_omitted_count": 0,
            "compact_owned_loop_deferred_sample_learner_last_model_state_applied": True,
            "compact_owned_loop_deferred_sample_learner_request_host_only": True,
            "compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count": 0,
            "compact_owned_loop_deferred_sample_learner_result_host_only": True,
            "compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count": 0,
            "compact_owned_loop_deferred_sample_learner_request_bytes": 100,
            "compact_owned_loop_deferred_sample_learner_result_bytes": 100,
            "compact_owned_loop_deferred_sample_learner_worker_owns_model_state": True,
            "compact_owned_loop_deferred_sample_learner_worker_owns_replay_store": True,
            "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent": False,
            ("compact_owned_loop_deferred_sample_learner_full_replay_snapshot_submit_count"): 0,
            "compact_owned_loop_deferred_sample_learner_replay_append_entry_count": 2,
            "compact_owned_loop_deferred_sample_learner_replay_append_index_row_count": 4,
            "compact_owned_loop_deferred_sample_learner_replay_append_entry_bytes": 100,
            ("compact_owned_loop_deferred_sample_learner_replay_append_host_observation_bytes"): 0,
            ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_count"): 0,
            "compact_owned_loop_deferred_sample_learner_worker_replay_append_count": 2,
            "compact_owned_loop_deferred_sample_learner_worker_replay_index_row_count": 4,
            ("compact_owned_loop_deferred_sample_learner_worker_model_initialized_count"): 1,
            "compact_owned_loop_deferred_sample_learner_worker_completed_count": 2,
            "compact_rollout_slab_sample_gate_calls": 2,
            "compact_rollout_slab_learner_gate_calls": 2,
            "compact_rollout_slab_learner_gate_updates": 2,
        }
    )
    return payload


def _owner_search_slab_proxy_fields() -> dict[str, Any]:
    return {
        "compact_owned_loop_entrypoint_enabled": False,
        "compact_rollout_slab_learner_gate_real_muzero_update": False,
        "compact_rollout_slab_learner_gate_calls": 0,
        "compact_rollout_slab_learner_gate_updates": 0,
        "compact_rollout_slab_learner_gate_sample_row_count": 0,
        "compact_rollout_slab_sample_gate_calls": 0,
        "compact_rollout_slab_sample_gate_sample_row_count": 0,
        "compact_rollout_slab_policy_refresh_after_learner_gate_enabled": False,
        "compact_rollout_slab_policy_refresh_after_learner_gate_calls": 0,
        "compact_owner_search_slab_proxy": True,
        "compact_owner_search_lazy_slab_proxy": True,
        "compact_owner_search_inline_slab_proxy": False,
        "compact_owner_search_inline_background_slab_proxy": False,
        "compact_owner_search_threaded_slab_proxy": False,
        "compact_owner_search_slab_bypass": False,
        "compact_owner_search_slab_bypass_kind": "none",
        "compact_rollout_slab_bypassed": False,
        "compact_rollout_slab_general_replay_row_builder_used": True,
        "compact_owner_search_slab_bypass_parent_committed_index_rows": 0,
        "compact_owner_search_slab_bypass_parent_stored_index_rows": 0,
        "compact_owner_search_slab_proxy_initialized": True,
        "compact_owner_search_boundary_kind": "worker_search_parent_slab_commit",
        "compact_owner_search_parent_slab_commits_replay": True,
        "compact_owner_search_worker_kind": "local_process_owner_search_v1",
        "compact_owner_search_worker_resource_scope": "persistent_process",
        "compact_owner_search_worker_resource_distinct_from_actor": True,
        "compact_owner_search_worker_hardware_resource_distinct_from_actor": False,
        "compact_owner_search_owner_pid": 1234,
        "compact_owner_search_root_slot_count": 2,
        "compact_owner_search_active_root_count": 2,
        "compact_owner_search_request_bytes": 55,
        "compact_owner_search_result_bytes": 88,
        "compact_owner_search_fixed_action_result_buffer_requested": False,
        "compact_owner_search_fixed_action_result_buffer_used": False,
        "compact_owner_search_fixed_action_result_buffer_slot_count": 0,
        "compact_owner_search_fixed_action_result_buffer_acquire_count": 0,
        "compact_owner_search_fixed_action_result_buffer_write_count": 0,
        "compact_owner_search_fixed_action_result_buffer_read_count": 0,
        "compact_owner_search_fixed_action_result_buffer_slot_id": -1,
        "compact_owner_search_fixed_action_result_buffer_last_slot_id": -1,
        "compact_owner_search_fixed_action_result_buffer_wire_result_bytes": 0,
        "compact_owner_search_fixed_action_result_buffer_full_result_bytes": 0,
        "compact_owner_search_fixed_action_result_buffer_pending_slot_count": 0,
        "compact_owner_search_request_cuda_tensor_count": 0,
        "compact_owner_search_result_cuda_tensor_count": 0,
        "compact_owner_search_root_observation_bytes_sent": 0,
        "compact_owner_search_parent_reconstructed_search_result": True,
        "compact_owner_search_model_state_bytes": 0,
        "compact_owner_search_model_state_return_count": 0,
        "compact_owner_search_model_state_snapshot_return_count": 0,
        "compact_owner_search_action_feedback_verified": True,
        "compact_owner_search_action_feedback_transition_count": 2,
        "compact_owner_search_action_feedback_action_count": 2,
        "compact_owner_search_action_feedback_mismatch_count": 0,
        "compact_owner_search_expected_joint_action_checksum": 11,
        "compact_owner_search_applied_joint_action_checksum": 11,
        "compact_owner_search_replay_action_checksum": 11,
        "compact_owner_search_search_result_payload_bytes": 77,
        "compact_owner_search_search_result_payload_transport_kind": "numpy_ndarray_ipc_v1",
        "compact_owner_search_search_result_payload_json_safe": False,
        "compact_owner_search_selected_action_bytes": 8,
        "compact_owner_search_visit_policy_bytes": 24,
        "compact_owner_search_root_value_bytes": 8,
        "compact_owner_search_optional_array_bytes": 24,
        "compact_owner_search_inner_two_phase_action_step": False,
        "compact_owner_search_inner_device_replay_payload_deferred": False,
        "compact_owner_search_use_inner_two_phase_device_replay": False,
        "compact_owner_search_replay_payload_handle_present": True,
        "compact_owner_search_worker_owns_search_state": True,
        "compact_owner_search_worker_owns_replay_state": True,
        "compact_owner_search_worker_owns_model_state": True,
        "compact_owner_search_consumed_learner_update": True,
        "compact_owner_search_search_refresh_update_count": 1,
        "compact_owner_search_replay_append_entry_count": 2,
        "compact_owner_search_replay_append_transport_entry_count": 2,
        "compact_owner_search_replay_append_transition_batch_count": 0,
        "compact_owner_search_replay_append_transition_batch_entry_count": 0,
        "compact_owner_search_replay_append_count": 2,
        "compact_owner_search_learner_update_count": 1,
        "compact_owner_search_model_owner_ref_returned": True,
        "compact_owner_search_model_owner_ref_digest": "owner-digest",
        "compact_owner_search_owner_replay_append_enabled": True,
        "compact_owner_search_owner_sample_batch_size": 2,
        "compact_owner_search_owner_train_steps": 1,
        "compact_owner_search_owner_train_interval": 2,
        "compact_owner_search_owner_model_refresh_interval": 1,
        "compact_owner_search_owner_expected_train_request_count": 1,
        "compact_owner_search_owner_defer_maintenance": False,
        "compact_owner_search_owner_loop_schema_id": (
            "curvyzero_compact_owner_search_priority_loop/v1"
        ),
        "compact_owner_search_owner_loop_kind": "persistent_priority_owner_loop_v1",
        "compact_owner_search_owner_loop_persistent": True,
        "compact_owner_search_owner_action_priority_enabled": True,
        "compact_owner_search_owner_background_maintenance_thread": False,
        "compact_owner_search_owner_background_overlap_enabled": False,
        "compact_owner_search_owner_action_request_count": 2,
        "compact_owner_search_owner_maintenance_request_count": 0,
        "compact_owner_search_owner_run_request_count": 0,
        "compact_owner_search_owner_sample_telemetry": {
            "compact_rollout_slab_sample_gate_sample_row_count": 2,
            "compact_rollout_slab_sample_gate_target_row_count": 2,
            "compact_rollout_slab_sample_gate_requested_sample_row_count": 2,
            "compact_rollout_slab_sample_gate_require_next_targets": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
        },
        "compact_owner_search_owner_learner_telemetry": {
            "compact_owner_search_owner_train_wall_sec": 0.011,
            "compact_owner_search_owner_train_sample_sec": 0.001,
            "compact_owner_search_owner_train_learner_update_sec": 0.002,
            "compact_owner_search_owner_train_model_state_digest_sec": 0.003,
            "compact_owner_search_owner_train_model_state_dict_sec": 0.004,
            "compact_owner_search_owner_train_owner_ref_build_sec": 0.0005,
            "compact_owner_search_owner_train_accounted_sec": 0.0105,
            "compact_owner_search_owner_train_residual_sec": 0.0005,
            "compact_owner_search_owner_train_timing_aggregate_count": 1,
        },
        "compact_owner_search_owner_replay_append_staged_entry_count": 2,
        "compact_owner_search_owner_replay_append_staged_transport_entry_count": 2,
        "compact_owner_search_owner_replay_append_suppressed_transport_entry_count": 0,
        "compact_owner_search_owner_replay_append_submitted_entry_count": 2,
        "compact_owner_search_owner_replay_append_submitted_transport_entry_count": 2,
        "compact_owner_search_owner_replay_transport_entry_count": 2,
        "compact_owner_search_owner_replay_transport_kind": "per_transition_entry_v1",
        "compact_owner_search_owner_replay_transition_batch_enabled": False,
        "compact_owner_search_owner_replay_transition_batch_count": 0,
        "compact_owner_search_owner_replay_transition_batch_transition_count": 0,
        "compact_owner_search_owner_replay_transition_legacy_entry_count": 0,
        "compact_owner_search_transition_batch_transport_requested": False,
        "compact_owner_search_transition_batch_transport_enabled": False,
        "compact_owner_search_transition_batch_transport_kind": "per_transition_entry_v1",
        "compact_owner_search_transition_batch_schema_id": "",
        "compact_owner_search_transition_batch_count": 0,
        "compact_owner_search_transition_batch_entry_count": 0,
        "compact_owner_search_transition_batch_transport_entry_count": 2,
        "compact_owner_search_transition_batch_max_entries_per_batch": 1,
        "compact_owner_search_transition_batch_fixed_capacity": 0,
        "compact_owner_search_transition_batch_padding_count": 0,
        "compact_owner_search_transition_batch_overflow_count": 0,
        "compact_owner_search_transition_batch_fallback_count": 0,
        "compact_owner_search_transition_batch_fallback_reason": "none",
        "compact_owner_search_transition_batch_pending_count": 0,
        "compact_owner_search_transition_batch_transport_bytes": 0,
        "compact_owner_search_transition_batch_digest": "",
        "compact_owner_search_transition_batch_digest_verified": True,
        "compact_owner_search_transition_batch_build_sec": 0.0,
        "compact_owner_search_transition_batch_submit_sec": 0.0,
        "compact_owner_search_owner_replay_append_request_count": 2,
        "compact_owner_search_owner_replay_append_count": 2,
        "compact_owner_search_owner_train_request_count": 1,
        "compact_owner_search_owner_model_refresh_request_count": 1,
        "compact_owner_search_owner_model_refresh_skipped_count": 0,
        "compact_owner_search_owner_submitted_learner_update_count": 1,
        "compact_owner_search_owner_learner_update_count": 1,
        "compact_owner_search_owner_pending_replay_append_entry_count": 0,
        "compact_owner_search_owner_maintenance_drain_request_count": 0,
        "compact_owner_search_owner_maintenance_staged_work_item_count": 0,
        "compact_owner_search_owner_maintenance_drained_count": 0,
        "compact_owner_search_owner_maintenance_drained_work_item_count": 0,
        "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": 0,
        "compact_owner_search_owner_maintenance_drained_replay_append_transport_entry_count": 0,
        "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_count": 0,
        "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_entry_count": 0,
        "compact_owner_search_owner_maintenance_drained_replay_append_count": 0,
        "compact_owner_search_owner_maintenance_pending_work_count": 0,
        "compact_owner_search_owner_maintenance_inflight": False,
        "compact_owner_search_owner_maintenance_final_drain_sec": 0.0,
        "compact_owner_search_owner_maintenance_final_drain_in_measured_sec": False,
        "compact_owner_search_owner_maintenance_coalescing_kind": "",
        "compact_owner_search_owner_maintenance_coalesced_skip_count": 0,
        "compact_owner_search_owner_maintenance_eager_append_drain_count": 0,
        "compact_owner_search_owner_async_learner_worker_enabled": False,
        "compact_owner_search_owner_async_learner_worker_kind": "none",
        "compact_owner_search_owner_async_learner_worker_resource_scope": "",
        "compact_owner_search_owner_async_learner_worker_resource_id": "",
        "compact_owner_search_owner_async_learner_actor_resource_id": "",
        "compact_owner_search_owner_async_learner_worker_parent_pid": 0,
        "compact_owner_search_owner_async_learner_resource_distinct_from_owner": False,
        ("compact_owner_search_owner_async_learner_hardware_resource_distinct_from_owner"): False,
        "compact_owner_search_owner_async_learner_max_pending": 0,
        "compact_owner_search_owner_async_learner_submit_count": 0,
        "compact_owner_search_owner_async_learner_completed_count": 0,
        "compact_owner_search_owner_async_learner_pending_count": 0,
        "compact_owner_search_owner_async_learner_max_pending_observed": 0,
        "compact_owner_search_owner_async_learner_wait_count": 0,
        "compact_owner_search_owner_async_learner_wait_sec": 0.0,
        "compact_owner_search_owner_action_while_async_learner_pending_count": 0,
        "compact_owner_search_owner_async_learner_failed": False,
        "compact_owner_search_owner_async_learner_request_host_only": False,
        "compact_owner_search_owner_async_learner_request_cuda_tensor_count": -1,
        "compact_owner_search_owner_async_learner_result_host_only": False,
        "compact_owner_search_owner_async_learner_result_cuda_tensor_count": -1,
        "compact_owner_search_owner_async_learner_request_bytes": 0,
        "compact_owner_search_owner_async_learner_result_bytes": 0,
        "compact_owner_search_owner_async_learner_worker_pid": 0,
        "compact_owner_search_owner_async_learner_worker_job_wall_sec": 0.0,
        "compact_owner_search_owner_async_learner_payload_prepare_sec": 0.0,
        ("compact_owner_search_owner_async_learner_worker_pid_distinct_from_owner"): False,
        "compact_owner_search_owner_async_learner_worker_owns_model_state": False,
        "compact_owner_search_owner_policy_lag_current": 0,
        "compact_owner_search_owner_policy_lag_max": 0,
        "compact_owner_search_owner_maintenance_actor_steps_while_pending": 0,
        "compact_owner_search_owner_maintenance_actor_steps_while_policy_lagged": 0,
        "compact_owner_search_owner_action_while_maintenance_pending_count": 0,
        "compact_owner_search_owner_action_while_policy_lagged_count": 0,
        "compact_owner_search_owner_action_served_before_maintenance_count": 0,
        "compact_owner_search_owner_fifo_blocked_action_count": 0,
        "compact_owner_search_owner_maintenance_failed": False,
        "compact_owner_search_parent_publish_sec": 0.001,
        "compact_owner_search_parent_submit_sec": 0.002,
        "compact_owner_search_parent_wait_sec": 0.01,
        "compact_owner_search_parent_wall_sec": 0.012,
        "compact_owner_search_worker_wall_sec": 0.02,
        "compact_owner_search_worker_root_resolve_sec": 0.003,
        "compact_owner_search_worker_search_sec": 0.015,
        "compact_owner_search_worker_replay_append_sec": 0.004,
        "compact_owner_search_worker_learner_train_sec": 0.011,
        "compact_owner_search_owner_train_wall_sec": 0.011,
        "compact_owner_search_owner_train_sample_sec": 0.001,
        "compact_owner_search_owner_train_learner_update_sec": 0.002,
        "compact_owner_search_owner_train_model_state_digest_sec": 0.003,
        "compact_owner_search_owner_train_model_state_dict_sec": 0.004,
        "compact_owner_search_owner_train_owner_ref_build_sec": 0.0005,
        "compact_owner_search_owner_train_accounted_sec": 0.0105,
        "compact_owner_search_owner_train_residual_sec": 0.0005,
        "compact_owner_search_owner_train_timing_aggregate_count": 1,
        "compact_owner_search_worker_search_refresh_sec": 0.006,
        "compact_owner_search_resident_root_bridge_ready": True,
        "compact_owner_search_resident_root_bridge_kind": (
            "shared_memory_host_root_to_owner_resident_tensor_v1"
        ),
        "compact_owner_search_resident_root_bridge_device": "cpu",
        "compact_owner_search_resident_root_bridge_h2d_bytes": 1024.0,
        "compact_owner_search_resident_root_bridge_generation_id": 7,
        "compact_owner_search_resident_root_bridge_final_storage": "sparse_rows",
        "compact_owner_search_resident_root_bridge_final_sparse_row_count": 1,
        "compact_owner_search_resident_root_bridge_final_sparse_bytes": 256,
        "compact_owner_search_resident_root_bridge_final_dense_clone_avoided_bytes": 768,
        "compact_rollout_slab_committed_index_row_count": 8,
        "compact_rollout_slab_stored_index_row_count": 8,
    }


def _owner_action_step_boundary_fields(total_iterations: int = 5) -> dict[str, Any]:
    return {
        "compact_owner_action_step_boundary_enabled": True,
        "compact_owner_action_step_boundary_proof_passed": True,
        "compact_owner_action_step_boundary_step_count": total_iterations,
        "compact_owner_action_step_boundary_seeded_action_count": 1,
        "compact_owner_action_step_boundary_feedback_action_count": total_iterations - 1,
        "compact_owner_action_step_boundary_action_verified_count": total_iterations,
        "compact_owner_action_step_boundary_next_action_count": total_iterations,
        "compact_owner_action_step_boundary_last_action_source": "search_feedback",
        "compact_owner_action_step_boundary_last_applied_action_checksum": 101,
        "compact_owner_action_step_boundary_last_next_action_checksum": 202,
        "compact_owner_action_step_boundary_failure_reason": "none",
    }


def _owner_action_dispatch_step_overlap_fields(total_iterations: int = 5) -> dict[str, Any]:
    return {
        "compact_owner_action_dispatch_step_overlap_enabled": True,
        "compact_owner_action_dispatch_step_overlap_proof_passed": True,
        "compact_rollout_slab_action_dispatch_step_overlap_supported": True,
        "compact_rollout_slab_action_dispatch_step_overlap_used": True,
        "compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait": True,
        "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper": False,
        "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count": 0,
        "compact_rollout_slab_action_dispatch_step_overlap_submit_count": total_iterations,
        "compact_rollout_slab_action_dispatch_step_overlap_resolve_count": total_iterations,
        "compact_rollout_slab_action_dispatch_step_overlap_pending_count": 0,
        "compact_rollout_slab_action_dispatch_step_overlap_max_pending_count": 1,
        "compact_rollout_slab_action_dispatch_step_overlap_submit_wall_sec": 0.001,
        "compact_rollout_slab_action_dispatch_step_overlap_resolve_wall_sec": 0.002,
        "compact_rollout_slab_action_dispatch_step_overlap_submit_to_resolve_elapsed_sec": 0.004,
        "compact_rollout_slab_action_dispatch_step_overlap_parent_work_sec": 0.003,
        "compact_owner_search_action_dispatch_handle_boundary_supported": True,
        "compact_owner_search_action_dispatch_handle_used": True,
        "compact_owner_search_action_dispatch_handle_schema_id": (
            "curvyzero_compact_owner_action_dispatch_handle/v1"
        ),
        "compact_owner_search_action_dispatch_handle_id": 5,
        "compact_owner_search_action_dispatch_handle_submit_no_wait": True,
        "compact_owner_search_action_dispatch_handle_sync_wrapper": False,
        "compact_owner_search_action_dispatch_handle_sync_wrapper_count": 0,
        "compact_owner_search_action_dispatch_handle_completed_at_submit_count": 0,
        "compact_owner_search_action_dispatch_handle_submit_count": total_iterations,
        "compact_owner_search_action_dispatch_handle_resolve_count": total_iterations,
        "compact_owner_search_action_dispatch_handle_pending_count": 0,
        "compact_owner_search_action_dispatch_handle_max_pending_count": 1,
        "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count": 0,
        "compact_owner_search_action_dispatch_handle_result_wait_sec": 0.002,
    }


def _lean_oracle_payload(owner: str) -> dict[str, Any]:
    return {
        "schema_id": "curvyzero_source_state_hybrid_actor_zero_observation_profile/v0",
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "compact_owned_training_loop_owner": owner,
        "death_mode": "profile_no_death",
        "env_action_checksum_total": 11,
        "env_done_checksum_total": 0,
        "env_reward_checksum_total": 500000,
        "env_action_mask_checksum_total": 33,
        "env_trajectory_checksum_total": 654321,
        "env_trajectory_ordered_checksum_total": 777777,
        "env_terminal_row_checksum_total": 44,
        "env_autoreset_row_checksum_total": 55,
        "env_terminal_reason_checksum_total": 66,
        "env_death_count_checksum_total": 77,
        "env_death_cause_checksum_total": 88,
        "env_death_hit_owner_checksum_total": 99,
        "last_env_action_checksum": 7,
        "last_env_trajectory_checksum": 123456,
        "last_env_terminal_row_checksum": 4,
        "last_env_autoreset_row_checksum": 5,
        "done_rows": 0,
        "terminal_row_count": 0,
        "death_row_count": 0,
        "terminated_row_count": 0,
        "truncated_row_count": 0,
        "terminal_final_observation_row_count": 0,
        "terminal_final_observation_before_autoreset_verified": False,
        "terminal_final_reward_map_verified": False,
        "autoreset_row_count": 0,
        "done_semantics_verified": True,
        "death_count_total": 0,
        "normal_collision_death_causes": [],
        "normal_collision_death_hit_owner_present": False,
        "normal_collision_death_evidence_rows": [],
        "compact_rollout_slab_committed_index_row_count": 8,
        "compact_rollout_slab_sample_gate_calls": 1,
        "compact_rollout_slab_sample_gate_sample_row_count": 2,
        "compact_rollout_slab_sample_gate_target_row_count": 2,
        "compact_rollout_slab_learner_gate_calls": 1,
        "compact_rollout_slab_learner_gate_updates": 1,
        "compact_rollout_slab_learner_gate_sample_row_count": 2,
        "compact_rollout_slab_policy_refresh_after_learner_gate_calls": 1,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": 1,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest": (
            "c" * 64
        ),
        "timings": {"resident_observation_host_fallback_count": 0.0},
    }


def _normal_death_profile_payload() -> dict[str, Any]:
    payload = _profile_payload(borrow_render_state=True)
    payload["death_mode"] = "normal"
    payload["compact_rollout_slab_learner_gate_num_unroll_steps"] = 2
    payload["terminal_row_count"] = 3
    payload["death_row_count"] = 3
    payload["terminated_row_count"] = 3
    payload["truncated_row_count"] = 0
    payload["death_count_total"] = 3
    payload["done_semantics_verified"] = True
    payload["normal_collision_death_causes"] = ["opponent_trail"]
    payload["normal_collision_death_hit_owner_present"] = True
    payload["normal_collision_death_evidence_rows"] = [
        {
            "death_cause": ["opponent_trail"],
            "death_count": 1,
            "death_hit_owner": [1, -1],
            "death_player": [0, -1],
            "done": True,
            "draw": False,
            "final_observation_row": True,
            "final_reward_map": [-1.0, 1.0],
            "final_reward_map_matches_reward": True,
            "global_row": 1,
            "reward": [-1.0, 1.0],
            "terminal_reason": 1,
            "terminated": True,
            "truncated": False,
            "winner": 1,
        }
    ]
    payload["terminal_final_observation_row_count"] = 3
    payload["terminal_final_observation_before_autoreset_verified"] = True
    payload["terminal_final_reward_map_verified"] = True
    payload["contract"]["render_state_copy_steps"] = 0
    payload["contract"]["render_state_row_overlay_steps"] = 1
    payload["contract"]["render_state_row_overlay_rows"] = 3
    payload["contract"]["render_state_row_overlay_bytes"] = 1024
    payload["compact_rollout_slab_sample_gate_last_telemetry"] = {
        "compact_rollout_slab_sample_gate_terminal_sample_row_count": 1,
        "compact_rollout_slab_sample_gate_next_final_observation_row_count": 1,
        "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used": True,
        "compact_rollout_slab_sample_gate_device_replay_index_rows": True,
        "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
        "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count": 1,
        "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode": (
            "stock_terminal_no_bootstrap_return_discount_1.0"
        ),
    }
    payload["compact_rollout_slab_learner_gate_last_telemetry"] = {
        "compact_rollout_slab_learner_gate_compact_muzero_telemetry": {
            "compact_muzero_learner_done_count": 1,
            "compact_muzero_learner_truncated_count": 0,
            "compact_muzero_learner_value_valid_count": 2,
        }
    }
    return payload


def _summary_args() -> SimpleNamespace:
    return SimpleNamespace(
        run_id="unit-speed-row",
        search_service_kind="compact_torch_search_service",
        hybrid_persistent_compact_render_state_buffer=False,
        hybrid_borrow_single_actor_render_state=False,
        compact_owned_loop_deferred_learner=False,
        compact_owned_loop_deferred_sample_learner=False,
        compact_owned_loop_deferred_sample_learner_max_pending=1,
        compact_owned_loop_sample_learner_worker_kind="in_process_thread",
        compact_owned_loop_fused_learner_batch=False,
        compact_muzero_learner_batch_unroll2_specialized_builder=False,
        compact_muzero_learner_batch_learner_ready_unroll2_cache=False,
        compact_muzero_learner_batch_tensor_native_replay=False,
        learner_num_unroll_steps=1,
        compact_torch_initial_inference_mode="direct_core",
        compact_torch_observation_memory_format="contiguous",
        compact_torch_model_memory_format="contiguous",
        compact_profile_bounded_diagnostics=False,
        compact_profile_cuda_sync_timing_diagnostics=False,
        compact_profile_runtime_step_timing_diagnostics=False,
        compact_owner_action_step_boundary=False,
        owner_search_fixed_action_result_buffer=False,
        owner_search_action_result_slot_capacity=4,
        actor_count=1,
        batch_size=2,
        steps=4,
        warmup_steps=1,
        death_mode="profile_no_death",
    )


def _loaded_identity() -> dict[str, Any]:
    return {
        "scope": "candidate_loaded_checkpoint",
        "identity_source": "unit_loaded_checkpoint",
        "candidate_loaded_checkpoint": True,
        "checkpoint_id": "unit-compact-ckpt",
        "trainer_id": "unit-compact-ckpt:trainer",
        "policy_version_ref": "unit-compact-ckpt:policy-update-1",
        "model_version_ref": "unit-compact-ckpt:model-update-1",
        "policy_source": "unit_policy_source",
        "learner_update_count": 1,
        "model_state_digest": "a" * 64,
        "compact_checkpoint_path": "/tmp/unit-compact-checkpoint.pt",
        "compact_checkpoint_sha256": "b" * 64,
        "support_scale": 300,
    }
