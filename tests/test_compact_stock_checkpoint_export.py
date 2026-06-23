from __future__ import annotations

import json

import pytest

from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod
from curvyzero.tournament import curvytron_checkpoint_tournament as arena
from curvyzero.training import lightzero_checkpoint_opponent_provider as provider
from curvyzero.training import compact_stock_checkpoint_export as export_mod
from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import compact_owned_loop_replay_store_metadata
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_EXPORT_EVIDENCE_BUNDLE_SCHEMA_ID,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    build_compact_stock_export_evidence_bundle_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    build_compact_stock_export_payload_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    build_current_policy_observation_sidecar_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    compact_stock_export_evidence_bundle_path,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    save_compact_stock_export_evidence_bundle_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    save_compact_stock_export_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    validate_compact_stock_export_evidence_bundle_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    validate_policy_observation_sidecar_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    verify_compact_stock_export_model_contract_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    CompactTrainerResumeStateV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    build_compact_trainer_checkpoint_v1,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _CompactReplayRingV1,
)


def _torch_module():
    return pytest.importorskip("torch")


def test_compact_stock_export_payload_exposes_model_state_under_stock_key():
    checkpoint = _compact_checkpoint()
    sidecar = build_current_policy_observation_sidecar_v1()

    payload = build_compact_stock_export_payload_v1(
        checkpoint,
        policy_metadata=sidecar,
    )

    state_key, state_dict = provider._state_dict_from_payload(
        payload,
        state_key=None,
    )
    assert state_key == COMPACT_STOCK_EXPORT_MODEL_STATE_KEY
    assert state_dict == payload[COMPACT_STOCK_EXPORT_MODEL_STATE_KEY]
    assert payload["metadata"]["stock_payload_mapping"] is True
    assert payload["metadata"]["stock_state_dict_discovery_verified"] is True
    assert payload["metadata"]["stock_eval_tournament_loadable"] is False
    assert payload["metadata"]["stock_eval_tournament_load_status"] == (
        "strict_stock_model_load_not_run"
    )


def test_compact_stock_export_writes_required_policy_metadata_sidecar(tmp_path):
    checkpoint = _compact_checkpoint()
    sidecar = build_current_policy_observation_sidecar_v1()
    path = tmp_path / "iteration_12.pth.tar"

    result = save_compact_stock_export_v1(
        checkpoint,
        path,
        policy_metadata=sidecar,
    )

    assert result["checkpoint_path"] == path
    assert result["sidecar_path"] == provider.checkpoint_policy_metadata_sidecar_path(
        path
    )
    sidecar_payload = json.loads(result["sidecar_path"].read_text(encoding="utf-8"))
    assert sidecar_payload["schema_id"] == "curvyzero_checkpoint_policy_metadata/v0"
    assert sidecar_payload["model_env_variant"] == "source_state_fixed_opponent"
    assert sidecar_payload["model_reward_variant"] == (
        "survival_plus_bonus_no_outcome"
    )
    assert sidecar_payload["observation_contract"]["single_frame_shape"] == [1, 64, 64]
    loaded_payload = _torch_module().load(path, map_location="cpu", weights_only=False)
    eval_found = eval_mod._find_state_dict(loaded_payload)
    assert eval_found is not None
    assert eval_found[0] == COMPACT_STOCK_EXPORT_MODEL_STATE_KEY
    accepted = provider.require_checkpoint_policy_observation_metadata(
        checkpoint_path=path,
        checkpoint_payload=result["payload"],
    )
    assert accepted["policy_observation_backend"] == "cpu_oracle"


def test_compact_stock_export_sidecar_preserves_model_contract_and_runtime_settings():
    sidecar = build_current_policy_observation_sidecar_v1(
        extra_metadata={"decision_source_frames": 1},
    )

    assert sidecar["model_env_variant"] == "source_state_fixed_opponent"
    assert sidecar["model_reward_variant"] == "survival_plus_bonus_no_outcome"
    assert sidecar["env_variant"] == "source_state_fixed_opponent"
    assert sidecar["reward_variant"] == "survival_plus_bonus_no_outcome"
    assert sidecar["decision_source_frames"] == 1
    assert sidecar["source_physics_step_ms"] == pytest.approx(1000.0 / 60.0)
    assert sidecar["decision_ms"] == pytest.approx(1000.0 / 60.0)
    assert sidecar["source_max_steps"] == 1048576
    assert sidecar["source_max_steps_semantics"] == "source_physics_steps"
    assert sidecar["learner_seat_mode"] == "random_per_episode"
    assert sidecar["observation_contract"]["contract_id"] == (
        "curvyzero_policy_observation_surface/v1"
    )
    assert sidecar["observation_contract"]["backend"] == "cpu_oracle"
    assert sidecar["observation_contract"]["stack_shape"] == [4, 64, 64]
    assert sidecar["observation_contract"]["single_frame_shape"] == [1, 64, 64]


def test_compact_stock_export_payload_preserves_tournament_loader_metadata():
    checkpoint = _compact_checkpoint()
    sidecar = build_current_policy_observation_sidecar_v1()

    payload = build_compact_stock_export_payload_v1(
        checkpoint,
        policy_metadata=sidecar,
    )

    assert arena._checkpoint_model_contract_from_payload(payload) == {
        "model_env_variant": "source_state_fixed_opponent",
        "model_reward_variant": "survival_plus_bonus_no_outcome",
    }
    runtime = arena._checkpoint_runtime_settings_from_payload(payload)
    assert runtime["decision_source_frames"] == 1
    assert runtime["source_physics_step_ms"] == pytest.approx(1000.0 / 60.0)
    assert runtime["decision_ms"] == pytest.approx(1000.0 / 60.0)
    assert runtime["source_max_steps"] == 1048576
    assert runtime["source_max_steps_semantics"] == "source_physics_steps"


def test_compact_stock_export_loads_through_tournament_policy_loader(
    monkeypatch,
    tmp_path,
):
    torch = _torch_module()
    checkpoint = _compact_checkpoint()
    ref = (
        "training/lightzero-curvytron-visual-survival/compact-export-run/"
        "attempts/attempt-a/train/lightzero_exp/ckpt/iteration_16.pth.tar"
    )
    path = tmp_path / ref
    sidecar = build_current_policy_observation_sidecar_v1(
        decision_source_frames=2,
        source_physics_step_ms=1000.0 / 120.0,
        source_max_steps=1234,
    )
    result = save_compact_stock_export_v1(
        checkpoint,
        path,
        policy_metadata=sidecar,
    )
    captured = {}

    class FakeEnv:
        def close(self) -> None:
            captured["closed"] = True

    def fake_make_policy_and_env(**kwargs):
        captured.update(kwargs)
        return object(), FakeEnv(), {"schema": "fake-loader-surface"}

    monkeypatch.setattr(eval_mod, "_make_policy_and_env", fake_make_policy_and_env)

    loaded = arena._load_policy_from_checkpoint(
        checkpoint_ref=ref,
        checkpoint_state_key=COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        seed=17,
        source_max_steps=16,
        num_simulations=3,
        batch_size=2,
        telemetry_path=tmp_path / "loader_telemetry.jsonl",
        mount=tmp_path,
        remote_root=None,
        model_env_variant=None,
        model_reward_variant=None,
    )

    assert captured["closed"] is True
    assert captured["seed"] == 17
    assert captured["num_simulations"] == 3
    assert captured["batch_size"] == 2
    assert captured["source_max_steps"] == 1234
    assert captured["decision_source_frames"] == 2
    assert captured["source_physics_step_ms"] == pytest.approx(1000.0 / 120.0)
    assert captured["decision_ms"] == pytest.approx(1000.0 / 60.0)
    assert captured["source_max_steps_semantics"] == "source_physics_steps"
    assert captured["model_env_variant"] == "source_state_fixed_opponent"
    assert captured["model_reward_variant"] == "survival_plus_bonus_no_outcome"
    assert set(captured["state_dict"]) == set(result["payload"]["model"])
    for key, expected in result["payload"]["model"].items():
        assert torch.equal(captured["state_dict"][key], expected)

    assert loaded["checkpoint_ref"] == ref
    assert loaded["checkpoint_resolution"]["source_kind"] == "volume_ref"
    assert loaded["checkpoint_state_key"] == COMPACT_STOCK_EXPORT_MODEL_STATE_KEY
    assert loaded["policy_observation_backend"] == "cpu_oracle"
    assert loaded["policy_observation_contract_id"] == (
        "curvyzero_policy_observation_surface/v1"
    )
    assert loaded["runtime_settings"]["source_max_steps"] == 1234
    assert loaded["runtime_settings"]["decision_source_frames"] == 2
    assert loaded["runtime_settings"]["source_physics_step_ms"] == pytest.approx(
        1000.0 / 120.0
    )
    assert loaded["runtime_settings"]["decision_ms"] == pytest.approx(1000.0 / 60.0)
    assert loaded["model_contract_source"]["payload"] == {
        "model_env_variant": "source_state_fixed_opponent",
        "model_reward_variant": "survival_plus_bonus_no_outcome",
    }


def test_compact_stock_export_rejected_without_policy_observation_metadata():
    sidecar = build_current_policy_observation_sidecar_v1()
    del sidecar["policy_observation_backend"]

    with pytest.raises(ValueError, match="policy_observation_backend"):
        validate_policy_observation_sidecar_v1(sidecar)


def test_compact_stock_export_rejected_without_runtime_metadata():
    sidecar = build_current_policy_observation_sidecar_v1()
    del sidecar["model_env_variant"]

    with pytest.raises(ValueError, match="model_env_variant"):
        validate_policy_observation_sidecar_v1(sidecar)


def test_compact_stock_export_does_not_claim_stock_resume_without_optimizer_state():
    checkpoint = _compact_checkpoint()
    sidecar = build_current_policy_observation_sidecar_v1()

    payload = build_compact_stock_export_payload_v1(
        checkpoint,
        policy_metadata=sidecar,
    )

    assert payload["metadata"]["stock_eval_tournament_loadable"] is False
    assert payload["metadata"]["stock_model_contract_verified"] is False
    assert payload["metadata"]["stock_model_contract_verification_required"] is True
    assert payload["metadata"]["stock_resume_claim"] is False
    assert payload["metadata"]["optimizer_resume_supported"] is False
    assert payload["metadata"]["stock_optimizer_state_in_payload"] is False
    assert "optimizer_state_dict" not in payload


def test_compact_stock_export_model_contract_verifier_reports_strict_load(
    monkeypatch,
    tmp_path,
):
    checkpoint = _compact_checkpoint()
    path = tmp_path / "iteration_13.pth.tar"
    save_compact_stock_export_v1(
        checkpoint,
        path,
        policy_metadata=build_current_policy_observation_sidecar_v1(),
    )
    calls = []

    def fake_load_lightzero_policy(**kwargs):
        calls.append(dict(kwargs))
        return (
            object(),
            "cpu",
            {
                "ok": True,
                "strict": True,
                "candidate": "as_is",
                "state_key": kwargs["state_key"],
                "checkpoint_path": str(kwargs["checkpoint_path"]),
                "checkpoint_inferred_model_support_config": {"support_scale": 1},
            },
        )

    monkeypatch.setattr(
        export_mod.provider,
        "load_lightzero_curvytron_visual_survival_policy",
        fake_load_lightzero_policy,
    )

    report = verify_compact_stock_export_model_contract_v1(
        path,
        seed=13,
        num_simulations=5,
        batch_size=4,
    )

    assert report["schema_id"] == COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID
    assert report["ok"] is True
    assert report["stock_model_contract_verified"] is True
    assert report["strict_stock_model_load_verified"] is True
    assert report["stock_eval_tournament_loadable"] is False
    assert report["stock_eval_tournament_load_status"] == (
        "strict_stock_model_load_verified_gameplay_not_run"
    )
    assert report["state_key"] == COMPACT_STOCK_EXPORT_MODEL_STATE_KEY
    assert report["source_max_steps_semantics"] == "source_physics_steps"
    assert report["tensor_count"] > 0
    assert calls == [
        {
            "checkpoint_path": path,
            "seed": 13,
            "num_simulations": 5,
            "batch_size": 4,
            "use_cuda": False,
            "state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        }
    ]


def test_compact_stock_export_model_contract_verifier_fails_closed(
    monkeypatch,
    tmp_path,
):
    checkpoint = _compact_checkpoint()
    path = tmp_path / "iteration_14.pth.tar"
    save_compact_stock_export_v1(
        checkpoint,
        path,
        policy_metadata=build_current_policy_observation_sidecar_v1(),
    )

    def fake_load_lightzero_policy(**kwargs):
        del kwargs
        raise RuntimeError("strict LightZero checkpoint load failed")

    monkeypatch.setattr(
        export_mod.provider,
        "load_lightzero_curvytron_visual_survival_policy",
        fake_load_lightzero_policy,
    )

    report = verify_compact_stock_export_model_contract_v1(path)

    assert report["ok"] is False
    assert report["stock_model_contract_verified"] is False
    assert report["stock_eval_tournament_loadable"] is False
    assert report["failure_reason"] == "strict_stock_model_load_failed"
    assert report["error_type"] == "RuntimeError"


def test_compact_stock_export_model_contract_verifier_requires_sidecar(tmp_path):
    checkpoint = _compact_checkpoint()
    path = tmp_path / "iteration_15.pth.tar"
    result = save_compact_stock_export_v1(
        checkpoint,
        path,
        policy_metadata=build_current_policy_observation_sidecar_v1(),
    )
    result["sidecar_path"].unlink()

    report = verify_compact_stock_export_model_contract_v1(path)

    assert report["ok"] is False
    assert report["failure_reason"] == "policy_metadata_sidecar_missing"
    assert report["stock_model_contract_verified"] is False


def test_compact_stock_export_evidence_bundle_attaches_reports_without_mutation(
    tmp_path,
):
    export_result, verifier_path, loader_path = _export_with_evidence_reports(tmp_path)
    saved = save_compact_stock_export_evidence_bundle_v1(
        export_result["checkpoint_path"],
        verification_report_path=verifier_path,
        tournament_loader_report_path=loader_path,
    )

    bundle = saved["bundle"]
    assert saved["path"] == compact_stock_export_evidence_bundle_path(
        export_result["checkpoint_path"]
    )
    assert bundle["schema_id"] == COMPACT_STOCK_EXPORT_EVIDENCE_BUNDLE_SCHEMA_ID
    assert bundle["ok"] is True
    assert bundle["base_export_claims_mutated"] is False
    assert bundle["attached_claims"] == {
        "stock_model_contract_verified_by_report": True,
        "strict_stock_model_load_verified": True,
        "tournament_loader_constructed": True,
        "stock_eval_tournament_loadable_by_evidence": False,
        "eval_gif_tournament_loadable_by_evidence": False,
        "promotion_claim": False,
        "training_speed_claim": False,
        "optimizer_resume_claim": False,
        "base_export_claims_mutated": False,
    }
    assert export_result["payload"]["metadata"]["stock_model_contract_verified"] is False
    assert export_result["payload"]["metadata"]["stock_eval_tournament_loadable"] is False
    assert export_result["payload"]["metadata"]["promotion_claim"] is False
    validate_compact_stock_export_evidence_bundle_v1(bundle)
    validate_compact_stock_export_evidence_bundle_v1(
        json.loads(saved["path"].read_text(encoding="utf-8"))
    )


def test_compact_stock_export_evidence_bundle_rejects_stale_hash(tmp_path):
    export_result, verifier_path, loader_path = _export_with_evidence_reports(tmp_path)
    saved = save_compact_stock_export_evidence_bundle_v1(
        export_result["checkpoint_path"],
        verification_report_path=verifier_path,
        tournament_loader_report_path=loader_path,
    )

    loader_payload = json.loads(loader_path.read_text(encoding="utf-8"))
    loader_payload["ok"] = False
    _write_json(loader_path, loader_payload)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_compact_stock_export_evidence_bundle_v1(saved["bundle"])


def test_compact_stock_export_evidence_bundle_rejects_failed_verifier(tmp_path):
    export_result, verifier_path, loader_path = _export_with_evidence_reports(tmp_path)
    verifier_payload = json.loads(verifier_path.read_text(encoding="utf-8"))
    verifier_payload["ok"] = False
    _write_json(verifier_path, verifier_payload)

    with pytest.raises(ValueError, match="verifier ok not true"):
        build_compact_stock_export_evidence_bundle_v1(
            export_result["checkpoint_path"],
            verification_report_path=verifier_path,
            tournament_loader_report_path=loader_path,
        )


def test_compact_stock_export_evidence_bundle_rejects_loader_runtime_drift(
    tmp_path,
):
    export_result, verifier_path, loader_path = _export_with_evidence_reports(tmp_path)
    loader_payload = json.loads(loader_path.read_text(encoding="utf-8"))
    loader_payload["runtime_settings"]["source_max_steps"] = 99
    _write_json(loader_path, loader_payload)

    with pytest.raises(ValueError, match="loader runtime source_max_steps mismatch"):
        build_compact_stock_export_evidence_bundle_v1(
            export_result["checkpoint_path"],
            verification_report_path=verifier_path,
            tournament_loader_report_path=loader_path,
        )


def test_compact_stock_export_evidence_bundle_rejects_promotion_claim(tmp_path):
    export_result, verifier_path, loader_path = _export_with_evidence_reports(tmp_path)
    bundle = build_compact_stock_export_evidence_bundle_v1(
        export_result["checkpoint_path"],
        verification_report_path=verifier_path,
        tournament_loader_report_path=loader_path,
    )
    bundle["attached_claims"]["promotion_claim"] = True

    with pytest.raises(ValueError, match="promotion_claim"):
        validate_compact_stock_export_evidence_bundle_v1(bundle)


def test_compact_stock_export_rejects_protected_extra_metadata_overrides():
    checkpoint = _compact_checkpoint()
    sidecar = build_current_policy_observation_sidecar_v1()

    for key in (
        "stock_eval_tournament_loadable",
        "stock_model_contract_verified",
        "promotion_claim",
        "training_speed_claim",
        "calls_train_muzero",
        "touches_live_runs",
        "eval_gif_tournament_load",
        "compact_current_chain_eval_gif_tournament_load_evidence",
    ):
        with pytest.raises(ValueError, match="protected metadata"):
            build_compact_stock_export_payload_v1(
                checkpoint,
                policy_metadata=sidecar,
                extra_metadata={key: True},
            )


def _export_with_evidence_reports(tmp_path):
    checkpoint = _compact_checkpoint()
    path = tmp_path / "iteration_16.pth.tar"
    sidecar = build_current_policy_observation_sidecar_v1()
    export_result = save_compact_stock_export_v1(
        checkpoint,
        path,
        policy_metadata=sidecar,
    )
    metadata = export_result["payload"]["metadata"]
    verifier_path = tmp_path / "verification_report.json"
    loader_path = tmp_path / "tournament_loader_report.json"
    _write_json(
        verifier_path,
        {
            "schema_id": COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID,
            "compact_stock_model_contract_verification": True,
            "checkpoint_path": str(path),
            "sidecar_path": str(export_result["sidecar_path"]),
            "sidecar_present": True,
            "sidecar_required": True,
            "state_key_requested": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
            "state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
            "ok": True,
            "strict_load": True,
            "stock_model_contract_verified": True,
            "strict_stock_model_load_verified": True,
            "stock_eval_tournament_loadable": False,
            "stock_eval_tournament_load_status": (
                "strict_stock_model_load_verified_gameplay_not_run"
            ),
            "failure_reason": None,
            "tensor_count": 2,
            "policy_observation_metadata": sidecar,
            "checkpoint_inferred_model_support_config": {"support_scale": 1},
            "load_summary": {
                "ok": True,
                "strict": True,
                "candidate": "as_is",
                "state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
                "missing_keys": [],
                "unexpected_keys": [],
            },
            **{
                key: metadata[key]
                for key in (
                    "source_compact_checkpoint_id",
                    "source_trainer_id",
                    "policy_version_ref",
                    "model_version_ref",
                    "policy_source",
                    "stock_model_state_key",
                    "model_env_variant",
                    "model_reward_variant",
                    "env_variant",
                    "reward_variant",
                    "decision_source_frames",
                    "source_physics_step_ms",
                    "decision_ms",
                    "source_max_steps",
                    "source_max_steps_semantics",
                    "learner_seat_mode",
                )
            },
        },
    )
    _write_json(
        loader_path,
        {
            "schema_id": COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID,
            "ok": True,
            "smoke_scope": "unit_loader_smoke_no_game_or_gif",
            "checkpoint_ref": str(path),
            "checkpoint_path": str(path),
            "checkpoint_state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
            "policy_trail_render_mode": sidecar["policy_trail_render_mode"],
            "policy_bonus_render_mode": sidecar["policy_bonus_render_mode"],
            "policy_observation_backend": sidecar["policy_observation_backend"],
            "policy_observation_contract_id": sidecar[
                "policy_observation_contract_id"
            ],
            "policy_observation_perspective_schema_id": sidecar[
                "policy_observation_perspective_schema_id"
            ],
            "model_env_variant": sidecar["model_env_variant"],
            "model_reward_variant": sidecar["model_reward_variant"],
            "runtime_settings": {
                key: sidecar[key]
                for key in (
                    "decision_source_frames",
                    "source_physics_step_ms",
                    "decision_ms",
                    "source_max_steps",
                    "source_max_steps_semantics",
                )
            },
            "surface": {
                "load_state_dict": {
                    "ok": True,
                    "strict": True,
                    "candidate": "as_is",
                    "missing_keys": [],
                    "unexpected_keys": [],
                }
            },
        },
    )
    return export_result, verifier_path, loader_path


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_checkpoint():
    torch = _torch_module()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    replay_state = _owned_replay_state(
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
    )
    resume = CompactTrainerResumeStateV1(
        trainer_id="unit-compact-trainer",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
        policy_source="unit_test_compact_stock_checkpoint_export",
        loop_counters={},
    )
    return build_compact_trainer_checkpoint_v1(
        checkpoint_id="unit-compact-ckpt-export",
        trainer_config={"schema_id": "unit_config"},
        resume_state=resume,
        model=model,
        optimizer=optimizer,
        replay_store_state=replay_state,
        metrics={"loss": 0.5},
    )


def _owned_replay_state(*, policy_version_ref: str, model_version_ref: str):
    policy = CompactPolicyVersionRefV1(
        policy_version_ref=policy_version_ref,
        policy_source="unit_test_compact_stock_checkpoint_export",
        model_version_ref=model_version_ref,
    )
    metadata = compact_owned_loop_replay_store_metadata(policy)
    ring = _CompactReplayRingV1(capacity=2, metadata=metadata)
    return ring.snapshot_durable_state(
        policy_version_ref=policy.policy_version_ref,
        policy_source=policy.policy_source,
        model_version_ref=policy.model_version_ref,
        metadata=metadata,
    )
