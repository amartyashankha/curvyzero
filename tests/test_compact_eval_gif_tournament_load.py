from __future__ import annotations

import json

import pytest

from curvyzero.training.compact_eval_gif_tournament_load import (
    COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_STATUS_VERIFIED,
)
from curvyzero.training.compact_eval_gif_tournament_load import (
    build_compact_current_chain_eval_gif_tournament_load_evidence_v1,
)
from curvyzero.training.compact_eval_gif_tournament_load import (
    compact_current_chain_eval_gif_tournament_load_evidence_path,
)
from curvyzero.training.compact_eval_gif_tournament_load import (
    compact_current_chain_eval_gif_tournament_load_evidence_ref,
)
from curvyzero.training.compact_eval_gif_tournament_load import (
    save_compact_current_chain_eval_gif_tournament_load_evidence_v1,
)
from curvyzero.training.compact_eval_gif_tournament_load import (
    validate_compact_current_chain_eval_gif_tournament_load_evidence_v1,
)
from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import (
    compact_owned_loop_replay_store_metadata,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    build_current_policy_observation_sidecar_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    save_compact_stock_export_evidence_bundle_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    save_compact_stock_export_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    CompactTrainerResumeStateV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    build_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    save_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    validate_compact_trainer_checkpoint_v1,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _CompactReplayRingV1,
)


def _torch_module():
    return pytest.importorskip("torch")


def test_current_chain_eval_gif_tournament_evidence_binds_exact_artifacts(tmp_path):
    artifacts = _current_chain_artifacts(tmp_path)

    evidence = build_compact_current_chain_eval_gif_tournament_load_evidence_v1(
        compact_checkpoint_path=artifacts["compact_checkpoint_path"],
        stock_export_path=artifacts["stock_export_path"],
        stock_export_evidence_bundle_path=artifacts["stock_bundle_path"],
        gameplay_smoke_report_path=artifacts["gameplay_report_path"],
    )

    assert evidence["status"] == (
        COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_STATUS_VERIFIED
    )
    assert evidence["current_chain_identity"]["checkpoint_id"] == (
        "unit-current-chain-ckpt"
    )
    assert evidence["attached_claims"]["eval_gif_tournament_load"] is True
    assert evidence["attached_claims"]["promotion_claim"] is False
    assert evidence["gameplay_summary"]["physical_steps"] == 4
    assert evidence["gameplay_summary"]["first_joint_action"] == [1, 1]
    assert evidence["standalone_eval_summary"]["steps_survived"] == 8
    validate_compact_current_chain_eval_gif_tournament_load_evidence_v1(evidence)
    assert compact_current_chain_eval_gif_tournament_load_evidence_ref(
        evidence
    ).startswith("compact_current_chain_eval_gif_tournament_load:")


def test_current_chain_eval_gif_tournament_evidence_saves_sibling_bundle(tmp_path):
    artifacts = _current_chain_artifacts(tmp_path)

    saved = save_compact_current_chain_eval_gif_tournament_load_evidence_v1(
        compact_checkpoint_path=artifacts["compact_checkpoint_path"],
        stock_export_path=artifacts["stock_export_path"],
        stock_export_evidence_bundle_path=artifacts["stock_bundle_path"],
        gameplay_smoke_report_path=artifacts["gameplay_report_path"],
    )

    assert saved["path"] == compact_current_chain_eval_gif_tournament_load_evidence_path(
        artifacts["stock_export_path"]
    )
    payload = json.loads(saved["path"].read_text(encoding="utf-8"))
    assert payload["schema_id"] == (
        "curvyzero_compact_current_chain_eval_gif_tournament_load_evidence/v1"
    )
    validate_compact_current_chain_eval_gif_tournament_load_evidence_v1(payload)


def test_current_chain_eval_gif_tournament_evidence_rejects_stale_gameplay_ref(
    tmp_path,
):
    artifacts = _current_chain_artifacts(tmp_path)
    gameplay = json.loads(
        artifacts["gameplay_report_path"].read_text(encoding="utf-8")
    )
    gameplay["checkpoint_ref"] = str(tmp_path / "other_iteration.pth.tar")
    _write_json(artifacts["gameplay_report_path"], gameplay)

    with pytest.raises(ValueError, match="gameplay checkpoint_ref mismatch"):
        build_compact_current_chain_eval_gif_tournament_load_evidence_v1(
            compact_checkpoint_path=artifacts["compact_checkpoint_path"],
            stock_export_path=artifacts["stock_export_path"],
            stock_export_evidence_bundle_path=artifacts["stock_bundle_path"],
            gameplay_smoke_report_path=artifacts["gameplay_report_path"],
        )


def test_compact_trainer_checkpoint_rejects_tampered_eval_gif_gate(tmp_path):
    checkpoint = _compact_checkpoint()
    checkpoint.metadata[
        "compact_coach_compatibility_gate_eval_gif_tournament_load"
    ] = True
    checkpoint.metadata["compact_coach_compatibility_evidence"][
        "eval_gif_tournament_load"
    ] = "unit_loader_smoke_no_game_or_gif"

    with pytest.raises(ValueError, match="eval/GIF Coach gate mismatch"):
        validate_compact_trainer_checkpoint_v1(checkpoint)


def _current_chain_artifacts(tmp_path):
    checkpoint = _compact_checkpoint()
    compact_checkpoint_path = save_compact_trainer_checkpoint_v1(
        checkpoint,
        tmp_path / "compact_checkpoint.pt",
    )
    sidecar = build_current_policy_observation_sidecar_v1()
    stock_export = save_compact_stock_export_v1(
        checkpoint,
        tmp_path / "iteration_0.pth.tar",
        policy_metadata=sidecar,
    )
    stock_export_path = stock_export["checkpoint_path"]
    metadata = stock_export["payload"]["metadata"]
    verifier_path = tmp_path / "verification_report.json"
    loader_path = tmp_path / "tournament_loader_report.json"
    _write_json(verifier_path, _verification_report(stock_export_path, sidecar, metadata))
    _write_json(loader_path, _loader_report(stock_export_path, sidecar))
    saved_bundle = save_compact_stock_export_evidence_bundle_v1(
        stock_export_path,
        verification_report_path=verifier_path,
        tournament_loader_report_path=loader_path,
    )
    gameplay_report_path = tmp_path / "one_game_gif_smoke_report.json"
    _write_gameplay_artifacts(
        root=tmp_path,
        gameplay_report_path=gameplay_report_path,
        stock_export_path=stock_export_path,
        sidecar_path=stock_export["sidecar_path"],
        verifier_path=verifier_path,
        loader_path=loader_path,
        stock_bundle_path=saved_bundle["path"],
        sidecar=sidecar,
    )
    return {
        "compact_checkpoint_path": compact_checkpoint_path,
        "stock_export_path": stock_export_path,
        "stock_bundle_path": saved_bundle["path"],
        "gameplay_report_path": gameplay_report_path,
    }


def _verification_report(path, sidecar, metadata):
    return {
        "schema_id": COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID,
        "compact_stock_model_contract_verification": True,
        "checkpoint_path": str(path),
        "sidecar_path": f"{path}.metadata.json",
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
        "load_summary": _strict_load_summary(),
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
    }


def _loader_report(path, sidecar):
    return {
        "schema_id": COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID,
        "ok": True,
        "smoke_scope": "unit_loader_smoke_no_game_or_gif",
        "checkpoint_ref": str(path),
        "checkpoint_path": str(path),
        "checkpoint_state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        "policy_trail_render_mode": sidecar["policy_trail_render_mode"],
        "policy_bonus_render_mode": sidecar["policy_bonus_render_mode"],
        "policy_observation_backend": sidecar["policy_observation_backend"],
        "policy_observation_contract_id": sidecar["policy_observation_contract_id"],
        "policy_observation_perspective_schema_id": sidecar[
            "policy_observation_perspective_schema_id"
        ],
        "model_env_variant": sidecar["model_env_variant"],
        "model_reward_variant": sidecar["model_reward_variant"],
        "runtime_settings": _runtime_settings(sidecar),
        "surface": {"load_state_dict": _strict_load_summary()},
    }


def _write_gameplay_artifacts(
    *,
    root,
    gameplay_report_path,
    stock_export_path,
    sidecar_path,
    verifier_path,
    loader_path,
    stock_bundle_path,
    sidecar,
):
    game_ref = (
        "tournaments/curvytron/unit-current-chain/battles/unit/games/"
        "game-000000/summary.json"
    )
    gif_ref = (
        "tournaments/curvytron/unit-current-chain/battles/unit/games/"
        "game-000000/game.gif"
    )
    frames_ref = (
        "tournaments/curvytron/unit-current-chain/battles/unit/games/"
        "game-000000/frames.npz"
    )
    game_summary_path = root / game_ref
    gif_path = root / gif_ref
    frames_path = root / frames_ref
    eval_summary_path = root / "eval" / "summary.json"
    game_summary_path.parent.mkdir(parents=True, exist_ok=True)
    eval_summary_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.write_bytes(b"GIF89a unit smoke\n")
    frames_path.write_bytes(b"npz unit smoke\n")
    _write_json(
        game_summary_path,
        {
            "ok": True,
            "summary_ref": game_ref,
            "gif_ref": gif_ref,
            "frames_ref": frames_ref,
        },
    )
    _write_json(eval_summary_path, {"ok": True})
    policy_loads = [
        _policy_load(stock_export_path, sidecar),
        _policy_load(stock_export_path, sidecar),
    ]
    _write_json(
        gameplay_report_path,
        {
            "schema_id": "curvyzero_compact_stock_export_one_game_gif_smoke/v1",
            "ok": True,
            "checkpoint_ref": str(stock_export_path),
            "sidecar_ref": str(sidecar_path),
            "evidence_bundle_ref": str(stock_bundle_path),
            "verification_report_ref": str(verifier_path),
            "tournament_loader_report_ref": str(loader_path),
            "policy_loads": policy_loads,
            "first_action_trace": [
                {"physical_step": 1, "joint_action": [1, 1], "done": False}
            ],
            "physical_steps": 4,
            "action_counts": {"seat_0": {"1": 4}, "seat_1": {"1": 4}},
            "gif_ref": gif_ref,
            "frames_ref": frames_ref,
            "game_summary_ref": game_ref,
            "standalone_eval": {
                "ok": True,
                "status": {
                    "cap": 8,
                    "checkpoint_load_ok": True,
                    "env_reset_ok": True,
                    "policy_could_act_in_real_env": True,
                    "steps_survived": 8,
                    "strict_policy_model_load_ok": True,
                },
                "load_state_dict": _strict_load_summary(),
                "artifact": {"path": str(eval_summary_path), "ref": str(eval_summary_path)},
            },
            "promotion_claim": False,
            "training_speed_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        },
    )


def _policy_load(path, sidecar):
    return {
        "checkpoint_path": str(path),
        "checkpoint_ref": str(path),
        "checkpoint_state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        "policy_trail_render_mode": sidecar["policy_trail_render_mode"],
        "policy_bonus_render_mode": sidecar["policy_bonus_render_mode"],
        "policy_observation_backend": sidecar["policy_observation_backend"],
        "policy_observation_contract_id": sidecar["policy_observation_contract_id"],
        "policy_observation_perspective_schema_id": sidecar[
            "policy_observation_perspective_schema_id"
        ],
        "model_env_variant": sidecar["model_env_variant"],
        "model_reward_variant": sidecar["model_reward_variant"],
        "runtime_settings": _runtime_settings(sidecar),
        "surface": {"load_state_dict": _strict_load_summary()},
    }


def _strict_load_summary():
    return {
        "ok": True,
        "strict": True,
        "candidate": "as_is",
        "missing_keys": [],
        "unexpected_keys": [],
    }


def _runtime_settings(sidecar):
    return {
        key: sidecar[key]
        for key in (
            "decision_source_frames",
            "source_physics_step_ms",
            "decision_ms",
            "source_max_steps",
            "source_max_steps_semantics",
        )
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _compact_checkpoint():
    torch = _torch_module()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    replay_state = _owned_replay_state(
        policy_version_ref="unit-policy:update-current-chain",
        model_version_ref="unit-model:update-current-chain",
    )
    resume = CompactTrainerResumeStateV1(
        trainer_id="unit-current-chain-trainer",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref="unit-policy:update-current-chain",
        model_version_ref="unit-model:update-current-chain",
        policy_source="unit_test_compact_eval_gif_tournament_load",
        loop_counters={},
    )
    return build_compact_trainer_checkpoint_v1(
        checkpoint_id="unit-current-chain-ckpt",
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
        policy_source="unit_test_compact_eval_gif_tournament_load",
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
