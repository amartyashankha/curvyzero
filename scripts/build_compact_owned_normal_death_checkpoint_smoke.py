#!/usr/bin/env python3
"""Build a local compact-owned checkpoint with payload-derived normal-death proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from curvyzero.env.vector_runtime import DEATH_MODE_NORMAL
from curvyzero.training.compact_death_terminal_contract import (
    COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP,
)
from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import compact_owned_loop_replay_store_metadata
from curvyzero.training.compact_trainer_checkpoint import (
    CompactTrainerResumeStateV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    build_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    load_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    save_compact_trainer_checkpoint_v1,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _CompactReplayRingV1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_owned_normal_death_checkpoint_results"
)
DEFAULT_RUN_ID = "optimizer-compact-owned-normal-death-checkpoint-smoke-20260530"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--profile-result",
        type=Path,
        default=None,
        help=(
            "Optional hybrid profile result JSON. If provided, the compact payload "
            "is consumed as the normal-death evidence source."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_root / str(args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "compact_normal_death_checkpoint.pt"
    report_path = output_dir / "normal_death_checkpoint_smoke_report.json"

    import torch

    torch.manual_seed(20260530)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    resume = CompactTrainerResumeStateV1(
        trainer_id="compact-owned-normal-death-smoke",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref="normal-death-policy:update-1",
        model_version_ref="normal-death-model:update-1",
        policy_source="compact_owned_normal_death_checkpoint_smoke",
        loop_counters={},
    )
    profile_payload = _load_profile_payload(args.profile_result)
    evidence_refs = [
        "source_collision_head_head_reverse_order_single_death_step.json",
        "payload-derived-normal-death-evidence-local-20260530",
    ]
    if args.profile_result is not None:
        evidence_refs.append(str(args.profile_result))
    checkpoint = build_compact_trainer_checkpoint_v1(
        checkpoint_id=str(args.run_id),
        trainer_config={"schema_id": "compact_owned_normal_death_smoke_config/v1"},
        resume_state=resume,
        model=model,
        optimizer=optimizer,
        replay_store_state=_owned_replay_state(
            policy_version_ref=resume.policy_version_ref,
            model_version_ref=resume.model_version_ref,
        ),
        metrics={"normal_death_checkpoint_smoke_loss": 1.25},
        death_mode=DEATH_MODE_NORMAL,
        normal_collision_death_profile_result=profile_payload,
        normal_collision_death_evidence_id=f"{args.run_id}:normal-death-profile",
        normal_collision_death_evidence_refs=evidence_refs,
    )
    save_compact_trainer_checkpoint_v1(checkpoint, checkpoint_path)
    loaded = load_compact_trainer_checkpoint_v1(checkpoint_path)
    metadata = loaded.metadata
    report = {
        "schema_id": "curvyzero_compact_owned_normal_death_checkpoint_smoke/v1",
        "ok": True,
        "run_id": str(args.run_id),
        "profile_result_path": None if args.profile_result is None else str(args.profile_result),
        "checkpoint_path": str(checkpoint_path),
        "death_mode": metadata["death_mode"],
        "death_terminal_contract": metadata["death_terminal_contract"],
        "death_terminal_contract_status": metadata["death_terminal_contract_status"],
        "normal_collision_death_supported": metadata[
            "normal_collision_death_supported"
        ],
        "normal_collision_death_evidence_id": metadata[
            "compact_death_terminal_contract"
        ]["normal_collision_death_evidence_id"],
        "compact_coach_compatibility_gate_death_terminal_contract": metadata[
            "compact_coach_compatibility_gate_death_terminal_contract"
        ],
        "compact_coach_compatibility_promotion_eligible": metadata[
            "compact_coach_compatibility_promotion_eligible"
        ],
        "compact_coach_compatibility_missing_required_gates": metadata[
            "compact_coach_compatibility_missing_required_gates"
        ],
        "compact_coach_compatibility_evidence": metadata[
            "compact_coach_compatibility_evidence"
        ],
        "promotion_claim": metadata["promotion_claim"],
        "training_speed_claim": metadata["training_speed_claim"],
        "calls_train_muzero": metadata["calls_train_muzero"],
        "touches_live_runs": metadata["touches_live_runs"],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "report_path": str(report_path)}, sort_keys=True))
    return 0


def _load_profile_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _normal_death_profile_payload()
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("compact"), dict):
        return payload["compact"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("profile result must be a JSON object")


def _owned_replay_state(*, policy_version_ref: str, model_version_ref: str):
    policy = CompactPolicyVersionRefV1(
        policy_version_ref=policy_version_ref,
        policy_source="compact_owned_normal_death_checkpoint_smoke",
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


def _normal_death_profile_payload() -> dict[str, Any]:
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


if __name__ == "__main__":
    raise SystemExit(main())
