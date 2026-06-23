"""Checkpoint envelope for the compact-owned trainer candidate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_REQUIRED_PROMOTION_GATES,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
)
from curvyzero.training.compact_coach_compatibility import (
    build_compact_coach_compatibility_report_v1,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_compact_death_terminal_contract_v1,
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
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_policy_refresh_handoff_evidence_ref,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    validate_compact_policy_refresh_handoff_v1,
)
from curvyzero.training.compact_reward_rnd_contract import (
    build_compact_reward_rnd_contract_v1,
)
from curvyzero.training.compact_reward_rnd_contract import (
    compact_reward_rnd_contract_evidence_ref,
)
from curvyzero.training.compact_reward_rnd_contract import (
    validate_compact_reward_rnd_contract_v1,
)
from curvyzero.training.compact_training_metrics_lineage import (
    compact_training_metrics_lineage_evidence_ref,
)
from curvyzero.training.compact_training_metrics_lineage import (
    validate_compact_training_metrics_lineage_v1,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    COMPACT_REPLAY_STORE_STATE_SCHEMA_ID,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    CompactReplayStoreStateV1,
)


COMPACT_TRAINER_CHECKPOINT_SCHEMA_ID = "curvyzero_compact_trainer_checkpoint/v1"
COMPACT_TRAINER_CHECKPOINT_KIND = "compact_owned_trainer_checkpoint"
_PROTECTED_EXTRA_METADATA_PREFIXES = (
    "compact_coach_compatibility_",
    "compact_current_chain_eval_gif_tournament_load",
    "compact_death_terminal_contract",
    "compact_policy_refresh_handoff",
    "compact_reward_rnd_contract",
    "compact_training_metrics_lineage",
    "death_terminal_contract",
    "eval_gif_tournament_load",
    "policy_refresh_handoff",
    "reward_rnd_contract",
    "training_metrics_lineage",
)
_PROTECTED_EXTRA_METADATA_KEYS = frozenset(
    (
        "calls_train_muzero",
        "coach_integration_claim",
        "compact_eval_adapter_required",
        "death_mode",
        "eval_gif_tournament_load",
        "eval_gif_tournament_load_status",
        "exploration_bonus_enabled",
        "exploration_bonus_mode",
        "lightzero_training_integration_claim",
        "max_ticks_terminal_supported",
        "normal_collision_death_supported",
        "profile_no_death_supported",
        "profile_only",
        "profile_only_terminal_contract",
        "promotion_claim",
        "reward_perspective",
        "reward_schema_hash",
        "reward_schema_id",
        "reward_target_effect",
        "reward_variant",
        "rnd_enabled",
        "rnd_state_dict_present",
        "rnd_state_present",
        "rnd_state_required",
        "stock_eval_tournament_load_status",
        "stock_eval_tournament_loadable",
        "terminal_unroll_value_target_mode",
        "terminated_supported",
        "touches_live_runs",
        "training_metrics_lineage",
        "training_speed_claim",
        "truncated_supported",
    )
)


@dataclass(frozen=True, slots=True)
class CompactTrainerResumeStateV1:
    """Trainer counters and lineage needed to resume compact-owned training."""

    trainer_id: str
    train_step: int
    learner_update_count: int
    sample_batch_count: int
    policy_version_ref: str
    model_version_ref: str
    policy_source: str
    loop_counters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactOwnedLoopCountersV1:
    """Serializable copy of compact-owned loop counters."""

    sample_gate_calls: int
    sample_gate_opportunities: int
    sample_gate_skipped_count: int
    sample_gate_index_rows: int
    sample_gate_target_rows: int
    sample_gate_sample_rows: int
    sample_gate_sec: float
    sample_gate_last_telemetry: dict[str, Any]
    sample_gate_last_sample_metadata: dict[str, Any]
    learner_gate_calls: int
    learner_gate_updates: int
    learner_gate_sample_rows: int
    learner_gate_input_bytes: int
    learner_gate_sec: float
    learner_gate_last_telemetry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactOwnedLoopRuntimeStateV1:
    """Runtime state needed so post-resume record_step does not warmup-drop."""

    previous_step: Any | None
    counters: CompactOwnedLoopCountersV1


@dataclass(frozen=True, slots=True)
class CompactTrainerCheckpointV1:
    """Atomic local checkpoint payload for the compact-owned trainer candidate."""

    metadata: dict[str, Any]
    trainer_config: dict[str, Any]
    resume_state: CompactTrainerResumeStateV1
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    replay_store_state: CompactReplayStoreStateV1
    metrics: dict[str, Any]
    loop_runtime_state: CompactOwnedLoopRuntimeStateV1 | None = None
    rng_state: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    rnd_state_dict: dict[str, Any] | None = None


def build_compact_trainer_checkpoint_v1(
    *,
    checkpoint_id: str,
    trainer_config: Mapping[str, Any],
    resume_state: CompactTrainerResumeStateV1,
    model: Any,
    optimizer: Any,
    replay_store_state: CompactReplayStoreStateV1,
    metrics: Mapping[str, Any] | None = None,
    loop_runtime_state: CompactOwnedLoopRuntimeStateV1 | None = None,
    rng_state: Mapping[str, Any] | None = None,
    scheduler: Any | None = None,
    rnd_state_dict: Mapping[str, Any] | None = None,
    exploration_bonus_config: Any = None,
    death_mode: str = "profile_no_death",
    normal_collision_death_evidence: Mapping[str, Any] | None = None,
    normal_collision_death_profile_result: Mapping[str, Any] | None = None,
    normal_collision_death_evidence_id: str = "",
    normal_collision_death_evidence_refs: list[str] | tuple[str, ...] | None = None,
    policy_refresh_handoff: Mapping[str, Any] | None = None,
    training_metrics_lineage: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> CompactTrainerCheckpointV1:
    """Build a compact trainer checkpoint and fail closed on overclaims."""

    checkpoint_id_value = str(checkpoint_id).strip()
    if not checkpoint_id_value:
        raise ValueError("checkpoint_id must be non-empty")
    _validate_resume_state(resume_state)
    replay_metadata = _validate_replay_store_state(
        replay_store_state,
        expected_policy_version_ref=resume_state.policy_version_ref,
        expected_model_version_ref=resume_state.model_version_ref,
    )
    model_state_dict = _clone_state_dict(model.state_dict())
    optimizer_state_dict = _clone_state_dict(optimizer.state_dict())
    scheduler_state_dict = (
        None if scheduler is None else _clone_state_dict(scheduler.state_dict())
    )
    rnd_state = None if rnd_state_dict is None else _clone_state_dict(rnd_state_dict)
    reward_rnd_contract = build_compact_reward_rnd_contract_v1(
        exploration_bonus_config=exploration_bonus_config,
        rnd_state_dict=rnd_state,
    )
    normal_collision_death_evidence = _resolve_normal_collision_death_evidence(
        checkpoint_id=checkpoint_id_value,
        normal_collision_death_evidence=normal_collision_death_evidence,
        normal_collision_death_profile_result=normal_collision_death_profile_result,
        normal_collision_death_evidence_id=normal_collision_death_evidence_id,
        normal_collision_death_evidence_refs=normal_collision_death_evidence_refs,
    )
    death_terminal_contract = build_compact_death_terminal_contract_v1(
        death_mode=death_mode,
        normal_collision_evidence=normal_collision_death_evidence,
    )
    metrics_dict = _plain_mapping(metrics or {})
    config_dict = _plain_mapping(trainer_config)
    policy_refresh_handoff_contract = (
        None if policy_refresh_handoff is None else _plain_mapping(policy_refresh_handoff)
    )
    if policy_refresh_handoff_contract is not None:
        validate_compact_policy_refresh_handoff_v1(policy_refresh_handoff_contract)
        _validate_policy_refresh_handoff_matches_checkpoint(
            policy_refresh_handoff_contract,
            checkpoint_id=checkpoint_id_value,
            resume_state=resume_state,
        )
    training_metrics_lineage_contract = (
        None
        if training_metrics_lineage is None
        else _plain_mapping(training_metrics_lineage)
    )
    if training_metrics_lineage_contract is not None:
        validate_compact_training_metrics_lineage_v1(
            training_metrics_lineage_contract
        )
    compatibility = _checkpoint_compatibility_report(
        checkpoint_id=checkpoint_id_value,
        resume_state=resume_state,
        has_metrics=bool(metrics_dict),
        reward_rnd_contract=reward_rnd_contract,
        death_terminal_contract=death_terminal_contract,
        policy_refresh_handoff=policy_refresh_handoff_contract,
        training_metrics_lineage=training_metrics_lineage_contract,
    )
    metadata: dict[str, Any] = {
        "schema_id": COMPACT_TRAINER_CHECKPOINT_SCHEMA_ID,
        "compact_trainer_checkpoint_schema_id": COMPACT_TRAINER_CHECKPOINT_SCHEMA_ID,
        "checkpoint_kind": COMPACT_TRAINER_CHECKPOINT_KIND,
        "compact_trainer_checkpoint": True,
        "checkpoint_id": checkpoint_id_value,
        "trainer_id": resume_state.trainer_id,
        "policy_version_ref": resume_state.policy_version_ref,
        "model_version_ref": resume_state.model_version_ref,
        "policy_source": resume_state.policy_source,
        "train_step": int(resume_state.train_step),
        "learner_update_count": int(resume_state.learner_update_count),
        "sample_batch_count": int(resume_state.sample_batch_count),
        "replay_store_state_schema_id": replay_metadata[
            "compact_replay_store_state_schema_id"
        ],
        "replay_store_entry_count": replay_metadata.get(
            "compact_replay_store_entry_count"
        ),
        "loop_runtime_state_present": loop_runtime_state is not None,
        "loop_previous_step_present": (
            loop_runtime_state is not None
            and loop_runtime_state.previous_step is not None
        ),
        "compact_reward_rnd_contract": reward_rnd_contract,
        "compact_reward_rnd_contract_schema_id": reward_rnd_contract[
            "compact_reward_rnd_contract_schema_id"
        ],
        "compact_reward_rnd_contract_mode": reward_rnd_contract[
            "compact_reward_rnd_contract_mode"
        ],
        "compact_reward_rnd_contract_verified": True,
        "reward_rnd_contract": True,
        "reward_variant": reward_rnd_contract["reward_variant"],
        "reward_schema_id": reward_rnd_contract["reward_schema_id"],
        "reward_schema_hash": reward_rnd_contract["reward_schema_hash"],
        "reward_perspective": reward_rnd_contract["reward_perspective"],
        "reward_target_effect": reward_rnd_contract["reward_target_effect"],
        "exploration_bonus_mode": reward_rnd_contract["exploration_bonus_mode"],
        "exploration_bonus_enabled": False,
        "rnd_enabled": False,
        "rnd_state_required": False,
        "rnd_state_present": False,
        "rnd_state_dict_present": False,
        "compact_death_terminal_contract": death_terminal_contract,
        "compact_death_terminal_contract_schema_id": death_terminal_contract[
            "compact_death_terminal_contract_schema_id"
        ],
        "compact_death_terminal_contract_mode": death_terminal_contract[
            "compact_death_terminal_contract_mode"
        ],
        "compact_death_terminal_contract_verified": True,
        "compact_death_terminal_contract_promotion_gate_satisfied": (
            death_terminal_contract[
                "compact_death_terminal_contract_promotion_gate_satisfied"
            ]
        ),
        "compact_death_terminal_contract_blocker": death_terminal_contract[
            "compact_death_terminal_contract_blocker"
        ],
        "death_terminal_contract": death_terminal_contract[
            "compact_death_terminal_contract_promotion_gate_satisfied"
        ],
        "death_terminal_contract_status": death_terminal_contract[
            "death_terminal_contract_status"
        ],
        "death_mode": death_terminal_contract["death_mode"],
        "profile_only_terminal_contract": death_terminal_contract[
            "profile_only_terminal_contract"
        ],
        "normal_collision_death_supported": death_terminal_contract[
            "normal_collision_death_supported"
        ],
        "profile_no_death_supported": death_terminal_contract[
            "profile_no_death_supported"
        ],
        "max_ticks_terminal_supported": death_terminal_contract[
            "max_ticks_terminal_supported"
        ],
        "terminated_supported": death_terminal_contract["terminated_supported"],
        "truncated_supported": death_terminal_contract["truncated_supported"],
        "terminal_unroll_value_target_mode": death_terminal_contract[
            "terminal_unroll_value_target_mode"
        ],
        "compact_current_chain_eval_gif_tournament_load_evidence": None,
        "compact_current_chain_eval_gif_tournament_load_verified": False,
        "eval_gif_tournament_load": False,
        "eval_gif_tournament_load_status": (
            "missing_current_chain_eval_gif_tournament_load_evidence"
        ),
        "compact_policy_refresh_handoff": policy_refresh_handoff_contract,
        "compact_policy_refresh_handoff_schema_id": (
            ""
            if policy_refresh_handoff_contract is None
            else policy_refresh_handoff_contract[
                "compact_policy_refresh_handoff_schema_id"
            ]
        ),
        "compact_policy_refresh_handoff_verified": (
            policy_refresh_handoff_contract is not None
        ),
        "policy_refresh_handoff": policy_refresh_handoff_contract is not None,
        "policy_refresh_handoff_status": (
            "missing_policy_refresh_handoff_contract"
            if policy_refresh_handoff_contract is None
            else policy_refresh_handoff_contract["policy_refresh_handoff_status"]
        ),
        "compact_training_metrics_lineage": training_metrics_lineage_contract,
        "compact_training_metrics_lineage_schema_id": (
            ""
            if training_metrics_lineage_contract is None
            else training_metrics_lineage_contract[
                "compact_training_metrics_lineage_schema_id"
            ]
        ),
        "compact_training_metrics_lineage_verified": (
            training_metrics_lineage_contract is not None
        ),
        "training_metrics_lineage": training_metrics_lineage_contract is not None,
        "training_metrics_lineage_status": (
            "missing_training_metrics_lineage_contract"
            if training_metrics_lineage_contract is None
            else training_metrics_lineage_contract[
                "training_metrics_lineage_status"
            ]
        ),
        "checkpoint_save_load": True,
        "resume_metadata": True,
        "stock_eval_tournament_loadable": False,
        "stock_eval_tournament_load_status": "adapter_missing",
        "compact_eval_adapter_required": True,
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
        "training_speed_claim": False,
        "coach_integration_claim": False,
        "lightzero_training_integration_claim": False,
    }
    metadata.update(compatibility.as_metadata())
    extra_metadata_dict = _plain_mapping(extra_metadata or {})
    _validate_checkpoint_extra_metadata(extra_metadata_dict)
    metadata.update(extra_metadata_dict)
    checkpoint = CompactTrainerCheckpointV1(
        metadata=metadata,
        trainer_config=config_dict,
        resume_state=resume_state,
        model_state_dict=model_state_dict,
        optimizer_state_dict=optimizer_state_dict,
        replay_store_state=replay_store_state,
        metrics=metrics_dict,
        loop_runtime_state=loop_runtime_state,
        rng_state=None if rng_state is None else _clone_state_dict(dict(rng_state)),
        scheduler_state_dict=scheduler_state_dict,
        rnd_state_dict=rnd_state,
    )
    validate_compact_trainer_checkpoint_v1(checkpoint)
    return checkpoint


def save_compact_trainer_checkpoint_v1(
    checkpoint: CompactTrainerCheckpointV1,
    path: str | Path,
) -> Path:
    """Persist a compact trainer checkpoint with torch serialization."""

    validate_compact_trainer_checkpoint_v1(checkpoint)
    import torch

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path_obj)
    return path_obj


def load_compact_trainer_checkpoint_v1(path: str | Path) -> CompactTrainerCheckpointV1:
    """Load and validate a compact trainer checkpoint."""

    import torch

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_compact_trainer_checkpoint_v1(checkpoint)
    return checkpoint


def restore_compact_trainer_checkpoint_v1(
    checkpoint: CompactTrainerCheckpointV1,
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
) -> CompactReplayStoreStateV1:
    """Restore model/optimizer/scheduler state and return replay state."""

    validate_compact_trainer_checkpoint_v1(checkpoint)
    model.load_state_dict(checkpoint.model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    if scheduler is not None and checkpoint.scheduler_state_dict is not None:
        scheduler.load_state_dict(checkpoint.scheduler_state_dict)
    return checkpoint.replay_store_state


def validate_compact_trainer_checkpoint_v1(checkpoint: Any) -> None:
    """Validate the compact checkpoint envelope, not just its replay snapshot."""

    metadata = getattr(checkpoint, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("compact trainer checkpoint metadata must be a mapping")
    schema_id = metadata.get("schema_id") or metadata.get(
        "compact_trainer_checkpoint_schema_id"
    )
    if schema_id != COMPACT_TRAINER_CHECKPOINT_SCHEMA_ID:
        raise ValueError("compact trainer checkpoint schema mismatch")
    if metadata.get("checkpoint_kind") != COMPACT_TRAINER_CHECKPOINT_KIND:
        raise ValueError("compact trainer checkpoint kind mismatch")
    if metadata.get("promotion_claim") is not False:
        raise ValueError("compact trainer checkpoint must not claim promotion")
    if metadata.get("calls_train_muzero") is not False:
        raise ValueError("compact trainer checkpoint must not call train_muzero")
    if metadata.get("touches_live_runs") is not False:
        raise ValueError("compact trainer checkpoint must not touch live runs")
    if metadata.get("checkpoint_save_load") is not True:
        raise ValueError("compact trainer checkpoint must prove checkpoint save/load")
    if metadata.get("resume_metadata") is not True:
        raise ValueError("compact trainer checkpoint must include resume metadata")
    reward_rnd_contract = metadata.get("compact_reward_rnd_contract")
    validate_compact_reward_rnd_contract_v1(
        reward_rnd_contract,
        rnd_state_dict=getattr(checkpoint, "rnd_state_dict", None),
    )
    if metadata.get("compact_reward_rnd_contract_verified") is not True:
        raise ValueError("compact trainer checkpoint missing reward/RND proof")
    if metadata.get("reward_rnd_contract") is not True:
        raise ValueError("compact trainer checkpoint must prove reward/RND contract")
    for key in (
        "reward_variant",
        "reward_schema_id",
        "reward_schema_hash",
        "reward_perspective",
        "reward_target_effect",
        "exploration_bonus_mode",
        "rnd_enabled",
        "rnd_state_required",
        "rnd_state_present",
    ):
        if metadata.get(key) != reward_rnd_contract.get(key):
            raise ValueError(f"compact trainer checkpoint reward/RND {key} mismatch")
    if metadata.get("exploration_bonus_enabled") is not False:
        raise ValueError("compact trainer checkpoint must keep exploration bonus disabled")
    if metadata.get("rnd_state_dict_present") is not False:
        raise ValueError("compact trainer checkpoint must not claim RND state")
    death_terminal_contract = metadata.get("compact_death_terminal_contract")
    validate_compact_death_terminal_contract_v1(death_terminal_contract)
    if metadata.get("compact_death_terminal_contract_verified") is not True:
        raise ValueError("compact trainer checkpoint missing death/terminal proof")
    if metadata.get("death_terminal_contract") != death_terminal_contract.get(
        "compact_death_terminal_contract_promotion_gate_satisfied"
    ):
        raise ValueError("compact trainer checkpoint death/terminal gate mismatch")
    if metadata.get("death_terminal_contract_status") != death_terminal_contract.get(
        "death_terminal_contract_status"
    ):
        raise ValueError("compact trainer checkpoint death/terminal status mismatch")
    for key in (
        "compact_death_terminal_contract_promotion_gate_satisfied",
        "compact_death_terminal_contract_blocker",
        "death_mode",
        "profile_only_terminal_contract",
        "normal_collision_death_supported",
        "profile_no_death_supported",
        "max_ticks_terminal_supported",
        "terminated_supported",
        "truncated_supported",
        "terminal_unroll_value_target_mode",
    ):
        if metadata.get(key) != death_terminal_contract.get(key):
            raise ValueError(
                f"compact trainer checkpoint death/terminal {key} mismatch"
            )
    eval_gif_tournament_load = metadata.get(
        "compact_current_chain_eval_gif_tournament_load_evidence"
    )
    if eval_gif_tournament_load is not None:
        raise ValueError(
            "compact trainer checkpoint must attach eval/GIF evidence as a "
            "sibling artifact, not mutate checkpoint metadata"
        )
    if (
        metadata.get("compact_current_chain_eval_gif_tournament_load_verified")
        is not False
    ):
        raise ValueError("compact trainer checkpoint eval/GIF verification mismatch")
    if metadata.get("eval_gif_tournament_load") is not False:
        raise ValueError(
            "compact trainer checkpoint must not claim eval/GIF/tournament load"
        )
    if metadata.get("compact_coach_compatibility_gate_eval_gif_tournament_load") is not False:
        raise ValueError("compact trainer checkpoint eval/GIF Coach gate mismatch")
    if metadata.get("stock_eval_tournament_loadable") is not False:
        raise ValueError("stock eval/tournament loadability must be explicit false")
    policy_refresh_handoff = metadata.get("compact_policy_refresh_handoff")
    if policy_refresh_handoff is None:
        if metadata.get("compact_policy_refresh_handoff_verified") is not False:
            raise ValueError(
                "compact trainer checkpoint policy refresh verified mismatch"
            )
        if metadata.get("policy_refresh_handoff") is not False:
            raise ValueError(
                "compact trainer checkpoint must not claim policy refresh handoff"
            )
    else:
        validate_compact_policy_refresh_handoff_v1(policy_refresh_handoff)
        _validate_policy_refresh_handoff_matches_metadata(
            policy_refresh_handoff,
            metadata=metadata,
        )
        if metadata.get("compact_policy_refresh_handoff_verified") is not True:
            raise ValueError(
                "compact trainer checkpoint missing policy refresh verification"
            )
        if metadata.get("policy_refresh_handoff") is not True:
            raise ValueError(
                "compact trainer checkpoint policy refresh gate mismatch"
            )
        if metadata.get("policy_refresh_handoff_status") != (
            policy_refresh_handoff.get("policy_refresh_handoff_status")
        ):
            raise ValueError(
                "compact trainer checkpoint policy refresh status mismatch"
            )
    training_metrics_lineage = metadata.get("compact_training_metrics_lineage")
    if training_metrics_lineage is None:
        if metadata.get("compact_training_metrics_lineage_verified") is not False:
            raise ValueError(
                "compact trainer checkpoint metrics lineage verified mismatch"
            )
        if metadata.get("training_metrics_lineage") is not False:
            raise ValueError(
                "compact trainer checkpoint must not claim metrics lineage"
            )
    else:
        validate_compact_training_metrics_lineage_v1(training_metrics_lineage)
        if metadata.get("compact_training_metrics_lineage_verified") is not True:
            raise ValueError(
                "compact trainer checkpoint missing metrics lineage verification"
            )
        if metadata.get("training_metrics_lineage") is not True:
            raise ValueError(
                "compact trainer checkpoint metrics lineage gate mismatch"
            )
        if metadata.get("training_metrics_lineage_status") != (
            training_metrics_lineage.get("training_metrics_lineage_status")
        ):
            raise ValueError(
                "compact trainer checkpoint metrics lineage status mismatch"
            )
    if not isinstance(getattr(checkpoint, "model_state_dict", None), Mapping):
        raise ValueError("compact trainer checkpoint missing model_state_dict")
    if not isinstance(getattr(checkpoint, "optimizer_state_dict", None), Mapping):
        raise ValueError("compact trainer checkpoint missing optimizer_state_dict")
    resume_state = getattr(checkpoint, "resume_state", None)
    if not isinstance(resume_state, CompactTrainerResumeStateV1):
        raise ValueError("compact trainer checkpoint missing resume_state")
    _validate_resume_state(resume_state)
    loop_runtime_state = getattr(checkpoint, "loop_runtime_state", None)
    if loop_runtime_state is not None and not isinstance(
        loop_runtime_state,
        CompactOwnedLoopRuntimeStateV1,
    ):
        raise ValueError("compact trainer checkpoint loop runtime state mismatch")
    _validate_replay_store_state(
        getattr(checkpoint, "replay_store_state", None),
        expected_policy_version_ref=resume_state.policy_version_ref,
        expected_model_version_ref=resume_state.model_version_ref,
    )


def capture_compact_owned_loop_runtime_state_v1(
    loop: Any,
) -> CompactOwnedLoopRuntimeStateV1:
    """Capture counters and previous-step state from ``CompactOwnedLoopV1``."""

    counters = CompactOwnedLoopCountersV1(
        sample_gate_calls=int(getattr(loop, "sample_gate_calls")),
        sample_gate_opportunities=int(getattr(loop, "sample_gate_opportunities")),
        sample_gate_skipped_count=int(getattr(loop, "sample_gate_skipped_count")),
        sample_gate_index_rows=int(getattr(loop, "sample_gate_index_rows")),
        sample_gate_target_rows=int(getattr(loop, "sample_gate_target_rows")),
        sample_gate_sample_rows=int(getattr(loop, "sample_gate_sample_rows")),
        sample_gate_sec=float(getattr(loop, "sample_gate_sec")),
        sample_gate_last_telemetry=_plain_mapping(
            getattr(loop, "sample_gate_last_telemetry", {}) or {}
        ),
        sample_gate_last_sample_metadata=_plain_mapping(
            getattr(loop, "sample_gate_last_sample_metadata", {}) or {}
        ),
        learner_gate_calls=int(getattr(loop, "learner_gate_calls")),
        learner_gate_updates=int(getattr(loop, "learner_gate_updates")),
        learner_gate_sample_rows=int(getattr(loop, "learner_gate_sample_rows")),
        learner_gate_input_bytes=int(getattr(loop, "learner_gate_input_bytes")),
        learner_gate_sec=float(getattr(loop, "learner_gate_sec")),
        learner_gate_last_telemetry=_plain_mapping(
            getattr(loop, "learner_gate_last_telemetry", {}) or {}
        ),
    )
    return CompactOwnedLoopRuntimeStateV1(
        previous_step=getattr(loop, "_previous_step", None),
        counters=counters,
    )


def restore_compact_owned_loop_runtime_state_v1(
    loop: Any,
    state: CompactOwnedLoopRuntimeStateV1,
) -> None:
    """Restore counters and previous-step state into ``CompactOwnedLoopV1``."""

    if not isinstance(state, CompactOwnedLoopRuntimeStateV1):
        raise ValueError("compact-owned loop runtime state schema mismatch")
    loop._previous_step = state.previous_step
    counters = state.counters
    loop.sample_gate_calls = int(counters.sample_gate_calls)
    loop.sample_gate_opportunities = int(counters.sample_gate_opportunities)
    loop.sample_gate_skipped_count = int(counters.sample_gate_skipped_count)
    loop.sample_gate_index_rows = int(counters.sample_gate_index_rows)
    loop.sample_gate_target_rows = int(counters.sample_gate_target_rows)
    loop.sample_gate_sample_rows = int(counters.sample_gate_sample_rows)
    loop.sample_gate_sec = float(counters.sample_gate_sec)
    loop.sample_gate_last_telemetry = dict(counters.sample_gate_last_telemetry)
    loop.sample_gate_last_sample_metadata = dict(
        counters.sample_gate_last_sample_metadata
    )
    loop.learner_gate_calls = int(counters.learner_gate_calls)
    loop.learner_gate_updates = int(counters.learner_gate_updates)
    loop.learner_gate_sample_rows = int(counters.learner_gate_sample_rows)
    loop.learner_gate_input_bytes = int(counters.learner_gate_input_bytes)
    loop.learner_gate_sec = float(counters.learner_gate_sec)
    loop.learner_gate_last_telemetry = dict(counters.learner_gate_last_telemetry)


def _checkpoint_compatibility_report(
    *,
    checkpoint_id: str,
    resume_state: CompactTrainerResumeStateV1,
    has_metrics: bool,
    reward_rnd_contract: Mapping[str, Any],
    death_terminal_contract: Mapping[str, Any],
    policy_refresh_handoff: Mapping[str, Any] | None,
    training_metrics_lineage: Mapping[str, Any] | None,
) -> Any:
    validate_compact_reward_rnd_contract_v1(reward_rnd_contract)
    validate_compact_death_terminal_contract_v1(death_terminal_contract)
    if policy_refresh_handoff is not None:
        validate_compact_policy_refresh_handoff_v1(policy_refresh_handoff)
    if training_metrics_lineage is not None:
        validate_compact_training_metrics_lineage_v1(training_metrics_lineage)
    metrics_present = bool(has_metrics)
    gates = {gate: False for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    gates.update(
        {
            "trainer_entrypoint": True,
            "checkpoint_save_load": True,
            "resume_metadata": True,
            "reward_rnd_contract": True,
            "death_terminal_contract": bool(
                death_terminal_contract[
                    "compact_death_terminal_contract_promotion_gate_satisfied"
                ]
            ),
            "policy_refresh_handoff": policy_refresh_handoff is not None,
            "training_metrics_lineage": training_metrics_lineage is not None,
        }
    )
    evidence = {
        "trainer_entrypoint": f"compact_owned_trainer:{resume_state.trainer_id}",
        "checkpoint_save_load": checkpoint_id,
        "resume_metadata": f"resume_state:{checkpoint_id}",
        "reward_rnd_contract": compact_reward_rnd_contract_evidence_ref(
            reward_rnd_contract
        ),
    }
    if bool(gates["death_terminal_contract"]):
        evidence["death_terminal_contract"] = compact_death_terminal_contract_evidence_ref(
            death_terminal_contract
        )
    if policy_refresh_handoff is not None:
        evidence["policy_refresh_handoff"] = (
            compact_policy_refresh_handoff_evidence_ref(policy_refresh_handoff)
        )
    if training_metrics_lineage is not None:
        evidence["training_metrics_lineage"] = (
            compact_training_metrics_lineage_evidence_ref(
                training_metrics_lineage
            )
        )
    if metrics_present:
        evidence["training_metrics_lineage_partial_metrics_present"] = (
            f"metrics_present:{checkpoint_id}"
        )
    return build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="compact_trainer_checkpoint_no_speed",
        gates=gates,
        evidence=evidence,
        promotion_claim=False,
    )


def _validate_policy_refresh_handoff_matches_checkpoint(
    policy_refresh_handoff: Mapping[str, Any],
    *,
    checkpoint_id: str,
    resume_state: CompactTrainerResumeStateV1,
) -> None:
    if str(policy_refresh_handoff.get("checkpoint_id", "")) != str(checkpoint_id):
        raise ValueError("policy refresh handoff checkpoint_id mismatch")
    for key in (
        "trainer_id",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "learner_update_count",
    ):
        if str(policy_refresh_handoff.get(key, "")) != str(
            getattr(resume_state, key)
        ):
            raise ValueError(f"policy refresh handoff {key} mismatch")


def _validate_policy_refresh_handoff_matches_metadata(
    policy_refresh_handoff: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> None:
    for key in (
        "checkpoint_id",
        "trainer_id",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "learner_update_count",
    ):
        if str(policy_refresh_handoff.get(key, "")) != str(metadata.get(key, "")):
            raise ValueError(f"policy refresh handoff metadata {key} mismatch")


def _resolve_normal_collision_death_evidence(
    *,
    checkpoint_id: str,
    normal_collision_death_evidence: Mapping[str, Any] | None,
    normal_collision_death_profile_result: Mapping[str, Any] | None,
    normal_collision_death_evidence_id: str,
    normal_collision_death_evidence_refs: list[str] | tuple[str, ...] | None,
) -> Mapping[str, Any] | None:
    if (
        normal_collision_death_evidence is not None
        and normal_collision_death_profile_result is not None
    ):
        raise ValueError(
            "provide either normal_collision_death_evidence or "
            "normal_collision_death_profile_result, not both"
        )
    if normal_collision_death_profile_result is None:
        return normal_collision_death_evidence
    evidence_id = str(normal_collision_death_evidence_id).strip() or (
        f"{checkpoint_id}:normal-death-profile"
    )
    refs = normal_collision_death_evidence_refs
    if not refs:
        refs = (f"checkpoint:{checkpoint_id}",)
    return build_normal_collision_death_evidence_from_profile_result_v1(
        normal_collision_death_profile_result,
        evidence_id=evidence_id,
        evidence_refs=tuple(str(ref) for ref in refs),
    )


def _validate_resume_state(resume_state: CompactTrainerResumeStateV1) -> None:
    if not str(resume_state.trainer_id).strip():
        raise ValueError("trainer_id must be non-empty")
    if int(resume_state.train_step) < 0:
        raise ValueError("train_step must be non-negative")
    if int(resume_state.learner_update_count) < 0:
        raise ValueError("learner_update_count must be non-negative")
    if int(resume_state.sample_batch_count) < 0:
        raise ValueError("sample_batch_count must be non-negative")
    if not str(resume_state.policy_version_ref).strip():
        raise ValueError("policy_version_ref must be non-empty")
    if not str(resume_state.model_version_ref).strip():
        raise ValueError("model_version_ref must be non-empty")
    if not str(resume_state.policy_source).strip():
        raise ValueError("policy_source must be non-empty")


def _validate_replay_store_state(
    replay_store_state: Any,
    *,
    expected_policy_version_ref: str,
    expected_model_version_ref: str,
) -> dict[str, Any]:
    if not isinstance(replay_store_state, CompactReplayStoreStateV1):
        raise ValueError("compact trainer checkpoint requires replay store state")
    metadata = dict(getattr(replay_store_state, "metadata", {}) or {})
    schema_id = metadata.get("schema_id") or metadata.get(
        "compact_replay_store_state_schema_id"
    )
    if schema_id != COMPACT_REPLAY_STORE_STATE_SCHEMA_ID:
        raise ValueError("compact trainer replay store state schema mismatch")
    policy_ref = str(
        metadata.get("policy_version_ref")
        or metadata.get("compact_replay_store_policy_version_ref")
        or ""
    )
    if policy_ref != str(expected_policy_version_ref):
        raise ValueError("compact trainer replay store policy lineage mismatch")
    model_ref = str(metadata.get("model_version_ref") or "")
    if model_ref != str(expected_model_version_ref):
        raise ValueError("compact trainer replay store model lineage mismatch")
    if (
        metadata.get("calls_train_muzero") is not False
        or metadata.get("touches_live_runs") is not False
    ):
        raise ValueError("compact trainer replay store non-claims mismatch")
    if metadata.get("compact_owned_loop_replay_store_owned") is not True:
        raise ValueError("compact trainer replay store must be owned by compact loop")
    if metadata.get("compact_owned_loop_policy_version_handoff") is not True:
        raise ValueError("compact trainer replay store missing policy handoff metadata")
    if not str(metadata.get("compact_owned_loop_schema_id") or "").strip():
        raise ValueError("compact trainer replay store missing compact-owned loop schema")
    metadata["compact_replay_store_state_schema_id"] = schema_id
    return metadata


def _validate_checkpoint_extra_metadata(extra_metadata: Mapping[str, Any]) -> None:
    protected_keys = sorted(
        key
        for key in extra_metadata
        if key in _PROTECTED_EXTRA_METADATA_KEYS
        or any(key.startswith(prefix) for prefix in _PROTECTED_EXTRA_METADATA_PREFIXES)
    )
    if protected_keys:
        raise ValueError(
            "compact trainer checkpoint extra_metadata cannot override protected "
            f"metadata keys: {protected_keys}"
        )


def _clone_state_dict(value: Any) -> Any:
    import torch

    if isinstance(value, Mapping):
        return {str(key): _clone_state_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_dict(item) for item in value)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _plain_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(value) for key, value in dict(metadata).items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "COMPACT_TRAINER_CHECKPOINT_KIND",
    "COMPACT_TRAINER_CHECKPOINT_SCHEMA_ID",
    "CompactOwnedLoopCountersV1",
    "CompactOwnedLoopRuntimeStateV1",
    "CompactTrainerCheckpointV1",
    "CompactTrainerResumeStateV1",
    "build_compact_trainer_checkpoint_v1",
    "capture_compact_owned_loop_runtime_state_v1",
    "load_compact_trainer_checkpoint_v1",
    "restore_compact_trainer_checkpoint_v1",
    "restore_compact_owned_loop_runtime_state_v1",
    "save_compact_trainer_checkpoint_v1",
    "validate_compact_trainer_checkpoint_v1",
]
