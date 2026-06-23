"""Minimal compact-owned trainer envelope.

This is not stock LightZero ``train_muzero``.  It gives the compact path a
trainer-shaped owner for learner updates, policy/model version lineage, and
checkpoint/resume state so later Coach compatibility gates have an artifact to
inspect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curvyzero.training.compact_trainer_checkpoint import (
    CompactTrainerCheckpointV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    CompactTrainerResumeStateV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    build_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    capture_compact_owned_loop_runtime_state_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    save_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_death_terminal_contract import (
    build_compact_death_terminal_contract_v1,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_normal_collision_death_evidence_from_profile_result_v1,
)
from curvyzero.training.compact_reward_rnd_contract import (
    build_compact_reward_rnd_contract_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    save_compact_stock_export_v1,
)
from curvyzero.training.compact_training_metrics_lineage import (
    build_compact_training_metrics_lineage_v1,
)


COMPACT_OWNED_TRAINER_SCHEMA_ID = "curvyzero_compact_owned_trainer/v1"


@dataclass(frozen=True, slots=True)
class CompactOwnedTrainerConfigV1:
    """Local trainer-envelope identity and non-claim knobs."""

    trainer_id: str
    policy_source: str
    initial_policy_version_ref: str
    initial_model_version_ref: str
    calls_train_muzero: bool = False
    touches_live_runs: bool = False
    promotion_claim: bool = False
    exploration_bonus_config: Any = None
    death_mode: str = "profile_no_death"
    normal_collision_death_evidence: Mapping[str, Any] | None = None
    normal_collision_death_profile_result: Mapping[str, Any] | None = None
    normal_collision_death_evidence_id: str = ""
    normal_collision_death_evidence_refs: tuple[str, ...] = ()
    allow_pending_normal_death_contract: bool = False


@dataclass(frozen=True, slots=True)
class CompactOwnedTrainerStepResultV1:
    """Result of one compact-owned learner update."""

    telemetry: dict[str, Any]
    policy_version_ref: str
    model_version_ref: str


class CompactOwnedTrainerV1:
    """Own compact learner-edge updates and checkpoint lineage."""

    schema_id = COMPACT_OWNED_TRAINER_SCHEMA_ID
    calls_train_muzero = False
    touches_live_runs = False

    def __init__(
        self,
        *,
        config: CompactOwnedTrainerConfigV1,
        learner: Any,
        loop: Any | None = None,
    ) -> None:
        _validate_config(config)
        self.config = config
        self.learner = learner
        self.loop = loop
        self.train_step = 0
        self.learner_update_count = 0
        self.sample_batch_count = 0
        self.record_step_calls = 0
        self.appended_replay_entry_count = 0
        self.sampled_count = 0
        self.trained_count = 0
        self.policy_refresh_count = 0
        self.checkpoint_save_count = 0
        self.resume_count = 0
        self.last_record_index: int | None = None
        self.policy_version_ref = str(config.initial_policy_version_ref)
        self.model_version_ref = str(config.initial_model_version_ref)
        self.last_learner_telemetry: dict[str, Any] = {}
        self._sync_loop_policy_version()

    @property
    def model(self) -> Any:
        return getattr(self.learner, "model")

    @property
    def optimizer(self) -> Any:
        return getattr(self.learner, "optimizer")

    @property
    def metadata(self) -> dict[str, Any]:
        reward_rnd_contract = build_compact_reward_rnd_contract_v1(
            exploration_bonus_config=self.config.exploration_bonus_config,
        )
        death_terminal_contract = build_compact_death_terminal_contract_v1(
            death_mode=self.config.death_mode,
            normal_collision_evidence=_normal_collision_death_evidence_from_config(self.config),
        )
        return {
            "schema_id": COMPACT_OWNED_TRAINER_SCHEMA_ID,
            "compact_owned_trainer_schema_id": COMPACT_OWNED_TRAINER_SCHEMA_ID,
            "compact_owned_trainer": True,
            "trainer_id": self.config.trainer_id,
            "policy_source": self.config.policy_source,
            "policy_version_ref": self.policy_version_ref,
            "model_version_ref": self.model_version_ref,
            "train_step": int(self.train_step),
            "learner_update_count": int(self.learner_update_count),
            "sample_batch_count": int(self.sample_batch_count),
            "record_step_calls": int(self.record_step_calls),
            "appended_replay_entry_count": int(self.appended_replay_entry_count),
            "sampled_count": int(self.sampled_count),
            "trained_count": int(self.trained_count),
            "policy_refresh_count": int(self.policy_refresh_count),
            "checkpoint_save_count": int(self.checkpoint_save_count),
            "resume_count": int(self.resume_count),
            "last_record_index": self.last_record_index,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "promotion_claim": False,
            "training_speed_claim": False,
            "stock_lightzero_integrated": False,
            "stock_eval_tournament_loadable": False,
            "compact_reward_rnd_contract": reward_rnd_contract,
            "compact_reward_rnd_contract_schema_id": reward_rnd_contract[
                "compact_reward_rnd_contract_schema_id"
            ],
            "compact_reward_rnd_contract_verified": True,
            "reward_rnd_contract": True,
            "reward_variant": reward_rnd_contract["reward_variant"],
            "reward_schema_id": reward_rnd_contract["reward_schema_id"],
            "reward_schema_hash": reward_rnd_contract["reward_schema_hash"],
            "exploration_bonus_mode": reward_rnd_contract["exploration_bonus_mode"],
            "exploration_bonus_enabled": False,
            "rnd_enabled": False,
            "compact_death_terminal_contract": death_terminal_contract,
            "compact_death_terminal_contract_schema_id": death_terminal_contract[
                "compact_death_terminal_contract_schema_id"
            ],
            "compact_death_terminal_contract_verified": True,
            "compact_death_terminal_contract_promotion_gate_satisfied": (
                death_terminal_contract["compact_death_terminal_contract_promotion_gate_satisfied"]
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
            "profile_no_death_supported": death_terminal_contract["profile_no_death_supported"],
            "max_ticks_terminal_supported": death_terminal_contract["max_ticks_terminal_supported"],
            "terminated_supported": death_terminal_contract["terminated_supported"],
            "truncated_supported": death_terminal_contract["truncated_supported"],
            "terminal_unroll_value_target_mode": death_terminal_contract[
                "terminal_unroll_value_target_mode"
            ],
            "policy_lineage_ref_propagated": True,
            "learner_updated_policy_consumed_by_search": False,
        }

    def record_step(
        self,
        *,
        current_step: Any,
        index_rows: Any | None,
    ) -> Any:
        """Record an actor/search step through the owned compact loop."""

        if self.loop is None:
            raise ValueError("compact-owned trainer record_step requires a loop")
        record_index = _record_index_from_rows(index_rows)
        if (
            record_index is not None
            and self.last_record_index is not None
            and int(record_index) <= int(self.last_record_index)
        ):
            raise ValueError("compact-owned trainer rejected stale record_index")
        previous_updates = int(getattr(self.loop, "learner_gate_updates", 0))
        result = self.loop.record_step(
            current_step=current_step,
            index_rows=index_rows,
        )
        self.record_step_calls += 1
        if getattr(result, "appended_replay_rows", False):
            self.appended_replay_entry_count += 1
        if getattr(result, "sampled", False):
            self.sampled_count += 1
            self.sample_batch_count += 1
        if getattr(result, "trained", False):
            self.trained_count += 1
            self.last_learner_telemetry = dict(
                getattr(self.loop, "learner_gate_last_telemetry", {}) or {}
            )
        self._record_loop_learner_progress(previous_updates)
        if record_index is not None:
            self.last_record_index = int(record_index)
        return result

    def consume_completed_learner_result(
        self,
        *,
        wait: bool = False,
    ) -> Any | None:
        """Drain deferred loop learner work while keeping trainer lineage in sync."""

        if self.loop is None:
            raise ValueError("compact-owned trainer consume requires a loop")
        consume = getattr(self.loop, "consume_completed_learner_result", None)
        if not callable(consume):
            raise ValueError("compact-owned trainer loop must expose consume_completed_learner_result")
        previous_updates = int(getattr(self.loop, "learner_gate_updates", 0))
        result = consume(wait=wait)
        if result is None:
            return None
        self.trained_count += int(result.get("compact_owned_loop_learner_result_aggregate_count", 1))
        self.last_learner_telemetry = dict(
            getattr(self.loop, "learner_gate_last_telemetry", {}) or {}
        )
        self._record_loop_learner_progress(previous_updates)
        return result

    def consume_completed_sample_learner_result(
        self,
        *,
        wait: bool = False,
    ) -> Any | None:
        """Drain staged replay sample + learner work through trainer ownership."""

        if self.loop is None:
            raise ValueError("compact-owned trainer staged consume requires a loop")
        consume = getattr(self.loop, "consume_completed_sample_learner_result", None)
        if not callable(consume):
            raise ValueError(
                "compact-owned trainer loop must expose consume_completed_sample_learner_result"
            )
        previous_updates = int(getattr(self.loop, "learner_gate_updates", 0))
        result = consume(wait=wait)
        if result is None:
            return None
        if getattr(result, "sampled", False):
            self.sampled_count += 1
            self.sample_batch_count += 1
        if getattr(result, "trained", False):
            self.trained_count += 1
            self.last_learner_telemetry = dict(
                getattr(self.loop, "learner_gate_last_telemetry", {}) or {}
            )
        self._record_loop_learner_progress(previous_updates)
        return result

    def train_on_sample_batch(
        self,
        sample_batch: Any,
        *,
        train_steps: int = 1,
    ) -> CompactOwnedTrainerStepResultV1:
        """Run the compact learner edge and advance local lineage refs."""

        train_steps_int = int(train_steps)
        if train_steps_int <= 0:
            raise ValueError("train_steps must be positive")
        result = self.learner.train_on_sample_batch(
            sample_batch,
            train_steps=train_steps_int,
        )
        return self._record_direct_learner_result(result, train_steps=train_steps_int)

    def train_on_learner_batch(
        self,
        learner_batch: Any,
        *,
        train_steps: int = 1,
    ) -> CompactOwnedTrainerStepResultV1:
        """Run the compact learner edge from an already-built learner batch."""

        train_steps_int = int(train_steps)
        if train_steps_int <= 0:
            raise ValueError("train_steps must be positive")
        train_on_learner_batch = getattr(self.learner, "train_on_learner_batch", None)
        if not callable(train_on_learner_batch):
            raise ValueError("compact learner does not support prebuilt learner batches")
        result = train_on_learner_batch(
            learner_batch,
            train_steps=train_steps_int,
        )
        return self._record_direct_learner_result(result, train_steps=train_steps_int)

    def _record_direct_learner_result(
        self,
        result: Any,
        *,
        train_steps: int,
    ) -> CompactOwnedTrainerStepResultV1:
        telemetry = _telemetry_from_result(result)
        if telemetry.get("compact_muzero_learner_calls_train_muzero") is not False:
            raise ValueError("compact-owned trainer learner must not call train_muzero")
        self.train_step += 1
        self.learner_update_count += int(
            telemetry.get("compact_muzero_learner_train_steps", int(train_steps))
        )
        self.sample_batch_count += 1
        self.policy_refresh_count += 1
        self._advance_policy_refs()
        telemetry.update(self.metadata)
        telemetry.update(
            {
                "compact_owned_trainer_update_claim": True,
                "compact_owned_trainer_checkpoint_required": True,
                "compact_owned_trainer_policy_refresh_consumed_by_search": False,
            }
        )
        self.last_learner_telemetry = dict(telemetry)
        return CompactOwnedTrainerStepResultV1(
            telemetry=telemetry,
            policy_version_ref=self.policy_version_ref,
            model_version_ref=self.model_version_ref,
        )

    def resume_state(
        self,
        *,
        loop_counters: Mapping[str, Any] | None = None,
    ) -> CompactTrainerResumeStateV1:
        return CompactTrainerResumeStateV1(
            trainer_id=self.config.trainer_id,
            train_step=int(self.train_step),
            learner_update_count=int(self.learner_update_count),
            sample_batch_count=int(self.sample_batch_count),
            policy_version_ref=self.policy_version_ref,
            model_version_ref=self.model_version_ref,
            policy_source=self.config.policy_source,
            loop_counters={str(key): value for key, value in dict(loop_counters or {}).items()},
        )

    def checkpoint(
        self,
        *,
        checkpoint_id: str,
        replay_store_state: Any,
        metrics: Mapping[str, Any] | None = None,
        policy_refresh_handoff: Mapping[str, Any] | None = None,
        training_metrics_lineage: Mapping[str, Any] | None = None,
        training_metrics_lineage_evidence_refs: tuple[str, ...] = (),
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> CompactTrainerCheckpointV1:
        metrics_dict = dict(metrics or {})
        if self.last_learner_telemetry:
            metrics_dict.setdefault("last_learner_telemetry", self.last_learner_telemetry)
        self.checkpoint_save_count += 1
        loop_runtime_state = (
            None if self.loop is None else capture_compact_owned_loop_runtime_state_v1(self.loop)
        )
        resume_state = self.resume_state(
            loop_counters={
                "last_learner_train_step": self.train_step,
                "last_learner_update_count": self.learner_update_count,
            }
        )
        metrics_lineage_contract = training_metrics_lineage
        if metrics_lineage_contract is None and training_metrics_lineage_evidence_refs:
            metrics_lineage_contract = build_compact_training_metrics_lineage_v1(
                checkpoint_id=checkpoint_id,
                resume_state=resume_state,
                replay_store_state=replay_store_state,
                metrics=metrics_dict,
                loop_runtime_state=loop_runtime_state,
                evidence_refs=training_metrics_lineage_evidence_refs,
            )
        return build_compact_trainer_checkpoint_v1(
            checkpoint_id=checkpoint_id,
            trainer_config=self.metadata,
            resume_state=resume_state,
            model=self.model,
            optimizer=self.optimizer,
            replay_store_state=replay_store_state,
            metrics=metrics_dict,
            loop_runtime_state=loop_runtime_state,
            exploration_bonus_config=self.config.exploration_bonus_config,
            death_mode=self.config.death_mode,
            normal_collision_death_evidence=(self.config.normal_collision_death_evidence),
            normal_collision_death_profile_result=(
                self.config.normal_collision_death_profile_result
            ),
            normal_collision_death_evidence_id=(self.config.normal_collision_death_evidence_id),
            normal_collision_death_evidence_refs=(self.config.normal_collision_death_evidence_refs),
            policy_refresh_handoff=policy_refresh_handoff,
            training_metrics_lineage=metrics_lineage_contract,
            extra_metadata=extra_metadata,
        )

    def save_checkpoint(
        self,
        *,
        checkpoint_id: str,
        replay_store_state: Any,
        path: str | Path,
        metrics: Mapping[str, Any] | None = None,
        policy_refresh_handoff: Mapping[str, Any] | None = None,
        training_metrics_lineage: Mapping[str, Any] | None = None,
        training_metrics_lineage_evidence_refs: tuple[str, ...] = (),
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        checkpoint = self.checkpoint(
            checkpoint_id=checkpoint_id,
            replay_store_state=replay_store_state,
            metrics=metrics,
            policy_refresh_handoff=policy_refresh_handoff,
            training_metrics_lineage=training_metrics_lineage,
            training_metrics_lineage_evidence_refs=(training_metrics_lineage_evidence_refs),
            extra_metadata=extra_metadata,
        )
        return save_compact_trainer_checkpoint_v1(checkpoint, path)

    def save_stock_eval_export(
        self,
        *,
        checkpoint_id: str,
        replay_store_state: Any,
        path: str | Path,
        policy_metadata: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
        checkpoint_extra_metadata: Mapping[str, Any] | None = None,
        export_extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save an eval-only stock-shaped export derived from a compact checkpoint."""

        checkpoint = self.checkpoint(
            checkpoint_id=checkpoint_id,
            replay_store_state=replay_store_state,
            metrics=metrics,
            extra_metadata=checkpoint_extra_metadata,
        )
        return save_compact_stock_export_v1(
            checkpoint,
            path,
            policy_metadata=policy_metadata,
            extra_metadata=export_extra_metadata,
        )

    def _advance_policy_refs(self) -> None:
        self.policy_version_ref = _version_ref(
            self.config.initial_policy_version_ref,
            self.learner_update_count,
        )
        self.model_version_ref = _version_ref(
            self.config.initial_model_version_ref,
            self.learner_update_count,
        )
        self._sync_loop_policy_version()

    def _record_loop_learner_progress(self, previous_updates: int) -> int:
        current_updates = int(getattr(self.loop, "learner_gate_updates", 0))
        delta = max(0, current_updates - int(previous_updates))
        if delta:
            self.train_step += 1
            self.learner_update_count += delta
            self.policy_refresh_count += 1
            self._advance_policy_refs()
        return delta

    def update_policy_version(self, policy_version: CompactPolicyVersionRefV1) -> None:
        """Adopt an externally refreshed policy/model ref."""

        self.policy_version_ref = str(policy_version.policy_version_ref)
        self.model_version_ref = str(policy_version.model_version_ref)
        self._sync_loop_policy_version()

    def _sync_loop_policy_version(self) -> None:
        if self.loop is None:
            return
        self.loop.policy_version = CompactPolicyVersionRefV1(
            policy_version_ref=self.policy_version_ref,
            policy_source=self.config.policy_source,
            model_version_ref=self.model_version_ref,
        )


def _validate_config(config: CompactOwnedTrainerConfigV1) -> None:
    if not str(config.trainer_id).strip():
        raise ValueError("trainer_id must be non-empty")
    if not str(config.policy_source).strip():
        raise ValueError("policy_source must be non-empty")
    if not str(config.initial_policy_version_ref).strip():
        raise ValueError("initial_policy_version_ref must be non-empty")
    if not str(config.initial_model_version_ref).strip():
        raise ValueError("initial_model_version_ref must be non-empty")
    if bool(config.calls_train_muzero):
        raise ValueError("compact-owned trainer does not call train_muzero")
    if bool(config.touches_live_runs):
        raise ValueError("compact-owned trainer must not touch live runs")
    if bool(config.promotion_claim):
        raise ValueError("compact-owned trainer cannot claim promotion")
    build_compact_reward_rnd_contract_v1(
        exploration_bonus_config=config.exploration_bonus_config,
    )
    if (
        str(config.death_mode) == "normal"
        and bool(config.allow_pending_normal_death_contract)
        and config.normal_collision_death_evidence is None
        and config.normal_collision_death_profile_result is None
    ):
        return
    build_compact_death_terminal_contract_v1(
        death_mode=config.death_mode,
        normal_collision_evidence=_normal_collision_death_evidence_from_config(config),
    )


def _normal_collision_death_evidence_from_config(
    config: CompactOwnedTrainerConfigV1,
) -> Mapping[str, Any] | None:
    if (
        config.normal_collision_death_evidence is not None
        and config.normal_collision_death_profile_result is not None
    ):
        raise ValueError(
            "provide either normal_collision_death_evidence or "
            "normal_collision_death_profile_result, not both"
        )
    if config.normal_collision_death_profile_result is None:
        return config.normal_collision_death_evidence
    evidence_id = str(config.normal_collision_death_evidence_id).strip() or (
        f"{config.trainer_id}:normal-death-profile"
    )
    refs = config.normal_collision_death_evidence_refs or (f"trainer:{config.trainer_id}",)
    return build_normal_collision_death_evidence_from_profile_result_v1(
        config.normal_collision_death_profile_result,
        evidence_id=evidence_id,
        evidence_refs=tuple(str(ref) for ref in refs),
    )


def _telemetry_from_result(result: Any) -> dict[str, Any]:
    telemetry = getattr(result, "telemetry", result)
    if not isinstance(telemetry, Mapping):
        raise ValueError("compact learner result telemetry must be a mapping")
    return {str(key): value for key, value in telemetry.items()}


def _version_ref(base: str, update_count: int) -> str:
    return f"{str(base)}:update-{int(update_count)}"


def _record_index_from_rows(index_rows: Any | None) -> int | None:
    if index_rows is None or not hasattr(index_rows, "record_index"):
        return None
    return int(getattr(index_rows, "record_index"))


__all__ = [
    "COMPACT_OWNED_TRAINER_SCHEMA_ID",
    "CompactOwnedTrainerConfigV1",
    "CompactOwnedTrainerStepResultV1",
    "CompactOwnedTrainerV1",
]
