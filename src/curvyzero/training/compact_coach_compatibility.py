"""Fail-closed Coach compatibility attestation for compact-owned candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_SPEED_ROW_EVIDENCE_REF_PREFIX,
)
from curvyzero.training.compact_coach_speed_row import (
    CompactCoachSpeedRowEvidenceError,
)
from curvyzero.training.compact_coach_speed_row import (
    validate_compact_coach_speed_row_evidence_matches_report_v1,
)


COMPACT_COACH_COMPATIBILITY_SCHEMA_ID = "curvyzero_compact_coach_compatibility/v1"

COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE = "stock_train_muzero_bridge"
COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER = "compact_owned_trainer"
COMPACT_COACH_ROUTES = (
    COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE,
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
)

COMPACT_COACH_LIFECYCLE_GATES = (
    "trainer_entrypoint",
    "checkpoint_save_load",
    "resume_metadata",
    "eval_gif_tournament_load",
    "reward_rnd_contract",
    "death_terminal_contract",
    "policy_refresh_handoff",
    "training_metrics_lineage",
)
COMPACT_COACH_GATE_COACH_SPEED_ROW = "coach_speed_row"
COMPACT_COACH_REQUIRED_PROMOTION_GATES = (
    *COMPACT_COACH_LIFECYCLE_GATES,
    COMPACT_COACH_GATE_COACH_SPEED_ROW,
)

COMPACT_COACH_SUPPORT_GATES = (
    "matched_denominator",
    "split_profile_loop_entrypoint",
    "durable_replay_store_snapshot",
    "public_stock_sample_diff",
    "terminal_nstep_targets",
    "compact_muzero_learner_edge",
)
COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_REF_PREFIX = (
    "compact_current_chain_eval_gif_tournament_load:"
)
COMPACT_COACH_PROMOTION_READINESS_BLOCKER = (
    "post_compatibility_promotion_readiness_required"
)


class CompactCoachCompatibilityError(ValueError):
    """Raised when a compact candidate overclaims Coach compatibility."""


@dataclass(frozen=True, slots=True)
class CompactCoachCompatibilityReportV1:
    """Structured promotion-readiness report for compact training candidates."""

    route: str
    profile_only: bool
    calls_train_muzero: bool
    touches_live_runs: bool
    speed_currency: str
    gates: Mapping[str, bool]
    evidence: Mapping[str, str]
    coach_speed_row_evidence: Mapping[str, Any] | None = None
    promotion_claim: bool = False

    def __post_init__(self) -> None:
        if bool(self.promotion_claim):
            raise CompactCoachCompatibilityError(
                "promotion_claim requires post-compatibility promotion-readiness "
                "evidence; construct reports through "
                "build_compact_coach_compatibility_report_v1"
            )

    @property
    def required_gates(self) -> tuple[str, ...]:
        return COMPACT_COACH_REQUIRED_PROMOTION_GATES

    @property
    def missing_required_gates(self) -> tuple[str, ...]:
        return tuple(
            gate for gate in self.required_gates if not bool(self.gates.get(gate))
        )

    @property
    def missing_required_evidence(self) -> tuple[str, ...]:
        return tuple(
            gate
            for gate in self.required_gates
            if not str(self.evidence.get(gate, "")).strip()
        )

    @property
    def promotion_eligible(self) -> bool:
        return (
            not bool(self.profile_only)
            and not self.missing_required_gates
            and not self.missing_required_evidence
            and _route_has_trainer_entrypoint(self)
        )

    @property
    def promotion_blocker(self) -> str:
        if self.promotion_eligible:
            return ""
        if bool(self.profile_only):
            return "profile_only_candidate"
        if not _route_has_trainer_entrypoint(self):
            return "missing_trainer_entrypoint"
        if self.missing_required_gates:
            return "missing_required_gates"
        if self.missing_required_evidence:
            return "missing_required_evidence"
        return "not_promotion_eligible"

    def as_metadata(self) -> dict[str, Any]:
        gates = {str(key): bool(value) for key, value in self.gates.items()}
        evidence = {str(key): str(value) for key, value in self.evidence.items()}
        metadata: dict[str, Any] = {
            "compact_coach_compatibility_schema_id": (
                COMPACT_COACH_COMPATIBILITY_SCHEMA_ID
            ),
            "compact_coach_compatibility_route": str(self.route),
            "compact_coach_compatibility_profile_only": bool(self.profile_only),
            "compact_coach_compatibility_calls_train_muzero": bool(
                self.calls_train_muzero
            ),
            "compact_coach_compatibility_touches_live_runs": bool(
                self.touches_live_runs
            ),
            "compact_coach_compatibility_speed_currency": str(self.speed_currency),
            "compact_coach_compatibility_promotion_claim": bool(
                self.promotion_claim
            ),
            "compact_coach_compatibility_promotion_eligible": bool(
                self.promotion_eligible
            ),
            "compact_coach_compatibility_promotion_blocker": (
                self.promotion_blocker
            ),
            "compact_coach_compatibility_required_gates": list(
                COMPACT_COACH_REQUIRED_PROMOTION_GATES
            ),
            "compact_coach_compatibility_support_gates": list(
                COMPACT_COACH_SUPPORT_GATES
            ),
            "compact_coach_compatibility_missing_required_gates": list(
                self.missing_required_gates
            ),
            "compact_coach_compatibility_missing_required_evidence": list(
                self.missing_required_evidence
            ),
            "compact_coach_compatibility_evidence": evidence,
        }
        for gate, passed in gates.items():
            metadata[f"compact_coach_compatibility_gate_{gate}"] = bool(passed)
        return metadata


def build_compact_coach_compatibility_report_v1(
    *,
    route: str,
    profile_only: bool,
    calls_train_muzero: bool,
    touches_live_runs: bool,
    speed_currency: str,
    gates: Mapping[str, bool],
    evidence: Mapping[str, str] | None = None,
    coach_speed_row_evidence: Mapping[str, Any] | None = None,
    promotion_claim: bool = False,
) -> CompactCoachCompatibilityReportV1:
    """Build and validate a Coach-compatibility report."""

    route_value = str(route)
    if route_value not in COMPACT_COACH_ROUTES:
        raise CompactCoachCompatibilityError(
            f"unknown compact Coach compatibility route: {route_value}"
        )
    speed_currency_value = str(speed_currency).strip()
    if not speed_currency_value:
        raise CompactCoachCompatibilityError("speed_currency must be non-empty")
    if (
        bool(calls_train_muzero)
        and route_value != COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE
    ):
        raise CompactCoachCompatibilityError(
            "calls_train_muzero=true requires stock_train_muzero_bridge route"
        )
    report = CompactCoachCompatibilityReportV1(
        route=route_value,
        profile_only=bool(profile_only),
        calls_train_muzero=bool(calls_train_muzero),
        touches_live_runs=bool(touches_live_runs),
        speed_currency=speed_currency_value,
        gates={str(key): bool(value) for key, value in dict(gates).items()},
        evidence={str(key): str(value) for key, value in dict(evidence or {}).items()},
        coach_speed_row_evidence=coach_speed_row_evidence,
        promotion_claim=False,
    )
    _validate_required_gate_evidence_semantics(report)
    if bool(promotion_claim) and not report.promotion_eligible:
        raise CompactCoachCompatibilityError(
            "promotion_claim requires non-profile trainer entrypoint and all "
            f"required gates; blocker={report.promotion_blocker}; "
            f"missing_gates={list(report.missing_required_gates)}; "
            f"missing_evidence={list(report.missing_required_evidence)}"
        )
    if bool(promotion_claim):
        raise CompactCoachCompatibilityError(
            "promotion_claim requires post-compatibility promotion-readiness "
            "evidence; local compact Coach compatibility eligibility is not "
            f"a promotion claim; blocker={COMPACT_COACH_PROMOTION_READINESS_BLOCKER}"
        )
    return report


def _validate_required_gate_evidence_semantics(
    report: CompactCoachCompatibilityReportV1,
) -> None:
    if bool(report.gates.get("eval_gif_tournament_load")):
        evidence_ref = str(report.evidence.get("eval_gif_tournament_load", "")).strip()
        if evidence_ref and not evidence_ref.startswith(
            COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_REF_PREFIX
        ):
            raise CompactCoachCompatibilityError(
                "eval_gif_tournament_load requires current-chain eval/GIF/tournament "
                "evidence ref"
            )
    if bool(report.gates.get(COMPACT_COACH_GATE_COACH_SPEED_ROW)):
        evidence_ref = str(
            report.evidence.get(COMPACT_COACH_GATE_COACH_SPEED_ROW, "")
        ).strip()
        if not evidence_ref:
            raise CompactCoachCompatibilityError(
                "coach_speed_row requires compact Coach speed-row evidence ref"
            )
        if not evidence_ref.startswith(
            COMPACT_COACH_SPEED_ROW_EVIDENCE_REF_PREFIX
        ):
            raise CompactCoachCompatibilityError(
                "coach_speed_row requires compact Coach speed-row evidence ref"
            )
        if report.coach_speed_row_evidence is None:
            raise CompactCoachCompatibilityError(
                "coach_speed_row requires structured compact Coach speed-row evidence"
            )
        try:
            validate_compact_coach_speed_row_evidence_matches_report_v1(
                report.coach_speed_row_evidence,
                evidence_ref=evidence_ref,
                route=report.route,
                speed_currency=report.speed_currency,
            )
        except CompactCoachSpeedRowEvidenceError as exc:
            raise CompactCoachCompatibilityError(
                f"coach_speed_row evidence invalid: {exc}"
            ) from exc


def build_profile_only_compact_coach_report_v1(
    *,
    speed_currency: str,
    evidence: Mapping[str, str] | None = None,
    promotion_claim: bool = False,
) -> CompactCoachCompatibilityReportV1:
    """Current compact-owned profile attestation.

    This intentionally records useful support gates while keeping every
    Coach-promotion gate closed.
    """

    gates = {
        "matched_denominator": True,
        "split_profile_loop_entrypoint": True,
        "durable_replay_store_snapshot": True,
        "public_stock_sample_diff": True,
        "terminal_nstep_targets": True,
        "compact_muzero_learner_edge": True,
        "trainer_entrypoint": False,
        "checkpoint_save_load": False,
        "resume_metadata": False,
        "eval_gif_tournament_load": False,
        "reward_rnd_contract": False,
        "death_terminal_contract": False,
        "policy_refresh_handoff": False,
        "training_metrics_lineage": False,
        COMPACT_COACH_GATE_COACH_SPEED_ROW: False,
    }
    default_evidence = {
        "matched_denominator": "curvytron-stock-vs-compact-owned-no-rnd-h100-20260528",
        "split_profile_loop_entrypoint": "curvyzero_compact_owned_loop/v1",
        "durable_replay_store_snapshot": "curvyzero_compact_replay_store_state/v1",
        "public_stock_sample_diff": "test_compact_multi_record_sample_batch_matches_stock_muzero_public_sample_for_terminal_nstep",
        "terminal_nstep_targets": "optimizer-compact-owned-terminal-nstep-next-20260528d",
        "compact_muzero_learner_edge": "curvyzero_compact_muzero_learner_edge/v1",
    }
    default_evidence.update(
        {str(key): str(value) for key, value in dict(evidence or {}).items()}
    )
    return build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=True,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency=speed_currency,
        gates=gates,
        evidence=default_evidence,
        promotion_claim=promotion_claim,
    )


def _route_has_trainer_entrypoint(
    report: CompactCoachCompatibilityReportV1,
) -> bool:
    if report.route == COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE:
        return bool(report.calls_train_muzero) and bool(
            report.gates.get("trainer_entrypoint")
        )
    if report.route == COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER:
        return bool(report.gates.get("trainer_entrypoint"))
    return False


__all__ = [
    "COMPACT_COACH_COMPATIBILITY_SCHEMA_ID",
    "COMPACT_COACH_GATE_COACH_SPEED_ROW",
    "COMPACT_COACH_LIFECYCLE_GATES",
    "COMPACT_COACH_REQUIRED_PROMOTION_GATES",
    "COMPACT_COACH_SPEED_ROW_EVIDENCE_REF_PREFIX",
    "COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_REF_PREFIX",
    "COMPACT_COACH_PROMOTION_READINESS_BLOCKER",
    "COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER",
    "COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE",
    "COMPACT_COACH_ROUTES",
    "COMPACT_COACH_SUPPORT_GATES",
    "CompactCoachCompatibilityError",
    "CompactCoachCompatibilityReportV1",
    "build_compact_coach_compatibility_report_v1",
    "build_profile_only_compact_coach_report_v1",
]
