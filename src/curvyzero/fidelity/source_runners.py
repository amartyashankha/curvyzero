"""Narrow CurvyTron source-fidelity scenario runners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from curvyzero.env.config import CurvyTronReferenceDefaults
from curvyzero.env.source_env import CurvyTronSourceEnv
from curvyzero.env.scenario_schema import (
    PYTHON_SCENARIO_TRACE_SCHEMA,
    LoadedScenario,
    ScenarioError,
    _coerce_scenario,
    _extract_alive,
    _extract_headings,
    _extract_positions,
    _scenario_player_ids,
)
from curvyzero.env.tracing import TRACE_SCHEMA_VERSION

SOURCE_KINEMATICS_RUNNER = "curvytron-v1-python-source-kinematics-runner"
SOURCE_KINEMATICS_TRACE_SCOPE = "curvytron-v1-python-source-kinematics"
SOURCE_KINEMATICS_RULES_HASH = "source-kinematics-v1"
SOURCE_KINEMATICS_MESSAGE = (
    "Python source kinematics for simple forced movement scenarios; "
    "collisions, trails, bonuses, and full game rules are not implemented."
)
SOURCE_KINEMATICS_FIDELITY_SCOPE = "movement kinematics only"
SOURCE_NORMAL_WALL_RUNNER = "curvytron-v1-python-source-normal-wall-runner"
SOURCE_NORMAL_WALL_TRACE_SCOPE = "curvytron-v1-python-source-normal-wall"
SOURCE_NORMAL_WALL_RULES_HASH = "source-normal-wall-v1"
SOURCE_NORMAL_WALL_MESSAGE = (
    "Python source movement plus normal-wall death state/events for forced wall scenarios; "
    "body collisions, trails, bonuses, and full game rules are not implemented."
)
SOURCE_NORMAL_WALL_FIDELITY_SCOPE = "movement plus normal-wall death state/events only"
SOURCE_BORDERLESS_WRAP_RUNNER = "curvytron-v1-python-source-borderless-wrap-runner"
SOURCE_BORDERLESS_WRAP_TRACE_SCOPE = "curvytron-v1-python-source-borderless-wrap"
SOURCE_BORDERLESS_WRAP_RULES_HASH = "source-borderless-wrap-v1"
SOURCE_BORDERLESS_WRAP_MESSAGE = (
    "Python source movement plus borderless wrap state/events for forced borderless scenarios; "
    "includes exact-edge/corner-axis and the first destination-body skip canaries; "
    "bonuses, broader trail cases, and full game rules are not implemented."
)
SOURCE_BORDERLESS_WRAP_FIDELITY_SCOPE = (
    "movement plus source borderless wrap, exact-edge/corner-axis, "
    "and first destination-body skip state/events only"
)
SOURCE_BODY_CANARY_RUNNER = "curvytron-v1-python-source-body-canary-runner"
SOURCE_BODY_CANARY_TRACE_SCOPE = "curvytron-v1-python-source-body-canary"
SOURCE_BODY_CANARY_RULES_HASH = "source-body-canary-v2"
SOURCE_BODY_CANARY_MESSAGE = (
    "Python source movement plus seeded body canary collisions for the opponent "
    "strict-overlap, own-latency, same-frame point, old-body metadata source_body_* "
    "scenarios, and the source collision-order death-point and head-head canaries; "
    "print-manager holes, bonuses, and full game rules are not implemented."
)
SOURCE_BODY_CANARY_FIDELITY_SCOPE = (
    "movement plus seeded opponent, own-latency, same-frame point, old-body metadata, "
    "and collision-order body canary state/events only"
)
SOURCE_PRINT_MANAGER_RUNNER = "curvytron-v1-python-source-print-manager-runner"
SOURCE_PRINT_MANAGER_TRACE_SCOPE = "curvytron-v1-python-source-print-manager"
SOURCE_PRINT_MANAGER_RULES_HASH = "source-print-manager-v8"
SOURCE_PRINT_MANAGER_MESSAGE = (
    "Python source movement plus deterministic PrintManager toggle/exact-zero, "
    "delayed start, random-tape call order/cadence, wall-death stop, and seeded "
    "body-collision death-stop canaries; broader trail cadence, bonuses, and "
    "full game rules are not implemented."
)
SOURCE_PRINT_MANAGER_FIDELITY_SCOPE = (
    "movement plus deterministic PrintManager toggle/exact-zero, delayed start, "
    "random-tape call order/cadence, and wall/body death-stop state/events only"
)
SOURCE_TRAIL_CADENCE_RUNNER = "curvytron-v1-python-source-trail-cadence-runner"
SOURCE_TRAIL_CADENCE_TRACE_SCOPE = "curvytron-v1-python-source-trail-cadence"
SOURCE_TRAIL_CADENCE_RULES_HASH = "source-trail-cadence-v1"
SOURCE_TRAIL_CADENCE_MESSAGE = (
    "Python source movement plus normal trail cadence canaries; "
    "trail gaps, broader collisions, bonuses, and full game rules are not implemented."
)
SOURCE_TRAIL_CADENCE_FIDELITY_SCOPE = (
    "movement plus normal trail point cadence state/events only"
)
SOURCE_TRAIL_GAP_RUNNER = "curvytron-v1-python-source-trail-gap-runner"
SOURCE_TRAIL_GAP_TRACE_SCOPE = "curvytron-v1-python-source-trail-gap"
SOURCE_TRAIL_GAP_RULES_HASH = "source-trail-gap-v2"
SOURCE_TRAIL_GAP_MESSAGE = (
    "Python source movement plus forced trail-gap body absence/collision canaries "
    "and one natural multi-step PrintManager hole-crossing canary; "
    "broader gap transitions, bonuses, and full game rules are not implemented."
)
SOURCE_TRAIL_GAP_FIDELITY_SCOPE = (
    "movement plus forced trail-gap body absence/collision and one natural "
    "multi-step hole-crossing state/events only"
)
SOURCE_LIFECYCLE_RUNNER = "curvytron-v1-python-source-lifecycle-runner"
SOURCE_LIFECYCLE_TRACE_SCOPE = "curvytron-v1-python-source-lifecycle"
SOURCE_LIFECYCLE_TRACE_SCHEMA = "environment-lifecycle-trace-v0"
SOURCE_LIFECYCLE_RULES_HASH = "source-lifecycle-v25"
SOURCE_LIFECYCLE_MESSAGE = (
    "Python source-project lifecycle runner for pinned source_lifecycle_* "
    "fixtures only; covers pinned 2P/3P/4P natural spawn RNG/order, heading "
    "rejection retry, 3P/4P present/absent round-new, survivor scoring, and "
    "next-round state, 2P/3P delayed PrintManager start, 2P "
    "warmdown/next-round facts, one focused 3P "
    "forced wall-death warmdown/next-round fixture, one focused 3P survivor "
    "warmdown/next-round fixture, one focused 3P survivor-scoring round-end "
    "fixture, one focused 3P warmdown removeAvatar/next-round fixture, "
    "one focused 3P dead-then-removeAvatar round-end/next-round fixture, "
    "focused 3P/4P present/absent survivor-scoring round-end and "
    "tie-at-max-score continuation "
    "fixtures, one 2P max-score match-end fact, one 3P max-score match-end "
    "fact, one 4P max-score match-end fact, "
    "one 3P tie-at-max-score continuation fact, one 4P tie-at-max-score "
    "continuation fact, one focused 3P all-present multi-round match-end "
    "fact, one focused 4P all-present multi-round match-end fact, one "
    "focused 4P all-present "
    "all-dead warmdown/next-round fact, and one focused 4P survivor "
    "warmdown/next-round fact."
)
SOURCE_LIFECYCLE_FIDELITY_SCOPE = (
    "source lifecycle event/random/snapshot facts for pinned 2P/3P/4P fixtures, "
    "including one focused 3P all-dead warmdown/next-round fixture, one focused "
    "3P survivor warmdown/next-round fixture, one focused 3P survivor-scoring "
    "round-end fixture, one focused 3P warmdown removeAvatar/next-round "
    "fixture, one focused 3P dead-then-removeAvatar round-end/next-round "
    "fixture, focused 3P/4P present/absent round-new, "
    "survivor-scoring round-end, tie-at-max-score continuation, and "
    "next-round fixtures, one focused 4P "
    "all-present all-dead warmdown/next-round fixture, one focused 4P "
    "survivor warmdown/next-round fixture, "
    "one 2P max-score/end fixture, one 3P max-score/end fixture, one 4P "
    "max-score/end fixture, one 3P "
    "tie-at-max-score continuation fixture, one 4P tie-at-max-score "
    "continuation fixture, plus focused 3P and 4P all-present multi-round "
    "match-end fixtures only"
)
SOURCE_BODY_CANARY_SCENARIO_IDS = frozenset(
    {
        "source_body_opponent_tangent_safe_step",
        "source_body_opponent_overlap_kills_step",
        "source_body_own_delta3_safe_step",
        "source_body_own_delta4_kills_step",
        "source_body_same_frame_point_kills_step",
        "source_body_same_frame_point_control_safe_step",
        "source_body_old_opponent_overlap_kills_step",
        "source_collision_death_point_kills_later_player_step",
        "source_collision_head_head_reverse_order_single_death_step",
    }
)
SOURCE_PRINT_MANAGER_SCENARIO_IDS = frozenset(
    {
        "source_print_manager_print_to_hole_step",
        "source_print_manager_hole_to_print_step",
        "source_print_manager_exact_zero_toggle_step",
        "source_print_manager_no_toggle_control_step",
        "source_print_manager_active_stop_on_death_step",
        "source_print_manager_active_hole_stop_on_death_step",
        "source_print_manager_body_collision_stop_on_death_step",
        "source_print_manager_delayed_start_timer_step",
        "source_print_manager_random_call_order_step",
        "source_print_manager_random_cadence_multistep",
    }
)
SOURCE_TRAIL_CADENCE_SCENARIO_IDS = frozenset(
    {
        "source_trail_normal_point_step",
        "source_trail_no_point_below_radius_step",
    }
)
SOURCE_TRAIL_GAP_SCENARIO_IDS = frozenset(
    {
        "source_trail_gap_hole_space_safe_step",
        "source_trail_gap_stored_body_still_kills_step",
        "source_trail_gap_print_to_hole_boundary_kills_step",
        "source_trail_gap_hole_to_print_boundary_kills_step",
        "source_trail_gap_natural_multistep_hole_crossing",
    }
)
SOURCE_LIFECYCLE_SCENARIO_IDS = frozenset(
    {
        "source_lifecycle_spawn_rng_warmup_print_start_2p",
        "source_lifecycle_spawn_rng_2p_next_round",
        "source_lifecycle_spawn_heading_rejection_retry_2p",
        "source_lifecycle_spawn_rng_order_3p",
        "source_lifecycle_spawn_rng_warmup_print_start_3p",
        "source_lifecycle_spawn_rng_3p_next_round",
        "source_lifecycle_survivor_score_3p_round_end",
        "source_lifecycle_survivor_score_3p_next_round",
        "source_lifecycle_remove_avatar_during_warmdown_3p",
        "source_lifecycle_remove_avatar_to_single_present_3p",
        "source_lifecycle_spawn_rng_order_4p",
        "source_lifecycle_spawn_rng_4p_next_round",
        "source_lifecycle_survivor_score_4p_next_round",
        "source_lifecycle_present_absent_3p_round_new",
        "source_lifecycle_present_absent_3p_survivor_score_round_end",
        "source_lifecycle_present_absent_3p_next_round",
        "source_lifecycle_present_absent_3p_tie_at_max_score",
        "source_lifecycle_present_absent_4p_round_new",
        "source_lifecycle_present_absent_4p_survivor_score_round_end",
        "source_lifecycle_present_absent_4p_next_round",
        "source_lifecycle_present_absent_4p_tie_at_max_score",
        "source_lifecycle_match_end_at_max_score_2p",
        "source_lifecycle_match_end_at_max_score_3p",
        "source_lifecycle_match_end_at_max_score_4p",
        "source_lifecycle_tie_at_max_score_3p",
        "source_lifecycle_tie_at_max_score_4p",
        "source_lifecycle_multi_round_match_end_3p",
        "source_lifecycle_multi_round_match_end_4p",
    }
)
_SOURCE_LIFECYCLE_3P_SCENARIO_IDS = frozenset(
    {
        "source_lifecycle_spawn_rng_order_3p",
        "source_lifecycle_spawn_rng_warmup_print_start_3p",
        "source_lifecycle_spawn_rng_3p_next_round",
        "source_lifecycle_survivor_score_3p_round_end",
        "source_lifecycle_survivor_score_3p_next_round",
        "source_lifecycle_remove_avatar_during_warmdown_3p",
        "source_lifecycle_remove_avatar_to_single_present_3p",
        "source_lifecycle_present_absent_3p_round_new",
        "source_lifecycle_present_absent_3p_survivor_score_round_end",
        "source_lifecycle_present_absent_3p_next_round",
        "source_lifecycle_present_absent_3p_tie_at_max_score",
        "source_lifecycle_match_end_at_max_score_3p",
        "source_lifecycle_tie_at_max_score_3p",
        "source_lifecycle_multi_round_match_end_3p",
    }
)
_SOURCE_LIFECYCLE_4P_SCENARIO_IDS = frozenset(
    {
        "source_lifecycle_spawn_rng_order_4p",
        "source_lifecycle_spawn_rng_4p_next_round",
        "source_lifecycle_survivor_score_4p_next_round",
        "source_lifecycle_present_absent_4p_round_new",
        "source_lifecycle_present_absent_4p_survivor_score_round_end",
        "source_lifecycle_present_absent_4p_next_round",
        "source_lifecycle_present_absent_4p_tie_at_max_score",
        "source_lifecycle_match_end_at_max_score_4p",
        "source_lifecycle_tie_at_max_score_4p",
        "source_lifecycle_multi_round_match_end_4p",
    }
)
SOURCE_BORDERLESS_PRINT_MANAGER_SCENARIO_ID = (
    "source_borderless_print_manager_wrap_toggle_step"
)
SOURCE_BORDERLESS_BODY_SKIP_SCENARIO_ID = (
    "source_borderless_wrap_skips_destination_body_then_next_frame_kills"
)
_SOURCE_BODY_CANARY_SAME_FRAME_SCENARIO_IDS = frozenset(
    {
        "source_body_same_frame_point_kills_step",
        "source_body_same_frame_point_control_safe_step",
        "source_collision_head_head_reverse_order_single_death_step",
    }
)
_SOURCE_BODY_CANARY_COLLISION_ORDER_SCENARIO_IDS = frozenset(
    {
        "source_collision_death_point_kills_later_player_step",
        "source_collision_head_head_reverse_order_single_death_step",
    }
)
_SOURCE_PRINT_MANAGER_DEATH_SCENARIO_IDS = frozenset(
    {
        "source_print_manager_active_stop_on_death_step",
        "source_print_manager_active_hole_stop_on_death_step",
        "source_print_manager_body_collision_stop_on_death_step",
    }
)
_SOURCE_PRINT_MANAGER_DELAYED_START_SCENARIO_ID = (
    "source_print_manager_delayed_start_timer_step"
)
_SOURCE_PRINT_MANAGER_RANDOM_SCENARIO_ID = (
    "source_print_manager_random_call_order_step"
)
_SOURCE_ROUND_DIGITS = 6
_SOURCE_MISSING = object()

__all__ = [
    "SOURCE_BODY_CANARY_FIDELITY_SCOPE",
    "SOURCE_BODY_CANARY_MESSAGE",
    "SOURCE_BODY_CANARY_RULES_HASH",
    "SOURCE_BODY_CANARY_RUNNER",
    "SOURCE_BODY_CANARY_SCENARIO_IDS",
    "SOURCE_BODY_CANARY_TRACE_SCOPE",
    "SOURCE_BORDERLESS_WRAP_FIDELITY_SCOPE",
    "SOURCE_BORDERLESS_WRAP_MESSAGE",
    "SOURCE_BORDERLESS_BODY_SKIP_SCENARIO_ID",
    "SOURCE_BORDERLESS_PRINT_MANAGER_SCENARIO_ID",
    "SOURCE_BORDERLESS_WRAP_RULES_HASH",
    "SOURCE_BORDERLESS_WRAP_RUNNER",
    "SOURCE_BORDERLESS_WRAP_TRACE_SCOPE",
    "SOURCE_KINEMATICS_FIDELITY_SCOPE",
    "SOURCE_KINEMATICS_MESSAGE",
    "SOURCE_KINEMATICS_RULES_HASH",
    "SOURCE_KINEMATICS_RUNNER",
    "SOURCE_KINEMATICS_TRACE_SCOPE",
    "SOURCE_LIFECYCLE_FIDELITY_SCOPE",
    "SOURCE_LIFECYCLE_MESSAGE",
    "SOURCE_LIFECYCLE_RULES_HASH",
    "SOURCE_LIFECYCLE_RUNNER",
    "SOURCE_LIFECYCLE_SCENARIO_IDS",
    "SOURCE_LIFECYCLE_TRACE_SCHEMA",
    "SOURCE_LIFECYCLE_TRACE_SCOPE",
    "SOURCE_NORMAL_WALL_FIDELITY_SCOPE",
    "SOURCE_NORMAL_WALL_MESSAGE",
    "SOURCE_NORMAL_WALL_RULES_HASH",
    "SOURCE_NORMAL_WALL_RUNNER",
    "SOURCE_NORMAL_WALL_TRACE_SCOPE",
    "SOURCE_PRINT_MANAGER_FIDELITY_SCOPE",
    "SOURCE_PRINT_MANAGER_MESSAGE",
    "SOURCE_PRINT_MANAGER_RULES_HASH",
    "SOURCE_PRINT_MANAGER_RUNNER",
    "SOURCE_PRINT_MANAGER_SCENARIO_IDS",
    "SOURCE_PRINT_MANAGER_TRACE_SCOPE",
    "SOURCE_TRAIL_CADENCE_FIDELITY_SCOPE",
    "SOURCE_TRAIL_CADENCE_MESSAGE",
    "SOURCE_TRAIL_CADENCE_RULES_HASH",
    "SOURCE_TRAIL_CADENCE_RUNNER",
    "SOURCE_TRAIL_CADENCE_SCENARIO_IDS",
    "SOURCE_TRAIL_CADENCE_TRACE_SCOPE",
    "SOURCE_TRAIL_GAP_FIDELITY_SCOPE",
    "SOURCE_TRAIL_GAP_MESSAGE",
    "SOURCE_TRAIL_GAP_RULES_HASH",
    "SOURCE_TRAIL_GAP_RUNNER",
    "SOURCE_TRAIL_GAP_SCENARIO_IDS",
    "SOURCE_TRAIL_GAP_TRACE_SCOPE",
    "SourceBodyCanaryScenarioRun",
    "SourceBorderlessWrapScenarioRun",
    "SourceKinematicsScenarioRun",
    "SourceLifecycleScenarioRun",
    "SourceNormalWallScenarioRun",
    "SourcePrintManagerScenarioRun",
    "SourceTrailCadenceScenarioRun",
    "SourceTrailGapScenarioRun",
    "run_source_body_canary_scenario",
    "run_source_border_rules_scenario",
    "run_source_borderless_wrap_scenario",
    "run_source_kinematics_first_scenario",
    "run_source_kinematics_scenario",
    "run_source_lifecycle_scenario",
    "run_source_normal_wall_scenario",
    "run_source_print_manager_scenario",
    "run_source_trail_cadence_scenario",
    "run_source_trail_gap_scenario",
]


@dataclass(frozen=True, slots=True)
class SourceKinematicsScenarioRun:
    """Result of the narrow source-kinematics movement runner."""

    scenario: LoadedScenario
    trace: Mapping[str, Any]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PYTHON_SCENARIO_TRACE_SCHEMA,
            "runner": SOURCE_KINEMATICS_RUNNER,
            "scenario_id": self.scenario.scenario_id,
            "source_ruleset_id": self.scenario.ruleset_id,
            "python_ruleset_id": SOURCE_KINEMATICS_TRACE_SCOPE,
            "provenance": self.scenario.provenance,
            "toy_v0_behavior": False,
            "source_fidelity": True,
            "source_fidelity_scope": SOURCE_KINEMATICS_FIDELITY_SCOPE,
            "message": SOURCE_KINEMATICS_MESSAGE,
            "scenario": _source_kinematics_scenario_payload(self.scenario),
            "trace_scope": self.trace["scope"],
            "trace_schema_version": self.trace["schema_version"],
            "rules_hash": self.trace["rules_hash"],
            "trace_fingerprint": _fingerprint_payload(self.trace),
            "trace": self.trace,
        }


@dataclass(frozen=True, slots=True)
class SourceNormalWallScenarioRun:
    """Result of the narrow source normal-wall death runner."""

    scenario: LoadedScenario
    trace: Mapping[str, Any]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PYTHON_SCENARIO_TRACE_SCHEMA,
            "runner": SOURCE_NORMAL_WALL_RUNNER,
            "scenario_id": self.scenario.scenario_id,
            "source_ruleset_id": self.scenario.ruleset_id,
            "python_ruleset_id": SOURCE_NORMAL_WALL_TRACE_SCOPE,
            "provenance": self.scenario.provenance,
            "toy_v0_behavior": False,
            "source_fidelity": True,
            "source_fidelity_scope": SOURCE_NORMAL_WALL_FIDELITY_SCOPE,
            "message": SOURCE_NORMAL_WALL_MESSAGE,
            "scenario": _source_kinematics_scenario_payload(self.scenario),
            "trace_scope": self.trace["scope"],
            "trace_schema_version": self.trace["schema_version"],
            "rules_hash": self.trace["rules_hash"],
            "trace_fingerprint": _fingerprint_payload(self.trace),
            "trace": self.trace,
        }


@dataclass(frozen=True, slots=True)
class SourceBorderlessWrapScenarioRun:
    """Result of the narrow source borderless wrap runner."""

    scenario: LoadedScenario
    trace: Mapping[str, Any]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PYTHON_SCENARIO_TRACE_SCHEMA,
            "runner": SOURCE_BORDERLESS_WRAP_RUNNER,
            "scenario_id": self.scenario.scenario_id,
            "source_ruleset_id": self.scenario.ruleset_id,
            "python_ruleset_id": SOURCE_BORDERLESS_WRAP_TRACE_SCOPE,
            "provenance": self.scenario.provenance,
            "toy_v0_behavior": False,
            "source_fidelity": True,
            "source_fidelity_scope": SOURCE_BORDERLESS_WRAP_FIDELITY_SCOPE,
            "message": SOURCE_BORDERLESS_WRAP_MESSAGE,
            "scenario": _source_kinematics_scenario_payload(self.scenario),
            "trace_scope": self.trace["scope"],
            "trace_schema_version": self.trace["schema_version"],
            "rules_hash": self.trace["rules_hash"],
            "trace_fingerprint": _fingerprint_payload(self.trace),
            "trace": self.trace,
        }


@dataclass(frozen=True, slots=True)
class SourceBodyCanaryScenarioRun:
    """Result of the narrow source seeded-body canary runner."""

    scenario: LoadedScenario
    trace: Mapping[str, Any]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PYTHON_SCENARIO_TRACE_SCHEMA,
            "runner": SOURCE_BODY_CANARY_RUNNER,
            "scenario_id": self.scenario.scenario_id,
            "source_ruleset_id": self.scenario.ruleset_id,
            "python_ruleset_id": SOURCE_BODY_CANARY_TRACE_SCOPE,
            "provenance": self.scenario.provenance,
            "toy_v0_behavior": False,
            "source_fidelity": True,
            "source_fidelity_scope": SOURCE_BODY_CANARY_FIDELITY_SCOPE,
            "message": SOURCE_BODY_CANARY_MESSAGE,
            "scenario": _source_kinematics_scenario_payload(self.scenario),
            "trace_scope": self.trace["scope"],
            "trace_schema_version": self.trace["schema_version"],
            "rules_hash": self.trace["rules_hash"],
            "trace_fingerprint": _fingerprint_payload(self.trace),
            "trace": self.trace,
        }


@dataclass(frozen=True, slots=True)
class SourcePrintManagerScenarioRun:
    """Result of the narrow source PrintManager canary runner."""

    scenario: LoadedScenario
    trace: Mapping[str, Any]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PYTHON_SCENARIO_TRACE_SCHEMA,
            "runner": SOURCE_PRINT_MANAGER_RUNNER,
            "scenario_id": self.scenario.scenario_id,
            "source_ruleset_id": self.scenario.ruleset_id,
            "python_ruleset_id": SOURCE_PRINT_MANAGER_TRACE_SCOPE,
            "provenance": self.scenario.provenance,
            "toy_v0_behavior": False,
            "source_fidelity": True,
            "source_fidelity_scope": SOURCE_PRINT_MANAGER_FIDELITY_SCOPE,
            "message": SOURCE_PRINT_MANAGER_MESSAGE,
            "scenario": _source_kinematics_scenario_payload(self.scenario),
            "trace_scope": self.trace["scope"],
            "trace_schema_version": self.trace["schema_version"],
            "rules_hash": self.trace["rules_hash"],
            "trace_fingerprint": _fingerprint_payload(self.trace),
            "randomCalls": list(self.trace.get("randomCalls", [])),
            "trace": self.trace,
        }


@dataclass(frozen=True, slots=True)
class SourceTrailCadenceScenarioRun:
    """Result of the narrow source trail cadence runner."""

    scenario: LoadedScenario
    trace: Mapping[str, Any]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PYTHON_SCENARIO_TRACE_SCHEMA,
            "runner": SOURCE_TRAIL_CADENCE_RUNNER,
            "scenario_id": self.scenario.scenario_id,
            "source_ruleset_id": self.scenario.ruleset_id,
            "python_ruleset_id": SOURCE_TRAIL_CADENCE_TRACE_SCOPE,
            "provenance": self.scenario.provenance,
            "toy_v0_behavior": False,
            "source_fidelity": True,
            "source_fidelity_scope": SOURCE_TRAIL_CADENCE_FIDELITY_SCOPE,
            "message": SOURCE_TRAIL_CADENCE_MESSAGE,
            "scenario": _source_kinematics_scenario_payload(self.scenario),
            "trace_scope": self.trace["scope"],
            "trace_schema_version": self.trace["schema_version"],
            "rules_hash": self.trace["rules_hash"],
            "trace_fingerprint": _fingerprint_payload(self.trace),
            "trace": self.trace,
        }


@dataclass(frozen=True, slots=True)
class SourceTrailGapScenarioRun:
    """Result of the narrow source trail-gap runner."""

    scenario: LoadedScenario
    trace: Mapping[str, Any]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PYTHON_SCENARIO_TRACE_SCHEMA,
            "runner": SOURCE_TRAIL_GAP_RUNNER,
            "scenario_id": self.scenario.scenario_id,
            "source_ruleset_id": self.scenario.ruleset_id,
            "python_ruleset_id": SOURCE_TRAIL_GAP_TRACE_SCOPE,
            "provenance": self.scenario.provenance,
            "toy_v0_behavior": False,
            "source_fidelity": True,
            "source_fidelity_scope": SOURCE_TRAIL_GAP_FIDELITY_SCOPE,
            "message": SOURCE_TRAIL_GAP_MESSAGE,
            "scenario": _source_kinematics_scenario_payload(self.scenario),
            "trace_scope": self.trace["scope"],
            "trace_schema_version": self.trace["schema_version"],
            "rules_hash": self.trace["rules_hash"],
            "trace_fingerprint": _fingerprint_payload(self.trace),
            "randomCalls": list(self.trace.get("randomCalls", [])),
            "trace": self.trace,
        }


@dataclass(frozen=True, slots=True)
class SourceLifecycleScenarioRun:
    """Result of the narrow source lifecycle runner."""

    scenario: Mapping[str, Any]
    trace: Mapping[str, Any]

    def to_payload(self) -> dict[str, object]:
        scenario_id = str(self.scenario["scenario_id"])
        return {
            "schema": PYTHON_SCENARIO_TRACE_SCHEMA,
            "runner": SOURCE_LIFECYCLE_RUNNER,
            "scenario_id": scenario_id,
            "source_ruleset_id": str(self.scenario["ruleset_id"]),
            "python_ruleset_id": SOURCE_LIFECYCLE_TRACE_SCOPE,
            "provenance": self.scenario.get("provenance", "unresolved"),
            "toy_v0_behavior": False,
            "source_fidelity": True,
            "source_fidelity_scope": SOURCE_LIFECYCLE_FIDELITY_SCOPE,
            "message": SOURCE_LIFECYCLE_MESSAGE,
            "scenario": _source_lifecycle_scenario_payload(self.scenario),
            "trace_scope": self.trace["scope"],
            "trace_schema_version": self.trace["schema_version"],
            "rules_hash": self.trace["rules_hash"],
            "trace_fingerprint": _fingerprint_payload(self.trace),
            "lifecycleMode": True,
            "playerCount": self.trace["playerCount"],
            "newRoundTimeMs": self.trace["newRoundTimeMs"],
            "timerAdvancesMs": list(self.trace["timerAdvancesMs"]),
            "events": list(self.trace["events"]),
            "snapshots": list(self.trace["snapshots"]),
            "randomCalls": list(self.trace["randomCalls"]),
            "expectations": dict(self.trace["expectations"]),
            "trace": self.trace,
        }


@dataclass(frozen=True, slots=True)
class _SourceSeededBody:
    owner_index: int
    owner_id: str
    x: float
    y: float
    radius: float
    num: int
    age_ms: float = 0.0


@dataclass(slots=True)
class _SourceRandomSource:
    sequence: tuple[float, ...] | None
    constant: float
    calls: list[dict[str, object]]

    def random(self) -> float:
        index = len(self.calls)
        if self.sequence is None:
            value = self.constant
        else:
            if index >= len(self.sequence):
                raise ScenarioError(f"Math.random tape exhausted after {index} calls")
            value = self.sequence[index]
        self.calls.append({"index": index, "value": value})
        return value


@dataclass(slots=True)
class _SourceLifecycleAvatar:
    id: int
    name: str
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    alive: bool = True
    present: bool = True
    printing: bool = False
    score: int = 0
    round_score: int = 0
    trail_point_count: int = 0
    trail_last_x: float | None = None
    trail_last_y: float | None = None
    body_count: int = 0
    print_manager: dict[str, object] | None = None
    velocity: float = 16.0

    def __post_init__(self) -> None:
        if self.print_manager is None:
            self.print_manager = _source_lifecycle_empty_print_manager()


@dataclass(slots=True)
class _SourceLifecycleGame:
    size: int
    started: bool = False
    in_round: bool = False
    world_active: bool = False
    world_body_count: int = 0
    frame_scheduled: bool = False
    rendered: int | None = None
    game_start_due_ms: float | None = None
    print_start_due_ms: float | None = None
    game_stop_due_ms: float | None = None
    max_score: float = 10.0
    world_exists: bool = True
    death_ids: list[int] = field(default_factory=list)


def run_source_lifecycle_scenario(
    scenario: Mapping[str, Any] | str | Path,
) -> SourceLifecycleScenarioRun:
    """Run one of the pinned source lifecycle fixtures."""

    loaded = _source_lifecycle_load_scenario(scenario)
    scenario_id = str(loaded.get("scenario_id", loaded.get("id", "")))
    if scenario_id not in SOURCE_LIFECYCLE_SCENARIO_IDS:
        raise ScenarioError(
            "source-lifecycle runner supports "
            "source_lifecycle_spawn_rng_warmup_print_start_2p, "
            "source_lifecycle_spawn_rng_2p_next_round, and "
            "source_lifecycle_spawn_heading_rejection_retry_2p, plus "
            "source_lifecycle_spawn_rng_order_3p, "
            "source_lifecycle_spawn_rng_warmup_print_start_3p, "
            "source_lifecycle_spawn_rng_3p_next_round, "
            "source_lifecycle_survivor_score_3p_round_end, "
            "source_lifecycle_survivor_score_3p_next_round, "
            "source_lifecycle_remove_avatar_during_warmdown_3p, "
            "source_lifecycle_remove_avatar_to_single_present_3p, "
            "source_lifecycle_present_absent_3p_round_new, "
            "source_lifecycle_present_absent_3p_survivor_score_round_end, "
            "source_lifecycle_present_absent_3p_next_round, "
            "source_lifecycle_present_absent_3p_tie_at_max_score, "
            "source_lifecycle_spawn_rng_order_4p, "
            "source_lifecycle_spawn_rng_4p_next_round, "
            "source_lifecycle_survivor_score_4p_next_round, "
            "source_lifecycle_present_absent_4p_round_new, "
            "source_lifecycle_present_absent_4p_survivor_score_round_end, "
            "source_lifecycle_present_absent_4p_next_round, "
            "source_lifecycle_present_absent_4p_tie_at_max_score, "
            "source_lifecycle_match_end_at_max_score_2p, "
            "source_lifecycle_match_end_at_max_score_3p, "
            "source_lifecycle_match_end_at_max_score_4p, "
            "source_lifecycle_tie_at_max_score_3p, "
            "source_lifecycle_tie_at_max_score_4p, and "
            "source_lifecycle_multi_round_match_end_3p and "
            "source_lifecycle_multi_round_match_end_4p"
        )
    expected_player_count = _source_lifecycle_expected_player_count(scenario_id)
    if loaded.get("player_count") != expected_player_count:
        raise ScenarioError(
            "source-lifecycle runner currently supports exactly "
            f"{expected_player_count} players for {scenario_id}"
        )

    trace = _trace_source_lifecycle(loaded)
    return SourceLifecycleScenarioRun(scenario=loaded, trace=trace)


def run_source_kinematics_scenario(
    scenario: LoadedScenario | Mapping[str, Any] | str | Path,
) -> SourceKinematicsScenarioRun:
    """Run a simple forced movement scenario with source-derived kinematics."""

    loaded = _coerce_scenario(scenario)
    if not _is_source_kinematics_movement_scenario(loaded.scenario_id):
        raise ScenarioError(
            "source-kinematics runner supports source_kinematics_* "
            "and forced_two_player_turn_step"
        )
    if loaded.player_count != 2:
        raise ScenarioError("source-kinematics runner currently supports exactly 2 players")
    if loaded.action_encoding != "source-move":
        raise ScenarioError("source-kinematics runner requires source-move actions")

    step_ms_values = _source_kinematics_step_ms_values(loaded)
    positions = _extract_positions(loaded.initial_state)
    headings = _extract_headings(loaded.initial_state)
    alive = _extract_alive(loaded.initial_state)

    if positions is None:
        raise ScenarioError("source-kinematics runner requires forced positions")
    if headings is None:
        raise ScenarioError("source-kinematics runner requires forced headings")
    if alive is None:
        alive = [True, True]
    _validate_source_player_state_lengths(
        loaded,
        positions=positions,
        headings=headings,
        alive=alive,
        runner_name="source-kinematics",
    )

    trace = _trace_source_kinematics(loaded, positions, headings, alive, step_ms_values)
    return SourceKinematicsScenarioRun(scenario=loaded, trace=trace)


def run_source_kinematics_first_scenario(
    scenario: LoadedScenario | Mapping[str, Any] | str | Path,
) -> SourceKinematicsScenarioRun:
    """Backward-compatible name for the source-kinematics movement runner."""

    return run_source_kinematics_scenario(scenario)


def run_source_normal_wall_scenario(
    scenario: LoadedScenario | Mapping[str, Any] | str | Path,
) -> SourceNormalWallScenarioRun:
    """Run a forced source normal-wall death scenario."""

    loaded = _coerce_scenario(scenario)
    if not loaded.scenario_id.startswith("source_normal_wall_"):
        raise ScenarioError("source-normal-wall runner supports source_normal_wall_* scenarios")
    if loaded.player_count < 2 or loaded.player_count > 4:
        raise ScenarioError("source-normal-wall runner currently supports 2 to 4 players")
    if loaded.action_encoding != "source-move":
        raise ScenarioError("source-normal-wall runner requires source-move actions")

    step_ms = _source_kinematics_step_ms(loaded)
    positions = _extract_positions(loaded.initial_state)
    headings = _extract_headings(loaded.initial_state)
    alive = _extract_alive(loaded.initial_state)

    if positions is None:
        raise ScenarioError("source-normal-wall runner requires forced positions")
    if headings is None:
        raise ScenarioError("source-normal-wall runner requires forced headings")
    if alive is None:
        alive = [True for _ in range(loaded.player_count)]
    _validate_source_player_state_lengths(
        loaded,
        positions=positions,
        headings=headings,
        alive=alive,
        runner_name="source-normal-wall",
    )

    map_size = _source_map_size(loaded)
    trace = _trace_source_normal_wall(loaded, positions, headings, alive, step_ms, map_size)
    return SourceNormalWallScenarioRun(scenario=loaded, trace=trace)


def run_source_borderless_wrap_scenario(
    scenario: LoadedScenario | Mapping[str, Any] | str | Path,
) -> SourceBorderlessWrapScenarioRun:
    """Run a forced source borderless wrap scenario."""

    loaded = _coerce_scenario(scenario)
    if not loaded.scenario_id.startswith("source_borderless_"):
        raise ScenarioError("source-borderless-wrap runner supports source_borderless_* scenarios")
    is_print_manager_wrap = loaded.scenario_id == SOURCE_BORDERLESS_PRINT_MANAGER_SCENARIO_ID
    is_body_skip = loaded.scenario_id == SOURCE_BORDERLESS_BODY_SKIP_SCENARIO_ID
    if is_print_manager_wrap:
        if loaded.player_count != 1:
            raise ScenarioError(
                "source-borderless-wrap PrintManager fixture currently supports exactly 1 player"
            )
    elif is_body_skip:
        if loaded.player_count != 3:
            raise ScenarioError(
                "source-borderless-wrap body-skip fixture currently supports exactly 3 players"
            )
    elif loaded.player_count != 2:
        raise ScenarioError("source-borderless-wrap runner currently supports exactly 2 players")
    if loaded.action_encoding != "source-move":
        raise ScenarioError("source-borderless-wrap runner requires source-move actions")
    if not _source_setup_borderless(loaded):
        raise ScenarioError("source-borderless-wrap runner requires source_setup.game.borderless")

    step_ms = None if is_body_skip else _source_kinematics_step_ms(loaded)
    step_ms_values = (
        _source_step_ms_values(loaded, runner_name="source-borderless-wrap")
        if is_body_skip
        else None
    )
    if is_print_manager_wrap:
        positions, headings, alive = _source_single_player_forced_state(
            loaded,
            runner_name="source-borderless-wrap",
        )
    else:
        positions = _extract_positions(loaded.initial_state)
        headings = _extract_headings(loaded.initial_state)
        alive = _extract_alive(loaded.initial_state)

    if positions is None:
        raise ScenarioError("source-borderless-wrap runner requires forced positions")
    if headings is None:
        raise ScenarioError("source-borderless-wrap runner requires forced headings")
    if alive is None:
        alive = [True for _ in range(loaded.player_count)]
    _validate_source_player_state_lengths(
        loaded,
        positions=positions,
        headings=headings,
        alive=alive,
        runner_name="source-borderless-wrap",
    )

    map_size = _source_map_size(loaded)
    if is_print_manager_wrap:
        assert step_ms is not None
        trace = _trace_source_borderless_print_manager_wrap(
            loaded,
            positions,
            headings,
            alive,
            step_ms,
            map_size,
            _source_player_ids(loaded),
        )
    elif is_body_skip:
        assert step_ms_values is not None
        player_ids = _source_player_ids(loaded)
        seeded_bodies = _source_seeded_world_bodies(loaded, player_ids)
        trace = _trace_source_borderless_body_skip(
            loaded,
            positions,
            headings,
            alive,
            step_ms_values,
            map_size,
            seeded_bodies,
            player_ids,
        )
    else:
        assert step_ms is not None
        trace = _trace_source_borderless_wrap(loaded, positions, headings, alive, step_ms, map_size)
    return SourceBorderlessWrapScenarioRun(scenario=loaded, trace=trace)


def run_source_body_canary_scenario(
    scenario: LoadedScenario | Mapping[str, Any] | str | Path,
) -> SourceBodyCanaryScenarioRun:
    """Run one of the forced seeded-body canary scenarios."""

    loaded = _coerce_scenario(scenario)
    if loaded.scenario_id not in SOURCE_BODY_CANARY_SCENARIO_IDS:
        raise ScenarioError(
            "source-body-canary runner supports source_body_opponent_tangent_safe_step "
            "source_body_opponent_overlap_kills_step, source_body_own_delta3_safe_step, "
            "source_body_own_delta4_kills_step, "
            "source_body_same_frame_point_kills_step, and "
            "source_body_same_frame_point_control_safe_step, plus "
            "source_body_old_opponent_overlap_kills_step, "
            "source_collision_death_point_kills_later_player_step and "
            "source_collision_head_head_reverse_order_single_death_step"
        )
    expected_player_count = (
        2
        if loaded.scenario_id in _SOURCE_BODY_CANARY_COLLISION_ORDER_SCENARIO_IDS
        else 3
    )
    if loaded.player_count != expected_player_count:
        raise ScenarioError(
            "source-body-canary runner currently supports exactly "
            f"{expected_player_count} players for {loaded.scenario_id}"
        )
    if loaded.action_encoding != "source-move":
        raise ScenarioError("source-body-canary runner requires source-move actions")
    if _source_setup_borderless(loaded):
        raise ScenarioError("source-body-canary runner requires non-borderless source setup")

    step_ms = _source_kinematics_step_ms(loaded)
    positions = _extract_positions(loaded.initial_state)
    headings = _extract_headings(loaded.initial_state)
    alive = _extract_alive(loaded.initial_state)

    if positions is None:
        raise ScenarioError("source-body-canary runner requires forced positions")
    if headings is None:
        raise ScenarioError("source-body-canary runner requires forced headings")
    if alive is None:
        alive = [True for _ in range(loaded.player_count)]
    _validate_source_player_state_lengths(
        loaded,
        positions=positions,
        headings=headings,
        alive=alive,
        runner_name="source-body-canary",
    )

    map_size = _source_map_size(loaded)
    player_ids = _source_player_ids(loaded)
    seeded_bodies = _source_seeded_world_bodies(
        loaded,
        player_ids,
        require_non_empty=loaded.scenario_id not in _SOURCE_BODY_CANARY_SAME_FRAME_SCENARIO_IDS,
    )
    trace = _trace_source_body_canary(
        loaded,
        positions,
        headings,
        alive,
        step_ms,
        map_size,
        seeded_bodies,
        player_ids,
    )
    return SourceBodyCanaryScenarioRun(scenario=loaded, trace=trace)


def run_source_print_manager_scenario(
    scenario: LoadedScenario | Mapping[str, Any] | str | Path,
) -> SourcePrintManagerScenarioRun:
    """Run one of the forced source PrintManager canary scenarios."""

    loaded = _coerce_scenario(scenario)
    if loaded.scenario_id not in SOURCE_PRINT_MANAGER_SCENARIO_IDS:
        raise ScenarioError(
            "source-print-manager-canary runner supports "
            "source_print_manager_print_to_hole_step, "
            "source_print_manager_hole_to_print_step, "
            "source_print_manager_exact_zero_toggle_step, "
            "source_print_manager_no_toggle_control_step, "
            "source_print_manager_active_stop_on_death_step, "
            "source_print_manager_active_hole_stop_on_death_step, and "
            "source_print_manager_body_collision_stop_on_death_step, plus "
            "source_print_manager_delayed_start_timer_step and "
            "source_print_manager_random_call_order_step and "
            "source_print_manager_random_cadence_multistep"
        )
    if loaded.scenario_id in _SOURCE_PRINT_MANAGER_DEATH_SCENARIO_IDS:
        expected_player_count = 3
    elif loaded.scenario_id == _SOURCE_PRINT_MANAGER_RANDOM_SCENARIO_ID:
        expected_player_count = 2
    else:
        expected_player_count = 1
    if loaded.player_count != expected_player_count:
        raise ScenarioError(
            "source-print-manager-canary runner currently supports exactly "
            f"{expected_player_count} players for {loaded.scenario_id}"
        )
    if loaded.action_encoding != "source-move":
        raise ScenarioError("source-print-manager-canary runner requires source-move actions")
    if _source_setup_borderless(loaded):
        raise ScenarioError("source-print-manager-canary runner requires non-borderless source setup")

    step_ms = _source_kinematics_step_ms(loaded)
    positions, headings, alive = _source_print_manager_forced_state(loaded)
    _validate_source_player_state_lengths(
        loaded,
        positions=positions,
        headings=headings,
        alive=alive,
        runner_name="source-print-manager-canary",
    )

    map_size = _source_map_size(loaded)
    player_ids = _source_player_ids(loaded)
    trace = _trace_source_print_manager(
        loaded,
        positions,
        headings,
        alive,
        step_ms,
        map_size,
        player_ids,
    )
    return SourcePrintManagerScenarioRun(scenario=loaded, trace=trace)


def run_source_trail_cadence_scenario(
    scenario: LoadedScenario | Mapping[str, Any] | str | Path,
) -> SourceTrailCadenceScenarioRun:
    """Run one of the forced source trail cadence canary scenarios."""

    loaded = _coerce_scenario(scenario)
    if loaded.scenario_id not in SOURCE_TRAIL_CADENCE_SCENARIO_IDS:
        raise ScenarioError(
            "source-trail-cadence-canary runner supports "
            "source_trail_normal_point_step and "
            "source_trail_no_point_below_radius_step"
        )
    if loaded.player_count != 1:
        raise ScenarioError("source-trail-cadence-canary runner currently supports exactly 1 player")
    if loaded.action_encoding != "source-move":
        raise ScenarioError("source-trail-cadence-canary runner requires source-move actions")
    if _source_setup_borderless(loaded):
        raise ScenarioError("source-trail-cadence-canary runner requires non-borderless source setup")

    step_ms = _source_kinematics_step_ms(loaded)
    positions, headings, alive = _source_single_player_forced_state(
        loaded,
        runner_name="source-trail-cadence-canary",
    )
    _validate_source_player_state_lengths(
        loaded,
        positions=positions,
        headings=headings,
        alive=alive,
        runner_name="source-trail-cadence-canary",
    )

    map_size = _source_map_size(loaded)
    player_ids = _source_player_ids(loaded)
    trace = _trace_source_trail_cadence(
        loaded,
        positions,
        headings,
        alive,
        step_ms,
        map_size,
        player_ids,
    )
    return SourceTrailCadenceScenarioRun(scenario=loaded, trace=trace)


def run_source_trail_gap_scenario(
    scenario: LoadedScenario | Mapping[str, Any] | str | Path,
) -> SourceTrailGapScenarioRun:
    """Run one of the forced source trail-gap canary scenarios."""

    loaded = _coerce_scenario(scenario)
    if loaded.scenario_id not in SOURCE_TRAIL_GAP_SCENARIO_IDS:
        raise ScenarioError(
            "source-trail-gap-canary runner supports "
            "source_trail_gap_hole_space_safe_step, "
            "source_trail_gap_stored_body_still_kills_step, "
            "source_trail_gap_print_to_hole_boundary_kills_step, "
            "source_trail_gap_hole_to_print_boundary_kills_step, and "
            "source_trail_gap_natural_multistep_hole_crossing"
        )
    if loaded.player_count != 3:
        raise ScenarioError("source-trail-gap-canary runner currently supports exactly 3 players")
    if loaded.action_encoding != "source-move":
        raise ScenarioError("source-trail-gap-canary runner requires source-move actions")
    if _source_setup_borderless(loaded):
        raise ScenarioError("source-trail-gap-canary runner requires non-borderless source setup")

    step_ms = _source_kinematics_step_ms(loaded)
    positions = _extract_positions(loaded.initial_state)
    headings = _extract_headings(loaded.initial_state)
    alive = _extract_alive(loaded.initial_state)

    if positions is None:
        raise ScenarioError("source-trail-gap-canary runner requires forced positions")
    if headings is None:
        raise ScenarioError("source-trail-gap-canary runner requires forced headings")
    if alive is None:
        alive = [True for _ in range(loaded.player_count)]
    _validate_source_player_state_lengths(
        loaded,
        positions=positions,
        headings=headings,
        alive=alive,
        runner_name="source-trail-gap-canary",
    )

    map_size = _source_map_size(loaded)
    player_ids = _source_player_ids(loaded)
    seeded_bodies = _source_seeded_world_bodies(
        loaded,
        player_ids,
        require_non_empty=False,
    )
    trace = _trace_source_trail_gap(
        loaded,
        positions,
        headings,
        alive,
        step_ms,
        map_size,
        seeded_bodies,
        player_ids,
    )
    return SourceTrailGapScenarioRun(scenario=loaded, trace=trace)


def run_source_border_rules_scenario(
    scenario: LoadedScenario | Mapping[str, Any] | str | Path,
) -> SourceNormalWallScenarioRun | SourceBorderlessWrapScenarioRun:
    """Run one forced source border-rule scenario through the matching runner."""

    loaded = _coerce_scenario(scenario)
    if loaded.scenario_id.startswith("source_normal_wall_"):
        return run_source_normal_wall_scenario(loaded)
    if loaded.scenario_id.startswith("source_borderless_"):
        return run_source_borderless_wrap_scenario(loaded)
    raise ScenarioError(
        "source-border-rules runner supports source_normal_wall_* "
        "and source_borderless_* scenarios"
    )


def _trace_source_lifecycle(scenario: Mapping[str, Any]) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    scenario_id = str(scenario["scenario_id"])
    _source_lifecycle_players(scenario, reference)
    random_values = _source_lifecycle_random_sequence(scenario)
    timer_advances_ms = _source_lifecycle_timer_advances_ms(scenario)
    actions = _source_lifecycle_actions(scenario)
    warmup_ms = _source_lifecycle_new_round_time_ms(scenario)
    include_deaths_snapshot = _source_lifecycle_include_deaths_snapshot(scenario)

    env = CurvyTronSourceEnv(
        reference=reference,
        random_values=random_values,
        max_score=_source_lifecycle_max_score(scenario),
        include_deaths_snapshot=include_deaths_snapshot,
        drain_frame_timers=True,
    )
    player_count = _source_lifecycle_expected_player_count(scenario_id)
    snapshots: list[dict[str, object]] = [
        env.reset(
            player_count=player_count,
            players=scenario["players"],  # type: ignore[arg-type]
            warmup_ms=warmup_ms,
        )
    ]

    if actions is None:
        for index, advance_ms in enumerate(timer_advances_ms):
            env.advance_timers(advance_ms)
            snapshots.append(
                env.snapshot(f"after_advance_{index}", advance_ms=advance_ms)
            )
    else:
        for index, action in enumerate(actions):
            action_type = _source_lifecycle_action_type(action)
            if action_type in {"advance_timers", "advance"}:
                env.advance_timers(_source_lifecycle_action_number(action, "ms", index))
            elif action_type == "set_avatar_state":
                _source_lifecycle_apply_env_avatar_state(env, action, index)
            elif action_type in {"remove_avatar", "removeAvatar"}:
                env.remove_avatar(_source_lifecycle_action_avatar_id(action, index))
            elif action_type == "update":
                env.step({}, _source_lifecycle_action_number(action, "step_ms", index))
            else:
                raise ScenarioError(f"unsupported lifecycle action type: {action_type}")
            snapshots.append(
                env.snapshot(f"after_action_{index}_{action_type}", action=action)
            )

    expectations = {
        "eventOrder": _source_lifecycle_validate_event_order(
            env.events,
            scenario.get("expectations", {}).get("event_order")
            if isinstance(scenario.get("expectations"), Mapping)
            else None,
        )
    }
    trace: dict[str, object] = {
        "scope": SOURCE_LIFECYCLE_TRACE_SCOPE,
        "schema_version": SOURCE_LIFECYCLE_TRACE_SCHEMA,
        "ruleset": SOURCE_LIFECYCLE_TRACE_SCOPE,
        "rules_hash": SOURCE_LIFECYCLE_RULES_HASH,
        "scenario": scenario_id,
        "lifecycleMode": True,
        "playerCount": len(env.avatars),
        "newRoundTimeMs": _source_lifecycle_number(warmup_ms),
        "timerAdvancesMs": [_source_lifecycle_number(value) for value in timer_advances_ms],
        "events": env.events,
        "snapshots": snapshots,
        "randomCalls": env.random_calls,
        "expectations": expectations,
    }
    if actions is not None:
        trace["lifecycleActions"] = [dict(action) for action in actions]
    return trace


def _source_lifecycle_load_scenario(scenario: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(scenario, Mapping):
        loaded = dict(scenario)
    else:
        with Path(scenario).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, Mapping):
            raise ScenarioError("source-lifecycle scenario JSON must be an object")
        loaded = dict(raw)

    scenario_id = loaded.get("scenario_id", loaded.get("id"))
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ScenarioError("scenario_id must be a non-empty string")
    ruleset_id = loaded.get("ruleset_id")
    if not isinstance(ruleset_id, str) or not ruleset_id:
        raise ScenarioError("ruleset_id must be a non-empty string")
    loaded["scenario_id"] = scenario_id
    return loaded


def _source_lifecycle_scenario_payload(scenario: Mapping[str, Any]) -> dict[str, object]:
    return dict(scenario)


def _source_lifecycle_players(
    scenario: Mapping[str, Any],
    reference: CurvyTronReferenceDefaults,
) -> list[_SourceLifecycleAvatar]:
    players = scenario.get("players")
    scenario_id = str(scenario.get("scenario_id", scenario.get("id", "")))
    expected_player_count = _source_lifecycle_expected_player_count(scenario_id)
    if not isinstance(players, list) or len(players) != expected_player_count:
        raise ScenarioError(
            "source-lifecycle runner requires exactly "
            f"{expected_player_count} players for {scenario_id}"
        )
    avatars: list[_SourceLifecycleAvatar] = []
    for index, raw_player in enumerate(players):
        if not isinstance(raw_player, Mapping):
            raise ScenarioError("source-lifecycle players entries must be objects")
        avatar_id = _source_lifecycle_avatar_id(raw_player, index)
        name = str(raw_player.get("name") or raw_player.get("id") or f"p{index}")
        present = _source_lifecycle_player_present(raw_player, index)
        avatar = _SourceLifecycleAvatar(id=avatar_id, name=name, present=present)
        if not present:
            avatar.alive = False
            avatar.x = reference.avatar_radius
            avatar.y = reference.avatar_radius
        avatars.append(avatar)
    return avatars


def _source_lifecycle_expected_player_count(scenario_id: str) -> int:
    if scenario_id in _SOURCE_LIFECYCLE_4P_SCENARIO_IDS:
        return 4
    if scenario_id in _SOURCE_LIFECYCLE_3P_SCENARIO_IDS:
        return 3
    if scenario_id in SOURCE_LIFECYCLE_SCENARIO_IDS:
        return 2
    raise ScenarioError(f"unsupported source-lifecycle scenario: {scenario_id}")


def _source_lifecycle_player_present(player: Mapping[str, Any], index: int) -> bool:
    value = _source_first_owned(player, ("present", "avatar_present", "avatarPresent"))
    if value is _SOURCE_MISSING:
        return True
    if not isinstance(value, bool):
        raise ScenarioError(
            f"source-lifecycle players[{index}].present must be a boolean"
        )
    return value


def _source_lifecycle_avatar_id(player: Mapping[str, Any], index: int) -> int:
    value = player.get("avatar_id", player.get("avatarId"))
    if value is None:
        value = player.get("id")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioError(
            f"source-lifecycle players[{index}].avatar_id must be an integer"
        )
    return int(value)


def _source_lifecycle_random_sequence(scenario: Mapping[str, Any]) -> tuple[float, ...]:
    source_setup = scenario.get("source_setup")
    if not isinstance(source_setup, Mapping):
        raise ScenarioError("source-lifecycle runner requires source_setup")
    random_setup = source_setup.get("random")
    if not isinstance(random_setup, Mapping):
        raise ScenarioError("source-lifecycle runner requires source_setup.random")
    raw_sequence = _source_first_owned(
        random_setup,
        (
            "math_random_sequence",
            "mathRandomSequence",
            "math_random_tape",
            "mathRandomTape",
        ),
    )
    if not isinstance(raw_sequence, list):
        raise ScenarioError("source_setup.random.math_random_sequence must be an array")
    return tuple(
        _source_required_random_value(
            value,
            f"source_setup.random.math_random_sequence[{index}]",
        )
        for index, value in enumerate(raw_sequence)
    )


def _source_lifecycle_max_score(scenario: Mapping[str, Any]) -> float:
    source_setup = scenario.get("source_setup")
    if not isinstance(source_setup, Mapping):
        raise ScenarioError("source-lifecycle runner requires source_setup")
    room = source_setup.get("room")
    if not isinstance(room, Mapping):
        return 10.0
    value = _source_first_owned(room, ("max_score", "maxScore"))
    if value is _SOURCE_MISSING:
        return 10.0
    return _source_lifecycle_number_field(value, "source_setup.room.max_score")


def _source_lifecycle_mapping(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    lifecycle = scenario.get("lifecycle")
    if lifecycle is None:
        return {}
    if not isinstance(lifecycle, Mapping):
        raise ScenarioError("lifecycle must be an object")
    return lifecycle


def _source_lifecycle_new_round_time_ms(scenario: Mapping[str, Any]) -> float:
    lifecycle = _source_lifecycle_mapping(scenario)
    value = lifecycle.get("new_round_time_ms", scenario.get("new_round_time_ms", 0))
    return _source_lifecycle_number_field(value, "lifecycle.new_round_time_ms")


def _source_lifecycle_timer_advances_ms(scenario: Mapping[str, Any]) -> list[float]:
    lifecycle = _source_lifecycle_mapping(scenario)
    raw_values = scenario.get("timer_advances_ms", lifecycle.get("timer_advances_ms", [0, 3000]))
    if not isinstance(raw_values, list):
        raise ScenarioError("lifecycle.timer_advances_ms must be an array")
    return [
        _source_lifecycle_number_field(
            value,
            f"lifecycle.timer_advances_ms[{index}]",
        )
        for index, value in enumerate(raw_values)
    ]


def _source_lifecycle_actions(
    scenario: Mapping[str, Any],
) -> list[Mapping[str, object]] | None:
    lifecycle = _source_lifecycle_mapping(scenario)
    raw_actions = lifecycle.get("actions", scenario.get("actions"))
    if raw_actions is None:
        return None
    if not isinstance(raw_actions, list):
        raise ScenarioError("lifecycle.actions must be an array")
    actions: list[Mapping[str, object]] = []
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, Mapping):
            raise ScenarioError(f"lifecycle.actions[{index}] must be an object")
        actions.append(raw_action)
    return actions


def _source_lifecycle_include_deaths_snapshot(scenario: Mapping[str, Any]) -> bool:
    lifecycle = _source_lifecycle_mapping(scenario)
    return (
        scenario.get("include_deaths_snapshot") is True
        or lifecycle.get("include_deaths_snapshot") is True
    )


def _source_lifecycle_number_field(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenarioError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ScenarioError(f"{field} must be a finite non-negative number")
    return result


def _source_lifecycle_number(value: float) -> int | float:
    rounded = _source_round(float(value))
    nearest = round(rounded)
    if math.isclose(rounded, nearest, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest)
    return rounded


def _source_lifecycle_empty_print_manager() -> dict[str, object]:
    return {"active": False, "distance": 0.0, "lastX": 0.0, "lastY": 0.0}


def _source_lifecycle_manager(avatar: _SourceLifecycleAvatar) -> dict[str, object]:
    if avatar.print_manager is None:
        avatar.print_manager = _source_lifecycle_empty_print_manager()
    return avatar.print_manager


def _source_lifecycle_game_snapshot(
    game: _SourceLifecycleGame,
    *,
    include_deaths: bool = False,
) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "size": game.size,
        "started": game.started,
        "inRound": game.in_round,
        "worldActive": game.world_active if game.world_exists else None,
        "worldBodyCount": game.world_body_count if game.world_exists else None,
        "frameScheduled": game.frame_scheduled,
        "rendered": game.rendered,
    }
    if include_deaths:
        snapshot["deathCount"] = len(game.death_ids)
        snapshot["deaths"] = list(game.death_ids)
    return snapshot


def _source_lifecycle_avatar_snapshot(avatar: _SourceLifecycleAvatar) -> dict[str, object]:
    manager = _source_lifecycle_manager(avatar)
    return {
        "id": avatar.id,
        "name": avatar.name,
        "x": _source_lifecycle_number(avatar.x),
        "y": _source_lifecycle_number(avatar.y),
        "angle": _source_lifecycle_number(avatar.angle),
        "alive": avatar.alive,
        "present": avatar.present,
        "printing": avatar.printing,
        "score": avatar.score,
        "roundScore": avatar.round_score,
        "trailPointCount": avatar.trail_point_count,
        "printManager": {
            "active": bool(manager["active"]),
            "distance": _source_lifecycle_number(float(manager["distance"])),
            "lastX": _source_lifecycle_number(float(manager["lastX"])),
            "lastY": _source_lifecycle_number(float(manager["lastY"])),
        },
    }


def _source_lifecycle_direction_valid(
    angle: float,
    x: float,
    y: float,
    *,
    tolerance: float,
    size: float,
) -> bool:
    quarter = math.pi / 2.0
    margin = tolerance * size
    for border in range(4):
        start = quarter * border
        end = quarter * (border + 1)
        if angle >= start and angle < end:
            if (
                _source_lifecycle_hypotenuse(
                    angle - start,
                    _source_lifecycle_distance_to_border(border, x, y, size),
                )
                < margin
            ):
                return False
            next_border = border + 1 if border < 3 else 0
            if (
                _source_lifecycle_hypotenuse(
                    end - angle,
                    _source_lifecycle_distance_to_border(next_border, x, y, size),
                )
                < margin
            ):
                return False
            return True
    return False


def _source_lifecycle_hypotenuse(angle: float, adjacent: float) -> float:
    return adjacent / math.cos(angle)


def _source_lifecycle_distance_to_border(
    border: int,
    x: float,
    y: float,
    size: float,
) -> float:
    if border == 0:
        return size - x
    if border == 1:
        return size - y
    if border == 2:
        return x
    return y


def _source_lifecycle_is_time_to_draw(
    avatar: _SourceLifecycleAvatar,
    radius: float,
) -> bool:
    if avatar.trail_last_x is None or avatar.trail_last_y is None:
        return True
    return math.hypot(avatar.x - avatar.trail_last_x, avatar.y - avatar.trail_last_y) > radius


def _source_lifecycle_action_type(action: Mapping[str, object]) -> str:
    value = action.get("type")
    if not isinstance(value, str) or not value:
        raise ScenarioError("lifecycle action type must be a non-empty string")
    return value


def _source_lifecycle_action_avatar_id(
    action: Mapping[str, object],
    index: int,
) -> int | str:
    avatar_id = action.get("avatar", action.get("avatar_id", action.get("avatarId")))
    if isinstance(avatar_id, bool) or not isinstance(avatar_id, int | str):
        raise ScenarioError(f"lifecycle.actions[{index}].avatar is required")
    return avatar_id


def _source_lifecycle_action_number(
    action: Mapping[str, object],
    key: str,
    index: int,
) -> float:
    return _source_lifecycle_number_field(action.get(key), f"lifecycle.actions[{index}].{key}")


def _source_lifecycle_apply_avatar_state(
    avatars: list[_SourceLifecycleAvatar],
    action: Mapping[str, object],
    index: int,
) -> None:
    avatar_id = action.get("avatar", action.get("avatar_id", action.get("avatarId")))
    avatar = _source_lifecycle_find_avatar(avatars, avatar_id)
    if "x" in action:
        avatar.x = _source_lifecycle_number_field(action["x"], f"lifecycle.actions[{index}].x")
    if "y" in action:
        avatar.y = _source_lifecycle_number_field(action["y"], f"lifecycle.actions[{index}].y")
    if "angle_rad" in action:
        avatar.angle = _source_lifecycle_number_field(
            action["angle_rad"],
            f"lifecycle.actions[{index}].angle_rad",
        )
    elif "angle" in action:
        avatar.angle = _source_lifecycle_number_field(
            action["angle"],
            f"lifecycle.actions[{index}].angle",
        )
    if "velocity" in action:
        velocity = _source_lifecycle_number_field(
            action["velocity"],
            f"lifecycle.actions[{index}].velocity",
        )
        avatar.velocity = max(velocity, 8.0)


def _source_lifecycle_apply_env_avatar_state(
    env: CurvyTronSourceEnv,
    action: Mapping[str, object],
    index: int,
) -> None:
    avatar_id = _source_lifecycle_action_avatar_id(action, index)
    updates: dict[str, float] = {}
    if "x" in action:
        updates["x"] = _source_lifecycle_number_field(
            action["x"],
            f"lifecycle.actions[{index}].x",
        )
    if "y" in action:
        updates["y"] = _source_lifecycle_number_field(
            action["y"],
            f"lifecycle.actions[{index}].y",
        )
    if "angle_rad" in action:
        updates["angle"] = _source_lifecycle_number_field(
            action["angle_rad"],
            f"lifecycle.actions[{index}].angle_rad",
        )
    elif "angle" in action:
        updates["angle"] = _source_lifecycle_number_field(
            action["angle"],
            f"lifecycle.actions[{index}].angle",
        )
    if "velocity" in action:
        updates["velocity"] = _source_lifecycle_number_field(
            action["velocity"],
            f"lifecycle.actions[{index}].velocity",
        )
    env.set_avatar_state(avatar_id, **updates)


def _source_lifecycle_find_avatar(
    avatars: list[_SourceLifecycleAvatar],
    raw_avatar_id: object,
) -> _SourceLifecycleAvatar:
    for avatar in avatars:
        if str(avatar.id) == str(raw_avatar_id):
            return avatar
    raise ScenarioError(f"avatar not found: {raw_avatar_id}")


def _source_lifecycle_validate_event_order(
    events: list[dict[str, object]],
    expected: object,
) -> dict[str, object]:
    if expected is None:
        return {"status": "not_checked"}
    if not isinstance(expected, list):
        raise ScenarioError("expectations.event_order must be an array")
    if len(events) != len(expected):
        return {
            "status": "fail",
            "message": f"event count mismatch: expected {len(expected)}, got {len(events)}",
        }
    for index, raw_wanted in enumerate(expected):
        if not isinstance(raw_wanted, Mapping):
            raise ScenarioError(f"expectations.event_order[{index}] must be an object")
        actual = events[index]
        for event_field, wanted_value in raw_wanted.items():
            if actual.get(event_field) != wanted_value:
                return {
                    "status": "fail",
                    "message": f"event {index} field {event_field} mismatch",
                    "expected": dict(raw_wanted),
                    "actual": actual,
                }
    return {"status": "pass"}


def _trace_source_kinematics(
    scenario: LoadedScenario,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    step_ms_values: list[float],
) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    current_positions = [[float(x), float(y)] for x, y in positions]
    current_headings = [float(heading) for heading in headings]
    player_ids = _source_player_ids(scenario)
    include_events = _source_include_events(scenario)

    frames = [
        _source_kinematics_frame(
            tick=0,
            positions=current_positions,
            headings=current_headings,
            alive=alive,
            rounded=False,
        )
    ]

    for tick, (actions, step_ms) in enumerate(
        zip(scenario.raw_action_script, step_ms_values, strict=True),
        start=1,
    ):
        tick_events: list[dict[str, object]] = []
        for player_index in reversed(range(scenario.player_count)):
            move = actions[f"player_{player_index}"]
            current_headings[player_index] += (
                move * reference.angular_velocity_radians_per_ms * step_ms
            )
            distance = reference.avatar_velocity_units_per_s * step_ms / 1000.0
            current_positions[player_index][0] += math.cos(current_headings[player_index]) * distance
            current_positions[player_index][1] += math.sin(current_headings[player_index]) * distance
            if include_events:
                if move != 0:
                    tick_events.append(
                        _source_angle_event(
                            player_ids[player_index],
                            current_headings[player_index],
                        )
                    )
                tick_events.append(
                    _source_position_event(player_ids[player_index], current_positions[player_index])
                )

        frames.append(
            _source_kinematics_frame(
                tick=tick,
                positions=current_positions,
                headings=current_headings,
                alive=alive,
                rounded=True,
                step_ms=step_ms,
                events=tick_events if include_events else None,
            )
        )

    return {
        "scope": SOURCE_KINEMATICS_TRACE_SCOPE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "ruleset": SOURCE_KINEMATICS_TRACE_SCOPE,
        "rules_hash": SOURCE_KINEMATICS_RULES_HASH,
        "seed": scenario.seed,
        "scripted_actions": scenario.raw_action_script,
        "frames": frames,
    }


def _trace_source_normal_wall(
    scenario: LoadedScenario,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    step_ms: float,
    map_size: float,
) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    current_positions = [[float(x), float(y)] for x, y in positions]
    current_headings = [float(heading) for heading in headings]
    current_alive = [bool(value) for value in alive]
    scores = [0 for _ in range(scenario.player_count)]
    round_scores = [0 for _ in range(scenario.player_count)]
    player_ids = _source_player_ids(scenario)
    include_events = _source_include_events(scenario)

    frames = [
        _source_kinematics_frame(
            tick=0,
            positions=current_positions,
            headings=current_headings,
            alive=current_alive,
            rounded=False,
            scores=scores,
            round_scores=round_scores,
        )
    ]

    for tick, actions in enumerate(scenario.raw_action_script, start=1):
        deaths = [False for _ in range(scenario.player_count)]
        tick_events: list[dict[str, object]] = []
        frame_start_deaths = sum(not value for value in current_alive)
        for player_index in reversed(range(scenario.player_count)):
            if not current_alive[player_index]:
                continue
            move = actions[f"player_{player_index}"]
            current_headings[player_index] += (
                move * reference.angular_velocity_radians_per_ms * step_ms
            )
            distance = reference.avatar_velocity_units_per_s * step_ms / 1000.0
            current_positions[player_index][0] += (
                math.cos(current_headings[player_index]) * distance
            )
            current_positions[player_index][1] += (
                math.sin(current_headings[player_index]) * distance
            )
            tick_events.append(
                _source_position_event(player_ids[player_index], current_positions[player_index])
            )
            if _source_normal_wall_hit(
                current_positions[player_index],
                reference.avatar_radius,
                map_size,
            ):
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )
                tick_events.append(_source_die_event(player_ids[player_index]))
                current_alive[player_index] = False
                deaths[player_index] = True
                round_scores[player_index] += frame_start_deaths
                tick_events.append(
                    _source_score_round_event(
                        player_ids[player_index],
                        scores[player_index],
                        round_scores[player_index],
                    )
                )

        if any(deaths) and sum(current_alive) <= 1:
            winner: int | None = None
            if sum(current_alive) == 1:
                winner = current_alive.index(True)
                round_scores[winner] += max(scenario.player_count - 1, 1)
                tick_events.append(
                    _source_score_round_event(
                        player_ids[winner],
                        scores[winner],
                        round_scores[winner],
                    )
                )
            for player_index in reversed(range(scenario.player_count)):
                scores[player_index] += round_scores[player_index]
                tick_events.append(
                    _source_score_event(
                        player_ids[player_index],
                        scores[player_index],
                        round_scores[player_index],
                    )
                )
            round_scores = [0 for _ in range(scenario.player_count)]
            tick_events.append(_source_round_end_event(None if winner is None else player_ids[winner]))

        frames.append(
            _source_kinematics_frame(
                tick=tick,
                positions=current_positions,
                headings=current_headings,
                alive=current_alive,
                rounded=True,
                scores=scores,
                round_scores=round_scores,
                events=tick_events if include_events else None,
            )
        )

    return {
        "scope": SOURCE_NORMAL_WALL_TRACE_SCOPE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "ruleset": SOURCE_NORMAL_WALL_TRACE_SCOPE,
        "rules_hash": SOURCE_NORMAL_WALL_RULES_HASH,
        "seed": scenario.seed,
        "scripted_actions": scenario.raw_action_script,
        "frames": frames,
    }


def _trace_source_borderless_wrap(
    scenario: LoadedScenario,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    step_ms: float,
    map_size: float,
) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    current_positions = [[float(x), float(y)] for x, y in positions]
    current_headings = [float(heading) for heading in headings]
    current_alive = [bool(value) for value in alive]
    scores = [0 for _ in range(scenario.player_count)]
    round_scores = [0 for _ in range(scenario.player_count)]
    player_ids = _source_player_ids(scenario)
    include_events = _source_include_events(scenario)

    frames = [
        _source_kinematics_frame(
            tick=0,
            positions=current_positions,
            headings=current_headings,
            alive=current_alive,
            rounded=False,
            scores=scores,
            round_scores=round_scores,
        )
    ]

    for tick, actions in enumerate(scenario.raw_action_script, start=1):
        tick_events: list[dict[str, object]] = []
        for player_index in reversed(range(scenario.player_count)):
            if not current_alive[player_index]:
                continue
            move = actions[f"player_{player_index}"]
            current_headings[player_index] += (
                move * reference.angular_velocity_radians_per_ms * step_ms
            )
            distance = reference.avatar_velocity_units_per_s * step_ms / 1000.0
            current_positions[player_index][0] += (
                math.cos(current_headings[player_index]) * distance
            )
            current_positions[player_index][1] += (
                math.sin(current_headings[player_index]) * distance
            )
            tick_events.append(
                _source_position_event(player_ids[player_index], current_positions[player_index])
            )
            wrapped = _source_apply_borderless_wrap(current_positions[player_index], map_size)
            if wrapped:
                tick_events.append(
                    _source_position_event(player_ids[player_index], current_positions[player_index])
                )

        frames.append(
            _source_kinematics_frame(
                tick=tick,
                positions=current_positions,
                headings=current_headings,
                alive=current_alive,
                rounded=True,
                scores=scores,
                round_scores=round_scores,
                events=tick_events if include_events else None,
            )
        )

    return {
        "scope": SOURCE_BORDERLESS_WRAP_TRACE_SCOPE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "ruleset": SOURCE_BORDERLESS_WRAP_TRACE_SCOPE,
        "rules_hash": SOURCE_BORDERLESS_WRAP_RULES_HASH,
        "seed": scenario.seed,
        "scripted_actions": scenario.raw_action_script,
        "frames": frames,
    }


def _trace_source_borderless_print_manager_wrap(
    scenario: LoadedScenario,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    step_ms: float,
    map_size: float,
    player_ids: tuple[str, ...],
) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    current_positions = [[float(x), float(y)] for x, y in positions]
    current_headings = [float(heading) for heading in headings]
    current_alive = [bool(value) for value in alive]
    scores = [0 for _ in range(scenario.player_count)]
    round_scores = [0 for _ in range(scenario.player_count)]
    world_bodies: list[_SourceSeededBody] = []
    body_counts, body_nums = _source_initial_body_state(
        scenario.player_count,
        world_bodies,
        scenario.initial_state,
        player_ids,
    )
    (
        trail_point_counts,
        last_trail_points,
        trail_draw_points,
    ) = _source_initial_trail_runtime_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    printing = _source_initial_printing_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    print_managers = _source_initial_print_manager_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    world_body_count = len(world_bodies)
    include_events = _source_include_events(scenario)

    frames = [
        _source_kinematics_frame(
            tick=0,
            positions=current_positions,
            headings=current_headings,
            alive=current_alive,
            rounded=False,
            scores=scores,
            round_scores=round_scores,
            world_body_count=world_body_count,
            trail_point_counts=trail_point_counts,
            last_trail_points=last_trail_points,
            body_nums=body_nums,
            body_counts=body_counts,
            printing=printing,
            print_managers=print_managers,
        )
    ]

    for tick, actions in enumerate(scenario.raw_action_script, start=1):
        tick_events: list[dict[str, object]] = []
        for player_index in reversed(range(scenario.player_count)):
            if not current_alive[player_index]:
                continue
            move = actions[f"player_{player_index}"]
            current_headings[player_index] += (
                move * reference.angular_velocity_radians_per_ms * step_ms
            )
            distance = reference.avatar_velocity_units_per_s * step_ms / 1000.0
            current_positions[player_index][0] += (
                math.cos(current_headings[player_index]) * distance
            )
            current_positions[player_index][1] += (
                math.sin(current_headings[player_index]) * distance
            )
            body_nums[player_index] = body_counts[player_index]
            tick_events.append(
                _source_position_event(player_ids[player_index], current_positions[player_index])
            )

            if printing[player_index] and _source_is_time_to_draw(
                trail_draw_points[player_index],
                current_positions[player_index],
                reference.avatar_radius,
            ):
                _source_add_avatar_point(
                    world_bodies=world_bodies,
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    radius=reference.avatar_radius,
                    body_num=body_counts[player_index],
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                )
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )

            wrapped = _source_apply_borderless_wrap(current_positions[player_index], map_size)
            if wrapped:
                body_nums[player_index] = body_counts[player_index]
                tick_events.append(
                    _source_position_event(player_ids[player_index], current_positions[player_index])
                )

            if current_alive[player_index] and print_managers[player_index]["active"]:
                toggled, point_inserted, manager_world_delta = _source_update_print_manager(
                    manager=print_managers[player_index],
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    printing=printing,
                    world_bodies=world_bodies,
                    radius=reference.avatar_radius,
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                )
                world_body_count += manager_world_delta
                if point_inserted:
                    tick_events.append(
                        _source_point_event(
                            player_ids[player_index],
                            current_positions[player_index],
                            important=True,
                        )
                    )
                if toggled:
                    tick_events.append(
                        _source_property_event(
                            player_ids[player_index],
                            "printing",
                            printing[player_index],
                        )
                    )

        frames.append(
            _source_kinematics_frame(
                tick=tick,
                positions=current_positions,
                headings=current_headings,
                alive=current_alive,
                rounded=True,
                scores=scores,
                round_scores=round_scores,
                world_body_count=world_body_count,
                trail_point_counts=trail_point_counts,
                last_trail_points=last_trail_points,
                body_nums=body_nums,
                body_counts=body_counts,
                printing=printing,
                print_managers=print_managers,
                events=tick_events if include_events else None,
            )
        )

    return {
        "scope": SOURCE_BORDERLESS_WRAP_TRACE_SCOPE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "ruleset": SOURCE_BORDERLESS_WRAP_TRACE_SCOPE,
        "rules_hash": SOURCE_BORDERLESS_WRAP_RULES_HASH,
        "seed": scenario.seed,
        "scripted_actions": scenario.raw_action_script,
        "frames": frames,
    }


def _trace_source_borderless_body_skip(
    scenario: LoadedScenario,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    step_ms_values: list[float],
    map_size: float,
    seeded_bodies: list[_SourceSeededBody],
    player_ids: tuple[str, ...],
) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    current_positions = [[float(x), float(y)] for x, y in positions]
    current_headings = [float(heading) for heading in headings]
    current_alive = [bool(value) for value in alive]
    scores = [0 for _ in range(scenario.player_count)]
    round_scores = [0 for _ in range(scenario.player_count)]
    body_counts, body_nums = _source_initial_body_state(
        scenario.player_count,
        seeded_bodies,
        scenario.initial_state,
        player_ids,
    )
    (
        trail_point_counts,
        last_trail_points,
        trail_draw_points,
    ) = _source_initial_trail_runtime_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    printing = _source_initial_printing_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    world_bodies = list(seeded_bodies)
    world_body_count = len(world_bodies)
    include_events = _source_include_events(scenario)

    frames = [
        _source_kinematics_frame(
            tick=0,
            positions=current_positions,
            headings=current_headings,
            alive=current_alive,
            rounded=False,
            scores=scores,
            round_scores=round_scores,
            world_body_count=world_body_count,
            trail_point_counts=trail_point_counts,
            last_trail_points=last_trail_points,
            body_nums=body_nums,
            body_counts=body_counts,
        )
    ]

    for tick, (actions, step_ms) in enumerate(
        zip(scenario.raw_action_script, step_ms_values, strict=True),
        start=1,
    ):
        deaths = [False for _ in range(scenario.player_count)]
        tick_events: list[dict[str, object]] = []
        frame_start_deaths = sum(not value for value in current_alive)
        for player_index in reversed(range(scenario.player_count)):
            if not current_alive[player_index]:
                continue
            move = actions[f"player_{player_index}"]
            current_headings[player_index] += (
                move * reference.angular_velocity_radians_per_ms * step_ms
            )
            distance = reference.avatar_velocity_units_per_s * step_ms / 1000.0
            current_positions[player_index][0] += (
                math.cos(current_headings[player_index]) * distance
            )
            current_positions[player_index][1] += (
                math.sin(current_headings[player_index]) * distance
            )
            body_nums[player_index] = body_counts[player_index]
            tick_events.append(
                _source_position_event(player_ids[player_index], current_positions[player_index])
            )

            if printing[player_index] and _source_is_time_to_draw(
                trail_draw_points[player_index],
                current_positions[player_index],
                reference.avatar_radius,
            ):
                _source_add_avatar_point(
                    world_bodies=world_bodies,
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    radius=reference.avatar_radius,
                    body_num=body_counts[player_index],
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                )
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )

            hit_body: _SourceSeededBody | None = None
            wrapped = _source_apply_borderless_wrap(current_positions[player_index], map_size)
            if wrapped:
                body_nums[player_index] = body_counts[player_index]
                tick_events.append(
                    _source_position_event(player_ids[player_index], current_positions[player_index])
                )
            else:
                hit_body = _source_body_canary_hit_body(
                    player_index,
                    current_positions[player_index],
                    reference.avatar_radius,
                    body_nums[player_index],
                    world_bodies,
                    reference.trail_latency_points,
                )
                if hit_body is not None:
                    current_alive[player_index] = False
                    deaths[player_index] = True

            if deaths[player_index]:
                _source_add_avatar_point(
                    world_bodies=world_bodies,
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    radius=reference.avatar_radius,
                    body_num=body_counts[player_index],
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                )
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )
                tick_events.append(
                    _source_die_event(
                        player_ids[player_index],
                        killer_id=hit_body.owner_id if hit_body is not None else None,
                        old=_source_body_old(hit_body) if hit_body is not None else None,
                    )
                )
                round_scores[player_index] += frame_start_deaths
                tick_events.append(
                    _source_score_round_event(
                        player_ids[player_index],
                        scores[player_index],
                        round_scores[player_index],
                    )
                )

        if any(deaths) and sum(current_alive) <= 1:
            winner: int | None = None
            if sum(current_alive) == 1:
                winner = current_alive.index(True)
                round_scores[winner] += max(scenario.player_count - 1, 1)
                tick_events.append(
                    _source_score_round_event(
                        player_ids[winner],
                        scores[winner],
                        round_scores[winner],
                    )
                )
            for player_index in reversed(range(scenario.player_count)):
                scores[player_index] += round_scores[player_index]
                tick_events.append(
                    _source_score_event(
                        player_ids[player_index],
                        scores[player_index],
                        round_scores[player_index],
                    )
                )
            round_scores = [0 for _ in range(scenario.player_count)]
            tick_events.append(_source_round_end_event(None if winner is None else player_ids[winner]))

        frames.append(
            _source_kinematics_frame(
                tick=tick,
                step_ms=step_ms,
                positions=current_positions,
                headings=current_headings,
                alive=current_alive,
                rounded=True,
                scores=scores,
                round_scores=round_scores,
                world_body_count=world_body_count,
                trail_point_counts=trail_point_counts,
                last_trail_points=last_trail_points,
                body_nums=body_nums,
                body_counts=body_counts,
                events=tick_events if include_events else None,
            )
        )

    return {
        "scope": SOURCE_BORDERLESS_WRAP_TRACE_SCOPE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "ruleset": SOURCE_BORDERLESS_WRAP_TRACE_SCOPE,
        "rules_hash": SOURCE_BORDERLESS_WRAP_RULES_HASH,
        "seed": scenario.seed,
        "scripted_actions": scenario.raw_action_script,
        "frames": frames,
    }


def _trace_source_body_canary(
    scenario: LoadedScenario,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    step_ms: float,
    map_size: float,
    seeded_bodies: list[_SourceSeededBody],
    player_ids: tuple[str, ...],
) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    current_positions = [[float(x), float(y)] for x, y in positions]
    current_headings = [float(heading) for heading in headings]
    current_alive = [bool(value) for value in alive]
    scores = [0 for _ in range(scenario.player_count)]
    round_scores = [0 for _ in range(scenario.player_count)]
    body_counts, body_nums = _source_initial_body_state(
        scenario.player_count,
        seeded_bodies,
        scenario.initial_state,
        player_ids,
    )
    (
        trail_point_counts,
        last_trail_points,
        trail_draw_points,
    ) = _source_initial_trail_runtime_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    printing = _source_initial_printing_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    world_bodies = list(seeded_bodies)
    world_body_count = len(world_bodies)
    include_events = _source_include_events(scenario)

    frames = [
        _source_kinematics_frame(
            tick=0,
            positions=current_positions,
            headings=current_headings,
            alive=current_alive,
            rounded=False,
            scores=scores,
            round_scores=round_scores,
            world_body_count=world_body_count,
            trail_point_counts=trail_point_counts,
            last_trail_points=last_trail_points,
            body_nums=body_nums,
            body_counts=body_counts,
        )
    ]

    for tick, actions in enumerate(scenario.raw_action_script, start=1):
        deaths = [False for _ in range(scenario.player_count)]
        tick_events: list[dict[str, object]] = []
        frame_start_deaths = sum(not value for value in current_alive)
        for player_index in reversed(range(scenario.player_count)):
            if not current_alive[player_index]:
                continue
            move = actions[f"player_{player_index}"]
            current_headings[player_index] += (
                move * reference.angular_velocity_radians_per_ms * step_ms
            )
            distance = reference.avatar_velocity_units_per_s * step_ms / 1000.0
            current_positions[player_index][0] += (
                math.cos(current_headings[player_index]) * distance
            )
            current_positions[player_index][1] += (
                math.sin(current_headings[player_index]) * distance
            )
            body_nums[player_index] = body_counts[player_index]
            tick_events.append(
                _source_position_event(player_ids[player_index], current_positions[player_index])
            )

            if printing[player_index] and _source_is_time_to_draw(
                trail_draw_points[player_index],
                current_positions[player_index],
                reference.avatar_radius,
            ):
                _source_insert_avatar_body(
                    world_bodies,
                    player_index,
                    player_ids,
                    current_positions[player_index],
                    reference.avatar_radius,
                    body_nums[player_index],
                )
                trail_point_counts[player_index] += 1
                last_trail_points[player_index] = list(current_positions[player_index])
                trail_draw_points[player_index] = list(current_positions[player_index])
                body_counts[player_index] += 1
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )

            hit_body: _SourceSeededBody | None = None
            if _source_normal_wall_hit(
                current_positions[player_index],
                reference.avatar_radius,
                map_size,
            ):
                current_alive[player_index] = False
                deaths[player_index] = True
            else:
                hit_body = _source_body_canary_hit_body(
                    player_index,
                    current_positions[player_index],
                    reference.avatar_radius,
                    body_nums[player_index],
                    world_bodies,
                    reference.trail_latency_points,
                )
                if hit_body is not None:
                    current_alive[player_index] = False
                    deaths[player_index] = True

            if deaths[player_index]:
                _source_insert_avatar_body(
                    world_bodies,
                    player_index,
                    player_ids,
                    current_positions[player_index],
                    reference.avatar_radius,
                    body_counts[player_index],
                )
                trail_point_counts[player_index] += 1
                last_trail_points[player_index] = list(current_positions[player_index])
                trail_draw_points[player_index] = list(current_positions[player_index])
                body_counts[player_index] += 1
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )
                killer_id = hit_body.owner_id if hit_body is not None else None
                old = _source_body_old(hit_body) if hit_body is not None else None
                tick_events.append(
                    _source_die_event(player_ids[player_index], killer_id=killer_id, old=old)
                )
                round_scores[player_index] += frame_start_deaths
                tick_events.append(
                    _source_score_round_event(
                        player_ids[player_index],
                        scores[player_index],
                        round_scores[player_index],
                    )
                )

        if any(deaths) and sum(current_alive) <= 1:
            winner: int | None = None
            if sum(current_alive) == 1:
                winner = current_alive.index(True)
                round_scores[winner] += max(scenario.player_count - 1, 1)
                tick_events.append(
                    _source_score_round_event(
                        player_ids[winner],
                        scores[winner],
                        round_scores[winner],
                    )
                )
            for player_index in reversed(range(scenario.player_count)):
                scores[player_index] += round_scores[player_index]
                tick_events.append(
                    _source_score_event(
                        player_ids[player_index],
                        scores[player_index],
                        round_scores[player_index],
                    )
                )
            round_scores = [0 for _ in range(scenario.player_count)]
            tick_events.append(_source_round_end_event(None if winner is None else player_ids[winner]))

        frames.append(
            _source_kinematics_frame(
                tick=tick,
                positions=current_positions,
                headings=current_headings,
                alive=current_alive,
                rounded=True,
                scores=scores,
                round_scores=round_scores,
                world_body_count=world_body_count,
                trail_point_counts=trail_point_counts,
                last_trail_points=last_trail_points,
                body_nums=body_nums,
                body_counts=body_counts,
                events=tick_events if include_events else None,
            )
        )

    return {
        "scope": SOURCE_BODY_CANARY_TRACE_SCOPE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "ruleset": SOURCE_BODY_CANARY_TRACE_SCOPE,
        "rules_hash": SOURCE_BODY_CANARY_RULES_HASH,
        "seed": scenario.seed,
        "scripted_actions": scenario.raw_action_script,
        "frames": frames,
    }


def _trace_source_print_manager(
    scenario: LoadedScenario,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    step_ms: float,
    map_size: float,
    player_ids: tuple[str, ...],
) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    current_positions = [[float(x), float(y)] for x, y in positions]
    current_headings = [float(heading) for heading in headings]
    current_alive = [bool(value) for value in alive]
    scores = [0 for _ in range(scenario.player_count)]
    round_scores = [0 for _ in range(scenario.player_count)]
    world_bodies = _source_seeded_world_bodies(
        scenario,
        player_ids,
        require_non_empty=False,
    )
    body_counts, body_nums = _source_initial_body_state(
        scenario.player_count,
        world_bodies,
        scenario.initial_state,
        player_ids,
    )
    (
        trail_point_counts,
        last_trail_points,
        trail_draw_points,
    ) = _source_initial_trail_runtime_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    printing = _source_initial_printing_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    print_managers = _source_initial_print_manager_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    world_body_count = len(world_bodies)
    include_events = _source_include_events(scenario)
    random_source = _source_random_source(scenario)
    timer_advance_ms_values = _source_print_manager_timer_advance_ms_values(scenario)
    delayed_start_enabled = (
        scenario.scenario_id == _SOURCE_PRINT_MANAGER_DELAYED_START_SCENARIO_ID
    )
    delayed_start_elapsed_ms = 0.0
    delayed_start_fired = not delayed_start_enabled

    frames = [
        _source_kinematics_frame(
            tick=0,
            positions=current_positions,
            headings=current_headings,
            alive=current_alive,
            rounded=False,
            scores=scores,
            round_scores=round_scores,
            world_body_count=world_body_count,
            trail_point_counts=trail_point_counts,
            last_trail_points=last_trail_points,
            body_nums=body_nums,
            body_counts=body_counts,
            printing=printing,
            print_managers=print_managers,
        )
    ]

    for tick, (actions, timer_advance_ms) in enumerate(
        zip(scenario.raw_action_script, timer_advance_ms_values, strict=True),
        start=1,
    ):
        deaths = [False for _ in range(scenario.player_count)]
        tick_events: list[dict[str, object]] = []
        frame_start_deaths = sum(not value for value in current_alive)
        delayed_start_elapsed_ms += timer_advance_ms
        if not delayed_start_fired and delayed_start_elapsed_ms >= 3000:
            for player_index in reversed(range(scenario.player_count)):
                point_inserted, property_emitted, start_world_delta = (
                    _source_start_print_manager(
                        manager=print_managers[player_index],
                        player_index=player_index,
                        player_ids=player_ids,
                        position=current_positions[player_index],
                        printing=printing,
                        world_bodies=world_bodies,
                        radius=reference.avatar_radius,
                        trail_point_counts=trail_point_counts,
                        last_trail_points=last_trail_points,
                        trail_draw_points=trail_draw_points,
                        body_counts=body_counts,
                        random_source=random_source,
                    )
                )
                world_body_count += start_world_delta
                if point_inserted:
                    tick_events.append(
                        _source_point_event(
                            player_ids[player_index],
                            current_positions[player_index],
                            important=True,
                        )
                    )
                if property_emitted:
                    tick_events.append(
                        _source_property_event(
                            player_ids[player_index],
                            "printing",
                            printing[player_index],
                        )
                    )
            delayed_start_fired = True
        for player_index in reversed(range(scenario.player_count)):
            if not current_alive[player_index]:
                continue
            move = actions["player_0"]
            if scenario.player_count > 1:
                move = actions[f"player_{player_index}"]
            current_headings[player_index] += (
                move * reference.angular_velocity_radians_per_ms * step_ms
            )
            distance = reference.avatar_velocity_units_per_s * step_ms / 1000.0
            current_positions[player_index][0] += (
                math.cos(current_headings[player_index]) * distance
            )
            current_positions[player_index][1] += (
                math.sin(current_headings[player_index]) * distance
            )
            body_nums[player_index] = body_counts[player_index]
            tick_events.append(
                _source_position_event(player_ids[player_index], current_positions[player_index])
            )

            if printing[player_index] and _source_is_time_to_draw(
                trail_draw_points[player_index],
                current_positions[player_index],
                reference.avatar_radius,
            ):
                _source_add_avatar_point(
                    world_bodies=world_bodies,
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    radius=reference.avatar_radius,
                    body_num=body_counts[player_index],
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                )
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )

            hit_body: _SourceSeededBody | None = None
            if _source_normal_wall_hit(
                current_positions[player_index],
                reference.avatar_radius,
                map_size,
            ):
                current_alive[player_index] = False
                deaths[player_index] = True
            else:
                hit_body = _source_body_canary_hit_body(
                    player_index,
                    current_positions[player_index],
                    reference.avatar_radius,
                    body_nums[player_index],
                    world_bodies,
                    reference.trail_latency_points,
                )
                if hit_body is not None:
                    current_alive[player_index] = False
                    deaths[player_index] = True

            if deaths[player_index]:
                _source_add_avatar_point(
                    world_bodies=world_bodies,
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    radius=reference.avatar_radius,
                    body_num=body_counts[player_index],
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                )
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )
                point_inserted, property_emitted, stop_world_delta = (
                    _source_stop_print_manager_after_death(
                        manager=print_managers[player_index],
                        player_index=player_index,
                        player_ids=player_ids,
                        position=current_positions[player_index],
                        printing=printing,
                        world_bodies=world_bodies,
                        radius=reference.avatar_radius,
                        trail_point_counts=trail_point_counts,
                        last_trail_points=last_trail_points,
                        trail_draw_points=trail_draw_points,
                        body_counts=body_counts,
                        random_source=random_source,
                    )
                )
                world_body_count += stop_world_delta
                if point_inserted:
                    tick_events.append(
                        _source_point_event(
                            player_ids[player_index],
                            current_positions[player_index],
                            important=True,
                        )
                    )
                if property_emitted:
                    tick_events.append(
                        _source_property_event(
                            player_ids[player_index],
                            "printing",
                            printing[player_index],
                        )
                    )
                tick_events.append(
                    _source_die_event(
                        player_ids[player_index],
                        killer_id=hit_body.owner_id if hit_body is not None else None,
                        old=_source_body_old(hit_body) if hit_body is not None else None,
                    )
                )
                round_scores[player_index] += frame_start_deaths
                tick_events.append(
                    _source_score_round_event(
                        player_ids[player_index],
                        scores[player_index],
                        round_scores[player_index],
                    )
                )
            elif print_managers[player_index]["active"]:
                toggled, point_inserted, manager_world_delta = _source_update_print_manager(
                    manager=print_managers[player_index],
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    printing=printing,
                    world_bodies=world_bodies,
                    radius=reference.avatar_radius,
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                    random_source=random_source,
                )
                world_body_count += manager_world_delta
                if point_inserted:
                    tick_events.append(
                        _source_point_event(
                            player_ids[player_index],
                            current_positions[player_index],
                            important=True,
                        )
                    )
                if toggled:
                    tick_events.append(
                        _source_property_event(
                            player_ids[player_index],
                            "printing",
                            printing[player_index],
                        )
                    )

        frames.append(
            _source_kinematics_frame(
                tick=tick,
                positions=current_positions,
                headings=current_headings,
                alive=current_alive,
                rounded=True,
                scores=scores,
                round_scores=round_scores,
                world_body_count=world_body_count,
                trail_point_counts=trail_point_counts,
                last_trail_points=last_trail_points,
                body_nums=body_nums,
                body_counts=body_counts,
                printing=printing,
                print_managers=print_managers,
                events=tick_events if include_events else None,
            )
        )

    return {
        "scope": SOURCE_PRINT_MANAGER_TRACE_SCOPE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "ruleset": SOURCE_PRINT_MANAGER_TRACE_SCOPE,
        "rules_hash": SOURCE_PRINT_MANAGER_RULES_HASH,
        "seed": scenario.seed,
        "scripted_actions": scenario.raw_action_script,
        "randomCalls": list(random_source.calls),
        "frames": frames,
    }


def _trace_source_trail_cadence(
    scenario: LoadedScenario,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    step_ms: float,
    map_size: float,
    player_ids: tuple[str, ...],
) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    current_positions = [[float(x), float(y)] for x, y in positions]
    current_headings = [float(heading) for heading in headings]
    current_alive = [bool(value) for value in alive]
    scores = [0 for _ in range(scenario.player_count)]
    round_scores = [0 for _ in range(scenario.player_count)]
    body_counts = [0 for _ in range(scenario.player_count)]
    body_nums = [0 for _ in range(scenario.player_count)]
    (
        trail_point_counts,
        last_trail_points,
        trail_draw_points,
    ) = _source_initial_trail_runtime_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    printing = _source_initial_printing_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    world_body_count = 0
    include_events = _source_include_events(scenario)

    frames = [
        _source_kinematics_frame(
            tick=0,
            positions=current_positions,
            headings=current_headings,
            alive=current_alive,
            rounded=False,
            scores=scores,
            round_scores=round_scores,
            world_body_count=world_body_count,
            trail_point_counts=trail_point_counts,
            last_trail_points=last_trail_points,
            body_nums=body_nums,
            body_counts=body_counts,
            printing=printing,
        )
    ]

    for tick, actions in enumerate(scenario.raw_action_script, start=1):
        tick_events: list[dict[str, object]] = []
        player_index = 0
        if current_alive[player_index]:
            move = actions["player_0"]
            current_headings[player_index] += (
                move * reference.angular_velocity_radians_per_ms * step_ms
            )
            distance = reference.avatar_velocity_units_per_s * step_ms / 1000.0
            current_positions[player_index][0] += (
                math.cos(current_headings[player_index]) * distance
            )
            current_positions[player_index][1] += (
                math.sin(current_headings[player_index]) * distance
            )
            body_nums[player_index] = body_counts[player_index]
            tick_events.append(
                _source_position_event(player_ids[player_index], current_positions[player_index])
            )

            if printing[player_index] and _source_is_time_to_draw(
                trail_draw_points[player_index],
                current_positions[player_index],
                reference.avatar_radius,
            ):
                trail_point_counts[player_index] += 1
                last_trail_points[player_index] = list(current_positions[player_index])
                trail_draw_points[player_index] = list(current_positions[player_index])
                body_counts[player_index] += 1
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )

            if _source_normal_wall_hit(
                current_positions[player_index],
                reference.avatar_radius,
                map_size,
            ):
                current_alive[player_index] = False

        frames.append(
            _source_kinematics_frame(
                tick=tick,
                positions=current_positions,
                headings=current_headings,
                alive=current_alive,
                rounded=True,
                scores=scores,
                round_scores=round_scores,
                world_body_count=world_body_count,
                trail_point_counts=trail_point_counts,
                last_trail_points=last_trail_points,
                body_nums=body_nums,
                body_counts=body_counts,
                printing=printing,
                events=tick_events if include_events else None,
            )
        )

    return {
        "scope": SOURCE_TRAIL_CADENCE_TRACE_SCOPE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "ruleset": SOURCE_TRAIL_CADENCE_TRACE_SCOPE,
        "rules_hash": SOURCE_TRAIL_CADENCE_RULES_HASH,
        "seed": scenario.seed,
        "scripted_actions": scenario.raw_action_script,
        "frames": frames,
    }


def _trace_source_trail_gap(
    scenario: LoadedScenario,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    step_ms: float,
    map_size: float,
    seeded_bodies: list[_SourceSeededBody],
    player_ids: tuple[str, ...],
) -> dict[str, object]:
    reference = CurvyTronReferenceDefaults()
    current_positions = [[float(x), float(y)] for x, y in positions]
    current_headings = [float(heading) for heading in headings]
    current_alive = [bool(value) for value in alive]
    scores = [0 for _ in range(scenario.player_count)]
    round_scores = [0 for _ in range(scenario.player_count)]
    body_counts, body_nums = _source_initial_body_state(
        scenario.player_count,
        seeded_bodies,
        scenario.initial_state,
        player_ids,
    )
    (
        trail_point_counts,
        last_trail_points,
        trail_draw_points,
    ) = _source_initial_trail_runtime_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    printing = _source_initial_printing_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    print_managers = _source_initial_print_manager_state(
        scenario.player_count,
        scenario.initial_state,
        player_ids,
    )
    world_bodies = list(seeded_bodies)
    world_body_count = len(world_bodies)
    include_events = _source_include_events(scenario)
    random_source = _source_random_source(scenario)

    frames = [
        _source_kinematics_frame(
            tick=0,
            positions=current_positions,
            headings=current_headings,
            alive=current_alive,
            rounded=False,
            scores=scores,
            round_scores=round_scores,
            world_body_count=world_body_count,
            trail_point_counts=trail_point_counts,
            last_trail_points=last_trail_points,
            body_nums=body_nums,
            body_counts=body_counts,
            printing=printing,
            print_managers=print_managers,
        )
    ]

    for tick, actions in enumerate(scenario.raw_action_script, start=1):
        deaths = [False for _ in range(scenario.player_count)]
        tick_events: list[dict[str, object]] = []
        frame_start_deaths = sum(not value for value in current_alive)
        for player_index in reversed(range(scenario.player_count)):
            if not current_alive[player_index]:
                continue
            move = actions[f"player_{player_index}"]
            current_headings[player_index] += (
                move * reference.angular_velocity_radians_per_ms * step_ms
            )
            distance = reference.avatar_velocity_units_per_s * step_ms / 1000.0
            current_positions[player_index][0] += (
                math.cos(current_headings[player_index]) * distance
            )
            current_positions[player_index][1] += (
                math.sin(current_headings[player_index]) * distance
            )
            body_nums[player_index] = body_counts[player_index]
            tick_events.append(
                _source_position_event(player_ids[player_index], current_positions[player_index])
            )

            if printing[player_index] and _source_is_time_to_draw(
                trail_draw_points[player_index],
                current_positions[player_index],
                reference.avatar_radius,
            ):
                _source_add_avatar_point(
                    world_bodies=world_bodies,
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    radius=reference.avatar_radius,
                    body_num=body_counts[player_index],
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                )
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )

            hit_body: _SourceSeededBody | None = None
            if _source_normal_wall_hit(
                current_positions[player_index],
                reference.avatar_radius,
                map_size,
            ):
                current_alive[player_index] = False
                deaths[player_index] = True
            else:
                hit_body = _source_body_canary_hit_body(
                    player_index,
                    current_positions[player_index],
                    reference.avatar_radius,
                    body_nums[player_index],
                    world_bodies,
                    reference.trail_latency_points,
                )
                if hit_body is not None:
                    current_alive[player_index] = False
                    deaths[player_index] = True

            if deaths[player_index]:
                _source_add_avatar_point(
                    world_bodies=world_bodies,
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    radius=reference.avatar_radius,
                    body_num=body_counts[player_index],
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                )
                world_body_count += 1
                tick_events.append(
                    _source_point_event(player_ids[player_index], current_positions[player_index])
                )
                killer_id = hit_body.owner_id if hit_body is not None else None
                old = _source_body_old(hit_body) if hit_body is not None else None
                tick_events.append(
                    _source_die_event(player_ids[player_index], killer_id=killer_id, old=old)
                )
                round_scores[player_index] += frame_start_deaths
                tick_events.append(
                    _source_score_round_event(
                        player_ids[player_index],
                        scores[player_index],
                        round_scores[player_index],
                    )
                )

            if current_alive[player_index] and print_managers[player_index]["active"]:
                toggled, point_inserted, manager_world_delta = _source_update_print_manager(
                    manager=print_managers[player_index],
                    player_index=player_index,
                    player_ids=player_ids,
                    position=current_positions[player_index],
                    printing=printing,
                    world_bodies=world_bodies,
                    radius=reference.avatar_radius,
                    trail_point_counts=trail_point_counts,
                    last_trail_points=last_trail_points,
                    trail_draw_points=trail_draw_points,
                    body_counts=body_counts,
                    random_source=random_source,
                )
                world_body_count += manager_world_delta
                if point_inserted:
                    tick_events.append(
                        _source_point_event(
                            player_ids[player_index],
                            current_positions[player_index],
                            important=True,
                        )
                    )
                if toggled:
                    tick_events.append(
                        _source_property_event(
                            player_ids[player_index],
                            "printing",
                            printing[player_index],
                        )
                    )

        frames.append(
            _source_kinematics_frame(
                tick=tick,
                positions=current_positions,
                headings=current_headings,
                alive=current_alive,
                rounded=True,
                scores=scores,
                round_scores=round_scores,
                world_body_count=world_body_count,
                trail_point_counts=trail_point_counts,
                last_trail_points=last_trail_points,
                body_nums=body_nums,
                body_counts=body_counts,
                printing=printing,
                print_managers=print_managers,
                events=tick_events if include_events else None,
            )
        )

    return {
        "scope": SOURCE_TRAIL_GAP_TRACE_SCOPE,
        "schema_version": TRACE_SCHEMA_VERSION,
        "ruleset": SOURCE_TRAIL_GAP_TRACE_SCOPE,
        "rules_hash": SOURCE_TRAIL_GAP_RULES_HASH,
        "seed": scenario.seed,
        "scripted_actions": scenario.raw_action_script,
        "randomCalls": list(random_source.calls),
        "frames": frames,
    }


def _is_source_kinematics_movement_scenario(scenario_id: str) -> bool:
    return scenario_id == "forced_two_player_turn_step" or scenario_id.startswith(
        "source_kinematics_"
    )


def _source_kinematics_frame(
    *,
    tick: int,
    step_ms: float | None = None,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    rounded: bool,
    scores: list[int] | None = None,
    round_scores: list[int] | None = None,
    world_body_count: int | None = None,
    trail_point_counts: list[int] | None = None,
    last_trail_points: list[list[float] | None] | None = None,
    body_nums: list[int] | None = None,
    body_counts: list[int] | None = None,
    printing: list[bool] | None = None,
    print_managers: list[dict[str, object]] | None = None,
    events: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if rounded:
        frame_positions = [
            [_source_round(position[0]), _source_round(position[1])] for position in positions
        ]
        frame_headings = [_source_round(heading) for heading in headings]
    else:
        frame_positions = [[float(position[0]), float(position[1])] for position in positions]
        frame_headings = [float(heading) for heading in headings]

    frame: dict[str, object] = {
        "tick": tick,
        "positions": frame_positions,
        "headings": frame_headings,
        "alive": [bool(value) for value in alive],
        "scores": list(scores) if scores is not None else [0 for _ in positions],
        "roundScores": list(round_scores) if round_scores is not None else [0 for _ in positions],
        "rewards": None,
        "terminated": None,
        "truncated": None,
    }
    if events is not None:
        frame["events"] = events
    if step_ms is not None:
        frame["stepMs"] = _source_round(float(step_ms))
    if world_body_count is not None:
        frame["worldBodyCount"] = int(world_body_count)
    if trail_point_counts is not None:
        frame["trailPointCounts"] = list(trail_point_counts)
    if last_trail_points is not None:
        frame["lastTrailPoints"] = [
            None
            if point is None
            else [_source_round(point[0]), _source_round(point[1])]
            for point in last_trail_points
        ]
    if body_nums is not None:
        frame["bodyNums"] = list(body_nums)
    if body_counts is not None:
        frame["bodyCounts"] = list(body_counts)
    if printing is not None:
        frame["printing"] = [bool(value) for value in printing]
    if print_managers is not None:
        frame["printManagers"] = [
            {
                "active": bool(manager["active"]),
                "distance": _source_round(float(manager["distance"])),
                "lastX": _source_round(float(manager["lastX"])),
                "lastY": _source_round(float(manager["lastY"])),
            }
            for manager in print_managers
        ]
    return frame


def _source_kinematics_step_ms(scenario: LoadedScenario) -> float:
    time_policy = scenario.time_policy
    if time_policy.get("kind") != "fixed":
        raise ScenarioError("source-kinematics runner requires fixed time_policy")
    value = time_policy.get("step_ms")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenarioError("source-kinematics runner requires time_policy.step_ms")
    return float(value)


def _source_kinematics_step_ms_values(scenario: LoadedScenario) -> list[float]:
    values = _source_step_ms_values(scenario, runner_name="source-kinematics")
    total = scenario.time_policy.get("total_step_ms")
    if total is None:
        return values
    if isinstance(total, bool) or not isinstance(total, int | float):
        raise ScenarioError("source-kinematics runner requires finite time_policy.total_step_ms")
    if not math.isfinite(float(total)):
        raise ScenarioError("source-kinematics runner requires finite time_policy.total_step_ms")
    if not math.isclose(sum(values), float(total), rel_tol=0.0, abs_tol=1e-9):
        raise ScenarioError(
            "source-kinematics runner requires time_policy.total_step_ms "
            "to match time_policy.step_ms_sequence"
        )
    return values


def _source_step_ms_values(scenario: LoadedScenario, *, runner_name: str) -> list[float]:
    time_policy = scenario.time_policy
    kind = time_policy.get("kind")
    if kind == "fixed":
        return [_source_kinematics_step_ms(scenario) for _ in scenario.raw_action_script]
    if kind == "per-step":
        values = time_policy.get("step_ms_sequence")
        if not isinstance(values, list) or len(values) != len(scenario.raw_action_script):
            raise ScenarioError(
                f"{runner_name} runner requires one time_policy.step_ms_sequence value per step"
            )
        result = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ScenarioError(
                    f"{runner_name} runner requires finite time_policy.step_ms_sequence[{index}]"
                )
            step_ms = float(value)
            if not math.isfinite(step_ms):
                raise ScenarioError(
                    f"{runner_name} runner requires finite time_policy.step_ms_sequence[{index}]"
                )
            result.append(step_ms)
        return result
    raise ScenarioError(f"{runner_name} runner requires fixed or per-step time_policy")


def _source_map_size(scenario: LoadedScenario) -> float:
    map_size = scenario.initial_state.get("map_size", scenario.initial_state.get("size"))
    if isinstance(map_size, bool) or not isinstance(map_size, int | float):
        raise ScenarioError("source-normal-wall runner requires initial_state.map_size")
    return float(map_size)


def _validate_source_player_state_lengths(
    scenario: LoadedScenario,
    *,
    positions: list[list[float]],
    headings: list[float],
    alive: list[bool],
    runner_name: str,
) -> None:
    if len(positions) != scenario.player_count:
        raise ScenarioError(
            f"{runner_name} runner requires one forced position per player"
        )
    if len(headings) != scenario.player_count:
        raise ScenarioError(
            f"{runner_name} runner requires one forced heading per player"
        )
    if len(alive) != scenario.player_count:
        raise ScenarioError(
            f"{runner_name} runner requires one alive flag per player"
        )


def _source_normal_wall_hit(position: list[float], radius: float, map_size: float) -> bool:
    x, y = position
    return x - radius < 0 or x + radius > map_size or y - radius < 0 or y + radius > map_size


def _source_apply_borderless_wrap(position: list[float], map_size: float) -> bool:
    x, y = position
    if x < 0:
        position[0] = map_size
        return True
    elif x > map_size:
        position[0] = 0.0
        return True
    elif y < 0:
        position[1] = map_size
        return True
    elif y > map_size:
        position[1] = 0.0
        return True
    return False


def _source_seeded_world_bodies(
    scenario: LoadedScenario,
    player_ids: tuple[str, ...],
    *,
    require_non_empty: bool = True,
) -> list[_SourceSeededBody]:
    raw_bodies = scenario.initial_state.get("world_bodies")
    if raw_bodies is None and not require_non_empty:
        return []
    if not isinstance(raw_bodies, list) or (require_non_empty and not raw_bodies):
        raise ScenarioError("source-body-canary runner requires initial_state.world_bodies")

    bodies: list[_SourceSeededBody] = []
    for index, raw_body in enumerate(raw_bodies):
        if not isinstance(raw_body, Mapping):
            raise ScenarioError("initial_state.world_bodies entries must be objects")
        owner = _source_world_body_owner(raw_body)
        if owner is None or owner == "":
            raise ScenarioError(f"initial_state.world_bodies[{index}].player_id is required")
        owner_index = _source_player_index(owner, player_ids)
        bodies.append(
            _SourceSeededBody(
                owner_index=owner_index,
                owner_id=player_ids[owner_index],
                x=float(raw_body["x"]),
                y=float(raw_body["y"]),
                radius=float(raw_body["radius"]),
                num=int(raw_body["num"]),
                age_ms=float(raw_body.get("age_ms", raw_body.get("ageMs", 0.0))),
            )
        )
    return bodies


def _source_insert_avatar_body(
    world_bodies: list[_SourceSeededBody],
    player_index: int,
    player_ids: tuple[str, ...],
    position: list[float],
    radius: float,
    body_num: int,
) -> None:
    world_bodies.append(
        _SourceSeededBody(
            owner_index=player_index,
            owner_id=player_ids[player_index],
            x=float(position[0]),
            y=float(position[1]),
            radius=float(radius),
            num=int(body_num),
            age_ms=0.0,
        )
    )


def _source_add_avatar_point(
    *,
    world_bodies: list[_SourceSeededBody],
    player_index: int,
    player_ids: tuple[str, ...],
    position: list[float],
    radius: float,
    body_num: int,
    trail_point_counts: list[int],
    last_trail_points: list[list[float] | None],
    trail_draw_points: list[list[float] | None],
    body_counts: list[int],
) -> None:
    _source_insert_avatar_body(
        world_bodies,
        player_index,
        player_ids,
        position,
        radius,
        body_num,
    )
    trail_point_counts[player_index] += 1
    last_trail_points[player_index] = list(position)
    trail_draw_points[player_index] = list(position)
    body_counts[player_index] += 1


def _source_update_print_manager(
    *,
    manager: dict[str, object],
    player_index: int,
    player_ids: tuple[str, ...],
    position: list[float],
    printing: list[bool],
    world_bodies: list[_SourceSeededBody],
    radius: float,
    trail_point_counts: list[int],
    last_trail_points: list[list[float] | None],
    trail_draw_points: list[list[float] | None],
    body_counts: list[int],
    random_source: _SourceRandomSource | None = None,
) -> tuple[bool, bool, int]:
    manager["distance"] = float(manager["distance"]) - math.hypot(
        float(manager["lastX"]) - position[0],
        float(manager["lastY"]) - position[1],
    )
    manager["lastX"] = position[0]
    manager["lastY"] = position[1]

    if float(manager["distance"]) > 0:
        return False, False, 0

    printing[player_index] = not printing[player_index]
    _source_add_avatar_point(
        world_bodies=world_bodies,
        player_index=player_index,
        player_ids=player_ids,
        position=position,
        radius=radius,
        body_num=body_counts[player_index],
        trail_point_counts=trail_point_counts,
        last_trail_points=last_trail_points,
        trail_draw_points=trail_draw_points,
        body_counts=body_counts,
    )
    if not printing[player_index]:
        trail_point_counts[player_index] = 0
        last_trail_points[player_index] = None
        trail_draw_points[player_index] = None
    manager["distance"] = _source_print_manager_random_distance(
        printing[player_index],
        random_source=random_source,
    )
    return True, True, 1


def _source_start_print_manager(
    *,
    manager: dict[str, object],
    player_index: int,
    player_ids: tuple[str, ...],
    position: list[float],
    printing: list[bool],
    world_bodies: list[_SourceSeededBody],
    radius: float,
    trail_point_counts: list[int],
    last_trail_points: list[list[float] | None],
    trail_draw_points: list[list[float] | None],
    body_counts: list[int],
    random_source: _SourceRandomSource | None = None,
) -> tuple[bool, bool, int]:
    if manager["active"]:
        return False, False, 0

    manager["active"] = True
    manager["lastX"] = position[0]
    manager["lastY"] = position[1]
    point_inserted = False
    world_delta = 0
    if not printing[player_index]:
        printing[player_index] = True
        _source_add_avatar_point(
            world_bodies=world_bodies,
            player_index=player_index,
            player_ids=player_ids,
            position=position,
            radius=radius,
            body_num=body_counts[player_index],
            trail_point_counts=trail_point_counts,
            last_trail_points=last_trail_points,
            trail_draw_points=trail_draw_points,
            body_counts=body_counts,
        )
        point_inserted = True
        world_delta = 1
    manager["distance"] = _source_print_manager_random_distance(
        printing[player_index],
        random_source=random_source,
    )
    return point_inserted, True, world_delta


def _source_stop_print_manager_after_death(
    *,
    manager: dict[str, object],
    player_index: int,
    player_ids: tuple[str, ...],
    position: list[float],
    printing: list[bool],
    world_bodies: list[_SourceSeededBody],
    radius: float,
    trail_point_counts: list[int],
    last_trail_points: list[list[float] | None],
    trail_draw_points: list[list[float] | None],
    body_counts: list[int],
    random_source: _SourceRandomSource | None = None,
) -> tuple[bool, bool, int]:
    if not manager["active"]:
        return False, False, 0

    manager["active"] = False
    point_inserted = False
    world_delta = 0
    if printing[player_index]:
        printing[player_index] = False
        _source_add_avatar_point(
            world_bodies=world_bodies,
            player_index=player_index,
            player_ids=player_ids,
            position=position,
            radius=radius,
            body_num=body_counts[player_index],
            trail_point_counts=trail_point_counts,
            last_trail_points=last_trail_points,
            trail_draw_points=trail_draw_points,
            body_counts=body_counts,
        )
        trail_point_counts[player_index] = 0
        last_trail_points[player_index] = None
        trail_draw_points[player_index] = None
        point_inserted = True
        world_delta = 1

    _source_print_manager_random_distance(
        printing[player_index],
        random_source=random_source,
    )
    manager["active"] = False
    manager["distance"] = 0.0
    manager["lastX"] = 0.0
    manager["lastY"] = 0.0
    return point_inserted, True, world_delta


def _source_world_body_owner(body: Mapping[str, Any]) -> Any:
    for key in (
        "player_id",
        "playerId",
        "owner_id",
        "ownerId",
        "avatar_id",
        "avatarId",
        "avatar",
    ):
        if key in body:
            return body[key]
    return None


def _source_player_index(owner: Any, player_ids: tuple[str, ...]) -> int:
    text = str(owner)
    for index, player_id in enumerate(player_ids):
        if text in {player_id, f"player_{index}", f"p{index}", str(index), str(index + 1)}:
            return index
    raise ScenarioError(f"initial_state.world_bodies references unknown player {owner}")


def _source_body_canary_hit_body(
    player_index: int,
    position: list[float],
    radius: float,
    body_num: int,
    seeded_bodies: list[_SourceSeededBody],
    trail_latency: int,
) -> _SourceSeededBody | None:
    for body in seeded_bodies:
        own_body = body.owner_index == player_index
        if own_body and body_num - body.num <= trail_latency:
            continue
        distance = math.hypot(position[0] - body.x, position[1] - body.y)
        if distance < radius + body.radius:
            return body
    return None


def _source_body_old(body: _SourceSeededBody) -> bool:
    return body.age_ms >= 2000.0


def _source_initial_body_state(
    player_count: int,
    seeded_bodies: list[_SourceSeededBody],
    initial_state: Mapping[str, Any],
    player_ids: tuple[str, ...],
) -> tuple[list[int], list[int]]:
    body_counts = [0 for _ in range(player_count)]
    for body in seeded_bodies:
        body_counts[body.owner_index] = max(body_counts[body.owner_index], body.num + 1)
    body_nums = [0 for _ in range(player_count)]

    players = initial_state.get("players")
    if not isinstance(players, list):
        return body_counts, body_nums

    for fallback_index, raw_player in enumerate(players[:player_count]):
        if not isinstance(raw_player, Mapping):
            continue
        player_index = _source_initial_player_index(raw_player, fallback_index, player_ids)
        body_count = _source_initial_player_counter(raw_player, ("body_count", "bodyCount"))
        body_num = _source_initial_player_counter(raw_player, ("body_num", "bodyNum"))
        if body_count is not None:
            body_counts[player_index] = body_count
            body_nums[player_index] = body_count
        if body_num is not None:
            body_nums[player_index] = body_num
    return body_counts, body_nums


def _source_initial_trail_state(
    player_count: int,
    initial_state: Mapping[str, Any],
    player_ids: tuple[str, ...],
) -> tuple[list[int], list[list[float] | None]]:
    trail_point_counts, last_trail_points, _trail_draw_points = (
        _source_initial_trail_runtime_state(player_count, initial_state, player_ids)
    )
    return trail_point_counts, last_trail_points


def _source_initial_trail_runtime_state(
    player_count: int,
    initial_state: Mapping[str, Any],
    player_ids: tuple[str, ...],
) -> tuple[list[int], list[list[float] | None], list[list[float] | None]]:
    trail_point_counts = [0 for _ in range(player_count)]
    last_trail_points: list[list[float] | None] = [None for _ in range(player_count)]
    trail_draw_points: list[list[float] | None] = [None for _ in range(player_count)]

    players = initial_state.get("players")
    if not isinstance(players, list):
        return trail_point_counts, last_trail_points, trail_draw_points

    for fallback_index, raw_player in enumerate(players[:player_count]):
        if not isinstance(raw_player, Mapping):
            continue
        state = _source_initial_player_state(raw_player)
        trail = state.get("trail")
        if trail is None:
            continue
        if not isinstance(trail, Mapping):
            raise ScenarioError(
                f"initial_state.players[{fallback_index}].initial.trail must be an object"
            )
        player_index = _source_initial_player_index(raw_player, fallback_index, player_ids)
        points = trail.get("points")
        if points is not None:
            if not isinstance(points, list):
                raise ScenarioError(
                    f"initial_state.players[{fallback_index}].initial.trail.points must be a list"
                )
            trail_point_counts[player_index] = len(points)
            if points:
                point = _source_trail_point(
                    points[-1],
                    (
                        "initial_state.players"
                        f"[{fallback_index}].initial.trail.points[{len(points) - 1}]"
                    ),
                )
                last_trail_points[player_index] = point
                trail_draw_points[player_index] = point
        last_x = trail.get("last_x", trail.get("lastX"))
        last_y = trail.get("last_y", trail.get("lastY"))
        if last_x is not None or last_y is not None:
            trail_draw_points[player_index] = [
                _source_required_finite_number(
                    last_x,
                    f"initial_state.players[{fallback_index}].initial.trail.last_x",
                ),
                _source_required_finite_number(
                    last_y,
                    f"initial_state.players[{fallback_index}].initial.trail.last_y",
                ),
            ]
    return trail_point_counts, last_trail_points, trail_draw_points


def _source_initial_printing_state(
    player_count: int,
    initial_state: Mapping[str, Any],
    player_ids: tuple[str, ...],
) -> list[bool]:
    printing = [False for _ in range(player_count)]

    players = initial_state.get("players")
    if not isinstance(players, list):
        return printing

    for fallback_index, raw_player in enumerate(players[:player_count]):
        if not isinstance(raw_player, Mapping):
            continue
        state = _source_initial_player_state(raw_player)
        if "printing" not in state:
            continue
        if not isinstance(state["printing"], bool):
            raise ScenarioError(
                f"initial_state.players[{fallback_index}].initial.printing must be a boolean"
            )
        player_index = _source_initial_player_index(raw_player, fallback_index, player_ids)
        printing[player_index] = state["printing"]
    return printing


def _source_initial_print_manager_state(
    player_count: int,
    initial_state: Mapping[str, Any],
    player_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    managers: list[dict[str, object]] = [
        {"active": False, "distance": 0.0, "lastX": 0.0, "lastY": 0.0}
        for _ in range(player_count)
    ]

    players = initial_state.get("players")
    if not isinstance(players, list):
        return managers

    for fallback_index, raw_player in enumerate(players[:player_count]):
        if not isinstance(raw_player, Mapping):
            continue
        state = _source_initial_player_state(raw_player)
        manager = state.get("print_manager", state.get("printManager"))
        if manager is None:
            continue
        if not isinstance(manager, Mapping):
            raise ScenarioError(
                f"initial_state.players[{fallback_index}].initial.print_manager must be an object"
            )
        player_index = _source_initial_player_index(raw_player, fallback_index, player_ids)
        managers[player_index] = {
            "active": _source_print_manager_bool(
                manager,
                "active",
                f"initial_state.players[{fallback_index}].initial.print_manager.active",
            ),
            "distance": _source_print_manager_number(
                manager,
                "distance",
                f"initial_state.players[{fallback_index}].initial.print_manager.distance",
            ),
            "lastX": _source_print_manager_number(
                manager,
                ("last_x", "lastX"),
                f"initial_state.players[{fallback_index}].initial.print_manager.last_x",
            ),
            "lastY": _source_print_manager_number(
                manager,
                ("last_y", "lastY"),
                f"initial_state.players[{fallback_index}].initial.print_manager.last_y",
            ),
        }
    return managers


def _source_print_manager_bool(
    manager: Mapping[str, Any],
    key: str,
    field: str,
) -> bool:
    value = manager.get(key)
    if not isinstance(value, bool):
        raise ScenarioError(f"{field} must be a boolean")
    return value


def _source_print_manager_number(
    manager: Mapping[str, Any],
    keys: str | tuple[str, str],
    field: str,
) -> float:
    key_tuple = (keys,) if isinstance(keys, str) else keys
    value = None
    for key in key_tuple:
        if key in manager:
            value = manager[key]
            break
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenarioError(f"{field} must be a finite number")
    if not math.isfinite(float(value)):
        raise ScenarioError(f"{field} must be a finite number")
    return float(value)


def _source_print_manager_random_distance(
    printing: bool,
    *,
    random_source: _SourceRandomSource | None = None,
) -> float:
    random_value = random_source.random() if random_source is not None else 0.5
    if printing:
        return 60 * (0.3 + random_value * 0.7)
    return 5 * (0.8 + random_value * 0.5)


def _source_random_source(scenario: LoadedScenario) -> _SourceRandomSource:
    source_setup = scenario.source_setup
    raw_random = source_setup.get("random") if isinstance(source_setup, Mapping) else None
    random_setup = raw_random if isinstance(raw_random, Mapping) else {}

    raw_sequence = _source_first_owned(
        random_setup,
        (
            "math_random_sequence",
            "mathRandomSequence",
            "math_random_tape",
            "mathRandomTape",
        ),
    )
    if raw_sequence is not _SOURCE_MISSING:
        if not isinstance(raw_sequence, list):
            raise ScenarioError("source_setup.random.math_random_sequence must be an array")
        return _SourceRandomSource(
            sequence=tuple(
                _source_required_random_value(
                    value,
                    f"source_setup.random.math_random_sequence[{index}]",
                )
                for index, value in enumerate(raw_sequence)
            ),
            constant=0.5,
            calls=[],
        )

    raw_constant = _source_first_owned(random_setup, ("math_random", "mathRandom"))
    return _SourceRandomSource(
        sequence=None,
        constant=_source_required_random_value(
            0.5 if raw_constant is _SOURCE_MISSING else raw_constant,
            "source_setup.random.math_random",
        ),
        calls=[],
    )


def _source_first_owned(source: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return _SOURCE_MISSING


def _source_required_random_value(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenarioError(f"{field} must be a finite number in [0, 1)")
    random_value = float(value)
    if not math.isfinite(random_value) or random_value < 0 or random_value >= 1:
        raise ScenarioError(f"{field} must be a finite number in [0, 1)")
    return random_value


def _source_print_manager_timer_advance_ms_values(scenario: LoadedScenario) -> list[float]:
    raw_values = scenario.time_policy.get(
        "timer_advance_ms_sequence",
        scenario.time_policy.get("timerAdvanceMsSequence"),
    )
    if raw_values is None:
        return [0.0 for _ in scenario.raw_action_script]
    if not isinstance(raw_values, list) or len(raw_values) != len(scenario.raw_action_script):
        raise ScenarioError(
            "source-print-manager-canary runner requires one "
            "time_policy.timer_advance_ms_sequence value per step"
        )

    values: list[float] = []
    for index, raw_value in enumerate(raw_values):
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ScenarioError(
                "source-print-manager-canary runner requires finite non-negative "
                f"time_policy.timer_advance_ms_sequence[{index}]"
            )
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ScenarioError(
                "source-print-manager-canary runner requires finite non-negative "
                f"time_policy.timer_advance_ms_sequence[{index}]"
            )
        values.append(value)
    return values


def _source_print_manager_forced_state(
    scenario: LoadedScenario,
) -> tuple[list[list[float]], list[float], list[bool]]:
    if (
        scenario.scenario_id in _SOURCE_PRINT_MANAGER_DEATH_SCENARIO_IDS
        or scenario.scenario_id == _SOURCE_PRINT_MANAGER_RANDOM_SCENARIO_ID
    ):
        positions = _extract_positions(scenario.initial_state)
        headings = _extract_headings(scenario.initial_state)
        alive = _extract_alive(scenario.initial_state)
        if positions is None:
            raise ScenarioError("source-print-manager-canary runner requires forced positions")
        if headings is None:
            raise ScenarioError("source-print-manager-canary runner requires forced headings")
        if alive is None:
            alive = [True for _ in range(scenario.player_count)]
        return positions, headings, alive

    return _source_single_player_forced_state(
        scenario,
        runner_name="source-print-manager-canary",
    )


def _source_single_player_forced_state(
    scenario: LoadedScenario,
    *,
    runner_name: str,
) -> tuple[list[list[float]], list[float], list[bool]]:
    players = scenario.initial_state.get("players")
    if not isinstance(players, list) or len(players) != 1:
        raise ScenarioError(f"{runner_name} runner requires one forced player")
    raw_player = players[0]
    if not isinstance(raw_player, Mapping):
        raise ScenarioError("initial_state.players entries must be objects")
    state = _source_initial_player_state(raw_player)
    x = _source_required_finite_number(
        state.get("x", raw_player.get("x")),
        "initial_state.players[0].initial.x",
    )
    y = _source_required_finite_number(
        state.get("y", raw_player.get("y")),
        "initial_state.players[0].initial.y",
    )
    heading = _source_required_finite_number(
        state.get("angle_rad", state.get("angle", raw_player.get("heading"))),
        "initial_state.players[0].initial.angle_rad",
    )
    alive = state.get("alive", raw_player.get("alive", True))
    if not isinstance(alive, bool):
        raise ScenarioError("initial_state.players[0].initial.alive must be a boolean")
    return [[x, y]], [heading], [alive]


def _source_required_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenarioError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ScenarioError(f"{field} must be a finite number")
    return result


def _source_initial_player_state(player: Mapping[str, Any]) -> Mapping[str, Any]:
    state = player.get("state") or player.get("initial") or player
    if not isinstance(state, Mapping):
        return {}
    return state


def _source_trail_point(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ScenarioError(f"{field} must be a coordinate pair")
    return [float(value[0]), float(value[1])]


def _source_is_time_to_draw(
    last_trail_point: list[float] | None,
    position: list[float],
    radius: float,
) -> bool:
    if last_trail_point is None:
        return True
    return math.hypot(
        position[0] - last_trail_point[0],
        position[1] - last_trail_point[1],
    ) > radius


def _source_initial_player_index(
    player: Mapping[str, Any],
    fallback_index: int,
    player_ids: tuple[str, ...],
) -> int:
    for key in ("id", "player_id", "playerId", "name"):
        if key in player:
            return _source_player_index(player[key], player_ids)
    return fallback_index


def _source_initial_player_counter(
    player: Mapping[str, Any],
    keys: tuple[str, str],
) -> int | None:
    state = player.get("state") or player.get("initial") or player
    if not isinstance(state, Mapping):
        state = {}
    for source in (state, player):
        for key in keys:
            if key in source:
                return int(source[key])
    return None


def _source_include_events(scenario: LoadedScenario) -> bool:
    return scenario.comparison.get("include_events") is True


def _source_player_ids(scenario: LoadedScenario) -> tuple[str, ...]:
    player_ids = _scenario_player_ids(scenario.to_payload())
    if len(player_ids) >= scenario.player_count:
        return player_ids
    return tuple(f"p{index}" for index in range(scenario.player_count))


def _source_position_event(player_id: str, position: list[float]) -> dict[str, object]:
    return {
        "event": "position",
        "player_id": player_id,
        "x": _source_round(position[0]),
        "y": _source_round(position[1]),
    }


def _source_angle_event(player_id: str, angle: float) -> dict[str, object]:
    return {
        "event": "angle",
        "player_id": player_id,
        "angle": _source_round(angle),
    }


def _source_point_event(
    player_id: str,
    position: list[float],
    *,
    important: bool = False,
) -> dict[str, object]:
    return {
        "event": "point",
        "player_id": player_id,
        "x": _source_round(position[0]),
        "y": _source_round(position[1]),
        "important": important,
    }


def _source_property_event(player_id: str, property_name: str, value: object) -> dict[str, object]:
    return {
        "event": "property",
        "player_id": player_id,
        "property": property_name,
        "value": value,
    }


def _source_die_event(
    player_id: str,
    *,
    killer_id: str | None = None,
    old: bool | None = None,
) -> dict[str, object]:
    return {
        "event": "die",
        "player_id": player_id,
        "killer_id": killer_id,
        "old": old,
    }


def _source_score_round_event(
    player_id: str,
    score: int,
    round_score: int,
) -> dict[str, object]:
    return {
        "event": "score:round",
        "player_id": player_id,
        "score": score,
        "roundScore": round_score,
    }


def _source_score_event(
    player_id: str,
    score: int,
    round_score: int,
) -> dict[str, object]:
    return {
        "event": "score",
        "player_id": player_id,
        "score": score,
        "roundScore": round_score,
    }


def _source_round_end_event(winner_id: str | None) -> dict[str, object]:
    return {
        "event": "round:end",
        "winner_id": winner_id,
    }


def _source_setup_borderless(scenario: LoadedScenario) -> bool:
    return scenario.initial_state.get("borderless") is True


def _source_kinematics_scenario_payload(scenario: LoadedScenario) -> dict[str, object]:
    payload = scenario.to_payload()
    time_policy = dict(scenario.time_policy)
    if "step_ms" in time_policy:
        time_policy["step_ms"] = _source_round(float(time_policy["step_ms"]))
    if isinstance(time_policy.get("step_ms_sequence"), list):
        time_policy["step_ms_sequence"] = [
            _source_round(float(value)) for value in time_policy["step_ms_sequence"]
        ]
    payload["time_policy"] = time_policy
    return payload


def _source_round(value: float) -> float:
    scale = 10**_SOURCE_ROUND_DIGITS
    return math.floor(value * scale + 0.5) / scale


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
