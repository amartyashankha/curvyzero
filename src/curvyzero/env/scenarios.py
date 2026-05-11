"""Scenario facade and Python runner CLI entry point.

The shared scenario schema/parser, toy-v0 runner, and narrow source-fidelity
runners live in smaller modules. This module preserves the historical import
surface and ``python -m curvyzero.env.scenarios`` behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from curvyzero.env.scenario_schema import (
    PYTHON_SCENARIO_TRACE_SCHEMA,
    LoadedScenario,
    ScenarioError,
    _SOURCE_MOVE_TO_TOY_ACTION,
    _agent_id,
    _coerce_scenario,
    _extract_alive,
    _extract_headings,
    _extract_positions,
    _extract_step_actions,
    _float_pair,
    _normalize_action_encoding,
    _normalize_step_actions,
    _optional_int,
    _optional_mapping,
    _parse_action_script,
    _player_index,
    _required_int,
    _required_str,
    _scenario_initial_state,
    _scenario_player_ids,
    _scenario_provenance,
    _scenario_time_policy,
    _to_toy_actions,
    _two_float_pairs,
    _validate_actions,
    load_scenario,
    parse_scenario,
)
from curvyzero.env.toy_runner import (
    PYTHON_SCENARIO_RUNNER,
    TOY_V0_MESSAGE,
    ScenarioRun,
    _config_for_scenario,
    _extract_size,
    _force_initial_state,
    _trace_loaded_scenario,
    run_scenario,
)
from curvyzero.fidelity.source_runners import (
    SOURCE_BORDERLESS_WRAP_FIDELITY_SCOPE,
    SOURCE_BORDERLESS_WRAP_MESSAGE,
    SOURCE_BORDERLESS_WRAP_RULES_HASH,
    SOURCE_BORDERLESS_WRAP_RUNNER,
    SOURCE_BORDERLESS_WRAP_TRACE_SCOPE,
    SOURCE_BODY_CANARY_FIDELITY_SCOPE,
    SOURCE_BODY_CANARY_MESSAGE,
    SOURCE_BODY_CANARY_RULES_HASH,
    SOURCE_BODY_CANARY_RUNNER,
    SOURCE_BODY_CANARY_SCENARIO_IDS,
    SOURCE_BODY_CANARY_TRACE_SCOPE,
    SOURCE_KINEMATICS_FIDELITY_SCOPE,
    SOURCE_KINEMATICS_MESSAGE,
    SOURCE_KINEMATICS_RULES_HASH,
    SOURCE_KINEMATICS_RUNNER,
    SOURCE_KINEMATICS_TRACE_SCOPE,
    SOURCE_LIFECYCLE_FIDELITY_SCOPE,
    SOURCE_LIFECYCLE_MESSAGE,
    SOURCE_LIFECYCLE_RULES_HASH,
    SOURCE_LIFECYCLE_RUNNER,
    SOURCE_LIFECYCLE_SCENARIO_IDS,
    SOURCE_LIFECYCLE_TRACE_SCHEMA,
    SOURCE_LIFECYCLE_TRACE_SCOPE,
    SOURCE_NORMAL_WALL_FIDELITY_SCOPE,
    SOURCE_NORMAL_WALL_MESSAGE,
    SOURCE_NORMAL_WALL_RULES_HASH,
    SOURCE_NORMAL_WALL_RUNNER,
    SOURCE_NORMAL_WALL_TRACE_SCOPE,
    SOURCE_PRINT_MANAGER_FIDELITY_SCOPE,
    SOURCE_PRINT_MANAGER_MESSAGE,
    SOURCE_PRINT_MANAGER_RULES_HASH,
    SOURCE_PRINT_MANAGER_RUNNER,
    SOURCE_PRINT_MANAGER_SCENARIO_IDS,
    SOURCE_PRINT_MANAGER_TRACE_SCOPE,
    SOURCE_TRAIL_CADENCE_FIDELITY_SCOPE,
    SOURCE_TRAIL_CADENCE_MESSAGE,
    SOURCE_TRAIL_CADENCE_RULES_HASH,
    SOURCE_TRAIL_CADENCE_RUNNER,
    SOURCE_TRAIL_CADENCE_SCENARIO_IDS,
    SOURCE_TRAIL_CADENCE_TRACE_SCOPE,
    SOURCE_TRAIL_GAP_FIDELITY_SCOPE,
    SOURCE_TRAIL_GAP_MESSAGE,
    SOURCE_TRAIL_GAP_RULES_HASH,
    SOURCE_TRAIL_GAP_RUNNER,
    SOURCE_TRAIL_GAP_SCENARIO_IDS,
    SOURCE_TRAIL_GAP_TRACE_SCOPE,
    SourceBodyCanaryScenarioRun,
    SourceBorderlessWrapScenarioRun,
    SourceKinematicsScenarioRun,
    SourceLifecycleScenarioRun,
    SourceNormalWallScenarioRun,
    SourcePrintManagerScenarioRun,
    SourceTrailCadenceScenarioRun,
    SourceTrailGapScenarioRun,
    _SOURCE_ROUND_DIGITS,
    _fingerprint_payload,
    _is_source_kinematics_movement_scenario,
    _source_apply_borderless_wrap,
    _source_die_event,
    _source_include_events,
    _source_kinematics_frame,
    _source_kinematics_scenario_payload,
    _source_kinematics_step_ms,
    _source_map_size,
    _source_normal_wall_hit,
    _source_player_ids,
    _source_point_event,
    _source_position_event,
    _source_round,
    _source_round_end_event,
    _source_score_event,
    _source_score_round_event,
    _source_setup_borderless,
    _trace_source_body_canary,
    _trace_source_borderless_wrap,
    _trace_source_kinematics,
    _trace_source_normal_wall,
    _validate_source_player_state_lengths,
    run_source_body_canary_scenario,
    run_source_border_rules_scenario,
    run_source_borderless_wrap_scenario,
    run_source_kinematics_first_scenario,
    run_source_kinematics_scenario,
    run_source_lifecycle_scenario,
    run_source_normal_wall_scenario,
    run_source_print_manager_scenario,
    run_source_trail_cadence_scenario,
    run_source_trail_gap_scenario,
)

__all__ = [
    "PYTHON_SCENARIO_RUNNER",
    "PYTHON_SCENARIO_TRACE_SCHEMA",
    "SOURCE_BORDERLESS_WRAP_FIDELITY_SCOPE",
    "SOURCE_BORDERLESS_WRAP_MESSAGE",
    "SOURCE_BORDERLESS_WRAP_RULES_HASH",
    "SOURCE_BORDERLESS_WRAP_RUNNER",
    "SOURCE_BORDERLESS_WRAP_TRACE_SCOPE",
    "SOURCE_BODY_CANARY_FIDELITY_SCOPE",
    "SOURCE_BODY_CANARY_MESSAGE",
    "SOURCE_BODY_CANARY_RULES_HASH",
    "SOURCE_BODY_CANARY_RUNNER",
    "SOURCE_BODY_CANARY_SCENARIO_IDS",
    "SOURCE_BODY_CANARY_TRACE_SCOPE",
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
    "TOY_V0_MESSAGE",
    "LoadedScenario",
    "ScenarioError",
    "ScenarioRun",
    "SourceBodyCanaryScenarioRun",
    "SourceBorderlessWrapScenarioRun",
    "SourceKinematicsScenarioRun",
    "SourceLifecycleScenarioRun",
    "SourceNormalWallScenarioRun",
    "SourcePrintManagerScenarioRun",
    "SourceTrailCadenceScenarioRun",
    "SourceTrailGapScenarioRun",
    "_SOURCE_MOVE_TO_TOY_ACTION",
    "_SOURCE_ROUND_DIGITS",
    "_agent_id",
    "_coerce_scenario",
    "_config_for_scenario",
    "_extract_alive",
    "_extract_headings",
    "_extract_positions",
    "_extract_size",
    "_extract_step_actions",
    "_fingerprint_payload",
    "_float_pair",
    "_force_initial_state",
    "_is_source_kinematics_movement_scenario",
    "_normalize_action_encoding",
    "_normalize_step_actions",
    "_optional_int",
    "_optional_mapping",
    "_parse_action_script",
    "_player_index",
    "_required_int",
    "_required_str",
    "_scenario_initial_state",
    "_scenario_player_ids",
    "_scenario_provenance",
    "_scenario_time_policy",
    "_source_apply_borderless_wrap",
    "_source_die_event",
    "_source_include_events",
    "_source_kinematics_frame",
    "_source_kinematics_scenario_payload",
    "_source_kinematics_step_ms",
    "_source_map_size",
    "_source_normal_wall_hit",
    "_source_player_ids",
    "_source_point_event",
    "_source_position_event",
    "_source_round",
    "_source_round_end_event",
    "_source_score_event",
    "_source_score_round_event",
    "_source_setup_borderless",
    "_to_toy_actions",
    "_trace_loaded_scenario",
    "_trace_source_body_canary",
    "_trace_source_borderless_wrap",
    "_trace_source_kinematics",
    "_trace_source_normal_wall",
    "_two_float_pairs",
    "_validate_actions",
    "_validate_source_player_state_lengths",
    "load_scenario",
    "main",
    "parse_scenario",
    "run_scenario",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one shared scenario through a Python scenario runner."
    )
    parser.add_argument("scenario", help="Path to a scenario JSON file")
    parser.add_argument(
        "--runner",
        choices=(
            "toy-v0",
            "source-kinematics",
            "source-normal-wall",
            "source-borderless-wrap",
            "source-body-canary",
            "source-print-manager-canary",
            "source-trail-cadence-canary",
            "source-trail-gap-canary",
            "source-lifecycle",
            "source-border-rules",
        ),
        default="toy-v0",
        help=(
            "Python runner mode. source-kinematics is movement only; "
            "source-normal-wall, source-borderless-wrap, source-body-canary, "
            "source-print-manager-canary, source-trail-cadence-canary, "
            "source-trail-gap-canary, and source-lifecycle are narrow "
            "source-fidelity runners."
        ),
    )
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON")
    args = parser.parse_args(argv)

    try:
        if args.runner == "source-kinematics":
            payload = run_source_kinematics_scenario(args.scenario).to_payload()
        elif args.runner == "source-normal-wall":
            payload = run_source_normal_wall_scenario(args.scenario).to_payload()
        elif args.runner == "source-borderless-wrap":
            payload = run_source_borderless_wrap_scenario(args.scenario).to_payload()
        elif args.runner == "source-body-canary":
            payload = run_source_body_canary_scenario(args.scenario).to_payload()
        elif args.runner == "source-print-manager-canary":
            payload = run_source_print_manager_scenario(args.scenario).to_payload()
        elif args.runner == "source-trail-cadence-canary":
            payload = run_source_trail_cadence_scenario(args.scenario).to_payload()
        elif args.runner == "source-trail-gap-canary":
            payload = run_source_trail_gap_scenario(args.scenario).to_payload()
        elif args.runner == "source-lifecycle":
            payload = run_source_lifecycle_scenario(args.scenario).to_payload()
        elif args.runner == "source-border-rules":
            payload = run_source_border_rules_scenario(args.scenario).to_payload()
        else:
            payload = run_scenario(args.scenario).to_payload()
    except ScenarioError as error:
        print(f"Scenario error: {error}", file=sys.stderr)
        return 2

    indent = None if args.compact else 2
    text = json.dumps(payload, indent=indent, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
