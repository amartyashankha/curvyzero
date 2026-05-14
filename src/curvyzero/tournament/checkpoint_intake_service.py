"""Pure helpers for the CurvyTron checkpoint-intake service contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SUBMIT_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "allow_rating_overrides",
        "checkpoint_iteration",
        "checkpoint_selection",
        "collect_epsilon",
        "collect_temperature",
        "continue_from_latest",
        "decision_ms",
        "decision_source_frames",
        "enqueue_existing",
        "games_per_pair",
        "games_per_shard",
        "gif_sample_games_per_pair",
        "gif_sample_strategy",
        "initial_rating",
        "active_pool_limit",
        "max_events",
        "max_runs",
        "max_steps",
        "num_simulations",
        "pair_selection",
        "pairs_per_round",
        "placement_min_games",
        "placement_min_opponents",
        "policy_mode",
        "policy_trail_render_mode",
        "reuse_policies_per_shard",
        "round_count",
        "save_gif",
        "seed",
        "source_physics_step_ms",
        "spawn_if_empty",
        "spawn_if_existing",
        "spawn_rating",
        "stop_when_stable",
    }
)

RATING_PAYLOAD_OVERRIDE_KEYS = frozenset(
    {
        "collect_epsilon",
        "collect_temperature",
        "decision_ms",
        "decision_source_frames",
        "games_per_pair",
        "games_per_shard",
        "gif_sample_games_per_pair",
        "gif_sample_strategy",
        "initial_rating",
        "active_pool_limit",
        "max_steps",
        "num_simulations",
        "pair_selection",
        "pairs_per_round",
        "placement_min_games",
        "placement_min_opponents",
        "policy_mode",
        "policy_trail_render_mode",
        "reuse_policies_per_shard",
        "round_count",
        "save_gif",
        "seed",
        "source_physics_step_ms",
        "stop_when_stable",
    }
)


def parse_run_ids_value(run_ids: Any) -> list[str]:
    if isinstance(run_ids, str):
        return [
            item.strip()
            for item in run_ids.replace("\n", ",").split(",")
            if item.strip()
        ]
    if isinstance(run_ids, Sequence) and not isinstance(run_ids, (str, bytes)):
        return [str(item).strip() for item in run_ids if str(item).strip()]
    return []


def validate_submit_payload(payload: Mapping[str, Any]) -> None:
    for key in sorted(SUBMIT_FORBIDDEN_PAYLOAD_KEYS):
        if key in payload:
            raise ValueError(f"submit payload may not set scheduler/service knob: {key}")


def submit_has_candidate_input(payload: Mapping[str, Any]) -> bool:
    return bool(
        str(payload.get("checkpoint_refs") or "").strip()
        or parse_run_ids_value(payload.get("run_ids"))
        or str(payload.get("run_id_prefix") or "").strip()
    )


def merge_submit_scan_spec(
    existing_scan_spec: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    default_checkpoint_selection: str,
) -> dict[str, Any]:
    merged = dict(existing_scan_spec)
    submitted_run_ids = parse_run_ids_value(payload.get("run_ids"))
    submitted_prefix = str(payload.get("run_id_prefix") or "").strip()
    if not submitted_run_ids and not submitted_prefix:
        return merged

    merged.pop("checkpoint_refs", None)
    if submitted_run_ids:
        existing_run_ids = parse_run_ids_value(merged.get("run_ids"))
        merged["run_ids"] = ",".join(sorted({*existing_run_ids, *submitted_run_ids}))
    if submitted_prefix:
        existing_prefix = str(merged.get("run_id_prefix") or "").strip()
        if existing_prefix and existing_prefix != submitted_prefix:
            raise ValueError(
                "submit cannot change run_id_prefix; reconfigure the service"
            )
        merged["run_id_prefix"] = submitted_prefix
    if not merged.get("checkpoint_selection"):
        merged["checkpoint_selection"] = default_checkpoint_selection
    return merged


def rating_overrides_from_payload(
    payload: Mapping[str, Any],
    *,
    allow_rating_overrides: bool,
) -> dict[str, Any]:
    allowed = {"continue_from_latest"}
    if allow_rating_overrides:
        allowed = allowed.union(RATING_PAYLOAD_OVERRIDE_KEYS)
    return {key: value for key, value in payload.items() if key in allowed}
