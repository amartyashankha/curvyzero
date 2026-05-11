"""Replay stored CurvyTron visual-survival eval episodes for death inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv,
)


ACTION_NAMES = {0: "left", 1: "straight", 2: "right"}
SUPPORTED_OPPONENT_POLICY_KINDS = {"fixed_straight", None}
SUPPORTED_ENV_VARIANTS = {"fixed_opponent", None}


def inspect_episode_artifact(path: str | Path) -> dict[str, Any]:
    """Replay one old per-episode eval artifact and summarize the terminal death."""

    artifact_path = Path(path)
    try:
        payload = _load_json_object(artifact_path)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "artifact_path": artifact_path.as_posix(),
            "reason": "invalid_json",
            "plain_read": f"This artifact is not valid JSON: {exc.msg}.",
        }
    except OSError as exc:
        return {
            "ok": False,
            "artifact_path": artifact_path.as_posix(),
            "reason": "read_failed",
            "plain_read": f"This artifact could not be read: {exc}.",
        }
    except ValueError as exc:
        return {
            "ok": False,
            "artifact_path": artifact_path.as_posix(),
            "reason": "invalid_artifact",
            "plain_read": str(exc),
        }
    episode = _dict(payload.get("episode"))
    config = _dict(payload.get("config"))
    actions = _episode_actions(episode, artifact_path)
    seed = _int_or_default(episode.get("seed"), _int_or_default(config.get("seed"), 0))
    env_variant = config.get("env_variant")
    if env_variant not in SUPPORTED_ENV_VARIANTS:
        return {
            "ok": False,
            "artifact_path": artifact_path.as_posix(),
            "reason": "unsupported_env_variant",
            "plain_read": (
                "This episode used a different env variant than the local debug replay "
                "inspector can reproduce."
            ),
            "env_variant": env_variant,
        }
    opponent_policy_kind = config.get("opponent_policy_kind", "fixed_straight")
    if opponent_policy_kind not in SUPPORTED_OPPONENT_POLICY_KINDS:
        return {
            "ok": False,
            "artifact_path": artifact_path.as_posix(),
            "reason": "unsupported_opponent_policy_kind",
            "plain_read": (
                "This episode used an opponent policy that cannot be replayed from "
                "ego actions alone."
            ),
            "opponent_policy_kind": opponent_policy_kind,
        }
    if not actions:
        return {
            "ok": False,
            "artifact_path": artifact_path.as_posix(),
            "reason": "missing_actions",
            "plain_read": "This artifact does not contain the ordered actions needed to replay it.",
        }

    env = CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv(
        {
            "seed": seed,
            "source_max_steps": _int_or_default(config.get("source_max_steps"), 1024),
            "source_step_ms": float(config.get("source_step_ms", 16.666666666666668)),
            "opponent_policy_kind": "fixed_straight",
        }
    )
    env.reset(seed=seed)
    timestep = None
    replayed_actions = []
    for action in actions:
        timestep = env.step(action)
        replayed_actions.append(action)
        if timestep.done:
            break

    if timestep is None:
        return {
            "ok": False,
            "artifact_path": artifact_path.as_posix(),
            "reason": "no_replay_steps",
            "plain_read": "Replay did not run any steps.",
        }

    info = _dict(timestep.info)
    expected_trace_hash = _expected_final_trace_hash(artifact_path, episode)
    actual_trace_hash = _string_or_none(info.get("trace_hash"))
    trace_hash_match = (
        expected_trace_hash is not None and actual_trace_hash == expected_trace_hash
    )
    first_death = _first_death(info)
    return {
        "ok": True,
        "artifact_path": artifact_path.as_posix(),
        "seed": seed,
        "actions_recorded": len(actions),
        "actions_replayed": len(replayed_actions),
        "action_read": _action_read(replayed_actions),
        "done": bool(timestep.done),
        "terminal_reason": info.get("terminal_reason"),
        "winner_ids": _plain(info.get("winner_ids")),
        "loser_ids": _plain(info.get("loser_ids")),
        "death_count": _plain(info.get("death_count")),
        "death_player": _plain(info.get("death_player")),
        "death_cause_name": _plain(info.get("death_cause_name")),
        "death_hit_owner": _plain(info.get("death_hit_owner")),
        "first_death": first_death,
        "replay_trace_hash": actual_trace_hash,
        "expected_trace_hash": expected_trace_hash,
        "trace_hash_match": trace_hash_match,
        "plain_read": _plain_read(
            first_death=first_death,
            info=info,
            actions=replayed_actions,
            trace_hash_match=trace_hash_match,
            expected_trace_hash=expected_trace_hash,
        ),
    }


def _episode_actions(episode: dict[str, Any], artifact_path: Path) -> list[int]:
    actions = episode.get("actions")
    if isinstance(actions, list):
        return [int(action) for action in actions]
    telemetry_path = artifact_path.with_suffix(".env_steps.jsonl")
    if not telemetry_path.exists():
        return []
    rows = _load_jsonl(telemetry_path)
    return [
        int(row["ego_action"])
        for row in rows
        if isinstance(row, dict) and row.get("ego_action") is not None
    ]


def _expected_final_trace_hash(
    artifact_path: Path,
    episode: dict[str, Any],
) -> str | None:
    telemetry_path = artifact_path.with_suffix(".env_steps.jsonl")
    if telemetry_path.exists():
        rows = _load_jsonl(telemetry_path)
        for row in reversed(rows):
            value = _string_or_none(row.get("trace_hash"))
            if value:
                return value
    steps = episode.get("steps")
    if isinstance(steps, list):
        for step in reversed(steps):
            if not isinstance(step, dict):
                continue
            info = step.get("info")
            if isinstance(info, dict):
                value = _string_or_none(info.get("trace_hash"))
                if value:
                    return value
    return None


def _first_death(info: dict[str, Any]) -> dict[str, Any] | None:
    count = _row_scalar(info.get("death_count"))
    if _int_or_default(count, 0) <= 0:
        return None
    player = _row_slot(info.get("death_player"), 0)
    cause_name = _row_slot(info.get("death_cause_name"), 0)
    hit_owner = _row_slot(info.get("death_hit_owner"), 0)
    return {
        "player": player,
        "player_id": _player_id(player),
        "cause_name": cause_name,
        "hit_owner": hit_owner,
        "hit_owner_id": _player_id(hit_owner),
    }


def _plain_read(
    *,
    first_death: dict[str, Any] | None,
    info: dict[str, Any],
    actions: list[int],
    trace_hash_match: bool,
    expected_trace_hash: str | None,
) -> str:
    action_text = _action_read(actions)
    if first_death is None:
        death_text = "Replay ended without a recorded death."
    else:
        player = first_death.get("player_id") or f"player_{first_death.get('player')}"
        cause_name = first_death.get("cause_name")
        if cause_name == "wall":
            death_text = f"{player} hit the wall."
        elif cause_name == "own_trail":
            death_text = f"{player} hit their own trail."
        elif cause_name == "opponent_trail":
            owner = first_death.get("hit_owner_id") or "the opponent"
            death_text = f"{player} hit {owner}'s trail."
        else:
            death_text = f"{player} died, but the replay only knows it was a body hit."
    terminal = info.get("terminal_reason")
    hash_text = (
        "The final trace hash matches the stored artifact."
        if trace_hash_match
        else "No stored final trace hash was found."
        if expected_trace_hash is None
        else "The final trace hash does not match the stored artifact."
    )
    return f"{death_text} Terminal reason: {terminal}. Actions: {action_text}. {hash_text}"


def _action_read(actions: list[int]) -> str:
    if not actions:
        return "no actions"
    counts: dict[int, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    if len(counts) == 1:
        action = actions[0]
        return f"{ACTION_NAMES.get(action, str(action))} on all {len(actions)} decisions"
    parts = [
        f"{ACTION_NAMES.get(action, str(action))}:{count}"
        for action, count in sorted(counts.items())
    ]
    return ", ".join(parts)


def _row_scalar(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _row_slot(value: Any, slot: int) -> Any:
    row = _row_scalar(value)
    if isinstance(row, list):
        if slot < len(row):
            return row[slot]
        return None
    return row


def _player_id(value: Any) -> str | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if index < 0:
        return None
    return f"player_{index}"


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# CurvyTron Replay Inspection", ""]
    for row in rows:
        lines.append(f"- `{row.get('artifact_path')}`")
        lines.append(f"  - ok: {row.get('ok')}")
        lines.append(f"  - read: {row.get('plain_read')}")
        if row.get("ok"):
            lines.append(
                "  - trace: "
                f"{row.get('replay_trace_hash')} "
                f"(match={row.get('trace_hash_match')})"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay CurvyTron visual-survival eval artifacts from stored action traces."
        )
    )
    parser.add_argument("artifacts", nargs="+", help="Per-episode eval JSON artifact(s).")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    args = parser.parse_args(argv)

    rows = [inspect_episode_artifact(path) for path in args.artifacts]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(_format_markdown(rows))
    return 0 if all(row.get("ok") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
