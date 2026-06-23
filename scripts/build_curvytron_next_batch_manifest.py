#!/usr/bin/env python3
"""Build the current CurvyTron Grid A/Grid B training manifest.

This is the current launch builder for the `cz26` family. It emits manifests
that can be submitted by `scripts/submit_curvytron_survivaldiag_manifest.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from curvyzero.contracts.curvytron import (
    COMPUTE_L4_T4_CPU40,
    CURVYTRON_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER,
    CURVYTRON_BACKGROUND_GIF_FPS,
    CURVYTRON_COMMIT_ON_CHECKPOINT,
    CURVYTRON_DECISION_MS,
    CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM,
    CURVYTRON_DEFAULT_MAX_ENV_STEP,
    CURVYTRON_DEFAULT_MAX_TRAIN_ITER,
    CURVYTRON_DEFAULT_N_EPISODE,
    CURVYTRON_DEFAULT_NUM_SIMULATIONS,
    CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE,
    CURVYTRON_POLICY_BONUS_RENDER_MODE,
    CURVYTRON_POLICY_TRAIL_RENDER_MODE,
    CURVYTRON_SAVE_CKPT_AFTER_ITER,
    CURVYTRON_SOURCE_MAX_STEPS,
    CURVYTRON_TRAINING_TASK_ID,
    DEFAULT_CURVYTRON_TRAIN_APP_NAME,
    DEFAULT_LEARNER_SEAT_MODE,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
    TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT,
)
from curvyzero.contracts.curvytron_naming import (
    CURVYTRON_CANARY_BATCH,
    CURVYTRON_GRID_A_BATCH,
    CURVYTRON_GRID_B_BATCH,
    action_noise_tag,
    curvytron_attempt_id,
    curvytron_run_id,
    leaderboard_immortal_tag,
    reward_alpha_tag,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_MIXTURE_SCHEMA_ID,
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    OPPONENT_RUNTIME_MODE_NORMAL,
    parse_opponent_mixture_spec,
)
from curvyzero.training.opponent_registry import (
    OPPONENT_ASSIGNMENT_SCHEMA_ID,
    canonical_assignment_json_sha256,
    parse_opponent_assignment_snapshot,
)


SCHEMA_ID = "curvyzero_curvytron_next_batch_manifest/v0"
ROW_SCHEMA_ID = "curvyzero_curvytron_next_batch_row/v0"
MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
TASK_ID = CURVYTRON_TRAINING_TASK_ID
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_next_batch_manifests")
DEFAULT_TOURNAMENT_ID = "cz26-live-20260517a"
DEFAULT_RATING_RUN_ID = "elo-cz26-live-20260517a"
DEFAULT_LEADERBOARD_ID = f"{DEFAULT_TOURNAMENT_ID}-{DEFAULT_RATING_RUN_ID}-training"
DEFAULT_ASSIGNMENT_BANK_RUN_ID = "cz26-training-candidates"
DEFAULT_ASSIGNMENT_BANK_ATTEMPT_ID = "try-cz26-training-candidates"
DEFAULT_POINTER_RUN_ID = "cz26-control"
DEFAULT_POINTER_ATTEMPT_ID = "try-cz26-control"
DEFAULT_REFRESH_CONFIG_REF = (
    f"control:training/{TASK_ID}/{DEFAULT_POINTER_RUN_ID}/attempts/"
    f"{DEFAULT_POINTER_ATTEMPT_ID}/opponents/training_candidate_refresh_config.json"
)
MAX_MODAL_RUN_ID_LEN = 96


@dataclass(frozen=True)
class Recipe:
    code: str
    role: str
    description: str
    slot_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class NoiseMode:
    probability: float
    tag: str
    straight_override_probability: float
    repeat_min: int
    repeat_max: int
    repeat_extra_probability: float
    profile_id: str


GRID_A_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "b20w05r1",
        "grid_a_anchor",
        "20% blank, 5% wall-avoidant, 75% rank1",
        (("blank", 13), ("wall", 3), ("rank1", 48)),
    ),
    Recipe(
        "b10w05r1",
        "grid_a_blank_dose",
        "10% blank, 5% wall-avoidant, 85% rank1",
        (("blank", 7), ("wall", 3), ("rank1", 54)),
    ),
    Recipe(
        "b20w10r1",
        "grid_a_wall_dose",
        "20% blank, 10% wall-avoidant, 70% rank1",
        (("blank", 13), ("wall", 6), ("rank1", 45)),
    ),
    Recipe(
        "b20w05top2",
        "grid_a_rank_diversity",
        "20% blank, 5% wall-avoidant, 50% rank1, 25% rank2",
        (("blank", 13), ("wall", 3), ("rank1", 32), ("rank2", 16)),
    ),
)

GRID_B_RECIPES: tuple[Recipe, ...] = (
    Recipe("b100", "grid_b_pure_control", "100% blank", (("blank", 64),)),
    Recipe("w100", "grid_b_pure_control", "100% wall-avoidant", (("wall", 64),)),
    Recipe("r1", "grid_b_pure_control", "100% rank1", (("rank1", 64),)),
    Recipe("b50r1", "grid_b_core", "50% blank, 50% rank1", (("blank", 32), ("rank1", 32))),
    Recipe(
        "b25w25r1",
        "grid_b_core",
        "25% blank, 25% wall-avoidant, 50% rank1",
        (("blank", 16), ("wall", 16), ("rank1", 32)),
    ),
    Recipe(
        "b20w20lad4s",
        "grid_b_stress",
        "20% blank, 20% wall, rank1/rank2/rank3/rank4 ladder",
        (("blank", 13), ("wall", 13), ("rank1", 19), ("rank2", 13), ("rank3", 3), ("rank4", 3)),
    ),
    GRID_A_RECIPES[0],
    Recipe(
        "b30w05r1",
        "grid_b_blank_dose",
        "30% blank, 5% wall-avoidant, 65% rank1",
        (("blank", 19), ("wall", 3), ("rank1", 42)),
    ),
    GRID_A_RECIPES[3],
    Recipe(
        "b20w05lad4",
        "grid_b_ladder",
        "20% blank, 5% wall, rank1/rank2/rank3/rank4 ladder",
        (("blank", 13), ("wall", 3), ("rank1", 19), ("rank2", 13), ("rank3", 10), ("rank4", 6)),
    ),
)

GRID_A_ALPHAS = (0.0, 0.33, 0.67, 1.0)
GRID_B_ALPHAS = (0.5,)
GRID_A_NOISES = (0.0, 0.10, 0.20)
GRID_B_NOISES = (0.0, 0.10)
LEADERBOARD_IMMORTAL_PROBABILITIES = (0.0, 0.10)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_id(value: str, *, label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not safe:
        raise ValueError(f"{label} became empty")
    if len(safe) > MAX_MODAL_RUN_ID_LEN:
        digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:10]
        prefix_len = MAX_MODAL_RUN_ID_LEN - len(digest) - 1
        safe = f"{safe[:prefix_len].rstrip('-_.')}-{digest}"
    return safe


def _ref(*parts: str | Path) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _checkpoint_iteration_from_ref(checkpoint_ref: str) -> int | None:
    match = re.fullmatch(r"iteration_(\d+)\.pth\.tar", Path(checkpoint_ref).name)
    return int(match.group(1)) if match else None


def _checkpoint_run_attempt_from_ref(checkpoint_ref: str) -> tuple[str | None, str | None]:
    parts = Path(checkpoint_ref).parts
    try:
        training_index = parts.index("training")
        attempts_index = parts.index("attempts")
    except ValueError:
        return None, None
    if attempts_index <= training_index + 2 or attempts_index + 1 >= len(parts):
        return None, None
    return parts[training_index + 2], parts[attempts_index + 1]


def _validate_exact_checkpoint_ref(checkpoint_ref: str, *, label: str) -> str:
    ref = str(checkpoint_ref).strip()
    if not ref:
        raise ValueError(f"{label} checkpoint ref is empty")
    if "latest" in ref or "ckpt_best" in ref:
        raise ValueError(f"{label} checkpoint ref is mutable: {ref}")
    if _checkpoint_iteration_from_ref(ref) is None:
        raise ValueError(f"{label} ref must end in iteration_N.pth.tar: {ref}")
    return ref


def _load_checkpoint_refs_payload(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    if text[0] in "[{":
        payload = json.loads(text)
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            values = (
                payload.get("checkpoint_refs")
                or payload.get("refs")
                or payload.get("rows")
                or payload.get("ratings")
            )
        else:
            values = None
        if not isinstance(values, list):
            raise ValueError(
                f"{path} JSON must be a list or contain checkpoint_refs/refs/rows/ratings"
            )
        return [
            str(item.get("checkpoint_ref") if isinstance(item, Mapping) else item).strip()
            for item in values
        ]
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith("rank=")
    ]


def _load_top_checkpoints_from_refs(path: Path) -> dict[str, dict[str, Any]]:
    top: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    refs = _load_checkpoint_refs_payload(path)
    if len(refs) < 4:
        raise ValueError(f"{path} must contain at least four checkpoint refs")
    for rank, raw_ref in enumerate(refs[:4], start=1):
        ref = _validate_exact_checkpoint_ref(raw_ref, label=f"rank{rank}")
        if ref in seen:
            raise ValueError(f"duplicate checkpoint ref in {path}: {ref}")
        seen.add(ref)
        run_id, attempt_id = _checkpoint_run_attempt_from_ref(ref)
        digest = hashlib.sha1(ref.encode("utf-8")).hexdigest()[:10]
        top[f"rank{rank}"] = {
            "rank": rank,
            "checkpoint_id": f"curated-rank{rank}-{digest}",
            "rating": None,
            "status": "curated_exact_ref",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "iteration": _checkpoint_iteration_from_ref(ref),
            "checkpoint_ref": ref,
        }
    return top


def _load_top_checkpoints_from_ratings(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("ratings") or payload.get("rows")
    if not isinstance(rows, list) or len(rows) < 4:
        raise ValueError(f"{path} must contain at least four ratings or rows")
    top: dict[str, dict[str, Any]] = {}
    for expected_rank, row in enumerate(rows[:4], start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"rating row {expected_rank} is not an object")
        rank = int(row.get("rank", expected_rank))
        if rank != expected_rank:
            raise ValueError(f"ratings[:4] must be ordered by rank; got {rank}")
        ref = _validate_exact_checkpoint_ref(
            str(row.get("checkpoint_ref") or ""),
            label=f"rank{rank}",
        )
        top[f"rank{rank}"] = {
            "rank": rank,
            "checkpoint_id": row.get("checkpoint_id"),
            "rating": row.get("rating"),
            "status": row.get("status") or "active",
            "run_id": row.get("run_id"),
            "attempt_id": row.get("attempt_id"),
            "iteration": row.get("iteration"),
            "checkpoint_ref": ref,
        }
    return top


def _load_top_checkpoints(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.checkpoint_refs_file:
        return _load_top_checkpoints_from_refs(args.checkpoint_refs_file)
    if args.ratings_snapshot:
        return _load_top_checkpoints_from_ratings(args.ratings_snapshot)
    raise ValueError("one of --checkpoint-refs-file or --ratings-snapshot is required")


def _noise_mode(probability: float) -> NoiseMode:
    tag = action_noise_tag(probability)
    if probability == 0.0:
        return NoiseMode(probability, tag, 0.0, 1, 1, 0.0, "none")
    return NoiseMode(
        probability,
        tag,
        float(probability),
        1,
        2,
        float(probability),
        f"straight_override_p{int(probability * 100):02d}_repeat_p{int(probability * 100):02d}",
    )


def _rank_number(component: str) -> int | None:
    match = re.fullmatch(r"rank([1-9][0-9]*)", component)
    return int(match.group(1)) if match else None


def _leaderboard_immortal_counts(
    slot_counts: Sequence[tuple[str, int]],
    probability: float,
) -> dict[str, int]:
    rank_counts = [(name, int(count)) for name, count in slot_counts if _rank_number(name)]
    total = sum(count for _name, count in rank_counts)
    if total <= 0 or probability <= 0.0:
        return {name: 0 for name, _count in rank_counts}
    target = max(0, min(total, int(round(total * float(probability)))))
    floors: dict[str, int] = {
        name: min(count, int(count * float(probability))) for name, count in rank_counts
    }
    allocated = sum(floors.values())
    remainders = sorted(
        (
            (count * float(probability) - floors[name], name, count)
            for name, count in rank_counts
        ),
        reverse=True,
    )
    for _remainder, name, count in remainders:
        if allocated >= target:
            break
        if floors[name] < count:
            floors[name] += 1
            allocated += 1
    return floors


def _checkpoint_entry(
    *,
    name: str,
    rank_name: str,
    weight: int,
    top_checkpoints: Mapping[str, Mapping[str, Any]],
    seed: int,
    opponent_immortal: bool,
    leaderboard_immortal_probability: float,
) -> dict[str, Any]:
    checkpoint = top_checkpoints[rank_name]
    checkpoint_ref = str(checkpoint["checkpoint_ref"])
    rank = int(checkpoint["rank"])
    return {
        "name": name,
        "age_label": name,
        "weight": weight,
        "opponent_policy_kind": OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
        "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_NORMAL,
        "opponent_immortal": bool(opponent_immortal),
        "opponent_checkpoint_ref": checkpoint_ref,
        "opponent_snapshot_ref": f"cz26_rank{rank}_{Path(checkpoint_ref).stem}",
        "opponent_policy_seed": _derived_seed("opponent", name, seed),
        "tags": {
            "rank": rank,
            "source_slot": rank_name,
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "rating": checkpoint.get("rating"),
            "leaderboard_immortal_probability": float(leaderboard_immortal_probability),
            "immortal": bool(opponent_immortal),
        },
    }


def _assignment_entries(
    recipe: Recipe,
    *,
    top_checkpoints: Mapping[str, Mapping[str, Any]],
    seed: int,
    leaderboard_immortal_probability: float,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    immortal_counts = _leaderboard_immortal_counts(
        recipe.slot_counts,
        leaderboard_immortal_probability,
    )
    for component, raw_count in recipe.slot_counts:
        count = int(raw_count)
        if component == "blank":
            entries.append(
                {
                    "name": "blank",
                    "age_label": "blank_canvas",
                    "weight": count,
                    "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                    "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                    "opponent_immortal": True,
                    "tags": {"source_slot": "blank", "hardcoded": True},
                }
            )
            continue
        if component == "wall":
            entries.append(
                {
                    "name": "wall_avoidant",
                    "age_label": "hardcoded_wall_avoidant",
                    "weight": count,
                    "opponent_policy_kind": OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
                    "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_NORMAL,
                    "opponent_immortal": True,
                    "opponent_wall_avoidant_safe_margin": 20.0,
                    "tags": {"source_slot": "wall", "hardcoded": True},
                }
            )
            continue
        rank = _rank_number(component)
        if rank is None:
            raise ValueError(f"unknown recipe component: {component!r}")
        rank_name = f"rank{rank}"
        immortal_count = immortal_counts.get(component, 0)
        mortal_count = count - immortal_count
        if mortal_count > 0:
            entries.append(
                _checkpoint_entry(
                    name=rank_name,
                    rank_name=rank_name,
                    weight=mortal_count,
                    top_checkpoints=top_checkpoints,
                    seed=seed,
                    opponent_immortal=False,
                    leaderboard_immortal_probability=leaderboard_immortal_probability,
                )
            )
        if immortal_count > 0:
            entries.append(
                _checkpoint_entry(
                    name=f"{rank_name}_immortal",
                    rank_name=rank_name,
                    weight=immortal_count,
                    top_checkpoints=top_checkpoints,
                    seed=seed,
                    opponent_immortal=True,
                    leaderboard_immortal_probability=leaderboard_immortal_probability,
                )
            )
    return entries


def _assignment_artifact(
    *,
    args: argparse.Namespace,
    recipe: Recipe,
    top_checkpoints: Mapping[str, Mapping[str, Any]],
    leaderboard_immortal_probability: float,
) -> dict[str, Any]:
    immortal_tag = leaderboard_immortal_tag(leaderboard_immortal_probability)
    assignment_id = _safe_id(
        f"{args.matrix_name}-{recipe.code}-{immortal_tag}-initial",
        label="assignment_id",
    )
    seed = _derived_seed(args.matrix_name, recipe.code, immortal_tag, "assignment")
    assignment = {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": assignment_id,
        "source_epoch": 0,
        "source_ref": args.assignment_source_ref,
        "created_at": _utc_timestamp(),
        "seed": seed,
        "entries": _assignment_entries(
            recipe,
            top_checkpoints=top_checkpoints,
            seed=seed,
            leaderboard_immortal_probability=leaderboard_immortal_probability,
        ),
    }
    parse_opponent_assignment_snapshot(assignment)
    assignment_sha256 = canonical_assignment_json_sha256(assignment)
    assignment_ref = _ref(
        "control:training",
        TASK_ID,
        args.assignment_bank_run_id,
        "attempts",
        args.assignment_bank_attempt_id,
        "opponents",
        "assignments",
        assignment_id,
        "assignment.json",
    )
    pointer_ref = _ref(
        "control:training",
        TASK_ID,
        args.pointer_run_id,
        "attempts",
        args.pointer_attempt_id,
        "opponents",
        "refresh_pointers",
        recipe.code,
        immortal_tag,
        "refresh_pointer.json",
    )
    mixture = parse_opponent_mixture_spec({"seed": seed, "entries": assignment["entries"]})
    assert mixture is not None
    hardcoded_immortal_slots = sum(
        float(entry["weight"])
        for entry in mixture["entries"]
        if entry["name"] in {"blank", "wall_avoidant"}
    )
    leaderboard_immortal_slots = sum(
        float(entry["weight"])
        for entry in mixture["entries"]
        if bool(entry.get("opponent_immortal"))
        and entry["opponent_policy_kind"] == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
    )
    source_leaderboard: dict[str, Any] | None = None
    if args.ratings_snapshot:
        snapshot_bytes = args.ratings_snapshot.read_bytes()
        snapshot_payload = json.loads(snapshot_bytes.decode("utf-8"))
        source_leaderboard = {
            "leaderboard_id": args.leaderboard_id,
            "snapshot_id": str(
                snapshot_payload.get("snapshot_id")
                or snapshot_payload.get("round_id")
                or args.ratings_snapshot.stem
            ),
            "snapshot_ref": str(args.assignment_source_ref),
            "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            "seed_rank1_checkpoint_ref": top_checkpoints["rank1"]["checkpoint_ref"],
        }
    audit = {
        "schema_id": "curvyzero_opponent_assignment_audit/v0",
        "assignment_id": assignment_id,
        "assignment_sha256": assignment_sha256,
        "selection": {
            "strategy_id": "cz26_slot_count_recipe_v0",
            "recipe_code": recipe.code,
            "recipe_role": recipe.role,
            "recipe_slot_counts": dict(recipe.slot_counts),
            "leaderboard_immortal_probability": float(leaderboard_immortal_probability),
            "slot_count_total": int(mixture["total_weight"]),
            "hardcoded_immortal_slots": hardcoded_immortal_slots,
            "leaderboard_immortal_slots": leaderboard_immortal_slots,
            "total_immortal_slots": hardcoded_immortal_slots + leaderboard_immortal_slots,
        },
    }
    if source_leaderboard is not None:
        audit["source_leaderboard"] = source_leaderboard
    else:
        audit["source_checkpoint_refs"] = {
            "source_ref": args.assignment_source_ref,
            "seed_rank1_checkpoint_ref": top_checkpoints["rank1"]["checkpoint_ref"],
        }
    return {
        "recipe_id": f"{recipe.code}-{immortal_tag}",
        "recipe_code": recipe.code,
        "immortal_tag": immortal_tag,
        "assignment_id": assignment_id,
        "assignment_ref": assignment_ref,
        "assignment_sha256": assignment_sha256,
        "pointer_ref": pointer_ref,
        "assignment": assignment,
        "audit": audit,
        "refresh_pointer": {
            "schema_id": "curvyzero_opponent_assignment_refresh_pointer/v0",
            "pointer_ref": pointer_ref,
            "pointer_volume": "control",
            "assignment_ref": assignment_ref,
            "assignment_sha256": assignment_sha256,
            "recipe_id": f"{recipe.code}-{immortal_tag}",
            "audit": {
                "schema_id": "curvyzero_opponent_assignment_refresh_pointer_audit/v0",
                "reason": "initial cz26 current-grid pointer",
                "assignment_id": assignment_id,
                "assignment_ref": assignment_ref,
                "assignment_sha256": assignment_sha256,
                "recipe_code": recipe.code,
                "immortal_tag": immortal_tag,
            },
        },
    }


def _train_function_name(compute: str) -> str:
    if compute == "cpu":
        return "lightzero_curvytron_visual_survival_cpu"
    if compute == COMPUTE_L4_T4_CPU40:
        return "lightzero_curvytron_visual_survival_gpu_cpu40"
    if compute == "gpu-h100-cpu40":
        return "lightzero_curvytron_visual_survival_h100_cpu40"
    raise ValueError(f"unsupported compute {compute!r}")


def _train_kwargs(
    args: argparse.Namespace,
    *,
    run_id: str,
    attempt_id: str,
    seed: int,
    reward_alpha: float,
    noise: NoiseMode,
    assignment_ref: str,
    pointer_ref: str,
    initial_policy_checkpoint_ref: str,
) -> dict[str, Any]:
    return {
        "mode": "train",
        "seed": seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "max_env_step": args.max_env_step,
        "max_train_iter": args.max_train_iter,
        "source_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "decision_ms": CURVYTRON_DECISION_MS,
        "collector_env_num": args.collector_env_num,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": args.n_episode,
        "num_simulations": args.num_simulations,
        "batch_size": args.batch_size,
        "lightzero_eval_freq": 0,
        "skip_lightzero_eval_in_profile": True,
        "profile_cuda_sync_enabled": False,
        "profile_allow_auto_resume": False,
        "profile_volume_commit": False,
        "lightzero_multi_gpu": False,
        "save_ckpt_after_iter": args.save_ckpt_after_iter,
        "commit_on_checkpoint": CURVYTRON_COMMIT_ON_CHECKPOINT,
        "stop_after_learner_train_calls": args.stop_after_learner_train_calls,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        "reward_outcome_alpha": float(reward_alpha),
        "source_state_trail_render_mode": CURVYTRON_POLICY_TRAIL_RENDER_MODE,
        "source_state_bonus_render_mode": CURVYTRON_POLICY_BONUS_RENDER_MODE,
        "learner_seat_mode": args.learner_seat_mode,
        "ego_action_straight_override_probability": noise.straight_override_probability,
        "policy_action_repeat_min": noise.repeat_min,
        "policy_action_repeat_max": noise.repeat_max,
        "policy_action_repeat_extra_probability": noise.repeat_extra_probability,
        "control_noise_profile_id": noise.profile_id,
        "disable_death_for_profile": False,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "normal",
        "env_telemetry_stride": 1,
        "env_manager_type": args.env_manager_type,
        "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_use_cuda": False,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_report_ref": None,
        "opponent_checkpoint_state_key": None,
        "initial_policy_checkpoint_ref": initial_policy_checkpoint_ref,
        "initial_policy_checkpoint_state_key": None,
        "initial_policy_checkpoint_load_mode": "matching_shape",
        "opponent_mixture_spec": None,
        "opponent_assignment_ref": assignment_ref,
        "opponent_assignment_refresh_interval_train_iter": (
            args.assignment_refresh_interval_train_iter
        ),
        "opponent_assignment_refresh_ref": pointer_ref,
        "background_eval_enabled": True,
        "background_eval_launch_kind": "poller",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "live_checkpoint",
        "background_eval_seed_count": args.background_eval_seed_count,
        "background_eval_seed_rng_seed": _derived_seed("eval", seed),
        "background_eval_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": args.background_eval_num_simulations,
        "background_eval_batch_size": args.background_eval_batch_size,
        "background_gif_enabled": True,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": 4096,
        "background_gif_frame_stride": 4,
        "background_gif_fps": CURVYTRON_BACKGROUND_GIF_FPS,
        "background_gif_scale": 4,
        "background_gif_frame_size": 128,
        "background_gif_collect_temperature": 1.0,
        "background_gif_collect_epsilon": 0.25,
    }


def _poller_kwargs(
    args: argparse.Namespace,
    *,
    run_id: str,
    attempt_id: str,
    seed: int,
    reward_alpha: float,
    assignment_ref: str,
) -> dict[str, Any]:
    exp_name_ref = _ref(
        "training",
        TASK_ID,
        run_id,
        "attempts",
        attempt_id,
        "train",
        "lightzero_exp",
    )
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "exp_name_ref": exp_name_ref,
        "seed": seed,
        "source_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        "reward_outcome_alpha": float(reward_alpha),
        "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
        "opponent_mixture_spec": None,
        "opponent_assignment_ref": assignment_ref,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "normal",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "live_checkpoint",
        "background_eval_seed_count": args.background_eval_seed_count,
        "background_eval_seed_rng_seed": _derived_seed("eval", seed),
        "background_eval_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": args.background_eval_num_simulations,
        "background_eval_batch_size": args.background_eval_batch_size,
        "background_gif_enabled": True,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": 4096,
        "background_gif_frame_stride": 4,
        "background_gif_fps": CURVYTRON_BACKGROUND_GIF_FPS,
        "background_gif_scale": 4,
        "background_gif_frame_size": 128,
        "background_gif_collect_temperature": 1.0,
        "background_gif_collect_epsilon": 0.25,
        "poll_interval_sec": args.background_eval_poll_interval_sec,
        "stable_polls": 1,
        "max_runtime_sec": args.background_eval_poller_max_runtime_sec,
        "idle_after_train_done_sec": 60.0,
    }


def _row(
    args: argparse.Namespace,
    *,
    row_number: int,
    run_row_number: int,
    grid: str,
    recipe: Recipe,
    reward_alpha: float,
    noise: NoiseMode,
    leaderboard_immortal_probability: float,
    artifact: Mapping[str, Any],
    initial_policy_checkpoint_ref: str,
) -> dict[str, Any]:
    batch = {
        "grid_a": CURVYTRON_GRID_A_BATCH,
        "grid_b": CURVYTRON_GRID_B_BATCH,
        "canary": CURVYTRON_CANARY_BATCH,
    }[grid]
    reward_tag = reward_alpha_tag(reward_alpha)
    immortal_tag = leaderboard_immortal_tag(leaderboard_immortal_probability)
    run_id = curvytron_run_id(
        batch=batch,
        row_number=run_row_number,
        reward_tag=reward_tag,
        noise_tag=noise.tag,
        immortal_tag=immortal_tag,
        recipe_code=recipe.code,
    )
    attempt_id = curvytron_attempt_id(run_id)
    seed = _derived_seed(args.matrix_name, grid, row_number, run_row_number, run_id)
    train_kwargs = _train_kwargs(
        args,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        reward_alpha=reward_alpha,
        noise=noise,
        assignment_ref=str(artifact["assignment_ref"]),
        pointer_ref=str(artifact["pointer_ref"]),
        initial_policy_checkpoint_ref=initial_policy_checkpoint_ref,
    )
    poller_kwargs = _poller_kwargs(
        args,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        reward_alpha=reward_alpha,
        assignment_ref=str(artifact["assignment_ref"]),
    )
    train_ref = _ref("training", TASK_ID, run_id, "attempts", attempt_id, "train")
    return {
        "schema_id": ROW_SCHEMA_ID,
        "matrix_name": args.matrix_name,
        "row_id": f"r{row_number:03d}",
        "grid_row_id": f"r{run_row_number:03d}",
        "grid": grid,
        "row_kind": f"{grid}_training",
        "status": "ready_for_operator_launch_gate",
        "label": run_id,
        "plain_name": run_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "mode": "train",
        "canonical_launcher": "scripts/submit_curvytron_survivaldiag_manifest.py",
        "calls_stock_train_muzero": True,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        "reward_outcome_alpha": float(reward_alpha),
        "reward_tag": reward_tag,
        "noise_tag": noise.tag,
        "leaderboard_immortal_probability": float(leaderboard_immortal_probability),
        "leaderboard_immortal_tag": immortal_tag,
        "recipe_code": recipe.code,
        "recipe_role": recipe.role,
        "recipe_description": recipe.description,
        "recipe_slot_counts": dict(recipe.slot_counts),
        "opponent_assignment_ref": artifact["assignment_ref"],
        "opponent_assignment_refresh_ref": artifact["pointer_ref"],
        "initial_policy_checkpoint_ref": initial_policy_checkpoint_ref,
        "source_state_trail_render_mode": CURVYTRON_POLICY_TRAIL_RENDER_MODE,
        "source_state_bonus_render_mode": CURVYTRON_POLICY_BONUS_RENDER_MODE,
        "learner_seat_mode": args.learner_seat_mode,
        "training_seed": seed,
        "deployed_app_submission": {
            "app_name": DEFAULT_CURVYTRON_TRAIN_APP_NAME,
            "train_function": _train_function_name(args.compute),
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
            "spawn_order": ["poller", "train"],
        },
        "train_kwargs": train_kwargs,
        "poller_kwargs": poller_kwargs,
        "artifact_refs": {
            "summary": _ref(train_ref, "summary.json"),
            "progress_latest": _ref(train_ref, "progress_latest.json"),
            "checkpoint_root": _ref(train_ref, "lightzero_exp", "ckpt"),
            "background_eval_status": _ref(train_ref, "checkpoint_eval_poller.json"),
            "background_gif_status": _ref(train_ref, "background_gif_jobs.json"),
            "assignment_refresh_events": _ref(
                train_ref,
                "opponent_assignment_refresh_events.jsonl",
            ),
        },
    }


def _iter_grid_rows(args: argparse.Namespace) -> Iterable[tuple[str, Recipe, float, NoiseMode, float]]:
    include_grid_a = args.profile in {"grid-a", "full", "canary"}
    include_grid_b = args.profile in {"grid-b", "full"}
    if include_grid_a:
        recipes = GRID_A_RECIPES[:1] if args.profile == "canary" else GRID_A_RECIPES
        alphas = (1.0,) if args.profile == "canary" else GRID_A_ALPHAS
        noises = (0.0,) if args.profile == "canary" else GRID_A_NOISES
        immortal_probabilities = (0.0,) if args.profile == "canary" else LEADERBOARD_IMMORTAL_PROBABILITIES
        for recipe in recipes:
            for alpha in alphas:
                for noise_probability in noises:
                    for immortal_probability in immortal_probabilities:
                        yield (
                            "grid_a" if args.profile != "canary" else "canary",
                            recipe,
                            alpha,
                            _noise_mode(noise_probability),
                            immortal_probability,
                        )
    if include_grid_b:
        for recipe in GRID_B_RECIPES:
            for alpha in GRID_B_ALPHAS:
                for noise_probability in GRID_B_NOISES:
                    for immortal_probability in LEADERBOARD_IMMORTAL_PROBABILITIES:
                        yield ("grid_b", recipe, alpha, _noise_mode(noise_probability), immortal_probability)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    top_checkpoints = _load_top_checkpoints(args)
    initial_policy_checkpoint_ref = str(top_checkpoints["rank1"]["checkpoint_ref"])
    args.assignment_source_ref = (
        args.assignment_source_ref
        or str(args.checkpoint_refs_file or args.ratings_snapshot or "unknown_source")
    )
    artifacts: dict[tuple[str, float], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    row_number = 0
    grid_row_numbers: dict[str, int] = {"grid_a": 0, "grid_b": 0, "canary": 0}
    for grid, recipe, alpha, noise, immortal_probability in _iter_grid_rows(args):
        key = (recipe.code, float(immortal_probability))
        if key not in artifacts:
            artifacts[key] = _assignment_artifact(
                args=args,
                recipe=recipe,
                top_checkpoints=top_checkpoints,
                leaderboard_immortal_probability=immortal_probability,
            )
        row_number += 1
        grid_row_numbers[grid] += 1
        rows.append(
            _row(
                args,
                row_number=row_number,
                run_row_number=grid_row_numbers[grid],
                grid=grid,
                recipe=recipe,
                reward_alpha=alpha,
                noise=noise,
                leaderboard_immortal_probability=immortal_probability,
                artifact=artifacts[key],
                initial_policy_checkpoint_ref=initial_policy_checkpoint_ref,
            )
        )
    assignments = {
        str(artifact["recipe_id"]): {
            "assignment_id": artifact["assignment_id"],
            "assignment_ref": artifact["assignment_ref"],
            "assignment_sha256": artifact["assignment_sha256"],
            "recipe_id": artifact["recipe_id"],
            "assignment": artifact["assignment"],
            "audit": artifact["audit"],
        }
        for artifact in artifacts.values()
    }
    refresh_pointers = {
        str(artifact["recipe_id"]): artifact["refresh_pointer"]
        for artifact in artifacts.values()
    }
    refresh_pointer_refs = [
        str(artifact["pointer_ref"])
        for artifact in sorted(artifacts.values(), key=lambda item: str(item["recipe_id"]))
    ]
    config = {
        "schema_id": "curvyzero_training_candidate_refresh_config/v0",
        "active": True,
        "written_at": _utc_timestamp(),
        "tournament_id": args.tournament_id,
        "rating_run_id": args.rating_run_id,
        "leaderboard_id": args.leaderboard_id,
        "assignment_bank_run_id": args.training_candidate_assignment_bank_run_id,
        "assignment_bank_attempt_id": args.training_candidate_assignment_bank_attempt_id,
        "assignment_id_prefix": args.training_candidate_assignment_id_prefix,
        "assignment_seed": args.training_candidate_assignment_seed,
        "min_active_count": args.training_candidate_min_active_count,
        "active_min_valid_games": args.training_candidate_active_min_valid_games,
        "active_min_distinct_opponents": (
            args.training_candidate_active_min_distinct_opponents
        ),
        "max_active_rank": args.training_candidate_max_active_rank,
        "allow_partial_assignment": False,
        "refresh_pointers": refresh_pointer_refs,
    }
    seed_checkpoint_refs = [
        top_checkpoints[f"rank{rank}"]["checkpoint_ref"] for rank in range(1, 5)
    ]
    tournament_run_ids = sorted(str(row["run_id"]) for row in rows)
    intake_seed_spec = {
        "tournament_id": args.tournament_id,
        "rating_run_id": args.rating_run_id,
        "checkpoint_refs": seed_checkpoint_refs,
        "run_ids": tournament_run_ids,
        "run_id_prefix": "",
        "checkpoint_selection": "latest" if args.profile == "canary" else "all",
        "max_runs": 0,
    }
    manifest = {
        "schema_id": SCHEMA_ID,
        "status": "ready_for_operator_launch_gate",
        "matrix_name": args.matrix_name,
        "matrix_profile": args.profile,
        "generated_at": _utc_timestamp(),
        "row_count": len(rows),
        "grid_a_row_count": sum(1 for row in rows if row["grid"] == "grid_a"),
        "grid_b_row_count": sum(1 for row in rows if row["grid"] == "grid_b"),
        "canary_row_count": sum(1 for row in rows if row["grid"] == "canary"),
        "tournament": {
            "tournament_id": args.tournament_id,
            "rating_run_id": args.rating_run_id,
            "leaderboard_id": args.leaderboard_id,
            "seed_checkpoint_refs": seed_checkpoint_refs,
            "intake_seed_spec": intake_seed_spec,
            "run_id_prefixes": sorted(
                {CURVYTRON_GRID_A_BATCH + "-", CURVYTRON_GRID_B_BATCH + "-"}
                if args.profile != "canary"
                else {CURVYTRON_CANARY_BATCH + "-"}
            ),
        },
        "top_checkpoint_source": top_checkpoints,
        "assignment_bank": {
            "run_id": args.assignment_bank_run_id,
            "attempt_id": args.assignment_bank_attempt_id,
            "source_ref": args.assignment_source_ref,
            "target_volume": "control",
            "assignments": assignments,
            "refresh_pointer_volume": "control",
            "refresh_pointer_run_id": args.pointer_run_id,
            "refresh_pointer_attempt_id": args.pointer_attempt_id,
            "refresh_pointers": refresh_pointers,
        },
        "training_candidate_refresh_controller": {
            "config_ref": args.training_candidate_refresh_config_ref,
            "config_volume": "control",
            "config": config,
        },
        "axes": {
            "grid_a_recipes": [
                {"code": recipe.code, "slot_counts": dict(recipe.slot_counts)}
                for recipe in GRID_A_RECIPES
            ],
            "grid_b_recipes": [
                {"code": recipe.code, "slot_counts": dict(recipe.slot_counts)}
                for recipe in GRID_B_RECIPES
            ],
            "grid_a_reward_outcome_alpha": list(GRID_A_ALPHAS),
            "grid_b_reward_outcome_alpha": list(GRID_B_ALPHAS),
            "grid_a_noise": list(GRID_A_NOISES),
            "grid_b_noise": list(GRID_B_NOISES),
            "leaderboard_immortal_probability": list(
                LEADERBOARD_IMMORTAL_PROBABILITIES
            ),
        },
        "fixed_knobs": {
            "compute": args.compute,
            "collector_env_num": args.collector_env_num,
            "n_episode": args.n_episode,
            "num_simulations": args.num_simulations,
            "batch_size": args.batch_size,
            "max_train_iter": args.max_train_iter,
            "max_env_step": args.max_env_step,
            "source_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
            "save_ckpt_after_iter": args.save_ckpt_after_iter,
            "assignment_refresh_interval_train_iter": (
                args.assignment_refresh_interval_train_iter
            ),
            "source_state_trail_render_mode": CURVYTRON_POLICY_TRAIL_RENDER_MODE,
            "source_state_bonus_render_mode": CURVYTRON_POLICY_BONUS_RENDER_MODE,
            "learner_seat_mode": args.learner_seat_mode,
            "initial_policy_checkpoint_ref": initial_policy_checkpoint_ref,
        },
        "guards": {
            "deployed_app_name": DEFAULT_CURVYTRON_TRAIN_APP_NAME,
            "launch_script": "scripts/submit_curvytron_survivaldiag_manifest.py",
            "operator_launch_gate_required": True,
            "expected_row_count": _expected_row_count(args.profile),
            "modal_launch_performed": False,
            "all_rows_seed_from_rank1": True,
            "current_code_names": True,
            "refresh_controller_config_required": True,
        },
        "rows": rows,
    }
    _validate_manifest(manifest)
    return manifest


def _expected_row_count(profile: str) -> int:
    return {
        "canary": 1,
        "grid-a": 96,
        "grid-b": 40,
        "full": 136,
    }[profile]


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    rows = list(manifest["rows"])
    expected = int(manifest["guards"]["expected_row_count"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, got {len(rows)}")
    run_ids = [str(row["run_id"]) for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id values")
    initial_refs = {
        str(row["train_kwargs"]["initial_policy_checkpoint_ref"]) for row in rows
    }
    if len(initial_refs) != 1:
        raise ValueError("all rows must seed from the same rank1 checkpoint")
    for row in rows:
        train_kwargs = row["train_kwargs"]
        poller_kwargs = row["poller_kwargs"]
        missing = [
            key for key in TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT if key not in train_kwargs
        ]
        if missing:
            raise ValueError(f"row {row['row_id']} missing train kwargs {missing}")
        if (
            len(row["run_id"]) > MAX_MODAL_RUN_ID_LEN
            or len(row["attempt_id"]) > MAX_MODAL_RUN_ID_LEN
        ):
            raise ValueError(f"row {row['row_id']} has overlong Modal id")
        if train_kwargs["opponent_assignment_ref"] != poller_kwargs["opponent_assignment_ref"]:
            raise ValueError(f"row {row['row_id']} train/poller assignment refs differ")
        if (
            train_kwargs["opponent_mixture_spec"] is not None
            or poller_kwargs["opponent_mixture_spec"] is not None
        ):
            raise ValueError(f"row {row['row_id']} must use assignment refs, not inline mixture")
        expected_refresh_interval = int(
            manifest["fixed_knobs"]["assignment_refresh_interval_train_iter"]
        )
        if (
            train_kwargs["opponent_assignment_refresh_interval_train_iter"]
            != expected_refresh_interval
        ):
            raise ValueError(f"row {row['row_id']} has wrong refresh interval")
        expected_checkpoint_cadence = int(manifest["fixed_knobs"]["save_ckpt_after_iter"])
        if train_kwargs["save_ckpt_after_iter"] != expected_checkpoint_cadence:
            raise ValueError(f"row {row['row_id']} has wrong checkpoint cadence")
        if train_kwargs["learner_seat_mode"] != DEFAULT_LEARNER_SEAT_MODE:
            raise ValueError(f"row {row['row_id']} has wrong learner seat mode")
    assignment_bank = manifest["assignment_bank"]
    pointer_refs = {
        str(row["train_kwargs"]["opponent_assignment_refresh_ref"]) for row in rows
    }
    configured_refs = set(
        manifest["training_candidate_refresh_controller"]["config"]["refresh_pointers"]
    )
    if pointer_refs - configured_refs:
        raise ValueError("training candidate config does not cover all row pointers")
    for artifact in assignment_bank["assignments"].values():
        parse_opponent_assignment_snapshot(artifact["assignment"])
        if artifact["assignment_sha256"] != canonical_assignment_json_sha256(
            artifact["assignment"]
        ):
            raise ValueError("assignment sha mismatch")
    for pointer in assignment_bank["refresh_pointers"].values():
        if pointer["assignment_ref"] not in {
            artifact["assignment_ref"] for artifact in assignment_bank["assignments"].values()
        }:
            raise ValueError("refresh pointer references unknown assignment")


def _write_outputs(manifest: Mapping[str, Any], output_root: Path) -> dict[str, str]:
    output_dir = output_root / str(manifest["matrix_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{manifest['matrix_name']}.json"
    rows_path = output_dir / f"{manifest['matrix_name']}.rows.jsonl"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in manifest["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"manifest_json": str(manifest_path), "rows_jsonl": str(rows_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("canary", "grid-a", "grid-b", "full"), default="full")
    parser.add_argument("--matrix-name", default="cz26-full-20260516a")
    parser.add_argument("--ratings-snapshot", type=Path)
    parser.add_argument("--checkpoint-refs-file", type=Path)
    parser.add_argument("--assignment-source-ref", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--compute", default=COMPUTE_L4_T4_CPU40)
    parser.add_argument("--collector-env-num", type=int, default=CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM)
    parser.add_argument("--n-episode", type=int, default=CURVYTRON_DEFAULT_N_EPISODE)
    parser.add_argument("--num-simulations", type=int, default=CURVYTRON_DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--batch-size", type=int, default=CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--max-env-step", type=int, default=CURVYTRON_DEFAULT_MAX_ENV_STEP)
    parser.add_argument("--max-train-iter", type=int, default=CURVYTRON_DEFAULT_MAX_TRAIN_ITER)
    parser.add_argument("--save-ckpt-after-iter", type=int, default=CURVYTRON_SAVE_CKPT_AFTER_ITER)
    parser.add_argument("--stop-after-learner-train-calls", type=int, default=0)
    parser.add_argument("--env-manager-type", choices=("base", "subprocess"), default="subprocess")
    parser.add_argument("--learner-seat-mode", default=DEFAULT_LEARNER_SEAT_MODE)
    parser.add_argument(
        "--assignment-refresh-interval-train-iter",
        type=int,
        default=CURVYTRON_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER,
    )
    parser.add_argument("--background-eval-seed-count", type=int, default=3)
    parser.add_argument("--background-eval-num-simulations", type=int, default=8)
    parser.add_argument("--background-eval-batch-size", type=int, default=64)
    parser.add_argument("--background-eval-poll-interval-sec", type=float, default=120.0)
    parser.add_argument("--background-eval-poller-max-runtime-sec", type=float, default=6 * 60 * 60)
    parser.add_argument("--tournament-id", default=DEFAULT_TOURNAMENT_ID)
    parser.add_argument("--rating-run-id", default=DEFAULT_RATING_RUN_ID)
    parser.add_argument("--leaderboard-id", default=DEFAULT_LEADERBOARD_ID)
    parser.add_argument("--assignment-bank-run-id", default=DEFAULT_ASSIGNMENT_BANK_RUN_ID)
    parser.add_argument("--assignment-bank-attempt-id", default=DEFAULT_ASSIGNMENT_BANK_ATTEMPT_ID)
    parser.add_argument("--pointer-run-id", default=DEFAULT_POINTER_RUN_ID)
    parser.add_argument("--pointer-attempt-id", default=DEFAULT_POINTER_ATTEMPT_ID)
    parser.add_argument("--training-candidate-refresh-config-ref", default=DEFAULT_REFRESH_CONFIG_REF)
    parser.add_argument("--training-candidate-assignment-bank-run-id", default="cz26-training-candidates")
    parser.add_argument("--training-candidate-assignment-bank-attempt-id", default="try-cz26-training-candidates")
    parser.add_argument("--training-candidate-assignment-id-prefix", default="cz26-auto")
    parser.add_argument("--training-candidate-assignment-seed", type=int, default=20260516)
    parser.add_argument("--training-candidate-min-active-count", type=int, default=1)
    parser.add_argument("--training-candidate-active-min-valid-games", type=int, default=21)
    parser.add_argument("--training-candidate-active-min-distinct-opponents", type=int, default=1)
    parser.add_argument("--training-candidate-max-active-rank", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_manifest(args)
    if args.stdout_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    outputs = _write_outputs(manifest, args.output_root)
    print(
        json.dumps(
            {
                "ok": True,
                "matrix_name": manifest["matrix_name"],
                "profile": manifest["matrix_profile"],
                "row_count": manifest["row_count"],
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
