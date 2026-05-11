"""Coach-facing no-train smoke for the multiplayer ego-row wrapper.

This module is intentionally metadata-only. It does not import LightZero,
register a trainable env, or call any trainer entrypoint. Its only job is to
prove that Coach/optimizer plumbing can inspect reset shape, ego action-map
shape, and wrapper sidecars through the newly landed multiplayer wrapper path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
import json
from typing import Any

import numpy as np

from curvyzero.env.multiplayer_ego_wrapper import (
    MULTIPLAYER_EGO_ACTION_SIDECAR_SCHEMA_ID,
    MULTIPLAYER_EGO_WRAPPER_ID,
    MetadataOnlyMultiplayerEgoWrapper,
)
from curvyzero.env.vector_multiplayer_env import (
    ACTION_COUNT,
    DEBUG_METADATA_OBSERVATION_SCHEMA_ID,
    VectorMultiplayerBatch,
    VectorMultiplayerEnv,
)
from curvyzero.training.multiplayer_opponent_policy import (
    FIXED_ACTION_OPPONENT_POLICY_ID,
    FixedActionOpponentPolicy,
)


MULTIPLAYER_EGO_LIGHTZERO_COACH_SMOKE_ID = (
    "curvyzero_metadata_only_multiplayer_ego_lightzero_coach_smoke/v0"
)
WRAPPER_CLASS_PATH = "curvyzero.env.multiplayer_ego_wrapper.MetadataOnlyMultiplayerEgoWrapper"
OPPONENT_POLICY_CLASS_PATH = (
    "curvyzero.training.multiplayer_opponent_policy.FixedActionOpponentPolicy"
)
PUBLIC_ENV_CLASS_PATH = "curvyzero.env.vector_multiplayer_env.VectorMultiplayerEnv"


@dataclass(frozen=True, slots=True)
class MultiplayerEgoLightZeroCoachSmokeRequest:
    """Tiny metadata-only request for the wrapper-backed Coach smoke."""

    seed: int = 0
    batch_size: int = 2
    player_count: int = 3
    ego_player_id: int | tuple[int, ...] = 0
    selected_ego_action_id: int = 0
    fixed_opponent_action_id: int = 1
    decision_ms: float = 50.0
    max_ticks: int = 64
    body_capacity: int = 128
    event_capacity: int = 16
    pad_to: int | None = None


def build_multiplayer_ego_lightzero_coach_smoke_report(
    request: MultiplayerEgoLightZeroCoachSmokeRequest | None = None,
    *,
    include_step_info_keys: bool = False,
) -> dict[str, Any]:
    """Build and validate a no-train wrapper smoke report."""

    request = request or MultiplayerEgoLightZeroCoachSmokeRequest()
    _validate_request(request)

    env = VectorMultiplayerEnv(
        batch_size=request.batch_size,
        player_count=request.player_count,
        seed=request.seed,
        decision_ms=request.decision_ms,
        max_ticks=request.max_ticks,
        body_capacity=request.body_capacity,
        event_capacity=request.event_capacity,
    )
    wrapper = MetadataOnlyMultiplayerEgoWrapper(
        env,
        ego_player_id=_ego_player_id_for_wrapper(request.ego_player_id),
        opponent_policy=FixedActionOpponentPolicy(action_id=request.fixed_opponent_action_id),
        pad_to=request.pad_to,
    )

    reset_batch = wrapper.reset(seed=request.seed)
    rows = wrapper.observe()
    selected_ego_actions = np.full(
        rows.mapping.active_count,
        int(request.selected_ego_action_id),
        dtype=np.int16,
    )
    action_map = wrapper.build_action_map(selected_ego_actions, rows=rows)
    step_batch = wrapper.step(selected_ego_actions)

    validation_problems = _validate_smoke_surfaces(
        request=request,
        reset_batch=reset_batch,
        rows=rows,
        selected_ego_actions=selected_ego_actions,
        action_map=action_map,
        step_batch=step_batch,
    )

    result: dict[str, Any] = {
        "ok": not validation_problems,
        "smoke_id": MULTIPLAYER_EGO_LIGHTZERO_COACH_SMOKE_ID,
        "label": "CurvyZero multiplayer ego-row LightZero/Coach metadata-only smoke",
        "mode": "wrapper_reset_action_map_sidecar_only",
        "call_policy": "does_not_train; does_not_import_lightzero; does_not_call_trainer",
        "metadata_only": True,
        "trainer_observation_claim": False,
        "trainer_replay_claim": False,
        "learned_observation_claim": False,
        "joint_action_mcts_claim": False,
        "quality_claim": "none",
        "source_fidelity_claim": "none",
        "uses_ale": False,
        "uses_atari": False,
        "registered_lightzero_env": False,
        "called_training_entrypoint": False,
        "request": asdict(request),
        "semantics": {
            "single_wrapper_source": WRAPPER_CLASS_PATH,
            "public_env_source": PUBLIC_ENV_CLASS_PATH,
            "opponent_policy_source": OPPONENT_POLICY_CLASS_PATH,
            "wrapper_id": MULTIPLAYER_EGO_WRAPPER_ID,
            "forked_curvytron_trainer_path": False,
        },
        "coach_config_surface": _coach_config_surface(request),
        "reset_surface": _reset_surface(reset_batch),
        "policy_rows_surface": _policy_rows_surface(rows),
        "action_map_surface": _action_map_surface(action_map),
        "step_surface": _step_surface(step_batch, include_info_keys=include_step_info_keys),
        "optimizer_coach_alignment": _optimizer_coach_alignment_note(),
        "validation_problems": validation_problems,
    }
    return _to_plain(result)


def _validate_request(request: MultiplayerEgoLightZeroCoachSmokeRequest) -> None:
    if int(request.batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    if int(request.player_count) not in (2, 3, 4):
        raise ValueError("player_count must be 2, 3, or 4")
    for name in ("selected_ego_action_id", "fixed_opponent_action_id"):
        value = int(getattr(request, name))
        if value < 0 or value >= ACTION_COUNT:
            raise ValueError(f"{name} must be in [0, {ACTION_COUNT})")
    if request.pad_to is not None and int(request.pad_to) < int(request.batch_size):
        raise ValueError("pad_to must be at least batch_size for the reset smoke")


def _validate_smoke_surfaces(
    *,
    request: MultiplayerEgoLightZeroCoachSmokeRequest,
    reset_batch: VectorMultiplayerBatch,
    rows: Any,
    selected_ego_actions: np.ndarray,
    action_map: Any,
    step_batch: VectorMultiplayerBatch,
) -> list[str]:
    problems: list[str] = []
    expected_batch_player_shape = (int(request.batch_size), int(request.player_count))
    expected_action_shape = (*expected_batch_player_shape, ACTION_COUNT)

    if reset_batch.observation.shape != (*expected_batch_player_shape, 6):
        problems.append(
            f"reset observation shape {reset_batch.observation.shape!r}, "
            f"expected {(*expected_batch_player_shape, 6)!r}"
        )
    if reset_batch.action_mask.shape != expected_action_shape:
        problems.append(
            f"reset action_mask shape {reset_batch.action_mask.shape!r}, "
            f"expected {expected_action_shape!r}"
        )
    _expect_metadata_only_info(
        reset_batch.info,
        "reset",
        problems,
        required_false_keys=("trainer_observation_claim",),
    )
    if reset_batch.info.get("observation_schema_id") != DEBUG_METADATA_OBSERVATION_SCHEMA_ID:
        problems.append(
            "reset observation_schema_id must stay "
            f"{DEBUG_METADATA_OBSERVATION_SCHEMA_ID!r}"
        )

    if rows.wrapper_id != MULTIPLAYER_EGO_WRAPPER_ID:
        problems.append(f"rows.wrapper_id={rows.wrapper_id!r}, expected {MULTIPLAYER_EGO_WRAPPER_ID!r}")
    if rows.metadata_only is not True:
        problems.append("policy rows metadata_only must be true")
    if rows.learned_observation_claim is not False:
        problems.append("policy rows learned_observation_claim must be false")
    if rows.joint_action_mcts_claim is not False:
        problems.append("policy rows joint_action_mcts_claim must be false")
    if rows.observation_schema_id != DEBUG_METADATA_OBSERVATION_SCHEMA_ID:
        problems.append("policy rows must carry only debug metadata observation schema")
    if rows.mapping.active_count != int(request.batch_size):
        problems.append(
            f"active ego rows {rows.mapping.active_count!r}, expected {int(request.batch_size)!r}"
        )
    if selected_ego_actions.shape != (rows.mapping.active_count,):
        problems.append("selected ego actions must have one value per active ego row")

    if action_map.joint_action.shape != expected_batch_player_shape:
        problems.append(
            f"joint_action shape {action_map.joint_action.shape!r}, "
            f"expected {expected_batch_player_shape!r}"
        )
    if action_map.joint_action.dtype != np.int16:
        problems.append(f"joint_action dtype {action_map.joint_action.dtype}, expected int16")
    _expect_wrapper_sidecar(action_map.action_sidecar, "action_map", problems)
    if action_map.opponent_policy_sidecar.get("policy_id") != FIXED_ACTION_OPPONENT_POLICY_ID:
        problems.append("opponent sidecar must use the fixed-action policy")
    if action_map.opponent_policy_sidecar.get("learned_observation_claim") is not False:
        problems.append("opponent sidecar learned_observation_claim must be false")

    _expect_metadata_only_info(
        step_batch.info,
        "step",
        problems,
        required_false_keys=(
            "trainer_observation_claim",
            "trainer_replay_claim",
            "learned_observation_claim",
        ),
    )
    if step_batch.info.get("multiplayer_ego_wrapper_id") != MULTIPLAYER_EGO_WRAPPER_ID:
        problems.append("step info must name the multiplayer ego wrapper id")
    if step_batch.info.get("joint_action_mcts_claim") is not False:
        problems.append("step joint_action_mcts_claim must be false")
    _expect_wrapper_sidecar(
        step_batch.info.get("multiplayer_ego_action_sidecar", {}),
        "step",
        problems,
    )
    wrapper_joint_action = np.asarray(step_batch.info.get("wrapper_joint_action"))
    if wrapper_joint_action.shape != expected_batch_player_shape:
        problems.append(
            f"step wrapper_joint_action shape {wrapper_joint_action.shape!r}, "
            f"expected {expected_batch_player_shape!r}"
        )
    return problems


def _expect_metadata_only_info(
    info: dict[str, Any],
    label: str,
    problems: list[str],
    *,
    required_false_keys: tuple[str, ...],
) -> None:
    if info.get("metadata_only") is not True:
        problems.append(f"{label} metadata_only must be true")
    for key in required_false_keys:
        if info.get(key) is not False:
            problems.append(f"{label} {key} must be false")


def _expect_wrapper_sidecar(
    sidecar: dict[str, Any],
    label: str,
    problems: list[str],
) -> None:
    if sidecar.get("schema_id") != MULTIPLAYER_EGO_ACTION_SIDECAR_SCHEMA_ID:
        problems.append(f"{label} sidecar schema_id must be {MULTIPLAYER_EGO_ACTION_SIDECAR_SCHEMA_ID}")
    if sidecar.get("wrapper_id") != MULTIPLAYER_EGO_WRAPPER_ID:
        problems.append(f"{label} sidecar wrapper_id must be {MULTIPLAYER_EGO_WRAPPER_ID}")
    for key in (
        "metadata_only",
        "trainer_observation_claim",
        "learned_observation_claim",
        "joint_action_mcts_claim",
    ):
        expected = key == "metadata_only"
        if sidecar.get(key) is not expected:
            problems.append(f"{label} sidecar {key} must be {expected}")


def _coach_config_surface(request: MultiplayerEgoLightZeroCoachSmokeRequest) -> dict[str, Any]:
    return {
        "trainable": False,
        "registered_lightzero_env": False,
        "target_surface": "metadata_only_multiplayer_ego_wrapper",
        "wrapper_class": WRAPPER_CLASS_PATH,
        "opponent_policy_class": OPPONENT_POLICY_CLASS_PATH,
        "player_count": int(request.player_count),
        "action_space_size": ACTION_COUNT,
        "model_observation_shape": None,
        "model_observation_schema_id": None,
        "public_observation_schema_id": DEBUG_METADATA_OBSERVATION_SCHEMA_ID,
        "reason_model_observation_absent": (
            "public multiplayer observations are debug metadata only; no learned "
            "2P/3P/4P observation schema is claimed here"
        ),
    }


def _reset_surface(batch: VectorMultiplayerBatch) -> dict[str, Any]:
    return {
        "observation": _array_summary(batch.observation),
        "action_mask": _array_summary(batch.action_mask),
        "reward": _array_summary(batch.reward),
        "done": _array_summary(batch.done),
        "observation_schema_id": batch.info.get("observation_schema_id"),
        "observation_schema_hash": batch.info.get("observation_schema_hash"),
        "metadata_only": batch.info.get("metadata_only"),
        "trainer_observation_claim": batch.info.get("trainer_observation_claim"),
        "learned_observation_claim": batch.info.get("learned_observation_claim"),
    }


def _policy_rows_surface(rows: Any) -> dict[str, Any]:
    mapping = rows.mapping
    return {
        "wrapper_id": rows.wrapper_id,
        "metadata_only": rows.metadata_only,
        "trainer_observation_claim": False,
        "learned_observation_claim": rows.learned_observation_claim,
        "joint_action_mcts_claim": rows.joint_action_mcts_claim,
        "observation_schema_id": rows.observation_schema_id,
        "observation_schema_hash": rows.observation_schema_hash,
        "mapping_schema": mapping.schema,
        "source_shape": mapping.source_shape,
        "capacity": mapping.capacity,
        "active_count": mapping.active_count,
        "observations": _array_summary(mapping.observations),
        "legal_action_mask": _array_summary(mapping.legal_action_mask),
        "env_row_id": mapping.env_row_id,
        "player_id": mapping.player_id,
        "row_mask": mapping.row_mask,
        "ego_player_id": rows.ego_player_id,
    }


def _action_map_surface(action_map: Any) -> dict[str, Any]:
    sidecar = action_map.action_sidecar
    opponent_sidecar = action_map.opponent_policy_sidecar
    return {
        "joint_action": action_map.joint_action,
        "joint_action_summary": _array_summary(action_map.joint_action),
        "ego_joint_action": action_map.ego_joint_action,
        "ego_joint_action_summary": _array_summary(action_map.ego_joint_action),
        "sidecar_schema_id": sidecar.get("schema_id"),
        "sidecar_wrapper_id": sidecar.get("wrapper_id"),
        "sidecar_flags": _metadata_flags(sidecar),
        "action_source": sidecar.get("action_source"),
        "opponent_policy_id": sidecar.get("opponent_policy_id"),
        "opponent_policy_version": sidecar.get("opponent_policy_version"),
        "opponent_policy_sidecar_schema_id": opponent_sidecar.get("schema_id"),
        "opponent_policy_sidecar_flags": _metadata_flags(opponent_sidecar),
        "opponent_actions": opponent_sidecar.get("actions"),
        "opponent_mask": opponent_sidecar.get("opponent_mask"),
    }


def _step_surface(
    batch: VectorMultiplayerBatch,
    *,
    include_info_keys: bool,
) -> dict[str, Any]:
    sidecar = batch.info.get("multiplayer_ego_action_sidecar", {})
    result = {
        "observation": _array_summary(batch.observation),
        "action_mask": _array_summary(batch.action_mask),
        "reward": _array_summary(batch.reward),
        "done": _array_summary(batch.done),
        "wrapper_joint_action": batch.info.get("wrapper_joint_action"),
        "wrapper_joint_action_summary": _array_summary(batch.info.get("wrapper_joint_action")),
        "multiplayer_ego_wrapper_id": batch.info.get("multiplayer_ego_wrapper_id"),
        "sidecar_schema_id": sidecar.get("schema_id"),
        "sidecar_wrapper_id": sidecar.get("wrapper_id"),
        "sidecar_flags": _metadata_flags(sidecar),
        "metadata_only": batch.info.get("metadata_only"),
        "trainer_observation_claim": batch.info.get("trainer_observation_claim"),
        "trainer_replay_claim": batch.info.get("trainer_replay_claim"),
        "learned_observation_claim": batch.info.get("learned_observation_claim"),
        "joint_action_mcts_claim": batch.info.get("joint_action_mcts_claim"),
    }
    if include_info_keys:
        result["info_keys"] = sorted(str(key) for key in batch.info)
    return result


def _optimizer_coach_alignment_note() -> dict[str, Any]:
    return {
        "single_semantics_source": WRAPPER_CLASS_PATH,
        "opponent_policy_source": OPPONENT_POLICY_CLASS_PATH,
        "coach_instruction": (
            "Use this only as a shape/metadata smoke for reset, action maps, and "
            "sidecars; do not schedule collection or training from it."
        ),
        "optimizer_instruction": (
            "Treat the report as a guard that Coach is still entering through the "
            "metadata-only ego wrapper, not a duplicate CurvyTron trainer path."
        ),
        "observation_boundary": (
            "No learned multiplayer observation shape is claimed until a real schema "
            "lands and replaces the debug metadata-only surface."
        ),
    }


def _metadata_flags(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata_only": mapping.get("metadata_only"),
        "trainer_observation_claim": mapping.get("trainer_observation_claim"),
        "trainer_replay_claim": mapping.get("trainer_replay_claim"),
        "learned_observation_claim": mapping.get("learned_observation_claim"),
        "joint_action_mcts_claim": mapping.get("joint_action_mcts_claim"),
    }


def _ego_player_id_for_wrapper(value: int | tuple[int, ...]) -> int | np.ndarray:
    if isinstance(value, tuple):
        return np.asarray(value, dtype=np.int16)
    return int(value)


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
    }


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--player-count", type=int, default=3)
    parser.add_argument("--ego-player-id", type=int, default=0)
    parser.add_argument("--selected-ego-action-id", type=int, default=0)
    parser.add_argument("--fixed-opponent-action-id", type=int, default=1)
    parser.add_argument("--include-step-info-keys", action="store_true")
    args = parser.parse_args(argv)

    request = MultiplayerEgoLightZeroCoachSmokeRequest(
        seed=args.seed,
        batch_size=args.batch_size,
        player_count=args.player_count,
        ego_player_id=args.ego_player_id,
        selected_ego_action_id=args.selected_ego_action_id,
        fixed_opponent_action_id=args.fixed_opponent_action_id,
    )
    report = build_multiplayer_ego_lightzero_coach_smoke_report(
        request,
        include_step_info_keys=args.include_step_info_keys,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


__all__ = [
    "MULTIPLAYER_EGO_LIGHTZERO_COACH_SMOKE_ID",
    "MultiplayerEgoLightZeroCoachSmokeRequest",
    "build_multiplayer_ego_lightzero_coach_smoke_report",
]


if __name__ == "__main__":
    main()
