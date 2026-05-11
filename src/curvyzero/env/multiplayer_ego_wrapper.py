"""Metadata-only ego-row wrapper over the public multiplayer env."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import numpy as np

from curvyzero.env.vector_multiplayer_env import VectorMultiplayerBatch
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.training.multiplayer_opponent_policy import FixedActionOpponentPolicy
from curvyzero.training.multiplayer_opponent_policy import MultiplayerOpponentPolicy
from curvyzero.training.multiplayer_opponent_policy import OpponentPolicySelection
from curvyzero.training.policy_row_mapping import NOOP_ACTION_ID
from curvyzero.training.policy_row_mapping import PolicyRowMapping
from curvyzero.training.policy_row_mapping import build_policy_row_mapping
from curvyzero.training.policy_row_mapping import policy_rows_to_joint_action


MULTIPLAYER_EGO_WRAPPER_ID = "curvyzero_metadata_only_multiplayer_ego_wrapper/v0"
MULTIPLAYER_EGO_ACTION_SIDECAR_SCHEMA_ID = (
    "curvyzero_metadata_only_multiplayer_ego_action_sidecar/v0"
)
MULTIPLAYER_EGO_ACTION_MAP_POLICY_ID = (
    "ego_policy_plus_named_opponent_policy_full_joint_action_no_mcts/v0"
)


@dataclass(frozen=True, slots=True)
class MultiplayerEgoPolicyRows:
    """Policy rows for one configured ego seat per public env row."""

    mapping: PolicyRowMapping
    ego_player_id: np.ndarray
    observation_schema_id: str | None
    observation_schema_hash: str | None
    metadata_only: bool = True
    learned_observation_claim: bool = False
    joint_action_mcts_claim: bool = False
    wrapper_id: str = MULTIPLAYER_EGO_WRAPPER_ID


@dataclass(frozen=True, slots=True)
class MultiplayerEgoActionMap:
    """Full env action map built from ego actions plus named opponent fills."""

    joint_action: np.ndarray
    ego_joint_action: np.ndarray
    opponent_selection: OpponentPolicySelection
    action_sidecar: dict[str, Any]
    opponent_policy_sidecar: dict[str, Any]


class MetadataOnlyMultiplayerEgoWrapper:
    """Thin wrapper that keeps multiplayer self-play plumbing metadata-only."""

    def __init__(
        self,
        env: VectorMultiplayerEnv,
        *,
        ego_player_id: int | np.ndarray = 0,
        opponent_policy: MultiplayerOpponentPolicy | None = None,
        pad_to: int | None = None,
    ) -> None:
        self.env = env
        self.ego_player_id = normalize_ego_player_id(
            ego_player_id,
            batch_size=env.batch_size,
            player_count=env.player_count,
        )
        self.opponent_policy = opponent_policy or FixedActionOpponentPolicy()
        self.pad_to = pad_to
        self._last_batch: VectorMultiplayerBatch | None = None
        self._decision_index = 0

    @property
    def decision_index(self) -> int:
        """Number of ego-wrapper decisions already sent to the public env."""

        return self._decision_index

    def reset(self, *args: Any, **kwargs: Any) -> VectorMultiplayerBatch:
        """Reset the wrapped env and keep its metadata-only public batch."""

        batch = self.env.reset(*args, **kwargs)
        self._last_batch = batch
        self._decision_index = 0
        return batch

    def observe(
        self,
        batch: VectorMultiplayerBatch | None = None,
        *,
        pad_to: int | None = None,
    ) -> MultiplayerEgoPolicyRows:
        """Build one metadata-only ego policy row per live configured ego seat."""

        source = self._batch_or_last(batch)
        return build_multiplayer_ego_policy_rows(
            source.observation,
            source.action_mask,
            ego_player_id=self.ego_player_id,
            observation_schema_id=_optional_info_string(source.info, "observation_schema_id"),
            observation_schema_hash=_optional_info_string(source.info, "observation_schema_hash"),
            pad_to=self.pad_to if pad_to is None else pad_to,
        )

    def build_action_map(
        self,
        selected_ego_action_ids: np.ndarray,
        *,
        batch: VectorMultiplayerBatch | None = None,
        rows: MultiplayerEgoPolicyRows | None = None,
    ) -> MultiplayerEgoActionMap:
        """Return the full ``[B,P]`` wrapper action map without stepping."""

        source = self._batch_or_last(batch)
        ego_rows = self.observe(source) if rows is None else rows
        return build_multiplayer_ego_action_map(
            ego_rows,
            selected_ego_action_ids,
            source.action_mask,
            opponent_policy=self.opponent_policy,
            decision_index=self._decision_index,
            observation=source.observation,
            present=source.info.get("present"),
            alive=source.info.get("alive"),
            done=source.done,
        )

    def step(self, selected_ego_action_ids: np.ndarray) -> VectorMultiplayerBatch:
        """Step the public env using ego actions and wrapper-filled opponents."""

        action_map = self.build_action_map(selected_ego_action_ids)
        batch = self.env.step(action_map.joint_action)
        info = dict(batch.info)
        info["multiplayer_ego_wrapper_id"] = MULTIPLAYER_EGO_WRAPPER_ID
        info["wrapper_joint_action"] = action_map.joint_action.copy()
        info["multiplayer_ego_action_sidecar"] = action_map.action_sidecar
        info["opponent_policy_sidecar"] = action_map.opponent_policy_sidecar
        info["metadata_only"] = True
        info["trainer_observation_claim"] = False
        info["trainer_replay_claim"] = False
        info["learned_observation_claim"] = False
        info["joint_action_mcts_claim"] = False
        wrapped = replace(batch, info=info)
        self._last_batch = wrapped
        self._decision_index += 1
        return wrapped

    def _batch_or_last(
        self,
        batch: VectorMultiplayerBatch | None,
    ) -> VectorMultiplayerBatch:
        source = self._last_batch if batch is None else batch
        if source is None:
            raise RuntimeError("reset must be called before observing or stepping")
        return source


def build_multiplayer_ego_policy_rows(
    observation: np.ndarray,
    legal_action_mask: np.ndarray,
    *,
    ego_player_id: int | np.ndarray,
    observation_schema_id: str | None = None,
    observation_schema_hash: str | None = None,
    pad_to: int | None = None,
) -> MultiplayerEgoPolicyRows:
    """Project public ``obs[B,P,...]`` to one configured ego row per env row."""

    obs = np.asarray(observation)
    legal = np.asarray(legal_action_mask, dtype=bool)
    if obs.ndim < 2:
        raise ValueError("observation must have shape [B,P,...]")
    if legal.ndim != 3:
        raise ValueError("legal_action_mask must have shape [B,P,A]")
    if legal.shape[:2] != obs.shape[:2]:
        raise ValueError(
            "legal_action_mask must share observation batch/player shape; "
            f"got obs={obs.shape!r}, legal={legal.shape!r}"
        )
    batch_size, player_count = int(obs.shape[0]), int(obs.shape[1])
    ego_ids = normalize_ego_player_id(
        ego_player_id,
        batch_size=batch_size,
        player_count=player_count,
    )
    ego_live = np.zeros((batch_size, player_count), dtype=bool)
    env_rows = np.arange(batch_size, dtype=np.int64)
    ego_live[env_rows, ego_ids] = legal[env_rows, ego_ids].any(axis=1)
    mapping = build_policy_row_mapping(obs, ego_live, legal, pad_to=pad_to)
    return MultiplayerEgoPolicyRows(
        mapping=mapping,
        ego_player_id=ego_ids,
        observation_schema_id=observation_schema_id,
        observation_schema_hash=observation_schema_hash,
    )


def build_multiplayer_ego_action_map(
    rows: MultiplayerEgoPolicyRows,
    selected_ego_action_ids: np.ndarray,
    legal_action_mask: np.ndarray,
    *,
    opponent_policy: MultiplayerOpponentPolicy,
    decision_index: int = 0,
    observation: np.ndarray | None = None,
    present: np.ndarray | None = None,
    alive: np.ndarray | None = None,
    done: np.ndarray | None = None,
    noop_action_id: int = NOOP_ACTION_ID,
) -> MultiplayerEgoActionMap:
    """Build full wrapper actions and sidecars for one ego-row decision batch."""

    legal = np.asarray(legal_action_mask, dtype=bool)
    if legal.ndim != 3 or legal.shape[:2] != rows.mapping.source_shape:
        raise ValueError(
            "legal_action_mask must have shape [B,P,A] matching ego rows; "
            f"got {legal.shape!r}, expected {rows.mapping.source_shape!r}"
        )
    batch_size, player_count = rows.mapping.source_shape
    live = legal.any(axis=2)
    present_mask = _status_mask(present, "present", fallback=live | _configured_ego_mask(rows))
    alive_mask = _status_mask(alive, "alive", fallback=live)
    done_mask = _done_mask(done, batch_size=batch_size)

    ego_joint_action = policy_rows_to_joint_action(
        rows.mapping,
        selected_ego_action_ids,
        noop_action_id=noop_action_id,
        validate_legal=True,
        dtype=np.int16,
    )
    ego_action_mask = _active_ego_slots(rows.mapping)
    opponent_mask = live & ~ego_action_mask
    opponent_selection = opponent_policy.select_actions(
        legal,
        opponent_mask,
        decision_index=decision_index,
        observation=observation,
    )

    joint_action = ego_joint_action.astype(np.int16, copy=True)
    joint_action[opponent_mask] = opponent_selection.actions[opponent_mask]
    action_source = _initial_action_source(
        present_mask,
        alive_mask,
        done_mask,
        player_count=player_count,
    )
    action_source[opponent_mask] = "opponent_policy"
    action_source[ego_action_mask] = "ego_policy"

    opponent_sidecar = opponent_selection.sidecar()
    opponent_sidecar.update(
        {
            "ego_player_id": rows.ego_player_id.astype(np.int16, copy=True),
            "opponent_player_ids": _opponent_player_ids(
                opponent_mask,
                player_count=player_count,
            ),
            "trainer_observation_claim": False,
        }
    )
    action_sidecar = {
        "schema_id": MULTIPLAYER_EGO_ACTION_SIDECAR_SCHEMA_ID,
        "wrapper_id": MULTIPLAYER_EGO_WRAPPER_ID,
        "action_map_policy_id": MULTIPLAYER_EGO_ACTION_MAP_POLICY_ID,
        "joint_action_semantics": "full_simultaneous_env_action_map_not_mcts/v0",
        "branching_policy": "ego_rows_only_opponents_are_sidecar_fills/v0",
        "metadata_only": True,
        "trainer_observation_claim": False,
        "learned_observation_claim": False,
        "joint_action_mcts_claim": False,
        "observation_schema_id": rows.observation_schema_id,
        "observation_schema_hash": rows.observation_schema_hash,
        "source_shape": rows.mapping.source_shape,
        "ego_player_id": rows.ego_player_id.astype(np.int16, copy=True),
        "ego_row_env_id": rows.mapping.env_row_id[rows.mapping.row_mask].astype(
            np.int32,
            copy=True,
        ),
        "ego_row_player_id": rows.mapping.player_id[rows.mapping.row_mask].astype(
            np.int16,
            copy=True,
        ),
        "ego_row_mask": rows.mapping.row_mask.astype(bool, copy=True),
        "ego_action": ego_joint_action.astype(np.int16, copy=True),
        "wrapper_joint_action": joint_action.astype(np.int16, copy=True),
        "opponent_action": opponent_selection.actions.astype(np.int16, copy=True),
        "action_source": action_source.copy(),
        "opponent_policy_id": opponent_selection.policy_id,
        "opponent_policy_version": opponent_selection.policy_version,
        "opponent_policy_seed": int(opponent_selection.seed),
    }
    if opponent_selection.policy_metadata is not None:
        action_sidecar["opponent_policy_metadata"] = dict(
            opponent_selection.policy_metadata
        )
    return MultiplayerEgoActionMap(
        joint_action=joint_action,
        ego_joint_action=ego_joint_action,
        opponent_selection=opponent_selection,
        action_sidecar=action_sidecar,
        opponent_policy_sidecar=opponent_sidecar,
    )


def normalize_ego_player_id(
    ego_player_id: int | np.ndarray,
    *,
    batch_size: int,
    player_count: int,
) -> np.ndarray:
    """Return one ego player id per env row."""

    value = np.asarray(ego_player_id)
    if value.ndim == 0:
        if isinstance(value.item(), bool):
            raise ValueError("ego_player_id must contain integer player indexes")
        ids = np.full(batch_size, int(value.item()), dtype=np.int16)
    elif value.shape == (batch_size,):
        if value.dtype == np.dtype(bool) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError("ego_player_id must contain integer player indexes")
        ids = value.astype(np.int16, copy=True)
    else:
        raise ValueError("ego_player_id must be scalar or shape [B]")
    invalid = (ids < 0) | (ids >= player_count)
    if bool(invalid.any()):
        raise ValueError(
            f"ego_player_id entries must be in [0, {player_count}); "
            f"got {ids.astype(int).tolist()}"
        )
    return ids


def _active_ego_slots(mapping: PolicyRowMapping) -> np.ndarray:
    mask = np.zeros(mapping.source_shape, dtype=bool)
    active = np.asarray(mapping.row_mask, dtype=bool)
    if bool(active.any()):
        env_ids = np.asarray(mapping.env_row_id, dtype=np.int64)[active]
        player_ids = np.asarray(mapping.player_id, dtype=np.int64)[active]
        mask[env_ids, player_ids] = True
    return mask


def _configured_ego_mask(rows: MultiplayerEgoPolicyRows) -> np.ndarray:
    mask = np.zeros(rows.mapping.source_shape, dtype=bool)
    env_rows = np.arange(rows.mapping.source_shape[0], dtype=np.int64)
    mask[env_rows, rows.ego_player_id.astype(np.int64, copy=False)] = True
    return mask


def _status_mask(
    value: Any,
    name: str,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    if value is None:
        return np.asarray(fallback, dtype=bool).copy()
    mask = np.asarray(value, dtype=bool)
    if mask.shape != fallback.shape:
        raise ValueError(f"{name} must have shape {fallback.shape!r}")
    return mask.copy()


def _done_mask(value: Any, *, batch_size: int) -> np.ndarray:
    if value is None:
        return np.zeros(batch_size, dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.shape != (batch_size,):
        raise ValueError(f"done must have shape {(batch_size,)!r}")
    return mask.copy()


def _initial_action_source(
    present: np.ndarray,
    alive: np.ndarray,
    done: np.ndarray,
    *,
    player_count: int,
) -> np.ndarray:
    source = np.full(present.shape, "inactive_noop", dtype=object)
    source[~present] = "absent_noop"
    source[present & ~alive] = "dead_noop"
    terminal = np.repeat(done[:, None], player_count, axis=1)
    source[terminal] = "terminal_padding"
    return source


def _opponent_player_ids(
    opponent_mask: np.ndarray,
    *,
    player_count: int,
) -> tuple[tuple[int, ...], ...]:
    del player_count
    return tuple(tuple(int(player) for player in np.flatnonzero(row)) for row in opponent_mask)


def _optional_info_string(info: dict[str, Any], key: str) -> str | None:
    value = info.get(key)
    return value if isinstance(value, str) else None
