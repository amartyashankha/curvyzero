"""Pure policy-row mapping helpers for simultaneous CurvyTron batches.

The helpers here are intentionally independent from LightZero, MCTS, and the
fixture actor bridge. They only translate between environment-shaped multiplayer
arrays and one-row-per-ego policy decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


POLICY_ROW_MAPPING_SCHEMA = "curvyzero_policy_row_mapping/v0"
ACTION_COUNT = 3
NOOP_ACTION_ID = 1
PADDED_ROW_ID = -1


@dataclass(frozen=True)
class PolicyRowMapping:
    """A compact or padded view of live ego-player policy rows."""

    observations: np.ndarray
    legal_action_mask: np.ndarray
    env_row_id: np.ndarray
    player_id: np.ndarray
    row_mask: np.ndarray
    source_shape: tuple[int, int]
    action_count: int
    schema: str = POLICY_ROW_MAPPING_SCHEMA

    @property
    def capacity(self) -> int:
        """Total rows exposed to the policy, including padding."""

        return int(self.row_mask.shape[0])

    @property
    def active_count(self) -> int:
        """Number of real live/legal policy rows."""

        return int(self.row_mask.sum())


def build_policy_row_mapping(
    obs: np.ndarray,
    live_mask: np.ndarray,
    legal_action_mask: np.ndarray,
    *,
    pad_to: int | None = None,
) -> PolicyRowMapping:
    """Map ``obs[B,P,...]`` into compact or padded policy rows.

    A policy row is emitted only when the player is live and has at least one
    legal action after applying the live mask. Dead, inactive, terminal, and
    no-legal-action rows are filtered out. When ``pad_to`` is provided, active
    rows are placed first and the remaining rows carry ``row_mask=False``.
    """

    obs_array = np.asarray(obs)
    if obs_array.ndim < 2:
        raise ValueError("obs must have shape [B,P,...]")
    batch_size, player_count = (int(obs_array.shape[0]), int(obs_array.shape[1]))

    live = _bool_array(live_mask, "live_mask")
    if live.shape != (batch_size, player_count):
        raise ValueError(
            "live_mask must have shape [B,P]; "
            f"got {live.shape!r} for obs shape {obs_array.shape!r}"
        )

    legal = _bool_array(legal_action_mask, "legal_action_mask")
    if legal.ndim != 3 or legal.shape[:2] != (batch_size, player_count):
        raise ValueError(
            "legal_action_mask must have shape [B,P,A]; "
            f"got {legal.shape!r} for obs shape {obs_array.shape!r}"
        )
    action_count = int(legal.shape[2])
    if action_count <= 0:
        raise ValueError("legal_action_mask must include at least one action")

    filtered_legal = legal & live[:, :, None]
    decision_mask = live & filtered_legal.any(axis=2)
    active_count = int(decision_mask.sum())
    capacity = _policy_row_capacity(active_count, pad_to)

    env_ids = np.repeat(np.arange(batch_size, dtype=np.int32)[:, None], player_count, axis=1)
    player_ids = np.repeat(
        np.arange(player_count, dtype=np.int16)[None, :],
        batch_size,
        axis=0,
    )

    active_obs = obs_array[decision_mask]
    active_legal = filtered_legal[decision_mask]
    active_env_ids = env_ids[decision_mask]
    active_player_ids = player_ids[decision_mask]

    observations = np.zeros((capacity, *obs_array.shape[2:]), dtype=obs_array.dtype)
    output_legal = np.zeros((capacity, action_count), dtype=bool)
    output_env_ids = np.full(capacity, PADDED_ROW_ID, dtype=np.int32)
    output_player_ids = np.full(capacity, PADDED_ROW_ID, dtype=np.int16)
    row_mask = np.zeros(capacity, dtype=bool)

    if active_count:
        observations[:active_count] = active_obs
        output_legal[:active_count] = active_legal
        output_env_ids[:active_count] = active_env_ids
        output_player_ids[:active_count] = active_player_ids
        row_mask[:active_count] = True

    return PolicyRowMapping(
        observations=observations,
        legal_action_mask=output_legal,
        env_row_id=output_env_ids,
        player_id=output_player_ids,
        row_mask=row_mask,
        source_shape=(batch_size, player_count),
        action_count=action_count,
    )


def policy_rows_to_joint_action(
    mapping: PolicyRowMapping,
    selected_action_ids: np.ndarray,
    *,
    noop_action_id: int = NOOP_ACTION_ID,
    validate_legal: bool = True,
    dtype: np.dtype | type[np.integer] = np.int8,
) -> np.ndarray:
    """Map selected ego action ids back to ``joint_action[B,P]``.

    Dead, inactive, filtered, and padded rows are left as ``noop_action_id``.
    ``selected_action_ids`` may have either one value per mapping row or one
    value per active row.
    """

    selected = _integer_1d(selected_action_ids, "selected_action_ids")
    selected_full = _selected_actions_for_capacity(mapping, selected, noop_action_id)
    active_rows = np.asarray(mapping.row_mask, dtype=bool)
    active_selected = selected_full[active_rows]

    if active_selected.size:
        out_of_range = (active_selected < 0) | (active_selected >= mapping.action_count)
        if out_of_range.any():
            bad_offsets = np.nonzero(out_of_range)[0].astype(int).tolist()
            raise ValueError(f"selected_action_ids out of range for active rows: {bad_offsets}")
        if validate_legal:
            legal = np.asarray(mapping.legal_action_mask, dtype=bool)[active_rows]
            legal_selected = legal[np.arange(active_selected.size), active_selected]
            if not legal_selected.all():
                bad_offsets = np.nonzero(~legal_selected)[0].astype(int).tolist()
                raise ValueError(
                    f"selected_action_ids contains illegal active-row actions: {bad_offsets}"
                )

    batch_size, player_count = mapping.source_shape
    joint_action = np.full((batch_size, player_count), noop_action_id, dtype=dtype)
    if active_selected.size:
        env_ids = np.asarray(mapping.env_row_id, dtype=np.int64)[active_rows]
        player_ids = np.asarray(mapping.player_id, dtype=np.int64)[active_rows]
        joint_action[env_ids, player_ids] = active_selected.astype(dtype, copy=False)
    return joint_action


def _policy_row_capacity(active_count: int, pad_to: int | None) -> int:
    if pad_to is None:
        return active_count
    if not isinstance(pad_to, int):
        raise ValueError("pad_to must be an integer when provided")
    if pad_to < active_count:
        raise ValueError(
            f"pad_to={pad_to} is smaller than active policy row count {active_count}"
        )
    return pad_to


def _bool_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        array = array.astype(bool)
    return array


def _integer_1d(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [R]")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer action ids")
    return array.astype(np.int64, copy=False)


def _selected_actions_for_capacity(
    mapping: PolicyRowMapping,
    selected: np.ndarray,
    noop_action_id: int,
) -> np.ndarray:
    if selected.shape == (mapping.capacity,):
        return selected
    if selected.shape == (mapping.active_count,):
        selected_full = np.full(mapping.capacity, noop_action_id, dtype=np.int64)
        selected_full[np.asarray(mapping.row_mask, dtype=bool)] = selected
        return selected_full
    raise ValueError(
        "selected_action_ids must have one value per policy row or active row; "
        f"got {selected.shape!r}, capacity={mapping.capacity}, active_count={mapping.active_count}"
    )
