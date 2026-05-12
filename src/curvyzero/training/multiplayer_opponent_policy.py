"""Small deterministic opponent policies for metadata-only multiplayer rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from curvyzero.training.multiplayer_replay_contract import (
    MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID,
)


FIXED_ACTION_OPPONENT_POLICY_ID = "curvyzero_fixed_action_opponent"
RANDOM_LEGAL_OPPONENT_POLICY_ID = "curvyzero_seeded_random_legal_opponent"
SNAPSHOT_BACKED_OPPONENT_POLICY_ID = "curvyzero_snapshot_backed_learned_opponent"
OPPONENT_POLICY_VERSION = "v0.2026-05-10"
NO_OPPONENT_ACTION = -1
_U64_MASK = (1 << 64) - 1
_I63_MASK = (1 << 63) - 1


class MultiplayerOpponentPolicy(Protocol):
    """Protocol for wrapper-owned, metadata-only opponent action suppliers."""

    policy_id: str
    policy_version: str
    seed: int

    def select_actions(
        self,
        legal_action_mask: np.ndarray,
        opponent_mask: np.ndarray,
        *,
        decision_index: int = 0,
        observation: np.ndarray | None = None,
    ) -> "OpponentPolicySelection":
        """Choose action ids for the true entries of ``opponent_mask``."""


class SnapshotOpponentActionProvider(Protocol):
    """Inference hook used by snapshot-backed learned opponent policies."""

    provider_id: str
    provider_version: str

    def select_action(
        self,
        *,
        observation: np.ndarray | None,
        legal_action_mask: np.ndarray,
        decision_index: int,
        env_row: int,
        player_id: int,
        action_seed: int,
        snapshot_ref: str,
        checkpoint_ref: str | None = None,
    ) -> "OpponentActionChoice | int":
        """Return one legal action for one live opponent slot."""


@dataclass(frozen=True, slots=True)
class OpponentActionChoice:
    """One provider action choice, optionally with model log-prob metadata."""

    action_id: int
    action_logp: float | None = None


@dataclass(frozen=True, slots=True)
class OpponentPolicySelection:
    """Opponent policy choices plus replay-compatible metadata."""

    policy_id: str
    policy_version: str
    seed: int
    actions: np.ndarray
    action_seed: np.ndarray
    action_logp: np.ndarray
    opponent_mask: np.ndarray
    decision_index: int
    policy_metadata: Mapping[str, Any] | None = None

    def sidecar(self) -> dict[str, Any]:
        """Return the batch-level sidecar consumed by metadata replay helpers."""

        batch_size = int(self.actions.shape[0])
        sidecar = {
            "schema_id": MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "seed": np.full(batch_size, int(self.seed), dtype=np.int64),
            "actions": self.actions.astype(np.int16, copy=True),
            "action_seed": self.action_seed.astype(np.int64, copy=True),
            "action_logp": self.action_logp.astype(np.float32, copy=True),
            "opponent_mask": self.opponent_mask.astype(bool, copy=True),
            "decision_index": int(self.decision_index),
            "metadata_only": True,
            "trainer_replay_claim": False,
            "learned_observation_claim": False,
        }
        if self.policy_metadata is not None:
            sidecar["policy_metadata"] = dict(self.policy_metadata)
        return sidecar


@dataclass(frozen=True, slots=True)
class FixedActionOpponentPolicy:
    """Always play one legal action for every live opponent slot."""

    action_id: int = 1
    policy_id: str = FIXED_ACTION_OPPONENT_POLICY_ID
    policy_version: str = OPPONENT_POLICY_VERSION
    seed: int = 0

    def select_actions(
        self,
        legal_action_mask: np.ndarray,
        opponent_mask: np.ndarray,
        *,
        decision_index: int = 0,
        observation: np.ndarray | None = None,
    ) -> OpponentPolicySelection:
        del observation
        legal, opponents = _validated_policy_inputs(legal_action_mask, opponent_mask)
        action_id = int(self.action_id)
        if action_id < 0 or action_id >= legal.shape[2]:
            raise ValueError(f"fixed opponent action {action_id!r} is outside action space")
        illegal = opponents & ~legal[:, :, action_id]
        if bool(illegal.any()):
            rows = np.argwhere(illegal).astype(int).tolist()
            raise ValueError(
                f"fixed opponent action {action_id!r} is illegal for opponent slots {rows}"
            )

        actions = np.full(legal.shape[:2], NO_OPPONENT_ACTION, dtype=np.int16)
        actions[opponents] = action_id
        action_seed = np.full(legal.shape[:2], NO_OPPONENT_ACTION, dtype=np.int64)
        action_logp = np.full(legal.shape[:2], np.nan, dtype=np.float32)
        action_logp[opponents] = 0.0
        return OpponentPolicySelection(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            seed=_nonnegative_seed(self.seed),
            actions=actions,
            action_seed=action_seed,
            action_logp=action_logp,
            opponent_mask=opponents,
            decision_index=int(decision_index),
        )


@dataclass(frozen=True, slots=True)
class SeededRandomOpponentPolicy:
    """Deterministically sample one legal action per opponent slot from a seed."""

    seed: int
    policy_id: str = RANDOM_LEGAL_OPPONENT_POLICY_ID
    policy_version: str = OPPONENT_POLICY_VERSION

    def select_actions(
        self,
        legal_action_mask: np.ndarray,
        opponent_mask: np.ndarray,
        *,
        decision_index: int = 0,
        observation: np.ndarray | None = None,
    ) -> OpponentPolicySelection:
        del observation
        legal, opponents = _validated_policy_inputs(legal_action_mask, opponent_mask)
        seed = _nonnegative_seed(self.seed)
        actions = np.full(legal.shape[:2], NO_OPPONENT_ACTION, dtype=np.int16)
        action_seed = np.full(legal.shape[:2], NO_OPPONENT_ACTION, dtype=np.int64)
        action_logp = np.full(legal.shape[:2], np.nan, dtype=np.float32)

        for env_row, player_id in np.argwhere(opponents):
            legal_ids = np.flatnonzero(legal[int(env_row), int(player_id)])
            if legal_ids.size == 0:
                raise ValueError(
                    "opponent slot has no legal actions: "
                    f"env_row={int(env_row)}, player_id={int(player_id)}"
                )
            slot_seed = stable_opponent_action_seed(
                seed,
                int(decision_index),
                int(env_row),
                int(player_id),
            )
            actions[int(env_row), int(player_id)] = int(legal_ids[slot_seed % legal_ids.size])
            action_seed[int(env_row), int(player_id)] = slot_seed
            action_logp[int(env_row), int(player_id)] = -float(np.log(legal_ids.size))

        return OpponentPolicySelection(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            seed=seed,
            actions=actions,
            action_seed=action_seed,
            action_logp=action_logp,
            opponent_mask=opponents,
            decision_index=int(decision_index),
        )


@dataclass(frozen=True, slots=True)
class SnapshotBackedOpponentPolicy:
    """Delegate opponent actions to a frozen learned snapshot/checkpoint provider."""

    provider: SnapshotOpponentActionProvider
    snapshot_ref: str
    checkpoint_ref: str | None = None
    model_id: str | None = None
    seed: int = 0
    policy_id: str = SNAPSHOT_BACKED_OPPONENT_POLICY_ID
    policy_version: str = OPPONENT_POLICY_VERSION

    def select_actions(
        self,
        legal_action_mask: np.ndarray,
        opponent_mask: np.ndarray,
        *,
        decision_index: int = 0,
        observation: np.ndarray | None = None,
    ) -> OpponentPolicySelection:
        legal, opponents = _validated_policy_inputs(legal_action_mask, opponent_mask)
        obs = _optional_policy_observation(observation, legal_shape=legal.shape)
        seed = _nonnegative_seed(self.seed)
        snapshot_ref = _nonempty_string(self.snapshot_ref, "snapshot_ref")
        checkpoint_ref = _optional_string(self.checkpoint_ref, "checkpoint_ref")
        model_id = _optional_string(self.model_id, "model_id")
        actions = np.full(legal.shape[:2], NO_OPPONENT_ACTION, dtype=np.int16)
        action_seed = np.full(legal.shape[:2], NO_OPPONENT_ACTION, dtype=np.int64)
        action_logp = np.full(legal.shape[:2], np.nan, dtype=np.float32)

        for env_row, player_id in np.argwhere(opponents):
            row = int(env_row)
            player = int(player_id)
            slot_seed = stable_opponent_action_seed(seed, int(decision_index), row, player)
            slot_observation = None if obs is None else obs[row, player].copy()
            choice = self.provider.select_action(
                observation=slot_observation,
                legal_action_mask=legal[row, player].copy(),
                decision_index=int(decision_index),
                env_row=row,
                player_id=player,
                action_seed=slot_seed,
                snapshot_ref=snapshot_ref,
                checkpoint_ref=checkpoint_ref,
            )
            action_id, logp = _provider_choice_parts(choice)
            legal_choice = (
                0 <= action_id < legal.shape[2]
                and bool(legal[row, player, action_id])
            )
            if not legal_choice:
                raise ValueError(
                    "snapshot-backed opponent provider returned illegal action "
                    f"{action_id!r} for env_row={row}, player_id={player}"
                )
            actions[row, player] = action_id
            action_seed[row, player] = slot_seed
            if logp is not None:
                action_logp[row, player] = float(logp)

        provider_id = _nonempty_string(
            getattr(self.provider, "provider_id", self.provider.__class__.__name__),
            "provider_id",
        )
        provider_version = _nonempty_string(
            getattr(self.provider, "provider_version", "unknown"),
            "provider_version",
        )
        metadata = {
            "snapshot_ref": snapshot_ref,
            "checkpoint_ref": checkpoint_ref,
            "model_id": model_id,
            "provider_id": provider_id,
            "provider_version": provider_version,
            "opponent_kind": "snapshot_backed_learned_policy",
        }
        load_summary = getattr(self.provider, "load_summary", None)
        if callable(load_summary):
            load_summary = load_summary()
        if load_summary is not None:
            metadata["provider_load_summary"] = dict(load_summary)
        return OpponentPolicySelection(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            seed=seed,
            actions=actions,
            action_seed=action_seed,
            action_logp=action_logp,
            opponent_mask=opponents,
            decision_index=int(decision_index),
            policy_metadata=metadata,
        )


def random_legal_opponent_policy(seed: int) -> SeededRandomOpponentPolicy:
    """Named constructor for the seeded random legal opponent."""

    return SeededRandomOpponentPolicy(seed=seed)


def stable_opponent_action_seed(
    seed: int,
    decision_index: int,
    env_row: int,
    player_id: int,
) -> int:
    """Derive a stable non-negative slot seed without relying on Python hash."""

    x = (_nonnegative_seed(seed) + 0x9E3779B97F4A7C15) & _U64_MASK
    for value in (decision_index, env_row, player_id):
        item = (int(value) + 0x9E3779B97F4A7C15) & _U64_MASK
        x ^= (item + ((x << 6) & _U64_MASK) + (x >> 2)) & _U64_MASK
        x &= _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    x = (x ^ (x >> 31)) & _U64_MASK
    return int(x & _I63_MASK)


def _validated_policy_inputs(
    legal_action_mask: np.ndarray,
    opponent_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    legal = np.asarray(legal_action_mask, dtype=bool)
    if legal.ndim != 3:
        raise ValueError("legal_action_mask must have shape [B,P,A]")
    if legal.shape[2] <= 0:
        raise ValueError("legal_action_mask must include at least one action")
    opponents = np.asarray(opponent_mask, dtype=bool)
    if opponents.shape != legal.shape[:2]:
        raise ValueError(
            "opponent_mask must have shape [B,P]; "
            f"got {opponents.shape!r} for legal mask {legal.shape!r}"
        )
    no_legal = opponents & ~legal.any(axis=2)
    if bool(no_legal.any()):
        rows = np.argwhere(no_legal).astype(int).tolist()
        raise ValueError(f"opponent slots require at least one legal action: {rows}")
    return legal, opponents


def _optional_policy_observation(
    observation: np.ndarray | None,
    *,
    legal_shape: tuple[int, int, int],
) -> np.ndarray | None:
    if observation is None:
        return None
    obs = np.asarray(observation)
    if obs.ndim < 2 or obs.shape[:2] != legal_shape[:2]:
        raise ValueError(
            "opponent policy observation must have shape [B,P,...] matching legal mask; "
            f"got {obs.shape!r}, expected prefix {legal_shape[:2]!r}"
        )
    return obs


def _provider_choice_parts(choice: OpponentActionChoice | int) -> tuple[int, float | None]:
    if isinstance(choice, OpponentActionChoice):
        if isinstance(choice.action_id, bool):
            raise ValueError("provider action_id must be an integer action id")
        action_id = int(choice.action_id)
        logp = None if choice.action_logp is None else float(choice.action_logp)
        return action_id, logp
    if isinstance(choice, bool):
        raise ValueError("provider action choice must be an integer action id")
    return int(choice), None


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be None or a non-empty string")
    return value


def _nonnegative_seed(seed: int) -> int:
    if isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer")
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed_int
