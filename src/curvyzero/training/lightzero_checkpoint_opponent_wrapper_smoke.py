"""No-train smoke for snapshot-backed opponents inside the ego wrapper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.env.multiplayer_ego_wrapper import MetadataOnlyMultiplayerEgoWrapper
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerBatch
from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
)
from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    LIGHTZERO_CHECKPOINT_OPPONENT_PROVIDER_ID,
    snapshot_backed_lightzero_checkpoint_opponent_policy,
)
from curvyzero.training.multiplayer_opponent_policy import (
    OPPONENT_POLICY_VERSION,
    OpponentActionChoice,
    SnapshotBackedOpponentPolicy,
)


SNAPSHOT_OPPONENT_WRAPPER_SMOKE_ID = (
    "curvyzero_lightzero_checkpoint_opponent_wrapper_smoke/v0"
)
FAKE_PROVIDER_ID = "curvyzero_fake_visual_checkpoint_provider_for_wrapper_smoke"
FAKE_PROVIDER_VERSION = OPPONENT_POLICY_VERSION
VISUAL_ROW_SCHEMA_ID = "curvyzero_debug_visual_multiplayer_rows_for_wrapper_smoke/v0"


@dataclass(frozen=True, slots=True)
class FakeVisualCheckpointProvider:
    """Small provider used when no real checkpoint path is passed."""

    action_id: int = 1
    provider_id: str = FAKE_PROVIDER_ID
    provider_version: str = FAKE_PROVIDER_VERSION

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
    ) -> OpponentActionChoice:
        del decision_index, env_row, player_id, action_seed, snapshot_ref, checkpoint_ref
        if observation is None:
            raise ValueError("fake visual checkpoint provider requires observation")
        obs = np.asarray(observation)
        if obs.shape != DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE:
            raise ValueError(
                f"observation shape {obs.shape!r}; expected {DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE!r}"
            )
        legal = np.asarray(legal_action_mask, dtype=bool)
        action_id = int(self.action_id)
        if action_id < 0 or action_id >= legal.shape[0] or not bool(legal[action_id]):
            action_id = int(np.flatnonzero(legal)[0])
        return OpponentActionChoice(action_id=action_id, action_logp=0.0)


class TinyVisualMultiplayerEnv:
    """Tiny env stub with visual rows for wrapper-only smoke tests."""

    def __init__(self, *, batch_size: int = 1, player_count: int = 2) -> None:
        self.batch_size = int(batch_size)
        self.player_count = int(player_count)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.player_count < 2:
            raise ValueError("player_count must be at least 2")
        self._step_index = 0
        self.last_joint_action: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> VectorMultiplayerBatch:
        del seed
        self._step_index = 0
        self.last_joint_action = None
        return self._batch()

    def step(self, joint_action: np.ndarray) -> VectorMultiplayerBatch:
        actions = np.asarray(joint_action, dtype=np.int16)
        if actions.shape != (self.batch_size, self.player_count):
            raise ValueError(
                f"joint_action shape {actions.shape!r}; "
                f"expected {(self.batch_size, self.player_count)!r}"
            )
        self.last_joint_action = actions.copy()
        self._step_index += 1
        return self._batch(joint_action=actions)

    def _batch(
        self,
        *,
        joint_action: np.ndarray | None = None,
    ) -> VectorMultiplayerBatch:
        obs = np.zeros(
            (self.batch_size, self.player_count, *DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
            dtype=np.float32,
        )
        for env_row in range(self.batch_size):
            for player_id in range(self.player_count):
                obs[env_row, player_id, 0, 0, 0] = float(env_row + 1) / 10.0
                obs[env_row, player_id, 0, 0, 1] = float(player_id + 1) / 10.0
                obs[env_row, player_id, 0, 0, 2] = float(self._step_index) / 10.0
        action_mask = np.ones((self.batch_size, self.player_count, 3), dtype=bool)
        reward = np.zeros((self.batch_size, self.player_count), dtype=np.float32)
        done = np.zeros(self.batch_size, dtype=bool)
        info: dict[str, Any] = {
            "observation_schema_id": VISUAL_ROW_SCHEMA_ID,
            "observation_schema_hash": "smoke",
            "present": np.ones((self.batch_size, self.player_count), dtype=bool),
            "alive": np.ones((self.batch_size, self.player_count), dtype=bool),
            "joint_action_seen_by_env": None if joint_action is None else joint_action.copy(),
        }
        return VectorMultiplayerBatch(
            observation=obs,
            action_mask=action_mask,
            reward=reward,
            done=done,
            terminated=done.copy(),
            truncated=np.zeros_like(done),
            final_observation=None,
            final_reward=None,
            info=info,
        )


def run_lightzero_checkpoint_opponent_wrapper_smoke(
    *,
    checkpoint_path: str | None = None,
    snapshot_ref: str = "wrapper_smoke_snapshot",
    checkpoint_ref: str | None = None,
    seed: int = 0,
    ego_action_id: int = 0,
    fake_action_id: int = 1,
    num_simulations: int = 8,
    batch_size: int = 16,
    state_key: str | None = None,
) -> dict[str, Any]:
    env = TinyVisualMultiplayerEnv(batch_size=1, player_count=2)
    if checkpoint_path:
        opponent_policy = snapshot_backed_lightzero_checkpoint_opponent_policy(
            checkpoint_path=Path(checkpoint_path),
            snapshot_ref=snapshot_ref,
            checkpoint_ref=checkpoint_ref,
            seed=seed,
            num_simulations=num_simulations,
            batch_size=batch_size,
            state_key=state_key,
            illegal_action_policy="raise",
        )
        provider_kind = "lightzero_checkpoint"
    else:
        opponent_policy = SnapshotBackedOpponentPolicy(
            provider=FakeVisualCheckpointProvider(action_id=fake_action_id),
            snapshot_ref=snapshot_ref,
            checkpoint_ref=checkpoint_ref or "fake://wrapper-smoke",
            model_id="fake_visual_checkpoint_provider_for_wrapper_smoke",
            seed=seed,
        )
        provider_kind = "fake_visual_checkpoint"

    wrapper = MetadataOnlyMultiplayerEgoWrapper(
        env,
        ego_player_id=0,
        opponent_policy=opponent_policy,
    )
    reset_batch = wrapper.reset(seed=seed)
    rows = wrapper.observe()
    selected_ego_actions = np.full(rows.mapping.active_count, int(ego_action_id), dtype=np.int16)
    action_map = wrapper.build_action_map(selected_ego_actions, rows=rows)
    step_batch = wrapper.step(selected_ego_actions)
    problems = _validate_report(
        provider_kind=provider_kind,
        action_map=action_map,
        step_batch=step_batch,
        ego_action_id=ego_action_id,
    )
    return _to_plain(
        {
            "ok": not problems,
            "smoke_id": SNAPSHOT_OPPONENT_WRAPPER_SMOKE_ID,
            "mode": "no_train_wrapper_action_map_only",
            "provider_kind": provider_kind,
            "checkpoint_path": checkpoint_path,
            "snapshot_ref": snapshot_ref,
            "num_simulations": int(num_simulations),
            "batch_size": int(batch_size),
            "state_key": state_key,
            "reset_observation_shape": list(reset_batch.observation.shape),
            "opponent_provider_observation_shape": list(
                reset_batch.observation[0, 1].shape
            ),
            "active_ego_rows": int(rows.mapping.active_count),
            "joint_action": action_map.joint_action,
            "opponent_actions": action_map.opponent_policy_sidecar.get("actions"),
            "opponent_policy_metadata": action_map.action_sidecar.get(
                "opponent_policy_metadata"
            ),
            "step_wrapper_joint_action": step_batch.info.get("wrapper_joint_action"),
            "validation_problems": problems,
            "note": (
                "Without --checkpoint-path this only proves wrapper/provider plumbing. "
                "With --checkpoint-path it also loads and calls a frozen LightZero policy."
            ),
        }
    )


def _validate_report(
    *,
    provider_kind: str,
    action_map: Any,
    step_batch: VectorMultiplayerBatch,
    ego_action_id: int,
) -> list[str]:
    problems: list[str] = []
    if action_map.joint_action.shape != (1, 2):
        problems.append("joint_action must have shape [1,2]")
    if int(action_map.joint_action[0, 0]) != int(ego_action_id):
        problems.append("ego action did not land in player 0")
    if int(action_map.joint_action[0, 1]) < 0:
        problems.append("opponent action did not land in player 1")
    metadata = action_map.action_sidecar.get("opponent_policy_metadata") or {}
    expected_provider = (
        LIGHTZERO_CHECKPOINT_OPPONENT_PROVIDER_ID
        if provider_kind == "lightzero_checkpoint"
        else FAKE_PROVIDER_ID
    )
    if metadata.get("provider_id") != expected_provider:
        problems.append("opponent metadata provider_id mismatch")
    if step_batch.info.get("opponent_policy_sidecar") is None:
        problems.append("step info missing opponent_policy_sidecar")
    return problems


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-ref", default=None)
    parser.add_argument("--snapshot-ref", default="wrapper_smoke_snapshot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ego-action-id", type=int, default=0)
    parser.add_argument("--fake-action-id", type=int, default=1)
    parser.add_argument("--num-simulations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--state-key", default=None)
    args = parser.parse_args()
    report = run_lightzero_checkpoint_opponent_wrapper_smoke(
        checkpoint_path=args.checkpoint_path,
        checkpoint_ref=args.checkpoint_ref,
        snapshot_ref=args.snapshot_ref,
        seed=args.seed,
        ego_action_id=args.ego_action_id,
        fake_action_id=args.fake_action_id,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        state_key=args.state_key,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "FakeVisualCheckpointProvider",
    "SNAPSHOT_OPPONENT_WRAPPER_SMOKE_ID",
    "TinyVisualMultiplayerEnv",
    "run_lightzero_checkpoint_opponent_wrapper_smoke",
]
