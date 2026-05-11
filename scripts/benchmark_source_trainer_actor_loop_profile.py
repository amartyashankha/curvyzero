"""Profile a tiny source-stepped, trainer-shaped 1v1 actor loop.

This is a plumbing profile. Source snapshots provide positions, headings, and
alive flags. Occupancy defaults to empty. The center-cell mode is an approximate
raster; the circle-ray mode uses source body circle metadata but is still not
browser-visible trail history or bonus-body geometry.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from datetime import timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.source_env import CurvyTronSourceEnv  # noqa: E402
from curvyzero.env.source_trainer_adapter import config_from_source_snapshot  # noqa: E402
from curvyzero.env.source_trainer_adapter import source_snapshot_player_ids  # noqa: E402
from curvyzero.env.source_trainer_adapter import source_snapshot_to_env_state  # noqa: E402
from curvyzero.env.source_trainer_adapter import source_snapshot_to_vector_trainer_state  # noqa: E402
from curvyzero.env.trainer_contract import ACTION_NAMES  # noqa: E402
from curvyzero.env.trainer_contract import ACTION_SPACE_HASH  # noqa: E402
from curvyzero.env.trainer_contract import ACTION_SPACE_ID  # noqa: E402
from curvyzero.env.trainer_contract import LIGHTZERO_FLAT_OBSERVATION_SHAPE  # noqa: E402
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_HASH  # noqa: E402
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_ID  # noqa: E402
from curvyzero.env.trainer_contract import REWARD_SCHEMA_HASH  # noqa: E402
from curvyzero.env.trainer_contract import REWARD_SCHEMA_ID  # noqa: E402
from curvyzero.env.trainer_observation import observe_1v1_egocentric_rays_v0  # noqa: E402
from curvyzero.env.vector_reset import TERMINAL_REASON_ALL_DEAD_DRAW  # noqa: E402
from curvyzero.env.vector_reset import TERMINAL_REASON_SURVIVOR_WIN  # noqa: E402
from curvyzero.env.vector_trainer_observation import (  # noqa: E402
    observe_vector_1v1_egocentric_rays_batch_arrays_v0,
)
from curvyzero.training.policy_row_mapping import build_policy_row_mapping  # noqa: E402
from curvyzero.training.policy_row_mapping import policy_rows_to_joint_action  # noqa: E402
from curvyzero.training.replay_chunk_v0 import REPLAY_CONTRACT_ID  # noqa: E402
from curvyzero.training.replay_chunk_v0 import build_replay_chunk_v0_metadata  # noqa: E402
from curvyzero.training.replay_chunk_v0 import read_replay_chunk_v0  # noqa: E402
from curvyzero.training.replay_chunk_v0 import write_replay_chunk_v0  # noqa: E402


SCHEMA_ID = "curvyzero_source_trainer_actor_loop_profile/v0"
OPTIMIZER_PROFILE_SCHEMA_ID = "curvyzero_optimizer_profile_report/v0"
PLAYER_COUNT = 2
ACTION_COUNT = len(ACTION_NAMES)
ACTION_ID_TO_SOURCE_MOVE = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)


class TinyNumpyPolicy:
    """Small deterministic policy/value model for timing the policy box."""

    def __init__(self, *, obs_dim: int, hidden_dim: int, action_count: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.04, size=(obs_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.wp = rng.normal(0.0, 0.04, size=(hidden_dim, action_count)).astype(np.float32)
        self.bp = np.zeros(action_count, dtype=np.float32)
        self.wv = rng.normal(0.0, 0.03, size=hidden_dim).astype(np.float32)

    def __call__(
        self,
        observation: np.ndarray,
        legal_action_mask: np.ndarray,
        row_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = np.ones(row_mask.shape[0], dtype=np.int64)
        probs = np.zeros((row_mask.shape[0], ACTION_COUNT), dtype=np.float32)
        values = np.zeros(row_mask.shape[0], dtype=np.float32)
        active = np.asarray(row_mask, dtype=np.bool_)
        if not active.any():
            return selected, probs, values

        hidden = np.tanh(observation[active] @ self.w1 + self.b1)
        logits = hidden @ self.wp + self.bp
        legal = legal_action_mask[active]
        masked_logits = np.where(legal, logits, -1.0e9)
        shifted = masked_logits - masked_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted).astype(np.float32) * legal.astype(np.float32)
        denom = exp_logits.sum(axis=1, keepdims=True)
        active_probs = np.divide(
            exp_logits,
            denom,
            out=np.zeros_like(exp_logits),
            where=denom > 0.0,
        )
        active_selected = active_probs.argmax(axis=1)
        selected[active] = active_selected.astype(np.int64)
        probs[active] = active_probs
        values[active] = np.tanh(hidden @ self.wv).astype(np.float32)
        return selected, probs, values


def run_profile(
    *,
    batch_size: int,
    rollout_steps: int,
    step_ms: float,
    warmup_ms: float,
    seed: int,
    hidden_dim: int,
    occupancy_policy: str,
    artifact_root: Path,
    source_drive_mode: str = "direct_step",
    source_setup_mode: str = "default",
    profile_observation_phases: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if source_drive_mode not in ("direct_step", "timer_frame"):
        raise ValueError("source_drive_mode must be 'direct_step' or 'timer_frame'")
    if source_setup_mode not in ("default", "controlled_trail"):
        raise ValueError("source_setup_mode must be 'default' or 'controlled_trail'")
    profile_started = time.perf_counter()
    artifact_root.mkdir(parents=True, exist_ok=True)

    timers = {
        "reset_autoreset_sec": 0.0,
        "source_adapter_sec": 0.0,
        "pre_observation_packing_sec": 0.0,
        "post_observation_reward_packing_sec": 0.0,
        "observation_packing_sec": 0.0,
        "row_compaction_sec": 0.0,
        "policy_forward_sec": 0.0,
        "action_scatter_sec": 0.0,
        "env_step_sec": 0.0,
        "replay_or_rollout_stage_sec": 0.0,
        "replay_write_or_learner_handoff_sec": 0.0,
        "replay_read_validate_sec": 0.0,
        "search_sec": 0.0,
        "target_construction_sec": 0.0,
        "learner_update_sec": 0.0,
    }
    obs_dim = int(LIGHTZERO_FLAT_OBSERVATION_SHAPE[0])
    policy = TinyNumpyPolicy(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_count=ACTION_COUNT,
        seed=seed + 10_000,
    )

    started = time.perf_counter()
    envs = [
        CurvyTronSourceEnv(
            random_constant=0.5,
            include_deaths_snapshot=True,
            drain_frame_timers=source_drive_mode == "timer_frame",
        )
        for _ in range(batch_size)
    ]
    reset_seed = np.asarray([seed + row for row in range(batch_size)], dtype=np.int64)
    episode_id = np.asarray([f"source-profile-row-{row}-episode-0" for row in range(batch_size)])
    reset_source_value = (
        "CurvyTronSourceEnv(random_constant=0.5,"
        f" source_drive_mode={source_drive_mode},"
        f" source_setup_mode={source_setup_mode})"
    )
    reset_source = np.asarray([reset_source_value for _ in range(batch_size)])
    first_snapshot: dict[str, Any] | None = None
    for row, env in enumerate(envs):
        snapshot = env.reset(player_count=PLAYER_COUNT, warmup_ms=warmup_ms)
        env.advance_timers(warmup_ms)
        if source_setup_mode == "controlled_trail":
            _prepare_controlled_trail_profile_env(env, row=row)
        first_snapshot = env.snapshot("after_warmup")
    timers["reset_autoreset_sec"] += time.perf_counter() - started
    if first_snapshot is None:
        raise RuntimeError("no source env rows were initialized")

    config = config_from_source_snapshot(first_snapshot)
    player_ids = source_snapshot_player_ids(first_snapshot)
    uses_circle_rays = occupancy_policy == "source_world_bodies_circle_rays_v0"

    observation = np.zeros((rollout_steps, batch_size, PLAYER_COUNT, obs_dim), dtype=np.float32)
    legal_action_mask = np.zeros((rollout_steps, batch_size, PLAYER_COUNT, ACTION_COUNT), dtype=np.bool_)
    reward = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    action = np.ones((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.int16)
    action_weights = np.zeros((rollout_steps, batch_size, PLAYER_COUNT, ACTION_COUNT), dtype=np.float32)
    root_value = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    done = np.zeros((rollout_steps, batch_size), dtype=np.bool_)
    terminated = np.zeros((rollout_steps, batch_size), dtype=np.bool_)
    truncated = np.zeros((rollout_steps, batch_size), dtype=np.bool_)
    final_observation = np.zeros((batch_size, PLAYER_COUNT, obs_dim), dtype=np.float32)
    final_reward_map = np.zeros((batch_size, PLAYER_COUNT), dtype=np.float32)
    body_trail_samples: list[dict[str, Any]] = []
    observation_phase_timers: dict[str, float] | None = {} if profile_observation_phases else None

    completed_games = 0
    active_policy_rows = 0
    action_latency_samples: list[float] = []
    loop_started = time.perf_counter()
    for step_index in range(rollout_steps):
        started = time.perf_counter()
        snapshots = [env.snapshot("pre_action") for env in envs]
        states = []
        for row, snapshot in enumerate(snapshots):
            world_bodies = (
                envs[row].world_bodies_snapshot()
                if occupancy_policy != "empty"
                else None
            )
            if uses_circle_rays:
                states.append(
                    source_snapshot_to_vector_trainer_state(
                        snapshot,
                        config,
                        world_bodies=world_bodies or (),
                        avatar_body_metadata=envs[row].avatar_body_metadata_snapshot(),
                        decision_ms=step_ms,
                    )
                )
            else:
                states.append(
                    source_snapshot_to_env_state(
                        snapshot,
                        config,
                        occupancy_policy=occupancy_policy,
                        world_bodies=world_bodies,
                        seed=int(reset_seed[row]),
                    )
                )
            _record_body_trail_sample(
                body_trail_samples,
                world_bodies=world_bodies,
                state=states[-1],
            )
        timers["source_adapter_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        if uses_circle_rays:
            batched_state = _stack_vector_trainer_row_states(states)
            (
                observation[step_index],
                legal_action_mask[step_index],
                _lightzero_action_mask,
                _to_play,
            ) = observe_vector_1v1_egocentric_rays_batch_arrays_v0(
                batched_state,
                player_ids=player_ids,
                decision_ms=step_ms,
                max_ticks=config.max_ticks,
                allow_straight=config.allow_straight_action,
                profile_timers=observation_phase_timers,
            )
        else:
            row_batches = []
            for state in states:
                row_batches.append(
                    observe_1v1_egocentric_rays_v0(
                        state,
                        config,
                        player_ids=player_ids,
                        needs_reset=False,
                        profile_timers=observation_phase_timers,
                    )
                )
            observation[step_index] = np.stack([batch.observation for batch in row_batches])
            legal_action_mask[step_index] = np.stack([batch.action_mask for batch in row_batches])
        live_mask = legal_action_mask[step_index].any(axis=-1)
        elapsed = time.perf_counter() - started
        timers["pre_observation_packing_sec"] += elapsed
        timers["observation_packing_sec"] += elapsed

        action_started = time.perf_counter()
        started = time.perf_counter()
        mapping = build_policy_row_mapping(
            observation[step_index],
            live_mask,
            legal_action_mask[step_index],
            pad_to=batch_size * PLAYER_COUNT,
        )
        active_policy_rows += mapping.active_count
        timers["row_compaction_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        selected, probs, values = policy(
            mapping.observations,
            mapping.legal_action_mask,
            mapping.row_mask,
        )
        timers["policy_forward_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        joint_action = policy_rows_to_joint_action(mapping, selected, dtype=np.int16)
        timers["action_scatter_sec"] += time.perf_counter() - started
        action_latency_samples.append(time.perf_counter() - action_started)

        started = time.perf_counter()
        if mapping.active_count:
            active_rows = np.asarray(mapping.row_mask, dtype=np.bool_)
            env_ids = mapping.env_row_id[active_rows]
            player_rows = mapping.player_id[active_rows]
            action[step_index, env_ids, player_rows] = selected[active_rows].astype(np.int16)
            action_weights[step_index, env_ids, player_rows] = probs[active_rows]
            root_value[step_index, env_ids, player_rows] = values[active_rows]
        timers["replay_or_rollout_stage_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        post_snapshots = []
        for row, env in enumerate(envs):
            source_actions = [
                float(ACTION_ID_TO_SOURCE_MOVE[int(joint_action[row, player])])
                for player in range(PLAYER_COUNT)
            ]
            if source_drive_mode == "timer_frame":
                env.step(source_actions, elapsed_ms=0.0)
                env.advance_timers(step_ms)
                post_snapshots.append(
                    env.snapshot("after_timer_frame_step", advance_ms=step_ms)
                )
            else:
                post_snapshots.append(env.step(source_actions, elapsed_ms=step_ms))
        timers["env_step_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        post_states = []
        for row, snapshot in enumerate(post_snapshots):
            world_bodies = (
                envs[row].world_bodies_snapshot()
                if occupancy_policy != "empty"
                else None
            )
            if uses_circle_rays:
                post_states.append(
                    source_snapshot_to_vector_trainer_state(
                        snapshot,
                        config,
                        world_bodies=world_bodies or (),
                        avatar_body_metadata=envs[row].avatar_body_metadata_snapshot(),
                        decision_ms=step_ms,
                    )
                )
            else:
                post_states.append(
                    source_snapshot_to_env_state(
                        snapshot,
                        config,
                        occupancy_policy=occupancy_policy,
                        world_bodies=world_bodies,
                        seed=int(reset_seed[row]),
                    )
                )
            _record_body_trail_sample(
                body_trail_samples,
                world_bodies=world_bodies,
                state=post_states[-1],
            )
        timers["source_adapter_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        if uses_circle_rays:
            post_batched_state = _stack_vector_trainer_row_states(post_states)
            post_observation, _post_action_mask, _post_lightzero_action_mask, _post_to_play = (
                observe_vector_1v1_egocentric_rays_batch_arrays_v0(
                    post_batched_state,
                    player_ids=player_ids,
                    decision_ms=step_ms,
                    max_ticks=config.max_ticks,
                    allow_straight=config.allow_straight_action,
                    profile_timers=observation_phase_timers,
                )
            )
            post_reward, post_done, post_terminated, post_truncated = (
                _vector_trainer_rewards_and_done(post_batched_state)
            )
            reward[step_index] = post_reward
            done[step_index] = post_done
            terminated[step_index] = post_terminated
            truncated[step_index] = post_truncated
            completed_games += int(np.count_nonzero(post_done))
            if bool(post_done.any()):
                final_observation[post_done] = post_observation[post_done]
                final_reward_map[post_done] = post_reward[post_done]
        else:
            for row, state in enumerate(post_states):
                post = observe_1v1_egocentric_rays_v0(
                    state,
                    config,
                    player_ids=player_ids,
                    needs_reset=False,
                    profile_timers=observation_phase_timers,
                )
                reward[step_index, row] = post.rewards
                done[step_index, row] = post.done
                terminated[step_index, row] = post.terminated
                truncated[step_index, row] = post.truncated
                if post.done:
                    completed_games += 1
                    final_observation[row] = post.observation
                    final_reward_map[row] = post.rewards
        elapsed = time.perf_counter() - started
        timers["post_observation_reward_packing_sec"] += elapsed
        timers["observation_packing_sec"] += elapsed

        started = time.perf_counter()
        for row, row_done in enumerate(done[step_index]):
            if row_done:
                envs[row].reset(player_count=PLAYER_COUNT, warmup_ms=warmup_ms)
                envs[row].advance_timers(warmup_ms)
        timers["reset_autoreset_sec"] += time.perf_counter() - started
    loop_elapsed_sec = time.perf_counter() - loop_started

    arrays = {
        "observation": observation,
        "reward": reward,
        "action": action,
        "action_weights": action_weights,
        "root_value": root_value,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "episode_id": episode_id.astype(str),
        "reset_seed": reset_seed,
        "reset_source": reset_source.astype(str),
        "final_observation": final_observation,
        "final_reward_map": final_reward_map,
    }
    metadata = build_replay_chunk_v0_metadata(
        arrays,
        rules_hash=config.rules_hash,
        observation_schema_hash=OBSERVATION_SCHEMA_HASH,
        action_space_hash=ACTION_SPACE_HASH,
        reward_schema_hash=REWARD_SCHEMA_HASH,
        ruleset_id=config.ruleset,
        observation_schema_id=OBSERVATION_SCHEMA_ID,
        action_space_id=ACTION_SPACE_ID,
        reward_schema_id=REWARD_SCHEMA_ID,
        producer=SCHEMA_ID,
        created_at=_utc_now(),
    )

    replay_path = artifact_root / "replay_v0_chunk.npz"
    started = time.perf_counter()
    write_replay_chunk_v0(replay_path, arrays=arrays, metadata=metadata)
    timers["replay_write_or_learner_handoff_sec"] += time.perf_counter() - started

    started = time.perf_counter()
    read_replay_chunk_v0(replay_path, expected_metadata=metadata)
    timers["replay_read_validate_sec"] += time.perf_counter() - started
    total_elapsed_sec = time.perf_counter() - profile_started

    source_derived_fields = ["positions", "headings", "alive"]
    occupancy_source_fields: list[str] = []
    approximate_fields = ["occupancy"]
    occupancy_caveat = "occupancy is empty, so trail/body ray channels are not source-faithful"
    if occupancy_policy in (
        "source_world_bodies_center_cell_v0",
        "source_world_bodies_circle_rays_v0",
    ):
        occupancy_source_fields = ["world_bodies"]
        if occupancy_policy == "source_world_bodies_circle_rays_v0":
            occupancy_source_fields = [
                "world_bodies",
                "avatar.bodyNum",
                "avatar.trailLatency",
                "avatar.radius",
            ]
            approximate_fields = ["visible_trail_history", "bonus_world_bodies"]
            occupancy_caveat = (
                "body/trail rays use source world body circles with own-body latency; "
                "they are not browser-visible trail history or bonus-body geometry"
            )
        else:
            occupancy_caveat = (
                "occupancy is a center-cell source body raster, not exact source circle geometry"
            )

    report = {
        "schema_id": SCHEMA_ID,
        "optimizer_profile_schema": OPTIMIZER_PROFILE_SCHEMA_ID,
        "run": _run_metadata(seed=seed),
        "lane": "source_trainer_actor_loop_profile",
        "contracts": {
            "environment_impl_id": "CurvyTronSourceEnv",
            "source_drive_mode": source_drive_mode,
            "source_setup_mode": source_setup_mode,
            "source_adapter": (
                "source_snapshot_to_vector_trainer_state("
                f"occupancy_policy={occupancy_policy!r})"
                if uses_circle_rays
                else f"source_snapshot_to_env_state(occupancy_policy={occupancy_policy!r})"
            ),
            "occupancy_policy": occupancy_policy,
            "source_derived_fields": source_derived_fields,
            "occupancy_source_fields": occupancy_source_fields,
            "approximate_fields": approximate_fields,
            "config_derived_observation_fields": [
                "arena size normalization",
                "tick fraction",
                "speed/radius/action constants",
            ],
            "ruleset_id": config.ruleset,
            "rules_hash": config.rules_hash,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
            "action_space_id": ACTION_SPACE_ID,
            "action_space_hash": ACTION_SPACE_HASH,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "reward_schema_hash": REWARD_SCHEMA_HASH,
            "replay_contract_id": REPLAY_CONTRACT_ID,
            "terminal_semantics": "done == terminated OR truncated",
            "action_id_to_source_move": ACTION_ID_TO_SOURCE_MOVE.tolist(),
            "player_ids": list(player_ids),
        },
        "loop_shape": {
            "batch_size": batch_size,
            "player_count": PLAYER_COUNT,
            "rollout_steps": rollout_steps,
            "observation_shape": list(observation.shape),
            "action_shape": list(action.shape),
        },
        "timing_sec": {
            **timers,
            "loop_elapsed_sec": loop_elapsed_sec,
            "wall_elapsed_sec": total_elapsed_sec,
        },
        "latency_sec": {"policy_action": _latency_summary(action_latency_samples)},
        "throughput": {
            "env_transitions_per_sec": (
                batch_size * rollout_steps / loop_elapsed_sec if loop_elapsed_sec > 0 else 0.0
            ),
            "ego_decisions_per_sec": active_policy_rows / loop_elapsed_sec if loop_elapsed_sec > 0 else 0.0,
            "completed_games_per_min": (
                60.0 * completed_games / loop_elapsed_sec if loop_elapsed_sec > 0 else 0.0
            ),
        },
        "denominators": {
            "env_transitions": batch_size * rollout_steps,
            "player_ticks": batch_size * rollout_steps * PLAYER_COUNT,
            "ego_decisions": int(active_policy_rows),
            "policy_rows": batch_size * rollout_steps * PLAYER_COUNT,
            "rollout_rows": batch_size * rollout_steps * PLAYER_COUNT,
            "completed_games": int(completed_games),
            "mcts_roots": 0,
            "mcts_simulations": 0,
            "learner_updates": 0,
        },
        "integrity": {
            "done_terminated_truncated_invariant_failures": int(
                np.count_nonzero(done != np.logical_or(terminated, truncated))
            ),
            "masked_action_violations": _masked_action_violations(action, legal_action_mask),
            "nan_or_inf_count": _nan_or_inf_count(arrays),
            "rows_with_nonzero_final_observation_buffer": int(
                np.count_nonzero(np.any(final_observation != 0.0, axis=(1, 2)))
            ),
            "replay_schema_read_validated": True,
            "replay_semantic_validation": "not performed",
        },
        "replay_or_rollout": {
            "path": str(replay_path),
            "schema_id": metadata["replay_schema_id"],
            "schema_hash": metadata["replay_schema_hash"],
            "field_specs": _field_specs(arrays),
        },
        "policy_search": {
            "policy_kind": "tiny_numpy_mlp_argmax",
            "hidden_dim": hidden_dim,
            "search": "none",
        },
        "source_body_trail": _body_trail_summary(body_trail_samples, player_ids),
        "observation_phase_profile": {
            "enabled": profile_observation_phases,
            "timing_sec": dict(sorted((observation_phase_timers or {}).items())),
            "note": (
                "optional observation helper sub-timers; enabling them adds timing overhead"
            ),
        },
        "artifacts": {
            "artifact_root": str(artifact_root),
            "report_json": str(artifact_root / "profile_report.json"),
            "replay_v0_chunk": str(replay_path),
        },
        "caveats": [
            "source-derived fields are listed in contracts.source_derived_fields",
            occupancy_caveat,
            "tiny NumPy policy is for timing/plumbing only",
            "no learner and no training-quality claim",
            "config-derived scalar fields are not source snapshot fields",
            "replay read validation checks schema/metadata, not source semantics",
            "episode_id/reset metadata are row-level, not per-transition",
            "final_observation/final_reward_map are row-level terminal snapshots for this chunk",
            "own-body latency semantics are not validated against source rendering yet",
        ],
    }
    report_path = artifact_root / "profile_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _stack_vector_trainer_row_states(
    states: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stack 1-row vector trainer states, padding variable body capacity."""

    if not states:
        raise ValueError("states must be non-empty")
    body_capacity = max(int(np.asarray(state["body_active"]).shape[1]) for state in states)
    batched: dict[str, np.ndarray] = {}
    body_pad_values = {
        "body_active": False,
        "body_pos": 0.0,
        "body_radius": 0.0,
        "body_owner": -1,
        "body_num": -1,
    }
    body_keys = set(body_pad_values)
    keys = tuple(states[0].keys())
    for key in keys:
        first = np.asarray(states[0][key])
        if first.shape[0] != 1:
            raise ValueError(f"{key} must be a 1-row state array")
        if key not in body_keys:
            rows = []
            trailing_shape = first.shape[1:]
            for state in states:
                row = np.asarray(state[key])
                if row.shape[0] != 1 or row.shape[1:] != trailing_shape:
                    raise ValueError(f"{key} rows must share trailing shape")
                rows.append(row)
            batched[key] = np.concatenate(rows, axis=0)
            continue

        output_shape = (len(states), body_capacity, *first.shape[2:])
        output = np.full(output_shape, body_pad_values[key], dtype=first.dtype)
        for row_index, state in enumerate(states):
            row = np.asarray(state[key])
            if row.shape[0] != 1 or row.shape[2:] != first.shape[2:]:
                raise ValueError(f"{key} rows must share trailing shape")
            capacity = int(row.shape[1])
            output[row_index, :capacity] = row[0]
        batched[key] = output
    return batched


def _vector_trainer_rewards_and_done(
    state: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    done = np.asarray(state["done"], dtype=np.bool_)
    terminated = np.asarray(state["terminated"], dtype=np.bool_)
    truncated = np.asarray(state["truncated"], dtype=np.bool_)
    if done.shape != terminated.shape or done.shape != truncated.shape:
        raise ValueError("done, terminated, and truncated must share shape")
    if bool(np.any(done != np.logical_or(terminated, truncated))):
        raise ValueError("done must equal terminated or truncated")

    rewards = np.zeros((done.shape[0], PLAYER_COUNT), dtype=np.float32)
    terminal_reason = np.asarray(state["terminal_reason"])
    winner = np.asarray(state["winner"])
    survivor_rows = terminated & (terminal_reason == TERMINAL_REASON_SURVIVOR_WIN)
    draw_rows = terminated & (terminal_reason == TERMINAL_REASON_ALL_DEAD_DRAW)
    unsupported_rows = terminated & ~(survivor_rows | draw_rows)
    if bool(np.any(unsupported_rows)):
        raise ValueError("unsupported terminal_reason for source vector trainer rewards")
    for row in np.flatnonzero(survivor_rows):
        winner_index = int(winner[int(row)])
        if winner_index not in (0, 1):
            raise ValueError("survivor terminal winner must be 0 or 1")
        rewards[int(row)] = -1.0
        rewards[int(row), winner_index] = 1.0
    return rewards, done.copy(), terminated.copy(), truncated.copy()


def _masked_action_violations(action: np.ndarray, legal_action_mask: np.ndarray) -> int:
    expanded = np.take_along_axis(legal_action_mask, action[..., None], axis=-1)
    return int(np.count_nonzero(~expanded))


def _prepare_controlled_trail_profile_env(env: CurvyTronSourceEnv, *, row: int) -> None:
    """Create short nonempty trail/body profiles without running a long rollout."""

    offset = float((row % 4) * 4.0)
    env.set_avatar_state(1, x=10.0, y=20.0 + offset, angle=0.0)
    env.set_avatar_state(2, x=78.0, y=60.0 - offset, angle=math.pi)
    env.advance_timers(env.reference.trail_start_delay_ms)


def _nan_or_inf_count(arrays: dict[str, np.ndarray]) -> int:
    total = 0
    for array in arrays.values():
        if np.issubdtype(array.dtype, np.floating):
            total += int(np.count_nonzero(~np.isfinite(array)))
    return total


def _field_specs(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
            "checksum": hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()[:16]
            if array.dtype.kind != "U"
            else hashlib.sha256(json.dumps(array.tolist(), sort_keys=True).encode("utf-8")).hexdigest()[:16],
        }
        for key, array in arrays.items()
    }


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _record_body_trail_sample(
    samples: list[dict[str, Any]],
    *,
    world_bodies: Sequence[Mapping[str, Any]] | None,
    state: Any,
) -> None:
    if isinstance(state, Mapping):
        occupancy = np.zeros((0,), dtype=np.int16)
        circle_body_count = int(np.count_nonzero(np.asarray(state.get("body_active", []))))
    else:
        occupancy = np.asarray(state.occupancy)
        circle_body_count = 0
    samples.append(
        {
            "world_bodies_count": 0 if world_bodies is None else len(world_bodies),
            "body_circle_count": circle_body_count,
            "occupancy_nonzero_count": int(np.count_nonzero(occupancy)),
            "occupancy_owner_cell_count": [
                int(np.count_nonzero(occupancy == owner))
                for owner in range(1, PLAYER_COUNT + 1)
            ],
        }
    )


def _body_trail_summary(
    samples: list[dict[str, Any]],
    player_ids: Sequence[str],
) -> dict[str, Any]:
    world_counts = [int(sample["world_bodies_count"]) for sample in samples]
    circle_counts = [int(sample.get("body_circle_count", 0)) for sample in samples]
    occupancy_counts = [int(sample["occupancy_nonzero_count"]) for sample in samples]
    if samples:
        owner_counts = np.asarray(
            [sample["occupancy_owner_cell_count"] for sample in samples],
            dtype=np.int64,
        )
    else:
        owner_counts = np.zeros((0, PLAYER_COUNT), dtype=np.int64)
    return {
        "sample_count": len(samples),
        "world_bodies_count": _count_summary(world_counts),
        "body_circle_count": _count_summary(circle_counts),
        "occupancy_nonzero_count": _count_summary(occupancy_counts),
        "occupancy_owner_cell_count": {
            str(player_id): _count_summary(owner_counts[:, index].tolist())
            for index, player_id in enumerate(player_ids)
        },
        "nonempty_world_body_samples": int(sum(count > 0 for count in world_counts)),
        "nonempty_body_circle_samples": int(sum(count > 0 for count in circle_counts)),
        "nonempty_occupancy_samples": int(sum(count > 0 for count in occupancy_counts)),
        "note": (
            "counts summarize source body snapshots and adapted trainer occupancy; "
            "they do not prove exact source trail/body fidelity"
        ),
    }


def _count_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "total": 0}
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "total": int(array.sum()),
    }


def _git_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, check=False, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _run_metadata(*, seed: int) -> dict[str, Any]:
    status = _git_text(["git", "status", "--short"])
    status_lines = status.splitlines() if status else []
    return {
        "run_id": f"source-trainer-profile-{uuid.uuid4().hex[:12]}",
        "created_at_utc": _utc_now(),
        "git_ref": _git_text(["git", "rev-parse", "--short", "HEAD"]) or "unavailable",
        "git_dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
        "command": sys.argv,
        "local_or_modal": "local",
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device_backend": "cpu",
        "debug_event_mode": "no-event",
        "seed": seed,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--rollout-steps", type=_positive_int, default=2)
    parser.add_argument("--step-ms", type=_nonnegative_float, default=1000.0 / 60.0)
    parser.add_argument("--warmup-ms", type=_nonnegative_float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--hidden-dim", type=_positive_int, default=64)
    parser.add_argument(
        "--occupancy-policy",
        choices=(
            "empty",
            "source_world_bodies_center_cell_v0",
            "source_world_bodies_circle_rays_v0",
        ),
        default="empty",
    )
    parser.add_argument(
        "--source-drive-mode",
        choices=("direct_step", "timer_frame"),
        default="direct_step",
        help=(
            "direct_step calls CurvyTronSourceEnv.step(elapsed_ms); timer_frame "
            "sets held actions with elapsed_ms=0 then advances source frame timers"
        ),
    )
    parser.add_argument(
        "--source-setup-mode",
        choices=("default", "controlled_trail"),
        default="default",
        help=(
            "default uses source reset/spawn as-is; controlled_trail force-places "
            "safe live avatars and fires the source trail-start timer before profiling"
        ),
    )
    parser.add_argument(
        "--profile-observation-phases",
        action="store_true",
        help="record optional sub-timers inside trainer observation packing",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/private/tmp/curvy-source-trainer-actor-loop-profile"),
    )
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_profile(
        batch_size=int(args.batch_size),
        rollout_steps=int(args.rollout_steps),
        step_ms=float(args.step_ms),
        warmup_ms=float(args.warmup_ms),
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        occupancy_policy=str(args.occupancy_policy),
        artifact_root=args.artifact_root,
        source_drive_mode=str(args.source_drive_mode),
        source_setup_mode=str(args.source_setup_mode),
        profile_observation_phases=bool(args.profile_observation_phases),
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(
        "source_trainer_actor_loop_profile "
        f"B={report['loop_shape']['batch_size']} T={report['loop_shape']['rollout_steps']} "
        f"env_transitions_per_sec={report['throughput']['env_transitions_per_sec']:.1f} "
        f"ego_decisions_per_sec={report['throughput']['ego_decisions_per_sec']:.1f} "
        f"artifact_root={report['artifacts']['artifact_root']}"
    )


if __name__ == "__main__":
    main()
