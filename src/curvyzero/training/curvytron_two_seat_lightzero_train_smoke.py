"""Bounded two-seat CurvyTron smoke using one live LightZero policy.

This is not ``train_muzero`` and not full distributed current-policy self-play
training. It proves the next small local boundary: one installed LightZero
``MuZeroPolicy`` object chooses actions for both CurvyTron seats, replay is
built from those rows, learner updates mutate that same object, and later
collection can use the refreshed policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping

import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_GRAY64_NORMALIZED_DTYPE,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_DEFAULT
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_ORDER
from curvyzero.env.trainer_contract import stable_contract_hash
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    ACTION_COUNT,
    NOOP_ACTION_ID,
    PLAYER_PERSPECTIVE_SCHEMA_ID,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
    SourceStateGray64Stack4,
    player_perspective_value_map,
    source_state_gray64_stack4_render_metadata,
    validate_stack_trail_render_mode,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_profile import (
    _action_weights,
    _build_lightzero_policy,
    _compact_mcts_output,
    _exception_result,
    _extract_eval_action,
    _learn_mode_batches,
    _learn_mode_forward_loss,
    _loss_summary,
    _model_hash,
    _policy_eval_action,
    _policy_model_device,
    _root_value,
    _strip_large_arrays,
    _strip_runtime_object,
    _temporary_policy_batch_size,
    _to_plain,
)
from curvyzero.training.policy_row_mapping import build_policy_row_mapping
from curvyzero.training.policy_row_mapping import policy_rows_to_joint_action


TWO_SEAT_LIGHTZERO_TRAIN_SMOKE_SCHEMA_ID = (
    "curvyzero_two_seat_lightzero_train_smoke/v0"
)
TWO_SEAT_LIGHTZERO_REPLAY_ROW_SCHEMA_ID = (
    "curvyzero_two_seat_lightzero_replay_row/v0"
)
TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_ID = (
    "terminal_winner_keeps_survival_loser_zero/v0"
)
TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_HASH = stable_contract_hash(
    {
        "schema_id": TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_ID,
        "base_step_reward": "alive_reward while alive after step else dead_reward",
        "terminal_survivor_win_return": "accumulated shaped survival return",
        "terminal_loser_return": 0.0,
        "draw_or_truncation_return": "unmodified shaped survival return",
        "scope": "per_player_two_seat_return_targets",
    }
)
LEARN_BATCH_BLOCKER = (
    "If MuZeroPolicy.learn_mode.forward fails, inspect the exact installed "
    "LightZero learner contract; the local adapter now preserves two-seat "
    "metadata and builds discounted survival value targets for those rows."
)
REPLAY_SCOPE_CURRENT_ITERATION = "current_iteration"
REPLAY_SCOPE_ACCUMULATED = "accumulated"
REPLAY_SCOPE_CHOICES = (REPLAY_SCOPE_CURRENT_ITERATION, REPLAY_SCOPE_ACCUMULATED)
DEFAULT_ALIVE_REWARD = 1.0
DEFAULT_DEAD_REWARD = 0.0
DEFAULT_ENV_MAX_TICKS = 2_000
DEFAULT_DEATH_MODE = vector_runtime.DEATH_MODE_NORMAL
DEFAULT_CHECKPOINT_EVERY_ITERATIONS = 100
DEFAULT_PROGRESS_EVERY_ITERATIONS = 100
DEFAULT_PROGRESS_COMMIT_EVERY_ITERATIONS = 100
DEFAULT_ACTION_NOOP_PROBABILITY = 0.0
DEFAULT_ACTION_NOOP_WARMUP_ITERATIONS = 0
DEFAULT_POLICY_ACTION_REPEAT_MIN = 1
DEFAULT_POLICY_ACTION_REPEAT_MAX = 3
DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY = 0.20
DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS = 0
DEFAULT_OBSERVATION_NOISE_STD = 0.10
POLICY_ACTION_REPEAT_RNG_SALT = 0xC0A7A11
CONTROL_STOCHASTICITY_SCHEMA_ID = "curvyzero_two_seat_policy_action_repeat/v0"
ACTION_SELECTION_MODE_COLLECT = "collect"
ACTION_SELECTION_MODE_EVAL = "eval"
ACTION_SELECTION_MODE_CHOICES = (
    ACTION_SELECTION_MODE_COLLECT,
    ACTION_SELECTION_MODE_EVAL,
)


def run_curvytron_two_seat_lightzero_train_smoke(
    *,
    seed: int = 0,
    batch_size: int = 1,
    steps: int = 4,
    outer_iterations: int = 1,
    collect_steps_per_iteration: int | None = None,
    updates_per_iteration: int | None = None,
    num_simulations: int = 2,
    learner_updates: int = 1,
    allow_optimizer_step: bool = False,
    checkpoint_dir: str | Path | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    checkpoint_every_iterations: int = DEFAULT_CHECKPOINT_EVERY_ITERATIONS,
    save_initial_checkpoint: bool = False,
    progress_path: str | Path | None = None,
    progress_every_iterations: int = DEFAULT_PROGRESS_EVERY_ITERATIONS,
    progress_commit_every_iterations: int = DEFAULT_PROGRESS_COMMIT_EVERY_ITERATIONS,
    progress_commit_callback: Callable[[], None] | None = None,
    progress_print: bool = True,
    max_ticks: int | None = None,
    death_mode: str = DEFAULT_DEATH_MODE,
    decision_ms: float = 300.0,
    replay_scope: str = REPLAY_SCOPE_CURRENT_ITERATION,
    learner_sample_size: int | None = None,
    max_replay_rows: int | None = 4096,
    record_log_limit: int = 512,
    replay_row_log_limit: int = 256,
    alive_reward: float = DEFAULT_ALIVE_REWARD,
    dead_reward: float = DEFAULT_DEAD_REWARD,
    action_selection_mode: str = ACTION_SELECTION_MODE_COLLECT,
    collect_temperature: float = 1.0,
    collect_epsilon: float = 0.25,
    action_noop_probability: float = DEFAULT_ACTION_NOOP_PROBABILITY,
    action_noop_warmup_iterations: int = DEFAULT_ACTION_NOOP_WARMUP_ITERATIONS,
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN,
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX,
    policy_action_repeat_extra_probability: float = (
        DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    ),
    policy_action_repeat_warmup_iterations: int = (
        DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS
    ),
    observation_noise_std: float = DEFAULT_OBSERVATION_NOISE_STD,
    trail_render_mode: str = TRAIL_RENDER_MODE_DEFAULT,
    use_cuda: bool = False,
    require_installed_lightzero: bool = True,
) -> dict[str, Any]:
    """Run a tiny iterative collect/replay/learn-forward smoke."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if outer_iterations < 1:
        raise ValueError("outer_iterations must be >= 1")
    resolved_collect_steps = (
        int(steps)
        if collect_steps_per_iteration is None
        else int(collect_steps_per_iteration)
    )
    if resolved_collect_steps < 1:
        raise ValueError("collect_steps_per_iteration must be >= 1")
    resolved_updates_per_iteration = (
        int(learner_updates)
        if updates_per_iteration is None
        else int(updates_per_iteration)
    )
    if resolved_updates_per_iteration < 0:
        raise ValueError("updates_per_iteration must be >= 0")
    if num_simulations < 1:
        raise ValueError("num_simulations must be >= 1")
    if learner_updates < 0:
        raise ValueError("learner_updates must be >= 0")
    if replay_scope not in REPLAY_SCOPE_CHOICES:
        choices = ", ".join(REPLAY_SCOPE_CHOICES)
        raise ValueError(f"replay_scope must be one of: {choices}")
    resolved_learner_sample_size = (
        None if learner_sample_size is None else int(learner_sample_size)
    )
    if (
        resolved_learner_sample_size is not None
        and resolved_learner_sample_size < 1
    ):
        raise ValueError("learner_sample_size must be >= 1 when set")
    resolved_max_replay_rows = None if max_replay_rows is None else int(max_replay_rows)
    if resolved_max_replay_rows is not None and resolved_max_replay_rows < 1:
        raise ValueError("max_replay_rows must be >= 1 when set")
    if int(record_log_limit) < 0:
        raise ValueError("record_log_limit must be >= 0")
    if int(replay_row_log_limit) < 0:
        raise ValueError("replay_row_log_limit must be >= 0")
    resolved_max_ticks = DEFAULT_ENV_MAX_TICKS if max_ticks is None else int(max_ticks)
    if resolved_max_ticks < 0:
        raise ValueError("max_ticks must be >= 0")
    resolved_death_mode = str(death_mode)
    if resolved_death_mode not in vector_runtime.DEATH_MODES:
        raise ValueError(
            f"death_mode must be one of {tuple(vector_runtime.DEATH_MODES)!r}; "
            f"got {death_mode!r}"
        )
    resolved_action_noop_probability = float(action_noop_probability)
    if not 0.0 <= resolved_action_noop_probability <= 1.0:
        raise ValueError("action_noop_probability must be in [0, 1]")
    resolved_action_noop_warmup_iterations = int(action_noop_warmup_iterations)
    if resolved_action_noop_warmup_iterations < 0:
        raise ValueError("action_noop_warmup_iterations must be >= 0")
    resolved_policy_action_repeat_min = int(policy_action_repeat_min)
    resolved_policy_action_repeat_max = int(policy_action_repeat_max)
    resolved_policy_action_repeat_extra_probability = float(
        policy_action_repeat_extra_probability
    )
    resolved_policy_action_repeat_warmup_iterations = int(
        policy_action_repeat_warmup_iterations
    )
    _validate_policy_action_repeat_config(
        min_repeat=resolved_policy_action_repeat_min,
        max_repeat=resolved_policy_action_repeat_max,
        extra_probability=resolved_policy_action_repeat_extra_probability,
        warmup_iterations=resolved_policy_action_repeat_warmup_iterations,
    )
    resolved_observation_noise_std = float(observation_noise_std)
    if resolved_observation_noise_std < 0.0:
        raise ValueError("observation_noise_std must be >= 0")
    resolved_trail_render_mode = validate_stack_trail_render_mode(trail_render_mode)
    resolved_alive_reward = float(alive_reward)
    resolved_dead_reward = float(dead_reward)
    if action_selection_mode not in ACTION_SELECTION_MODE_CHOICES:
        choices = ", ".join(ACTION_SELECTION_MODE_CHOICES)
        raise ValueError(f"action_selection_mode must be one of: {choices}")
    resolved_collect_temperature = float(collect_temperature)
    if resolved_collect_temperature <= 0.0:
        raise ValueError("collect_temperature must be > 0")
    resolved_collect_epsilon = float(collect_epsilon)
    if not 0.0 <= resolved_collect_epsilon <= 1.0:
        raise ValueError("collect_epsilon must be in [0, 1]")
    resolved_checkpoint_every_iterations = int(checkpoint_every_iterations)
    if resolved_checkpoint_every_iterations < 1:
        raise ValueError("checkpoint_every_iterations must be >= 1")
    resolved_progress_path = None if progress_path is None else Path(progress_path)
    resolved_progress_every_iterations = int(progress_every_iterations)
    if resolved_progress_every_iterations < 1:
        raise ValueError("progress_every_iterations must be >= 1")
    resolved_progress_commit_every_iterations = int(progress_commit_every_iterations)
    if resolved_progress_commit_every_iterations < 1:
        raise ValueError("progress_commit_every_iterations must be >= 1")
    run_started = time.perf_counter()

    problems: list[str] = []
    checkpoint_records: list[dict[str, Any]] = []
    policy_context = _build_lightzero_policy(
        seed=seed,
        num_simulations=num_simulations,
        require_installed_lightzero=require_installed_lightzero,
        use_cuda=use_cuda,
    )
    policy = policy_context.get("policy")
    if policy is None:
        problems.append(
            "installed LightZero policy setup did not complete; collect loop was not run"
        )
        return _to_plain(
            _result_payload(
                ok=False,
                seed=seed,
                batch_size=batch_size,
                steps=steps,
                outer_iterations=outer_iterations,
                collect_steps_per_iteration=resolved_collect_steps,
                updates_per_iteration=resolved_updates_per_iteration,
                num_simulations=num_simulations,
                learner_updates=learner_updates,
                allow_optimizer_step=allow_optimizer_step,
                replay_scope=replay_scope,
                learner_sample_size=resolved_learner_sample_size,
                max_replay_rows=resolved_max_replay_rows,
                record_log_limit=int(record_log_limit),
                replay_row_log_limit=int(replay_row_log_limit),
                alive_reward=resolved_alive_reward,
                dead_reward=resolved_dead_reward,
                action_selection_mode=action_selection_mode,
                collect_temperature=resolved_collect_temperature,
                collect_epsilon=resolved_collect_epsilon,
                action_noop_probability=resolved_action_noop_probability,
                action_noop_warmup_iterations=(
                    resolved_action_noop_warmup_iterations
                ),
                policy_action_repeat_min=resolved_policy_action_repeat_min,
                policy_action_repeat_max=resolved_policy_action_repeat_max,
                policy_action_repeat_extra_probability=(
                    resolved_policy_action_repeat_extra_probability
                ),
                policy_action_repeat_warmup_iterations=(
                    resolved_policy_action_repeat_warmup_iterations
                ),
                observation_noise_std=resolved_observation_noise_std,
                trail_render_mode=resolved_trail_render_mode,
                use_cuda=use_cuda,
                elapsed_sec=time.perf_counter() - run_started,
                checkpoint_every_iterations=resolved_checkpoint_every_iterations,
                save_initial_checkpoint=save_initial_checkpoint,
                env_max_ticks=resolved_max_ticks,
                death_mode=resolved_death_mode,
                policy_context=policy_context,
                problems=problems,
                records=[],
                replay_rows=[],
                total_steps_collected=0,
                total_replay_rows_collected=0,
                iteration_summaries=[],
                learner_forwards=[],
                learner_forward={
                    "status": "blocked",
                    "reason": "LightZero policy unavailable",
                    "blocker": LEARN_BATCH_BLOCKER,
                },
                final_observation_shape=[batch_size, 2, *STACKED_SOURCE_STATE_GRAY64_SHAPE],
                action_counts=Counter(),
                per_player_action_counts={"player_0": Counter(), "player_1": Counter()},
                checkpoint_records=checkpoint_records,
            )
        )

    resolved_checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    if allow_optimizer_step and resolved_checkpoint_dir is None:
        resolved_checkpoint_dir = Path(
            "artifacts/local/curvytron-two-seat-lightzero-real-train-smoke/"
            "checkpoints/lightzero"
        )
    if resolved_checkpoint_dir is not None and bool(save_initial_checkpoint):
        checkpoint_records.append(
            _save_lightzero_policy_checkpoint(
                policy,
                resolved_checkpoint_dir,
                iteration=0,
                metadata={
                    "kind": "initial",
                    "seed": int(seed),
                    "batch_size": int(batch_size),
                    "steps": int(steps),
                    "outer_iterations": int(outer_iterations),
                    "collect_steps_per_iteration": int(resolved_collect_steps),
                    "updates_per_iteration": int(resolved_updates_per_iteration),
                    "num_simulations": int(num_simulations),
                    "allow_optimizer_step": bool(allow_optimizer_step),
                    "replay_scope": replay_scope,
                    "learner_sample_size": resolved_learner_sample_size,
                    "action_selection_mode": action_selection_mode,
                    "collect_temperature": float(resolved_collect_temperature),
                    "collect_epsilon": float(resolved_collect_epsilon),
                    "death_mode": resolved_death_mode,
                    "trail_render_mode": resolved_trail_render_mode,
                    "policy_action_repeat_min": int(resolved_policy_action_repeat_min),
                    "policy_action_repeat_max": int(resolved_policy_action_repeat_max),
                    "policy_action_repeat_extra_probability": float(
                        resolved_policy_action_repeat_extra_probability
                    ),
                    "policy_action_repeat_warmup_iterations": int(
                        resolved_policy_action_repeat_warmup_iterations
                    ),
                    **dict(checkpoint_metadata or {}),
                },
            )
        )
    if resolved_progress_path is not None:
        _append_progress_line(
            resolved_progress_path,
            {
                "event": "start",
                "timestamp": _utc_timestamp(),
                "seed": int(seed),
                "batch_size": int(batch_size),
                "outer_iterations": int(outer_iterations),
                "collect_steps_per_iteration": int(resolved_collect_steps),
                "updates_per_iteration": int(resolved_updates_per_iteration),
                "checkpoint_every_iterations": int(
                    resolved_checkpoint_every_iterations
                ),
                "save_initial_checkpoint": bool(save_initial_checkpoint),
                "progress_every_iterations": int(resolved_progress_every_iterations),
                "requested_max_ticks": None if max_ticks is None else int(max_ticks),
                "env_max_ticks": int(resolved_max_ticks),
                "death_mode": resolved_death_mode,
                "action_selection_mode": action_selection_mode,
                "collect_temperature": float(resolved_collect_temperature),
                "collect_epsilon": float(resolved_collect_epsilon),
                "trail_render_mode": resolved_trail_render_mode,
            },
            print_line=progress_print,
        )
        if progress_commit_callback is not None:
            progress_commit_callback()

    env = VectorMultiplayerEnv(
        batch_size=batch_size,
        player_count=2,
        seed=seed,
        decision_ms=decision_ms,
        max_ticks=resolved_max_ticks,
        death_mode=resolved_death_mode,
    )
    visual_stack = SourceStateGray64Stack4(
        batch_size=batch_size,
        player_count=2,
        trail_render_mode=resolved_trail_render_mode,
    )

    # The run seed initializes the RNG, but each row reset gets its own generated
    # reset seed. This keeps training starts varied while remaining reproducible.
    batch = env.reset(seed=None)
    observation = visual_stack.update(env)
    problems.extend(_validate_visual_batch(observation, batch.action_mask, label="reset"))

    records: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    replay_rows_for_output: list[dict[str, Any]] = []
    total_steps_collected = 0
    total_replay_rows_collected = 0
    action_counts: Counter[int] = Counter()
    per_player_action_counts: dict[str, Counter[int]] = {
        "player_0": Counter(),
        "player_1": Counter(),
    }

    learner_forwards: list[dict[str, Any]] = []
    iteration_summaries: list[dict[str, Any]] = []
    learner_forward: dict[str, Any] = {
        "status": "skipped",
        "reason": "no outer iteration ran",
        "blocker": LEARN_BATCH_BLOCKER,
    }
    final_learner_replay_rows: list[dict[str, Any]] = []
    decision_offset = 0
    cumulative_update_index = 0

    for outer_index in range(outer_iterations):
        iteration_number = outer_index + 1
        effective_action_noop_probability = _scheduled_action_noop_probability(
            final_probability=resolved_action_noop_probability,
            warmup_iterations=resolved_action_noop_warmup_iterations,
            iteration=iteration_number,
        )
        effective_policy_action_repeat_extra_probability = (
            _scheduled_policy_action_repeat_extra_probability(
                final_probability=resolved_policy_action_repeat_extra_probability,
                warmup_iterations=resolved_policy_action_repeat_warmup_iterations,
                iteration=iteration_number,
            )
        )
        collection = _collect_current_policy_iteration(
            policy,
            env,
            visual_stack,
            batch=batch,
            observation=observation,
            iteration=iteration_number,
            decision_offset=decision_offset,
            collect_steps=resolved_collect_steps,
            alive_reward=resolved_alive_reward,
            dead_reward=resolved_dead_reward,
            action_selection_mode=action_selection_mode,
            collect_temperature=resolved_collect_temperature,
            collect_epsilon=resolved_collect_epsilon,
            action_noop_probability=effective_action_noop_probability,
            action_noise_rng=_action_noise_rng(
                seed=seed,
                iteration=iteration_number,
            ),
            policy_action_repeat_min=resolved_policy_action_repeat_min,
            policy_action_repeat_max=resolved_policy_action_repeat_max,
            policy_action_repeat_extra_probability=(
                effective_policy_action_repeat_extra_probability
            ),
            policy_action_repeat_rng=_policy_action_repeat_rng(
                seed=seed,
                iteration=iteration_number,
            ),
            observation_noise_std=resolved_observation_noise_std,
            observation_noise_rng=_observation_noise_rng(
                seed=seed,
                iteration=iteration_number,
            ),
        )
        batch = collection["batch"]
        observation = collection["observation"]
        decision_offset += int(collection["steps_collected"])
        problems.extend(collection["problems"])

        iteration_records = collection["records"]
        iteration_replay_rows = collection["replay_rows"]
        records.extend(iteration_records)
        total_steps_collected += int(collection["steps_collected"])
        total_replay_rows_collected += int(len(iteration_replay_rows))
        if len(records) > int(record_log_limit):
            del records[int(record_log_limit) :]
        replay_rows_for_output.extend(iteration_replay_rows)
        if len(replay_rows_for_output) > int(replay_row_log_limit):
            del replay_rows_for_output[int(replay_row_log_limit) :]
        if replay_scope == REPLAY_SCOPE_ACCUMULATED:
            replay_rows.extend(iteration_replay_rows)
            if (
                resolved_max_replay_rows is not None
                and len(replay_rows) > resolved_max_replay_rows
            ):
                del replay_rows[: len(replay_rows) - resolved_max_replay_rows]
        action_counts.update(collection["action_counts"])
        for player, counts in collection["per_player_action_counts"].items():
            per_player_action_counts[player].update(counts)

        learner_replay_rows = (
            replay_rows
            if replay_scope == REPLAY_SCOPE_ACCUMULATED
            else iteration_replay_rows
        )
        final_learner_replay_rows = learner_replay_rows
        replay_rows_available = len(learner_replay_rows)
        learner_batch_size = _resolved_learner_batch_size(
            replay_rows_available,
            learner_sample_size=resolved_learner_sample_size,
        )
        iteration_updates: list[dict[str, Any]] = []
        if resolved_updates_per_iteration < 1:
            learner_forward = {
                "status": "skipped",
                "reason": "updates_per_iteration is 0",
                "blocker": LEARN_BATCH_BLOCKER,
                "iteration": int(iteration_number),
                "replay_scope": replay_scope,
                "replay_rows_available": int(replay_rows_available),
                "learner_batch_size": int(learner_batch_size),
            }
        elif not learner_replay_rows:
            learner_forward = {
                "status": "blocked",
                "reason": "no replay rows available",
                "blocker": LEARN_BATCH_BLOCKER,
                "iteration": int(iteration_number),
                "replay_scope": replay_scope,
                "replay_rows_available": int(replay_rows_available),
                "learner_batch_size": int(learner_batch_size),
            }
        else:
            for iteration_update_index in range(resolved_updates_per_iteration):
                sample = _sample_replay_batch(
                    learner_replay_rows,
                    replay_scope=replay_scope,
                    learner_sample_size=resolved_learner_sample_size,
                    rng=_replay_sample_rng(
                        seed=seed,
                        iteration=iteration_number,
                        update_index=cumulative_update_index,
                    ),
                )
                if allow_optimizer_step:
                    update = _learn_mode_forward_update(
                        policy,
                        sample,
                        update_index=cumulative_update_index,
                    )
                else:
                    update = _learn_mode_forward_loss(policy, sample)
                    update["update_index"] = int(cumulative_update_index)
                update["iteration"] = int(iteration_number)
                update["iteration_update_index"] = int(iteration_update_index)
                update["replay_scope"] = replay_scope
                update["replay_rows_available"] = int(sample["replay_rows_available"])
                update["learner_batch_size"] = int(sample["learner_batch_size"])
                if isinstance(update.get("sample"), dict):
                    update["sample"].update(_sample_replay_metadata(sample))
                cumulative_update_index += 1
                learner_forwards.append(update)
                iteration_updates.append(update)
                learner_forward = update
                if not update.get("ok"):
                    update["blocker"] = LEARN_BATCH_BLOCKER
                    problems.append("LightZero learn_mode.forward failed")
                    break
                if allow_optimizer_step and not bool(update.get("model_parameters_changed")):
                    problems.append(
                        "real optimizer step ran but model parameters did not change"
                    )
                    break

        iteration_checkpoint: dict[str, Any] | None = None
        if (
            allow_optimizer_step
            and resolved_checkpoint_dir is not None
            and iteration_updates
            and bool(iteration_updates[-1].get("ok"))
            and (
                iteration_number % resolved_checkpoint_every_iterations == 0
                or iteration_number == outer_iterations
            )
        ):
            iteration_checkpoint = _save_lightzero_policy_checkpoint(
                policy,
                resolved_checkpoint_dir,
                iteration=iteration_number,
                metadata={
                    "kind": "after_outer_iteration",
                    "iteration": int(iteration_number),
                    "seed": int(seed),
                    "batch_size": int(batch_size),
                    "steps": int(steps),
                    "outer_iterations": int(outer_iterations),
                    "collect_steps_per_iteration": int(resolved_collect_steps),
                    "updates_per_iteration": int(resolved_updates_per_iteration),
                    "num_simulations": int(num_simulations),
                    "learner_updates": int(learner_updates),
                    "allow_optimizer_step": True,
                    "replay_scope": replay_scope,
                    "learner_sample_size": resolved_learner_sample_size,
                    "alive_reward": resolved_alive_reward,
                    "dead_reward": resolved_dead_reward,
                    "action_selection_mode": action_selection_mode,
                    "collect_temperature": float(resolved_collect_temperature),
                    "collect_epsilon": float(resolved_collect_epsilon),
                    "action_noop_probability": resolved_action_noop_probability,
                    "effective_action_noop_probability": float(
                        effective_action_noop_probability
                    ),
                    "action_noop_warmup_iterations": int(
                        resolved_action_noop_warmup_iterations
                    ),
                    "policy_action_repeat_min": int(resolved_policy_action_repeat_min),
                    "policy_action_repeat_max": int(resolved_policy_action_repeat_max),
                    "policy_action_repeat_extra_probability": float(
                        resolved_policy_action_repeat_extra_probability
                    ),
                    "effective_policy_action_repeat_extra_probability": float(
                        effective_policy_action_repeat_extra_probability
                    ),
                    "policy_action_repeat_warmup_iterations": int(
                        resolved_policy_action_repeat_warmup_iterations
                    ),
                    "observation_noise_std": float(resolved_observation_noise_std),
                    "checkpoint_every_iterations": int(
                        resolved_checkpoint_every_iterations
                    ),
                    "learner_forwards": [
                        _strip_large_arrays(update) for update in iteration_updates
                    ],
                    **dict(checkpoint_metadata or {}),
                },
            )
            checkpoint_records.append(iteration_checkpoint)

        iteration_summaries.append(
            _iteration_summary(
                iteration=iteration_number,
                collect_steps_requested=resolved_collect_steps,
                records=iteration_records,
                replay_rows=iteration_replay_rows,
                action_counts=collection["action_counts"],
                per_player_action_counts=collection["per_player_action_counts"],
                learner_forwards=iteration_updates
                if iteration_updates
                else [learner_forward],
                checkpoint=iteration_checkpoint,
                replay_scope=replay_scope,
                learner_replay_rows=learner_replay_rows,
                learner_sample_size=resolved_learner_sample_size,
                replay_rows_available=replay_rows_available,
                learner_batch_size=learner_batch_size,
            )
        )
        iteration_summaries[-1]["effective_action_noop_probability"] = float(
            effective_action_noop_probability
        )
        iteration_summaries[-1]["effective_policy_action_repeat_extra_probability"] = float(
            effective_policy_action_repeat_extra_probability
        )
        iteration_summaries[-1]["control_stochasticity"] = collection[
            "control_stochasticity"
        ]
        iteration_summaries[-1]["collect_timing_sec"] = collection["timing_sec"]
        iteration_summaries[-1]["policy_batching_counts"] = collection[
            "policy_batching_counts"
        ]
        iteration_summaries[-1]["policy_search_call_count"] = int(
            collection["policy_search_call_count"]
        )
        iteration_summaries[-1]["policy_search_row_count"] = int(
            collection["policy_search_row_count"]
        )
        if (
            resolved_progress_path is not None
            and (
                iteration_number % resolved_progress_every_iterations == 0
                or iteration_checkpoint is not None
                or bool(problems)
            )
        ):
            _append_progress_line(
                resolved_progress_path,
                _iteration_progress_line(
                    iteration=iteration_number,
                    iteration_summary=iteration_summaries[-1],
                    total_steps_collected=total_steps_collected,
                    total_replay_rows_collected=total_replay_rows_collected,
                    elapsed_sec=time.perf_counter() - run_started,
                    checkpoint=iteration_checkpoint,
                    problems=problems,
                ),
                print_line=progress_print,
            )
            if (
                progress_commit_callback is not None
                and (
                    iteration_number % resolved_progress_commit_every_iterations == 0
                    or iteration_checkpoint is not None
                    or bool(problems)
                )
            ):
                progress_commit_callback()
        if problems:
            break

    if not learner_forwards:
        learner_forwards = [learner_forward]

    return _to_plain(
        _result_payload(
            ok=not problems,
            seed=seed,
            batch_size=batch_size,
            steps=steps,
            outer_iterations=outer_iterations,
            collect_steps_per_iteration=resolved_collect_steps,
            updates_per_iteration=resolved_updates_per_iteration,
            num_simulations=num_simulations,
            learner_updates=learner_updates,
            allow_optimizer_step=allow_optimizer_step,
            replay_scope=replay_scope,
            learner_sample_size=resolved_learner_sample_size,
            max_replay_rows=resolved_max_replay_rows,
            record_log_limit=int(record_log_limit),
            replay_row_log_limit=int(replay_row_log_limit),
            alive_reward=resolved_alive_reward,
            dead_reward=resolved_dead_reward,
            action_selection_mode=action_selection_mode,
            collect_temperature=resolved_collect_temperature,
            collect_epsilon=resolved_collect_epsilon,
            action_noop_probability=resolved_action_noop_probability,
            action_noop_warmup_iterations=(
                resolved_action_noop_warmup_iterations
            ),
            policy_action_repeat_min=resolved_policy_action_repeat_min,
            policy_action_repeat_max=resolved_policy_action_repeat_max,
            policy_action_repeat_extra_probability=(
                resolved_policy_action_repeat_extra_probability
            ),
            policy_action_repeat_warmup_iterations=(
                resolved_policy_action_repeat_warmup_iterations
            ),
            observation_noise_std=resolved_observation_noise_std,
            trail_render_mode=resolved_trail_render_mode,
            use_cuda=use_cuda,
            elapsed_sec=time.perf_counter() - run_started,
            checkpoint_every_iterations=resolved_checkpoint_every_iterations,
            save_initial_checkpoint=save_initial_checkpoint,
            env_max_ticks=resolved_max_ticks,
            death_mode=resolved_death_mode,
            policy_context=policy_context,
            problems=problems,
            records=records,
            replay_rows=replay_rows_for_output,
            learner_replay_rows=final_learner_replay_rows,
            total_steps_collected=total_steps_collected,
            total_replay_rows_collected=total_replay_rows_collected,
            iteration_summaries=iteration_summaries,
            learner_forwards=learner_forwards,
            learner_forward=learner_forward,
            final_observation_shape=list(observation.shape),
            action_counts=action_counts,
            per_player_action_counts=per_player_action_counts,
            checkpoint_records=checkpoint_records,
        )
    )


def compact_curvytron_two_seat_lightzero_train_smoke_summary(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the small handoff-friendly surface."""

    replay = result.get("replay", {})
    learner = result.get("learner_forward", {})
    policy = result.get("lightzero_policy", {})
    inputs = result.get("inputs", {})
    return _to_plain(
        {
            "ok": result.get("ok"),
            "schema_id": result.get("schema_id"),
            "mode": result.get("mode"),
            "problems": result.get("problems", []),
            "inputs": {
                "steps_requested": inputs.get("steps_requested"),
                "outer_iterations": inputs.get("outer_iterations"),
                "collect_steps_per_iteration": inputs.get(
                    "collect_steps_per_iteration"
                ),
                "updates_per_iteration": inputs.get("updates_per_iteration"),
                "learner_updates": inputs.get("learner_updates"),
                "env_max_ticks": inputs.get("env_max_ticks"),
                "replay_scope": inputs.get("replay_scope"),
                "learner_sample_size": inputs.get("learner_sample_size"),
                "death_mode": inputs.get("death_mode"),
                "death_suppression_for_profile": inputs.get(
                    "death_suppression_for_profile"
                ),
                "action_noop_probability": inputs.get("action_noop_probability"),
                "action_noop_warmup_iterations": inputs.get(
                    "action_noop_warmup_iterations"
                ),
                "policy_action_repeat_min": inputs.get("policy_action_repeat_min"),
                "policy_action_repeat_max": inputs.get("policy_action_repeat_max"),
                "policy_action_repeat_extra_probability": inputs.get(
                    "policy_action_repeat_extra_probability"
                ),
                "policy_action_repeat_warmup_iterations": inputs.get(
                    "policy_action_repeat_warmup_iterations"
                ),
                "trail_render_mode": inputs.get("trail_render_mode"),
                "use_cuda": inputs.get("use_cuda"),
                "checkpoint_every_iterations": inputs.get(
                    "checkpoint_every_iterations"
                ),
            },
            "lightzero_policy_status": policy.get("status"),
            "lightzero_policy_requested_cuda": policy.get("requested_cuda"),
            "lightzero_policy_model_device": policy.get("model_device"),
            "lightzero_policy_surface": policy.get("surface"),
            "elapsed_sec": result.get("elapsed_sec"),
            "steps_survived": result.get("steps_survived"),
            "collect_timing_summary": result.get("collect_timing_summary"),
            "episode_duration_summary": result.get("episode_duration_summary"),
            "surface": {
                "render": result.get("surface", {}).get("render"),
                "stack_schema_id": result.get("surface", {}).get("stack_schema_id"),
                "player_perspective_schema_id": result.get("surface", {}).get(
                    "player_perspective_schema_id"
                ),
            },
            "iteration_count": len(result.get("iterations", []))
            if isinstance(result.get("iterations"), list)
            else None,
            "iteration_edges": _iteration_edges(result.get("iterations", [])),
            "replay": {
                "status": replay.get("status"),
                "row_count": replay.get("row_count"),
                "scope": replay.get("scope"),
                "replay_rows_available": replay.get("replay_rows_available"),
                "learner_batch_size": replay.get("learner_batch_size"),
            },
            "optimizer_step_allowed": result.get("real_optimizer_step_allowed"),
            "learner_forward": {
                "status": learner.get("status"),
                "ok": learner.get("ok"),
                "api": learner.get("api"),
                "optimizer_step": learner.get("optimizer_step"),
                "model_hash_before": learner.get("model_hash_before"),
                "model_hash_after": learner.get("model_hash_after"),
                "model_parameters_changed": learner.get("model_parameters_changed"),
                "replay_scope": learner.get("replay_scope"),
                "replay_rows_available": learner.get("replay_rows_available"),
                "learner_batch_size": learner.get("learner_batch_size"),
                "blocker": learner.get("blocker"),
            },
            "checkpoints": result.get("checkpoints"),
            "next_command": result.get("next_command"),
        }
    )


def _iteration_edges(iterations: Any, *, edge_count: int = 2) -> dict[str, Any]:
    if not isinstance(iterations, list):
        return {"head": [], "tail": []}
    edge = int(edge_count)
    compact = [_compact_iteration_for_summary(item) for item in iterations]
    return {
        "head": compact[:edge],
        "tail": compact[-edge:] if len(compact) > edge else [],
    }


def _compact_iteration_for_summary(iteration: Any) -> dict[str, Any]:
    if not isinstance(iteration, Mapping):
        return {}
    return {
        "iteration": iteration.get("iteration"),
        "steps_survived": iteration.get("steps_survived"),
        "completed_episode_count": iteration.get("completed_episode_count"),
        "mean_completed_episode_steps": iteration.get("mean_completed_episode_steps"),
        "max_completed_episode_steps": iteration.get("max_completed_episode_steps"),
        "survival_reward_sum": iteration.get("survival_reward_sum"),
        "control_stochasticity": iteration.get("control_stochasticity"),
        "effective_policy_action_repeat_extra_probability": iteration.get(
            "effective_policy_action_repeat_extra_probability"
        ),
        "collect_timing_sec": iteration.get("collect_timing_sec"),
        "policy_batching_counts": iteration.get("policy_batching_counts"),
        "policy_search_call_count": iteration.get("policy_search_call_count"),
        "policy_search_row_count": iteration.get("policy_search_row_count"),
    }


def _result_payload(
    *,
    ok: bool,
    seed: int,
    batch_size: int,
    steps: int,
    outer_iterations: int,
    collect_steps_per_iteration: int,
    updates_per_iteration: int,
    num_simulations: int,
    learner_updates: int,
    allow_optimizer_step: bool,
    replay_scope: str,
    learner_sample_size: int | None,
    max_replay_rows: int | None,
    record_log_limit: int,
    replay_row_log_limit: int,
    alive_reward: float,
    dead_reward: float,
    action_selection_mode: str,
    collect_temperature: float,
    collect_epsilon: float,
    action_noop_probability: float,
    action_noop_warmup_iterations: int,
    policy_action_repeat_min: int,
    policy_action_repeat_max: int,
    policy_action_repeat_extra_probability: float,
    policy_action_repeat_warmup_iterations: int,
    observation_noise_std: float,
    trail_render_mode: str,
    use_cuda: bool,
    elapsed_sec: float,
    checkpoint_every_iterations: int,
    save_initial_checkpoint: bool,
    env_max_ticks: int | None = None,
    death_mode: str = DEFAULT_DEATH_MODE,
    policy_context: Mapping[str, Any],
    problems: list[str],
    records: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    total_steps_collected: int,
    total_replay_rows_collected: int,
    learner_replay_rows: list[dict[str, Any]] | None = None,
    iteration_summaries: list[dict[str, Any]],
    learner_forwards: list[dict[str, Any]],
    learner_forward: Mapping[str, Any],
    final_observation_shape: list[int],
    action_counts: Counter[int],
    per_player_action_counts: dict[str, Counter[int]],
    checkpoint_records: list[dict[str, Any]],
) -> dict[str, Any]:
    render_metadata = source_state_gray64_stack4_render_metadata(trail_render_mode)
    resolved_learner_replay_rows = (
        replay_rows if learner_replay_rows is None else learner_replay_rows
    )
    replay_rows_available = len(resolved_learner_replay_rows)
    learner_batch_size = _resolved_learner_batch_size(
        replay_rows_available,
        learner_sample_size=learner_sample_size,
    )
    sample = _summarize_replay_sample(
        resolved_learner_replay_rows,
        replay_scope=replay_scope,
        learner_sample_size=learner_sample_size,
    )
    optimizer_limit = (
        "does not call LightZero train_muzero"
        if allow_optimizer_step
        else "learn_mode.forward uses a guarded no-op optimizer-step patch"
    )
    return {
        "ok": bool(ok),
        "schema_id": TWO_SEAT_LIGHTZERO_TRAIN_SMOKE_SCHEMA_ID,
        "elapsed_sec": round(float(elapsed_sec), 6),
        "mode": (
            "bounded_two_seat_lightzero_collect_replay_real_train_smoke"
            if allow_optimizer_step
            else "bounded_two_seat_lightzero_collect_replay_learn_forward_smoke"
        ),
        "called_train_muzero": False,
        "true_lightzero_current_policy_self_play_training": False,
        "real_optimizer_step_allowed": bool(allow_optimizer_step),
        "simple_label": (
            "one shared LightZero policy object chooses both CurvyTron player "
            "actions in a bounded local joint-action smoke"
        ),
        "honest_limits": [
            "does not call LightZero train_muzero",
            "does not use the LightZero collector",
            "does not implement distributed actor weight refresh",
            optimizer_limit,
            "two-seat replay uses a local learn_mode.forward adapter, not LightZero's upstream GameBuffer target builder",
            "visuals are source-state gray64, not browser pixel fidelity",
        ],
        "what_works": [
            "builds float32 [B,P,4,64,64] observations",
            "remaps source player pixels into self/other perspective per policy row",
            "uses the same live LightZero MuZeroPolicy object for both seats",
            "repeats collect -> replay/sample -> learner update(s) -> checkpoint",
            "maps policy rows back to joint_action [B,P]",
            "steps VectorMultiplayerEnv with external joint actions",
            "records two-seat replay rows with iteration, env_row_id, player_id, decision_index, observation, action mask, action, action_weights, root_value, and survival reward",
            "can sample learner rows from either the current iteration or accumulated replay rows collected so far",
            "samples two-seat metadata through to the learner adapter for discounted survival value targets",
            (
                "samples replay rows for a real learn_mode.forward optimizer step"
                if allow_optimizer_step
                else "samples replay rows for a guarded learn_mode.forward attempt"
            ),
        ],
        "problems": list(problems),
        "inputs": {
            "seed": int(seed),
            "batch_size": int(batch_size),
            "player_count": 2,
            "steps_requested": int(steps),
            "outer_iterations": int(outer_iterations),
            "collect_steps_per_iteration": int(collect_steps_per_iteration),
            "updates_per_iteration": int(updates_per_iteration),
            "num_simulations": int(num_simulations),
            "learner_updates": int(learner_updates),
            "allow_optimizer_step": bool(allow_optimizer_step),
            "replay_scope": replay_scope,
            "learner_sample_size": learner_sample_size,
            "max_replay_rows": max_replay_rows,
            "record_log_limit": int(record_log_limit),
            "replay_row_log_limit": int(replay_row_log_limit),
            "initial_reset_seed_policy": "generated_from_env_rng",
            "autoreset_seed_policy": "generated_from_env_rng",
            "env_max_ticks": None if env_max_ticks is None else int(env_max_ticks),
            "death_mode": death_mode,
            "death_suppression_for_profile": (
                death_mode == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
            ),
            "death_suppression_claim": (
                "profile_only_not_source_fidelity"
                if death_mode == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
                else None
            ),
            "max_ticks_default_policy": (
                "DEFAULT_ENV_MAX_TICKS when --max-ticks is omitted; independent "
                "from collect_steps_per_iteration"
            ),
            "alive_reward": float(alive_reward),
            "dead_reward": float(dead_reward),
            "action_selection_mode": action_selection_mode,
            "action_selection_mode_semantics": _action_selection_mode_semantics(
                action_selection_mode
            ),
            "collect_temperature": float(collect_temperature),
            "collect_epsilon": float(collect_epsilon),
            "action_noop_probability": float(action_noop_probability),
            "action_noop_warmup_iterations": int(action_noop_warmup_iterations),
            "policy_action_repeat_min": int(policy_action_repeat_min),
            "policy_action_repeat_max": int(policy_action_repeat_max),
            "policy_action_repeat_extra_probability": float(
                policy_action_repeat_extra_probability
            ),
            "policy_action_repeat_warmup_iterations": int(
                policy_action_repeat_warmup_iterations
            ),
            "policy_action_repeat_semantics": (
                "per-env-row/per-seat action hold in the collector; when enabled, "
                "a seat can reuse its last executed action without a fresh policy "
                "search on that physical env step"
            ),
            "observation_noise_std": float(observation_noise_std),
            "trail_render_mode": render_metadata["trail_render_mode"],
            "default_trail_render_mode": render_metadata["default_trail_render_mode"],
            "supported_trail_render_modes": render_metadata[
                "supported_trail_render_modes"
            ],
            "use_cuda": bool(use_cuda),
            "checkpoint_every_iterations": int(checkpoint_every_iterations),
            "save_initial_checkpoint": bool(save_initial_checkpoint),
        },
        "lightzero_policy": _strip_runtime_object(policy_context),
        "surface": {
            "observation_shape": final_observation_shape,
            "per_policy_row_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "observation_dtype": SOURCE_STATE_GRAY64_NORMALIZED_DTYPE,
            "value_range": list(SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE),
            "single_frame_schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
            "single_frame_schema_hash": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH,
            "render": render_metadata,
            "stack_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
            "player_perspective_schema_id": PLAYER_PERSPECTIVE_SCHEMA_ID,
            "player_perspective": {
                "shape_preserved": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
                "semantics": (
                    "controlled player color becomes self grayscale before RGB-to-gray; "
                    "other visible player colors become other grayscale"
                ),
                "value_map": player_perspective_value_map(),
            },
            "action_space_size": ACTION_COUNT,
            "joint_action_shape": [int(batch_size), 2],
        },
        "checkpoints": {
            "status": "ok" if checkpoint_records else "not_requested_or_empty",
            "count": int(len(checkpoint_records)),
            "files": checkpoint_records,
        },
        "steps_survived": int(total_steps_collected),
        "total_steps_collected": int(total_steps_collected),
        "collect_timing_summary": _collect_timing_summary(iteration_summaries),
        "episode_duration_summary": _episode_duration_summary(iteration_summaries),
        "iterations": iteration_summaries,
        "action_counts": _counter_dict(action_counts),
        "action_counts_by_player": {
            player: _counter_dict(counts)
            for player, counts in sorted(per_player_action_counts.items())
        },
        "records": [_strip_large_arrays(record) for record in records],
        "replay": {
            "status": "ok" if replay_rows else "empty",
            "row_count": int(total_replay_rows_collected),
            "row_count_semantics": "total collected replay rows",
            "rows_in_payload": int(len(replay_rows)),
            "scope": replay_scope,
            "replay_rows_available": int(replay_rows_available),
            "learner_batch_size": int(learner_batch_size),
            "sample_semantics": "final learner-visible replay rows",
            "row_schema": TWO_SEAT_LIGHTZERO_REPLAY_ROW_SCHEMA_ID,
            "reward": "steps_survived_per_player; 1.0 while alive after step else 0.0",
            "reward_values": {
                "alive_reward": float(alive_reward),
                "dead_reward": float(dead_reward),
            },
            "return_schema_id": TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_ID,
            "return_schema_hash": TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_HASH,
            "value_target": (
                "discounted survival return grouped by "
                "episode_id/env_row_id/player_id/decision_index when episode "
                "metadata is present, with terminal survivor-win returns "
                "rewritten so the winner keeps its shaped survival return "
                "and the loser receives 0"
            ),
            "sample": sample,
            "rows": [_strip_large_arrays(row) for row in replay_rows],
        },
        "learner_forwards": [_strip_large_arrays(item) for item in learner_forwards],
        "learner_forward": _strip_large_arrays(learner_forward),
        "blocker": LEARN_BATCH_BLOCKER,
        "next_command": (
            "uv run --extra modal modal run "
            "-m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train "
            "--mode two-seat-selfplay --compute gpu-l4-t4 --seed 0 "
            "--batch-size 1 --outer-iterations 2 --collect-steps-per-iteration 4 "
            "--updates-per-iteration 1 --num-simulations 2 --replay-scope accumulated "
            "--learner-sample-size 32 --allow-optimizer-step "
            f"--two-seat-death-mode {death_mode} "
            f"--two-seat-trail-render-mode {render_metadata['trail_render_mode']}"
        ),
    }


def _collect_current_policy_iteration(
    policy: Any,
    env: VectorMultiplayerEnv,
    visual_stack: SourceStateGray64Stack4,
    *,
    batch: Any,
    observation: np.ndarray,
    iteration: int,
    decision_offset: int,
    collect_steps: int,
    alive_reward: float,
    dead_reward: float,
    action_selection_mode: str,
    collect_temperature: float,
    collect_epsilon: float,
    action_noop_probability: float,
    action_noise_rng: np.random.Generator,
    policy_action_repeat_min: int,
    policy_action_repeat_max: int,
    policy_action_repeat_extra_probability: float,
    policy_action_repeat_rng: np.random.Generator,
    observation_noise_std: float,
    observation_noise_rng: np.random.Generator,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    action_counts: Counter[int] = Counter()
    per_player_action_counts: dict[str, Counter[int]] = {
        "player_0": Counter(),
        "player_1": Counter(),
    }
    timing_sec: Counter[str] = Counter()
    policy_batching_counts: Counter[str] = Counter()
    policy_search_call_count = 0
    policy_search_row_count = 0
    physical_action_counts: Counter[int] = Counter()
    physical_per_player_action_counts: dict[str, Counter[int]] = {
        "player_0": Counter(),
        "player_1": Counter(),
    }
    stochasticity_counts: Counter[str] = Counter()
    repeat_count_histogram: Counter[int] = Counter()
    last_executed_action = np.full(
        (env.batch_size, env.player_count),
        NOOP_ACTION_ID,
        dtype=np.int16,
    )
    last_action_valid = np.zeros((env.batch_size, env.player_count), dtype=bool)
    repeat_remaining = np.zeros((env.batch_size, env.player_count), dtype=np.int16)

    def add_elapsed(name: str, started: float) -> None:
        timing_sec[name] += time.perf_counter() - started

    def clear_control_state(row_mask: np.ndarray) -> None:
        if not bool(np.asarray(row_mask, dtype=bool).any()):
            return
        last_executed_action[row_mask, :] = NOOP_ACTION_ID
        last_action_valid[row_mask, :] = False
        repeat_remaining[row_mask, :] = 0

    needs_reset = np.asarray(batch.info.get("needs_reset", batch.done), dtype=bool)
    if bool(needs_reset.any()):
        started = time.perf_counter()
        reset_batch = env.autoreset_done_rows(row_mask=needs_reset)
        observation = _refresh_reset_rows_in_visual_stack(
            visual_stack,
            env,
            row_mask=needs_reset,
        )
        add_elapsed("initial_autoreset_sec", started)
        batch = reset_batch
        clear_control_state(needs_reset)
        problems.extend(
            _validate_visual_batch(
                observation,
                batch.action_mask,
                label=f"iteration_{iteration}_initial_autoreset",
            )
        )

    for iteration_step in range(collect_steps):
        decision_index = int(decision_offset + iteration_step)
        started = time.perf_counter()
        policy_observation = _apply_observation_noise(
            observation,
            std=observation_noise_std,
            rng=observation_noise_rng,
        )
        add_elapsed("observation_noise_sec", started)
        started = time.perf_counter()
        mapping = build_policy_row_mapping(
            policy_observation,
            batch.action_mask.any(axis=2),
            batch.action_mask,
        )
        add_elapsed("policy_row_mapping_sec", started)
        active = np.asarray(mapping.row_mask, dtype=bool)
        if not bool(active.any()):
            break

        active_observations = mapping.observations[active]
        active_legal = mapping.legal_action_mask[active]
        active_env_row_id = mapping.env_row_id[active]
        active_player_id = mapping.player_id[active]

        active_count = int(mapping.active_count)
        repeat_remaining_before = repeat_remaining[
            active_env_row_id,
            active_player_id,
        ].astype(np.int16, copy=True)
        last_valid_for_active = last_action_valid[
            active_env_row_id,
            active_player_id,
        ]
        needs_policy_decision = (repeat_remaining_before <= 0) | (~last_valid_for_active)
        fresh_active_indices = np.flatnonzero(needs_policy_decision)
        repeated_active_indices = np.flatnonzero(~needs_policy_decision)
        stochasticity_counts["active_policy_rows"] += active_count
        stochasticity_counts["fresh_policy_decision_rows"] += int(
            fresh_active_indices.size
        )
        stochasticity_counts["reused_action_rows"] += int(
            repeated_active_indices.size
        )

        selected_actions: list[int] = []
        search_records_by_active: list[dict[str, Any] | None] = [None] * active_count
        batched_search: dict[str, Any] = {
            "ok": True,
            "records": [],
            "reason": None,
        }
        policy_batching_label = "reused_previous_actions"
        started = time.perf_counter()
        if fresh_active_indices.size:
            fresh_observations = active_observations[fresh_active_indices]
            fresh_legal = active_legal[fresh_active_indices]
            fresh_env_row_id = active_env_row_id[fresh_active_indices]
            fresh_player_id = active_player_id[fresh_active_indices]
            batched_search = _policy_actions_batch(
                policy,
                fresh_observations,
                fresh_legal,
                player_id=fresh_player_id,
                step_index=decision_index,
                mode=action_selection_mode,
                temperature=collect_temperature,
                epsilon=collect_epsilon,
            )
            policy_search_row_count += int(fresh_observations.shape[0])
            if batched_search.get("ok"):
                policy_search_call_count += 1
                policy_batching_counts["batched_fresh_rows"] += 1
                policy_batching_label = "batched_fresh_rows"
                fresh_search_records = list(batched_search["records"])
                selected_actions = [
                    int(record["action"]) for record in fresh_search_records
                ]
                for fresh_row, search_record in enumerate(fresh_search_records):
                    active_row = int(fresh_active_indices[fresh_row])
                    search_record["policy_row"] = active_row
                    search_record["search_batch_row"] = int(fresh_row)
                    search_record["env_row_id"] = int(fresh_env_row_id[fresh_row])
                    search_record["player_id"] = int(fresh_player_id[fresh_row])
                    search_record["iteration"] = int(iteration)
                    search_record["iteration_step"] = int(iteration_step)
                    action = int(search_record["action"])
                    if not bool(fresh_legal[fresh_row, action]):
                        problems.append(
                            "Batched LightZero selected an illegal action for "
                            f"iteration {iteration}, decision {decision_index}, "
                            f"row {active_row}: {action}"
                        )
                        break
                    search_records_by_active[active_row] = search_record
            else:
                policy_batching_counts["row_fallback"] += 1
                policy_batching_label = "row_fallback"
                for fresh_row in range(fresh_observations.shape[0]):
                    active_row = int(fresh_active_indices[fresh_row])
                    policy_search_call_count += 1
                    search_record = _policy_action(
                        policy,
                        {
                            "observation": fresh_observations[fresh_row],
                            "action_mask": fresh_legal[fresh_row].astype(np.int8),
                            "to_play": int(fresh_player_id[fresh_row]),
                        },
                        step_index=decision_index,
                        mode=action_selection_mode,
                        temperature=collect_temperature,
                        epsilon=collect_epsilon,
                    )
                    search_record["policy_row"] = active_row
                    search_record["search_batch_row"] = int(fresh_row)
                    search_record["env_row_id"] = int(fresh_env_row_id[fresh_row])
                    search_record["player_id"] = int(fresh_player_id[fresh_row])
                    search_record["iteration"] = int(iteration)
                    search_record["iteration_step"] = int(iteration_step)
                    search_record["batch_fallback_reason"] = batched_search.get("reason")
                    if not search_record.get("ok"):
                        problems.append(
                            f"LightZero {action_selection_mode} action selection failed for "
                            f"iteration {iteration}, decision {decision_index}, "
                            f"row {active_row}"
                        )
                        break
                    action = int(search_record["action"])
                    if not bool(fresh_legal[fresh_row, action]):
                        problems.append(
                            "LightZero selected an illegal action for "
                            f"iteration {iteration}, decision {decision_index}, "
                            f"row {active_row}: {action}"
                        )
                        break
                    selected_actions.append(action)
                    search_records_by_active[active_row] = search_record
        add_elapsed("policy_search_sec", started)
        if problems:
            break
        if len(selected_actions) != int(fresh_active_indices.size):
            break

        started = time.perf_counter()
        policy_selected = np.full(active_count, -1, dtype=np.int16)
        selected = last_executed_action[
            active_env_row_id,
            active_player_id,
        ].astype(np.int16, copy=True)
        if fresh_active_indices.size:
            policy_selected[fresh_active_indices] = np.asarray(
                selected_actions,
                dtype=np.int16,
            )
            selected[fresh_active_indices] = policy_selected[fresh_active_indices]
        action_noise_mask = np.zeros(active_count, dtype=bool)
        if action_noop_probability > 0.0 and fresh_active_indices.size:
            fresh_noise_mask = (
                action_noise_rng.random(fresh_active_indices.shape)
                < float(action_noop_probability)
            )
            action_noise_mask[fresh_active_indices] = fresh_noise_mask
            selected[fresh_active_indices[fresh_noise_mask]] = NOOP_ACTION_ID
        action_repeat_started = np.zeros(active_count, dtype=bool)
        action_repeat_count = np.zeros(active_count, dtype=np.int16)
        if fresh_active_indices.size:
            fresh_repeat_counts = _sample_policy_action_repeats(
                count=int(fresh_active_indices.size),
                min_repeat=policy_action_repeat_min,
                max_repeat=policy_action_repeat_max,
                extra_probability=policy_action_repeat_extra_probability,
                rng=policy_action_repeat_rng,
            )
            for fresh_row, active_row in enumerate(fresh_active_indices):
                env_row = int(active_env_row_id[active_row])
                player = int(active_player_id[active_row])
                repeat_count = int(fresh_repeat_counts[fresh_row])
                last_executed_action[env_row, player] = int(selected[active_row])
                last_action_valid[env_row, player] = True
                repeat_remaining[env_row, player] = repeat_count
                action_repeat_started[active_row] = True
                action_repeat_count[active_row] = repeat_count
                repeat_count_histogram[repeat_count] += 1
        for active_row, action in enumerate(selected):
            env_row = int(active_env_row_id[active_row])
            player = int(active_player_id[active_row])
            physical_action_counts[int(action)] += 1
            physical_per_player_action_counts[f"player_{player}"][int(action)] += 1
        joint_action = policy_rows_to_joint_action(
            mapping,
            selected,
            noop_action_id=NOOP_ACTION_ID,
            validate_legal=True,
            dtype=np.int16,
        )
        add_elapsed("joint_action_mapping_sec", started)

        before_alive = batch.info.get("alive")
        started = time.perf_counter()
        step_batch = env.step(joint_action)
        add_elapsed("env_step_sec", started)
        started = time.perf_counter()
        next_observation = visual_stack.update(env)
        add_elapsed("visual_stack_update_sec", started)
        started = time.perf_counter()
        replay_next_observation = _apply_observation_noise(
            next_observation,
            std=observation_noise_std,
            rng=observation_noise_rng,
        )
        add_elapsed("replay_observation_noise_sec", started)
        alive_after = step_batch.info.get("alive")
        done_by_row = np.asarray(step_batch.done, dtype=bool)
        winner_by_row = _winner_by_row(step_batch.info)
        for active_row in range(active_count):
            env_row = int(active_env_row_id[active_row])
            player = int(active_player_id[active_row])
            if repeat_remaining[env_row, player] > 0:
                repeat_remaining[env_row, player] -= 1
        repeat_remaining_after = repeat_remaining[
            active_env_row_id,
            active_player_id,
        ].astype(np.int16, copy=True)

        started = time.perf_counter()
        for policy_row in fresh_active_indices:
            search_record = search_records_by_active[int(policy_row)]
            if search_record is None:
                problems.append(
                    "missing search record for fresh policy decision at "
                    f"iteration {iteration}, decision {decision_index}, "
                    f"row {int(policy_row)}"
                )
                break
            env_row = int(active_env_row_id[policy_row])
            player = int(active_player_id[policy_row])
            action = int(selected[policy_row])
            policy_action = int(policy_selected[policy_row])
            replay_rows.append(
                {
                    "schema_id": TWO_SEAT_LIGHTZERO_REPLAY_ROW_SCHEMA_ID,
                    "iteration": int(iteration),
                    "iteration_step": int(iteration_step),
                    "episode_id": _episode_id_for_row(batch.info, env_row),
                    "decision_index": int(decision_index),
                    "env_row_id": env_row,
                    "player_id": player,
                    "to_play": player,
                    "observation": active_observations[policy_row].copy(),
                    "next_observation": replay_next_observation[env_row, player].copy(),
                    "action_mask": active_legal[policy_row].copy(),
                    "legal_action_mask": active_legal[policy_row].copy(),
                    "action": action,
                    "policy_selected_action": policy_action,
                    "executed_action": action,
                    "action_noise_noop_applied": bool(action_noise_mask[policy_row]),
                    "control_stochasticity_schema_id": CONTROL_STOCHASTICITY_SCHEMA_ID,
                    "policy_action_repeat_min": int(policy_action_repeat_min),
                    "policy_action_repeat_max": int(policy_action_repeat_max),
                    "policy_action_repeat_extra_probability": float(
                        policy_action_repeat_extra_probability
                    ),
                    "policy_action_repeat_started": bool(
                        action_repeat_started[policy_row]
                    ),
                    "policy_action_repeat_count": int(action_repeat_count[policy_row]),
                    "policy_action_repeat_remaining_after_step": int(
                        repeat_remaining_after[policy_row]
                    ),
                    "replay_row_for_fresh_policy_decision": True,
                    "observation_noise_std": float(observation_noise_std),
                    "action_weights": _action_weights(search_record, policy_action),
                    "root_value": _root_value(search_record),
                    "reward": _survival_reward(
                        alive_after,
                        env_row=env_row,
                        player_id=player,
                        alive_reward=alive_reward,
                        dead_reward=dead_reward,
                    ),
                    "done": bool(done_by_row[env_row]),
                    "terminal_winner": _terminal_winner_for_row(
                        winner_by_row,
                        env_row=env_row,
                        done=bool(done_by_row[env_row]),
                    ),
                    "return_schema_id": TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_ID,
                    "return_schema_hash": TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_HASH,
                    "policy_api": search_record.get("api"),
                    "action_selection_mode": action_selection_mode,
                    "policy_object": "shared_live_lightzero_policy",
                }
            )
            action_counts[action] += 1
            per_player_action_counts[f"player_{player}"][action] += 1
        add_elapsed("replay_row_build_sec", started)
        if problems:
            break

        record = {
            "iteration": int(iteration),
            "iteration_step": int(iteration_step),
            "decision_index": int(decision_index),
            "step_index": np.asarray(
                step_batch.info.get("step_index", np.full(env.batch_size, decision_index)),
                dtype=np.int64,
            ).copy(),
            "active_policy_row_shape": list(active_observations.shape),
            "active_policy_rows": int(active_count),
            "fresh_policy_input_shape": list(
                active_observations[fresh_active_indices].shape
            ),
            "fresh_policy_input_rows": int(fresh_active_indices.size),
            "reused_action_rows": int(repeated_active_indices.size),
            "policy_input_shape": list(active_observations.shape),
            "policy_input_rows": int(fresh_active_indices.size),
            "policy_api": (
                _action_selection_policy_api(action_selection_mode)
            ),
            "policy_batching": policy_batching_label,
            "policy_batch_fallback_reason": None
            if batched_search.get("ok")
            else batched_search.get("reason"),
            "action_selection_mode": action_selection_mode,
            "collect_temperature": float(collect_temperature),
            "collect_epsilon": float(collect_epsilon),
            "same_policy_object_for_both_seats": True,
            "joint_action": joint_action.copy(),
            "policy_selected_actions": policy_selected.copy(),
            "executed_actions": selected.copy(),
            "action_noise_noop_probability": float(action_noop_probability),
            "action_noise_noop_count": int(action_noise_mask.sum()),
            "control_stochasticity_schema_id": CONTROL_STOCHASTICITY_SCHEMA_ID,
            "policy_action_repeat_min": int(policy_action_repeat_min),
            "policy_action_repeat_max": int(policy_action_repeat_max),
            "policy_action_repeat_extra_probability": float(
                policy_action_repeat_extra_probability
            ),
            "policy_action_repeat_remaining_before": repeat_remaining_before.copy(),
            "policy_action_repeat_remaining_after": repeat_remaining_after.copy(),
            "policy_action_repeat_started_mask": action_repeat_started.copy(),
            "policy_action_repeat_counts": action_repeat_count.copy(),
            "observation_noise_std": float(observation_noise_std),
            "reward": step_batch.reward.copy(),
            "done": step_batch.done.copy(),
            "needs_reset": np.asarray(
                step_batch.info.get("needs_reset", step_batch.done),
                dtype=bool,
            ).copy(),
            "alive_before": None
            if before_alive is None
            else np.asarray(before_alive).copy(),
            "alive_after": alive_after,
            "search": [
                _compact_search_record(record)
                for record in search_records_by_active
                if record is not None
            ],
        }
        records.append(record)

        batch = step_batch
        observation = next_observation
        problems.extend(
            _validate_visual_batch(
                observation,
                batch.action_mask,
                label=f"iteration_{iteration}_step_{iteration_step}",
            )
        )
        needs_reset = np.asarray(batch.info.get("needs_reset", batch.done), dtype=bool)
        if bool(needs_reset.any()):
            clear_control_state(needs_reset)
        if bool(needs_reset.any()) and iteration_step + 1 < collect_steps:
            started = time.perf_counter()
            reset_batch = env.autoreset_done_rows(row_mask=needs_reset)
            observation = _refresh_reset_rows_in_visual_stack(
                visual_stack,
                env,
                row_mask=needs_reset,
            )
            add_elapsed("loop_autoreset_sec", started)
            batch = reset_batch
            record["autoreset_rows"] = reset_batch.info["reset_rows"].copy()
            problems.extend(
                _validate_visual_batch(
                    observation,
                    batch.action_mask,
                    label=f"iteration_{iteration}_autoreset_{iteration_step}",
                )
            )
        elif bool(batch.done.all()):
            break
        if problems:
            break

    return {
        "batch": batch,
        "observation": observation,
        "records": records,
        "replay_rows": replay_rows,
        "problems": problems,
        "steps_collected": int(len(records)),
        "action_counts": action_counts,
        "per_player_action_counts": per_player_action_counts,
        "physical_action_counts": physical_action_counts,
        "physical_per_player_action_counts": physical_per_player_action_counts,
        "control_stochasticity": {
            "schema_id": CONTROL_STOCHASTICITY_SCHEMA_ID,
            "policy_action_repeat_min": int(policy_action_repeat_min),
            "policy_action_repeat_max": int(policy_action_repeat_max),
            "policy_action_repeat_extra_probability": float(
                policy_action_repeat_extra_probability
            ),
            "policy_action_repeat_schedule": (
                "per-env-row/per-seat draws; repeated seats reuse the last "
                "executed action without a fresh policy search"
            ),
            "counts": _counter_dict(stochasticity_counts),
            "repeat_count_histogram": _counter_dict(repeat_count_histogram),
            "physical_action_counts": _counter_dict(physical_action_counts),
            "physical_action_counts_by_player": {
                player: _counter_dict(counts)
                for player, counts in sorted(physical_per_player_action_counts.items())
            },
        },
        "timing_sec": {
            key: round(float(timing_sec[key]), 6) for key in sorted(timing_sec)
        },
        "policy_batching_counts": _counter_dict(policy_batching_counts),
        "policy_search_call_count": int(policy_search_call_count),
        "policy_search_row_count": int(policy_search_row_count),
    }


def _policy_actions_batch(
    policy: Any,
    observations: np.ndarray,
    legal_action_mask: np.ndarray,
    *,
    player_id: np.ndarray,
    step_index: int,
    mode: str,
    temperature: float,
    epsilon: float,
) -> dict[str, Any]:
    if mode == ACTION_SELECTION_MODE_EVAL:
        return _policy_eval_actions_batch(
            policy,
            observations,
            legal_action_mask,
            player_id=player_id,
            step_index=step_index,
        )
    if mode != ACTION_SELECTION_MODE_COLLECT:
        return {
            "ok": False,
            "status": "failed",
            "step_index": int(step_index),
            "reason": f"unknown action selection mode: {mode}",
        }
    try:
        import torch

        obs_array = np.asarray(observations, dtype=np.float32)
        action_mask = np.asarray(legal_action_mask, dtype=np.float32)
        if obs_array.shape[0] != action_mask.shape[0]:
            raise ValueError("observations and legal_action_mask batch sizes differ")
        players = np.asarray(player_id, dtype=np.int64).reshape(-1)
        if players.shape != (obs_array.shape[0],):
            raise ValueError("player_id must have one value per observation")
        obs_tensor = torch.as_tensor(
            obs_array,
            dtype=torch.float32,
            device=_policy_model_device(policy),
        )
        ready_env_id = np.arange(obs_array.shape[0])
        to_play = [int(item) for item in players]
        with torch.no_grad():
            output = policy.collect_mode.forward(
                obs_tensor,
                action_mask=action_mask,
                temperature=float(temperature),
                to_play=to_play,
                epsilon=float(epsilon),
                ready_env_id=ready_env_id,
            )
        records = []
        for row in range(obs_array.shape[0]):
            row_output = _policy_output_row(output, row)
            records.append(
                {
                    "ok": True,
                    "status": "ok",
                    "step_index": int(step_index),
                    "api": "MuZeroPolicy.collect_mode.forward",
                    "action": _extract_eval_action(row_output),
                    "data_shape": [int(item) for item in obs_tensor.shape],
                    "row_data_shape": [int(item) for item in obs_array[row].shape],
                    "action_mask_shape": [int(item) for item in action_mask.shape],
                    "row_action_mask_shape": [
                        int(item) for item in action_mask[row].shape
                    ],
                    "to_play": [int(to_play[row])],
                    "ready_env_id": [int(ready_env_id[row])],
                    "batch_size": int(obs_array.shape[0]),
                    "temperature": float(temperature),
                    "epsilon": float(epsilon),
                    "compact_output": _compact_mcts_output(row_output),
                    "policy_batching": "batched_active_rows",
                }
            )
        return {
            "ok": True,
            "status": "ok",
            "step_index": int(step_index),
            "records": records,
        }
    except Exception as exc:  # pragma: no cover - installed runtime diagnosis.
        return {
            "ok": False,
            "status": "failed",
            "step_index": int(step_index),
            "reason": "batched MuZeroPolicy.collect_mode.forward failed",
            "exception": _exception_result(exc),
        }


def _action_selection_policy_api(mode: str) -> str:
    if mode == ACTION_SELECTION_MODE_COLLECT:
        return "MuZeroPolicy.collect_mode.forward"
    if mode == ACTION_SELECTION_MODE_EVAL:
        return "MuZeroPolicy.eval_mode.forward"
    return f"unknown:{mode}"


def _action_selection_mode_semantics(mode: str) -> str:
    if mode == ACTION_SELECTION_MODE_COLLECT:
        return "LightZero collect-mode MCTS with batched active policy rows"
    if mode == ACTION_SELECTION_MODE_EVAL:
        return "LightZero eval-mode MCTS; currently row fallback in this adapter"
    return "unknown action selection mode"


def _policy_eval_actions_batch(
    policy: Any,
    observations: np.ndarray,
    legal_action_mask: np.ndarray,
    *,
    player_id: np.ndarray,
    step_index: int,
) -> dict[str, Any]:
    try:
        records = []
        observation_array = np.asarray(observations)
        legal_array = np.asarray(legal_action_mask)
        players = np.asarray(player_id, dtype=np.int64).reshape(-1)
        if players.shape != (observation_array.shape[0],):
            raise ValueError("player_id must have one value per observation")
        for row in range(observation_array.shape[0]):
            record = _policy_eval_action(
                policy,
                {
                    "observation": observation_array[row],
                    "action_mask": legal_array[row].astype(np.int8),
                    "to_play": int(players[row]),
                },
                step_index=step_index,
            )
            record["policy_batching"] = "eval_row_fallback"
            records.append(record)
        return {
            "ok": True,
            "status": "ok",
            "step_index": int(step_index),
            "records": records,
        }
    except Exception as exc:  # pragma: no cover - installed runtime diagnosis.
        return {
            "ok": False,
            "status": "failed",
            "step_index": int(step_index),
            "reason": "batched eval action fallback failed",
            "exception": _exception_result(exc),
        }


def _policy_output_row(output: Any, row: int) -> Any:
    plain = _to_plain(output)
    if isinstance(plain, Mapping):
        for key in (int(row), str(int(row))):
            if key in plain:
                return plain[key]
        row_item: dict[str, Any] = {}
        for key, value in plain.items():
            if isinstance(value, Mapping):
                row_item[key] = _policy_output_row(value, row)
                continue
            try:
                array = np.asarray(value)
            except Exception:
                row_item[key] = value
                continue
            if array.ndim > 0 and array.shape[0] > int(row):
                row_item[key] = array[int(row)]
            else:
                row_item[key] = value
        return row_item
    if isinstance(plain, list) and len(plain) > int(row):
        return plain[int(row)]
    return output


def _policy_action(
    policy: Any,
    observation: Mapping[str, Any],
    *,
    step_index: int,
    mode: str,
    temperature: float,
    epsilon: float,
) -> dict[str, Any]:
    if mode == ACTION_SELECTION_MODE_EVAL:
        return _policy_eval_action(policy, observation, step_index=step_index)
    if mode != ACTION_SELECTION_MODE_COLLECT:
        return {
            "ok": False,
            "status": "failed",
            "step_index": int(step_index),
            "reason": f"unknown action selection mode: {mode}",
        }
    try:
        import torch

        obs_tensor = torch.as_tensor(
            np.asarray([observation["observation"]]),
            dtype=torch.float32,
            device=_policy_model_device(policy),
        )
        action_mask = np.asarray([observation["action_mask"]], dtype=np.float32)
        to_play = [int(np.asarray(observation.get("to_play", -1)).reshape(-1)[0])]
        ready_env_id = np.asarray([0])
        with torch.no_grad():
            output = policy.collect_mode.forward(
                obs_tensor,
                action_mask=action_mask,
                temperature=float(temperature),
                to_play=to_play,
                epsilon=float(epsilon),
                ready_env_id=ready_env_id,
            )
        return {
            "ok": True,
            "status": "ok",
            "step_index": int(step_index),
            "api": "MuZeroPolicy.collect_mode.forward",
            "action": _extract_eval_action(output),
            "data_shape": [int(item) for item in obs_tensor.shape],
            "action_mask_shape": [int(item) for item in action_mask.shape],
            "to_play": to_play,
            "ready_env_id": [0],
            "temperature": float(temperature),
            "epsilon": float(epsilon),
            "compact_output": _compact_mcts_output(output),
        }
    except Exception as exc:  # pragma: no cover - installed runtime diagnosis.
        return {
            "ok": False,
            "status": "failed",
            "step_index": int(step_index),
            "reason": "MuZeroPolicy.collect_mode.forward failed",
            "exception": _exception_result(exc),
        }


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _append_progress_line(
    path: Path,
    payload: Mapping[str, Any],
    *,
    print_line: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plain_payload = _to_plain(dict(payload))
    line = json.dumps(plain_payload, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")
    latest_path = path.with_name("progress_latest.json")
    latest_path.write_text(
        json.dumps(plain_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if print_line:
        print(f"TRAIN_PROGRESS {line}", flush=True)


def _iteration_progress_line(
    *,
    iteration: int,
    iteration_summary: Mapping[str, Any],
    total_steps_collected: int,
    total_replay_rows_collected: int,
    elapsed_sec: float,
    checkpoint: Mapping[str, Any] | None,
    problems: list[str],
) -> dict[str, Any]:
    learner_forwards = iteration_summary.get("learner_forwards")
    last_learner = {}
    if isinstance(learner_forwards, list) and learner_forwards:
        candidate = learner_forwards[-1]
        if isinstance(candidate, dict):
            last_learner = candidate
    compact_replay = _compact_progress_replay(iteration_summary.get("replay"))
    return {
        "event": "iteration",
        "timestamp": _utc_timestamp(),
        "iteration": int(iteration),
        "elapsed_sec": round(float(elapsed_sec), 6),
        "total_steps_collected": int(total_steps_collected),
        "total_replay_rows_collected": int(total_replay_rows_collected),
        "steps_collected_this_iteration": iteration_summary.get("steps_survived"),
        "completed_episode_count": iteration_summary.get("completed_episode_count"),
        "mean_completed_episode_steps": iteration_summary.get(
            "mean_completed_episode_steps"
        ),
        "max_completed_episode_steps": iteration_summary.get(
            "max_completed_episode_steps"
        ),
        "survival_reward_sum": iteration_summary.get("survival_reward_sum"),
        "action_counts": iteration_summary.get("action_counts"),
        "action_counts_by_player": iteration_summary.get("action_counts_by_player"),
        "effective_action_noop_probability": iteration_summary.get(
            "effective_action_noop_probability"
        ),
        "effective_policy_action_repeat_extra_probability": iteration_summary.get(
            "effective_policy_action_repeat_extra_probability"
        ),
        "control_stochasticity": iteration_summary.get("control_stochasticity"),
        "collect_timing_sec": iteration_summary.get("collect_timing_sec"),
        "policy_batching_counts": iteration_summary.get("policy_batching_counts"),
        "policy_search_call_count": iteration_summary.get("policy_search_call_count"),
        "policy_search_row_count": iteration_summary.get("policy_search_row_count"),
        "replay": compact_replay,
        "last_learner": last_learner,
        "checkpoint_saved": checkpoint is not None,
        "checkpoint": None if checkpoint is None else dict(checkpoint),
        "problem_count": int(len(problems)),
        "problems_tail": list(problems[-5:]),
    }


def _compact_progress_replay(replay: Any) -> dict[str, Any]:
    if not isinstance(replay, Mapping):
        return {}
    iteration_sample = replay.get("iteration_sample")
    learner_sample = replay.get("learner_sample")
    return {
        "status": replay.get("status"),
        "row_count": replay.get("row_count"),
        "scope": replay.get("scope"),
        "replay_rows_available": replay.get("replay_rows_available"),
        "learner_batch_size": replay.get("learner_batch_size"),
        "iteration_sample": _compact_progress_sample(iteration_sample),
        "learner_sample": _compact_progress_sample(learner_sample),
    }


def _compact_progress_sample(sample: Any) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        return {}
    return {
        "status": sample.get("status"),
        "sample_count": sample.get("sample_count"),
        "reward_sum": sample.get("reward_sum"),
        "reward_mean": sample.get("reward_mean"),
        "terminal_count": sample.get("terminal_count"),
        "action_counts": sample.get("action_counts"),
    }


def _iteration_summary(
    *,
    iteration: int,
    collect_steps_requested: int,
    records: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    action_counts: Counter[int],
    per_player_action_counts: dict[str, Counter[int]],
    learner_forwards: list[Mapping[str, Any]],
    checkpoint: Mapping[str, Any] | None,
    replay_scope: str,
    learner_replay_rows: list[dict[str, Any]],
    learner_sample_size: int | None,
    replay_rows_available: int,
    learner_batch_size: int,
) -> dict[str, Any]:
    rewards = np.asarray([row["reward"] for row in replay_rows], dtype=np.float32)
    episode_steps = _completed_episode_steps(records)
    return _to_plain(
        {
            "iteration": int(iteration),
            "collect_steps_requested": int(collect_steps_requested),
            "steps_survived": int(len(records)),
            "completed_episode_count": int(len(episode_steps)),
            "mean_completed_episode_steps": (
                float(np.mean(episode_steps)) if episode_steps else None
            ),
            "max_completed_episode_steps": int(max(episode_steps)) if episode_steps else None,
            "survival_reward_sum": float(rewards.sum()) if rewards.size else 0.0,
            "replay": {
                "status": "ok" if replay_rows else "empty",
                "row_count": int(len(replay_rows)),
                "scope": replay_scope,
                "replay_rows_available": int(replay_rows_available),
                "learner_batch_size": int(learner_batch_size),
                "iteration_sample": _summarize_replay_sample(replay_rows),
                "learner_sample": _summarize_replay_sample(
                    learner_replay_rows,
                    replay_scope=replay_scope,
                    learner_sample_size=learner_sample_size,
                ),
            },
            "action_counts": _counter_dict(action_counts),
            "action_counts_by_player": {
                player: _counter_dict(counts)
                for player, counts in sorted(per_player_action_counts.items())
            },
            "learner_forwards": [
                {
                    "status": item.get("status"),
                    "ok": item.get("ok"),
                    "update_index": item.get("update_index"),
                    "iteration_update_index": item.get("iteration_update_index"),
                    "optimizer_step": item.get("optimizer_step"),
                    "model_hash_before": item.get("model_hash_before"),
                    "model_hash_after": item.get("model_hash_after"),
                    "model_parameters_changed": item.get("model_parameters_changed"),
                    "replay_scope": item.get("replay_scope"),
                    "replay_rows_available": item.get("replay_rows_available"),
                    "learner_batch_size": item.get("learner_batch_size"),
                    "blocker": item.get("blocker"),
                    "reason": item.get("reason"),
                }
                for item in learner_forwards
            ],
            "checkpoint": None if checkpoint is None else dict(checkpoint),
        }
    )


def _completed_episode_steps(records: list[dict[str, Any]]) -> list[int]:
    completed: list[int] = []
    for record in records:
        done = np.asarray(record.get("done"), dtype=bool)
        step_index = np.asarray(record.get("step_index"), dtype=np.int64)
        if done.size == 0 or step_index.size == 0:
            continue
        for row in np.flatnonzero(done.any(axis=1) if done.ndim == 2 else done):
            try:
                completed.append(int(step_index[int(row)]) + 1)
            except Exception:
                completed.append(int(record.get("decision_index", 0)) + 1)
    return completed


def _episode_duration_summary(iteration_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        int(item["max_completed_episode_steps"])
        for item in iteration_summaries
        if item.get("max_completed_episode_steps") is not None
    ]
    means = [
        float(item["mean_completed_episode_steps"])
        for item in iteration_summaries
        if item.get("mean_completed_episode_steps") is not None
    ]
    return {
        "completed_iteration_count": int(len(completed)),
        "best_iteration_max_steps": int(max(completed)) if completed else None,
        "mean_of_iteration_mean_steps": float(np.mean(means)) if means else None,
        "first_iterations": [
            {
                "iteration": item.get("iteration"),
                "steps_collected": item.get("steps_survived"),
                "completed_episode_count": item.get("completed_episode_count"),
                "mean_completed_episode_steps": item.get("mean_completed_episode_steps"),
                "max_completed_episode_steps": item.get("max_completed_episode_steps"),
            }
            for item in iteration_summaries[:5]
        ],
        "last_iterations": [
            {
                "iteration": item.get("iteration"),
                "steps_collected": item.get("steps_survived"),
                "completed_episode_count": item.get("completed_episode_count"),
                "mean_completed_episode_steps": item.get("mean_completed_episode_steps"),
                "max_completed_episode_steps": item.get("max_completed_episode_steps"),
            }
            for item in iteration_summaries[-5:]
        ],
    }


def _collect_timing_summary(iteration_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    timings = [
        item.get("collect_timing_sec", {})
        for item in iteration_summaries
        if isinstance(item.get("collect_timing_sec"), Mapping)
    ]
    if not timings:
        return {
            "iteration_count": 0,
            "policy_search_call_count": 0,
            "policy_search_row_count": 0,
            "timing_sec": {},
        }
    names = sorted({str(name) for timing in timings for name in timing})
    timing_summary = {}
    for name in names:
        values = [
            float(timing[name])
            for timing in timings
            if name in timing and timing[name] is not None
        ]
        if not values:
            continue
        timing_summary[name] = {
            "sum": round(float(sum(values)), 6),
            "mean": round(float(np.mean(values)), 6),
            "min": round(float(min(values)), 6),
            "max": round(float(max(values)), 6),
            "first": round(float(values[0]), 6),
            "last": round(float(values[-1]), 6),
        }
    search_calls = sum(
        int(item.get("policy_search_call_count") or 0) for item in iteration_summaries
    )
    search_rows = sum(
        int(item.get("policy_search_row_count") or 0) for item in iteration_summaries
    )
    total_instrumented = sum(
        float(bucket.get("sum", 0.0)) for bucket in timing_summary.values()
    )
    policy_search_total = float(
        timing_summary.get("policy_search_sec", {}).get("sum", 0.0)
    )
    return {
        "iteration_count": int(len(timings)),
        "policy_search_call_count": int(search_calls),
        "policy_search_row_count": int(search_rows),
        "policy_rows_per_call": (
            round(float(search_rows) / float(search_calls), 6)
            if search_calls
            else None
        ),
        "total_instrumented_sec": round(float(total_instrumented), 6),
        "policy_search_fraction_of_instrumented": (
            round(float(policy_search_total) / float(total_instrumented), 6)
            if total_instrumented > 0.0
            else None
        ),
        "timing_sec": timing_summary,
    }


def _learn_mode_forward_update(
    policy: Any,
    sample: Mapping[str, Any],
    *,
    update_index: int,
) -> dict[str, Any]:
    try:
        model = getattr(policy, "_model", None)
        before_hash = _model_hash(model)
        current_batch, target_batch, sample_summary = _learn_mode_batches(policy, sample)
        with _temporary_policy_batch_size(
            policy,
            int(sample_summary["batch_size"]),
        ) as batch_size_patched:
            output = policy.learn_mode.forward([current_batch, target_batch])
        after_hash = _model_hash(model)
        return {
            "ok": True,
            "status": "updated",
            "api": "MuZeroPolicy.learn_mode.forward",
            "update_index": int(update_index),
            "optimizer_step": "allowed",
            "trainer_entrypoint_called": False,
            "model_hash_before": before_hash,
            "model_hash_after": after_hash,
            "model_parameters_changed": before_hash != after_hash,
            "policy_batch_size_patched": batch_size_patched,
            "sample": sample_summary,
            "loss": _loss_summary(output),
        }
    except Exception as exc:  # pragma: no cover - installed runtime diagnosis.
        return {
            "ok": False,
            "status": "failed",
            "reason": "MuZeroPolicy.learn_mode.forward failed",
            "blocking_call": "MuZeroPolicy.learn_mode.forward",
            "blocking_stage": "real optimizer update",
            "update_index": int(update_index),
            "optimizer_step": "allowed",
            "trainer_entrypoint_called": False,
            "exception": _exception_result(exc),
        }


def _save_lightzero_policy_checkpoint(
    policy: Any,
    checkpoint_dir: Path,
    *,
    iteration: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"iteration_{int(iteration)}.pth.tar"
    model = getattr(policy, "_model", None)
    target_model = getattr(policy, "_target_model", None)
    optimizer = getattr(policy, "_optimizer", None)
    payload: dict[str, Any] = {
        "model": model.state_dict() if model is not None else {},
        "metadata": _to_plain(dict(metadata)),
    }
    if target_model is not None and hasattr(target_model, "state_dict"):
        payload["target_model"] = target_model.state_dict()
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    best_path = checkpoint_dir / "ckpt_best.pth.tar"
    latest_path = checkpoint_dir / "latest.pth.tar"
    shutil.copy2(path, best_path)
    shutil.copy2(path, latest_path)
    record = {
        "iteration": int(iteration),
        "name": path.name,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "best_name": best_path.name,
        "best_path": str(best_path),
        "best_bytes": int(best_path.stat().st_size),
        "latest_name": latest_path.name,
        "latest_path": str(latest_path),
        "latest_bytes": int(latest_path.stat().st_size),
    }
    checkpoint_ref_prefix = metadata.get("checkpoint_root_ref")
    if checkpoint_ref_prefix is not None:
        prefix = str(checkpoint_ref_prefix).rstrip("/")
        record["ref"] = f"{prefix}/{path.name}"
        record["best_ref"] = f"{prefix}/{best_path.name}"
        record["latest_ref"] = f"{prefix}/{latest_path.name}"
    return record


def _sample_replay_batch(
    rows: list[dict[str, Any]],
    *,
    replay_scope: str = REPLAY_SCOPE_CURRENT_ITERATION,
    learner_sample_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if not rows:
        return {
            "ok": False,
            "reason": "no replay rows available",
            "replay_scope": replay_scope,
            "replay_rows_available": 0,
            "learner_sample_size": learner_sample_size,
            "learner_batch_size": 0,
        }
    selected_rows, sample_indices, sampled_without_replacement = _select_replay_rows(
        rows,
        learner_sample_size=learner_sample_size,
        rng=rng,
    )
    sample = {
        "ok": True,
        "batch_size": len(selected_rows),
        "learner_batch_size": len(selected_rows),
        "replay_scope": replay_scope,
        "replay_rows_available": len(rows),
        "learner_sample_size": learner_sample_size,
        "sampled_without_replacement": bool(sampled_without_replacement),
        "sample_indices": sample_indices,
        "players": sorted({int(row["player_id"]) for row in selected_rows}),
        "iteration_batch": np.asarray(
            [row["iteration"] for row in selected_rows],
            dtype=np.int64,
        ),
        "env_row_id_batch": np.asarray(
            [row["env_row_id"] for row in selected_rows],
            dtype=np.int64,
        ),
        "player_id_batch": np.asarray(
            [row["player_id"] for row in selected_rows],
            dtype=np.int64,
        ),
        "decision_index_batch": np.asarray(
            [row["decision_index"] for row in selected_rows],
            dtype=np.int64,
        ),
        "return_context_iteration_batch": np.asarray(
            [row["iteration"] for row in rows],
            dtype=np.int64,
        ),
        "return_context_env_row_id_batch": np.asarray(
            [row["env_row_id"] for row in rows],
            dtype=np.int64,
        ),
        "return_context_player_id_batch": np.asarray(
            [row["player_id"] for row in rows],
            dtype=np.int64,
        ),
        "return_context_decision_index_batch": np.asarray(
            [row["decision_index"] for row in rows],
            dtype=np.int64,
        ),
        "return_context_reward_batch": np.asarray(
            [row["reward"] for row in rows],
            dtype=np.float32,
        ),
        "return_context_done_batch": np.asarray(
            [row["done"] for row in rows],
            dtype=np.bool_,
        ),
        "return_context_terminal_winner_batch": np.asarray(
            [row.get("terminal_winner", -1) for row in rows],
            dtype=np.int64,
        ),
        "return_context_return_schema_id_batch": np.asarray(
            [row.get("return_schema_id", "") for row in rows],
            dtype=object,
        ),
        "observation_batch": np.stack(
            [row["observation"] for row in selected_rows],
            axis=0,
        ),
        "next_observation_batch": np.stack(
            [row["next_observation"] for row in selected_rows],
            axis=0,
        ),
        "action_batch": np.asarray(
            [row["action"] for row in selected_rows],
            dtype=np.int64,
        ),
        "reward_batch": np.asarray(
            [row["reward"] for row in selected_rows],
            dtype=np.float32,
        ),
        "done_batch": np.asarray([row["done"] for row in selected_rows], dtype=np.bool_),
        "terminal_winner_batch": np.asarray(
            [row.get("terminal_winner", -1) for row in selected_rows],
            dtype=np.int64,
        ),
        "return_schema_id_batch": np.asarray(
            [row.get("return_schema_id", "") for row in selected_rows],
            dtype=object,
        ),
        "policy_batch": np.stack(
            [row["action_weights"] for row in selected_rows],
            axis=0,
        ),
    }
    if all("episode_id" in row for row in selected_rows):
        sample["episode_id_batch"] = np.asarray(
            [row["episode_id"] for row in selected_rows],
            dtype=np.int64,
        )
    if all("episode_id" in row for row in rows):
        sample["return_context_episode_id_batch"] = np.asarray(
            [row["episode_id"] for row in rows],
            dtype=np.int64,
        )
    return sample


def _summarize_replay_sample(
    rows: list[dict[str, Any]],
    *,
    replay_scope: str = REPLAY_SCOPE_CURRENT_ITERATION,
    learner_sample_size: int | None = None,
) -> dict[str, Any]:
    sample = _sample_replay_batch(
        rows,
        replay_scope=replay_scope,
        learner_sample_size=learner_sample_size,
    )
    if not sample.get("ok"):
        return sample
    return {
        "ok": True,
        "batch_size": int(sample["batch_size"]),
        "learner_batch_size": int(sample["learner_batch_size"]),
        "replay_scope": sample["replay_scope"],
        "replay_rows_available": int(sample["replay_rows_available"]),
        "learner_sample_size": sample["learner_sample_size"],
        "sampled_without_replacement": bool(sample["sampled_without_replacement"]),
        "sample_indices": _indices_summary(sample["sample_indices"]),
        "players": sample["players"],
        "metadata": {
            "iteration_batch_shape": list(sample["iteration_batch"].shape),
            "episode_id_batch_shape": list(sample["episode_id_batch"].shape)
            if "episode_id_batch" in sample
            else None,
            "env_row_id_batch_shape": list(sample["env_row_id_batch"].shape),
            "player_id_batch_shape": list(sample["player_id_batch"].shape),
            "decision_index_batch_shape": list(sample["decision_index_batch"].shape),
            "return_context_rows": int(sample["return_context_reward_batch"].shape[0]),
            "return_context_episode_id_batch_shape": list(
                sample["return_context_episode_id_batch"].shape
            )
            if "return_context_episode_id_batch" in sample
            else None,
        },
        "observation_batch_shape": list(sample["observation_batch"].shape),
        "next_observation_batch_shape": list(sample["next_observation_batch"].shape),
        "action_batch_shape": list(sample["action_batch"].shape),
        "reward_batch_shape": list(sample["reward_batch"].shape),
        "policy_batch_shape": list(sample["policy_batch"].shape),
        "reward_sum": float(np.asarray(sample["reward_batch"], dtype=np.float32).sum()),
    }


def _indices_summary(indices: Any, *, edge_count: int = 8) -> dict[str, Any]:
    if indices is None:
        return {"count": 0, "head": [], "tail": []}
    values = [int(index) for index in list(indices)]
    edge = int(edge_count)
    return {
        "count": int(len(values)),
        "head": values[:edge],
        "tail": values[-edge:] if len(values) > edge else [],
    }


def _select_replay_rows(
    rows: list[dict[str, Any]],
    *,
    learner_sample_size: int | None,
    rng: np.random.Generator | None,
) -> tuple[list[dict[str, Any]], list[int], bool]:
    available = len(rows)
    if learner_sample_size is None or int(learner_sample_size) >= available:
        return list(rows), list(range(available)), False

    sample_size = int(learner_sample_size)
    generator = rng if rng is not None else np.random.default_rng(0)
    indices = sorted(
        int(index)
        for index in generator.choice(available, size=sample_size, replace=False)
    )
    return [rows[index] for index in indices], indices, True


def _resolved_learner_batch_size(
    replay_rows_available: int,
    *,
    learner_sample_size: int | None,
) -> int:
    if replay_rows_available < 1:
        return 0
    if learner_sample_size is None:
        return int(replay_rows_available)
    return min(int(learner_sample_size), int(replay_rows_available))


def _replay_sample_rng(
    *,
    seed: int,
    iteration: int,
    update_index: int,
) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [
            int(seed) & 0xFFFFFFFF,
            (int(seed) >> 32) & 0xFFFFFFFF,
            int(iteration) & 0xFFFFFFFF,
            int(update_index) & 0xFFFFFFFF,
        ]
    )
    return np.random.default_rng(seed_sequence)


def _action_noise_rng(
    *,
    seed: int,
    iteration: int,
) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [
            int(seed) & 0xFFFFFFFF,
            (int(seed) >> 32) & 0xFFFFFFFF,
            int(iteration) & 0xFFFFFFFF,
            0xA17C10,
        ]
    )
    return np.random.default_rng(seed_sequence)


def _policy_action_repeat_rng(
    *,
    seed: int,
    iteration: int,
) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [
            int(seed) & 0xFFFFFFFF,
            (int(seed) >> 32) & 0xFFFFFFFF,
            int(iteration) & 0xFFFFFFFF,
            POLICY_ACTION_REPEAT_RNG_SALT,
        ]
    )
    return np.random.default_rng(seed_sequence)


def _observation_noise_rng(
    *,
    seed: int,
    iteration: int,
) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [
            int(seed) & 0xFFFFFFFF,
            (int(seed) >> 32) & 0xFFFFFFFF,
            int(iteration) & 0xFFFFFFFF,
            0x0B5E9A7E,
        ]
    )
    return np.random.default_rng(seed_sequence)


def _apply_observation_noise(
    observation: np.ndarray,
    *,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    resolved_std = float(std)
    if resolved_std <= 0.0:
        return observation
    noisy = observation.astype(np.float32, copy=True)
    noisy += rng.normal(loc=0.0, scale=resolved_std, size=noisy.shape).astype(
        np.float32
    )
    return np.clip(noisy, 0.0, 1.0)


def _scheduled_action_noop_probability(
    *,
    final_probability: float,
    warmup_iterations: int,
    iteration: int,
) -> float:
    if warmup_iterations <= 0:
        return float(final_probability)
    progress = min(max(int(iteration), 0), int(warmup_iterations))
    return float(final_probability) * (float(progress) / float(warmup_iterations))


def _scheduled_policy_action_repeat_extra_probability(
    *,
    final_probability: float,
    warmup_iterations: int,
    iteration: int,
) -> float:
    if warmup_iterations <= 0:
        return float(final_probability)
    progress = min(max(int(iteration), 0), int(warmup_iterations))
    return float(final_probability) * (float(progress) / float(warmup_iterations))


def _validate_policy_action_repeat_config(
    *,
    min_repeat: int,
    max_repeat: int,
    extra_probability: float,
    warmup_iterations: int,
) -> None:
    if int(min_repeat) < 1:
        raise ValueError("policy_action_repeat_min must be at least 1")
    if int(max_repeat) < int(min_repeat):
        raise ValueError(
            "policy_action_repeat_max must be greater than or equal to "
            "policy_action_repeat_min"
        )
    if not 0.0 <= float(extra_probability) <= 1.0:
        raise ValueError("policy_action_repeat_extra_probability must be in [0, 1]")
    if int(warmup_iterations) < 0:
        raise ValueError("policy_action_repeat_warmup_iterations must be >= 0")


def _sample_policy_action_repeats(
    *,
    count: int,
    min_repeat: int,
    max_repeat: int,
    extra_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    repeats = np.full(int(count), int(min_repeat), dtype=np.int16)
    if int(count) < 1 or int(max_repeat) <= int(min_repeat):
        return repeats
    probability = float(extra_probability)
    if probability <= 0.0:
        return repeats
    while True:
        can_extend = repeats < int(max_repeat)
        if not bool(can_extend.any()):
            return repeats
        draws = rng.random(int(count))
        extend = can_extend & (draws < probability)
        if not bool(extend.any()):
            return repeats
        repeats[extend] += 1


def _sample_replay_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "replay_scope": sample.get("replay_scope"),
        "replay_rows_available": sample.get("replay_rows_available"),
        "learner_sample_size": sample.get("learner_sample_size"),
        "sampled_without_replacement": sample.get("sampled_without_replacement"),
        "sample_indices": _indices_summary(sample.get("sample_indices")),
    }


def _refresh_reset_rows_in_visual_stack(
    visual_stack: SourceStateGray64Stack4,
    env: VectorMultiplayerEnv,
    *,
    row_mask: np.ndarray,
) -> np.ndarray:
    """Refresh only reset env rows after autoreset, without rolling live rows."""

    mask = np.asarray(row_mask, dtype=bool)
    if mask.shape != (env.batch_size,):
        raise ValueError("row_mask must have shape [B]")
    if not bool(mask.any()):
        return visual_stack.stack.copy()

    reset_stack = SourceStateGray64Stack4(
        batch_size=env.batch_size,
        player_count=env.player_count,
        trail_render_mode=visual_stack.trail_render_mode,
    )
    reset_observation = reset_stack.update(env)
    visual_stack.stack[mask] = reset_observation[mask]
    return visual_stack.stack.copy()


def _validate_visual_batch(
    observation: np.ndarray,
    action_mask: np.ndarray,
    *,
    label: str,
) -> list[str]:
    problems: list[str] = []
    obs = np.asarray(observation)
    mask = np.asarray(action_mask)
    if obs.ndim != 5 or obs.shape[1:] != (2, *STACKED_SOURCE_STATE_GRAY64_SHAPE):
        problems.append(f"{label}: observation shape {obs.shape!r}, expected [B,2,4,64,64]")
    if obs.dtype != np.float32:
        problems.append(f"{label}: observation dtype {obs.dtype}, expected float32")
    if obs.size and (float(obs.min()) < 0.0 or float(obs.max()) > 1.0):
        problems.append(f"{label}: observation values are outside [0,1]")
    if mask.ndim != 3 or mask.shape[:2] != obs.shape[:2] or mask.shape[2] != ACTION_COUNT:
        problems.append(f"{label}: action_mask shape {mask.shape!r}, expected [B,2,3]")
    if (
        obs.ndim == 5
        and obs.shape[1:] == (2, *STACKED_SOURCE_STATE_GRAY64_SHAPE)
        and mask.ndim == 3
        and mask.shape[:2] == obs.shape[:2]
    ):
        active_pair_rows = mask.any(axis=2).all(axis=1)
        if bool(active_pair_rows.any()):
            delta = np.max(
                np.abs(obs[active_pair_rows, 0] - obs[active_pair_rows, 1])
            )
            if float(delta) <= 0.0:
                problems.append(f"{label}: active player visual frames are identical")
    return problems


def _survival_reward(
    alive_after: Any,
    *,
    env_row: int,
    player_id: int,
    alive_reward: float = DEFAULT_ALIVE_REWARD,
    dead_reward: float = DEFAULT_DEAD_REWARD,
) -> float:
    alive = np.asarray(alive_after, dtype=bool)
    if alive.ndim != 2:
        raise ValueError("step info alive must have shape [B,P] for survival reward")
    return float(alive_reward) if bool(alive[int(env_row), int(player_id)]) else float(dead_reward)


def _winner_by_row(info: Mapping[str, Any]) -> np.ndarray | None:
    winners = info.get("winner")
    if winners is None:
        return None
    array = np.asarray(winners, dtype=np.int64)
    return array if array.ndim == 1 else None


def _terminal_winner_for_row(
    winner_by_row: np.ndarray | None,
    *,
    env_row: int,
    done: bool,
) -> int:
    if not done or winner_by_row is None or winner_by_row.shape[0] <= int(env_row):
        return -1
    return int(winner_by_row[int(env_row)])


def _episode_id_for_row(info: Mapping[str, Any], env_row: int) -> int:
    episode_ids = info.get("episode_id")
    if episode_ids is None:
        return -1
    array = np.asarray(episode_ids, dtype=np.int64)
    if array.ndim != 1 or array.shape[0] <= int(env_row):
        return -1
    return int(array[int(env_row)])


def _compact_search_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(record.get("ok")),
        "status": record.get("status"),
        "policy_row": int(record.get("policy_row", -1)),
        "env_row_id": int(record.get("env_row_id", -1)),
        "player_id": int(record.get("player_id", -1)),
        "action": record.get("action"),
        "api": record.get("api"),
        "compact_output": record.get("compact_output"),
    }


def _counter_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--outer-iterations", type=int, default=1)
    parser.add_argument(
        "--collect-steps-per-iteration",
        type=int,
        default=None,
        help="Collection steps before each learner phase. Defaults to --steps.",
    )
    parser.add_argument(
        "--updates-per-iteration",
        type=int,
        default=None,
        help="Learner updates after each collection phase. Defaults to --learner-updates.",
    )
    parser.add_argument("--num-simulations", type=int, default=2)
    parser.add_argument("--learner-updates", type=int, default=1)
    parser.add_argument(
        "--replay-scope",
        choices=REPLAY_SCOPE_CHOICES,
        default=REPLAY_SCOPE_CURRENT_ITERATION,
        help=(
            "Rows available to each learner update. Defaults to current_iteration "
            "to preserve the original smoke behavior."
        ),
    )
    parser.add_argument(
        "--learner-sample-size",
        type=int,
        default=None,
        help=(
            "Optional cap for learner rows. If smaller than the available replay "
            "rows, samples without replacement using a deterministic seed."
        ),
    )
    parser.add_argument(
        "--allow-optimizer-step",
        action="store_true",
        help="Allow learn_mode.forward to update the LightZero policy weights.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional directory for iteration_*.pth.tar policy checkpoints.",
    )
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--decision-ms", type=float, default=300.0)
    parser.add_argument("--alive-reward", type=float, default=DEFAULT_ALIVE_REWARD)
    parser.add_argument("--dead-reward", type=float, default=DEFAULT_DEAD_REWARD)
    parser.add_argument(
        "--action-selection-mode",
        choices=ACTION_SELECTION_MODE_CHOICES,
        default=ACTION_SELECTION_MODE_COLLECT,
        help="Use collect mode for training data by default; eval mode is greedy.",
    )
    parser.add_argument("--collect-temperature", type=float, default=1.0)
    parser.add_argument("--collect-epsilon", type=float, default=0.25)
    parser.add_argument(
        "--action-noop-probability",
        type=float,
        default=DEFAULT_ACTION_NOOP_PROBABILITY,
    )
    parser.add_argument(
        "--action-noop-warmup-iterations",
        type=int,
        default=DEFAULT_ACTION_NOOP_WARMUP_ITERATIONS,
        help="Linearly ramp action no-op probability from 0 to the requested value over this many outer iterations.",
    )
    parser.add_argument(
        "--policy-action-repeat-min",
        type=int,
        default=DEFAULT_POLICY_ACTION_REPEAT_MIN,
        help="Minimum physical env steps to hold a fresh policy action for each row/seat.",
    )
    parser.add_argument(
        "--policy-action-repeat-max",
        type=int,
        default=DEFAULT_POLICY_ACTION_REPEAT_MAX,
        help="Maximum physical env steps to hold a fresh policy action for each row/seat.",
    )
    parser.add_argument(
        "--policy-action-repeat-extra-probability",
        type=float,
        default=DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
        help="Chance to extend a row/seat action hold by one more physical step, repeated until max.",
    )
    parser.add_argument(
        "--policy-action-repeat-warmup-iterations",
        type=int,
        default=DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS,
        help="Linearly ramp action-repeat extra probability from 0 to the requested value.",
    )
    parser.add_argument(
        "--observation-noise-std",
        type=float,
        default=DEFAULT_OBSERVATION_NOISE_STD,
        help="Gaussian noise added to policy visual inputs and replay frames, clipped to [0, 1].",
    )
    parser.add_argument(
        "--trail-render-mode",
        choices=TRAIL_RENDER_MODE_ORDER,
        default=TRAIL_RENDER_MODE_DEFAULT,
        help="Visual stack renderer. browser_lines is the canonical RGB-to-gray path.",
    )
    parser.add_argument(
        "--death-mode",
        choices=tuple(vector_runtime.DEATH_MODES),
        default=DEFAULT_DEATH_MODE,
        help="Use profile_no_death only for optimizer long-survival profiling.",
    )
    parser.add_argument(
        "--checkpoint-every-iterations",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_ITERATIONS,
        help="Save iteration checkpoints every N outer iterations. Final checkpoint is still saved.",
    )
    parser.add_argument(
        "--save-initial-checkpoint",
        action="store_true",
        help="Also save iteration_0 before any optimizer update. Off by default to avoid a large startup write.",
    )
    parser.add_argument("--progress-path", default=None)
    parser.add_argument(
        "--progress-every-iterations",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_ITERATIONS,
    )
    parser.add_argument(
        "--progress-commit-every-iterations",
        type=int,
        default=DEFAULT_PROGRESS_COMMIT_EVERY_ITERATIONS,
    )
    parser.add_argument(
        "--no-progress-print",
        action="store_true",
        help="Write progress JSONL without printing TRAIN_PROGRESS lines.",
    )
    parser.add_argument("--max-replay-rows", type=int, default=4096)
    parser.add_argument("--record-log-limit", type=int, default=512)
    parser.add_argument("--replay-row-log-limit", type=int, default=256)
    parser.add_argument(
        "--allow-missing-lightzero",
        action="store_true",
        help="Return a blocked result instead of requiring installed LightZero.",
    )
    parser.add_argument(
        "--output",
        choices=("full", "summary"),
        default="summary",
    )
    args = parser.parse_args()
    result = run_curvytron_two_seat_lightzero_train_smoke(
        seed=args.seed,
        batch_size=args.batch_size,
        steps=args.steps,
        outer_iterations=args.outer_iterations,
        collect_steps_per_iteration=args.collect_steps_per_iteration,
        updates_per_iteration=args.updates_per_iteration,
        num_simulations=args.num_simulations,
        learner_updates=args.learner_updates,
        allow_optimizer_step=args.allow_optimizer_step,
        checkpoint_dir=args.checkpoint_dir,
        max_ticks=args.max_ticks,
        decision_ms=args.decision_ms,
        replay_scope=args.replay_scope,
        learner_sample_size=args.learner_sample_size,
        alive_reward=args.alive_reward,
        dead_reward=args.dead_reward,
        action_selection_mode=args.action_selection_mode,
        collect_temperature=args.collect_temperature,
        collect_epsilon=args.collect_epsilon,
        action_noop_probability=args.action_noop_probability,
        action_noop_warmup_iterations=args.action_noop_warmup_iterations,
        policy_action_repeat_min=args.policy_action_repeat_min,
        policy_action_repeat_max=args.policy_action_repeat_max,
        policy_action_repeat_extra_probability=(
            args.policy_action_repeat_extra_probability
        ),
        policy_action_repeat_warmup_iterations=(
            args.policy_action_repeat_warmup_iterations
        ),
        observation_noise_std=args.observation_noise_std,
        trail_render_mode=args.trail_render_mode,
        death_mode=args.death_mode,
        checkpoint_every_iterations=args.checkpoint_every_iterations,
        save_initial_checkpoint=args.save_initial_checkpoint,
        progress_path=args.progress_path,
        progress_every_iterations=args.progress_every_iterations,
        progress_commit_every_iterations=args.progress_commit_every_iterations,
        progress_print=not args.no_progress_print,
        max_replay_rows=args.max_replay_rows,
        record_log_limit=args.record_log_limit,
        replay_row_log_limit=args.replay_row_log_limit,
        require_installed_lightzero=not args.allow_missing_lightzero,
    )
    payload = (
        compact_curvytron_two_seat_lightzero_train_smoke_summary(result)
        if args.output == "summary"
        else result
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not result["ok"] and not args.allow_missing_lightzero:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "ACTION_SELECTION_MODE_CHOICES",
    "ACTION_SELECTION_MODE_COLLECT",
    "ACTION_SELECTION_MODE_EVAL",
    "DEFAULT_ACTION_NOOP_PROBABILITY",
    "DEFAULT_ACTION_NOOP_WARMUP_ITERATIONS",
    "DEFAULT_OBSERVATION_NOISE_STD",
    "LEARN_BATCH_BLOCKER",
    "REPLAY_SCOPE_ACCUMULATED",
    "REPLAY_SCOPE_CHOICES",
    "REPLAY_SCOPE_CURRENT_ITERATION",
    "TWO_SEAT_LIGHTZERO_REPLAY_ROW_SCHEMA_ID",
    "TWO_SEAT_LIGHTZERO_TRAIN_SMOKE_SCHEMA_ID",
    "compact_curvytron_two_seat_lightzero_train_smoke_summary",
    "run_curvytron_two_seat_lightzero_train_smoke",
]
