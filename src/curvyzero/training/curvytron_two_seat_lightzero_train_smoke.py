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
from curvyzero.env.trainer_contract import stable_contract_hash
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    ACTION_COUNT,
    NOOP_ACTION_ID,
    PLAYER_PERSPECTIVE_SCHEMA_ID,
    STACK_RENDER_MODE_DEFAULT,
    STACK_RENDER_MODE_ORDER,
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
from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    snapshot_backed_lightzero_checkpoint_opponent_policy,
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
    "curvyzero_two_seat_dense_survival_plus_sparse_outcome/v0"
)
TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_HASH = stable_contract_hash(
    {
        "schema_id": TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_ID,
        "trainer_reward": (
            "dense alive/dead helper + immediate same-step bonus pickup helper "
            "+ env sparse terminal outcome * terminal_outcome_reward_per_step "
            "* episode_step_count"
        ),
        "dense_helper": "alive_reward while alive after step else dead_reward",
        "bonus_pickup_helper": "bonus_pickup_reward_per_catch on the exact catch step",
        "sparse_outcome": "VectorMultiplayerEnv reward for that player",
        "terminal_outcome_scale": "terminal_outcome_reward_per_step * episode_step_count",
        "return_target": "discounted sum of trainer rewards per player trajectory",
        "scope": "per_player_two_seat_training_reward_targets",
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
DEFAULT_ALIVE_REWARD = 0.01
DEFAULT_DEAD_REWARD = 0.0
DEFAULT_TERMINAL_OUTCOME_REWARD_PER_STEP = DEFAULT_ALIVE_REWARD
DEFAULT_BONUS_PICKUP_REWARD_PER_CATCH = 0.05
DEFAULT_RETURN_TARGET_DISCOUNT = 1.0
DEFAULT_ENV_MAX_TICKS = 65_536
DEFAULT_DEATH_MODE = vector_runtime.DEATH_MODE_NORMAL
DEFAULT_NATURAL_BONUS_SPAWN = True
DEFAULT_CHECKPOINT_EVERY_ITERATIONS = 100
DEFAULT_PROGRESS_EVERY_ITERATIONS = 100
DEFAULT_PROGRESS_COMMIT_EVERY_ITERATIONS = 100
DEFAULT_ACTION_NOOP_PROBABILITY = 0.0
DEFAULT_ACTION_NOOP_WARMUP_ITERATIONS = 0
DEFAULT_POLICY_ACTION_REPEAT_MIN = 1
DEFAULT_POLICY_ACTION_REPEAT_MAX = 1
DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY = 0.0
DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS = 0
DEFAULT_OBSERVATION_NOISE_STD = 0.10
POLICY_ACTION_REPEAT_RNG_SALT = 0xC0A7A11
CONTROL_STOCHASTICITY_SCHEMA_ID = "curvyzero_two_seat_policy_noop_skip/v0"
OPPONENT_MIX_SCHEMA_ID = "curvyzero_two_seat_opponent_mix/v0"
ROLLOUT_KIND_CURRENT_POLICY_SELFPLAY = "current_policy_selfplay"
ROLLOUT_KIND_CURRENT_POLICY_VS_FROZEN = "current_policy_vs_frozen_checkpoint"
ACTION_SOURCE_CURRENT_POLICY = "current_policy"
ACTION_SOURCE_FROZEN_CHECKPOINT = "frozen_checkpoint"
ACTION_SOURCE_ABSENT_NOOP = "absent_noop"
DEFAULT_FROZEN_OPPONENT_PROBABILITY = 0.0
DEFAULT_FROZEN_OPPONENT_PLAYER_ID = 1
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
    natural_bonus_spawn: bool = DEFAULT_NATURAL_BONUS_SPAWN,
    decision_ms: float = 300.0,
    replay_scope: str = REPLAY_SCOPE_CURRENT_ITERATION,
    learner_sample_size: int | None = None,
    max_replay_rows: int | None = 4096,
    record_log_limit: int = 512,
    replay_row_log_limit: int = 256,
    alive_reward: float = DEFAULT_ALIVE_REWARD,
    dead_reward: float = DEFAULT_DEAD_REWARD,
    terminal_outcome_reward_per_step: float = (
        DEFAULT_TERMINAL_OUTCOME_REWARD_PER_STEP
    ),
    bonus_pickup_reward_per_catch: float = DEFAULT_BONUS_PICKUP_REWARD_PER_CATCH,
    return_target_discount: float = DEFAULT_RETURN_TARGET_DISCOUNT,
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
    trail_render_mode: str = STACK_RENDER_MODE_DEFAULT,
    learning_rate: float | None = None,
    frozen_opponent_probability: float = DEFAULT_FROZEN_OPPONENT_PROBABILITY,
    frozen_opponent_checkpoint_path: str | Path | None = None,
    frozen_opponent_checkpoint_ref: str | None = None,
    frozen_opponent_snapshot_ref: str | None = None,
    frozen_opponent_checkpoint_state_key: str | None = None,
    frozen_opponent_player_id: int = DEFAULT_FROZEN_OPPONENT_PLAYER_ID,
    frozen_opponent_num_simulations: int | None = None,
    frozen_opponent_use_cuda: bool | None = None,
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
    resolved_natural_bonus_spawn = bool(natural_bonus_spawn)
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
    resolved_frozen_opponent_probability = float(frozen_opponent_probability)
    if not 0.0 <= resolved_frozen_opponent_probability <= 1.0:
        raise ValueError("frozen_opponent_probability must be in [0, 1]")
    resolved_frozen_opponent_player_id = int(frozen_opponent_player_id)
    if resolved_frozen_opponent_player_id not in (0, 1):
        raise ValueError("frozen_opponent_player_id must be 0 or 1")
    resolved_frozen_opponent_num_simulations = (
        int(num_simulations)
        if frozen_opponent_num_simulations is None
        else int(frozen_opponent_num_simulations)
    )
    if resolved_frozen_opponent_num_simulations < 1:
        raise ValueError("frozen_opponent_num_simulations must be >= 1")
    resolved_frozen_opponent_use_cuda = (
        bool(use_cuda)
        if frozen_opponent_use_cuda is None
        else bool(frozen_opponent_use_cuda)
    )
    resolved_trail_render_mode = validate_stack_trail_render_mode(trail_render_mode)
    resolved_alive_reward = float(alive_reward)
    resolved_dead_reward = float(dead_reward)
    resolved_terminal_outcome_reward_per_step = float(terminal_outcome_reward_per_step)
    if not np.isfinite(resolved_terminal_outcome_reward_per_step):
        raise ValueError("terminal_outcome_reward_per_step must be finite")
    resolved_bonus_pickup_reward_per_catch = float(bonus_pickup_reward_per_catch)
    if (
        not np.isfinite(resolved_bonus_pickup_reward_per_catch)
        or resolved_bonus_pickup_reward_per_catch < 0.0
    ):
        raise ValueError("bonus_pickup_reward_per_catch must be finite and >= 0")
    resolved_return_target_discount = float(return_target_discount)
    if not 0.0 <= resolved_return_target_discount <= 1.0:
        raise ValueError("return_target_discount must be in [0, 1]")
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
        learning_rate=learning_rate,
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
            terminal_outcome_reward_per_step=(
                resolved_terminal_outcome_reward_per_step
            ),
            bonus_pickup_reward_per_catch=resolved_bonus_pickup_reward_per_catch,
            return_target_discount=resolved_return_target_discount,
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
                learning_rate=learning_rate,
                frozen_opponent_probability=resolved_frozen_opponent_probability,
                frozen_opponent_checkpoint_ref=frozen_opponent_checkpoint_ref,
                frozen_opponent_snapshot_ref=frozen_opponent_snapshot_ref,
                frozen_opponent_checkpoint_state_key=frozen_opponent_checkpoint_state_key,
                frozen_opponent_player_id=resolved_frozen_opponent_player_id,
                frozen_opponent_num_simulations=resolved_frozen_opponent_num_simulations,
                frozen_opponent_use_cuda=resolved_frozen_opponent_use_cuda,
                frozen_opponent_metadata=_frozen_opponent_metadata(
                    enabled=False,
                    probability=resolved_frozen_opponent_probability,
                    checkpoint_path=frozen_opponent_checkpoint_path,
                    checkpoint_ref=frozen_opponent_checkpoint_ref,
                    snapshot_ref=frozen_opponent_snapshot_ref,
                    state_key=frozen_opponent_checkpoint_state_key,
                    player_id=resolved_frozen_opponent_player_id,
                    num_simulations=resolved_frozen_opponent_num_simulations,
                    use_cuda=resolved_frozen_opponent_use_cuda,
                ),
                use_cuda=use_cuda,
                elapsed_sec=time.perf_counter() - run_started,
                checkpoint_every_iterations=resolved_checkpoint_every_iterations,
                save_initial_checkpoint=save_initial_checkpoint,
                env_max_ticks=resolved_max_ticks,
                death_mode=resolved_death_mode,
                natural_bonus_spawn=resolved_natural_bonus_spawn,
                policy_context=policy_context,
                problems=problems,
                records=[],
                replay_rows=[],
                total_steps_collected=0,
                total_replay_rows_collected=0,
                initial_reset_seed_sample=[],
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

    frozen_opponent_policy = None
    frozen_opponent_metadata = _frozen_opponent_metadata(
        enabled=False,
        probability=resolved_frozen_opponent_probability,
        checkpoint_path=frozen_opponent_checkpoint_path,
        checkpoint_ref=frozen_opponent_checkpoint_ref,
        snapshot_ref=frozen_opponent_snapshot_ref,
        state_key=frozen_opponent_checkpoint_state_key,
        player_id=resolved_frozen_opponent_player_id,
        num_simulations=resolved_frozen_opponent_num_simulations,
        use_cuda=resolved_frozen_opponent_use_cuda,
    )
    if resolved_frozen_opponent_probability > 0.0:
        if frozen_opponent_checkpoint_path is None:
            raise ValueError(
                "frozen_opponent_checkpoint_path is required when "
                "frozen_opponent_probability > 0"
            )
        snapshot_ref = frozen_opponent_snapshot_ref or "curvytron_two_seat_frozen_opponent"
        checkpoint_ref = frozen_opponent_checkpoint_ref or str(frozen_opponent_checkpoint_path)
        frozen_opponent_policy = snapshot_backed_lightzero_checkpoint_opponent_policy(
            checkpoint_path=frozen_opponent_checkpoint_path,
            snapshot_ref=snapshot_ref,
            checkpoint_ref=checkpoint_ref,
            seed=int(seed),
            num_simulations=resolved_frozen_opponent_num_simulations,
            batch_size=int(batch_size),
            use_cuda=resolved_frozen_opponent_use_cuda,
            state_key=frozen_opponent_checkpoint_state_key,
        )
        frozen_opponent_metadata = _frozen_opponent_metadata(
            enabled=True,
            probability=resolved_frozen_opponent_probability,
            checkpoint_path=frozen_opponent_checkpoint_path,
            checkpoint_ref=checkpoint_ref,
            snapshot_ref=snapshot_ref,
            state_key=frozen_opponent_checkpoint_state_key,
            player_id=resolved_frozen_opponent_player_id,
            num_simulations=resolved_frozen_opponent_num_simulations,
            use_cuda=resolved_frozen_opponent_use_cuda,
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
                    "natural_bonus_spawn": bool(resolved_natural_bonus_spawn),
                    "alive_reward": float(resolved_alive_reward),
                    "dead_reward": float(resolved_dead_reward),
                    "terminal_outcome_reward_per_step": float(
                        resolved_terminal_outcome_reward_per_step
                    ),
                    "bonus_pickup_reward_per_catch": float(
                        resolved_bonus_pickup_reward_per_catch
                    ),
                    "return_target_discount": float(resolved_return_target_discount),
                    "trail_render_mode": resolved_trail_render_mode,
                    "learning_rate": None if learning_rate is None else float(learning_rate),
                    "policy_action_repeat_min": int(resolved_policy_action_repeat_min),
                    "policy_action_repeat_max": int(resolved_policy_action_repeat_max),
                    "policy_action_repeat_extra_probability": float(
                        resolved_policy_action_repeat_extra_probability
                    ),
                    "policy_action_repeat_warmup_iterations": int(
                        resolved_policy_action_repeat_warmup_iterations
                    ),
                    "frozen_opponent": frozen_opponent_metadata,
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
                "natural_bonus_spawn": bool(resolved_natural_bonus_spawn),
                "alive_reward": float(resolved_alive_reward),
                "dead_reward": float(resolved_dead_reward),
                "terminal_outcome_reward_per_step": float(
                    resolved_terminal_outcome_reward_per_step
                ),
                "bonus_pickup_reward_per_catch": float(
                    resolved_bonus_pickup_reward_per_catch
                ),
                "return_target_discount": float(resolved_return_target_discount),
                "action_selection_mode": action_selection_mode,
                "collect_temperature": float(resolved_collect_temperature),
                "collect_epsilon": float(resolved_collect_epsilon),
                "trail_render_mode": resolved_trail_render_mode,
                "learning_rate": None if learning_rate is None else float(learning_rate),
                "frozen_opponent": frozen_opponent_metadata,
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
        natural_bonus_spawn=resolved_natural_bonus_spawn,
    )
    visual_stack = SourceStateGray64Stack4(
        batch_size=batch_size,
        player_count=2,
        trail_render_mode=resolved_trail_render_mode,
    )

    # The run seed initializes the RNG, but each row reset gets its own generated
    # reset seed. This keeps training starts varied while remaining reproducible.
    batch = env.reset(seed=None)
    frozen_opponent_mix_rng = _frozen_opponent_mix_rng(seed=seed)
    frozen_opponent_row_mask = _sample_frozen_opponent_rows(
        batch_size=env.batch_size,
        probability=resolved_frozen_opponent_probability,
        rng=frozen_opponent_mix_rng,
        enabled=frozen_opponent_policy is not None,
    )
    initial_reset_seed_sample = _reset_seed_sample(batch.info)
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
            terminal_outcome_reward_per_step=(
                resolved_terminal_outcome_reward_per_step
            ),
            bonus_pickup_reward_per_catch=resolved_bonus_pickup_reward_per_catch,
            return_target_discount=resolved_return_target_discount,
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
            frozen_opponent_policy=frozen_opponent_policy,
            frozen_opponent_row_mask=frozen_opponent_row_mask,
            frozen_opponent_probability=resolved_frozen_opponent_probability,
            frozen_opponent_mix_rng=frozen_opponent_mix_rng,
            frozen_opponent_player_id=resolved_frozen_opponent_player_id,
            frozen_opponent_metadata=frozen_opponent_metadata,
        )
        batch = collection["batch"]
        observation = collection["observation"]
        frozen_opponent_row_mask = collection["frozen_opponent_row_mask"]
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
                    "terminal_outcome_reward_per_step": float(
                        resolved_terminal_outcome_reward_per_step
                    ),
                    "bonus_pickup_reward_per_catch": float(
                        resolved_bonus_pickup_reward_per_catch
                    ),
                    "return_target_discount": float(resolved_return_target_discount),
                    "learning_rate": None if learning_rate is None else float(learning_rate),
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
                    "frozen_opponent": frozen_opponent_metadata,
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
        iteration_summaries[-1]["visual_stack_dirty_render"] = collection[
            "visual_stack_dirty_render"
        ]
        iteration_summaries[-1]["policy_batching_counts"] = collection[
            "policy_batching_counts"
        ]
        iteration_summaries[-1]["policy_search_call_count"] = int(
            collection["policy_search_call_count"]
        )
        iteration_summaries[-1]["policy_search_row_count"] = int(
            collection["policy_search_row_count"]
        )
        iteration_summaries[-1]["opponent_mix"] = collection["opponent_mix"]
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
            initial_reset_seed_sample=initial_reset_seed_sample,
            alive_reward=resolved_alive_reward,
            dead_reward=resolved_dead_reward,
            terminal_outcome_reward_per_step=(
                resolved_terminal_outcome_reward_per_step
            ),
            bonus_pickup_reward_per_catch=(
                resolved_bonus_pickup_reward_per_catch
            ),
            return_target_discount=resolved_return_target_discount,
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
            learning_rate=learning_rate,
            frozen_opponent_probability=resolved_frozen_opponent_probability,
            frozen_opponent_checkpoint_ref=frozen_opponent_checkpoint_ref,
            frozen_opponent_snapshot_ref=frozen_opponent_snapshot_ref,
            frozen_opponent_checkpoint_state_key=frozen_opponent_checkpoint_state_key,
            frozen_opponent_player_id=resolved_frozen_opponent_player_id,
            frozen_opponent_num_simulations=resolved_frozen_opponent_num_simulations,
            frozen_opponent_use_cuda=resolved_frozen_opponent_use_cuda,
            frozen_opponent_metadata=frozen_opponent_metadata,
            use_cuda=use_cuda,
            elapsed_sec=time.perf_counter() - run_started,
            checkpoint_every_iterations=resolved_checkpoint_every_iterations,
            save_initial_checkpoint=save_initial_checkpoint,
            env_max_ticks=resolved_max_ticks,
            death_mode=resolved_death_mode,
            natural_bonus_spawn=resolved_natural_bonus_spawn,
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
                "natural_bonus_spawn": inputs.get("natural_bonus_spawn"),
                "death_suppression_for_profile": inputs.get(
                    "death_suppression_for_profile"
                ),
                "action_noop_probability": inputs.get("action_noop_probability"),
                "action_noop_warmup_iterations": inputs.get(
                    "action_noop_warmup_iterations"
                ),
                "alive_reward": inputs.get("alive_reward"),
                "dead_reward": inputs.get("dead_reward"),
                "terminal_outcome_reward_per_step": inputs.get(
                    "terminal_outcome_reward_per_step"
                ),
                "bonus_pickup_reward_per_catch": inputs.get(
                    "bonus_pickup_reward_per_catch"
                ),
                "return_target_discount": inputs.get("return_target_discount"),
                "policy_action_repeat_min": inputs.get("policy_action_repeat_min"),
                "policy_action_repeat_max": inputs.get("policy_action_repeat_max"),
                "policy_action_repeat_extra_probability": inputs.get(
                    "policy_action_repeat_extra_probability"
                ),
                "policy_action_repeat_warmup_iterations": inputs.get(
                    "policy_action_repeat_warmup_iterations"
                ),
                "trail_render_mode": inputs.get("trail_render_mode"),
                "frozen_opponent": inputs.get("frozen_opponent"),
                "frozen_opponent_probability": inputs.get(
                    "frozen_opponent_probability"
                ),
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
            "opponent_mix": result.get("opponent_mix"),
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
        "training_reward_sum": iteration.get("training_reward_sum"),
        "survival_reward_sum": iteration.get("survival_reward_sum"),
        "bonus_pickup_reward_sum": iteration.get("bonus_pickup_reward_sum"),
        "bonus_pickup_count": iteration.get("bonus_pickup_count"),
        "sparse_outcome_reward_sum": iteration.get("sparse_outcome_reward_sum"),
        "terminal_outcome_reward_sum": iteration.get("terminal_outcome_reward_sum"),
        "fresh_policy_action_summary": iteration.get("fresh_policy_action_summary"),
        "control_stochasticity": iteration.get("control_stochasticity"),
        "effective_policy_action_repeat_extra_probability": iteration.get(
            "effective_policy_action_repeat_extra_probability"
        ),
        "collect_timing_sec": iteration.get("collect_timing_sec"),
        "visual_stack_dirty_render": iteration.get("visual_stack_dirty_render"),
        "policy_batching_counts": iteration.get("policy_batching_counts"),
        "policy_search_call_count": iteration.get("policy_search_call_count"),
        "policy_search_row_count": iteration.get("policy_search_row_count"),
        "opponent_mix": iteration.get("opponent_mix"),
    }


def _reset_seed_sample(
    info: Mapping[str, Any],
    *,
    row_mask: np.ndarray | None = None,
    limit: int = 8,
) -> list[dict[str, int]]:
    reset_seed = info.get("reset_seed")
    if reset_seed is None:
        return []
    array = np.asarray(reset_seed)
    if array.ndim != 1:
        return []
    if row_mask is None:
        rows = np.arange(array.shape[0], dtype=np.int64)
    else:
        mask = np.asarray(row_mask, dtype=bool)
        if mask.shape != (array.shape[0],):
            return []
        rows = np.flatnonzero(mask)
    return [
        {"env_row_id": int(row), "reset_seed": int(array[int(row)])}
        for row in rows[: max(0, int(limit))]
    ]


def _frozen_opponent_metadata(
    *,
    enabled: bool,
    probability: float,
    checkpoint_path: str | Path | None,
    checkpoint_ref: str | None,
    snapshot_ref: str | None,
    state_key: str | None,
    player_id: int,
    num_simulations: int,
    use_cuda: bool,
) -> dict[str, Any]:
    return {
        "schema_id": OPPONENT_MIX_SCHEMA_ID,
        "enabled": bool(enabled),
        "probability": float(probability),
        "checkpoint_path": None
        if checkpoint_path is None
        else str(checkpoint_path),
        "checkpoint_ref": checkpoint_ref,
        "snapshot_ref": snapshot_ref,
        "checkpoint_state_key": state_key,
        "frozen_player_id": int(player_id),
        "current_policy_player_id": 1 - int(player_id),
        "num_simulations": int(num_simulations),
        "use_cuda": bool(use_cuda),
        "semantics": (
            "frozen checkpoint supplies opponent actions for selected env rows; "
            "only current-policy seats create learner replay rows"
        ),
    }


def _frozen_opponent_mix_rng(*, seed: int) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [
            int(seed) & 0xFFFFFFFF,
            (int(seed) >> 32) & 0xFFFFFFFF,
            0xF20A0A,
        ]
    )
    return np.random.default_rng(seed_sequence)


def _sample_frozen_opponent_rows(
    *,
    batch_size: int,
    probability: float,
    rng: np.random.Generator,
    enabled: bool,
) -> np.ndarray:
    rows = np.arange(int(batch_size), dtype=np.int64)
    return _sample_frozen_opponent_rows_from_candidates(
        rows,
        probability=probability,
        rng=rng,
        enabled=enabled,
        batch_size=int(batch_size),
    )


def _resample_frozen_opponent_rows(
    frozen_opponent_row_mask: np.ndarray,
    *,
    row_mask: np.ndarray,
    probability: float,
    rng: np.random.Generator,
    enabled: bool,
) -> np.ndarray:
    current = np.asarray(frozen_opponent_row_mask, dtype=bool).copy()
    resets = np.asarray(row_mask, dtype=bool)
    if current.shape != resets.shape:
        raise ValueError("frozen opponent row mask and reset mask must both be [B]")
    rows = np.flatnonzero(resets).astype(np.int64)
    current[rows] = False
    replacement = _sample_frozen_opponent_rows_from_reset_candidates(
        rows,
        probability=probability,
        rng=rng,
        enabled=enabled,
        batch_size=current.shape[0],
    )
    current |= replacement
    return current


def _sample_frozen_opponent_rows_from_candidates(
    rows: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
    enabled: bool,
    batch_size: int,
) -> np.ndarray:
    mask = np.zeros(int(batch_size), dtype=bool)
    if not bool(enabled):
        return mask
    probability = float(probability)
    if probability <= 0.0 or rows.size == 0:
        return mask
    count = int(round(float(rows.size) * probability))
    if probability > 0.0 and count == 0:
        count = 1
    count = min(max(count, 0), int(rows.size))
    if count < 1:
        return mask
    chosen = rng.choice(rows, size=count, replace=False)
    mask[np.asarray(chosen, dtype=np.int64)] = True
    return mask


def _sample_frozen_opponent_rows_from_reset_candidates(
    rows: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
    enabled: bool,
    batch_size: int,
) -> np.ndarray:
    mask = np.zeros(int(batch_size), dtype=bool)
    if not bool(enabled):
        return mask
    probability = float(probability)
    if probability <= 0.0 or rows.size == 0:
        return mask
    draws = rng.random(int(rows.size)) < min(probability, 1.0)
    if not bool(np.any(draws)):
        return mask
    mask[np.asarray(rows[draws], dtype=np.int64)] = True
    return mask


def _current_policy_live_mask(
    live_mask: np.ndarray,
    *,
    frozen_opponent_row_mask: np.ndarray,
    frozen_opponent_player_id: int,
) -> np.ndarray:
    current = np.asarray(live_mask, dtype=bool).copy()
    frozen_rows = np.asarray(frozen_opponent_row_mask, dtype=bool)
    if current.ndim != 2 or current.shape[1] != 2:
        raise ValueError("live_mask must have shape [B,2]")
    if frozen_rows.shape != (current.shape[0],):
        raise ValueError("frozen_opponent_row_mask must have shape [B]")
    current[frozen_rows, int(frozen_opponent_player_id)] = False
    return current


def _frozen_opponent_slot_mask(
    legal_action_mask: np.ndarray,
    *,
    frozen_opponent_row_mask: np.ndarray,
    frozen_opponent_player_id: int,
) -> np.ndarray:
    legal = np.asarray(legal_action_mask, dtype=bool)
    if legal.ndim != 3 or legal.shape[1] != 2:
        raise ValueError("legal_action_mask must have shape [B,2,A]")
    frozen_rows = np.asarray(frozen_opponent_row_mask, dtype=bool)
    if frozen_rows.shape != (legal.shape[0],):
        raise ValueError("frozen_opponent_row_mask must have shape [B]")
    mask = np.zeros(legal.shape[:2], dtype=bool)
    mask[frozen_rows, int(frozen_opponent_player_id)] = True
    return mask & legal.any(axis=2)


def _select_frozen_opponent_actions(
    frozen_opponent_policy: Any,
    *,
    legal_action_mask: np.ndarray,
    opponent_mask: np.ndarray,
    observation: np.ndarray,
    decision_index: int,
) -> dict[str, Any]:
    empty_actions = np.full(
        np.asarray(legal_action_mask).shape[:2],
        NOOP_ACTION_ID,
        dtype=np.int16,
    )
    if frozen_opponent_policy is None or not bool(np.asarray(opponent_mask).any()):
        return {
            "ok": True,
            "actions": empty_actions,
            "sidecar": None,
            "selection": None,
            "opponent_action_count": 0,
        }
    try:
        selection = frozen_opponent_policy.select_actions(
            np.asarray(legal_action_mask, dtype=bool),
            np.asarray(opponent_mask, dtype=bool),
            decision_index=int(decision_index),
            observation=np.asarray(observation, dtype=np.float32),
        )
        actions = np.asarray(selection.actions, dtype=np.int16)
        if actions.shape != empty_actions.shape:
            raise ValueError(
                f"frozen opponent actions shape {actions.shape!r}; "
                f"expected {empty_actions.shape!r}"
            )
        legal = np.asarray(legal_action_mask, dtype=bool)
        opponent_slots = np.asarray(opponent_mask, dtype=bool)
        for env_row, player_id in np.argwhere(opponent_slots):
            action = int(actions[int(env_row), int(player_id)])
            if action < 0 or action >= legal.shape[2] or not bool(
                legal[int(env_row), int(player_id), action]
            ):
                raise ValueError(
                    "frozen opponent selected illegal action "
                    f"{action} for env_row={int(env_row)}, player={int(player_id)}"
                )
        full_actions = empty_actions.copy()
        full_actions[opponent_slots] = actions[opponent_slots]
        return {
            "ok": True,
            "actions": full_actions,
            "sidecar": selection.sidecar(),
            "selection": selection,
            "opponent_action_count": int(opponent_slots.sum()),
        }
    except Exception as exc:  # pragma: no cover - runtime checkpoint diagnosis.
        return {
            "ok": False,
            "actions": empty_actions,
            "sidecar": None,
            "selection": None,
            "opponent_action_count": 0,
            "exception": _exception_result(exc),
        }


def _action_source_by_slot(
    *,
    live_mask: np.ndarray,
    current_policy_live_mask: np.ndarray,
    frozen_opponent_slot_mask: np.ndarray,
) -> np.ndarray:
    live = np.asarray(live_mask, dtype=bool)
    current = np.asarray(current_policy_live_mask, dtype=bool)
    frozen = np.asarray(frozen_opponent_slot_mask, dtype=bool)
    source = np.full(live.shape, ACTION_SOURCE_ABSENT_NOOP, dtype=object)
    source[current] = ACTION_SOURCE_CURRENT_POLICY
    source[frozen] = ACTION_SOURCE_FROZEN_CHECKPOINT
    return source


def _rollout_kind_by_row(
    *,
    batch_size: int,
    frozen_opponent_row_mask: np.ndarray,
) -> np.ndarray:
    kinds = np.full(
        int(batch_size),
        ROLLOUT_KIND_CURRENT_POLICY_SELFPLAY,
        dtype=object,
    )
    kinds[np.asarray(frozen_opponent_row_mask, dtype=bool)] = (
        ROLLOUT_KIND_CURRENT_POLICY_VS_FROZEN
    )
    return kinds


def _opponent_mix_summary(
    *,
    frozen_opponent_row_mask: np.ndarray,
    frozen_opponent_player_id: int,
    frozen_opponent_probability: float,
    frozen_opponent_metadata: Mapping[str, Any],
    step_counts: Counter[str] | None = None,
) -> dict[str, Any]:
    frozen_rows = np.asarray(frozen_opponent_row_mask, dtype=bool)
    batch_size = int(frozen_rows.shape[0])
    frozen_count = int(frozen_rows.sum())
    summary = {
        "schema_id": OPPONENT_MIX_SCHEMA_ID,
        "enabled": bool(frozen_opponent_metadata.get("enabled")),
        "configured_probability": float(frozen_opponent_probability),
        "batch_size": batch_size,
        "current_policy_selfplay_rows": int(batch_size - frozen_count),
        "current_policy_vs_frozen_rows": frozen_count,
        "frozen_opponent_player_id": int(frozen_opponent_player_id),
        "current_policy_player_id_in_mixed_rows": 1 - int(frozen_opponent_player_id),
        "frozen_opponent": dict(frozen_opponent_metadata),
    }
    if step_counts is not None:
        summary["step_counts"] = _counter_dict(step_counts)
    return summary


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
    initial_reset_seed_sample: list[dict[str, int]],
    alive_reward: float,
    dead_reward: float,
    terminal_outcome_reward_per_step: float,
    bonus_pickup_reward_per_catch: float,
    return_target_discount: float,
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
    learning_rate: float | None,
    frozen_opponent_probability: float = DEFAULT_FROZEN_OPPONENT_PROBABILITY,
    frozen_opponent_checkpoint_ref: str | None = None,
    frozen_opponent_snapshot_ref: str | None = None,
    frozen_opponent_checkpoint_state_key: str | None = None,
    frozen_opponent_player_id: int = DEFAULT_FROZEN_OPPONENT_PLAYER_ID,
    frozen_opponent_num_simulations: int = 1,
    frozen_opponent_use_cuda: bool = False,
    frozen_opponent_metadata: Mapping[str, Any] | None = None,
    use_cuda: bool,
    elapsed_sec: float,
    checkpoint_every_iterations: int,
    save_initial_checkpoint: bool,
    env_max_ticks: int | None = None,
    death_mode: str = DEFAULT_DEATH_MODE,
    natural_bonus_spawn: bool = DEFAULT_NATURAL_BONUS_SPAWN,
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
    resolved_frozen_opponent_metadata = dict(
        frozen_opponent_metadata
        or _frozen_opponent_metadata(
            enabled=False,
            probability=frozen_opponent_probability,
            checkpoint_path=None,
            checkpoint_ref=frozen_opponent_checkpoint_ref,
            snapshot_ref=frozen_opponent_snapshot_ref,
            state_key=frozen_opponent_checkpoint_state_key,
            player_id=frozen_opponent_player_id,
            num_simulations=frozen_opponent_num_simulations,
            use_cuda=frozen_opponent_use_cuda,
        )
    )
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
            "records two-seat replay rows with iteration, env_row_id, player_id, decision_index, observation, action mask, action, action_weights, root_value, and shaped training reward components",
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
            "reset_seed_strategy": "run_seed_initializes_env_rng_then_per_reset_row_draws/v0",
            "initial_reset_seed_sample": list(initial_reset_seed_sample),
            "env_max_ticks": None if env_max_ticks is None else int(env_max_ticks),
            "death_mode": death_mode,
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "ruleset_hint": (
                "source-default natural bonus spawning is enabled by default; "
                "disable only for no-bonus ablations"
                if natural_bonus_spawn
                else "no-bonus ablation; not the default Coach ruleset"
            ),
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
            "terminal_outcome_reward_per_step": float(
                terminal_outcome_reward_per_step
            ),
            "bonus_pickup_reward_per_catch": float(bonus_pickup_reward_per_catch),
            "return_target_discount": float(return_target_discount),
            "training_reward_formula": (
                "dense alive/dead helper plus same-step bonus pickup helper plus "
                "sparse terminal outcome scaled by terminal_outcome_reward_per_step "
                "* episode_step_count"
            ),
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
                "legacy flag names; semantics are per-env-row/per-seat no-op "
                "policy skips. After a fresh policy action, a seat can skip later "
                "policy chances by sending NOOP to the env. Skipped no-op ticks do "
                "not create replay rows or reward targets."
            ),
            "observation_noise_std": float(observation_noise_std),
            "trail_render_mode": render_metadata["trail_render_mode"],
            "learning_rate": None if learning_rate is None else float(learning_rate),
            "frozen_opponent": resolved_frozen_opponent_metadata,
            "frozen_opponent_probability": float(frozen_opponent_probability),
            "frozen_opponent_checkpoint_ref": frozen_opponent_checkpoint_ref,
            "frozen_opponent_snapshot_ref": frozen_opponent_snapshot_ref,
            "frozen_opponent_checkpoint_state_key": frozen_opponent_checkpoint_state_key,
            "frozen_opponent_player_id": int(frozen_opponent_player_id),
            "frozen_opponent_num_simulations": int(frozen_opponent_num_simulations),
            "frozen_opponent_use_cuda": bool(frozen_opponent_use_cuda),
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
        "opponent_mix": {
            "schema_id": OPPONENT_MIX_SCHEMA_ID,
            "frozen_opponent": resolved_frozen_opponent_metadata,
            "iteration_edges": [
                item.get("opponent_mix")
                for item in iteration_summaries[-5:]
                if isinstance(item.get("opponent_mix"), Mapping)
            ],
        },
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
            "reward": (
                "training reward per policy decision row: dense survival helper "
                "plus same-step bonus pickup helper plus scaled sparse terminal outcome"
            ),
            "reward_values": {
                "alive_reward": float(alive_reward),
                "dead_reward": float(dead_reward),
                "bonus_pickup_reward_per_catch": float(
                    bonus_pickup_reward_per_catch
                ),
                "terminal_outcome_reward_per_step": float(
                    terminal_outcome_reward_per_step
                ),
                "return_target_discount": float(return_target_discount),
            },
            "return_schema_id": TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_ID,
            "return_schema_hash": TWO_SEAT_TERMINAL_SHAPED_RETURN_SCHEMA_HASH,
            "value_target": (
                "discounted shaped training return grouped by "
                "episode_id/env_row_id/player_id/decision_index when episode "
                "metadata is present"
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
            f"--two-seat-natural-bonus-spawn {str(bool(natural_bonus_spawn)).lower()} "
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
    terminal_outcome_reward_per_step: float,
    bonus_pickup_reward_per_catch: float,
    return_target_discount: float,
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
    frozen_opponent_policy: Any | None = None,
    frozen_opponent_row_mask: np.ndarray | None = None,
    frozen_opponent_probability: float = DEFAULT_FROZEN_OPPONENT_PROBABILITY,
    frozen_opponent_mix_rng: np.random.Generator | None = None,
    frozen_opponent_player_id: int = DEFAULT_FROZEN_OPPONENT_PLAYER_ID,
    frozen_opponent_metadata: Mapping[str, Any] | None = None,
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
    opponent_mix_step_counts: Counter[str] = Counter()
    noop_skip_interval_histogram: Counter[int] = Counter()
    repeat_remaining = np.zeros((env.batch_size, env.player_count), dtype=np.int16)
    frozen_enabled = (
        frozen_opponent_policy is not None and float(frozen_opponent_probability) > 0.0
    )
    if frozen_opponent_mix_rng is None:
        frozen_opponent_mix_rng = _frozen_opponent_mix_rng(seed=iteration)
    if frozen_opponent_row_mask is None:
        active_frozen_opponent_row_mask = _sample_frozen_opponent_rows(
            batch_size=env.batch_size,
            probability=frozen_opponent_probability,
            rng=frozen_opponent_mix_rng,
            enabled=frozen_enabled,
        )
    else:
        active_frozen_opponent_row_mask = np.asarray(
            frozen_opponent_row_mask,
            dtype=bool,
        ).copy()
        if active_frozen_opponent_row_mask.shape != (env.batch_size,):
            raise ValueError("frozen_opponent_row_mask must have shape [B]")
        if not frozen_enabled:
            active_frozen_opponent_row_mask[:] = False
    resolved_frozen_opponent_metadata = dict(
        frozen_opponent_metadata
        or _frozen_opponent_metadata(
            enabled=frozen_enabled,
            probability=frozen_opponent_probability,
            checkpoint_path=None,
            checkpoint_ref=None,
            snapshot_ref=None,
            state_key=None,
            player_id=frozen_opponent_player_id,
            num_simulations=1,
            use_cuda=False,
        )
    )

    def add_elapsed(name: str, started: float) -> None:
        timing_sec[name] += time.perf_counter() - started

    def clear_control_state(row_mask: np.ndarray) -> None:
        if not bool(np.asarray(row_mask, dtype=bool).any()):
            return
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
        active_frozen_opponent_row_mask = _resample_frozen_opponent_rows(
            active_frozen_opponent_row_mask,
            row_mask=needs_reset,
            probability=frozen_opponent_probability,
            rng=frozen_opponent_mix_rng,
            enabled=frozen_enabled,
        )
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
        live_mask = batch.action_mask.any(axis=2)
        current_policy_live = _current_policy_live_mask(
            live_mask,
            frozen_opponent_row_mask=active_frozen_opponent_row_mask,
            frozen_opponent_player_id=frozen_opponent_player_id,
        )
        frozen_slot_mask = _frozen_opponent_slot_mask(
            batch.action_mask,
            frozen_opponent_row_mask=active_frozen_opponent_row_mask,
            frozen_opponent_player_id=frozen_opponent_player_id,
        )
        action_source_by_slot = _action_source_by_slot(
            live_mask=live_mask,
            current_policy_live_mask=current_policy_live,
            frozen_opponent_slot_mask=frozen_slot_mask,
        )
        rollout_kind_by_row = _rollout_kind_by_row(
            batch_size=env.batch_size,
            frozen_opponent_row_mask=active_frozen_opponent_row_mask,
        )
        opponent_mix_step_counts[ROLLOUT_KIND_CURRENT_POLICY_SELFPLAY] += int(
            (~active_frozen_opponent_row_mask & live_mask.any(axis=1)).sum()
        )
        opponent_mix_step_counts[ROLLOUT_KIND_CURRENT_POLICY_VS_FROZEN] += int(
            (active_frozen_opponent_row_mask & live_mask.any(axis=1)).sum()
        )
        mapping = build_policy_row_mapping(
            policy_observation,
            current_policy_live,
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
        needs_policy_decision = repeat_remaining_before <= 0
        fresh_active_indices = np.flatnonzero(needs_policy_decision)
        skipped_active_indices = np.flatnonzero(~needs_policy_decision)
        stochasticity_counts["active_policy_rows"] += active_count
        stochasticity_counts["fresh_policy_decision_rows"] += int(
            fresh_active_indices.size
        )
        stochasticity_counts["policy_noop_skip_rows"] += int(
            skipped_active_indices.size
        )

        selected_actions: list[int] = []
        search_records_by_active: list[dict[str, Any] | None] = [None] * active_count
        batched_search: dict[str, Any] = {
            "ok": True,
            "records": [],
            "reason": None,
        }
        policy_batching_label = "skipped_policy_noop_rows"
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
        selected = np.full(active_count, NOOP_ACTION_ID, dtype=np.int16)
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
        policy_noop_skip_started = np.zeros(active_count, dtype=bool)
        policy_noop_skip_interval = np.zeros(active_count, dtype=np.int16)
        if fresh_active_indices.size:
            fresh_policy_intervals = _sample_policy_action_repeats(
                count=int(fresh_active_indices.size),
                min_repeat=policy_action_repeat_min,
                max_repeat=policy_action_repeat_max,
                extra_probability=policy_action_repeat_extra_probability,
                rng=policy_action_repeat_rng,
            )
            for fresh_row, active_row in enumerate(fresh_active_indices):
                env_row = int(active_env_row_id[active_row])
                player = int(active_player_id[active_row])
                interval_steps = int(fresh_policy_intervals[fresh_row])
                repeat_remaining[env_row, player] = interval_steps
                policy_noop_skip_started[active_row] = True
                policy_noop_skip_interval[active_row] = interval_steps
                noop_skip_interval_histogram[interval_steps] += 1
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
        frozen_opponent_actions = np.full(
            (env.batch_size, env.player_count),
            NOOP_ACTION_ID,
            dtype=np.int16,
        )
        frozen_opponent_sidecar = None
        if bool(frozen_slot_mask.any()):
            started = time.perf_counter()
            frozen_selection = _select_frozen_opponent_actions(
                frozen_opponent_policy,
                legal_action_mask=batch.action_mask,
                opponent_mask=frozen_slot_mask,
                observation=policy_observation,
                decision_index=decision_index,
            )
            add_elapsed("frozen_opponent_action_sec", started)
            if not frozen_selection.get("ok"):
                problems.append(
                    "frozen opponent action selection failed: "
                    f"{frozen_selection.get('exception')}"
                )
                break
            frozen_opponent_actions = np.asarray(
                frozen_selection["actions"],
                dtype=np.int16,
            )
            frozen_opponent_sidecar = frozen_selection.get("sidecar")
            joint_action[frozen_slot_mask] = frozen_opponent_actions[frozen_slot_mask]
            for env_row, player in np.argwhere(frozen_slot_mask):
                action = int(frozen_opponent_actions[int(env_row), int(player)])
                physical_action_counts[action] += 1
                physical_per_player_action_counts[f"player_{int(player)}"][action] += 1

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
            action_source = str(action_source_by_slot[env_row, player])
            if action_source != ACTION_SOURCE_CURRENT_POLICY:
                problems.append(
                    "refusing to write non-current-policy row into learner replay: "
                    f"env_row={env_row}, player={player}, source={action_source}"
                )
                break
            rollout_kind = str(rollout_kind_by_row[env_row])
            sparse_outcome_reward = _sparse_outcome_reward(
                step_batch.reward,
                env_row=env_row,
                player_id=player,
            )
            episode_step_count = _episode_step_count_for_row(
                step_batch.info,
                env_row=env_row,
                fallback=decision_index + 1,
            )
            dense_survival_helper = _survival_reward(
                alive_after,
                env_row=env_row,
                player_id=player,
                alive_reward=alive_reward,
                dead_reward=dead_reward,
            )
            terminal_outcome_reward = _terminal_outcome_reward(
                sparse_outcome_reward,
                done=bool(done_by_row[env_row]),
                episode_step_count=episode_step_count,
                reward_per_step=terminal_outcome_reward_per_step,
            )
            bonus_pickup_count = _bonus_pickup_count_for_row(
                step_batch.info,
                env_row=env_row,
                player_id=player,
            )
            bonus_pickup_reward = (
                float(bonus_pickup_count) * float(bonus_pickup_reward_per_catch)
            )
            training_reward = (
                dense_survival_helper
                + bonus_pickup_reward
                + terminal_outcome_reward
            )
            replay_rows.append(
                {
                    "schema_id": TWO_SEAT_LIGHTZERO_REPLAY_ROW_SCHEMA_ID,
                    "iteration": int(iteration),
                    "iteration_step": int(iteration_step),
                    "episode_id": _episode_id_for_row(batch.info, env_row),
                    "reset_seed": _reset_seed_for_row(batch.info, env_row),
                    "decision_index": int(decision_index),
                    "env_row_id": env_row,
                    "player_id": player,
                    "to_play": player,
                    "learner_controlled": True,
                    "action_source": ACTION_SOURCE_CURRENT_POLICY,
                    "rollout_kind": rollout_kind,
                    "opponent_mix_schema_id": OPPONENT_MIX_SCHEMA_ID,
                    "frozen_opponent_player_id": int(frozen_opponent_player_id),
                    "frozen_opponent_checkpoint_ref": (
                        resolved_frozen_opponent_metadata.get("checkpoint_ref")
                    ),
                    "frozen_opponent_snapshot_ref": (
                        resolved_frozen_opponent_metadata.get("snapshot_ref")
                    ),
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
                    "policy_noop_skip_started": bool(
                        policy_noop_skip_started[policy_row]
                    ),
                    "policy_noop_skip_interval_steps": int(
                        policy_noop_skip_interval[policy_row]
                    ),
                    "policy_noop_skip_count_after_action": int(
                        max(int(policy_noop_skip_interval[policy_row]) - 1, 0)
                    ),
                    "policy_noop_skip_remaining_after_step": int(
                        repeat_remaining_after[policy_row]
                    ),
                    "replay_row_for_fresh_policy_decision": True,
                    "policy_noop_skip_ticks_create_replay_rows": False,
                    "observation_noise_std": float(observation_noise_std),
                    "action_weights": _action_weights(search_record, policy_action),
                    "root_value": _root_value(search_record),
                    "reward": float(training_reward),
                    "dense_survival_helper_reward": float(dense_survival_helper),
                    "bonus_pickup_count": int(bonus_pickup_count),
                    "bonus_pickup_reward": float(bonus_pickup_reward),
                    "bonus_pickup_reward_per_catch": float(
                        bonus_pickup_reward_per_catch
                    ),
                    "sparse_outcome_reward": float(sparse_outcome_reward),
                    "terminal_outcome_reward": float(terminal_outcome_reward),
                    "terminal_outcome_reward_per_step": float(
                        terminal_outcome_reward_per_step
                    ),
                    "episode_step_count": int(episode_step_count),
                    "return_target_discount": float(return_target_discount),
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
            "policy_noop_skip_rows": int(skipped_active_indices.size),
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
            "same_policy_object_for_both_seats": not bool(
                active_frozen_opponent_row_mask.any()
            ),
            "opponent_mix_schema_id": OPPONENT_MIX_SCHEMA_ID,
            "opponent_mix": _opponent_mix_summary(
                frozen_opponent_row_mask=active_frozen_opponent_row_mask,
                frozen_opponent_player_id=frozen_opponent_player_id,
                frozen_opponent_probability=frozen_opponent_probability,
                frozen_opponent_metadata=resolved_frozen_opponent_metadata,
            ),
            "rollout_kind_by_row": rollout_kind_by_row.copy(),
            "frozen_opponent_row_mask": active_frozen_opponent_row_mask.copy(),
            "current_policy_live_mask": current_policy_live.copy(),
            "frozen_opponent_slot_mask": frozen_slot_mask.copy(),
            "action_source_by_slot": action_source_by_slot.copy(),
            "joint_action": joint_action.copy(),
            "policy_selected_actions": policy_selected.copy(),
            "executed_actions": selected.copy(),
            "frozen_opponent_actions": frozen_opponent_actions.copy(),
            "frozen_opponent_policy_sidecar": frozen_opponent_sidecar,
            "policy_noop_skip_actions_are_noop": True,
            "action_noise_noop_probability": float(action_noop_probability),
            "action_noise_noop_count": int(action_noise_mask.sum()),
            "control_stochasticity_schema_id": CONTROL_STOCHASTICITY_SCHEMA_ID,
            "policy_action_repeat_min": int(policy_action_repeat_min),
            "policy_action_repeat_max": int(policy_action_repeat_max),
            "policy_action_repeat_extra_probability": float(
                policy_action_repeat_extra_probability
            ),
            "policy_noop_skip_remaining_before": repeat_remaining_before.copy(),
            "policy_noop_skip_remaining_after": repeat_remaining_after.copy(),
            "policy_noop_skip_started_mask": policy_noop_skip_started.copy(),
            "policy_noop_skip_interval_steps": policy_noop_skip_interval.copy(),
            "observation_noise_std": float(observation_noise_std),
            "reward": step_batch.reward.copy(),
            "done": step_batch.done.copy(),
            "needs_reset": np.asarray(
                step_batch.info.get("needs_reset", step_batch.done),
                dtype=bool,
            ).copy(),
            "reset_seed": _reset_seed_array_from_info(step_batch.info),
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
            active_frozen_opponent_row_mask = _resample_frozen_opponent_rows(
                active_frozen_opponent_row_mask,
                row_mask=needs_reset,
                probability=frozen_opponent_probability,
                rng=frozen_opponent_mix_rng,
                enabled=frozen_enabled,
            )
            record["autoreset_rows"] = reset_batch.info["reset_rows"].copy()
            record["autoreset_reset_seed_sample"] = _reset_seed_sample(
                reset_batch.info,
                row_mask=needs_reset,
            )
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
        "frozen_opponent_row_mask": active_frozen_opponent_row_mask,
        "opponent_mix": _opponent_mix_summary(
            frozen_opponent_row_mask=active_frozen_opponent_row_mask,
            frozen_opponent_player_id=frozen_opponent_player_id,
            frozen_opponent_probability=frozen_opponent_probability,
            frozen_opponent_metadata=resolved_frozen_opponent_metadata,
            step_counts=opponent_mix_step_counts,
        ),
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
            "policy_noop_skip_schedule": (
                "per-env-row/per-seat draws; skipped seats send NOOP without "
                "fresh policy search, replay row, or reward target"
            ),
            "counts": _counter_dict(stochasticity_counts),
            "policy_noop_skip_interval_histogram": _counter_dict(
                noop_skip_interval_histogram
            ),
            "physical_action_counts": _counter_dict(physical_action_counts),
            "physical_action_counts_by_player": {
                player: _counter_dict(counts)
                for player, counts in sorted(physical_per_player_action_counts.items())
            },
            "physical_action_summary": {
                **_action_histogram_summary(
                    physical_action_counts,
                    physical_per_player_action_counts,
                    count_field="executed_action_count",
                ),
                "count_semantics": (
                    "physical/executed env actions; policy no-op skips send NOOP "
                    "here, so this is not the primary training-collapse gate"
                ),
            },
        },
        "timing_sec": {
            key: round(float(timing_sec[key]), 6) for key in sorted(timing_sec)
        },
        "policy_batching_counts": _counter_dict(policy_batching_counts),
        "policy_search_call_count": int(policy_search_call_count),
        "policy_search_row_count": int(policy_search_row_count),
        "visual_stack_dirty_render": visual_stack.dirty_render_stats(),
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
        "training_reward_sum": iteration_summary.get("training_reward_sum"),
        "survival_reward_sum": iteration_summary.get("survival_reward_sum"),
        "dense_survival_helper_reward_sum": iteration_summary.get(
            "dense_survival_helper_reward_sum"
        ),
        "bonus_pickup_reward_sum": iteration_summary.get("bonus_pickup_reward_sum"),
        "bonus_pickup_count": iteration_summary.get("bonus_pickup_count"),
        "sparse_outcome_reward_sum": iteration_summary.get(
            "sparse_outcome_reward_sum"
        ),
        "terminal_outcome_reward_sum": iteration_summary.get(
            "terminal_outcome_reward_sum"
        ),
        "action_counts": iteration_summary.get("action_counts"),
        "action_counts_by_player": iteration_summary.get("action_counts_by_player"),
        "fresh_policy_action_summary": iteration_summary.get(
            "fresh_policy_action_summary"
        ),
        "effective_action_noop_probability": iteration_summary.get(
            "effective_action_noop_probability"
        ),
        "effective_policy_action_repeat_extra_probability": iteration_summary.get(
            "effective_policy_action_repeat_extra_probability"
        ),
        "control_stochasticity": iteration_summary.get("control_stochasticity"),
        "opponent_mix": iteration_summary.get("opponent_mix"),
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
    dense_helpers = np.asarray(
        [row.get("dense_survival_helper_reward", row["reward"]) for row in replay_rows],
        dtype=np.float32,
    )
    sparse_outcomes = np.asarray(
        [row.get("sparse_outcome_reward", 0.0) for row in replay_rows],
        dtype=np.float32,
    )
    terminal_outcomes = np.asarray(
        [row.get("terminal_outcome_reward", 0.0) for row in replay_rows],
        dtype=np.float32,
    )
    bonus_pickups = np.asarray(
        [row.get("bonus_pickup_reward", 0.0) for row in replay_rows],
        dtype=np.float32,
    )
    bonus_pickup_counts = np.asarray(
        [row.get("bonus_pickup_count", 0) for row in replay_rows],
        dtype=np.int32,
    )
    rollout_kind_counts = Counter(
        str(row.get("rollout_kind", ROLLOUT_KIND_CURRENT_POLICY_SELFPLAY))
        for row in replay_rows
    )
    action_source_counts = Counter(
        str(row.get("action_source", ACTION_SOURCE_CURRENT_POLICY))
        for row in replay_rows
    )
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
            "training_reward_sum": float(rewards.sum()) if rewards.size else 0.0,
            "survival_reward_sum": float(dense_helpers.sum())
            if dense_helpers.size
            else 0.0,
            "dense_survival_helper_reward_sum": float(dense_helpers.sum())
            if dense_helpers.size
            else 0.0,
            "bonus_pickup_reward_sum": float(bonus_pickups.sum())
            if bonus_pickups.size
            else 0.0,
            "bonus_pickup_count": int(bonus_pickup_counts.sum())
            if bonus_pickup_counts.size
            else 0,
            "sparse_outcome_reward_sum": float(sparse_outcomes.sum())
            if sparse_outcomes.size
            else 0.0,
            "terminal_outcome_reward_sum": float(terminal_outcomes.sum())
            if terminal_outcomes.size
            else 0.0,
            "replay": {
                "status": "ok" if replay_rows else "empty",
                "row_count": int(len(replay_rows)),
                "rollout_kind_counts": _counter_dict(rollout_kind_counts),
                "action_source_counts": _counter_dict(action_source_counts),
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
            "fresh_policy_action_summary": _action_histogram_summary(
                action_counts,
                per_player_action_counts,
            ),
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
    dirty_stats = [
        item.get("visual_stack_dirty_render")
        for item in iteration_summaries
        if isinstance(item.get("visual_stack_dirty_render"), Mapping)
    ]
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
        "visual_stack_dirty_render": _summarize_dirty_render_stats(dirty_stats),
        "timing_sec": timing_summary,
    }


def _summarize_dirty_render_stats(stats: list[Any]) -> dict[str, Any]:
    if not stats:
        return {}
    attempts = [int(item.get("attempts") or 0) for item in stats if isinstance(item, Mapping)]
    hits = [int(item.get("hits") or 0) for item in stats if isinstance(item, Mapping)]
    fallbacks = [int(item.get("fallbacks") or 0) for item in stats if isinstance(item, Mapping)]
    dirty_blocks = [
        int(item.get("dirty_blocks_total") or 0)
        for item in stats
        if isinstance(item, Mapping)
    ]
    total_attempts = int(sum(attempts))
    total_hits = int(sum(hits))
    return {
        "enabled": any(bool(item.get("enabled")) for item in stats if isinstance(item, Mapping)),
        "attempts": total_attempts,
        "hits": total_hits,
        "fallbacks": int(sum(fallbacks)),
        "dirty_blocks_total": int(sum(dirty_blocks)),
        "hit_rate": (
            round(float(total_hits) / float(total_attempts), 6)
            if total_attempts
            else None
        ),
        "last": stats[-1] if stats else {},
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
    bad_rows = [
        index
        for index, row in enumerate(rows)
        if row.get("action_source") != ACTION_SOURCE_CURRENT_POLICY
        or row.get("learner_controlled") is not True
    ]
    if bad_rows:
        raise ValueError(
            "learner replay rows must all be current-policy controlled; "
            f"bad row indices: {bad_rows[:8]}"
        )
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
        "return_target_discount": _common_row_float(
            rows,
            key="return_target_discount",
            default=DEFAULT_RETURN_TARGET_DISCOUNT,
        ),
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
        "return_target_discount": float(sample["return_target_discount"]),
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


def _common_row_float(
    rows: list[dict[str, Any]],
    *,
    key: str,
    default: float,
) -> float:
    for row in rows:
        if key not in row:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            break
    return float(default)


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
        "return_target_discount": sample.get("return_target_discount"),
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


def _sparse_outcome_reward(
    reward: Any,
    *,
    env_row: int,
    player_id: int,
) -> float:
    rewards = np.asarray(reward, dtype=np.float32)
    if rewards.ndim != 2:
        raise ValueError("step reward must have shape [B,P]")
    return float(rewards[int(env_row), int(player_id)])


def _bonus_pickup_count_for_row(
    info: Mapping[str, Any],
    *,
    env_row: int,
    player_id: int,
) -> int:
    counts = info.get("bonus_catch_count_step")
    if counts is None:
        return 0
    array = np.asarray(counts, dtype=np.int64)
    if array.ndim != 2:
        return 0
    if array.shape[0] <= int(env_row) or array.shape[1] <= int(player_id):
        return 0
    return max(int(array[int(env_row), int(player_id)]), 0)


def _episode_step_count_for_row(
    info: Mapping[str, Any],
    *,
    env_row: int,
    fallback: int,
) -> int:
    step_index = info.get("step_index")
    if step_index is None:
        return max(int(fallback), 1)
    values = np.asarray(step_index, dtype=np.int64)
    if values.ndim != 1 or values.shape[0] <= int(env_row):
        return max(int(fallback), 1)
    return max(int(values[int(env_row)]) + 1, 1)


def _terminal_outcome_reward(
    sparse_outcome_reward: float,
    *,
    done: bool,
    episode_step_count: int,
    reward_per_step: float,
) -> float:
    if not bool(done):
        return 0.0
    return (
        float(sparse_outcome_reward)
        * float(reward_per_step)
        * float(max(int(episode_step_count), 1))
    )


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


def _reset_seed_for_row(info: Mapping[str, Any], env_row: int) -> int:
    reset_seeds = info.get("reset_seed")
    if reset_seeds is None:
        return -1
    array = np.asarray(reset_seeds, dtype=np.uint64)
    if array.ndim != 1 or array.shape[0] <= int(env_row):
        return -1
    return int(array[int(env_row)])


def _reset_seed_array_from_info(info: Mapping[str, Any]) -> np.ndarray | None:
    reset_seeds = info.get("reset_seed")
    if reset_seeds is None:
        return None
    array = np.asarray(reset_seeds, dtype=np.uint64)
    if array.ndim != 1:
        return None
    return array.copy()


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


def _action_histogram_summary(
    action_counts: Counter[int],
    per_player_action_counts: dict[str, Counter[int]],
    *,
    collapse_threshold: float = 0.95,
    count_field: str = "decision_count",
) -> dict[str, Any]:
    threshold = float(collapse_threshold)
    top_action_fraction_by_player: dict[str, float | None] = {}
    action_entropy_by_player: dict[str, float | None] = {}
    action_collapse_players: list[str] = []
    for player, counts in sorted(per_player_action_counts.items()):
        player_total = int(sum(counts.values()))
        if player_total < 1:
            top_action_fraction_by_player[player] = None
            action_entropy_by_player[player] = None
            continue
        probabilities = np.asarray(
            [
                float(count) / float(player_total)
                for count in counts.values()
                if int(count) > 0
            ],
            dtype=np.float64,
        )
        top_fraction = float(probabilities.max()) if probabilities.size else 0.0
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        top_action_fraction_by_player[player] = top_fraction
        action_entropy_by_player[player] = entropy
        if top_fraction >= threshold:
            action_collapse_players.append(player)
    return {
        count_field: int(sum(action_counts.values())),
        "action_counts_by_player": {
            player: _counter_dict(counts)
            for player, counts in sorted(per_player_action_counts.items())
        },
        "top_action_fraction_by_player": top_action_fraction_by_player,
        "action_entropy_by_player": action_entropy_by_player,
        "collapse_threshold": threshold,
        "action_collapse_warning": bool(action_collapse_players),
        "action_collapse_players": action_collapse_players,
    }


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
        "--terminal-outcome-reward-per-step",
        type=float,
        default=DEFAULT_TERMINAL_OUTCOME_REWARD_PER_STEP,
        help=(
            "Scale terminal +1/-1 sparse outcome by this value times episode steps."
        ),
    )
    parser.add_argument(
        "--bonus-pickup-reward-per-catch",
        type=float,
        default=DEFAULT_BONUS_PICKUP_REWARD_PER_CATCH,
        help=(
            "Immediate reward on the exact policy step where that player catches "
            "a bonus. Progress reports also sum this for logging."
        ),
    )
    parser.add_argument(
        "--return-target-discount",
        type=float,
        default=DEFAULT_RETURN_TARGET_DISCOUNT,
        help="Discount used for shaped return targets built from replay rows.",
    )
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
        help=(
            "Legacy name. Minimum macro length for each row/seat: one fresh "
            "policy action, then optional skipped policy chances that send NOOP."
        ),
    )
    parser.add_argument(
        "--policy-action-repeat-max",
        type=int,
        default=DEFAULT_POLICY_ACTION_REPEAT_MAX,
        help=(
            "Legacy name. Maximum macro length for each row/seat: one fresh "
            "policy action, then optional skipped policy chances that send NOOP."
        ),
    )
    parser.add_argument(
        "--policy-action-repeat-extra-probability",
        type=float,
        default=DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
        help=(
            "Legacy name. Chance to add one more skipped policy chance that "
            "sends NOOP, repeated until max."
        ),
    )
    parser.add_argument(
        "--policy-action-repeat-warmup-iterations",
        type=int,
        default=DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS,
        help=(
            "Linearly ramp policy no-op skip extra probability from 0 to the "
            "requested value."
        ),
    )
    parser.add_argument(
        "--observation-noise-std",
        type=float,
        default=DEFAULT_OBSERVATION_NOISE_STD,
        help="Gaussian noise added to policy visual inputs and replay frames, clipped to [0, 1].",
    )
    parser.add_argument(
        "--trail-render-mode",
        choices=STACK_RENDER_MODE_ORDER,
        default=STACK_RENDER_MODE_DEFAULT,
        help=(
            "Visual stack renderer. browser_lines is the reference RGB-to-gray "
            "path; fast_gray64_direct is the approximate speed path."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional LightZero policy learning_rate override. Omit to use the stock config default.",
    )
    parser.add_argument(
        "--frozen-opponent-probability",
        type=float,
        default=DEFAULT_FROZEN_OPPONENT_PROBABILITY,
        help="Fraction of env rows that use one frozen checkpoint opponent seat.",
    )
    parser.add_argument("--frozen-opponent-checkpoint-path", default=None)
    parser.add_argument("--frozen-opponent-checkpoint-ref", default=None)
    parser.add_argument("--frozen-opponent-snapshot-ref", default=None)
    parser.add_argument("--frozen-opponent-checkpoint-state-key", default=None)
    parser.add_argument(
        "--frozen-opponent-player-id",
        type=int,
        default=DEFAULT_FROZEN_OPPONENT_PLAYER_ID,
    )
    parser.add_argument("--frozen-opponent-num-simulations", type=int, default=None)
    parser.add_argument(
        "--frozen-opponent-use-cuda",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--death-mode",
        choices=tuple(vector_runtime.DEATH_MODES),
        default=DEFAULT_DEATH_MODE,
        help="Use profile_no_death only for optimizer long-survival profiling.",
    )
    parser.add_argument(
        "--natural-bonus-spawn",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NATURAL_BONUS_SPAWN,
        help="Enable source-default natural bonus timers, type draws, and position draws.",
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
        terminal_outcome_reward_per_step=args.terminal_outcome_reward_per_step,
        bonus_pickup_reward_per_catch=args.bonus_pickup_reward_per_catch,
        return_target_discount=args.return_target_discount,
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
        learning_rate=args.learning_rate,
        frozen_opponent_probability=args.frozen_opponent_probability,
        frozen_opponent_checkpoint_path=args.frozen_opponent_checkpoint_path,
        frozen_opponent_checkpoint_ref=args.frozen_opponent_checkpoint_ref,
        frozen_opponent_snapshot_ref=args.frozen_opponent_snapshot_ref,
        frozen_opponent_checkpoint_state_key=args.frozen_opponent_checkpoint_state_key,
        frozen_opponent_player_id=args.frozen_opponent_player_id,
        frozen_opponent_num_simulations=args.frozen_opponent_num_simulations,
        frozen_opponent_use_cuda=args.frozen_opponent_use_cuda,
        death_mode=args.death_mode,
        natural_bonus_spawn=args.natural_bonus_spawn,
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
    "DEFAULT_ALIVE_REWARD",
    "DEFAULT_DEAD_REWARD",
    "DEFAULT_FROZEN_OPPONENT_PLAYER_ID",
    "DEFAULT_FROZEN_OPPONENT_PROBABILITY",
    "DEFAULT_NATURAL_BONUS_SPAWN",
    "DEFAULT_OBSERVATION_NOISE_STD",
    "DEFAULT_RETURN_TARGET_DISCOUNT",
    "DEFAULT_TERMINAL_OUTCOME_REWARD_PER_STEP",
    "LEARN_BATCH_BLOCKER",
    "REPLAY_SCOPE_ACCUMULATED",
    "REPLAY_SCOPE_CHOICES",
    "REPLAY_SCOPE_CURRENT_ITERATION",
    "TWO_SEAT_LIGHTZERO_REPLAY_ROW_SCHEMA_ID",
    "TWO_SEAT_LIGHTZERO_TRAIN_SMOKE_SCHEMA_ID",
    "compact_curvytron_two_seat_lightzero_train_smoke_summary",
    "run_curvytron_two_seat_lightzero_train_smoke",
]
