"""Modal train-attempt artifact for CurvyTron stacked debug-visual survival MuZero.

This is the canonical attempt-shaped launch point for the first CurvyTron
debug-fidelity visual survival trainer. It delegates the implementation to the
bounded sibling trainer module, which validates the configured surface and calls
``lzero.entry.train_muzero`` only when ``mode="train"``.

It is a trainer plumbing artifact, not a policy-quality claim:

- env: stacked debug visual survival CurvyTron LightZero wrapper
- observation: wrapper-owned ``float32[4,64,64]`` debug occupancy stack
- reward: survival-time only, with no terminal outcome shaping
- checkpoints: saved frequently and mirrored into the ``curvyzero-runs`` Volume
- action observability: env-step JSONL summarized into action histograms

Dry config check:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train_attempt --mode dry

Bounded train launch:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train_attempt --mode train
"""

from __future__ import annotations

import json
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    CHEAP_GPU_RESOURCE,
    BACKGROUND_EVAL_LAUNCH_HOOK,
    COMPUTE_CHOICES,
    DEFAULT_ATTEMPT_ID,
    DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    DEFAULT_BACKGROUND_EVAL_COMPUTE,
    DEFAULT_BACKGROUND_EVAL_ENABLED,
    DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
    DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
    DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
    DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
    DEFAULT_BACKGROUND_GIF_ENABLED,
    DEFAULT_BACKGROUND_GIF_FPS,
    DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
    DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    DEFAULT_BACKGROUND_GIF_SCALE,
    DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTOR_ENV_NUM,
    DEFAULT_COMPUTE,
    DEFAULT_CONTROL_NOISE_PROFILE_ID,
    DEFAULT_DECISION_MS,
    DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY,
    DEFAULT_ENV_MANAGER_TYPE,
    DEFAULT_ENV_TELEMETRY_STRIDE,
    DEFAULT_ENV_VARIANT,
    DEFAULT_EVALUATOR_ENV_NUM,
    DEFAULT_MAX_ENV_STEP,
    DEFAULT_MAX_TRAIN_ITER,
    DEFAULT_MODE,
    DEFAULT_N_EPISODE,
    DEFAULT_N_EVALUATOR_EPISODE,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_RUN_ID,
    DEFAULT_SAVE_CKPT_AFTER_ITER,
    DEFAULT_SEED,
    DEFAULT_SOURCE_MAX_STEPS,
    DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    DEFAULT_OPPONENT_POLICY_KIND,
    DEFAULT_REWARD_VARIANT,
    RUNS_MOUNT,
    TASK_ID,
    _run_visual_survival_train,
    image,
    runs_volume,
)


APP_NAME = "curvyzero-lightzero-curvytron-visual-survival-train-attempt"
app = modal.App(APP_NAME)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=20 * 60, cpu=2.0)
def lightzero_curvytron_visual_survival_train_attempt_cpu(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    decision_ms: float = DEFAULT_DECISION_MS,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE,
    n_episode: int = DEFAULT_N_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_ckpt_after_iter: int = DEFAULT_SAVE_CKPT_AFTER_ITER,
    stop_after_learner_train_calls: int = DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    ego_action_straight_override_probability: float = (
        DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
    ),
    control_noise_profile_id: str = DEFAULT_CONTROL_NOISE_PROFILE_ID,
    disable_death_for_profile: bool = DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    env_telemetry_stride: int = DEFAULT_ENV_TELEMETRY_STRIDE,
    env_manager_type: str = DEFAULT_ENV_MANAGER_TYPE,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    background_eval_enabled: bool = DEFAULT_BACKGROUND_EVAL_ENABLED,
    background_eval_compute: str = DEFAULT_BACKGROUND_EVAL_COMPUTE,
    background_eval_id_prefix: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    background_eval_seed_count: int = DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
    background_eval_seed_rng_seed: int | None = DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
    background_eval_max_steps: int = DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
    background_eval_step_detail_limit: int | None = DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
    background_eval_num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    background_eval_batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    background_gif_enabled: bool = DEFAULT_BACKGROUND_GIF_ENABLED,
    background_gif_seed_offset: int = DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    background_gif_max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    background_gif_frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    background_gif_fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    background_gif_scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    background_gif_frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
) -> dict[str, Any]:
    return _run_visual_survival_train(
        mode=mode,
        compute="cpu",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        source_max_steps=source_max_steps,
        decision_ms=decision_ms,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        n_episode=n_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        save_ckpt_after_iter=save_ckpt_after_iter,
        stop_after_learner_train_calls=stop_after_learner_train_calls,
        env_variant=env_variant,
        reward_variant=reward_variant,
        ego_action_straight_override_probability=ego_action_straight_override_probability,
        control_noise_profile_id=control_noise_profile_id,
        disable_death_for_profile=disable_death_for_profile,
        env_telemetry_stride=env_telemetry_stride,
        env_manager_type=env_manager_type,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_report_ref=None,
        opponent_checkpoint_state_key=None,
        background_eval_enabled=background_eval_enabled,
        background_eval_launch_kind=BACKGROUND_EVAL_LAUNCH_HOOK,
        background_eval_compute=background_eval_compute,
        background_eval_id_prefix=background_eval_id_prefix,
        background_eval_seed_count=background_eval_seed_count,
        background_eval_seed_rng_seed=background_eval_seed_rng_seed,
        background_eval_max_steps=background_eval_max_steps,
        background_eval_step_detail_limit=background_eval_step_detail_limit,
        background_eval_num_simulations=background_eval_num_simulations,
        background_eval_batch_size=background_eval_batch_size,
        background_gif_enabled=background_gif_enabled,
        background_gif_seed_offset=background_gif_seed_offset,
        background_gif_max_steps=background_gif_max_steps,
        background_gif_frame_stride=background_gif_frame_stride,
        background_gif_fps=background_gif_fps,
        background_gif_scale=background_gif_scale,
        background_gif_frame_size=background_gif_frame_size,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=8 * 60 * 60,
    cpu=8.0,
    memory=32768,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_train_attempt_gpu(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    decision_ms: float = DEFAULT_DECISION_MS,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE,
    n_episode: int = DEFAULT_N_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_ckpt_after_iter: int = DEFAULT_SAVE_CKPT_AFTER_ITER,
    stop_after_learner_train_calls: int = DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    ego_action_straight_override_probability: float = (
        DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
    ),
    control_noise_profile_id: str = DEFAULT_CONTROL_NOISE_PROFILE_ID,
    disable_death_for_profile: bool = DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    env_telemetry_stride: int = DEFAULT_ENV_TELEMETRY_STRIDE,
    env_manager_type: str = DEFAULT_ENV_MANAGER_TYPE,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    background_eval_enabled: bool = DEFAULT_BACKGROUND_EVAL_ENABLED,
    background_eval_compute: str = DEFAULT_BACKGROUND_EVAL_COMPUTE,
    background_eval_id_prefix: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    background_eval_seed_count: int = DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
    background_eval_seed_rng_seed: int | None = DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
    background_eval_max_steps: int = DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
    background_eval_step_detail_limit: int | None = DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
    background_eval_num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    background_eval_batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    background_gif_enabled: bool = DEFAULT_BACKGROUND_GIF_ENABLED,
    background_gif_seed_offset: int = DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    background_gif_max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    background_gif_frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    background_gif_fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    background_gif_scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    background_gif_frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
) -> dict[str, Any]:
    return _run_visual_survival_train(
        mode=mode,
        compute="gpu-l4-t4",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        source_max_steps=source_max_steps,
        decision_ms=decision_ms,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        n_episode=n_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        save_ckpt_after_iter=save_ckpt_after_iter,
        stop_after_learner_train_calls=stop_after_learner_train_calls,
        env_variant=env_variant,
        reward_variant=reward_variant,
        ego_action_straight_override_probability=ego_action_straight_override_probability,
        control_noise_profile_id=control_noise_profile_id,
        disable_death_for_profile=disable_death_for_profile,
        env_telemetry_stride=env_telemetry_stride,
        env_manager_type=env_manager_type,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_report_ref=None,
        opponent_checkpoint_state_key=None,
        background_eval_enabled=background_eval_enabled,
        background_eval_launch_kind=BACKGROUND_EVAL_LAUNCH_HOOK,
        background_eval_compute=background_eval_compute,
        background_eval_id_prefix=background_eval_id_prefix,
        background_eval_seed_count=background_eval_seed_count,
        background_eval_seed_rng_seed=background_eval_seed_rng_seed,
        background_eval_max_steps=background_eval_max_steps,
        background_eval_step_detail_limit=background_eval_step_detail_limit,
        background_eval_num_simulations=background_eval_num_simulations,
        background_eval_batch_size=background_eval_batch_size,
        background_gif_enabled=background_gif_enabled,
        background_gif_seed_offset=background_gif_seed_offset,
        background_gif_max_steps=background_gif_max_steps,
        background_gif_frame_stride=background_gif_frame_stride,
        background_gif_fps=background_gif_fps,
        background_gif_scale=background_gif_scale,
        background_gif_frame_size=background_gif_frame_size,
    )


@app.local_entrypoint()
def main(
    mode: str = DEFAULT_MODE,
    compute: str = DEFAULT_COMPUTE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    decision_ms: float = DEFAULT_DECISION_MS,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE,
    n_episode: int = DEFAULT_N_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_ckpt_after_iter: int = DEFAULT_SAVE_CKPT_AFTER_ITER,
    stop_after_learner_train_calls: int = DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    ego_action_straight_override_probability: float = (
        DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
    ),
    control_noise_profile_id: str = DEFAULT_CONTROL_NOISE_PROFILE_ID,
    disable_death_for_profile: bool = DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    env_telemetry_stride: int = DEFAULT_ENV_TELEMETRY_STRIDE,
    env_manager_type: str = DEFAULT_ENV_MANAGER_TYPE,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    background_eval_enabled: bool = DEFAULT_BACKGROUND_EVAL_ENABLED,
    background_eval_compute: str = DEFAULT_BACKGROUND_EVAL_COMPUTE,
    background_eval_id_prefix: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    background_eval_seed_count: int = DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
    background_eval_seed_rng_seed: int | None = DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
    background_eval_max_steps: int = DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
    background_eval_step_detail_limit: int | None = DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
    background_eval_num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    background_eval_batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    background_gif_enabled: bool = DEFAULT_BACKGROUND_GIF_ENABLED,
    background_gif_seed_offset: int = DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    background_gif_max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    background_gif_frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    background_gif_fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    background_gif_scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    background_gif_frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
    wait_for_train: bool = False,
) -> None:
    if compute == "cpu":
        train_fn = lightzero_curvytron_visual_survival_train_attempt_cpu
    elif compute == "gpu-l4-t4":
        train_fn = lightzero_curvytron_visual_survival_train_attempt_gpu
    else:
        raise ValueError(f"unknown compute {compute!r}; expected one of {COMPUTE_CHOICES!r}")

    kwargs = {
        "mode": mode,
        "seed": seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "source_max_steps": source_max_steps,
        "decision_ms": decision_ms,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "n_evaluator_episode": n_evaluator_episode,
        "n_episode": n_episode,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "save_ckpt_after_iter": save_ckpt_after_iter,
        "stop_after_learner_train_calls": stop_after_learner_train_calls,
        "env_variant": env_variant,
        "reward_variant": reward_variant,
        "ego_action_straight_override_probability": ego_action_straight_override_probability,
        "control_noise_profile_id": control_noise_profile_id,
        "disable_death_for_profile": disable_death_for_profile,
        "env_telemetry_stride": env_telemetry_stride,
        "env_manager_type": env_manager_type,
        "opponent_policy_kind": opponent_policy_kind,
        "background_eval_enabled": background_eval_enabled,
        "background_eval_compute": background_eval_compute,
        "background_eval_id_prefix": background_eval_id_prefix,
        "background_eval_seed_count": background_eval_seed_count,
        "background_eval_seed_rng_seed": background_eval_seed_rng_seed,
        "background_eval_max_steps": background_eval_max_steps,
        "background_eval_step_detail_limit": background_eval_step_detail_limit,
        "background_eval_num_simulations": background_eval_num_simulations,
        "background_eval_batch_size": background_eval_batch_size,
        "background_gif_enabled": background_gif_enabled,
        "background_gif_seed_offset": background_gif_seed_offset,
        "background_gif_max_steps": background_gif_max_steps,
        "background_gif_frame_stride": background_gif_frame_stride,
        "background_gif_fps": background_gif_fps,
        "background_gif_scale": background_gif_scale,
        "background_gif_frame_size": background_gif_frame_size,
    }
    if mode == "train" and not wait_for_train:
        call = train_fn.spawn(**kwargs)
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print(
            json.dumps(
                {
                    "schema_id": (
                        "curvyzero_lightzero_curvytron_visual_survival_attempt_"
                        "background_launch/v0"
                    ),
                    "status": "spawned",
                    "app_name": APP_NAME,
                    "mode": mode,
                    "compute": compute,
                    "seed": seed,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "function_call_id": call_id,
                    "summary_ref": (
                        runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "summary.json"
                    ).as_posix(),
                    "action_observability_ref": (
                        runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
                        / "action_observability.json"
                    ).as_posix(),
                    "command": kwargs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = train_fn.remote(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
