"""Bounded CurvyTron stacked debug-visual survival profile.

This is an Optimizer artifact, not a trainer and not a learning claim. It times
the current wrapper-stacked debug visual survival surface and, when LightZero is
installed, attempts one tiny MuZero eval/search path against ``float32[4,64,64]``.
It never calls ``train_muzero`` and never steps an optimizer.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import json
import time
import traceback
from dataclasses import asdict, is_dataclass
from importlib import metadata
from typing import Any, Mapping

import numpy as np

from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_LABEL,
    DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
    DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
    DEBUG_OCCUPANCY_GRAY64_SHAPE,
    DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
    DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env import (
    LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID,
    LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
    LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_IMPORT_NAMES,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv,
    STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER,
    STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH,
    STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
)
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    SURVIVAL_TIME_REWARD_SCHEMA_HASH,
    SURVIVAL_TIME_REWARD_SCHEMA_ID,
)


PROFILE_SCHEMA_ID = "curvyzero_stacked_debug_visual_survival_profile/v0"
LIGHTZERO_VERSION = "0.2.0"
DEFAULT_SURVIVAL_TARGET_DISCOUNT = 0.997
SURVIVAL_TARGET_METADATA_KEYS = (
    "iteration_batch",
    "env_row_id_batch",
    "player_id_batch",
    "decision_index_batch",
)


class _Timer:
    def __init__(self) -> None:
        self.timing_sec: dict[str, float] = {}

    @contextlib.contextmanager
    def time(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timing_sec[name] = self.timing_sec.get(name, 0.0) + (
                time.perf_counter() - started
            )

    def rounded(self) -> dict[str, float]:
        return {key: round(value, 6) for key, value in sorted(self.timing_sec.items())}


def run_curvyzero_stacked_debug_visual_survival_profile(
    *,
    seed: int = 0,
    steps: int = 4,
    num_simulations: int = 2,
    require_installed_lightzero: bool = False,
    attempt_installed_lightzero: bool = True,
) -> dict[str, Any]:
    """Run a small profile over the stacked debug visual survival boundary."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")

    timer = _Timer()
    started = time.perf_counter()
    problems: list[str] = []
    policy_context: dict[str, Any] = {
        "status": "not_requested" if not attempt_installed_lightzero else "not_run",
    }
    policy = None

    if attempt_installed_lightzero:
        with timer.time("lightzero_policy_setup_sec"):
            policy_context = _build_lightzero_policy(
                seed=seed,
                num_simulations=num_simulations,
                require_installed_lightzero=require_installed_lightzero,
            )
            policy = policy_context.get("policy")
        if require_installed_lightzero and policy is None:
            problems.append("installed LightZero policy/search setup did not complete")

    with timer.time("env_create_sec"):
        env = CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv(
            {"seed": seed, "source_max_steps": max(steps + 1, 2)}
        )
    with timer.time("reset_render_stack_sec"):
        observation = env.reset(seed=seed)

    replay_rows: list[dict[str, Any]] = []
    step_summaries: list[dict[str, Any]] = []
    policy_search_records: list[dict[str, Any]] = []
    current_observation = observation
    policy_search_ok_count = 0

    for step_index in range(steps):
        action = 1
        search_record: dict[str, Any] = {
            "step_index": step_index,
            "status": "not_run",
            "reason": "LightZero policy unavailable; fixed straight action used",
        }
        if policy is not None:
            with timer.time("policy_search_sec"):
                search_record = _policy_eval_action(
                    policy,
                    current_observation,
                    step_index=step_index,
                )
            if search_record.get("ok"):
                policy_search_ok_count += 1
                action = int(search_record["action"])
            else:
                problems.append(
                    f"policy/search failed at step {step_index}: "
                    f"{search_record.get('reason') or search_record.get('exception', {})}"
                )
        policy_search_records.append(search_record)

        previous_observation = current_observation
        with timer.time("env_step_render_stack_sec"):
            timestep = env.step(action)
        current_observation = timestep.obs
        with timer.time("replay_row_build_sec"):
            row = _build_replay_row(
                step_index=step_index,
                previous_observation=previous_observation,
                timestep=timestep,
                action=action,
                search_record=search_record,
            )
            replay_rows.append(row)
        step_summaries.append(_step_summary(row))
        if bool(row["done"]):
            break

    with timer.time("replay_sample_batch_sec"):
        sample = _sample_replay_batch(replay_rows)

    learner_forward_loss = {
        "status": "not_run",
        "reason": "LightZero policy unavailable",
    }
    if policy is not None and sample["ok"]:
        with timer.time("learner_forward_loss_sec"):
            learner_forward_loss = _learn_mode_forward_loss(policy, sample)
        if not learner_forward_loss.get("ok"):
            problems.append(
                "LightZero learn_mode forward/loss profile failed: "
                f"{learner_forward_loss.get('reason') or learner_forward_loss.get('exception', {})}"
            )

    policy_search_reached = policy_search_ok_count > 0
    if require_installed_lightzero and not policy_search_reached:
        problems.append("required installed LightZero policy/search was not reached")

    elapsed = time.perf_counter() - started
    result = {
        "ok": not problems,
        "schema": PROFILE_SCHEMA_ID,
        "mode": "bounded_debug_visual_collect_search_replay_sample_forward_loss_profile",
        "call_policy": (
            "does_not_call_train_muzero; does_not_step_optimizer; "
            "does_not_claim_learning_or_source_visual_fidelity"
        ),
        "called_train_muzero": False,
        "trainer_entrypoint": "lzero.entry.train_muzero",
        "trainer_claim": "none",
        "quality_claim": "none",
        "problems": problems,
        "inputs": {
            "seed": int(seed),
            "steps_requested": int(steps),
            "num_simulations": int(num_simulations),
            "require_installed_lightzero": bool(require_installed_lightzero),
            "attempt_installed_lightzero": bool(attempt_installed_lightzero),
        },
        "surface": _surface_summary(),
        "packages": {
            "LightZero": _version_or_missing("LightZero", "lightzero"),
            "DI-engine": _version_or_missing("DI-engine", "ding"),
            "torch": _version_or_missing("torch"),
            "gym": _version_or_missing("gym"),
            "numpy": _version_or_missing("numpy"),
        },
        "timed_components": {
            "reset": True,
            "env_step": True,
            "render": True,
            "stack": True,
            "policy_search": policy_search_reached,
            "replay": bool(replay_rows),
            "sample": bool(sample.get("ok")),
            "learner_forward_loss": bool(learner_forward_loss.get("ok")),
            "optimizer_step": False,
            "train_muzero": False,
            "checkpoint": False,
            "eval": False,
        },
        "timing_sec": timer.rounded(),
        "elapsed_sec": round(elapsed, 6),
        "throughput": {
            "rows_collected": len(replay_rows),
            "rows_per_sec_wall": round(len(replay_rows) / elapsed, 6) if elapsed else None,
        },
        "lightzero_policy": _strip_runtime_object(policy_context),
        "policy_search": {
            "status": "ok" if policy_search_reached else "not_reached",
            "ok_count": policy_search_ok_count,
            "records": [_strip_large_arrays(record) for record in policy_search_records],
        },
        "replay": {
            "status": "ok" if replay_rows else "empty",
            "row_count": len(replay_rows),
            "row_schema": "debug_visual_mu_zero_profile_row/v0",
            "rows": [_strip_large_arrays(row) for row in replay_rows],
            "sample": _strip_large_arrays(sample),
        },
        "learner_forward_loss": _strip_large_arrays(learner_forward_loss),
        "steps": step_summaries,
    }
    return _to_plain(result)


def run_stacked_debug_visual_survival_profile(
    *,
    seed: int = 0,
    steps: int = 4,
    num_simulations: int = 2,
    batch_size: int | None = None,
    require_installed_lightzero: bool = False,
) -> dict[str, Any]:
    """Compatibility entrypoint used by the Modal wrapper.

    ``batch_size`` is accepted for CLI symmetry with trainer-like profiles, but
    this bounded profile samples exactly the rows it collects.
    """

    del batch_size
    return run_curvyzero_stacked_debug_visual_survival_profile(
        seed=seed,
        steps=steps,
        num_simulations=num_simulations,
        require_installed_lightzero=require_installed_lightzero,
        attempt_installed_lightzero=True,
    )


def compact_stacked_debug_visual_survival_profile_summary(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the small log-friendly profile surface while preserving full results."""

    surface = result.get("surface", {})
    policy_search = result.get("policy_search", {})
    replay = result.get("replay", {})
    learner_forward_loss = result.get("learner_forward_loss", {})
    lightzero_policy = result.get("lightzero_policy", {})
    return _to_plain(
        {
            "ok": result.get("ok"),
            "schema": result.get("schema"),
            "mode": result.get("mode"),
            "call_policy": result.get("call_policy"),
            "problems": result.get("problems", []),
            "inputs": result.get("inputs", {}),
            "surface": {
                "surface": surface.get("surface"),
                "env_id": surface.get("env_id"),
                "env_type": surface.get("env_type"),
                "observation_shape": surface.get("observation_shape"),
                "dtype": surface.get("dtype"),
                "uses_ale": surface.get("uses_ale"),
                "source_fidelity_claim": surface.get("source_fidelity_claim"),
            },
            "timed_components": result.get("timed_components", {}),
            "timing_sec": result.get("timing_sec", {}),
            "elapsed_sec": result.get("elapsed_sec"),
            "throughput": result.get("throughput", {}),
            "lightzero_policy_status": lightzero_policy.get("status"),
            "policy_search": {
                "status": policy_search.get("status"),
                "ok_count": policy_search.get("ok_count"),
            },
            "replay": {
                "status": replay.get("status"),
                "row_count": replay.get("row_count"),
            },
            "learner_forward_loss": {
                "status": learner_forward_loss.get("status"),
                "ok": learner_forward_loss.get("ok"),
            },
            "step_count": len(result.get("steps", [])),
        }
    )


def profile_output_payload(result: Mapping[str, Any], output: str) -> dict[str, Any] | None:
    if output == "none":
        return None
    if output == "summary":
        return compact_stacked_debug_visual_survival_profile_summary(result)
    if output == "full":
        return _to_plain(result)
    raise ValueError("output must be one of: full, summary, none")


def _build_lightzero_policy(
    *,
    seed: int,
    num_simulations: int,
    require_installed_lightzero: bool,
    use_cuda: bool = False,
) -> dict[str, Any]:
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
    }
    if packages["LightZero"] == "missing":
        result = {
            "status": "blocked",
            "reason": "LightZero is not installed in this runtime",
            "packages": packages,
        }
        if require_installed_lightzero:
            result["required"] = True
        return result

    try:
        from ding.config import compile_config
        from lzero.policy.muzero import MuZeroPolicy

        import curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env  # noqa: F401

        atari_config = importlib.import_module("zoo.atari.config.atari_muzero_config")
        main_config = copy.deepcopy(atari_config.main_config)
        create_config = copy.deepcopy(atari_config.create_config)
        patches = _patch_stacked_visual_config(
            main_config,
            create_config,
            seed=seed,
            num_simulations=num_simulations,
            use_cuda=use_cuda,
        )
        cfg = compile_config(
            main_config,
            seed=seed,
            auto=True,
            create_cfg=create_config,
            save_cfg=False,
        )
        cfg.policy.cuda = bool(use_cuda)
        cfg.policy.device = "cuda" if use_cuda else "cpu"
        policy = MuZeroPolicy(cfg.policy)
        model = getattr(policy, "_model", None)
        if model is not None and hasattr(model, "eval"):
            model.eval()
        return {
            "status": "ok",
            "packages": packages,
            "policy": policy,
            "surface": _compiled_surface(cfg, main_config, create_config),
            "patches": patches,
            "policy_class": type(policy).__module__ + "." + type(policy).__name__,
            "model_class": (
                type(model).__module__ + "." + type(model).__name__
                if model is not None
                else None
            ),
            "requested_cuda": bool(use_cuda),
            "model_device": str(_policy_model_device(policy)),
        }
    except Exception as exc:  # pragma: no cover - exercised in installed runtime.
        return {
            "status": "failed",
            "packages": packages,
            "exception": _exception_result(exc),
        }


def _patch_stacked_visual_config(
    main_config: Any,
    create_config: Any,
    *,
    seed: int,
    num_simulations: int,
    use_cuda: bool = False,
) -> list[dict[str, Any]]:
    patches = [
        _set_path(
            create_config,
            ("env", "type"),
            LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
        ),
        _set_path(
            create_config,
            ("env", "import_names"),
            list(LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_IMPORT_NAMES),
        ),
        _set_path(create_config, ("env_manager", "type"), "base"),
        _set_path(main_config, ("exp_name",), "/tmp/curvyzero-stacked-debug-visual-profile"),
        _set_path(main_config, ("env", "env_id"), LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID),
        _set_path(main_config, ("env", "observation_shape"), DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
        _set_path(main_config, ("env", "frame_stack_num"), 1),
        _set_path(main_config, ("env", "gray_scale"), True),
        _set_path(main_config, ("env", "image_channel"), 4),
        _set_path(main_config, ("env", "collector_env_num"), 1),
        _set_path(main_config, ("env", "evaluator_env_num"), 1),
        _set_path(main_config, ("env", "n_evaluator_episode"), 1),
        _set_path(main_config, ("env", "dynamic_seed"), False),
        _set_path(main_config, ("env", "seed"), int(seed)),
        _set_path(main_config, ("policy", "cuda"), bool(use_cuda)),
        _set_path(main_config, ("policy", "env_type"), "not_board_games"),
        _set_path(main_config, ("policy", "collector_env_num"), 1),
        _set_path(main_config, ("policy", "evaluator_env_num"), 1),
        _set_path(main_config, ("policy", "n_episode"), 1),
        _set_path(main_config, ("policy", "num_simulations"), int(num_simulations)),
        _set_path(main_config, ("policy", "batch_size"), 2),
        _set_path(main_config, ("policy", "num_unroll_steps"), 1),
        _set_path(main_config, ("policy", "td_steps"), 1),
        _set_path(main_config, ("policy", "update_per_collect"), 1),
        _set_path(main_config, ("policy", "game_segment_length"), 4),
        _set_path(main_config, ("policy", "model", "model_type"), "conv"),
        _set_path(
            main_config,
            ("policy", "model", "observation_shape"),
            DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
        ),
        _set_path(main_config, ("policy", "model", "action_space_size"), 3),
        _set_path(main_config, ("policy", "model", "frame_stack_num"), 1),
        _set_path(main_config, ("policy", "model", "gray_scale"), True),
        _set_path(main_config, ("policy", "model", "image_channel"), 4),
        _set_path(main_config, ("policy", "model", "self_supervised_learning_loss"), True),
        _set_path(main_config, ("policy", "use_augmentation"), False),
    ]
    return patches


def _policy_eval_action(policy: Any, observation: Mapping[str, Any], *, step_index: int) -> dict[str, Any]:
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
            output = policy.eval_mode.forward(
                obs_tensor,
                action_mask=action_mask,
                to_play=to_play,
                ready_env_id=ready_env_id,
            )
        return {
            "ok": True,
            "status": "ok",
            "step_index": int(step_index),
            "api": "MuZeroPolicy.eval_mode.forward",
            "action": _extract_eval_action(output),
            "data_shape": [int(item) for item in obs_tensor.shape],
            "action_mask_shape": [int(item) for item in action_mask.shape],
            "to_play": to_play,
            "ready_env_id": [0],
            "compact_output": _compact_mcts_output(output),
        }
    except Exception as exc:  # pragma: no cover - installed runtime diagnosis.
        return {
            "ok": False,
            "status": "failed",
            "step_index": int(step_index),
            "reason": "MuZeroPolicy.eval_mode.forward failed",
            "exception": _exception_result(exc),
        }


def _learn_mode_forward_loss(policy: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    patch = _patch_no_training_step(policy)
    try:
        model = getattr(policy, "_model", None)
        before_hash = _model_hash(model)
        model_snapshot = _clone_state_dict(model)
        current_batch, target_batch, sample_summary = _learn_mode_batches(policy, sample)
        with _temporary_policy_batch_size(
            policy,
            int(sample_summary["batch_size"]),
        ) as batch_size_patched:
            output = policy.learn_mode.forward([current_batch, target_batch])
        restored = _restore_state_dict(model, model_snapshot)
        after_hash = _model_hash(model)
        return {
            "ok": True,
            "status": "run",
            "api": "MuZeroPolicy.learn_mode.forward",
            "optimizer_step": "blocked_by_noop_patch",
            "trainer_entrypoint_called": False,
            "model_hash_before": before_hash,
            "model_hash_after": after_hash,
            "model_parameters_changed": before_hash != after_hash,
            "model_state_restored": restored,
            "policy_batch_size_patched": batch_size_patched,
            "sample": sample_summary,
            "optimizer_step_block": patch["summary"](),
            "loss": _loss_summary(output),
        }
    except Exception as exc:  # pragma: no cover - installed runtime diagnosis.
        return {
            "ok": False,
            "status": "failed",
            "reason": "MuZeroPolicy.learn_mode.forward failed",
            "blocking_call": "MuZeroPolicy.learn_mode.forward",
            "blocking_stage": "learner recurrent dynamics/loss",
            "owner_file": (
                "src/curvyzero/training/"
                "curvyzero_stacked_debug_visual_survival_profile.py"
            ),
            "exception": _exception_result(exc),
            "optimizer_step": "blocked_by_noop_patch",
            "trainer_entrypoint_called": False,
            "optimizer_step_block": patch["summary"](),
        }
    finally:
        patch["restore"]()


def _learn_mode_batches(policy: Any, sample: Mapping[str, Any]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    configured_batch_size = int(getattr(getattr(policy, "_cfg", {}), "batch_size", 2))
    batch_size = _sample_learner_batch_size(sample, default=configured_batch_size)
    num_unroll_steps = int(getattr(getattr(policy, "_cfg", {}), "num_unroll_steps", 1))
    observations = np.asarray(sample["observation_batch"], dtype=np.float32)[:batch_size]
    next_observations = np.asarray(
        sample.get("next_observation_batch", sample["observation_batch"]),
        dtype=np.float32,
    )[:batch_size]
    actions = np.asarray(sample["action_batch"], dtype=np.int64)[:batch_size]
    rewards = np.asarray(sample["reward_batch"], dtype=np.float32)[:batch_size]
    policies = np.asarray(sample["policy_batch"], dtype=np.float32)[:batch_size]
    if observations.shape[0] < batch_size:
        raise ValueError(f"sample has {observations.shape[0]} rows, expected {batch_size}")
    model_cfg = getattr(getattr(policy, "_cfg", {}), "model", {})
    use_ssl_targets = bool(
        getattr(model_cfg, "self_supervised_learning_loss", False)
        if not isinstance(model_cfg, Mapping)
        else model_cfg.get("self_supervised_learning_loss", False)
    )
    learn_observations = (
        np.concatenate([observations, next_observations], axis=1).astype(np.float32)
        if use_ssl_targets and num_unroll_steps >= 1
        else observations
    )
    action_batch = np.repeat(actions[:, None], num_unroll_steps, axis=1).astype(np.int64)
    mask_batch = np.ones((batch_size, num_unroll_steps + 1), dtype=np.float32)
    indices = np.arange(batch_size, dtype=np.int64)
    weights = np.ones((batch_size,), dtype=np.float32)
    make_time = np.zeros((batch_size,), dtype=np.float32)
    target_reward = np.repeat(rewards[:, None], num_unroll_steps, axis=1).astype(np.float32)
    target_value, target_value_summary = _target_value_batch(
        policy,
        sample,
        batch_size=batch_size,
        num_unroll_steps=num_unroll_steps,
    )
    target_policy = np.repeat(policies[:, None, :], num_unroll_steps + 1, axis=1).astype(
        np.float32
    )
    current_batch = [learn_observations, action_batch, mask_batch, indices, weights, make_time]
    target_batch = [target_reward, target_value, target_policy]
    summary = {
        "batch_size": batch_size,
        "configured_policy_batch_size": configured_batch_size,
        "sample_batch_size": int(sample.get("batch_size", len(np.asarray(sample["reward_batch"])))),
        "num_unroll_steps": num_unroll_steps,
        "observation_batch_shape": [int(item) for item in observations.shape],
        "next_observation_batch_shape": [int(item) for item in next_observations.shape],
        "learn_observation_batch_shape": [int(item) for item in learn_observations.shape],
        "self_supervised_targets": use_ssl_targets,
        "action_batch_shape": [int(item) for item in action_batch.shape],
        "target_reward_shape": [int(item) for item in target_reward.shape],
        "target_value_shape": [int(item) for item in target_value.shape],
        "target_policy_shape": [int(item) for item in target_policy.shape],
        "target_semantics": target_value_summary["semantics"],
        "target_value_source": target_value_summary["source"],
        "target_value_discount": target_value_summary.get("discount"),
        "target_value_preview": target_value[: min(batch_size, 4)].copy(),
    }
    return current_batch, target_batch, summary


def _sample_learner_batch_size(sample: Mapping[str, Any], *, default: int) -> int:
    available = len(np.asarray(sample["reward_batch"]))
    requested = sample.get("learner_batch_size", default)
    try:
        batch_size = int(requested)
    except (TypeError, ValueError):
        batch_size = int(default)
    if batch_size < 1:
        batch_size = int(default)
    return min(int(batch_size), int(available))


def _target_value_batch(
    policy: Any,
    sample: Mapping[str, Any],
    *,
    batch_size: int,
    num_unroll_steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rewards = np.asarray(sample["reward_batch"], dtype=np.float32)
    if not _has_survival_target_metadata(sample):
        legacy = np.concatenate(
            [
                np.zeros((batch_size, 1), dtype=np.float32),
                rewards[:batch_size, None],
            ],
            axis=1,
        )
        return legacy, {
            "source": "legacy_immediate_reward",
            "discount": None,
            "semantics": (
                "target_reward is the immediate replay-row reward repeated across "
                "the configured unroll steps; target_value keeps the legacy "
                "[0.0, immediate_reward] adapter shape because replay metadata "
                "for discounted survival returns is absent."
            ),
        }

    discount = _policy_discount(policy)
    returns = _discounted_survival_returns_from_sample(sample, discount=discount)
    target_value = _survival_return_value_targets(
        sample,
        returns,
        batch_size=batch_size,
        num_unroll_steps=num_unroll_steps,
        discount=discount,
    )
    return target_value, {
        "source": "discounted_survival_return",
        "discount": float(discount),
        "semantics": (
            "target_reward is the immediate replay-row reward repeated across "
            "the configured unroll steps; target_value is remaining discounted "
            "survival return per episode_id/env_row_id/player_id/decision_index "
            "trajectory when episode metadata is present, with the legacy "
            "iteration/env_row_id/player_id trajectory as fallback."
        ),
    }


def _has_survival_target_metadata(sample: Mapping[str, Any]) -> bool:
    return all(key in sample for key in SURVIVAL_TARGET_METADATA_KEYS)


def _policy_discount(policy: Any) -> float:
    cfg = getattr(policy, "_cfg", {})
    for key in ("discount_factor", "discount"):
        raw = _cfg_get(cfg, key, None)
        if raw is None:
            continue
        try:
            discount = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(discount):
            return discount
    return DEFAULT_SURVIVAL_TARGET_DISCOUNT


@contextlib.contextmanager
def _temporary_policy_batch_size(policy: Any, batch_size: int):
    cfg = getattr(policy, "_cfg", None)
    if cfg is None or not hasattr(cfg, "batch_size"):
        yield False
        return
    old_batch_size = cfg.batch_size
    cfg.batch_size = int(batch_size)
    try:
        yield int(old_batch_size) != int(batch_size)
    finally:
        cfg.batch_size = old_batch_size


def _discounted_survival_returns_from_sample(
    sample: Mapping[str, Any],
    *,
    discount: float,
) -> np.ndarray:
    context_lookup = _return_context_lookup(sample, discount=discount)
    if context_lookup is not None:
        iterations = np.asarray(sample["iteration_batch"], dtype=np.int64)
        decisions = np.asarray(sample["decision_index_batch"], dtype=np.int64)
        return np.asarray(
            [
                context_lookup.get(
                    _sample_return_key(sample, index),
                    0.0,
                )
                for index in range(len(iterations))
            ],
            dtype=np.float32,
        )

    rewards = np.asarray(sample["reward_batch"], dtype=np.float32)
    dones = np.asarray(sample.get("done_batch", np.zeros_like(rewards)), dtype=bool)
    order = _sample_chronological_order(sample)
    returns = np.zeros((rewards.shape[0],), dtype=np.float32)
    running: dict[tuple[int, ...], float] = {}
    for index in reversed(order):
        key = _sample_trajectory_key(sample, index)
        if bool(dones[index]):
            running[key] = 0.0
        value = float(rewards[index]) + float(discount) * running.get(key, 0.0)
        returns[index] = np.float32(value)
        running[key] = value
    return returns


def _survival_return_value_targets(
    sample: Mapping[str, Any],
    returns: np.ndarray,
    *,
    batch_size: int,
    num_unroll_steps: int,
    discount: float | None = None,
) -> np.ndarray:
    target_value = np.zeros((batch_size, num_unroll_steps + 1), dtype=np.float32)
    lookup = (
        _return_context_lookup(sample, discount=discount)
        if discount is not None
        else None
    )
    if lookup is None:
        lookup = _sample_return_lookup(sample, returns)
    decisions = np.asarray(sample["decision_index_batch"], dtype=np.int64)
    for row in range(batch_size):
        trajectory_key = _sample_trajectory_key(sample, row)
        for offset in range(num_unroll_steps + 1):
            target_value[row, offset] = np.float32(
                lookup.get(
                    (*trajectory_key, int(decisions[row]) + int(offset)),
                    0.0,
                )
            )
    return target_value


def _return_context_lookup(
    sample: Mapping[str, Any],
    *,
    discount: float | None,
) -> dict[tuple[int, ...], float] | None:
    required = (
        "return_context_iteration_batch",
        "return_context_env_row_id_batch",
        "return_context_player_id_batch",
        "return_context_decision_index_batch",
        "return_context_reward_batch",
        "return_context_done_batch",
    )
    if discount is None or not all(key in sample for key in required):
        return None
    iterations = np.asarray(sample["return_context_iteration_batch"], dtype=np.int64)
    env_rows = np.asarray(sample["return_context_env_row_id_batch"], dtype=np.int64)
    players = np.asarray(sample["return_context_player_id_batch"], dtype=np.int64)
    decisions = np.asarray(sample["return_context_decision_index_batch"], dtype=np.int64)
    rewards = np.asarray(sample["return_context_reward_batch"], dtype=np.float32)
    dones = np.asarray(sample["return_context_done_batch"], dtype=bool)
    count = int(rewards.shape[0])
    episode_ids = (
        np.asarray(sample["return_context_episode_id_batch"], dtype=np.int64)
        if "return_context_episode_id_batch" in sample
        else None
    )
    if not (
        iterations.shape[0]
        == env_rows.shape[0]
        == players.shape[0]
        == decisions.shape[0]
        == dones.shape[0]
        == count
    ):
        return None
    if episode_ids is not None and episode_ids.shape[0] != count:
        return None
    decorated = [
        (
            int(iterations[index]),
            int(decisions[index]),
            int(env_rows[index]),
            int(players[index]),
            index,
        )
        for index in range(count)
    ]
    returns = np.zeros((count,), dtype=np.float32)
    running: dict[tuple[int, ...], float] = {}
    for *_, index in reversed(sorted(decorated)):
        key = _context_trajectory_key(
            iterations=iterations,
            env_rows=env_rows,
            players=players,
            episode_ids=episode_ids,
            index=index,
        )
        if bool(dones[index]):
            running[key] = 0.0
        value = float(rewards[index]) + float(discount) * running.get(key, 0.0)
        returns[index] = np.float32(value)
        running[key] = value
    return {
        (
            *_context_trajectory_key(
                iterations=iterations,
                env_rows=env_rows,
                players=players,
                episode_ids=episode_ids,
                index=index,
            ),
            int(decisions[index]),
        ): float(returns[index])
        for index in range(count)
    }


def _sample_chronological_order(sample: Mapping[str, Any]) -> list[int]:
    count = len(np.asarray(sample["reward_batch"]))
    decorated = [
        (
            int(np.asarray(sample["iteration_batch"])[index]),
            int(np.asarray(sample["decision_index_batch"])[index]),
            int(np.asarray(sample["env_row_id_batch"])[index]),
            int(np.asarray(sample["player_id_batch"])[index]),
            index,
        )
        for index in range(count)
    ]
    return [item[-1] for item in sorted(decorated)]


def _sample_return_lookup(
    sample: Mapping[str, Any],
    returns: np.ndarray,
) -> dict[tuple[int, ...], float]:
    return {
        _sample_return_key(sample, index): float(returns[index])
        for index in range(len(returns))
    }


def _sample_return_key(sample: Mapping[str, Any], index: int) -> tuple[int, ...]:
    decisions = np.asarray(sample["decision_index_batch"], dtype=np.int64)
    return (*_sample_trajectory_key(sample, index), int(decisions[index]))


def _sample_trajectory_key(sample: Mapping[str, Any], index: int) -> tuple[int, ...]:
    iterations = np.asarray(sample["iteration_batch"], dtype=np.int64)
    env_rows = np.asarray(sample["env_row_id_batch"], dtype=np.int64)
    players = np.asarray(sample["player_id_batch"], dtype=np.int64)
    episode_ids = (
        np.asarray(sample["episode_id_batch"], dtype=np.int64)
        if "episode_id_batch" in sample
        else None
    )
    return _context_trajectory_key(
        iterations=iterations,
        env_rows=env_rows,
        players=players,
        episode_ids=episode_ids,
        index=index,
    )


def _context_trajectory_key(
    *,
    iterations: np.ndarray,
    env_rows: np.ndarray,
    players: np.ndarray,
    episode_ids: np.ndarray | None,
    index: int,
) -> tuple[int, ...]:
    if episode_ids is not None:
        return (
            int(episode_ids[index]),
            int(env_rows[index]),
            int(players[index]),
        )
    return (
        int(iterations[index]),
        int(env_rows[index]),
        int(players[index]),
    )


def toy_alive_survival_target_diagnostic(
    *,
    steps: int = 20,
    num_unroll_steps: int = 1,
    discount: float = DEFAULT_SURVIVAL_TARGET_DISCOUNT,
) -> dict[str, Any]:
    """Describe LightZero target rows for a toy all-alive trajectory.

    Metadata-bearing two-seat samples now train the value head on remaining
    discounted survival return instead of the legacy ``[0.0, immediate_reward]``
    adapter target.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if num_unroll_steps < 1:
        raise ValueError("num_unroll_steps must be >= 1")
    if not np.isfinite(discount):
        raise ValueError("discount must be finite")

    rewards = np.ones((steps,), dtype=np.float32)
    current_target_reward = np.repeat(
        rewards[:, None],
        num_unroll_steps,
        axis=1,
    ).astype(np.float32)
    current_target_value = np.concatenate(
        [np.zeros((steps, 1), dtype=np.float32), rewards[:, None]],
        axis=1,
    )

    sample = {
        "reward_batch": rewards,
        "done_batch": np.zeros((steps,), dtype=np.bool_),
        "iteration_batch": np.ones((steps,), dtype=np.int64),
        "env_row_id_batch": np.zeros((steps,), dtype=np.int64),
        "player_id_batch": np.zeros((steps,), dtype=np.int64),
        "decision_index_batch": np.arange(steps, dtype=np.int64),
    }
    discounted_returns = _discounted_survival_returns_from_sample(
        sample,
        discount=float(discount),
    )
    metadata_target_value = _survival_return_value_targets(
        sample,
        discounted_returns,
        batch_size=steps,
        num_unroll_steps=num_unroll_steps,
    )

    return _to_plain(
        {
            "toy": {
                "steps": int(steps),
                "reward_per_alive_step": 1.0,
                "terminal_reward": 0.0,
                "discount": float(discount),
            },
            "metadata_adapter_targets": {
                "target_value_shape": list(metadata_target_value.shape),
                "target_value_rows": metadata_target_value,
                "first_column_descends": bool(
                    np.all(np.diff(metadata_target_value[:, 0]) < 0.0)
                ),
                "meaning": (
                    "With replay metadata present, target_value[step, 0] is "
                    "the remaining discounted alive count from that decision."
                ),
            },
            "legacy_adapter_targets_without_metadata": {
                "target_reward_shape": list(current_target_reward.shape),
                "target_reward_rows": current_target_reward,
                "target_value_shape": list(current_target_value.shape),
                "target_value_rows": current_target_value,
                "meaning": (
                    "Every alive row trains immediate reward as 1.0. The value "
                    "head sees a root target of 0.0 and a one-step target of "
                    "1.0, so remaining lifetime is not backed up."
                ),
            },
            "return_backed_value_reference": {
                "discounted_survival_returns": discounted_returns,
                "undiscounted_survival_returns": np.arange(
                    steps,
                    0,
                    -1,
                    dtype=np.float32,
                ),
                "meaning": (
                    "A full survival value target for row t would be the "
                    "remaining discounted count of alive rewards from t."
                ),
            },
        }
    )


def split_iteration_survival_target_proof() -> dict[str, Any]:
    """Tiny proof that one episode split across iterations keeps one return tail."""

    sample = {
        "reward_batch": np.ones((4,), dtype=np.float32),
        "done_batch": np.asarray([False, False, False, True], dtype=np.bool_),
        "iteration_batch": np.asarray([1, 1, 2, 2], dtype=np.int64),
        "episode_id_batch": np.asarray([7, 7, 7, 7], dtype=np.int64),
        "env_row_id_batch": np.zeros((4,), dtype=np.int64),
        "player_id_batch": np.zeros((4,), dtype=np.int64),
        "decision_index_batch": np.arange(4, dtype=np.int64),
        "return_context_iteration_batch": np.asarray([1, 1, 2, 2], dtype=np.int64),
        "return_context_episode_id_batch": np.asarray([7, 7, 7, 7], dtype=np.int64),
        "return_context_env_row_id_batch": np.zeros((4,), dtype=np.int64),
        "return_context_player_id_batch": np.zeros((4,), dtype=np.int64),
        "return_context_decision_index_batch": np.arange(4, dtype=np.int64),
        "return_context_reward_batch": np.ones((4,), dtype=np.float32),
        "return_context_done_batch": np.asarray([False, False, False, True], dtype=np.bool_),
    }
    returns = _discounted_survival_returns_from_sample(sample, discount=1.0)
    targets = _survival_return_value_targets(
        sample,
        returns,
        batch_size=4,
        num_unroll_steps=1,
        discount=1.0,
    )
    expected = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    return _to_plain(
        {
            "ok": bool(np.array_equal(returns, expected))
            and bool(np.array_equal(targets[:, 0], expected)),
            "iterations": sample["iteration_batch"],
            "episode_ids": sample["episode_id_batch"],
            "discount": 1.0,
            "returns": returns,
            "target_value_first_column": targets[:, 0],
            "expected": expected,
        }
    )


def _patch_no_training_step(policy: Any) -> dict[str, Any]:
    optimizer = getattr(policy, "_optimizer", None)
    scheduler = getattr(policy, "lr_scheduler", None)
    target_model = getattr(policy, "_target_model", None)
    calls = {"optimizer_step": 0, "scheduler_step": 0, "target_update": 0}
    original_optimizer_step = getattr(optimizer, "step", None)
    original_scheduler_step = getattr(scheduler, "step", None)
    original_target_update = getattr(target_model, "update", None)

    def noop_optimizer_step(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        calls["optimizer_step"] += 1

    def noop_scheduler_step(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        calls["scheduler_step"] += 1

    def noop_target_update(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        calls["target_update"] += 1

    if optimizer is not None and callable(original_optimizer_step):
        optimizer.step = noop_optimizer_step
    if scheduler is not None and callable(original_scheduler_step):
        scheduler.step = noop_scheduler_step
    if target_model is not None and callable(original_target_update):
        try:
            target_model.update = noop_target_update
        except Exception:
            original_target_update = None

    def restore() -> None:
        if optimizer is not None and callable(original_optimizer_step):
            optimizer.step = original_optimizer_step
        if scheduler is not None and callable(original_scheduler_step):
            scheduler.step = original_scheduler_step
        if target_model is not None and callable(original_target_update):
            try:
                target_model.update = original_target_update
            except Exception:
                pass

    def summary() -> dict[str, int]:
        return {
            "optimizer_step_calls_blocked": calls["optimizer_step"],
            "scheduler_step_calls_blocked": calls["scheduler_step"],
            "target_update_calls_blocked": calls["target_update"],
        }

    return {"restore": restore, "summary": summary}


def _loss_summary(output: Any) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        return {"raw_type": type(output).__name__}
    keys = (
        "weighted_total_loss",
        "total_loss",
        "policy_loss",
        "reward_loss",
        "value_loss",
        "target_reward",
        "target_value",
        "predicted_rewards",
        "predicted_values",
    )
    return {key: _to_plain(output[key]) for key in keys if key in output}


def _model_hash(model: Any) -> str | None:
    if model is None or not hasattr(model, "state_dict"):
        return None
    import hashlib

    digest = hashlib.sha256()
    try:
        for key, value in sorted(model.state_dict().items(), key=lambda item: str(item[0])):
            digest.update(str(key).encode("utf-8"))
            if hasattr(value, "detach"):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(str(array.dtype).encode("utf-8"))
                digest.update(str(array.shape).encode("utf-8"))
                digest.update(array.tobytes())
        return digest.hexdigest()[:16]
    except Exception:
        return None


def _clone_state_dict(model: Any) -> dict[str, Any] | None:
    if model is None or not hasattr(model, "state_dict"):
        return None
    try:
        return {
            str(key): value.detach().clone()
            if hasattr(value, "detach")
            else copy.deepcopy(value)
            for key, value in model.state_dict().items()
        }
    except Exception:
        return None


def _restore_state_dict(model: Any, snapshot: dict[str, Any] | None) -> bool:
    if model is None or snapshot is None or not hasattr(model, "load_state_dict"):
        return False
    try:
        model.load_state_dict(snapshot, strict=True)
        return True
    except Exception:
        return False


def _build_replay_row(
    *,
    step_index: int,
    previous_observation: Mapping[str, Any],
    timestep: Any,
    action: int,
    search_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "step_index": int(step_index),
        "observation": np.asarray(previous_observation["observation"], dtype=np.float32).copy(),
        "next_observation": np.asarray(timestep.obs["observation"], dtype=np.float32).copy(),
        "action_mask": np.asarray(previous_observation["action_mask"], dtype=np.int8).copy(),
        "action": int(action),
        "action_weights": _action_weights(search_record, int(action)),
        "root_value": _root_value(search_record),
        "reward": float(timestep.reward),
        "done": bool(timestep.done),
        "observation_schema_id": timestep.info.get("observation_schema_id"),
        "reward_schema_id": timestep.info.get("reward_schema_id"),
        "frame_stack_owner": timestep.info.get("frame_stack_owner"),
        "terminal_reason": timestep.info.get("terminal_reason"),
    }


def _sample_replay_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"ok": False, "reason": "no replay rows available"}
    return {
        "ok": True,
        "batch_size": len(rows),
        "observation_batch": np.stack([row["observation"] for row in rows], axis=0),
        "next_observation_batch": np.stack([row["next_observation"] for row in rows], axis=0),
        "action_batch": np.asarray([row["action"] for row in rows], dtype=np.int64),
        "reward_batch": np.asarray([row["reward"] for row in rows], dtype=np.float32),
        "done_batch": np.asarray([row["done"] for row in rows], dtype=np.bool_),
        "policy_batch": np.stack([row["action_weights"] for row in rows], axis=0),
    }


def _action_weights(search_record: Mapping[str, Any], action: int) -> np.ndarray:
    compact = search_record.get("compact_output")
    if isinstance(compact, Mapping):
        for key in ("visit_count_distribution", "visit_count_distributions"):
            value = compact.get(key)
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.size == 3 and float(array.sum()) > 0:
                return array / float(array.sum())
    weights = np.zeros((3,), dtype=np.float32)
    if 0 <= action < 3:
        weights[action] = 1.0
    return weights


def _root_value(search_record: Mapping[str, Any]) -> float:
    compact = search_record.get("compact_output")
    if isinstance(compact, Mapping):
        for key in ("searched_value", "predicted_value", "value"):
            if key in compact:
                try:
                    return float(np.asarray(compact[key]).reshape(-1)[0])
                except (TypeError, ValueError, IndexError):
                    pass
    return 0.0


def _step_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step_index": int(row["step_index"]),
        "action": int(row["action"]),
        "reward": float(row["reward"]),
        "done": bool(row["done"]),
        "observation_shape": [int(item) for item in row["observation"].shape],
        "next_observation_shape": [int(item) for item in row["next_observation"].shape],
        "observation_schema_id": row["observation_schema_id"],
        "reward_schema_id": row["reward_schema_id"],
        "frame_stack_owner": row["frame_stack_owner"],
    }


def _surface_summary() -> dict[str, Any]:
    return {
        "surface": "debug_visual_tensor",
        "env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
        "env_id": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID,
        "env_import_names": list(LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_IMPORT_NAMES),
        "observation_schema_id": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
        "observation_schema_hash": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH,
        "raw_observation_schema_id": DEBUG_OCCUPANCY_GRAY64_LABEL,
        "raw_observation_schema_hash": DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
        "renderer_impl_id": DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
        "raw_frame_shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
        "observation_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
        "dtype": "float32",
        "value_range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
        "frame_stack_owner": STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER,
        "frame_stack_proof": "wrapper_owned_fifo_stack; not LightZero env-manager stacking",
        "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
        "reward_schema_hash": SURVIVAL_TIME_REWARD_SCHEMA_HASH,
        "debug_fidelity_only": True,
        "source_fidelity_claim": "none",
        "source_fidelity_level": "none",
        "browser_pixel_fidelity": False,
        "uses_ale": False,
        "ale_usage": "none",
    }


def _compiled_surface(cfg: Any, main_config: Any, create_config: Any) -> dict[str, Any]:
    return {
        "env_type": _cfg_get(_cfg_get(create_config, "env", {}), "type", None),
        "env_import_names": _to_plain(_cfg_get(_cfg_get(create_config, "env", {}), "import_names", ())),
        "env_manager_type": _cfg_get(_cfg_get(create_config, "env_manager", {}), "type", None),
        "policy_type": _cfg_get(_cfg_get(create_config, "policy", {}), "type", None),
        "policy_import_names": _to_plain(_cfg_get(_cfg_get(create_config, "policy", {}), "import_names", ())),
        "env_id": _cfg_get(_cfg_get(main_config, "env", {}), "env_id", None),
        "env_frame_stack_num": _cfg_get(_cfg_get(main_config, "env", {}), "frame_stack_num", None),
        "env_image_channel": _cfg_get(_cfg_get(main_config, "env", {}), "image_channel", None),
        "policy_env_type": _cfg_get(_cfg_get(main_config, "policy", {}), "env_type", None),
        "compiled_policy_cuda": bool(getattr(cfg.policy, "cuda", False)),
        "compiled_policy_device": str(getattr(cfg.policy, "device", "missing")),
        "model_type": str(cfg.policy.model.model_type),
        "observation_shape": _to_plain(cfg.policy.model.observation_shape),
        "model_image_channel": int(cfg.policy.model.image_channel),
        "model_frame_stack_num": int(cfg.policy.model.frame_stack_num),
        "model_self_supervised_learning_loss": bool(
            cfg.policy.model.self_supervised_learning_loss
        ),
        "action_space_size": int(cfg.policy.model.action_space_size),
        "num_simulations": int(cfg.policy.num_simulations),
        "batch_size": int(cfg.policy.batch_size),
    }


def _set_path(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        current = current[part]
    old = current.get(path[-1], "<missing>") if isinstance(current, dict) else current[path[-1]]
    current[path[-1]] = value
    return {"path": ".".join(path), "old": _to_plain(old), "new": _to_plain(value)}


def _extract_eval_action(output: Any) -> int:
    plain = _to_plain(output)
    if isinstance(plain, Mapping):
        if "action" in plain:
            return int(np.asarray(plain["action"]).reshape(-1)[0])
        for key in (0, "0"):
            if key in plain:
                return _extract_eval_action(plain[key])
        for key in ("actions", "selected_action", "selected_actions"):
            if key in plain:
                return int(np.asarray(plain[key]).reshape(-1)[0])
    if isinstance(plain, list) and plain:
        return _extract_eval_action(plain[0])
    raise ValueError(f"could not extract action from policy eval output: {plain!r}")


def _compact_mcts_output(output: Any) -> dict[str, Any]:
    root = _root_output(output)
    if not isinstance(root, Mapping):
        return {"raw": root}
    keys = (
        "action",
        "visit_count_distribution",
        "visit_count_distributions",
        "visit_count_distribution_entropy",
        "visit_counts",
        "predicted_policy_logits",
        "policy_logits",
        "predicted_value",
        "searched_value",
        "value",
    )
    compact = {key: root.get(key) for key in keys if key in root}
    compact["output_keys"] = sorted(str(key) for key in root.keys())
    return _to_plain(compact)


def _root_output(output: Any) -> Any:
    plain = _to_plain(output)
    if isinstance(plain, Mapping):
        for key in (0, "0"):
            if key in plain:
                return plain[key]
    return plain


def _policy_model_device(policy: Any) -> Any:
    try:
        import torch

        model = getattr(policy, "_model", None)
        if model is None:
            return torch.device("cpu")
        return next(model.parameters()).device
    except Exception:
        import torch

        return torch.device("cpu")


def _tensor_stats(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        array = np.asarray(value)
        return {
            "shape": [int(item) for item in array.shape],
            "dtype": str(array.dtype),
            "min": float(array.min()) if array.size else None,
            "max": float(array.max()) if array.size else None,
            "mean": float(array.mean()) if array.size else None,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _strip_runtime_object(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_runtime_object(item)
            for key, item in value.items()
            if key != "policy"
        }
    return _to_plain(value)


def _strip_large_arrays(value: Any) -> Any:
    if isinstance(value, Mapping):
        stripped: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"observation", "next_observation", "observation_batch", "next_observation_batch"}:
                stripped[str(key)] = _array_summary(item)
            else:
                stripped[str(key)] = _strip_large_arrays(item)
        return stripped
    if isinstance(value, (list, tuple)):
        return [_strip_large_arrays(item) for item in value]
    if hasattr(value, "shape") and not np.isscalar(value):
        return _array_summary(value)
    return _to_plain(value)


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
        "nonzero": int(np.count_nonzero(array)) if array.size else 0,
    }


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "detach"):
        return _to_plain(value.detach().cpu().numpy())
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-simulations", type=int, default=2)
    parser.add_argument("--require-installed-lightzero", action="store_true")
    parser.add_argument("--skip-installed-lightzero", action="store_true")
    parser.add_argument(
        "--output",
        choices=("full", "summary", "none"),
        default="full",
        help="Printed JSON payload; the in-process result object is always full.",
    )
    args = parser.parse_args()
    result = run_curvyzero_stacked_debug_visual_survival_profile(
        seed=args.seed,
        steps=args.steps,
        num_simulations=args.num_simulations,
        require_installed_lightzero=args.require_installed_lightzero,
        attempt_installed_lightzero=not args.skip_installed_lightzero,
    )
    payload = profile_output_payload(result, args.output)
    if payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "PROFILE_SCHEMA_ID",
    "compact_stacked_debug_visual_survival_profile_summary",
    "profile_output_payload",
    "run_curvyzero_stacked_debug_visual_survival_profile",
    "run_stacked_debug_visual_survival_profile",
    "toy_alive_survival_target_diagnostic",
]
