"""Deterministic LightZero oracle probe for dummy Pong contact-pressure states.

This is an eval/debug probe only. It does not train, does not modify rewards,
and does not run pytest. It samples concrete contact-pressure reset seeds,
logs the tabular features sent to LightZero, compares direct env action
effects, and queries policy-head logits plus eval-mode MCTS outputs for saved
LightZero checkpoints.

Run from the repository root, for example:

    uv run --extra modal modal run \
      -m curvyzero.infra.modal.lightzero_dummy_pong_contact_pressure_oracle
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs


APP_NAME = "curvyzero-lightzero-dummy-pong-contact-pressure-oracle"
TASK_ID = "lightzero-dummy-pong"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
LIGHTZERO_VERSION = "0.2.0"

DEFAULT_RUN_ID = "lz-dpong-20260509T175407Z-77159cc3a6b4"
DEFAULT_ATTEMPT_ID = "attempt-20260509T175407Z-8105d62c1e00"
DEFAULT_CHECKPOINT_REFS = (
    "iteration_0=training/lightzero-dummy-pong/"
    "lz-dpong-20260509T175407Z-77159cc3a6b4/checkpoints/lightzero/iteration_0.pth.tar,"
    "iteration_3=training/lightzero-dummy-pong/"
    "lz-dpong-20260509T175407Z-77159cc3a6b4/checkpoints/lightzero/iteration_3.pth.tar,"
    "ckpt_best=training/lightzero-dummy-pong/"
    "lz-dpong-20260509T175407Z-77159cc3a6b4/checkpoints/lightzero/ckpt_best.pth.tar"
)
DEFAULT_STATE_SEEDS = "20260510,20260515,20260523"
DEFAULT_NUM_SIMULATIONS = "2,8,16,25"
FEATURE_NAMES = (
    "ego_paddle_y",
    "opponent_paddle_y",
    "ego_paddle_x",
    "opponent_paddle_x",
    "ball_dx_forward",
    "ball_dy_from_ego_center",
    "ball_vx_forward",
    "ball_vy",
    "ball_y",
    "step",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}", "numpy>=1.26")
    .env({"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}"})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "detach"):
        return _to_plain(value.detach().cpu())
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-16:],
    }


def _parse_checkpoint_refs(text: str) -> list[tuple[str, str]]:
    refs = []
    for item in text.replace("\n", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"checkpoint ref must be LABEL=REF: {item!r}")
        label, ref = item.split("=", 1)
        label = label.strip()
        ref = ref.strip()
        if not label or not ref:
            raise ValueError(f"checkpoint ref must be LABEL=REF: {item!r}")
        refs.append((label, ref))
    if not refs:
        raise ValueError("at least one checkpoint ref is required")
    return refs


def _parse_ints(text: str, *, label: str) -> list[int]:
    values = []
    for item in text.replace("\n", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"{label} must include at least one integer")
    return values


def _output_ref(
    *,
    output_ref: str | None,
    run_id: str,
    attempt_id: str,
    eval_id: str,
) -> PurePosixPath:
    explicit_ref = runs.explicit_volume_ref(output_ref or "") if output_ref else None
    if explicit_ref is not None:
        return explicit_ref
    if output_ref:
        return runs.require_relative_ref(output_ref)
    return (
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, eval_id)
        / "contact_pressure_state_action_oracle.json"
    )


def _snapshot(env: Any) -> dict[str, Any]:
    return {
        "player_0_y": int(env._paddle_y["player_0"]),
        "player_1_y": int(env._paddle_y["player_1"]),
        "ball_x": int(env._ball_x),
        "ball_y": int(env._ball_y),
        "ball_vx": int(env._ball_vx),
        "ball_vy": int(env._ball_vy),
        "step": int(env._step),
        "reset_info": dict(env._reset_info),
    }


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent: {agent!r}")


def _track_ball_oracle_action(observation: Any) -> int:
    if observation.ball_dy_from_ego_center < 0:
        return 0
    if observation.ball_dy_from_ego_center > 0:
        return 2
    return 1


def _direct_step_effect(
    *,
    config: Any,
    seed: int,
    ego_agent: str,
    action: int,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong import ACTION_LABELS
    from curvyzero.training.dummy_pong import PongEnv
    from curvyzero.training.dummy_pong_eval import LaggedTrackBallPolicy

    opponent_agent = _opponent(ego_agent)
    env = PongEnv(config)
    observations = env.reset(seed=seed)
    before = _snapshot(env)
    opponent_policy = LaggedTrackBallPolicy(delay=1)
    opponent_policy.reset(seed, opponent_agent)
    raster_grid = env.raster_observation()
    opponent_action = int(
        opponent_policy.action(observations[opponent_agent], raster_grid, opponent_agent)
    )
    joint_action = {
        ego_agent: int(action),
        opponent_agent: opponent_action,
    }
    step = env.step(joint_action)
    return {
        "action": int(action),
        "action_label": ACTION_LABELS[int(action)],
        "ego_paddle_delta": int(action) - 1,
        "opponent_policy": "lagged_track_ball_1",
        "joint_action": joint_action,
        "joint_action_labels": {
            agent: ACTION_LABELS[int(action_id)]
            for agent, action_id in joint_action.items()
        },
        "before": before,
        "after": _snapshot(env),
        "reward": float(step.rewards[ego_agent]),
        "terminated": bool(step.terminated),
        "truncated": bool(step.truncated),
        "last_hit": step.infos["last_hit"],
        "last_hit_impact": step.infos["last_hit_impact"],
        "winner": step.infos["winner"],
    }


def _candidate_rollout(
    *,
    config: Any,
    seed: int,
    ego_agent: str,
    action: int,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong import ACTION_LABELS
    from curvyzero.training.dummy_pong import PongEnv
    from curvyzero.training.dummy_pong_eval import LaggedTrackBallPolicy
    from curvyzero.training.dummy_pong_eval import TrackBallPolicy

    opponent_agent = _opponent(ego_agent)
    env = PongEnv(config)
    env.reset(seed=seed)
    ego_policy = TrackBallPolicy()
    opponent_policy = LaggedTrackBallPolicy(delay=1)
    ego_policy.reset(seed, ego_agent)
    opponent_policy.reset(seed, opponent_agent)
    released_to_ego_baseline = False
    first_contact = None
    score_return = 0.0
    action_trace_prefix = []
    final_step = None

    for rollout_step in range(config.max_steps):
        observations = env.observations()
        raster_grid = env.raster_observation()
        if released_to_ego_baseline:
            ego_action = int(ego_policy.action(observations[ego_agent], raster_grid, ego_agent))
        else:
            ego_action = int(action)
        opponent_action = int(
            opponent_policy.action(observations[opponent_agent], raster_grid, opponent_agent)
        )
        joint_action = {
            ego_agent: ego_action,
            opponent_agent: opponent_action,
        }
        if len(action_trace_prefix) < 12:
            action_trace_prefix.append(dict(joint_action))
        final_step = env.step(joint_action)
        score_return += float(final_step.rewards[ego_agent])

        impact = final_step.infos["last_hit_impact"]
        if first_contact is None and impact is not None:
            first_contact = {
                "step": int(next(iter(final_step.observations.values())).step),
                **dict(impact),
            }
            released_to_ego_baseline = True
        if final_step.terminated or final_step.truncated:
            break

    if final_step is None:
        raise RuntimeError("rollout did not step")
    terminal_step = int(next(iter(final_step.observations.values())).step)
    return {
        "action": int(action),
        "action_label": ACTION_LABELS[int(action)],
        "opponent_policy": "lagged_track_ball_1",
        "ego_action_mode": "until_contact_then_track_ball",
        "score_return": float(score_return),
        "ego_reward": float(final_step.rewards[ego_agent]),
        "winner": final_step.infos["winner"],
        "terminated": bool(final_step.terminated),
        "truncated": bool(final_step.truncated),
        "survival_steps": terminal_step,
        "first_contact": first_contact,
        "action_trace_prefix": action_trace_prefix,
        "final_state": {
            "ball": dict(final_step.infos["ball"]),
            "paddles": dict(final_step.infos["paddles"]),
        },
    }


def _state_probe_row(*, seed: int, pressure_agent: str, max_env_step: int) -> dict[str, Any]:
    from curvyzero.training.dummy_pong import ACTION_LABELS
    from curvyzero.training.dummy_pong import PongConfig
    from curvyzero.training.dummy_pong import PongEnv
    from curvyzero.training.lightzero_dummy_pong_features import (
        TABULAR_FEATURE_SCHEMA_ID,
    )
    from curvyzero.training.lightzero_dummy_pong_features import (
        encode_tabular_ego_observation,
    )

    config = PongConfig(
        max_steps=max_env_step,
        reset_profile="contact_pressure",
        reset_pressure_agent=pressure_agent,
    )
    env = PongEnv(config)
    observations = env.reset(seed=seed)
    observation = observations[pressure_agent]
    features = encode_tabular_ego_observation(observation, config)
    direct_step_effects = [
        _direct_step_effect(config=config, seed=seed, ego_agent=pressure_agent, action=action)
        for action in range(config.action_count)
    ]
    rollouts = [
        _candidate_rollout(config=config, seed=seed, ego_agent=pressure_agent, action=action)
        for action in range(config.action_count)
    ]
    scores_by_action = {
        row["action_label"]: float(row["score_return"])
        for row in rollouts
    }
    return {
        "state_id": f"{pressure_agent}-seed-{seed}",
        "seed": int(seed),
        "pressure_agent": pressure_agent,
        "ego_agent": pressure_agent,
        "opponent_agent": _opponent(pressure_agent),
        "initial_snapshot": _snapshot(env),
        "initial_observation": asdict(observation),
        "tabular_features": {
            "schema_id": TABULAR_FEATURE_SCHEMA_ID,
            "feature_names": list(FEATURE_NAMES),
            "values": [float(value) for value in features.tolist()],
        },
        "track_ball_oracle_action": int(_track_ball_oracle_action(observation)),
        "track_ball_oracle_action_label": ACTION_LABELS[
            int(_track_ball_oracle_action(observation))
        ],
        "direct_step_effects": direct_step_effects,
        "scoreability_rollouts": rollouts,
        "score_return_by_action_label": scores_by_action,
        "down_changes_score_return": bool(
            scores_by_action["down"] != scores_by_action["up"]
            or scores_by_action["down"] != scores_by_action["stay"]
        ),
    }


def _action_order(logits: list[float]) -> list[int]:
    return sorted(range(len(logits)), key=logits.__getitem__, reverse=True)


def _softmax(values: list[float]) -> list[float]:
    import math

    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _policy_logits_for_states(
    *,
    checkpoint_path: Path,
    checkpoint_label: str,
    state_rows: list[dict[str, Any]],
    env: str,
    feature_mode: str,
    config_opponent_policy: str,
    seed: int,
    max_env_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from curvyzero.training.dummy_pong import ACTION_LABELS
    from curvyzero.training.lightzero_dummy_pong_policy import (
        load_lightzero_policy_head_greedy_checkpoint,
    )

    spec = load_lightzero_policy_head_greedy_checkpoint(
        policy_id=f"lightzero_{checkpoint_label}",
        checkpoint_path=checkpoint_path,
        env=env,
        feature_mode=feature_mode,
        opponent_policy=config_opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
    )
    model = spec.policy._model
    obs = [row["tabular_features"]["values"] for row in state_rows]
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        output = model.initial_inference(obs_tensor)
    logits_tensor = output.policy_logits.detach().cpu().float()
    rows = []
    argmax_counts = Counter()
    for index, state in enumerate(state_rows):
        logits = [float(value) for value in logits_tensor[index].reshape(-1).tolist()]
        order = _action_order(logits)
        argmax_counts[int(order[0])] += 1
        rows.append(
            {
                "state_id": state["state_id"],
                "policy_logits_up_stay_down": logits,
                "policy_softmax_up_stay_down": _softmax(logits),
                "argmax_action": int(order[0]),
                "argmax_action_label": ACTION_LABELS[int(order[0])],
                "ranked_actions": [
                    {"action": int(action), "action_label": ACTION_LABELS[int(action)]}
                    for action in order
                ],
                "top1_top2_margin": float(logits[order[0]] - logits[order[1]]),
                "down_minus_best_not_down": float(
                    logits[2] - max(logits[0], logits[1])
                ),
            }
        )
    summary = {
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata": spec.checkpoint_metadata,
        "argmax_counts_up_stay_down": [int(argmax_counts[i]) for i in range(3)],
    }
    return summary, {"checkpoint_label": checkpoint_label, "rows": rows}


def _extract_action(output: Any) -> int:
    if isinstance(output, dict):
        for key in (0, "0"):
            if key in output:
                return _extract_action(output[key])
        if "action" in output:
            return int(_to_plain(output["action"]))
        for key in ("actions", "selected_action", "selected_actions"):
            if key in output:
                value = _to_plain(output[key])
                return int(value[0] if isinstance(value, list) else value)
    if isinstance(output, list) and output:
        return _extract_action(output[0])
    raise ValueError(f"could not extract action from eval output: {output!r}")


def _root_output(output: Any) -> Any:
    plain = _to_plain(output)
    if isinstance(plain, dict):
        for key in (0, "0"):
            if key in plain:
                return plain[key]
    return plain


def _compact_mcts_output(output: Any) -> dict[str, Any]:
    root = _root_output(output)
    if not isinstance(root, dict):
        return {"raw": root}
    keys = [
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
    ]
    compact = {key: root.get(key) for key in keys if key in root}
    compact["output_keys"] = sorted(str(key) for key in root.keys())
    return compact


def _flat_action_vector(value: Any) -> list[float] | None:
    plain = _to_plain(value)
    if plain is None:
        return None
    if isinstance(plain, (int, float)):
        return [float(plain)]
    if not isinstance(plain, list):
        return None
    current = plain
    while (
        len(current) == 1
        and isinstance(current[0], list)
    ):
        current = current[0]
    if len(current) < 3:
        return None
    try:
        return [float(value) for value in current[:3]]
    except (TypeError, ValueError):
        return None


def _visit_vector(compact: dict[str, Any]) -> tuple[str | None, list[float] | None]:
    for field in (
        "visit_count_distributions",
        "visit_count_distribution",
        "visit_counts",
    ):
        vector = _flat_action_vector(compact.get(field))
        if vector is not None:
            return field, vector
    return None, None


def _vector_summary(vector: list[float] | None) -> dict[str, Any] | None:
    from curvyzero.training.dummy_pong import ACTION_LABELS

    if vector is None:
        return None
    order = sorted(range(len(vector)), key=vector.__getitem__, reverse=True)
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else top
    return {
        "up_stay_down": vector[:3],
        "top_action": top,
        "top_action_label": ACTION_LABELS[top],
        "top_minus_second": float(vector[top] - vector[second]),
        "down_minus_best_not_down": float(vector[2] - max(vector[0], vector[1])),
        "is_top_tie_at_1e_12": bool(
            len(order) > 1 and abs(vector[top] - vector[second]) <= 1e-12
        ),
    }


def _mcts_for_states(
    *,
    checkpoint_path: Path,
    checkpoint_label: str,
    state_rows: list[dict[str, Any]],
    env: str,
    feature_mode: str,
    config_opponent_policy: str,
    seed: int,
    max_env_step: int,
    num_simulations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch

    from curvyzero.training.dummy_pong import ACTION_LABELS
    from curvyzero.training.lightzero_dummy_pong_policy import (
        load_lightzero_mcts_eval_mode_checkpoint,
    )

    spec = load_lightzero_mcts_eval_mode_checkpoint(
        policy_id=f"lightzero_{checkpoint_label}_mcts{num_simulations}",
        checkpoint_path=checkpoint_path,
        env=env,
        feature_mode=feature_mode,
        opponent_policy=config_opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        num_simulations=num_simulations,
    )
    policy = spec.policy._policy
    rows = []
    action_counts = Counter()
    for state in state_rows:
        obs_tensor = torch.as_tensor(
            [state["tabular_features"]["values"]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = policy.eval_mode.forward(
                obs_tensor,
                action_mask=np.ones((1, 3), dtype=np.float32),
                to_play=[-1],
                ready_env_id=np.asarray([0]),
            )
        action = _extract_action(output)
        action_counts[int(action)] += 1
        compact = _compact_mcts_output(output)
        visit_field, visit_vector = _visit_vector(compact)
        prior_vector = _flat_action_vector(compact.get("predicted_policy_logits"))
        rows.append(
            {
                "state_id": state["state_id"],
                "num_simulations": int(num_simulations),
                "selected_action": int(action),
                "selected_action_label": ACTION_LABELS[int(action)],
                "visit_field": visit_field,
                "visit_summary": _vector_summary(visit_vector),
                "predicted_policy_logit_summary": _vector_summary(prior_vector),
                "mcts_output": compact,
            }
        )
    summary = {
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "num_simulations": int(num_simulations),
        "checkpoint_metadata": spec.checkpoint_metadata,
        "selected_counts_up_stay_down": [int(action_counts[i]) for i in range(3)],
    }
    return summary, {
        "checkpoint_label": checkpoint_label,
        "num_simulations": int(num_simulations),
        "rows": rows,
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=30 * 60)
def run_contact_pressure_oracle(
    checkpoint_refs: str = DEFAULT_CHECKPOINT_REFS,
    state_seeds: str = DEFAULT_STATE_SEEDS,
    num_simulations: str = DEFAULT_NUM_SIMULATIONS,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    eval_id: str = "state-action-oracle-contact-pressure",
    output_ref: str | None = None,
    pressure_agent: str = "player_0",
    max_env_step: int = 64,
    lightzero_env: str = "dummy_pong_lag1",
    feature_mode: str = "tabular_ego",
    config_opponent_policy: str = "random_uniform",
    seed: int = 71,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    output_path = runs.volume_path(
        RUNS_MOUNT,
        _output_ref(
            output_ref=output_ref,
            run_id=run_id,
            attempt_id=attempt_id,
            eval_id=eval_id,
        ),
    )
    try:
        checkpoint_inputs = []
        checkpoints = []
        for checkpoint_label, ref in _parse_checkpoint_refs(checkpoint_refs):
            path = runs.volume_path(RUNS_MOUNT, ref)
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint file not found: {path}")
            checkpoint_inputs.append(
                {
                    "checkpoint_label": checkpoint_label,
                    "ref": ref,
                    "file": runs.file_summary(path, mount=RUNS_MOUNT),
                }
            )
            checkpoints.append((checkpoint_label, path))

        state_rows = [
            _state_probe_row(
                seed=state_seed,
                pressure_agent=pressure_agent,
                max_env_step=max_env_step,
            )
            for state_seed in _parse_ints(state_seeds, label="state_seeds")
        ]
        policy_logit_summaries = []
        policy_logits = []
        mcts_summaries = []
        mcts_evals = []
        for checkpoint_label, checkpoint_path in checkpoints:
            logits_summary, logits_rows = _policy_logits_for_states(
                checkpoint_path=checkpoint_path,
                checkpoint_label=checkpoint_label,
                state_rows=state_rows,
                env=lightzero_env,
                feature_mode=feature_mode,
                config_opponent_policy=config_opponent_policy,
                seed=seed,
                max_env_step=max_env_step,
            )
            policy_logit_summaries.append(logits_summary)
            policy_logits.append(logits_rows)
            for sims in _parse_ints(num_simulations, label="num_simulations"):
                mcts_summary, mcts_rows = _mcts_for_states(
                    checkpoint_path=checkpoint_path,
                    checkpoint_label=checkpoint_label,
                    state_rows=state_rows,
                    env=lightzero_env,
                    feature_mode=feature_mode,
                    config_opponent_policy=config_opponent_policy,
                    seed=seed,
                    max_env_step=max_env_step,
                    num_simulations=sims,
                )
                mcts_summaries.append(mcts_summary)
                mcts_evals.append(mcts_rows)

        result = {
            "schema": "curvyzero_lightzero_dummy_pong_contact_pressure_oracle/v0",
            "ok": True,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": {
                "app_name": APP_NAME,
                "volume_name": VOLUME_NAME,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "checkpoint_refs": checkpoint_refs,
                "state_seeds": state_seeds,
                "num_simulations": num_simulations,
                "pressure_agent": pressure_agent,
                "opponent_policy_for_direct_env": "lagged_track_ball_1",
                "max_env_step": max_env_step,
                "lightzero_env": lightzero_env,
                "feature_mode": feature_mode,
                "config_opponent_policy_for_checkpoint_reconstruction": (
                    config_opponent_policy
                ),
                "seed": seed,
                "modal_task_id": os.environ.get("MODAL_TASK_ID"),
                "no_training": True,
                "reward": "PongEnv sparse score reward only",
            },
            "action_labels": ["up", "stay", "down"],
            "feature_names": list(FEATURE_NAMES),
            "checkpoint_inputs": checkpoint_inputs,
            "states": state_rows,
            "policy_logit_summaries": policy_logit_summaries,
            "policy_logits": policy_logits,
            "mcts_summaries": mcts_summaries,
            "mcts_evals": mcts_evals,
        }
        write_summary = runs.write_json(output_path, result)
        runs_volume.commit()
        result["output"] = {
            "result_ref": runs.file_ref(output_path, mount=RUNS_MOUNT),
            "result_path": str(output_path),
            "file": write_summary,
            "committed": True,
        }
        return result
    except Exception as exc:
        result = {
            "schema": "curvyzero_lightzero_dummy_pong_contact_pressure_oracle/v0",
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": {
                "app_name": APP_NAME,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "checkpoint_refs": checkpoint_refs,
                "state_seeds": state_seeds,
                "num_simulations": num_simulations,
                "pressure_agent": pressure_agent,
                "max_env_step": max_env_step,
            },
            **_exception_result(exc),
        }
        runs.write_json(output_path, result)
        runs_volume.commit()
        result["output"] = {
            "result_ref": runs.file_ref(output_path, mount=RUNS_MOUNT),
            "result_path": str(output_path),
            "committed": True,
        }
        return result


@app.local_entrypoint()
def main(
    checkpoint_refs: str = DEFAULT_CHECKPOINT_REFS,
    state_seeds: str = DEFAULT_STATE_SEEDS,
    num_simulations: str = DEFAULT_NUM_SIMULATIONS,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    eval_id: str = "state-action-oracle-contact-pressure",
    output_ref: str | None = None,
    pressure_agent: str = "player_0",
    max_env_step: int = 64,
    lightzero_env: str = "dummy_pong_lag1",
    feature_mode: str = "tabular_ego",
    config_opponent_policy: str = "random_uniform",
    seed: int = 71,
) -> None:
    started = time.perf_counter()
    result = run_contact_pressure_oracle.remote(
        checkpoint_refs=checkpoint_refs,
        state_seeds=state_seeds,
        num_simulations=num_simulations,
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=eval_id,
        output_ref=output_ref,
        pressure_agent=pressure_agent,
        max_env_step=max_env_step,
        lightzero_env=lightzero_env,
        feature_mode=feature_mode,
        config_opponent_policy=config_opponent_policy,
        seed=seed,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
