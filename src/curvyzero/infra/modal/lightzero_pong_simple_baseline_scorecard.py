"""Official ALE/LightZero Atari Pong simple baseline scorecard.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_simple_baseline_scorecard

This evaluates simple control policies in the same stock LightZero/ALE
``PongNoFrameskip-v4`` environment/config lane used by the official visual Pong
checkpoint eval smoke. It intentionally does not import or use any custom dummy
Pong policies.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_atari_rom_image import build_lightzero_atari_rom_image
from curvyzero.infra.modal.lightzero_pong_checkpoint_probe import (
    DEFAULT_ATTEMPT_ID,
    DEFAULT_RUN_ID,
    _summarize_value,
)
from curvyzero.infra.modal.lightzero_pong_dry_config_smoke import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTOR_ENV_NUM,
    DEFAULT_ENV_ID,
    DEFAULT_EVALUATOR_ENV_NUM,
    DEFAULT_MAX_ENV_STEP,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_SEED,
    DEFAULT_UPDATE_PER_COLLECT,
    LIGHTZERO_VERSION,
)
from curvyzero.infra.modal.lightzero_pong_eval_smoke import (
    _discover_action_meanings,
    _float_or_plain,
    _gym_action_meanings,
    _timestep_parts,
)
from curvyzero.infra.modal.lightzero_pong_tiny_train_smoke import (
    DEFAULT_GAME_SEGMENT_LENGTH,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_MAX_TRAIN_ITER,
    TASK_ID,
    VOLUME_NAME,
    _patched_stock_atari_pong_configs,
    _runtime_compute_summary,
    _to_plain,
    _version_or_missing,
)


APP_NAME = "curvyzero-lightzero-pong-simple-baseline-scorecard"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_EPISODES = 1
DEFAULT_MAX_EVAL_STEPS = 512
DEFAULT_EVAL_ID = "official-pong-simple-baselines"
DEFAULT_POLICY_SET = "random_legal,noop,fixed_actions"

image = (
    build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _safe_policy_suffix(text: str) -> str:
    cleaned = "".join(
        char.lower() if char.isalnum() else "_" for char in text
    ).strip("_")
    return cleaned or "action"


def _default_output_ref(
    *,
    run_id: str,
    attempt_id: str,
    eval_id: str,
    max_eval_steps: int,
    episodes: int,
    seed: int,
) -> str:
    return (
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, eval_id)
        / (
            "lightzero_visual_pong_simple_baselines"
            f"_episodes{episodes}_steps{max_eval_steps}_seed{seed}_{runs.utc_stamp()}.json"
        )
    ).as_posix()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _round_or_none(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _extract_meanings(action_meanings: dict[str, Any], action_space_size: int) -> list[str]:
    for key in ("env_chain", "gym_direct"):
        candidate = action_meanings.get(key)
        if isinstance(candidate, dict) and candidate.get("ok"):
            meanings = candidate.get("meanings")
            if isinstance(meanings, list) and meanings:
                return [str(item) for item in meanings]
    return [f"ACTION_{index}" for index in range(action_space_size)]


def _noop_action(meanings: list[str]) -> tuple[int, str]:
    for index, meaning in enumerate(meanings):
        if meaning.upper() == "NOOP":
            return index, "action_meaning_NOOP"
    return 0, "fallback_action_0_no_noop_meaning_found"


def _policy_specs(*, policy_set: str, meanings: list[str]) -> list[dict[str, Any]]:
    requested = {item.strip() for item in policy_set.split(",") if item.strip()}
    specs: list[dict[str, Any]] = []
    if "random_legal" in requested:
        specs.append(
            {
                "policy": "random_legal",
                "kind": "random_legal",
                "description": "uniform random over the current LightZero action_mask",
            }
        )
    if "noop" in requested:
        action, source = _noop_action(meanings)
        specs.append(
            {
                "policy": "noop",
                "kind": "fixed_action",
                "action": action,
                "action_meaning": meanings[action] if action < len(meanings) else None,
                "action_source": source,
            }
        )
    if "fixed_actions" in requested:
        for action, meaning in enumerate(meanings):
            specs.append(
                {
                    "policy": f"fixed_action_{action}_{_safe_policy_suffix(meaning)}",
                    "kind": "fixed_action",
                    "action": action,
                    "action_meaning": meaning,
                    "action_source": "enumerated_action_meanings",
                }
            )
    unknown = sorted(requested - {"random_legal", "noop", "fixed_actions"})
    if unknown:
        raise ValueError(
            "unknown policy_set entries: "
            + ", ".join(unknown)
            + "; expected random_legal, noop, fixed_actions"
        )
    return specs


def _legal_actions(observation: Any, *, action_space_size: int) -> list[int]:
    if isinstance(observation, dict) and "action_mask" in observation:
        mask = observation["action_mask"]
        try:
            values = list(mask.tolist()) if hasattr(mask, "tolist") else list(mask)
            legal = [index for index, value in enumerate(values) if float(value) > 0.0]
            if legal:
                return legal
        except Exception:
            pass
    return list(range(action_space_size))


def _select_baseline_action(
    *,
    spec: dict[str, Any],
    observation: Any,
    rng: Any,
    action_space_size: int,
) -> tuple[int, dict[str, Any]]:
    legal = _legal_actions(observation, action_space_size=action_space_size)
    if spec["kind"] == "random_legal":
        action = int(rng.choice(legal))
        return action, {"legal_actions": legal, "selected_from_legal": True}
    if spec["kind"] == "fixed_action":
        action = int(spec["action"])
        return action, {
            "legal_actions": legal,
            "selected_from_legal": action in legal,
            "fixed_action_meaning": spec.get("action_meaning"),
        }
    raise ValueError(f"unsupported policy spec kind: {spec['kind']!r}")


def _make_env(
    *,
    main_config: Any,
    create_config: Any,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    from ding.config import compile_config
    from ding.envs import get_vec_env_setting

    cfg = compile_config(
        copy.deepcopy(main_config),
        seed=seed,
        auto=True,
        create_cfg=copy.deepcopy(create_config),
        save_cfg=False,
    )
    if hasattr(cfg.env, "evaluator_env_num"):
        cfg.env.evaluator_env_num = 1
    if hasattr(cfg.env, "n_evaluator_episode"):
        cfg.env.n_evaluator_episode = 1
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    one_env_cfg = evaluator_env_cfg[0] if evaluator_env_cfg else collector_env_cfg[0]
    try:
        env = env_fn(cfg=one_env_cfg)
    except TypeError:
        env = env_fn(one_env_cfg)
    if hasattr(env, "seed"):
        env.seed(seed, dynamic_seed=False)
    surface = {
        "compiled_env": {
            "env_id": str(cfg.env.env_id),
            "collector_env_num": _to_plain(getattr(cfg.env, "collector_env_num", None)),
            "evaluator_env_num": _to_plain(getattr(cfg.env, "evaluator_env_num", None)),
            "n_evaluator_episode": _to_plain(getattr(cfg.env, "n_evaluator_episode", None)),
            "eval_max_episode_steps": _to_plain(getattr(cfg.env, "eval_max_episode_steps", None)),
        },
        "compiled_policy_surface": {
            "model_type": str(cfg.policy.model.model_type),
            "observation_shape": _to_plain(cfg.policy.model.observation_shape),
            "action_space_size": int(cfg.policy.model.action_space_size),
        },
        "env_factory": {
            "env_fn": getattr(env_fn, "__module__", "") + "." + getattr(env_fn, "__name__", repr(env_fn)),
            "collector_env_cfg_count": len(collector_env_cfg),
            "evaluator_env_cfg_count": len(evaluator_env_cfg),
            "env_type": type(env).__module__ + "." + type(env).__name__,
            "action_meanings": {
                "env_chain": _discover_action_meanings(env),
                "gym_direct": _gym_action_meanings(str(cfg.env.env_id)),
            },
            "action_space": _summarize_value(getattr(env, "action_space", None)),
            "observation_space": _summarize_value(getattr(env, "observation_space", None)),
        },
    }
    return env, surface


def _run_one_episode(
    *,
    env: Any,
    spec: dict[str, Any],
    seed: int,
    episode_index: int,
    max_eval_steps: int,
    step_detail_limit: int | None,
    action_space_size: int,
) -> dict[str, Any]:
    import numpy as np

    episode_seed = seed + episode_index
    if hasattr(env, "seed"):
        try:
            env.seed(episode_seed, dynamic_seed=False)
        except TypeError:
            env.seed(episode_seed)
    rng = np.random.default_rng(episode_seed)
    observation = env.reset()
    total_reward = 0.0
    step_summaries: list[dict[str, Any]] = []
    step_details: list[dict[str, Any]] = []
    done = False

    for step_index in range(max_eval_steps):
        action, action_info = _select_baseline_action(
            spec=spec,
            observation=observation,
            rng=rng,
            action_space_size=action_space_size,
        )
        timestep = env.step(action)
        next_observation, reward, done, info = _timestep_parts(timestep)
        reward_float = float(_float_or_plain(reward))
        total_reward += reward_float
        summary = {
            "step_index": step_index,
            "action": action,
            "reward": reward_float,
            "done": bool(done),
            "selected_from_legal": bool(action_info["selected_from_legal"]),
            "info": _to_plain(info),
        }
        step_summaries.append(summary)
        if step_detail_limit is None or len(step_details) < step_detail_limit:
            step_details.append(
                {
                    **summary,
                    "legal_actions": action_info["legal_actions"],
                    "before_observation": _summarize_value(observation),
                    "after_observation": _summarize_value(next_observation),
                }
            )
        observation = next_observation
        if done:
            break

    actions = [int(step["action"]) for step in step_summaries]
    rewards = [float(step["reward"]) for step in step_summaries]
    nonzero_reward_steps = [
        {"step_index": int(step["step_index"]), "reward": float(step["reward"])}
        for step in step_summaries
        if float(step["reward"]) != 0.0
    ]
    illegal_action_steps = [
        {"step_index": int(step["step_index"]), "action": int(step["action"])}
        for step in step_summaries
        if not bool(step["selected_from_legal"])
    ]
    steps_survived = len(step_summaries)
    return {
        "episode_index": episode_index,
        "seed": episode_seed,
        "return": total_reward,
        "steps_survived": steps_survived,
        "episode_length": steps_survived,
        "survived_to_cap": bool(steps_survived >= max_eval_steps),
        "done": bool(done),
        "actions": actions,
        "action_histogram": dict(sorted(Counter(actions).items())),
        "rewards": rewards,
        "reward_histogram": dict(sorted(Counter(rewards).items())),
        "nonzero_reward_steps": nonzero_reward_steps,
        "illegal_action_steps": illegal_action_steps,
        "step_summaries": step_summaries,
        "steps": step_details,
        "steps_truncated": bool(
            step_detail_limit is not None and len(step_details) < len(step_summaries)
        ),
    }


def _row_from_episodes(
    *,
    spec: dict[str, Any],
    episodes: list[dict[str, Any]],
    max_eval_steps: int,
    env_id: str,
    action_meanings: list[str],
) -> dict[str, Any]:
    returns = [float(episode["return"]) for episode in episodes]
    lengths = [float(episode["episode_length"]) for episode in episodes]
    steps = [float(episode["steps_survived"]) for episode in episodes]
    action_counter: Counter[int] = Counter()
    reward_counter: Counter[float] = Counter()
    nonzero_reward_count = 0
    illegal_action_count = 0
    for episode in episodes:
        action_counter.update({int(key): int(value) for key, value in episode["action_histogram"].items()})
        reward_counter.update({float(key): int(value) for key, value in episode["reward_histogram"].items()})
        nonzero_reward_count += len(episode["nonzero_reward_steps"])
        illegal_action_count += len(episode["illegal_action_steps"])
    return {
        "baseline_policy": spec["policy"],
        "return_mean": _round_or_none(_mean(returns)),
        "return_std": _round_or_none(_std(returns)),
        "return_min": min(returns) if returns else None,
        "return_max": max(returns) if returns else None,
        "steps_survived_mean": _round_or_none(_mean(steps)),
        "episode_length_mean": _round_or_none(_mean(lengths)),
        "survival_rate": _round_or_none(_mean([value / max_eval_steps for value in steps])),
        "episodes": len(episodes),
        "max_eval_steps": max_eval_steps,
        "env_id": env_id,
        "kind": spec["kind"],
        "fixed_action": spec.get("action"),
        "fixed_action_meaning": spec.get("action_meaning"),
        "action_meanings": action_meanings,
        "action_histogram": dict(sorted(action_counter.items())),
        "reward_histogram": dict(sorted(reward_counter.items())),
        "nonzero_reward_count": nonzero_reward_count,
        "illegal_action_count": illegal_action_count,
        "episode_returns": returns,
        "episode_lengths": [int(value) for value in lengths],
        "steps_survived": [int(value) for value in steps],
    }


def _run_policy(
    *,
    spec: dict[str, Any],
    main_config: Any,
    create_config: Any,
    seed: int,
    episodes: int,
    max_eval_steps: int,
    step_detail_limit: int | None,
    action_space_size: int,
    env_id: str,
    action_meanings: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    env = None
    try:
        env, surface = _make_env(
            main_config=main_config,
            create_config=create_config,
            seed=seed,
        )
        episode_rows = [
            _run_one_episode(
                env=env,
                spec=spec,
                seed=seed,
                episode_index=index,
                max_eval_steps=max_eval_steps,
                step_detail_limit=step_detail_limit,
                action_space_size=action_space_size,
            )
            for index in range(episodes)
        ]
        row = _row_from_episodes(
            spec=spec,
            episodes=episode_rows,
            max_eval_steps=max_eval_steps,
            env_id=env_id,
            action_meanings=action_meanings,
        )
        return {
            "ok": True,
            "policy": spec["policy"],
            "spec": spec,
            "row": row,
            "episodes": episode_rows,
            "surface": surface,
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:  # pragma: no cover - remote runtime diagnosis.
        return {
            "ok": False,
            "policy": spec.get("policy"),
            "spec": spec,
            "row": {
                "baseline_policy": spec.get("policy"),
                "return_mean": None,
                "steps_survived_mean": None,
                "episode_length_mean": None,
                "episodes": 0,
            },
            "error": _exception_result(exc),
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass


def _run_scorecard(
    *,
    run_id: str,
    attempt_id: str,
    output_ref: str | None,
    eval_id: str,
    env_id: str,
    seed: int,
    episodes: int,
    max_env_step: int,
    max_train_iter: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    max_episode_steps: int,
    game_segment_length: int,
    max_eval_steps: int,
    step_detail_limit: int | None,
    policy_set: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    output_ref = output_ref or _default_output_ref(
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=eval_id,
        max_eval_steps=max_eval_steps,
        episodes=episodes,
        seed=seed,
    )
    output_path = runs.volume_path(RUNS_MOUNT, output_ref)
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "gym": _version_or_missing("gym"),
        "gymnasium": _version_or_missing("gymnasium"),
        "ale-py": _version_or_missing("ale-py", "ale_py"),
        "AutoROM": _version_or_missing("AutoROM"),
    }
    config = {
        "job_kind": "lightzero_official_visual_pong_simple_baseline_scorecard",
        "lane": "official_control_lightzero_atari_pong",
        "claim": (
            "Simple baselines are evaluated only in the official ALE-backed "
            "LightZero Atari Pong env/config lane."
        ),
        "non_claim": (
            "These rows are not dummy Pong baselines and do not use privileged "
            "track-ball or project-owned dummy Pong state."
        ),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "output_ref": output_ref,
        "eval_id": eval_id,
        "env_id": env_id,
        "seed": seed,
        "episodes": episodes,
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "update_per_collect": update_per_collect,
        "max_episode_steps": max_episode_steps,
        "game_segment_length": game_segment_length,
        "max_eval_steps": max_eval_steps,
        "step_detail_limit": step_detail_limit,
        "policy_set": policy_set,
        "runtime_compute": _runtime_compute_summary(requested_compute="cpu"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        patched = _patched_stock_atari_pong_configs(
            env_id=env_id,
            seed=seed,
            run_id=run_id,
            attempt_id=attempt_id,
            max_env_step=max_env_step,
            max_train_iter=max_train_iter,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            num_simulations=num_simulations,
            batch_size=batch_size,
            update_per_collect=update_per_collect,
            max_episode_steps=max_episode_steps,
            game_segment_length=game_segment_length,
            use_cuda=False,
        )
        probe_env, probe_surface = _make_env(
            main_config=patched["main_config"],
            create_config=patched["create_config"],
            seed=seed,
        )
        try:
            action_space_size = int(patched["patched_surface"]["action_space_size"])
            action_meanings = probe_surface["env_factory"]["action_meanings"]
            meanings = _extract_meanings(action_meanings, action_space_size)
        finally:
            if hasattr(probe_env, "close"):
                probe_env.close()
        specs = _policy_specs(policy_set=policy_set, meanings=meanings)
        policy_results = [
            _run_policy(
                spec=spec,
                main_config=patched["main_config"],
                create_config=patched["create_config"],
                seed=seed,
                episodes=episodes,
                max_eval_steps=max_eval_steps,
                step_detail_limit=step_detail_limit,
                action_space_size=action_space_size,
                env_id=env_id,
                action_meanings=meanings,
            )
            for spec in specs
        ]
        result: dict[str, Any] = {
            "schema": "curvyzero_lightzero_visual_pong_simple_baseline_scorecard/v0",
            "ok": all(bool(item.get("ok")) for item in policy_results),
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "stock_example": {
                "task": env_id,
                "algorithm": "MuZero env/config only; no checkpoint policy",
                "module": patched["module"],
                "original_surface": patched["original_surface"],
                "patched_surface": patched["patched_surface"],
                "patches": patched["patches"],
            },
            "surface": probe_surface,
            "action_meanings": meanings,
            "table_fields": [
                "baseline_policy",
                "return_mean",
                "return_std",
                "steps_survived_mean",
                "episode_length_mean",
                "survival_rate",
                "episodes",
                "max_eval_steps",
                "env_id",
                "kind",
                "fixed_action",
                "fixed_action_meaning",
                "action_meanings",
                "action_histogram",
                "nonzero_reward_count",
                "illegal_action_count",
            ],
            "table": [item["row"] for item in policy_results],
            "policy_results": policy_results,
        }
    except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
        result = {
            "schema": "curvyzero_lightzero_visual_pong_simple_baseline_scorecard/v0",
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "wrapper_error": _exception_result(exc),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(output_path, _to_plain(result))
    result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60, cpu=1.0)
def lightzero_pong_simple_baseline_scorecard(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    output_ref: str | None = None,
    eval_id: str = DEFAULT_EVAL_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    episodes: int = DEFAULT_EPISODES,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = 4,
    policy_set: str = DEFAULT_POLICY_SET,
) -> dict[str, Any]:
    return _run_scorecard(
        run_id=run_id,
        attempt_id=attempt_id,
        output_ref=output_ref,
        eval_id=eval_id,
        env_id=env_id,
        seed=seed,
        episodes=episodes,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        policy_set=policy_set,
    )


@app.local_entrypoint()
def main(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    output_ref: str | None = None,
    eval_id: str = DEFAULT_EVAL_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    episodes: int = DEFAULT_EPISODES,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = 4,
    policy_set: str = DEFAULT_POLICY_SET,
) -> None:
    result = lightzero_pong_simple_baseline_scorecard.remote(
        run_id=run_id,
        attempt_id=attempt_id,
        output_ref=output_ref,
        eval_id=eval_id,
        env_id=env_id,
        seed=seed,
        episodes=episodes,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        policy_set=policy_set,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
