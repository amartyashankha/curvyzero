"""Strict stock-evaluator eval for LightZero MuZeroAgent 96x96 Pong checkpoints.

This module is intentionally separate from ``lightzero_pong_eval_smoke``.  The
smoke eval loads the 64x64 ``zoo.atari.config.atari_muzero_config`` surface.
This eval loads the 96x96 ``lzero.agent.config.muzero.supported_env_cfg`` surface
used by ``lightzero_pong_muzero_agent_reproduction`` and compares checkpoints
from one Agent96 run against that run's ``iteration_0`` baseline.
"""

from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib
import json
import math
import os
import random
import re
import tempfile
import time
import traceback
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_atari_rom_image import build_lightzero_atari_rom_image
from curvyzero.infra.modal.lightzero_pong_checkpoint_probe import (
    _checkpoint_summary,
    _find_state_dict,
    _load_state_dict_probe,
    _summarize_value,
    _torch_load,
)
from curvyzero.infra.modal.lightzero_pong_dry_config_smoke import LIGHTZERO_VERSION
from curvyzero.infra.modal.lightzero_pong_muzero_agent_reproduction import (
    DEFAULT_ATTEMPT_ID,
    DEFAULT_ENV_ID,
    DEFAULT_RUN_ID,
    DEFAULT_SEED,
    TASK_ID as AGENT96_TASK_ID,
    VOLUME_NAME,
)

APP_NAME = "curvyzero-lightzero-pong-muzero-agent96-eval"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_COMPUTE = "gpu-l4-t4-cpu8"
DEFAULT_SELECTED_ITERATIONS = "0,1000"
DEFAULT_EVAL_ID = "agent96_strict_stock_curve"
DEFAULT_EVAL_SEED_COUNT = 8
DEFAULT_EVAL_SEED_RNG_SEED = 20260511
DEFAULT_MAX_EVAL_STEPS = 512
DEFAULT_STEP_RECORD_LIMIT = 64
EVAL_SEED_MIN = 0
EVAL_SEED_MAX = (2**31) - 1
EVAL_FUNCTION_TIMEOUT_SEC = 18 * 60
CHEAP_GPU_RESOURCE = ["L4", "T4"]

image = (
    build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_to_plain(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-16:],
    }


@contextlib.contextmanager
def _quiet_framework_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _runtime_compute_summary(*, requested_compute: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested_compute": requested_compute,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        summary.update(
            {
                "torch_cuda_available": cuda_available,
                "torch_cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            }
        )
        if cuda_available:
            device = int(torch.cuda.current_device())
            summary.update(
                {
                    "torch_cuda_current_device": device,
                    "torch_cuda_device_name": torch.cuda.get_device_name(device),
                    "torch_cuda_capability": list(torch.cuda.get_device_capability(device)),
                }
            )
    except Exception as exc:
        summary["torch_cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _agent96_checkpoint_template(*, run_id: str, attempt_id: str) -> str:
    return (
        runs.attempt_train_ref(AGENT96_TASK_ID, run_id, attempt_id)
        / "agent_exp"
        / "ckpt"
        / "iteration_{iteration}.pth.tar"
    ).as_posix()


def _safe_generated_id(raw: str, *, fallback: str) -> str:
    cleaned = "".join(char if char in runs.SAFE_ID_CHARS else "_" for char in raw).strip("._-")
    if not cleaned:
        cleaned = fallback
    if not cleaned[0].isalnum():
        cleaned = f"{fallback}_{cleaned}"
    return runs.clean_id(cleaned[:80], label=fallback)


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_iterations(value: str | None) -> list[int]:
    iterations = []
    for item in _split_csv(value):
        try:
            iteration = int(item)
        except ValueError as exc:
            raise ValueError(f"selected iteration must be an int, got {item!r}") from exc
        if iteration < 0:
            raise ValueError("selected iterations must be non-negative")
        iterations.append(iteration)
    return iterations


def _infer_checkpoint_template(checkpoint_ref: str) -> str:
    template = re.sub(
        r"iteration_\d+(\.pth\.tar|\.tar|\.pth|\.pt|\.bin)$",
        r"iteration_{iteration}\1",
        checkpoint_ref,
    )
    if template == checkpoint_ref:
        raise ValueError(
            "selected_iterations requires checkpoint_ref_template when checkpoint_ref "
            "does not end with iteration_<n>.pth.tar"
        )
    return template


def _selected_checkpoint_refs(
    *,
    run_id: str,
    attempt_id: str,
    checkpoint_ref: str | None,
    checkpoint_refs: str | None,
    checkpoint_ref_template: str | None,
    selected_iterations: str | None,
) -> list[str]:
    explicit_refs = _split_csv(checkpoint_refs)
    iterations = _parse_iterations(selected_iterations)
    if explicit_refs and (checkpoint_ref or checkpoint_ref_template or iterations):
        raise ValueError("use either checkpoint_refs or a single/template checkpoint selection")
    if explicit_refs:
        return explicit_refs
    if iterations:
        template = (
            checkpoint_ref_template
            or (_infer_checkpoint_template(checkpoint_ref) if checkpoint_ref else None)
            or _agent96_checkpoint_template(run_id=run_id, attempt_id=attempt_id)
        )
        return [template.format(iteration=iteration) for iteration in iterations]
    if checkpoint_ref_template:
        raise ValueError("checkpoint_ref_template requires selected_iterations")
    if checkpoint_ref:
        return [checkpoint_ref]
    return [_agent96_checkpoint_template(run_id=run_id, attempt_id=attempt_id).format(iteration=0)]


def _checkpoint_label(checkpoint_ref: str, *, index: int) -> str:
    name = Path(checkpoint_ref).name
    for suffix in (".pth.tar", ".tar", ".pth", ".pt", ".bin"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return _safe_generated_id(name or f"checkpoint_{index:03d}", fallback=f"checkpoint_{index:03d}")


def _parse_eval_seeds(
    *,
    seed: int,
    eval_seeds: str | None,
    eval_seed_count: int | None,
    eval_seed_rng_seed: int,
) -> tuple[list[int], int | None]:
    if eval_seeds is None:
        if eval_seed_count is None:
            return [seed], None
        if eval_seed_count < 1:
            raise ValueError("eval_seed_count must be >= 1")
        rng = random.Random(eval_seed_rng_seed)
        return (
            rng.sample(range(EVAL_SEED_MIN, EVAL_SEED_MAX + 1), eval_seed_count),
            eval_seed_rng_seed,
        )
    if eval_seed_count is not None:
        raise ValueError("eval_seeds cannot be combined with eval_seed_count")
    parsed: list[int] = []
    for item in _split_csv(eval_seeds):
        try:
            parsed_seed = int(item)
        except ValueError as exc:
            raise ValueError(f"eval seed must be an int, got {item!r}") from exc
        if parsed_seed not in parsed:
            parsed.append(parsed_seed)
    if not parsed:
        raise ValueError("eval_seeds must include at least one integer")
    return parsed, None


def _seed_panel_label(seeds: list[int]) -> str:
    payload = ",".join(str(seed) for seed in seeds)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"n{len(seeds)}_{digest}"


def _checkpoint_sort_key(label: Any) -> tuple[int, int | str]:
    text = str(label)
    match = re.search(r"iteration_(\d+)", text)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, text)


def _agent96_config_surface(cfg: Any) -> dict[str, Any]:
    main_config = cfg.main_config
    create_config = cfg.create_config
    policy = main_config.policy
    env = main_config.env
    model = policy.model
    return {
        "env_id": env.env_id,
        "env_type": create_config.env.type,
        "env_import_names": _to_plain(create_config.env.get("import_names")),
        "env_manager_type": create_config.env_manager.type,
        "policy_type": create_config.policy.type,
        "policy_import_names": _to_plain(create_config.policy.get("import_names")),
        "model_type": model.get("model_type", "conv"),
        "observation_shape": _to_plain(model.observation_shape),
        "env_obs_shape": _to_plain(env.obs_shape),
        "action_space_size": int(model.action_space_size),
        "downsample": bool(model.get("downsample", False)),
        "self_supervised_learning_loss": bool(model.get("self_supervised_learning_loss", False)),
        "collector_env_num": int(env.collector_env_num),
        "evaluator_env_num": int(env.evaluator_env_num),
        "n_evaluator_episode": int(env.n_evaluator_episode),
        "num_simulations": int(policy.num_simulations),
        "batch_size": int(policy.batch_size),
        "update_per_collect": int(policy.update_per_collect),
        "game_segment_length": int(policy.game_segment_length),
        "eval_freq": int(policy.eval_freq),
        "cuda": bool(policy.cuda),
    }


def _validate_agent96_surface(surface: dict[str, Any]) -> list[str]:
    expected = {
        "env_id": "PongNoFrameskip-v4",
        "env_type": "atari_lightzero",
        "env_import_names": ["zoo.atari.envs.atari_lightzero_env"],
        "env_manager_type": "subprocess",
        "policy_type": "muzero",
        "policy_import_names": ["lzero.policy.muzero"],
        "model_type": "conv",
        "observation_shape": [4, 96, 96],
        "env_obs_shape": [4, 96, 96],
        "action_space_size": 6,
        "downsample": True,
        "self_supervised_learning_loss": True,
    }
    problems = []
    for key, expected_value in expected.items():
        if surface.get(key) != expected_value:
            problems.append(
                f"Agent96 eval surface {key}={surface.get(key)!r}, expected {expected_value!r}"
            )
    return problems


def _patched_agent96_configs(
    *,
    env_id: str,
    use_cuda: bool,
    max_eval_steps: int,
) -> dict[str, Any]:
    config_module = importlib.import_module("lzero.agent.config.muzero")
    supported_env_cfg = getattr(config_module, "supported_env_cfg")
    if env_id not in supported_env_cfg:
        raise ValueError(f"{env_id!r} not present in MuZeroAgent supported_env_cfg")
    cfg = copy.deepcopy(supported_env_cfg[env_id])
    original_surface = _agent96_config_surface(cfg)
    problems = _validate_agent96_surface(original_surface)
    if problems:
        raise ValueError("; ".join(problems))

    cfg.main_config.exp_name = str(Path(tempfile.gettempdir()) / "curvyzero-agent96-eval")
    cfg.main_config.policy.cuda = use_cuda
    cfg.main_config.env.evaluator_env_num = 1
    cfg.main_config.policy.evaluator_env_num = 1
    cfg.main_config.env.n_evaluator_episode = 1
    cfg.main_config.policy.n_evaluator_episode = 1
    cfg.main_config.env.eval_max_episode_steps = max_eval_steps
    cfg.main_config.policy.eval_freq = 1
    patched_surface = _agent96_config_surface(cfg)
    patched_surface["eval_max_episode_steps"] = max_eval_steps
    patched_surface["eval_cuda"] = use_cuda
    return {
        "module": "lzero.agent.config.muzero",
        "config_key": env_id,
        "main_config": cfg.main_config,
        "create_config": cfg.create_config,
        "original_surface": original_surface,
        "patched_surface": patched_surface,
        "patch_scope": (
            "Eval-only patches: exp_name artifact scratch path, cuda/device, one evaluator "
            "episode/env, eval cap, and eval_freq. Observation shape/env/policy surface remains Agent96."
        ),
    }


def _float_or_plain(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return _to_plain(value)


def _timestep_parts(timestep: Any) -> tuple[Any, Any, bool, Any]:
    if hasattr(timestep, "obs"):
        return timestep.obs, timestep.reward, bool(timestep.done), timestep.info
    if isinstance(timestep, (list, tuple)) and len(timestep) >= 4:
        obs, reward, done, info = timestep[:4]
        return obs, reward, bool(done), info
    raise TypeError(f"unsupported timestep type: {type(timestep).__name__}")


def _action_int_or_none(action: Any) -> int | None:
    plain = _to_plain(action)
    if isinstance(plain, dict):
        for key in ("action", 0, "0"):
            if key in plain:
                return _action_int_or_none(plain[key])
        return None
    if isinstance(plain, (list, tuple)):
        if len(plain) == 1:
            return _action_int_or_none(plain[0])
        return None
    try:
        return int(plain)
    except (TypeError, ValueError):
        return None


def _truncation_from_info(info: Any) -> bool | None:
    plain = _to_plain(info)
    if not isinstance(plain, dict):
        return None
    for key in ("TimeLimit.truncated", "truncated", "timeout", "TimeLimit.truncation"):
        if key in plain:
            return bool(plain[key])
    return None


def _terminal_reason_from_info(info: Any, *, done: bool, truncated: bool | None) -> str | None:
    plain = _to_plain(info)
    if isinstance(plain, dict):
        for key in ("terminal_reason", "end_reason", "episode_end_reason", "reason"):
            value = plain.get(key)
            if value is not None:
                return str(value)
    if truncated is True:
        return "truncated"
    if done:
        return "done"
    return None


def _extract_eval_action(output: Any) -> int:
    plain = _to_plain(output)
    if isinstance(plain, dict):
        if "action" in plain:
            return int(plain["action"])
        for key in (0, "0"):
            if key in plain:
                return _extract_eval_action(plain[key])
    if isinstance(plain, (list, tuple)) and plain:
        return _extract_eval_action(plain[0])
    raise ValueError(f"could not extract action from policy eval output: {plain!r}")


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


def _make_recording_env_thunk(
    env_fn: Any,
    env_cfg: Any,
    records: list[dict[str, Any]],
    *,
    record_path: Path,
) -> Any:
    def _thunk() -> Any:
        cfg_copy = copy.deepcopy(env_cfg)
        try:
            inner = env_fn(cfg=cfg_copy)
        except TypeError:
            inner = env_fn(cfg_copy)

        class _RecordingEnv:
            def __init__(self, wrapped: Any) -> None:
                self._wrapped = wrapped
                self._reset_count = 0

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

            def reset(self, *args: Any, **kwargs: Any) -> Any:
                self._reset_count += 1
                return self._wrapped.reset(*args, **kwargs)

            def step(self, action: Any) -> Any:
                step_index = len(records)
                started = time.perf_counter()
                timestep = self._wrapped.step(action)
                elapsed_sec = time.perf_counter() - started
                try:
                    _obs, reward, done, info = _timestep_parts(timestep)
                    reward_float = float(_float_or_plain(reward))
                    info_plain = _to_plain(info)
                    truncated = _truncation_from_info(info_plain)
                    record = {
                        "step_index": step_index,
                        "reset_count": self._reset_count,
                        "action": _action_int_or_none(action),
                        "action_raw": _to_plain(action),
                        "reward": reward_float,
                        "done": bool(done),
                        "truncated": truncated,
                        "terminal_reason": _terminal_reason_from_info(
                            info_plain,
                            done=bool(done),
                            truncated=truncated,
                        ),
                        "info": info_plain,
                        "env_step_elapsed_sec": round(elapsed_sec, 6),
                    }
                except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
                    record = {
                        "step_index": step_index,
                        "action": _action_int_or_none(action),
                        "action_raw": _to_plain(action),
                        "recording_error": _exception_result(exc),
                        "env_step_elapsed_sec": round(elapsed_sec, 6),
                    }
                records.append(_to_plain(record))
                try:
                    with record_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(_to_plain(record), sort_keys=True) + "\n")
                except Exception:
                    pass
                return timestep

        return _RecordingEnv(inner)

    return _thunk


def _load_stock_step_records(records: list[dict[str, Any]], record_path: Path) -> list[dict[str, Any]]:
    if not record_path.exists():
        return records
    loaded = []
    with record_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                loaded.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return loaded or records


def _stock_rollout_summary(records: list[dict[str, Any]], *, max_eval_steps: int) -> dict[str, Any]:
    ok_steps = [step for step in records if "recording_error" not in step]
    actions = [int(step["action"]) for step in ok_steps if step.get("action") is not None]
    rewards = []
    for step in ok_steps:
        try:
            rewards.append(float(step.get("reward", 0.0)))
        except (TypeError, ValueError):
            pass
    terminal_steps = [step for step in ok_steps if bool(step.get("done"))]
    terminal_step = terminal_steps[-1] if terminal_steps else {}
    truncated_values = [
        bool(step["truncated"])
        for step in ok_steps
        if step.get("truncated") is not None
    ]
    return {
        "ok": bool(ok_steps) and len(ok_steps) == len(records),
        "source": "stock_env_wrapper_under_lzero_worker_MuZeroEvaluator",
        "selection_metric_rank": ["stock_survival_steps", "stock_return"],
        "max_eval_steps": max_eval_steps,
        "stock_survival_steps": len(ok_steps),
        "steps_run": len(ok_steps),
        "episode_length": len(ok_steps),
        "done": bool(terminal_steps),
        "truncated": bool(any(truncated_values)) if truncated_values else None,
        "terminal_reason": terminal_step.get("terminal_reason"),
        "terminal_info": terminal_step.get("info"),
        "stock_return": round(sum(rewards), 6) if rewards else 0.0,
        "total_reward": round(sum(rewards), 6) if rewards else 0.0,
        "nonzero_reward_count": sum(1 for reward in rewards if reward != 0.0),
        "positive_reward_count": sum(1 for reward in rewards if reward > 0.0),
        "negative_reward_count": sum(1 for reward in rewards if reward < 0.0),
        "nonzero_reward_steps": [
            {"step_index": int(step["step_index"]), "reward": float(step["reward"])}
            for step in ok_steps
            if float(step.get("reward", 0.0)) != 0.0
        ],
        "action_histogram": dict(sorted(Counter(actions).items())),
        "reward_histogram": dict(sorted(Counter(rewards).items())),
        "step_records": records[:DEFAULT_STEP_RECORD_LIMIT],
        "step_records_truncated": len(records) > DEFAULT_STEP_RECORD_LIMIT,
        "recording_error_count": len(records) - len(ok_steps),
    }


def _stock_evaluator_return(eval_output: Any) -> float | None:
    plain = _to_plain(eval_output)
    if (
        isinstance(plain, (list, tuple))
        and len(plain) >= 2
        and isinstance(plain[1], dict)
    ):
        raw = plain[1].get("eval_episode_return_mean")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return None


def _run_stock_agent96_eval(
    *,
    main_config: Any,
    create_config: Any,
    state_dict: dict[str, Any],
    seed: int,
    max_eval_steps: int,
    use_cuda: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    eval_mode_records: list[dict[str, Any]] = []
    env_manager = None
    temp_root = Path(tempfile.mkdtemp(prefix="curvyzero_agent96_stock_eval_"))
    record_path = temp_root / "stock_step_records.jsonl"
    env_manager_patch: dict[str, Any] | None = None
    try:
        from ding.config import compile_config
        from ding.envs import create_env_manager, get_vec_env_setting
        from lzero.policy.muzero import MuZeroPolicy
        from lzero.worker import MuZeroEvaluator

        cfg = compile_config(
            copy.deepcopy(main_config),
            seed=seed,
            auto=True,
            create_cfg=copy.deepcopy(create_config),
            save_cfg=False,
        )
        if not hasattr(cfg.policy, "device"):
            cfg.policy.device = "cuda" if use_cuda else "cpu"
        elif use_cuda and str(cfg.policy.device) == "cpu":
            cfg.policy.device = "cuda"
        elif not use_cuda:
            cfg.policy.device = "cpu"
        cfg.policy.cuda = use_cuda
        cfg.env.evaluator_env_num = 1
        cfg.env.n_evaluator_episode = 1
        cfg.env.eval_max_episode_steps = max_eval_steps
        manager_cfg = getattr(cfg.env, "manager", None)
        if isinstance(manager_cfg, dict):
            old_manager_type = manager_cfg.get("type")
            manager_cfg["type"] = "base"
        else:
            old_manager_type = getattr(manager_cfg, "type", None)
            if manager_cfg is not None:
                manager_cfg.type = "base"
        env_manager_patch = {
            "path": "compiled.env.manager.type",
            "old": _to_plain(old_manager_type),
            "new": "base",
            "reason": (
                "Eval uses one in-process stock Atari env so ALE banner output cannot "
                "corrupt DI-engine subprocess pipes; train config remains stock subprocess."
            ),
        }

        policy = MuZeroPolicy(cfg.policy)
        policy_model = getattr(policy, "_model", None)
        if policy_model is None:
            raise AttributeError("MuZeroPolicy has no _model attribute")
        load = _load_state_dict_probe(policy_model, state_dict)
        if hasattr(policy_model, "eval"):
            policy_model.eval()
        if not load["ok"] or not load.get("strict"):
            raise RuntimeError(f"strict policy model checkpoint load failed: {load}")

        class _RecordingEvalMode:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                output = self._inner.forward(*args, **kwargs)
                if len(eval_mode_records) < 32:
                    data = args[0] if args else kwargs.get("data")
                    eval_mode_records.append(
                        {
                            "call_index": len(eval_mode_records),
                            "data": _summarize_value(data),
                            "action_mask": _summarize_value(kwargs.get("action_mask")),
                            "action_mask_values": _to_plain(kwargs.get("action_mask")),
                            "ready_env_id": _to_plain(kwargs.get("ready_env_id")),
                            "to_play": _to_plain(kwargs.get("to_play")),
                            "action": _extract_eval_action(output),
                            "mcts_output": _compact_mcts_output(output),
                        }
                    )
                return output

        env_fn, _collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
        if not evaluator_env_cfg:
            raise RuntimeError("get_vec_env_setting returned no evaluator env configs")
        env_fns = [
            _make_recording_env_thunk(
                env_fn,
                evaluator_env_cfg[0],
                records,
                record_path=record_path,
            )
        ]
        env_manager = create_env_manager(cfg.env.manager, env_fns)
        if hasattr(env_manager, "seed"):
            env_manager.seed(seed, dynamic_seed=False)

        evaluator = MuZeroEvaluator(
            eval_freq=int(cfg.policy.eval_freq),
            n_evaluator_episode=1,
            stop_value=float(getattr(cfg.env, "stop_value", 1e6)),
            env=env_manager,
            policy=_RecordingEvalMode(policy.eval_mode),
            tb_logger=None,
            exp_name=str(temp_root),
            instance_name="curvyzero_agent96_stock_eval",
            policy_config=cfg.policy,
        )

        eval_attempts: list[dict[str, Any]] = []

        def save_ckpt_fn(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        eval_output = None
        eval_ok = False
        for label, call in (
            ("keyword_train_iter_envstep", lambda: evaluator.eval(save_ckpt_fn, train_iter=0, envstep=0)),
            ("positional_save_train_envstep", lambda: evaluator.eval(save_ckpt_fn, 0, 0)),
            ("positional_train_envstep_save", lambda: evaluator.eval(0, 0, save_ckpt_fn)),
        ):
            try:
                eval_output = call()
                eval_attempts.append({"label": label, "ok": True})
                eval_ok = True
                break
            except Exception as exc:  # pragma: no cover - remote API diagnosis.
                eval_attempts.append(
                    {
                        "label": label,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if not eval_ok:
            raise RuntimeError(f"stock evaluator eval calls failed: {eval_attempts}")
        records = _load_stock_step_records(records, record_path)
        rollout = _stock_rollout_summary(records, max_eval_steps=max_eval_steps)
        stock_return = _stock_evaluator_return(eval_output)
        if stock_return is not None:
            rollout["stock_return"] = stock_return
        return {
            "ok": True,
            "path": "lzero.worker.MuZeroEvaluator",
            "strict_policy_model_load": load,
            "eval_attempts": eval_attempts,
            "eval_output": _to_plain(eval_output),
            "stock_rollout": rollout,
            "env_manager_patch": env_manager_patch,
            "recorded_eval_mode_calls": eval_mode_records,
            "recorded_call_count": len(eval_mode_records),
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        records = _load_stock_step_records(records, record_path)
        return {
            "ok": False,
            "path": "lzero.worker.MuZeroEvaluator",
            "blocker": _exception_result(exc),
            "stock_rollout": _stock_rollout_summary(records, max_eval_steps=max_eval_steps),
            "env_manager_patch": env_manager_patch,
            "recorded_eval_mode_calls": eval_mode_records,
            "recorded_call_count": len(eval_mode_records),
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    finally:
        if env_manager is not None and hasattr(env_manager, "close"):
            try:
                env_manager.close()
            except Exception:
                pass


def _eval_one_checkpoint(
    *,
    checkpoint_ref: str,
    output_ref: str,
    run_id: str,
    attempt_id: str,
    env_id: str,
    seed: int,
    max_eval_steps: int,
    compute: str,
    use_cuda: bool,
    quiet_framework_logs: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
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
        "job_kind": "lightzero_muzero_agent96_strict_stock_pong_eval",
        "agent96_task_id": AGENT96_TASK_ID,
        "compute": compute,
        "use_cuda": use_cuda,
        "runtime_compute": _runtime_compute_summary(requested_compute=compute),
        "checkpoint_ref": checkpoint_ref,
        "output_ref": output_ref,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "env_id": env_id,
        "seed": seed,
        "max_eval_steps": max_eval_steps,
        "allow_model_fallback": False,
        "manual_rollout": False,
        "selection_metric_order": ["stock_survival_steps", "stock_return"],
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        with _quiet_framework_output(quiet_framework_logs):
            checkpoint = _torch_load(checkpoint_path)
            state_candidate = _find_state_dict(checkpoint)
            if state_candidate is None:
                raise ValueError("no tensor state dict found under common LightZero checkpoint keys")
            state_path, state_dict = state_candidate
            patched = _patched_agent96_configs(
                env_id=env_id,
                use_cuda=use_cuda,
                max_eval_steps=max_eval_steps,
            )
            stock_eval = _run_stock_agent96_eval(
                main_config=patched["main_config"],
                create_config=patched["create_config"],
                state_dict=state_dict,
                seed=seed,
                max_eval_steps=max_eval_steps,
                use_cuda=use_cuda,
            )
        stock_rollout = (
            stock_eval.get("stock_rollout")
            if isinstance(stock_eval.get("stock_rollout"), dict)
            else {}
        )
        result: dict[str, Any] = {
            "schema": "curvyzero_lightzero_muzero_agent96_strict_stock_eval/v0",
            "ok": bool(stock_eval.get("ok")),
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "checkpoint": _checkpoint_summary(checkpoint_path, include_sha256=False),
            "load": {
                "ok": True,
                "state_dict_path": state_path,
                "tensor_count": len(state_dict),
                "keys_sample": list(state_dict)[:40],
            },
            "agent96_surface": {
                "module": patched["module"],
                "config_key": patched["config_key"],
                "original_surface": patched["original_surface"],
                "patched_surface": patched["patched_surface"],
                "patch_scope": patched["patch_scope"],
                "non_mix_assertion": (
                    "This eval did not import or patch zoo.atari.config.atari_muzero_config; "
                    "it compiled the Agent96 supported_env_cfg surface."
                ),
            },
            "stock_evaluator": stock_eval,
            "status": {
                "checkpoint_load_ok": True,
                "strict_policy_model_load_ok": bool(
                    stock_eval.get("strict_policy_model_load", {}).get("strict")
                ),
                "env_reset_and_step_ok": bool(stock_rollout.get("steps_run")),
                "model_fallback_used": False,
                "stock_survival_steps": stock_rollout.get("stock_survival_steps"),
                "stock_return": stock_rollout.get("stock_return"),
            },
        }
    except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
        result = {
            "schema": "curvyzero_lightzero_muzero_agent96_strict_stock_eval/v0",
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "checkpoint": _checkpoint_summary(checkpoint_path, include_sha256=False),
            "status": {
                "checkpoint_load_ok": False,
                "strict_policy_model_load_ok": False,
                "env_reset_and_step_ok": False,
                "model_fallback_used": False,
                "stock_survival_steps": 0,
                "stock_return": None,
            },
            "wrapper_error": _exception_result(exc),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(output_path, _to_plain(result))
    result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


def _parallel_output_ref(
    *,
    run_id: str,
    attempt_id: str,
    eval_id: str,
    checkpoint_label: str,
    max_eval_steps: int,
    seed: int,
    stamp: str,
) -> str:
    leaf = (
        f"lightzero_agent96_stock_eval_{checkpoint_label}"
        f"_steps{max_eval_steps}_seed{seed}_{stamp}.json"
    )
    return (
        runs.attempt_eval_ref(AGENT96_TASK_ID, run_id, attempt_id, eval_id)
        / f"{checkpoint_label}_steps{max_eval_steps}_seed{seed}"
        / leaf
    ).as_posix()


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _histogram_total(histogram: Any) -> int:
    if not isinstance(histogram, dict):
        return 0
    total = 0
    for value in histogram.values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            pass
    return total


def _normalized_histogram_entropy(histogram: Any) -> float | None:
    total = _histogram_total(histogram)
    if not isinstance(histogram, dict) or total <= 0:
        return None
    counts = []
    for value in histogram.values():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts.append(count)
    if len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts)
    return round(entropy / math.log(len(counts)), 6)


def _result_table_row(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    stock_eval = result.get("stock_evaluator") if isinstance(result.get("stock_evaluator"), dict) else {}
    stock_rollout = (
        stock_eval.get("stock_rollout")
        if isinstance(stock_eval.get("stock_rollout"), dict)
        else {}
    )
    histogram = stock_rollout.get("action_histogram")
    return {
        "index": job.get("index"),
        "checkpoint": job.get("checkpoint_label"),
        "seed": job.get("seed"),
        "stock_survival_steps": stock_rollout.get("stock_survival_steps"),
        "stock_return": stock_rollout.get("stock_return"),
        "eval_cap_steps": stock_rollout.get("max_eval_steps"),
        "ok": bool(result.get("ok")),
        "strict_load": result.get("status", {}).get("strict_policy_model_load_ok")
        if isinstance(result.get("status"), dict)
        else None,
        "model_fallback_used": False,
        "stock_done": stock_rollout.get("done"),
        "stock_truncated": stock_rollout.get("truncated"),
        "stock_terminal_reason": stock_rollout.get("terminal_reason"),
        "stock_nonzero_reward_count": stock_rollout.get("nonzero_reward_count"),
        "stock_positive_reward_count": stock_rollout.get("positive_reward_count"),
        "stock_negative_reward_count": stock_rollout.get("negative_reward_count"),
        "stock_action_histogram": histogram,
        "stock_action_entropy": _normalized_histogram_entropy(histogram),
        "elapsed_sec": result.get("remote_elapsed_sec"),
        "artifact_ref": result.get("artifact", {}).get("ref")
        if isinstance(result.get("artifact"), dict)
        else None,
        "checkpoint_ref": job.get("checkpoint_ref"),
    }


def _aggregate_table(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in table:
        checkpoint = str(row.get("checkpoint") or "")
        by_checkpoint.setdefault(checkpoint, []).append(row)

    rows = []
    for checkpoint, group in sorted(by_checkpoint.items(), key=lambda item: _checkpoint_sort_key(item[0])):
        steps = [
            value
            for value in (_number(row.get("stock_survival_steps")) for row in group)
            if value is not None
        ]
        returns = [
            value
            for value in (_number(row.get("stock_return")) for row in group)
            if value is not None
        ]
        rows.append(
            {
                "checkpoint": checkpoint,
                "seeds": len(group),
                "ok": sum(1 for row in group if row.get("ok")),
                "mean_stock_survival_steps": _mean(steps),
                "median_stock_survival_steps": _median(steps),
                "min_stock_survival_steps": min(steps) if steps else None,
                "max_stock_survival_steps": max(steps) if steps else None,
                "mean_stock_return": _mean(returns),
            }
        )
    baseline = next((row for row in rows if row.get("checkpoint") == "iteration_0"), rows[0] if rows else None)
    if baseline is not None:
        base_steps = _number(baseline.get("mean_stock_survival_steps"))
        base_return = _number(baseline.get("mean_stock_return"))
        for row in rows:
            row["baseline_checkpoint"] = baseline.get("checkpoint")
            steps = _number(row.get("mean_stock_survival_steps"))
            stock_return = _number(row.get("mean_stock_return"))
            row["delta_mean_stock_survival_steps"] = (
                steps - base_steps if steps is not None and base_steps is not None else None
            )
            row["delta_mean_stock_return"] = (
                stock_return - base_return
                if stock_return is not None and base_return is not None
                else None
            )
    return rows


def _tsv_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["\t".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            elif isinstance(value, (dict, list, tuple)):
                values.append(json.dumps(_to_plain(value), sort_keys=True, separators=(",", ":")))
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    return "\n".join(lines)


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=EVAL_FUNCTION_TIMEOUT_SEC,
    cpu=1.0,
)
def lightzero_pong_agent96_eval_cpu(
    checkpoint_ref: str,
    output_ref: str,
    run_id: str,
    attempt_id: str,
    env_id: str,
    seed: int,
    max_eval_steps: int,
    quiet_framework_logs: bool = True,
) -> dict[str, Any]:
    return _eval_one_checkpoint(
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=seed,
        max_eval_steps=max_eval_steps,
        compute="cpu",
        use_cuda=False,
        quiet_framework_logs=quiet_framework_logs,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=EVAL_FUNCTION_TIMEOUT_SEC,
    cpu=8.0,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_agent96_eval_l4_cpu8(
    checkpoint_ref: str,
    output_ref: str,
    run_id: str,
    attempt_id: str,
    env_id: str,
    seed: int,
    max_eval_steps: int,
    quiet_framework_logs: bool = True,
) -> dict[str, Any]:
    return _eval_one_checkpoint(
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=seed,
        max_eval_steps=max_eval_steps,
        compute="gpu-l4-t4-cpu8",
        use_cuda=True,
        quiet_framework_logs=quiet_framework_logs,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=EVAL_FUNCTION_TIMEOUT_SEC,
    cpu=40.0,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_agent96_eval_l4_cpu40(
    checkpoint_ref: str,
    output_ref: str,
    run_id: str,
    attempt_id: str,
    env_id: str,
    seed: int,
    max_eval_steps: int,
    quiet_framework_logs: bool = True,
) -> dict[str, Any]:
    return _eval_one_checkpoint(
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=seed,
        max_eval_steps=max_eval_steps,
        compute="gpu-l4-t4-cpu40",
        use_cuda=True,
        quiet_framework_logs=quiet_framework_logs,
    )


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=2 * 60, cpu=0.25)
def lightzero_pong_agent96_eval_manifest(
    manifest_ref: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    output_path = runs.volume_path(RUNS_MOUNT, manifest_ref)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(output_path, _to_plain(manifest))
    summary = runs.file_summary(output_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    return {"ok": True, "manifest_ref": manifest_ref, "artifact": summary}


@app.local_entrypoint()
def main(
    compute: str = DEFAULT_COMPUTE,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    checkpoint_ref: str | None = None,
    checkpoint_refs: str | None = None,
    checkpoint_ref_template: str | None = None,
    selected_iterations: str | None = DEFAULT_SELECTED_ITERATIONS,
    eval_id: str = DEFAULT_EVAL_ID,
    manifest_ref: str | None = None,
    seed: int = DEFAULT_SEED,
    eval_seeds: str | None = None,
    eval_seed_count: int | None = DEFAULT_EVAL_SEED_COUNT,
    eval_seed_rng_seed: int = DEFAULT_EVAL_SEED_RNG_SEED,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    summary_only: bool = False,
    slim_manifest: bool = True,
    quiet_framework_logs: bool = True,
) -> None:
    if compute == "cpu":
        eval_fn = lightzero_pong_agent96_eval_cpu
    elif compute == "gpu-l4-t4-cpu8":
        eval_fn = lightzero_pong_agent96_eval_l4_cpu8
    elif compute == "gpu-l4-t4-cpu40":
        eval_fn = lightzero_pong_agent96_eval_l4_cpu40
    else:
        raise ValueError("compute must be one of: cpu, gpu-l4-t4-cpu8, gpu-l4-t4-cpu40")
    selected_refs = _selected_checkpoint_refs(
        run_id=run_id,
        attempt_id=attempt_id,
        checkpoint_ref=checkpoint_ref,
        checkpoint_refs=checkpoint_refs,
        checkpoint_ref_template=checkpoint_ref_template,
        selected_iterations=selected_iterations,
    )
    checkpoint_labels = [
        _checkpoint_label(ref, index=index) for index, ref in enumerate(selected_refs)
    ]
    if "iteration_0" not in checkpoint_labels:
        raise ValueError("Agent96 strict curve must include iteration_0 for same-run comparison")
    if len(selected_refs) < 2:
        raise ValueError("Agent96 strict curve needs iteration_0 plus at least one later checkpoint")

    eval_seed_values, eval_seed_sampler_seed = _parse_eval_seeds(
        seed=seed,
        eval_seeds=eval_seeds,
        eval_seed_count=eval_seed_count,
        eval_seed_rng_seed=eval_seed_rng_seed,
    )
    clean_eval_id = _safe_generated_id(eval_id, fallback="agent96_eval")
    stamp = runs.utc_stamp()
    jobs = [
        {
            "index": job_index,
            "checkpoint_ref": ref,
            "checkpoint_label": checkpoint_labels[checkpoint_index],
            "output_ref": _parallel_output_ref(
                run_id=run_id,
                attempt_id=attempt_id,
                eval_id=clean_eval_id,
                checkpoint_label=checkpoint_labels[checkpoint_index],
                max_eval_steps=max_eval_steps,
                seed=eval_seed,
                stamp=stamp,
            ),
            "seed": eval_seed,
        }
        for job_index, (eval_seed, checkpoint_index, ref) in enumerate(
            (eval_seed, checkpoint_index, ref)
            for eval_seed in eval_seed_values
            for checkpoint_index, ref in enumerate(selected_refs)
        )
    ]
    starmap_inputs = [
        (
            job["checkpoint_ref"],
            job["output_ref"],
            run_id,
            attempt_id,
            env_id,
            job["seed"],
            max_eval_steps,
        )
        for job in jobs
    ]
    results = list(
        eval_fn.starmap(
            starmap_inputs,
            kwargs={"quiet_framework_logs": quiet_framework_logs},
        )
    )
    table = [_result_table_row(job, result) for job, result in zip(jobs, results, strict=True)]
    aggregate = _aggregate_table(table)
    manifest_stamp = runs.utc_stamp()
    manifest_ref = manifest_ref or (
        runs.attempt_eval_ref(AGENT96_TASK_ID, run_id, attempt_id, clean_eval_id)
        / (
            f"manifest_steps{max_eval_steps}_seeds"
            f"{_seed_panel_label(eval_seed_values)}_{manifest_stamp}.json"
        )
    ).as_posix()
    manifest = {
        "schema": "curvyzero_lightzero_muzero_agent96_strict_stock_parallel_eval/v0",
        "ok": all(bool(result.get("ok")) for result in results),
        "created_at": runs.utc_timestamp(),
        "job_kind": "lightzero_muzero_agent96_same_run_strict_stock_checkpoint_curve",
        "agent96_task_id": AGENT96_TASK_ID,
        "eval_id": clean_eval_id,
        "selection_metric_order": ["stock_survival_steps", "stock_return"],
        "no_model_fallback": True,
        "stock_env": {
            "env_id": env_id,
            "source": "lzero.agent.config.muzero.supported_env_cfg",
            "expected_surface": "PongNoFrameskip-v4 atari_lightzero [4,96,96] downsample=True",
            "not_used": "zoo.atari.config.atari_muzero_config 64x64 eval path",
        },
        "selection": {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_refs": checkpoint_refs,
            "checkpoint_ref_template": checkpoint_ref_template,
            "selected_iterations": selected_iterations,
            "eval_seeds": eval_seed_values,
            "eval_seed_count": eval_seed_count,
            "eval_seed_sampler_seed": eval_seed_sampler_seed,
            "jobs": jobs,
        },
        "config": {
            "compute": compute,
            "max_eval_steps": max_eval_steps,
            "summary_only": summary_only,
            "slim_manifest": slim_manifest,
            "quiet_framework_logs": quiet_framework_logs,
        },
        "table_fields_ordered_by_priority": [
            "stock_survival_steps",
            "stock_return",
            "ok",
            "strict_load",
            "model_fallback_used",
        ],
        "aggregate_table": aggregate,
        "table": table,
        "output_refs": [job["output_ref"] for job in jobs],
        "artifacts": [
            result.get("artifact")
            for result in results
            if isinstance(result, dict) and result.get("artifact")
        ],
    }
    if slim_manifest:
        manifest["results_omitted"] = True
        manifest["result_count"] = len(results)
    else:
        manifest["results"] = results
    manifest_result = lightzero_pong_agent96_eval_manifest.remote(
        manifest_ref=manifest_ref,
        manifest=manifest,
    )
    if summary_only:
        print("# aggregate_by_checkpoint")
        print(
            _tsv_table(
                aggregate,
                [
                    "checkpoint",
                    "seeds",
                    "ok",
                    "mean_stock_survival_steps",
                    "median_stock_survival_steps",
                    "min_stock_survival_steps",
                    "max_stock_survival_steps",
                    "mean_stock_return",
                    "baseline_checkpoint",
                    "delta_mean_stock_survival_steps",
                    "delta_mean_stock_return",
                ],
            )
        )
        print("# per_checkpoint_seed_curve")
        print(
            _tsv_table(
                table,
                [
                    "checkpoint",
                    "seed",
                    "stock_survival_steps",
                    "stock_return",
                    "eval_cap_steps",
                    "ok",
                    "strict_load",
                    "model_fallback_used",
                    "stock_terminal_reason",
                    "artifact_ref",
                ],
            )
        )
        print(f"manifest_ref\t{manifest_ref}")
        return
    print(
        json.dumps(
            {
                "aggregate_table": aggregate,
                "table": table,
                "manifest": manifest_result,
                "manifest_ref": manifest_ref,
                "jobs": jobs,
                "output_refs": [job["output_ref"] for job in jobs],
                "results": results if not slim_manifest else "omitted_by_slim_manifest",
            },
            indent=2,
            sort_keys=True,
        )
    )
