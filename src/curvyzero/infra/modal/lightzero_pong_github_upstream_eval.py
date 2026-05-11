"""Strict eval for pinned GitHub upstream LightZero Atari Pong segment checkpoints.

Run a tiny survival-first smoke from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_github_upstream_eval \
      --selected-iterations 0 --eval-seeds 0 --max-eval-steps 64 --num-simulations 5
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import random
import re
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_pong_checkpoint_probe import (
    _checkpoint_summary,
    _find_state_dict,
    _load_state_dict_probe,
    _torch_load,
)
from curvyzero.infra.modal.lightzero_pong_github_upstream_dry_check import (
    AUTOROM_VERSION,
    LIGHTZERO_GITHUB_COMMIT,
    LIGHTZERO_GITHUB_REPO,
    OPENCV_PYTHON_HEADLESS_VERSION,
    TASK_ID,
)

APP_NAME = "curvyzero-lightzero-pong-github-upstream-eval"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
VOLUME_NAME = "curvyzero-runs"

DEFAULT_RUN_ID = "lz-visual-pong-github-upstream-segment-20260511-s1-wait"
DEFAULT_ATTEMPT_ID = "train-muzero-segment-ale-pong-v5-50k-ckpt1000-l4cpu40-wait"
DEFAULT_ENV_ID = "ALE/Pong-v5"
DEFAULT_SEED = 0
DEFAULT_EVAL_ID = "upstream_segment_survival"
DEFAULT_MAX_EVAL_STEPS = 512
DEFAULT_NUM_SIMULATIONS = 50
DEFAULT_SELECTED_ITERATIONS = "0"
DEFAULT_STEP_RECORD_LIMIT = 2048
DEFAULT_COMPUTE = "gpu-l4-t4-cpu40"
DEFAULT_EVAL_SEED_RNG_SEED = 20260511
EVAL_SEED_MIN = 0
EVAL_SEED_MAX = (2**31) - 1
SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "g++")
    .uv_pip_install("Cython>=3", "numpy>=1.24.1,<2")
    .uv_pip_install(
        f"git+{LIGHTZERO_GITHUB_REPO}@{LIGHTZERO_GITHUB_COMMIT}",
        f"opencv-python-headless=={OPENCV_PYTHON_HEADLESS_VERSION}",
        f"AutoROM[accept-rom-license]=={AUTOROM_VERSION}",
    )
    .run_commands("AutoROM --accept-license")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


class _ConfigCaptured(RuntimeError):
    """Intentional stop after upstream segment config construction."""


@dataclass(frozen=True)
class EvalJob:
    checkpoint_ref: str
    checkpoint_label: str
    seed: int
    output_ref: str


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_to_plain(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
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


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _runtime_compute_summary(*, requested_compute: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested_compute": requested_compute,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        summary["torch_cuda_available"] = cuda_available
        summary["torch_cuda_device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available:
            device = int(torch.cuda.current_device())
            summary["torch_cuda_current_device"] = device
            summary["torch_cuda_device_name"] = torch.cuda.get_device_name(device)
            summary["torch_cuda_capability"] = list(torch.cuda.get_device_capability(device))
    except Exception as exc:  # pragma: no cover - remote runtime diagnosis.
        summary["torch_cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _default_checkpoint_dir(run_id: str, attempt_id: str) -> str:
    return (
        f"training/{TASK_ID}/{run_id}/attempts/{attempt_id}"
        "/train/lightzero_segment_exp/ckpt"
    )


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_iterations(value: str | None) -> list[int]:
    iterations: list[int] = []
    for item in _split_csv(value):
        try:
            iteration = int(item)
        except ValueError as exc:
            raise ValueError(f"selected iteration must be an int, got {item!r}") from exc
        if iteration < 0:
            raise ValueError("selected iterations must be non-negative")
        iterations.append(iteration)
    return iterations


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


def _checkpoint_label(checkpoint_ref: str) -> str:
    label = Path(checkpoint_ref).name
    for suffix in (".pth.tar", ".tar", ".pth", ".pt", ".bin"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    label = SAFE_LABEL_RE.sub("_", label).strip("._-") or "checkpoint"
    if not label[0].isalnum():
        label = f"checkpoint_{label}"
    return runs.clean_id(label[:80], label="checkpoint_label")


def _selected_checkpoint_refs(
    *,
    checkpoint_dir: str,
    checkpoint_refs: str | None,
    selected_iterations: str | None,
) -> list[str]:
    refs = _split_csv(checkpoint_refs)
    iterations = _parse_iterations(selected_iterations)
    if refs and iterations:
        raise ValueError("use either checkpoint_refs or selected_iterations, not both")
    if refs:
        return refs
    if not iterations:
        iterations = _parse_iterations(DEFAULT_SELECTED_ITERATIONS)
    return [f"{checkpoint_dir.rstrip('/')}/iteration_{iteration}.pth.tar" for iteration in iterations]


def _seed_panel_label(seeds: list[int]) -> str:
    payload = ",".join(str(seed) for seed in seeds)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"n{len(seeds)}_{digest}"


def _output_ref(
    *,
    run_id: str,
    attempt_id: str,
    eval_id: str,
    checkpoint_label: str,
    seed: int,
    max_eval_steps: int,
    stamp: str,
) -> str:
    leaf = (
        f"github_upstream_segment_eval_{checkpoint_label}"
        f"_steps{max_eval_steps}_seed{seed}_{stamp}.json"
    )
    return (
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, eval_id)
        / f"{checkpoint_label}_steps{max_eval_steps}_seed{seed}"
        / leaf
    ).as_posix()


def _manifest_ref(
    *,
    run_id: str,
    attempt_id: str,
    eval_id: str,
    max_eval_steps: int,
    seeds: list[int],
    stamp: str,
) -> str:
    return (
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, eval_id)
        / f"manifest_steps{max_eval_steps}_seeds{_seed_panel_label(seeds)}_{stamp}.json"
    ).as_posix()


def _capture_upstream_segment_configs(
    *,
    env_id: str,
    seed: int,
    exp_name_ref: str,
    num_simulations: int,
) -> dict[str, Any]:
    module_name = "zoo.atari.config.atari_muzero_segment_config"
    module = importlib.import_module(module_name)
    entry_module = importlib.import_module("lzero.entry")
    original_train = entry_module.train_muzero_segment
    captured: dict[str, Any] = {}

    def intercepted_train(configs: Any, *args: Any, **kwargs: Any) -> Any:
        del args
        main_config, create_config = configs
        main_config = copy.deepcopy(main_config)
        create_config = copy.deepcopy(create_config)
        old_exp_name = main_config["exp_name"]
        main_config["exp_name"] = exp_name_ref
        old_simulations = main_config["policy"]["num_simulations"]
        main_config["policy"]["num_simulations"] = int(num_simulations)
        captured.update(
            {
                "main_config": main_config,
                "create_config": create_config,
                "stock_max_env_step": int(kwargs.get("max_env_step")),
                "patches": [
                    {
                        "path": "main_config.exp_name",
                        "old": _to_plain(old_exp_name),
                        "new": exp_name_ref,
                        "reason": "eval artifact path only",
                    },
                    {
                        "path": "main_config.policy.num_simulations",
                        "old": _to_plain(old_simulations),
                        "new": int(num_simulations),
                        "reason": "eval-time search budget",
                    },
                ],
            }
        )
        raise _ConfigCaptured("captured upstream segment configs before train")

    try:
        entry_module.train_muzero_segment = intercepted_train
        try:
            module.main(env_id, seed)
        except _ConfigCaptured:
            pass
    finally:
        entry_module.train_muzero_segment = original_train
    if not captured:
        raise RuntimeError("failed to capture upstream segment configs")
    captured["module"] = module_name
    return captured


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


def _record_step(records: list[dict[str, Any]], record: dict[str, Any], record_path: Path) -> None:
    plain = _to_plain(record)
    records.append(plain)
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(plain, sort_keys=True) + "\n")


def _load_step_records(records: list[dict[str, Any]], record_path: Path) -> list[dict[str, Any]]:
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
                    _record_step(
                        records,
                        {
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
                        },
                        record_path,
                    )
                except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
                    _record_step(
                        records,
                        {
                            "step_index": step_index,
                            "action": _action_int_or_none(action),
                            "action_raw": _to_plain(action),
                            "recording_error": _exception_result(exc),
                            "env_step_elapsed_sec": round(elapsed_sec, 6),
                        },
                        record_path,
                    )
                return timestep

        return _RecordingEnv(inner)

    return _thunk


def _extract_eval_action(output: Any) -> int | None:
    plain = _to_plain(output)
    if isinstance(plain, dict):
        if "action" in plain:
            return _action_int_or_none(plain["action"])
        for key in (0, "0"):
            if key in plain:
                return _extract_eval_action(plain[key])
    if isinstance(plain, (list, tuple)) and plain:
        return _extract_eval_action(plain[0])
    return None


def _summarize_value(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        summary["shape"] = [int(item) for item in shape]
    if dtype is not None:
        summary["dtype"] = str(dtype)
    return summary


def _run_stock_evaluator(
    *,
    main_config: Any,
    create_config: Any,
    state_dict: dict[str, Any],
    seed: int,
    max_eval_steps: int,
    use_cuda: bool,
    step_record_limit: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    eval_mode_records: list[dict[str, Any]] = []
    env_manager = None
    temp_root = Path(tempfile.mkdtemp(prefix="curvyzero_github_upstream_eval_"))
    record_path = temp_root / "step_records.jsonl"
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
        cfg.policy.cuda = use_cuda
        cfg.policy.device = "cuda" if use_cuda else "cpu"
        cfg.env.evaluator_env_num = 1
        cfg.env.n_evaluator_episode = 1
        cfg.env.eval_max_episode_steps = int(max_eval_steps)
        manager_cfg = getattr(cfg.env, "manager", None)
        env_manager_patch: dict[str, Any] | None = None
        if isinstance(manager_cfg, dict):
            old_manager_type = manager_cfg.get("type")
            manager_cfg["type"] = "base"
            env_manager_patch = {"path": "env.manager.type", "old": old_manager_type, "new": "base"}
        elif manager_cfg is not None and hasattr(manager_cfg, "type"):
            old_manager_type = manager_cfg.type
            manager_cfg.type = "base"
            env_manager_patch = {"path": "env.manager.type", "old": old_manager_type, "new": "base"}

        policy = MuZeroPolicy(cfg.policy)
        model = getattr(policy, "_model", None)
        if model is None:
            raise AttributeError("MuZeroPolicy has no _model attribute")
        load = _load_state_dict_probe(model, state_dict)
        if hasattr(model, "eval"):
            model.eval()
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
                            "action": _extract_eval_action(output),
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
            instance_name="curvyzero_github_upstream_segment_eval",
            policy_config=cfg.policy,
        )

        def save_ckpt_fn(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        eval_output = evaluator.eval(save_ckpt_fn, train_iter=0, envstep=0)
        records = _load_step_records(records, record_path)
        rollout = _rollout_summary(records, max_eval_steps=max_eval_steps, limit=step_record_limit)
        return {
            "ok": True,
            "path": "lzero.worker.MuZeroEvaluator",
            "load_state_dict": load,
            "eval_output": _to_plain(eval_output),
            "stock_rollout": rollout,
            "env_manager_patch": env_manager_patch,
            "recorded_eval_mode_calls": eval_mode_records,
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:  # pragma: no cover - remote diagnosis.
        records = _load_step_records(records, record_path)
        return {
            "ok": False,
            "path": "lzero.worker.MuZeroEvaluator",
            "blocker": _exception_result(exc),
            "stock_rollout": _rollout_summary(records, max_eval_steps=max_eval_steps, limit=step_record_limit),
            "recorded_eval_mode_calls": eval_mode_records,
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    finally:
        if env_manager is not None and hasattr(env_manager, "close"):
            try:
                env_manager.close()
            except Exception:
                pass


def _rollout_summary(
    records: list[dict[str, Any]],
    *,
    max_eval_steps: int,
    limit: int,
) -> dict[str, Any]:
    ok_steps = [step for step in records if "recording_error" not in step]
    rewards: list[float] = []
    actions: list[int] = []
    for step in ok_steps:
        if step.get("action") is not None:
            actions.append(int(step["action"]))
        try:
            rewards.append(float(step.get("reward", 0.0)))
        except (TypeError, ValueError):
            pass
    terminal_steps = [step for step in ok_steps if bool(step.get("done"))]
    terminal_step = terminal_steps[-1] if terminal_steps else {}
    return {
        "ok": bool(ok_steps) and len(ok_steps) == len(records),
        "max_eval_steps": max_eval_steps,
        "steps_run": len(ok_steps),
        "score": round(sum(rewards), 6) if rewards else 0.0,
        "total_reward": round(sum(rewards), 6) if rewards else 0.0,
        "done": bool(terminal_steps),
        "survived_to_cap": len(ok_steps) >= max_eval_steps,
        "terminal_reason": terminal_step.get("terminal_reason"),
        "terminal_info": terminal_step.get("info"),
        "nonzero_reward_count": sum(1 for reward in rewards if reward != 0.0),
        "positive_reward_count": sum(1 for reward in rewards if reward > 0.0),
        "negative_reward_count": sum(1 for reward in rewards if reward < 0.0),
        "action_histogram": dict(sorted(Counter(actions).items())),
        "reward_histogram": dict(sorted(Counter(rewards).items())),
        "step_records": ok_steps[:limit],
        "step_records_truncated": len(ok_steps) > limit,
        "recording_error_count": len(records) - len(ok_steps),
    }


def _eval_checkpoint(job: EvalJob, *, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_path = runs.volume_path(RUNS_MOUNT, job.checkpoint_ref)
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "gym": _version_or_missing("gym"),
        "gymnasium": _version_or_missing("gymnasium"),
        "ale-py": _version_or_missing("ale-py", "ale_py"),
        "opencv-python-headless": _version_or_missing("opencv-python-headless"),
        "AutoROM": _version_or_missing("AutoROM"),
    }
    result: dict[str, Any]
    try:
        checkpoint = _torch_load(checkpoint_path)
        state_candidate = _find_state_dict(checkpoint)
        if state_candidate is None:
            raise ValueError("no tensor state dict found under common LightZero checkpoint keys")
        state_path, state_dict = state_candidate
        captured = _capture_upstream_segment_configs(
            env_id=str(config["env_id"]),
            seed=job.seed,
            exp_name_ref=(
                runs.attempt_eval_ref(
                    TASK_ID,
                    str(config["run_id"]),
                    str(config["attempt_id"]),
                    str(config["eval_id"]),
                )
                / "lightzero_segment_eval_exp"
            ).as_posix(),
            num_simulations=int(config["num_simulations"]),
        )
        evaluator_result = _run_stock_evaluator(
            main_config=captured["main_config"],
            create_config=captured["create_config"],
            state_dict=state_dict,
            seed=job.seed,
            max_eval_steps=int(config["max_eval_steps"]),
            use_cuda=bool(config["use_cuda"]),
            step_record_limit=int(config["step_record_limit"]),
        )
        rollout = evaluator_result.get("stock_rollout", {})
        result = {
            "schema": "curvyzero_lightzero_github_upstream_segment_eval/v0",
            "ok": bool(evaluator_result.get("ok")),
            "started_at": config["started_at"],
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "source": {
                "kind": "github",
                "repo": LIGHTZERO_GITHUB_REPO,
                "commit": LIGHTZERO_GITHUB_COMMIT,
                "official_source_module": captured["module"],
                "official_env_arg_used": f"--env {config['env_id']}",
            },
            "config": {**config, "seed": job.seed, "checkpoint_ref": job.checkpoint_ref},
            "packages": packages,
            "checkpoint": _checkpoint_summary(checkpoint_path, include_sha256=False),
            "load": {
                "ok": True,
                "state_dict_path": state_path,
                "tensor_count": len(state_dict),
                "keys_sample": list(state_dict)[:40],
            },
            "stock_example": {
                "trainer_entrypoint": "lzero.entry.train_muzero_segment",
                "stock_max_env_step": captured["stock_max_env_step"],
                "patches": captured["patches"],
            },
            "stock_evaluator": evaluator_result,
            "status": {
                "checkpoint_load_ok": True,
                "strict_policy_model_load_ok": bool(
                    evaluator_result.get("load_state_dict", {}).get("strict")
                ),
                "env_steps_run": rollout.get("steps_run"),
                "survival_steps": rollout.get("steps_run"),
                "score": rollout.get("score"),
                "done": rollout.get("done"),
                "survived_to_cap": rollout.get("survived_to_cap"),
            },
        }
    except Exception as exc:  # pragma: no cover - remote diagnosis.
        result = {
            "schema": "curvyzero_lightzero_github_upstream_segment_eval/v0",
            "ok": False,
            "started_at": config["started_at"],
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": {**config, "seed": job.seed, "checkpoint_ref": job.checkpoint_ref},
            "packages": packages,
            "checkpoint": _checkpoint_summary(checkpoint_path, include_sha256=False),
            "status": {
                "checkpoint_load_ok": False,
                "strict_policy_model_load_ok": False,
                "env_steps_run": 0,
                "survival_steps": 0,
                "score": None,
                "done": None,
                "survived_to_cap": False,
            },
            "wrapper_error": _exception_result(exc),
        }
    output_path = runs.volume_path(RUNS_MOUNT, job.output_ref)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(output_path, _to_plain(result))
    result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    return _to_plain(result)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _table_row(job: EvalJob, result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("status") if isinstance(result.get("status"), dict) else {}
    artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
    return {
        "checkpoint": job.checkpoint_label,
        "seed": job.seed,
        "ok": bool(result.get("ok")),
        "strict_load": status.get("strict_policy_model_load_ok"),
        "survival_steps": status.get("survival_steps"),
        "score": status.get("score"),
        "done": status.get("done"),
        "survived_to_cap": status.get("survived_to_cap"),
        "artifact_ref": artifact.get("ref"),
        "checkpoint_ref": job.checkpoint_ref,
    }


def _aggregate_rows(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in table:
        by_checkpoint.setdefault(str(row["checkpoint"]), []).append(row)
    rows = []
    for checkpoint, items in sorted(by_checkpoint.items()):
        steps = [value for value in (_number(item.get("survival_steps")) for item in items) if value is not None]
        scores = [value for value in (_number(item.get("score")) for item in items) if value is not None]
        rows.append(
            {
                "checkpoint": checkpoint,
                "seeds": len(items),
                "ok": sum(1 for item in items if item.get("ok")),
                "mean_survival_steps": _mean(steps),
                "median_survival_steps": _median(steps),
                "min_survival_steps": min(steps) if steps else None,
                "max_survival_steps": max(steps) if steps else None,
                "mean_score": _mean(scores),
            }
        )
    return rows


def _run_eval_manifest(
    *,
    compute: str,
    run_id: str,
    attempt_id: str,
    eval_id: str,
    env_id: str,
    seed: int,
    checkpoint_refs: str | None,
    selected_iterations: str | None,
    eval_seeds: str | None,
    eval_seed_count: int | None,
    eval_seed_rng_seed: int,
    max_eval_steps: int,
    num_simulations: int,
    step_record_limit: int,
) -> dict[str, Any]:
    if compute not in {"cpu", "gpu-l4-t4-cpu40"}:
        raise ValueError("compute must be 'cpu' or 'gpu-l4-t4-cpu40'")
    use_cuda = compute == "gpu-l4-t4-cpu40"
    seeds, seed_sampler = _parse_eval_seeds(
        seed=seed,
        eval_seeds=eval_seeds,
        eval_seed_count=eval_seed_count,
        eval_seed_rng_seed=eval_seed_rng_seed,
    )
    checkpoint_dir = _default_checkpoint_dir(run_id, attempt_id)
    refs = _selected_checkpoint_refs(
        checkpoint_dir=checkpoint_dir,
        checkpoint_refs=checkpoint_refs,
        selected_iterations=selected_iterations,
    )
    stamp = runs.utc_stamp()
    started_at = runs.utc_timestamp()
    base_config = {
        "job_kind": "lightzero_github_upstream_segment_eval",
        "compute": compute,
        "use_cuda": use_cuda,
        "runtime_compute": _runtime_compute_summary(requested_compute=compute),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "eval_id": eval_id,
        "env_id": env_id,
        "max_eval_steps": max_eval_steps,
        "num_simulations": num_simulations,
        "step_record_limit": step_record_limit,
        "started_at": started_at,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    jobs = [
        EvalJob(
            checkpoint_ref=ref,
            checkpoint_label=_checkpoint_label(ref),
            seed=job_seed,
            output_ref=_output_ref(
                run_id=run_id,
                attempt_id=attempt_id,
                eval_id=eval_id,
                checkpoint_label=_checkpoint_label(ref),
                seed=job_seed,
                max_eval_steps=max_eval_steps,
                stamp=stamp,
            ),
        )
        for ref in refs
        for job_seed in seeds
    ]
    results = [_eval_checkpoint(job, config=base_config) for job in jobs]
    table = [_table_row(job, result) for job, result in zip(jobs, results, strict=True)]
    manifest = {
        "schema": "curvyzero_lightzero_github_upstream_segment_eval_manifest/v0",
        "ok": all(row["ok"] for row in table),
        "started_at": started_at,
        "ended_at": runs.utc_timestamp(),
        "source": {
            "kind": "github",
            "repo": LIGHTZERO_GITHUB_REPO,
            "commit": LIGHTZERO_GITHUB_COMMIT,
            "official_source_module": "zoo.atari.config.atari_muzero_segment_config",
            "official_env_arg_used": f"--env {env_id}",
        },
        "config": {
            **base_config,
            "checkpoint_refs": refs,
            "eval_seeds": seeds,
            "eval_seed_sampler": seed_sampler,
        },
        "aggregate": _aggregate_rows(table),
        "table": table,
        "output_refs": [job.output_ref for job in jobs],
    }
    manifest_path = runs.volume_path(
        RUNS_MOUNT,
        _manifest_ref(
            run_id=run_id,
            attempt_id=attempt_id,
            eval_id=eval_id,
            max_eval_steps=max_eval_steps,
            seeds=seeds,
            stamp=stamp,
        ),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(manifest_path, _to_plain(manifest))
    manifest["artifact"] = runs.file_summary(manifest_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    return _to_plain(manifest)


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=30 * 60,
    cpu=1.0,
)
def lightzero_pong_github_upstream_eval_cpu(
    run_id: str,
    attempt_id: str,
    eval_id: str,
    env_id: str,
    seed: int,
    checkpoint_refs: str | None,
    selected_iterations: str | None,
    eval_seeds: str | None,
    eval_seed_count: int | None,
    eval_seed_rng_seed: int,
    max_eval_steps: int,
    num_simulations: int,
    step_record_limit: int,
) -> dict[str, Any]:
    return _run_eval_manifest(
        compute="cpu",
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=eval_id,
        env_id=env_id,
        seed=seed,
        checkpoint_refs=checkpoint_refs,
        selected_iterations=selected_iterations,
        eval_seeds=eval_seeds,
        eval_seed_count=eval_seed_count,
        eval_seed_rng_seed=eval_seed_rng_seed,
        max_eval_steps=max_eval_steps,
        num_simulations=num_simulations,
        step_record_limit=step_record_limit,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=90 * 60,
    gpu=["L4", "T4"],
    cpu=40.0,
)
def lightzero_pong_github_upstream_eval_gpu_l4_cpu40(
    run_id: str,
    attempt_id: str,
    eval_id: str,
    env_id: str,
    seed: int,
    checkpoint_refs: str | None,
    selected_iterations: str | None,
    eval_seeds: str | None,
    eval_seed_count: int | None,
    eval_seed_rng_seed: int,
    max_eval_steps: int,
    num_simulations: int,
    step_record_limit: int,
) -> dict[str, Any]:
    return _run_eval_manifest(
        compute="gpu-l4-t4-cpu40",
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=eval_id,
        env_id=env_id,
        seed=seed,
        checkpoint_refs=checkpoint_refs,
        selected_iterations=selected_iterations,
        eval_seeds=eval_seeds,
        eval_seed_count=eval_seed_count,
        eval_seed_rng_seed=eval_seed_rng_seed,
        max_eval_steps=max_eval_steps,
        num_simulations=num_simulations,
        step_record_limit=step_record_limit,
    )


@app.local_entrypoint()
def main(
    compute: str = DEFAULT_COMPUTE,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    eval_id: str = DEFAULT_EVAL_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    checkpoint_refs: str | None = None,
    selected_iterations: str | None = DEFAULT_SELECTED_ITERATIONS,
    eval_seeds: str | None = None,
    eval_seed_count: int | None = None,
    eval_seed_rng_seed: int = DEFAULT_EVAL_SEED_RNG_SEED,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    step_record_limit: int = DEFAULT_STEP_RECORD_LIMIT,
) -> None:
    eval_fn = (
        lightzero_pong_github_upstream_eval_gpu_l4_cpu40
        if compute == "gpu-l4-t4-cpu40"
        else lightzero_pong_github_upstream_eval_cpu
        if compute == "cpu"
        else None
    )
    if eval_fn is None:
        raise ValueError("compute must be 'cpu' or 'gpu-l4-t4-cpu40'")
    result = eval_fn.remote(
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=eval_id,
        env_id=env_id,
        seed=seed,
        checkpoint_refs=checkpoint_refs,
        selected_iterations=selected_iterations,
        eval_seeds=eval_seeds,
        eval_seed_count=eval_seed_count,
        eval_seed_rng_seed=eval_seed_rng_seed,
        max_eval_steps=max_eval_steps,
        num_simulations=num_simulations,
        step_record_limit=step_record_limit,
    )
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
