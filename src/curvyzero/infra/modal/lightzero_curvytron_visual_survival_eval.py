"""Modal eval harness for CurvyTron debug-visual survival LightZero checkpoints.

This scores one or more mirrored LightZero checkpoints by running the
registered stacked visual CurvyTron env. Survival length is reported from
episode length, separate from the trainer reward variant.

Example:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
      --run-id curvytron-visual-survival-debug-lz-s0 \
      --attempt-id train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510 \
      --selected-iterations 0,1,2 \
      --seed 0 \
      --max-eval-steps 256 \
      --parallel \
      --summary-only
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import time
import traceback
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    CHEAP_GPU_RESOURCE,
    DEFAULT_ATTEMPT_ID as TRAIN_DEFAULT_ATTEMPT_ID,
    DEFAULT_COLLECTOR_ENV_NUM,
    DEFAULT_DECISION_MS,
    DEFAULT_ENV_MANAGER_TYPE,
    DEFAULT_ENV_TELEMETRY_STRIDE,
    DEFAULT_ENV_VARIANT,
    DEFAULT_LIGHTZERO_EVAL_FREQ,
    DEFAULT_MAX_TRAIN_ITER,
    DEFAULT_N_EPISODE,
    DEFAULT_N_EVALUATOR_EPISODE,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_OPPONENT_POLICY_KIND,
    DEFAULT_REWARD_VARIANT,
    DEFAULT_RUN_ID as TRAIN_DEFAULT_RUN_ID,
    DEFAULT_SAVE_CKPT_AFTER_ITER,
    DEFAULT_SOURCE_MAX_STEPS,
    LIGHTZERO_VERSION,
    REMOTE_ROOT,
    RUNS_MOUNT,
    TASK_ID,
    VOLUME_NAME,
    _build_visual_survival_configs,
    _normalize_opponent_policy_kind_for_env,
    _normalize_reward_variant_for_env,
    _resolve_opponent_checkpoint_for_env,
    _to_plain,
    image,
    runs_volume,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    ENV_VARIANT_CHOICES,
    OPPONENT_POLICY_KIND_CHOICES,
)


APP_NAME = "curvyzero-lightzero-curvytron-visual-survival-eval"

DEFAULT_RUN_ID = TRAIN_DEFAULT_RUN_ID
DEFAULT_ATTEMPT_ID = TRAIN_DEFAULT_ATTEMPT_ID
DEFAULT_CHECKPOINT_REF = (
    "training/lightzero-curvytron-visual-survival/"
    "curvytron-visual-survival-debug-lz-s0/"
    "checkpoints/lightzero/iteration_0.pth.tar"
)
DEFAULT_EVAL_ID = "checkpoint_curve"
GPU_L4_T4_CPU40_COMPUTE = "gpu-l4-t4-cpu40"
DEFAULT_COMPUTE = GPU_L4_T4_CPU40_COMPUTE
DEFAULT_MAX_EVAL_STEPS = 256
DEFAULT_STEP_DETAIL_LIMIT = 4
DEFAULT_QUIET_FRAMEWORK_LOGS = True
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_EVAL_SEED_COUNT = 8
DEFAULT_EVAL_SEED_RNG_SEED: int | None = None
EVAL_SEED_RNG_SEED_MAX = (2**63) - 1
EVAL_SEED_MIN = 0
EVAL_SEED_MAX = (2**31) - 1
EVAL_TIMEOUT_SEC = 20 * 60
COMPUTE_CHOICES = ("cpu", "gpu-l4-t4", GPU_L4_T4_CPU40_COMPUTE)

app = modal.App(APP_NAME)


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-16:],
    }


class _QuietFrameworkOutput:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._stack: Any = None

    def __enter__(self) -> None:
        if not self.enabled:
            return
        import contextlib

        self._sink = open(os.devnull, "w", encoding="utf-8")
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(contextlib.redirect_stdout(self._sink))
        self._stack.enter_context(contextlib.redirect_stderr(self._sink))

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._stack is not None:
            self._stack.close()
            self._sink.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_summary(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        summary["bytes"] = path.stat().st_size
        if include_sha256:
            summary["sha256"] = _sha256(path)
    return summary


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _find_state_dict(payload: Any) -> tuple[str, dict[str, Any]] | None:
    candidates: list[tuple[str, Any]] = [("<root>", payload)]
    if isinstance(payload, dict):
        for key in (
            "model",
            "state_dict",
            "model_state_dict",
            "network",
            "target_model",
            "policy",
            "_model",
            "_learn_model",
        ):
            if key in payload:
                candidates.append((key, payload[key]))
        for key, value in payload.items():
            if isinstance(value, dict):
                for nested_key in ("model", "state_dict", "model_state_dict", "_model", "_learn_model"):
                    if nested_key in value:
                        candidates.append((f"{key}.{nested_key}", value[nested_key]))

    best: tuple[str, dict[str, Any], int] | None = None
    for path, value in candidates:
        if not isinstance(value, dict):
            continue
        tensor_count = sum(1 for item in value.values() if _is_tensor_like(item))
        if tensor_count == 0:
            continue
        keys = [str(key) for key in value]
        score = tensor_count
        if any("representation_network" in key for key in keys):
            score += 1000
        if any("prediction_network" in key for key in keys):
            score += 1000
        if any("dynamics_network" in key for key in keys):
            score += 1000
        if best is None or score > best[2]:
            best = (path, value, score)
    if best is None:
        return None
    return best[0], best[1]


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not any(str(key).startswith(prefix) for key in state_dict):
        return state_dict
    return {str(key).removeprefix(prefix): value for key, value in state_dict.items()}


def _load_state_dict_probe(module: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        ("as_is", state_dict),
        ("strip_module", _strip_prefix(state_dict, "module.")),
        ("strip_model", _strip_prefix(state_dict, "model.")),
        ("strip_learn_model", _strip_prefix(state_dict, "_learn_model.")),
    ]
    errors = []
    for name, candidate in candidates:
        try:
            loaded = module.load_state_dict(candidate, strict=True)
            return {
                "ok": True,
                "candidate": name,
                "strict": True,
                "missing_keys": list(getattr(loaded, "missing_keys", [])),
                "unexpected_keys": list(getattr(loaded, "unexpected_keys", [])),
            }
        except Exception as exc:
            errors.append({"candidate": name, "strict": True, "error": str(exc)})

    for name, candidate in candidates:
        try:
            loaded = module.load_state_dict(candidate, strict=False)
            return {
                "ok": True,
                "candidate": name,
                "strict": False,
                "missing_keys": list(getattr(loaded, "missing_keys", []))[:80],
                "unexpected_keys": list(getattr(loaded, "unexpected_keys", []))[:80],
                "note": "non-strict load is reported only; this eval does not score it",
            }
        except Exception as exc:
            errors.append({"candidate": name, "strict": False, "error": str(exc)})

    return {"ok": False, "strict": False, "errors": errors[-8:]}


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


def _parse_eval_seeds(
    *,
    seed: int,
    eval_seeds: str | None,
    eval_seed_count: int | None,
    eval_seed_rng_seed: int | None,
) -> tuple[list[int], int | None]:
    if eval_seeds is None:
        if eval_seed_count is None:
            return [seed], None
        if eval_seed_count < 1:
            raise ValueError("eval_seed_count must be >= 1")
        sampler_seed = eval_seed_rng_seed
        if sampler_seed is None:
            sampler_seed = random.SystemRandom().randrange(
                0, EVAL_SEED_RNG_SEED_MAX + 1
            )
        rng = random.Random(sampler_seed)
        return (
            rng.sample(range(EVAL_SEED_MIN, EVAL_SEED_MAX + 1), eval_seed_count),
            sampler_seed,
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


def _safe_generated_id(raw: str, *, fallback: str) -> str:
    cleaned = "".join(char if char in runs.SAFE_ID_CHARS else "_" for char in raw).strip("._-")
    if not cleaned:
        cleaned = fallback
    if not cleaned[0].isalnum():
        cleaned = f"{fallback}_{cleaned}"
    return runs.clean_id(cleaned[:80], label=fallback)


def _checkpoint_label(checkpoint_ref: str, *, index: int) -> str:
    name = Path(checkpoint_ref).name
    for suffix in (".pth.tar", ".tar", ".pth", ".pt", ".bin"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name:
        name = f"checkpoint_{index:03d}"
    return _safe_generated_id(name, fallback=f"checkpoint_{index:03d}")


def _checkpoint_ref_for_iteration(*, run_id: str, iteration: int) -> str:
    return (
        runs.checkpoints_root_ref(TASK_ID, run_id)
        / "lightzero"
        / f"iteration_{iteration}.pth.tar"
    ).as_posix()


def _infer_checkpoint_template(checkpoint_ref: str) -> str:
    template = re.sub(
        r"iteration_\d+(\.pth\.tar|\.tar|\.pth|\.pt|\.bin)$",
        r"iteration_{iteration}\1",
        checkpoint_ref,
    )
    if template == checkpoint_ref:
        raise ValueError(
            "selected_iterations needs checkpoint_ref_template when checkpoint_ref "
            "does not end with iteration_<n>.pth.tar"
        )
    return template


def _selected_checkpoint_refs(
    *,
    run_id: str,
    checkpoint_ref: str,
    checkpoint_refs: str | None,
    checkpoint_ref_template: str | None,
    selected_iterations: str | None,
) -> list[str]:
    explicit_refs = _split_csv(checkpoint_refs)
    iterations = _parse_iterations(selected_iterations)
    if explicit_refs and (checkpoint_ref_template or iterations):
        raise ValueError("use either checkpoint_refs or checkpoint template/selected_iterations")
    if explicit_refs:
        return explicit_refs
    if iterations:
        if checkpoint_ref_template:
            return [checkpoint_ref_template.format(iteration=iteration) for iteration in iterations]
        if checkpoint_ref != DEFAULT_CHECKPOINT_REF:
            template = _infer_checkpoint_template(checkpoint_ref)
            return [template.format(iteration=iteration) for iteration in iterations]
        return [_checkpoint_ref_for_iteration(run_id=run_id, iteration=iteration) for iteration in iterations]
    if checkpoint_ref_template:
        raise ValueError("checkpoint_ref_template requires selected_iterations")
    return [checkpoint_ref]


def _output_ref(
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
        f"curvytron_visual_survival_eval_{checkpoint_label}"
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
    seed_label = _seed_panel_label(seeds)
    return (
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, eval_id)
        / f"manifest_steps{max_eval_steps}_seeds{seed_label}_{stamp}.json"
    ).as_posix()


def _seed_panel_label(seeds: list[int]) -> str:
    payload = ",".join(str(seed) for seed in seeds)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"n{len(seeds)}_{digest}"


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


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_zero(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _death_slot_value(info: dict[str, Any], key: str, *, slot: int = 0) -> Any:
    value = info.get(key)
    if value is None:
        return None
    row_value = _row_zero(value)
    if isinstance(row_value, list):
        if slot < 0 or slot >= len(row_value):
            return None
        return row_value[slot]
    return row_value


def _first_death_from_info(info: dict[str, Any]) -> dict[str, Any]:
    death_count = _int_or_none(_row_zero(info.get("death_count")))
    if death_count is not None and death_count <= 0:
        return {
            "death_player": None,
            "death_cause": None,
            "death_cause_name": None,
            "death_hit_owner": None,
        }
    return {
        "death_player": _death_slot_value(info, "death_player"),
        "death_cause": _death_slot_value(info, "death_cause"),
        "death_cause_name": _death_slot_value(info, "death_cause_name"),
        "death_hit_owner": _death_slot_value(info, "death_hit_owner"),
    }


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


def _extract_eval_action(output: Any) -> int:
    if isinstance(output, dict):
        if "action" in output:
            return int(_float_or_plain(output["action"]))
        if 0 in output:
            return _extract_eval_action(output[0])
        if "0" in output:
            return _extract_eval_action(output["0"])
    if isinstance(output, (list, tuple)) and output:
        return _extract_eval_action(output[0])
    raise ValueError(f"could not extract action from policy eval output: {output!r}")


def _policy_model_device(policy: Any) -> Any:
    import torch

    model = getattr(policy, "_model", None)
    if model is None:
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _policy_eval_action(policy: Any, observation: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
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
        "source": "policy_eval_mode",
        "action": _extract_eval_action(output),
        "compact_output": _compact_mcts_output(output),
    }


def _summarize_observation(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {"type": type(observation).__name__}
    raw = observation.get("observation")
    mask = observation.get("action_mask")
    return {
        "keys": sorted(str(key) for key in observation),
        "observation_shape": [int(item) for item in getattr(raw, "shape", [])],
        "observation_dtype": str(getattr(raw, "dtype", "missing")),
        "action_mask": _to_plain(mask),
        "to_play": _float_or_plain(observation.get("to_play")),
        "timestep": _float_or_plain(observation.get("timestep")),
    }


def _infer_model_support_config_from_state_dict(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    def first_output_size(suffixes: tuple[str, ...]) -> int | None:
        for key, value in state_dict.items():
            key_text = str(key)
            if not any(key_text.endswith(suffix) for suffix in suffixes):
                continue
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) < 1:
                continue
            return int(shape[0])
        return None

    reward_size = first_output_size(
        (
            "dynamics_network.fc_reward_head.3.weight",
            "dynamics_network.reward_head.3.weight",
            "reward_head.3.weight",
        )
    )
    value_size = first_output_size(
        (
            "prediction_network.fc_value.3.weight",
            "prediction_network.value_head.3.weight",
            "value_head.3.weight",
        )
    )
    config: dict[str, Any] = {}
    if reward_size is not None:
        config["model_reward_support_size"] = reward_size
    if value_size is not None:
        config["model_value_support_size"] = value_size
    sizes = [size for size in (reward_size, value_size) if size is not None]
    if sizes and all(size == sizes[0] for size in sizes) and sizes[0] % 2 == 1:
        config["model_support_scale"] = (sizes[0] - 1) // 2
    return config


def _make_policy_and_env(
    *,
    state_dict: dict[str, Any],
    seed: int,
    use_cuda: bool,
    source_max_steps: int,
    num_simulations: int,
    batch_size: int,
    telemetry_path: Path,
    env_variant: str,
    reward_variant: str,
    model_env_variant: str | None = None,
    model_reward_variant: str | None = None,
    opponent_policy_kind: str,
    opponent_checkpoint: dict[str, Any] | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
) -> tuple[Any, Any, dict[str, Any]]:
    from ding.config import compile_config
    from ding.envs import get_vec_env_setting
    from lzero.policy.muzero import MuZeroPolicy

    patched = _build_visual_survival_configs(
        seed=seed,
        exp_name=Path("/tmp/curvytron_visual_survival_eval_exp"),
        telemetry_path=telemetry_path,
        cuda=use_cuda,
        max_env_step=source_max_steps,
        source_max_steps=source_max_steps,
        decision_ms=DEFAULT_DECISION_MS,
        collector_env_num=DEFAULT_COLLECTOR_ENV_NUM,
        evaluator_env_num=1,
        n_evaluator_episode=DEFAULT_N_EVALUATOR_EPISODE,
        n_episode=DEFAULT_N_EPISODE,
        num_simulations=num_simulations,
        batch_size=batch_size,
        lightzero_eval_freq=DEFAULT_LIGHTZERO_EVAL_FREQ,
        lightzero_multi_gpu=False,
        max_train_iter=DEFAULT_MAX_TRAIN_ITER,
        save_ckpt_after_iter=DEFAULT_SAVE_CKPT_AFTER_ITER,
        env_variant=env_variant,
        reward_variant=reward_variant,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id="none",
        disable_death_for_profile=False,
        env_telemetry_stride=DEFAULT_ENV_TELEMETRY_STRIDE,
        env_manager_type=DEFAULT_ENV_MANAGER_TYPE,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint=opponent_checkpoint,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
    )
    model_env_variant = model_env_variant or env_variant
    # Only used to rebuild the checkpoint model shape; survival scoring stays steps-based.
    model_reward_variant = model_reward_variant or reward_variant
    model_target_patches: list[dict[str, Any]] = []
    if model_env_variant != env_variant or model_reward_variant != reward_variant:
        from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
            _lightzero_target_config_for_reward,
            _target_config_patches,
        )

        model_target_config = _lightzero_target_config_for_reward(
            env_variant=model_env_variant,
            reward_variant=model_reward_variant,
            source_max_steps=source_max_steps,
        )
        model_target_patches.extend(_target_config_patches(
            patched["main_config"],
            model_target_config,
        ))
        patched["surface"]["model_env_variant"] = model_env_variant
        patched["surface"]["model_reward_variant"] = model_reward_variant
        patched["surface"]["model_lightzero_target_config"] = model_target_config
    inferred_support_config = _infer_model_support_config_from_state_dict(state_dict)
    if inferred_support_config:
        from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
            _target_config_patches,
        )

        model_target_patches.extend(_target_config_patches(
            patched["main_config"],
            inferred_support_config,
        ))
        patched["surface"]["checkpoint_inferred_model_support_config"] = (
            inferred_support_config
        )
    if model_target_patches:
        patched["surface"]["model_target_config_patches"] = model_target_patches
    cfg = compile_config(
        copy.deepcopy(patched["main_config"]),
        seed=seed,
        auto=True,
        create_cfg=copy.deepcopy(patched["create_config"]),
        save_cfg=False,
    )
    if not hasattr(cfg.policy, "device"):
        cfg.policy.device = "cuda" if use_cuda else "cpu"
    elif use_cuda and str(cfg.policy.device) == "cpu":
        cfg.policy.device = "cuda"
    elif not use_cuda:
        cfg.policy.device = "cpu"
    cfg.policy.cuda = use_cuda

    policy = MuZeroPolicy(cfg.policy)
    policy_model = getattr(policy, "_model", None)
    if policy_model is None:
        raise AttributeError("MuZeroPolicy has no _model attribute")
    load = _load_state_dict_probe(policy_model, state_dict)
    if hasattr(policy_model, "eval"):
        policy_model.eval()
    if not load.get("ok") or not load.get("strict"):
        raise RuntimeError(f"strict policy model checkpoint load failed: {load}")

    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    one_env_cfg = evaluator_env_cfg[0] if evaluator_env_cfg else collector_env_cfg[0]
    try:
        env = env_fn(cfg=one_env_cfg)
    except TypeError:
        env = env_fn(one_env_cfg)
    if hasattr(env, "seed"):
        env.seed(seed, dynamic_seed=False)
    surface = {
        "compiled_policy": {
            "cuda": bool(cfg.policy.cuda),
            "device": str(getattr(cfg.policy, "device", "missing")),
            "num_simulations": int(cfg.policy.num_simulations),
            "batch_size": int(cfg.policy.batch_size),
            "model_type": str(cfg.policy.model.model_type),
            "observation_shape": _to_plain(cfg.policy.model.observation_shape),
            "image_channel": int(cfg.policy.model.image_channel),
            "frame_stack_num": int(cfg.policy.model.frame_stack_num),
            "self_supervised_learning_loss": bool(
                cfg.policy.model.self_supervised_learning_loss
            ),
            "action_space_size": int(cfg.policy.model.action_space_size),
        },
        "env": {
            "type": type(env).__module__ + "." + type(env).__name__,
            "env_type": str(cfg.env.type),
            "env_id": str(cfg.env.env_id),
            "source_max_steps": int(cfg.env.source_max_steps),
        },
        "load_state_dict": load,
        "patched_surface": patched["surface"],
    }
    return policy, env, surface


def _run_survival_episode(
    *,
    policy: Any,
    env: Any,
    seed: int,
    max_eval_steps: int,
    step_detail_limit: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        observation = env.reset(seed=seed)
    except TypeError:
        observation = env.reset()

    reset_summary = _summarize_observation(observation)
    steps: list[dict[str, Any]] = []
    total_reward = 0.0
    actions: list[int] = []
    done = False
    last_info: dict[str, Any] = {}
    failure: dict[str, Any] | None = None

    for step_index in range(max_eval_steps):
        try:
            action_result = _policy_eval_action(policy, observation)
            action = int(action_result["action"])
            timestep = env.step(action)
            next_observation, reward, done, info = _timestep_parts(timestep)
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            failure = {"step_index": step_index, **_exception_result(exc)}
            break

        reward_float = float(_float_or_plain(reward))
        total_reward += reward_float
        actions.append(action)
        last_info = _to_plain(info) if isinstance(info, dict) else {"raw_info": _to_plain(info)}
        if step_detail_limit is None or len(steps) < step_detail_limit:
            steps.append(
                {
                    "step_index": step_index,
                    "action": action,
                    "reward": reward_float,
                    "done": bool(done),
                    "terminal_reason": last_info.get("terminal_reason"),
                    "info": last_info,
                    "mcts_output": action_result.get("compact_output"),
                }
            )
        observation = next_observation
        if done:
            break

    close_result: dict[str, Any] = {"called": False}
    if hasattr(env, "close"):
        try:
            env.close()
            close_result = {"called": True, "ok": True}
        except Exception as exc:  # pragma: no cover - remote cleanup diagnosis.
            close_result = {"called": True, **_exception_result(exc)}

    terminal_reason = last_info.get("terminal_reason")
    if terminal_reason in (None, "", "none"):
        terminal_reason = "cap" if len(actions) >= max_eval_steps and not done else "unknown"
    death = _first_death_from_info(last_info)
    steps_survived = len(actions)
    elapsed_sec = time.perf_counter() - started
    return {
        "ok": failure is None and bool(actions),
        "seed": seed,
        "cap": max_eval_steps,
        "steps_run": len(actions),
        "steps_survived": steps_survived,
        "total_reward": total_reward,
        "done": bool(done),
        "terminal_reason": terminal_reason,
        "death_player": death["death_player"],
        "death_cause": death["death_cause"],
        "death_cause_name": death["death_cause_name"],
        "death_hit_owner": death["death_hit_owner"],
        "final_action": actions[-1] if actions else None,
        "decision_ms": last_info.get("decision_ms"),
        "action_histogram": dict(sorted(Counter(str(action) for action in actions).items())),
        "actions": actions,
        "reset_observation": reset_summary,
        "steps": steps,
        "steps_truncated": step_detail_limit is not None and len(actions) > step_detail_limit,
        "failure": failure,
        "close": close_result,
        "elapsed_sec": round(elapsed_sec, 6),
        "steps_per_sec": round(len(actions) / elapsed_sec, 6) if elapsed_sec > 0 else None,
    }


def _eval_checkpoint(
    *,
    checkpoint_path: Path,
    seed: int,
    max_eval_steps: int,
    step_detail_limit: int | None,
    use_cuda: bool,
    source_max_steps: int,
    num_simulations: int,
    batch_size: int,
    telemetry_path: Path,
    env_variant: str,
    eval_reward_variant: str,
    model_reward_variant: str | None,
    opponent_policy_kind: str,
    opponent_checkpoint: dict[str, Any] | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path)
    state_candidate = _find_state_dict(checkpoint)
    if state_candidate is None:
        raise ValueError("no tensor state dict found under common LightZero checkpoint keys")
    state_path, state_dict = state_candidate
    policy, env, surface = _make_policy_and_env(
        state_dict=state_dict,
        seed=seed,
        use_cuda=use_cuda,
        source_max_steps=source_max_steps,
        num_simulations=num_simulations,
        batch_size=batch_size,
        telemetry_path=telemetry_path,
        env_variant=env_variant,
        reward_variant=eval_reward_variant,
        model_reward_variant=model_reward_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint=opponent_checkpoint,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
    )
    episode = _run_survival_episode(
        policy=policy,
        env=env,
        seed=seed,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
    )
    return {
        "checkpoint": _checkpoint_summary(checkpoint_path, include_sha256=False),
        "load": {
            "ok": True,
            "state_dict_path": state_path,
            "tensor_count": len(state_dict),
            "keys_sample": list(state_dict)[:40],
        },
        "surface": surface,
        "episode": episode,
    }


def _row_from_result(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    episode = result.get("episode") if isinstance(result.get("episode"), dict) else {}
    status = result.get("status") if isinstance(result.get("status"), dict) else {}
    artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
    return {
        "index": job.get("index"),
        "checkpoint_label": job.get("checkpoint_label"),
        "seed": job.get("seed"),
        "steps_survived": status.get("steps_survived", episode.get("steps_survived")),
        "cap": status.get("cap", episode.get("cap")),
        "terminal_reason": episode.get("terminal_reason"),
        "death_player": episode.get("death_player"),
        "death_cause": episode.get("death_cause"),
        "death_cause_name": episode.get("death_cause_name"),
        "death_hit_owner": episode.get("death_hit_owner"),
        "final_action": episode.get("final_action"),
        "decision_ms": episode.get("decision_ms"),
        "ok": bool(result.get("ok")),
        "strict_load": status.get("strict_policy_model_load_ok"),
        "training_reward": episode.get("total_reward"),
        "action_histogram": episode.get("action_histogram"),
        "elapsed_seconds": result.get("remote_elapsed_sec"),
        "artifact_ref": artifact.get("ref"),
        "checkpoint_ref": job.get("checkpoint_ref"),
        "error_type": result.get("wrapper_error", {}).get("error_type")
        if isinstance(result.get("wrapper_error"), dict)
        else None,
    }


def _table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_to_plain(value), sort_keys=True, separators=(",", ":"))
    return str(value)


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


def _tsv_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_table_value(row.get(column)) for column in columns))
    return "\n".join(lines)


def _checkpoint_sort_key(label: Any) -> tuple[int, int | str]:
    text = str(label)
    match = re.search(r"iteration_(\d+)", text)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, text)


def _survival_aggregate_table(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in table:
        checkpoint = str(row.get("checkpoint_label") or "")
        by_checkpoint.setdefault(checkpoint, []).append(row)

    aggregate_rows = []
    for checkpoint, rows in sorted(by_checkpoint.items(), key=lambda item: _checkpoint_sort_key(item[0])):
        steps = [
            value
            for value in (_number(row.get("steps_survived")) for row in rows)
            if value is not None
        ]
        scores = [
            value
            for value in (_number(row.get("training_reward")) for row in rows)
            if value is not None
        ]
        ok_count = sum(1 for row in rows if row.get("ok"))
        capped_count = sum(
            1
            for row in rows
            if row.get("terminal_reason") == "cap"
            or (
                _number(row.get("steps_survived")) is not None
                and _number(row.get("cap")) is not None
                and _number(row.get("steps_survived")) >= _number(row.get("cap"))
            )
        )
        elapsed = [
            value
            for value in (_number(row.get("elapsed_seconds")) for row in rows)
            if value is not None
        ]
        aggregate_rows.append(
            {
                "checkpoint": checkpoint,
                "seeds": len(rows),
                "mean_steps": _mean(steps),
                "median_steps": _median(steps),
                "min_steps": min(steps) if steps else None,
                "max_steps": max(steps) if steps else None,
                "ok_count": ok_count,
                "capped_count": capped_count,
                "failure_count": len(rows) - ok_count,
                "mean_training_reward": _mean(scores),
                "mean_elapsed_sec": _mean(elapsed),
            }
        )
    return aggregate_rows


def _survival_table(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": row.get("checkpoint_label"),
            "seed": row.get("seed"),
            "steps": row.get("steps_survived"),
            "cap": row.get("cap"),
            "terminal": row.get("terminal_reason"),
            "death_player": row.get("death_player"),
            "death_cause": row.get("death_cause"),
            "death_cause_name": row.get("death_cause_name"),
            "death_hit_owner": row.get("death_hit_owner"),
            "final_action": row.get("final_action"),
            "decision_ms": row.get("decision_ms"),
            "actions": row.get("action_histogram"),
            "ok": row.get("ok"),
            "strict": row.get("strict_load"),
            "elapsed_sec": row.get("elapsed_seconds"),
            "artifact_ref": row.get("artifact_ref"),
        }
        for row in table
    ]


def _run_eval(
    *,
    compute: str,
    checkpoint_ref: str,
    output_ref: str,
    run_id: str,
    attempt_id: str,
    seed: int,
    max_eval_steps: int,
    step_detail_limit: int | None,
    source_max_steps: int,
    num_simulations: int,
    batch_size: int,
    emit_result_json: bool,
    quiet_framework_logs: bool,
    env_variant: str,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    model_reward_variant: str | None = None,
    opponent_policy_kind: str,
    opponent_checkpoint_ref: str | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
) -> dict[str, Any]:
    if compute not in COMPUTE_CHOICES:
        raise ValueError(f"unknown compute {compute!r}; expected one of {COMPUTE_CHOICES!r}")
    if env_variant not in ENV_VARIANT_CHOICES:
        raise ValueError(
            f"unknown env_variant {env_variant!r}; expected one of {ENV_VARIANT_CHOICES!r}"
        )
    opponent_policy_kind = _normalize_opponent_policy_kind_for_env(
        env_variant=env_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
    )
    eval_reward_variant = _normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    if model_reward_variant is not None:
        # Only used to reconstruct checkpoint model/support shape, not to score eval.
        model_reward_variant = _normalize_reward_variant_for_env(
            env_variant=env_variant,
            reward_variant=model_reward_variant,
        )
    effective_model_reward_variant = model_reward_variant or eval_reward_variant
    if opponent_policy_kind not in OPPONENT_POLICY_KIND_CHOICES:
        raise ValueError(
            f"unknown opponent_policy_kind {opponent_policy_kind!r}; "
            f"expected one of {OPPONENT_POLICY_KIND_CHOICES!r}"
        )
    use_cuda = compute in ("gpu-l4-t4", GPU_L4_T4_CPU40_COMPUTE)
    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    output_path = runs.volume_path(RUNS_MOUNT, output_ref)
    telemetry_path = output_path.with_suffix(".env_steps.jsonl")
    opponent_checkpoint = _resolve_opponent_checkpoint_for_env(
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_checkpoint_report_ref=None,
    )
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "gym": _version_or_missing("gym"),
    }
    config = {
        "job_kind": "lightzero_curvytron_visual_survival_checkpoint_eval",
        "compute": compute,
        "use_cuda": use_cuda,
        "checkpoint_ref": checkpoint_ref,
        "output_ref": output_ref,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "seed": seed,
        "max_eval_steps": max_eval_steps,
        "step_detail_limit": step_detail_limit,
        "source_max_steps": source_max_steps,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "eval_primary_metric": "steps_survived",
        "env_variant": env_variant,
        "eval_reward_variant": eval_reward_variant,
        "env_reward_variant": eval_reward_variant,
        "reward_variant": eval_reward_variant,
        "reward_variant_role": "backward_compatible_alias_for_eval_reward_variant",
        "model_reward_variant": model_reward_variant,
        "effective_model_reward_variant": effective_model_reward_variant,
        "model_reward_variant_role": "checkpoint_model_reconstruction_only_not_scoring",
        "training_reward_telemetry_field": "episode.total_reward",
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "opponent_policy_kind": opponent_policy_kind,
        "opponent_checkpoint_ref": opponent_checkpoint_ref,
        "opponent_snapshot_ref": opponent_snapshot_ref,
        "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
    }
    try:
        with _QuietFrameworkOutput(quiet_framework_logs):
            eval_result = _eval_checkpoint(
                checkpoint_path=checkpoint_path,
                seed=seed,
                max_eval_steps=max_eval_steps,
                step_detail_limit=step_detail_limit,
                use_cuda=use_cuda,
                source_max_steps=source_max_steps,
                num_simulations=num_simulations,
                batch_size=batch_size,
                telemetry_path=telemetry_path,
                env_variant=env_variant,
                eval_reward_variant=eval_reward_variant,
                model_reward_variant=model_reward_variant,
                opponent_policy_kind=opponent_policy_kind,
                opponent_checkpoint=opponent_checkpoint,
                opponent_snapshot_ref=opponent_snapshot_ref,
                opponent_checkpoint_state_key=opponent_checkpoint_state_key,
            )
        episode = eval_result["episode"]
        load_state = eval_result["surface"]["load_state_dict"]
        result: dict[str, Any] = {
            "schema": "curvyzero_lightzero_curvytron_visual_survival_eval/v0",
            **eval_result,
            "ok": bool(episode.get("ok")),
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "status": {
                "checkpoint_load_ok": True,
                "strict_policy_model_load_ok": bool(load_state.get("ok") and load_state.get("strict")),
                "env_reset_ok": bool(episode.get("reset_observation")),
                "policy_could_act_in_real_env": bool(episode.get("actions")),
                "steps_survived": episode.get("steps_survived"),
                "cap": max_eval_steps,
            },
        }
    except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
        result = {
            "schema": "curvyzero_lightzero_curvytron_visual_survival_eval/v0",
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "checkpoint": _checkpoint_summary(checkpoint_path, include_sha256=False),
            "status": {
                "checkpoint_load_ok": checkpoint_path.is_file(),
                "strict_policy_model_load_ok": False,
                "env_reset_ok": False,
                "policy_could_act_in_real_env": False,
                "steps_survived": 0,
                "cap": max_eval_steps,
            },
            "wrapper_error": _exception_result(exc),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(output_path, _to_plain(result))
    result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    if telemetry_path.exists():
        result["telemetry_artifact"] = runs.file_summary(telemetry_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    if emit_result_json:
        print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=EVAL_TIMEOUT_SEC, cpu=2.0)
def curvytron_visual_survival_eval_cpu(
    checkpoint_ref: str,
    output_ref: str,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    seed: int = 0,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    emit_result_json: bool = True,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    model_reward_variant: str | None = None,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
) -> dict[str, Any]:
    return _run_eval(
        compute="cpu",
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        source_max_steps=source_max_steps,
        num_simulations=num_simulations,
        batch_size=batch_size,
        emit_result_json=emit_result_json,
        quiet_framework_logs=quiet_framework_logs,
        env_variant=env_variant,
        reward_variant=reward_variant,
        model_reward_variant=model_reward_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=EVAL_TIMEOUT_SEC,
    cpu=8.0,
    memory=32768,
    gpu=CHEAP_GPU_RESOURCE,
)
def curvytron_visual_survival_eval_gpu(
    checkpoint_ref: str,
    output_ref: str,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    seed: int = 0,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    emit_result_json: bool = True,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    model_reward_variant: str | None = None,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
) -> dict[str, Any]:
    return _run_eval(
        compute="gpu-l4-t4",
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        source_max_steps=source_max_steps,
        num_simulations=num_simulations,
        batch_size=batch_size,
        emit_result_json=emit_result_json,
        quiet_framework_logs=quiet_framework_logs,
        env_variant=env_variant,
        reward_variant=reward_variant,
        model_reward_variant=model_reward_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=EVAL_TIMEOUT_SEC,
    cpu=40.0,
    memory=65536,
    gpu=CHEAP_GPU_RESOURCE,
)
def curvytron_visual_survival_eval_gpu_cpu40(
    checkpoint_ref: str,
    output_ref: str,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    seed: int = 0,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    emit_result_json: bool = True,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    model_reward_variant: str | None = None,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
) -> dict[str, Any]:
    return _run_eval(
        compute=GPU_L4_T4_CPU40_COMPUTE,
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        source_max_steps=source_max_steps,
        num_simulations=num_simulations,
        batch_size=batch_size,
        emit_result_json=emit_result_json,
        quiet_framework_logs=quiet_framework_logs,
        env_variant=env_variant,
        reward_variant=reward_variant,
        model_reward_variant=model_reward_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
    )


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=5 * 60, cpu=1.0)
def curvytron_visual_survival_eval_manifest(
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
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    checkpoint_refs: str | None = None,
    checkpoint_ref_template: str | None = None,
    selected_iterations: str | None = None,
    output_ref: str | None = None,
    manifest_ref: str | None = None,
    parallel: bool = False,
    eval_id: str = DEFAULT_EVAL_ID,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    seed: int = 0,
    eval_seeds: str | None = None,
    eval_seed_count: int | None = None,
    eval_seed_rng_seed: int | None = DEFAULT_EVAL_SEED_RNG_SEED,
    max_eval_steps: int = DEFAULT_MAX_EVAL_STEPS,
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    summary_only: bool = False,
    slim_manifest: bool = True,
    quiet_framework_logs: bool = DEFAULT_QUIET_FRAMEWORK_LOGS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    model_reward_variant: str | None = None,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
) -> None:
    if compute == "cpu":
        eval_fn = curvytron_visual_survival_eval_cpu
    elif compute == "gpu-l4-t4":
        eval_fn = curvytron_visual_survival_eval_gpu
    elif compute == GPU_L4_T4_CPU40_COMPUTE:
        eval_fn = curvytron_visual_survival_eval_gpu_cpu40
    else:
        raise ValueError(f"unknown compute {compute!r}; expected one of {COMPUTE_CHOICES!r}")
    eval_reward_variant = _normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    resolved_model_reward_variant = (
        _normalize_reward_variant_for_env(
            env_variant=env_variant,
            reward_variant=model_reward_variant,
        )
        if model_reward_variant is not None
        else eval_reward_variant
    )

    selected_refs = _selected_checkpoint_refs(
        run_id=run_id,
        checkpoint_ref=checkpoint_ref,
        checkpoint_refs=checkpoint_refs,
        checkpoint_ref_template=checkpoint_ref_template,
        selected_iterations=selected_iterations,
    )
    effective_eval_seed_count = (
        eval_seed_count
        if eval_seed_count is not None or eval_seeds is not None
        else DEFAULT_EVAL_SEED_COUNT
    )
    eval_seed_values, eval_seed_sampler_seed = _parse_eval_seeds(
        seed=seed,
        eval_seeds=eval_seeds,
        eval_seed_count=effective_eval_seed_count,
        eval_seed_rng_seed=eval_seed_rng_seed,
    )
    single_job = len(selected_refs) == 1 and len(eval_seed_values) == 1
    clean_eval_id = _safe_generated_id(eval_id, fallback="eval")
    stamp = runs.utc_stamp()

    if output_ref is not None and not single_job:
        raise ValueError("output_ref is only valid for a single checkpoint/seed eval")
    output_refs = [
        output_ref
        or _output_ref(
            run_id=run_id,
            attempt_id=attempt_id,
            eval_id=clean_eval_id,
            checkpoint_label=_checkpoint_label(ref, index=index),
            max_eval_steps=max_eval_steps,
            seed=eval_seed,
            stamp=stamp,
        )
        for eval_seed in eval_seed_values
        for index, ref in enumerate(selected_refs)
    ]
    jobs = [
        {
            "index": job_index,
            "checkpoint_ref": ref,
            "checkpoint_label": _checkpoint_label(ref, index=checkpoint_index),
            "output_ref": output_refs[job_index],
            "seed": eval_seed,
        }
        for job_index, (eval_seed, checkpoint_index, ref) in enumerate(
            (eval_seed, checkpoint_index, ref)
            for eval_seed in eval_seed_values
            for checkpoint_index, ref in enumerate(selected_refs)
        )
    ]
    remote_kwargs = {
        "source_max_steps": max(int(source_max_steps), int(max_eval_steps)),
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "max_eval_steps": max_eval_steps,
        "step_detail_limit": step_detail_limit,
        "emit_result_json": not summary_only,
        "quiet_framework_logs": quiet_framework_logs,
        "env_variant": env_variant,
        "reward_variant": eval_reward_variant,
        "model_reward_variant": resolved_model_reward_variant,
        "opponent_policy_kind": opponent_policy_kind,
        "opponent_checkpoint_ref": opponent_checkpoint_ref,
        "opponent_snapshot_ref": opponent_snapshot_ref,
        "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
    }

    if parallel or len(jobs) > 1:
        starmap_inputs = [
            (job["checkpoint_ref"], job["output_ref"], run_id, attempt_id, job["seed"])
            for job in jobs
        ]
        results = list(eval_fn.starmap(starmap_inputs, kwargs=remote_kwargs))
    else:
        job = jobs[0]
        results = [
            eval_fn.remote(
                checkpoint_ref=job["checkpoint_ref"],
                output_ref=job["output_ref"],
                run_id=run_id,
                attempt_id=attempt_id,
                seed=job["seed"],
                **remote_kwargs,
            )
        ]

    table = [_row_from_result(job, result) for job, result in zip(jobs, results, strict=True)]
    survival_aggregate_table = _survival_aggregate_table(table)
    survival_table = _survival_table(table)
    manifest_ref = manifest_ref or _manifest_ref(
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=clean_eval_id,
        max_eval_steps=max_eval_steps,
        seeds=eval_seed_values,
        stamp=runs.utc_stamp(),
    )
    manifest = {
        "schema": "curvyzero_lightzero_curvytron_visual_survival_parallel_eval/v0",
        "ok": all(bool(result.get("ok")) for result in results),
        "created_at": runs.utc_timestamp(),
        "job_kind": "lightzero_curvytron_visual_survival_checkpoint_curve_eval",
        "eval_id": clean_eval_id,
        "retention_policy": (
            "Keep per-checkpoint eval JSON artifacts and this manifest under the "
            "attempt eval root until the run is archived."
        ),
        "selection": {
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_refs": checkpoint_refs,
            "checkpoint_ref_template": checkpoint_ref_template,
            "selected_iterations": selected_iterations,
            "eval_seeds": eval_seed_values,
            "eval_seed_count": effective_eval_seed_count,
            "eval_seed_sampler_seed": eval_seed_sampler_seed,
            "eval_primary_metric": "steps_survived",
            "env_variant": env_variant,
            "eval_reward_variant": eval_reward_variant,
            "env_reward_variant": eval_reward_variant,
            "reward_variant": eval_reward_variant,
            "reward_variant_role": "backward_compatible_alias_for_eval_reward_variant",
            "model_reward_variant": resolved_model_reward_variant,
            "effective_model_reward_variant": resolved_model_reward_variant,
            "model_reward_variant_role": "checkpoint_model_reconstruction_only_not_scoring",
            "opponent_policy_kind": opponent_policy_kind,
            "opponent_checkpoint_ref": opponent_checkpoint_ref,
            "opponent_snapshot_ref": opponent_snapshot_ref,
            "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
            "jobs": jobs,
        },
        "config": {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "compute": compute,
            "max_eval_steps": max_eval_steps,
            "step_detail_limit": step_detail_limit,
            "source_max_steps": max(int(source_max_steps), int(max_eval_steps)),
            "num_simulations": num_simulations,
            "batch_size": batch_size,
            "default_eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
            "slim_manifest": slim_manifest,
            "eval_primary_metric": "steps_survived",
            "env_variant": env_variant,
            "eval_reward_variant": eval_reward_variant,
            "env_reward_variant": eval_reward_variant,
            "reward_variant": eval_reward_variant,
            "reward_variant_role": "backward_compatible_alias_for_eval_reward_variant",
            "model_reward_variant": resolved_model_reward_variant,
            "effective_model_reward_variant": resolved_model_reward_variant,
            "model_reward_variant_role": "checkpoint_model_reconstruction_only_not_scoring",
            "training_reward_telemetry_field": "episode.total_reward",
            "LightZero": LIGHTZERO_VERSION,
            "remote_root": str(REMOTE_ROOT),
            "volume_name": VOLUME_NAME,
            "opponent_policy_kind": opponent_policy_kind,
            "opponent_checkpoint_ref": opponent_checkpoint_ref,
            "opponent_snapshot_ref": opponent_snapshot_ref,
            "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
        },
        "table_fields": [
            "checkpoint_label",
            "seed",
            "steps_survived",
            "cap",
            "terminal_reason",
            "death_player",
            "death_cause",
            "death_cause_name",
            "death_hit_owner",
            "final_action",
            "decision_ms",
            "ok",
            "strict_load",
            "training_reward",
            "action_histogram",
            "elapsed_seconds",
            "artifact_ref",
            "checkpoint_ref",
        ],
        "survival_aggregate_table": survival_aggregate_table,
        "survival_table": survival_table,
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
    manifest_result = curvytron_visual_survival_eval_manifest.remote(
        manifest_ref=manifest_ref,
        manifest=manifest,
    )
    summary = {
        "survival_aggregate_table": survival_aggregate_table,
        "survival_table": survival_table,
        "table": table,
        "manifest": manifest_result,
        "manifest_ref": manifest_ref,
        "jobs": jobs,
        "output_refs": [job["output_ref"] for job in jobs],
    }
    if not summary_only:
        summary["results"] = results
    else:
        print("# aggregate_by_checkpoint")
        print(_tsv_table(
            survival_aggregate_table,
            [
                "checkpoint",
                "seeds",
                "mean_steps",
                "median_steps",
                "min_steps",
                "max_steps",
                "ok_count",
                "capped_count",
                "failure_count",
                "mean_training_reward",
                "mean_elapsed_sec",
            ],
        ))
        print("# per_checkpoint_seed_curve")
        print(_tsv_table(
            survival_table,
            [
                "checkpoint",
                "seed",
                "steps",
                "cap",
                "terminal",
                "actions",
                "ok",
                "strict",
                "elapsed_sec",
                "artifact_ref",
            ],
        ))
        if eval_seed_sampler_seed is not None:
            print(f"eval_seed_sampler_seed\t{eval_seed_sampler_seed}")
            print("eval_seeds\t" + ",".join(str(item) for item in eval_seed_values))
        print(f"manifest_ref\t{manifest_ref}")
        return
    print(json.dumps(_to_plain(summary), indent=2))
