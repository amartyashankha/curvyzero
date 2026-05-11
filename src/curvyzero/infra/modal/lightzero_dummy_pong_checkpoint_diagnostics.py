"""Modal checkpoint diagnostics for the 512/8 LightZero dummy Pong run.

Run from the repository root:

    uv run --extra modal modal run \
      -m curvyzero.infra.modal.lightzero_dummy_pong_checkpoint_diagnostics
"""

from __future__ import annotations

import copy
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTOR_ENV_NUM,
    DEFAULT_ENV,
    DEFAULT_EVALUATOR_ENV_NUM,
    DEFAULT_FEATURE_MODE,
    DEFAULT_UPDATE_PER_COLLECT,
    LIGHTZERO_VERSION,
    patched_dummy_pong_configs,
)


APP_NAME = "curvyzero-lightzero-dummy-pong-checkpoint-diagnostics"
TASK_ID = "lightzero-dummy-pong"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_RUN_ID = "lz-dpong-20260509T144635Z-eb5a0ed35de0"
DEFAULT_ATTEMPT_ID = "attempt-20260509T144635Z-ece79bad80d0"
DEFAULT_CHECKPOINT_REFS = (
    "iteration_0=training/lightzero-dummy-pong/"
    "lz-dpong-20260509T144635Z-eb5a0ed35de0/checkpoints/lightzero/iteration_0.pth.tar,"
    "iteration_8=training/lightzero-dummy-pong/"
    "lz-dpong-20260509T144635Z-eb5a0ed35de0/checkpoints/lightzero/iteration_8.pth.tar,"
    "ckpt_best=training/lightzero-dummy-pong/"
    "lz-dpong-20260509T144635Z-eb5a0ed35de0/checkpoints/lightzero/ckpt_best.pth.tar"
)
DEFAULT_STATE_VARIANTS = "model,target_model"

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
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
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


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_state_dict(payload: Any) -> tuple[str, dict[str, Any]]:
    from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import _find_state_dict

    found = _find_state_dict(payload)
    if found is None:
        raise ValueError("no tensor state dict found")
    return found


def _state_dict_variants(payload: Any, requested: str) -> list[tuple[str, str, dict[str, Any]]]:
    variants = []
    requested_names = [
        item.strip()
        for item in requested.replace("\n", ",").split(",")
        if item.strip()
    ]
    for name in requested_names:
        value = payload.get(name) if isinstance(payload, dict) else None
        if isinstance(value, dict) and _tensor_count(value) > 0:
            variants.append((name, name, value))
    if variants:
        return variants
    state_path, state_dict = _find_state_dict(payload)
    return [("auto", state_path, state_dict)]


def _tensor_count(value: dict[str, Any]) -> int:
    return sum(1 for item in value.values() if hasattr(item, "shape") and hasattr(item, "dtype"))


def _model_surface(
    *,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    num_simulations: int,
) -> dict[str, Any]:
    patched = patched_dummy_pong_configs(
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        collector_env_num=DEFAULT_COLLECTOR_ENV_NUM,
        evaluator_env_num=DEFAULT_EVALUATOR_ENV_NUM,
        n_evaluator_episode=1,
        num_simulations=num_simulations,
        batch_size=DEFAULT_BATCH_SIZE,
        update_per_collect=DEFAULT_UPDATE_PER_COLLECT,
    )
    model_cfg = dict(patched["main_config"]["policy"]["model"])
    model_cfg.pop("model_type", None)
    return {
        "main_config": patched["main_config"],
        "create_config": patched["create_config"],
        "model_cfg": model_cfg,
        "patched_surface": patched["patched_surface"],
    }


def _needs_residual_dynamics(state_dict: dict[str, Any]) -> bool:
    keys = [str(key) for key in state_dict]
    return any(key.startswith("dynamics_network.fc_dynamics_1.") for key in keys) and any(
        key.startswith("dynamics_network.fc_dynamics_2.") for key in keys
    )


def _model_cfg_for_state(model_cfg: dict[str, Any], state_dict: dict[str, Any]) -> dict[str, Any]:
    if _needs_residual_dynamics(state_dict):
        return {**model_cfg, "res_connection_in_dynamics": True}
    return dict(model_cfg)


def _load_model(model_cfg: dict[str, Any], state_dict: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from lzero.model.muzero_model_mlp import MuZeroModelMLP

    cfg = _model_cfg_for_state(model_cfg, state_dict)
    model = MuZeroModelMLP(**cfg)
    loaded = model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, {
        "ok": True,
        "strict": True,
        "model_config_delta": (
            {"res_connection_in_dynamics": True}
            if cfg.get("res_connection_in_dynamics") is True
            else {}
        ),
        "missing_keys": list(getattr(loaded, "missing_keys", [])),
        "unexpected_keys": list(getattr(loaded, "unexpected_keys", [])),
    }


def _load_mcts_policy(
    *,
    main_config: Any,
    create_config: Any,
    state_dict: dict[str, Any],
    seed: int,
) -> Any:
    from ding.config import compile_config
    from lzero.policy.muzero import MuZeroPolicy

    config = copy.deepcopy(main_config)
    if _needs_residual_dynamics(state_dict):
        config["policy"]["model"]["res_connection_in_dynamics"] = True
    cfg = compile_config(
        config,
        seed=seed,
        auto=True,
        create_cfg=copy.deepcopy(create_config),
        save_cfg=False,
    )
    if not hasattr(cfg.policy, "device"):
        cfg.policy.device = "cpu"
    policy = MuZeroPolicy(cfg.policy)
    policy_model = getattr(policy, "_model", None)
    if policy_model is None:
        raise AttributeError("MuZeroPolicy has no _model attribute")
    policy_model.load_state_dict(state_dict, strict=True)
    policy_model.eval()
    return policy


def _sample_eval_observations(*, seed: int, max_env_step: int, limit: int) -> list[dict[str, Any]]:
    from curvyzero.training.dummy_pong import AGENTS
    from curvyzero.training.dummy_pong import PongConfig
    from curvyzero.training.dummy_pong import PongEnv
    from curvyzero.training.dummy_pong_eval import LaggedTrackBallPolicy
    from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
    from curvyzero.training.dummy_pong_eval import TrackBallPolicy
    from curvyzero.training.lightzero_dummy_pong_features import encode_tabular_ego_observation

    config = PongConfig(max_steps=max_env_step)
    matchup_factories = [
        ("random_uniform", lambda offset: RandomUniformPolicy(seed + offset), "track_ball", lambda offset: TrackBallPolicy()),
        ("lagged_track_ball_1", lambda offset: LaggedTrackBallPolicy(delay=1), "random_uniform", lambda offset: RandomUniformPolicy(seed + 1000 + offset)),
        ("track_ball", lambda offset: TrackBallPolicy(), "lagged_track_ball_1", lambda offset: LaggedTrackBallPolicy(delay=1)),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    episode_index = 0
    for p0_name, p0_factory, p1_name, p1_factory in matchup_factories:
        for local_episode in range(4):
            episode_seed = seed + 37 * episode_index
            env = PongEnv(config)
            observations = env.reset(seed=episode_seed)
            policies = {
                "player_0": p0_factory(local_episode),
                "player_1": p1_factory(local_episode),
            }
            policy_names = {"player_0": p0_name, "player_1": p1_name}
            for agent in AGENTS:
                policies[agent].reset(episode_seed, agent)
            while len(rows) < limit:
                raster = env.raster_observation()
                joint_action = {}
                for agent in AGENTS:
                    obs = observations[agent]
                    encoded = encode_tabular_ego_observation(obs, config)
                    key = tuple(round(float(value), 5) for value in encoded.tolist())
                    if key not in seen:
                        seen.add(key)
                        rows.append(
                            {
                                "sample_index": len(rows),
                                "episode_seed": episode_seed,
                                "step": int(obs.step),
                                "agent": agent,
                                "policy_context": policy_names[agent],
                                "matchup": f"{p0_name}_p0__{p1_name}_p1",
                                "track_ball_oracle_action": _track_ball_oracle_action(obs),
                                "encoded_observation": [float(value) for value in encoded.tolist()],
                            }
                        )
                    joint_action[agent] = policies[agent].action(obs, raster, agent)
                if len(rows) >= limit:
                    break
                step = env.step(joint_action)
                observations = step.observations
                if step.terminated or step.truncated:
                    break
            episode_index += 1
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    return rows


def _track_ball_oracle_action(observation: Any) -> int:
    if observation.ball_dy_from_ego_center < 0:
        return 0
    if observation.ball_dy_from_ego_center > 0:
        return 2
    return 1


def _checkpoint_summary(path: Path, payload: Any, state_path: str, state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "ref": runs.file_ref(path, mount=RUNS_MOUNT),
        "exists": path.is_file(),
        "bytes": path.stat().st_size,
        "sha256": runs.sha256_file(path),
        "payload_keys": list(payload.keys()) if isinstance(payload, dict) else None,
        "last_iter": _to_plain(payload.get("last_iter")) if isinstance(payload, dict) else None,
        "last_step": _to_plain(payload.get("last_step")) if isinstance(payload, dict) else None,
        "optimizer_present": bool(isinstance(payload, dict) and payload.get("optimizer") is not None),
        "state_dict_path": state_path,
        "tensor_count": len(state_dict),
        "prefix_counts": dict(Counter(str(key).split(".", 1)[0] for key in state_dict)),
        "policy_tensors": _tensor_stats(state_dict, key_predicate=_is_policy_tensor),
    }


def _is_policy_tensor(key: str) -> bool:
    lowered = key.lower()
    return "policy" in lowered or lowered.endswith("fc_policy.weight") or lowered.endswith("fc_policy.bias")


def _tensor_stats(state_dict: dict[str, Any], *, key_predicate: Any) -> dict[str, Any]:
    import torch

    stats = {}
    for key, value in state_dict.items():
        key_text = str(key)
        if not key_predicate(key_text) or not hasattr(value, "detach"):
            continue
        tensor = value.detach().cpu().float().reshape(-1)
        if tensor.numel() == 0:
            continue
        stats[key_text] = {
            "shape": [int(item) for item in value.shape],
            "numel": int(tensor.numel()),
            "min": float(torch.min(tensor).item()),
            "max": float(torch.max(tensor).item()),
            "mean": float(torch.mean(tensor).item()),
            "std": float(torch.std(tensor, unbiased=False).item()),
            "l2": float(torch.linalg.vector_norm(tensor).item()),
            "nonzero": int(torch.count_nonzero(tensor).item()),
        }
    return stats


def _compare_state_dicts(left: dict[str, Any], right: dict[str, Any], *, left_label: str, right_label: str) -> dict[str, Any]:
    import torch

    common = sorted(set(left) & set(right))
    changed = []
    identical = 0
    shape_mismatch = []
    total_sq = 0.0
    right_sq = 0.0
    for key in common:
        left_value = left[key]
        right_value = right[key]
        if not hasattr(left_value, "detach") or not hasattr(right_value, "detach"):
            continue
        if tuple(left_value.shape) != tuple(right_value.shape):
            shape_mismatch.append(
                {
                    "key": str(key),
                    "left_shape": [int(item) for item in left_value.shape],
                    "right_shape": [int(item) for item in right_value.shape],
                }
            )
            continue
        left_tensor = left_value.detach().cpu().float().reshape(-1)
        right_tensor = right_value.detach().cpu().float().reshape(-1)
        diff = right_tensor - left_tensor
        l2 = float(torch.linalg.vector_norm(diff).item())
        max_abs = float(torch.max(torch.abs(diff)).item()) if diff.numel() else 0.0
        total_sq += l2 * l2
        right_norm = float(torch.linalg.vector_norm(right_tensor).item())
        right_sq += right_norm * right_norm
        if max_abs <= 1e-12:
            identical += 1
        else:
            changed.append(
                {
                    "key": str(key),
                    "max_abs_delta": max_abs,
                    "l2_delta": l2,
                    "right_l2": right_norm,
                    "relative_l2_delta": l2 / right_norm if right_norm > 0 else None,
                    "is_policy_tensor": _is_policy_tensor(str(key)),
                }
            )
    changed.sort(key=lambda item: float(item["l2_delta"]), reverse=True)
    policy_changed = [item for item in changed if item["is_policy_tensor"]]
    total_l2 = total_sq ** 0.5
    right_l2 = right_sq ** 0.5
    return {
        "left": left_label,
        "right": right_label,
        "common_tensor_keys": len(common),
        "left_only_keys": sorted(str(key) for key in set(left) - set(right))[:40],
        "right_only_keys": sorted(str(key) for key in set(right) - set(left))[:40],
        "shape_mismatch_count": len(shape_mismatch),
        "shape_mismatch_sample": shape_mismatch[:20],
        "identical_tensor_count_at_1e_12": identical,
        "changed_tensor_count": len(changed),
        "total_l2_delta": total_l2,
        "right_l2": right_l2,
        "relative_total_l2_delta": total_l2 / right_l2 if right_l2 > 0 else None,
        "top_changed_tensors": changed[:20],
        "policy_changed_tensors": policy_changed,
    }


def _policy_head_eval(model: Any, observation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    encoded = [row["encoded_observation"] for row in observation_rows]
    obs_tensor = torch.as_tensor(encoded, dtype=torch.float32)
    with torch.no_grad():
        output = model.initial_inference(obs_tensor)
    logits_tensor = output.policy_logits.detach().cpu().float()
    rows = []
    argmax_counts = Counter()
    oracle_counts = Counter()
    margins = []
    spreads = []
    for index, sample in enumerate(observation_rows):
        logits = [float(value) for value in logits_tensor[index].reshape(-1).tolist()]
        order = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
        action = int(order[0])
        argmax_counts[action] += 1
        oracle_counts[int(sample["track_ball_oracle_action"])] += 1
        margin = float(logits[order[0]] - logits[order[1]])
        spread = float(max(logits) - min(logits))
        margins.append(margin)
        spreads.append(spread)
        rows.append(
            {
                "sample_index": sample["sample_index"],
                "argmax_action": action,
                "track_ball_oracle_action": int(sample["track_ball_oracle_action"]),
                "policy_logits": logits,
                "top1_top2_margin": margin,
                "spread": spread,
            }
        )
    return {
        "sample_count": len(observation_rows),
        "argmax_counts_up_stay_down": [int(argmax_counts[i]) for i in range(3)],
        "track_ball_oracle_counts_up_stay_down": [int(oracle_counts[i]) for i in range(3)],
        "mean_top1_top2_margin": sum(margins) / len(margins) if margins else None,
        "min_top1_top2_margin": min(margins) if margins else None,
        "max_top1_top2_margin": max(margins) if margins else None,
        "mean_spread": sum(spreads) / len(spreads) if spreads else None,
        "max_spread": max(spreads) if spreads else None,
        "rows_sample": rows[:16],
        "all_rows": rows,
    }


def _mcts_eval(policy: Any, observation_rows: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    import numpy as np
    import torch

    rows = []
    action_counts = Counter()
    for sample in observation_rows[:limit]:
        obs_tensor = torch.as_tensor([sample["encoded_observation"]], dtype=torch.float32)
        with torch.no_grad():
            output = policy.eval_mode.forward(
                obs_tensor,
                action_mask=np.ones((1, 3), dtype=np.float32),
                to_play=[-1],
                ready_env_id=np.asarray([0]),
            )
        plain = _to_plain(output)
        action = _extract_action(plain)
        action_counts[action] += 1
        rows.append(
            {
                "sample_index": sample["sample_index"],
                "action": action,
                "track_ball_oracle_action": int(sample["track_ball_oracle_action"]),
                "output": _compact_mcts_output(plain),
            }
        )
    return {
        "sample_count": len(rows),
        "action_counts_up_stay_down": [int(action_counts[i]) for i in range(3)],
        "rows": rows,
    }


def _extract_action(output: Any) -> int:
    if isinstance(output, dict):
        for key in (0, "0"):
            if key in output:
                return _extract_action(output[key])
        if "action" in output:
            return int(output["action"])
        for key in ("actions", "selected_action", "selected_actions"):
            if key in output:
                value = output[key]
                return int(value[0] if isinstance(value, list) else value)
    if isinstance(output, list) and output:
        return _extract_action(output[0])
    raise ValueError(f"could not extract action from eval output: {output!r}")


def _compact_mcts_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        value = output.get(0, output.get("0", output))
        if isinstance(value, dict):
            return {
                key: value.get(key)
                for key in (
                    "action",
                    "visit_count_distribution",
                    "policy_logits",
                    "predicted_value",
                    "searched_value",
                    "value",
                )
                if key in value
            }
    return {"raw": output}


def _compare_policy_outputs(evals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = list(evals)
    comparisons = []
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            left_rows = evals[left_label]["all_rows"]
            right_rows = evals[right_label]["all_rows"]
            changed_actions = 0
            max_abs_deltas = []
            for left_row, right_row in zip(left_rows, right_rows):
                if left_row["argmax_action"] != right_row["argmax_action"]:
                    changed_actions += 1
                deltas = [
                    abs(float(a) - float(b))
                    for a, b in zip(left_row["policy_logits"], right_row["policy_logits"])
                ]
                max_abs_deltas.append(max(deltas))
            comparisons.append(
                {
                    "left": left_label,
                    "right": right_label,
                    "sample_count": len(left_rows),
                    "changed_argmax_count": changed_actions,
                    "mean_max_abs_logit_delta": (
                        sum(max_abs_deltas) / len(max_abs_deltas) if max_abs_deltas else None
                    ),
                    "max_abs_logit_delta": max(max_abs_deltas) if max_abs_deltas else None,
                }
            )
    return {"pairwise": comparisons}


def _compare_model_target_within_checkpoint(
    states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons = []
    by_checkpoint: dict[str, dict[str, dict[str, Any]]] = {}
    for label, state in states.items():
        if ":" not in label:
            continue
        checkpoint_label, variant = label.rsplit(":", 1)
        by_checkpoint.setdefault(checkpoint_label, {})[variant] = state
    for checkpoint_label, variants in sorted(by_checkpoint.items()):
        if "model" not in variants or "target_model" not in variants:
            continue
        comparisons.append(
            _compare_state_dicts(
                variants["model"],
                variants["target_model"],
                left_label=f"{checkpoint_label}:model",
                right_label=f"{checkpoint_label}:target_model",
            )
        )
    return comparisons


def _read_json_ref(ref: str) -> dict[str, Any] | None:
    path = runs.volume_path(RUNS_MOUNT, ref)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _train_artifact_summary(run_id: str, attempt_id: str) -> dict[str, Any]:
    train_root = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    refs = {
        "summary": train_root / "summary.json",
        "signals": train_root / "lightzero_training_signals.json",
        "artifacts": train_root / "lightzero_artifacts_manifest.json",
    }
    result: dict[str, Any] = {}
    for name, ref in refs.items():
        path = runs.volume_path(RUNS_MOUNT, ref)
        if not path.is_file():
            result[name] = {"exists": False, "ref": ref.as_posix()}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[name] = {
            "exists": True,
            "ref": ref.as_posix(),
            "file": runs.file_summary(path, mount=RUNS_MOUNT),
            "compact": _compact_train_payload(name, payload),
        }
    return result


def _compact_train_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name == "summary":
        episode_summary = payload.get("episode_summary", {})
        return {
            "ok": payload.get("ok"),
            "config": payload.get("config"),
            "checkpoint_mirror": payload.get("checkpoint_mirror"),
            "episode_summary": episode_summary,
            "lightzero_result_type": type(payload.get("lightzero_result")).__name__,
        }
    if name == "signals":
        return payload
    if name == "artifacts":
        return {
            "keys": list(payload.keys()),
            "artifact_summary": payload.get("artifact_summary", payload),
        }
    return payload


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=20 * 60)
def run_lightzero_checkpoint_diagnostics(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    checkpoint_refs: str = DEFAULT_CHECKPOINT_REFS,
    state_variants: str = DEFAULT_STATE_VARIANTS,
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    opponent_policy: str = "random_uniform",
    seed: int = 0,
    max_env_step: int = 512,
    num_simulations: int = 8,
    observation_limit: int = 48,
    mcts_observation_limit: int = 12,
    output_ref: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    output_ref = output_ref or (
        runs.attempt_root_ref(TASK_ID, run_id, attempt_id)
        / "probe"
        / f"lightzero_checkpoint_diagnostics_{runs.utc_stamp()}.json"
    ).as_posix()
    try:
        surface = _model_surface(
            env=env,
            feature_mode=feature_mode,
            opponent_policy=opponent_policy,
            seed=seed,
            max_env_step=max_env_step,
            num_simulations=num_simulations,
        )
        observation_rows = _sample_eval_observations(
            seed=seed,
            max_env_step=max_env_step,
            limit=observation_limit,
        )
        checkpoints: dict[str, dict[str, Any]] = {}
        checkpoint_files: dict[str, dict[str, Any]] = {}
        states: dict[str, dict[str, Any]] = {}
        policy_head_evals: dict[str, dict[str, Any]] = {}
        mcts_evals: dict[str, dict[str, Any]] = {}

        for label, ref in _parse_checkpoint_refs(checkpoint_refs):
            path = runs.volume_path(RUNS_MOUNT, ref)
            payload = _torch_load(path)
            checkpoint_files[label] = {
                "path": str(path),
                "ref": runs.file_ref(path, mount=RUNS_MOUNT),
                "exists": path.is_file(),
                "bytes": path.stat().st_size,
                "sha256": runs.sha256_file(path),
                "payload_keys": list(payload.keys()) if isinstance(payload, dict) else None,
                "last_iter": _to_plain(payload.get("last_iter")) if isinstance(payload, dict) else None,
                "last_step": _to_plain(payload.get("last_step")) if isinstance(payload, dict) else None,
                "state_variants_requested": state_variants,
            }
            for variant_name, state_path, state_dict in _state_dict_variants(payload, state_variants):
                variant_label = f"{label}:{variant_name}"
                states[variant_label] = state_dict
                model, load_state_dict = _load_model(surface["model_cfg"], state_dict)
                checkpoints[variant_label] = {
                    **_checkpoint_summary(path, payload, state_path, state_dict),
                    "checkpoint_label": label,
                    "state_variant": variant_name,
                    "load_state_dict": load_state_dict,
                }
                policy_head_evals[variant_label] = _policy_head_eval(model, observation_rows)
                mcts_policy = _load_mcts_policy(
                    main_config=surface["main_config"],
                    create_config=surface["create_config"],
                    state_dict=state_dict,
                    seed=seed,
                )
                mcts_evals[variant_label] = _mcts_eval(
                    mcts_policy,
                    observation_rows,
                    limit=mcts_observation_limit,
                )

        labels = list(states)
        tensor_deltas = []
        for left_index, left_label in enumerate(labels):
            for right_label in labels[left_index + 1 :]:
                tensor_deltas.append(
                    _compare_state_dicts(
                        states[left_label],
                        states[right_label],
                        left_label=left_label,
                        right_label=right_label,
                    )
                )

        result = {
            "schema": "curvyzero_lightzero_pong_checkpoint_diagnostics/v0",
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
                "state_variants": state_variants,
                "env": env,
                "feature_mode": feature_mode,
                "opponent_policy_for_config": opponent_policy,
                "seed": seed,
                "max_env_step": max_env_step,
                "num_simulations": num_simulations,
                "observation_limit": observation_limit,
                "mcts_observation_limit": mcts_observation_limit,
                "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            },
            "model_surface": _to_plain(surface["patched_surface"]),
            "train_artifacts": _train_artifact_summary(run_id, attempt_id),
            "observation_samples": observation_rows,
            "checkpoint_files": checkpoint_files,
            "checkpoints": checkpoints,
            "tensor_deltas": tensor_deltas,
            "model_target_deltas": _compare_model_target_within_checkpoint(states),
            "policy_head_eval": policy_head_evals,
            "policy_head_pairwise": _compare_policy_outputs(policy_head_evals),
            "mcts_eval": mcts_evals,
            "interpretive_flags": _interpretive_flags(checkpoints, tensor_deltas, policy_head_evals, mcts_evals),
        }
    except Exception as exc:  # pragma: no cover - remote diagnostics.
        result = {
            "schema": "curvyzero_lightzero_pong_checkpoint_diagnostics/v0",
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "wrapper_error": _exception_result(exc),
        }

    output_path = runs.volume_path(RUNS_MOUNT, output_ref)
    safe_result = json.loads(json.dumps(_to_plain(result)))
    runs.write_json(output_path, safe_result)
    safe_result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    runs.write_json(output_path, safe_result)
    runs_volume.commit()
    print(json.dumps(safe_result, indent=2, sort_keys=True))
    return safe_result


def _interpretive_flags(
    checkpoints: dict[str, dict[str, Any]],
    tensor_deltas: list[dict[str, Any]],
    policy_head_evals: dict[str, dict[str, Any]],
    mcts_evals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    del checkpoints
    return {
        "all_policy_head_argmax_up": {
            label: eval_result["argmax_counts_up_stay_down"][0] == eval_result["sample_count"]
            for label, eval_result in policy_head_evals.items()
        },
        "all_mcts_actions_up_on_sample": {
            label: eval_result["action_counts_up_stay_down"][0] == eval_result["sample_count"]
            for label, eval_result in mcts_evals.items()
        },
        "policy_head_margins_are_small": {
            label: {
                "mean_top1_top2_margin": eval_result["mean_top1_top2_margin"],
                "max_top1_top2_margin": eval_result["max_top1_top2_margin"],
            }
            for label, eval_result in policy_head_evals.items()
        },
        "checkpoint_pairs_with_policy_tensor_change": [
            {
                "left": delta["left"],
                "right": delta["right"],
                "policy_changed_tensor_count": len(delta["policy_changed_tensors"]),
                "changed_argmax_expected": None,
            }
            for delta in tensor_deltas
        ],
    }


@app.local_entrypoint()
def main(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    checkpoint_refs: str = DEFAULT_CHECKPOINT_REFS,
    state_variants: str = DEFAULT_STATE_VARIANTS,
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    opponent_policy: str = "random_uniform",
    seed: int = 0,
    max_env_step: int = 512,
    num_simulations: int = 8,
    observation_limit: int = 48,
    mcts_observation_limit: int = 12,
    output_ref: str | None = None,
) -> None:
    result = run_lightzero_checkpoint_diagnostics.remote(
        run_id=run_id,
        attempt_id=attempt_id,
        checkpoint_refs=checkpoint_refs,
        state_variants=state_variants,
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        num_simulations=num_simulations,
        observation_limit=observation_limit,
        mcts_observation_limit=mcts_observation_limit,
        output_ref=output_ref,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
