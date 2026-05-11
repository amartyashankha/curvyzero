"""No-fallback eval smoke for the OpenDILab HF 96x96 Pong MuZero checkpoint.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_pretrained96_eval_smoke

This wrapper is the eval companion to
``lightzero_pong_pretrained96_checkpoint_probe`` and deliberately does not use
the current 64x64 stock checkpoint/eval defaults.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_atari_rom_image import build_lightzero_atari_rom_image
from curvyzero.infra.modal.lightzero_pong_checkpoint_probe import (
    _checkpoint_summary,
    _exception_result,
    _find_state_dict,
    _torch_load,
    _to_plain,
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
    DEFAULT_MAX_EVAL_STEPS,
    DEFAULT_RUN_STOCK_EVALUATOR,
    DEFAULT_STEP_DETAIL_LIMIT,
    _compare_manual_and_stock,
    _make_policy_and_env,
    _run_eval_episode,
    _run_stock_evaluator_probe,
)
from curvyzero.infra.modal.lightzero_pong_pretrained96_checkpoint_probe import (
    DEFAULT_ATTEMPT_ID,
    DEFAULT_CHECKPOINT_REF,
    DEFAULT_POLICY_CONFIG_REF,
    DEFAULT_RUN_ID,
    TASK_ID,
    _patched_pretrained96_atari_pong_configs,
    _validate_pretrained96_surface,
)
from curvyzero.infra.modal.lightzero_pong_tiny_train_smoke import (
    DEFAULT_GAME_SEGMENT_LENGTH,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_MAX_TRAIN_ITER,
    VOLUME_NAME,
    _version_or_missing,
)


APP_NAME = "curvyzero-lightzero-pong-pretrained96-eval-smoke"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_ALLOW_MODEL_FALLBACK = False

image = (
    build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _default_output_ref() -> str:
    return (
        Path("training")
        / TASK_ID
        / "pretrained"
        / "OpenDILabCommunity"
        / "PongNoFrameskip-v4-MuZero"
        / "eval"
        / f"lightzero_visual_pong_pretrained96_eval_{runs.utc_stamp()}.json"
    ).as_posix()


def _eval_pretrained96_checkpoint(
    *,
    checkpoint_path: Path,
    run_id: str,
    attempt_id: str,
    env_id: str,
    seed: int,
    policy_config_ref: str | None,
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
    allow_model_fallback: bool,
    run_stock_evaluator: bool,
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path)
    state_candidate = _find_state_dict(checkpoint)
    if state_candidate is None:
        raise ValueError("no tensor state dict found under common LightZero checkpoint keys")
    state_path, state_dict = state_candidate
    patched = _patched_pretrained96_atari_pong_configs(
        env_id=env_id,
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        policy_config_ref=policy_config_ref,
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
    surface_problems = _validate_pretrained96_surface(
        patched["patched_surface"],
        use_cuda=False,
    )
    if surface_problems:
        raise ValueError(f"pretrained96 config surface mismatch: {surface_problems}")
    policy, env, surface = _make_policy_and_env(
        main_config=patched["main_config"],
        create_config=patched["create_config"],
        state_dict=state_dict,
        seed=seed,
        use_cuda=False,
    )
    episode = _run_eval_episode(
        policy=policy,
        env=env,
        seed=seed,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        allow_model_fallback=allow_model_fallback,
    )
    stock_evaluator = None
    if run_stock_evaluator:
        stock_evaluator = _run_stock_evaluator_probe(
            main_config=patched["main_config"],
            create_config=patched["create_config"],
            state_dict=state_dict,
            seed=seed,
            max_eval_steps=max_eval_steps,
        )
    return {
        "schema": "curvyzero_lightzero_visual_pong_pretrained96_eval_smoke/v0",
        "checkpoint": _checkpoint_summary(checkpoint_path),
        "load": {
            "ok": True,
            "state_dict_path": state_path,
            "tensor_count": len(state_dict),
            "keys_sample": list(state_dict)[:40],
        },
        "pretrained_example": {
            "source": "OpenDILabCommunity/PongNoFrameskip-v4-MuZero",
            "checkpoint_config_surface": (
                patched["captured"].get("policy_config_ref")
                or "policy_config.py model-card-compatible recreation"
            ),
            "module": patched["module"],
            "original_surface": patched["original_surface"],
            "patched_surface": patched["patched_surface"],
            "patches": patched["patches"],
        },
        "surface": surface,
        "episode": episode,
        "stock_evaluator": stock_evaluator,
        "manual_vs_stock": _compare_manual_and_stock(episode, stock_evaluator),
    }


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60, cpu=1.0)
def lightzero_pong_pretrained96_eval_smoke(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    policy_config_ref: str | None = DEFAULT_POLICY_CONFIG_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
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
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    allow_model_fallback: bool = DEFAULT_ALLOW_MODEL_FALLBACK,
    run_stock_evaluator: bool = DEFAULT_RUN_STOCK_EVALUATOR,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    output_ref = output_ref or _default_output_ref()
    output_path = runs.volume_path(RUNS_MOUNT, output_ref)
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
    config = {
        "job_kind": "lightzero_official_visual_pong_pretrained96_eval_smoke",
        "checkpoint_ref": checkpoint_ref,
        "policy_config_ref": policy_config_ref,
        "output_ref": output_ref,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "env_id": env_id,
        "seed": seed,
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
        "allow_model_fallback": allow_model_fallback,
        "run_stock_evaluator": run_stock_evaluator,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        eval_result = _eval_pretrained96_checkpoint(
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            attempt_id=attempt_id,
            env_id=env_id,
            seed=seed,
            policy_config_ref=policy_config_ref,
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
            allow_model_fallback=allow_model_fallback,
            run_stock_evaluator=run_stock_evaluator,
        )
        episode = eval_result["episode"]
        result: dict[str, Any] = {
            **eval_result,
            "ok": bool(
                episode["ok"]
                and episode["policy_could_act_in_real_env"]
                and episode["fallback_step_count"] == 0
            ),
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "status": {
                "checkpoint_load_ok": True,
                "strict_policy_model_load_ok": True,
                "env_reset_ok": bool(episode["steps"]),
                "policy_could_act_in_real_env": bool(episode["policy_could_act_in_real_env"]),
                "model_fallback_used": bool(episode["fallback_step_count"] > 0),
                "steps_run": int(episode["steps_run"]),
            },
        }
    except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
        result = {
            "schema": "curvyzero_lightzero_visual_pong_pretrained96_eval_smoke/v0",
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "checkpoint": _checkpoint_summary(checkpoint_path),
            "status": {
                "checkpoint_load_ok": False,
                "strict_policy_model_load_ok": False,
                "env_reset_ok": False,
                "policy_could_act_in_real_env": False,
                "model_fallback_used": False,
                "steps_run": 0,
            },
            "wrapper_error": _exception_result(exc),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(output_path, _to_plain(result))
    result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.local_entrypoint()
def main(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    policy_config_ref: str | None = DEFAULT_POLICY_CONFIG_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
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
    step_detail_limit: int | None = DEFAULT_STEP_DETAIL_LIMIT,
    allow_model_fallback: bool = DEFAULT_ALLOW_MODEL_FALLBACK,
    run_stock_evaluator: bool = DEFAULT_RUN_STOCK_EVALUATOR,
) -> None:
    result = lightzero_pong_pretrained96_eval_smoke.remote(
        checkpoint_ref=checkpoint_ref,
        policy_config_ref=policy_config_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=seed,
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
        allow_model_fallback=allow_model_fallback,
        run_stock_evaluator=run_stock_evaluator,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
