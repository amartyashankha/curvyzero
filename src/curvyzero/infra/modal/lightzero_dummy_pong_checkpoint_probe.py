"""Modal wrapper for probing a mirrored LightZero dummy Pong checkpoint.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_checkpoint_probe
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke import (
    DEFAULT_ENV,
    DEFAULT_FEATURE_MODE,
    DEFAULT_MAX_ENV_STEP,
    DEFAULT_OPPONENT_POLICY,
    DEFAULT_SEED,
    LIGHTZERO_VERSION,
)
from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import (
    DEFAULT_TABULAR_OBSERVATION,
    probe_lightzero_dummy_pong_checkpoint,
)


APP_NAME = "curvyzero-lightzero-dummy-pong-checkpoint-probe"
TASK_ID = "lightzero-dummy-pong"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_RUN_ID = "lz-dpong-20260509T141607Z-3696aa333028"
DEFAULT_ATTEMPT_ID = "attempt-20260509T141607Z-98662e4917b4"
DEFAULT_CHECKPOINT_REF = (
    "training/lightzero-dummy-pong/"
    "lz-dpong-20260509T141607Z-3696aa333028/"
    "checkpoints/lightzero/ckpt_best.pth.tar"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
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


def _parse_observation_csv(text: str | None) -> list[float]:
    if text is None or not text.strip():
        return list(DEFAULT_TABULAR_OBSERVATION)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _default_output_ref(*, run_id: str, attempt_id: str) -> str:
    return (
        runs.attempt_root_ref(TASK_ID, run_id, attempt_id)
        / "probe"
        / f"lightzero_checkpoint_probe_{runs.utc_stamp()}.json"
    ).as_posix()


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _direct_policy_head_possible(probe: dict[str, Any]) -> bool:
    action_probe = probe.get("action_probe", {})
    if not action_probe.get("ok"):
        return False
    load_state_dict = action_probe.get("load_state_dict", {})
    if load_state_dict.get("strict"):
        return True
    missing = [str(key) for key in load_state_dict.get("missing_keys", [])]
    unexpected = [str(key) for key in load_state_dict.get("unexpected_keys", [])]
    return bool(missing or unexpected) and all(
        key.startswith("dynamics_network.") for key in [*missing, *unexpected]
    )


def _probe_status(probe: dict[str, Any]) -> dict[str, Any]:
    action_probe = probe.get("action_probe", {})
    load_state_dict = action_probe.get("load_state_dict", {})
    return {
        "load_ok": bool(probe.get("load", {}).get("ok")),
        "state_dict_ok": bool(probe.get("state_dict", {}).get("ok")),
        "action_probe_ok": bool(action_probe.get("ok")),
        "strict_full_model_load_ok": bool(
            load_state_dict.get("ok") and load_state_dict.get("strict")
        ),
        "direct_policy_head_possible": _direct_policy_head_possible(probe),
        "direct_policy_head_note": (
            "Direct initial_inference policy-head eval is technically possible "
            "when mismatches are dynamics_network-only; full MuZero/MCTS eval still "
            "needs the dynamics key/config mismatch resolved."
        ),
    }


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=10 * 60)
def lightzero_dummy_pong_checkpoint_probe(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    opponent_policy: str = DEFAULT_OPPONENT_POLICY,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    observation: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    output_ref = output_ref or _default_output_ref(run_id=run_id, attempt_id=attempt_id)
    output_path = runs.volume_path(RUNS_MOUNT, output_ref)
    started_at = runs.utc_timestamp()
    config = {
        "job_kind": "lightzero_dummy_pong_checkpoint_probe",
        "checkpoint_ref": checkpoint_ref,
        "output_ref": output_ref,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "env": env,
        "feature_mode": feature_mode,
        "opponent_policy": opponent_policy,
        "seed": seed,
        "max_env_step": max_env_step,
        "observation": observation,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }

    try:
        probe = probe_lightzero_dummy_pong_checkpoint(
            checkpoint_path=checkpoint_path,
            feature_mode=feature_mode,
            env=env,
            opponent_policy=opponent_policy,
            seed=seed,
            max_env_step=max_env_step,
            observation=_parse_observation_csv(observation),
        )
        status = _probe_status(probe)
        ok = bool(status["load_ok"] and status["state_dict_ok"] and status["action_probe_ok"])
        result: dict[str, Any] = {
            "schema": "curvyzero_lightzero_dummy_pong_modal_checkpoint_probe/v0",
            "ok": ok,
            "status": status,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "config": config,
            "probe": probe,
        }
    except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
        result = {
            "schema": "curvyzero_lightzero_dummy_pong_modal_checkpoint_probe/v0",
            "ok": False,
            "status": {
                "load_ok": False,
                "state_dict_ok": False,
                "action_probe_ok": False,
                "strict_full_model_load_ok": False,
                "direct_policy_head_possible": False,
            },
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "config": config,
            "probe": None,
            "wrapper_error": _exception_result(exc),
        }

    runs.write_json(output_path, _to_plain(result))
    artifact = runs.file_summary(output_path, mount=RUNS_MOUNT)
    result["artifact"] = artifact
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.local_entrypoint()
def main(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    opponent_policy: str = DEFAULT_OPPONENT_POLICY,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    observation: str | None = None,
) -> None:
    result = lightzero_dummy_pong_checkpoint_probe.remote(
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        observation=observation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
