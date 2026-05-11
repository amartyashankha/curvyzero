"""Compare two stock LightZero Atari Pong checkpoints on the Modal Volume.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_checkpoint_diff

This is a metadata/state probe only. It does not construct an env, run MCTS, or
score gameplay.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_atari_rom_image import build_lightzero_atari_rom_image
from curvyzero.infra.modal.lightzero_pong_checkpoint_probe import (
    _checkpoint_summary,
    _find_state_dict,
    _summarize_value,
    _to_plain,
    _torch_load,
)
from curvyzero.infra.modal.lightzero_pong_dry_config_smoke import LIGHTZERO_VERSION
from curvyzero.infra.modal.lightzero_pong_tiny_train_smoke import TASK_ID, VOLUME_NAME


APP_NAME = "curvyzero-lightzero-pong-checkpoint-diff"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_RUN_ID = "lz-visual-pong-8192-sim25-s0"
DEFAULT_ATTEMPT_ID = "train-8192-sim25-b64-env4-auto"
DEFAULT_LEFT_REF = (
    "training/lightzero-official-visual-pong/"
    "lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/iteration_932.pth.tar"
)
DEFAULT_RIGHT_REF = (
    "training/lightzero-official-visual-pong/"
    "lz-visual-pong-8192-sim25-s0/checkpoints/lightzero/ckpt_best.pth.tar"
)
DEFAULT_LEFT_LABEL = "iteration_932"
DEFAULT_RIGHT_LABEL = "ckpt_best"

image = (
    build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _default_output_ref(*, run_id: str, attempt_id: str, left_label: str, right_label: str) -> str:
    return (
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, "checkpoint_state_diff")
        / f"{left_label}_vs_{right_label}_{runs.utc_stamp()}.json"
    ).as_posix()


def _tensor_float_stats(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach().cpu().float().reshape(-1)
    stats = {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }
    if int(tensor.numel()) > 0:
        stats.update(
            {
                "mean": float(torch.mean(tensor).item()),
                "std": float(torch.std(tensor).item()) if int(tensor.numel()) > 1 else 0.0,
                "min": float(torch.min(tensor).item()),
                "max": float(torch.max(tensor).item()),
            }
        )
    return stats


def _state_summary(payload: Any, path: Path) -> dict[str, Any]:
    state_candidate = _find_state_dict(payload)
    if state_candidate is None:
        return {
            "checkpoint": _checkpoint_summary(path),
            "state_dict": {"ok": False, "error": "no tensor state dict found"},
            "top_level": _summarize_value(payload),
        }
    state_path, state_dict = state_candidate
    keys = [str(key) for key in state_dict]
    prefix_counts = dict(sorted(Counter(key.split(".", 1)[0] for key in keys).items()))
    tensor_numel = sum(int(value.numel()) for value in state_dict.values() if hasattr(value, "numel"))
    sample_stats = {
        str(key): _tensor_float_stats(value)
        for key, value in list(state_dict.items())[:12]
        if hasattr(value, "detach")
    }
    top_level_keys = sorted(str(key) for key in payload.keys()) if isinstance(payload, dict) else []
    top_level_metadata = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"model", "target_model", "optimizer"}:
                top_level_metadata[str(key)] = _summarize_value(value)
            elif not hasattr(value, "detach"):
                top_level_metadata[str(key)] = _to_plain(value)
    return {
        "checkpoint": _checkpoint_summary(path),
        "top_level_keys": top_level_keys,
        "top_level_metadata": top_level_metadata,
        "state_dict": {
            "ok": True,
            "path": state_path,
            "tensor_count": len(state_dict),
            "tensor_numel": tensor_numel,
            "prefix_counts": prefix_counts,
            "keys_sample": keys[:40],
            "sample_tensor_stats": sample_stats,
        },
    }


def _compare_states(left_payload: Any, right_payload: Any) -> dict[str, Any]:
    import torch

    left_candidate = _find_state_dict(left_payload)
    right_candidate = _find_state_dict(right_payload)
    if left_candidate is None or right_candidate is None:
        return {
            "ok": False,
            "left_state_found": left_candidate is not None,
            "right_state_found": right_candidate is not None,
        }
    left_path, left_state = left_candidate
    right_path, right_state = right_candidate
    left_keys = set(str(key) for key in left_state)
    right_keys = set(str(key) for key in right_state)
    common_keys = sorted(left_keys & right_keys)
    left_only = sorted(left_keys - right_keys)
    right_only = sorted(right_keys - left_keys)
    shape_mismatches = []
    diffs = []
    total_numel = 0
    weighted_abs_sum = 0.0
    max_abs_diff = 0.0
    for key in common_keys:
        left_value = left_state[key]
        right_value = right_state[key]
        if not (hasattr(left_value, "detach") and hasattr(right_value, "detach")):
            continue
        left_shape = tuple(int(item) for item in left_value.shape)
        right_shape = tuple(int(item) for item in right_value.shape)
        if left_shape != right_shape:
            shape_mismatches.append(
                {"key": key, "left_shape": list(left_shape), "right_shape": list(right_shape)}
            )
            continue
        left_tensor = left_value.detach().cpu().float()
        right_tensor = right_value.detach().cpu().float()
        abs_diff = torch.abs(left_tensor - right_tensor)
        numel = int(abs_diff.numel())
        mean_abs = float(torch.mean(abs_diff).item()) if numel else 0.0
        key_max = float(torch.max(abs_diff).item()) if numel else 0.0
        total_numel += numel
        weighted_abs_sum += mean_abs * numel
        max_abs_diff = max(max_abs_diff, key_max)
        diffs.append(
            {
                "key": key,
                "numel": numel,
                "mean_abs_diff": mean_abs,
                "max_abs_diff": key_max,
            }
        )
    largest_mean_abs = sorted(diffs, key=lambda item: item["mean_abs_diff"], reverse=True)[:20]
    largest_max_abs = sorted(diffs, key=lambda item: item["max_abs_diff"], reverse=True)[:20]
    return {
        "ok": True,
        "left_state_path": left_path,
        "right_state_path": right_path,
        "left_key_count": len(left_keys),
        "right_key_count": len(right_keys),
        "common_key_count": len(common_keys),
        "left_only_count": len(left_only),
        "right_only_count": len(right_only),
        "left_only_sample": left_only[:40],
        "right_only_sample": right_only[:40],
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatches_sample": shape_mismatches[:40],
        "comparable_tensor_count": len(diffs),
        "total_comparable_numel": total_numel,
        "global_mean_abs_diff": weighted_abs_sum / total_numel if total_numel else None,
        "global_max_abs_diff": max_abs_diff,
        "largest_mean_abs_diff": largest_mean_abs,
        "largest_max_abs_diff": largest_max_abs,
    }


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60, cpu=1.0)
def lightzero_pong_checkpoint_diff(
    left_ref: str = DEFAULT_LEFT_REF,
    right_ref: str = DEFAULT_RIGHT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    left_label: str = DEFAULT_LEFT_LABEL,
    right_label: str = DEFAULT_RIGHT_LABEL,
) -> dict[str, Any]:
    started = time.perf_counter()
    left_path = runs.volume_path(RUNS_MOUNT, left_ref)
    right_path = runs.volume_path(RUNS_MOUNT, right_ref)
    output_ref = output_ref or _default_output_ref(
        run_id=run_id,
        attempt_id=attempt_id,
        left_label=left_label,
        right_label=right_label,
    )
    output_path = runs.volume_path(RUNS_MOUNT, output_ref)
    left_payload = _torch_load(left_path)
    right_payload = _torch_load(right_path)
    result = {
        "schema": "curvyzero_lightzero_visual_pong_checkpoint_diff/v0",
        "ok": True,
        "created_at": runs.utc_timestamp(),
        "job_kind": "lightzero_official_visual_pong_checkpoint_state_diff",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "left": {"label": left_label, "ref": left_ref, **_state_summary(left_payload, left_path)},
        "right": {"label": right_label, "ref": right_ref, **_state_summary(right_payload, right_path)},
        "comparison": _compare_states(left_payload, right_payload),
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(output_path, _to_plain(result))
    result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.local_entrypoint()
def main(
    left_ref: str = DEFAULT_LEFT_REF,
    right_ref: str = DEFAULT_RIGHT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    left_label: str = DEFAULT_LEFT_LABEL,
    right_label: str = DEFAULT_RIGHT_LABEL,
) -> None:
    result = lightzero_pong_checkpoint_diff.remote(
        left_ref=left_ref,
        right_ref=right_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        left_label=left_label,
        right_label=right_label,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
