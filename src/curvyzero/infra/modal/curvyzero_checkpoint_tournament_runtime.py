"""Modal runtime objects for the CurvyTron checkpoint tournament app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from curvyzero.contracts.curvytron import modal_volume_kwargs_for_name
from curvyzero.infra.modal.curvyzero_checkpoint_tournament_settings import (
    APP_NAME,
    CHECKPOINT_INTAKE_DICT_NAME,
    CHECKPOINT_INTAKE_QUEUE_NAME,
    CHECKPOINT_VOLUME_NAME,
    CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH,
    LIGHTZERO_VERSION,
    OPPONENT_LEADERBOARD_DICT_NAME,
    REMOTE_ROOT,
    RUNS_MOUNT,
    TOURNAMENT_MOUNT,
    TOURNAMENT_VOLUME_NAME,
)


LOW_LEVEL_WORKER_RETRIES = modal.Retries(
    max_retries=2,
    initial_delay=5.0,
    backoff_coefficient=2.0,
    max_delay=60.0,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"LightZero=={LIGHTZERO_VERSION}",
        "numpy>=1.26",
        "cloudpickle>=3",
        "pillow>=10",
        "fastapi>=0.110",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_file(
        Path.cwd() / CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH,
        remote_path=str(REMOTE_ROOT / CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH),
        copy=True,
    )
)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME,
    **modal_volume_kwargs_for_name(CHECKPOINT_VOLUME_NAME),
).read_only()
tournament_volume = modal.Volume.from_name(
    TOURNAMENT_VOLUME_NAME,
    **modal_volume_kwargs_for_name(TOURNAMENT_VOLUME_NAME),
)
checkpoint_intake_state = modal.Dict.from_name(
    CHECKPOINT_INTAKE_DICT_NAME,
    create_if_missing=True,
)
checkpoint_intake_queue = modal.Queue.from_name(
    CHECKPOINT_INTAKE_QUEUE_NAME,
    create_if_missing=True,
)
opponent_leaderboard_state = modal.Dict.from_name(
    OPPONENT_LEADERBOARD_DICT_NAME,
    create_if_missing=True,
)
app = modal.App(APP_NAME)


def checkpoint_volumes() -> dict[str, Any]:
    return {RUNS_MOUNT.as_posix(): checkpoint_volume}


def tournament_volumes() -> dict[str, Any]:
    return {TOURNAMENT_MOUNT.as_posix(): tournament_volume}


def game_volumes() -> dict[str, Any]:
    return {**checkpoint_volumes(), **tournament_volumes()}
