#!/usr/bin/env python3
"""Build a dry-run manifest for CurvyTron stock LightZero train launches.

This script never calls Modal. It emits explicit run ids, attempt ids, artifact
refs, and launch commands for the canonical CurvyTron Modal trainer so a matrix
can be reviewed before anyone launches it.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
TASK_ID = "lightzero-curvytron-visual-survival"
SCHEMA_ID = "curvyzero_curvytron_stock_train_dry_run_manifest/v0"

MODE_TRAIN = "train"
ENV_SOURCE_STATE_FIXED_OPPONENT = "source_state_fixed_opponent"
ENV_SOURCE_STATE_JOINT_ACTION = "source_state_joint_action"
OPPONENT_FIXED_STRAIGHT = "fixed_straight"
OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT = "frozen_lightzero_checkpoint"
OPPONENT_NONE_CENTRALIZED = "none_centralized_joint_action"
REWARD_SPARSE_OUTCOME = "sparse_outcome"
REWARD_DENSE_SURVIVAL_PLUS_OUTCOME = "dense_survival_plus_outcome"
REWARD_ALL_PLAYERS_ALIVE_DIAGNOSTIC = "all_players_alive_diagnostic"
SOURCE_STATE_RENDER_BROWSER_LINES = "browser_lines"
SOURCE_STATE_RENDER_BODY_CIRCLES_FAST = "body_circles_fast"

DEFAULT_RECENT_OPPONENT_CHECKPOINT_REF = (
    "training/lightzero-curvytron-visual-survival/"
    "curvytron-dense-ckpt1-iter10000-sanity-20260512a/"
    "checkpoints/lightzero/iteration_32.pth.tar"
)
DEFAULT_MID_OPPONENT_CHECKPOINT_REF = (
    "training/lightzero-curvytron-visual-survival/"
    "curvytron-dense-ckpt1-iter10000-sanity-20260512a/"
    "checkpoints/lightzero/iteration_16.pth.tar"
)
DEFAULT_OLD_OPPONENT_CHECKPOINT_REF = (
    "training/lightzero-curvytron-visual-survival/"
    "curvytron-dense-ckpt1-iter10000-sanity-20260512a/"
    "checkpoints/lightzero/iteration_0.pth.tar"
)

DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_stock_train_manifests")


@dataclass(frozen=True)
class Row:
    row_id: str
    label: str
    env_variant: str
    reward_variant: str
    opponent_policy_kind: str
    source_state_trail_render_mode: str
    compute: str
    seed: int
    max_train_iter: int
    save_ckpt_after_iter: int
    max_env_step: int
    collector_env_num: int
    evaluator_env_num: int
    n_evaluator_episode: int
    n_episode: int
    source_max_steps: int
    batch_size: int
    num_simulations: int
    opponent_checkpoint_ref: str | None = None
    opponent_snapshot_ref: str | None = None
    opponent_checkpoint_state_key: str | None = None
    background_eval_enabled: bool = True
    background_gif_enabled: bool = True
    background_eval_launch_kind: str = "poller"
    background_eval_seed_count: int = 8
    background_eval_max_steps: int = 4096
    background_eval_num_simulations: int = 8
    background_eval_batch_size: int = 64
    background_gif_max_steps: int = 2048
    background_gif_frame_stride: int = 4
    lightzero_eval_freq: int = 0
    env_manager_type: str = "base"
    ego_action_straight_override_probability: float = 0.0
    control_noise_profile_id: str = "none"


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_id(raw: str, *, label: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    if (
        not raw
        or len(raw) > 96
        or raw in {".", ".."}
        or not raw[0].isalnum()
        or any(char not in allowed for char in raw)
    ):
        raise ValueError(
            f"{label} must be 1-96 chars of letters, numbers, dash, underscore, or dot"
        )
    return raw


def _ref(*parts: str) -> str:
    return "/".join(parts)


def _base_rows(
    *,
    compute: str,
    max_train_iter: int,
    save_ckpt_after_iter: int,
    source_state_trail_render_mode: str,
) -> list[Row]:
    common: dict[str, Any] = {
        "compute": compute,
        "max_train_iter": max_train_iter,
        "save_ckpt_after_iter": save_ckpt_after_iter,
        "max_env_step": 262144,
        "collector_env_num": 1,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": 1,
        "source_max_steps": 256,
        "batch_size": 32,
        "num_simulations": 8,
        "source_state_trail_render_mode": source_state_trail_render_mode,
    }
    return [
        Row(
            row_id="01",
            label="fixed-sparse-s0",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_SPARSE_OUTCOME,
            opponent_policy_kind=OPPONENT_FIXED_STRAIGHT,
            seed=3101,
            **common,
        ),
        Row(
            row_id="02",
            label="fixed-dense-s0",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            opponent_policy_kind=OPPONENT_FIXED_STRAIGHT,
            seed=3102,
            **common,
        ),
        Row(
            row_id="03",
            label="joint-diagnostic-s0",
            env_variant=ENV_SOURCE_STATE_JOINT_ACTION,
            reward_variant=REWARD_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
            opponent_policy_kind=OPPONENT_NONE_CENTRALIZED,
            seed=3103,
            **common,
        ),
    ]


def _frozen_snapshot_ref(tag: str) -> str:
    return f"curvytron_stock_manifest_{tag}"


def _high_signal_rows(
    *,
    compute: str,
    max_train_iter: int,
    save_ckpt_after_iter: int,
    source_state_trail_render_mode: str,
    recent_opponent_checkpoint_ref: str,
    mid_opponent_checkpoint_ref: str,
    old_opponent_checkpoint_ref: str,
    include_joint_diagnostics: bool,
) -> list[Row]:
    common: dict[str, Any] = {
        "compute": compute,
        "max_train_iter": max_train_iter,
        "save_ckpt_after_iter": save_ckpt_after_iter,
        "max_env_step": 262144,
        "collector_env_num": 1,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": 1,
        "source_max_steps": 256,
        "background_eval_seed_count": 8,
        "background_eval_max_steps": 4096,
        "background_eval_num_simulations": 8,
        "background_eval_batch_size": 64,
        "background_gif_enabled": True,
        "background_gif_max_steps": 2048,
        "background_gif_frame_stride": 4,
        "source_state_trail_render_mode": source_state_trail_render_mode,
    }
    frozen_recent = {
        "opponent_policy_kind": OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT,
        "opponent_checkpoint_ref": recent_opponent_checkpoint_ref,
        "opponent_snapshot_ref": _frozen_snapshot_ref("recent_iteration_32"),
    }
    rows = [
        Row(
            row_id="01",
            label="fixed-straight-sparse-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_SPARSE_OUTCOME,
            opponent_policy_kind=OPPONENT_FIXED_STRAIGHT,
            batch_size=32,
            num_simulations=8,
            seed=410,
            **common,
        ),
        Row(
            row_id="02",
            label="fixed-straight-dense-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            opponent_policy_kind=OPPONENT_FIXED_STRAIGHT,
            batch_size=32,
            num_simulations=8,
            seed=410,
            **common,
        ),
        Row(
            row_id="03",
            label="frozen-recent-dense-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            batch_size=32,
            num_simulations=8,
            seed=411,
            **frozen_recent,
            **common,
        ),
        Row(
            row_id="04",
            label="frozen-recent-sparse-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_SPARSE_OUTCOME,
            batch_size=32,
            num_simulations=8,
            seed=411,
            **frozen_recent,
            **common,
        ),
        Row(
            row_id="05",
            label="frozen-mid-dense-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            opponent_policy_kind=OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT,
            opponent_checkpoint_ref=mid_opponent_checkpoint_ref,
            opponent_snapshot_ref=_frozen_snapshot_ref("mid_iteration_16"),
            batch_size=32,
            num_simulations=8,
            seed=412,
            **common,
        ),
        Row(
            row_id="06",
            label="frozen-old-dense-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            opponent_policy_kind=OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT,
            opponent_checkpoint_ref=old_opponent_checkpoint_ref,
            opponent_snapshot_ref=_frozen_snapshot_ref("old_iteration_0"),
            batch_size=32,
            num_simulations=8,
            seed=413,
            **common,
        ),
        Row(
            row_id="07",
            label="frozen-recent-dense-b32-sim16",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            batch_size=32,
            num_simulations=16,
            seed=414,
            **frozen_recent,
            **common,
        ),
        Row(
            row_id="08",
            label="frozen-recent-dense-b64-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            batch_size=64,
            num_simulations=8,
            seed=415,
            **frozen_recent,
            **common,
        ),
    ]
    if include_joint_diagnostics:
        rows.extend(
            [
                Row(
                    row_id="09",
                    label="joint-diagnostic-b32-sim8",
                    env_variant=ENV_SOURCE_STATE_JOINT_ACTION,
                    reward_variant=REWARD_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
                    opponent_policy_kind=OPPONENT_NONE_CENTRALIZED,
                    batch_size=32,
                    num_simulations=8,
                    seed=416,
                    **common,
                ),
                Row(
                    row_id="10",
                    label="joint-diagnostic-b32-sim16",
                    env_variant=ENV_SOURCE_STATE_JOINT_ACTION,
                    reward_variant=REWARD_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
                    opponent_policy_kind=OPPONENT_NONE_CENTRALIZED,
                    batch_size=32,
                    num_simulations=16,
                    seed=417,
                    **common,
                ),
            ]
        )
    return rows


def _long_stock_rows(
    *,
    compute: str,
    max_train_iter: int,
    save_ckpt_after_iter: int,
    source_state_trail_render_mode: str,
    recent_opponent_checkpoint_ref: str,
    old_opponent_checkpoint_ref: str,
) -> list[Row]:
    body_common: dict[str, Any] = {
        "compute": compute,
        "max_train_iter": max_train_iter,
        "save_ckpt_after_iter": save_ckpt_after_iter,
        "max_env_step": 2_000_000,
        "collector_env_num": 32,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": 32,
        "source_max_steps": 256,
        "batch_size": 32,
        "num_simulations": 8,
        "background_eval_seed_count": 8,
        "background_eval_max_steps": 4096,
        "background_eval_num_simulations": 8,
        "background_eval_batch_size": 64,
        "background_gif_enabled": True,
        "background_gif_max_steps": 2048,
        "background_gif_frame_stride": 4,
        "source_state_trail_render_mode": source_state_trail_render_mode,
        "env_manager_type": "subprocess",
    }
    browser_common = {
        **body_common,
        "collector_env_num": 16,
        "n_episode": 16,
        "source_state_trail_render_mode": SOURCE_STATE_RENDER_BROWSER_LINES,
    }
    frozen_recent = {
        "opponent_policy_kind": OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT,
        "opponent_checkpoint_ref": recent_opponent_checkpoint_ref,
        "opponent_snapshot_ref": _frozen_snapshot_ref("recent_iteration_32"),
    }
    return [
        Row(
            row_id="01",
            label="c32-fast-fixed-sparse-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_SPARSE_OUTCOME,
            opponent_policy_kind=OPPONENT_FIXED_STRAIGHT,
            seed=510,
            **body_common,
        ),
        Row(
            row_id="02",
            label="c32-fast-fixed-dense-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            opponent_policy_kind=OPPONENT_FIXED_STRAIGHT,
            seed=511,
            **body_common,
        ),
        Row(
            row_id="03",
            label="c32-fast-frozen-recent-dense-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            seed=512,
            **frozen_recent,
            **body_common,
        ),
        Row(
            row_id="04",
            label="c32-fast-frozen-recent-sparse-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_SPARSE_OUTCOME,
            seed=513,
            **frozen_recent,
            **body_common,
        ),
        Row(
            row_id="05",
            label="c32-fast-frozen-old-dense-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            opponent_policy_kind=OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT,
            opponent_checkpoint_ref=old_opponent_checkpoint_ref,
            opponent_snapshot_ref=_frozen_snapshot_ref("old_iteration_0"),
            seed=514,
            **body_common,
        ),
        Row(
            row_id="06",
            label="c32-fast-frozen-recent-dense-b32-sim16",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            seed=515,
            num_simulations=16,
            **frozen_recent,
            **{key: value for key, value in body_common.items() if key != "num_simulations"},
        ),
        Row(
            row_id="07",
            label="c16-browser-frozen-recent-dense-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
            seed=516,
            **frozen_recent,
            **browser_common,
        ),
        Row(
            row_id="08",
            label="c16-browser-fixed-sparse-b32-sim8",
            env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
            reward_variant=REWARD_SPARSE_OUTCOME,
            opponent_policy_kind=OPPONENT_FIXED_STRAIGHT,
            seed=517,
            **browser_common,
        ),
    ]


def _stock_tensor_rows(
    *,
    compute: str,
    max_train_iter: int,
    save_ckpt_after_iter: int,
    source_state_trail_render_mode: str,
    recent_opponent_checkpoint_ref: str,
    mid_opponent_checkpoint_ref: str,
    old_opponent_checkpoint_ref: str,
) -> list[Row]:
    common: dict[str, Any] = {
        "compute": compute,
        "max_train_iter": max_train_iter,
        "save_ckpt_after_iter": save_ckpt_after_iter,
        "max_env_step": 10_000_000,
        "collector_env_num": 32,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": 32,
        "source_max_steps": 256,
        "batch_size": 32,
        "num_simulations": 8,
        "background_eval_seed_count": 8,
        "background_eval_max_steps": 4096,
        "background_eval_num_simulations": 8,
        "background_eval_batch_size": 64,
        "background_gif_enabled": True,
        "background_gif_max_steps": 2048,
        "background_gif_frame_stride": 4,
        "source_state_trail_render_mode": source_state_trail_render_mode,
        "env_manager_type": "subprocess",
    }

    opponent_specs = {
        "fixed": {
            "opponent_policy_kind": OPPONENT_FIXED_STRAIGHT,
            "opponent_checkpoint_ref": None,
            "opponent_snapshot_ref": None,
        },
        "recent": {
            "opponent_policy_kind": OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT,
            "opponent_checkpoint_ref": recent_opponent_checkpoint_ref,
            "opponent_snapshot_ref": _frozen_snapshot_ref("recent_iteration_32"),
        },
        "mid": {
            "opponent_policy_kind": OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT,
            "opponent_checkpoint_ref": mid_opponent_checkpoint_ref,
            "opponent_snapshot_ref": _frozen_snapshot_ref("mid_iteration_16"),
        },
        "old": {
            "opponent_policy_kind": OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT,
            "opponent_checkpoint_ref": old_opponent_checkpoint_ref,
            "opponent_snapshot_ref": _frozen_snapshot_ref("old_iteration_0"),
        },
    }
    reward_specs = {
        "sparse": REWARD_SPARSE_OUTCOME,
        "dense": REWARD_DENSE_SURVIVAL_PLUS_OUTCOME,
    }

    rows: list[Row] = []

    def add(
        label: str,
        *,
        opponent: str,
        reward: str,
        seed: int,
        **overrides: Any,
    ) -> None:
        row_id = f"{len(rows) + 1:02d}"
        kwargs = {**common, **opponent_specs[opponent], **overrides}
        rows.append(
            Row(
                row_id=row_id,
                label=label,
                env_variant=ENV_SOURCE_STATE_FIXED_OPPONENT,
                reward_variant=reward_specs[reward],
                seed=seed,
                **kwargs,
            )
        )

    seed = 610
    for opponent in ("fixed", "recent", "mid", "old"):
        for reward in ("sparse", "dense"):
            add(
                f"c32-fast-{opponent}-{reward}-b32-sim8",
                opponent=opponent,
                reward=reward,
                seed=seed,
            )
            seed += 1

    for opponent in ("fixed", "recent", "mid", "old"):
        add(
            f"c32-fast-{opponent}-dense-b32-sim16",
            opponent=opponent,
            reward="dense",
            seed=seed,
            num_simulations=16,
        )
        seed += 1

    for opponent in ("fixed", "recent", "mid", "old"):
        add(
            f"c64-fast-{opponent}-dense-b32-sim8",
            opponent=opponent,
            reward="dense",
            seed=seed,
            collector_env_num=64,
            n_episode=64,
        )
        seed += 1

    for opponent in ("fixed", "recent", "mid", "old"):
        add(
            f"c32-fast-{opponent}-dense-b64-sim8",
            opponent=opponent,
            reward="dense",
            seed=seed,
            batch_size=64,
        )
        seed += 1

    for opponent, reward in (
        ("fixed", "sparse"),
        ("fixed", "dense"),
        ("recent", "sparse"),
        ("recent", "dense"),
        ("old", "dense"),
    ):
        add(
            f"c16-browser-{opponent}-{reward}-b32-sim8",
            opponent=opponent,
            reward=reward,
            seed=seed,
            collector_env_num=16,
            n_episode=16,
            source_state_trail_render_mode=SOURCE_STATE_RENDER_BROWSER_LINES,
        )
        seed += 1

    for opponent, reward in (
        ("fixed", "sparse"),
        ("fixed", "dense"),
        ("recent", "dense"),
        ("mid", "dense"),
        ("old", "dense"),
    ):
        add(
            f"c32-fast-{opponent}-{reward}-max1024-b32-sim8",
            opponent=opponent,
            reward=reward,
            seed=seed,
            source_max_steps=1024,
            background_eval_max_steps=8192,
            background_gif_max_steps=4096,
        )
        seed += 1

    for opponent, reward in (("fixed", "sparse"), ("recent", "dense")):
        add(
            f"c32-fast-{opponent}-{reward}-straight005-b32-sim8",
            opponent=opponent,
            reward=reward,
            seed=seed,
            ego_action_straight_override_probability=0.05,
            control_noise_profile_id="straight_override_0.05",
        )
        seed += 1

    return rows


def _rows_for_matrix(
    matrix_name: str,
    *,
    compute: str,
    max_train_iter: int,
    save_ckpt_after_iter: int,
    source_state_trail_render_mode: str,
    recent_opponent_checkpoint_ref: str,
    mid_opponent_checkpoint_ref: str,
    old_opponent_checkpoint_ref: str,
    include_joint_diagnostics: bool,
) -> list[Row]:
    if matrix_name == "stock-control-v1":
        return _base_rows(
            compute=compute,
            max_train_iter=max_train_iter,
            save_ckpt_after_iter=save_ckpt_after_iter,
            source_state_trail_render_mode=source_state_trail_render_mode,
        )
    if matrix_name in {"stock-high-signal-v1", "stock-high-signal-v1b"}:
        return _high_signal_rows(
            compute=compute,
            max_train_iter=max_train_iter,
            save_ckpt_after_iter=save_ckpt_after_iter,
            source_state_trail_render_mode=source_state_trail_render_mode,
            recent_opponent_checkpoint_ref=recent_opponent_checkpoint_ref,
            mid_opponent_checkpoint_ref=mid_opponent_checkpoint_ref,
            old_opponent_checkpoint_ref=old_opponent_checkpoint_ref,
            include_joint_diagnostics=include_joint_diagnostics,
        )
    if matrix_name == "stock-long-v1c":
        return _long_stock_rows(
            compute=compute,
            max_train_iter=max_train_iter,
            save_ckpt_after_iter=save_ckpt_after_iter,
            source_state_trail_render_mode=source_state_trail_render_mode,
            recent_opponent_checkpoint_ref=recent_opponent_checkpoint_ref,
            old_opponent_checkpoint_ref=old_opponent_checkpoint_ref,
        )
    if matrix_name == "stock-tensor-v1d":
        return _stock_tensor_rows(
            compute=compute,
            max_train_iter=max_train_iter,
            save_ckpt_after_iter=save_ckpt_after_iter,
            source_state_trail_render_mode=source_state_trail_render_mode,
            recent_opponent_checkpoint_ref=recent_opponent_checkpoint_ref,
            mid_opponent_checkpoint_ref=mid_opponent_checkpoint_ref,
            old_opponent_checkpoint_ref=old_opponent_checkpoint_ref,
        )
    if matrix_name == "stock-fixed-sparse-seeds-v1":
        rows = _base_rows(
            compute=compute,
            max_train_iter=max_train_iter,
            save_ckpt_after_iter=save_ckpt_after_iter,
            source_state_trail_render_mode=source_state_trail_render_mode,
        )
        base = rows[0]
        return [
            Row(
                row_id=f"{index:02d}",
                label=f"fixed-sparse-s{index - 1}",
                env_variant=base.env_variant,
                reward_variant=base.reward_variant,
                opponent_policy_kind=base.opponent_policy_kind,
                source_state_trail_render_mode=base.source_state_trail_render_mode,
                compute=base.compute,
                seed=3200 + index,
                max_train_iter=base.max_train_iter,
                save_ckpt_after_iter=base.save_ckpt_after_iter,
                max_env_step=base.max_env_step,
                collector_env_num=base.collector_env_num,
                evaluator_env_num=base.evaluator_env_num,
                n_evaluator_episode=base.n_evaluator_episode,
                n_episode=base.n_episode,
                source_max_steps=base.source_max_steps,
                batch_size=base.batch_size,
                num_simulations=base.num_simulations,
                background_eval_enabled=base.background_eval_enabled,
                background_gif_enabled=base.background_gif_enabled,
                background_eval_launch_kind=base.background_eval_launch_kind,
                background_eval_seed_count=base.background_eval_seed_count,
                background_eval_max_steps=base.background_eval_max_steps,
                background_eval_num_simulations=base.background_eval_num_simulations,
                background_eval_batch_size=base.background_eval_batch_size,
                background_gif_max_steps=base.background_gif_max_steps,
                background_gif_frame_stride=base.background_gif_frame_stride,
                lightzero_eval_freq=base.lightzero_eval_freq,
                env_manager_type=base.env_manager_type,
            )
            for index in range(1, 5)
        ]
    raise ValueError(
        "unknown matrix_name "
        f"{matrix_name!r}; expected 'stock-control-v1', 'stock-high-signal-v1', "
        "'stock-high-signal-v1b', 'stock-long-v1c', 'stock-tensor-v1d', or "
        "'stock-fixed-sparse-seeds-v1'"
    )


def _command_for_row(row: Row, *, run_id: str, attempt_id: str, detach: bool) -> list[str]:
    command = [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "--quiet",
    ]
    if detach:
        command.append("--detach")
    command.extend(
        [
            "-m",
            MODULE,
            "--mode",
            MODE_TRAIN,
            "--compute",
            row.compute,
            "--seed",
            str(row.seed),
            "--run-id",
            run_id,
            "--attempt-id",
            attempt_id,
            "--env-variant",
            row.env_variant,
            "--reward-variant",
            row.reward_variant,
            "--source-state-trail-render-mode",
            row.source_state_trail_render_mode,
            "--ego-action-straight-override-probability",
            str(row.ego_action_straight_override_probability),
            "--control-noise-profile-id",
            row.control_noise_profile_id,
            "--opponent-policy-kind",
            row.opponent_policy_kind,
            "--max-train-iter",
            str(row.max_train_iter),
            "--max-env-step",
            str(row.max_env_step),
            "--save-ckpt-after-iter",
            str(row.save_ckpt_after_iter),
            "--collector-env-num",
            str(row.collector_env_num),
            "--evaluator-env-num",
            str(row.evaluator_env_num),
            "--n-evaluator-episode",
            str(row.n_evaluator_episode),
            "--n-episode",
            str(row.n_episode),
            "--source-max-steps",
            str(row.source_max_steps),
            "--batch-size",
            str(row.batch_size),
            "--num-simulations",
            str(row.num_simulations),
            "--lightzero-eval-freq",
            str(row.lightzero_eval_freq),
            "--env-manager-type",
            row.env_manager_type,
            "--background-eval-launch-kind",
            row.background_eval_launch_kind,
            "--background-eval-seed-count",
            str(row.background_eval_seed_count),
            "--background-eval-max-steps",
            str(row.background_eval_max_steps),
            "--background-eval-num-simulations",
            str(row.background_eval_num_simulations),
            "--background-eval-batch-size",
            str(row.background_eval_batch_size),
            "--background-gif-max-steps",
            str(row.background_gif_max_steps),
            "--background-gif-frame-stride",
            str(row.background_gif_frame_stride),
            "--output-detail",
            "compact",
        ]
    )
    if row.opponent_checkpoint_ref is not None:
        command.extend(["--opponent-checkpoint-ref", row.opponent_checkpoint_ref])
    if row.opponent_snapshot_ref is not None:
        command.extend(["--snapshot-ref", row.opponent_snapshot_ref])
    if row.opponent_checkpoint_state_key is not None:
        command.extend(["--state-key", row.opponent_checkpoint_state_key])
    if not row.background_eval_enabled:
        command.append("--no-background-eval-enabled")
    if not row.background_gif_enabled:
        command.append("--no-background-gif-enabled")
    return command


def _manifest_row(
    row: Row,
    *,
    matrix_name: str,
    run_prefix: str,
    attempt_prefix: str,
    detach: bool,
) -> dict[str, Any]:
    run_id = _safe_id(
        f"{run_prefix}-{row.row_id}-{row.label}",
        label="run_id",
    )
    attempt_id = _safe_id(
        f"{attempt_prefix}-{row.row_id}-{row.label}",
        label="attempt_id",
    )
    command = _command_for_row(row, run_id=run_id, attempt_id=attempt_id, detach=detach)
    command_text = shlex.join(command)
    if "two-seat-selfplay" in command_text:
        raise ValueError(f"refusing stale two-seat-selfplay command for row {row.row_id}")
    if row.env_variant not in {
        ENV_SOURCE_STATE_FIXED_OPPONENT,
        ENV_SOURCE_STATE_JOINT_ACTION,
    }:
        raise ValueError(f"refusing non-stock env_variant {row.env_variant!r}")
    if row.opponent_policy_kind == OPPONENT_FROZEN_LIGHTZERO_CHECKPOINT:
        if not row.opponent_checkpoint_ref:
            raise ValueError(f"row {row.row_id} requires an immutable frozen checkpoint ref")
        if "latest" in row.opponent_checkpoint_ref or "ckpt_best" in row.opponent_checkpoint_ref:
            raise ValueError(
                f"row {row.row_id} refuses mutable frozen checkpoint ref "
                f"{row.opponent_checkpoint_ref!r}"
            )
    elif row.opponent_checkpoint_ref is not None:
        raise ValueError(
            f"row {row.row_id} has opponent_checkpoint_ref but opponent kind "
            f"{row.opponent_policy_kind!r}"
        )
    train_ref = _ref("training", TASK_ID, run_id, "attempts", attempt_id, "train")
    return {
        "row_id": row.row_id,
        "label": row.label,
        "matrix_name": matrix_name,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "mode": MODE_TRAIN,
        "canonical_launcher": MODULE,
        "calls_stock_train_muzero": True,
        "env_variant": row.env_variant,
        "reward_variant": row.reward_variant,
        "source_state_trail_render_mode": row.source_state_trail_render_mode,
        "opponent_policy_kind": row.opponent_policy_kind,
        "opponent_checkpoint_ref": row.opponent_checkpoint_ref,
        "opponent_snapshot_ref": row.opponent_snapshot_ref,
        "compute": row.compute,
        "seed": row.seed,
        "flags": {
            "max_env_step": row.max_env_step,
            "max_train_iter": row.max_train_iter,
            "save_ckpt_after_iter": row.save_ckpt_after_iter,
            "collector_env_num": row.collector_env_num,
            "evaluator_env_num": row.evaluator_env_num,
            "n_evaluator_episode": row.n_evaluator_episode,
            "n_episode": row.n_episode,
            "source_max_steps": row.source_max_steps,
            "batch_size": row.batch_size,
            "num_simulations": row.num_simulations,
            "lightzero_eval_freq": row.lightzero_eval_freq,
            "env_manager_type": row.env_manager_type,
            "ego_action_straight_override_probability": (
                row.ego_action_straight_override_probability
            ),
            "control_noise_profile_id": row.control_noise_profile_id,
            "background_eval_enabled": row.background_eval_enabled,
            "background_eval_launch_kind": row.background_eval_launch_kind,
            "background_eval_seed_count": row.background_eval_seed_count,
            "background_eval_max_steps": row.background_eval_max_steps,
            "background_eval_num_simulations": row.background_eval_num_simulations,
            "background_eval_batch_size": row.background_eval_batch_size,
            "background_gif_enabled": row.background_gif_enabled,
            "background_gif_max_steps": row.background_gif_max_steps,
            "background_gif_frame_stride": row.background_gif_frame_stride,
        },
        "artifact_refs": {
            "run_manifest": _ref("training", TASK_ID, run_id, "run.json"),
            "attempt_manifest": _ref(
                "training", TASK_ID, run_id, "attempts", attempt_id, "attempt.json"
            ),
            "latest_attempt": _ref("training", TASK_ID, run_id, "latest_attempt.json"),
            "summary": _ref(train_ref, "summary.json"),
            "command": _ref(
                "training", TASK_ID, run_id, "attempts", attempt_id, "command.json"
            ),
            "action_observability": _ref(train_ref, "action_observability.json"),
            "checkpoint_root": _ref("training", TASK_ID, run_id, "checkpoints", "lightzero"),
            "background_eval_status": _ref(train_ref, "checkpoint_eval_poller.json"),
        },
        "command": command,
        "command_text": command_text,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    matrix_name = _safe_id(args.matrix_name, label="matrix_name")
    run_prefix = _safe_id(args.run_prefix or f"curvytron-stock-{matrix_name}", label="run_prefix")
    attempt_prefix = _safe_id(
        args.attempt_prefix or f"{matrix_name}-attempt",
        label="attempt_prefix",
    )
    rows = _rows_for_matrix(
        matrix_name,
        compute=args.compute,
        max_train_iter=args.max_train_iter,
        save_ckpt_after_iter=args.save_ckpt_after_iter,
        source_state_trail_render_mode=args.source_state_trail_render_mode,
        recent_opponent_checkpoint_ref=args.recent_opponent_checkpoint_ref,
        mid_opponent_checkpoint_ref=args.mid_opponent_checkpoint_ref,
        old_opponent_checkpoint_ref=args.old_opponent_checkpoint_ref,
        include_joint_diagnostics=args.include_joint_diagnostics,
    )
    manifest_rows = [
        _manifest_row(
            row,
            matrix_name=matrix_name,
            run_prefix=run_prefix,
            attempt_prefix=attempt_prefix,
            detach=not args.no_detach,
        )
        for row in rows
    ]
    run_ids = [row["run_id"] for row in manifest_rows]
    attempt_ids = [row["attempt_id"] for row in manifest_rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("generated duplicate run_id values")
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("generated duplicate attempt_id values")
    command_text = "\n".join(row["command_text"] for row in manifest_rows)
    if "two-seat-selfplay" in command_text:
        raise ValueError("refusing manifest containing stale two-seat-selfplay")
    return {
        "schema_id": SCHEMA_ID,
        "generated_at": _utc_timestamp(),
        "dry_run_only": True,
        "launches_modal": False,
        "matrix_name": matrix_name,
        "task_id": TASK_ID,
        "canonical_launcher": MODULE,
        "run_prefix": run_prefix,
        "attempt_prefix": attempt_prefix,
        "row_count": len(manifest_rows),
        "guards": {
            "mode_required": MODE_TRAIN,
            "forbidden_mode": "two-seat-selfplay",
            "allowed_env_variants": [
                ENV_SOURCE_STATE_FIXED_OPPONENT,
                ENV_SOURCE_STATE_JOINT_ACTION,
            ],
            "source_state_trail_render_mode": args.source_state_trail_render_mode,
            "frozen_checkpoint_refs_must_be_immutable": True,
            "commands_reviewed_only": True,
            "include_joint_diagnostics": bool(args.include_joint_diagnostics),
        },
        "rows": manifest_rows,
    }


def _write_outputs(manifest: dict[str, Any], *, output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / str(manifest["matrix_name"])
    json_path = base.with_suffix(".json")
    jsonl_path = base.with_suffix(".commands.jsonl")
    sh_path = base.with_suffix(".commands.sh")
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    sh_path.write_text(
        "#!/usr/bin/env sh\n"
        "# Dry-run review artifact only. Do not execute without a fresh readiness check.\n"
        + "\n".join(row["command_text"] for row in manifest["rows"])
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_json": str(json_path),
        "commands_jsonl": str(jsonl_path),
        "commands_sh": str(sh_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dry-run manifest for CurvyTron stock Modal train launches."
    )
    parser.add_argument(
        "--matrix-name",
        default="stock-high-signal-v1",
        help=(
            "Named matrix: stock-high-signal-v1, stock-control-v1, "
            "stock-high-signal-v1b, stock-long-v1c, stock-tensor-v1d, or "
            "stock-fixed-sparse-seeds-v1."
        ),
    )
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--attempt-prefix", default=None)
    parser.add_argument("--compute", default="gpu-l4-t4")
    parser.add_argument("--max-train-iter", type=int, default=3000)
    parser.add_argument("--save-ckpt-after-iter", type=int, default=100)
    parser.add_argument(
        "--source-state-trail-render-mode",
        default=SOURCE_STATE_RENDER_BODY_CIRCLES_FAST,
        choices=(SOURCE_STATE_RENDER_BROWSER_LINES, SOURCE_STATE_RENDER_BODY_CIRCLES_FAST),
        help="Stock source-state render surface; body_circles_fast is the fast approximation.",
    )
    parser.add_argument(
        "--recent-opponent-checkpoint-ref",
        default=DEFAULT_RECENT_OPPONENT_CHECKPOINT_REF,
    )
    parser.add_argument(
        "--mid-opponent-checkpoint-ref",
        default=DEFAULT_MID_OPPONENT_CHECKPOINT_REF,
    )
    parser.add_argument(
        "--old-opponent-checkpoint-ref",
        default=DEFAULT_OLD_OPPONENT_CHECKPOINT_REF,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--include-joint-diagnostics",
        action="store_true",
        help=(
            "Include source_state_joint_action diagnostic rows. Default is off because "
            "fixed/frozen stock rows are the trusted training lane."
        ),
    )
    parser.add_argument(
        "--no-detach",
        action="store_true",
        help="Omit --detach from generated command text.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print the manifest and do not write artifact files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    if args.stdout_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    outputs = _write_outputs(manifest, output_root=args.output_root)
    print(
        json.dumps(
            {
                "ok": True,
                "dry_run_only": True,
                "launches_modal": False,
                "matrix_name": manifest["matrix_name"],
                "row_count": manifest["row_count"],
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
