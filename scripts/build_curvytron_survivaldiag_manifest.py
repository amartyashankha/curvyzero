#!/usr/bin/env python3
"""Build a dry-run-only manifest for the current CurvyTron survivaldiag matrix.

This generator is intentionally separate from
``build_curvytron_stock_train_manifest.py``. The older generator is historical
May 12 control material; this one encodes the current blank-canvas
survival-plus-bonus diagnostic lane and never launches Modal.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from curvyzero.env.vector_multiplayer_env import SOURCE_PHYSICS_STEP_MS


MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
APP_NAME = "curvyzero-lightzero-curvytron-visual-survival-train"
TASK_ID = "lightzero-curvytron-visual-survival"
SCHEMA_ID = "curvyzero_curvytron_survivaldiag_dry_run_manifest/v0"
ROW_SCHEMA_ID = "curvyzero_curvytron_survivaldiag_dry_run_manifest_row/v0"

MODE_TRAIN = "train"
FORBIDDEN_MODE = "two-seat-selfplay"
ENV_SOURCE_STATE_FIXED_OPPONENT = "source_state_fixed_opponent"
OPPONENT_POLICY_FIXED_STRAIGHT = "fixed_straight"
OPPONENT_POLICY_FROZEN_LIGHTZERO_CHECKPOINT = "frozen_lightzero_checkpoint"
OPPONENT_POLICY_PROACTIVE_WALL_AVOIDANT = "proactive_wall_avoidant"
OPPONENT_RUNTIME_NORMAL = "normal"
OPPONENT_RUNTIME_BLANK_CANVAS_NOOP = "blank_canvas_noop"
OPPONENT_DEATH_NORMAL = "normal"
OPPONENT_DEATH_IMMORTAL = "immortal"
REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME = "survival_plus_bonus_no_outcome"
REWARD_SURVIVAL_ONLY_GATED = "survival_only"
RENDER_FAST = "body_circles_fast"
RENDER_BROWSER = "browser_lines"
SOURCE_MAX_STEPS = 65_536
DECISION_MS = SOURCE_PHYSICS_STEP_MS
ACTION_REPEAT_SEED_OFFSET = 2027
STRAIGHT_OVERRIDE_SEED_OFFSET = 1009

TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT: tuple[str, ...] = (
    "mode",
    "seed",
    "run_id",
    "attempt_id",
    "max_env_step",
    "max_train_iter",
    "source_max_steps",
    "decision_ms",
    "collector_env_num",
    "evaluator_env_num",
    "n_evaluator_episode",
    "n_episode",
    "num_simulations",
    "batch_size",
    "lightzero_eval_freq",
    "skip_lightzero_eval_in_profile",
    "profile_cuda_sync_enabled",
    "profile_allow_auto_resume",
    "profile_volume_commit",
    "lightzero_multi_gpu",
    "save_ckpt_after_iter",
    "stop_after_learner_train_calls",
    "env_variant",
    "reward_variant",
    "source_state_trail_render_mode",
    "ego_action_straight_override_probability",
    "policy_action_repeat_min",
    "policy_action_repeat_max",
    "policy_action_repeat_extra_probability",
    "control_noise_profile_id",
    "disable_death_for_profile",
    "opponent_death_mode",
    "opponent_runtime_mode",
    "env_telemetry_stride",
    "env_manager_type",
    "opponent_policy_kind",
    "opponent_use_cuda",
    "opponent_checkpoint_ref",
    "opponent_snapshot_ref",
    "opponent_checkpoint_report_ref",
    "opponent_checkpoint_state_key",
    "background_eval_enabled",
    "background_eval_launch_kind",
    "background_eval_compute",
    "background_eval_id_prefix",
    "background_eval_seed_count",
    "background_eval_seed_rng_seed",
    "background_eval_max_steps",
    "background_eval_step_detail_limit",
    "background_eval_num_simulations",
    "background_eval_batch_size",
    "background_gif_enabled",
    "background_gif_seed_offset",
    "background_gif_max_steps",
    "background_gif_frame_stride",
    "background_gif_fps",
    "background_gif_scale",
    "background_gif_frame_size",
    "background_gif_collect_temperature",
    "background_gif_collect_epsilon",
)

DEFAULT_MATRIX_NAME = "curvy-survive-bonus-large"
DEFAULT_RUN_PREFIX = "curvy-survive-bonus"
DEFAULT_ATTEMPT_PREFIX = "try"
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_survivaldiag_manifests")
DEFAULT_RECENT_OPPONENT_CHECKPOINT_REF = (
    "training/lightzero-curvytron-visual-survival/"
    "curvytron-dense-ckpt1-iter10000-sanity-20260512a/"
    "checkpoints/lightzero/iteration_32.pth.tar"
)
MATRIX_PROFILE_FIRST_WAVE = "first_wave"
MATRIX_PROFILE_BLANK_REPEAT_EXPANSION = "blank_repeat_expansion"
MATRIX_PROFILE_LARGE_READY = "large_ready"
MATRIX_PROFILE_CHOICES = (
    MATRIX_PROFILE_LARGE_READY,
    MATRIX_PROFILE_FIRST_WAVE,
    MATRIX_PROFILE_BLANK_REPEAT_EXPANSION,
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


@dataclass(frozen=True)
class StochasticityProfile:
    profile_id: str
    tag: str
    description: str
    ego_action_straight_override_probability: float
    policy_action_repeat_min: int
    policy_action_repeat_max: int
    policy_action_repeat_extra_probability: float
    control_noise_profile_id: str


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    stage_index: int
    hypothesis_id: str
    description: str
    stochasticity_profile_ids: tuple[str, ...]
    copy_ids: tuple[str, ...]


@dataclass(frozen=True)
class Row:
    row_id: str
    row_kind: str
    block_id: str
    stage_index: int
    hypothesis_id: str
    label: str
    logical_pair_id: str
    render_pair_role: str
    source_state_trail_render_mode: str
    copy_id: str
    training_seed: int
    reset_seed: int
    opponent_policy_seed: int | None
    opponent_behavior_seed: int | None
    eval_seed: int
    straight_override_seed: int
    action_repeat_seed: int
    stochasticity: StochasticityProfile
    reward_variant: str
    opponent_policy_kind: str
    opponent_runtime_mode: str
    opponent_death_mode: str
    compute: str
    max_train_iter: int
    max_env_step: int
    save_ckpt_after_iter: int
    collector_env_num: int
    evaluator_env_num: int
    n_evaluator_episode: int
    n_episode: int
    batch_size: int
    num_simulations: int
    env_manager_type: str
    background_eval_launch_kind: str
    background_eval_compute: str
    background_eval_id_prefix: str
    background_eval_seed_count: int
    background_eval_max_steps: int
    background_eval_num_simulations: int
    background_eval_batch_size: int
    background_eval_poll_interval_sec: float
    background_eval_poll_stable_polls: int
    background_eval_poller_max_runtime_sec: float
    background_eval_poller_idle_after_done_sec: float
    background_gif_seed_offset: int
    background_gif_max_steps: int
    background_gif_frame_stride: int
    background_gif_fps: float
    background_gif_scale: int
    background_gif_frame_size: int
    background_gif_collect_temperature: float
    background_gif_collect_epsilon: float
    opponent_checkpoint_ref: str | None = None
    opponent_snapshot_ref: str | None = None
    row_note: str | None = None
    lightzero_eval_freq: int = 0
    background_eval_enabled: bool = True
    background_gif_enabled: bool = True


STOCHASTICITY_PROFILES: dict[str, StochasticityProfile] = {
    "none": StochasticityProfile(
        profile_id="none",
        tag="ar0",
        description="deterministic policy action, no action repeat beyond one source step",
        ego_action_straight_override_probability=0.0,
        policy_action_repeat_min=1,
        policy_action_repeat_max=1,
        policy_action_repeat_extra_probability=0.0,
        control_noise_profile_id="none",
    ),
    "low": StochasticityProfile(
        profile_id="low",
        tag="arlow",
        description="light held-action repeat inside one LightZero env step",
        ego_action_straight_override_probability=0.0,
        policy_action_repeat_min=1,
        policy_action_repeat_max=2,
        policy_action_repeat_extra_probability=0.10,
        control_noise_profile_id="policy_action_repeat_low",
    ),
    "medium": StochasticityProfile(
        profile_id="medium",
        tag="armed",
        description="medium held-action repeat pressure",
        ego_action_straight_override_probability=0.0,
        policy_action_repeat_min=1,
        policy_action_repeat_max=3,
        policy_action_repeat_extra_probability=0.20,
        control_noise_profile_id="policy_action_repeat_medium",
    ),
    "high": StochasticityProfile(
        profile_id="high",
        tag="arhigh",
        description="aggressive action-repeat pressure for survival robustness",
        ego_action_straight_override_probability=0.0,
        policy_action_repeat_min=1,
        policy_action_repeat_max=3,
        policy_action_repeat_extra_probability=0.35,
        control_noise_profile_id="policy_action_repeat_high",
    ),
}

BLOCK_SPECS: tuple[BlockSpec, ...] = (
    BlockSpec(
        block_id="b00_exact_preflight",
        stage_index=0,
        hypothesis_id="h00_exact_lane_preflight_before_first_wave",
        description=(
            "Exact lane preflight rows with the same high-cap reward/opponent/render "
            "contract and medium action-repeat pressure as the first wave."
        ),
        stochasticity_profile_ids=("medium",),
        copy_ids=("c00", "c01"),
    ),
    BlockSpec(
        block_id="b01_blank_canvas_core",
        stage_index=1,
        hypothesis_id="h01_blank_canvas_action_repeat_survival_core",
        description=(
            "The 32-row blank-canvas core: two render twins, four stochasticity "
            "levels, four copies, no reward/opponent/compute crossing."
        ),
        stochasticity_profile_ids=("none", "low", "medium", "high"),
        copy_ids=("c01", "c02", "c03", "c04"),
    ),
    BlockSpec(
        block_id="b02_blank_canvas_extra_repeats",
        stage_index=2,
        hypothesis_id="h02_blank_canvas_medium_high_repeat_stability",
        description=(
            "Extra blank-canvas repeats on medium/high stochasticity so the first "
            "wave spends spare rows on seed stability instead of gated opponents."
        ),
        stochasticity_profile_ids=("medium", "high"),
        copy_ids=("c05", "c06"),
    ),
)

RENDER_PAIRS: tuple[tuple[str, str], ...] = (
    ("fast", RENDER_FAST),
    ("browser", RENDER_BROWSER),
)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_id(raw: str, *, label: str, max_length: int = 96) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    if (
        not raw
        or len(raw) > max_length
        or raw in {".", ".."}
        or not raw[0].isalnum()
        or any(char not in allowed for char in raw)
    ):
        raise ValueError(
            f"{label} must be 1-{max_length} chars of letters, numbers, dash, "
            "underscore, or dot"
        )
    return raw


def _ref(*parts: str) -> str:
    return "/".join(parts)


def _derived_reset_seed(run_seed: int, reset_index: int = 0) -> int:
    seed_sequence = np.random.SeedSequence(
        [
            int(run_seed) & 0xFFFFFFFF,
            (int(run_seed) >> 32) & 0xFFFFFFFF,
            int(reset_index) & 0xFFFFFFFF,
            (int(reset_index) >> 32) & 0xFFFFFFFF,
        ]
    )
    rng = np.random.default_rng(seed_sequence)
    return int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))


def _copy_number(copy_id: str) -> int:
    if not copy_id.startswith("c"):
        raise ValueError(f"copy_id must start with 'c': {copy_id!r}")
    return int(copy_id[1:])


def _copy_ids(start: int, end: int) -> tuple[str, ...]:
    if start < 1 or end < start:
        raise ValueError(f"invalid copy range {start}..{end}")
    return tuple(f"c{index:02d}" for index in range(start, end + 1))


def _seed_base(*, block: BlockSpec, stochasticity_index: int, copy_id: str) -> int:
    return 910_000 + block.stage_index * 10_000 + stochasticity_index * 1_000 + (
        _copy_number(copy_id) * 10
    )


def _render_tag(render_mode: str) -> str:
    if render_mode == RENDER_FAST:
        return "fast"
    if render_mode == RENDER_BROWSER:
        return "browser"
    return render_mode.replace("_", "-")


def _compute_tag(compute: str) -> str:
    return (
        compute.replace("gpu-", "")
        .replace("cpu", "c")
        .replace("-", "")
        .replace("_", "")
    )


def _stochasticity_name(profile: StochasticityProfile) -> str:
    return {
        "none": "steady",
        "low": "light",
        "medium": "medium",
        "high": "heavy",
    }.get(profile.profile_id, profile.profile_id.replace("_", "-"))


def _opponent_name(row: Row) -> str:
    if row.opponent_runtime_mode == OPPONENT_RUNTIME_BLANK_CANVAS_NOOP:
        return "blank"
    if row.opponent_policy_kind == OPPONENT_POLICY_PROACTIVE_WALL_AVOIDANT:
        return "scripted"
    if row.opponent_policy_kind == OPPONENT_POLICY_FROZEN_LIGHTZERO_CHECKPOINT:
        return "ancestor"
    if row.opponent_death_mode == OPPONENT_DEATH_IMMORTAL:
        return "passive"
    return "fixed"


def _compute_name(row: Row) -> str:
    labels: list[str] = []
    if row.compute == "gpu-h100-cpu40":
        labels.append("h100")
    elif row.compute == "gpu-h100x2-cpu40":
        labels.append("h100x2")
    if row.num_simulations == 16:
        labels.append("search16")
    if row.collector_env_num == 64 or row.n_episode == 64:
        labels.append("collect64")
    if row.batch_size == 64:
        labels.append("batch64")
    return "-".join(labels) if labels else "base"


def _row_copy_name(row: Row) -> str:
    return f"r{int(row.row_id):03d}"


def _human_label(row: Row) -> str:
    return "-".join(
        [
            _opponent_name(row),
            _render_tag(row.source_state_trail_render_mode),
            _stochasticity_name(row.stochasticity),
            _compute_name(row),
            _row_copy_name(row),
        ]
    )


def _train_function_name(compute: str) -> str:
    if compute == "cpu":
        return "lightzero_curvytron_visual_survival_cpu"
    if compute == "cpu64":
        return "lightzero_curvytron_visual_survival_cpu64"
    if compute == "gpu-l4-t4":
        return "lightzero_curvytron_visual_survival_gpu"
    if compute == "gpu-l4-t4-cpu40":
        return "lightzero_curvytron_visual_survival_gpu_cpu40"
    if compute == "gpu-h100-cpu40":
        return "lightzero_curvytron_visual_survival_h100_cpu40"
    if compute == "gpu-h100x2-cpu40":
        return "lightzero_curvytron_visual_survival_h100x2_cpu40"
    raise ValueError(f"unknown compute {compute!r}")


def _reward_contract(reward_variant: str) -> dict[str, Any]:
    if reward_variant != REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
        raise ValueError(f"unsupported executable reward_variant: {reward_variant!r}")
    return {
        "schema_id": "curvyzero_survival_plus_bonus_no_outcome/v0",
        "reward_variant": REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        "reward_survival_weight": 1.0,
        "reward_bonus_weight": 1.0,
        "reward_outcome_weight": 0.0,
        "trainer_reward_terms": [
            "dense_alive_helper_for_ego_player",
            "same_step_bonus_pickup_helper_for_ego_player",
        ],
        "reward_components": {
            "dense_alive_helper_for_ego_player": 1.0,
            "same_step_bonus_pickup_helper_for_ego_player": 1.0,
            "bonus_pickup_reward_per_catch": 1.0,
            "terminal_outcome_bonus": 0.0,
            "loser_penalty": 0.0,
            "winner_bonus": 0.0,
            "draw_bonus": 0.0,
            "truncation_bonus": 0.0,
        },
        "outcome_is_telemetry_only": True,
    }


def _opponent_contract(row: Row) -> dict[str, Any]:
    if row.opponent_runtime_mode == OPPONENT_RUNTIME_BLANK_CANVAS_NOOP:
        trail_mode = "none_blank_canvas_scrubbed"
        collision_effect = "disabled_no_player_1_movement_trail_collision_bonus_side_effects"
        visibility_mode = "hidden_from_model_gif_raw_render_views_public_present_alive"
        claim = "blank_canvas_noop_player_1_inert_hidden_no_trail_no_collision_no_bonus"
        policy_seed_kind = "not_applicable_blank_canvas_noop"
        behavior_seed_kind = "not_applicable_blank_canvas_noop"
    else:
        trail_mode = "normal"
        collision_effect = "normal"
        visibility_mode = "visible_if_present_alive"
        claim = (
            "passive_immortal_dirty_control_not_main_claim"
            if row.opponent_death_mode == OPPONENT_DEATH_IMMORTAL
            else "normal_frozen_ancestor_control_not_main_claim"
        )
        policy_seed_kind = "immutable_checkpoint_ref" if row.opponent_checkpoint_ref else "fixed_policy"
        behavior_seed_kind = "deterministic_fixed_or_checkpoint_policy"
    return {
        "opponent_runtime_mode": row.opponent_runtime_mode,
        "opponent_policy_kind": row.opponent_policy_kind,
        "opponent_death_mode": row.opponent_death_mode,
        "opponent_trail_mode": trail_mode,
        "opponent_collision_effect": collision_effect,
        "opponent_visibility_mode": visibility_mode,
        "opponent_contract_claim": claim,
        "opponent_policy_seed_kind": policy_seed_kind,
        "opponent_behavior_seed_kind": behavior_seed_kind,
        "opponent_checkpoint_ref": row.opponent_checkpoint_ref,
        "opponent_snapshot_ref": row.opponent_snapshot_ref,
    }


def _rows(args: argparse.Namespace) -> list[Row]:
    rows: list[Row] = []

    def add_render_pair(
        *,
        block: BlockSpec,
        row_kind: str,
        profile: StochasticityProfile,
        stochasticity_index: int,
        copy_id: str,
        reward_variant: str = REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        opponent_policy_kind: str = OPPONENT_POLICY_FIXED_STRAIGHT,
        opponent_runtime_mode: str = OPPONENT_RUNTIME_BLANK_CANVAS_NOOP,
        opponent_death_mode: str = OPPONENT_DEATH_NORMAL,
        opponent_checkpoint_ref: str | None = None,
        opponent_snapshot_ref: str | None = None,
        compute: str | None = None,
        num_simulations: int | None = None,
        collector_env_num: int | None = None,
        n_episode: int | None = None,
        batch_size: int | None = None,
        label_suffix: str | None = None,
        row_note: str | None = None,
    ) -> None:
        seed_base = _seed_base(
            block=block,
            stochasticity_index=stochasticity_index,
            copy_id=copy_id,
        )
        if label_suffix:
            seed_base += sum(ord(char) for char in label_suffix)
        training_seed = seed_base + 1
        reset_seed = _derived_reset_seed(training_seed, reset_index=0)
        eval_seed = seed_base + 7
        opponent_tag = (
            "blanknoop"
            if opponent_runtime_mode == OPPONENT_RUNTIME_BLANK_CANVAS_NOOP
            else (
                "immdirty"
                if opponent_death_mode == OPPONENT_DEATH_IMMORTAL
                else "ancestor"
            )
        )
        logical_pair_id = (
            f"{block.block_id}-{opponent_tag}-{profile.tag}-{copy_id}-"
            f"train{training_seed}-eval{eval_seed}"
        )
        for render_role, render_mode in RENDER_PAIRS:
            row_number = len(rows) + 1
            label_parts = [
                "survbonusnoout",
                opponent_tag,
                _render_tag(render_mode),
                profile.tag,
                copy_id,
            ]
            if label_suffix:
                label_parts.append(label_suffix)
            label = "-".join(label_parts)
            rows.append(
                Row(
                    row_id=f"{row_number:03d}",
                    row_kind=row_kind,
                    block_id=block.block_id,
                    stage_index=block.stage_index,
                    hypothesis_id=block.hypothesis_id,
                    label=label,
                    logical_pair_id=logical_pair_id,
                    render_pair_role=render_role,
                    source_state_trail_render_mode=render_mode,
                    copy_id=copy_id,
                    training_seed=training_seed,
                    reset_seed=reset_seed,
                    opponent_policy_seed=None,
                    opponent_behavior_seed=None,
                    eval_seed=eval_seed,
                    straight_override_seed=reset_seed + STRAIGHT_OVERRIDE_SEED_OFFSET,
                    action_repeat_seed=reset_seed + ACTION_REPEAT_SEED_OFFSET,
                    stochasticity=profile,
                    reward_variant=reward_variant,
                    opponent_policy_kind=opponent_policy_kind,
                    opponent_runtime_mode=opponent_runtime_mode,
                    opponent_death_mode=opponent_death_mode,
                    compute=compute or args.compute,
                    max_train_iter=args.max_train_iter,
                    max_env_step=args.max_env_step,
                    save_ckpt_after_iter=args.save_ckpt_after_iter,
                    collector_env_num=collector_env_num or args.collector_env_num,
                    evaluator_env_num=args.evaluator_env_num,
                    n_evaluator_episode=args.n_evaluator_episode,
                    n_episode=n_episode or args.n_episode,
                    batch_size=batch_size or args.batch_size,
                    num_simulations=num_simulations or args.num_simulations,
                    env_manager_type=args.env_manager_type,
                    background_eval_launch_kind=args.background_eval_launch_kind,
                    background_eval_compute=args.background_eval_compute,
                    background_eval_id_prefix=args.background_eval_id_prefix,
                    background_eval_seed_count=args.background_eval_seed_count,
                    background_eval_max_steps=args.background_eval_max_steps,
                    background_eval_num_simulations=args.background_eval_num_simulations,
                    background_eval_batch_size=args.background_eval_batch_size,
                    background_eval_poll_interval_sec=args.background_eval_poll_interval_sec,
                    background_eval_poll_stable_polls=args.background_eval_poll_stable_polls,
                    background_eval_poller_max_runtime_sec=(
                        args.background_eval_poller_max_runtime_sec
                    ),
                    background_eval_poller_idle_after_done_sec=(
                        args.background_eval_poller_idle_after_done_sec
                    ),
                    background_gif_seed_offset=args.background_gif_seed_offset,
                    background_gif_max_steps=args.background_gif_max_steps,
                    background_gif_frame_stride=args.background_gif_frame_stride,
                    background_gif_fps=args.background_gif_fps,
                    background_gif_scale=args.background_gif_scale,
                    background_gif_frame_size=args.background_gif_frame_size,
                    background_gif_collect_temperature=(
                        args.background_gif_collect_temperature
                    ),
                    background_gif_collect_epsilon=args.background_gif_collect_epsilon,
                    opponent_checkpoint_ref=opponent_checkpoint_ref,
                    opponent_snapshot_ref=opponent_snapshot_ref,
                    row_note=row_note,
                )
            )

    if args.matrix_profile == MATRIX_PROFILE_LARGE_READY:
        blank_all_block = BlockSpec(
            block_id="b20_blank_canvas_all_level_repeats",
            stage_index=20,
            hypothesis_id="h20_blank_canvas_all_stochasticity_large_repeat",
            description=(
                "Main large batch block: blank canvas, both renders, all four "
                "action-repeat levels, twenty fresh copies."
            ),
            stochasticity_profile_ids=("none", "low", "medium", "high"),
            copy_ids=_copy_ids(1, 20),
        )
        for stochasticity_index, profile_id in enumerate(
            blank_all_block.stochasticity_profile_ids
        ):
            for copy_id in blank_all_block.copy_ids:
                add_render_pair(
                    block=blank_all_block,
                    row_kind="blank_canvas_large_repeat",
                    profile=STOCHASTICITY_PROFILES[profile_id],
                    stochasticity_index=stochasticity_index,
                    copy_id=copy_id,
                )

        blank_focus_block = BlockSpec(
            block_id="b21_blank_canvas_medium_high_extra",
            stage_index=21,
            hypothesis_id="h21_blank_canvas_medium_high_extra_repeat",
            description=(
                "Extra blank-canvas repeats on medium/high action-repeat levels, "
                "where the first diagnostic batch most wanted more copies."
            ),
            stochasticity_profile_ids=("medium", "high"),
            copy_ids=_copy_ids(21, 30),
        )
        for stochasticity_index, profile_id in enumerate(
            blank_focus_block.stochasticity_profile_ids
        ):
            for copy_id in blank_focus_block.copy_ids:
                add_render_pair(
                    block=blank_focus_block,
                    row_kind="blank_canvas_large_repeat",
                    profile=STOCHASTICITY_PROFILES[profile_id],
                    stochasticity_index=stochasticity_index,
                    copy_id=copy_id,
                )

        passive_block = BlockSpec(
            block_id="b22_passive_immortal_dirty_controls",
            stage_index=22,
            hypothesis_id="h22_passive_immortal_dirty_control_large",
            description=(
                "Passive immortal fixed-straight dirty controls. They are trail "
                "pressure checks, not the main learning claim."
            ),
            stochasticity_profile_ids=("none", "low", "medium", "high"),
            copy_ids=_copy_ids(1, 5),
        )
        for stochasticity_index, profile_id in enumerate(
            passive_block.stochasticity_profile_ids
        ):
            for copy_id in passive_block.copy_ids:
                add_render_pair(
                    block=passive_block,
                    row_kind="dirty_control",
                    profile=STOCHASTICITY_PROFILES[profile_id],
                    stochasticity_index=stochasticity_index,
                    copy_id=copy_id,
                    opponent_runtime_mode=OPPONENT_RUNTIME_NORMAL,
                    opponent_death_mode=OPPONENT_DEATH_IMMORTAL,
                    label_suffix="dirty",
                    row_note="passive_immortal_dirty_control_not_main_claim",
                )

        compute_block = BlockSpec(
            block_id="b23_blank_canvas_compute_sentinels",
            stage_index=23,
            hypothesis_id="h23_blank_canvas_small_compute_sentinels",
            description=(
                "Small compute sentinels on blank-canvas medium/high rows: "
                "search16, collector64, and batch64."
            ),
            stochasticity_profile_ids=("medium", "high"),
            copy_ids=_copy_ids(1, 5),
        )
        for stochasticity_index, profile_id in enumerate(
            compute_block.stochasticity_profile_ids
        ):
            for copy_id in compute_block.copy_ids:
                profile = STOCHASTICITY_PROFILES[profile_id]
                add_render_pair(
                    block=compute_block,
                    row_kind="compute_sentinel",
                    profile=profile,
                    stochasticity_index=stochasticity_index,
                    copy_id=copy_id,
                    num_simulations=16,
                    label_suffix="search16",
                    row_note="blank_canvas_search16_sentinel",
                )
                add_render_pair(
                    block=compute_block,
                    row_kind="compute_sentinel",
                    profile=profile,
                    stochasticity_index=stochasticity_index,
                    copy_id=copy_id,
                    collector_env_num=64,
                    n_episode=64,
                    label_suffix="collect64",
                    row_note="blank_canvas_collect64_sentinel",
                )
                add_render_pair(
                    block=compute_block,
                    row_kind="compute_sentinel",
                    profile=profile,
                    stochasticity_index=stochasticity_index,
                    copy_id=copy_id,
                    batch_size=64,
                    label_suffix="batch64",
                    row_note="blank_canvas_batch64_sentinel",
                )
        return rows

    if args.matrix_profile == MATRIX_PROFILE_BLANK_REPEAT_EXPANSION:
        repeat_all_block = BlockSpec(
            block_id="b10_blank_canvas_scale_all",
            stage_index=10,
            hypothesis_id="h10_blank_canvas_repeat_all_stochasticity_scale",
            description=(
                "Second-wave clean repeat block: blank canvas, all stochasticity "
                "levels, four new copies, matched fast/browser render pairs."
            ),
            stochasticity_profile_ids=("none", "low", "medium", "high"),
            copy_ids=("c07", "c08", "c09", "c10"),
        )
        for stochasticity_index, profile_id in enumerate(
            repeat_all_block.stochasticity_profile_ids
        ):
            for copy_id in repeat_all_block.copy_ids:
                add_render_pair(
                    block=repeat_all_block,
                    row_kind="blank_canvas_scale_repeat",
                    profile=STOCHASTICITY_PROFILES[profile_id],
                    stochasticity_index=stochasticity_index,
                    copy_id=copy_id,
                )

        repeat_focus_block = BlockSpec(
            block_id="b11_blank_canvas_medium_high_scale",
            stage_index=11,
            hypothesis_id="h11_blank_canvas_medium_high_extra_scale",
            description=(
                "Extra clean repeat block on medium/high action-repeat pressure."
            ),
            stochasticity_profile_ids=("medium", "high"),
            copy_ids=("c11", "c12"),
        )
        for stochasticity_index, profile_id in enumerate(
            repeat_focus_block.stochasticity_profile_ids
        ):
            for copy_id in repeat_focus_block.copy_ids:
                add_render_pair(
                    block=repeat_focus_block,
                    row_kind="blank_canvas_scale_repeat",
                    profile=STOCHASTICITY_PROFILES[profile_id],
                    stochasticity_index=stochasticity_index,
                    copy_id=copy_id,
                )

        dirty_extension_block = BlockSpec(
            block_id="b12_passive_immortal_dirty_extension",
            stage_index=12,
            hypothesis_id="h12_passive_immortal_low_high_dirty_control_extension",
            description=(
                "Small passive-immortal dirty control extension; not a clean "
                "opponent claim."
            ),
            stochasticity_profile_ids=("low", "high"),
            copy_ids=("c03",),
        )
        for stochasticity_index, profile_id in enumerate(
            dirty_extension_block.stochasticity_profile_ids
        ):
            for copy_id in dirty_extension_block.copy_ids:
                add_render_pair(
                    block=dirty_extension_block,
                    row_kind="dirty_control",
                    profile=STOCHASTICITY_PROFILES[profile_id],
                    stochasticity_index=stochasticity_index,
                    copy_id=copy_id,
                    opponent_runtime_mode=OPPONENT_RUNTIME_NORMAL,
                    opponent_death_mode=OPPONENT_DEATH_IMMORTAL,
                    label_suffix="dirty",
                    row_note="dirty_control_extension_only_not_promoted_to_main_claim",
                )

        compute_extension_block = BlockSpec(
            block_id="b13_compute_sentinel_extension",
            stage_index=13,
            hypothesis_id="h13_small_compute_sentinel_extension",
            description=(
                "Small compute sentinels on clean blank-canvas rows: sim16, "
                "collector64, and batch64."
            ),
            stochasticity_profile_ids=("medium",),
            copy_ids=("c01",),
        )
        add_render_pair(
            block=compute_extension_block,
            row_kind="compute_sentinel",
            profile=STOCHASTICITY_PROFILES["medium"],
            stochasticity_index=0,
            copy_id="c02",
            num_simulations=16,
            label_suffix="sim16",
            row_note="sim16_sentinel_extension_only",
        )
        add_render_pair(
            block=compute_extension_block,
            row_kind="compute_sentinel",
            profile=STOCHASTICITY_PROFILES["medium"],
            stochasticity_index=1,
            copy_id="c01",
            collector_env_num=64,
            n_episode=64,
            label_suffix="c64",
            row_note="collector64_sentinel_extension_only",
        )
        add_render_pair(
            block=compute_extension_block,
            row_kind="compute_sentinel",
            profile=STOCHASTICITY_PROFILES["medium"],
            stochasticity_index=2,
            copy_id="c01",
            batch_size=64,
            label_suffix="b64",
            row_note="batch64_sentinel_extension_only",
        )
        return rows

    for block in BLOCK_SPECS:
        for stochasticity_index, profile_id in enumerate(block.stochasticity_profile_ids):
            profile = STOCHASTICITY_PROFILES[profile_id]
            for copy_id in block.copy_ids:
                add_render_pair(
                    stochasticity_index=stochasticity_index,
                    block=block,
                    row_kind=(
                        "exact_preflight"
                        if block.block_id == "b00_exact_preflight"
                        else (
                            "blank_canvas_extra_repeat"
                            if block.block_id == "b02_blank_canvas_extra_repeats"
                            else "blank_canvas_core"
                        )
                    ),
                    profile=profile,
                    copy_id=copy_id,
                )

    passive_block = BlockSpec(
        block_id="b03_passive_immortal_dirty_control",
        stage_index=3,
        hypothesis_id="h03_passive_immortal_dirty_trail_control",
        description="Tiny dirty control; passive immortal is not a main learning claim.",
        stochasticity_profile_ids=("medium",),
        copy_ids=("c01", "c02"),
    )
    for stochasticity_index, profile_id in enumerate(passive_block.stochasticity_profile_ids):
        for copy_id in passive_block.copy_ids:
            add_render_pair(
                block=passive_block,
                row_kind="dirty_control",
                profile=STOCHASTICITY_PROFILES[profile_id],
                stochasticity_index=stochasticity_index,
                copy_id=copy_id,
                opponent_runtime_mode=OPPONENT_RUNTIME_NORMAL,
                opponent_death_mode=OPPONENT_DEATH_IMMORTAL,
                label_suffix="dirty",
                row_note="dirty_control_only_not_promoted_to_main_claim",
            )

    compute_block = BlockSpec(
        block_id="b04_compute_sentinel_sim16",
        stage_index=4,
        hypothesis_id="h04_sim16_sentinel_on_strong_blank_cell",
        description="Minimal sim16 sentinel on one matched medium-stochasticity blank cell.",
        stochasticity_profile_ids=("medium",),
        copy_ids=("c01",),
    )
    add_render_pair(
        block=compute_block,
        row_kind="compute_sentinel",
        profile=STOCHASTICITY_PROFILES["medium"],
        stochasticity_index=0,
        copy_id="c01",
        num_simulations=16,
        label_suffix="sim16",
        row_note="sim16_sentinel_only_no_broad_compute_expansion_without_blank_core_readout",
    )
    return rows


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
            str(row.training_seed),
            "--run-id",
            run_id,
            "--attempt-id",
            attempt_id,
            "--env-variant",
            ENV_SOURCE_STATE_FIXED_OPPONENT,
            "--reward-variant",
            row.reward_variant,
            "--opponent-policy-kind",
            row.opponent_policy_kind,
            "--opponent-runtime-mode",
            row.opponent_runtime_mode,
            "--opponent-death-mode",
            row.opponent_death_mode,
            "--source-state-trail-render-mode",
            row.source_state_trail_render_mode,
            "--ego-action-straight-override-probability",
            str(row.stochasticity.ego_action_straight_override_probability),
            "--control-noise-profile-id",
            row.stochasticity.control_noise_profile_id,
            "--policy-action-repeat-min",
            str(row.stochasticity.policy_action_repeat_min),
            "--policy-action-repeat-max",
            str(row.stochasticity.policy_action_repeat_max),
            "--policy-action-repeat-extra-probability",
            str(row.stochasticity.policy_action_repeat_extra_probability),
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
            str(SOURCE_MAX_STEPS),
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
            "--background-eval-compute",
            row.background_eval_compute,
            "--background-eval-id-prefix",
            row.background_eval_id_prefix,
            "--background-eval-seed-count",
            str(row.background_eval_seed_count),
            "--background-eval-seed-rng-seed",
            str(row.eval_seed),
            "--background-eval-max-steps",
            str(row.background_eval_max_steps),
            "--background-eval-num-simulations",
            str(row.background_eval_num_simulations),
            "--background-eval-batch-size",
            str(row.background_eval_batch_size),
            "--background-eval-poll-interval-sec",
            str(row.background_eval_poll_interval_sec),
            "--background-eval-poll-stable-polls",
            str(row.background_eval_poll_stable_polls),
            "--background-eval-poller-max-runtime-sec",
            str(row.background_eval_poller_max_runtime_sec),
            "--background-eval-poller-idle-after-done-sec",
            str(row.background_eval_poller_idle_after_done_sec),
            "--background-gif-seed-offset",
            str(row.background_gif_seed_offset),
            "--background-gif-max-steps",
            str(row.background_gif_max_steps),
            "--background-gif-frame-stride",
            str(row.background_gif_frame_stride),
            "--background-gif-fps",
            str(row.background_gif_fps),
            "--background-gif-scale",
            str(row.background_gif_scale),
            "--background-gif-frame-size",
            str(row.background_gif_frame_size),
            "--background-gif-collect-temperature",
            str(row.background_gif_collect_temperature),
            "--background-gif-collect-epsilon",
            str(row.background_gif_collect_epsilon),
            "--output-detail",
            "compact",
        ]
    )
    if row.opponent_checkpoint_ref is not None:
        command.extend(["--opponent-checkpoint-ref", row.opponent_checkpoint_ref])
    if row.opponent_snapshot_ref is not None:
        command.extend(["--snapshot-ref", row.opponent_snapshot_ref])
    return command


def _train_kwargs_for_row(row: Row, *, run_id: str, attempt_id: str) -> dict[str, Any]:
    return {
        "mode": MODE_TRAIN,
        "seed": row.training_seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "max_env_step": row.max_env_step,
        "max_train_iter": row.max_train_iter,
        "source_max_steps": SOURCE_MAX_STEPS,
        "decision_ms": DECISION_MS,
        "collector_env_num": row.collector_env_num,
        "evaluator_env_num": row.evaluator_env_num,
        "n_evaluator_episode": row.n_evaluator_episode,
        "n_episode": row.n_episode,
        "num_simulations": row.num_simulations,
        "batch_size": row.batch_size,
        "lightzero_eval_freq": row.lightzero_eval_freq,
        "skip_lightzero_eval_in_profile": True,
        "profile_cuda_sync_enabled": False,
        "profile_allow_auto_resume": False,
        "profile_volume_commit": False,
        "lightzero_multi_gpu": False,
        "save_ckpt_after_iter": row.save_ckpt_after_iter,
        "stop_after_learner_train_calls": 0,
        "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": row.reward_variant,
        "source_state_trail_render_mode": row.source_state_trail_render_mode,
        "ego_action_straight_override_probability": (
            row.stochasticity.ego_action_straight_override_probability
        ),
        "policy_action_repeat_min": row.stochasticity.policy_action_repeat_min,
        "policy_action_repeat_max": row.stochasticity.policy_action_repeat_max,
        "policy_action_repeat_extra_probability": (
            row.stochasticity.policy_action_repeat_extra_probability
        ),
        "control_noise_profile_id": row.stochasticity.control_noise_profile_id,
        "opponent_death_mode": row.opponent_death_mode,
        "opponent_runtime_mode": row.opponent_runtime_mode,
        "disable_death_for_profile": False,
        "env_telemetry_stride": 1,
        "env_manager_type": row.env_manager_type,
        "opponent_policy_kind": row.opponent_policy_kind,
        "opponent_use_cuda": False,
        "opponent_checkpoint_ref": row.opponent_checkpoint_ref,
        "opponent_snapshot_ref": row.opponent_snapshot_ref,
        "opponent_checkpoint_report_ref": None,
        "opponent_checkpoint_state_key": None,
        "background_eval_enabled": row.background_eval_enabled,
        "background_eval_launch_kind": row.background_eval_launch_kind,
        "background_eval_compute": row.background_eval_compute,
        "background_eval_id_prefix": row.background_eval_id_prefix,
        "background_eval_seed_count": row.background_eval_seed_count,
        "background_eval_seed_rng_seed": row.eval_seed,
        "background_eval_max_steps": row.background_eval_max_steps,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": row.background_eval_num_simulations,
        "background_eval_batch_size": row.background_eval_batch_size,
        "background_gif_enabled": row.background_gif_enabled,
        "background_gif_seed_offset": row.background_gif_seed_offset,
        "background_gif_max_steps": row.background_gif_max_steps,
        "background_gif_frame_stride": row.background_gif_frame_stride,
        "background_gif_fps": row.background_gif_fps,
        "background_gif_scale": row.background_gif_scale,
        "background_gif_frame_size": row.background_gif_frame_size,
        "background_gif_collect_temperature": row.background_gif_collect_temperature,
        "background_gif_collect_epsilon": row.background_gif_collect_epsilon,
    }


def _poller_kwargs_for_row(row: Row, *, run_id: str, attempt_id: str) -> dict[str, Any]:
    exp_name_ref = _ref(
        "training",
        TASK_ID,
        run_id,
        "attempts",
        attempt_id,
        "train",
        "lightzero_exp",
    )
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "exp_name_ref": exp_name_ref,
        "seed": row.training_seed,
        "source_max_steps": SOURCE_MAX_STEPS,
        "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": row.reward_variant,
        "opponent_policy_kind": row.opponent_policy_kind,
        "opponent_checkpoint_ref": row.opponent_checkpoint_ref,
        "opponent_snapshot_ref": row.opponent_snapshot_ref,
        "opponent_checkpoint_state_key": None,
        "opponent_death_mode": row.opponent_death_mode,
        "opponent_runtime_mode": row.opponent_runtime_mode,
        "background_eval_compute": row.background_eval_compute,
        "background_eval_id_prefix": row.background_eval_id_prefix,
        "background_eval_seed_count": row.background_eval_seed_count,
        "background_eval_seed_rng_seed": row.eval_seed,
        "background_eval_max_steps": row.background_eval_max_steps,
        "background_eval_num_simulations": row.background_eval_num_simulations,
        "background_eval_batch_size": row.background_eval_batch_size,
        "background_gif_enabled": row.background_gif_enabled,
        "background_gif_seed_offset": row.background_gif_seed_offset,
        "background_gif_max_steps": row.background_gif_max_steps,
        "background_gif_frame_stride": row.background_gif_frame_stride,
        "background_gif_fps": row.background_gif_fps,
        "background_gif_scale": row.background_gif_scale,
        "background_gif_frame_size": row.background_gif_frame_size,
        "background_gif_collect_temperature": row.background_gif_collect_temperature,
        "background_gif_collect_epsilon": row.background_gif_collect_epsilon,
        "poll_interval_sec": row.background_eval_poll_interval_sec,
        "stable_polls": row.background_eval_poll_stable_polls,
        "max_runtime_sec": row.background_eval_poller_max_runtime_sec,
        "idle_after_train_done_sec": row.background_eval_poller_idle_after_done_sec,
    }


def _validate_train_kwargs_shape(row_id: str, train_kwargs: dict[str, Any]) -> None:
    missing = [
        key for key in TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT if key not in train_kwargs
    ]
    if missing:
        raise ValueError(
            f"row {row_id} grouped train kwargs missing required keys: {missing}"
        )


def _manifest_row(
    row: Row,
    *,
    matrix_name: str,
    run_prefix: str,
    attempt_prefix: str,
    detach: bool,
) -> dict[str, Any]:
    human_label = _human_label(row)
    run_id = _safe_id(f"{run_prefix}-{human_label}-s{row.training_seed}", label="run_id")
    attempt_id = _safe_id(
        f"{attempt_prefix}-{human_label}-s{row.training_seed}",
        label="attempt_id",
    )
    command = _command_for_row(row, run_id=run_id, attempt_id=attempt_id, detach=detach)
    command_text = shlex.join(command)
    if FORBIDDEN_MODE in command_text:
        raise ValueError(f"refusing stale two-seat command for row {row.row_id}")
    if command[command.index("--mode") + 1] != MODE_TRAIN:
        raise ValueError(f"row {row.row_id} must use --mode train")
    if row.source_state_trail_render_mode not in {RENDER_FAST, RENDER_BROWSER}:
        raise ValueError(f"unexpected render mode in row {row.row_id}")
    reward_contract = _reward_contract(row.reward_variant)
    opponent_contract = _opponent_contract(row)
    train_ref = _ref("training", TASK_ID, run_id, "attempts", attempt_id, "train")
    train_kwargs = _train_kwargs_for_row(row, run_id=run_id, attempt_id=attempt_id)
    _validate_train_kwargs_shape(row.row_id, train_kwargs)
    poller_kwargs = _poller_kwargs_for_row(row, run_id=run_id, attempt_id=attempt_id)
    return {
        "schema_id": ROW_SCHEMA_ID,
        "matrix_name": matrix_name,
        "row_id": row.row_id,
        "status": "planned_dry_run_only",
        "row_kind": row.row_kind,
        "block_id": row.block_id,
        "stage_index": row.stage_index,
        "hypothesis_id": row.hypothesis_id,
        "label": human_label,
        "internal_label": row.label,
        "plain_name": human_label,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "mode": MODE_TRAIN,
        "canonical_launcher": MODULE,
        "calls_stock_train_muzero": True,
        "two_seat_self_play": False,
        "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": row.reward_variant,
        "reward_survival_weight": reward_contract["reward_survival_weight"],
        "reward_bonus_weight": reward_contract["reward_bonus_weight"],
        "reward_outcome_weight": reward_contract["reward_outcome_weight"],
        "reward_components": reward_contract["reward_components"],
        "reward_contract": reward_contract,
        "opponent_runtime_mode": row.opponent_runtime_mode,
        "opponent_policy_kind": row.opponent_policy_kind,
        "opponent_death_mode": row.opponent_death_mode,
        "opponent_trail_mode": opponent_contract["opponent_trail_mode"],
        "opponent_collision_effect": opponent_contract["opponent_collision_effect"],
        "opponent_visibility_mode": opponent_contract["opponent_visibility_mode"],
        "opponent_contract": opponent_contract,
        "opponent_checkpoint_ref": row.opponent_checkpoint_ref,
        "opponent_snapshot_ref": row.opponent_snapshot_ref,
        "source_max_steps": SOURCE_MAX_STEPS,
        "source_state_trail_render_mode": row.source_state_trail_render_mode,
        "logical_pair_id": row.logical_pair_id,
        "render_pair_role": row.render_pair_role,
        "copy_id": row.copy_id,
        "training_seed": row.training_seed,
        "reset_seed": row.reset_seed,
        "reset_seed_index": 0,
        "reset_seed_strategy": "dynamic_seed_sequence_from_run_seed_and_reset_index/v0",
        "opponent_policy_seed": row.opponent_policy_seed,
        "opponent_behavior_seed": row.opponent_behavior_seed,
        "eval_seed": row.eval_seed,
        "straight_override_seed": row.straight_override_seed,
        "action_repeat_seed": row.action_repeat_seed,
        "seed_fields": {
            "training_seed": row.training_seed,
            "reset_seed": row.reset_seed,
            "reset_seed_index": 0,
            "opponent_policy_seed": row.opponent_policy_seed,
            "opponent_behavior_seed": row.opponent_behavior_seed,
            "eval_seed": row.eval_seed,
            "straight_override_seed": row.straight_override_seed,
            "action_repeat_seed": row.action_repeat_seed,
        },
        "stochasticity_profile_id": row.stochasticity.profile_id,
        "stochasticity_description": row.stochasticity.description,
        "ego_action_straight_override_probability": (
            row.stochasticity.ego_action_straight_override_probability
        ),
        "policy_action_repeat_min": row.stochasticity.policy_action_repeat_min,
        "policy_action_repeat_max": row.stochasticity.policy_action_repeat_max,
        "policy_action_repeat_extra_probability": (
            row.stochasticity.policy_action_repeat_extra_probability
        ),
        "policy_action_repeat_semantics": (
            "repeat_selected_policy_action_inside_one_lightzero_env_step"
        ),
        "control_noise_profile_id": row.stochasticity.control_noise_profile_id,
        "action_repeat_affected_actor": "ego_selected_policy_action",
        "compute": row.compute,
        "compute_label": _compute_name(row),
        "row_note": row.row_note,
        "deployed_app_submission": {
            "app_name": APP_NAME,
            "train_function": _train_function_name(row.compute),
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
            "spawn_order": ["poller", "train"],
        },
        "train_kwargs": train_kwargs,
        "poller_kwargs": poller_kwargs,
        "flags": {
            "max_env_step": row.max_env_step,
            "max_train_iter": row.max_train_iter,
            "save_ckpt_after_iter": row.save_ckpt_after_iter,
            "collector_env_num": row.collector_env_num,
            "evaluator_env_num": row.evaluator_env_num,
            "n_evaluator_episode": row.n_evaluator_episode,
            "n_episode": row.n_episode,
            "source_max_steps": SOURCE_MAX_STEPS,
            "batch_size": row.batch_size,
            "num_simulations": row.num_simulations,
            "lightzero_eval_freq": row.lightzero_eval_freq,
            "env_manager_type": row.env_manager_type,
            "background_eval_enabled": row.background_eval_enabled,
            "background_eval_launch_kind": row.background_eval_launch_kind,
            "background_eval_compute": row.background_eval_compute,
            "background_eval_id_prefix": row.background_eval_id_prefix,
            "background_eval_seed_count": row.background_eval_seed_count,
            "background_eval_seed_rng_seed": row.eval_seed,
            "background_eval_max_steps": row.background_eval_max_steps,
            "background_eval_num_simulations": row.background_eval_num_simulations,
            "background_eval_batch_size": row.background_eval_batch_size,
            "background_eval_poll_interval_sec": row.background_eval_poll_interval_sec,
            "background_eval_poll_stable_polls": row.background_eval_poll_stable_polls,
            "background_eval_poller_max_runtime_sec": (
                row.background_eval_poller_max_runtime_sec
            ),
            "background_eval_poller_idle_after_done_sec": (
                row.background_eval_poller_idle_after_done_sec
            ),
            "background_gif_enabled": row.background_gif_enabled,
            "background_gif_seed_offset": row.background_gif_seed_offset,
            "background_gif_max_steps": row.background_gif_max_steps,
            "background_gif_frame_stride": row.background_gif_frame_stride,
            "background_gif_fps": row.background_gif_fps,
            "background_gif_scale": row.background_gif_scale,
            "background_gif_frame_size": row.background_gif_frame_size,
            "background_gif_collect_temperature": row.background_gif_collect_temperature,
            "background_gif_collect_epsilon": row.background_gif_collect_epsilon,
        },
        "artifact_refs": {
            "run_manifest": _ref("training", TASK_ID, run_id, "run.json"),
            "attempt_manifest": _ref(
                "training", TASK_ID, run_id, "attempts", attempt_id, "attempt.json"
            ),
            "latest_attempt": _ref("training", TASK_ID, run_id, "latest_attempt.json"),
            "summary": _ref(train_ref, "summary.json"),
            "action_observability": _ref(train_ref, "action_observability.json"),
            "checkpoint_root": _ref("training", TASK_ID, run_id, "checkpoints", "lightzero"),
            "background_eval_status": _ref(train_ref, "checkpoint_eval_poller.json"),
            "background_gif_status": _ref(train_ref, "background_gif_jobs.json"),
        },
        "command": command,
        "command_text": command_text,
        "review_command_only": True,
    }


def _validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
    run_ids = [row["run_id"] for row in rows]
    attempt_ids = [row["attempt_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("generated duplicate run_id values")
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("generated duplicate attempt_id values")
    for row in rows:
        command_text = str(row["command_text"])
        if FORBIDDEN_MODE in command_text:
            raise ValueError("refusing manifest containing stale two-seat-selfplay")
        if row["mode"] != MODE_TRAIN or " --mode train " not in f" {command_text} ":
            raise ValueError(f"row {row['row_id']} is not stock --mode train")
        if row["env_variant"] != ENV_SOURCE_STATE_FIXED_OPPONENT:
            raise ValueError(f"row {row['row_id']} is not source-state fixed-opponent")
        if row["reward_variant"] != REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
            raise ValueError(f"row {row['row_id']} has unexpected reward variant")
        if row["opponent_runtime_mode"] not in {
            OPPONENT_RUNTIME_BLANK_CANVAS_NOOP,
            OPPONENT_RUNTIME_NORMAL,
        }:
            raise ValueError(f"row {row['row_id']} has unexpected opponent runtime")
        if row["opponent_policy_kind"] == OPPONENT_POLICY_FROZEN_LIGHTZERO_CHECKPOINT:
            checkpoint_ref = row.get("opponent_checkpoint_ref")
            if not checkpoint_ref:
                raise ValueError(f"row {row['row_id']} requires a checkpoint ref")
            if "latest" in str(checkpoint_ref) or "ckpt_best" in str(checkpoint_ref):
                raise ValueError(f"row {row['row_id']} refuses mutable checkpoint ref")
        if row["source_max_steps"] != SOURCE_MAX_STEPS:
            raise ValueError(f"row {row['row_id']} has unexpected source_max_steps")
        if not row["flags"]["background_eval_enabled"]:
            raise ValueError(f"row {row['row_id']} must keep background eval enabled")
        if not row["flags"]["background_gif_enabled"]:
            raise ValueError(f"row {row['row_id']} must keep background GIF enabled")

    pair_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pair_groups.setdefault(str(row["logical_pair_id"]), []).append(row)
    for logical_pair_id, group in pair_groups.items():
        roles = {row["render_pair_role"] for row in group}
        if roles != {"fast", "browser"} or len(group) != 2:
            raise ValueError(f"logical_pair_id {logical_pair_id!r} is not a render twin")
        comparable_fields = (
            "block_id",
            "hypothesis_id",
            "copy_id",
            "training_seed",
            "reset_seed",
            "opponent_policy_seed",
            "opponent_behavior_seed",
            "eval_seed",
            "stochasticity_profile_id",
            "policy_action_repeat_min",
            "policy_action_repeat_max",
            "policy_action_repeat_extra_probability",
        )
        first = group[0]
        for row in group[1:]:
            for field in comparable_fields:
                if row[field] != first[field]:
                    raise ValueError(
                        f"logical_pair_id {logical_pair_id!r} mismatches field {field}"
                    )


def _gated_survival_only_ablation_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for copy_id in ("c01", "c02"):
        for render_role, render_mode in RENDER_PAIRS:
            specs.append(
                {
                    "schema_id": "curvyzero_curvytron_survivaldiag_gated_row_spec/v0",
                    "status": "gated_not_commanded",
                    "block_id": "b02_survival_only_ablation_gated",
                    "stage_index": 2,
                    "hypothesis_id": "h02_survival_only_ablation_if_stock_lane_exists",
                    "label": (
                        f"survonly-blanknoop-{_render_tag(render_mode)}-"
                        f"{STOCHASTICITY_PROFILES['medium'].tag}-{copy_id}"
                    ),
                    "reward_variant": REWARD_SURVIVAL_ONLY_GATED,
                    "intended_reward_weights": {
                        "reward_survival_weight": 1.0,
                        "reward_bonus_weight": 0.0,
                        "reward_outcome_weight": 0.0,
                    },
                    "opponent_runtime_mode": OPPONENT_RUNTIME_BLANK_CANVAS_NOOP,
                    "source_max_steps": SOURCE_MAX_STEPS,
                    "render_pair_role": render_role,
                    "source_state_trail_render_mode": render_mode,
                    "copy_id": copy_id,
                    "command_omitted": True,
                    "gate": (
                        "stock source_state_fixed_opponent currently does not accept "
                        "a literal survival_only reward variant; add only after "
                        "first-class trainer wiring and canary"
                    ),
                }
            )
    return specs


def _gated_ancestor_checkpoint_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    refs = (
        ("old", args.old_opponent_checkpoint_ref, "ancestor_old_iteration_0"),
        ("mid", args.mid_opponent_checkpoint_ref, "ancestor_mid_iteration_16"),
        ("recent", args.recent_opponent_checkpoint_ref, "ancestor_recent_iteration_32"),
    )
    specs: list[dict[str, Any]] = []
    for tag, checkpoint_ref, snapshot_ref in refs:
        for render_role, render_mode in RENDER_PAIRS:
            specs.append(
                {
                    "schema_id": "curvyzero_curvytron_survivaldiag_gated_row_spec/v0",
                    "status": "gated_not_commanded",
                    "block_id": "b05_ancestor_checkpoint_sentinels_gated",
                    "stage_index": 5,
                    "hypothesis_id": "h05_minimal_ancestor_checkpoint_controls",
                    "label": (
                        f"survbonusnoout-ancestor-{_render_tag(render_mode)}-"
                        f"{STOCHASTICITY_PROFILES['medium'].tag}-c01-{tag}"
                    ),
                    "reward_variant": REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
                    "opponent_policy_kind": OPPONENT_POLICY_FROZEN_LIGHTZERO_CHECKPOINT,
                    "opponent_runtime_mode": OPPONENT_RUNTIME_NORMAL,
                    "opponent_death_mode": OPPONENT_DEATH_NORMAL,
                    "opponent_checkpoint_ref": checkpoint_ref,
                    "opponent_snapshot_ref": f"curvytron_survivaldiag_{snapshot_ref}",
                    "source_max_steps": SOURCE_MAX_STEPS,
                    "render_pair_role": render_role,
                    "source_state_trail_render_mode": render_mode,
                    "copy_id": "c01",
                    "command_omitted": True,
                    "gate": (
                        "ancestor checkpoint controls need their own exact-lane e2e "
                        "canary and immutable identity review before joining the "
                        "first executable survivaldiag wave"
                    ),
                }
            )
    return specs


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    matrix_name = _safe_id(args.matrix_name, label="matrix_name")
    run_prefix = _safe_id(args.run_prefix, label="run_prefix")
    attempt_prefix = _safe_id(args.attempt_prefix, label="attempt_prefix")
    rows = _rows(args)
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
    _validate_manifest_rows(manifest_rows)
    block_descriptions = {
        block.block_id: block.description
        for block in BLOCK_SPECS
    } | {
        "b03_passive_immortal_dirty_control": (
            "Tiny passive-immortal dirty control; not promoted to the main claim."
        ),
        "b04_compute_sentinel_sim16": (
            "Minimal sim16 compute sentinel on one matched blank-canvas cell."
        ),
        "b10_blank_canvas_scale_all": (
            "Clean v1c repeat rows for blank canvas across all stochasticity levels."
        ),
        "b11_blank_canvas_medium_high_scale": (
            "Extra v1c repeat rows on medium/high action-repeat pressure."
        ),
        "b12_passive_immortal_dirty_extension": (
            "Small v1c passive-immortal dirty-control extension."
        ),
        "b13_compute_sentinel_extension": (
            "Small v1c compute sentinel extension: sim16, collector64, batch64."
        ),
        "b20_blank_canvas_all_level_repeats": (
            "Large-ready blank-canvas repeats across all action-repeat levels."
        ),
        "b21_blank_canvas_medium_high_extra": (
            "Large-ready extra repeats on medium/high action-repeat levels."
        ),
        "b22_passive_immortal_dirty_controls": (
            "Large-ready passive immortal dirty controls."
        ),
        "b23_blank_canvas_compute_sentinels": (
            "Large-ready blank-canvas search16, collector64, and batch64 sentinels."
        ),
    }
    block_ids = sorted({str(row["block_id"]) for row in manifest_rows})
    blocks = []
    for block_id in block_ids:
        group = [row for row in manifest_rows if row["block_id"] == block_id]
        blocks.append(
            {
                "block_id": block_id,
                "stage_index": min(int(row["stage_index"]) for row in group),
                "hypothesis_ids": sorted({str(row["hypothesis_id"]) for row in group}),
                "description": block_descriptions.get(block_id, ""),
                "row_count": len(group),
                "row_kinds": sorted({str(row["row_kind"]) for row in group}),
                "stochasticity_profile_ids": sorted(
                    {str(row["stochasticity_profile_id"]) for row in group}
                ),
                "copy_ids": sorted({str(row["copy_id"]) for row in group}),
                "render_pair_roles": ["fast", "browser"],
            }
        )
    gated_specs = _gated_survival_only_ablation_specs() + _gated_ancestor_checkpoint_specs(args)
    if args.matrix_profile == MATRIX_PROFILE_LARGE_READY:
        default_rows = "300 executable large-ready rows plus gated specs"
        executable_row_shape = {
            "blank_canvas_all_level_repeats": (
                "160 rows: 2 renders x 4 stochasticity levels x 20 copies"
            ),
            "blank_canvas_medium_high_extra": (
                "40 rows: 2 renders x medium/high stochasticity x 10 extra copies"
            ),
            "passive_immortal_dirty_controls": (
                "40 rows: 2 renders x 4 stochasticity levels x 5 dirty-control copies"
            ),
            "blank_canvas_compute_sentinels": (
                "60 rows: 2 renders x medium/high stochasticity x 5 copies x "
                "search16/collector64/batch64"
            ),
        }
    elif args.matrix_profile == MATRIX_PROFILE_BLANK_REPEAT_EXPANSION:
        default_rows = "50 executable v1c blank-repeat expansion rows plus gated specs"
        executable_row_shape = {
            "blank_canvas_scale_all": (
                "32 rows: 2 renders x 4 stochasticity levels x 4 new copies"
            ),
            "blank_canvas_medium_high_scale": (
                "8 rows: 2 renders x medium/high stochasticity x 2 new copies"
            ),
            "passive_immortal_dirty_extension": (
                "4 rows: 2 renders x low/high stochasticity x 1 dirty-control copy"
            ),
            "compute_sentinel_extension": (
                "6 rows: matched render pairs for sim16, collector64, and batch64"
            ),
        }
    else:
        default_rows = "50 executable review rows plus gated ablation/control specs"
        executable_row_shape = {
            "exact_preflight": "4 rows: 2 renders x medium stochasticity x 2 copies",
            "blank_canvas_core": "32 rows: 2 renders x 4 stochasticity levels x 4 copies",
            "blank_canvas_extra_repeats": (
                "8 rows: 2 renders x medium/high stochasticity x 2 extra copies"
            ),
            "passive_immortal_dirty_control": (
                "4 rows: 2 renders x medium stochasticity x 2 copies"
            ),
            "compute_sentinel": "2 rows: sim16 matched render pair",
        }
    return {
        "schema_id": SCHEMA_ID,
        "generated_at": _utc_timestamp(),
        "dry_run_only": True,
        "launches_modal": False,
        "current_launch_approved": False,
        "matrix_name": matrix_name,
        "matrix_profile": args.matrix_profile,
        "task_id": TASK_ID,
        "canonical_launcher": MODULE,
        "run_prefix": run_prefix,
        "attempt_prefix": attempt_prefix,
        "row_count": len(manifest_rows),
        "block_count": len(blocks),
        "logical_pair_count": len({row["logical_pair_id"] for row in manifest_rows}),
        "shape": {
            "staged_not_cartesian": True,
            "default_rows": default_rows,
            "executable_row_shape": executable_row_shape,
            "gated_shape": {
                "survival_only_ablation": (
                    "4 specs: 2 renders x 1 stochasticity x 2 copies, not commanded"
                ),
                "ancestor_checkpoint_controls": (
                    "6 specs: old/mid/recent x 2 renders, not commanded"
                ),
                "scripted_random_frozen_expansions": (
                    "omitted until first-class wiring, immutable identity, and e2e canary"
                ),
            },
            "render_pairing": "every logical setting has fast/browser twins",
            "default_source_max_steps": SOURCE_MAX_STEPS,
            "reward_variant": REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "main_opponent_runtime_mode": OPPONENT_RUNTIME_BLANK_CANVAS_NOOP,
        },
        "rich_readout_expectations": {
            "survival": ["latest", "best", "slope", "late_bloom", "peak_then_crash"],
            "reward": ["trainer_reward", "reward_components", "bonus_pickup_count"],
            "terminal": ["terminal_cause_histogram", "death_cause_histogram"],
            "actions": [
                "action_histogram",
                "action_entropy",
                "straight_left_right_rates",
                "repeated_action_collapse",
            ],
            "artifacts": ["checkpoint_eval_health", "background_eval_health", "gif_health"],
            "secondary_only": ["outcome_win_loss_draw"],
        },
        "schema_contract": {
            "seed_fields": [
                "training_seed",
                "reset_seed",
                "opponent_policy_seed",
                "opponent_behavior_seed",
                "eval_seed",
                "copy_id",
            ],
            "render_pair_fields": ["logical_pair_id", "render_pair_role"],
            "reward_fields": [
                "reward_survival_weight",
                "reward_bonus_weight",
                "reward_outcome_weight",
                "reward_components",
            ],
            "opponent_contract_fields": [
                "opponent_runtime_mode",
                "opponent_trail_mode",
                "opponent_death_mode",
                "opponent_collision_effect",
                "opponent_visibility_mode",
            ],
            "stochasticity_fields": [
                "ego_action_straight_override_probability",
                "policy_action_repeat_min",
                "policy_action_repeat_max",
                "policy_action_repeat_extra_probability",
                "control_noise_profile_id",
            ],
        },
        "guards": {
            "commands_reviewed_only": True,
            "deployed_app_grouped_submitter_required": True,
            "deployed_app_name": APP_NAME,
            "no_modal_run_per_row_for_large_batch": True,
            "mode_required": MODE_TRAIN,
            "forbidden_mode": FORBIDDEN_MODE,
            "stock_train_muzero_only": True,
            "max_train_iter_semantics": (
                "not_strict_for_tiny_canaries_checked_after_collect_update_block"
            ),
            "tiny_canary_stop_required": "--stop-after-learner-train-calls",
            "forced_stop_status_warning": (
                "heartbeat_or_poller_running_may_be_stale_verify_modal_app_state"
            ),
            "allowed_env_variants": [ENV_SOURCE_STATE_FIXED_OPPONENT],
            "allowed_reward_variants": [REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME],
            "allowed_opponent_runtime_modes": [
                OPPONENT_RUNTIME_BLANK_CANVAS_NOOP,
                OPPONENT_RUNTIME_NORMAL,
            ],
            "no_two_seat_mode": True,
            "background_eval_required": True,
            "background_gif_required": True,
            "source_max_steps_required": SOURCE_MAX_STEPS,
            "launch_blocked_by_default": True,
            "gated_expansions": [
                "survival_only_ablation_until_stock_reward_variant_exists",
                "scripted_opponents_until_remote_e2e_canaried",
                "ancestor_checkpoint_controls_until_exact_lane_canaried",
                "random_or_broad_frozen_expansions_until_immutable_and_canaried",
            ],
        },
        "blocks": blocks,
        "gated_specs": gated_specs,
        "rows": manifest_rows,
    }


def _write_outputs(manifest: dict[str, Any], *, output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / str(manifest["matrix_name"])
    json_path = base.with_suffix(".json")
    jsonl_path = base.with_suffix(".rows.jsonl")
    commands_path = base.with_suffix(".review_commands.txt")
    json_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    commands_path.write_text(
        "# Dry-run review commands only. Do not execute without a fresh launch gate.\n"
        + "\n".join(row["command_text"] for row in manifest["rows"])
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_json": str(json_path),
        "rows_jsonl": str(jsonl_path),
        "review_commands_txt": str(commands_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dry-run-only CurvyTron survivaldiag manifest. This writes "
            "review artifacts only and never launches Modal."
        )
    )
    parser.add_argument("--matrix-name", default=DEFAULT_MATRIX_NAME)
    parser.add_argument(
        "--matrix-profile",
        choices=MATRIX_PROFILE_CHOICES,
        default=MATRIX_PROFILE_LARGE_READY,
        help=(
            "Which staged row set to emit. large_ready emits the current "
            "300-row batch; first_wave and blank_repeat_expansion are kept "
            "for historical review."
        ),
    )
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--attempt-prefix", default=DEFAULT_ATTEMPT_PREFIX)
    parser.add_argument("--compute", default="gpu-l4-t4-cpu40")
    parser.add_argument("--max-train-iter", type=int, default=300_000)
    parser.add_argument("--max-env-step", type=int, default=30_000_000)
    parser.add_argument("--save-ckpt-after-iter", type=int, default=15_000)
    parser.add_argument("--collector-env-num", type=int, default=32)
    parser.add_argument("--evaluator-env-num", type=int, default=1)
    parser.add_argument("--n-evaluator-episode", type=int, default=1)
    parser.add_argument("--n-episode", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-simulations", type=int, default=8)
    parser.add_argument("--env-manager-type", default="subprocess")
    parser.add_argument("--background-eval-launch-kind", default="poller")
    parser.add_argument("--background-eval-compute", default="cpu")
    parser.add_argument("--background-eval-id-prefix", default="live_checkpoint")
    parser.add_argument("--background-eval-seed-count", type=int, default=8)
    parser.add_argument("--background-eval-max-steps", type=int, default=SOURCE_MAX_STEPS)
    parser.add_argument("--background-eval-num-simulations", type=int, default=8)
    parser.add_argument("--background-eval-batch-size", type=int, default=64)
    parser.add_argument("--background-eval-poll-interval-sec", type=float, default=10.0)
    parser.add_argument("--background-eval-poll-stable-polls", type=int, default=1)
    parser.add_argument(
        "--background-eval-poller-max-runtime-sec",
        type=float,
        default=18 * 60 * 60,
    )
    parser.add_argument(
        "--background-eval-poller-idle-after-done-sec",
        type=float,
        default=60.0,
    )
    parser.add_argument("--background-gif-seed-offset", type=int, default=10_000)
    parser.add_argument("--background-gif-max-steps", type=int, default=4096)
    parser.add_argument("--background-gif-frame-stride", type=int, default=4)
    parser.add_argument("--background-gif-fps", type=float, default=8.0)
    parser.add_argument("--background-gif-scale", type=int, default=4)
    parser.add_argument("--background-gif-frame-size", type=int, default=128)
    parser.add_argument("--background-gif-collect-temperature", type=float, default=1.0)
    parser.add_argument("--background-gif-collect-epsilon", type=float, default=0.25)
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
        "--no-detach",
        action="store_true",
        help="Omit --detach from generated review command text.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print the manifest and do not write local artifact files.",
    )
    return parser.parse_args(argv)


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
                "current_launch_approved": False,
                "matrix_name": manifest["matrix_name"],
                "row_count": manifest["row_count"],
                "block_count": manifest["block_count"],
                "logical_pair_count": manifest["logical_pair_count"],
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
