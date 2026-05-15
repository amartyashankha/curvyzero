#!/usr/bin/env python3
"""Build CurvyTron opponent-mixture training manifests.

The rows produced here use the same deployed Modal trainer app as the current
survival diagnostic batch. This file only writes review artifacts; launch still
goes through ``submit_curvytron_survivaldiag_manifest.py``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from curvyzero.contracts.curvytron import (
    CURVYTRON_DECISION_MS as DECISION_MS,
    CURVYTRON_POLICY_BONUS_RENDER_MODE as BONUS_RENDER_SIMPLE_SYMBOLS,
    CURVYTRON_POLICY_TRAIL_RENDER_MODE as RENDER_BROWSER,
    CURVYTRON_SAVE_CKPT_AFTER_ITER as SAVE_CKPT_AFTER_ITER,
    CURVYTRON_SOURCE_MAX_STEPS as SOURCE_MAX_STEPS,
    CURVYTRON_TRAINING_TASK_ID,
    DEFAULT_CURVYTRON_TRAIN_APP_NAME,
    LEARNER_SEAT_MODE_RANDOM_PER_EPISODE,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME as REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_MIXTURE_SCHEMA_ID,
    parse_opponent_mixture_spec,
)


MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
APP_NAME = DEFAULT_CURVYTRON_TRAIN_APP_NAME
TASK_ID = CURVYTRON_TRAINING_TASK_ID
SCHEMA_ID = "curvyzero_curvytron_opponent_mixture_manifest/v0"
ROW_SCHEMA_ID = "curvyzero_curvytron_opponent_mixture_manifest_row/v0"

MODE_TRAIN = "train"
ENV_SOURCE_STATE_FIXED_OPPONENT = "source_state_fixed_opponent"
RENDER_FAST = RENDER_BROWSER
DEFAULT_MATRIX_NAME = "curvy-mix2-20260513a"
DEFAULT_RUN_PREFIX = "curvy-mix2"
DEFAULT_ATTEMPT_PREFIX = "try-mix2"
DEFAULT_NEXT_WAVE_MATRIX_NAME = "curvy-mix3-nextwave-20260513a"
DEFAULT_NEXT_WAVE_RUN_PREFIX = "curvy-mix3nw"
DEFAULT_NEXT_WAVE_ATTEMPT_PREFIX = "try-mix3nw"
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_opponent_mixture_manifests")
MAIN_CORE_COPY_COUNT = 3
MAIN_SENTINEL_COPY_COUNT = 1
CONTROL_COPY_COUNT = 1
NEXT_WAVE_MAIN_COPY_COUNT = 5
NEXT_WAVE_CONTROL_COPY_COUNT = 2
NEXT_WAVE_COMPUTE_COPY_COUNT = 1
BATCH_SCOPE_FULL = "full"
BATCH_SCOPE_CORE = "core"
BATCH_SCOPE_CHOICES = (BATCH_SCOPE_FULL, BATCH_SCOPE_CORE)

DEFAULT_RECENT_OPPONENT_CHECKPOINT_REF = (
    "training/lightzero-curvytron-visual-survival/"
    "survivaldiag-v1b-20260513h-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40/"
    "attempts/sdv1bh-a-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40/"
    "train/lightzero_exp/ckpt/iteration_20000.pth.tar"
)
DEFAULT_MID_OPPONENT_CHECKPOINT_REF = (
    "training/lightzero-curvytron-visual-survival/"
    "survivaldiag-v1b-20260513h-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40/"
    "attempts/sdv1bh-a-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40/"
    "train/lightzero_exp/ckpt/iteration_10000.pth.tar"
)
DEFAULT_OLD_OPPONENT_CHECKPOINT_REF = (
    "training/lightzero-curvytron-visual-survival/"
    "survivaldiag-v1b-20260513h-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40/"
    "attempts/sdv1bh-a-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40/"
    "train/lightzero_exp/ckpt/iteration_0.pth.tar"
)


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    weights: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class BaseProfile:
    token: str
    profile_kind: str
    render_token: str
    render_mode: str
    num_simulations: int
    collector_env_num: int
    n_episode: int
    batch_size: int
    repeat_token: str
    policy_action_repeat_min: int
    policy_action_repeat_max: int
    policy_action_repeat_extra_probability: float
    control_noise_profile_id: str
    save_ckpt_after_iter: int = SAVE_CKPT_AFTER_ITER


MAIN_RECIPES: tuple[Recipe, ...] = (
    Recipe("r50-blank50", (("recent", 50), ("blank", 50))),
    Recipe("r50-mid50", (("recent", 50), ("mid", 50))),
    Recipe("r50-old50", (("recent", 50), ("old", 50))),
    Recipe("r50-scr50", (("recent", 50), ("scripted", 50))),
    Recipe("r50-pass50", (("recent", 50), ("passive", 50))),
    Recipe("r50-blank25-scr25", (("recent", 50), ("blank", 25), ("scripted", 25))),
    Recipe("r50-mid25-old25", (("recent", 50), ("mid", 25), ("old", 25))),
    Recipe(
        "r50-blank20-mid15-scr15",
        (("recent", 50), ("blank", 20), ("mid", 15), ("scripted", 15)),
    ),
)

CONTROL_RECIPES: tuple[Recipe, ...] = (
    Recipe("recent100", (("recent", 100),)),
    Recipe("mid100", (("mid", 100),)),
    Recipe("old100", (("old", 100),)),
    Recipe("blank100", (("blank", 100),)),
    Recipe("scr100", (("scripted", 100),)),
    Recipe("pass100", (("passive", 100),)),
)

RECIPES: tuple[Recipe, ...] = (*MAIN_RECIPES, *CONTROL_RECIPES)
MAIN_RECIPE_IDS = frozenset(recipe.recipe_id for recipe in MAIN_RECIPES)
CONTROL_RECIPE_IDS = frozenset(recipe.recipe_id for recipe in CONTROL_RECIPES)

NEXT_WAVE_MAIN_RECIPES: tuple[Recipe, ...] = (
    Recipe("r25-blank75", (("recent", 25), ("blank", 75))),
    Recipe("r50-blank50", (("recent", 50), ("blank", 50))),
    Recipe("r75-blank25", (("recent", 75), ("blank", 25))),
    Recipe("r50-scr50", (("recent", 50), ("scripted", 50))),
    Recipe("r50-mid25-old25", (("recent", 50), ("mid", 25), ("old", 25))),
    Recipe(
        "r40-blank20-mid20-scr20",
        (("recent", 40), ("blank", 20), ("mid", 20), ("scripted", 20)),
    ),
)

NEXT_WAVE_CONTROL_RECIPES: tuple[Recipe, ...] = (
    Recipe("recent100", (("recent", 100),)),
    Recipe("blank100", (("blank", 100),)),
    Recipe("scr100", (("scripted", 100),)),
    Recipe("mid100", (("mid", 100),)),
    Recipe("old100", (("old", 100),)),
)

NEXT_WAVE_RECIPES: tuple[Recipe, ...] = (
    *NEXT_WAVE_MAIN_RECIPES,
    *NEXT_WAVE_CONTROL_RECIPES,
)
NEXT_WAVE_MAIN_RECIPE_IDS = frozenset(recipe.recipe_id for recipe in NEXT_WAVE_MAIN_RECIPES)
NEXT_WAVE_CONTROL_RECIPE_IDS = frozenset(recipe.recipe_id for recipe in NEXT_WAVE_CONTROL_RECIPES)
NEXT_WAVE_COMPUTE_RECIPE_IDS: frozenset[str] = frozenset(
    (
        "r50-blank50",
        "r75-blank25",
        "r50-scr50",
        "r50-mid25-old25",
        "r40-blank20-mid20-scr20",
    )
)


def _base_profile(
    *,
    render_token: str,
    num_simulations: int,
    collector_env_num: int,
    batch_size: int,
    repeat_token: str,
    profile_kind: str,
) -> BaseProfile:
    render_modes = {"rf": RENDER_FAST, "rb": RENDER_BROWSER}
    repeat_settings = {
        "rep0": (1, 1, 0.0, "none"),
        "repM": (1, 3, 0.20, "policy_action_repeat_medium"),
        "repH": (1, 3, 0.35, "policy_action_repeat_high"),
    }
    repeat_min, repeat_max, repeat_extra, control_noise = repeat_settings[repeat_token]
    token = (
        f"{render_token}-s{num_simulations}-c{collector_env_num}-l{batch_size}-{repeat_token}-k10"
    )
    return BaseProfile(
        token=token,
        profile_kind=profile_kind,
        render_token=render_token,
        render_mode=render_modes[render_token],
        num_simulations=num_simulations,
        collector_env_num=collector_env_num,
        n_episode=collector_env_num,
        batch_size=batch_size,
        repeat_token=repeat_token,
        policy_action_repeat_min=repeat_min,
        policy_action_repeat_max=repeat_max,
        policy_action_repeat_extra_probability=repeat_extra,
        control_noise_profile_id=control_noise,
    )


CORE_BASE_PROFILES: tuple[BaseProfile, ...] = tuple(
    _base_profile(
        render_token=render_token,
        num_simulations=8,
        collector_env_num=32,
        batch_size=32,
        repeat_token=repeat_token,
        profile_kind="core",
    )
    for render_token in ("rf", "rb")
    for repeat_token in ("rep0", "repM", "repH")
)

SENTINEL_BASE_PROFILES: tuple[BaseProfile, ...] = tuple(
    _base_profile(
        render_token=render_token,
        num_simulations=num_simulations,
        collector_env_num=collector_env_num,
        batch_size=batch_size,
        repeat_token="repM",
        profile_kind="sentinel",
    )
    for render_token in ("rf", "rb")
    for num_simulations, collector_env_num, batch_size in (
        (16, 32, 32),
        (8, 64, 32),
        (8, 32, 64),
    )
)

BASE_PROFILES: tuple[BaseProfile, ...] = (*CORE_BASE_PROFILES, *SENTINEL_BASE_PROFILES)
BASE_PROFILES_BY_TOKEN: dict[str, BaseProfile] = {
    profile.token: profile for profile in BASE_PROFILES
}

NEXT_WAVE_COMPUTE_PROBE_BASE_PROFILES: tuple[BaseProfile, ...] = tuple(
    _base_profile(
        render_token=render_token,
        num_simulations=num_simulations,
        collector_env_num=collector_env_num,
        batch_size=batch_size,
        repeat_token=repeat_token,
        profile_kind="compute_probe",
    )
    for render_token in ("rf", "rb")
    for repeat_token in ("repM", "repH")
    for num_simulations, collector_env_num, batch_size in (
        (16, 32, 32),
        (8, 64, 32),
        (8, 32, 64),
    )
)

ALL_BASE_PROFILES_BY_TOKEN_KIND: dict[tuple[str, str], BaseProfile] = {
    (profile.token, profile.profile_kind): profile
    for profile in (*BASE_PROFILES, *NEXT_WAVE_COMPUTE_PROBE_BASE_PROFILES)
}
RECIPES_BY_ID: dict[str, Recipe] = {recipe.recipe_id: recipe for recipe in RECIPES}
RECIPE_INDEX_BY_ID: dict[str, int] = {
    recipe.recipe_id: index for index, recipe in enumerate(RECIPES, start=1)
}
NEXT_WAVE_RECIPES_BY_ID: dict[str, Recipe] = {
    recipe.recipe_id: recipe for recipe in NEXT_WAVE_RECIPES
}
NEXT_WAVE_RECIPE_INDEX_BY_ID: dict[str, int] = {
    recipe.recipe_id: index for index, recipe in enumerate(NEXT_WAVE_RECIPES, start=1)
}

CANARY_PLAN: tuple[tuple[str, str], ...] = (
    ("r50-blank25-scr25", "rf-s8-c32-l32-repM-k10"),
    ("r50-blank25-scr25", "rb-s8-c32-l32-repM-k10"),
    ("r50-mid25-old25", "rf-s8-c32-l32-repM-k10"),
    ("r50-mid25-old25", "rb-s8-c32-l32-repM-k10"),
    ("r50-pass50", "rf-s8-c32-l32-repH-k10"),
    ("r50-scr50", "rb-s8-c32-l32-repH-k10"),
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
            f"{label} must be 1-{max_length} chars of letters, numbers, dash, underscore, or dot"
        )
    return raw


def _ref(*parts: str) -> str:
    return "/".join(parts)


def _seed(*, recipe_index: int, copy_index: int, profile: str) -> int:
    profile_offset = {
        "batch": 2_100_000,
        "canary": 2_190_000,
        "next-wave": 2_300_000,
    }[profile]
    return profile_offset + recipe_index * 1_000 + copy_index * 10 + 1


def _derived_seed(label: str, base_seed: int) -> int:
    payload = f"{label}:{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _checkpoint_refs(args: argparse.Namespace) -> dict[str, str]:
    refs = {
        "recent": str(args.recent_opponent_checkpoint_ref),
        "mid": str(args.mid_opponent_checkpoint_ref),
        "old": str(args.old_opponent_checkpoint_ref),
    }
    for label, ref in refs.items():
        if "latest" in ref or "ckpt_best" in ref:
            raise ValueError(f"{label} checkpoint ref is mutable: {ref}")
        if not re.fullmatch(r"iteration_\d+\.pth\.tar", Path(ref).name):
            raise ValueError(f"{label} checkpoint ref must end in iteration_N.pth.tar: {ref}")
    return refs


def _component_entry(
    component: str,
    *,
    weight: int,
    refs: dict[str, str],
    seed: int,
) -> dict[str, Any]:
    if component == "blank":
        return {
            "name": "blank",
            "age_label": "blank",
            "weight": weight,
            "opponent_policy_kind": "fixed_straight",
            "opponent_runtime_mode": "blank_canvas_noop",
            "opponent_immortal": True,
        }
    if component == "passive":
        return {
            "name": "passive",
            "age_label": "scripted_static",
            "weight": weight,
            "opponent_policy_kind": "fixed_straight",
            "opponent_runtime_mode": "normal",
            "opponent_immortal": True,
        }
    if component == "scripted":
        return {
            "name": "scripted",
            "age_label": "scripted_wall_avoidant",
            "weight": weight,
            "opponent_policy_kind": "proactive_wall_avoidant",
            "opponent_runtime_mode": "normal",
            "opponent_immortal": True,
            "opponent_policy_seed": _derived_seed("scripted_opponent", seed),
            "opponent_wall_avoidant_safe_margin": 96.0,
        }
    if component in {"recent", "mid", "old"}:
        return {
            "name": component,
            "age_label": component,
            "weight": weight,
            "opponent_policy_kind": "frozen_lightzero_checkpoint",
            "opponent_runtime_mode": "normal",
            "opponent_immortal": True,
            "opponent_checkpoint_ref": refs[component],
            "opponent_snapshot_ref": f"curvy_mix_{component}_{Path(refs[component]).stem}",
            "opponent_policy_seed": _derived_seed(f"{component}_opponent", seed),
        }
    raise ValueError(f"unknown mixture component {component!r}")


def _mixture_spec(recipe: Recipe, *, refs: dict[str, str], seed: int) -> dict[str, Any]:
    spec = {
        "schema_id": OPPONENT_MIXTURE_SCHEMA_ID,
        "selection_unit": "episode_reset",
        "seed": _derived_seed("mixture", seed),
        "entries": [
            _component_entry(component, weight=weight, refs=refs, seed=seed)
            for component, weight in recipe.weights
        ],
    }
    parsed = parse_opponent_mixture_spec(spec)
    if parsed is None:
        raise ValueError(f"recipe {recipe.recipe_id} produced an empty mixture")
    return _public_mixture_spec(parsed)


def _public_mixture_spec(parsed: dict[str, Any]) -> dict[str, Any]:
    public = copy.deepcopy(parsed)
    for entry in public.get("entries", []):
        if isinstance(entry, dict):
            entry.pop("opponent_death_mode", None)
    return public


def _train_function_name(compute: str) -> str:
    if compute == "gpu-l4-t4-cpu40":
        return "lightzero_curvytron_visual_survival_gpu_cpu40"
    if compute == "gpu-h100-cpu40":
        return "lightzero_curvytron_visual_survival_h100_cpu40"
    if compute == "gpu-h100x2-cpu40":
        return "lightzero_curvytron_visual_survival_h100x2_cpu40"
    raise ValueError(f"unsupported compute for mixture manifest: {compute!r}")


def _command_for_row(
    *,
    args: argparse.Namespace,
    base: BaseProfile,
    run_id: str,
    attempt_id: str,
    seed: int,
    mixture_spec: dict[str, Any],
) -> list[str]:
    command = [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "--quiet",
    ]
    if not args.no_detach:
        command.append("--detach")
    command.extend(
        [
            "-m",
            MODULE,
            "--mode",
            MODE_TRAIN,
            "--compute",
            args.compute,
            "--seed",
            str(seed),
            "--run-id",
            run_id,
            "--attempt-id",
            attempt_id,
            "--env-variant",
            ENV_SOURCE_STATE_FIXED_OPPONENT,
            "--reward-variant",
            REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "--opponent-policy-kind",
            "fixed_straight",
            "--opponent-runtime-mode",
            "normal",
            "--opponent-death-mode",
            "normal",
            "--opponent-mixture-spec",
            json.dumps(mixture_spec, sort_keys=True, separators=(",", ":")),
            "--source-state-trail-render-mode",
            base.render_mode,
            "--source-state-bonus-render-mode",
            BONUS_RENDER_SIMPLE_SYMBOLS,
            "--ego-action-straight-override-probability",
            "0.0",
            "--control-noise-profile-id",
            base.control_noise_profile_id,
            "--policy-action-repeat-min",
            str(base.policy_action_repeat_min),
            "--policy-action-repeat-max",
            str(base.policy_action_repeat_max),
            "--policy-action-repeat-extra-probability",
            str(base.policy_action_repeat_extra_probability),
            "--max-train-iter",
            str(args.max_train_iter),
            "--max-env-step",
            str(args.max_env_step),
            "--save-ckpt-after-iter",
            str(base.save_ckpt_after_iter),
            "--collector-env-num",
            str(base.collector_env_num),
            "--evaluator-env-num",
            "1",
            "--n-evaluator-episode",
            "1",
            "--n-episode",
            str(base.n_episode),
            "--source-max-steps",
            str(SOURCE_MAX_STEPS),
            "--batch-size",
            str(base.batch_size),
            "--num-simulations",
            str(base.num_simulations),
            "--lightzero-eval-freq",
            "0",
            "--env-manager-type",
            args.env_manager_type,
            "--background-eval-launch-kind",
            "poller",
            "--background-eval-compute",
            "cpu",
            "--background-eval-id-prefix",
            "live_checkpoint",
            "--background-eval-seed-count",
            str(args.background_eval_seed_count),
            "--background-eval-seed-rng-seed",
            str(_derived_seed("eval", seed)),
            "--background-eval-max-steps",
            str(SOURCE_MAX_STEPS),
            "--background-eval-step-detail-limit",
            "4",
            "--background-eval-num-simulations",
            "8",
            "--background-eval-batch-size",
            "64",
            "--background-eval-poll-interval-sec",
            str(args.background_eval_poll_interval_sec),
            "--background-eval-poll-stable-polls",
            "1",
            "--background-eval-poller-max-runtime-sec",
            str(args.background_eval_poller_max_runtime_sec),
            "--background-eval-poller-idle-after-done-sec",
            "60.0",
            "--background-gif-seed-offset",
            "10000",
            "--background-gif-max-steps",
            "4096",
            "--background-gif-frame-stride",
            "4",
            "--background-gif-fps",
            "8.0",
            "--background-gif-scale",
            "4",
            "--background-gif-frame-size",
            "128",
            "--background-gif-collect-temperature",
            "1.0",
            "--background-gif-collect-epsilon",
            "0.25",
            "--output-detail",
            "compact",
        ]
    )
    return command


def _train_kwargs(
    *,
    args: argparse.Namespace,
    base: BaseProfile,
    run_id: str,
    attempt_id: str,
    seed: int,
    mixture_spec: dict[str, Any],
    initial_policy_checkpoint_ref: str,
) -> dict[str, Any]:
    return {
        "mode": MODE_TRAIN,
        "seed": seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "max_env_step": args.max_env_step,
        "max_train_iter": args.max_train_iter,
        "source_max_steps": SOURCE_MAX_STEPS,
        "decision_ms": DECISION_MS,
        "collector_env_num": base.collector_env_num,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": base.n_episode,
        "num_simulations": base.num_simulations,
        "batch_size": base.batch_size,
        "lightzero_eval_freq": 0,
        "skip_lightzero_eval_in_profile": True,
        "profile_cuda_sync_enabled": False,
        "profile_allow_auto_resume": False,
        "profile_volume_commit": False,
        "lightzero_multi_gpu": False,
        "save_ckpt_after_iter": base.save_ckpt_after_iter,
        "commit_on_checkpoint": True,
        "stop_after_learner_train_calls": args.stop_after_learner_train_calls,
        "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        "source_state_trail_render_mode": base.render_mode,
        "source_state_bonus_render_mode": BONUS_RENDER_SIMPLE_SYMBOLS,
        "learner_seat_mode": LEARNER_SEAT_MODE_RANDOM_PER_EPISODE,
        "ego_action_straight_override_probability": 0.0,
        "policy_action_repeat_min": base.policy_action_repeat_min,
        "policy_action_repeat_max": base.policy_action_repeat_max,
        "policy_action_repeat_extra_probability": base.policy_action_repeat_extra_probability,
        "control_noise_profile_id": base.control_noise_profile_id,
        "disable_death_for_profile": False,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "normal",
        "env_telemetry_stride": 1,
        "env_manager_type": args.env_manager_type,
        "opponent_policy_kind": "fixed_straight",
        "opponent_use_cuda": False,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_report_ref": None,
        "opponent_checkpoint_state_key": None,
        "initial_policy_checkpoint_ref": initial_policy_checkpoint_ref,
        "initial_policy_checkpoint_state_key": None,
        "initial_policy_checkpoint_load_mode": "matching_shape",
        "opponent_mixture_spec": mixture_spec,
        "background_eval_enabled": True,
        "background_eval_launch_kind": "poller",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "live_checkpoint",
        "background_eval_seed_count": args.background_eval_seed_count,
        "background_eval_seed_rng_seed": _derived_seed("eval", seed),
        "background_eval_max_steps": SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": 8,
        "background_eval_batch_size": 64,
        "background_gif_enabled": True,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": 4096,
        "background_gif_frame_stride": 4,
        "background_gif_fps": 8.0,
        "background_gif_scale": 4,
        "background_gif_frame_size": 128,
        "background_gif_collect_temperature": 1.0,
        "background_gif_collect_epsilon": 0.25,
    }


def _poller_kwargs(
    *,
    args: argparse.Namespace,
    run_id: str,
    attempt_id: str,
    seed: int,
    mixture_spec: dict[str, Any],
) -> dict[str, Any]:
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
        "seed": seed,
        "source_max_steps": SOURCE_MAX_STEPS,
        "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        "opponent_policy_kind": "fixed_straight",
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
        "opponent_mixture_spec": mixture_spec,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "normal",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "live_checkpoint",
        "background_eval_seed_count": args.background_eval_seed_count,
        "background_eval_seed_rng_seed": _derived_seed("eval", seed),
        "background_eval_max_steps": SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": 8,
        "background_eval_batch_size": 64,
        "background_gif_enabled": True,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": 4096,
        "background_gif_frame_stride": 4,
        "background_gif_fps": 8.0,
        "background_gif_scale": 4,
        "background_gif_frame_size": 128,
        "background_gif_collect_temperature": 1.0,
        "background_gif_collect_epsilon": 0.25,
        "poll_interval_sec": args.background_eval_poll_interval_sec,
        "stable_polls": 1,
        "max_runtime_sec": args.background_eval_poller_max_runtime_sec,
        "idle_after_train_done_sec": 60.0,
    }


def _row(
    *,
    args: argparse.Namespace,
    matrix_name: str,
    run_prefix: str,
    attempt_prefix: str,
    row_number: int,
    recipe: Recipe,
    recipe_index: int,
    copy_index: int,
    refs: dict[str, str],
    base: BaseProfile,
    profile: str,
) -> dict[str, Any]:
    seed = _seed(recipe_index=recipe_index, copy_index=copy_index, profile=profile)
    mixture_spec = _mixture_spec(recipe, refs=refs, seed=seed)
    row_tag = f"r{row_number:03d}"
    run_id = _safe_id(
        f"{run_prefix}-{recipe.recipe_id}-{base.token}-c{copy_index}-s{seed}",
        label="run_id",
    )
    attempt_id = _safe_id(
        f"{attempt_prefix}-{recipe.recipe_id}-{base.token}-c{copy_index}-s{seed}",
        label="attempt_id",
    )
    command = _command_for_row(
        args=args,
        base=base,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        mixture_spec=mixture_spec,
    )
    train_kwargs = _train_kwargs(
        args=args,
        base=base,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        mixture_spec=mixture_spec,
        initial_policy_checkpoint_ref=refs["recent"],
    )
    poller_kwargs = _poller_kwargs(
        args=args,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        mixture_spec=mixture_spec,
    )
    train_ref = _ref("training", TASK_ID, run_id, "attempts", attempt_id, "train")
    if profile == "canary":
        row_kind = "opponent_mixture_canary"
    elif profile == "next-wave":
        row_kind = "opponent_mixture_next_wave_candidate"
    else:
        row_kind = "opponent_mixture"
    return {
        "schema_id": ROW_SCHEMA_ID,
        "matrix_name": matrix_name,
        "row_id": row_tag,
        "row_kind": row_kind,
        "status": "planned_dry_run_only",
        "recipe_id": recipe.recipe_id,
        "recipe_weights": dict(recipe.weights),
        "base_token": base.token,
        "base_profile_kind": base.profile_kind,
        "plain_name": f"{recipe.recipe_id}-{base.token}-c{copy_index}",
        "label": f"{recipe.recipe_id}-{base.token}-c{copy_index}",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "mode": MODE_TRAIN,
        "canonical_launcher": MODULE,
        "calls_stock_train_muzero": True,
        "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        "source_max_steps": SOURCE_MAX_STEPS,
        "source_state_trail_render_mode": base.render_mode,
        "source_state_bonus_render_mode": BONUS_RENDER_SIMPLE_SYMBOLS,
        "render_role": "target_a" if base.render_token == "rf" else "target_b",
        "training_seed": seed,
        "mixture_seed": mixture_spec["seed"],
        "eval_seed": _derived_seed("eval", seed),
        "copy_index": copy_index,
        "opponent_mixture_enabled": True,
        "opponent_mixture_spec": mixture_spec,
        "opponent_components": [entry["name"] for entry in mixture_spec["entries"]],
        "opponent_component_age_labels": [
            entry.get("age_label") for entry in mixture_spec["entries"]
        ],
        "moving_opponents_are_immortal": True,
        "base_settings": {
            "compute": args.compute,
            "token": base.token,
            "render_token": base.render_token,
            "source_state_trail_render_mode": base.render_mode,
            "source_state_bonus_render_mode": BONUS_RENDER_SIMPLE_SYMBOLS,
            "num_simulations": base.num_simulations,
            "collector_env_num": base.collector_env_num,
            "n_episode": base.n_episode,
            "batch_size": base.batch_size,
            "policy_action_repeat_min": base.policy_action_repeat_min,
            "policy_action_repeat_max": base.policy_action_repeat_max,
            "policy_action_repeat_extra_probability": (base.policy_action_repeat_extra_probability),
            "control_noise_profile_id": base.control_noise_profile_id,
            "max_train_iter": args.max_train_iter,
            "max_env_step": args.max_env_step,
            "save_ckpt_after_iter": base.save_ckpt_after_iter,
        },
        "deployed_app_submission": {
            "app_name": APP_NAME,
            "train_function": _train_function_name(args.compute),
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
            "spawn_order": ["poller", "train"],
        },
        "train_kwargs": train_kwargs,
        "poller_kwargs": poller_kwargs,
        "artifact_refs": {
            "run_manifest": _ref("training", TASK_ID, run_id, "run.json"),
            "attempt_manifest": _ref(
                "training", TASK_ID, run_id, "attempts", attempt_id, "attempt.json"
            ),
            "latest_attempt": _ref("training", TASK_ID, run_id, "latest_attempt.json"),
            "summary": _ref(train_ref, "summary.json"),
            "progress_latest": _ref(train_ref, "progress_latest.json"),
            "checkpoint_root": _ref("training", TASK_ID, run_id, "checkpoints", "lightzero"),
            "background_eval_status": _ref(train_ref, "checkpoint_eval_poller.json"),
        },
        "command": command,
        "command_text": shlex.join(command),
        "review_command_only": True,
    }


def _profile_default_ids(args: argparse.Namespace) -> tuple[str, str, str]:
    matrix_name = str(args.matrix_name)
    run_prefix = str(args.run_prefix)
    attempt_prefix = str(args.attempt_prefix)
    if args.profile == "next-wave":
        if matrix_name == DEFAULT_MATRIX_NAME:
            matrix_name = DEFAULT_NEXT_WAVE_MATRIX_NAME
        if run_prefix == DEFAULT_RUN_PREFIX:
            run_prefix = DEFAULT_NEXT_WAVE_RUN_PREFIX
        if attempt_prefix == DEFAULT_ATTEMPT_PREFIX:
            attempt_prefix = DEFAULT_NEXT_WAVE_ATTEMPT_PREFIX
    return matrix_name, run_prefix, attempt_prefix


def _recipes_catalog_for_profile(profile: str) -> tuple[Recipe, ...]:
    return NEXT_WAVE_RECIPES if profile == "next-wave" else RECIPES


def _main_recipe_ids_for_profile(profile: str) -> frozenset[str]:
    return NEXT_WAVE_MAIN_RECIPE_IDS if profile == "next-wave" else MAIN_RECIPE_IDS


def _control_recipe_ids_for_profile(profile: str) -> frozenset[str]:
    return NEXT_WAVE_CONTROL_RECIPE_IDS if profile == "next-wave" else CONTROL_RECIPE_IDS


def _recipes_for_profile(args: argparse.Namespace) -> tuple[Recipe, ...]:
    if args.profile == "canary":
        seen: set[str] = set()
        recipes: list[Recipe] = []
        for recipe_id, _base_token in CANARY_PLAN:
            if recipe_id in seen:
                continue
            seen.add(recipe_id)
            recipes.append(RECIPES_BY_ID[recipe_id])
        return tuple(recipes)
    if args.profile == "next-wave":
        if args.recipe_id:
            raise ValueError("--recipe-id is only supported for mix2 batch manifests")
        return NEXT_WAVE_RECIPES
    if args.recipe_id:
        recipes = []
        for recipe_id in args.recipe_id:
            if recipe_id not in RECIPES_BY_ID:
                choices = ", ".join(sorted(RECIPES_BY_ID))
                raise ValueError(f"unknown recipe id {recipe_id!r}; choices: {choices}")
            recipes.append(RECIPES_BY_ID[recipe_id])
        if len({recipe.recipe_id for recipe in recipes}) != len(recipes):
            raise ValueError("--recipe-id contains duplicates")
        return tuple(recipes)
    return RECIPES


def _base_by_render_pair(
    profiles: Sequence[BaseProfile],
) -> dict[tuple[int, int, int, str], dict[str, BaseProfile]]:
    grouped: dict[tuple[int, int, int, str], dict[str, BaseProfile]] = {}
    for profile in profiles:
        key = (
            profile.num_simulations,
            profile.collector_env_num,
            profile.batch_size,
            profile.repeat_token,
        )
        grouped.setdefault(key, {})[profile.render_token] = profile
    for key, by_render in grouped.items():
        if set(by_render) != {"rf", "rb"}:
            raise ValueError(f"base profile pair {key!r} is missing a render")
    return grouped


def _append_next_wave_pair(
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    matrix_name: str,
    run_prefix: str,
    attempt_prefix: str,
    recipe: Recipe,
    recipe_index: int,
    copy_index: int,
    refs: dict[str, str],
    rf_base: BaseProfile,
    rb_base: BaseProfile,
    block: str,
    pair_number: int,
) -> int:
    pair_id = f"p{pair_number:03d}"
    pair_key = (
        f"{block}:{recipe.recipe_id}:"
        f"s{rf_base.num_simulations}:c{rf_base.collector_env_num}:"
        f"l{rf_base.batch_size}:{rf_base.repeat_token}:c{copy_index}"
    )
    lead_render = "target_a" if pair_number % 2 else "target_b"
    bases = (rf_base, rb_base) if lead_render == "target_a" else (rb_base, rf_base)
    for position, base in enumerate(bases, start=1):
        row = _row(
            args=args,
            matrix_name=matrix_name,
            run_prefix=run_prefix,
            attempt_prefix=attempt_prefix,
            row_number=len(rows) + 1,
            recipe=recipe,
            recipe_index=recipe_index,
            copy_index=copy_index,
            refs=refs,
            base=base,
            profile=args.profile,
        )
        row.update(
            {
                "next_wave_block": block,
                "launch_order_index": len(rows) + 1,
                "launch_order_strategy": "paired_target_replicates_alternating_lead",
                "launch_pair_id": pair_id,
                "launch_pair_key": pair_key,
                "launch_pair_render_lead": lead_render,
                "launch_pair_position": position,
                "render_pairing_note": (
                    "Paired target-render rows are kept as matched replicates; "
                    "alternating launch lead keeps startup timing measurements clean."
                ),
            }
        )
        rows.append(row)
    return pair_number + 1


def _build_next_wave_rows(
    *,
    args: argparse.Namespace,
    matrix_name: str,
    run_prefix: str,
    attempt_prefix: str,
    refs: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pair_number = 1
    core_pairs = _base_by_render_pair(CORE_BASE_PROFILES)
    compute_pairs = _base_by_render_pair(NEXT_WAVE_COMPUTE_PROBE_BASE_PROFILES)

    for recipe in NEXT_WAVE_MAIN_RECIPES:
        recipe_index = NEXT_WAVE_RECIPE_INDEX_BY_ID[recipe.recipe_id]
        for repeat_token in ("rep0", "repM", "repH"):
            pair = core_pairs[(8, 32, 32, repeat_token)]
            for copy_index in range(1, NEXT_WAVE_MAIN_COPY_COUNT + 1):
                pair_number = _append_next_wave_pair(
                    rows=rows,
                    args=args,
                    matrix_name=matrix_name,
                    run_prefix=run_prefix,
                    attempt_prefix=attempt_prefix,
                    recipe=recipe,
                    recipe_index=recipe_index,
                    copy_index=copy_index,
                    refs=refs,
                    rf_base=pair["rf"],
                    rb_base=pair["rb"],
                    block="main",
                    pair_number=pair_number,
                )

    for recipe in NEXT_WAVE_CONTROL_RECIPES:
        recipe_index = NEXT_WAVE_RECIPE_INDEX_BY_ID[recipe.recipe_id]
        for repeat_token in ("rep0", "repM", "repH"):
            pair = core_pairs[(8, 32, 32, repeat_token)]
            for copy_index in range(1, NEXT_WAVE_CONTROL_COPY_COUNT + 1):
                pair_number = _append_next_wave_pair(
                    rows=rows,
                    args=args,
                    matrix_name=matrix_name,
                    run_prefix=run_prefix,
                    attempt_prefix=attempt_prefix,
                    recipe=recipe,
                    recipe_index=recipe_index,
                    copy_index=copy_index,
                    refs=refs,
                    rf_base=pair["rf"],
                    rb_base=pair["rb"],
                    block="control",
                    pair_number=pair_number,
                )

    for recipe_id in (
        "r50-blank50",
        "r75-blank25",
        "r50-scr50",
        "r50-mid25-old25",
        "r40-blank20-mid20-scr20",
    ):
        recipe = NEXT_WAVE_RECIPES_BY_ID[recipe_id]
        recipe_index = NEXT_WAVE_RECIPE_INDEX_BY_ID[recipe.recipe_id]
        for repeat_token in ("repM", "repH"):
            for probe_key in ((16, 32, 32), (8, 64, 32), (8, 32, 64)):
                pair = compute_pairs[(*probe_key, repeat_token)]
                for copy_index in range(1, NEXT_WAVE_COMPUTE_COPY_COUNT + 1):
                    pair_number = _append_next_wave_pair(
                        rows=rows,
                        args=args,
                        matrix_name=matrix_name,
                        run_prefix=run_prefix,
                        attempt_prefix=attempt_prefix,
                        recipe=recipe,
                        recipe_index=recipe_index,
                        copy_index=copy_index,
                        refs=refs,
                        rf_base=pair["rf"],
                        rb_base=pair["rb"],
                        block="compute_probe",
                        pair_number=pair_number,
                    )
    return rows


def _base_profile_payload(profile: BaseProfile) -> dict[str, Any]:
    return {
        "token": profile.token,
        "profile_kind": profile.profile_kind,
        "render_token": profile.render_token,
        "source_state_trail_render_mode": profile.render_mode,
        "num_simulations": profile.num_simulations,
        "collector_env_num": profile.collector_env_num,
        "n_episode": profile.n_episode,
        "batch_size": profile.batch_size,
        "repeat_token": profile.repeat_token,
        "policy_action_repeat_min": profile.policy_action_repeat_min,
        "policy_action_repeat_max": profile.policy_action_repeat_max,
        "policy_action_repeat_extra_probability": (profile.policy_action_repeat_extra_probability),
        "control_noise_profile_id": profile.control_noise_profile_id,
        "save_ckpt_after_iter": profile.save_ckpt_after_iter,
    }


def _validate_mix2_args(args: argparse.Namespace) -> None:
    if args.save_ckpt_after_iter != SAVE_CKPT_AFTER_ITER:
        raise ValueError("curvy-mix2 only defines k10 bases with save_ckpt_after_iter=10000")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    _validate_mix2_args(args)
    matrix_name_raw, run_prefix_raw, attempt_prefix_raw = _profile_default_ids(args)
    matrix_name = _safe_id(matrix_name_raw, label="matrix_name")
    run_prefix = _safe_id(run_prefix_raw, label="run_prefix")
    attempt_prefix = _safe_id(attempt_prefix_raw, label="attempt_prefix")
    refs = _checkpoint_refs(args)
    recipes = _recipes_for_profile(args)
    recipe_catalog = _recipes_catalog_for_profile(args.profile)
    main_recipe_ids = _main_recipe_ids_for_profile(args.profile)
    control_recipe_ids = _control_recipe_ids_for_profile(args.profile)
    selected_main_recipes = tuple(
        recipe for recipe in recipes if recipe.recipe_id in main_recipe_ids
    )
    selected_control_recipes = tuple(
        recipe for recipe in recipes if recipe.recipe_id in control_recipe_ids
    )
    include_sentinels = args.profile == "batch" and args.batch_scope == BATCH_SCOPE_FULL
    rows: list[dict[str, Any]] = []
    if args.profile == "canary":
        for recipe_id, base_token in CANARY_PLAN:
            rows.append(
                _row(
                    args=args,
                    matrix_name=matrix_name,
                    run_prefix=run_prefix,
                    attempt_prefix=attempt_prefix,
                    row_number=len(rows) + 1,
                    recipe=RECIPES_BY_ID[recipe_id],
                    recipe_index=RECIPE_INDEX_BY_ID[recipe_id],
                    copy_index=1,
                    refs=refs,
                    base=BASE_PROFILES_BY_TOKEN[base_token],
                    profile=args.profile,
                )
            )
    elif args.profile == "next-wave":
        rows = _build_next_wave_rows(
            args=args,
            matrix_name=matrix_name,
            run_prefix=run_prefix,
            attempt_prefix=attempt_prefix,
            refs=refs,
        )
    else:
        for recipe in selected_main_recipes:
            recipe_index = RECIPE_INDEX_BY_ID[recipe.recipe_id]
            for base in CORE_BASE_PROFILES:
                for copy_index in range(1, MAIN_CORE_COPY_COUNT + 1):
                    rows.append(
                        _row(
                            args=args,
                            matrix_name=matrix_name,
                            run_prefix=run_prefix,
                            attempt_prefix=attempt_prefix,
                            row_number=len(rows) + 1,
                            recipe=recipe,
                            recipe_index=recipe_index,
                            copy_index=copy_index,
                            refs=refs,
                            base=base,
                            profile=args.profile,
                        )
                    )
            if include_sentinels:
                for base in SENTINEL_BASE_PROFILES:
                    for copy_index in range(1, MAIN_SENTINEL_COPY_COUNT + 1):
                        rows.append(
                            _row(
                                args=args,
                                matrix_name=matrix_name,
                                run_prefix=run_prefix,
                                attempt_prefix=attempt_prefix,
                                row_number=len(rows) + 1,
                                recipe=recipe,
                                recipe_index=recipe_index,
                                copy_index=copy_index,
                                refs=refs,
                                base=base,
                                profile=args.profile,
                            )
                        )
        for recipe in selected_control_recipes:
            recipe_index = RECIPE_INDEX_BY_ID[recipe.recipe_id]
            for base in CORE_BASE_PROFILES:
                for copy_index in range(1, CONTROL_COPY_COUNT + 1):
                    rows.append(
                        _row(
                            args=args,
                            matrix_name=matrix_name,
                            run_prefix=run_prefix,
                            attempt_prefix=attempt_prefix,
                            row_number=len(rows) + 1,
                            recipe=recipe,
                            recipe_index=recipe_index,
                            copy_index=copy_index,
                            refs=refs,
                            base=base,
                            profile=args.profile,
                        )
                    )
    main_rows_per_recipe = len(CORE_BASE_PROFILES) * MAIN_CORE_COPY_COUNT + (
        len(SENTINEL_BASE_PROFILES) * MAIN_SENTINEL_COPY_COUNT if include_sentinels else 0
    )
    control_rows_per_recipe = len(CORE_BASE_PROFILES) * CONTROL_COPY_COUNT
    if args.profile == "next-wave":
        expected_count = 300
    else:
        expected_count = (
            len(CANARY_PLAN)
            if args.profile == "canary"
            else len(selected_main_recipes) * main_rows_per_recipe
            + len(selected_control_recipes) * control_rows_per_recipe
        )
    _validate_manifest(rows, expected_count=expected_count)
    sentinel_profiles = list(SENTINEL_BASE_PROFILES) if include_sentinels else []
    compute_probe_profiles = (
        list(NEXT_WAVE_COMPUTE_PROBE_BASE_PROFILES) if args.profile == "next-wave" else []
    )
    active_base_profiles = [*CORE_BASE_PROFILES, *sentinel_profiles, *compute_probe_profiles]
    batch_shape = (
        f"{len(selected_main_recipes)} main recipes x "
        f"(6 core bases x {MAIN_CORE_COPY_COUNT} seeds"
        + (f" + 6 sentinel bases x {MAIN_SENTINEL_COPY_COUNT} seed)" if include_sentinels else ")")
        + f" + {len(selected_control_recipes)} controls x 6 core bases x "
        f"{CONTROL_COPY_COUNT} seed = {len(rows)} rows"
    )
    selected_recipe_ids = {recipe.recipe_id for recipe in recipes}
    if args.profile == "next-wave":
        shape = (
            "6 main recipes x 2 target replicate tokens x 3 repeat levels x 5 seeds"
            " + 5 controls x 2 target replicate tokens x 3 repeat levels x 2 seeds"
            " + 5 selected recipes x 2 target replicate tokens x 2 repeat levels x "
            "3 compute probes x 1 seed = 300 rows"
        )
    elif args.profile == "batch":
        shape = batch_shape
    else:
        shape = "six fixed canary rows across selected recipes and bases"
    return {
        "schema_id": SCHEMA_ID,
        "generated_at": _utc_timestamp(),
        "dry_run_only": True,
        "launches_modal": False,
        "current_launch_approved": False,
        "matrix_name": matrix_name,
        "profile": args.profile,
        "batch_scope": args.batch_scope,
        "task_id": TASK_ID,
        "canonical_launcher": MODULE,
        "run_prefix": run_prefix,
        "attempt_prefix": attempt_prefix,
        "row_count": len(rows),
        "recipe_count": len(recipes),
        "main_recipe_count": len(selected_main_recipes),
        "control_recipe_count": len(selected_control_recipes),
        "compute_probe_recipe_count": (
            len(NEXT_WAVE_COMPUTE_RECIPE_IDS) if args.profile == "next-wave" else 0
        ),
        "excluded_recipe_ids": [
            recipe.recipe_id
            for recipe in recipe_catalog
            if recipe.recipe_id not in selected_recipe_ids
        ],
        "core_profile_count": len(CORE_BASE_PROFILES),
        "sentinel_profile_count": len(sentinel_profiles),
        "compute_probe_profile_count": len(compute_probe_profiles),
        "main_core_copies": (
            NEXT_WAVE_MAIN_COPY_COUNT if args.profile == "next-wave" else MAIN_CORE_COPY_COUNT
        ),
        "main_sentinel_copies": MAIN_SENTINEL_COPY_COUNT if include_sentinels else 0,
        "control_copies": (
            NEXT_WAVE_CONTROL_COPY_COUNT if args.profile == "next-wave" else CONTROL_COPY_COUNT
        ),
        "compute_probe_copies": (
            NEXT_WAVE_COMPUTE_COPY_COUNT if args.profile == "next-wave" else 0
        ),
        "shape": shape,
        "launch_order_strategy": (
            "paired_target_replicates_alternating_lead"
            if args.profile == "next-wave"
            else "recipe_then_base"
        ),
        "render_pairing_rationale": (
            "Paired target-render rows are retained as matched replicates. "
            "Both tokens now use browser_lines + simple_symbols; alternating "
            "launch lead keeps timing measurement clean."
            if args.profile == "next-wave"
            else None
        ),
        "base_settings": {
            "trainer": "stock LightZero train_muzero",
            "mode": MODE_TRAIN,
            "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
            "reward_variant": REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "source_max_steps": SOURCE_MAX_STEPS,
            "compute": args.compute,
            "base_token_shape": "<render>-s<SIMS>-c<COLLECT>-l<LEARN_BATCH>-<repeat>-k10",
            "source_state_trail_render_modes": {
                "rf": RENDER_FAST,
                "rb": RENDER_BROWSER,
            },
            "source_state_bonus_render_mode": BONUS_RENDER_SIMPLE_SYMBOLS,
            "num_simulations_values": sorted(
                {profile.num_simulations for profile in active_base_profiles}
            ),
            "collector_env_num_values": sorted(
                {profile.collector_env_num for profile in active_base_profiles}
            ),
            "n_episode_matches_collector_env_num": True,
            "batch_size_values": sorted({profile.batch_size for profile in active_base_profiles}),
            "repeat_profiles": {
                "rep0": {
                    "policy_action_repeat_min": 1,
                    "policy_action_repeat_max": 1,
                    "policy_action_repeat_extra_probability": 0.0,
                    "control_noise_profile_id": "none",
                },
                "repM": {
                    "policy_action_repeat_min": 1,
                    "policy_action_repeat_max": 3,
                    "policy_action_repeat_extra_probability": 0.20,
                    "control_noise_profile_id": "policy_action_repeat_medium",
                },
                "repH": {
                    "policy_action_repeat_min": 1,
                    "policy_action_repeat_max": 3,
                    "policy_action_repeat_extra_probability": 0.35,
                    "control_noise_profile_id": "policy_action_repeat_high",
                },
            },
            "max_train_iter": args.max_train_iter,
            "max_env_step": args.max_env_step,
            "save_ckpt_after_iter": SAVE_CKPT_AFTER_ITER,
            "background_eval_enabled": True,
            "background_gif_enabled": True,
            "stock_lightzero_eval_freq": 0,
        },
        "base_profiles": {
            "core": [_base_profile_payload(profile) for profile in CORE_BASE_PROFILES],
            "sentinel": [_base_profile_payload(profile) for profile in sentinel_profiles],
            "compute_probe": [_base_profile_payload(profile) for profile in compute_probe_profiles],
        },
        "checkpoint_refs": refs,
        "component_contract": {
            "recent_mid_old": (
                "frozen LightZero checkpoints by exact iteration_N.pth.tar refs, "
                "immortal in this diagnostic so weak opponents do not disappear"
            ),
            "scripted": "proactive wall-avoidant hand-coded opponent, immortal",
            "passive": "fixed-straight immortal trail source",
            "blank": "hidden no-op immortal opponent, no trail and no collision pressure",
        },
        "guards": {
            "deployed_app_grouped_submitter_required": True,
            "deployed_app_name": APP_NAME,
            "no_modal_run_per_row_for_large_batch": True,
            "mode_required": MODE_TRAIN,
            "stock_train_muzero_only": True,
            "background_eval_required": True,
            "background_gif_required": True,
            "source_max_steps_required": SOURCE_MAX_STEPS,
            "no_top_level_opponent_checkpoint_ref": True,
            "frozen_refs_must_be_exact_iteration_files": True,
            "tiny_remote_canary_required_before_batch": True,
        },
        "recipes": [
            {"recipe_id": recipe.recipe_id, "weights": dict(recipe.weights)} for recipe in recipes
        ],
        "rows": rows,
    }


def _validate_manifest(
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"manifest produced {len(rows)} rows, expected {expected_count}")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate run_id values")
    if len({row["attempt_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate attempt_id values")
    for row in rows:
        base = ALL_BASE_PROFILES_BY_TOKEN_KIND.get(
            (str(row.get("base_token")), str(row.get("base_profile_kind")))
        )
        if base is None:
            base = BASE_PROFILES_BY_TOKEN.get(str(row.get("base_token")))
        if base is None:
            raise ValueError(f"row {row['row_id']} has unknown base token")
        if row["mode"] != MODE_TRAIN:
            raise ValueError(f"row {row['row_id']} is not stock train mode")
        if row["env_variant"] != ENV_SOURCE_STATE_FIXED_OPPONENT:
            raise ValueError(f"row {row['row_id']} has wrong env variant")
        if row["reward_variant"] != REWARD_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
            raise ValueError(f"row {row['row_id']} has wrong reward")
        if row["source_max_steps"] != SOURCE_MAX_STEPS:
            raise ValueError(f"row {row['row_id']} has wrong cap")
        if row["train_kwargs"]["opponent_checkpoint_ref"] is not None:
            raise ValueError(f"row {row['row_id']} uses a top-level checkpoint ref")
        if (
            row["poller_kwargs"]["opponent_mixture_spec"]
            != row["train_kwargs"]["opponent_mixture_spec"]
        ):
            raise ValueError(f"row {row['row_id']} train/poller mixture mismatch")
        _validate_row_base_settings(row, base)
        mixture = parse_opponent_mixture_spec(row["opponent_mixture_spec"])
        if mixture is None:
            raise ValueError(f"row {row['row_id']} has empty mixture")
        if abs(float(mixture["total_weight"]) - 100.0) > 1e-9:
            raise ValueError(f"row {row['row_id']} mixture weights must sum to 100")
        for entry in mixture["entries"]:
            if entry["opponent_policy_kind"] != "frozen_lightzero_checkpoint":
                continue
            checkpoint_ref = str(entry["opponent_checkpoint_ref"])
            if "latest" in checkpoint_ref or "ckpt_best" in checkpoint_ref:
                raise ValueError(f"row {row['row_id']} uses mutable checkpoint ref")
            if not re.fullmatch(r"iteration_\d+\.pth\.tar", Path(checkpoint_ref).name):
                raise ValueError(f"row {row['row_id']} uses a non-iteration checkpoint")
        if row["deployed_app_submission"]["app_name"] != APP_NAME:
            raise ValueError(f"row {row['row_id']} does not target the grouped app")
        if not row["train_kwargs"]["background_eval_enabled"]:
            raise ValueError(f"row {row['row_id']} disabled background eval")
        if not row["train_kwargs"]["background_gif_enabled"]:
            raise ValueError(f"row {row['row_id']} disabled background GIF")


def _validate_row_base_settings(row: dict[str, Any], base: BaseProfile) -> None:
    expected = {
        "token": base.token,
        "render_token": base.render_token,
        "source_state_trail_render_mode": base.render_mode,
        "source_state_bonus_render_mode": BONUS_RENDER_SIMPLE_SYMBOLS,
        "num_simulations": base.num_simulations,
        "collector_env_num": base.collector_env_num,
        "n_episode": base.n_episode,
        "batch_size": base.batch_size,
        "policy_action_repeat_min": base.policy_action_repeat_min,
        "policy_action_repeat_max": base.policy_action_repeat_max,
        "policy_action_repeat_extra_probability": (base.policy_action_repeat_extra_probability),
        "control_noise_profile_id": base.control_noise_profile_id,
        "save_ckpt_after_iter": base.save_ckpt_after_iter,
    }
    for key, value in expected.items():
        if row["base_settings"].get(key) != value:
            raise ValueError(f"row {row['row_id']} base setting {key} drifted")

    train_kwargs = row["train_kwargs"]
    for key in (
        "source_state_trail_render_mode",
        "source_state_bonus_render_mode",
        "num_simulations",
        "collector_env_num",
        "n_episode",
        "batch_size",
        "policy_action_repeat_min",
        "policy_action_repeat_max",
        "policy_action_repeat_extra_probability",
        "control_noise_profile_id",
        "save_ckpt_after_iter",
    ):
        if train_kwargs.get(key) != expected[key]:
            raise ValueError(f"row {row['row_id']} train kwarg {key} drifted")


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
        "# Review commands only. Use submit_curvytron_survivaldiag_manifest.py to launch.\n"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("canary", "batch", "next-wave"),
        default="batch",
    )
    parser.add_argument(
        "--batch-scope",
        choices=BATCH_SCOPE_CHOICES,
        default=BATCH_SCOPE_FULL,
        help="Use 'core' to omit sim16/C64/B64 sentinel rows from batch manifests.",
    )
    parser.add_argument("--matrix-name", default=DEFAULT_MATRIX_NAME)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--attempt-prefix", default=DEFAULT_ATTEMPT_PREFIX)
    parser.add_argument("--compute", default="gpu-l4-t4-cpu40")
    parser.add_argument("--max-train-iter", type=int, default=300_000)
    parser.add_argument("--max-env-step", type=int, default=30_000_000)
    parser.add_argument("--save-ckpt-after-iter", type=int, default=10_000)
    parser.add_argument(
        "--recipe-id",
        action="append",
        default=[],
        help=(
            "Optional batch recipe allow-list. Repeat for each recipe to include. "
            "Canary profile ignores this and uses the fixed canary plan."
        ),
    )
    parser.add_argument("--env-manager-type", default="subprocess")
    parser.add_argument("--background-eval-seed-count", type=int, default=8)
    parser.add_argument("--background-eval-poll-interval-sec", type=float, default=10.0)
    parser.add_argument(
        "--background-eval-poller-max-runtime-sec",
        type=float,
        default=18 * 60 * 60,
    )
    parser.add_argument("--stop-after-learner-train-calls", type=int, default=0)
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
    parser.add_argument("--no-detach", action="store_true")
    parser.add_argument("--stdout-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_manifest(args)
    if args.stdout_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    outputs = _write_outputs(manifest, output_root=args.output_root)
    print(json.dumps({"manifest": manifest["matrix_name"], **outputs}, indent=2))


if __name__ == "__main__":
    main()
