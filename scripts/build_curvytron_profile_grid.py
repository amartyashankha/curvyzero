#!/usr/bin/env python3
"""Build a small, flexible CurvyTron optimizer profile manifest.

This is intentionally simpler than the historical matrix builders. It only
creates profile commands for the current trusted stock LightZero lane, and it
lets the caller choose the few axes needed for the next question.
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.contracts.curvytron import CURVYTRON_TRAINING_TASK_ID
from curvyzero.env.observation_surface_contract import (
    DEFAULT_POLICY_OBSERVATION_BACKEND,
    POLICY_BONUS_RENDER_MODE,
    POLICY_TRAIL_RENDER_MODE,
)
from curvyzero.training import exploration_bonus as xb

MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train::main"
TASK_ID = CURVYTRON_TRAINING_TASK_ID
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_optimizer_profile_manifests")
COMPUTE_ALIASES = {
    "l4": "gpu-l4-t4-cpu40",
    "gpu-l4": "gpu-l4-t4-cpu40",
    "h100": "gpu-h100-cpu40",
    "gpu-h100": "gpu-h100-cpu40",
    "h100x2": "gpu-h100x2-cpu40",
    "gpu-h100x2": "gpu-h100x2-cpu40",
}
COMPUTE_CHOICES = {
    "cpu",
    "cpu64",
    "gpu-l4-t4",
    "gpu-l4-t4-cpu40",
    "gpu-h100-cpu40",
    "gpu-h100x2-cpu40",
}
COLLECT_SEARCH_BACKEND_CHOICES = {"stock", "direct_ctree_gpu_latent"}
COLLECT_SEARCH_CTREE_BACKEND_CHOICES = {"lightzero", "flat_a3"}
MATCHED_DENOMINATOR_ID = "curvytron-stock-vs-compact-owned-no-rnd-h100-20260528"
MATCHED_DENOMINATOR_ROW_PURPOSE = "matched_denominator_speed"
MATCHED_STOCK_SPEED_CURRENCY = "stock_train_muzero_profile_env_steps_per_sec"
MATCHED_COMPACT_MANIFEST_REF = (
    "artifacts/local/curvytron_hybrid_observation_profile_manifests/"
    "optimizer-matched-denominator-compact-owned-20260528/manifest.json"
)


def apply_next_matched_denominator_stock_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the selected stock side of the denominator pair."""

    args.family = "matched_denominator_stock_reference"
    args.seed = 304
    args.seeds = None
    args.computes = ["gpu-h100-cpu40"]
    args.env_manager_types = ["subprocess"]
    args.collectors = [512]
    args.batch_sizes = [64]
    args.num_simulations = [8]
    args.exploration_bonus_modes = ["none"]
    args.exploration_bonus_weight = 0.0
    args.source_max_steps = 512
    args.max_train_iter = 96
    args.max_env_step = 200_000
    args.save_ckpt_after_iter = 999_999
    args.stop_after_learner_train_calls = 12
    args.env_telemetry_stride = 256
    args.env_variant = "source_state_fixed_opponent"
    args.reward_variant = "sparse_outcome"
    args.opponent_policy_kind = "fixed_straight"
    args.collect_search_backends = ["stock"]
    args.collect_search_ctree_backends = ["lightzero"]
    args.disable_death_for_profile = True
    args.detached = True
    args.matched_denominator_id = MATCHED_DENOMINATOR_ID
    args.matched_pair_role = "stock_reference"
    args.matched_speed_currency = MATCHED_STOCK_SPEED_CURRENCY
    args.matched_counterpart_manifest_ref = MATCHED_COMPACT_MANIFEST_REF
    args.matched_counterpart_row_id = "001"
    args.matched_row_purpose = MATCHED_DENOMINATOR_ROW_PURPOSE
    args.matched_promotion_claim = False


def _csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _csv_strings(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _normalize_compute(raw: str) -> str:
    compute = COMPUTE_ALIASES.get(raw, raw)
    if compute not in COMPUTE_CHOICES:
        expected = ", ".join(sorted(COMPUTE_CHOICES.union(COMPUTE_ALIASES)))
        raise argparse.ArgumentTypeError(
            f"unknown compute {raw!r}; expected one of: {expected}"
        )
    return compute


def _normalize_choice(raw: str, *, choices: set[str], label: str) -> str:
    value = raw.strip()
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise argparse.ArgumentTypeError(
            f"unknown {label} {raw!r}; expected one of: {expected}"
        )
    return value


def _csv_collect_search_backends(raw: str) -> list[str]:
    values = [
        _normalize_choice(
            part,
            choices=COLLECT_SEARCH_BACKEND_CHOICES,
            label="collect search backend",
        )
        for part in raw.split(",")
        if part.strip()
    ]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one collect search backend")
    return values


def _csv_collect_search_ctree_backends(raw: str) -> list[str]:
    values = [
        _normalize_choice(
            part,
            choices=COLLECT_SEARCH_CTREE_BACKEND_CHOICES,
            label="collect search ctree backend",
        )
        for part in raw.split(",")
        if part.strip()
    ]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one collect search ctree backend")
    return values


def _csv_computes(raw: str) -> list[str]:
    values = [_normalize_compute(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one compute")
    return values


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_id(raw: str, *, label: str, limit: int = 96) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    clean = "".join(char if char in allowed else "-" for char in raw)
    clean = clean.strip("-.")
    if len(clean) > limit:
        clean = clean[:limit].rstrip("-.")
    if not clean or not clean[0].isalnum():
        raise ValueError(f"{label} cannot be made into a safe id: {raw!r}")
    return clean


def _compute_tag(compute: str) -> str:
    return compute.replace("gpu-", "").replace("-cpu", "c").replace("-", "")


def _row_command(args: argparse.Namespace, *, row: dict[str, Any]) -> list[str]:
    command = ["uv", "run", "--extra", "modal", "modal", "run"]
    if args.detached:
        command.append("--detach")
    command.extend(
        [
            "--quiet",
            "-m",
            MODULE,
            "--mode",
            "profile",
            "--compute",
            row["compute"],
            "--seed",
            str(row["seed"]),
            "--run-id",
            row["run_id"],
            "--attempt-id",
            row["attempt_id"],
            "--env-variant",
            args.env_variant,
            "--reward-variant",
            args.reward_variant,
            "--opponent-policy-kind",
            args.opponent_policy_kind,
            "--no-opponent-use-cuda",
            "--env-manager-type",
            row["env_manager_type"],
            "--collector-env-num",
            str(row["collector_env_num"]),
            "--evaluator-env-num",
            str(row["evaluator_env_num"]),
            "--n-episode",
            str(row["n_episode"]),
            "--batch-size",
            str(row["batch_size"]),
            "--num-simulations",
            str(row["num_simulations"]),
            "--source-state-trail-render-mode",
            args.trail_render_mode,
            "--source-state-bonus-render-mode",
            args.bonus_render_mode,
            "--policy-observation-backend",
            args.policy_observation_backend,
            "--collect-search-backend",
            row["collect_search_backend"],
            "--collect-search-ctree-backend",
            row["collect_search_ctree_backend"],
            "--exploration-bonus-mode",
            row["exploration_bonus_mode"],
            "--exploration-bonus-weight",
            str(args.exploration_bonus_weight),
            "--source-max-steps",
            str(args.source_max_steps),
            "--max-train-iter",
            str(args.max_train_iter),
            "--max-env-step",
            str(args.max_env_step),
            "--save-ckpt-after-iter",
            str(args.save_ckpt_after_iter),
            "--stop-after-learner-train-calls",
            str(args.stop_after_learner_train_calls),
            "--env-telemetry-stride",
            str(args.env_telemetry_stride),
            "--lightzero-eval-freq",
            "0",
            "--skip-lightzero-eval-in-profile",
            "--no-background-eval-enabled",
            "--no-background-gif-enabled",
            "--output-detail",
            "compact",
        ]
    )
    if args.disable_death_for_profile:
        command.append("--disable-death-for-profile")
    if row["exploration_bonus_mode"] != "none":
        command.extend(
            [
                "--exploration-bonus-rnd-batch-size",
                str(args.exploration_bonus_rnd_batch_size),
                "--exploration-bonus-rnd-update-per-collect",
                str(args.exploration_bonus_rnd_update_per_collect),
                "--require-rnd-metrics",
            ]
        )
    if args.detached:
        command.append("--profile-spawn")
    return command


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    row_number = 1
    seeds = args.seeds if args.seeds is not None else [args.seed]
    multi_seed = len(seeds) > 1
    evaluator_env_num = int(getattr(args, "evaluator_env_num", 2))
    for seed in seeds:
        for compute in args.computes:
            for env_manager_type in args.env_manager_types:
                for collect_search_backend in args.collect_search_backends:
                    for collect_search_ctree_backend in args.collect_search_ctree_backends:
                        if (
                            collect_search_ctree_backend != "lightzero"
                            and collect_search_backend != "direct_ctree_gpu_latent"
                        ):
                            raise ValueError(
                                "collect_search_ctree_backend='flat_a3' requires "
                                "collect_search_backend='direct_ctree_gpu_latent'"
                            )
                        for collectors in args.collectors:
                            for batch_size in args.batch_sizes:
                                for sims in args.num_simulations:
                                    for exploration_bonus_mode in args.exploration_bonus_modes:
                                        row_id = f"{row_number:03d}"
                                        manager_tag = (
                                            "sub"
                                            if env_manager_type == "subprocess"
                                            else env_manager_type
                                        )
                                        collect_tag = (
                                            "stocksearch"
                                            if collect_search_backend == "stock"
                                            else "directsearch"
                                        )
                                        ctree_tag = (
                                            "lzctree"
                                            if collect_search_ctree_backend == "lightzero"
                                            else collect_search_ctree_backend
                                        )
                                        bonus_tag = str(exploration_bonus_mode).replace("_", "")
                                        if exploration_bonus_mode != "none":
                                            bonus_tag = (
                                                f"{bonus_tag}"
                                                f"-rndbs{args.exploration_bonus_rnd_batch_size}"
                                                f"-rndupd{args.exploration_bonus_rnd_update_per_collect}"
                                                f"-rndw{args.exploration_bonus_weight:g}"
                                            )
                                        seed_tag = f"-s{seed}" if multi_seed else ""
                                        tag = (
                                            f"{_compute_tag(compute)}-{manager_tag}-c{collectors}"
                                            f"-b{batch_size}-sim{sims}-{collect_tag}-{ctree_tag}"
                                            f"-{bonus_tag}{seed_tag}"
                                            f"-{'nodeath' if args.disable_death_for_profile else 'normal'}"
                                        )
                                        run_id = _safe_id(
                                            f"{args.run_prefix}-{row_id}-{tag}",
                                            label="run_id",
                                        )
                                        attempt_id = _safe_id(
                                            f"{args.attempt_prefix}-{row_id}-{tag}",
                                            label="attempt_id",
                                        )
                                        row = {
                                            "schema_id": "curvyzero_optimizer_profile_manifest_row/v1",
                                            "experiment_id": args.experiment_id,
                                            "row_id": row_id,
                                            "family": args.family,
                                            "label": tag,
                                            "run_id": run_id,
                                            "attempt_id": attempt_id,
                                            "seed": seed,
                                            "compute": compute,
                                            "env_manager_type": env_manager_type,
                                            "collector_env_num": collectors,
                                            "evaluator_env_num": evaluator_env_num,
                                            "n_episode": collectors,
                                            "batch_size": batch_size,
                                            "num_simulations": sims,
                                            "collect_search_backend": collect_search_backend,
                                            "collect_search_ctree_backend": (
                                                collect_search_ctree_backend
                                            ),
                                            "exploration_bonus_mode": exploration_bonus_mode,
                                            "exploration_bonus_weight": (
                                                args.exploration_bonus_weight
                                            ),
                                            "exploration_bonus_rnd_batch_size": (
                                                args.exploration_bonus_rnd_batch_size
                                            ),
                                            "exploration_bonus_rnd_update_per_collect": (
                                                args.exploration_bonus_rnd_update_per_collect
                                            ),
                                            "source_max_steps": args.source_max_steps,
                                            "disable_death_for_profile": (
                                                args.disable_death_for_profile
                                            ),
                                            "readback": (
                                                "modal_function_call_result"
                                                if args.detached
                                                else "local_compact_stdout"
                                            ),
                                        }
                                        matched_denominator_id = str(
                                            getattr(args, "matched_denominator_id", "") or ""
                                        )
                                        if matched_denominator_id:
                                            row["matched_denominator_id"] = (
                                                matched_denominator_id
                                            )
                                            row["matched_pair_role"] = str(
                                                getattr(
                                                    args,
                                                    "matched_pair_role",
                                                    "stock_reference",
                                                )
                                            )
                                            row["speed_currency"] = str(
                                                getattr(
                                                    args,
                                                    "matched_speed_currency",
                                                    MATCHED_STOCK_SPEED_CURRENCY,
                                                )
                                            )
                                            row["counterpart_manifest_ref"] = str(
                                                getattr(
                                                    args,
                                                    "matched_counterpart_manifest_ref",
                                                    "",
                                                )
                                                or ""
                                            )
                                            row["counterpart_row_id"] = str(
                                                getattr(
                                                    args,
                                                    "matched_counterpart_row_id",
                                                    "001",
                                                )
                                            )
                                            row["row_purpose"] = str(
                                                getattr(
                                                    args,
                                                    "matched_row_purpose",
                                                    MATCHED_DENOMINATOR_ROW_PURPOSE,
                                                )
                                            )
                                            row["promotion_claim"] = bool(
                                                getattr(
                                                    args,
                                                    "matched_promotion_claim",
                                                    False,
                                                )
                                            )
                                            row["fixed_denominator"] = {
                                                "batch_size": batch_size,
                                                "collector_env_num": collectors,
                                                "collect_search_backend": (
                                                    collect_search_backend
                                                ),
                                                "collect_search_ctree_backend": (
                                                    collect_search_ctree_backend
                                                ),
                                                "compute": compute,
                                                "disable_death_for_profile": (
                                                    args.disable_death_for_profile
                                                ),
                                                "env_manager_type": env_manager_type,
                                                "exploration_bonus_mode": (
                                                    exploration_bonus_mode
                                                ),
                                                "max_env_step": args.max_env_step,
                                                "max_train_iter": args.max_train_iter,
                                                "num_simulations": sims,
                                                "policy_observation_backend": (
                                                    args.policy_observation_backend
                                                ),
                                                "source_max_steps": args.source_max_steps,
                                                "speed_currency": row["speed_currency"],
                                                "stop_after_learner_train_calls": (
                                                    args.stop_after_learner_train_calls
                                                ),
                                            }
                                        command = _row_command(args, row=row)
                                        row["command"] = command
                                        row["command_text"] = shlex.join(command)
                                        row["result_ref"] = (
                                            f"training/{TASK_ID}/{run_id}/attempts/"
                                            f"{attempt_id}/train/summary.json"
                                        )
                                        rows.append(row)
                                        row_number += 1
    matched_denominator_id = str(getattr(args, "matched_denominator_id", "") or "")
    manifest = {
        "schema_id": "curvyzero_optimizer_profile_manifest/v1",
        "generated_at": _utc_timestamp(),
        "experiment_id": args.experiment_id,
        "family": args.family,
        "row_count": len(rows),
        "readback": "detached_profile_spawn" if args.detached else "direct_blocking_stdout",
        "guardrails": {
            "mode": "profile",
            "path": "stock LightZero train_muzero",
            "env_variant": args.env_variant,
            "policy_surface": f"{args.trail_render_mode} + {args.bonus_render_mode}",
            "policy_observation_backend": args.policy_observation_backend,
            "env_manager_types": args.env_manager_types,
            "collect_search_backends": args.collect_search_backends,
            "collect_search_ctree_backends": args.collect_search_ctree_backends,
            "exploration_bonus_modes": args.exploration_bonus_modes,
            "lightzero_eval_freq": 0,
            "background_eval_enabled": False,
            "background_gif_enabled": False,
            "checkpoint_writes_expected": False,
            "seeds": seeds,
        },
        "rows": rows,
    }
    if matched_denominator_id:
        manifest["matched_denominator"] = {
            "counterpart_manifest_ref": str(
                getattr(args, "matched_counterpart_manifest_ref", "") or ""
            ),
            "counterpart_row_id": str(
                getattr(args, "matched_counterpart_row_id", "001")
            ),
            "id": matched_denominator_id,
            "promotion_claim": bool(getattr(args, "matched_promotion_claim", False)),
            "role": str(getattr(args, "matched_pair_role", "stock_reference")),
            "row_purpose": str(
                getattr(args, "matched_row_purpose", MATCHED_DENOMINATOR_ROW_PURPOSE)
            ),
            "speed_currency": str(
                getattr(args, "matched_speed_currency", MATCHED_STOCK_SPEED_CURRENCY)
            ),
        }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--family", default="ad_hoc_grid")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--attempt-prefix", default="profile")
    parser.add_argument("--seed", type=int, default=304)
    parser.add_argument("--seeds", type=_csv_ints)
    parser.add_argument("--computes", type=_csv_computes)
    parser.add_argument("--env-manager-types", type=_csv_strings, default=["subprocess"])
    parser.add_argument("--collectors", type=_csv_ints)
    parser.add_argument("--evaluator-env-num", type=int, default=2)
    parser.add_argument("--batch-sizes", type=_csv_ints, default=[32])
    parser.add_argument("--num-simulations", type=_csv_ints, default=[8])
    parser.add_argument("--exploration-bonus-modes", type=_csv_strings, default=["none"])
    parser.add_argument("--exploration-bonus-weight", type=float, default=0.0)
    parser.add_argument("--exploration-bonus-rnd-batch-size", type=int, default=64)
    parser.add_argument(
        "--exploration-bonus-rnd-update-per-collect",
        type=int,
        default=xb.RND_DEFAULT_UPDATE_PER_COLLECT,
    )
    parser.add_argument("--source-max-steps", type=int, default=512)
    parser.add_argument("--max-train-iter", type=int, default=96)
    parser.add_argument("--max-env-step", type=int, default=200_000)
    parser.add_argument("--save-ckpt-after-iter", type=int, default=999_999)
    parser.add_argument("--stop-after-learner-train-calls", type=int, default=12)
    parser.add_argument("--env-telemetry-stride", type=int, default=256)
    parser.add_argument("--env-variant", default="source_state_fixed_opponent")
    parser.add_argument("--reward-variant", default="sparse_outcome")
    parser.add_argument("--opponent-policy-kind", default="fixed_straight")
    parser.add_argument("--trail-render-mode", default=POLICY_TRAIL_RENDER_MODE)
    parser.add_argument("--bonus-render-mode", default=POLICY_BONUS_RENDER_MODE)
    parser.add_argument("--policy-observation-backend", default=DEFAULT_POLICY_OBSERVATION_BACKEND)
    parser.add_argument(
        "--collect-search-backends",
        type=_csv_collect_search_backends,
        default=["stock"],
    )
    parser.add_argument(
        "--collect-search-ctree-backends",
        type=_csv_collect_search_ctree_backends,
        default=["lightzero"],
    )
    parser.add_argument("--disable-death-for-profile", action="store_true")
    parser.add_argument("--detached", action="store_true")
    parser.add_argument(
        "--next-matched-denominator-stock-preset",
        action="store_true",
        help=(
            "Emit the selected stock side of the matched denominator pair: "
            "H100 C512/b64/sim8 stock train_muzero, no RND, profile-only "
            "result capture."
        ),
    )
    parser.add_argument("--matched-denominator-id", default="")
    parser.add_argument("--matched-pair-role", default="stock_reference")
    parser.add_argument("--matched-speed-currency", default=MATCHED_STOCK_SPEED_CURRENCY)
    parser.add_argument("--matched-counterpart-manifest-ref", default="")
    parser.add_argument("--matched-counterpart-row-id", default="001")
    parser.add_argument("--matched-row-purpose", default=MATCHED_DENOMINATOR_ROW_PURPOSE)
    parser.add_argument(
        "--matched-promotion-claim",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stdout-only", action="store_true")
    args = parser.parse_args()
    if args.next_matched_denominator_stock_preset:
        apply_next_matched_denominator_stock_preset(args)
    if not args.computes:
        parser.error("--computes is required unless a preset supplies it")
    if not args.collectors:
        parser.error("--collectors is required unless a preset supplies it")
    return args


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    if args.stdout_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    base = args.output_root / manifest["experiment_id"]
    json_path = base.with_suffix(".json")
    jsonl_path = base.with_suffix(".commands.jsonl")
    sh_path = base.with_suffix(".commands.sh")
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    sh_path.write_text(
        "#!/usr/bin/env sh\n"
        + "\n".join(str(row["command_text"]) for row in manifest["rows"])
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "experiment_id": manifest["experiment_id"],
                "row_count": manifest["row_count"],
                "readback": manifest["readback"],
                "outputs": {
                    "manifest_json": str(json_path),
                    "commands_jsonl": str(jsonl_path),
                    "commands_sh": str(sh_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
