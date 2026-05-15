#!/usr/bin/env python3
"""Build a dry-run manifest for CurvyTron optimizer profile grids.

This script never launches Modal. It emits reviewed `--mode profile` commands
for the current trusted stock LightZero lane:

    train_muzero + source_state_fixed_opponent + frozen checkpoint opponent.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.env.observation_surface_contract import (
    DEFAULT_POLICY_OBSERVATION_BACKEND,
    POLICY_BONUS_RENDER_MODE,
    POLICY_TRAIL_RENDER_MODE,
)
from curvyzero.contracts.curvytron import CURVYTRON_TRAINING_TASK_ID

MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
TASK_ID = CURVYTRON_TRAINING_TASK_ID
SCHEMA_ID = "curvyzero_optimizer_profile_manifest/v0"

DEFAULT_OPPONENT_CHECKPOINT_REF = (
    "training/lightzero-curvytron-visual-survival/"
    "curvytron-dense-ckpt1-iter10000-sanity-20260512a/"
    "checkpoints/lightzero/iteration_32.pth.tar"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_optimizer_profile_manifests")
DEFAULT_BONUS_RENDER_MODE = POLICY_BONUS_RENDER_MODE


@dataclass(frozen=True)
class ProfileRow:
    row_id: str
    family: str
    label: str
    compute: str
    env_manager: str
    collectors: int
    n_episode: int
    batch_size: int
    sims: int
    render_mode: str
    death_mode: str
    bonus_render_mode: str = DEFAULT_BONUS_RENDER_MODE
    policy_observation_backend: str = DEFAULT_POLICY_OBSERVATION_BACKEND
    reward_variant: str = "sparse_outcome"
    opponent_policy_kind: str = "frozen_lightzero_checkpoint"
    seed: int = 304
    source_max_steps: int = 256
    env_telemetry_stride: int = 10000
    max_train_iter: int = 64
    max_env_step: int = 65_536
    save_ckpt_after_iter: int = 9999
    stop_after_learner_train_calls: int = 10
    expected_metrics: tuple[str, ...] = (
        "steps_per_sec",
        "collector_sec",
        "mcts_sec",
        "learner_sec",
    )

    @property
    def disable_death(self) -> bool:
        return self.death_mode == "nodeath"


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


def _render_tag(render_mode: str) -> str:
    if render_mode == POLICY_TRAIL_RENDER_MODE:
        return "browser"
    return render_mode.replace("_", "-")


def _compute_tag(compute: str) -> str:
    return compute.replace("gpu-", "").replace("-", "")


def _family_tag(family: str) -> str:
    tags = {
        "anatomy_base": "anat",
        "collector_width": "width",
        "sim_ladder": "sim",
        "long_render_lens": "render",
        "hardware_ladder": "hw",
        "reward_lens": "reward",
        "collector_width_ext": "wext",
        "high_width_search": "hws",
        "long_render_width": "lrw",
        "short_render_lens": "srl",
        "opponent_ablation": "opp",
        "cpu_shape": "cpu",
        "reward_lens_ext": "rewx",
    }
    return tags.get(family, family.replace("_", "-"))


def _reward_tag(reward_variant: str) -> str:
    tags = {
        "sparse_outcome": "sparse",
        "dense_survival_plus_outcome": "dense",
    }
    return tags.get(reward_variant, reward_variant.replace("_", "-"))


def _first_wave_rows() -> list[ProfileRow]:
    rows: list[ProfileRow] = [
        ProfileRow(
            "01",
            "anatomy_base",
            "base-c1-normal-browser",
            "gpu-l4-t4",
            "base",
            1,
            1,
            16,
            8,
            POLICY_TRAIL_RENDER_MODE,
            "normal",
            env_telemetry_stride=1,
            expected_metrics=(
                "env_step_sec",
                "env_render_rgb_canvas_sec",
                "policy_forward_collect_sec",
                "learner_train_sec",
            ),
        ),
        ProfileRow(
            "02",
            "anatomy_base",
            "base-c1-nodeath-browser",
            "gpu-l4-t4",
            "base",
            1,
            1,
            16,
            8,
            POLICY_TRAIL_RENDER_MODE,
            "nodeath",
            env_telemetry_stride=1,
            expected_metrics=("render_sec", "stack_sec", "mcts_sec", "opponent_sec"),
        ),
    ]
    for index, collectors in enumerate((8, 16, 32, 64, 96), start=4):
        rows.append(
            ProfileRow(
                f"{index:02d}",
                "collector_width",
                f"subproc-c{collectors}-normal-browser",
                "gpu-l4-t4-cpu40",
                "subprocess",
                collectors,
                collectors,
                16,
                8,
                POLICY_TRAIL_RENDER_MODE,
                "normal",
                expected_metrics=("steps_per_sec", "collector_sec", "mcts_sec", "gpu_max_pct"),
            )
        )
    sim_rows = [(32, 4), (32, 16), (32, 32)]
    for offset, (collectors, sims) in enumerate(sim_rows, start=9):
        rows.append(
            ProfileRow(
                f"{offset:02d}",
                "sim_ladder",
                f"subproc-c{collectors}-normal-browser-sim{sims}",
                "gpu-l4-t4-cpu40",
                "subprocess",
                collectors,
                collectors,
                16,
                sims,
                POLICY_TRAIL_RENDER_MODE,
                "normal",
                expected_metrics=("steps_per_sec", "mcts_sec", "mcts_sim_budget", "gpu_max_pct"),
            )
        )
    rows.extend(
        [
            ProfileRow(
                "12",
                "long_render_lens",
                "subproc-c16-nodeath-browser",
                "gpu-l4-t4-cpu40",
                "subprocess",
                16,
                16,
                16,
                8,
                POLICY_TRAIL_RENDER_MODE,
                "nodeath",
                env_telemetry_stride=64,
                expected_metrics=("telem_obs", "telem_opp", "telem_vec", "steps_per_sec"),
            ),
            ProfileRow(
                "14",
                "hardware_ladder",
                "l4cpu40-c64-normal-browser-sim16",
                "gpu-l4-t4-cpu40",
                "subprocess",
                64,
                64,
                16,
                16,
                POLICY_TRAIL_RENDER_MODE,
                "normal",
                expected_metrics=("steps_per_sec", "mcts_sec", "gpu_max_pct", "gpu_mem_mib"),
            ),
            ProfileRow(
                "15",
                "hardware_ladder",
                "h100cpu40-c64-normal-browser-sim16",
                "gpu-h100-cpu40",
                "subprocess",
                64,
                64,
                16,
                16,
                POLICY_TRAIL_RENDER_MODE,
                "normal",
                expected_metrics=("steps_per_sec", "mcts_sec", "gpu_max_pct", "gpu_mem_mib"),
            ),
            ProfileRow(
                "16",
                "hardware_ladder",
                "h100cpu40-c64-normal-browser-sim32",
                "gpu-h100-cpu40",
                "subprocess",
                64,
                64,
                16,
                32,
                POLICY_TRAIL_RENDER_MODE,
                "normal",
                expected_metrics=("steps_per_sec", "mcts_sec", "gpu_max_pct", "gpu_mem_mib"),
            ),
            ProfileRow(
                "17",
                "reward_lens",
                "subproc-c32-normal-browser-dense",
                "gpu-l4-t4-cpu40",
                "subprocess",
                32,
                32,
                16,
                8,
                POLICY_TRAIL_RENDER_MODE,
                "normal",
                reward_variant="dense_survival_plus_outcome",
                expected_metrics=("steps_per_sec", "collector_sec", "mcts_sec", "reward_variant"),
            ),
        ]
    )
    return rows


def _second_wave_rows() -> list[ProfileRow]:
    """Rows chosen after the first function-readback wave."""
    return [
        ProfileRow(
            "S01",
            "collector_width_ext",
            "subproc-c128-normal-browser",
            "gpu-l4-t4-cpu40",
            "subprocess",
            128,
            128,
            16,
            8,
            POLICY_TRAIL_RENDER_MODE,
            "normal",
            expected_metrics=("steps_per_sec", "collector_sec", "worker_oversubscription"),
        ),
        ProfileRow(
            "S02",
            "collector_width_ext",
            "subproc-c160-normal-browser",
            "gpu-l4-t4-cpu40",
            "subprocess",
            160,
            160,
            16,
            8,
            POLICY_TRAIL_RENDER_MODE,
            "normal",
            expected_metrics=("steps_per_sec", "collector_sec", "worker_oversubscription"),
        ),
        ProfileRow(
            "S03",
            "high_width_search",
            "subproc-c96-normal-browser-sim16",
            "gpu-l4-t4-cpu40",
            "subprocess",
            96,
            96,
            16,
            16,
            POLICY_TRAIL_RENDER_MODE,
            "normal",
            expected_metrics=("steps_per_sec", "mcts_sec", "gpu_max_pct"),
        ),
        ProfileRow(
            "S04",
            "long_render_width",
            "subproc-c32-nodeath-browser",
            "gpu-l4-t4-cpu40",
            "subprocess",
            32,
            32,
            16,
            8,
            POLICY_TRAIL_RENDER_MODE,
            "nodeath",
            env_telemetry_stride=128,
            expected_metrics=("steps_per_sec", "obs_sec", "collector_sec"),
        ),
        ProfileRow(
            "S07",
            "opponent_ablation",
            "subproc-c32-normal-fixed-straight",
            "gpu-l4-t4-cpu40",
            "subprocess",
            32,
            32,
            16,
            8,
            POLICY_TRAIL_RENDER_MODE,
            "normal",
            opponent_policy_kind="fixed_straight",
            expected_metrics=("steps_per_sec", "opponent_sec", "collector_sec"),
        ),
        ProfileRow(
            "S08",
            "opponent_ablation",
            "subproc-c96-normal-fixed-straight",
            "gpu-l4-t4-cpu40",
            "subprocess",
            96,
            96,
            16,
            8,
            POLICY_TRAIL_RENDER_MODE,
            "normal",
            opponent_policy_kind="fixed_straight",
            expected_metrics=("steps_per_sec", "opponent_sec", "collector_sec"),
        ),
        ProfileRow(
            "S09",
            "cpu_shape",
            "cpu64-c32-normal-browser",
            "cpu64",
            "subprocess",
            32,
            32,
            16,
            8,
            POLICY_TRAIL_RENDER_MODE,
            "normal",
            expected_metrics=("steps_per_sec", "mcts_sec", "learner_sec"),
        ),
        ProfileRow(
            "S10",
            "reward_lens_ext",
            "subproc-c96-normal-dense",
            "gpu-l4-t4-cpu40",
            "subprocess",
            96,
            96,
            16,
            8,
            POLICY_TRAIL_RENDER_MODE,
            "normal",
            reward_variant="dense_survival_plus_outcome",
            expected_metrics=("steps_per_sec", "reward_variant"),
        ),
    ]


def _command_for_row(
    row: ProfileRow,
    *,
    run_id: str,
    attempt_id: str,
    opponent_checkpoint_ref: str,
    detach: bool,
) -> list[str]:
    command = ["uv", "run", "--extra", "modal", "modal", "run"]
    if detach:
        command.append("--detach")
    command.append("--quiet")
    command.extend(
        [
            "-m",
            MODULE,
            "--mode",
            "profile",
            "--compute",
            row.compute,
            "--seed",
            str(row.seed),
            "--run-id",
            run_id,
            "--attempt-id",
            attempt_id,
            "--env-variant",
            "source_state_fixed_opponent",
            "--reward-variant",
            row.reward_variant,
            "--opponent-policy-kind",
            row.opponent_policy_kind,
            "--no-opponent-use-cuda",
            "--env-manager-type",
            row.env_manager,
            "--collector-env-num",
            str(row.collectors),
            "--n-episode",
            str(row.n_episode),
            "--batch-size",
            str(row.batch_size),
            "--num-simulations",
            str(row.sims),
            "--source-max-steps",
            str(row.source_max_steps),
            "--source-state-trail-render-mode",
            row.render_mode,
            "--source-state-bonus-render-mode",
            row.bonus_render_mode,
            "--policy-observation-backend",
            row.policy_observation_backend,
            "--max-train-iter",
            str(row.max_train_iter),
            "--max-env-step",
            str(row.max_env_step),
            "--save-ckpt-after-iter",
            str(row.save_ckpt_after_iter),
            "--stop-after-learner-train-calls",
            str(row.stop_after_learner_train_calls),
            "--env-telemetry-stride",
            str(row.env_telemetry_stride),
            "--lightzero-eval-freq",
            "0",
            "--skip-lightzero-eval-in-profile",
            "--no-background-eval-enabled",
            "--no-background-gif-enabled",
            "--output-detail",
            "compact",
        ]
    )
    if row.opponent_policy_kind == "frozen_lightzero_checkpoint":
        command.extend(["--opponent-checkpoint-ref", opponent_checkpoint_ref])
    if detach:
        command.append("--profile-spawn")
    if row.disable_death:
        command.append("--disable-death-for-profile")
    return command


def _manifest_row(
    row: ProfileRow,
    *,
    experiment_id: str,
    run_prefix: str,
    attempt_prefix: str,
    opponent_checkpoint_ref: str,
    detach: bool,
) -> dict[str, Any]:
    render = _render_tag(row.render_mode)
    compute = _compute_tag(row.compute)
    manager = "sub" if row.env_manager == "subprocess" else row.env_manager
    run_id = _safe_id(
        (
            f"{run_prefix}-{row.row_id}-{_family_tag(row.family)}-{compute}-{manager}"
            f"-c{row.collectors}-n{row.n_episode}-b{row.batch_size}-sim{row.sims}"
            f"-{row.death_mode}-{render}-{_reward_tag(row.reward_variant)}-sd{row.seed}"
        ),
        label="run_id",
    )
    attempt_id = _safe_id(
        f"{attempt_prefix}-{row.row_id}-{row.label}",
        label="attempt_id",
    )
    command = _command_for_row(
        row,
        run_id=run_id,
        attempt_id=attempt_id,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        detach=detach,
    )
    command_text = shlex.join(command)
    if "two-seat-selfplay" in command_text:
        raise ValueError(f"refusing stale two-seat command in row {row.row_id}")
    result_ref = (
        f"training/{TASK_ID}/{run_id}/attempts/{attempt_id}/train/summary.json"
    )
    return {
        "schema_id": "curvyzero_optimizer_profile_manifest_row/v0",
        "experiment_id": experiment_id,
        "row_id": row.row_id,
        "family": row.family,
        "label": row.label,
        "status": "planned",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "result_ref": result_ref,
        "path_claim": "stock_train_muzero_profile",
        "readback": (
            "modal_function_call_result" if detach else "local_compact_stdout"
        ),
        "command": command,
        "command_text": command_text,
        "hardware": row.compute,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": row.reward_variant,
        "opponent_policy_kind": row.opponent_policy_kind,
        "opponent_checkpoint_ref": (
            opponent_checkpoint_ref
            if row.opponent_policy_kind == "frozen_lightzero_checkpoint"
            else None
        ),
        "opponent_use_cuda": False,
        "env_manager": row.env_manager,
        "collectors": row.collectors,
        "n_episode": row.n_episode,
        "batch_size": row.batch_size,
        "sims": row.sims,
        "render_mode": row.render_mode,
        "bonus_render_mode": row.bonus_render_mode,
        "policy_observation_backend": row.policy_observation_backend,
        "render_mode_role": "current_policy_observation",
        "death_mode": row.death_mode,
        "source_max_steps": row.source_max_steps,
        "max_train_iter": row.max_train_iter,
        "max_env_step": row.max_env_step,
        "save_ckpt_after_iter": row.save_ckpt_after_iter,
        "stop_after_learner_train_calls": row.stop_after_learner_train_calls,
        "env_telemetry_stride": row.env_telemetry_stride,
        "expected_metrics": list(row.expected_metrics),
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    experiment_id = _safe_id(args.experiment_id, label="experiment_id")
    run_prefix = _safe_id(args.run_prefix, label="run_prefix")
    attempt_prefix = _safe_id(args.attempt_prefix, label="attempt_prefix")
    if args.matrix_name == "stock-profile-first-wave-v0":
        profile_rows = _first_wave_rows()
    elif args.matrix_name == "stock-profile-second-wave-v0":
        profile_rows = _second_wave_rows()
    else:
        raise ValueError(
            "matrix_name must be stock-profile-first-wave-v0 or "
            "stock-profile-second-wave-v0"
        )
    rows = [
        _manifest_row(
            row,
            experiment_id=experiment_id,
            run_prefix=run_prefix,
            attempt_prefix=attempt_prefix,
            opponent_checkpoint_ref=args.opponent_checkpoint_ref,
            detach=not args.no_detach,
        )
        for row in profile_rows
    ]
    return {
        "schema_id": SCHEMA_ID,
        "generated_at": _utc_timestamp(),
        "dry_run_only": True,
        "launches_modal": False,
        "matrix_name": args.matrix_name,
        "experiment_id": experiment_id,
        "row_count": len(rows),
        "guardrails": {
            "mode": "profile",
            "forbidden_mode": "two-seat-selfplay",
            "current_policy_observation": "browser_lines + simple_symbols",
            "background_eval_enabled": False,
            "background_gif_enabled": False,
            "opponent_use_cuda": False,
            "checkpoint_ref_is_immutable": "latest" not in args.opponent_checkpoint_ref,
            "profile_stop_after_learner_train_calls": 10,
        },
        "rows": rows,
    }


def _write_outputs(manifest: dict[str, Any], *, output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / str(manifest["experiment_id"])
    json_path = base.with_suffix(".json")
    jsonl_path = base.with_suffix(".commands.jsonl")
    sh_path = base.with_suffix(".commands.sh")
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in manifest["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    sh_path.write_text(
        "#!/usr/bin/env sh\n"
        "# Dry-run review artifact only. Launch rows deliberately after review.\n"
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
        description="Build a dry-run CurvyTron optimizer profile manifest."
    )
    parser.add_argument("--matrix-name", default="stock-profile-first-wave-v0")
    parser.add_argument(
        "--experiment-id",
        default="opt-stock-frozen-profile-first-wave-20260512e",
    )
    parser.add_argument(
        "--run-prefix",
        default="opt-sfp-fw0-20260512e",
    )
    parser.add_argument("--attempt-prefix", default="profile-fw0")
    parser.add_argument(
        "--opponent-checkpoint-ref",
        default=DEFAULT_OPPONENT_CHECKPOINT_REF,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-detach", action="store_true")
    parser.add_argument("--stdout-only", action="store_true")
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
                "experiment_id": manifest["experiment_id"],
                "row_count": manifest["row_count"],
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
