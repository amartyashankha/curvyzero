#!/usr/bin/env python3
"""Print live LightZero Pong eval commands for unevaluated checkpoints."""

from __future__ import annotations

import argparse
import random
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass


VOLUME_NAME = "curvyzero-runs"
TASK_ID = "lightzero-official-visual-pong"
EVAL_MODULE = "curvyzero.infra.modal.lightzero_pong_eval_smoke"
MANIFEST_SUMMARY_SCRIPT = "scripts/summarize_lightzero_pong_eval_manifest.py"
ITERATION_RE = re.compile(r"\biteration_(\d+)\.pth\.tar\b")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
EVAL_SEED_MIN = 0
EVAL_SEED_MAX = (2**31) - 1
EVAL_SEED_RNG_SEED_MAX = (2**63) - 1
DEFAULT_EVAL_SEED_COUNT = 16
DEFAULT_EVAL_SEED_RNG_SEED: int | None = None
SERIOUS_NUM_SIMULATIONS = 50
TELEMETRY_NUM_SIMULATIONS = 5


@dataclass(frozen=True)
class Checkpoint:
    iteration: int
    name: str
    ref: str


@dataclass(frozen=True)
class EvalSeedSelection:
    seeds: list[int]
    sampler_seed: int | None


def _default_checkpoint_dir(run_id: str, attempt_id: str) -> str:
    return (
        f"training/{TASK_ID}/{run_id}/attempts/{attempt_id}"
        "/train/lightzero_exp/ckpt"
    )


def _default_eval_root(run_id: str, attempt_id: str, eval_id: str) -> str:
    return f"training/{TASK_ID}/{run_id}/attempts/{attempt_id}/eval/{eval_id}"


def _modal_volume_ls(volume: str, ref: str, *, required: bool) -> str:
    cmd = ["uv", "run", "--extra", "modal", "modal", "volume", "ls", volume, ref]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return result.stdout
    if required:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    message = result.stderr.strip()
    if message:
        print(
            f"# eval root listing failed; assuming no completed eval dirs: {message}",
            file=sys.stderr,
        )
    return ""


def _iteration_names(listing: str) -> set[str]:
    names: set[str] = set()
    for match in ITERATION_RE.finditer(ANSI_RE.sub("", listing)):
        names.add(match.group(0))
    return names


def _listed_entries(listing: str) -> set[str]:
    entries: set[str] = set()
    for raw_line in ANSI_RE.sub("", listing).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for token in line.split():
            clean = token.rstrip("/")
            if clean and clean not in {".", ".."}:
                entries.add(clean.rsplit("/", 1)[-1])
    return entries


def _checkpoints_from_listing(listing: str, checkpoint_dir: str) -> list[Checkpoint]:
    checkpoints = []
    for name in _iteration_names(listing):
        match = ITERATION_RE.fullmatch(name)
        if match is None:
            continue
        checkpoints.append(
            Checkpoint(
                iteration=int(match.group(1)),
                name=name,
                ref=f"{checkpoint_dir.rstrip('/')}/{name}",
            )
        )
    return sorted(checkpoints, key=lambda checkpoint: checkpoint.iteration)


def _eval_steps(args: argparse.Namespace) -> int:
    if args.eval_pass == "low":
        return args.low_detail_max_eval_steps
    if args.eval_pass == "high":
        return args.high_detail_max_eval_steps
    return args.max_eval_steps


def _eval_seeds(args: argparse.Namespace) -> EvalSeedSelection:
    if args.eval_seeds is None:
        if args.eval_seed_count is None:
            return EvalSeedSelection(seeds=[args.seed], sampler_seed=None)
        sampler_seed = args.eval_seed_rng_seed
        if sampler_seed is None:
            sampler_seed = random.SystemRandom().randrange(
                0, EVAL_SEED_RNG_SEED_MAX + 1
            )
        rng = random.Random(sampler_seed)
        seeds = rng.sample(
            range(EVAL_SEED_MIN, EVAL_SEED_MAX + 1),
            args.eval_seed_count,
        )
        return EvalSeedSelection(seeds=seeds, sampler_seed=sampler_seed)
    seeds: list[int] = []
    for token in args.eval_seeds.split(","):
        clean = token.strip()
        if not clean:
            continue
        try:
            seed = int(clean)
            if seed not in seeds:
                seeds.append(seed)
        except ValueError as exc:
            raise SystemExit(
                f"--eval-seeds must be comma-separated integers: {args.eval_seeds!r}"
            ) from exc
    if not seeds:
        raise SystemExit("--eval-seeds must include at least one integer")
    return EvalSeedSelection(seeds=seeds, sampler_seed=None)


def _step_detail_flag(args: argparse.Namespace) -> tuple[str, str]:
    if args.eval_pass == "low":
        return "--low-detail-step-detail-limit", str(args.low_detail_step_detail_limit)
    if args.eval_pass == "high":
        return "--high-detail-step-detail-limit", str(args.high_detail_step_detail_limit)
    return "--step-detail-limit", str(args.step_detail_limit)


def _steps_flag(args: argparse.Namespace) -> tuple[str, str]:
    if args.eval_pass == "low":
        return "--low-detail-max-eval-steps", str(args.low_detail_max_eval_steps)
    if args.eval_pass == "high":
        return "--high-detail-max-eval-steps", str(args.high_detail_max_eval_steps)
    return "--max-eval-steps", str(args.max_eval_steps)


def _expected_eval_dir(checkpoint: Checkpoint, args: argparse.Namespace, *, seed: int) -> str:
    label = checkpoint.name.removesuffix(".pth.tar")
    return f"{label}_{args.eval_pass}_steps{_eval_steps(args)}_seed{seed}"


def _eval_command(checkpoints: list[Checkpoint], args: argparse.Namespace, *, seeds: list[int]) -> list[str]:
    if not checkpoints:
        raise ValueError("expected at least one checkpoint")
    if not seeds:
        raise ValueError("expected at least one eval seed")
    steps_flag, steps_value = _steps_flag(args)
    detail_flag, detail_value = _step_detail_flag(args)
    command = [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "-m",
        EVAL_MODULE,
        "--compute",
        args.compute,
        "--parallel",
        "--eval-pass",
        args.eval_pass,
        "--eval-id",
        args.eval_id,
        "--seed",
        str(seeds[0]),
        "--checkpoint-refs",
        ",".join(checkpoint.ref for checkpoint in checkpoints),
        "--run-id",
        args.run_id,
        "--attempt-id",
        args.attempt_id,
        steps_flag,
        steps_value,
        "--max-episode-steps",
        str(args.max_episode_steps),
        detail_flag,
        detail_value,
        "--max-env-step",
        str(args.max_env_step),
        "--max-train-iter",
        str(args.max_train_iter),
        "--collector-env-num",
        str(args.collector_env_num),
        "--evaluator-env-num",
        str(args.evaluator_env_num),
        "--num-simulations",
        str(args.num_simulations),
        "--batch-size",
        str(args.batch_size),
        "--update-per-collect",
        str(args.update_per_collect),
        "--game-segment-length",
        str(args.game_segment_length),
        "--no-allow-model-fallback",
        "--run-stock-evaluator",
    ]
    if args.stock_only:
        command.append("--stock-only")
    if len(seeds) > 1:
        command.extend(["--eval-seeds", ",".join(str(seed) for seed in seeds)])
    if args.summary_only:
        command.append("--summary-only")
    if args.quiet_framework_logs:
        command.append("--quiet-framework-logs")
    if args.optimizer_phase_timing:
        command.append("--optimizer-phase-timing")
    if args.slim_manifest:
        command.append("--slim-manifest")
    return command


def _chunked(items: list[Checkpoint], size: int) -> list[list[Checkpoint]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _selected_iterations(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    selected: set[int] = set()
    for token in raw.split(","):
        clean = token.strip()
        if not clean:
            continue
        try:
            selected.add(int(clean))
        except ValueError as exc:
            raise SystemExit(
                f"--selected-iterations must be comma-separated integers: {raw!r}"
            ) from exc
    if not selected:
        raise SystemExit("--selected-iterations must include at least one integer")
    return selected


def _fetch_manifests_command(eval_root: str, args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "volume",
        "get",
        args.volume,
        eval_root,
        str(args.local_manifest_dir),
        "--force",
    ]


def _summary_command(args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        MANIFEST_SUMMARY_SCRIPT,
        "--survival-curve",
        "--survival-aggregate",
        "--format",
        "tsv",
        f"{str(args.local_manifest_dir).rstrip('/')}/**/*.json",
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find iteration_*.pth.tar checkpoints in the Modal Volume and print "
            "eval commands for checkpoints without a matching eval output dir."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--volume", default=VOLUME_NAME)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--eval-root")
    parser.add_argument(
        "--compute",
        default="gpu-l4-t4-cpu40",
        choices=["cpu", "gpu-l4-t4", "gpu-l4-t4-cpu8", "gpu-l4-t4-cpu40"],
    )
    parser.add_argument("--eval-pass", default="low", choices=["low", "high", "custom"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-seeds",
        default=None,
        help=(
            "Comma-separated eval seeds to launch for every selected checkpoint, "
            "e.g. 0,1,2,3. "
            "Use for replay/manual debugging after recording a seed list."
        ),
    )
    parser.add_argument(
        "--eval-seed-count",
        type=int,
        default=None,
        help=(
            "Sample this many unique eval seeds from a wide integer range. "
            f"Default {DEFAULT_EVAL_SEED_COUNT} gives each checkpoint a small "
            "varied seed panel; replay later with the printed --eval-seeds list."
        ),
    )
    parser.add_argument(
        "--eval-seed-rng-seed",
        type=int,
        default=DEFAULT_EVAL_SEED_RNG_SEED,
        help=(
            "Sampler seed for --eval-seed-count. Omit it for a fresh random "
            "seed list; the helper prints the sampler seed and final eval "
            "seed list for replay."
        ),
    )
    parser.add_argument("--low-detail-max-eval-steps", type=int, default=512)
    parser.add_argument("--low-detail-step-detail-limit", type=int, default=8)
    parser.add_argument("--high-detail-max-eval-steps", type=int, default=512)
    parser.add_argument("--high-detail-step-detail-limit", type=int, default=8)
    parser.add_argument("--max-eval-steps", type=int, default=512)
    parser.add_argument("--step-detail-limit", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=512)
    parser.add_argument("--max-env-step", type=int, default=200000)
    parser.add_argument("--max-train-iter", type=int, default=1)
    parser.add_argument("--collector-env-num", type=int, default=8)
    parser.add_argument("--evaluator-env-num", type=int, default=3)
    parser.add_argument(
        "--eval-speed-profile",
        default="serious",
        choices=["serious", "telemetry"],
        help=(
            "Named eval speed posture. serious keeps the full stock search "
            f"default ({SERIOUS_NUM_SIMULATIONS} MCTS simulations/action). "
            "telemetry keeps the same survival-time artifact contract but uses "
            f"{TELEMETRY_NUM_SIMULATIONS} simulations/action unless "
            "--num-simulations is set explicitly."
        ),
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=None,
        help=(
            "MCTS simulations per policy action. Omit to use the selected "
            "--eval-speed-profile default."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--update-per-collect",
        type=int,
        default=-1,
        help=(
            "LightZero update_per_collect for the eval config. Default -1 "
            "restores stock None in the Modal eval helper."
        ),
    )
    parser.add_argument("--game-segment-length", type=int, default=400)
    parser.add_argument(
        "--local-manifest-dir",
        default=None,
        help="Local directory used in the printed fetch and summary commands.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help=(
            "Number of pending checkpoints per Modal run. Default 4 trims "
            "Modal app startup and root-manifest noise while keeping per-call "
            "manifests moderate; use 1 for fastest first checkpoint signal."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only print or launch the first N pending checkpoint/seed eval jobs "
            "after duplicate filtering. Jobs are ordered checkpoint-first, then "
            "seed, so small multi-seed previews keep seed sets together."
        ),
    )
    parser.add_argument(
        "--selected-iterations",
        default=None,
        help=(
            "Optional comma-separated checkpoint iterations to consider before "
            "duplicate filtering, e.g. 0,1000,5000,16829."
        ),
    )
    parser.add_argument(
        "--max-parallel-launches",
        type=int,
        default=64,
        help=(
            "With --execute, keep up to this many local modal run commands in "
            "flight. Default 64 aggressively fans out checkpoint evals."
        ),
    )
    parser.add_argument(
        "--verbose-listings",
        action="store_true",
        help="Print full checkpoint and existing eval-dir listings.",
    )
    parser.add_argument(
        "--skip-eval-root-listing",
        action="store_true",
        help=(
            "Do not list the eval root before planning jobs. Use only for a "
            "known-new eval id or an intentional full relaunch; duplicate "
            "checkpoint/seed dirs will not be skipped."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --summary-only to Modal eval commands so stdout contains only "
            "the compact table and manifest/artifact refs. Use --no-summary-only "
            "to print full result JSON."
        ),
    )
    parser.add_argument(
        "--quiet-framework-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --quiet-framework-logs to Modal eval commands so framework "
            "stdout/stderr from the per-checkpoint eval call is redirected to "
            "/dev/null. Use --no-quiet-framework-logs to keep LightZero/Gym "
            "INFO and warning chatter visible."
        ),
    )
    parser.add_argument(
        "--optimizer-phase-timing",
        action="store_true",
        help=(
            "Pass --optimizer-phase-timing to Modal eval commands so returned "
            "JSON includes opt-in phase timings for eval setup, stock eval, "
            "artifact write, and Volume commit."
        ),
    )
    parser.add_argument(
        "--slim-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --slim-manifest to Modal eval commands so combined root "
            "manifests contain the compact table and artifact refs, not the "
            "full per-job result payloads. Raw per-checkpoint artifacts still "
            "keep full detail. Use --no-slim-manifest for debug manifests."
        ),
    )
    parser.add_argument(
        "--stock-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass --stock-only to Modal eval commands. Default true skips the "
            "duplicate manual rollout and keeps the stock MuZeroEvaluator "
            "episode used for stock_steps_survived. Use --no-stock-only for "
            "manual+stock parity debugging."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.group_size < 1:
        raise SystemExit("--group-size must be >= 1")
    if args.max_parallel_launches < 1:
        raise SystemExit("--max-parallel-launches must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1 when provided")
    if args.eval_seed_count is not None and args.eval_seed_count < 1:
        raise SystemExit("--eval-seed-count must be >= 1 when provided")
    if args.eval_seed_count is None and args.eval_seeds is None:
        args.eval_seed_count = DEFAULT_EVAL_SEED_COUNT
    if args.eval_seed_count is not None and args.eval_seeds is not None:
        raise SystemExit("--eval-seed-count cannot be combined with --eval-seeds")
    if args.eval_seed_rng_seed is not None and args.eval_seeds is not None:
        raise SystemExit("--eval-seed-rng-seed cannot be combined with --eval-seeds")
    if args.eval_seed_rng_seed is not None and args.eval_seed_count is None:
        raise SystemExit("--eval-seed-rng-seed requires --eval-seed-count")
    seed_range_size = EVAL_SEED_MAX - EVAL_SEED_MIN + 1
    if args.eval_seed_count is not None and args.eval_seed_count > seed_range_size:
        raise SystemExit(f"--eval-seed-count must be <= {seed_range_size}")
    if args.num_simulations is None:
        if args.eval_speed_profile == "telemetry":
            args.num_simulations = TELEMETRY_NUM_SIMULATIONS
        else:
            args.num_simulations = SERIOUS_NUM_SIMULATIONS
    if args.num_simulations < 1:
        raise SystemExit("--num-simulations must be >= 1")


def _wait_for_one(
    running: list[tuple[list[int], list[Checkpoint], list[str], subprocess.Popen[str]]],
) -> None:
    seeds, group, command, process = running.pop(0)
    returncode = process.wait()
    labels = ", ".join(checkpoint.name for checkpoint in group)
    if returncode != 0:
        raise SystemExit(
            f"eval command failed with exit code {returncode} for seeds {seeds}, {labels}: "
            f"{shlex.join(command)}"
        )


def _execute_groups(groups: list[tuple[list[int], list[Checkpoint]]], args: argparse.Namespace) -> None:
    running: list[tuple[list[int], list[Checkpoint], list[str], subprocess.Popen[str]]] = []
    for seeds, group in groups:
        command = _eval_command(group, args, seeds=seeds)
        labels = ", ".join(checkpoint.name for checkpoint in group)
        print(f"# launching seeds {','.join(str(seed) for seed in seeds)}: {labels}")
        running.append((seeds, group, command, subprocess.Popen(command, text=True)))
        while len(running) >= args.max_parallel_launches:
            _wait_for_one(running)
    while running:
        _wait_for_one(running)


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    checkpoint_dir = args.checkpoint_dir or _default_checkpoint_dir(
        args.run_id, args.attempt_id
    )
    eval_root = args.eval_root or _default_eval_root(
        args.run_id, args.attempt_id, args.eval_id
    )
    eval_seed_selection = _eval_seeds(args)
    eval_seeds = eval_seed_selection.seeds
    if args.local_manifest_dir is None:
        args.local_manifest_dir = f"artifacts/local/lightzero-eval-manifests/{args.eval_id}"

    checkpoint_listing = _modal_volume_ls(args.volume, checkpoint_dir, required=True)
    checkpoints = _checkpoints_from_listing(checkpoint_listing, checkpoint_dir)
    selected_iterations = _selected_iterations(args.selected_iterations)
    if selected_iterations is not None:
        checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.iteration in selected_iterations
        ]
    eval_root_listing_status = "read"
    if args.skip_eval_root_listing:
        existing_eval_dirs: set[str] = set()
        eval_root_listing_status = "skipped_by_flag"
    elif not checkpoints:
        existing_eval_dirs = set()
        eval_root_listing_status = "skipped_no_selected_checkpoints"
    else:
        eval_listing = _modal_volume_ls(args.volume, eval_root, required=False)
        existing_eval_dirs = _listed_entries(eval_listing)
    pending_pairs = [
        (seed, checkpoint)
        for checkpoint in checkpoints
        for seed in eval_seeds
        if _expected_eval_dir(checkpoint, args, seed=seed) not in existing_eval_dirs
    ]
    if args.limit is not None:
        pending_pairs = pending_pairs[: args.limit]
    grouped_by_checkpoint: dict[Checkpoint, list[int]] = {}
    for seed, checkpoint in pending_pairs:
        grouped_by_checkpoint.setdefault(checkpoint, []).append(seed)
    grouped_by_seed_tuple: dict[tuple[int, ...], list[Checkpoint]] = {}
    for checkpoint, seeds in grouped_by_checkpoint.items():
        grouped_by_seed_tuple.setdefault(tuple(seeds), []).append(checkpoint)
    groups = [
        (list(seed_tuple), group)
        for seed_tuple, checkpoints_for_seeds in grouped_by_seed_tuple.items()
        for group in _chunked(checkpoints_for_seeds, args.group_size)
    ]

    print(f"# checkpoint_dir: {checkpoint_dir}")
    print(f"# eval_root: {eval_root}")
    print(f"# found iteration checkpoints: {len(checkpoints)}")
    if checkpoints and args.verbose_listings:
        print("# available checkpoints: " + ", ".join(checkpoint.name for checkpoint in checkpoints))
    elif checkpoints:
        print(f"# checkpoint range: {checkpoints[0].name} .. {checkpoints[-1].name}")
    if selected_iterations is not None:
        print("# selected iterations: " + ", ".join(str(item) for item in sorted(selected_iterations)))
    if eval_seed_selection.sampler_seed is None:
        print("# eval seed sampler seed: n/a (fixed eval seed list)")
    else:
        print(f"# eval seed sampler seed: {eval_seed_selection.sampler_seed}")
    print("# eval seeds: " + ", ".join(str(seed) for seed in eval_seeds))
    print(
        "# eval speed profile: "
        f"{args.eval_speed_profile} ({args.num_simulations} MCTS simulations/action)"
    )
    if args.eval_speed_profile == "telemetry":
        print(
            "# telemetry note: stock_steps_survived is still recorded, but lower-search "
            "runs are fast trend probes; confirm claims with the serious profile"
        )
    if eval_root_listing_status == "skipped_by_flag":
        print(
            "# eval root listing: skipped; duplicate filtering is disabled for this run"
        )
    elif eval_root_listing_status == "skipped_no_selected_checkpoints":
        print("# eval root listing: skipped; no selected checkpoints to de-duplicate")
    elif existing_eval_dirs and args.verbose_listings:
        print("# existing eval dirs: " + ", ".join(sorted(existing_eval_dirs)))
    elif existing_eval_dirs:
        print(f"# existing eval dirs: {len(existing_eval_dirs)} (use --verbose-listings to print)")
    print(f"# pending checkpoint/seed evals: {len(pending_pairs)}")
    print(f"# eval group size: {args.group_size} checkpoint(s), seeds mapped inside each Modal call")
    print(f"# eval modal calls: {len(groups)}")
    if groups:
        starmap_job_counts = [len(seeds) * len(group) for seeds, group in groups]
        print(
            "# remote Function.starmap jobs per Modal call: "
            f"{min(starmap_job_counts)}..{max(starmap_job_counts)}"
        )
    if args.stock_only:
        print("# strict contract: --no-allow-model-fallback --run-stock-evaluator --stock-only")
    else:
        print("# strict contract: --no-allow-model-fallback --run-stock-evaluator")
        print("# eval mode: full manual+stock debug rollout (--no-stock-only)")
    if args.summary_only:
        print("# stdout mode: --summary-only")
    else:
        print("# stdout mode: full result JSON")
    if args.quiet_framework_logs:
        print("# framework logs: quiet (--quiet-framework-logs)")
    else:
        print("# framework logs: verbose (--no-quiet-framework-logs)")
    if args.optimizer_phase_timing:
        print("# optimizer phase timing: enabled (--optimizer-phase-timing)")
    if args.group_size == 1:
        print("# manifest mode: one root manifest_*.json per checkpoint/seed set Modal call")
    else:
        print("# manifest mode: one root manifest_*.json per checkpoint group/seed set Modal call")
    if args.slim_manifest:
        print("# root manifest detail: slim table/artifact index (--slim-manifest)")
    else:
        print("# root manifest detail: full result payloads (--no-slim-manifest)")
    if not pending_pairs:
        print("# fetch manifests:")
        print(shlex.join(_fetch_manifests_command(eval_root, args)))
        print("# summarize fetched eval root as a stock survival curve:")
        print(
            "# summary input may include manifest_*.json and raw per-episode "
            "JSONs; duplicate artifacts are de-duplicated"
        )
        print(shlex.join(_summary_command(args)))
        return 0

    for index, (seeds, group) in enumerate(groups, start=1):
        expected = ", ".join(
            (
                f"{checkpoint.name}->"
                + "|".join(_expected_eval_dir(checkpoint, args, seed=seed) for seed in seeds)
            )
            for checkpoint in group
        )
        command = _eval_command(group, args, seeds=seeds)
        print(f"# group {index}/{len(groups)} seeds {','.join(str(seed) for seed in seeds)}: {expected}")
        print(shlex.join(command))
    if args.execute:
        print(f"# executing with max_parallel_launches={args.max_parallel_launches}")
        _execute_groups(groups, args)
    print("# fetch manifests after evals finish:")
    print(shlex.join(_fetch_manifests_command(eval_root, args)))
    print("# summarize fetched eval root as a stock survival curve:")
    print(
        "# summary input may include manifest_*.json and raw per-episode "
        "JSONs; duplicate artifacts are de-duplicated"
    )
    print(shlex.join(_summary_command(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
