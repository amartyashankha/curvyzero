#!/usr/bin/env python3
"""Compare stock LightZero MCTS facade with compact CTree array paths.

This is a profile/validation helper, not a trainer. It runs small LightZero
policies over deterministic synthetic CurvyTron observation batches and reports
distributional differences between:

- stock facade: public ``collect_mode.forward`` decoded to compact arrays;
- direct CTree: real model + real CTree MCTS + compact arrays out;
- GPU-latent direct CTree: same CPU CTree decisions, but recurrent latent
  tensors stay on device inside the search loop.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from curvyzero.infra.modal.source_state_batched_observation_boundary_profile import (
    ACTION_COUNT,
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
    LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
    _build_profile_lightzero_policy,
    _LightZeroCollectForwardStackProbe,
)


@dataclass(frozen=True)
class CompareRow:
    seed: int
    impl: str
    action_agreement: float
    mean_visit_l1: float
    max_visit_l1: float
    mean_value_abs_diff: float
    max_value_abs_diff: float
    mean_predicted_value_abs_diff: float
    max_predicted_value_abs_diff: float
    mean_policy_logit_abs_diff: float
    max_policy_logit_abs_diff: float
    stock_illegal_actions: float
    candidate_illegal_actions: float
    stock_policy_cuda: bool
    candidate_policy_cuda: bool
    candidate_gpu_latent_enabled: bool


IMPL_ALIASES = {
    "direct": LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
    "direct_ctree": LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
    "direct_ctree_arrays": LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
    "gpu_latent": LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
    "direct_ctree_gpu_latent": LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
}
ACTION_MASK_SCENARIOS = (
    "random",
    "all_legal",
    "single_legal_cycle",
    "mixed_legal_cycle",
)


def _parse_impls(raw: str) -> list[str]:
    impls: list[str] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        impl = IMPL_ALIASES.get(value)
        if impl is None:
            allowed = ", ".join(sorted(IMPL_ALIASES))
            raise argparse.ArgumentTypeError(
                f"unknown impl {value!r}; expected one of: {allowed}"
            )
        if impl not in impls:
            impls.append(impl)
    if not impls:
        raise argparse.ArgumentTypeError("expected at least one implementation")
    return impls


def _observation(*, seed: int, batch_rows: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(
        0,
        256,
        size=(batch_rows, 2, 4, 64, 64),
        dtype=np.uint8,
    )
    # Add row/player sentinels so accidental row-order swaps are visible.
    for row in range(batch_rows):
        for player in range(2):
            base[row, player, 0, 0, 0] = np.uint8((row * 17 + player * 53) % 256)
    return base


def _action_mask(*, seed: int, batch_rows: int, scenario: str = "random") -> np.ndarray:
    if scenario not in ACTION_MASK_SCENARIOS:
        allowed = ", ".join(ACTION_MASK_SCENARIOS)
        raise ValueError(f"unknown action mask scenario {scenario!r}; expected {allowed}")
    if scenario == "all_legal":
        return np.ones((batch_rows, 2, ACTION_COUNT), dtype=bool)
    if scenario == "single_legal_cycle":
        mask = np.zeros((batch_rows, 2, ACTION_COUNT), dtype=bool)
        for flat_index in range(batch_rows * 2):
            row = flat_index // 2
            player = flat_index % 2
            mask[row, player, flat_index % ACTION_COUNT] = True
        return mask
    if scenario == "mixed_legal_cycle":
        patterns = np.asarray(
            [
                [True, True, True],
                [True, False, True],
                [False, True, True],
                [True, True, False],
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ],
            dtype=bool,
        )
        mask = np.zeros((batch_rows, 2, ACTION_COUNT), dtype=bool)
        for flat_index in range(batch_rows * 2):
            row = flat_index // 2
            player = flat_index % 2
            mask[row, player] = patterns[flat_index % len(patterns)]
        return mask
    rng = np.random.default_rng(seed + 991)
    mask = rng.random((batch_rows, 2, ACTION_COUNT)) > 0.25
    empty = ~mask.any(axis=2)
    mask[empty, 0] = True
    return mask.astype(bool, copy=False)


def _run_impl(
    *,
    impl: str,
    seed: int,
    num_simulations: int,
    observation: np.ndarray,
    action_mask: np.ndarray,
    root_noise_weight: float,
    temperature: float,
    epsilon: float,
    use_cuda: bool,
) -> dict[str, Any]:
    torch = __import__("torch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_meta = _build_profile_lightzero_policy(
        seed=seed,
        use_cuda=use_cuda,
        num_simulations=num_simulations,
        collect_with_pure_policy=False,
        policy_batch_size=max(8, int(observation.shape[0]) * 2),
        max_ticks=8,
    )
    policy = policy_meta["policy"]
    policy._cfg.root_noise_weight = float(root_noise_weight)
    policy._cfg.eps.eps_greedy_exploration_in_collect = True

    probe = _LightZeroCollectForwardStackProbe(
        policy=policy,
        policy_metadata=policy_meta,
        num_simulations=num_simulations,
        temperature=temperature,
        epsilon=epsilon,
        arrays_boundary=True,
        arrays_boundary_impl=impl,
    )
    telemetry = probe.run(observation, action_mask).telemetry
    debug = telemetry["lightzero_mcts_arrays_boundary_debug_arrays"]
    if not bool(debug.get("included", False)):
        raise RuntimeError(
            "debug arrays were not included; reduce --batch-rows so roots <= debug cap"
        )
    return {
        "actions": np.asarray(debug["actions"], dtype=np.int64),
        "visits": np.asarray(debug["visit_distributions"], dtype=np.float32),
        "searched_values": np.asarray(debug["searched_values"], dtype=np.float32),
        "predicted_values": np.asarray(debug["predicted_values"], dtype=np.float32),
        "policy_logits": np.asarray(debug["policy_logits"], dtype=np.float32),
        "illegal_actions": float(telemetry["lightzero_illegal_action_count"]),
        "policy_cuda": bool(telemetry.get("lightzero_policy_cuda", False)),
        "gpu_latent_enabled": bool(
            telemetry.get("lightzero_mcts_arrays_boundary_gpu_latent_enabled", False)
        ),
    }


def compare_seed(
    *,
    seed: int,
    batch_rows: int,
    num_simulations: int,
    root_noise_weight: float,
    temperature: float,
    epsilon: float,
    use_cuda: bool,
    impls: list[str],
    action_mask_scenario: str,
) -> list[CompareRow]:
    observation = _observation(seed=seed, batch_rows=batch_rows)
    action_mask = _action_mask(
        seed=seed,
        batch_rows=batch_rows,
        scenario=action_mask_scenario,
    )

    stock = _run_impl(
        impl="stock_facade",
        seed=seed,
        num_simulations=num_simulations,
        observation=observation,
        action_mask=action_mask,
        root_noise_weight=root_noise_weight,
        temperature=temperature,
        epsilon=epsilon,
        use_cuda=use_cuda,
    )
    rows: list[CompareRow] = []
    for impl in impls:
        candidate = _run_impl(
            impl=impl,
            seed=seed,
            num_simulations=num_simulations,
            observation=observation,
            action_mask=action_mask,
            root_noise_weight=root_noise_weight,
            temperature=temperature,
            epsilon=epsilon,
            use_cuda=use_cuda,
        )
        visit_l1 = np.abs(stock["visits"] - candidate["visits"]).sum(axis=1)
        value_abs = np.abs(stock["searched_values"] - candidate["searched_values"])
        predicted_value_abs = np.abs(
            stock["predicted_values"] - candidate["predicted_values"]
        )
        policy_logit_abs = np.abs(stock["policy_logits"] - candidate["policy_logits"])
        action_agreement = float(np.mean(stock["actions"] == candidate["actions"]))
        rows.append(
            CompareRow(
                seed=seed,
                impl=impl,
                action_agreement=action_agreement,
                mean_visit_l1=float(visit_l1.mean()),
                max_visit_l1=float(visit_l1.max()),
                mean_value_abs_diff=float(value_abs.mean()),
                max_value_abs_diff=float(value_abs.max()),
                mean_predicted_value_abs_diff=float(predicted_value_abs.mean()),
                max_predicted_value_abs_diff=float(predicted_value_abs.max()),
                mean_policy_logit_abs_diff=float(policy_logit_abs.mean()),
                max_policy_logit_abs_diff=float(policy_logit_abs.max()),
                stock_illegal_actions=float(stock["illegal_actions"]),
                candidate_illegal_actions=float(candidate["illegal_actions"]),
                stock_policy_cuda=bool(stock["policy_cuda"]),
                candidate_policy_cuda=bool(candidate["policy_cuda"]),
                candidate_gpu_latent_enabled=bool(candidate["gpu_latent_enabled"]),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=20260521)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--batch-rows", type=int, default=4)
    parser.add_argument("--num-simulations", type=int, default=8)
    parser.add_argument(
        "--impls",
        type=_parse_impls,
        default=[
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT,
        ],
        help="Comma-separated candidate implementations to compare against stock.",
    )
    parser.add_argument("--root-noise-weight", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--action-mask-scenario",
        choices=ACTION_MASK_SCENARIOS,
        default="random",
        help="Synthetic action-mask pattern for forced-case gates.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Request CUDA policies. Use with --require-cuda to fail instead of falling back.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless every stock and candidate policy actually ran on CUDA.",
    )
    parser.add_argument(
        "--require-gpu-latent-enabled",
        action="store_true",
        help="Fail unless every direct_ctree_gpu_latent row reports the GPU-latent path.",
    )
    parser.add_argument(
        "--strict-exact",
        action="store_true",
        help="Default unset thresholds to exact equality for actions, visits, and values.",
    )
    parser.add_argument("--max-mean-visit-l1", type=float, default=None)
    parser.add_argument("--max-max-visit-l1", type=float, default=None)
    parser.add_argument("--min-action-agreement", type=float, default=None)
    parser.add_argument("--max-max-value-abs-diff", type=float, default=None)
    parser.add_argument("--max-max-predicted-value-abs-diff", type=float, default=None)
    parser.add_argument("--max-max-policy-logit-abs-diff", type=float, default=None)
    args = parser.parse_args()

    if args.batch_rows * 2 > 16:
        raise SystemExit("batch rows must create at most 16 roots for debug arrays")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive")
    if not 0.0 <= args.epsilon <= 1.0:
        raise SystemExit("--epsilon must be in [0, 1]")
    if args.require_cuda and not args.use_cuda:
        raise SystemExit("--require-cuda needs --use-cuda")
    if args.strict_exact:
        if args.max_mean_visit_l1 is None:
            args.max_mean_visit_l1 = 0.0
        if args.max_max_visit_l1 is None:
            args.max_max_visit_l1 = 0.0
        if args.min_action_agreement is None:
            args.min_action_agreement = 1.0
        if args.max_max_value_abs_diff is None:
            args.max_max_value_abs_diff = 0.0
        if args.max_max_predicted_value_abs_diff is None:
            args.max_max_predicted_value_abs_diff = 0.0
        if args.max_max_policy_logit_abs_diff is None:
            args.max_max_policy_logit_abs_diff = 0.0

    rows = [
        row
        for index in range(args.seeds)
        for row in compare_seed(
            seed=args.seed_start + index,
            batch_rows=args.batch_rows,
            num_simulations=args.num_simulations,
            root_noise_weight=args.root_noise_weight,
            temperature=args.temperature,
            epsilon=args.epsilon,
            use_cuda=args.use_cuda,
            impls=args.impls,
            action_mask_scenario=args.action_mask_scenario,
        )
    ]
    by_impl: dict[str, dict[str, float]] = {}
    for impl in args.impls:
        impl_rows = [row for row in rows if row.impl == impl]
        by_impl[impl] = {
            "mean_action_agreement": float(
                np.mean([row.action_agreement for row in impl_rows])
            ),
            "min_action_agreement": float(
                np.min([row.action_agreement for row in impl_rows])
            ),
            "mean_visit_l1": float(np.mean([row.mean_visit_l1 for row in impl_rows])),
            "max_visit_l1": float(np.max([row.max_visit_l1 for row in impl_rows])),
            "mean_value_abs_diff": float(
                np.mean([row.mean_value_abs_diff for row in impl_rows])
            ),
            "max_value_abs_diff": float(
                np.max([row.max_value_abs_diff for row in impl_rows])
            ),
            "mean_predicted_value_abs_diff": float(
                np.mean([row.mean_predicted_value_abs_diff for row in impl_rows])
            ),
            "max_predicted_value_abs_diff": float(
                np.max([row.max_predicted_value_abs_diff for row in impl_rows])
            ),
            "mean_policy_logit_abs_diff": float(
                np.mean([row.mean_policy_logit_abs_diff for row in impl_rows])
            ),
            "max_policy_logit_abs_diff": float(
                np.max([row.max_policy_logit_abs_diff for row in impl_rows])
            ),
            "stock_illegal_actions": float(
                sum(row.stock_illegal_actions for row in impl_rows)
            ),
            "candidate_illegal_actions": float(
                sum(row.candidate_illegal_actions for row in impl_rows)
            ),
            "stock_cuda_rows": float(sum(row.stock_policy_cuda for row in impl_rows)),
            "candidate_cuda_rows": float(
                sum(row.candidate_policy_cuda for row in impl_rows)
            ),
            "candidate_gpu_latent_enabled_rows": float(
                sum(row.candidate_gpu_latent_enabled for row in impl_rows)
            ),
        }
    primary = by_impl[args.impls[0]]
    summary = {
        "schema_id": "curvyzero_direct_ctree_stock_compare/v0",
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "impls": args.impls,
        "batch_rows": args.batch_rows,
        "root_count_per_seed": args.batch_rows * 2,
        "num_simulations": args.num_simulations,
        "root_noise_weight": args.root_noise_weight,
        "temperature": args.temperature,
        "epsilon": args.epsilon,
        "action_mask_scenario": args.action_mask_scenario,
        "use_cuda_requested": bool(args.use_cuda),
        "require_cuda": bool(args.require_cuda),
        "require_gpu_latent_enabled": bool(args.require_gpu_latent_enabled),
        "by_impl": by_impl,
        # Backwards-compatible top-level fields for the first requested impl.
        "mean_action_agreement": primary["mean_action_agreement"],
        "min_action_agreement": primary["min_action_agreement"],
        "mean_visit_l1": primary["mean_visit_l1"],
        "max_visit_l1": primary["max_visit_l1"],
        "mean_value_abs_diff": primary["mean_value_abs_diff"],
        "max_value_abs_diff": primary["max_value_abs_diff"],
        "mean_predicted_value_abs_diff": primary["mean_predicted_value_abs_diff"],
        "max_predicted_value_abs_diff": primary["max_predicted_value_abs_diff"],
        "mean_policy_logit_abs_diff": primary["mean_policy_logit_abs_diff"],
        "max_policy_logit_abs_diff": primary["max_policy_logit_abs_diff"],
        "stock_illegal_actions": primary["stock_illegal_actions"],
        "direct_illegal_actions": by_impl.get(
            LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE,
            primary,
        )["candidate_illegal_actions"],
        "candidate_illegal_actions": float(
            sum(row.candidate_illegal_actions for row in rows)
        ),
        "rows": [row.__dict__ for row in rows],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    failures: list[str] = []
    for impl, impl_summary in by_impl.items():
        if (
            args.max_mean_visit_l1 is not None
            and impl_summary["mean_visit_l1"] > args.max_mean_visit_l1
        ):
            failures.append(
                f"{impl} mean_visit_l1 {impl_summary['mean_visit_l1']:.6f} "
                f"> {args.max_mean_visit_l1}"
            )
        if (
            args.max_max_visit_l1 is not None
            and impl_summary["max_visit_l1"] > args.max_max_visit_l1
        ):
            failures.append(
                f"{impl} max_visit_l1 {impl_summary['max_visit_l1']:.6f} "
                f"> {args.max_max_visit_l1}"
            )
        if (
            args.min_action_agreement is not None
            and impl_summary["mean_action_agreement"] < args.min_action_agreement
        ):
            failures.append(
                f"{impl} mean_action_agreement "
                f"{impl_summary['mean_action_agreement']:.6f} "
                f"< {args.min_action_agreement}"
            )
        if (
            args.max_max_value_abs_diff is not None
            and impl_summary["max_value_abs_diff"] > args.max_max_value_abs_diff
        ):
            failures.append(
                f"{impl} max_value_abs_diff "
                f"{impl_summary['max_value_abs_diff']:.6g} "
                f"> {args.max_max_value_abs_diff}"
            )
        if (
            args.max_max_predicted_value_abs_diff is not None
            and impl_summary["max_predicted_value_abs_diff"]
            > args.max_max_predicted_value_abs_diff
        ):
            failures.append(
                f"{impl} max_predicted_value_abs_diff "
                f"{impl_summary['max_predicted_value_abs_diff']:.6g} "
                f"> {args.max_max_predicted_value_abs_diff}"
            )
        if (
            args.max_max_policy_logit_abs_diff is not None
            and impl_summary["max_policy_logit_abs_diff"]
            > args.max_max_policy_logit_abs_diff
        ):
            failures.append(
                f"{impl} max_policy_logit_abs_diff "
                f"{impl_summary['max_policy_logit_abs_diff']:.6g} "
                f"> {args.max_max_policy_logit_abs_diff}"
            )
        if args.require_cuda:
            expected_rows = float(args.seeds)
            if impl_summary["stock_cuda_rows"] != expected_rows:
                failures.append(f"{impl} stock policy did not run on CUDA")
            if impl_summary["candidate_cuda_rows"] != expected_rows:
                failures.append(f"{impl} candidate policy did not run on CUDA")
        if (
            args.require_gpu_latent_enabled
            and impl == LIGHTZERO_MCTS_ARRAYS_BOUNDARY_IMPL_DIRECT_CTREE_GPU_LATENT
        ):
            expected_rows = float(args.seeds)
            if impl_summary["candidate_gpu_latent_enabled_rows"] != expected_rows:
                failures.append(f"{impl} GPU-latent path did not run for every seed")
        if (
            impl_summary["candidate_illegal_actions"]
            or impl_summary["stock_illegal_actions"]
        ):
            failures.append(f"{impl} decoded illegal actions")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
