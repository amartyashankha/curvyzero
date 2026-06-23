#!/usr/bin/env python3
"""Benchmark LightZero MuZero CTree list ABI without env or model work.

This is an optimizer falsifier. It does not train, does not step CurvyTron, and
does not call a neural network. It prices the current LightZero CTree boundary:
root construction, list-shaped prepare/backprop payloads, traverse/backprop,
and output extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


ACTION_COUNT = 3
CTREE_FLAT_A3_BACKEND = "ctree-flat-a3"


def _parse_csv_ints(raw: str, *, name: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        value = part.strip()
        if not value:
            continue
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"{name} values must be positive")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


def _parse_csv_strings(raw: str, *, allowed: set[str], name: str) -> list[str]:
    values: list[str] = []
    for part in str(raw).split(","):
        value = part.strip()
        if not value:
            continue
        if value not in allowed:
            expected = ", ".join(sorted(allowed))
            raise argparse.ArgumentTypeError(
                f"unknown {name} {value!r}; expected one of: {expected}"
            )
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


def _legal_actions(root_count: int, legal_profile: str) -> list[list[int]]:
    if legal_profile == "all3":
        return [[0, 1, 2] for _ in range(root_count)]
    if legal_profile == "single1":
        return [[index % ACTION_COUNT] for index in range(root_count)]
    if legal_profile == "mixed_2of3":
        patterns = ((0, 1), (1, 2), (0, 2))
        return [list(patterns[index % len(patterns)]) for index in range(root_count)]
    raise ValueError(f"unknown legal_profile {legal_profile!r}")


def _synthetic_arrays(
    *,
    root_count: int,
    simulations: int,
    action_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    root_reward = rng.normal(0.0, 0.05, size=(root_count,)).astype(np.float32)
    root_policy_logits = rng.normal(0.0, 0.2, size=(root_count, action_count)).astype(np.float32)
    reward = rng.normal(0.0, 0.05, size=(simulations, root_count)).astype(np.float32)
    value = rng.normal(0.0, 0.2, size=(simulations, root_count)).astype(np.float32)
    policy_logits = rng.normal(
        0.0,
        0.2,
        size=(simulations, root_count, action_count),
    ).astype(np.float32)
    return {
        "root_reward": root_reward,
        "root_policy_logits": root_policy_logits,
        "reward": reward,
        "value": value,
        "policy_logits": policy_logits,
    }


def _root_noises(
    *,
    root_count: int,
    action_count: int,
    mode: str,
    alpha: float,
    seed: int,
) -> np.ndarray:
    if mode == "zero":
        return np.zeros((root_count, action_count), dtype=np.float32)
    if mode == "dirichlet":
        rng = np.random.default_rng(seed)
        return rng.dirichlet(
            np.full(action_count, float(alpha), dtype=np.float32),
            size=root_count,
        ).astype(np.float32)
    raise ValueError(f"unknown root_noise {mode!r}")


def _import_mz_tree(backend: str) -> Any:
    if backend == CTREE_FLAT_A3_BACKEND:
        try:
            from curvyzero.vendor.lightzero_ctree_a3.ctree_muzero import mz_tree_a3
        except ImportError as exc:
            raise RuntimeError(
                "ctree-flat-a3 requires the opt-in vendored extension. Build it with: "
                "uv run --with Cython --with numpy python "
                "scripts/build_lightzero_ctree_a3.py build_ext --inplace"
            ) from exc
        return mz_tree_a3

    from lzero.mcts.ctree.ctree_muzero import mz_tree

    return mz_tree


def _run_ctree_list_case(
    *,
    backend: str,
    root_count: int,
    simulations: int,
    iterations: int,
    warmup: int,
    legal_profile: str,
    root_noise: str,
    root_noise_weight: float,
    root_dirichlet_alpha: float,
    pb_c_base: int,
    pb_c_init: float,
    discount_factor: float,
    value_delta_max: float,
    seed: int,
    torch_device: str,
    flat_a3_parity_check: bool,
) -> dict[str, Any]:
    mz_tree = _import_mz_tree(backend)

    if backend not in {"ctree-list", "ctree-torch-d2h", CTREE_FLAT_A3_BACKEND}:
        raise ValueError(f"unsupported CTree backend {backend!r}")
    legal_actions = _legal_actions(root_count, legal_profile)
    arrays = _synthetic_arrays(
        root_count=root_count,
        simulations=simulations,
        action_count=ACTION_COUNT,
        seed=seed,
    )
    noises_array = _root_noises(
        root_count=root_count,
        action_count=ACTION_COUNT,
        mode=root_noise,
        alpha=root_dirichlet_alpha,
        seed=seed + 17,
    )
    to_play = [-1 for _ in range(root_count)]
    measured = {
        "root_ctor_sec": 0.0,
        "root_payload_listify_sec": 0.0,
        "roots_prepare_sec": 0.0,
        "traverse_sec": 0.0,
        "sim_payload_listify_sec": 0.0,
        "flat_payload_view_sec": 0.0,
        "backpropagate_sec": 0.0,
        "get_distributions_sec": 0.0,
        "get_values_sec": 0.0,
        "output_array_cast_sec": 0.0,
        "torch_payload_d2h_sec": 0.0,
    }
    torch_payload_d2h_bytes = 0
    resolved_torch_device = "none"
    torch_payload = None
    if backend == "ctree-torch-d2h":
        import torch

        resolved_torch_device = _resolve_torch_device(torch, torch_device)
        device = torch.device(resolved_torch_device)
        torch_payload = torch.cat(
            (
                torch.as_tensor(arrays["reward"], dtype=torch.float32, device=device)[..., None],
                torch.as_tensor(arrays["value"], dtype=torch.float32, device=device)[..., None],
                torch.as_tensor(arrays["policy_logits"], dtype=torch.float32, device=device),
            ),
            dim=2,
        )
    checksum_visit = 0.0
    checksum_value = 0.0
    traverse_calls = 0
    backpropagate_calls = 0
    started_total = time.perf_counter()
    for iteration in range(iterations + warmup):
        record = iteration >= warmup
        started = time.perf_counter()
        roots = mz_tree.Roots(root_count, legal_actions)
        if record:
            measured["root_ctor_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        reward_roots = arrays["root_reward"].tolist()
        root_policy_logits = arrays["root_policy_logits"].tolist()
        noises = noises_array.tolist()
        if record:
            measured["root_payload_listify_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        roots.prepare(root_noise_weight, noises, reward_roots, root_policy_logits, to_play)
        if record:
            measured["roots_prepare_sec"] += time.perf_counter() - started

        min_max_stats = mz_tree.MinMaxStatsList(root_count)
        min_max_stats.set_delta(value_delta_max)
        for simulation in range(simulations):
            results = mz_tree.ResultsWrapper(num=root_count)
            started = time.perf_counter()
            (
                _latent_path_indices,
                _latent_batch_indices,
                last_actions,
                virtual_to_play,
            ) = mz_tree.batch_traverse(
                roots,
                pb_c_base,
                pb_c_init,
                discount_factor,
                min_max_stats,
                results,
                to_play,
            )
            if len(last_actions) != root_count:
                raise RuntimeError("batch_traverse did not return one action per root")
            if record:
                measured["traverse_sec"] += time.perf_counter() - started
                traverse_calls += 1

            if backend == CTREE_FLAT_A3_BACKEND:
                model_output_np = None
                started = time.perf_counter()
                reward_batch = np.ascontiguousarray(
                    arrays["reward"][simulation],
                    dtype=np.float32,
                )
                value_batch = np.ascontiguousarray(
                    arrays["value"][simulation],
                    dtype=np.float32,
                )
                policy_batch = np.ascontiguousarray(
                    arrays["policy_logits"][simulation],
                    dtype=np.float32,
                )
                if record:
                    measured["flat_payload_view_sec"] += time.perf_counter() - started
            elif torch_payload is None:
                model_output_np = None
                started = time.perf_counter()
                reward_batch = arrays["reward"][simulation].tolist()
                value_batch = arrays["value"][simulation].tolist()
                policy_batch = arrays["policy_logits"][simulation].tolist()
            else:
                started = time.perf_counter()
                model_output_np = _torch_payload_to_numpy(torch_payload[simulation])
                if record:
                    measured["torch_payload_d2h_sec"] += time.perf_counter() - started
                    torch_payload_d2h_bytes += int(model_output_np.nbytes)
                started = time.perf_counter()
                reward_batch = model_output_np[:, 0].tolist()
                value_batch = model_output_np[:, 1].tolist()
                policy_batch = model_output_np[:, 2:].tolist()
            if record and backend != CTREE_FLAT_A3_BACKEND:
                measured["sim_payload_listify_sec"] += time.perf_counter() - started

            started = time.perf_counter()
            if backend == CTREE_FLAT_A3_BACKEND:
                mz_tree.batch_backpropagate_flat_a3(
                    simulation + 1,
                    discount_factor,
                    reward_batch,
                    value_batch,
                    policy_batch,
                    min_max_stats,
                    results,
                )
            else:
                mz_tree.batch_backpropagate(
                    simulation + 1,
                    discount_factor,
                    reward_batch,
                    value_batch,
                    policy_batch,
                    min_max_stats,
                    results,
                    virtual_to_play,
                )
            if record:
                measured["backpropagate_sec"] += time.perf_counter() - started
                backpropagate_calls += 1

        started = time.perf_counter()
        visits_raw = roots.get_distributions()
        if record:
            measured["get_distributions_sec"] += time.perf_counter() - started
        started = time.perf_counter()
        values_raw = roots.get_values()
        if record:
            measured["get_values_sec"] += time.perf_counter() - started
        started = time.perf_counter()
        visits = _visits_to_full_action_space(visits_raw, legal_actions)
        values = np.asarray(values_raw, dtype=np.float32)
        if visits.shape != (root_count, ACTION_COUNT):
            raise RuntimeError(f"unexpected visit shape {visits.shape!r}")
        if values.shape != (root_count,):
            raise RuntimeError(f"unexpected values shape {values.shape!r}")
        if not np.isfinite(values).all():
            raise RuntimeError("CTree values contain non-finite values")
        if bool((visits.sum(axis=1) <= 0).any()):
            raise RuntimeError("CTree visit rows must have positive mass")
        if record:
            measured["output_array_cast_sec"] += time.perf_counter() - started
            checksum_visit = float(visits.sum())
            checksum_value = float(values.sum())
    total_sec = time.perf_counter() - started_total
    # Warmup work is included in started_total above. Remove it by rerunning the
    # sum of measured buckets, which is what matters for no-model attribution.
    measured_total = float(sum(measured.values()))
    boundary_sec = float(
        measured["root_payload_listify_sec"]
        + measured["sim_payload_listify_sec"]
        + measured["flat_payload_view_sec"]
        + measured["get_distributions_sec"]
        + measured["get_values_sec"]
        + measured["output_array_cast_sec"]
    )
    core_sec = float(measured["traverse_sec"] + measured["backpropagate_sec"])
    return {
        "backend": backend,
        "torch_device": resolved_torch_device,
        "root_count": root_count,
        "simulations": simulations,
        "iterations": iterations,
        "warmup": warmup,
        "legal_profile": legal_profile,
        "root_noise": root_noise,
        "total_sec": measured_total,
        "wall_with_warmup_sec": total_sec,
        "roots_per_sec": float(iterations * root_count / measured_total)
        if measured_total > 0.0
        else 0.0,
        "nodes_per_sec": float(iterations * root_count * simulations / measured_total)
        if measured_total > 0.0
        else 0.0,
        "boundary_sec": boundary_sec,
        "ctree_core_sec": core_sec,
        "boundary_fraction": boundary_sec / measured_total if measured_total > 0.0 else 0.0,
        "core_fraction": core_sec / measured_total if measured_total > 0.0 else 0.0,
        "traverse_calls": traverse_calls,
        "backpropagate_calls": backpropagate_calls,
        "torch_payload_d2h_bytes": torch_payload_d2h_bytes,
        "checksum_visit": checksum_visit,
        "checksum_value": checksum_value,
        "flat_a3_parity": _flat_a3_parity_check(
            arrays=arrays,
            legal_actions=legal_actions,
            root_count=root_count,
            simulations=simulations,
            root_noise_weight=root_noise_weight,
            noises_array=noises_array,
            to_play=to_play,
            pb_c_base=pb_c_base,
            pb_c_init=pb_c_init,
            discount_factor=discount_factor,
            value_delta_max=value_delta_max,
        )
        if backend == CTREE_FLAT_A3_BACKEND and flat_a3_parity_check
        else None,
        **measured,
    }


def _flat_a3_parity_check(
    *,
    arrays: dict[str, np.ndarray],
    legal_actions: list[list[int]],
    root_count: int,
    simulations: int,
    root_noise_weight: float,
    noises_array: np.ndarray,
    to_play: list[int],
    pb_c_base: int,
    pb_c_init: float,
    discount_factor: float,
    value_delta_max: float,
) -> dict[str, Any]:
    flat_tree = _import_mz_tree(CTREE_FLAT_A3_BACKEND)
    flat_tree.set_deterministic_tie_breaking(True)
    try:
        reference_visits, reference_values = _run_ctree_once_for_parity(
            flat_tree,
            arrays=arrays,
            legal_actions=legal_actions,
            root_count=root_count,
            simulations=simulations,
            root_noise_weight=root_noise_weight,
            noises_array=noises_array,
            to_play=list(to_play),
            pb_c_base=pb_c_base,
            pb_c_init=pb_c_init,
            discount_factor=discount_factor,
            value_delta_max=value_delta_max,
            flat_a3=False,
        )
        flat_visits, flat_values = _run_ctree_once_for_parity(
            flat_tree,
            arrays=arrays,
            legal_actions=legal_actions,
            root_count=root_count,
            simulations=simulations,
            root_noise_weight=root_noise_weight,
            noises_array=noises_array,
            to_play=list(to_play),
            pb_c_base=pb_c_base,
            pb_c_init=pb_c_init,
            discount_factor=discount_factor,
            value_delta_max=value_delta_max,
            flat_a3=True,
        )
    finally:
        flat_tree.set_deterministic_tie_breaking(False)
    visit_abs_diff = np.abs(reference_visits - flat_visits)
    value_abs_diff = np.abs(reference_values - flat_values)
    return {
        "reference_backend": "vendored-ctree-list-deterministic-ties",
        "max_visit_abs_diff": float(visit_abs_diff.max(initial=0.0)),
        "max_value_abs_diff": float(value_abs_diff.max(initial=0.0)),
        "visit_sum_equal": bool(
            np.array_equal(reference_visits.sum(axis=1), flat_visits.sum(axis=1))
        ),
        "exact_visit_match": bool(np.array_equal(reference_visits, flat_visits)),
        "value_allclose_1e_6": bool(np.allclose(reference_values, flat_values, atol=1e-6)),
    }


def _run_ctree_once_for_parity(
    mz_tree: Any,
    *,
    arrays: dict[str, np.ndarray],
    legal_actions: list[list[int]],
    root_count: int,
    simulations: int,
    root_noise_weight: float,
    noises_array: np.ndarray,
    to_play: list[int],
    pb_c_base: int,
    pb_c_init: float,
    discount_factor: float,
    value_delta_max: float,
    flat_a3: bool,
) -> tuple[np.ndarray, np.ndarray]:
    roots = mz_tree.Roots(root_count, legal_actions)
    roots.prepare(
        root_noise_weight,
        noises_array.tolist(),
        arrays["root_reward"].tolist(),
        arrays["root_policy_logits"].tolist(),
        to_play,
    )
    min_max_stats = mz_tree.MinMaxStatsList(root_count)
    min_max_stats.set_delta(value_delta_max)
    for simulation in range(simulations):
        results = mz_tree.ResultsWrapper(num=root_count)
        (
            _latent_path_indices,
            _latent_batch_indices,
            _last_actions,
            virtual_to_play,
        ) = mz_tree.batch_traverse(
            roots,
            pb_c_base,
            pb_c_init,
            discount_factor,
            min_max_stats,
            results,
            to_play,
        )
        if flat_a3:
            mz_tree.batch_backpropagate_flat_a3(
                simulation + 1,
                discount_factor,
                np.ascontiguousarray(arrays["reward"][simulation], dtype=np.float32),
                np.ascontiguousarray(arrays["value"][simulation], dtype=np.float32),
                np.ascontiguousarray(arrays["policy_logits"][simulation], dtype=np.float32),
                min_max_stats,
                results,
            )
        else:
            mz_tree.batch_backpropagate(
                simulation + 1,
                discount_factor,
                arrays["reward"][simulation].tolist(),
                arrays["value"][simulation].tolist(),
                arrays["policy_logits"][simulation].tolist(),
                min_max_stats,
                results,
                virtual_to_play,
            )
    visits = _visits_to_full_action_space(roots.get_distributions(), legal_actions)
    values = np.asarray(roots.get_values(), dtype=np.float32)
    return visits, values


def _run_fake_flat_case(
    *,
    root_count: int,
    simulations: int,
    iterations: int,
    warmup: int,
    legal_profile: str,
    seed: int,
) -> dict[str, Any]:
    legal_actions = _legal_actions(root_count, legal_profile)
    arrays = _synthetic_arrays(
        root_count=root_count,
        simulations=simulations,
        action_count=ACTION_COUNT,
        seed=seed,
    )
    legal_mask = np.zeros((root_count, ACTION_COUNT), dtype=bool)
    for root, actions in enumerate(legal_actions):
        legal_mask[root, actions] = True
    visit_counts = np.zeros((root_count, ACTION_COUNT), dtype=np.float32)
    value_sum = np.zeros((root_count, ACTION_COUNT), dtype=np.float32)
    measured = {"search_update_sec": 0.0, "output_array_cast_sec": 0.0}
    checksum_visit = 0.0
    checksum_value = 0.0
    started_total = time.perf_counter()
    for iteration in range(iterations + warmup):
        record = iteration >= warmup
        visit_counts.fill(0.0)
        value_sum.fill(0.0)
        for simulation in range(simulations):
            started = time.perf_counter()
            score = (
                arrays["policy_logits"][simulation]
                + arrays["value"][simulation, :, None] * np.float32(0.01)
                - visit_counts * np.float32(0.001)
            )
            score = np.where(legal_mask, score, -1.0e9)
            action = np.argmax(score, axis=1)
            visit_counts[np.arange(root_count), action] += 1.0
            value_sum[np.arange(root_count), action] += arrays["value"][simulation]
            if record:
                measured["search_update_sec"] += time.perf_counter() - started
        started = time.perf_counter()
        values = np.divide(
            value_sum.sum(axis=1),
            np.maximum(visit_counts.sum(axis=1), 1.0),
        ).astype(np.float32)
        visits = visit_counts.astype(np.float32, copy=True)
        if bool((visits[~legal_mask] > 0).any()):
            raise RuntimeError("fake-flat emitted illegal visit mass")
        if record:
            measured["output_array_cast_sec"] += time.perf_counter() - started
            checksum_visit = float(visits.sum())
            checksum_value = float(values.sum())
    wall_with_warmup_sec = time.perf_counter() - started_total
    total_sec = float(sum(measured.values()))
    return {
        "backend": "fake-flat",
        "root_count": root_count,
        "simulations": simulations,
        "iterations": iterations,
        "warmup": warmup,
        "legal_profile": legal_profile,
        "root_noise": "none",
        "total_sec": total_sec,
        "wall_with_warmup_sec": wall_with_warmup_sec,
        "roots_per_sec": float(iterations * root_count / total_sec) if total_sec > 0 else 0.0,
        "nodes_per_sec": float(iterations * root_count * simulations / total_sec)
        if total_sec > 0
        else 0.0,
        "boundary_sec": measured["output_array_cast_sec"],
        "ctree_core_sec": 0.0,
        "boundary_fraction": measured["output_array_cast_sec"] / total_sec
        if total_sec > 0
        else 0.0,
        "core_fraction": 0.0,
        "traverse_calls": 0,
        "backpropagate_calls": 0,
        "checksum_visit": checksum_visit,
        "checksum_value": checksum_value,
        **measured,
    }


def _visits_to_full_action_space(
    visits_raw: Any,
    legal_actions: list[list[int]],
) -> np.ndarray:
    visits = np.asarray(visits_raw, dtype=np.float32)
    root_count = len(legal_actions)
    if visits.shape == (root_count, ACTION_COUNT):
        return visits
    if visits.ndim != 2 or visits.shape[0] != root_count:
        raise RuntimeError(f"unexpected visit shape {visits.shape!r}")
    full = np.zeros((root_count, ACTION_COUNT), dtype=np.float32)
    for root, actions in enumerate(legal_actions):
        if len(actions) != visits.shape[1]:
            raise RuntimeError(
                "legal-action visit width mismatch: "
                f"root={root}, legal={len(actions)}, visits={visits.shape[1]}"
            )
        full[root, actions] = visits[root]
    return full


def _resolve_torch_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if requested == "cuda" and not bool(torch.cuda.is_available()):
        raise RuntimeError("torch CUDA requested but unavailable")
    return requested


def _torch_payload_to_numpy(tensor: Any) -> np.ndarray:
    device = getattr(tensor, "device", None)
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)
        result = tensor.detach().cpu().numpy()
        torch.cuda.synchronize(device)
        return result
    return tensor.detach().cpu().numpy()


def _run_case(
    args: argparse.Namespace,
    *,
    backend: str,
    root_count: int,
    simulations: int,
    legal_profile: str,
) -> dict[str, Any]:
    if backend in {"ctree-list", "ctree-torch-d2h", CTREE_FLAT_A3_BACKEND}:
        return _run_ctree_list_case(
            backend=backend,
            root_count=root_count,
            simulations=simulations,
            iterations=args.iterations,
            warmup=args.warmup,
            legal_profile=legal_profile,
            root_noise=args.root_noise,
            root_noise_weight=args.root_noise_weight,
            root_dirichlet_alpha=args.root_dirichlet_alpha,
            pb_c_base=args.pb_c_base,
            pb_c_init=args.pb_c_init,
            discount_factor=args.discount_factor,
            value_delta_max=args.value_delta_max,
            seed=args.seed + root_count * 17 + simulations * 101,
            torch_device=args.torch_device,
            flat_a3_parity_check=bool(args.flat_a3_parity_check),
        )
    if backend == "fake-flat":
        return _run_fake_flat_case(
            root_count=root_count,
            simulations=simulations,
            iterations=args.iterations,
            warmup=args.warmup,
            legal_profile=legal_profile,
            seed=args.seed + root_count * 17 + simulations * 101,
        )
    raise ValueError(f"unknown backend {backend!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=lambda raw: _parse_csv_ints(raw, name="roots"), default=[512])
    parser.add_argument(
        "--simulations",
        type=lambda raw: _parse_csv_ints(raw, name="simulations"),
        default=[16],
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--pb-c-base", type=int, default=19652)
    parser.add_argument("--pb-c-init", type=float, default=1.25)
    parser.add_argument("--discount-factor", type=float, default=0.997)
    parser.add_argument("--value-delta-max", type=float, default=0.01)
    parser.add_argument("--root-noise", choices=("zero", "dirichlet"), default="zero")
    parser.add_argument("--root-noise-weight", type=float, default=0.0)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument(
        "--legal-profiles",
        type=lambda raw: _parse_csv_strings(
            raw,
            allowed={"all3", "single1", "mixed_2of3"},
            name="legal profile",
        ),
        default=["all3"],
    )
    parser.add_argument(
        "--backends",
        type=lambda raw: _parse_csv_strings(
            raw,
            allowed={"ctree-list", "ctree-torch-d2h", CTREE_FLAT_A3_BACKEND, "fake-flat"},
            name="backend",
        ),
        default=["ctree-list", "fake-flat"],
    )
    parser.add_argument("--torch-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--flat-a3-parity-check", action="store_true")
    parser.add_argument("--jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for backend in args.backends:
        for root_count in args.roots:
            for simulations in args.simulations:
                for legal_profile in args.legal_profiles:
                    row = _run_case(
                        args,
                        backend=backend,
                        root_count=root_count,
                        simulations=simulations,
                        legal_profile=legal_profile,
                    )
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
