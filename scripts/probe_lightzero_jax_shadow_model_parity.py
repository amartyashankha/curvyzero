#!/usr/bin/env python3
"""Probe raw LightZero PyTorch vs JAX shadow-model parity.

This is an optimizer proof gate, not a trainer.  It loads one immutable
checkpoint, compares raw model inference, and writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    load_lightzero_curvytron_visual_survival_policy,
)
from curvyzero.training.lightzero_jax_shadow_model_parity import (
    checkpoint_sha256,
    compare_arrays,
    deterministic_observation_batch,
    inverse_scalar_transform_logits,
    jax_shadow_from_state_dict,
    profile_only_report_base,
    require_immutable_checkpoint_ref,
)


def _torch_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _output_comparisons(
    *,
    prefix: str,
    torch_output: Any,
    jax_output: dict[str, Any],
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for name in ("policy_logits", "value", "reward", "latent_state"):
        torch_value = _torch_to_numpy(getattr(torch_output, name))
        jax_value = np.asarray(jax_output[name])
        comparisons.append(
            compare_arrays(
                f"{prefix}.{name}",
                torch_value,
                jax_value,
                atol=atol,
                rtol=rtol,
            ).as_dict()
        )
        if name in {"value", "reward"} and torch_value.ndim == 2 and jax_value.ndim == 2:
            support_scale = (int(torch_value.shape[1]) - 1) // 2
            comparisons.append(
                compare_arrays(
                    f"{prefix}.{name}_scalar",
                    inverse_scalar_transform_logits(
                        torch_value,
                        support_scale=support_scale,
                    ),
                    inverse_scalar_transform_logits(
                        jax_value,
                        support_scale=support_scale,
                    ),
                    atol=atol,
                    rtol=rtol,
                ).as_dict()
            )
    return comparisons


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    checkpoint_ref = require_immutable_checkpoint_ref(args.checkpoint_path)
    checkpoint_path = Path(checkpoint_ref)
    policy, device, load_summary = load_lightzero_curvytron_visual_survival_policy(
        checkpoint_path=checkpoint_path,
        seed=args.seed,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        use_cuda=args.device == "cuda",
        state_key=args.state_key,
    )
    model = getattr(policy, "_model")
    model.eval()
    state_dict = model.state_dict()
    shadow = jax_shadow_from_state_dict(state_dict, platform=args.jax_platform)

    comparisons: list[dict[str, Any]] = []
    for index, kind in enumerate(("zeros", "ramp", "random")):
        obs_np = deterministic_observation_batch(
            batch_size=args.batch_size,
            kind=kind,
            seed=args.seed + index,
        )
        obs_torch = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            torch_initial = model.initial_inference(obs_torch)
        jax_initial = shadow.initial_inference(obs_np)
        comparisons.extend(
            _output_comparisons(
                prefix=f"initial.{kind}",
                torch_output=torch_initial,
                jax_output=jax_initial,
                atol=args.atol,
                rtol=args.rtol,
            )
        )

        actions_np = np.arange(args.batch_size, dtype=np.int64) % shadow.action_space_size
        actions_torch = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        with torch.no_grad():
            torch_recurrent = model.recurrent_inference(
                torch_initial.latent_state, actions_torch
            )
        jax_recurrent = shadow.recurrent_inference(
            jax_initial["latent_state"], actions_np
        )
        jax_recurrent_from_torch_latent = shadow.recurrent_inference(
            _torch_to_numpy(torch_initial.latent_state), actions_np
        )
        comparisons.extend(
            _output_comparisons(
                prefix=f"recurrent.{kind}",
                torch_output=torch_recurrent,
                jax_output=jax_recurrent,
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        comparisons.extend(
            _output_comparisons(
                prefix=f"recurrent_from_torch_latent.{kind}",
                torch_output=torch_recurrent,
                jax_output=jax_recurrent_from_torch_latent,
                atol=args.atol,
                rtol=args.rtol,
            )
        )

    coverage = shadow.coverage_summary()
    allclose = all(bool(item["allclose"]) for item in comparisons)
    report = {
        **profile_only_report_base(),
        "ok": bool(allclose and coverage["ok"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_load_summary": load_summary,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_simulations": int(args.num_simulations),
        "jax_platform": args.jax_platform,
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "model_surface": {
            "model_class": type(model).__module__ + "." + type(model).__name__,
            "action_space_size": int(shadow.action_space_size),
            "reward_support_size": int(shadow.reward_support_size),
            "value_support_size": int(shadow.value_support_size),
            "state_dict_key_count": len(state_dict),
        },
        "coverage": coverage,
        "comparisons": comparisons,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--state-key")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-simulations", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--jax-platform", default=None)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    report = run_probe(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
