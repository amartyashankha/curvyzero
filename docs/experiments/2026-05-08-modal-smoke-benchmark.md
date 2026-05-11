# 2026-05-08 Modal Smoke Benchmark

## Question

Does the Modal smoke app still run repository tests, a small remote CPU environment benchmark, and a
cheap GPU visibility probe?

## Setup

- Modal app: `curvyzero-smoke`.
- Entrypoint: `curvyzero.infra.modal.smoke`.
- Local command runner: `uv run --extra modal modal`.
- CPU benchmark config: `CurvyTronConfig(action_repeat=1)`, random actions, `episodes=25`,
  `max_steps=500`, `seed=0`.
- GPU smoke: `nvidia-smi` only; no framework install or training loop.

## Commands And Results

### Remote Tests

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind tests
```

Result:

```text
Modal run: ap-88LM0Y5FYWAmDc4omylcFt
pytest: 4 passed in 0.47s
```

### Remote CPU Benchmark

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind benchmark --episodes 25 --max-steps 500
```

Result:

```json
{
  "elapsed_sec": 0.07019016899999997,
  "episodes": 25,
  "episodes_per_sec": 356.1752358795433,
  "rules_hash": "d1aab3da8c983fc4",
  "steps": 575,
  "steps_per_sec": 8192.030425229497
}
```

Modal run: `ap-HhifGZ9XakXcGBhs0A98e8`.

### Remote GPU Smoke

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind gpu
```

Result:

```json
{
  "nvidia_smi": "NVIDIA L4, 23034 MiB, 580.95.05"
}
```

Modal run: `ap-xak6TKbcJ2C3Pgr90fE8XW`.

## Interpretation

The Modal image builds, mounts the repo sources, runs the current test suite remotely, executes the
reference environment on remote CPU, and can see a Modal-provisioned NVIDIA L4. This remains a smoke
result only; it is not evidence for training throughput.

Keep simulator stepping, MCTS, replay sampling, and model updates inside long-running container
processes. Do not turn Modal Functions, Queues, or Dicts into per-action, per-step, or per-node
hot-loop primitives.
