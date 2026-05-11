# 2026-05-08 Env Smoke Benchmark

## Question

Does the initial reference environment run end-to-end and provide a baseline throughput number for future regressions?

## Setup

- Machine: local development machine.
- Python: managed by `uv`.
- Package: initial `curvyzero.env` reference implementation.
- Config: `CurvyTronConfig(action_repeat=1)`.
- Benchmark script: `scripts/benchmark_env.py`.

## Command

```sh
uv run python scripts/benchmark_env.py --episodes 100 --max-steps 500
```

## Results

```text
episodes=100
steps=2370
elapsed_sec=0.0630
steps_per_sec=37594.1
episodes_per_sec=1586.2
```

## Interpretation

The reference simulator is runnable and fast enough for early tests. This is not a performance claim for the final environment: the current implementation has simplified collisions, simple observations, no vectorization, no wrappers, and no source-derived trail gaps.

## Artifacts

- `scripts/benchmark_env.py`
- `src/curvyzero/env/`

## Follow-ups

- Add separate timers for physics, collision, observation, reset/autoreset, and wrappers.
- Add deterministic scripted collision benchmarks.
- Add `step_many`/`reset_many` only after single-env semantics are pinned down.
