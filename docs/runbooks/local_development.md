# Local Development

## Install And Test

```sh
uv run --extra dev pytest
```

## Environment Smoke Benchmark

```sh
uv run python scripts/benchmark_env.py --episodes 1000
```

This benchmark is intentionally simple. Treat it as a regression smoke, not as proof the simulator is fast enough for training.

## Modal Smokes

Run these from the repository root:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind tests
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind benchmark --episodes 25 --max-steps 500
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind gpu
```

Use local tests for the tight edit loop, but prefer Modal for benchmarks, GPU checks, and any experiment
that should resemble the eventual training runtime. Keep Modal smoke commands small unless you are
intentionally running a benchmark; Modal Functions, Queues, and Dicts are not per-step hot-loop
primitives.

If `modal` is not already installed in the active environment, use:

```sh
uv run --extra modal modal --version
```
