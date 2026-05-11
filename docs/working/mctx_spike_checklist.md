# Next Mctx Spike Checklist

Scope: first real JAX/Mctx evidence pass after the dummy survival loop and tiny line-duel harness. No trainer yet; prove imports, shapes, search throughput, and artifact hygiene.

Current status, 2026-05-09: Modal GPU dependency smoke passed, the tiny
synthetic benchmark passed, and the larger synthetic profile also passed on
Modal L4. Larger profile command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 64 \
  --num-simulations 16 \
  --hidden-dim 64 \
  --max-depth 16 \
  --warmup-runs 2 \
  --steady-runs 5
```

Larger profile result: Modal app `ap-ULhQNpnV6a1lsn0uQLUbnX`, backend
`gpu/cuda:0`, L4, compile plus first run `8.080801095000002s`, steady median
`0.005292786999998356s`, median `12091.928127850202` decisions/sec,
`193470.85004560323` simulations/sec, and finite normalized action weights.

Current repo state: `pyproject.toml` only has `numpy` as a runtime dependency
and `modal` as an optional extra. JAX, Mctx, Flax/Optax/Equinox are not pinned
yet; the spike should make that dependency diff explicit.

## Dependency Pins To Consider

- Python: `3.11` first, `3.12` only if the JAX/Mctx stack is clean.
- JAX: pin one CUDA extra for Modal, e.g. `jax[cuda12]` or `jax[cuda13]`, matching the Modal base image and driver.
- Search/model stack: `mctx`, plus exactly one module family for the toy net: `flax` + `optax` or `equinox` + `optax`.
- Logging/profiling: keep light at first: JSON/JSONL, `time.perf_counter`, `jax.devices()`, optional TensorBoard/W&B only after the smoke is stable.
- Record exact lockfile diff, package versions, CUDA visible devices, and JAX backend in the experiment note.

## Local CPU Import Smoke

- Add the smallest script/module that imports `jax`, `mctx`, and the chosen model library.
- Print `jax.devices()`, run one tiny `jit` function, and instantiate a dummy root batch.
- Run on CPU before Modal so dependency errors are separated from GPU/image errors.
- Success: import plus one synthetic search call returns stable-shaped outputs without hidden network downloads or GPU assumptions.

## Synthetic Root/Recurrent Benchmark Shape

- Fixed action space: `A=3`, matching left/straight/right from dummy survival and line duel.
- Root batch sweep: `64`, `256`, `1024`.
- Simulation sweep: `25`, `50`, `100`.
- Hidden dim: start `128`; keep observation/root tensors synthetic and static-shaped.
- Functions to define: `representation(obs) -> hidden`, `prediction(hidden) -> prior_logits,value`, `dynamics(hidden, action) -> next_hidden,reward`, `recurrent_fn(...)`.
- Report compile time separately from warm steady-state search time; include searches/sec and model recurrent calls/sec.

## Modal GPU Smoke Shape

- Done for dependency smoke and two synthetic profiles with one small GPU Function
  and no distributed services.
- Done: first remote command imported, printed package versions/devices, and ran
  the CPU-equivalent tiny search.
- Done: fixed-shape benchmark ran with explicit warmup for both tiny and larger
  profiles.
- Captured for the larger profile: Modal app name, GPU type/backend, remote
  compile time, and steady-state timing.
- Do not attach replay, queues, volumes, actors, or training orchestration yet.

## Artifacts To Record

- Done in dated docs for first pass: config, package versions where available,
  device/backend, compile time, steady-state time, throughput, and action-weight
  sanity.
- Still useful for the next sweep: `summary.json` with config, package versions,
  device/backend, batch/sim sweep, compile time, steady-state time, throughput.
- Still useful for the next sweep: `benchmark_metrics.jsonl` with one row per
  batch/simulation/device/config result.
- Still useful for the next sweep: `stdout.txt` or copied key log lines with
  package versions, `jax.devices()`, Modal run URL.
- Optional profiler trace only if timing is surprising.
- Experiment note under `docs/experiments/` with command, result, artifact paths, and accept/reject decision.

## Rejection Criteria

- Local CPU import or tiny search cannot run with pinned dependencies.
- Modal GPU import requires unpinned/manual package surgery.
- JAX sees no GPU or silently runs the benchmark on CPU in the GPU smoke.
- Compile time dominates every tested shape and warm steady-state throughput is not meaningfully better than a CPU baseline.
- Mctx integration forces environment logic into JAX before there is evidence env stepping is the bottleneck.
- The recurrent/root API cannot preserve the dummy harness contract: fixed `A=3`, ego-perspective observations, value/reward outputs, and static batch shapes.

## Connection To Dummy Harnesses

- Dummy survival provides the single-agent MuZero-shaped contract: `A=3`, sparse terminal reward, checkpoint/summary/metrics artifact pattern.
- Tiny line duel is the next harness for simultaneous actions, ego perspective, ties, and self-play-shaped value targets.
- The Mctx spike should consume synthetic tensors shaped like these harnesses, not real env rollouts yet.
- Accepted for synthetic fixed-shape search on Modal L4. The next implementation
  step is a real-observation-shaped synthetic benchmark or a thin adapter from
  line-duel root observations into Mctx root batches while keeping CPU env
  stepping outside search.
