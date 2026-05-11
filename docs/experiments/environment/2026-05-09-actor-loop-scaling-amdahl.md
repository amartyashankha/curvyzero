# Actor Loop Scaling Amdahl Scout - 2026-05-09

## Goal

Run a modest local timing sweep with the existing fixture-seeded actor-loop
benchmark to see whether env step, synthetic policy/search, replay staging, or
debug event logging dominates the current fixture-slice loop.

This is a local scout only. It is useful for current-loop Amdahl shape, not for
production self-play throughput.

## Practical Decision

Do not pick the next speed project from toy numbers alone. Use this scout to
decide what the next full actor-loop benchmark must answer.

Current decision:

1. Optimize the measured actor loop before rewriting any one component.
2. Treat env step as the leading suspect, not a solved verdict. It is large in
   the local debug runs, but real model/search/MCTS is still missing.
3. Replace synthetic policy/search with calibrated model/MCTS timing before
   spending major effort on GPU env, C++/Rust env, or distributed actors.
4. Keep observation packing and replay fixed-shape and cheap. They should stay
   simple until they become visible wall-time buckets or cause CPU/GPU copies.
5. Make reset/autoreset correctness and timing explicit. Optimize it only if
   terminal rows stall batches, inflate p95/p99 action latency, or corrupt
   replay ordering.

Plain latency versus throughput rule: throughput is how many useful games or
ego decisions the system finishes per minute; latency is how long a ready env
row waits for its next action. Bigger batches, process pools, or central
inference only help if throughput rises without p95/p99 action latency and
policy staleness becoming unacceptable.

Optimization order for the next benchmark:

| Component | Optimize first when | Why |
| --- | --- | --- |
| Env step | It remains the largest real bucket after debug events are off and real search timing is included. | It directly gates every transition, but Amdahl's law makes env-only work weak if search dominates. |
| Model/search/MCTS | Calibrated timing is large, GPU idle/queueing is visible, or action latency tracks search time. | Real MuZero-style search can easily dominate; the current NumPy stand-in is not evidence that it is cheap. |
| Observation packing | It takes meaningful wall time or creates extra copies/transfers. | Every live player becomes an ego row, so bad packing can erase env gains. |
| Replay | Serialization/write/learner handoff appears in the hot path. | In-memory array staging is cheap in debug scouts, but production writes are not yet measured. |
| Reset/autoreset | Finished rows stall the batch or final transitions/reset observations get mixed. | Correct terminal handling matters before terminal-row speed claims. |

## Commands

Baseline synthetic search, one simulation:

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py --batch-sizes 2 8 32 --repeat 20 --warmup 2 --rollout-steps 2 --body-capacity 4 --hidden-dim 8 --simulations 1 --chunk-steps 8 --event-modes debug-event no-event --format plain
```

Small follow-up with heavier synthetic search:

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py --batch-sizes 2 8 --repeat 20 --warmup 2 --rollout-steps 2 --body-capacity 4 --hidden-dim 8 --simulations 4 --chunk-steps 8 --event-modes debug-event no-event --format plain
```

Both commands completed with green preflight. That is a correctness guard for
the timing slice, not the headline speed result.

## Baseline Highlights

Baseline run: `repeat=20`, `warmup=2`, `rollout_steps=2`, `hidden_dim=8`,
`simulations=1`, `chunk_steps=8`.

Selected `B=32` rows, because they best show the amortized fixture-slice shape:

| Group | Event mode | Staged ego rows/s | Actor p50 ms | Env % | Debug pack % | Synthetic policy % | Replay % | Env us/env row | Debug us/ego | Policy us/ego | Replay us/ego | Event emit % env step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1_K4 | debug-event | 18,589.0 | 0.808 | 51.6 | 5.0 | 8.7 | 2.2 | 27.763 | 2.714 | 4.690 | 1.203 | 9.3 |
| P1_K4 | no-event | 47,148.9 | 0.486 | 44.8 | 8.7 | 10.4 | 2.2 | 9.512 | 1.843 | 2.212 | 0.458 | 0.0 |
| P2_K4 | debug-event | 127,316.5 | 0.431 | 62.5 | 7.0 | 7.2 | 1.6 | 9.826 | 0.549 | 0.567 | 0.124 | 36.4 |
| P2_K4 | no-event | 179,436.1 | 0.322 | 50.0 | 7.2 | 10.8 | 2.2 | 5.577 | 0.401 | 0.600 | 0.123 | 0.0 |
| P3_K4 | debug-event | 219,133.8 | 0.488 | 78.7 | 7.5 | 8.5 | 1.6 | 10.768 | 0.341 | 0.387 | 0.072 | 25.8 |
| P3_K4 | no-event | 318,643.6 | 0.301 | 72.9 | 8.0 | 11.7 | 2.1 | 6.866 | 0.252 | 0.367 | 0.067 | 0.0 |

Event logging comparison from the same baseline run:

| Group | B=2 no-event speedup | B=8 no-event speedup | B=32 no-event speedup |
| --- | ---: | ---: | ---: |
| P1_K4 | 1.381x | 1.015x | 2.536x |
| P2_K4 | 1.427x | 1.344x | 1.409x |
| P3_K4 | 1.156x | 1.449x | 1.454x |

The P1 B=32 debug-event row looks noisy or unusually event-sensitive in this
short run, but the P2/P3 rows are consistent: event logging is material, and
removing it improves total loop rate by about 1.3x to 1.45x at B=8/B=32.

## Simulations=4 Follow-up

Second run: same fixture loop, but `simulations=4` and only `B=2,8`.

Selected `B=8` rows:

| Group | Event mode | Staged ego rows/s | Actor p50 ms | Env % | Debug pack % | Synthetic policy % | Replay % | Synthetic policy p50 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1_K4 | debug-event | 29,470.3 | 0.264 | 43.1 | 7.6 | 17.6 | 2.1 | 47.334 |
| P1_K4 | no-event | 33,997.2 | 0.227 | 36.7 | 7.7 | 20.0 | 2.3 | 46.646 |
| P2_K4 | debug-event | 45,845.0 | 0.341 | 53.7 | 7.3 | 14.6 | 1.7 | 50.645 |
| P2_K4 | no-event | 56,935.5 | 0.271 | 45.1 | 7.0 | 18.0 | 2.0 | 50.062 |
| P3_K4 | debug-event | 71,605.1 | 0.332 | 70.5 | 7.2 | 16.3 | 1.8 | 54.000 |
| P3_K4 | no-event | 92,621.3 | 0.253 | 63.8 | 7.8 | 20.8 | 2.2 | 53.583 |

Increasing synthetic simulations from 1 to 4 moved synthetic policy/search into
the roughly 14% to 22% range, but it still did not overtake env step for P2/P3.

## Interpretation

Current fixture-slice loop dominance:

- Env step is the main bucket for P2/P3. At B=32, env step is 50.0% to 72.9%
  without debug events and 62.5% to 78.7% with debug events.
- P1 is more mixed and noisier in this short sweep. Env step is still the top
  bucket, but non-env work is close enough that small-run variance shows up.
- Debug event logging is significant. In P2/P3, event emit was 25.8% to 36.4%
  of env-step time at B=32, and no-event mode improved total loop rate by about
  1.3x to 1.45x at B=8/B=32.
- Synthetic policy/search is not the baseline bottleneck at `simulations=1`
  (about 7% to 16% of the loop), but becomes a meaningful second-tier bucket at
  `simulations=4` (about 14% to 22%).
- Replay staging is not the current bottleneck. It stayed around 1.5% to 3.4%
  of loop time in these runs.
- Debug obs/reward packing is not dominant, but it is visible: roughly 5% to
  10% of loop time in most rows.

## What Is Real

- The timed path connects the current fixture-seeded NumPy vector step, debug
  obs/reward packing, synthetic policy/search, action encoding, and fixed
  in-memory replay chunk staging.
- Source/common-trace state and fixed event rows are compared once per
  supported fixture before timing.
- B>1 batch preflight compares stacked fixture source moves against scalar
  comparator output.
- The measurements are local CPU/NumPy timings on the current machine.

## What Is Fake Or Incomplete

- The loop uses fixture-reset rollout blocks, not production self-play reset
  behavior.
- The internal autoreset is debug-only and happens after replay staging.
- Observations and rewards are from the debug packer, not the final training
  observation/reward schema.
- Policy/search is a synthetic NumPy stand-in, not a learned model, MCTS, Mctx,
  JAX, or GPU work.
- Replay is only an in-memory fixed ring, not the production replay schema,
  replay writer, or learner handoff.
- The source fidelity preflight covers supported fixture source moves, not
  arbitrary policy moves fed back after the first rollout step.
- P1, P2, and P3 fixture groups are timed separately instead of as one padded
  mixed-player production batch.
- No horizon truncation, learner transfer, device transfer, JAX compile, or GPU
  timing is measured.

## Most Important Next Measurement

Replace the synthetic policy/search stand-in with a production-calibrated model
and search timing, with debug event logging disabled, then rerun the same bucket
breakdown at a modest larger batch. That is the largest unknown: real search may
dominate the actor loop even though this local fixture-slice scout is currently
env-step dominated.

## Worker D Cheap Follow-Up

Goal: keep the speed story practical and check latency/throughput shape without
claiming GPU, real MCTS, or production self-play.

Commands run:

```sh
uv run pytest tests/test_benchmark_selfplay_parallel_bridge.py -q
```

```sh
uv run python scripts/benchmark_selfplay_parallel_bridge.py --batch 32 --steps 50 --warmup 5 --workers 2 --hidden-dim 32 --modes serial serial-sharded thread --format plain
```

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py --batch-sizes 8 32 --repeat 10 --warmup 1 --rollout-steps 2 --body-capacity 4 --hidden-dim 8 --simulations 1 --chunk-steps 8 --event-modes debug-event no-event --format plain
```

The helper smoke was green; the timing interpretation below is the part that
answers the bottleneck question.

### Toy Object-Env Parallel Bridge

This benchmark is local debug timing only. It uses the simplified toy object
env, synthetic NumPy policy logits, in-memory replay staging, and coarse local
sharding. It is not source-fidelity timing, not LightZero, not real MCTS, not a
GPU run, and not production throughput.

Run shape: `batch=32`, `steps=50`, `warmup=5`, `workers=2`, `hidden_dim=32`.

| Mode | Wall env steps/s | Measured-loop env steps/s | Actor p50 ms | Actor p95 ms | Actor p99 ms | Env % | Policy % | Replay % | Bottleneck summary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| serial | 19,937.0 | 26,669.3 | 1.149 | 1.391 | 1.678 | 94.5 | 1.2 | 0.4 | `env_step` 94.5%; top3 `env_step`, `autoreset`, `policy_batch` |
| serial-sharded | 22,725.6 | 25,774.8 | 1.201 | 1.402 | 1.606 | 93.2 | 1.7 | 0.7 | `env_step` 93.2%; top3 `env_step`, `autoreset`, `policy_batch` |
| thread | 18,088.0 | 23,033.9 | 1.297 | 3.786 | 6.953 | 55.6 | 40.6 | 0.4 | `env_step` 55.6%; top3 `env_step`, `policy_batch`, `autoreset` |

Practical read: coarse threading did not improve this cheap local workload and
made p95/p99 actor-step latency worse. That is the latency-vs-throughput
boundary to keep watching: a mode can look acceptable by average steps/sec but
still hurt action freshness.

### Fixture-Seeded Vector Actor Bridge

This benchmark is also local debug timing only. It cycles verified fixture
states, runs the current NumPy vector step, uses debug obs/reward packing, feeds
a synthetic NumPy policy/search stand-in, and stages replay in memory. There is
no real MCTS, no learned checkpoint, no LightZero collector, no GPU env step,
no device transfer timing, and no production reset/autoreset contract.

The run completed with green preflight; the useful part for speed planning is
the bucket split below.

Selected `B=32`, `simulations=1`, no-event rows:

| Group | Staged ego rows/s | Actor p50 ms | Actor p95 ms | Env % | Debug pack % | Synthetic policy % | Replay % | Env us/env row | Policy us/ego |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1_K4 | 124,131.3 | 0.247 | 0.263 | 37.4 | 7.5 | 7.8 | 2.1 | 3.011 | 0.632 |
| P2_K4 | 185,905.8 | 0.334 | 0.350 | 46.7 | 6.4 | 7.9 | 1.7 | 5.021 | 0.423 |
| P3_K4 | 290,361.0 | 0.324 | 0.371 | 66.2 | 6.8 | 9.8 | 1.8 | 6.840 | 0.337 |

Debug-event versus no-event total-loop speedup at `B=32` in this short run:

| Group | No-event total-loop speedup vs debug-event | No-event env-step speedup vs debug-event |
| --- | ---: | ---: |
| P1_K4 | 1.171x | 1.361x |
| P2_K4 | 1.361x | 1.687x |
| P3_K4 | 1.385x | 1.534x |

Practical read: for P2/P3, env step remains the largest bucket even with debug
events disabled, while synthetic policy/search and replay staging are not the
current bottleneck. The next useful speed proof is not more vector-only timing;
it is replacing the stand-in with calibrated model/search timing and measuring
the CPU env plus policy/search boundary, including transfer if a GPU is used.

## Worker M Calibrated Stand-In Check

Goal: cheaply expose latency-vs-throughput behavior at the actor-loop boundary
without claiming real MuZero/MCTS timing.

Existing script support was enough. `benchmark_vector_actor_loop_bridge.py`
already has `--simulations`, which repeats the synthetic NumPy
recurrent/search-shaped loop inside the timed actor step. Worker M only tightened
the CLI help text to label that knob as a calibrated stand-in. No benchmark
behavior changed.

Recommended scout command:

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py --batch-sizes 8 32 --repeat 5 --warmup 1 --rollout-steps 2 --body-capacity 4 --hidden-dim 16 --simulations 64 --chunk-steps 8 --event-modes no-event --format plain
```

Companion light baseline:

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py --batch-sizes 32 --repeat 5 --warmup 1 --rollout-steps 2 --body-capacity 4 --hidden-dim 16 --simulations 1 --chunk-steps 8 --event-modes no-event --format plain
```

The command completed with green preflight: `passed:19 failed:0 unsupported:0
batch_preflight_failed:False`. This remains a fixture-reset debug actor-loop
timing with debug obs/reward packing, synthetic policy/search, no real MCTS, no
learned model, no GPU, and no device transfer.

Selected no-event rows:

| Group | B | Sims | Staged ego rows/s | Actor p50 ms | Actor p95 ms | Env % | Synthetic policy % | Top bucket |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P2_K4 | 32 | 1 | 163,868.0 | 0.380 | 0.412 | 45.7 | 9.0 | `env_step` |
| P2_K4 | 8 | 64 | 16,124.6 | 0.918 | 1.241 | 18.5 | 64.4 | `policy_search` |
| P2_K4 | 32 | 64 | 48,304.0 | 1.281 | 1.491 | 14.3 | 71.3 | `policy_search` |
| P3_K4 | 32 | 1 | 269,896.7 | 0.337 | 0.435 | 64.5 | 11.4 | `env_step` |
| P3_K4 | 8 | 64 | 14,451.2 | 1.461 | 2.377 | 21.5 | 67.9 | `policy_search` |
| P3_K4 | 32 | 64 | 59,993.6 | 1.509 | 2.026 | 16.3 | 76.8 | `policy_search` |

What this tells us:

- The same actor-loop report can expose the boundary tradeoff. With a light
  stand-in, P2/P3 are env-step led; with heavier synthetic search, policy/search
  dominates and p95 actor-step latency grows.
- Larger batches improved throughput in the heavy stand-in rows, but did not
  erase action latency. That is the practical latency-vs-throughput shape the
  real model/search benchmark must preserve.
- This is only a calibrated stand-in for local CPU shape work. The next real
  benchmark should replace `--simulations 64` with measured model/MCTS timing on
  the same ego-row shapes, including host/device transfer if GPU search is used.
