# Architecture Re-Exploration

Date: 2026-05-12

Purpose: rethink CurvyTron training speed as a whole system, not as one local
render or MCTS patch.

## Current Plain Read

The trusted proof lane is stock LightZero `train_muzero` with
`source_state_fixed_opponent`. It is useful because it calls the real LightZero
loop and gives us honest profile data.

The old custom `two-seat-selfplay` path is historical until its native replay
and target semantics are fixed.

The next big question is architecture:

```text
actors collect searched games
  -> replay stores complete trajectory data
  -> learner samples and updates
  -> checkpoints are published
  -> actors/eval refresh from checkpoints
```

Single-container optimization helps, but it probably will not give the large
speedups alone. We need to understand how much can be batched, parallelized, or
moved to accelerators without changing the MuZero algorithm.

## Working Questions

- What exactly happens in one stock LightZero training iteration?
- Which parts are CPU-only today: env, render, opponent inference, tree
  bookkeeping, replay, learner input building, artifacts?
- Which parts can use GPU today: live model inference, learner forward/backward,
  some search model calls?
- Why does multi-GPU not help yet?
- How far can subprocess collector width scale before CPU/process/IPC overhead
  wins?
- What does a real AlphaZero/MuZero actor fleet do differently?
- Which external framework gives the best reusable pattern: LightZero,
  EfficientZero, MiniZero, MCTX/JAX, OpenSpiel, RLlib/Acme/Reverb?
- What is the smallest experiment that tests coarse parallel searched self-play
  without building a giant new service?

## Active Lanes

| Lane | Output | Owner |
| --- | --- | --- |
| LightZero stock dataflow | `lightzero_stock_dataflow.md` | subagent |
| EfficientZero/Ray architecture | `efficientzero_ray_architecture.md` | subagent |
| MiniZero/full-system Zero architecture | `minizero_architecture.md` | subagent |
| MCTX/JAX batched search | `mctx_jax_search.md` | subagent |
| CurvyTron architecture synthesis | `curvytron_system_design.md` | optimizer |
| Large-scale Zero systems | `large_scale_zero_architectures.md` | subagent |
| Collect-only fanout design | `collect_only_fanout_design.md` | subagent |
| MCTX visual-root benchmark | `mctx_visual_root_benchmark_plan.md` | subagent |
| Stock-frozen profile tensor | `current_profile_tensor.md` | optimizer |
| Profile validation results | `profile_validation_results.md` | optimizer |
| Second-wave profile tensor | `second_wave_profile_tensor.md` | optimizer |
| Coach speed recommendations | `coach_speed_recommendations.md` | optimizer |
| Optimizer task board | `task_board.md` | optimizer |

## Early Synthesis

The emerging answer is that large speedups probably come from architecture, not
from one local patch. The trusted stock LightZero lane is good for proof and
timing, but it is still mostly a single trainer loop. The large-scale pattern is
actor/search workers producing searched trajectory chunks, replay storing those
chunks, and a learner publishing checkpoints.

Near term, keep LightZero as the control path and test scale with coarse
synchronous fanout before building a permanent service. In parallel, use MCTX as
a search benchmark and MiniZero/EfficientZero/OpenSpiel as architecture
references.

## First-Wave Results

- LightZero stock path: one synchronous trainer process with vector env
  collection, in-process replay, learner, and checkpoint hooks. It has useful
  env-manager and MCTS knobs, but no ready distributed CurvyTron actor/replay
  service.
- EfficientZero: useful Ray decomposition reference with self-play workers,
  shared weights, replay, batch builders, reanalysis, and learner queues. Not a
  direct CurvyTron migration.
- MiniZero: strongest full-system open reference so far for actor batching and
  checkpointed self-play, but turn-based C++ game assumptions make independent
  simultaneous CurvyTron a real port.
- MCTX: credible fast batched search primitive, not a full trainer.

## Concrete Benchmark Notes

The immediate profile lane now has generated tensor manifests:
`current_profile_tensor.md` and `second_wave_profile_tensor.md`. They cover
base-manager attribution, subprocess collector width, MCTS simulation slope,
long no-death render lenses, fixed-opponent overhead, CPU/GPU shape, and reward
bookkeeping.

The validation and wave results are in `profile_validation_results.md`. Current
plain read: C96 is the current wide-profile starting point, long trajectories
are collection/render-bound, fast render is a real speed lens, subprocess
collection helps, frozen checkpoint opponent inference is costly in the stock
fixed-opponent lane, and MCTS is not the main limiter yet.

The MCTX lane now has a narrow benchmark plan:
`mctx_visual_root_benchmark_plan.md`. It should test
`float32[B,2,4,64,64]` visual roots through a tiny JAX CNN and
`mctx.gumbel_muzero_policy`, with timing split into visual setup, host/device
transfer, compile, steady search, output copy, and GPU memory. This is useful
only as a search primitive test; it should not touch the trainer yet.

## Current Verdict

Stay on LightZero for the trusted near-term CurvyTron proof path. Do not migrate
to EfficientZero, MiniZero, OpenSpiel, or MCTX right now. The useful move is to
copy the architecture pattern those systems show: searched actors, trajectory
chunks, replay import, learner work, checkpoint publication, and separate eval.

The next concrete Optimizer build should be a collect-only fanout prototype:
one checkpoint, many searched collectors, native LightZero `GameSegment`
chunks, strict manifests, then a merge/import smoke into `MuZeroGameBuffer`.
This answers the self-play scaling question directly without changing MuZero
semantics or the Coach learning lane.

The second concrete experiment should be an isolated MCTX visual-root benchmark.
It is lower priority than fanout because it only answers the search-kernel
question, but it is safe to run in parallel because it does not touch training.

## Guardrails

- Do not claim learning quality from speed profiles.
- Do not mutate live Coach training runs.
- Do not treat old custom two-seat timings as current stock LightZero truth.
- Do not change MuZero semantics to get speed.
- Keep all speed numbers tied to the exact path, render mode, env manager,
  collector count, search sims, and checkpoint/eval settings.
