# Optimizer Orchestration, 2026-05-22

## Current Goal

Speed up the CurvyTron compact visual + MCTX profile loop without touching live
Coach training runs. Keep every speed claim labeled by currency:

- production Coach training;
- stock/full-loop profile;
- profile-only compact boundary probe.

The current work is profile-only compact boundary probing.

## Current Best Denominator

```text
H100, B1024/P2, loop24, no-death compact loop,
native_actor_buffer=True, actor_count=1,
borrow_single_actor_render_state=True,
resident GPU stack, no root observation copy,
closed-loop replay-index on,
explicit resident-stack sync off.
vectorized delta pack off.
```

## Current Payload Split Read

Newest read after direct root-node extraction:

```text
sim16 full materialization, replay off:
  71.2k roots/sec, root_value_extract 0.014s
sim32 full materialization, replay off:
  43.5k roots/sec, root_value_extract 0.021s

sim16 replay-valid, replay index on:
  55.0k roots/sec, replay_index 0.010s, root_value_extract 0.019s
sim32 replay-valid, replay index on:
  38.1k roots/sec, replay_index 0.012s, root_value_extract 0.024s

longer stability:
  loop96 sim16 replay on/off: 50.6k / 53.6k roots/sec
```

Plain meaning:

```text
The old payload wall was mostly a bad root-value extractor. We were taking the
expensive MCTX summary/materialization path to get root values. Direct root
node extraction removes that wall. Replay rows themselves are small here; the
loop96 sim16 replay-on/off pair is the current clean sanity check.

Action-only, deferred payload, and overlap payload remain useful diagnostics,
but they are not the current recommended training/profile path.
```

Decision:

```text
Do not spend the next patch on serial deferred payload flushing. Do not promote
the Python-thread overlap canary; it caused contention. The next target is
env/observation/search-input handoff, and at sim32+ MCTS search itself.
```

Latest rows:

```text
current-code exact delta pack:
  sim16: 51.9k roots/sec
  sim32: 45.1k roots/sec

vectorized delta pack:
  sim16: 53.1k roots/sec
  sim32: 37.9k roots/sec
```

Decision: vectorized delta pack is opt-in only. It is validated, but the sim32
A/B regression means it is not a recommended speed setting.

The async internal renderer flag did not help:

```text
sim16: 50.2k roots/sec
sim32: 40.9k roots/sec
```

## Main-Thread Read

The loop still alternates between CPU and GPU:

```text
MCTX action on GPU
-> read selected action to CPU
-> CPU env step
-> CPU compact sidecars and render/search-input state
-> H2D renderer deltas/compose state
-> GPU persistent framebuffer/latest frame
-> GPU resident stack
-> MCTX search
```

The selected-action readback is small and semantically required while the env is
CPU-owned. The larger suspect cost is rebuilding and transferring the next
search input every decision.

Subagent synthesis:

```text
Bernoulli/dataflow:
  selected action is about 8 KiB and required before CPU env step.
  latest frame is about 8 MiB, resident stack about 32 MiB, and copied root
  observation would be about 32 MiB. Avoiding those large copies is still the
  right shape; now attack repeated compact/delta/H2D/search-input handoff.

Locke/validation:
  row-level compact replay tests are strong. Future action-only/deferred/overlap
  modes need a multi-record materialization parity canary before any speed
  number is replay-valid.

Goodall/external:
  fast RL systems keep hot state contiguous, batch model/search work, and push
  scalar framework objects to validation/logging boundaries. For us, that means
  compact state/search-input ownership, not another isolated renderer toggle.

Pascal/host-overhead:
  root-value summary fallback is a regression; keep direct root-node extraction
  guarded. Renderer H2D still blocks by default. Compact-batch construction was
  suspected as hidden work, so we added a timer; the measured loop48 cost was
  tiny.
```

## Amdahl Read

Do not spend the next local patch on raw drawing or game mechanics unless fresh
data disproves this.

Current likely wall:

```text
CPU production-to-compact
CPU delta pack
H2D delta/compose update
resident stack/root ownership
small public packaging
MCTX search
```

Raw GPU drawing is already a few milliseconds per loop row set. Game mechanics
is small in the current profile. At sim32, search is already large enough that
pure observation work cannot produce a 10x whole-loop win.

## Parallel Agents

- Banach: write a full one-iteration dataflow map with large/small data and
  sync points.
- Wegener: propose resident/GPU architecture options and cheap falsifiers.
- Plato: design validation gates so speed patches do not silently change the
  observation/search/replay contract.
- Erdos: external research on fast MCTS/self-play/GPU dataflow patterns.

## Next Local Decision

Do not promote the async renderer flag. The next architecture canary should be
one of:

1. Actor emits compact render deltas directly, bypassing
   `_persistent_compact_state_from_production` and most of `_persistent_delta_state`.
2. Renderer owns a compact CPU/GPU state and updates it in place from actor
   step outputs.
3. MCTX/search boundary consumes a resident compact root object with only
   tiny host sidecars rebuilt each step.

The falsifier is total roots/sec on the current denominator, not a local timer.
Any payload design must also prove replay parity before it becomes trainer-facing.

## Immediate To-Do

- [x] Correct world model/task board/experiment log with fresh sync-off and
  async rows.
- [x] Close stale Arendt agent after recording mechanics-vs-observation audit.
- [x] Collect Banach/Dirac deferred-payload design and Copernicus action-only
  critique docs.
- [x] Add Gauss GPU/MCTS payload research doc.
- [x] Fold the direct-root extraction finding into `world_model.md`,
  `experiment_log.md`, and `task_board.md`.
- [x] Fold the new full-dataflow subagent findings into the docs when they
  land.
- [ ] Choose one P0 canary with a disjoint write set: likely env/observation
  search-input ownership or search scaling, not serial deferred flush.
- [x] Run async-H2D canary through the existing renderer async flag. Result:
  small win only, about `1.05-1.06x`; keep opt-in.
- [x] Add compact batch build/probe wall timers and verify they are visible in
  repeated closed-loop rows. Result: only `0.0016s` / `0.0006s` over 48 steps.
- [ ] Run focused local tests first, then one matched H100 profile row.

## Full Dataflow Parallel Wave, 2026-05-22 Late

User ask:

```text
Map the full data flow, sync points, data sizes, and architecture options.
Keep the main thread clean, use parallel subagents, and keep docs current.
```

Current main-thread stance:

```text
Do not keep hammering raw rendering. Raw GPU draw is already tiny in the
profile-only compact loop. The remaining wall is the loop boundary:
CPU env state -> compact/search input -> GPU renderer/stack -> MCTX search,
plus search itself at higher simulation counts.
```

Active / recent subagents:

- Maxwell: direct visual-delta canary critique. Verdict: feasible but narrow;
  likely `1.05x-1.12x` at sim16, hard ceiling near `1.2x`; keep as profile-only
  unless it shrinks production-to-compact/delta/H2D materially.
- Helmholtz: current full dataflow map. Output target:
  `subagent_full_dataflow_map_20260522.md`.
- Averroes: GPU/host sync model and known patterns. Output target:
  `subagent_gpu_sync_model_20260522.md`.
- Laplace: ten architecture designs and critiques. Output target:
  `subagent_architecture_design_critiques_20260522.md`.

Fresh profile rows launched from current code:

```text
H100/B1024/P2/body4096/h64/depth16/loop48/native/borrowed/root-no-copy/
resident-stack/resident-sync-off/replay-valid:

sim16 refresh-on:  ap-XmQM6DmkFvDls76ZpU0xzE
sim32 refresh-on:  ap-ABmbtE2z87CaEfvcERadBS
sim16 refresh-off: ap-VG9v7RM7vWSytnaBrFWFzw
```

Result correction:

```text
sim16 refresh-on:  ap-XmQM6DmkFvDls76ZpU0xzE -> 62.7k roots/sec
sim32 refresh-on:  ap-ABmbtE2z87CaEfvcERadBS -> 49.1k roots/sec

The first refresh-off row was invalid because
borrow_single_actor_render_state=True requires refresh_observation_stack=True.
Reran with borrowed state disabled:

sim16 refresh-off: ap-Re7uaeDqBjNOI8LxQdRWeI -> 98.5k roots/sec
sim32 refresh-off: ap-DDZi5dTudchj4ZnaoDvWGw -> 74.9k roots/sec
```

Current decision:

```text
Observation refresh is still worth optimizing, but its measured ceiling is only
about 1.5-1.6x on the current denominator. The next main wave should price the
compact search-service / contiguous-buffer architecture. Direct visual-delta
stays a bounded P1 canary, not the main line.
```

## Compact Service Sidecar Wave Result

Fair current-code shape:

```text
H100, B512/A16, 60 measured, 15 warmup, direct_gray64,
uint8 stack, scalar timestep materialization off,
compact service replay proof on, root noise 0.0.
```

Rows:

```text
mock_search_service:      ap-XChnzStDiIYMLtprr7qQ0H -> 17,711.9 steps/sec
service_tax_probe:        ap-Yx4retayDwFuKDzFWSQgHM -> 12,461.6 steps/sec
direct_ctree_gpu_latent:  ap-AxV03FKEyemvd1okLy0Zxc ->  7,155.7 steps/sec
```

Decision:

```text
The service shape has real headroom:
  mock/direct = 2.48x
  service_tax/direct = 1.74x

This is not enough to claim 10x or trainer readiness. It is enough to stop
treating renderer polish as the main line. The next serious implementation
lane is one compact search-service boundary with strict replay/RND/player/
terminal parity gates.
```

Subagent synthesis:

```text
Parfit: mock is fake search, service_tax pays real model rollout but fake tree,
direct_ctree_gpu_latent is the current real-search comparator, and
compact_service_replay_proof is correctness/replay-edge proof rather than a
speed mode.

Curie: current tests are good local contracts, but promotion needs a closed
compact-loop parity test showing the search payload attaches to the exact same
replay facts as the existing target-row path.
```

Decision rule after these rows land:

1. If refresh-on is still close to refresh-off at sim16, stop small renderer
   canaries and put effort into search/service/array-native replay.
2. If refresh-on fell back far from refresh-off, split the remaining
   observation handoff and try the direct visual-delta canary.
3. If sim32 is search-heavy, run one search-boundary falsifier beside any
   observation work.
4. Do not turn action-only/deferred/overlap rows into trainer claims until the
   multi-record replay materialization parity canary exists.

## 2026-05-23 Full Dataflow Critique Wave

Purpose:

```text
Stop arguing from one timer. Map the whole iteration: what moves, where it
lives, when it syncs, what is large, what is small, and which architecture
changes can plausibly move the wall by 2x/5x/10x.
```

Local status before the wave:

```text
CompactSearchServiceV1 protocol exists.
Direct CTree compact service adapter exists.
Array-ceiling compact service adapter exists.
Full local contract tests pass:
  tests/test_compact_search_replay_contract.py -> 9 passed
  tests/test_source_state_batched_observation_boundary_profile.py -> 107 passed
```

Important warning:

```text
The adapters call their wrapped probe. The next implementation must avoid
double-running search. Either the profile loop calls the service once, or the
existing one probe call converts its arrays into CompactSearchResultV1.
```

Subagents launched:

```text
Kepler: full iteration dataflow critique
  -> subagent_full_iteration_dataflow_critique_20260523.md

Godel: sync/data-size budget and five low-risk experiments
  -> subagent_sync_budget_and_experiments_20260523.md

Anscombe: 10 architecture designs for 2x/5x/10x
  -> subagent_architecture_designs_2x_5x_10x_20260523.md

Feynman: validation gate audit before promotion
  -> subagent_optimizer_validation_gate_audit_20260523.md
```

Wave result:

```text
All four sidecars completed and were closed.

Main correction:
  The LightZero comparator lane is not the same as the older resident MCTX
  currency. It can still bounce through host stacks and CPU CTree/list work.

Main bottleneck hypothesis:
  direct_ctree_gpu_latent is suspicious because it repeats CPU CTree traverse,
  GPU recurrent inference, GPU-to-CPU output, Python listification, and CPU
  backprop once per simulation.

Main architecture ladder:
  2x  -> compact service made real and replay-valid
  5x  -> compact service + fixed-shape search + compact env/replay ownership
  10x -> service/device-resident architecture with many roots in flight

Main validation blocker:
  actual search-selected actions must drive the next env step and then land in
  the same replay/RND/player-perspective rows.
```

Local implementation follow-through:

```text
Added compact_search_result_v1_from_arrays.
Refactored direct/profile replay proof to use it.
Added no-double-run array validation tests.
Kept live Coach runs untouched.

Validation:
  ruff passed
  tests/test_compact_search_replay_contract.py -> 10 passed
  tests/test_source_state_batched_observation_boundary_profile.py -> 108 passed
  tests/test_source_state_hybrid_observation_profile.py -> 35 passed
```
