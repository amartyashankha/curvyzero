# Next Big Architecture Recommendation, 2026-05-22

Scope: read-only architecture recommendation for the CurvyTron optimizer lane.
No live Coach runs touched. No code changes.

## Plain Recommendation

The next change most likely to produce a `>3x` full-loop speedup is not another
renderer patch, CPU-count sweep, flat-A3 CTree patch, or stock LightZero wrapper
cleanup.

The next big change should be a compact batched actor/search/replay service:

```text
compact CurvyTron batch state
-> compact visual observations and legal masks
-> batched device-resident search
-> compact action / visit-policy / root-value arrays
-> compact replay/RND/target arrays
-> stock LightZero objects only for validation, eval adapters, or migration
```

In simpler words: keep the batch together. Stop turning every row/player view
back into Python dicts, lists, per-env timestep objects, and per-simulation
CPU/GPU handoffs inside the hot loop.

## Why This Is The Bigger Move

Current local docs say the small patch lane has a real but limited ceiling:

- matched stock-loop direct CTree/output-fast rows are about `1.28x-1.31x`;
- flat-A3 and dense Torch search are useful falsifiers, not a 10x path;
- real compact visual MCTX rows show `10x`-class search-boundary headroom, but
  the replay/env/observation edge becomes the next Amdahl wall;
- compact sidecars and compact RND/target rows are already cheap enough in
  local profiles that the next wall is the closed loop, not isolated target
  materialization.

External systems point in the same direction:

- MCTX is the clean reference for JAX-native, JIT-friendly, batched MCTS on
  accelerators: <https://github.com/google-deepmind/mctx>
- MiniZero keeps multiple MCTS instances alive per worker and batches leaf
  neural evaluation on GPU:
  <https://github.com/rlglab/minizero>
- KataGo gets speed from analyzing many positions at once through
  cross-position batching:
  <https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>
- OpenSpiel's faster AlphaZero path uses C++ threads, shared cache, batched
  inference, and GPU support instead of scalar Python CPU inference:
  <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>
- LightZero already uses C++ for core `batch_traverse` and
  `batch_backpropagate`, so "rewrite in C++" alone is not the missing idea:
  <https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html>
- SEED/IMPALA-style systems show the broader distributed lesson: actors and
  learners scale when inference/learning is centralized or batched, not when
  every actor owns tiny model calls:
  <https://research.google/pubs/seed-rl-scalable-and-efficient-deep-rl-with-accelerated-central-inference/>
  and
  <https://research.google/pubs/impala-scalable-distributed-deep-rl-with-importance-weighted-actor-learner-architectures/>
- PufferLib is not MuZero, but its speed story matches the systems lesson:
  static contiguous memory, chunked workers, pinned/async transfer, CUDA graphs,
  and avoiding scalar object churn:
  <https://puffer.ai/docs.html>

## What The Architecture Should Look Like

Near-term profile service:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> MCTX/JAX visual-root search or equivalent fixed-shape search service
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1 / CompactReplayChunkV1
-> compact RND latest-frame and target builders
```

Production-shaped destination:

```text
many CurvyTron physical rows alive at once
row/player roots flattened to M = B * P
fixed action count A = 3
static masks for inactive roots and illegal actions
batched initial inference
batched recurrent/search work
compact replay owner
learner samples compact batches
LightZero compatibility adapters are outside the measured hot path
```

MCTX is the best reference implementation for the search side because the tree,
recurrent function, masks, priors, values, visits, and embeddings are all arrays
inside the compiled loop. But MCTX is not a drop-in LightZero patch. If the
existing PyTorch model is called through host callbacks, we recreate the wall.

## Minimal Falsifier

Run one closed compact-loop profile, not another isolated roots/sec benchmark.

Required loop:

```text
seeded compact batch
for K measured decisions:
  build compact root batch
  search with selected implementation
  read back only selected action / visit policy / root value
  commit joint actions to compact CurvyTron state
  update compact visual observation stack
  write compact replay/RND/target edge
report aggregate bucket timings
```

Minimum rows:

| row | purpose |
| --- | --- |
| H100, B512, sim16, current compact MCTX/JAX | best first big-change signal |
| H100, B512, sim32, current compact MCTX/JAX | stress the real search setting |
| H100, B1024, sim16, current compact MCTX/JAX | test batch fill and edge scaling |
| H100, B512, sim16, no-search/mock-service-tax | isolate env/obs/replay edge |
| L4, B512, sim16, current compact MCTX/JAX | cost/perf sanity check |

Primary metric:

```text
closed_loop_active_roots_per_sec
```

Secondary bucket timings:

```text
root build
H2D / D2H
search
env step
render / stack update
compact replay index/chunk write
RND latest-frame / RND update if enabled
target builder / learner-facing adapter
scalar LightZero materialization, if any
```

Pass condition:

```text
The closed compact loop beats the matched stock/direct profile denominator by
at least 3x while preserving legal actions, row/player identity, terminal
final-observation semantics, and RND latest-frame semantics.
```

Kill condition:

```text
If closed compact MCTX/search is fast but env/obs/replay/RND edges swallow the
win, stop advertising search-boundary roots/sec and optimize the largest closed
loop bucket next.
```

## What Could Go Wrong

- MCTX search is fast, but JAX/PyTorch model ownership makes a real model bridge
  slow or brittle.
- The compact env/observation/replay edge dominates once search is accelerated.
- The independent per-seat `A=3` search changes the simultaneous-action problem
  if row/player identity or same-state action selection is mishandled.
- Replay/RND/target code quietly reintroduces scalar LightZero objects in the
  measured loop.
- Terminal and autoreset rows lose the true final observation.
- The profile uses a toy model that is too far from the real learner network.

These are not reasons to stop. They are reasons the falsifier must measure the
closed loop and expose bucket timings.

## What To Stop Doing

Stop spending primary time on:

- renderer-only work unless a fresh closed-loop profile says render/stack is
  again the largest wall;
- CPU-count sweeps;
- H100 vs L4 hardware sweeps before the architecture path is fixed;
- dense Torch MCTS polish without static-shape compile/CUDA-graph evidence;
- flat-A3 CTree polish as if it were the 10x lane;
- isolated roots/sec tables that do not include env/observation/replay/RND
  edges;
- hidden compatibility fallbacks or legacy mode names;
- train-facing claims from profile-only sidecars.

Keep the small patches that already helped, but treat them as scaffolding. The
large speedup requires owning the compact batch across actor/search/replay, not
patching one boundary at a time.

## Concrete Next Step

Finish the closed compact-loop denominator first.

If it clears the `>3x` gate, the next implementation should become:

```text
compact batched CurvyTron actor
-> MCTX-style or fixed-shape search service
-> compact replay/RND/target owner
-> parity adapter to stock LightZero for validation only
```

If it fails, do not guess. Use the measured largest bucket as the next P0:
search model bridge, compact env step, visual stack, replay/RND, or learner
adapter.

