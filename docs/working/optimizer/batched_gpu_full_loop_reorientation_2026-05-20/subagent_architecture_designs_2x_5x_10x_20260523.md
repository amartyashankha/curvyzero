# Architecture Designs For 2x, 5x, And 10x

Date: 2026-05-23

Role: parallel architecture critique sidecar. No source code, live Coach run,
checkpoint, eval, GIF, tournament, or Modal volume was touched.

## Starting Point

Current same-shape profile numbers:

```text
H100, B512/A16, sim16:
  direct_ctree_gpu_latent:  7,155 steps/sec
  service_tax_probe:      12,461 steps/sec
  mock_search_service:    17,712 steps/sec
```

Plain read:

```text
direct_ctree_gpu_latent is the real LightZero CTree comparator.
service_tax_probe pays real model work but avoids the real CTree.
mock_search_service is a fake-search ceiling.
```

Ratios:

```text
mock / direct:        2.48x
service_tax / direct: 1.74x
mock / service_tax:   1.42x
```

This says there is real headroom in the search/dataflow boundary, but it does
not prove a standalone 10x fix. A 10x full-loop speedup requires removing whole
contracts: scalar LightZero object fanout, CPU/list CTree boundaries, repeated
host/device handoffs, replay materialization, and possibly the current
synchronous one-batch collect/search shape.

## Safety Rules

Any speed path must keep these facts attached to the same record:

```text
root observation
legal mask
selected action
visit policy
root value
reward
done
terminal final observation
player perspective
RND input / RND reward if enabled
```

If a design cannot prove that, it is profile-only.

## Design Matrix

| # | Design | Expected Speedup Range | Main Bottleneck Removed | Semantic Risk | Implementation Risk | Validation Gate |
| ---: | --- | ---: | --- | --- | --- | --- |
| 1 | Compact search service boundary | `1.5-2.5x` if it reaches the service-tax/mock ceiling; `1.0-1.1x` if only used as a wrapper | Mode-specific glue, hidden scalar side channels, action/result plumbing, and repeated public LightZero collect surfaces | Medium: wrong action/result can attach to wrong env row or player | Medium | One `CompactRootBatchV1 -> CompactSearchResultV1 -> CompactReplayIndexRowsV1` path for direct, service-tax, and mock. Service-selected actions must drive the next env step. Non-prefix ids and terminal rows must materialize to trusted target rows. |
| 2 | Replay-row deferral / compact replay owner | `1.05-1.3x` alone; required support for `5x+` | Premature full observation row materialization, scalar target rows, replay object fanout, RND frame extraction in the collect hot path | Medium-high: replay targets can silently shift by one step or wrong player | Medium | Immediate compact rows and deferred compact rows materialize identical learner targets across live, terminal, autoreset, RND, and non-identity policy-env ids. |
| 3 | Device-resident observation stack and search input | `1.2-1.7x` in compact rows; smaller in full stock unless paired with service | Full stack/root observation copies, repeated host stack ownership, unnecessary host readiness waits before search | Medium: stale frame or wrong player perspective poisons training | Medium | `obs_h2d_bytes` near zero in hot path, sampled host/device stack parity, terminal final observation parity, total loop win at least `1.2x` in a same-shape profile. |
| 4 | Persistent env/render compact state | `1.2-1.8x` if it removes remaining observation handoff; bounded by refresh-off rows | Production-state-to-compact rebuilds, delta packing over unchanged state, repeated renderer state movement | Medium: trail wrap, reset, bonus, death, and final-frame bugs | Medium-high | Seeded fixtures covering long trails, resets, bonuses, death/no-death, and player views produce matching 64x64 observations against the trusted renderer; profile shows handoff time drops, not just raw draw time. |
| 5 | Fixed-action array-native CPU CTree | `1.1-1.6x`; maybe `2x` in list-heavy rows | Python legal-action lists, policy/value/reward `.tolist()`, root output dicts, dynamic action plumbing for fixed `A=3` | Low-medium: legal mask/action-index mismatch | Medium | Same CTree semantics, fixed `[N,3]` masks/visits/policies, same selected actions under deterministic tie-breaking, same root values within tolerance; kill if full-loop win is under `10%`. |
| 6 | Batch model calls across more roots / search-service leaf batching | `1.3-3x`; larger only if current GPU is underfilled | Underfilled GPU inference, repeated small recurrent calls, per-root search/model scheduling | Medium: queue order, policy freshness, root-noise handling, replay order | High | A local service with many compact producers keeps GPU occupancy high and returns ordered actions/visits/values. Must beat direct CTree at same sim count and keep replay parity. |
| 7 | More CPU parallelism for env/search shell | `0.8-1.5x` alone; useful only if CPU shell is proven wall | CPU env stepping, public packaging, CPU CTree shell contention | Low-medium: nondeterministic ordering, seeding, and replay row order | Low-medium | Scale actor count / CPU count sweep with fixed seeds. Keep only if throughput scales monotonically after warmup and replay row order remains deterministic. More CPUs getting slower is a bug signal, not a win. |
| 8 | Native SoA / C++ / Numba env and delta emitter | `1.2-2.5x`; support lane for `5x` | Python env wrappers, object payload merge, mechanics-side loops, and separate render-delta construction | Medium-high: game mechanics parity can drift | High | `step_many(action[B,P])` parity against scalar fixtures for collisions, trail crossing, bonuses, stochasticity, death/no-death, resets, masks, and final observations. Must reduce mechanics+handoff, not only mechanics. |
| 9 | Fixed-shape GPU search body behind compact service | `2-5x` if CTree/list/control is the wall; possible path to `10x` only when paired with compact replay/env ownership | CPU CTree traversal/backprop API, per-simulation recurrent output D2H, Python loop/listification, scalar search output | High: MCTS semantics, root noise, legal masks, value/reward transforms, and tie-breaking can diverge | High | MCTX/JAX, Torch/Triton, or custom kernel returns `action[N]`, `visits[N,3]`, `root_value[N]` through `CompactSearchServiceV1`; deterministic no-noise gate matches direct CTree closely enough, stochastic gate matches distribution, and same-shape profile beats direct by `>=2x`. |
| 10 | MiniZero/KataGo-style local search/inference service, eventually device-resident | `3-8x` if batch fill is high; `5-10x+` only after compact env/replay ownership also lands | Tight synchronous collect/search cadence, underfilled GPU, repeated model/search setup, per-env scalar action dicts, CPU env/search sync, scalar replay objects | Very high: request ordering, policy versioning, replay row attachment, latency, and eventually whole-loop semantic drift | Very high | Many env/root producers feed one service; service batches recurrent inference and tree work; ordered compact results drive env; compact replay parity passes. Radical version needs fixed-shape no-death, death/reset/bonus/stochastic canaries before any training claim. |

## Conservative To Radical Ranking

### 2x Class

Most realistic near-term path:

```text
compact search service
+ device-resident stack/search input
+ replay-row deferral at the edge
```

This is the path suggested by the existing numbers. `service_tax_probe` is
already `1.74x` faster than direct CTree on the same shape, and `mock` is
`2.48x`. The first goal is to make that comparison real and replay-valid, not
to polish one small timer.

### 5x Class

This needs multiple removals at once:

```text
compact env/render ownership
+ compact replay owner
+ fixed-shape search body or real batched search service
+ no hot scalar LightZero objects
```

A 5x win is not credible from renderer work alone, replay deferral alone, or
array-native CPU CTree alone. It becomes credible if search no longer crosses
CPU/list/Python every simulation and the observation/replay boundaries stay
compact.

### 10x Class

This is an architecture change:

```text
many compact actors/roots
-> batched search/inference service or device search
-> compact replay/RND owner
-> learner materializes tensors at coarse boundaries
```

External systems point in this direction:

- AlphaZero/MuZero-style systems scale by separating many self-play actors,
  search/inference, replay, and learner.
- MiniZero and KataGo-style systems keep many games/positions in flight and
  batch neural evaluation behind a service.
- MCTX is the clean all-device search reference, but only if the model/search
  boundary stays device-shaped.
- Puffer-style environments show the same lesson on the env side: static
  contiguous buffers beat scalar Python APIs.

The practical first version is not a distributed training rewrite. It is a
local compact search service with strict parity gates.

## What Could Go Wrong

The dangerous bugs are not type errors. They are record-attachment bugs:

```text
search result from root k gets attached to replay row k+1
player 0 and player 1 views swap
terminal final observation is captured after autoreset
RND sees a different frame than the policy saw
legal masks are stale
root noise / tie-breaking changes the action distribution
stochastic no-op or reward shaping is silently omitted
```

That is why every fast lane needs a closed-loop parity gate before it becomes
trainer-facing.

## Recommendation

Do not pick one giant rewrite yet.

Next serious sequence:

1. Finish the compact service denominator: direct CTree, service-tax, and mock
   all use the same compact root/result/replay boundary.
2. Make the service-selected actions drive the next env step and prove compact
   replay rows match the trusted target-row oracle.
3. Put one real fixed-shape search backend behind that service: MCTX/JAX first
   for the all-device ceiling, or array-native CTree first for a conservative
   compatibility bridge.
4. In parallel, reduce observation/env ownership overhead with resident stack
   and persistent compact render state.
5. Only after the compact service + fixed-shape search path clears `>=2x`, test
   many-producer batching for a `5x+` service architecture.

Short version:

```text
2x: compact service boundary made real.
5x: compact service + fixed-shape search + compact replay/env ownership.
10x: service or device-resident architecture with many roots in flight.
```
