# Next Architecture Design Matrix, 2026-05-22

Status: docs-only optimizer sidecar note. I reviewed the requested working
docs and did a narrow read of compact root/search/replay bridge code. I did
not change source code, launch runs, touch Coach training, checkpoints, evals,
GIFs, tournaments, or Modal volumes.

## Short Read

The current evidence does not support a renderer-only `5-10x` plan. The
refresh-off compact profile ceiling is now about `1.5-1.6x` over refresh-on,
while direct root-value extraction and replay-index rows are small. The active
wall is the full dataflow boundary:

```text
GPU search action
-> CPU joint_action
-> CPU env/render state update
-> CPU compact/delta preparation
-> H2D renderer/search-input update
-> GPU stack/search
-> small search payloads back to CPU
-> compact replay/RND/target edge
```

A `5-10x` win is still plausible, but only if multiple boundaries are removed
together: compact state ownership, fixed-shape/search-service ownership, and
compact replay/RND/sampler ownership. A single copy trim is now mostly a
`1.05-1.6x` lane.

## Status Legend

| Status | Meaning |
| --- | --- |
| Trainer-facing safe now | Can be used without changing trusted Coach training semantics. Almost none of the speed lanes qualify. |
| Profile-only | Safe as a bounded optimizer/profile experiment, not a training claim. |
| Blocked by parity gates | Do not promote until root/search, replay, terminal/final-observation, RND, and player-perspective gates pass. |

## Design Matrix

| # | Design | One-sentence Description | Expected Win Mechanism | Exact Sync/Copy Removed | Why It Could Fail | Smallest Falsifying Experiment | Status |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | Compact service API wrapper | Put `mock_search_service`, `service_tax_probe`, `direct_ctree_gpu_latent`, and future MCTX/Torch search behind one `CompactRootBatchV1 -> CompactSearchResultV1` service API. | Removes mode-specific side channels and lets all search designs share one compact root/result/replay denominator. | Public LightZero collect-output dict assembly, per-mode compact array extraction glue, and hidden scalar timestep fanout in service profiles. | It may only make comparisons cleaner without moving wall time; current mock/service-tax rows show headroom but not `10x`. | Same denominator, scalar/public output off, compact replay proof on: direct, mock, and service-tax all drive next env actions and write `CompactReplayIndexRowsV1`; kill if mock is not `>=1.7-2.0x` faster than direct. | Profile-only |
| 2 | Puffer-style compact slab | Make one contiguous owner for obs, masks, rewards, done/final flags, row/player ids, search output, and replay indices. | Keeps batches array-native across collect/search/replay instead of rebuilding scalar `BaseEnvTimestep` or per-env dict objects. | Scalar timestep materialization, per-env object allocation, row dict fanout, repeated reward/mask/id copies into LightZero-shaped surfaces. | If search or renderer handoff dominates, slab cleanup alone caps near `1.5x`; trainer integration can recreate the object boundary later. | A compact consumer reads static slabs for one closed loop, no scalar timesteps/public output in hot path, and materialized target rows match object/direct compact rows. | Blocked by parity gates |
| 3 | Persistent compact render-state owner | Keep the renderer's compact state alive and update it in place instead of rebuilding it from production env arrays every step. | Attacks the inclusive `env_step_sec` handoff: production-to-compact, delta pack, renderer H2D, and public packaging. | Full production-state-to-compact conversion, repeated trail-capacity scans, copied render-state rows, and large-ish delta/compose H2D payloads. | Borrowed render state already captured the largest copy; refresh-off ceiling says this lane alone is bounded to about `1.5-1.6x`. Terminal/autoreset snapshots are sharp. | No-death profile canary where compact state owner feeds persistent renderer directly; keep only if matched loop24/48 total roots/sec improves `>=1.2x` and sampled frames match copied path. | Profile-only |
| 4 | Actor-emitted visual deltas | Have the actor/env emit changed trail segments, head/player fields, reset masks, and bonus deltas directly in renderer layout. | Shrinks the next-search-input handoff by making changed bytes scale with live writes, not full trail capacity or parent buffers. | Actor env.state -> parent render-state copy, CPU exact delta pack over unchanged slots, and much of renderer delta H2D. | Cursor wrap, `break_before`, reset, bonus overwrite, and terminal final-frame bugs can silently poison observations. | Seeded no-death plus reset/bonus fixture: actor deltas render identical latest frames and stack checksums; kill if delta bytes or total loop wall do not drop materially. | Blocked by parity gates |
| 5 | Device-resident stack/root input | Treat the GPU resident stack as the search input owner and keep host root observations as sidecar metadata or sampled validation. | Avoids moving the `32 MiB` stack or root observation through host when MCTX already consumes device arrays. | Host stack shift, root observation copy, obs `device_put`, hot-loop host frame readback, and explicit resident-stack readiness blocks where avoidable. | The wait can move into `h2d_sec` or `search_sec`; stale-stack bugs are easy if row/player order drifts. | Report `obs_h2d_bytes == 0` with resident stack, sampled host/device frame parity, and total roots/sec `>=1.2x` over matched host-stack/no-copy row. | Profile-only |
| 6 | Persistent device mask/sidecar buffers | Keep legal masks and small root sidecars in preallocated device buffers, updating only changed slices before search. | Reduces allocation and host-visible readiness churn around tiny but ordering-critical arrays. | Per-loop mask `device_put`, mask `block_until_ready`, and repeated device allocation for `[R,3]` invalid actions. | Bytes are tiny, so the measurable win may be noise; correctness risk is stale legal masks. | Same loop with persistent mask buffer and legality checksum; kill if total roots/sec improves less than `5%` or any forced-mask test fails. | Profile-only |
| 7 | Action-critical split with chunked replay payloads | Read selected actions immediately for CPU env stepping while visit policies/root values commit into ordered compact replay chunks. | Shortens the action-critical path and prevents replay payloads from forcing action-loop synchronization. | Hot-loop blocking on replay-only `action_weights`/root-value payloads before env can step; premature scalar replay row visibility. | Direct root-node extraction made payloads small; serial deferred flushing already showed the cost comes back if ownership does not change. | Action-only loop plus ordered chunk commit, measured including flush; kill if replay-valid total wall is not faster and rows do not match immediate path by record/root identity. | Blocked by parity gates |
| 8 | Array-native fixed-`A=3` CTree | Replace Python legal-action lists and reward/value/policy listification with dense arrays for the fixed three-action CTree boundary. | Removes Python/list ABI overhead while keeping LightZero-compatible CPU CTree semantics. | Legal-action list construction, policy/reward/value `.tolist()`, nested root output dicts, and some per-root public output assembly. | Prior flat-A3 evidence won microbenches but did not win matched full-loop rows; CPU CTree traversal and per-simulation model-output D2H remain. | Same denominator sim16/sim32 direct CTree row with flat-A3 arrays; kill if full-loop throughput is not `>=1.1-1.2x` and legality outputs differ. | Profile-only |
| 9 | Fixed-shape device search body | Put tree arrays, masks, recurrent outputs, visits, values, and selected actions in one fixed-shape JAX/MCTX or Torch/Triton search backend. | Removes the per-simulation CPU CTree boundary and lets search operate over batched dense tensors. | `batch_traverse` Python control return, recurrent model-output D2H, reward/value/policy listification, CPU backprop API, and output dict assembly. | Naive GPU MCTS can be slower because simulations are sequential; PyTorch/JAX interop or model porting can erase the win. | Implement behind the compact service API; require sim16 and sim32 same-denominator rows to beat `direct_ctree_gpu_latent` by `>=2-3x` before deeper work. | Profile-only |
| 10 | Current-model MCTX realism spike | Replace toy MCTX model assumptions with a current-model-compatible fixed-shape search spike or a frozen-equivalence bridge. | Tests whether the large MCTX search-boundary signal survives contact with real model shapes. | LightZero CTree shell, PyTorch output-to-CPU handoff per simulation, and CPU tree object ownership if the model/search can stay device-owned. | A faithful model bridge may require weight conversion, recompiles, or cross-framework copies that recreate the old wall. | Fixed compact roots with legal masks, current/frozen model equivalence checks, and `CompactSearchResultV1` output; kill if interop copy makes it close to direct CTree. | Profile-only |
| 11 | Batched search service with many compact producers | Decouple compact actors from one GPU search/model owner so many row-seat roots keep the search service saturated. | Amortizes model/search work and removes the tight single-batch synchronous collect/search cadence. | Synchronous env-search interleave, per-env action dict boundary, underfilled GPU batches, and immediate scalar replay fanout. | Queue latency, policy staleness, replay ordering, and env-side bottlenecks can eat the win; if one batch already saturates the GPU, service overhead hurts. | Mock service first with multiple producer batches, compact replay proof on, and service actions driving env; kill if the rest of the loop is not several times faster than direct. | Blocked by parity gates |
| 12 | SoA/Numba native-ish CPU env + delta emitter | Keep the env CPU-owned but move hot mechanics, masks, row sidecars, and visual deltas into typed SoA buffers. | Removes Python wrapper/object overhead and gives the renderer/search boundary compact arrays directly. | Per-row Python stepping, payload merge objects, public batch packaging, some production-to-compact conversion, and separate action/mask/reward sidecar copies. | Recent timing says true mechanics are only about `8-11%` of `env_step_sec`; if observation handoff stays dominant, this is support work. | `step_many(action[B,P]) -> masks/rewards/done/deltas` parity against scalar fixtures; require env/update bucket `>=2x` faster and whole loop `>=1.3x`. | Profile-only |
| 13 | Full JAX env/render/search loop | Move CurvyTron state, render/update, stack, MCTX search, and replay summaries into one fixed-shape JAX loop. | Removes the selected-action D2H, CPU env step, CPU render-state packing, H2D deltas, and framework boundary entirely. | GPU action -> CPU env synchronization, CPU production-to-compact, renderer H2D, mask H2D, and most host-side search/replay readiness waits. | Dynamic collision/trail/bonus/reset semantics can cause semantic drift, recompiles, or memory blowup; trainer/learner integration is a separate project. | No-death fixed-shape JIT loop for many steps with sampled scalar oracle parity and compact replay summaries at chunk edge; kill if it recompiles or fails frame/mask/reward parity. | Blocked by parity gates |
| 14 | Compact replay/RND/sampler owner | Store compact search results and row identities in a replay owner, materializing learner tensors and RND inputs only at sampler or validation edges. | Prevents fast collect/search paths from falling back into scalar replay objects and host latest-frame copies. | Per-step target-row materialization, full observation copies into replay rows, RND latest-frame host extraction, and premature sample-visible partial rows. | Current replay-index rows are already small; this only matters if paired with compact service or trainer-like RND/sampler traffic. | Three-record chunk parity with terminal/live, non-prefix active roots, non-identity ids, inactive-root poison, and fake-RND latest-frame reward shaping. | Blocked by parity gates |

## Readout

The designs split into three classes:

| Class | Designs | Read |
| --- | --- | --- |
| Protect current gains | 1, 5, 6, 8 | Useful for cleaner denominators and regression prevention; unlikely to produce `5x` alone. |
| Remove current closed-loop handoff | 2, 3, 4, 12, 14 | Best near-term Amdahl target because refresh-on rows still spend heavily in env/observation/search-input handoff. |
| Try for `5-10x` architecture | 9, 10, 11, 13 | Only plausible if search ownership and compact replay/RND ownership are paired with lower-copy env/observation ownership. |

## Top 3 Recommendations

1. **Freeze the compact service denominator first.**
   Use one `CompactRootBatchV1 -> CompactSearchResultV1 -> CompactReplayIndexRowsV1`
   loop for `direct_ctree_gpu_latent`, `mock_search_service`, and
   `service_tax_probe`, with scalar/public output off and service actions
   driving the next env step. This is the fastest way to separate search
   headroom from env/observation/replay overhead.

2. **Prototype compact render/observation state ownership next.**
   The refresh-off ceiling says another renderer-only canary cannot be the
   whole answer, but the handoff is still the largest repeated closed-loop
   bucket. The smallest serious falsifier is actor-emitted or persistent
   compact deltas that bypass production-to-compact plus delta pack in a
   no-death profile row, with sampled frame parity.

3. **Put one real fixed-shape search body behind the service API.**
   Prefer MCTX/JAX for the first high-upside ceiling because it already showed
   strong search-boundary signal; keep dense Torch/Triton as the Torch-native
   fallback. Do not polish more wrappers unless they feed this API or the
   compact replay/RND edge.

## Tomorrow / Next

First tomorrow: define the single closed-loop comparison contract and run or
prepare the grid around it:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> CompactSearchServiceV1.run(...)
-> CompactSearchResultV1
-> env step with previous selected actions
-> CompactReplayIndexRowsV1
```

Use the same denominator for direct, mock, and service-tax. If mock/service-tax
cannot clear the existing kill criteria, move immediately to compact
render/observation ownership. If they clear the criteria, put MCTX/current-model
or dense fixed-shape search behind the same API before touching any
trainer-facing path.
