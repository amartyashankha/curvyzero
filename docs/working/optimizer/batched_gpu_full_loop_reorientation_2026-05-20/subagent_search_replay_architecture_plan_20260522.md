# Search Replay Architecture Plan, 2026-05-22

Status: implementation planning note. Do not touch live Coach runs from this
plan. Use profile-only jobs until a trust gate explicitly says otherwise.

## Plain Answer

The blocker is not only "MCTS is not on the GPU."

The blocker is the current search/replay shape:

```text
batched CurvyTron state
-> scalar LightZero env/timestep objects
-> Python dict/list MCTS inputs
-> CTree CPU/list tree calls
-> recurrent model outputs copied/listified each simulation
-> Python action outputs
-> replay/RND/target objects
```

The small `direct_ctree_gpu_latent + output-fast` lane proved that removing
some public LightZero wrapper work helps, but only at about `1.28x-1.31x` in
the matched profile loop. That is not a bad patch. It is also not a 5-10x
architecture.

The realistic 5-10x route is to keep roots, search state, search output, RND
inputs, and replay/target rows in compact batches for much longer. GPU MCTS is
one possible implementation of that route. It is not enough by itself if it
keeps the same scalar/object boundary.

## Current Evidence To Preserve

Existing docs this plan depends on:

- `README.md`: current denominator and plain truth.
- `gpu_parallel_mcts_research_synthesis_20260522.md`: external GPU/parallel
  MCTS synthesis.
- `compact_search_replay_contract_plan_20260522.md`: compact root/search/replay
  contract.
- `current_hot_path_bottleneck_map_20260522.md`: current hot boundary map.
- `search_boundary_next_fix_strategy_20260522.md`: direct CTree and dense-spike
  comparison notes.
- `direct_ctree_promotion_gates_20260522.md`: current direct-CTree trust gates.
- `radical_search_validation_harness_plan_20260522.md`: validation harness
  shape for replacement lanes.

Known numbers:

| Result | Read |
| --- | --- |
| Train-profile direct hook `~1.28x-1.31x` | Useful patch, not enough. |
| H100 direct CTree sidecar `~5k-7.5k roots/sec` class | Better than stock facade, still boundary-shaped. |
| H100 recurrent toy `~8.5k roots/sec` class | Model recurrent work is not impossible to batch. |
| Compact/mock sidecars `~58k-72k timesteps/sec` class | Compact sidecar bookkeeping can be cheap. |
| H100 no-model CTree list `~0.5M-0.8M nodes/sec` | Raw CTree is not the entire wall. |
| fake-flat no-model `~14M-19M nodes/sec` | Array-native tree/state updates have large headroom. |

Plain interpretation:

```text
The profile-only compact path proves there is headroom.
The stock-compatible patch path proves small wrapper removal is capped.
The missing piece is a trusted compact search/replay owner.
```

## Three Lanes

### Lane A: Squeeze Current LightZero CTree Boundary

Goal:

```text
Keep LightZero mostly intact and remove the most obvious avoidable overhead.
```

What this means:

- Continue improving `collect_search_backend=direct_ctree_gpu_latent`.
- Keep the real LightZero model and real LightZero CTree.
- Avoid public `collect_mode.forward` output fanout where possible.
- Reduce recurrent output D2H/list conversion and output assembly.
- Keep replay/learner stock unless a profile proves that boundary is now the
  dominant wall.

Smallest next prototypes:

1. **Precomputed recurrent-output falsifier**
   - Same root batch and CTree calls as `direct_ctree_gpu_latent`.
   - Replace `model.recurrent_inference` with resident synthetic tensors.
   - Purpose: split recurrent-model launch/output handling from CTree/list
     shell.

2. **Model-output conversion split**
   - Time GPU tensor creation, D2H copy, NumPy conversion, and Python list
     conversion separately.
   - Purpose: stop hiding listification inside a generic search bucket.

3. **Array-to-CTree payload slimmer**
   - Keep `value[N]`, `reward[N]`, `policy[N,3]` compact until the last possible
     call.
   - Convert only the active roots required by CTree.
   - Purpose: reduce Python allocation pressure without changing semantics.

4. **Direct hook full-loop repeat**
   - Matched H100 no-RND and RND rows.
   - Same collector count, simulation count, learner calls, death mode,
     checkpoint/eval/GIF off or accounted.
   - Purpose: verify any sidecar win survives the real profile loop.

Tests:

- Existing direct-CTree forced-mask and clear-preference tests.
- Stock-vs-direct statistical comparison with root noise on/off.
- Illegal-action mass must stay zero.
- `to_play=-1`, binary masks, reward support, value/reward transforms, and
  schema identity must be attested in summary output.
- Precomputed recurrent mode must be labeled as a profile falsifier, not a
  semantic training path.

Profiling metrics:

- `steps_per_sec` and `roots_per_sec`.
- `mcts_search_sec`.
- `ctree_batch_traverse_sec`.
- `ctree_batch_backpropagate_sec`.
- `model_initial_inference_sec`.
- `model_recurrent_inference_sec`.
- recurrent call count and mean batch size.
- model-output D2H bytes/sec.
- model-output listify sec.
- direct output assembly sec.
- replay/sample/learner time if run through `train_muzero`.

Promotion gate:

```text
Lane A is worth promoting only if it gives at least a stable 1.5x full-loop
profile win on the same workload with no-RND and RND, and no semantic gate
regresses.
```

Kill criteria:

- If precomputed recurrent output is still close to normal direct CTree, then
  recurrent inference is not the main wall. Stop polishing recurrent handling.
- If sidecar roots/sec improves but train-profile speed stays under `1.5x`,
  the stock replay/collector topology is swallowing the win. Move to Lane B.
- If exact neutral parity keeps failing but forced/statistical gates pass,
  do not block on neutral tie exactness. Keep the statistical gate.

Expected ceiling:

```text
1.3x already proven. Maybe 1.5x-2x if the remaining conversion buckets are
clean. Unlikely to reach 5x because the scalar/object topology remains.
```

### Lane B: Compact Batch/Search/Replay Sidecar In PyTorch/CTree

Goal:

```text
Own the hot training data path in compact arrays while keeping enough LightZero
semantics to compare and trust it.
```

This is the most realistic near-term 5-10x lane.

What this means:

- Treat `CompactSearchReplayV1` as a real contract, not a hidden optimization.
- Keep CurvyTron roots as compact arrays.
- Run search from compact arrays.
- Emit compact search results.
- Build compact replay chunks and target rows without materializing stock
  per-env timestep/action dicts in the hot path.
- Use stock LightZero objects as a validation adapter, not as the main path.

Smallest next prototypes:

1. **Closed compact search/RND/target loop**
   - Input: `HybridCompactBatch`.
   - Search: existing direct CTree compact arrays.
   - RND: latest-frame extraction from compact observation.
   - Target: direct compact target-row builder.
   - Output: compact target rows, no `BaseEnvTimestep`, no `PolicyRowRecordV0`
     unless parity mode is enabled.
   - Purpose: prove the compact sidecar can carry the real facts end to end.

2. **Two-record compact replay writer**
   - Store observations, selected actions, rewards, done/final observations,
     visit policy, searched value, `to_play`, row/player ids, RND reward.
   - Include one live row and one terminal/autoreset row.
   - Purpose: prove replay identity and terminal semantics before speed claims.

3. **Compact replay sampler**
   - Sample a small learner-shaped batch from compact arrays.
   - Compare shape and values against the stock target-row builder for a fixed
     deterministic fixture.
   - Purpose: prove we can feed the learner-shaped boundary without stock
     object fanout.

4. **Array-native fixed-A=3 search sketch**
   - Keep action width fixed at 3.
   - Store visits and priors as dense `[N,3]`.
   - Keep CTree only where still necessary, or build a small fixed-A=3
     selection/backup sketch.
   - Purpose: test whether replacing list/tree glue, not the neural model, is
     the next big win.

5. **Profile-manager full-loop sidecar**
   - Not Coach launch advice.
   - Run a fixed number of compact collect/search/replay/sample/learner-shaped
     steps.
   - Emit a denominator ledger matching the stock profile fields.
   - Purpose: measure full-loop speed without the stock scalar boundary.

Tests:

- `CompactRootBatchV1` validation:
  - observation shape/dtype/schema;
  - binary masks;
  - active-root mask;
  - `to_play=-1` for fixed-opponent only;
  - row/player/env ids;
  - terminal/autoreset/final-observation sidecars;
  - reward support/schema;
  - RND mode and normalization identity.

- Compact replay parity:
  - one-record, two-record, and three-record chunks;
  - mixed live and terminal rows;
  - final observation before autoreset;
  - non-prefix active roots;
  - non-identity row ids;
  - player-perspective swap rejection;
  - RND latest-frame sentinel.

- Search parity:
  - forced single-legal action exact;
  - clear preference exact;
  - root-noise statistical compare;
  - zero illegal visit mass;
  - visit distribution sums to one over legal actions.

- Learner-boundary shape:
  - sampled batch has the same observation/action/reward/value/policy tensor
    shapes expected by the current learner lane.
  - no hidden fallback to stock env/timestep objects in profile mode.

Profiling metrics:

- compact roots/sec.
- physical rows/sec.
- nodes/sec if tree search is included.
- compact replay write rows/sec.
- compact target build rows/sec.
- replay sample rows/sec.
- learner input preparation sec.
- object count allocated in hot path.
- bytes copied host->device and device->host.
- CUDA sync count.
- percent time in env step, render/stack, search, replay/RND/target, learner.

Promotion gate:

```text
Lane B becomes the main optimizer implementation lane if the closed compact
search/RND/target/replay prototype shows at least a 3x profile-loop win over
current direct CTree on matched root count/sim count, with all compact parity
tests passing.
```

Coach-facing gate:

```text
Only recommend it for real training after a small learning proof shows that
the compact path and trusted stock path produce comparable target rows and
the learner consumes the compact samples without hidden semantic drift.
```

Kill criteria:

- If compact search/replay cannot beat current direct CTree by `3x` on a
  matched profile denominator, it is not the 5-10x route.
- If terminal/autoreset, RND reward, or player perspective cannot be made
  exact and simple, stop before scaling.
- If most time moves into learner updates, stop optimizing search and profile
  learner/RND separately.

Expected ceiling:

```text
3x is the first serious gate.
5-10x is plausible only if compact replay and array-shaped search avoid most
stock Python object fanout and keep large root batches alive.
```

### Lane C: MCTX/JAX Or External Framework Sidecar

Goal:

```text
Measure a clean device-resident search architecture without pretending it is a
drop-in LightZero patch.
```

This is the cleanest research reference and the riskiest integration lane.

What this means:

- Prototype a scratch JAX/MCTX visual-root toy.
- Keep it separate from Coach training.
- Use it to answer whether true device-resident batched search is fast for our
  root/action scale.
- Do not start by porting the whole trainer.
- Compare against MiniZero/KataGo/Puffer-style patterns for system design.

Smallest next prototypes:

1. **MCTX visual-root toy**
   - Input: `[R,4,64,64]`, where `R=B*2`.
   - Tiny JAX encoder/prediction/dynamics.
   - `A=3` legal masks.
   - MCTX search with sim8/sim16/sim32.
   - Output: action, visit weights, root value.
   - Purpose: measure true all-device search shape.

2. **MCTX memory scaling sweep**
   - Sweep roots, sims, latent width, and action count.
   - Report compile time separately from steady-state time.
   - Purpose: avoid being fooled by JIT warmup or tree-memory blowups.

3. **Interop boundary test**
   - Copy compact CurvyTron observations from PyTorch/NumPy to JAX.
   - Run toy search.
   - Copy compact results back.
   - Purpose: price the framework boundary if we do not port the env/model.

4. **External architecture comparison note**
   - MiniZero: C++/Python zero-knowledge training loop patterns.
   - KataGo: batched inference/service architecture patterns.
   - PufferLib: contiguous env/replay buffers and async transfer patterns.
   - Purpose: decide what to borrow without cloning a full new project.

Tests:

- MCTX toy must be deterministic under fixed PRNG keys.
- Legal mask must zero illegal visit mass.
- Search output shapes must match `CompactSearchResultV1`.
- JIT compile time must be excluded from steady-state rows.
- PyTorch/JAX interop copy bytes and timings must be reported.

Profiling metrics:

- steady-state roots/sec.
- nodes/sec.
- JIT compile sec.
- device memory peak.
- H2D/D2H or framework-boundary copy sec.
- search result bytes.
- roots/sec per GPU type.

Promotion gate:

```text
Lane C deserves more investment based on real checkpoint-backed MCTX/JAX shadow
rows, not only the old scratch toy. Current evidence is positive on speed
(`2.20x` over direct CTree at B1024/A16/sim8 scalar-off), but semantic deltas
must be explained before Coach-facing promotion.
```

Kill criteria:

- If JAX interop overhead dominates and we are not ready to port the model/env,
  keep MCTX as a reference only.
- If MCTX only wins on toy dynamics but loses after visual-root CNN pressure,
  do not promote it.
- If the required rewrite is "port CurvyTron, model, replay, learner, and RND
  to JAX" before any useful signal, park it until Lane B is exhausted.

Expected ceiling:

```text
Potentially 5-10x or more in search-heavy regimes, but integration risk is
large. This is a research sidecar until it proves a big same-scale win.
```

## Staged Plan

### Stage 0: Make The Denominators Hard To Confuse

Owner: optimizer.

Scope:

- No live Coach runs.
- Profile-only.
- No default trainer changes.

Deliverables:

- One table per run family:
  - stock profile loop;
  - direct CTree hook;
  - compact sidecar;
  - MCTX/external toy.
- Every row must report:
  - workload: roots, physical rows, sims, learner calls, RND mode, death mode;
  - hardware;
  - whether `train_muzero` was called;
  - whether scalar timesteps were materialized;
  - whether stock replay was used;
  - whether outputs are semantic or synthetic.

Trust gate:

```text
No speed claim without a denominator label.
```

### Stage 1: Finish The Lane A Falsifiers

Run:

1. `direct_ctree_gpu_latent` vs
   `direct_ctree_gpu_latent_precomputed_recurrent`.
2. Same shape, H100, sim8 and sim16 if cheap.
3. Enough warmup to avoid first-use noise.
4. No-RND first; RND later only if the no-RND split is meaningful.

Decision:

- If precomputed recurrent is much faster:
  - recurrent call/output handling is a real wall;
  - optimize resident recurrent payloads before rewriting CTree.
- If it is only modestly faster:
  - CTree/list/control and stock topology dominate;
  - move quickly to Lane B.

Kill gate:

```text
Do not spend more than one small patch wave on Lane A unless it clears a
stable 1.5x train-profile win.
```

### Stage 2: Build The Closed Compact Search/RND/Target Prototype

Run:

```text
HybridCompactBatch
-> direct CTree compact search
-> RND latest-frame extraction
-> compact target-row builder
-> compact replay writer
-> compact replay sampler
```

Do not allocate:

- `BaseEnvTimestep` in the hot path.
- public LightZero action dicts in the hot path.
- `PolicyRowRecordV0` unless parity mode is explicitly enabled.

Trust gates:

- Compact replay parity tests pass.
- Terminal/autoreset/final observation tests pass.
- RND latest-frame sentinel passes.
- Forced-mask and clear-preference search tests pass.
- Statistical compare passes within documented tolerance.

Speed gate:

```text
At least 3x over current direct CTree on the same roots/sims, or kill as the
5-10x candidate.
```

### Stage 3: Attach A Learner-Shaped Boundary

Goal:

```text
Find out whether the compact sidecar win survives the point where training
would actually consume samples.
```

Prototype:

- compact replay sampler emits learner-shaped tensors;
- optional adapter converts those tensors to the current learner input format;
- no checkpoint/eval/GIF/tournament sidecars;
- RND enabled in meter mode after the no-RND row is stable.

Trust gates:

- same sample identities as compact replay tests;
- same reward/value/policy target semantics as stock target-row builder;
- deterministic fixture can be replayed twice with same outputs;
- no hidden scalar-object fallback.

Speed gate:

```text
At least 2.5x over current direct train-profile denominator after learner
input preparation is included. If learner/RND now dominates, stop optimizing
search and split that new wall.
```

### Stage 4: MCTX/JAX Toy In Parallel

Run this in parallel with Stage 2, not instead of it.

Minimum useful toy:

```text
[R,4,64,64] uint8/float input
-> tiny JAX encoder
-> MCTX sim8/sim16/sim32
-> action/visit/value arrays
```

Trust gates:

- deterministic PRNG;
- legal masks correct;
- compile time separated;
- interop copy priced;
- output can map to `CompactSearchResultV1`.

Promotion gate:

```text
Only promote if it clearly beats Lane B or current direct CTree by 3x+ at the
same root/sim scale after warmup.
```

### Stage 5: Decide The Real Architecture

Decision table:

| Outcome | Next move |
| --- | --- |
| Lane A reaches `1.5x-2x` but no more | Keep as a safe tactical patch, but do not call it the 5-10x plan. |
| Lane B reaches `3x+` before learner | Build compact replay/learner adapter and keep pushing. |
| Lane B stays under `3x` | The compact sidecar is not enough; use its tests but move to search-service/native-buffer design. |
| Lane C toy reaches `3x+` over direct | Consider a JAX/MCTX or external-framework search service spike. |
| Lane C only wins by toy cheating | Keep as research, not implementation. |
| Learner/RND becomes dominant | Stop search work and profile learner/RND cadence/data movement. |

## Trust Gates Before Any Coach Recommendation

Minimum:

- No hidden trainer defaults changed.
- No live run touched.
- Workload counts match stock profile row.
- Observation schema identity recorded.
- Search backend identity recorded.
- RND mode identity recorded.
- death/autoreset mode recorded.
- checkpoint/eval/GIF/tournament sidecars disabled or explicitly priced.
- illegal action count is zero.
- compact replay tests pass.
- forced-mask and clear-preference search tests pass.
- statistical stock-vs-replacement comparison passes.
- at least one no-RND and one RND meter-mode profile row.

For a 5-10x claim:

- Must include replay/sample/learner-shaped cost, not only search.
- Must run enough warmup.
- Must compare against the current best direct CTree baseline, not an old stock
  baseline.
- Must report Amdahl fractions after the speedup, because the next wall will
  move.

## Highest-Value Next Actions

1. Finish and record the precomputed recurrent-output falsifier.
2. Add explicit model-output listify timing if missing.
3. Build the closed compact search/RND/target/replay prototype.
4. Run same-denominator H100 profile rows:
   - current direct CTree;
   - precomputed recurrent falsifier;
   - closed compact search/RND/target sidecar;
   - recurrent toy ceiling.
5. Start the MCTX visual-root toy in parallel, but do not block Lane B on it.

## Recommendation

The optimizer should not keep trying to get 10x from small LightZero hook
patches.

Use Lane A to finish attribution and keep the best safe tactical speedup. Make
Lane B the main implementation lane because it keeps the current PyTorch/CTree
world but removes the scalar/object boundary that is eating the wins. Run Lane
C as a parallel research spike so we have a clean all-device reference and do
not miss a better architecture.

The current working hypothesis is:

```text
5-10x requires compact batched search/replay ownership.
GPU MCTS helps only if it comes with that ownership.
```
