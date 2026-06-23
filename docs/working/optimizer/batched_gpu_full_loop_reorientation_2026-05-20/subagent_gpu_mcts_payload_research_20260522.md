# GPU MCTS Payload / Host-Sync Research, 2026-05-22

Status: docs-only sidecar note. I inspected local code and existing working
docs first. I did not touch live runs, Modal jobs, checkpoints, volumes, source
code, or trainer defaults. I did not need fresh web browsing; external claims
below are synthesized from local docs and the source links already captured
there, plus the local MCTX source clone under `/private/tmp`.

## Short Read

Fast GPU search systems avoid host synchronization by making the search loop
one accelerator-owned array program:

```text
fixed root batch
-> device resident model params / observations / masks
-> device resident tree arrays and recurrent outputs
-> one final action or compact result edge
-> coarse replay / learner / metrics edge
```

They do not repeatedly ask Python for `reward`, `value`, `policy`, hidden
state, tree visits, or observation tensors during the inner search. When the
real environment is CPU-owned, selected actions are the legitimate per-step
readback. Full visit policies, root values, tree summaries, rendered frames, and
debug metrics should be read either at a coarse replay/debug boundary or not at
all on the hot cadence.

The current `mctx_synthetic_benchmark.py` has the right MCTX/JAX search core,
but the closed compact loop still has several explicit host/device ordering
points:

- root sidecars and legal masks are rebuilt on CPU;
- the resident device stack is reshaped for search, but `loop_obs` and
  `loop_invalid` are still explicitly blocked before search;
- search is timed by blocking on `action` or `action_weights`;
- `action`, `action_weights`, and root values are read back before CPU env step
  and CPU replay validation;
- the CPU env then renders/updates the next observation and pushes the next
  resident device stack.

That is a good attribution benchmark. It is not yet the minimal-payload loop
that external GPU search systems converge toward.

## Local Ground Truth

Primary code inspected:

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `/private/tmp/mctx-src-optimizer-20260521/mctx/_src/base.py`
- `/private/tmp/mctx-src-optimizer-20260521/mctx/_src/tree.py`
- `/private/tmp/mctx-src-optimizer-20260521/mctx/_src/search.py`

Most relevant local notes:

- `subagent_iteration_dataflow_deep_dive_20260522.md`
- `gpu_mcts_current_flow_explainer_20260522.md`
- `subagent_external_gpu_mcts_dataflow_research_20260522.md`
- `subagent_fast_rl_architecture_patterns_20260522.md`
- `subagent_gpu_sync_residency_designs_20260522.md`
- `subagent_gpu_mcts_implementation_patterns_20260522.md`
- `docs/research/mctx_integration.md`

Key current-code facts:

- `run_search` is a JIT-compiled JAX/MCTX call. It builds root hidden state,
  logits, and value, then calls `mctx.gumbel_muzero_policy`
  (`mctx_synthetic_benchmark.py:2169`).
- The steady-state search microbench blocks on `action_weights` for honest
  timing (`mctx_synthetic_benchmark.py:2203`, `:2217`, `:2232`).
- The fresh end-to-end path does `jax.device_put` for observation and mask,
  blocks on both, runs search, then reads `action` and `action_weights` with
  `np.asarray` (`mctx_synthetic_benchmark.py:2245`, `:2247`, `:2260`, `:2264`).
- The closed-loop path builds a CPU `CompactRootBatchV1`, converts root
  observation/legal/active masks with `np.asarray`, uses either resident stack
  reshape or host observation H2D, blocks on search input, blocks on search
  output, then reads action/policy/root values to CPU
  (`mctx_synthetic_benchmark.py:2531`, `:2549`, `:2571`, `:2578`, `:2580`,
  `:2593`, `:2600`, `:2604`, `:2648`).
- CPU joint actions drive the CPU compact env step
  (`mctx_synthetic_benchmark.py:2673`, `:2691`).
- The resident visual stack is updated from `renderer.last_output_device` with a
  JAX concat, and optionally blocked immediately
  (`mctx_synthetic_benchmark.py:823`, `:845`, `:2696`, `:2709`).

## Pattern 1: MCTX / JAX-Style Loops

MCTX's core type shape is exactly the anti-readback pattern:

```text
RootFnOutput:
  prior_logits [B,A]
  value        [B]
  embedding    [B,...]

RecurrentFn:
  action [B], embedding [B,...]
  -> reward [B], discount [B], prior_logits [B,A], value [B]
  -> next_embedding [B,...]

PolicyOutput:
  action [B]
  action_weights [B,A]
  search_tree Tree[B,...]
```

The MCTX tree is dense arrays over `[B,N]` and `[B,N,A]`: node visits, node
values, parents, child indices, child priors, child rewards, child discounts,
child values, root invalid-action masks, and embeddings. Search uses a JAX
loop, vmapped simulation over the batch, expansion through the recurrent
function, and JAX backup. The recurrent payload never needs to become a Python
list or NumPy array.

The useful rule:

```text
read final products, not inner products
```

Final products are actions, visit policy, root value, maybe compact tree
summaries. Inner products are per-simulation rewards, values, policy logits,
latent states, selected leaves, and backup path statistics. If those inner
products leave device each simulation, the loop has already stopped being an
MCTX-style loop.

For this repo, the current `run_search` satisfies the MCTX shape. The
surrounding closed loop does not yet fully satisfy the payload shape because it
still demands host-visible action weights and root values every step for CPU
validation/replay.

## Pattern 2: AlphaZero / MuZero Actor Loops

Classic AlphaZero/MuZero systems usually separate:

```text
actors/self-play
search or batched inference service
replay/storage
learner
evaluators/checkpoints
```

The common high-throughput trick is not "every game state is on GPU." Many
systems keep games and tree control on CPU. The trick is batching many roots or
leaf evaluations so the neural net sees a large stable batch, and accepting
sync only at leaf-batch or action/result boundaries.

Examples already summarized in local docs:

- OpenSpiel AlphaZero separates actors, learner, evaluators, replay, and
  checkpoints. Its C++ path adds threads, cache, batched inference, and GPU
  support.
- MiniZero self-play workers keep multiple MCTS instances alive, select leaves
  across them, and batch GPU inference for those leaves.
- LightZero/EfficientZero keep a CPU/C++ tree but batch model inference. This
  is compatible and pragmatic, but it accepts per-simulation CPU/GPU sync when
  recurrent outputs are copied back to CPU tree structures.
- Gumbel MuZero is relevant because low simulation counts can improve the
  system tradeoff: fewer simulations means fewer opportunities to hit sync and
  object overhead.

The practical CurvyTron read:

```text
B physical rows * P player views = R roots
fixed A = 3
fixed num_simulations
stable compact masks and row/player ids
one selected-action readback per physical tick if env is CPU
policy/root payload readback only where replay truly needs it
```

The current CPU-env loop is allowed to synchronize for selected actions. It
should not be forced to read the full search payload before the next env step
unless CPU replay validation is deliberately in the same denominator.

## Pattern 3: Fast RL Systems

Fast RL systems that are not MCTS-specific point at the same boundary discipline:

- SEED RL centralizes inference/training on the learner side. Actors send
  observations/unrolls and receive actions; the accepted sync is actor-service
  communication, paid to batch accelerator inference.
- PufferLib emphasizes static allocations, chunked buffers, pinned or async
  transfers, CUDA streams, and CUDA graph-friendly boundaries.
- EnvPool keeps environments CPU-side but uses a C++ batched env pool with
  sync/async `send`/`recv` rather than per-env Python stepping.
- Isaac Gym, Brax, and PixelBrax are the endpoint pattern: env, observation,
  policy, and training all live on the accelerator, so the hot loop only syncs
  at metrics/checkpoint/sample boundaries.

The recurring pattern is compact ownership:

```text
contiguous rows
preallocated buffers
fixed shapes
few Python objects
few host-visible tensors
coarse metadata/checkpoint/replay edges
```

For CurvyTron, `HybridCompactBatch`, `CompactRootBatchV1`,
`CompactSearchResultV1`, and `CompactReplayIndexRowsV1` are the right
direction. The remaining issue is cadence: current validation/replay makes the
full search result host-visible every closed-loop step.

## What Should Stay On Device

These should remain device-resident in the hot search/search-input path:

| Payload | Why |
| --- | --- |
| Visual observation stack `[B,P,4,64,64]` | It is large. At `B=1024`, `P=2`, `uint8`, it is about 32 MiB; as `float32`, about 128 MiB. Avoid GPU latest frame -> host stack -> H2D bounce. |
| Latest rendered frame `[B,P,1,64,64]` | It is about 8 MiB at `B=1024`, `P=2`, `uint8`; it should feed the resident stack directly. |
| Model params and representation/prediction/recurrent activations | JAX/MCTX only works as intended if the recurrent function is pure device code. |
| Hidden embeddings for all expanded nodes | MCTX tree memory is dominated by embeddings at larger `B`, `N`, and hidden sizes; reading them defeats the search architecture. |
| Tree arrays: visits, values, parents, child indices, rewards, discounts, priors | These are the search state. They should be updated by JAX/Torch/CUDA arrays, not Python objects or host lists. |
| Recurrent outputs per simulation: reward, value, policy logits, next embedding | These are the most important "do not read back" payloads. Reading them every sim is the LightZero CPU-tree sync pattern, not the MCTX pattern. |
| Invalid/legal mask if produced on device | Current masks are CPU-generated because env is CPU. If a future device env/render owner can emit masks on device, keep them there. |
| Full visit policy/root value until replay edge | CPU env only needs selected action. Policy/value are replay/training payload, so they can be chunked or staged. |

Small payloads are still sync risks:

| Payload | Approx size at `B=1024`, `P=2`, `A=3` | Why it still matters |
| --- | ---: | --- |
| legal/invalid mask `[R,3] bool` | about 6 KiB | A tiny `device_put` plus `block_until_ready()` can serialize the next search. |
| selected action `[R]` | a few KiB | Legitimate CPU-env barrier. |
| action weights `[R,3] float32` | about 24 KiB | Tiny by bytes, but a full search-completion fence. |
| root values `[R] float32` | about 8 KiB | Same: small payload, large ordering effect. |
| row/player ids, done/reward/final masks | KiB class | Needed by CPU env/replay, but should not force observation/tree readback. |

## Where Host Sync Is Unavoidable

Unavoidable or currently semantically real:

- **Selected action before CPU env step.** The authoritative CurvyTron mechanics
  are still CPU NumPy. The CPU env cannot step until it knows the action.
- **Terminal/final observation semantics while env/autoreset are CPU-owned.**
  Final observation must be captured before autoreset mutates the row.
- **CPU replay validation/writer if used in the same loop.** Current compact
  replay builders consume CPU arrays. If they remain in the hot denominator,
  action weights and root values must be host-visible before replay rows are
  built.
- **CPU CTree boundaries in LightZero-compatible probes.** If the tree is CPU
  C++/Cython, recurrent reward/value/policy payloads must cross back unless the
  tree API is redesigned.
- **Profiling attribution.** `block_until_ready()` and CUDA syncs are needed to
  assign time to buckets. They should be paired with minimal-fence total-wall
  rows before claiming algorithmic necessity.
- **Metrics, checksums, parity samples, artifacts, and debugging.** These should
  be sampled/coarse, not per-step mandatory.
- **JIT compile/profile shape changes.** Host work around compilation and shape
  specialization is real, but it should be separated from warmed steady-state
  timing.

Avoidable or suspicious in a minimal hot loop:

- full observation/frame D2H followed by H2D into search;
- immediate resident-stack `block_until_ready()` when search is the next real
  consumer;
- blocking on `loop_obs` just to label a bucket if total wall does not move;
- reading `action_weights` and root values before the CPU env step when replay
  can be staged;
- `np.asarray(loop_root_batch.observation)` in a resident-stack path except for
  sampled validation;
- per-simulation recurrent output `.cpu().numpy()`, `.tolist()`, or JAX
  `device_get`;
- scalar `.item()` conversions inside the hot search/replay cadence.

## Current MCTX Closed Loop Through The Payload Lens

Current loop:

```text
CPU HybridCompactBatch
-> CPU CompactRootBatchV1
-> CPU np.asarray(root observation / legal mask / active mask)
-> resident stack reshape or host obs device_put
-> mask device_put
-> block on search input
-> JAX/MCTX search
-> block on action or action_weights
-> np.asarray(action)
-> optionally np.asarray(action_weights) and root values
-> CPU compact search validation
-> CPU joint_action[B,P]
-> CPU env step
-> renderer/update latest device frame
-> JAX resident stack concat
-> optional resident stack block
-> CPU replay-index rows
```

The loop is already much better than a full host observation bounce when
resident mode is enabled. The remaining issue is that the "result payload" is
not split by consumer:

- CPU env needs only `action`.
- Replay/validation needs `action_weights`, root values, masks, ids, rewards,
  done/final markers.
- Debug/metrics may want tree summaries.

Right now, the non-action payload is read on the action-critical path whenever
`closed_loop_action_only_profile` is off. That is correct for replay-valid
attribution, but it likely overstates what a production-style fast loop would
need before the next env step.

## Practical Next Experiment

Run this as a profile-only design when live-run constraints allow; do not attach
it to Coach training first.

### Payload-Split MCTX Closed Loop

Add or use a profile mode with three matched variants:

1. **Current replay-valid baseline**

   ```text
   block action_weights
   D2H action + action_weights + root_value
   CPU validate
   CPU env step
   CPU replay-index row per step
   ```

2. **Action-critical ceiling**

   Use the existing `closed_loop_action_only_profile=True` /
   `closed_loop_replay_index=False` shape as the upper bound:

   ```text
   block action
   D2H action only
   CPU env step
   no replay-valid policy/root payload
   ```

   This is not a trainer/replay-valid lane. It prices the unavoidable CPU-env
   action barrier.

3. **Chunked payload staging**

   New profile-only middle lane:

   ```text
   per step:
     run_search
     D2H action only for env step
     keep action_weights/root_value as device arrays or append to pinned chunk
     CPU env step

   every K steps or at loop end:
     one batched device_get of action_weights/root_values
     build CompactReplayIndexRowsV1 for the chunk
   ```

   The goal is not to eliminate replay payload. The goal is to move policy/value
   payload readback out of the action-critical path and measure whether one
   coarser transfer plus replay build beats per-step readback/validation.

### Required Measurements

Use the same denominator across variants:

- active roots/sec and total loop wall;
- `search_sec`, `d2h_sec`, `env_step_sec`, `replay_index_sec`, residual;
- number of host syncs per loop step;
- bytes read D2H for action, policy, value, frames;
- bytes written H2D for observation, mask, render deltas;
- illegal selected-action count;
- row/player checksum for selected actions driving `joint_action`;
- replay row parity for the chunked variant against the current per-step writer.

Pass signal:

```text
action-only ceiling is materially faster than current replay-valid baseline,
and chunked payload staging recovers a meaningful part of that gap while
preserving compact replay rows.
```

Kill signal:

```text
action-only barely moves total wall, or chunked staging only moves waits into
search/env/residual without increasing active roots/sec.
```

Why this is the right next payload experiment:

- It directly tests the external-system pattern: final action is hot, full
  policy/value payload is replay-facing and can be coarser.
- It uses the current MCTX/JAX lane rather than inventing another search engine.
- It keeps CPU env semantics intact.
- It does not require claiming that MCTX proves production training.
- It separates "small payload bytes" from "large synchronization consequence."

## Sources / Pointers

Local pointers:

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `docs/research/mctx_integration.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_external_gpu_mcts_dataflow_research_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_iteration_dataflow_deep_dive_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_gpu_sync_residency_designs_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/gpu_mcts_current_flow_explainer_20260522.md`

External sources already captured in local docs:

- MCTX: <https://github.com/google-deepmind/mctx>
- MCTX core types: <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/base.py>
- MCTX tree: <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/tree.py>
- MCTX search: <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/search.py>
- AlphaZero paper: <https://arxiv.org/abs/1712.01815>
- MuZero paper: <https://arxiv.org/abs/1911.08265>
- Gumbel MuZero: <https://openreview.net/forum?id=bERaNdoegnO>
- MiniZero: <https://github.com/rlglab/minizero>
- OpenSpiel AlphaZero docs: <https://openspiel.readthedocs.io/en/stable/alpha_zero.html>
- LightZero tree docs: <https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html>
- PufferLib docs: <https://puffer.ai/docs.html>
- SEED RL: <https://github.com/google-research/seed_rl>
- EnvPool: <https://github.com/sail-sg/envpool>
- Isaac Gym paper: <https://arxiv.org/abs/2108.10470>
- Brax paper: <https://arxiv.org/abs/2106.13281>
