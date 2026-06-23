# GPU MCTS / Parallel Search Architecture Follow-Up

Date: 2026-05-22

Scope: research and code synthesis only. No live runs touched. No code edited.

## Plain Answer

The blocker is not simply "MCTS is not on GPU."

The blocker is the current boundary shape:

```text
compact CurvyTron batch
-> LightZero scalar/object collect surface
-> CPU/list CTree API
-> Python simulation loop
-> GPU recurrent model
-> reward/value/policy copied back to CPU lists every sim
-> stock replay/RND/learner objects
```

GPU MCTS becomes useful only if we also change that shape. A CUDA tree that
still copies tiny tensors to CPU each simulation and writes Python replay
objects will not produce a 10x full-loop speedup.

## What Other Systems Do

### MCTX

Pattern:

```text
Root arrays + embedding
-> JAX recurrent_fn inside search loop
-> dense batched tree arrays
-> compact action/action_weights/root summaries
```

MCTX is the clean all-device reference. It is JAX-native, supports JIT
compilation, and its search operates on batches in parallel. Its MuZero policy
returns selected actions and action weights that can be used as policy targets.

Repo fit:

- Good architecture target.
- Bad drop-in patch for us because the current learner/model stack is PyTorch
  LightZero. Calling PyTorch from JAX would recreate the host boundary.
- Useful as a scratch benchmark: `[B,2,4,64,64] -> tiny JAX model -> MCTX`.

Source: <https://github.com/google-deepmind/mctx>.

### MiniZero

Pattern:

```text
server
-> many self-play workers
-> each worker owns multiple MCTS instances
-> collect leaves across trees
-> batch GPU inference
-> send completed games to storage
-> optimizer samples replay and updates network
```

This is the closest conceptual match. The important bit is not just C++.
MiniZero keeps many MCTS instances alive so neural inference is batched.

Repo fit:

- Very good. CurvyTron naturally has `B physical rows * 2 player views` roots.
- The service shape should be many row/player roots, batched recurrent calls,
  compact visit/action/value arrays out.

Source: <https://github.com/rlglab/minizero>.

### KataGo

Pattern:

```text
many positions in parallel
-> cross-position neural batch
-> NN result cache
-> tune positions, search threads, and NN batch size
```

KataGo's analysis engine is faster when it can batch many positions on modern
GPUs. It also caches neural net results when search hits repeated positions.

Repo fit:

- Cross-position batching maps well to row/player roots.
- The cache matters less for CurvyTron because visual continuous-ish states
  probably repeat less than Go positions.
- The lesson is still strong: saturate the GPU with many roots, not one tiny
  tree at a time.

Sources:

- <https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>
- <https://github.com/lightvector/KataGo>

### OpenSpiel AlphaZero

Pattern:

```text
actors generate MCTS self-play
learner trains from generated games
evaluators measure progress
fast C++ path uses threads, shared cache, batched inference, GPU support
```

OpenSpiel explicitly contrasts the slower Python path with a C++ path that
adds batching and GPU support.

Repo fit:

- Supports separating actor/search/learner roles.
- Confirms that "more hardware" needs a batched evaluator/search boundary.
- This does not say "rewrite everything in C"; it says remove Python from the
  hot search/evaluator path.

Source: <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>.

### PufferLib

Pattern:

```text
static contiguous buffers
env chunks assigned to workers/streams
pinned async transfer
CUDA graph replay for stable GPU work
no per-step Python object allocation in the hot path
```

PufferLib is not a MuZero system, but it is highly relevant to our env/replay
side. It shows the buffer ownership pattern we are missing.

Repo fit:

- Very good for CurvyTron state/action/reward/done/mask buffers.
- Useful for replay/RND too: write compact chunks, do not create stock
  LightZero objects until a compatibility edge.

Source: <https://puffer.ai/docs.html>.

### Batch MCTS / Batched Neural Search

Pattern:

```text
select many leaves
-> evaluate them as one neural batch
-> store inference results in table/cache
-> update tree statistics after batch results return
```

The core idea is simple: GPU inference on a batch is much faster than one state
at a time. The tricky part is choosing leaves while some evaluations are
pending without changing search semantics too much.

Source: <https://arxiv.org/abs/2104.04278>.

### SEED RL

Pattern:

```text
many actors
-> centralized accelerator inference
-> optimized communication layer
-> learner updates centrally
```

SEED is not MuZero/MCTS, but it supports the same systems principle: keep
accelerator inference centralized and batched instead of distributing small
model calls through many Python actors.

Source: <https://research.google/pubs/seed-rl-scalable-and-efficient-deep-rl-with-accelerated-central-inference/>.

## What We Already Know Locally

From local docs:

- Current full-loop `direct_ctree_gpu_latent + output-fast` is only about
  `1.28x-1.31x` over matched stock profile rows.
- The profile-only `mock_search_service` sim16 row was about `2.20x` faster
  than `direct_ctree_gpu_latent`, but it skipped real MCTS and replay.
- H100 no-model CTree rows show current CTree/list at about `0.45M-0.76M`
  nodes/sec, while a fake flat array update reaches about `13.9M-19.4M`
  nodes/sec. The API is expensive, but raw CTree alone is not the whole wall.
- LightZero already uses C++ CTree for core `batch_traverse` and
  `batch_backpropagate`. The remaining problem is the Python/list/CPU contract
  around those calls.

Local source anchors:

- `current_hot_path_bottleneck_map_20260522.md`
- `gpu_parallel_mcts_research_synthesis_20260522.md`
- `array_native_ctree_opportunity_20260522.md`
- `compact_search_replay_contract_plan_20260522.md`
- `native_vector_buffer_architecture_plan_20260522.md`
- `search_boundary_next_wave_20260522.md`

## Most Realistic Pattern For This Repo

The realistic path is:

```text
MiniZero/KataGo-style batched search service
+ Puffer-style compact buffers
+ LightZero-compatible validation adapter
```

Not this:

```text
rewrite all of LightZero into JAX/MCTX immediately
```

And not this:

```text
add a naive CUDA MCTS kernel but keep the same Python/CPU/list/replay boundary
```

The practical staging should be:

1. Keep `HybridCompactBatch` as the boundary of truth.
2. Build compact roots in arrays, not env-id dictionaries.
3. Run batched search over `M = B * players` roots.
4. Return compact `selected_action[M]`, `visit_policy[M,3]`,
   `root_value[M]`.
5. Write compact replay chunks directly.
6. Materialize stock LightZero rows only for parity/eval/debug.

Near-term backend should stay close to LightZero semantics:

```text
real PyTorch LightZero model
+ real MuZero recurrent calls
+ fixed-A=3 array-native CTree boundary
```

That is more realistic than MCTX because it avoids a framework migration, and
more realistic than a full CUDA tree because it first removes the proven ABI
cost.

## What Could Go Wrong

- **Naive GPU search can be slower.** CurvyTron has only `A=3`; one tiny kernel
  per root or per simulation can lose to launch overhead.
- **Tree parallelism can serialize.** Shared-tree updates hit hot root edges
  and need atomics, locks, or virtual loss. Parallel roots are safer than many
  threads fighting inside one root.
- **MCTX requires JAX ownership.** A PyTorch-to-JAX bridge would lose the main
  benefit.
- **Dense Torch search can graph-break.** `nonzero`, dynamic active-root
  counts, variable legal lists, `.item()`, allocation inside the loop, and
  Python exceptions all fight `torch.compile` and CUDA graphs.
- **Array-native CTree can be too small a win.** If we only clean root/output
  and leave per-simulation list conversion, it will not move full-loop wall
  much.
- **Replay can become the new wall.** Faster search is not enough if every
  step still builds stock GameSegment/replay objects in Python.
- **RND can become the new wall.** RND latest-frame extraction, reward edits,
  CPU hashes, and metrics need the same compact-batch treatment.
- **Tie-heavy exact parity is a bad gate.** Separate CTree calls may choose
  different equal actions. Use forced-mask, clear-preference, legality, value,
  visit-distribution, and statistical gates.
- **Simultaneous-action semantics can drift.** The compact row/player roots
  must both search from the same physical pre-step snapshot, then commit one
  joint action.
- **Bigger batches can change training behavior.** In synchronous runs this is
  controlled, but huge batches delay policy feedback. Every chunk needs
  checkpoint/search metadata.

## Exact First Prototype To Build

Build a profile-only `CompactBatchedSearchReplayV0` lane.

Do not touch live runs. Do not make it Coach default.

### Input

Use existing `HybridCompactBatch` and convert it into:

```text
CompactRootBatchV1
  obs_uint8_or_float[M,4,64,64]
  legal_mask[M,3]
  active_root_mask[M]
  target_reward[M,1]
  done_root[M]
  row[M]
  player[M]
  to_play[M] = -1 for fixed-opponent lane
  terminal/final_observation/autoreset sidecars
```

### Search Service API

Add a profile-only service interface:

```text
search(batch: CompactRootBatchV1) -> CompactSearchResultV1
```

with output:

```text
selected_action[N]
visit_policy[N,3]
raw_visit_counts[N,3]
root_value[N]
predicted_value[N]
policy_logits[N,3]
implementation metadata
```

### Backend 1: Current CTree Service Wrapper

First backend should wrap the existing `direct_ctree_gpu_latent` logic but keep
the service boundary explicit. This is a control. It should prove that the new
contract can reproduce today's profile-only output without stock
`collect_mode.forward` or scalar timesteps.

Pass:

- same legality checks as current direct path;
- same target-row adapter output on deterministic fixtures;
- clear telemetry for root prep, recurrent model, CTree traverse/backprop,
  output arrays, replay chunk write.

### Backend 2: Precomputed Recurrent Falsifier

Same CTree service shape, but replace recurrent model calls with resident
synthetic reward/value/policy tensors.

Question:

```text
Is the wall recurrent model launch/output handling, or CTree/list/control?
```

If this barely speeds up, CTree/control/replay is the wall. If it jumps, model
launch/D2H/recurrent handling is the wall.

### Backend 3: Fixed-A=3 Array-Native CTree

Only after Backend 1 and 2 are measured, add an array-native CTree ABI:

```text
prepare_arrays(value[N], policy[N,3], legal[N,3], noise[N,3], to_play[N])
traverse_arrays() -> path_index[N], batch_index[N], action[N], virtual_to_play[N]
backprop_arrays(reward[N], value[N], policy[N,3], virtual_to_play[N])
output_arrays() -> visits[N,3], root_value[N]
```

This can be CPU C++/Cython first. It does not have to be CUDA on day one. The
first win is removing Python nested lists and `.tolist()` from the simulation
loop while keeping LightZero semantics close.

### Compact Replay Edge

In the same prototype, write:

```text
CompactReplayChunkV1
  obs/action/reward/done/to_play/action_mask
  visit_policy/raw_visit_counts/root_value/selected_action
  row/player/checkpoint/search metadata
  final_observation sidecars
```

Then compare compact target rows against the existing target-row builder on:

- live row;
- terminal/autoreset row;
- both players;
- non-prefix active roots;
- RND latest-frame sentinel.

### First Profile Matrix

Run only profile jobs:

```text
B512/A16, sim8 and sim16, H100
B2048/A16, sim16, H100 if B512 passes
no-RND first
rnd_meter_v0 as a separate axis after no-RND is clean
normal-death/autoreset after no-death is clean
```

Compare:

```text
stock facade
direct_ctree_gpu_latent service wrapper
precomputed recurrent falsifier
array-native A=3 CTree
mock compact search service ceiling
```

### Success Criteria

For a search-service prototype:

- `array-native A=3` beats `direct_ctree_gpu_latent` at sim16 on the same
  denominator by at least `1.3x`.
- compact replay construction does not erase the gain.
- legality, masks, row/player identity, `to_play`, terminal/final observation,
  and RND latest-frame sentinels pass.

For a 5-10x thesis:

- the closed compact env/search/replay profile must be at least `3x` above the
  current stock-compatible direct profile denominator before deeper CUDA/JAX
  work is justified.

## Recommendation

Do not spend the next major effort on "MCTS on GPU" as a slogan.

Build the compact batched search/replay service first, with the current CTree
as a control and fixed-A=3 array-native CTree as the first real replacement.

Use MCTX as the clean reference and optional toy benchmark. Use MiniZero and
KataGo as the production shape. Use PufferLib as the buffer/replay ownership
shape.

The core architecture to test is:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> batched search service over M=B*2 roots
-> CompactSearchResultV1
-> CompactReplayChunkV1
-> stock LightZero adapter only for validation/debug
```

That is the shortest path that could plausibly turn our current `~1.3x`
cleanup into a real multi-x training-loop improvement.
