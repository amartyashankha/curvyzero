# External Systems Patterns, Wave 2, 2026-05-23

Status: research-only optimizer note. No source code, live Coach runs,
checkpoints, evals, tournaments, Modal state, or trainer defaults were changed.

## Short Read

The external systems agree with the current local read: the next multiplier is
not "more GPU" or "rewrite CTree in C++" by itself. Fast RL systems choose one
owner for hot data, keep many roots/envs alive, batch accelerator work, and
defer scalar objects until logging, validation, or learner sampling.

For this repo, the target shape should be:

```text
CPU vector env writes compact row/player buffers
-> resident uint8 observation stack/root batch
-> device or array-native batched search service
-> CPU receives selected joint actions once per env tick
-> replay receives compact visit/value/index rows, possibly delayed
-> stock LightZero objects only at validation/sample/debug edges
```

Current repo anchors already point this way:

- `HybridCompactBatch` is the pre-scalar boundary of truth.
- `CompactRootBatchV1 -> CompactSearchResultV1 -> CompactReplayIndexRowsV1`
  is the right search/replay ABI.
- `CompactTorchSearchServiceV1` proves the service boundary and replay proof
  plumbing, but its eager Torch tree loop is not yet the big speed win.
- MCTX/JAX sidecars prove architecture headroom, but not Coach/trainer
  readiness with the current PyTorch/LightZero model.

## Sources Checked

Local docs:

- `current_state_audit_20260523.md`
- `subagent_dataflow_sync_budget_20260523.md`
- `full_iteration_dataflow_designs_20260523.md`
- `compact_torch_backend_integration_plan_20260523.md`
- `puffer_style_contiguous_buffer_attach_audit_20260522.md`
- `subagent_external_patterns_20260522_late.md`
- `subagent_external_dataflow_patterns_20260522.md`
- `subagent_fast_rl_architecture_patterns_20260522.md`
- `subagent_external_search_systems_20260522.md`
- `subagent_gpu_mcts_implementation_patterns_20260522.md`

Repo code anchors:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `src/curvyzero/training/compact_policy_row_bridge.py`
- `src/curvyzero/training/compact_search_service.py`
- `src/curvyzero/training/compact_torch_search_service.py`

Primary external sources:

- PufferLib docs: <https://puffer.ai/docs.html>
- EnvPool docs and paper: <https://envpool.readthedocs.io/>,
  <https://envpool.readthedocs.io/en/latest/content/python_interface.html>,
  <https://openreview.net/pdf?id=BubxnHpuMbG>
- Sample Factory architecture and double buffering:
  <https://www.samplefactory.dev/06-architecture/overview/>,
  <https://www.samplefactory.dev/07-advanced-topics/double-buffered/>
- MCTX README and source types:
  <https://github.com/google-deepmind/mctx>,
  <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/base.py>,
  <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/tree.py>
- MiniZero README/paper page: <https://github.com/rlglab/minizero>,
  <https://arxiv.org/abs/2310.11305>
- MuZero-CPP README: <https://github.com/tuero/muzero-cpp>
- LightZero CTree docs:
  <https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html>
- OpenSpiel AlphaZero docs:
  <https://openspiel.readthedocs.io/en/latest/alpha_zero.html>
- SEED RL page:
  <https://research.google/pubs/seed-rl-scalable-and-efficient-deep-rl-with-accelerated-central-inference/>
- EfficientZero supplement:
  <https://openreview.net/attachment?id=OKrNPg3xR3T&name=supplementary_material>
- KataGo analysis engine:
  <https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>

## Concrete Implementation Lessons

### 1. Buffer Layout

Copy PufferLib's memory lesson, not its learner: allocate stable slabs and let
producers write directly into them. PufferLib's current docs emphasize static
contiguous allocations, no reallocation after init, CUDA graph-friendly memory,
chunked env buffers, pinned memory, and async transfers.

CurvyTron hot slabs should be structure-of-arrays, not object lists:

```text
B = physical rows
P = 2 players
R = B * P roots
A = 3 actions

obs_uint8[B,P,4,64,64]              # policy stack
latest_frame_uint8[B,P,1,64,64]     # render/RND latest frame
legal_mask_bool[R,A] or [B,P,A]
reward_f32[B,P]
done_bool[B]
target_reward_f32[R]
joint_action_i16[B,P]
policy_env_id_i64[R]
env_row_i32[R]
player_i16[R]
to_play_i64[R]                      # still DEFAULT_TO_PLAY for this lane
active_root_bool[R]
terminal/autoreset/final_observation sidecars
checkpoint_id/search_impl/model_version metadata
```

Keep `HybridCompactBatch` and `CompactRootBatchV1` as compact facts. Do not
re-expand into `BaseEnvTimestep`, per-env dicts, `PolicyRowRecordV0`, or
GameSegment-shaped Python rows during collection. `CompactReplayIndexRowsV1`
already records `observation_materialized=False`; that is the right hot-path
contract.

Device side should use similar fixed arrays:

```text
root_logits[R,A]
root_value[R]
latent[R,...]
node_visits[R,N]
node_values[R,N]
children_index[R,N,A]
children_prior_logits[R,N,A]
children_visits[R,N,A]
children_rewards[R,N,A]
children_discounts[R,N,A]
children_values[R,N,A]
embeddings[R,N,...]
root_invalid_actions[R,A]
```

That is the MCTX tree shape, translated to CurvyTron fixed `A=3`.

### 2. Batching

Batch across row/player roots first. MiniZero's self-play workers keep multiple
MCTS instances alive and evaluate selected leaves through batched GPU
inference. MuZero-CPP recommends initial and recurrent inference batch sizes
matching actor count so threads batch inference instead of waiting on a model.
KataGo's analysis engine gets much of its serving speed from many positions in
flight.

CurvyTron has a naturally friendly batch:

```text
B physical rows * P player views = R roots
fixed A = 3
fixed simulation count
same observation shape
same legal-mask width
```

The implementation lesson is to preserve that batch until the search result.
Independent per-seat roots from the same physical snapshot are fine; committing
one simultaneous `joint_action[B,P]` is the semantic boundary. Do not silently
switch to sequential player decisions or a centralized `A=9` joint-action
search unless it is labeled as a different algorithm.

For CPU env scheduling, EnvPool and Sample Factory suggest two useful
patterns:

- Separate send/recv style env progress from policy/search progress when there
  is variance or waiting.
- Use double buffers or split workers so CPU env stepping can proceed while
  another batch waits on inference/search.

For this repo, that means producer/consumer chunks with stable row ids, not
anonymous async results. Every chunk must carry `env_row`, `player`,
`policy_env_id`, `record_index`, and model/search metadata.

### 3. Sync Cadence

Accept these syncs while the env remains CPU-owned:

- selected actions `[R]` or joint actions `[B,P]` back to CPU once per env tick;
- legal/action masks to the search device once per tick if they are not already
  resident;
- visit policy `[R,3]`, raw counts `[R,3]`, and root value `[R]` at replay
  commit or chunk flush;
- terminal final-observation snapshots when terminal rows exist;
- validation mirrors and timing barriers outside the promoted hot path.

Treat these as bad syncs:

- per-simulation recurrent output GPU -> CPU -> list -> CTree backprop;
- per-simulation action CPU -> GPU if it serializes the model/search loop;
- full visual stack GPU -> CPU -> GPU just to feed search;
- scalar LightZero timestep materialization before search/replay actually need
  a compatibility object;
- full observation/next-observation target row materialization during
  collection.

The desired service cadence is two phase:

```text
search_step(root_batch) -> selected_action_cpu + replay_payload_handle
flush_replay_payload(handle) -> visit_policy + raw_counts + root_value + diagnostics
```

The selected action is on the env-critical path. Visit/root-value payloads are
on the replay-critical path and can be delayed or overlapped if the chunk is not
sample-visible before flush.

### 4. Replay Ownership

Replay must be an owner, not a dump of whatever the collector happened to
materialize. The external Zero-family systems separate actors/search/replay/
learner; EfficientZero's supplement is especially explicit about self-play
actors, replay, CPU context workers, GPU batch/reanalysis workers, queues, and
learner.

For CurvyTron, keep replay rows index-first:

```text
record_index, next_record_index
compact_root_row
policy_env_id, env_row, player
action
action_mask
visit_policy / raw_visit_counts
root_value
reward, final_reward
done, terminated, truncated
next_final_observation_row
to_play
policy_source/search_impl/model_version
```

Rows become sample-visible only after action, reward, done, visit policy, root
value, final-observation sidecars, and RND sidecars are attached. Attach by
stable ids, not by current batch position. Terminal rows must use the final
observation before autoreset. RND must read the same row/player latest frame
that the policy saw.

The compatibility bridge to LightZero target rows is valuable as a parity test.
It should not be the collection hot path for the optimized lane.

### 5. Inference/Search Service Shape

Keep `CompactSearchServiceV1` as the public optimizer boundary, but evolve the
implementation so the service owns scheduling and search state:

```text
input:
  CompactRootBatchV1
    observation view or device handle
    legal_mask[R,3]
    active_root_mask[R]
    env_row/player/policy_env_id/to_play

service owns:
  initial inference
  root prep
  tree arrays
  recurrent loop
  root noise/eval-mode policy
  device sync and readback cadence

output:
  selected_action[active]
  visit_policy[active,3]
  raw_visit_counts[active,3]
  root_value[active]
  optional predicted logits/value diagnostics
  metadata with search_impl, num_simulations, model/checkpoint id, fallback counts
```

Direct LightZero CTree should stay the semantic oracle. A fixed-shape Torch,
Triton/CUDA, array-native CTree, or JAX/MCTX backend can sit behind the same
contract only if it returns a validated `CompactSearchResultV1` and passes the
same replay-index materialization gates.

MCTX is the cleanest design reference because its root output, recurrent
function, policy output, and tree are batch-first JAX arrays. It is not a cheap
patch if the current PyTorch model is called through host callbacks; that would
recreate the bad boundary.

### 6. What Not To Copy

- Do not copy PufferLib's PPO/V-trace-style learner or model choices. Copy
  static slabs, chunked buffers, pinned async transfer discipline, and profiling
  honesty.
- Do not copy EnvPool's Gym-compatible scalar surface as the optimized target.
  Copy C++/thread-pool batched env ownership and send/recv identity handling.
- Do not copy Sample Factory's PPO algorithm details. Copy component split,
  shared-memory buffer ids, and double-buffered sampling.
- Do not copy Sampled MuZero as a speed fix. CurvyTron has `A=3`; action
  enumeration is not the problem.
- Do not copy KataGo's transposition/NN-cache story as the main bet. Cross-root
  batching applies; Go-like repeated-state caching is a secondary experiment.
- Do not frame the fix as "make CTree C++." LightZero already has C++
  `batch_traverse` and `batch_backpropagate`; the repo's wall is the boundary
  around them.
- Do not promote eager dense Torch search just because it is on GPU. Dynamic
  indexing, `nonzero`, allocation, and many tiny ops can lose at sim32.
- Do not bridge PyTorch into JAX/MCTX per simulation and call it resident.
- Do not report roots/sec as training speed unless replay, RND, learner sample
  materialization, terminal handling, and stock-vs-candidate parity are inside
  the denominator or explicitly excluded.

## Recommended Next Shape

Build the next optimizer slice as a compact-service dataflow test, not as
another one-off microbench:

```text
HybridCompactBatch ring/slab
-> CompactRootBatchV1 with resident or copy-free observation view
-> CompactSearchServiceV1 two-phase action + replay payload
-> CPU env step consumes selected joint_action[B,P]
-> CompactReplayIndexRowsV1 writes complete index rows
-> sampler/validation materializes full tensors only on demand
```

Promotion gates:

- active-root order and non-prefix roots survive;
- selected actions are legal and drive the next env joint action;
- `env_row`, `player`, `policy_env_id`, and `record_index` attach correctly;
- terminal final observations are not autoreset frames;
- RND latest frames and target rewards match trusted behavior;
- no scalar LightZero rows or full observation target rows appear in the collect
  hot path;
- search backend reports action D2H, replay payload D2H, observation H2D, and
  fallback counts;
- same-denominator sim16 and sim32 rows beat direct CTree before any
  trainer-facing claim.

Plain conclusion:

```text
The real external pattern is compact ownership plus batched search, not a
specific framework. CurvyTron should keep CPU env actions visible, keep visual
stacks and search internals resident where possible, and make replay an
index-first compact owner. Stock LightZero should remain the oracle and
compatibility edge, not the optimized hot loop owner.
```
