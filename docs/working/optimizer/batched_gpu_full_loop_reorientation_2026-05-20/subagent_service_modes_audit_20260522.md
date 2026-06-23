# Service Modes Audit, 2026-05-22

Status: docs-only optimizer subagent note. I did not touch live Coach
training, trainer defaults, checkpoints, evals, GIFs, tournaments, Modal
volumes, or source code.

Scope:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- compact search/replay docs in this folder
- the profile-only modes named `mock_search_service`, `service_tax_probe`,
  `direct_ctree_gpu_latent`, and `hybrid_compact_service_replay_proof`

## Short Read

The current service modes are useful, but they are not all measuring the same
thing.

Plain map:

```text
mock_search_service
  real env/render/stack/model initial inference
  fake search
  answers: "If search were cheap, is the rest of the compact path fast enough?"

service_tax_probe
  real env/render/stack/model initial inference/recurrent inference
  fake tree/search bookkeeping
  answers: "If we paid model rollout cost but avoided CTree/list/search control,
  how much headroom is left?"

direct_ctree_gpu_latent
  real env/render/stack/model initial inference/recurrent inference/CTree MCTS
  compact output instead of stock public collect output
  answers: "What is the current best real-search profile comparator?"

compact_service_replay_proof
  validation and replay-edge timing
  not a search mode
  answers: "Can compact search outputs drive the next env step and produce
  replay-index rows with the right record timing?"
```

The strongest conclusion so far is:

```text
Mock and service-tax prove there is search-service topology headroom.
They do not prove a 10x win by themselves.
The next real implementation must own a compact root/result/replay boundary
and then put a real fixed-shape search body behind it.
```

## Mode-by-Mode Audit

### `mock_search_service`

Code owner:

```text
_LightZeroArrayCeilingStackProbe
mode = mock_search_service
```

How it is entered:

```text
hybrid_observation_canary=true
observation_renderer_backend=jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
hybrid_lightzero_array_ceiling_probe=true
hybrid_lightzero_array_ceiling_mode=mock_search_service
```

What is real:

- CurvyTron profile env step and legal masks from the hybrid profile harness.
- Current GPU/persistent observation renderer and `[B,P,4,64,64]` stack.
- Flattening/filtering active roots from real action masks.
- H2D and normalization according to the selected array-ceiling input mode.
- Scratch LightZero MuZero model construction.
- Real `model.initial_inference(obs_tensor)`.
- Compact output arrays, illegal-action validation, checksums, byte counts.
- Optional public collect-output materialization if
  `hybrid_lightzero_mock_service_materialize_public_output=true`.

What is fake:

- No `policy.collect_mode.forward`.
- No CTree roots.
- No `batch_traverse`.
- No `batch_backpropagate`.
- No recurrent rollout.
- No Dirichlet root noise.
- No real searched value.

Output semantics:

```text
visit_policy = masked softmax(initial policy logits)
selected_action = argmax(masked policy logits)
root_value = initial predicted value
actual_search_simulations = 0
```

What it measures:

It is a compact search-service ceiling. It prices the real input side and the
compact output side if search were cheap. It is best used to decide whether the
service boundary has enough room to justify deeper search-service work.

What it does not measure:

It does not measure MuZero search quality, tree cost, recurrent model cost, or
learning semantics. It must never be compared to real search as if it were a
valid algorithm.

### `service_tax_probe`

Code owner:

```text
_LightZeroArrayCeilingStackProbe
mode = service_tax_probe
```

What is real:

- Everything real in `mock_search_service`.
- Real `model.recurrent_inference(...)` called once per requested simulation.
- Device normalization/H2D/readback for compact output arrays.
- Compact arrays are stored for compact replay proof.

What is fake:

- No CTree.
- No real tree traversal.
- No tree expansion/backprop.
- No root noise.
- No selection from a growing tree.
- No real value backup.

Output semantics:

For each fake simulation:

```text
action_input = previous selected action
network_output = model.recurrent_inference(latent_state, action_input)
selected_action = argmax(masked recurrent policy logits)
visit_count[selected_action] += 1
value_accum += recurrent value
```

At the end:

```text
visit_policy = normalized fake visit counts
root_value = value_accum / fake visit count total
actual_search_simulations = requested simulations
real_ctree_calls = 0
```

What it measures:

It prices a compact service that pays real initial inference, real recurrent
inference, compact output readback, and simple array updates, but does not pay
CTree/list/Python tree control. It is the best current probe for "model tax
plus compact service tax."

What it does not measure:

It is still not MCTS. It does not model tree branching, root noise, visit
backup, PUCT selection, or LightZero CTree object cost. If it is close to
`direct_ctree_gpu_latent`, then CTree removal alone is unlikely to produce a
large win. If it is much faster, then the missing tree/control path is a real
target.

### `direct_ctree_gpu_latent`

Code owner:

```text
_LightZeroCollectForwardStackProbe
arrays_boundary_impl = direct_ctree_gpu_latent
```

What is real:

- Real hybrid env/render/stack inputs.
- Real active-root filtering and legal masks.
- Real LightZero MuZero initial inference.
- Real LightZero CTree roots and `roots.prepare(...)`.
- Real CTree `batch_traverse` and `batch_backpropagate`.
- Real recurrent inference for each simulation.
- Real searched values and visit counts from CTree roots.
- Real selected-action legality checks and compact visit policies.

What is improved versus the stock public facade:

- Latent roots stay on GPU instead of round-tripping full latent state through
  CPU for each search call.
- Output is compact arrays instead of full LightZero public dict output.
- The profile path exposes compact arrays for root/result/replay validation.

What is still not fixed:

- CTree still owns CPU-side tree objects.
- Per-simulation traversal returns Python/list-shaped indices/actions.
- Last actions are rebuilt and copied to device every simulation.
- Recurrent outputs are copied back to CPU every simulation.
- Reward/value/policy arrays are listified for CTree backprop.
- Root legal actions/noises are Python lists.

What it measures:

This is the current best real-search comparator in the profile harness. It is
not the 10x answer; it is the denominator for "can a real compact service beat
our best current LightZero-compatible search boundary?"

What it does not measure:

It is not stock `train_muzero` speed. It is a profile-only compact boundary.
It also does not prove that compact outputs are train-facing until the replay
contract and trainer integration gates pass.

### `direct_ctree_gpu_latent_precomputed_recurrent`

This is a falsifier variant of `direct_ctree_gpu_latent`.

What is fake:

- Recurrent inference is replaced by precomputed zero-like payloads.

What it measures:

It estimates how much of direct CTree wall is recurrent model work versus
CTree/list/control/output overhead.

How to read it:

If this only gives a small gain, recurrent model calls are not the whole wall.
If it gives a large gain, recurrent batching/model execution is the next hot
target. Existing notes say it helped, but did not open a 10x lane by itself.

### `hybrid_compact_service_replay_proof`

Code owner:

```text
run_hybrid_observation_profile(...)
_maybe_run_compact_service_replay_proof(...)
compact_policy_row_bridge.py
```

What is real:

- The previous compact search action is used as the next env step action.
- The proof checks that current `joint_action` equals previous selected search
  actions.
- It builds `CompactRootBatchV1`.
- It validates `CompactSearchResultV1`.
- It builds `CompactReplayIndexRowsV1`.
- It checks action, mask, reward, done, terminal/final flags, row ids, player
  ids, and `to_play` contracts.

What is fake or limited:

- It is still a profile proof, not a replay buffer owner.
- The current hot proof uses index rows, not a full production sampler.
- It does not make the learner train from compact replay.
- It does not make search asynchronous or resident by itself.

Important denominator detail:

The first measured env step can be driven by a warmup search action. The code
tracks warmup-seeded proof rows separately so they do not pollute measured
proof timing.

What it measures:

It prices the compact search-to-replay edge and catches basic record-ordering
bugs. It is a correctness gate and a small cost bucket, not a speedup mode.

## Fair Comparison Rules

Use these rules when comparing rows:

1. Same hardware, same batch size, same actor count, same simulations, same
   `steps`, same `warmup_steps`, same trail/body capacity, same renderer,
   same stack dtype, same input mode.
2. Use aggregate `compact.timings` / measured profile results, not a printed
   last-step telemetry field.
3. Compare `mock_search_service`, `service_tax_probe`, and
   `direct_ctree_gpu_latent` with `hybrid_compact_service_replay_proof` either
   on for all rows or off for all rows.
4. Keep `hybrid_materialize_scalar_timestep=false` for service-boundary rows.
   Scalar timestep materialization prices a different path.
5. Keep public-output materialization off unless the row is explicitly pricing
   the scalar/public output edge. Only `mock_search_service` supports that
   flag today.
6. Do not use `resident_torch_reuse` with compact replay proof. It is a stale
   input ceiling, and the code rejects it for replay proof.
7. Report actual search semantics:

   ```text
   mock_search_service: actual search simulations = 0
   service_tax_probe: actual recurrent calls = requested simulations, CTree = 0
   direct_ctree_gpu_latent: actual CTree MCTS = requested simulations
   ```

8. Separate three numbers in the summary:

   ```text
   total measured roots/sec
   search/probe boundary sec
   compact replay proof sec
   ```

## Current Same-Denominator Read

Latest documented closed compact service rows:

```text
H100 B512/A16, 80 measured, 20 warmup, compact replay proof on

sim16:
  direct_ctree_gpu_latent  5215 roots/sec
  mock_search_service      9767 roots/sec  (1.87x direct)
  service_tax_probe       10926 roots/sec  (2.10x direct)

sim32:
  direct_ctree_gpu_latent  4360 roots/sec
  mock_search_service      7950 roots/sec  (1.82x direct)
  service_tax_probe        5637 roots/sec  (1.29x direct)
```

Plain read:

- Mock says the compact boundary has about a `1.8x` ceiling over current direct
  if search becomes cheap.
- Service-tax says real recurrent/model cost matters, especially at sim32.
- Compact replay proof is small in these rows, around `0.8-1.5%` of measured
  wall.
- This evidence supports a real compact-buffer/search-service prototype, but
  does not yet support claiming a 10x win from wrapper cleanup.

## What Would Make This Real

The sidecar becomes a real compact-buffer/search-service prototype when these
are true:

1. A single `CompactSearchServiceV1` API owns the hot boundary:

   ```text
   CompactRootBatchV1 -> CompactSearchResultV1
   ```

   Implementations can be:

   ```text
   mock_search_service
   service_tax_probe
   direct_ctree_gpu_latent
   future fixed-shape MCTX/JAX or array-native CTree service
   ```

2. The closed profile loop uses service actions for every next env step and
   writes `CompactReplayIndexRowsV1` every step without public LightZero output
   dicts.

3. A sampler/materialization edge can rebuild the same target rows as the
   current object path from compact replay index rows.

4. The search service has a real fixed-shape backend that removes the current
   `direct_ctree_gpu_latent` walls:

   ```text
   Python/list root prep
   per-simulation model-output D2H
   policy/value listification
   CPU CTree API calls
   public output assembly
   ```

5. The train-facing bridge is only considered after the compact profile loop
   passes parity and beats the current direct comparator by a large enough
   margin.

## Concrete 3-Step Next Plan

### 1. Freeze the compact service API in profile code

Add a small protocol/module around the existing dataclasses:

```text
CompactSearchServiceV1.run(root_batch) -> CompactSearchResultV1
```

Then adapt the current three services behind it:

```text
mock_search_service
service_tax_probe
direct_ctree_gpu_latent
```

Goal: one service API, one compact root/result contract, no mode-specific
hidden side channels.

### 2. Build the closed compact loop denominator

Run a profile-only loop shaped like:

```text
env step with previous service actions
-> compact root batch
-> service.run(...)
-> compact replay index rows
-> repeat
```

Use the same command grid for direct, mock, and service-tax. Keep scalar
timesteps/public output off. Keep compact replay proof on. Report total
roots/sec plus the search/probe/replay buckets.

Goal: prove the compact service loop is not only a sidecar probe but a coherent
array-owned collect/replay loop.

### 3. Put one real fixed-shape search body behind the API

Pick one implementation spike:

```text
fixed-shape JAX/MCTX service
```

or:

```text
array-native fixed-A=3 CTree service
```

It must consume compact arrays and return `CompactSearchResultV1`. Keep the
existing `direct_ctree_gpu_latent` service as the oracle/control.

Goal: remove the actual direct CTree walls, not just rename them.

## Kill Criteria

Kill or park a lane if any of these happen:

1. Correctness kill:

   ```text
   illegal selected action
   illegal visit mass
   visit policy does not sum to one
   row/player/policy_env_id mismatch
   player perspective swap
   previous search action does not match next joint_action
   terminal/final observation ordering mismatch
   compact target rows do not match the object path
   ```

2. Boundary-headroom kill:

   ```text
   mock_search_service + compact replay proof is not at least about 1.7-2.0x
   faster than direct_ctree_gpu_latent on the same denominator
   ```

   If mock is not clearly faster, the compact service boundary is not the next
   big lever.

3. Model-tax kill:

   ```text
   service_tax_probe is near direct_ctree_gpu_latent, especially at sim32
   ```

   If this persists, do not polish wrapper code. Move directly to a real
   fixed-shape search body or stop the service lane.

4. Prototype-speed kill:

   ```text
   the closed compact service prototype cannot beat current direct_ctree_gpu_latent
   by at least 1.5x end-to-end after replay proof
   ```

   Then it is not worth promoting as train-facing.

5. Complexity kill:

   ```text
   implementation requires changing Coach training defaults or stock
   train_muzero semantics before profile parity passes
   ```

   Keep it in the sidecar until the contract is proven.
