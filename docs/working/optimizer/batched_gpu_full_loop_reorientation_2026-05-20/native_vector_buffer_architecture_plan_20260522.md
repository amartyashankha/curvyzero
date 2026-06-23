# Native Vector Buffer Architecture Plan

Date: 2026-05-22

Status: active optimizer planning doc. No live Coach runs touched.

## Plain Read

Our recent changes are too small because they mostly keep the old shape:

```text
Python env rows
-> Python/NumPy LightZero timesteps
-> public LightZero collect/search wrapper
-> per-root dictionaries and lists
```

That shape can be cleaned up, but it is not where 5-10x usually comes from.

The durable H100 falsifier says:

```text
mock_search_service sim16:       11648.29 roots/sec
direct_ctree_gpu_latent sim16:    5303.97 roots/sec
recurrent_toy sim16:              8512.57 roots/sec
```

So compact search-service output is about `2.20x` over current direct in a
profile-only denominator. Useful, but not the full answer.

The bigger architecture must keep more of the loop in compact batches:

```text
CurvyTron state batch
-> action buffer
-> reward/done/action-mask buffer
-> observation buffer
-> batched model/search service
-> compact replay/target rows
```

Scalar Python objects should exist at compatibility edges, not in the hot path.

## Why PufferLib Matters Here

PufferLib is not a MuZero package, so we should not copy its algorithm. The
useful pattern is systems-level:

- static memory;
- contiguous environment buffers;
- no redundant observation copies;
- environment writes directly into training buffers;
- pinned async transfer;
- CUDA graph replay for repeated GPU work;
- native C/CUDA paths where Python object churn would dominate.

Sources:

- <https://puffer.ai/docs.html>
- <https://puffer.ai/blog.html>
- <https://github.com/pufferai/pufferlib>

Fresh repo inspection, 2026-05-22:

- `src/vecenv.h` has a `StaticVec` that owns flat buffers for observations,
  actions, rewards, terminals, and optional action masks.
- GPU mode uses pinned host buffers plus device buffers. Env code writes into
  the host buffer slice it owns, then Puffer copies contiguous chunks to device.
- Each env struct receives pointers into the big shared buffers. The env does
  not allocate and return a new Python object per step.
- The threaded path chunks agents by buffer, calls a network callback, copies
  the buffer's actions down, steps envs with OMP, then copies observations,
  rewards, terminals, and masks back up on the buffer stream.
- `src/kernels.cu` uses a simple allocator pattern: register tensor shapes,
  allocate one contiguous block, then assign tensor pointers as offsets.

Plain translation:

```text
If CurvyTron remains a scalar Python env hidden behind LightZero timesteps, we
should expect small wins. If CurvyTron becomes a native/vector batch provider,
then search/model/replay can consume real batches instead of reconstructed
objects.
```

This does not mean "port to PufferLib" as a trainer. It means the first serious
CurvyTron architecture prototype should copy the buffer contract:

```text
one owner for obs/action/reward/done/mask arrays
workers write into assigned slices
consumer reads ready contiguous slices
scalar LightZero rows are optional compatibility output, not the hot path
```

## Candidate Architecture

### Component 1: Batch State Owner

Own `N` CurvyTron games in contiguous arrays:

```text
positions[N, players, 2]
angles[N, players]
alive[N, players]
trail/cursor buffers
bonus/effect state
reward[N, players]
done[N]
legal_mask[N, players, 3]
```

First implementation does not need to be perfect C. It needs to prove the
contract and the cost. Python/NumPy, JAX, Numba, Cython, or C are all candidates.

### Component 2: Observation Writer

Write observations into a contiguous buffer:

```text
obs_uint8[N, players, 4, H, W]
```

The current persistent GPU renderer can stay as the observation writer while we
test the bigger buffer shape. Later, a C/CUDA renderer can replace it if this
still matters.

### Component 3: Batched Search Service

Consume compact arrays:

```text
obs_uint8[M, 4, 64, 64]
legal_mask[M, 3]
to_play[M]
```

Return compact arrays:

```text
action[M]
visit_policy[M, 3]
root_value[M]
```

The service must preserve MuZero semantics. Mock service rows are ceilings only.

### Component 4: Replay Edge

Materialize stock-looking replay rows only when needed:

```text
compact chunk -> GameSegment/replay rows
```

Do not force the hot collector to create full Python dictionaries per root if a
compact replay chunk can carry the same information.

## Smallest Useful Falsifiers

### Falsifier A: Native Buffer Env-Only Driver

Goal:

```text
Can one process advance a large batch of CurvyTron states and write obs/masks
without scalar env objects dominating?
```

Inputs:

```text
N = 512, 1024, 2048
no death and normal death
1, 4, 12 physics ticks per action
obs64 and maybe obs96
```

Output:

```text
state steps/sec
obs writes/sec
bytes moved
terminal/autoreset counts
semantic mismatch against scalar env for seeded rows
```

Kill condition:

```text
If this is only close to the current vector profile manager, the native buffer
rewrite is not yet the main target.
```

### Falsifier B: Compact Search-Service Loop

Goal:

```text
Run a closed profile loop where compact batch state feeds compact search output
without scalar LightZero timestep fanout.
```

First version can use the current `mock_search_service` or `recurrent_toy`.
Second version must use real MCTS/search semantics.

Kill condition:

```text
If compact env/search/replay-shaped loop is not at least 3x above the current
direct train-profile denominator, the path is too expensive for its rewrite
risk.
```

### Falsifier C: Replay Materialization Edge

Goal:

```text
Take compact chunks and materialize exactly the replay/target fields LightZero
needs, then compare target rows against stock for forced masks and clear
preference cases.
```

Kill condition:

```text
If replay/target materialization forces us back into per-root Python objects
inside the hot path, the native buffer work must include replay ownership too.
```

## What Not To Do

- Do not rewrite CurvyTron in C before a compact-buffer prototype proves a big
  enough ceiling.
- Do not call mock search-service a learning algorithm.
- Do not chase exact neutral/tie-heavy visit parity.
- Do not use roots/sec alone as Coach launch advice.
- Do not touch live Coach runs while this is still architecture research.

## Next Work

Update, 2026-05-22:

The first opt-in native actor-buffer falsifier now exists:

```text
HybridObservationProfileConfig(native_actor_buffer=True)
scripts/profile_hybrid_batched_observation_manager.py --native-actor-buffer
```

Matched local zero-observation B512/A16 row:

```text
old actor payload + merge: 40477 timesteps/sec
native actor buffer:       67890 timesteps/sec
```

This is not the final architecture. It proves one PufferLib-shaped piece: actor
fields can be written into parent-owned compact arrays and consumed without
the per-step actor payload/merge path. Renderer-backed rows still need a
compact render-state contract.

1. Use the PufferLib/external architecture critique as the boundary contract:
   preallocated compact buffers, native row/player axes, async ready batches
   only when they beat the synchronous batch, and scalar LightZero objects only
   at measured compatibility edges.
2. Build or delegate a tiny local batch-buffer prototype with seeded scalar
   parity checks.
3. Build or delegate a compact loop profiler:
   batch state -> observation -> mock/recurrent search -> compact replay chunk.
4. Only then decide between:
   - array-native CTree bridge;
   - compact batched search service;
   - native/vector CurvyTron buffer rewrite.

Current preference:

```text
Do not spend a week vendoring CTree unless the native/vector buffer prototype
fails. The external fast-RL pattern suggests the larger object-boundary rewrite
is more likely to produce the missing multiplier.
```

## 2026-05-22 Sidecar Boundary Update

Implemented first compact boundary widening:

```text
HybridCompactBatch
HybridBatchedStackProbe.run_compact_batch(batch)
```

This keeps the old `run(observation, action_mask)` path alive for legacy
profile probes, but the preferred path now exposes enough state to represent a
real compact training chunk:

```text
observation[B,P,4,64,64]
action_mask[B,P,3]
reward[B,P]
done[B]
policy_env_id[M]
policy_env_row[M]
policy_player[M]
target_reward[M,1]
done_root[M]
final_observation
final_observation_row_mask[B]
terminal_row_mask[B]
autoreset_row_mask[B]
episode_step[B]
elapsed_ms[B]
round_id[B]
alive[B,P]
joint_action[B,P]
```

Why this matters:

```text
The previous probe could only answer "can a consumer read obs+mask cheaply?"
The new probe can answer "can a replay/search/RND-shaped consumer stay compact
without scalar LightZero timestep objects?"
```

Validation:

```text
tests/test_source_state_hybrid_observation_profile.py -> 19 passed
```

Local post-update profile:

```text
B512/A16/steps100/warmup20/uint8/no-pickle/no-scalar/native-vector-probe
payload+merge:        ~22.1k timesteps/sec
native actor buffer:  ~30.5k timesteps/sec
```

Next implementation target:

```text
Use HybridCompactBatch as the sidecar contract for larger search/replay/RND
prototypes. The next proof should be matched full-loop speed, not another
isolated helper unless it removes a specific uncertainty.
```

## 2026-05-22 Follow-Up: Real Search Consumer Hook

The first real-consumer hook now exists in the profile-only LightZero boundary:

```text
_LightZeroCollectForwardStackProbe.run_compact_batch(batch)
```

The sidecar has been extended again with:

```text
to_play[M]
active_root_mask[M]
```

Why these fields matter:

- `to_play` pins the fixed-opponent profile convention instead of leaving it
  implicit.
- `active_root_mask` separates "this row has legal action ids" from "this row
  should actually be searched"; done roots are filtered out even if a stale
  legal mask is present.

Current status:

```text
Profile-only direct CTree can consume HybridCompactBatch.
It validates row/player ids, target reward, active roots, and to_play.
It then enters the existing direct CTree arrays path with inactive masks zeroed.
```

This is the right next step because it asks whether real LightZero-shaped
search can use the compact row/player batch, rather than asking only whether a
toy obs+mask probe is fast.

Current status before trainer advice:

1. Corrected Modal smoke with the persistent policy framebuffer backend passed.
2. RND compact latest-frame input proof passed. RND cadence/training throughput
   remains separate.
3. Native actor-buffer row fill/ownership guard passed locally.
4. Compact target-row policy-record adapter passed locally.
5. Matched full-loop A/B after the sidecar gates pass is still missing.

## 2026-05-22 Closed Compact Consumer Update

The first closed compact consumer falsifier now exists locally:

```text
HybridCompactBatch
-> latest-frame RND input
-> mock compact legal action/visit/value arrays
-> compact target validation
```

The important fix was to make RND latest-frame extraction actually latest-only.
The old compact path normalized `[B,P,4,64,64]` and then kept only
`[B*P,1,64,64]`. The fixed path slices the latest channel first.

Best local rows:

```text
B512/A16 closed compact arrays:   57.9k-62.8k timesteps/sec
B512/A16 native-vector mock:      69.1k timesteps/sec
B2048/A16 closed compact arrays:  71.6k timesteps/sec
B2048/A16 native-vector mock:     80.4k timesteps/sec
```

Plain read:

```text
The compact sidecar path is close enough to the local native-vector mock that
it is not the next big standalone wall. The bigger architecture still has to
attack the real collect/search/replay boundary, where stock LightZero creates
CPU/list CTree contracts and per-env Python outputs.
```

Practical shape note:

```text
B2048/A16 was the best local shape in this probe. A32 regressed, so "more actor
partitions" is not automatically better. Prefer larger contiguous chunks unless
a future subprocess/native implementation changes the partition cost.
```
