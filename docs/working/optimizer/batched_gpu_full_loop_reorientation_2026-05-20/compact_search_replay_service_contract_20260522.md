# Compact Search Replay Service Contract, 2026-05-22

Status: optimizer working contract. Profile-only until the tests below pass.
No live Coach runs should use this directly.

## Plain Goal

The next speed lane is not "put one more function on the GPU." The next speed
lane is to stop tearing the training data apart into scalar LightZero objects
inside the hot loop.

The target shape is:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> batched search service
-> CompactSearchResultV1
-> CompactReplayChunkV1
-> learner/replay adapter
```

Stock LightZero objects may still exist for parity checks, evaluation, or
debugging. They should not be the hot-path owner if this lane is meant to chase
5-10x.

## Why This Is The Current Target

The latest H100 falsifier says removing recurrent model calls from direct CTree
only improved the stable B512/A16/sim16 row from `4920.30` to `6771.37`
roots/sec, about `1.38x`.

That means recurrent output handling matters, but it is not the whole wall.
The remaining wall is broader:

```text
CPU/list CTree inputs
Python search-loop control
per-simulation output readback/listification
scalar env-id dicts
replay/target/RND object materialization
```

The outside systems point to the same answer:

- MiniZero: multiple MCTS instances per worker plus batched GPU inference.
- KataGo: cross-position batching and search/inference service tuning.
- MCTX: dense accelerator-native search arrays.
- PufferLib: contiguous static env/replay buffers, scalar objects only at the
  edge.

## Contract Objects

### CompactRootBatchV1

Input to the search service. It is derived from `HybridCompactBatch`.

Required arrays:

```text
observation_uint8[M,4,H,W]
legal_mask[M,3]                 # binary
active_root_mask[M]             # not terminal and at least one legal action
to_play[M]                      # fixed-opponent lane: -1
env_row[M]
player[M]
policy_env_id[M]
target_reward[M,1]
done_root[M]
terminal/final/autoreset sidecars
```

Required metadata:

```text
schema_id
observation_schema
reward_schema
search_lane
fixed_opponent_or_two_seat_mode
renderer_surface
rnd_mode
death/autoreset settings
seed/checkpoint/model identity
```

Hard failures:

- non-binary legal masks;
- player perspective mismatch;
- row/player order mismatch;
- `to_play` not supported by the selected lane;
- done roots marked active;
- active roots with no legal action;
- missing final observation for a terminal/autoreset row.

### CompactSearchResultV1

Output from the search service over active roots.

Required arrays:

```text
root_index[N]
env_row[N]
player[N]
policy_env_id[N]
selected_action[N]
visit_policy[N,3]
raw_visit_counts[N,3]           # optional at first, required for deeper audit
root_value[N]
predicted_value[N]              # optional if service can provide it
predicted_policy_logits[N,3]    # optional if service can provide it
```

Required metadata:

```text
search_impl
num_simulations
temperature
epsilon
root_noise_weight
model_version
fallback_count
illegal_action_count
logical_model_eval_count
actual_model_eval_count
synthetic_eval_count
```

Hard failures:

- selected action is illegal;
- visit policy has illegal mass;
- visit policy row does not sum to one over legal actions;
- root ids do not map back to `CompactRootBatchV1`;
- output order cannot be explained by `root_index`.

### CompactReplayChunkV1

Time-major training records.

Required arrays:

```text
observation[T,B,P,4,H,W]
legal_mask[T,B,P,3]
selected_action[T,B,P]          # for searched live seats
visit_policy[T,B,P,3]
root_value[T,B,P]
reward[T,B,P]
done[T,B]
terminated[T,B]
truncated[T,B]
final_observation[T,B,P,4,H,W]
final_observation_mask[T,B]
to_play[T,B,P]
rnd_latest_frame[T,B,P,1,H,W]   # if RND is enabled
rnd_bonus[T,B,P]                # if RND is enabled
```

Required invariant:

```text
search at record k consumes observation[k]
selected_action[k] must equal the action stored for the transition into k+1
reward/done/next_observation come from record k+1
terminal next observation uses final_observation, not autoreset observation
```

## First Prototype

Build a profile-only closed compact loop:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> current direct_ctree_gpu_latent service-wrapper control
-> CompactSearchResultV1
-> CompactReplayChunkV1
-> compact target-row adapter
```

The first search service can use the current direct CTree hook. The point is
not to replace search yet; it is to prove the array contracts and measure the
whole compact path without public per-env collect output fanout.

## Trust Gates

### P0 Local Parity

- single legal action exact;
- masked clear-preference exact;
- illegal visit mass zero;
- mixed live and terminal rows;
- final observation before autoreset;
- non-prefix active roots;
- non-identity env ids;
- player-perspective swap sentinel;
- RND latest-frame sentinel;
- compact target rows match existing target-row builder.

### P0 Profile

Run the profile-only closed compact loop against current direct profile rows.

Kill condition:

```text
If the closed compact loop cannot plausibly beat current direct by a 3x-class
margin in a fair denominator, do not keep polishing this architecture as the
10x answer.
```

### P1 Search Replacement

Only after the contract works:

1. fixed-`A=3` array-native CTree payload API;
2. batched search service that collects many roots/leaves before recurrent
   inference;
3. MCTX/JAX visual-root toy as a scratch accelerator-native reference.

## What This Is Not

- Not Coach launch advice.
- Not a trainer default.
- Not a MuZero algorithm change.
- Not an excuse to bypass parity gates.
- Not a reason to touch live overnight runs.

## Current Decision

Use `direct_ctree_gpu_latent` as the control service and move the contract
forward. The precomputed recurrent falsifier already showed that deleting
recurrent calls alone is not enough. The next useful implementation is compact
search/replay ownership with explicit contracts and parity tests.

## Current Wiring Status

As of 2026-05-22, the profile-only boundary can wire:

```text
HybridCompactBatch
-> direct CTree compact arrays search
-> CompactRootBatchV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1 in the measured collection hot path
-> CompactReplayChunkV1 only as the heavy materialized validation edge
```

This lives in
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.
It validates the direct search output against the compact root sidecars and
emits `compact_service_*` telemetry. It is not a trainer default and not Coach
launch advice.

The old materialized replay edge uses:

```text
_LightZeroCollectForwardStackProbe.run_compact_batch_with_replay_chunk(...)
```

Plainly: `run_compact_batch()` proves root/result identity and legality. The
profile loop now uses `CompactReplayIndexRowsV1` for the measured compact replay
proof because it does not copy `observation` or `next_observation` during
collection. `run_compact_batch_with_replay_chunk()` remains a heavy validation
helper for callers that deliberately want full materialized target rows.

Same-denominator H100 rows were run. Materialized target rows were falsified as
too slow (`~52-54s` proof time over `61440` roots). Index rows fixed that proof
bucket (`~0.18-0.19s` over the same roots). This justifies the compact replay
writer shape, but it does not prove a 3x search-service speedup.

Important measurement constraint:

```text
The current hybrid profile loop steps CurvyTron with random actions, then runs
the search probe on the post-step observation. That is fine for search timing,
but it is not a valid replay-target timing loop by itself. A real compact
replay profile needs either:

1. search at record k, step the environment with the selected actions, then
   build targets at record k+1; or
2. a clearly labeled synthetic replay-materialization overhead row.
```

Do not claim a closed compact replay speedup from a row that merely validates
target rows against unrelated random next actions.

Update: the profile-only flag `hybrid_compact_service_replay_proof` implements
case 1. It saves the direct compact search output, uses those selected actions
for the next environment step, and writes `CompactReplayIndexRowsV1`. The flag
requires `warmup_steps >= 1` and rejects `resident_torch_reuse` because that
stale-input ceiling is not a valid replay denominator.

2026-05-22 safety refresh:

```text
CompactReplayIndexRowsV1 is now guarded against stale identity and terminal
edge mistakes:

- search_result.policy_env_id must match root_batch.policy_env_id at active roots;
- root_batch legal masks, env rows, players, and policy ids must match the
  compact batch it claims to describe;
- terminal next rows require next_final_observation_row_mask;
- final_reward is checked against final_reward_map for final rows and reward
  for non-final rows;
- non-prefix active roots keep both compact_root_row and compacted policy_row.
```

There is also a validation/sampler edge:

```text
materialize_compact_target_rows_from_index_rows_v1(...)
```

It rebuilds learner-shaped target rows from index rows plus the replay chunk
and matches `build_compact_target_rows_from_search_arrays_v0(...)` in focused
tests. This keeps the collection hot path index-only while proving the index
rows still contain enough information for later learner materialization.
