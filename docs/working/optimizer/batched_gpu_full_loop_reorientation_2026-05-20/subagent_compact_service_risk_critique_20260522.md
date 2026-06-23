# Compact Service Risk Critique, 2026-05-22

Scope: critique only. I inspected the compact service contract, compact target
row bridge, source-state replay/target builders, RND path, and current tests. I
did not edit runtime code and did not touch live runs.

## Plain Read

`CompactRootBatchV1 -> CompactSearchResultV1 -> CompactReplayChunkV1` is the
right direction if the goal is to escape the scalar LightZero object boundary.
It is also dangerous because it moves ownership of training facts out of stock
LightZero and into our own arrays.

The main silent failure is not shape. The dangerous failures are identity:

```text
which physical row,
which player,
which policy perspective,
which record index,
which final observation,
which reward target,
which RND latest frame,
which search output,
which replay transition.
```

Most of these can be wrong while every tensor still has the expected shape.

## Current Code Read

Good existing checks:

- `compact_policy_row_bridge.py` validates binary action masks, `done_root`,
  `active_root_mask`, `to_play=-1`, selected-action legality, zero illegal
  visit mass, target reward shape, and compact batch versus replay record
  observation/reward/done parity.
- `build_compact_target_rows_from_search_arrays_v0(...)` now preserves both
  identities: `compact_root_row` and replay `policy_row`.
- `tests/test_compact_search_replay_contract.py` covers two-record final
  observation, three-record `record_index=1`, non-prefix active roots, RND
  latest-frame order, and player-perspective swap rejection.
- `tests/test_multiplayer_source_state_target_rows.py` also covers non-identity
  `policy_env_id`, terminal final observations, and swapped observation
  rejection.
- `exploration_bonus.py` has explicit RND config hashing, latest-frame
  extraction shape checks, uint8 normalization, train/update counters, predictor
  hash checks, target hash checks, and reward-neutral meter checks.

Risky current limits:

- The compact target bridge is still a target-row proof, not a real compact
  replay writer/sampler.
- Current compact tests mostly compare against the current object bridge. That
  is good parity, but it can bless a shared mistake if both paths read the same
  wrong metadata.
- `policy_env_id` is allowed to be non-identity in some tests, but the service
  contract still needs an explicit rule: service response matching must use
  `request_id/root_index`, not arrival order.
- `CompactSearchResultV1` is not yet a typed, fail-closed object. If it is just
  arrays passed by convention, row swaps can look like valid speed wins.
- RND is currently coupled to LightZero reward-model entrypoint semantics. A
  compact path can accidentally skip `collect_data`, train too rarely, train on
  stale frames, or estimate on a different row order while still producing
  numeric rewards.

## What Can Silently Go Wrong

### 1. Root Identity Drift

The search service may receive active roots in compacted order but return
results in original `[B*P]`, queue arrival order, or sorted env id order.

Silent symptom:

- selected actions are legal;
- visits sum to one;
- throughput looks good;
- actions are applied to the wrong player or wrong env row.

Fail-closed invariant:

```text
Every search result row must carry request_id/root_index/env_row/player/policy_env_id.
The replay writer must reject any result whose identity tuple does not match
the root batch identity table.
```

### 2. Compact Root Row Versus Replay Policy Row Confusion

This already bit the bridge once: compact root row indexes `[B*P]`; replay
`policy_row` indexes only live policy rows. They are equal only for prefix-live
cases.

Fail-closed invariant:

```text
compact_root_row != policy_row is normal.
Both ids must be preserved and tested.
Any output that uses one id where the other is required must reject.
```

### 3. Terminal Roots Searched Or Terminal Results Dropped

Terminal rows can still have observations and masks in arrays. A compact search
service might search them, or a replay writer might drop their final observation
because no search result exists.

Fail-closed invariant:

```text
done_root -> active_root_mask false.
terminal/final rows are not searched as live roots.
terminal transition rows must still carry final_observation and final_reward
from record k+1.
```

### 4. Autoreset Observation Leaks Into Terminal Target

This is the most expensive kind of quiet bug: the model trains on a reset
observation as the terminal next observation.

Fail-closed invariant:

```text
If final_observation_row_mask[k+1, env_row] is true, next_observation must be
final_observation[k+1, env_row, player], not observation[k+1, env_row, player].
```

The tests already cover this locally. The new service needs the same check at
the `CompactReplayChunkV1` writer and sampler boundary.

### 5. Player Perspective Swap

The policy tensor is a controlled-player view. Player 0 and player 1 tensors
are not interchangeable even when they share global board coordinates.

Silent symptom:

- shape `[4,64,64]` is right;
- reward/action mask is right;
- model learns the wrong self/other visual convention.

Fail-closed invariant:

```text
observation[k, env_row, player] must match policy_player=player and the
controlled-player perspective schema.
Search/replay metadata must carry perspective_schema_id and controlled_player.
```

### 6. Joint Action / Selected Action Off By One Record

The contract says search at record `k` chooses the action that appears in the
transition into `k+1`. A compact writer can easily store the current
`joint_action[k]`, especially because reset rows often have placeholder
actions.

Fail-closed invariant:

```text
selected_action[k, env_row, player] == joint_action[k+1, env_row, player]
```

Reject if `record_index + 1` is missing.

### 7. Reward Target Misalignment

The root batch currently requires `target_reward == reward.reshape(B*P,1)` for
the current record. Target rows use `reward[k+1]` for the transition. A compact
service can mix these without shape errors.

Fail-closed invariant:

```text
root target_reward is the model/search root reward for observation[k].
target-row reward is the transition result reward from k+1.
final_reward uses final_reward_map only when final_observation_row_mask[k+1].
```

The metadata should name these separately.

### 8. RND Latest Frame Row Order Drift

RND uses the latest frame of the policy stack. If compact roots are filtered or
reordered before RND extraction, the intrinsic reward can attach to the wrong
target row.

Fail-closed invariant:

```text
rnd_latest_frame[root_index] must use the same root_index/env_row/player table
as search and replay.
rnd_bonus must carry the same identity table before being added to rewards.
```

Do not accept a bare `[N]` RND bonus vector without identity sidecars.

### 9. RND Cadence Becomes Meaningless

The current RND model trains in `train_with_data()` from collected frames and
estimates inside the reward-model path. A compact service can accidentally
estimate many times while training only a few times, or skip `collect_data`
entirely.

Fail-closed invariant:

```text
RND mode enabled -> collect_data_calls, train_with_data_calls, estimate_calls,
train_cnt_rnd, estimate_cnt_rnd, predictor hash delta, target hash stability,
and reward neutrality/augmentation status must be present.
```

Rows missing those fields are throughput smokes, not RND correctness rows.

### 10. Positive RND Reward Alters The Wrong Reward

`rnd_meter_v0` must not alter target rewards. `rnd_replay_target_v0` may alter
them, but only with explicit weight/schema. A compact replay writer can add RND
to `final_reward`, current root reward, or the wrong player reward.

Fail-closed invariant:

```text
rnd_meter_v0 -> target rewards byte/equality unchanged.
rnd_replay_target_v0 -> only transition target reward gets the weighted bonus,
with identity sidecar and bounded delta.
```

### 11. Search Result Legal Mask Uses The Wrong Mask

There are `legal_action_mask`, `lightzero_action_mask`, and possibly
service-local active masks. A result can be legal under one and illegal under
another.

Fail-closed invariant:

```text
selected_action must be legal under the replay policy_action_mask and the
root-batch legal_mask for the same env_row/player.
visit_policy illegal mass must be zero under that exact mask.
```

### 12. `to_play` Pretends Fixed-Opponent Semantics In A Two-Seat Lane

The current compact bridge rightly rejects anything except `to_play=-1`. A
future two-seat/current-policy lane cannot inherit that silently.

Fail-closed invariant:

```text
fixed-opponent lane requires to_play=-1.
two-seat/current-policy lane must have a separate explicit contract and tests.
```

Do not let `CompactRootBatchV1` become mode-polymorphic without a mode field
that changes validation.

### 13. Replay Chunk Time Axis Gets Compacted Per Row

If some env rows terminate and autoreset while others continue, a compact
writer may compact time independently per env row. That breaks MuZero unroll
semantics.

Fail-closed invariant:

```text
CompactReplayChunkV1 is time-major over shared record indices.
record_index and next_record_index are explicit arrays, not inferred from
physical storage offset.
```

### 14. Metadata Claims Drift

A profile-only compact service can accidentally emit metadata that sounds like
stock LightZero training integration.

Fail-closed invariant:

```text
Profile-only rows must say profile_only=true, lightzero_training_claim=false,
native_game_segment_claim=false, and no policy-improvement claim.
Training rows must prove the actual learner boundary they feed.
```

## Invariants That Must Be Hard Errors

`CompactRootBatchV1` should reject:

- missing or unknown schema id;
- observation not `[B,P,4,H,W]` with expected dtype/schema;
- missing controlled-player perspective metadata;
- non-binary legal masks;
- `policy_env_id` duplicates;
- `policy_env_row/policy_player` not matching the declared identity table;
- `done_root != repeat(done, player_count)`;
- `active_root_mask != (~done_root & legal_mask.any(-1))`;
- active root with no legal action;
- done root marked active;
- unsupported `to_play` for the lane;
- `target_reward` shape/order mismatch;
- terminal/final mask present without final observation;
- final observation shape/dtype/schema mismatch;
- reward schema/RND mode missing when RND is enabled.

`CompactSearchResultV1` should reject:

- result count not equal active root count unless explicitly partial with
  fallback accounting;
- duplicate or missing `root_index`;
- result identity tuple not found in root batch;
- selected action out of range or illegal;
- visit policy not finite, negative, unnormalized, or illegal-mass positive;
- raw visit counts shape/order mismatch;
- fallback count positive in a row claiming no fallback;
- model/search metadata missing: sim count, root noise, temperature, model id,
  actual/logical/synthetic eval counts.

`CompactReplayChunkV1` should reject:

- fewer than two records for target rows;
- record `k` with no `k+1` result record;
- selected action not equal `joint_action[k+1]`;
- next observation using autoreset when final observation is required;
- terminal row with missing final reward map;
- row/player ids outside chunk dimensions;
- duplicate policy rows for the same record/seat;
- policy observation not equal `observation[record, env_row, player]`;
- policy action mask not equal legal mask for that seat;
- RND bonus vector without matching identity sidecars;
- positive RND reward mutation when mode says meter-only.

## Tests To Add Next

### P0. Typed Result Identity Reorder Test

Build a root batch with active roots `[3, 1, 6]`, non-identity
`policy_env_id`, and service results returned in order `[1, 6, 3]`.

Pass condition:

- replay writer sorts/maps by `root_index` or request id;
- final target rows match the expected env/player/action mapping;
- arrival-order-only matching fails.

### P0. Missing And Duplicate Result Rejection

Same fixture, but omit one active root and duplicate another.

Pass condition:

- `CompactSearchResultV1` rejects before replay writing.

### P0. Terminal Plus Live Mixed Chunk Writer Test

Use `T=3`, `B=3`, `P=2`:

- row 0 live through all records;
- row 1 dies at record `k+1`;
- row 2 autoresets after final observation.

Pass condition:

- only live roots at record `k` are searched;
- terminal `next_observation` uses `final_observation[k+1]`;
- live row uses normal `observation[k+1]`;
- terminal row does not reappear as a searched reset row in the same transition.

### P0. Perspective Sentinel With Same Shape

Construct player 0 and player 1 observations with distinct latest-frame
sentinels and distinct self/other encodings. Swap only the observations, not
`policy_player`.

Pass condition:

- compact root validation or replay writer rejects;
- a shape-only test would pass, proving the sentinel matters.

### P0. RND Identity Sentinel

Use latest channel values encoding:

```text
latest = 0.10 * record + 0.01 * env_row + 0.001 * player
```

Filter to non-prefix active roots and reorder search results.

Pass condition:

- RND latest frames, search results, target rewards, and replay rows all carry
  the same identity order;
- any RND vector applied by positional order alone fails.

### P0. RND Meter Neutrality Through Compact Replay

Run a compact fixture with `rnd_meter_v0`.

Pass condition:

- predictor hash changes after train;
- target hash stays fixed;
- `last_target_reward_changed=false`;
- compact target reward arrays are exactly unchanged;
- `train_cnt_rnd`, `estimate_cnt_rnd`, and skip counters are surfaced.

### P0. Positive RND Reward Attachment Test

Run the same fixture with `rnd_replay_target_v0` and a tiny fixed synthetic RND
bonus.

Pass condition:

- only the intended transition reward rows change;
- deltas are bounded by weight;
- final_reward_map is not accidentally mutated unless explicitly part of the
  chosen contract;
- metadata says target reward was augmented.

### P0. LightZero Consumer Boundary Canary

For any profile claiming it feeds a learner-shaped boundary, sample from the
compact replay chunk and compare to the current target-row builder on:

- observation;
- action;
- action mask;
- policy target;
- root value;
- reward;
- final reward;
- done/terminated/truncated;
- next observation;
- `to_play`;
- env row;
- player;
- record index;
- policy row.

This should be an exact deterministic fixture test, not a Modal run.

### P1. Stochastic Search Statistical Gate

Once deterministic identity gates pass, compare stochastic/root-noise behavior
over many seeds:

- zero illegal actions;
- zero illegal visit mass;
- empirical action frequencies close to stock;
- finite values/logits;
- no hidden fallback.

Do not block on exact per-seed parity for tied neutral rows.

## Recommendation

Proceed with the compact service, but keep it profile-only until the P0 identity
and replay tests exist. The next implementation should not start by optimizing
the tree. It should start by making these three objects real checked contracts:

```text
CompactRootBatchV1
CompactSearchResultV1
CompactReplayChunkV1
```

The first speed prototype can still use current `direct_ctree_gpu_latent` as
the search service. The trust gate is whether the compact service can own
search, RND, replay, and target rows without losing row/player/final-observation
facts. If that cannot be made boringly exact on deterministic fixtures, any
5-10x speed number is not useful.
