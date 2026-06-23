# Puffer-Style Contiguous Buffer Attach Audit

Date: 2026-05-22

Scope: local-only static code inspection for the CurvyTron/LightZero optimizer
lane. No production behavior changes, no trainer rewrites, no live Coach runs.

## Short Read

The lowest-disruption attach point is inside
`HybridBatchedObservationProfileManager.step`, after actor payload merge,
observation stack update, and terminal `final_observation` capture, but before
`materialize_lightzero_scalar_timestep`.

That point already exists as `HybridBatchedStackProbe.run(observation,
action_mask)`. It is the right place to test a Puffer-style contiguous buffer
consumer because it sees row-major `[B, P, 4, 64, 64]` stacks and `[B, P, 3]`
legal masks before LightZero-shaped scalar objects are built. The current probe
is too narrow for a real compact loop because it does not receive row id,
player id, reward, done, final observation, autoreset, or RND sidecars.

Relevant anchors:

- Hybrid pre-scalar probe protocol:
  [source_state_hybrid_observation_profile.py](/Users/shankha/curvy/src/curvyzero/training/source_state_hybrid_observation_profile.py:198)
- Probe call before scalar materialization:
  [source_state_hybrid_observation_profile.py](/Users/shankha/curvy/src/curvyzero/training/source_state_hybrid_observation_profile.py:459)
- Scalar materialization call:
  [source_state_hybrid_observation_profile.py](/Users/shankha/curvy/src/curvyzero/training/source_state_hybrid_observation_profile.py:519)
- Existing mock native boundary:
  [profile_hybrid_batched_observation_manager.py](/Users/shankha/curvy/scripts/profile_hybrid_batched_observation_manager.py:90)

Follow-up external context from the local PufferLib clone matches this shape:
`StaticVec` owns flat observation/action/reward/terminal/action-mask buffers,
env structs write directly into their slices, GPU mode uses pinned host plus
device buffers, and work is chunked by buffer/stream. The local CurvyTron
translation should therefore be "manager owns contiguous row/player buffers and
the compact consumer reads them directly", not "create faster scalar
LightZero timesteps."

Recent local profile-only rows sharpen the boundary read:

| row | observation | shape | result | read |
| --- | --- | --- | --- | --- |
| zero obs + native probe | zero stack | `B256/A8/steps40/warmup10/uint8/no scalar` | `32764` timesteps/sec, measured `0.625s`; native probe `0.095s`, actor wall `0.472s`, observation `0.045s` | compact consumer is cheap; actor scheduling is already visible |
| CPU oracle + native probe | CPU renderer | `B128/A8/steps40/warmup10/uint8/no scalar` | `192.91` timesteps/sec, measured `53.08s`; renderer/render stack about `52.23s`, native probe `0.065s`, actor wall `0.640s` | dominated by old CPU renderer, not useful for judging the Puffer-style boundary |
| CPU oracle + scalar only | CPU renderer | same as above, scalar materialization only | `192.17` timesteps/sec, measured `53.29s`; renderer about `52.35s`, scalar materialization `0.133s` | scalarization is lost under CPU render cost in this row |

Interpretation: use zero-observation or real device-renderer rows to judge the
contiguous buffer boundary. CPU-oracle rows mostly say "the CPU renderer is
slow"; they do not falsify the Puffer-style attach point.

## First Native Actor-Buffer Implementation

Update: the first opt-in profile-only implementation now exists.

Code paths:

- `HybridObservationProfileConfig(native_actor_buffer=True)`
- `scripts/profile_hybrid_batched_observation_manager.py --native-actor-buffer`

Scope:

- zero-observation profile rows only;
- no stock trainer behavior changed;
- no live runs touched;
- renderer-backed rows still use actor payloads because render-state dicts need
  a separate compact-buffer contract.

Matched local result:

```text
B512/A16/steps100/warmup20/uint8/no-pickle/no-scalar/native-vector-probe
old actor payload + merge: 40477 timesteps/sec
native actor buffer:       67890 timesteps/sec
```

This confirms the attach audit's main claim: the actor payload/merge boundary
can be priced and removed locally before attempting a larger trainer rewrite.

## Current Hybrid Dataflow

Plain profile-only flow:

```text
profile_hybrid_batched_observation_manager.py
-> run_hybrid_observation_profile(config, renderer?, batched_stack_probe?)
-> HybridBatchedObservationProfileManager
-> row partitions [B] split across in-process actors
-> each actor owns VectorMultiplayerEnv(local_rows, P)
-> random joint_action[B,P] int16 from profile loop
-> actor.step(action[actor_rows])
-> VectorMultiplayerEnv.step(local_action)
-> HybridActorStepPayload arrays per actor
-> _merge_payloads into parent compact arrays
-> _update_observation builds observation[B,P,4,64,64]
-> capture final_observation before autoreset rows are reset
-> _reset_autoreset_observation updates reset rows in stack
-> optional batched_stack_probe(observation_for_timestep, action_mask)
-> optional materialize_lightzero_scalar_timestep(...)
-> optional policy_search_probe(flat_obs) after scalarization
-> JSON profile summary
```

Actor payload arrays are compact but actor-scoped:

```text
global_rows[A_i] int32
reward[A_i,P] float32
done[A_i] bool
episode_step[A_i] int32
elapsed_ms[A_i] float64
round_id[A_i] int32
alive[A_i,P] bool
action_mask[A_i,P,3] bool
joint_action[A_i,P] int16
terminal_global_rows[T_i] int32
autoreset_global_rows[R_i] int32
render_state/autoreset_render_state: dict[str, ndarray] when renderer-backed
```

Parent compact merge writes row-major arrays:

```text
reward[B,P] float32
done[B] bool
episode_step[B] int32
elapsed_ms[B] float64
round_id[B] int32
alive[B,P] bool
action_mask[B,P,3] bool
joint_action[B,P] int16
terminal_global_rows[T] int32
autoreset_global_rows[R] int32
```

See `_merge_payloads`:
[source_state_hybrid_observation_profile.py](/Users/shankha/curvy/src/curvyzero/training/source_state_hybrid_observation_profile.py:984).

Observation update is central:

```text
_zero_stack[B,P,4,64,64] float32 or uint8
shift stack fifo newest-last
if zero mode: latest frame fill 0
if renderer-backed: merge actor render_state dicts
-> SourceStateBatchedRenderRequest(state, row_indices, controlled_players, out)
-> frames[B*P,1,64,64] uint8
-> latest stack frame updated for every row/player
```

The renderer request shape and player-view order are already explicit in the
batched observation facade contract:
[source_state_batched_observation_profile.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_profile.py:577).

## Earliest Scalar Python Boundary

There are Python actor objects and per-actor payload dataclasses earlier, but
their hot data is still NumPy arrays. The earliest per-root LightZero-shaped
Python object boundary is `materialize_lightzero_scalar_timestep`.

The first scalar Python rows appear when it builds `info = []` and appends one
dict per row/player:
[source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:787).

Each per-root `info` dict carries:

```text
row: Python int
player: Python int
player_view: Python int, currently same as player
profile_only: True
final_observation_present: Python bool
final_observation: optional [4,64,64] float32 copy for terminal roots
```

The batch `MockBaseEnvTimestep` then carries:

```text
obs["observation"]: flat_obs[B*P,4,64,64] float32, contiguous
obs["action_mask"]: flat_action_mask[B*P,3] bool, contiguous
obs["to_play"]: [B*P] int64, all -1
reward: [B*P] float32
done: [B*P] bool, repeated from row-level done
info: list[dict], one dict per root
target_reward: [B*P,1] float32
```

The later env-manager-shaped bridge creates even more scalar fanout:

```text
ready_obs: dict[int, {"observation": [4,64,64] copy, "action_mask": [3] copy, "to_play": -1}]
timestep_by_env_id: dict[int, MockBaseEnvTimestep]
```

See `_ready_obs_by_env_id` and `_split_timestep_by_env_id`:
[source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:1069)
and
[source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:1102).

## Minimal Contiguous Buffer Prototype

Prototype shape: add a profile-only "native vector boundary batch" next to the
existing `batched_stack_probe`. Do not route it through stock LightZero
collector objects. Let the current manager remain the producer, and let a mock
or real compact search/replay/RND service be the consumer.

Constants:

```text
B = physical CurvyTron rows
P = players per row
A = action count = 3
C,H,W = 4,64,64
M = B * P root/player views
T = terminal row count this step
R = autoreset row count this step
```

Input arrays:

| name | shape | dtype | producer | consumer |
| --- | ---: | --- | --- | --- |
| `action_in` | `[B,P]` | `int16` | compact service output from previous step or profile RNG | env actor step |
| `obs_stack` | `[B,P,4,64,64]` | `uint8` preferred, `float32` accepted | `observation_for_timestep` from hybrid manager | model/search, RND, compact replay |
| `legal_mask` | `[B,P,3]` | `bool` | `compact["action_mask"]` | search/action sampler |
| `policy_env_row` | `[M]` | `int32` | row-major repeat `0..B-1` | output/replay mapping |
| `policy_player` | `[M]` | `int16` or `int32` | row-major tile `0..P-1` | perspective mapping |
| `policy_env_id` | `[M]` | `int32` | `row * P + player` | compat/debug edge |
| `to_play` | `[M]` | `int64` | constant `-1` | MuZero fixed-seat semantics |
| `reward` | `[B,P]` | `float32` | `compact["reward"]` | replay/RND target reward |
| `target_reward` | `[M,1]` | `float32` | reshape of reward row/player order | RND estimate / replay targets |
| `done_row` | `[B]` | `bool` | `compact["done"]` | terminal handling |
| `done_root` | `[M]` | `bool` | repeat `done_row` by player | replay/compat edge |
| `episode_step` | `[B]` | `int32` | compact merge | replay/debug |
| `elapsed_ms` | `[B]` | `float64` | compact merge | debug/timing |
| `round_id` | `[B]` | `int32` | compact merge | replay/debug |
| `alive` | `[B,P]` | `bool` | compact merge | mask validation/debug |
| `joint_action` | `[B,P]` | `int16` | compact merge | replay/action history |
| `final_observation` | `[B,P,4,64,64]` | same as `obs_stack` or normalized `float32` | terminal snapshot before reset | replay/compat terminal edge |
| `final_observation_row_mask` | `[B]` | `bool` | `done_row` when final obs present | terminal edge |
| `terminal_global_rows` | `[T]` | `int32` | compact merge | terminal audit |
| `autoreset_global_rows` | `[R]` | `int32` | compact merge | reset audit |
| `autoreset_row_mask` | `[B]` | `bool` | derived from autoreset rows | reset audit |

Output arrays:

| name | shape | dtype | producer | consumer |
| --- | ---: | --- | --- | --- |
| `selected_action` | `[M]` | `int16` | compact search/action service | joint-action builder / replay |
| `visit_policy` | `[M,3]` | `float32` | compact search/action service | replay/learner targets |
| `root_value` | `[M]` | `float32` | compact search/action service | replay/learner targets |
| `illegal_action_count` | scalar | `int64` | validation pass | profile summary |
| `active_root_count` | scalar | `int64` | validation pass | profile summary |

RND view:

```text
rnd_obs_segment[M,4,64,64] float32 normalized [0,1]
rnd_latest[M,1,64,64] float32, derived from channel -1
target_reward[M,1] float32
rnd_feature_source = "policy_gray64_latest/v0"
```

The current RND meter consumes `SimpleNamespace(obs_segment=flat_obs)` and
`target_reward`, then extracts the latest gray64 frame. See:
[source_state_batched_observation_mock_collector.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_mock_collector.py:674)
and
[exploration_bonus.py](/Users/shankha/curvy/src/curvyzero/training/exploration_bonus.py:488).

Minimal producer:

```text
HybridBatchedObservationProfileManager.step
  after _update_observation and final_observation snapshot
  before materialize_lightzero_scalar_timestep
```

This is close to PufferLib's concrete ownership pattern, but currently one
level higher: CurvyTron actors still own their env state and return compact
payload arrays to the parent. A later stronger prototype would move toward
`StaticVec`-style ownership by giving each actor or shared parent slab fixed
slices for `action_in`, `reward`, `done`, `terminal`, `legal_mask`, and
`obs_stack`, then letting env stepping write directly into those slices.

Minimal consumer:

```text
profile-only compact boundary service:
  validate C-contiguity/order/masks
  optionally normalize uint8 -> float32 on device
  run mock or real batched model/search
  return selected_action[M], visit_policy[M,3], root_value[M]
```

The existing `_NativeVectorBoundaryProbe` already proves the small array-only
call shape but only for `obs` and `mask`:
[profile_hybrid_batched_observation_manager.py](/Users/shankha/curvy/scripts/profile_hybrid_batched_observation_manager.py:101).

## Timing Fields

Reuse existing timing slots first:

```text
actor_step_sec
actor_step_wall_sec
actor_idle_wait_sec
parent_send_receive_sec
gather_merge_sec
observation_sec
renderer_render_sec
renderer_device_render_sec
renderer_stack_update_sec
stack_shift_sec
stack_latest_update_sec
batched_stack_probe_sec
batched_stack_probe_host_to_device_sec
batched_stack_probe_normalize_sec
batched_stack_probe_device_sec
batched_stack_probe_readback_sec
scalar_materialization_sec
compact_payload_pickle_sec
```

Add prototype-specific telemetry under the probe result rather than changing
trainer behavior:

```text
native_buffer_total_sec
native_buffer_pack_sec
native_buffer_validate_sec
native_buffer_host_to_device_sec
native_buffer_normalize_sec
native_buffer_model_search_sec
native_buffer_readback_sec
native_buffer_output_decode_sec
native_buffer_replay_pack_sec
native_buffer_rnd_collect_sec
native_buffer_rnd_train_sec
native_buffer_rnd_estimate_sec
native_buffer_input_bytes
native_buffer_transfer_bytes
native_buffer_output_bytes
native_buffer_root_count
native_buffer_active_root_count
native_buffer_terminal_row_count
native_buffer_autoreset_row_count
native_buffer_illegal_action_count
```

## Semantic Fields Not To Lose

Highest risk fields:

- Row identity: preserve `policy_env_row`, `policy_env_id`, `terminal_global_rows`,
  and `autoreset_global_rows`. Current env id order is `row * player_count +
  player`.
- Player identity and perspective: preserve `policy_player`, `player_view ==
  player`, renderer `controlled_players`, and `to_play=-1`. Do not swap player
  views while flattening `[B,P] -> [M]`.
- Legal mask: preserve `[B,P,3]` boolean masks exactly and ensure selected
  actions are legal for every active root. Terminal/empty masks need an explicit
  policy, not accidental argmax behavior.
- Reward/done/final observation/autoreset: preserve `reward[B,P]`, row-level
  `done[B]`, repeated root done, terminal `final_observation` before row reset,
  `final_observation_row_mask`, autoreset row masks, and reset-generation
  behavior. The trainer-surface materializer already fails closed when terminal
  final arrays are missing.
- RND hooks: preserve `flat_obs[M,4,64,64]` normalized to `[0,1]`, latest-frame
  extraction from channel `-1`, `target_reward[M,1]`, meter mode's
  target-reward-unchanged behavior, and positive-RND target mutation semantics
  when that mode is tested.
- Debug/replay sidecars: keep `episode_step`, `round_id`, `elapsed_ms`, `alive`,
  and `joint_action` available even if the first consumer ignores them.

## Recommendation

Do not attach a Puffer-style prototype after `materialize_lightzero_scalar_timestep`;
that is already past the scalar Python boundary. Do not attach it inside each
actor either; the parent manager is where row-major observations, legal masks,
terminal snapshots, and autoreset bookkeeping first coexist.

Attach at the current `batched_stack_probe` location, but promote the probe
input from two arrays to a profile-only contiguous batch sidecar. Keep scalar
LightZero materialization as a debug/compatibility edge, disabled in the hot
prototype with `materialize_scalar_timestep=False` only when the compact
consumer is present.

Next falsifier should avoid CPU-oracle rendering as the denominator. The useful
local row is zero observation or a resident/device renderer with the compact
consumer enabled, because that is where actor scheduling, buffer packing, and
consumer cost are visible instead of being buried under a CPU render loop.

## Review Addendum: Native Actor Buffer and Wide Probe

Current `native_actor_buffer` read:

- The patch is correctly profile-only and opt-in. It is rejected when an
  observation renderer is configured, so renderer-backed terminal/reset state is
  not accidentally lost.
- Terminal and autoreset ordering is currently correct for zero-observation
  rows: the manager snapshots `observation_for_timestep`/`final_observation`
  before resetting autoreset rows, then resets the manager stack from the
  collected autoreset row ids.
- The native path relies on manager-owned row partitions for the "each global
  row exactly once" invariant. The payload merge path validates that invariant;
  the native direct-write path does not. That is acceptable for the current
  sealed profile harness, but a widened sidecar should validate row ownership or
  expose a cheap checksum/filled-row mask in tests before subprocess/shared-slab
  experiments.
- The existing pre-scalar probe is intentionally narrow:
  `run(observation, action_mask)`. It cannot audit reward, done, final frame,
  autoreset, row/player ids, or replay/RND sidecars, so it should not be treated
  as a correctness proxy for the compact loop.

Recommended next contract:

```text
HybridCompactBatchProbe.run_compact_batch(batch) -> HybridBatchedStackProbeResult
```

Use a frozen profile-only batch object or typed namespace. Required fields:

```text
observation[B,P,4,64,64] uint8|float32, C-contiguous
action_mask[B,P,3] bool, C-contiguous
policy_env_id[M] int32, row-major row * P + player
policy_env_row[M] int32, row-major repeated rows
policy_player[M] int32, row-major tiled players
reward[B,P] float32
target_reward[M,1] float32, row-major reward view/copy
done[B] bool
done_root[M] bool, row-major repeat of done
final_observation[B,P,4,64,64] uint8|float32
final_observation_row_mask[B] bool
autoreset_row_mask[B] bool
terminal_global_rows[T] int32
autoreset_global_rows[R] int32
episode_step[B] int32
elapsed_ms[B] float64
round_id[B] int32
alive[B,P] bool
joint_action[B,P] int16
```

Backward compatibility: keep calling legacy probes with `run(observation,
action_mask)`. Prefer the wide path only when the probe has
`run_compact_batch`; report the active contract in telemetry as
`batched_stack_probe_contract = "compact_batch_v1"` or
`"legacy_observation_action_mask_v0"`.

Minimal tests before trusting the widened sidecar:

- Capture-probe parity against scalar materialization for
  `policy_env_id`, `policy_env_row`, `policy_player`, `reward`,
  `target_reward`, `done_root`, and `action_mask`.
- Terminal row with autoreset enabled proves the sidecar receives terminal
  `observation`/`final_observation` before reset plus `autoreset_row_mask`, and
  the manager stack is reset afterward.
- Native actor buffer versus payload merge parity on terminal/autoreset rows,
  not just non-terminal core arrays.
- Renderer-backed wide-probe test using sentinel frames, while still asserting
  `native_actor_buffer + renderer` raises.
- Legacy two-argument probe still runs unchanged, including
  `materialize_scalar_timestep=False` when a consumer is present.

## Implementation Update: Wide Sidecar Exists

Status: implemented in the profile-only harness.

Files:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `scripts/profile_hybrid_batched_observation_manager.py`
- `tests/test_source_state_hybrid_observation_profile.py`

Actual method name:

```text
HybridBatchedStackProbe.run_compact_batch(batch)
```

The harness uses `run_compact_batch` when present and falls back to legacy
`run(observation, action_mask)` otherwise.

Test coverage added:

- Wide sidecar capture with renderer-backed sentinel terminal frames.
- Native actor buffer versus payload merge on terminal/autoreset rows.
- Existing legacy two-array probes remain green.

Validation:

```text
uv run ruff check ... -> passed
uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py
-> 19 passed
```

Fresh profile read:

```text
B512/A16/steps100/warmup20/uint8/no-pickle/no-scalar/native-vector-probe
payload+merge:        ~22.1k timesteps/sec
native actor buffer:  ~30.5k timesteps/sec
```

Plain read:

```text
The sidecar is now semantically credible enough for the next falsifier. It can
see reward/done/final/autoreset/row/player fields, not just observation and
mask. The next question is whether a real search/replay/RND-shaped consumer can
use it without falling back to scalar Python rows.
```

## 2026-05-22 Addendum: Compact Batch Is Now Consumed By Direct CTree

The Puffer-style lesson has now been applied to the profile LightZero boundary:

```text
compact row/player batch -> direct CTree arrays profile hook
```

The important detail is not just speed. The compact batch now carries the
semantic facts that scalar LightZero timesteps used to carry implicitly:

```text
row/player ids
reward and target reward
done and done_root
to_play
active_root_mask
terminal/autoreset/final-observation masks
episode/alive/action metadata
```

This lets the next experiments ask the real question:

```text
Can search/replay/RND consume one compact batch without rebuilding thousands of
per-root Python objects?
```

Current risk:

```text
The compact hook is still profile-only. It is not a full trainer path and not a
Coach launch recommendation until matched full-loop gates pass.
```

2026-05-22 update:

```text
The old two-argument probe was too narrow. The widened HybridCompactBatch
sidecar now feeds direct CTree, RND latest-frame input, and a checked
target-row policy-record adapter. The remaining question is whether this shape
produces matched full-loop speed, not whether the sidecar can carry the basic
semantic fields.
```
