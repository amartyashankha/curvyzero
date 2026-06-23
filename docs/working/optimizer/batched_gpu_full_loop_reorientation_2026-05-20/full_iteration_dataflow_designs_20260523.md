# Full Iteration Dataflow And Architecture Designs, 2026-05-23

Status: active optimizer working memory. No live Coach training runs were
touched. This doc is for planning, profiling, and validation of optimizer
probes.

## Why This Exists

The project needs faster CurvyTron training without breaking the learning
contract. Recent optimizer work has produced real profile-only speed probes,
but the big lesson is that we must keep three currencies separate:

- Coach training speed: real `train_muzero` runs, checkpoint iterations per
  hour, learning curves, tournament checks.
- Stock full-loop profile speed: steps/sec through the stock LightZero
  training/profile path with all selected knobs recorded.
- Optimizer probe speed: roots/sec or steps/sec from compact/profile-only
  loops such as direct CTree, mock service, service-tax, dense Torch, MCTX/JAX,
  or compact Torch service.

Do not compare these as if they are the same thing.

## Current Plain Read

The latest compact-service profiles say the next big wall is not a single
small function. It is the dataflow shape:

```text
CPU/vector env state
-> observation/stack/root batch ownership
-> model/search service
-> selected actions
-> compact replay/RND/learner edges
```

The current direct LightZero CTree path still crosses the CPU/GPU boundary too
often. It sends small data by bytes, but it does so in a high-frequency loop:
root preparation, tree traversal, recurrent model calls, CPU readback, list
conversion, and backprop through LightZero CTree.

The newest `compact_torch_search_service` is useful because it uses the common
`CompactRootBatchV1 -> CompactSearchResultV1` boundary and proves replay
attachment remotely. It is not yet a Coach recommendation. The H100 result was
roughly direct CTree speed to about `1.4x` faster on one row, not a 5x or 10x
breakthrough. Its timing split points at the eager tree/recurrent loop as the
hot section.

Scale reminder for the common profile shapes:

```text
B = physical env rows
P = 2 players
R = B * P policy roots
stack = [4,64,64]

float32 stack = 64 KiB/root
uint8 stack   = 16 KiB/root

At B512/R1024:
  float32 roots ~= 64 MiB
  uint8 roots   ~= 16 MiB
```

Selected actions, masks, root values, and visit policies are tiny compared to
the observation stack. They can still be expensive if they force a GPU wait
inside a high-frequency loop.

## Leading Architecture Read

Popper's 2026-05-23 critique sharpens the preferred target:

```text
CPU batched env
-> resident GPU observation/stack tensor
-> device-resident search service
-> CPU receives only selected actions
-> compact replay/RND payload is flushed later
-> stock LightZero objects exist only at validation/debug/sample edges
```

The important rule is simple:

```text
One selected-action sync per env tick is acceptable while the env is CPU.
Per-simulation reward/value/policy readback for CTree backprop is the bad sync.
```

What should stay device-side:

- observation stack `[B,2,4,64,64]`, preferably uint8 until model input
  preparation;
- root logits/value/hidden state;
- search tree arrays: visit counts, priors, values, rewards, child links,
  legal masks;
- recurrent outputs inside the simulation loop;
- visit policy/root value until chunk flush;
- RND latest-frame ring if RND is enabled.

What can sync to CPU:

- selected joint actions `[B,2]` once per env tick;
- compact replay payload chunks every `K` steps, before sample visibility;
- sampled validation mirrors;
- metrics/checkpoint/eval summaries at coarse cadence.

The existing `CompactSearchServiceV1` should evolve toward a two-phase API:

```python
DeviceSearchStep:
    selected_action_cpu
    replay_payload_handle

flush_replay_payload(handle):
    visit_policy
    raw_visit_counts
    root_value
    predicted logits/value
```

That shape lets the CPU env keep moving without forcing visit policy/root value
readback on the action-critical path. It is still profile-only until replay
chunk parity, RND latest-frame parity, and terminal final-observation parity
pass.

## Non-Search Materialization Read

Ohm's 2026-05-23 audit is the warning for after search gets faster:

```text
Do not only remove CTree overhead and then keep rebuilding scalar LightZero
objects, full observation rows, replay chunks, and RND per-frame tensors in the
hot loop. That just moves the wall.
```

Ranked materialization offenders:

1. scalar LightZero timestep splitting and ready-observation dicts;
2. target rows copying `observation + next_observation`;
3. replay chunk stacking full `observation` and `final_observation`;
4. RND's per-frame Python/Torch tensor buffer and CPU metric/hash reads;
5. root-batch observation copies.

Plain implication:

```text
The next major implementation should store compact ids/sidecars and frame slabs
while collecting. Full Python rows should appear only at validation, debug,
sample, or stock compatibility edges.
```

## External Systems Read

Parfit's 2026-05-23 research pass matches the same shape:

```text
Fast RL systems stop treating every env step as Python objects. They use fixed
arrays, shared buffers, batched inference/search, and delayed synchronization.
```

Patterns to copy:

- PufferLib-style static memory and chunked buffers;
- EnvPool-style batched CPU env execution instead of scalar object stepping;
- Sample Factory-style rollout/inference/learner split with buffer ids rather
  than serialized observations;
- MCTX-style device-resident batched tree state;
- MiniZero/MuZero-CPP-style actor threads plus batched GPU leaf evaluation.

Patterns to avoid:

- claiming roots/sec as training speed when replay/RND/learner materialization
  is outside the denominator;
- using bigger GPUs as the primary fix while Python/object boundaries remain
  hot;
- bridging PyTorch into JAX/MCTX per simulation through host callbacks and
  calling it device-resident.

## Semantic Kill Criteria

Carver's 2026-05-23 critique defines the promotion gate:

```text
observation k
-> search root k
-> selected action k
-> env transition k+1
-> replay row k
-> learner-visible sample
```

Every aggressive service/device path must prove that chain with the same
`env_row`, `player`, `policy_env_id`, player perspective, legal mask, terminal
final-observation rule, reward rule, and RND latest frame.

Promotion should stop immediately if:

- selected actions are not proven to be the next env joint actions;
- results attach by batch position instead of stable ids;
- player perspective is chosen in optimizer code instead of the training/
  tournament contract;
- terminal next observations come from autoreset frames;
- RND meter changes target rewards or reads the wrong latest frame;
- rows become sample-visible before action, reward, done, visit policy, root
  value, final observation, and RND sidecars are complete;
- fallback count is nonzero in a promoted row;
- the summary cannot distinguish Coach training, stock full-loop profile, and
  optimizer probe currencies.

## Current Anchor Numbers

Same B512/A16/sim16 H100 profile family:

| Mode | Steps/sec | Probe sec | Model sec | Search/update sec | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `direct_ctree_gpu_latent` | `4,965.96` | `5.702` | `1.283` | `3.569` | real LightZero CTree boundary |
| `service_tax_probe` | `5,853.08` | `2.857` | `1.448` | `0.213` | ceiling/falsifier, not MCTS |
| `compact_torch_search_service` | `5,575.16` | `5.098` | `0.271` initial | `4.250` tree/recurrent | profile-only |

No-noise pair:

| Mode | Steps/sec | Probe sec | Model sec | Search/update sec |
| --- | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` | `3,955.36` | `6.944` | `1.528` | `4.404` |
| `compact_torch_search_service` | `5,704.22` | `5.077` | `0.271` initial | `4.235` |

Plain read: the common compact boundary works. The eager Torch body is not the
big win. The next search lane needs a genuinely compiled/fused fixed-shape
body, an array-native CTree, or an MCTX/JAX-style comparator.

## One Iteration Dataflow

This is the shape to keep in mind when deciding where to sync and what to
materialize.

| Stage | Data | Large or small | Current likely owner | Sync risk | Optimization read |
| --- | --- | --- | --- | --- | --- |
| Env mechanics | positions, trail cursors, bonuses, dones | medium, grows with trail history | CPU/vector env | selected actions must return to CPU if env stays CPU | Mechanics itself has not looked like the main wall, but state ownership can be expensive. |
| Render/observation | latest gray frame, stack `[B,P,4,H,W]` | large | renderer plus stack owner | GPU render returning to host; root observation copy | Raw draw is small; repeated stack/root handoff is the cost. |
| Root batch | obs, legal mask, active roots, ids, rewards/dones | obs large, sidecars small | `CompactRootBatchV1` | H2D if obs is host; D2H if resident must be summarized | Keep sidecars compact. Avoid full obs copies except at validation edges. |
| Initial model | root value/logits/hidden | hidden large, logits small | Torch/JAX GPU | CPU readback for LightZero roots | Stock CTree forces CPU root prep; compact services should avoid this. |
| MCTS/search | tree visits, priors, values, hidden states | tree medium, hidden large | LightZero CTree CPU today; probes vary | per-sim recurrent output readback and list APIs | Main search target: remove per-simulation CPU/list/control boundary. |
| Action feedback | selected actions `[B,P]` | tiny | CPU env if env stays CPU | one small sync is acceptable | This is not the byte problem; cadence and tree semantics matter more. |
| Replay rows | obs/action/reward/done/policy/value ids | full rows large, index rows small | stock GameSegments or compact index rows | materializing full rows in hot path | Keep compact index rows hot, materialize only for validation/learner gates. |
| RND | latest frames, predictor/target metrics | latest frame medium, metrics small | currently mixed CPU/Torch | CPU hashes/items/stats | Compatibility gate now; later resident latest-frame ring if RND becomes hot. |
| Learner | sampled tensors and targets | large | stock replay/learner | sample-to-device copy | Do not rewrite this until search/env ownership is clearly no longer the wall. |

Stage notes from the current code map:

- Pure CurvyTron mechanics have not looked like the main cost by themselves.
  The bucket labeled `env` often includes render state, observation stack,
  root/input packaging, final observations, and public metadata.
- Observation render/draw on the GPU can be tiny, but stack ownership and
  readback/copy choices can still dominate.
- The scalar LightZero timestep boundary is still expensive in the trusted full
  train path because it re-creates Python objects and full observation rows.
- Compact replay index rows are the right hot-path shape because they avoid
  copying `observation` and `next_observation`.
- Stock replay, target rows, learner samples, and RND still pull data back
  toward full observation materialization at the sample/training edge.

## Data Movement Rules

- Tiny by bytes can still be expensive if it happens once per simulation with a
  CPU/GPU sync.
- A selected action sync is probably fine while the environment is CPU.
- Recurrent reward/value/policy readback every MCTS simulation is not fine if
  the goal is a multi-x speedup.
- Full observation stacks should not bounce host/device unless a validation or
  stock compatibility edge requires it.
- Full replay rows should not be built during collection when compact index
  rows are enough.
- Profile timers with CUDA sync are attribution tools. Throughput claims need
  unsynchronized companion rows.

## Sync Budget

This is the current rule of thumb for where synchronization belongs.

| Boundary | Should sync? | Reason |
| --- | --- | --- |
| Search result action -> CPU env step | Yes, once per env tick while env mechanics are CPU. | The next CPU step needs the action. The payload is tiny, but it still waits for earlier GPU work. |
| Replay payload, visit policy, root value | Not on the action-critical path if avoidable. | Replay needs these before sample visibility, not before the next env action. Delaying or chunking is legal only if replay proof stays complete. |
| Full observation stack H2D/D2H | Avoid in hot loop. | `[B,P,4,H,W]` is the large payload. Host copies should be validation/debug edges or an explicit host-mode denominator. |
| Legal masks and action masks | Fine by bytes, but record it. | Small payloads can still reveal accidental host/device bouncing. |
| Resident GPU stack explicit fence | Diagnostic only. | A fence may make timing buckets cleaner while reducing overlap. Compare total roots/sec sync-on and sync-off. |
| Torch service phase fences | Diagnostic only except final result readback. | Current service uses phase-honest timing. A future production-like row should use CUDA events or unsynchronized companion throughput. |
| Python row/list materialization | Avoid in collection hot path. | This is allocation/control overhead, not a byte-size problem. It belongs at validation/sample/export edges. |
| RND hashes/metrics | Keep out of high-frequency collect if possible. | RND compatibility matters, but CPU hashes/items/stat reads should not become the hidden wall. |

Telemetry we should prefer adding before the next wave:

- `obs_h2d_bytes` / `lightzero_array_ceiling_obs_h2d_bytes`
- `mask_h2d_bytes` / `lightzero_array_ceiling_mask_h2d_bytes`
- `action_d2h_bytes` / `lightzero_array_ceiling_action_d2h_bytes`
- `replay_payload_d2h_bytes` /
  `lightzero_array_ceiling_replay_payload_d2h_bytes`
- `root_observation_copy_bytes` /
  `lightzero_array_ceiling_root_observation_copy_bytes`
- `python_rows_materialized` /
  `lightzero_array_ceiling_python_rows_materialized`
- `rnd_materialized_rows` /
  `lightzero_array_ceiling_rnd_materialized_rows`
- `resident_obs_reused` / `lightzero_array_ceiling_resident_obs_reused`
- `action_d2h_wait_sec`
- `replay_payload_readback_sec`

2026-05-23 status: the compact Torch search-service profile path now emits the
`lightzero_array_ceiling_*` byte/materialization ledger fields above. The
hybrid profile runner also aggregates these into
`batched_stack_probe_ledger_totals`, and the durable manifest summary carries
the compact fields forward. The time split still needs action-critical wait
versus replay-payload readback split if we continue down that lane.

The direct CTree profile path now emits matching
`lightzero_mcts_arrays_boundary_*` ledger fields for the same summary columns.
This lets the next direct-vs-compact table compare observation transfer,
mask/action readback, Python row materialization, and resident-observation
reuse without guessing from prose.

Fail closed if a row claims resident or compact mode while still copying full
host observations in the hot loop, or if replay-valid mode reads only actions
and omits visit policies/root values.

## Ten Designs To Keep Alive

| # | Design | Likely upside | Main risk | Next falsifier |
| ---: | --- | ---: | --- | --- |
| 1 | Compiled/fused fixed-shape Torch search behind `CompactSearchServiceV1` | `1.5-2.5x`, possibly more search-only | PUCT drift, masks, root noise, tie behavior | no-noise parity fixture plus H100 B512/sim16/sim32 same-denominator row |
| 2 | MCTX/JAX sidecar search | `2-5x` profile-only | semantics differ from LightZero | deterministic tiny model and fixed-mask comparator |
| 3 | Array-native fixed-`A=3` CTree, CPU first then CUDA | CPU `1.2-1.6x`, CUDA `2-5x` | CTree parity and RNG/tie behavior | no-model CTree list-vs-flat microbench, then compact boundary A/B |
| 4 | Multi-producer search service | `2-5x` if GPU underfill dominates | ordering/versioning/replay attachment | N-producer queue with ordered compact result checks |
| 5 | Actor-owned compact render state or env-emitted deltas | `1.5-3x` if handoff is hot | trail/reset/bonus parity | same-state render parity and refresh-on/off H100 A/B |
| 6 | Compact replay/RND ownership | `1.1-1.5x` alone; prerequisite for larger wins | off-by-one rows, final obs, RND wrong frame | multi-record replay/RND terminal canary |
| 7 | In-process actor slab instead of scalar/subprocess object flow | `1.2-2x` if IPC/object fanout is hot | GIL/crash/determinism | same batch subprocess vs in-process slab profile |
| 8 | Larger fixed root batches / sim cohorts | `1.2-2x` | quality or semantics if adaptive | fixed bucket only first; adaptive is research-only |
| 9 | Larger or semantic observations | quality upside; speed uncertain | checkpoint contract and model cost | profile `[4,64,64]` vs larger surfaces with same search settings |
| 10 | PufferLib/EnvPool-style native SoA buffer owner | `2-5x`, platform for `10x` | rewrite scope and parity | `step_many(action[B,P])` fixture plus B512/B1024/B2048 scaling |

Current ranking:

1. Keep compact service boundary and validation gates.
2. Build a real device-resident fixed-shape search attempt behind that
   boundary, with selected-action-only critical sync and delayed replay payload
   flush.
3. In parallel, test actor-owned compact render/input ownership if env/input
   handoff remains hot in the same denominator.
4. Keep MCTX/JAX and native SoA as bigger architecture probes, not quick Coach
   recommendations.

## Validation Gates

Before any optimizer backend becomes trainer-facing, it must prove:

- selected search actions are exactly the next env actions;
- `env_row`, `player`, `policy_env_id`, `to_play`, and active-root order stay
  attached through root batch, search result, replay rows, and target rows;
- illegal actions and illegal visit mass fail closed;
- terminal rows use final observations, not autoreset observations;
- RND meter mode does not change target rewards, while RND reward mode changes
  them intentionally and reports cadence metrics;
- no-noise deterministic fixtures pass before noisy statistical comparisons;
- fallbacks are zero in promotion rows;
- summaries record backend, input mode, root noise, seed, sims, collectors,
  batch size, death mode, RND mode, env manager, and replay proof status.

Preflight test bundle for the next optimizer backend:

```sh
uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_compact_torch_search_service.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_profile_cpu.py \
  tests/test_exploration_bonus.py \
  tests/test_lightzero_phase_profiler.py
```

The first remote profile grid should be small and same-denominator:

| Axis | First values |
| --- | --- |
| Backend | `stock`, `direct_ctree_gpu_latent`, `compact_torch_search_service` |
| Root noise | `0.0` exact gate first, then seeded normal noise |
| Death | normal and no-death profile |
| RND | `none` and `rnd_meter_v0` zero-weight compatibility |
| Sims | `8`, `16`, then `32` |
| Collectors | `256`, `512` first |
| Compute | L4 sanity, H100 final profile |

Kill criteria:

- speed claim mixes Coach speed, stock full-loop speed, and compact probe speed;
- candidate wins only with noise on and lacks no-noise parity;
- RND meter changes target rewards;
- terminal rows read reset observations;
- player perspective is inferred from row order;
- fallback calls are nonzero in a promotion row;
- summary cannot prove the denominator.

## Active Subagents

### Completed 2026-05-23 Architecture Critique Wave

- Popper: device-resident/search-service architecture. Question: what simple
  API and sync policy removes high-frequency CPU/GPU/control crossings?
  Folded above.
- Ohm: non-search materialization. Question: which env/observation/replay/RND/
  learner boundaries will dominate after search improves?
  Folded above; full doc:
  `subagent_non_search_materialization_audit_20260523.md`.
- Parfit: external high-throughput RL patterns. Question: what do PufferLib,
  EnvPool, Sample Factory, Brax/JAX, MCTX, MiniZero/EfficientZero, and
  AlphaZero-style systems do that we can copy?
  Folded above.
- Carver: failure-mode validation. Question: what goes wrong if we make the
  pipeline device-resident or service-based, and what smallest tests catch it?
  Folded above; full doc:
  `subagent_device_resident_semantic_kill_criteria_20260523.md`.

### Active 2026-05-23 Wave 2

- Anscombe: exact insertion points in the current code for a compact slab/search
  service.
- Russell: candidate architectures with speed ranges, sync choices, risks, and
  falsifiers.
- Fermat: validation ladder, existing tests, and missing gates.
- Turing: external systems follow-up focused on buffer layout, sync cadence,
  replay ownership, and service shape.

2026-05-23 result: all four completed and were folded into the world model.
Detailed docs:

- `subagent_insertion_points_20260523.md`
- `subagent_architecture_candidates_wave2_20260523.md`
- `subagent_validation_ladder_wave2_20260523.md`
- `subagent_external_patterns_wave2_20260523.md`

Shared conclusion:

```text
The first real insertion should stay in the profile path at
HybridBatchedObservationProfileManager.step, after compact state/root sidecars
exist and before scalar LightZero timestep materialization. The code should
make a named compact slab/search-service owner around the existing
HybridCompactBatch -> CompactRootBatchV1 -> CompactSearchServiceV1 ->
CompactSearchResultV1 -> CompactReplayIndexRowsV1 contracts.
```

Recommended next architecture lanes, in order:

1. Puffer/EnvPool-style static actor slab around the existing profile manager
   so the hot path owns compact arrays, not scalar timestep objects.
2. Two-phase compact search service: selected actions return immediately;
   visit policy/root value/RND replay payload flushes before sample visibility.
3. Fixed-shape compiled search or array-native fixed-`A=3` CTree behind
   `CompactSearchServiceV1`.
4. MCTX/JAX remains the architecture comparator, not a trainer-facing path
   unless the model/replay side moves with it.

Validation ladder:

```text
L0 labels/promotion lock
L1 root/player/perspective/legal mask
L2 selected action is the next env action
L3 terminal final observation before autoreset
L4 RND latest frame and reward-model behavior
L5 replay row and sample visibility
L6 no-noise exact gate, then noisy statistical gate
```

Missing gates to prioritize:

- out-of-order service result attachment by stable ids;
- incomplete compact replay row hidden from sampler until payload complete;
- mixed terminal/live real compact-service test;
- full-loop RND meter smoke with reward-model metrics;
- result summarizer hard fail for missing promotion fields.

2026-05-23 implementation update:

```text
src/curvyzero/training/compact_search_service.py now has the first two-phase
contract:

CompactSearchResultV1
-> CompactSearchActionStepV1        # selected actions and stable ids
-> CompactSearchReplayPayloadV1     # visit policy/root value/debug payload
```

`validate_compact_search_two_phase_payload_v1(...)` fails closed if the delayed
payload handle, `root_index`, `env_row`, `player`, or `policy_env_id` no longer
matches the action-critical step. This is still profile-only. It does not yet
queue, overlap, or hide incomplete replay rows; it is the small contract slice
that makes those next tests clean.

Follow-up:

```text
CompactSearchPayloadGateV1 now provides the profile-only sample-visibility
guard:

register_action_step(action_step)       # action can drive the CPU env
attach_replay_payload(replay_payload)   # row becomes sample-visible only here
require_replay_payload(handle)          # fails if payload is incomplete
```

It allows out-of-order completion by stable handle and rejects payloads that
arrive before their action step, duplicate handles, and stale/reordered ids.
This is still a local contract helper, not a queue or trainer integration.

The compact service replay proof now also stages the next env joint action from
`CompactSearchActionStepV1`, not from the full search arrays. The full arrays
are still used to build and flush the replay payload, but the action-critical
path has its own object and its own checksum:

```text
CompactSearchActionStepV1.selected_action
-> next joint_action[B,P]
-> next env step

CompactSearchReplayPayloadV1.visit_policy/root_value
-> payload gate
-> replay row visibility
```

Telemetry added:

- `compact_service_replay_action_step_drives_next_action_verified`
- `compact_service_replay_two_phase_payload_verified`
- `compact_service_replay_sample_visible_before_payload_flush`
- `compact_service_replay_sample_visible_after_payload_flush`
- `compact_service_replay_payload_gate_pending_count`
- `compact_service_replay_payload_gate_complete_count`

Validation:

```text
uv run ruff check src/curvyzero/training/compact_search_service.py \
  tests/test_compact_search_replay_contract.py \
  scripts/run_curvytron_hybrid_observation_profile_manifest.py \
  tests/test_curvytron_hybrid_observation_profile_manifest_runner.py

uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_compact_torch_search_service.py \
  tests/test_curvytron_hybrid_observation_profile_manifest_runner.py

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k 'real_direct_ctree_compact_service_drives_next_step_and_matches_rows or compact_torch_search_service_drives_next_step_and_matches_rows or compact_service'
```

Updated result: ruff clean, compact/search/manifest focused bundle `34 passed`,
source-state compact-service proof tests `4 passed`, focused boundary
compact-service tests `3 passed`.

2026-05-23 slab slice:

```text
src/curvyzero/training/compact_rollout_slab.py
```

This is the first profile-only compact owner. It is not wired into Coach or
Modal. It owns one local compact rollout stream:

```text
compact batch k
-> root batch k
-> CompactSearchServiceV1.run(...)
-> staged next_joint_action k+1
-> compact batch k+1 must apply that action
-> previous search commits to CompactReplayIndexRowsV1
```

It fails if the next batch ignored the staged selected actions. It also maps
selected active-root actions back to dense `[B,P]` joint actions with legal-mask
checks.

The slab is also now opt-in wired into
`HybridBatchedObservationProfileManager`. The manager takes
`compact_rollout_slab=None` by default, so existing profile and Coach behavior
does not change. When supplied, it builds the same `HybridCompactBatch` before
scalar LightZero materialization, calls the slab, records
`compact_rollout_slab_sec`, and exposes the slab step/telemetry on
`HybridObservationProfileStep`.

Validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_compact_torch_search_service.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_curvytron_hybrid_observation_profile_manifest_runner.py

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k 'real_direct_ctree_compact_service_drives_next_step_and_matches_rows or compact_torch_search_service_drives_next_step_and_matches_rows or compact_service'
```

Updated validation: ruff clean. Full local files passed:

```text
tests/test_source_state_hybrid_observation_profile.py -> 38 passed
tests/test_source_state_batched_observation_boundary_profile.py -> 115 passed
tests/test_compact_search_replay_contract.py -> 22 passed
```

Main-thread stance while they run:

```text
Do not touch live Coach runs.
Do not promote compact Torch to Coach.
Keep the next work profile-only until the denominator and validation gates are
clear.
Prefer one clean boundary with strong telemetry over five half-labeled speed
lanes.
```

The decision we need from this wave:

```text
P0 = compiled/fused search behind CompactSearchServiceV1
  vs array-native CTree
  vs MCTX/JAX sidecar
  vs actor/env/replay ownership first
```

The answer can be "two lanes in parallel" if each lane has a small falsifier.

### Closed 2026-05-23 Dataflow Wave

- Rawls: current full dataflow map through env, render stack, root batch,
  search, replay, RND, learner. Completed; major finding is that B512/R1024 is
  already tens of MiB of observation stack, while search outputs are tiny by
  bytes. The Amdahl trap is optimizing roots/sec while stock learner/replay/RND
  still materialize full observation rows.
- Bacon: sync/data movement audit, including host/device copies and object
  materialization. Completed; major finding is that one selected-action
  readback is legitimate while env remains CPU, but replay payload readback,
  full stack copies, root observation copies, Python row/list materialization,
  and RND materialization must be separated and measured.
- Beauvoir: design matrix and critique. Completed; major finding is that
  compact env/render/replay ownership plus compiled/fused search are the
  practical near lanes, while MCTX/JAX and native SoA are the honest larger
  architecture probes.
- Sartre: validation harness. Completed; major finding is to lock
  denominators, run deterministic local gates first, and treat compact Torch as
  profile-only until trainer-facing parity passes.

## Immediate Next Steps

1. Add a concise dataflow/sync summary to `world_model.md`.
2. Close the completed subagents.
3. Choose the next experiment row set from this map:
   - same-denominator direct CTree vs compact Torch service vs service-tax,
   - no-noise exact gate first,
   - sim16 and sim32,
   - RND meter compatibility row,
   - one input-handoff ceiling row if the denominator says env/input is hot.
4. Do not launch or modify live Coach runs from this lane.
