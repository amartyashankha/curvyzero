# OPT-132 Architecture Reset

Date: 2026-06-04

## Why This Reset Exists

The last several weeks proved the compact-owned loop can close end to end, but
did not prove a stable speedup. Repeated exact H100 rows kept the same work
identity and still swung hard. The moving bucket is broad sample-gate /
learner-batch-builder user CPU, and the current loop still spends too much time
rebuilding training tensors through Python/Torch orchestration.

The mistake was treating that as an endless attribution problem. It is now an
architecture problem.

## Current World Model

```text
game mechanics: CPU
search/model inference: GPU-oriented, via compact_torch_search_service
learner: GPU-oriented compact MuZero learner
replay/sample/proof: mixed CPU/GPU, too much Python
current stable speed claim: none
10x plausibility: unproven
latest replay result: sample gate fixed as a surface, full loop still slow
latest local loop proof: OPT-132BK closed search-feedback slab/replay/sample
  toy gate; still not speed evidence
```

The current accepted path is a benchmark target, not necessarily the design to
copy internally.

## Curvy Dataflow Today

- Env mechanics run on CPU in `vector_multiplayer_env.py` /
  `vector_runtime.py`, reached through the hybrid profile actor.
- The main profile loop in `source_state_hybrid_observation_profile.py`
  computes primary residual as outer step wall minus named actor, observation,
  slab/search, sample, learner, and policy-refresh timers. OPT-132BF's
  `122.679s` residual is therefore manager/slab/profile object, checksum, copy,
  telemetry, and deferred-bookkeeping suspicion until the fixed-buffer toy
  proves otherwise.
- Actor/autoreset runs through `HybridBatchedObservationProfileManager.step`,
  `InProcessHybridCurvyTronActor.step_into`, and
  `VectorMultiplayerEnv.step_compact_profile`. Reset concentration points are
  `autoreset_done_rows_compact_profile` and
  `_reset_selected_rows_compact_profile`.
- Env runtime / `step_many` runs through
  `VectorMultiplayerEnv._advance_runtime_for_public_step` into
  `vector_runtime.step_many` / `_step_many_kernel`; collision/body scans are the
  first inner mechanics suspects.
- Search/action is owned by `CompactRolloutSlab.step`, which builds compact
  roots and calls `CompactTorchSearchServiceV1`.
- Search service work starts at `CompactTorchSearchServiceV1.run_action_step`;
  root observation preparation and compact Torch tree search are the first
  dispatch/inference suspects if the env toy is fast.
- Replay append is split: `CompactRolloutSlab._commit_previous` materializes
  replay index rows, then `CompactOwnedLoopV1.record_step` appends them into
  `_CompactReplayRingV1`.
- Sample gate is still owned by `_CompactReplayRingV1.sample_from_snapshot` and
  `CompactOwnedLoopV1._sample_and_train`; it handles filtering, RNG, grouping,
  metadata, fallback, and learner-batch construction.
- Learner-batch construction still lives in the sample path. The resident fast
  path is `_build_compact_resident_grouped_device_learner_batch_fast`; host
  fallback is `build_compact_muzero_learner_batch_v1`.
- Learner update is separate again: `CompactOwnedLoopV1._train_learner_now`
  calls `CompactMuZeroLearner.train_on_learner_batch`.

Boundaries that must move for a plausible 5x-10x:

1. Replay sampling plus learner-batch construction should become a resident
   trainer/replay service that emits learner-ready batches without parent
   Python walking every sampled group.
2. Search/replay production should be tighter: OPT-132BK proves this locally
   for the toy, but the real loop still needs the same ownership compression
   without parent Python/object handoffs.
3. Parent-owned rich step objects should stop being the hot handoff format. Env,
   observation, action feedback, search, and replay append should move through
   fixed buffers or compact append deltas.

## External Patterns To Copy

AlphaZero/MuZero-style systems do not usually treat sample construction as a
per-row Python side task:

- Self-play/search is a first-class batched service.
- Neural network inference is batched and cached.
- Replay stores structured training targets or enough fixed-shape data to build
  them without per-sample Python work.
- Learner updates consume fixed-shape tensors.
- H100-scale measurements come after toy/mechanics proof, not before every
  architecture decision.

PufferLib/EnvPool/Sample Factory/Isaac-style systems point at the environment
side of the same lesson:

- Step many environments as dense batches.
- Keep per-env Python out of the hot loop.
- Use fixed contiguous buffers for env state, observations, actions, rewards,
  terminal/death flags, reset/autoreset metadata, and replay handoff.
- Use compiled/vectorized mechanics when CPU stepping is the bottleneck.
- Use shared-memory or buffer-id handoffs rather than serializing trajectories
  through process/Python boundaries.
- If mechanics can live on GPU, keep observations/actions/rewards/replay append
  resident and make transfer boundaries explicit.
- Separate the env-step ceiling from search, replay, and learner ceilings before
  deciding what to rewrite.

Primary references to keep in view:

- AlphaZero paper: https://arxiv.org/abs/1712.01815
- MuZero paper: https://arxiv.org/abs/1911.08265
- EfficientZero paper: https://arxiv.org/abs/2111.00210
- OpenSpiel AlphaZero docs: https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- LightZero repository/docs: https://github.com/opendilab/LightZero
- PufferLib docs: https://puffer.ai/docs.html
- PufferLib repository: https://github.com/PufferAI/PufferLib
- EnvPool paper: https://arxiv.org/abs/2206.10558
- EnvPool repository: https://github.com/sail-sg/envpool
- Sample Factory architecture: https://www.samplefactory.dev/06-architecture/overview/
- Sample Factory repository: https://github.com/alex-petrenko/sample-factory
- Isaac Gym paper: https://arxiv.org/abs/2108.10470
- Isaac Lab repository: https://github.com/isaac-sim/IsaacLab

## What Went Wrong

- We overused H100 rows as the first test of ideas.
- We kept adding proof/timer fields after the evidence already said the current
  loop shape was suspect.
- We optimized narrow builder buckets while the full sample-gate builder surface
  remained CPU-heavy and unstable.
- `goal.md` became a chronological ledger, so old "explain before patch" notes
  kept competing with newer decisions.
- Subagents were used for audits, but not enough for broad architecture search
  and toy proof design.

## Current Structural Patch

OPT-132BA added a default-off learner-ready unroll-2 cache for the fused
resident grouped learner-batch path. Local proof is good:

```text
cache_requested: true
cache_available: 2
cache_eligible: 2
cache_used: true
cache_call: 2
cache_fallback: 0
cache_fallback_reason: none
cache_impl: learner_ready_unroll2_cache_v1
unroll_path: learner_ready_unroll2_cache
```

The first accepted H100 cache attempt failed the fail-closed cache proof gate
before producing speed evidence. Treat that as proof/plumbing feedback, not a
performance result.

The maintained tensor-native replay table path now exists locally. When the
default-off tensor-native replay flag is requested, `_CompactReplayRingV1`
maintains per-record learner-table entries, snapshots them for sampling,
rebuilds them when metadata enables the path, invalidates them on eviction or
stale successor windows, and gathers sampled rows from maintained table state.
The 32x8/128-row CPU proof passed with `maintained_record_table_v1`, reused
records `32`, missing records `0`, and real sample-gate median `0.000867458s`
versus current ring sample/build `0.004377667s` (`5.047x`). Accepted-preset
proof projection is now wired locally through smoke, Modal, launcher,
standalone remote validation, and evidence validation, including the parent
learner-ready cache dependency. This is local architecture evidence, not H100
speed evidence.

OPT-132BD then passed the H100 maintained-table proof with zero fallbacks but
was slow (`3884.88 env steps/sec`, `285.727s`). OPT-132BF added the maintained
sample-universe layer and cut sample gate from `74.005s` to `4.116s`, but
full-loop speed was still `5362.68 env steps/sec`, only `0.42x` OPT-104. This
closes replay/sample as the current P0 surface. The active branch is now
owner-search deferred-maintenance action-only replay ownership, because it
removes the parent-side search-result/replay-row handoff rather than polishing
the sample builder again.

## Immediate Order Of Operations

1. Treat `scripts/benchmark_vector_fixed_action_tape.py` as a closed local
   toy ladder through OPT-132BK:
   the first deterministic zero-observation/no-search proof passes with current
   mapping-shaped CPU stepping versus fixed-buffer direct runtime. Terminal/
   autoreset rows now pass in the wall-death fixture. Rendered observation now
   passes with zero_observation_stub false, schema/hash/shape proof, nonzero
   observation content, and terminal/autoreset equality preserved. Fixed-shape
   search/root handoff now passes locally with root/action/search checksums,
   deterministic selected-action/replay-payload digests, and zero CTree/tolist/
   per-sim D2H. Search-feedback slab/replay/sample now passes locally with
   selected actions fed into the next joint action, replay appends, index-row
   checksums, replay-ring samples, and sample-batch checksums.
2. Promote the selected owner-search candidate:
   `CompactRolloutSlab.run_action_step`, `owner_defer_maintenance`,
   action-only search result, owner-materialized replay, no parent committed
   rows, owner maintenance drain proof, and truthful parent-commit metadata.
3. Keep the compact dataflow map current with CPU/GPU boundaries:
   env state layout, reset/autoreset, terminal/death identity, observation
   buffer ownership, action selection, search, replay append, replay storage,
   sample gate, learner batch, learner update, proof/reporting.
4. Keep remaining toy/mechanics benchmarks as bounded fallback lanes. Run them
   only if the owner-search candidate is falsified or the ownership map selects
   that handoff next.
5. Only run exact H100 rows after the real-loop architecture candidate passes
   local fail-closed proof fields.

## Closed Replay Toy Artifact

The first replay/learner-batch toy is now implemented locally. It targets the
measured sample-gate / learner-batch-builder CPU surface directly.

Artifact:

```text
script: scripts/benchmark_compact_tensor_native_unroll2_replay.py
default device: cpu
optional device: cuda
inputs: records, rows_per_record, sample_rows, iters, terminal_rate, seed
test: tests/test_benchmark_compact_tensor_native_unroll2_replay.py
```

Compare three paths:

```text
1. Current replay ring sample/build path:
   _CompactReplayRingV1.sample_from_snapshot(... build_compact_muzero_learner_batch=True)
2. Real tensor-native sample-gate path:
   _CompactReplayRingV1.sample_from_snapshot(... tensor-native flag true)
   using maintained per-record table entries plus row-index gather
3. Resident grouped builder with learner-ready unroll-2 targets:
   _build_compact_resident_grouped_device_learner_batch_fast(...)
4. Toy tensor-native path:
   prepacked replay tensors + prebuilt unroll-2 target tensors + one sampled
   row-index tensor -> CompactMuZeroLearnerBatchV1
```

Proof checks:

```text
sampled_flat_row_checksum
source_record_window_checksum
learner_batch_sample_row_checksum
num_unroll_steps == 2
host_fallback_allowed == false
sample order preserved
observation/action/action_mask equality
target_reward/target_value/target_policy equality
done/terminated/truncated/mask/weight equality
```

Reuse these code surfaces:

```text
_CompactReplayRingV1.sample_from_snapshot
_compact_replay_learner_ready_unroll2_targets_for_entry
_select_compact_learner_ready_unroll2_targets
_build_compact_resident_grouped_device_learner_batch_fast
CompactDeviceReplayIndexRowsV1
CompactMuZeroLearnerBatchV1
tests/test_source_state_hybrid_observation_profile.py resident fixtures
```

Build the fixed-action-tape env mechanics toy next, or earlier only if the
dataflow map shows env stepping is the first ownership boundary to move.

Local toy result from 2026-06-04:

```text
command:
  uv run python scripts/benchmark_compact_tensor_native_unroll2_replay.py \
    --records 128 --rows-per-record 16 --sample-rows 512 --iters 5 \
    --terminal-rate 0.02 --seed 132 --device cpu

proof:
  required_pass: true
  ring_vs_grouped_equal: true
  grouped_vs_flat_equal: true
  checksum_match: true
  ring_cache_used: true
  grouped_cache_used: true
  host_fallback_allowed: false
  device_replay_index_rows_sample_all: true

median wall:
  current ring sample/build: 0.01673870798549615s
  resident grouped learner-ready unroll-2: 0.013820332998875529s
  flat tensor-native gather: 0.000054791016736999154s

local toy ceiling:
  grouped vs current: 1.211x
  flat gather vs current: 305.501x
```

## Active Env/Outer-Loop Toy Artifact

The first fixed-action-tape env/outer-loop mechanics proof now passes locally.

```text
script: scripts/benchmark_vector_fixed_action_tape.py
scenario: source_borderless_wrap_skips_destination_body_then_next_frame_kills.json
input: deterministic row-major int action tape
latest local artifact:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132bg-local-fixed-action-env-outer-loop-toy-b1024-m180-w45-20260604.json
compare:
  current compact actor/env path
  fixed-buffer env/autoreset loop with zero observation and no-op search
  fixed-buffer mechanics + current resident observation update
  fixed-buffer mechanics + CompactTorchSearchServiceV1.run_action_step
  fixed-buffer mechanics + CompactRolloutSlab commit/root/replay index work
proof:
  state checksum
  body checksum
  reward/done/truncated/death cause
  autoreset rows
  action masks
  observation checksum
  replay index rows
latest result:
  required_pass true
  full_state_checksum_match true
  full_state_field_count 103
  per_step_state/body/death_checksum_match true
  uncompared_output_fields []
  compact_profile_wall 0.3585549167764839s
  fixed_buffer_direct_wall 0.21791812602896243s
  fixed_vs_compact_speedup 1.6453652723171366x
  terminal_row_count 0
  autoreset_row_count 0
  new_death_row_count 1024
  measured_new_death_row_count 0
  measured_initial_fixture_transition_exercised false
  death_cause_names opponent_trail
  observation is a zero stub in this BG artifact; superseded by BI for
  rendered-observation proof
terminal/autoreset result:
  scenario source_normal_wall_death_step
  artifact opt132bh-local-fixed-action-terminal-autoreset-toy-b1024-m1-w0-20260604.json
  required_pass true
  compact_profile_wall 0.22919812600594014s
  fixed_buffer_direct_wall 0.172821791988099s
  fixed_vs_compact_speedup 1.3262107941903727x
  terminal_row_count 1024
  autoreset_call_count 1
  autoreset_row_count 1024
  terminal_rows_equal_autoreset_rows true
  death_cause_names wall
  observation is a zero stub in this BH artifact; superseded by BI for
  rendered-observation proof
rendered-observation result:
  artifacts:
    opt132bi-local-fixed-action-rendered-observation-toy-b32-m3-w1-20260604.json
    opt132bi-local-fixed-action-rendered-observation-terminal-autoreset-toy-b4-m1-w0-20260604.json
  required_pass true
  zero_observation_stub false
  observation_schema_id curvyzero_source_state_canvas_gray64/v0
  observation_schema_hash d383dad88bdf0a0f
  borderless observation_shape [32, 3, 4, 64, 64]
  borderless render_row_count 288
  borderless observation_nonzero_count 1179648
  terminal observation_shape [4, 2, 4, 64, 64]
  terminal render_row_count 8
  terminal observation_nonzero_count 32768
  terminal_row_count 4
  autoreset_row_count 4
  terminal_rows_equal_autoreset_rows true
  renderer_backend cpu_oracle
promotion:
  only if a local/toy result removes a whole actor/autoreset/env/observation/
  search/replay surface; current pivot is owner-search action-only proof
  hardening plus scale/falsification
```

Interpretation:

```text
if fixed_buffer_outer_sec - env_runtime_sec - autoreset_sec - observation_sec
   - search_stub_sec collapses:
  primary residual is mostly the current manager/slab/profile object and
  bookkeeping path
if actor/autoreset remains large:
  chase VectorMultiplayerEnv reset/runtime first
if fixed loop is fast:
  add back exactly one subsystem at a time in the real candidate; the local
  toy has already closed observation, search/root, and slab/replay/sample
```

This is not H100 speed evidence. It is local architecture evidence only. The
fixed-action toy tests whether fixed-buffer env/autoreset/observation/search/
slab handoffs can remove the remaining OPT-132BF surfaces before any H100 row.

Latest maintained-table real-loop proof from 2026-06-04:

```text
command:
  uv run python scripts/benchmark_compact_tensor_native_unroll2_replay.py \
    --records 32 --rows-per-record 8 --sample-rows 128 --iters 3 \
    --terminal-rate 0.02 --seed 132 --device cpu

proof:
  required_pass: true
  ring_vs_grouped_equal: true
  real_tensor_native_vs_grouped_equal: true
  grouped_vs_flat_equal: true
  checksum_match: true
  real_tensor_native_table_source: maintained_record_table_v1
  real_tensor_native_table_reused_record_count: 32
  real_tensor_native_table_missing_record_count: 0
  host_fallback_allowed: false

median wall:
  current ring sample/build: 0.004377667006338015s
  real tensor-native maintained table: 0.0008674579730723053s
  resident grouped learner-ready unroll-2: 0.0035462919913697988s
  flat tensor-native gather: 0.00002733399742282927s

local architecture result:
  real tensor-native vs current: 5.046546509721368x
  flat gather vs current: 160.1546579016577x
```

Default-off real sample-gate prototype from 2026-06-04:

```text
source path:
  _CompactReplayRingV1.sample_from_snapshot

new proof keys:
  compact_muzero_learner_batch_tensor_native_replay_requested
  compact_muzero_learner_batch_tensor_native_replay_used
  compact_muzero_learner_batch_tensor_native_replay_call_count
  compact_muzero_learner_batch_tensor_native_replay_fallback_count
  compact_muzero_learner_batch_tensor_native_replay_fallback_reason
  compact_muzero_learner_batch_tensor_native_replay_impl

focused validation:
  uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py \
    tests/test_benchmark_compact_tensor_native_unroll2_replay.py \
    scripts/benchmark_compact_tensor_native_unroll2_replay.py
  uv run pytest tests/test_benchmark_compact_tensor_native_unroll2_replay.py -q
  uv run pytest tests/test_source_state_hybrid_observation_profile.py -q

benchmark:
  uv run python scripts/benchmark_compact_tensor_native_unroll2_replay.py \
    --records 32 --rows-per-record 8 --sample-rows 128 --iters 3 \
    --terminal-rate 0.02 --seed 132 --device cpu

proof:
  required_pass: true
  ring_vs_grouped_equal: true
  real_tensor_native_vs_grouped_equal: true
  grouped_vs_flat_equal: true
  checksum_match: true

median wall:
  current ring sample/build: 0.0048690410039853305s
  real tensor-native sample-gate path: 0.005500167026184499s
  resident grouped learner-ready unroll-2: 0.003923166019376367s
  flat tensor-native gather: 0.000029000017093494534s
```

Historical conclusion: correctness/proof was good, but speed shape was not
good because the prototype prebuilt the full flat table in each sample call.
This was superseded by maintained_record_table_v1 and OPT-132BD/BF. Do not
resume table-maintenance work unless the fixed-buffer design specifically needs
replay changes.

## Stop Conditions

Do not launch another H100 row unless the fixed-action env/outer-loop toy or
primary-residual map produces a specific fail-closed architecture candidate.
Do not claim speedup from local toys or proof-only rows. Do not stack unrelated
optimizations on top of an unstable row.
