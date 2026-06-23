# Measurement Ledger

Date: 2026-06-07

Purpose: keep speed comparisons honest. This file only keeps rows that affect
the current decision.

## Compare Rows Only When These Match

- hardware;
- batch, actor count, measured steps, warmup steps;
- death mode;
- search service;
- replay sample interval, sample batch, replay capacity;
- learner unroll, learner update count, refresh interval;
- host fallback count;
- whether the row touches the real trainer.

Long-window stability diagnostics intentionally change measured steps and
warmup steps. They can be compared to each other, but they do not replace
OPT-104 unless a matched long-window baseline is also produced.

## OPT-132 Fixed-SoA Slot-Candidate + Digest-Deferral Local Smoke

```text
run: opt132-local-fixed-soa-slot-digestdefer-smoke-20260607-r3
currency: local wiring proof, not speed evidence
death mode: profile_no_death
shape: B8/A1, 12 measured, 4 warmup, sample batch 8, unroll 2
status: ok=true
artifact:
  artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-local-fixed-soa-slot-digestdefer-smoke-20260607-r3/
    compact_coach_speed_row_smoke_report.json
```

Proof read:

```text
owner-search: threaded proxy
direct replay: 12 logical transitions -> 3 transport entries
search payload bytes: 0
fixed SoA requested / used: true / true
fixed SoA slot writes: 12
columnar append used / slot writes: false / 0
fixed SoA entry/step/learner-ready/table-entry/table-concat counts: 0/0/0/0/0
fixed SoA fallback: 0 / none
learner slot-candidate path: true
selected fixed-SoA records: 6
owner-ref digest deferred to search refresh: true
learner-side owner train digest sec: 0.0
```

Interpretation:

- This is a composition proof for two default-off support patches:
  fixed-SoA slot-candidate replay and owner-ref digest deferral.
- It is not normal-death speed evidence and cannot promote anything.
- The two failed preflights before r3 were command/shape checks, not mechanism
  failures: r1 forced local `direct_core` on a model surface without the needed
  methods, and r2 used a tiny normal-death shape without terminal/death proof.
- H100 is justified only as a labeled ablation/support row, or as part of a
  broader owner-buffer/root/search/learner-publication patch that preserves the
  normal-death/action/replay/terminal gates.

## OPT-132 Fixed-SoA Slot-Candidate H100 Support Row

```text
run: opt132-h100-slotcandidate-support-b1024a1-normal-unroll2-m724-w180-r1-20260607
currency: H100 support row, speed rejected
shape: B1024/A1, normal death, 724 measured, 180 warmup, sample batch 512, unroll 2
artifact:
  artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-h100-slotcandidate-support-b1024a1-normal-unroll2-m724-w180-r1-20260607/
    compact_coach_speed_row_modal_report.json
function call: fc-01KTJ9WS6XRJ506KSE41TCJR2G
```

Speed read:

```text
env/s: 10348.874836136198
wall: 71.638319309s
env steps: 741376
vs OPT-104: 0.8156x
vs columnar r2 15852.67 env/s: 0.6528x
projected replay-ceiling env/s from this row: 13968.78
projected reaches 2x: false
additional removed sec to 2x: 23.861s
```

Proof read:

```text
normal-death gate: true
terminal rows / death rows: 4926 / 4926
terminal sample rows: 4
terminal unroll target rows: 8
accepted-fast-path violations: []
tensor-native replay violations: []
learner-ready cache violations: []
unroll2 specialized-builder violations: []
fixed SoA requested / used: true / true
fixed SoA fallback: 0 / none
fixed SoA records / selected / table rows: 2155 / 373 / 1460766
fixed SoA slot writes: 720
fixed SoA entry/step/learner-ready/table-entry/table-concat counts: 0/0/0/0/0
tensor-native replay used: true
direct transition replay: 724 transitions -> 181 batches -> 181 transport entries
columnar append used: false
search payload bytes: 0
policy lag max: 45
```

Bucket read:

```text
source measured wall: 71.638s
primary accounted/residual: 56.464s / 15.174s
actor step wall: 23.140s
observation: 10.592s
compact rollout slab: 22.733s
owner parent wait total: 18.565s
owner worker search total: 13.933s
owner worker replay append: 6.131s
owner train wall: 60.232s
fixed-SoA total: 4.083s
fixed-SoA successor-index: 4.080s
tensor-native gather: 0.318s
learner gate: 4.922s
```

Interpretation:

- Fixed-SoA slot-candidate replay is H100 proof-clean, but it is a speed
  rejection. It is worse than OPT-104 and much worse than columnar r2.
- Do not repeat fixed-SoA exact/locality/slot-candidate unchanged as a variance
  or longer-run question. This row is too far below the target.
- Keep the proof as support for a broader fixed owner-buffer/root/search/
  learner-publication design; do not spend the main lane on gather tweaks alone.

## OPT-132 Owner-Ref Digest-Deferral Modal Plumbing Closure

```text
currency: launch/projection proof, not speed evidence
status: local launcher and H100 Modal producer surfaces pass tests
```

What changed:

```text
Modal producer config accepts:
  owner_search_defer_model_state_digest_to_refresh
Remote argv forwards:
  --owner-search-defer-model-state-digest-to-refresh
Result bundle projects:
  owner_search_defer_model_state_digest_to_refresh_requested
```

Validation:

```text
uv run ruff check src/curvyzero/infra/modal/compact_coach_speed_row.py \
  scripts/run_compact_coach_speed_row_modal_smoke.py \
  tests/test_compact_coach_speed_row_smoke.py
uv run pytest tests/test_compact_coach_speed_row_smoke.py -q -k \
  "owner_search_digest_deferral or remote_modal_owner_search_config_projects_digest_deferral or owner_search_fixed_soa_replay or accepted_fast_path_preset_rejects_owner_search_overrides or rejects_digest_deferral"
  -> 5 passed
uv run pytest tests/test_compact_coach_speed_row_smoke.py -q
  -> 90 passed
```

Interpretation:

- This closes a real launch-surface mismatch: before this patch, a remote
  digest-deferral run could not honestly prove the requested config.
- This is not a speed row. It only makes the next labeled H100 ablation or
  broader owner-boundary run mechanically meaningful.

## OPT-132 Direct Root Build-Request Core Local Gate

Local proof:

```text
status: local unit proof only, not speed evidence
code:
  CompactRootBuildRequestV1
  compact_root_build_request_v1_from_batch()
  build_compact_root_batch_v1_from_request()
  CompactDirectRootStoreV1.publish_root_build_request()
  CompactOwnerSearchSlabProxyV1.run_action_step_from_root_build_request()
  CompactLazyThreadedOwnerSearchSlabProxyV1 local routing
  CompactOwnerSearchDirectStepperV1(direct_root_build_request=True)
tests:
  test_root_build_request_builds_resident_stub_root_batch
  test_direct_root_store_builds_root_request_inside_owner
  test_owner_search_direct_stepper_root_build_request_avoids_parent_builder
  test_threaded_owner_search_direct_stepper_root_build_request_uses_owner_build
validation:
  focused 4-test slice passed
  full owner-search service suite: 45 passed
```

Local speed-row proof:

```text
run: opt132-local-owner-rootbuildrequest-threaded-directtransition-smoke-20260607-r3
status: ok=true, local proof only
root-build request schema/kind: curvyzero_compact_root_build_request/v1 / resident_root_view_build_request_v1
publish / resolve / owner-build: 15 / 15 / 15
parent builder used / calls: false / 0
parent root-batch build sec: 0.0
request observation included / bytes / host-observation bytes: false / 0 / 0
resident handle / resident-root-view proved: true / true
resident H2D/D2H: 0.0 / 0.0
host-observation stub requested/stubbed/materialized bytes: true / true / 0
parent committed/stored rows: 0 / 0
search payload bytes: 0
direct replay batches / transitions / transport: 3 / 12 / 3
action feedback mismatch count: 0
owner maintenance failed / pending / policy lag: false / 0 / 0
owner pid / request bytes / result bytes: reported / 356 / 738
local env/s: 96.98, not speed currency
```

H100 closeout:

```text
run: opt132-h100-rootbuildrequest-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607
status: proof clean, speed rejected
speed: 11327.75 env/s
wall: 65.4477s
baseline delta: 0.893x OPT-104 by env/s
current fastest columnar r2 delta: 0.715x by env/s
root-build request schema/kind: curvyzero_compact_root_build_request/v1 / resident_root_view_build_request_v1
publish / resolve / owner-build: 904 / 904 / 904
parent builder used / calls: false / 0
parent root-batch build sec: 0.0
parent root-batch objects sent: 0
request observation included / bytes / host-observation bytes: false / 0 / 0
resident root view proved / H2D / D2H / host fallback: true / 0.0 / 0.0 / 0.0
host-observation stub requested/stubbed/materialized bytes: true / true / 0
parent committed/stored rows: 0 / 0
search payload / visit / root-value bytes: 0 / 0 / 0
direct replay batches / transitions / transport: 181 / 724 / 181
action feedback mismatch count: 0
normal-death terminal gate: true
terminal sample / target rows: 512 / 512
owner maintenance failed / pending / policy lag: false / 0 / 0
```

Decision:

- This proves the direct stepper and the real local threaded owner proxy can
  avoid the parent `build_compact_root_batch_v1` symbol; tests monkeypatch that
  parent symbol to raise and still complete through
  `run_action_step_from_root_build_request`.
- Speed-row CLI/report proof is wired and locally closed but still proof-only. The
  default-off flag `--owner-search-direct-root-build-request` reaches
  `CompactOwnerSearchDirectStepperV1(direct_root_build_request=True)`, local
  and Modal report paths preserve the build-request proof fields, and the
  speed-row guard fails closed on nonzero parent builder calls, publish/resolve
  mismatch, owner-build mismatch, parent root-batch objects sent, request
  observation bytes, missing resident handle, and lost resident-root/stub/
  action-only gates. Validation: full speed-row smoke module `87 passed`.
- The H100 decision row keeps the proof closed but fails speed badly. It removes
  parent root-batch build time (`0.0s`, parent builder calls `0`) but the whole
  loop regresses: parent wait `26.151s`, replay append `22.783s`, learner train
  `15.526s`, worker search `16.422s`, observation `13.196s`, and slab
  `29.850s`, all worse than columnar r2. Decision: do not repeat
  root-build-request unchanged. Preserve it as a support gate and move to a
  larger owner-resident buffer/search/learner-publication surface.

## OPT-132 Resident Root Host-Observation Stub Local Gate

Local proof row:

```text
run: opt132-local-owner-resident-root-hoststub-threaded-directtransition-smoke-20260607-r1
status: ok=true, local proof only
stub requested/stubbed: true/true
stub kind: zero_stride_shape_only_v1
stub materialized bytes: 0
stub logical bytes: 262144 last step / 3932160 total
resident root view: required/proved true
resident H2D/D2H: 0.0/0.0
host fallback allowed: false
action-only / owner-materializes-replay: true / true
parent committed/stored rows: 0/0
search payload bytes: 0
transition entries / batches / transport entries: 12/3/3
final policy lag / pending maintenance / failed: 0/0/false
```

Decision:

- This closes the local proof that direct-root threaded owner mode can avoid
  materializing parent host-observation bytes while search consumes the source
  resident root handle.
- This is not H100 speed evidence and does not remove parent root-batch
  construction. The next speed-relevant patch must replace the parent
  root-batch handoff with a root-view/build-request owner boundary or combine
  root ownership with another measured owner-buffer/search/parent-wait/
  learner-publication/mechanics surface.

## OPT-132 Resident Root-View Local Gate

Local proof row:

```text
run: opt132-local-owner-resident-root-view-threaded-directtransition-smoke-20260607-r5
status: ok=true, local proof only
resident root view: required/proved true
kind: direct_root_batch_resident_handle_v1
direct root publish/resolve: 15/15
resident H2D/D2H: 0.0/0.0
host fallback allowed: false
action-only / owner-materializes-replay: true / true
parent committed/stored rows: 0/0
search payload / visit / root-value bytes: 0/0/0
transition entries / batches / transport entries: 12/3/3
final policy lag / pending maintenance / failed: 0/0/false
```

Decision:

- This closes the local proof that direct-root threaded owner mode can consume
  the source resident root handle and fail closed on host fallback.
- This is not H100 speed evidence and does not remove parent root-batch
  construction. The next speed-relevant patch must move a larger owner/root/
  search/mechanics/learner-publication surface while preserving these proof
  fields when resident ownership is claimed.

## OPT-132 Fixed SoA Exact H100 Read

Baseline:

```text
OPT-104 validation-only:
  12689.38 env steps/sec
  14.5255s wall
```

Rows:

```text
r4 fixed SoA row-level semantics bug:
  speed: 3123.45 physical rows/sec equivalent / not a valid speed candidate
  problem: normal proof failed and successor indexing cost was 191.93s

r5 row-level incremental successor cache:
  speed: 7466.24 env/s
  measured wall: 99.30s
  sample aggregate: 72.95s
  status: proof still missing mixed terminal no-bootstrap mask field

r6 proof/sync fix:
  speed: 8931.53 env/s
  measured wall: 83.01s
  sample aggregate: 58.35s
  status: proof clean

r7 delayed uint8 normalization:
  speed: 10662.84 env/s
  measured wall: 69.53s
  sample aggregate: 48.25s
  last gather: 0.2687s
  status: proof clean

r8 selected-group execution plan:
  speed: 11616.45 env/s
  measured wall: 63.82s
  sample aggregate: 43.14s
  last gather: 0.2247s
  selected records: about 373 per 512 sampled rows
  record count: 2155
  table rows: 1460766
  terminal samples: 4
  status: best proof-clean exact fixed-SoA row, still below OPT-104

r9 copy-trim:
  speed: 8491.45 env/s
  measured wall: 87.31s
  sample aggregate: 61.93s
  last gather: 0.3756s
  status: proof clean but speed-regressed; copy-trim reverted
```

Decision:

- This is not solved by "run it longer" unchanged. r8 is proof-clean and still
  loses to OPT-104.
- The current causal blocker is likely learner-batch gather fragmentation:
  a 512-row random sample touches hundreds of record groups.
- The labeled fixed-SoA locality sampler probe below confirms the fragmentation
  mechanism but rejects locality alone as the next whole-loop lever.
- Move the main lane away from fixed-SoA gather tweaks unless the next patch
  also changes broader owner-buffer surfaces.

## OPT-132 Fixed SoA Locality Probe Patch

Patch:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/build_compact_coach_speed_row_smoke.py
scripts/run_compact_coach_speed_row_modal_smoke.py
tests/test_source_state_hybrid_observation_profile.py
tests/test_compact_coach_speed_row_smoke.py
```

Mechanism:

- `compact_replay_fixed_soa_locality_sample_group_size` defaults to `1`.
- Values `>1` choose candidate groups proportional to their row count, then
  sample chunked local rows inside each chosen group.
- This preserves row marginals but changes minibatch correlation and can
  introduce duplicates; metadata therefore marks
  `fixed_soa_locality_sample_semantic_drift=true`.
- The Modal launcher threads
  `--owner-search-fixed-soa-locality-sample-group-size`.

Validation:

```text
python -m py_compile src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py
uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_fixed_soa_locality_sampler_reduces_selected_record_count_and_marks_drift tests/test_source_state_hybrid_observation_profile.py::test_fixed_soa_samples_row_level_successors_for_coalesced_transition_batch tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_launcher_threads_owner_search_fixed_soa_replay -q
```

Result:

```text
py_compile passed
ruff passed
pytest: 3 passed, 2 warnings
guard: group_size=1 matches the exact fixed-SoA RNG path on tensors/checksum
```

H100:

```text
r1:
  run: opt132-h100-fixed-soa-locality-g8-rowlevel-selectedgroups-b1024a1-normal-unroll2-m724-w180-r1-20260607
  status: launch failed
  cause: Modal app did not yet recognize --owner-search-fixed-soa-locality-sample-group-size
  decision: plumbing failure only; fixed in src/curvyzero/infra/modal/compact_coach_speed_row.py

r2:
  run: opt132-h100-fixed-soa-locality-g8-rowlevel-selectedgroups-b1024a1-normal-unroll2-m724-w180-r2-20260607
  status: proof clean, speed rejected
  speed: 10428.59 env/s
  measured wall: 71.0907s
  env steps: 741376
  locality group size: 8
  semantic drift: true
  selected groups: 64
  selected fixed-SoA records: 62
  learner action shape: [512, 2]
  fixed-SoA record/table rows: 2155 / 1460766
  gather: 0.23497s
  terminal sample rows: 8
  owner train sample: 32.03s
  owner train wall: 40.30s
  learner update: 6.52s
  replay append: 17.24s
  worker search: 16.68s
  parent wait: 21.62s
```

Decision:

```text
The probe worked as a causal diagnostic: selected-record fragmentation is real.
It did not improve the whole loop. Do not keep iterating on fixed-SoA
gather/locality as the main lane. A real flat/global replay row layout may
still be part of a larger fixed owner-buffer design, but locality alone does
not justify the expected 2x/10x speedup.
```

Projection cleanup:

```text
patched:
  scripts/build_compact_coach_speed_row_smoke.py
  scripts/run_compact_coach_speed_row_modal_smoke.py
  src/curvyzero/training/compact_coach_speed_row.py
  src/curvyzero/infra/modal/compact_coach_speed_row.py
  tests/test_compact_coach_speed_row_smoke.py
  tests/test_compact_coach_speed_row.py

new top-level fields:
  compact_rollout_slab_sample_gate_fixed_soa_record_count
  compact_rollout_slab_sample_gate_fixed_soa_selected_record_count
  compact_rollout_slab_sample_gate_fixed_soa_table_row_count
  compact_rollout_slab_sample_gate_fixed_soa_locality_*

validation:
  py_compile passed
  ruff passed
  pytest: 3 passed, 2 warnings
```

## OPT-132 Whole-Owner-Buffer Replay Ceiling Instrument

Patch:

```text
scripts/build_compact_coach_speed_row_smoke.py
scripts/run_compact_coach_speed_row_modal_smoke.py
src/curvyzero/training/compact_coach_speed_row.py
src/curvyzero/infra/modal/compact_coach_speed_row.py
scripts/compare_compact_coach_speed_rows.py
tests/test_compact_coach_speed_row_smoke.py
tests/test_compact_coach_speed_row.py
tests/test_compact_coach_speed_row_compare.py
```

Mechanism:

- Adds `compact_whole_owner_buffer_replay_ceiling_*` fields to speed-row
  summary/compact/report surfaces.
- Computes a conservative projection from measured wall, env steps, owner
  replay append, owner train sample, parent wait, worker search, and learner
  update.
- Caps removable wall at the parent-wait-visible portion bounded by replay
  append plus owner-train sample, and preserves a search/update floor.
- Marks the result as projection-only:
  `production_speed_claim=false`, `touches_live_training=false`,
  `speed_currency=local_projection_no_speed`,
  `h100_validation_status=not_run`, `promotion_eligible=false`.
- Evidence validation rejects projection fields that drift toward a production
  speed claim or production speed currency.

Validation:

```text
python -m py_compile scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py src/curvyzero/training/compact_coach_speed_row.py src/curvyzero/infra/modal/compact_coach_speed_row.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row.py
uv run ruff check scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py src/curvyzero/training/compact_coach_speed_row.py src/curvyzero/infra/modal/compact_coach_speed_row.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row.py
uv run pytest tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_projects_and_validates_owner_search_slab_proxy_fields tests/test_compact_coach_speed_row.py::test_compact_coach_speed_row_projection_fields_remain_nonproduction tests/test_compact_coach_speed_row.py::test_compact_coach_speed_row_requires_tensor_native_replay_proof tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_tensor_native_replay_violations_and_report_fields -q
uv run pytest tests/test_compact_coach_speed_row_compare.py -q
```

Result:

```text
py_compile passed
ruff passed
pytest: 4 passed, 2 warnings
compare pytest: 16 passed, 2 warnings
```

Generated review artifact:

```text
command:
  uv run python scripts/compare_compact_coach_speed_rows.py
    --baseline-label columnar-r2
    --row direct-table-r3=artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-direct-table-builder-b1024a1-normal-unroll2-m724-w180-r3-20260607/row_001_result.json
    --row columnar-r2=artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-columnar-append-direct-table-b1024a1-normal-unroll2-m724-w180-r2-20260607/row_001_result.json
    --row fixed-soa-locality-r2=artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-fixed-soa-locality-g8-rowlevel-selectedgroups-b1024a1-normal-unroll2-m724-w180-r2-20260607/row_001_result.json
    --output artifacts/local/curvytron_compact_coach_speed_row_results/opt132-whole-owner-buffer-ceiling-review-20260607/comparison.json

output:
  artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-whole-owner-buffer-ceiling-review-20260607/comparison.json
```

Generated H100 read:

```text
columnar/direct-table r2:
  observed: 15852.67 env/s, 46.7666s wall, 741376 env steps
  replay append + owner train sample: about 16.996s
  parent wait: 17.655s
  conservative projected wall: 29.770s
  conservative projected speed: 24903.25 env/s
  projected speedup vs OPT-104: 1.9625x
  2x target wall for same env steps: 29.212s
  remaining gap: 0.558s

direct-table r3:
  observed: 15681.59 env/s, 47.2768s wall, 741376 env steps
  conservative projected wall: 30.713s
  conservative projected speed: 24138.94 env/s
  projected speedup vs OPT-104: 1.9023x
  remaining gap: 1.500s

fixed-SoA locality r2:
  observed: 10428.59 env/s, 71.0907s wall, 741376 env steps
  conservative projected wall: 49.471s
  conservative projected speed: 14986.22 env/s
  projected speedup vs OPT-104: 1.1810x
  remaining gap: 20.258s
```

Decision:

```text
Replay/sample removal is plausibly close to 2x on the fastest exact stack, but
not enough by itself under the conservative wall model. The next production
rewrite must either prove better overlap/parent-wait removal on a real row or
also attack search, parent wait, learner update/publication, or another
measured non-replay surface.

The fastest columnar row already has action-only owner result transport:
owner materializes replay, parent reconstruction false, parent slab commits
false, search payload bytes / visit policy bytes / root value bytes all zero,
replay payload handle present, and inner two-phase deferred device replay true.
Therefore the next target is not "make the row action-only"; it is the owner
resident root/mechanics/search-dispatch surface or learner-publication overlap.
```

## OPT-132 Default-Off One-Simulation Replay Materialization Deferral

Patch:

```text
src/curvyzero/training/compact_torch_search_service.py
src/curvyzero/training/compact_rollout_slab.py
src/curvyzero/training/source_state_hybrid_observation_profile.py
tests/test_compact_torch_search_service.py
```

Mechanism:

- Adds default-off `CompactTorchCompileConfig.defer_one_simulation_replay_payload`.
- Applies only to `run_action_step()` when `num_simulations == 1` and root
  noise is zero.
- Action phase selects from masked initial policy logits and skips recurrent
  replay-target materialization.
- Replay payload flush materializes one-hot visit policy, root value, and raw
  counts by running recurrent inference later.
- Action-time and flush-time model identity fields must match before recurrent
  materialization. `refresh_model_state()` and `refresh_shared_model_state()`
  fail while deferred one-simulation handles are pending.
- Slab/profile telemetry now carries action-phase request/used/pending/model
  identity fields.

Validation:

```text
python -m py_compile src/curvyzero/training/compact_torch_search_service.py src/curvyzero/training/compact_rollout_slab.py src/curvyzero/training/source_state_hybrid_observation_profile.py tests/test_compact_torch_search_service.py
uv run ruff check src/curvyzero/training/compact_torch_search_service.py src/curvyzero/training/compact_rollout_slab.py src/curvyzero/training/source_state_hybrid_observation_profile.py tests/test_compact_torch_search_service.py
uv run pytest tests/test_compact_torch_search_service.py -q
```

Result:

```text
py_compile passed
ruff passed
pytest: 42 passed, 2 warnings
```

Decision:

```text
This is local proof of a plausible parent-wait/search-surface move, not speed
evidence. Do not launch H100 yet. Next proof must happen in owner-search:
deferred handles flushed before model refresh, action/flush identity match,
model-refresh-crossed count zero, final pending deferred count zero, and no
loss of action-feedback/replay/terminal proof.
```

## OPT-132 Direct Tensor-Native Prebuilt Sample Path

Artifacts:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/build_compact_coach_speed_row_smoke.py
src/curvyzero/training/compact_coach_speed_row.py
scripts/run_compact_coach_speed_row_modal_smoke.py
src/curvyzero/infra/modal/compact_coach_speed_row.py
tests/test_source_state_hybrid_observation_profile.py
tests/test_compact_coach_speed_row_smoke.py
tests/test_compact_coach_speed_row.py
```

Patch:

```text
_CompactReplayRingV1.sample_from_snapshot now has a guarded direct-prebuilt
path for compact MuZero learner-batch-only, tensor-native replay, learner-ready
unroll-2 cache, resident/device replay, maintained unroll-2 candidate universe,
and full maintained-table coverage.

When eligible, the sample gate still preserves sample/metadata proof but does
not construct per-sampled-group resident sample objects before calling the
maintained table gather builder.
```

Proof fields:

```text
compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested
compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible
compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used
compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count
compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason
compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count
compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped
```

Local validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py src/curvyzero/training/compact_coach_speed_row.py scripts/run_compact_coach_speed_row_modal_smoke.py src/curvyzero/infra/modal/compact_coach_speed_row.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row.py
uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_compact_replay_ring_batch_append_deduplicates_tensor_native_refresh tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_owner_search_fused_tensor_native_uses_owner_sample_telemetry tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_requires_tensor_native_replay_proof tests/test_compact_coach_speed_row_smoke.py::test_modal_result_bundle_tensor_native_replay_violations tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_tensor_native_replay_violations_and_report_fields tests/test_compact_coach_speed_row.py -q -k tensor_native
```

Decision:

```text
Do not run a standalone same-window H100 verification for this patch. It is
kept as a local, fail-closed support path because it only skips
per-sampled-group resident sample-object construction. The r3/r4 ring-batch
reports show the remaining replay append/materialization surface is still
about 19.420s, while the group-object/builder surface this patch can remove is
measured in milliseconds. Use the direct-prebuilt proof fields inside the next
fixed-buffer row if the path is active; do not treat this wrapper as the next
speed experiment.
```

## OPT-132 Ring-Batched Owner Replay Append / Cache Refresh

Artifacts:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/build_compact_coach_speed_row_smoke.py
tests/test_source_state_hybrid_observation_profile.py
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-local-owner-search-transitionbatch-ringbatch-compactmuzero-smoke-r2-20260606/compact_coach_speed_row_smoke_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-search-transitionbatch-ringbatch-slab-bypass-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r3-20260606/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-search-transitionbatch-ringbatch-slab-bypass-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r4-20260606/compact_coach_speed_row_modal_report.json
```

Patch:

```text
_CompactReplayRingV1.append_entries batches ring append work.
Owner-search replay store now hands fixed transition-batch entries to the ring
as one tuple.
Learner-ready/tensor-native unroll2 cache refresh candidates are de-duplicated
per batch.
```

Local proof:

```text
unit: test_compact_replay_ring_batch_append_deduplicates_tensor_native_refresh
smoke: opt132-local-owner-search-transitionbatch-ringbatch-compactmuzero-smoke-r2-20260606
local smoke status: ok
local transition batches/logical/transport: 4 / 16 / 4
local tensor-native table source: maintained_record_table_v1
local tensor-native missing records: 0
```

H100:

```text
same window: 724 measured / 180 warmup
same transport shape: 724 logical replay transitions / 181 transport entries
baseline r2 transition-batch-only speed/wall: 13618.91 env/s / 54.437s
baseline r2 replay append: 31.926s

r3 speed/wall: 14394.77 env/s / 51.503s
r3 replay append: 20.204s
r3 learner train: 11.380s
r3 tensor-native reused/missing: 719 / 0

r4 speed/wall: 14160.27 env/s / 52.356s
r4 replay append: 18.636s
r4 learner train: 10.855s
r4 tensor-native reused/missing: 719 / 0

r3/r4 average speed/wall: 14277.52 env/s / 51.930s
r3/r4 average replay append: 19.420s
average delta vs r2: +658.60 env/s, -2.508s wall, -12.506s replay append
```

Decision:

```text
This is a real speed patch and the current best same-window owner-search
candidate. It does not close the goal. The full row is still only about 1.13x
OPT-104, so the next implementation must cut another object/materialization
surface: true fixed SoA/ring-buffer owner storage. The direct-prebuilt sample
path remains support proof, not the next standalone row.
```

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py tests/test_source_state_hybrid_observation_profile.py
uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_compact_replay_ring_batch_append_deduplicates_tensor_native_refresh -q
uv run pytest tests/test_compact_coach_speed_row_smoke.py::test_owner_search_replay_store_metadata_receives_fused_tensor_native_flags tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_owner_search_fused_tensor_native_uses_owner_sample_telemetry tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_launcher_accepts_owner_search_slab_bypass tests/test_compact_owner_search_service.py -k "transition_batch or defer_owner_maintenance or slab_proxy" -q
uv run pytest tests/test_compact_coach_speed_row_smoke.py -k "owner_search and slab_bypass" -q
```

## OPT-132 Fixed Transition-Batch Local Proof

Artifacts:

```text
src/curvyzero/training/compact_rollout_slab.py
src/curvyzero/training/compact_owner_search_service.py
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/build_compact_coach_speed_row_smoke.py
scripts/run_compact_coach_speed_row_modal_smoke.py
tests/test_compact_owner_search_service.py
tests/test_compact_coach_speed_row_smoke.py
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-local-owner-search-transition-batch-smoke-r5-20260605/compact_coach_speed_row_smoke_report.json
```

Read:

```text
mode: owner-search slab bypass plus fixed transition-batch transport
flags: --owner-search-slab-bypass --owner-search-transition-batch-size 4
local speed claim: none; CPU proof only
local wall/speed: 0.0865s, 1110.16 env/s
logical replay transitions: 12
transport entries: 3
transition batches: 3
legacy transition entries: 0
parent committed/stored rows: 0/0
search payload bytes: 0
owner replay pending final: 0
action-feedback mismatches: 0
```

H100:

```text
run: opt132-h100-owner-search-transitionbatch-slab-bypass-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r2-20260605
window: 724 measured / 180 warmup
speed/wall: 13618.91 env/s / 54.437s
r32 comparison: 14002.12 env/s / 52.947s
r33 comparison: 13640.60 env/s / 54.351s
OPT-104 accepted baseline: 12689.38 env/s / 14.5255s at 180/45
logical replay transitions: 724
transport entries: 181
transition batches: 181
legacy transition entries: 0
parent committed/stored rows: 0/0
search payload bytes: 0
owner replay pending final: 0
action-feedback mismatches: 0
owner replay append requests: 181
worker replay append: 31.926s
worker learner train: 9.802s
parent wait total: 18.177s
worker search total: 13.656s
```

Decision:

```text
This proved the transport-count change locally and on H100, but it did not
move whole-row speed beyond r32/r33 noise. It reduced owner replay append
requests from 724 to 181 and moved replay append time modestly, but the row
remained dominated by owner replay materialization/search/actor/observation
surfaces. Do not run this same transition-batch shape unchanged. The next
architecture step is a true fixed SoA/ring-buffer owner boundary or
owner-search direct prebuilt tensor-native learner batches.
```

Validation:

```text
uv run ruff check src/curvyzero/training/compact_rollout_slab.py src/curvyzero/training/compact_owner_search_service.py src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py
uv run pytest tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py -k owner_search -q
```

## OPT-132 Inline-Background Owner Local Proof

Artifacts:

```text
src/curvyzero/training/compact_owner_search_service.py
scripts/build_compact_coach_speed_row_smoke.py
tests/test_compact_owner_search_service.py
tests/test_compact_coach_speed_row_smoke.py
```

Read:

```text
new mode: owner_search_inline_background_proxy
purpose: preserve direct inline action while draining owner maintenance on a background thread
speed claim: none yet; local proof only
next row: same-work H100 against OPT-104 and r32/r33/r34
decision: if this is not a material win, stop owner-worker polish and move to fixed-buffer slab bypass/coarse batched search
```

Validation:

```text
uv run ruff check src/curvyzero/training/compact_owner_search_service.py scripts/build_compact_coach_speed_row_smoke.py tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py
uv run pytest tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py -k "inline_background or lazy_inline_background or priority_loop_serves_action or speed_row_smoke_backend_factory_selects_floor_decomposition_services or threaded_owner_search_direct_root_proof_requires_background_overlap or inline_background_owner_search_direct_root_proof_requires_background_overlap" -q
```

## OPT-132 r35-r38 Inline-Background Owner H100

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-inlinebackground-coalesced-directautoreset-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r35-20260605/
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-inlinebackground-coalesced-directautoreset-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r36-20260605/
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-inlinebackground-coalesced-directautoreset-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r37-repeat-20260605/
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-inlinebackground-coalesced-directautoreset-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m1448-w360-r38-long-20260605/
```

Read:

```text
r35: rejected proof projection only; remote used inline-background but top summary dropped the mode flag
r36: 14906.20 env/s, implied wall 49.736s, 1.175x OPT-104
r37: 13679.66 env/s, implied wall 54.196s, 1.078x OPT-104
r38 long: 12361.15 env/s, implied wall 119.953s, 0.974x OPT-104
threaded r34 long comparison: 13145.34 env/s, implied wall 112.797s
proof: inline-background flag true after projection fix; actions while maintenance pending r36/r37/r38 = 563/501/1277; FIFO blocked 0; tensor fallback none; sample/target 512/512; maintenance clean
```

Decision:

```text
Inline-background answered the variance question but did not create stable headroom.
Do not launch another owner-worker variant as the next P0.
Next work should remove parent/slab hot data movement with fixed buffers or coarse batched search/replay.
```

## OPT-132AZ Local Remaining Builder CPU Diagnostic

Artifacts:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/build_compact_coach_speed_row_smoke.py
scripts/run_compact_coach_speed_row_modal_smoke.py
src/curvyzero/infra/modal/compact_coach_speed_row.py
scripts/compare_compact_coach_speed_rows.py
tests/test_source_state_hybrid_observation_profile.py
tests/test_compact_coach_speed_row_smoke.py
tests/test_compact_coach_speed_row_compare.py
artifacts/local/curvytron_compact_coach_speed_row_results/opt132az-local-remaining-builder-cpu-smoke-r2-20260604/row_001_result.json
```

Read:

```text
diagnostic route: nested in-process wall/process/thread CPU-time counters
new prepare split:
  group_loop_prepare_accounted
  group_loop_prepare_residual
  group_loop_prepare_snapshot
  group_loop_prepare_index
  group_loop_prepare_observation
new terminal split:
  terminal_metadata_accounted
  terminal_metadata_residual
  terminal_metadata_final_observation_accounted
  terminal_metadata_final_observation_residual
  terminal_metadata_final_observation_storage
new unroll split:
  unroll_fields_accounted
  unroll_fields_residual
  unroll_builder_select
  unroll_row_index_prepare
local smoke:
  ok true
  wall 0.813s
  prepare CPU 276000 ns
  terminal metadata CPU 64000 ns
  unroll CPU 706000 ns
```

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py src/curvyzero/infra/modal/compact_coach_speed_row.py scripts/compare_compact_coach_speed_rows.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_compare.py:
  passed
uv run pytest tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_compare.py -q:
  190 passed, 2 warnings
```

Decision:

- Local diagnostic-only plumbing is complete and has now been read on H100.
- These fields are not a speed patch.
- No speed claim.

## OPT-132AZ H100 Remaining Builder CPU No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132az-h100-generic-remaining-builder-cpu-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260604/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132az-h100-generic-remaining-builder-cpu-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260604/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132az-h100-generic-remaining-builder-cpu-nosampler-r1-r2-comparison-20260604/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
identity missing fields: []
identity mismatches: []
stable speed claim allowed: false
CUDA-sync diagnostics: false
GPU sampling: false
r1 wall/speed: 217.391s / 5106.07 env steps/sec
r2 wall/speed: 180.892s / 6136.36 env steps/sec
wall spread: 36.500s, 18.329% of median
sample gate: 136.622s / 109.260s
learner-batch build: 85.001s / 65.049s
builder group-loop: 84.177s / 64.278s
trace records: 135 / 135
terminal validate_only/materialized: 399/0 in both rows
```

Builder child process CPU:

```text
group loop: 80.37s / 61.70s
group-loop prepare: 10.44s / 8.33s
prepare snapshot/index/observation/residual:
  1.20s / 0.78s
  1.51s / 1.08s
  5.00s / 4.26s
  2.73s / 2.21s
terminal metadata: 15.58s / 11.44s
terminal metadata accounted/residual: 13.04s / 9.54s and 2.54s / 1.90s
terminal final observation: 7.38s / 5.46s
unroll fields: 42.13s / 32.24s
unroll accounted/residual: 30.29s / 24.21s and 11.84s / 8.03s
unroll stack/mask-build/terminal-value:
  6.60s / 6.11s
  9.31s / 7.49s
  8.41s / 6.46s
```

Comparison:

```text
runtime-step sum explained: 99.999% of wall delta
sample gate explained: about 74.96% of wall delta
learner-batch build explained: about 54.66% of wall delta
builder group-loop explained: about 54.52% of wall delta
largest deltas:
  group-loop process CPU: -18.67s r1 to r2
  unroll-fields process CPU: -9.89s
  terminal metadata process CPU: -4.14s
  unroll residual process CPU: -3.81s
```

Decision:

- Diagnostic only; no speed claim.
- The same-work instability is still broad user-CPU builder work.
- Stop pure attribution. Next implementation should remove or reshape the
  learner-batch builder surface with guarded learner-ready resident replay or a
  batch-level vectorized unroll-2 builder, then repeat exact `1084/270`.

## OPT-132AV Local Builder-Child CPU-Time Diagnostic

Artifacts:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/build_compact_coach_speed_row_smoke.py
scripts/run_compact_coach_speed_row_modal_smoke.py
src/curvyzero/infra/modal/compact_coach_speed_row.py
scripts/compare_compact_coach_speed_rows.py
tests/test_source_state_hybrid_observation_profile.py
tests/test_compact_coach_speed_row_smoke.py
tests/test_compact_coach_speed_row_compare.py
```

Read:

```text
diagnostic route: in-process process/thread CPU-time counters
remote perf route: closed by OPT-132AU perf_event ENODEV
builder phases:
  group_loop, group_loop_accounted, group_loop_residual
  terminal_metadata and terminal metadata subchildren
  unroll_terminal_window_hint, unroll_fields
  write_output, order_restore, finalize_outputs, metadata_sync, metadata_build
projected field family:
  compact_rollout_slab_sample_gate_learner_batch_builder_<phase>_<scope>_cpu_time_delta_ns
trace field family:
  builder_<phase>_<scope>_cpu_time_delta_ns
```

Validation:

```text
uv run ruff check ...: passed
focused pytest: 6 passed, 2 warnings
smoke/compare pytest: 78 passed, 2 warnings
```

Decision:

- Historical local diagnostic-only plumbing was ready for the AV exact generic
  `1084/270` H100 no-sampler row.
- That read split the then-current mostly user-CPU-backed sample-gate and
  learner-batch-build swing by builder child phase. Later AZ evidence
  supersedes this as the current decision point.
- No speed claim.

## OPT-132AW Local Final-Observation/Residual Split Diagnostic

Artifacts:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/build_compact_coach_speed_row_smoke.py
scripts/run_compact_coach_speed_row_modal_smoke.py
src/curvyzero/infra/modal/compact_coach_speed_row.py
scripts/compare_compact_coach_speed_rows.py
tests/test_source_state_hybrid_observation_profile.py
tests/test_compact_coach_speed_row_smoke.py
tests/test_compact_coach_speed_row_compare.py
```

Read:

```text
diagnostic route: in-process wall and process/thread CPU-time counters
new group-loop residual split:
  group_loop_prepare
  group_loop_terminal_value_bookkeeping
new terminal final-observation split:
  terminal_metadata_final_observation_presence
  terminal_metadata_final_observation_select_current
  terminal_metadata_final_observation_gather
new branch/storage proof:
  terminal_final_observation_group_count
  terminal_final_observation_index_fast_path_count
  terminal_final_observation_fallback_count
  terminal_final_observation_final_row_count_sum/max
  terminal_final_observation_dense/sparse/missing_storage_count
  terminal_final_observation_sparse_row_count_sum/max
```

Validation:

```text
uv run ruff check ...: passed
uv run pytest tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_compare.py -q:
  189 passed, 2 warnings
```

Decision:

- Local diagnostic-only plumbing was ready for exact generic `1084/270` H100
  and has now been read in r1/r2 below.
- No speed claim.

## OPT-132AW H100 Final-Observation/Residual Split No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132aw-h100-generic-finalobs-residual-split-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260604/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132aw-h100-generic-finalobs-residual-split-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260604/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132aw-h100-generic-finalobs-residual-split-nosampler-r1-r2-comparison-20260604/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
report violations: []
CUDA-sync diagnostics: false
GPU sampling: false
r1 function: fc-01KT824Q9507JCJZVPNJYEQ8SS
r2 function: fc-01KT82J00KDMXHGT7QX389A9N9
r1 wall/speed: 250.183s / 4436.81 env steps/sec
r2 wall/speed: 232.521s / 4773.82 env steps/sec
wall spread: 17.662s, 7.318% of median
runtime-step sum: 250.177s / 232.515s
sample gate: 157.724s / 146.441s
learner-batch build: 94.924s / 88.854s
builder group-loop: 94.002s / 87.987s
trace records: 135 / 135
```

Builder child process CPU:

```text
group loop: 89.13s / 83.17s (r1-r2 +5.96s)
group loop accounted: 74.68s / 69.10s (r1-r2 +5.58s)
group loop residual: 14.45s / 14.07s (r1-r2 +0.38s)
group loop prepare: 6.30s / 6.52s (r1-r2 -0.22s)
terminal value bookkeeping: 0.17s / 0.10s (r1-r2 +0.07s)
terminal metadata: 24.01s / 21.75s (r1-r2 +2.26s)
terminal final observation: 14.78s / 13.35s (r1-r2 +1.43s)
final-observation presence: 0.12s / 0.17s (r1-r2 -0.05s)
final-observation select current: 3.74s / 3.77s (r1-r2 -0.03s)
final-observation gather: 6.84s / 5.95s (r1-r2 +0.89s)
unroll fields: 38.18s / 34.98s (r1-r2 +3.20s)
write output: 2.61s / 2.97s (r1-r2 -0.36s)
```

Terminal final-observation proof:

```text
group count: 399 / 399
index fast path: 0 / 0
fallback: 399 / 399
final-row count sum/max: 512/4 in both rows
dense/sparse/missing storage: 0/399/0 in both rows
sparse-row count sum/max: 3102/16 in both rows
```

Decision:

- Exact work still does not repeat, so there is no speed claim.
- AW answers the branch-mix question: terminal final-observation work took the
  same fallback/sparse path with the same row counts in both repeats.
- The new accounting shrinks the unexplained group-loop residual movement
  (`14.45s / 14.07s` process CPU). The larger moving named buckets are
  unroll-fields CPU (`+3.20s` r1-r2), terminal metadata (`+2.26s`), and
  terminal final-observation (`+1.43s`) with gather (`+0.89s`).
- Historical next target was identical fallback final-observation gather work.
  Superseded by AX/AZ and the architecture reset: final-observation
  materialization is no longer the live blocker; learner-ready replay is one
  candidate lane inside the dataflow/toy-ceiling decision.

## OPT-132AX Local Validate-Only Final-Observation Cleanup

Artifacts:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/build_compact_coach_speed_row_smoke.py
scripts/run_compact_coach_speed_row_modal_smoke.py
src/curvyzero/infra/modal/compact_coach_speed_row.py
scripts/compare_compact_coach_speed_rows.py
tests/test_source_state_hybrid_observation_profile.py
tests/test_compact_coach_speed_row_smoke.py
tests/test_compact_coach_speed_row_compare.py
```

Read:

```text
local cleanup:
  grouped learner terminal-final-observation fallback now validates coverage
  instead of materializing a final-next-observation tensor whose return is
  discarded.
unchanged:
  sample paths still call _resident_final_next_observation_for_rows because
  they use the returned tensor to write next_observation.
new timing:
  terminal_metadata_final_observation_validate wall/per-call/process/thread CPU
new proof:
  terminal_final_observation_validate_only_count
  terminal_final_observation_materialized_count
comparator:
  terminal-final-observation proof counters are optional identity fields.
  Legacy rows may omit them; rows that emit them must match exactly.
```

Validation:

```text
uv run ruff check ...: passed
uv run pytest tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_compare.py -q:
  190 passed, 2 warnings
```

Decision:

- Local proof is ready for exact generic no-sampler `1084/270` H100.
- H100 r1/r2 has now been read below.
- No speed claim.

## OPT-132AX H100 Validate-Only Final-Observation No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ax-h100-generic-finalobs-validate-only-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260604/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ax-h100-generic-finalobs-validate-only-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260604/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ax-h100-generic-finalobs-validate-only-nosampler-r1-r2-comparison-20260604/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
optional terminal-final-observation proof missing fields: 0
report violations: []
CUDA-sync diagnostics: false
GPU sampling: false
r1 function: fc-01KT844HJ4XKMRJ75BQYNBJWTC
r2 function: fc-01KT84CPC5E473S2ACHGTJKDST
r1 wall/speed: 168.280s / 6596.25 env steps/sec
r2 wall/speed: 262.830s / 4223.32 env steps/sec
wall spread: 94.550s, 43.864% of median
sample gate: 98.894s / 173.517s
learner-batch build: 56.376s / 102.112s
builder group-loop: 55.699s / 101.183s
trace records: 135 / 135
```

Terminal final-observation proof:

```text
group/fallback/validate-only count: 399/399/399 in both rows
materialized count: 0 / 0
index fast path: 0 / 0
final-row count sum/max: 512/4 in both rows
dense/sparse/missing storage: 0/399/0 in both rows
sparse-row count sum/max: 3102/16 in both rows
select-current wall/CPU: 0 / 0 in both rows
gather wall/CPU: 0 / 0 in both rows
validate wall: 4.312s / 7.057s
validate process CPU: 3.33s / 5.76s
```

Builder process CPU:

```text
group loop: 53.21s / 96.04s (r2-r1 +42.83s)
group loop accounted: 41.83s / 78.06s (r2-r1 +36.23s)
group loop residual: 11.38s / 17.98s (r2-r1 +6.60s)
terminal metadata: 9.20s / 18.95s (r2-r1 +9.75s)
terminal final observation: 4.92s / 8.84s (r2-r1 +3.92s)
unroll fields: 22.70s / 42.99s (r2-r1 +20.29s)
```

Decision:

- The AX cleanup did what it was supposed to do: it removed grouped learner
  select-current/gather materialization and proved `validate_only_count ==
  fallback_count` with `materialized_count == 0`.
- Exact same-work timing stability got worse, not better. There is no speed
  claim.
- The remaining moving CPU is not final-observation materialization. The next
  target is unroll-fields CPU and broader builder group-loop CPU under exact
  identity.

## OPT-132AY Local Unroll Sub-CPU Diagnostic

Artifacts:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
scripts/build_compact_coach_speed_row_smoke.py
scripts/run_compact_coach_speed_row_modal_smoke.py
src/curvyzero/infra/modal/compact_coach_speed_row.py
scripts/compare_compact_coach_speed_rows.py
tests/test_source_state_hybrid_observation_profile.py
tests/test_compact_coach_speed_row_smoke.py
tests/test_compact_coach_speed_row_compare.py
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ay-local-unroll-subcpu-profile-no-death-smoke-20260604/row_001_result.json
```

Read:

```text
diagnostic route: nested in-process process/thread CPU-time counters
new child CPU names:
  unroll_identity
  unroll_stack_fields
  unroll_mask_build
  unroll_terminal_value
  unroll_mask_apply
  unroll_action_stack
accounting caveat:
  these are nested under unroll_fields and are not added to
  group_loop_accounted, so group-loop CPU is not double counted
local smoke:
  profile_no_death tiny fused unroll-2 row completed ok
  unroll_fields process CPU: 698000 ns
  identity/stack/mask-build/terminal-value/mask-apply/action-stack process CPU:
    15000 / 200000 / 159000 / 196000 / 50000 / 24000 ns
```

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py src/curvyzero/infra/modal/compact_coach_speed_row.py scripts/compare_compact_coach_speed_rows.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_compare.py:
  passed
uv run pytest tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_compare.py -q:
  190 passed, 2 warnings
```

Decision:

- Local diagnostic-only plumbing is ready and has now been read on H100 below.
- No speed claim.

## OPT-132AY H100 Unroll Sub-CPU No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ay-h100-generic-unroll-subcpu-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260604/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ay-h100-generic-unroll-subcpu-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260604/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ay-h100-generic-unroll-subcpu-nosampler-r1-r2-comparison-20260604/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
identity missing fields: []
identity mismatches: []
stable speed claim allowed: false
CUDA-sync diagnostics: false
GPU sampling: false
r1 function: fc-01KT85M9MPCV28WW0WVVRF2CQN
r2 function: fc-01KT85YVXN28K1DJJEXY55ZHQS
r1 wall/speed: 237.752s / 4668.80 env steps/sec
r2 wall/speed: 183.386s / 6052.91 env steps/sec
wall spread: 54.366s, 25.819% of median
runtime-step sum explains wall delta: 99.997%
sample gate: 148.271s / 110.841s
learner-batch build: 91.035s / 65.376s
builder group-loop: 89.834s / 64.694s
trace records: 135 / 135
terminal final-observation validate-only/materialized: 399/0 in both rows
select-current/gather wall/CPU: 0 / 0 in both rows
```

Builder process CPU:

```text
group loop: 85.30s / 62.04s (r1-r2 +23.26s)
group loop accounted: 76.22s / 55.78s (r1-r2 +20.44s)
group loop residual: 9.08s / 6.26s (r1-r2 +2.82s)
group loop prepare: 8.72s / 5.40s (r1-r2 +3.32s)
terminal metadata: 17.77s / 12.02s (r1-r2 +5.75s)
unroll fields: 43.86s / 34.14s (r1-r2 +9.72s)
unroll sub-CPU sum: 31.96s / 25.14s (r1-r2 +6.82s)
unroll residual under parent: 11.90s / 9.00s (r1-r2 +2.90s)
```

Unroll process CPU split:

```text
identity: 0.39s / 0.23s (r1-r2 +0.16s)
stack fields: 7.76s / 6.29s (r1-r2 +1.47s)
mask build: 10.08s / 8.47s (r1-r2 +1.61s)
terminal value: 8.51s / 6.55s (r1-r2 +1.96s)
mask apply: 3.25s / 2.29s (r1-r2 +0.96s)
action stack: 1.97s / 1.31s (r1-r2 +0.66s)
```

Decision:

- Exact work still does not repeat, so there is no speed claim.
- AY answers the AX follow-up: the moving unroll CPU is broad across several
  named subchildren rather than one isolated operation.
- The next target is the remaining unroll residual plus group-loop prepare,
  terminal metadata, and broad sample-gate CPU under exact identity.

## OPT-132AV H100 Builder-Child CPU-Time No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132av-h100-generic-builder-child-cputime-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132av-h100-generic-builder-child-cputime-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132av-h100-generic-builder-child-cputime-nosampler-r1-r2-comparison-20260603/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
report violations: []
CUDA-sync diagnostics: false
GPU sampling: false
r1 function: fc-01KT7ZWHYPG3342MWGDP7TFSHB
r2 function: fc-01KT809AAW2TY3DAEHPR8TV1B2
r1 wall/speed: 190.101s / 5839.10 env steps/sec
r2 wall/speed: 220.991s / 5022.90 env steps/sec
wall spread: 30.890s, 15.03% of median
runtime-step sum: 190.095s / 220.985s
sample gate: 114.928s / 135.240s
learner-batch build: 66.229s / 82.661s
builder group-loop: 65.505s / 81.792s
trace records: 135 / 135
```

Builder child process CPU:

```text
group loop: 62.48s / 77.36s (delta +14.88s)
group loop accounted: 47.35s / 57.95s (delta +10.60s)
group loop residual: 15.13s / 19.41s (delta +4.28s)
terminal metadata: 14.74s / 21.11s (delta +6.37s)
terminal final observation: 7.33s / 10.00s (delta +2.67s)
unroll terminal-window hint: 2.85s / 3.54s (delta +0.69s)
unroll fields: 27.70s / 30.46s (delta +2.76s)
write output: 2.06s / 2.84s (delta +0.78s)
```

Measured thirds, group-loop process CPU:

```text
early: 11.72s / 14.87s (delta +3.15s)
mid:   23.22s / 29.51s (delta +6.29s)
late:  27.54s / 32.98s (delta +5.44s)
```

Decision:

- Exact work still does not repeat, so there is no speed claim.
- The builder group-loop wall delta `16.287s` is mostly current-process CPU:
  process CPU delta `14.88s` is about `91%` of the group-loop wall delta.
- The largest moving child CPU bucket is terminal metadata (`+6.37s`), with
  terminal final-observation work contributing `+2.67s`; residual group-loop
  CPU also moves (`+4.28s`). Unroll fields move less (`+2.76s`).
- Historical next diagnostic target was terminal/final-observation and residual
  CPU movement. Superseded by AX/AZ; the current action is the guarded
  learner-ready resident replay / unroll-2 cache path.

## OPT-132AN H100 Active-Cadence No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132an-h100-generic-activecadence-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132an-h100-generic-activecadence-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132an-h100-generic-activecadence-nosampler-r1-r2-comparison-20260603/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
accepted-fast-path violations: []
runtime-step diagnostic violations: []
CUDA-sync diagnostics: false
GPU sampling: false
r1 function: fc-01KT7MD03R5852Y5BK8MW30W1Z
r2 function: fc-01KT7MRAV915YB7FZC4ERK34GY
r1 wall/speed: 206.746s / 5368.97 env steps/sec
r2 wall/speed: 166.095s / 6683.01 env steps/sec
wall spread: 40.651s, 21.806% of median
runtime-step sum: 206.741s / 166.090s
sample gate: 125.169s / 93.384s
learner-batch build: 68.851s / 49.643s
builder group-loop: 68.101s / 48.923s
active sample-gate p50 early/mid/late:
  r1 0.425s / 0.981s / 1.342s
  r2 0.327s / 0.699s / 1.025s
active sample-gate late p95:
  r1 1.714s
  r2 1.392s
```

Decision:

- Exact work still does not repeat.
- The slowdown is broad across active sample-gate calls and elevated across
  r1 versus r2, not just a few late spikes.
- Wall-swing attribution assigns `99.999%` of the slow-fast delta to
  runtime-step sum, `78.19%` to sample gate, `47.25%` to learner-batch build,
  and `47.18%` to builder group-loop.
- No speed claim. The next diagnostic is per-sample-gate replay-state trace.

## OPT-132AO H100 Replay-Trace No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ao-h100-generic-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ao-h100-generic-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ao-h100-generic-trace-nosampler-r1-r2-comparison-20260603/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
accepted-fast-path violations: []
runtime-step diagnostic violations: []
CUDA-sync diagnostics: false
GPU sampling: false
r1 function: fc-01KT7P4RMN2JK0CB98FZMMJTS6
r2 function: fc-01KT7PF9JAQDDB6RVWQK81AHC0
r1 wall/speed: 221.118s / 5020.01 env steps/sec
r2 wall/speed: 254.261s / 4365.66 env steps/sec
wall spread: 33.142s, 13.944% of median
runtime-step sum: 221.111s / 254.253s
sample gate: 132.338s / 163.614s
learner-batch build: 72.215s / 91.981s
builder group-loop: 71.350s / 91.030s
trace records: 135 / 135
trace state mismatch count: 0 / 135
slowest trace call:
  r1 call 120, measured iteration 960, sample_gate 3.155s
  r2 call 131, measured iteration 1048, sample_gate 2.164s
```

Trace comparison:

```text
state fields matched at every call index:
  measured iteration and measured third
  sample seed, sample rows, sampled pairs, terminal rows
  source/sample/action/order/checksum fields
  stored/raw/eligible/excluded/evicted replay counts
  replay ring pair capacity
sample-gate delta by measured third:
  early +6.224s
  mid +12.215s
  late +12.836s
```

Decision:

- Exact work still does not repeat.
- The exposed replay/sample state is identical, so the current swing is not
  replay-state drift or sample-shape drift at the AO trace level.
- Wall-swing attribution assigns `99.999%` of the slow-fast delta to
  runtime-step sum, `94.37%` to sample gate, `59.64%` to learner-batch build,
  and `59.38%` to builder group-loop.
- No speed claim. The next diagnostic needs to look below the trace state:
  allocator/cache/runtime timing or lower-level builder timing on identical
  replay/sample state.

## OPT-132AR H100 CPU-Time Trace No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ar-h100-generic-cputime-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ar-h100-generic-cputime-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ar-h100-generic-cputime-trace-nosampler-r1-r2-comparison-20260603/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
accepted-fast-path violations: []
runtime-step diagnostic violations: []
CUDA-sync diagnostics: false
GPU sampling: false
trace records: 135 / 135
r1 wall/speed: 189.915s / 5844.80 env steps/sec
r2 wall/speed: 217.364s / 5106.72 env steps/sec
wall spread: 27.449s, 13.479% of median
runtime-step sum spread: 27.449s, 13.480% of median
sample gate: 111.765s / 130.614s
learner-batch build: 60.527s / 71.857s
builder group-loop: 59.815s / 71.015s
```

CPU-time comparison:

```text
sample-gate process CPU sum: 104.77s / 122.10s
sample-gate thread CPU sum: 104.00s / 121.18s
sample-gate wall delta: 18.848s
sample-gate process/thread CPU delta: 17.33s / 17.18s
sample-gate CPU delta share of wall delta: 91.9% / 91.1%

learner-build process CPU sum: 56.85s / 67.29s
learner-build thread CPU sum: 56.44s / 66.83s
learner-build wall delta: 11.331s
learner-build process/thread CPU delta: 10.44s / 10.39s
learner-build CPU delta share of wall delta: 92.1% / 91.7%
```

Trace comparison:

```text
replay/sample identity mismatch: none across all 135 calls
CUDA allocator/memory mismatch: none across all 135 calls
sample-gate CPU-time deltas: quantized at 10ms in these reports
```

Decision:

- Exact work still does not repeat.
- The remaining sample-gate/build swing is CPU-time-backed rather than mostly
  off-CPU wait.
- AP already ruled out exposed replay/sample-state drift and simple CUDA
  allocator state; AQ ruled out actual GC collection counts as the primary
  cause; AR rules out mostly waiting.
- No speed claim. The next question is why identical trace state consumes
  different CPU time: lower-level CPU work, CPU frequency, cache/memory
  locality, or runtime/library behavior.

## OPT-132AT Local CPU Perf-Stat Diagnostic

Local change:

```text
purpose: explain AS's mostly-user-CPU sample-gate/build swing with lower-level
  CPU counters
speed claim: none
accepted-speed denominator change: none
new flag: --compact-profile-cpu-perf-stat-diagnostics
```

What changed:

- The Modal H100 producer can wrap the remote speed-row script with
  `perf stat -x,`.
- Requested events are task-clock, cycles, ref-cycles, instructions, branches,
  branch misses, cache references/misses, LLC loads/misses, dTLB loads/misses,
  page faults, context switches, and CPU migrations.
- Perf stdout/stderr are written beside the row as
  `cpu_perf_stat_stdout.txt` and `cpu_perf_stat_stderr.txt`.
- Available counters are parsed into `compact_profile_cpu_perf_stat_*` fields,
  including task-clock seconds, instructions-per-cycle, and cache-miss rate.
- The fields project into the row summary/compact payload, Modal report, and
  same-work comparison identity/timing surfaces.
- If `perf` is missing or fails before a row exists, the wrapper returns a
  structured diagnostic failure rather than silently producing a speed row.

Validation:

```text
ruff: passed
targeted pytest: 4 passed, 2 warnings
broader smoke/compare slice: 20 passed, 2 warnings
```

Decision:

- Local diagnostic only; no speed claim.
- The first evidence was a structured perf-unavailable H100 failure; see below.
- Useful outcomes: more retired instructions, worse IPC/cache/TLB/branch
  behavior, CPU migrations, lower effective CPU frequency, or proof that Modal
  cannot provide hardware counters for this diagnostic.

## OPT-132AT H100 Perf Missing Failure

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132at-h100-generic-perfstat-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132at-h100-generic-perfstat-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/remote_bundle.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132at-h100-generic-perfstat-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/launch.json
```

Read:

```text
function_call_id: fc-01KT7YGKDM2Y35NCG6C204TYMY
ok: false
status: failed
problem: perf stat diagnostic requested but perf was not found
compact_profile_cpu_perf_stat_diagnostics: true
compact_profile_cpu_perf_stat_available: false
compact_profile_cpu_perf_stat_returncode: 127
producer_result_present: false
producer_report_present: false
```

Decision:

- This is not a speed row and carries no timing evidence.
- It proves only that the current speed-row image lacks `perf`.
- Next step is OPT-132AU: install `linux-perf` in `speed_row_image` and rerun
  once. If the kernel or event set is denied, preserve that structured failure.

## OPT-132AU Local linux-perf Image Patch

Local change:

```text
purpose: make the OPT-132AT perf-stat diagnostic reachable on Modal
speed claim: none
accepted-speed denominator change: none
image scope: compact speed-row image only
```

What changed:

- `speed_row_image` in `src/curvyzero/infra/modal/compact_coach_speed_row.py`
  now calls `.apt_install("linux-perf")` before setting env/local files.
- The shared `gpu_lightzero_image` remains unchanged; this is scoped to the
  compact speed-row producer.

Validation:

```text
ruff: passed
focused pytest: 2 passed, 2 warnings
```

Decision:

- Local diagnostic hygiene only; no speed claim.
- The AU H100 result below proved perf events are unavailable in this Modal
  container.

## OPT-132AU H100 perf_event Unavailable Failure

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132au-h100-generic-perfstat-image-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132au-h100-generic-perfstat-image-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/remote_bundle.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132au-h100-generic-perfstat-image-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/launch.json
```

Read:

```text
function_call_id: fc-01KT7YWEDCC60CMGTFY6388QBE
ok: false
problem: perf stat diagnostic failed before speed-row result
perf binary: /usr/bin/perf
compact_profile_cpu_perf_stat_available: true
compact_profile_cpu_perf_stat_returncode: 255
parse_line_count: 0
parsed_event_count: 0
stderr: sys_perf_event_open() returned with 19 (No such device) for task-clock
producer_result_present: false
producer_report_present: false
```

Decision:

- This is not a speed row and carries no timing evidence.
- The external perf-counter route is unavailable in the current Modal
  container even after installing `linux-perf`.
- Do not retry perf unchanged. The next diagnostic should be in-process,
  splitting AS's user-CPU-backed sample-gate/learner-batch-build work by child
  phase or another runtime-visible bucket.

## OPT-132AS H100 Resource-Usage Trace No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132as-h100-generic-rusage-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132as-h100-generic-rusage-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132as-h100-generic-rusage-trace-nosampler-r1-r2-comparison-20260603/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
accepted-fast-path violations: []
runtime-step diagnostic violations: []
CUDA-sync diagnostics: false
GPU sampling: false
trace records: 135 / 135
r1 function: fc-01KT7WC6TNV26CF5AN0JSB4A6A
r2 function: fc-01KT7WQZ5MD5JCNRN9NS1DCWA5
r1 wall/speed: 211.257s / 5254.34 env steps/sec
r2 wall/speed: 182.483s / 6082.84 env steps/sec
wall spread: 28.774s, 14.616% of median
runtime-step sum spread: 28.773s, 14.616% of median
sample gate: 126.934s / 107.934s
learner-batch build: 70.377s / 58.933s
builder group-loop: 69.602s / 58.206s
```

Resource comparison:

```text
sample-gate process user/system CPU sum:
  r1 117.55s / 0.95s
  r2 101.38s / 0.46s
sample-gate thread user/system CPU sum:
  r1 116.93s / 0.93s
  r2 101.10s / 0.35s
sample-gate wall delta: 18.999s
sample-gate process user/system CPU delta: 16.17s / 0.49s
sample-gate thread user/system CPU delta: 15.83s / 0.58s

learner-build process user/system CPU sum:
  r1 64.89s / 0.94s
  r2 55.58s / 0.39s
learner-build thread user/system CPU sum:
  r1 64.52s / 0.93s
  r2 55.44s / 0.34s
learner-build wall delta: 11.444s
learner-build process user/system CPU delta: 9.31s / 0.55s
learner-build thread user/system CPU delta: 9.08s / 0.59s

process page-fault deltas: 0 minor, 0 major
process context-switch deltas: 0 voluntary, 0 involuntary
```

Decision:

- Exact work still does not repeat.
- AS rules out system CPU, page faults, and context switches as the main
  explanation of the AR CPU-time-backed swing.
- The remaining sample-gate/build movement is mostly user CPU on the same
  exposed replay/sample, allocator, and GC state.
- No speed claim. The next question is why identical trace state consumes
  different user CPU: CPU frequency, cache/memory locality, runtime/library
  behavior, or a still-hidden Python/Torch work bucket.

## OPT-132AS Local Resource-Usage Trace

Local change:

```text
purpose: split AR's CPU-time-backed sample-gate/build swing below aggregate CPU time
speed claim: none
accepted-speed denominator change: none
new trace fields: process/thread getrusage before/after/delta counters
```

What changed:

- `compact_rollout_slab_sample_gate_call_trace_records` now carries
  process/thread resource-usage counters before/after/delta when runtime
  diagnostics are enabled.
- The fields are emitted for both the full sample gate and the
  learner-batch-build slice.
- Counters include user CPU time, system CPU time, minor/major page faults, and
  voluntary/involuntary context switches.
- Thread counters populate when the platform exposes `RUSAGE_THREAD`; otherwise
  they remain zero.
- This is a diagnostic addition to AR's runtime trace. It does not add CUDA
  synchronization and does not change the accepted speed denominator.

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py
uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_compact_muzero_learner_gate_uses_explicit_next_targets tests/test_source_state_hybrid_observation_profile.py::test_runtime_step_timing_diagnostics_can_run_without_cuda_sync tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_projects_cuda_sync_diagnostics tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_launcher_threads_runtime_step_timing_diagnostics tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_report_projects_sample_learner_child_timers tests/test_compact_coach_speed_row_compare.py -q
result: ruff passed; 17 passed, 2 warnings
```

Decision:

- Local diagnostic only.
- Next evidence is exact generic no-sampler `1084/270` H100 AS r1/r2.
- Compare AS user/system CPU, page faults, and context switches against AR's
  CPU-time deltas and builder child wall fields before any new speed patch.

## OPT-132AR Local CPU-Time Trace

Local change:

```text
purpose: separate real CPU work from wall-time scheduling/wait after AQ ruled out GC collections
speed claim: none
accepted-speed denominator change: none
new trace fields: process/thread CPU-time before/after/delta counters
```

What changed:

- `compact_rollout_slab_sample_gate_call_trace_records` now carries process
  and current-thread CPU-time before/after/delta fields when runtime
  diagnostics are enabled.
- The fields are emitted for both the full sample gate and the
  learner-batch-build slice.
- This is a diagnostic addition to AQ's runtime trace. It does not add CUDA
  synchronization and does not change the accepted speed denominator.

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py
uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_compact_muzero_learner_gate_uses_explicit_next_targets tests/test_source_state_hybrid_observation_profile.py::test_runtime_step_timing_diagnostics_can_run_without_cuda_sync tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_projects_cuda_sync_diagnostics tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_launcher_threads_runtime_step_timing_diagnostics tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_report_projects_sample_learner_child_timers tests/test_compact_coach_speed_row_compare.py -q
result: ruff passed; 17 passed, 2 warnings
```

Decision:

- Local diagnostic only; superseded by the AR H100 r1/r2 read above.

## OPT-132AQ H100 GC-Stats Trace No-Sampler R1/R2

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132aq-h100-generic-gcstats-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132aq-h100-generic-gcstats-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132aq-h100-generic-gcstats-trace-nosampler-r1-r2-comparison-20260603/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
accepted-fast-path violations: []
runtime-step diagnostic violations: []
CUDA-sync diagnostics: false
GPU sampling: false
trace records: 135 / 135
r1 wall/speed: 218.320s / 5084.35 env steps/sec
r2 wall/speed: 163.265s / 6798.87 env steps/sec
wall spread: 55.055s, 28.856% of median
runtime-step sum spread: 55.055s, 28.857% of median
sample gate: 131.651s / 95.558s
learner-batch build: 72.267s / 51.322s
builder group-loop: 71.432s / 50.716s
```

GC comparison:

```text
gen0/gen1/gen2 collection totals:
  r1 7232 / 659 / 28
  r2 7234 / 655 / 28
gen0/gen1/gen2 collected-object totals:
  r1 240157 / 34125 / 1417
  r2 237744 / 36777 / 1466
same-GC-collection-delta calls: 100
sample-gate delta on same-GC-delta calls: 22.6695s
```

Decision:

- Exact work still does not repeat.
- Actual Python GC collection counters do not explain the timing swing.
- AP already ruled out exposed replay/sample-state drift and simple CUDA
  allocator state; AQ rules out actual GC collection counts as the primary
  cause.
- No speed claim. The next diagnostic is OPT-132AR process/thread CPU-time
  attribution.

## OPT-132AQ Local GC-Stats Trace

Local change:

```text
purpose: identify actual Python GC collections after AP showed allocator-exact timing spread
speed claim: none
accepted-speed denominator change: none
new trace fields: gc.get_stats() collections/collected/uncollectable counters
```

What changed:

- `compact_rollout_slab_sample_gate_call_trace_records` now carries
  per-generation Python GC stats before/after/delta when runtime diagnostics
  are enabled: collections, collected objects, and uncollectable objects for
  gen0/gen1/gen2.
- This is a diagnostic addition to AP's runtime trace. It does not add CUDA
  synchronization and does not change the accepted speed denominator.

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py
uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_compact_muzero_learner_gate_uses_explicit_next_targets tests/test_source_state_hybrid_observation_profile.py::test_runtime_step_timing_diagnostics_can_run_without_cuda_sync tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_projects_cuda_sync_diagnostics tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_launcher_threads_runtime_step_timing_diagnostics tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_report_projects_sample_learner_child_timers tests/test_compact_coach_speed_row_compare.py -q
result: ruff passed; 17 passed, 2 warnings
```

Decision:

- Local diagnostic only; superseded by the AQ H100 r1/r2 read above and the AR
  local CPU-time extension.

## OPT-132AP H100 Allocator/Runtime Trace No-Sampler R1/R2/R3

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ap-h100-generic-allocator-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ap-h100-generic-allocator-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ap-h100-generic-allocator-trace-nosampler-b1024a1-normal-unroll2-m1084-w270-r3-20260603/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132ap-h100-generic-allocator-trace-nosampler-r1-r2-r3-comparison-20260603/comparison.json
```

Read:

```text
window: stability_1084_270
identity: exact
accepted-fast-path violations: []
runtime-step diagnostic violations: []
CUDA-sync diagnostics: false
GPU sampling: false
trace records: 135 / 135 / 135
r1 wall/speed: 219.446s / 5058.27 env steps/sec
r2 wall/speed: 205.371s / 5404.94 env steps/sec
r3 wall/speed: 153.752s / 7219.51 env steps/sec
wall spread: 65.694s, 31.99% of median
runtime-step sum spread: 65.691s, 31.99% of median
sample gate: 130.199s / 120.714s / 86.639s
learner-batch build: 70.732s / 65.983s / 46.881s
builder group-loop: 69.882s / 65.174s / 46.350s
```

Trace comparison:

```text
replay/sample identity mismatch: none across all 135 calls
CUDA allocator/memory mismatch: none across all 135 calls
CUDA retry/OOM mismatch: none across all 135 calls
learner-batch-build CUDA memory mismatch: none across all 135 calls
Python GC count fields: varied
process max-RSS fields: varied
```

Decision:

- Exact work still does not repeat.
- AP rules out simple CUDA allocator state and exposed replay/sample state as
  the cause of the r1/r2/r3 wall swing.
- The biggest slow-minus-fast timing deltas are sample gate `43.560s`,
  learner-batch build `23.851s`, builder group-loop `23.532s`, unroll fields
  `14.645s`, terminal metadata `4.968s`, and sample-gate residual `12.176s`.
- No speed claim. The next diagnostic is actual Python GC collection counters
  via OPT-132AQ; if those do not explain the swing, suspect lower-level
  host/runtime scheduling.

## OPT-132AP Local Allocator/Runtime Trace

Local change:

```text
purpose: read below OPT-132AO's identical replay/sample trace state
speed claim: none
accepted-speed denominator change: none
new trace fields: allocator/runtime snapshots and deeper builder child timings
```

What changed:

- `compact_rollout_slab_sample_gate_call_trace_records` now carries additional
  per-call fields when runtime diagnostics request them: CUDA memory
  allocated/reserved/peak counters, allocator retry/OOM counters,
  learner-batch-build memory deltas, Python GC generation counts, process
  max-RSS raw state, and sample-gate CUDA-sync timing.
- The trace also includes already-measured deeper builder child timings:
  terminal metadata mask/tensor fallback/validate/final-observation, unroll
  terminal-window hint/identity/stack-fields/mask-build/terminal-value/
  mask-apply/action-stack, order-restore, finalize-outputs, metadata-sync, and
  metadata-build.
- The probe avoids new CUDA synchronization and remains diagnostic-only.

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py src/curvyzero/infra/modal/compact_coach_speed_row.py scripts/compare_compact_coach_speed_rows.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_compare.py
uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_compact_muzero_learner_gate_uses_explicit_next_targets tests/test_source_state_hybrid_observation_profile.py::test_runtime_step_timing_diagnostics_can_run_without_cuda_sync tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_projects_cuda_sync_diagnostics tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_launcher_threads_runtime_step_timing_diagnostics tests/test_compact_coach_speed_row_smoke.py::test_modal_speed_row_report_projects_sample_learner_child_timers tests/test_compact_coach_speed_row_compare.py -q
result: ruff passed; 17 passed, 2 warnings
```

Decision:

- Local diagnostic only; superseded by the AP H100 r1/r2/r3 read above and the
  AQ local GC-stats extension.

## OPT-132AO Local Sample-Gate Call Trace

Local change:

```text
purpose: correlate each active sample-gate call with replay/sample state
speed claim: none
accepted-speed denominator change: none
new report field: compact_rollout_slab_sample_gate_call_trace_records
```

What changed:

- Source profile emits a bounded list of per-call records with call index,
  iteration, measured iteration, measured third, sample seed/rows/checksums,
  stored/eligible/excluded/evicted replay counts, replay capacity, and per-call
  sample/builder/residual timing.
- Speed-row summary/compact payload and local/remote Modal reports preserve the
  records.

Validation:

```text
ruff: passed for touched source/profile, speed-row, Modal, comparator, tests
focused pytest: source-profile runtime-step diagnostic and speed-row projection tests passed
launcher/comparator slice: 13 passed
```

Decision:

- This is measurement tooling only.
- Superseded by the OPT-132AO H100 r1/r2 trace read above.

## Accepted Baseline: OPT-104

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt104-h100-normal-death-learner-validation-only-compacttorch-r1-20260531/row_001_result.json`

Read:

```text
speed: 12689.38 env steps/sec
wall:  14.5255s
sample gate: 3.3327s
learner gate: 1.2605s
resident host fallback: 0
```

Decision:

- OPT-104 is still the accepted speed baseline.
- Do not replace it with a single lucky row.

## OPT-132-I: Accepted-Preset Repeat Artifact, Useful Fast Row

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132i-h100-accepted-fast-path-repeat-proofguard-samework-b1024a1-normal-unroll2-speed-20260602-r1/row_001_result.json`

Read:

```text
speed: 13308.59 env steps/sec
wall:  13.8497s
actor: 4.613s
observation: 2.064s
observation other: 1.636s
sample gate: 2.604s
slab/search: 2.824s
learner gate: 1.163s
resident stack shift: 0.013s
sample/learner updates: 22/22
policy refresh after learner gate: 6 calls, forced final 1
```

Guard note:

- The artifact validates under the corrected accepted-fast-path guard.
- The original Modal report rejected it because the guard wrongly required a
  signed checksum to be positive. The correct rule is nonzero for checksums and
  positive for counters.

Decision:

- Useful fast row: it beats OPT-104 by about `0.676s` wall.
- Not accepted as the new baseline. OPT-132-J already ran as the corrected
  repeat and was slow, so the active issue is measurement stability, not another
  short same-preset row.

## OPT-132-J: Accepted-Preset Repeat, Clean Slow Row

Artifacts:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132j-h100-accepted-fast-path-repeat-proofguard-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132j-h100-accepted-fast-path-repeat-proofguard-samework-b1024a1-normal-unroll2-speed-20260602-r1/row_001_result.json`

Read:

```text
speed: 10057.98 env steps/sec
wall:  18.3258s
actor: 6.010s
observation: 1.911s
observation other: 1.444s
sample gate: 4.456s
slab/search: 3.325s
learner gate: 1.817s
resident stack shift: 0.018s
sample/learner updates: 22/22
policy refresh after learner gate: 6 calls, forced final 1
```

I -> J swing:

```text
wall:        +4.476s
actor:       +1.397s
sample gate: +1.852s
learner:     +0.654s
observation: -0.154s
```

Decision:

- Clean slow row. The corrected accepted-fast-path guard passed.
- Trajectory and sample-order checksums match OPT-132-I, so this is not hidden
  work drift.
- Current target is narrower: sample-gate learner-batch-build timing on exact
  `1084/270` work. Actor/autoreset and learner backward stay visible, but do
  not become speed targets until long-window timing is stable.

Comparison artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132-ghij-runtime-comparison-20260602/comparison.json`

Read:

```text
stable_speed_claim_allowed: false
clean exact comparison: I -> J
sample gate: +1.852s
actor: +1.397s
actor autoreset: +1.104s
learner batch build: +1.038s
learner: +0.654s
learner backward: +0.445s
sample RNG: +0.416s
```

## Long-Window Stability Diagnostic Support

Launcher support:

```text
--compact-owned-accepted-fast-path-preset
--compact-owned-accepted-fast-path-step-window stability_724_180
--compact-owned-accepted-fast-path-step-window stability_1084_270
```

Decision:

- This keeps the accepted fast-path flags and changes only the step window.
- `stability_724_180`, `stability_1084_270`, and `stability_1444_360` are
  diagnostics.
- Compare long-window rows only against rows with the same long window.
- Do not claim a speed win against OPT-104 from a long-window row alone.

Latest H100 diagnostics:

```text
724/180 bridge:
  run: opt132w-h100-latest-frame-bounded-bridge-stability-b1024a1-normal-unroll2-m724-w180-r1
  wall/speed: 104.186s / 7115.91 env steps/sec
  resident replay snapshot mode: latest_frame_history
  retained resident snapshot bytes: 6243287040
  accepted-fast-path violations: []
  decision: bridge passed, diagnostic only

1084/270 r1:
  run: opt132w-h100-latest-frame-bounded-midwindow-stability-b1024a1-normal-unroll2-m1084-w270-r1
  wall/speed: 244.168s / 4546.12 env steps/sec
  retained resident snapshot bytes: 9346744320
  sample gate: 155.937s
  learner-batch build: 86.207s

1084/270 r2:
  run: opt132w-h100-latest-frame-bounded-midwindow-stability-b1024a1-normal-unroll2-m1084-w270-r2
  wall/speed: 224.761s / 4938.66 env steps/sec
  retained resident snapshot bytes: 9346744320
  sample gate: 135.476s
  learner-batch build: 73.758s
```

Comparison artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132w-h100-latest-frame-bounded-midwindow-r1r2-comparison-20260602/comparison.json`

Decision:

- Memory fit is fixed for `1084/270`; both rows completed on H100 with latest-frame
  resident replay snapshots.
- Timing stability is still not fixed: wall spread is `8.28%`, sample gate
  spread is `14.04%`, learner-batch build spread is `15.57%`.
- Next work is to isolate sample-gate learner-batch-build timing, not to claim
  speed or start a new hardware lane.

## OPT-132W Local Learner-Batch Timing Diagnostic Patch

Local change:

```text
purpose: explain `1084/270` sample-gate learner-batch-build timing instability
speed claim: none
accepted-speed denominator change: none
```

What changed:

- The resident grouped device learner-batch builder now reports sub-timers for
  group loop, terminal metadata, unroll field build, output writes, order
  restore, output finalization, metadata sync/readback, and metadata build.
- The profile now records bounded per-call learner-batch-build stats: count,
  sum, min, max, p50, p95, and slowest-call seed/checksum context.
- Speed-row summary/compact payload and Modal reports project these fields.

Validation:

```text
ruff: passed for touched source/profile, speed-row, Modal, and tests
focused pytest: source-profile explicit-next-target test passed
focused pytest: speed-row fused proof and Modal sample/learner timer tests passed
broader pytest before final stats projection: source profile 110 passed; speed-row smoke 47 passed
```

Decision:

- This is measurement work, not speed evidence.
- Superseded by OPT-132Y r1/r2: the sync fields are visible and comparison now
  tracks them. Next evidence is exact OPT-132Y r3, then a three-row comparison.

## OPT-132Y H100 CUDA Sync Timing Diagnostic R1/R2/R3

Artifacts:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132y-h100-cuda-sync-timing-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1/row_001_result.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132y-h100-cuda-sync-timing-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2/row_001_result.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132y-h100-cuda-sync-timing-diagnostic-b1024a1-normal-unroll2-m1084-w270-r3/row_001_result.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132y-h100-cuda-sync-timing-diagnostic-r1r2r3-comparison-20260602/comparison.json`

Read:

```text
r1 wall/speed: 196.699s / 5643.23 env steps/sec
r2 wall/speed: 194.540s / 5705.85 env steps/sec
r3 wall/speed: 255.499s / 4344.51 env steps/sec
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
stable_speed_claim_allowed: false
wall spread: 60.959s, 30.99% of median
sample gate spread: 36.790s, 30.47% of median
learner-batch build spread: 19.588s, 28.01% of median
actor wall spread: 11.223s, 31.66% of median
observation spread: 8.268s, 62.14% of median
observation-other spread: 8.016s, 75.95% of median
sample rng spread: 0.939s, 24.68% of median
```

CUDA sync read:

```text
sample gate sync count/sec: 12 / 0.030s, 12 / 0.028s, 12 / 0.051s
builder sync count/sec: 11977 / 6.910s, 11977 / 5.520s, 11977 / 8.274s
learner sync count/sec: 28 / 0.845s, 28 / 0.831s, 28 / 0.611s
builder cuda_sync spread: 2.754s, 39.85% of median
```

Local guard/comparator follow-up:

```text
ruff: passed for launcher, Modal producer, comparator, and focused tests
pytest: comparator suite plus sync projection/violation tests passed
```

Decision:

- The r1/r2 pair looked better than OPT-132X for wall/sample/build A/A spread,
  but r3 failed timing stability hard.
- The sync proof is now fail-closed in the launcher and Modal producer. The
  comparator includes sync flags/counts in identity and sync seconds in timing
  ranges.
- Do not patch speed code yet. The slowdown is broad across actor/autoreset,
  observation, sample gate, builder, and learner. Isolate common-host/runtime
  variance before learner-batch builder group-loop optimization.

## OPT-132Y Runtime-Step Envelope Diagnostic Patch

Local change:

```text
purpose: isolate broad same-work runtime variance after OPT-132Y r3
speed claim: none
trigger: --compact-profile-cuda-sync-timing-diagnostics
new fields: compact_profile_runtime_step_*
```

What changed:

- The hybrid profile records measured-step wall timing when CUDA-sync timing
  diagnostics are requested.
- The stats include count, sum, min, max, p50, p95, and the slowest measured
  step's actor, observation, compact slab, sample gate, learner gate,
  policy-refresh, residual, and trajectory checksum context.
- Speed-row summary/compact payloads, Modal reports, and comparison artifacts
  project the flattened fields.
- Comparator identity includes the runtime diagnostic flag and measured-step
  count; timing ranges include the runtime wall/stat fields.

Validation:

```text
ruff: passed for touched source/profile, speed-row, Modal, comparator, and tests
pytest: tests/test_compact_coach_speed_row_compare.py plus focused smoke/profile tests
result: 11 passed, 2 warnings
```

Decision:

- This is diagnostic-only. It does not explain OPT-132Y by itself and does not
  authorize a learner-batch builder speed patch.
- The next exact `1084/270` H100 diagnostic should inspect these fields to tell
  whether the broad slowdown is a uniform measured-step shift or a few
  pathological steps.

## OPT-132Z H100 Runtime-Step Envelope Diagnostic R1/R2

Artifacts:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132z-h100-runtime-step-envelope-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1/row_001_result.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132z-h100-runtime-step-envelope-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2/row_001_result.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132z-h100-runtime-step-envelope-diagnostic-r1r2-comparison-20260603/comparison.json`

Read:

```text
r1 wall/speed: 197.292s / 5626.27 env steps/sec
r2 wall/speed: 273.865s / 4053.14 env steps/sec
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
runtime-step count: 1084 / 1084
wall spread: 76.574s, 32.50% of median
runtime-step sum spread: 76.573s, 32.50% of median
sample gate spread: 64.055s, 42.67% of median
learner-batch build spread: 38.792s, 44.14% of median
actor wall spread: 8.483s, 21.69% of median
actor autoreset spread: 8.118s, 31.17% of median
runtime p50: 0.063s / 0.073s
runtime p95: 1.150s / 1.784s
runtime max: 1.786s / 2.717s
slowest sample gate: 1.636s / 2.521s
```

Learner-batch per-call read:

```text
count: 135 / 135
p50: 0.564s / 0.898s
p95: 0.696s / 1.083s
max: 1.004s / 1.561s
sum: 68.491s / 107.283s
```

Decision:

- The runtime-step envelope worked and reconciled with wall, so the swing is
  inside measured loop iterations rather than final drain or untracked wall.
- The p50 step barely moved while p95/max moved hard. That points to
  sample-gate cadence steps, not uniform slowdown over every actor step.
- Do not patch speed code yet. Add/read sample-gate per-call child distribution
  stats before learner-batch builder optimization.

## OPT-132Z Local Sample-Gate Per-Call Distribution Patch

Local change:

```text
purpose: explain OPT-132Z sample-gate cadence-step instability
speed claim: none
accepted-speed denominator change: none
new fields: compact_rollout_slab_sample_gate_*_per_call_stats
flattened fields: total/candidate/rng/residual count/sum/min/max/p50/p95
```

What changed:

- The source profile now records a bounded record for every sample-gate call.
- It emits per-call stats for sample-gate total, candidate selection, RNG, and
  residual; existing learner-batch-build per-call stats now use the same record
  stream.
- Speed-row summaries/compact payloads and Modal reports carry nested stats and
  flattened distribution fields.
- The comparator includes the flattened fields in timing ranges.

Validation:

```text
ruff: passed for touched source/profile, speed-row, Modal, comparator, and tests
pytest: tests/test_compact_coach_speed_row_compare.py plus focused smoke/profile tests
result: 11 passed, 2 warnings
```

Decision:

- Diagnostic only.
- OPT-132AA used this field set on H100 and showed the sample-gate cost is
  broad across calls, not driven by only a few pathological sample calls.

## OPT-132AA H100 Sample-Gate Per-Call Diagnostic R1/R2

Artifacts:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132aa-h100-sample-gate-per-call-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1/compact_coach_speed_row_modal_report.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132aa-h100-sample-gate-per-call-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2/compact_coach_speed_row_modal_report.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132aa-h100-sample-gate-per-call-diagnostic-r1r2-comparison-20260603/comparison.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132aa-h100-sample-gate-per-call-diagnostic-zz-aa-comparison-20260603/comparison.json`

Read:

```text
identity: exact against each other and OPT-132Z
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
function calls: fc-01KT5Q4Z0KBGKSP4J17TX11Y0Q / fc-01KT5QJRPA1013JPFMHBP3BWR4
wall/speed: 213.543s / 5198.09, 224.408s / 4946.42
AA wall spread: 10.865s, 4.96% of median
AA sample gate: 128.039s -> 134.063s, spread 4.60%
AA learner-batch build: 76.980s -> 79.225s, spread 2.88%
runtime-step sum: 213.539s -> 224.404s
runtime p50/p95/max:
  0.068/1.250/1.934s -> 0.073/1.345/2.015s
sample-gate per-call p50/p95/max:
  1.032/1.630/1.723s -> 1.061/1.674/1.783s
learner-batch-build per-call p50/p95/max:
  0.652/0.804/1.094s -> 0.666/0.802/1.100s
candidate per-call p95: 0.263s -> 0.343s
rng per-call p95: 0.050s -> 0.056s
residual per-call p95: 0.566s -> 0.515s
builder cuda_sync: 7.241s -> 8.126s, count 11977/11977
stable speed claim allowed: false
```

Decision:

- No speed claim.
- The AA repeat was much tighter than the Z r1/r2 packet but still long-window
  diagnostic-only.
- Sample-gate cost is broad across sample calls. It is not explained by one or
  two pathological calls.
- Learner-batch builder work is the stable heavy center inside sample gate:
  build p50/p95 stayed around `0.65/0.80s` per call.
- Superseded by OPT-132AB: builder-child per-call attribution is now read on
  H100. The next work is isolating the builder group-loop/unroll/terminal
  swing, not a blind builder-group-loop patch.

## OPT-132AB Local Builder-Child Per-Call Attribution Patch

Local change:

```text
purpose: identify which learner-batch builder child drives broad sample-gate cost
speed claim: none
accepted-speed denominator change: none
new nested stats:
  compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_stats
  compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_per_call_stats
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_per_call_stats
  compact_rollout_slab_sample_gate_learner_batch_builder_write_output_per_call_stats
  compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_per_call_stats
  compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_per_call_stats
  compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_per_call_stats
  compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_per_call_stats
  compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_per_call_stats
flattened fields: count/sum/min/max/p50/p95/slowest_call_index
comparison fields: sum/min/max/p50/p95
```

What changed:

- Source profile uses the existing sample-gate call record stream to emit
  per-call stats for every builder child timer already recorded in telemetry.
- Speed-row summary and compact payloads flatten those stats and preserve the
  nested dictionaries.
- Local and remote Modal reports project the nested and flattened fields.
- The comparator includes builder-child per-call sum/min/max/p50/p95 timing
  fields, including existing learner-batch-build per-call p95.

Validation:

```text
ruff: passed for touched source/profile, speed-row, Modal, comparator, and tests
pytest: focused comparator, smoke, and source-profile tests
result: 12 passed, 2 warnings
```

Decision:

- Diagnostic only.
- Superseded by the OPT-132AB H100 read below. The next action is not a blind
  learner-batch builder speed patch; isolate the builder group-loop swing first.

## OPT-132AB H100 Builder-Child Per-Call Diagnostic R1/R2

Artifacts:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132ab-h100-builder-child-per-call-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1/compact_coach_speed_row_modal_report.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132ab-h100-builder-child-per-call-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2/compact_coach_speed_row_modal_report.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132ab-h100-builder-child-per-call-diagnostic-r1r2-comparison-20260603/comparison.json`

Read:

```text
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
function calls: fc-01KT5S9CDCRDZBKEXGXSP3QZMY / fc-01KT5SP000CGAS45NHKBZMQZ7E
wall/speed:
  r1 271.062s / 4095.07
  r2 170.468s / 6511.57
spread:
  wall 100.593s, 45.57% of median
  runtime-step sum 100.592s, 45.57%
  sample gate 77.654s, 54.93%
  learner-batch build 46.281s, 56.09%
  builder group-loop per-call sum 46.002s, 56.27%
  builder unroll-fields per-call sum 21.305s, 53.05%
builder-child sums:
  group-loop 104.758s -> 58.756s
  unroll fields 50.816s -> 29.511s
  terminal metadata 23.072s -> 12.377s
  write output 7.243s -> 4.326s
  builder cuda_sync 9.988s -> 5.930s
builder-child p50/p95:
  build 0.897/1.059s -> 0.496/0.598s
  group-loop 0.890/1.050s -> 0.490/0.592s
  unroll fields 0.431/0.507s -> 0.240/0.293s
  terminal metadata 0.197/0.231s -> 0.104/0.123s
stable speed claim allowed: false
```

Decision:

- No speed claim.
- Exact identity and proof fields held, so this is timing instability, not work
  drift.
- The broad timing swing is inside builder group-loop work.
- Unroll-field construction is the largest visible child, followed by terminal
  metadata. Builder CUDA sync is visible but smaller.
- Do not patch learner-batch builder speed code until the group-loop,
  unroll-field, or terminal-metadata swing is isolated enough to measure a
  change under exact `1084/270` repeatability.

## OPT-132AD Local Runtime Phase/Residual Diagnostic Patch

Local change:

```text
purpose: explain/bound the OPT-132AC r1c slow branch before speed code
speed claim: none
accepted-speed denominator change: none
trigger: --compact-profile-cuda-sync-timing-diagnostics
```

What changed:

- Runtime-step distribution stats now include actor env runtime and actor
  autoreset in addition to actor wall.
- Runtime-step distribution stats now include sample-gate residual,
  sample-gate CUDA sync, sample-gate builder group-loop, and sample-gate
  builder CUDA sync deltas derived from existing cumulative diagnostic timers.
- Existing runtime phase distributions for observation, compact rollout slab,
  sample gate, learner gate, policy refresh, primary accounted, and primary
  residual remain projected.
- Speed-row summary/compact payloads, local and remote Modal reports, and the
  comparator project
  `compact_profile_runtime_step_{phase}_{sum,min,max,p50,p95}_sec`.

Validation:

```text
ruff: passed for touched source/profile, speed-row, Modal, comparator, and tests
focused pytest:
  tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_compact_muzero_learner_gate_uses_explicit_next_targets
  tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_projects_cuda_sync_diagnostics
  tests/test_compact_coach_speed_row_compare.py::test_compare_speed_rows_includes_cuda_sync_diagnostic_fields
result: 3 passed, 2 warnings
broader compact speed-row slice: 16 passed, 2 warnings
```

Decision:

- Diagnostic only. It should be compared only against rows with the same AD
  field set.
- Superseded by the H100 r1/r2/r3 read and the OPT-132AE formal
  accounted/residual split below. No speed claim.

H100 launch/read:

```text
completed rows:
  opt132ad-h100-runtime-phase-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1
  opt132ad-h100-runtime-phase-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2
  opt132ad-h100-runtime-phase-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r3
reports:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ad-h100-runtime-phase-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1/compact_coach_speed_row_modal_report.json
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ad-h100-runtime-phase-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2/compact_coach_speed_row_modal_report.json
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ad-h100-runtime-phase-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r3/compact_coach_speed_row_modal_report.json
comparison:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ad-h100-runtime-phase-residual-diagnostic-r1-r2-r3-comparison-20260603/comparison.json
function calls:
  r1 fc-01KT70ANVD3N80TX0JXNN88RJP
  r2 fc-01KT72GZ503M27W7RPXRJDJ07H
  r3 fc-01KT73847VC31GWYJ4KXBTZFVK
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
stable speed claim allowed: false
wall/speed:
  r1 204.520s / 5427.42
  r2 231.874s / 4787.14
  r3 182.574s / 6079.81
spread:
  wall 49.300s, 24.11% of median
  runtime-step sum 49.300s, 24.11%
  sample gate 37.484s, 30.48%
  learner-batch build 24.948s, 34.32%
  sample-gate builder group-loop 24.801s, 34.49%
  unroll fields 13.121s, 36.13%
new-field sums:
  actor env runtime 10.628s / 13.338s / 12.224s
  actor autoreset 21.910s / 23.280s / 19.790s
  sample residual 25.939s / 29.594s / 23.927s
  sample CUDA sync 0.030s / 0.038s / 0.034s
  sample-gate builder group-loop 71.909s / 88.019s / 63.218s
  builder CUDA sync 10.863s / 12.768s / 9.848s
  terminal metadata 16.420s / 19.947s / 14.241s
  manual group-loop accounted 60.586s / 74.177s / 53.093s
  manual group-loop residual 11.323s / 13.842s / 10.125s
```

Decision:

- Exact identity and proof fields held for r1/r2/r3, but timing stability still
  failed. No speed claim.
- AD confirms the repeat swing is still mostly measured sample-gate /
  learner-batch builder group-loop work. Sample residual and actor
  env/autoreset move too, but they are secondary to the builder group-loop
  delta. Superseded by OPT-132AE, which makes the group-loop
  accounted/residual split first-class.

## OPT-132AE Local Builder Group-Loop Accounted/Residual Diagnostic Patch

Local change:

```text
purpose: formalize how much builder group-loop time is named child work versus
  residual before any speed patch
speed claim: none
accepted-speed denominator change: none
trigger: --compact-profile-cuda-sync-timing-diagnostics
```

What changed:

- The resident grouped learner-batch builder now reports
  `compact_muzero_learner_batch_builder_group_loop_accounted_sec` and
  `compact_muzero_learner_batch_builder_group_loop_residual_sec`.
- Sample-gate telemetry projects those fields as
  `compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec`
  and
  `compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec`.
- Accounted is terminal metadata + terminal-window hint + unroll fields +
  write output. Residual is group-loop minus accounted.
- Source profile, speed-row summary/compact payload, local/remote Modal
  reports, and the comparator project totals and per-call stats.

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py src/curvyzero/infra/modal/compact_coach_speed_row.py scripts/compare_compact_coach_speed_rows.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_compare.py
result: All checks passed.

uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_compact_muzero_learner_gate_uses_explicit_next_targets tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_projects_cuda_sync_diagnostics tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_requires_and_emits_fused_learner_batch_proof tests/test_compact_coach_speed_row_compare.py::test_compare_speed_rows_includes_cuda_sync_diagnostic_fields
result: 4 passed, 2 warnings

uv run pytest tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_compare.py
result: 63 passed, 2 warnings
```

H100 launch/read:

```text
completed rows:
  opt132ae-h100-builder-loop-accounted-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1
  opt132ae-h100-builder-loop-accounted-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2
reports:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ae-h100-builder-loop-accounted-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1/compact_coach_speed_row_modal_report.json
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ae-h100-builder-loop-accounted-residual-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2/compact_coach_speed_row_modal_report.json
comparison:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ae-h100-builder-loop-accounted-residual-diagnostic-r1-r2-comparison-20260603/comparison.json
function calls:
  r1 fc-01KT74133RFR458NBFCFWX0C3Q
  r2 fc-01KT74CKC3DAHGG3J2DVD94QGC
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
stable speed claim allowed: false
wall/speed:
  r1 271.920s / 4082.14
  r2 284.561s / 3900.80
spread:
  wall 12.641s, 4.54% of median
  sample gate 8.537s, 4.51%
  learner-batch build 6.056s, 5.14%
  sample-gate builder group-loop 5.882s, 5.03%
builder group-loop split:
  group-loop 113.971s -> 119.853s
  accounted child work 95.872s -> 101.300s
  residual 18.099s -> 18.553s
named child totals:
  unroll fields 57.127s -> 61.190s
  terminal metadata 26.666s -> 27.764s
  terminal-window hint 5.174s -> 5.415s
  write output 6.905s -> 6.931s
per-call p50/p95:
  group-loop 0.956s/1.162s -> 0.988s/1.219s
  accounted 0.805s/0.978s -> 0.837s/1.026s
  residual 0.151s/0.185s -> 0.154s/0.191s
```

Decision:

- Diagnostic only. No speed claim.
- Both AE rows are slow. The builder group-loop is mostly named child work, not
  residual. Unroll fields are the largest absolute child; terminal metadata is
  next.
- OPT-132AF now provides that default-off unroll-2 specialized builder locally;
  this local gate is superseded by the H100 r1/r2/r3 read below. Proof passed,
  but the stable speed claim was rejected.

## OPT-132AF H100 Guarded Unroll-2 Specialized Builder r1/r2/r3

Artifacts:

```text
r1: artifacts/local/curvytron_compact_coach_speed_row_results/opt132af-h100-unroll2-specialized-builder-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
r2: artifacts/local/curvytron_compact_coach_speed_row_results/opt132af-h100-unroll2-specialized-builder-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
r3: artifacts/local/curvytron_compact_coach_speed_row_results/opt132af-h100-unroll2-specialized-builder-diagnostic-b1024a1-normal-unroll2-m1084-w270-r3-20260603/row_001_result.json
comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132af-h100-unroll2-specialized-builder-diagnostic-r1-r2-r3-comparison-20260603/comparison.json
```

Read:

```text
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
unroll2 specialized builder violations: []
stable_speed_claim_allowed: false
window: stability_1084_270
r1 wall/speed: 211.682755496s / 5243.771498528963
r2 wall/speed: 225.02032744299999s / 4932.958780273657
r3 wall/speed: 234.04234284400002s / 4742.799898990401
wall spread: 22.35958734800002s / 9.93669665406737%
sample gate: 128.43333603900004s -> 139.11173540300007s -> 146.50472299799998s
learner-batch build: 78.60774218799986s -> 85.98325656999997s -> 90.698261082s
builder group-loop: 77.85183318599981s -> 85.16751857000008s -> 89.80348819599989s
group-loop accounted: 65.57660667099412s -> 71.75830154699142s -> 75.67058289498172s
group-loop residual: 12.275226515005691s -> 13.40921702300868s -> 14.13290530101817s
unroll fields: 39.99889293599863s -> 43.86709902599678s -> 46.46955278899776s
terminal metadata: 17.51481979900239s -> 19.229725937999618s -> 19.83223791699863s
proof: requested=true, used=true, eligible_count=399, call_count=399,
  fallback_count=0, fallback_reason=none, impl=unroll2_specialized_v1,
  path=unroll2_specialized
```

Decision:

- Diagnostic only. R1/R2/R3 prove the guarded path can run on H100 with intact
  proof, but the long-window wall spread is `9.94%`, sample gate spread is
  `12.99%`, learner-batch build spread is `14.06%`, and builder group-loop
  spread is `14.03%`.
- OPT-132AF is rejected as a stable speed claim under exact `1084/270`.

## OPT-132AG Local Comparator GPU Context Patch

Local change:

```text
purpose: compare AF reruns with hardware sampler context
speed claim: none
accepted-speed denominator change: none
next H100 shape: exact AF 1084/270 with GPU utilization sampling enabled
```

What changed:

- `scripts/compare_compact_coach_speed_rows.py` now adds a per-row
  `gpu_utilization` block for sampler enabled/interval/count/name, utilization,
  memory, power, and sampler errors.
- Numeric GPU sampler fields are also included in comparator timing/range
  output: sample count, max/mean utilization, nonzero/over-threshold counts,
  max memory utilization, max memory used, and max power draw.
- Identity matching is unchanged, so old rows without GPU sampling are not
  reclassified by this hardware context.

Validation:

```text
ruff: scripts/compare_compact_coach_speed_rows.py tests/test_compact_coach_speed_row_compare.py passed
pytest: tests/test_compact_coach_speed_row_compare.py passed, 10 passed, 2 warnings
refresh comparison:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132af-h100-unroll2-specialized-builder-diagnostic-r1-r2-r3-comparison-refresh-20260603/comparison.json
  identity: exact
  stable_speed_claim_allowed: false
  gpu sampling in source rows: disabled, zero/null hardware fields projected
```

Decision:

- Diagnostic tooling only. Superseded by the OPT-132AG H100 r4/r5/r6 packet
  below.

## OPT-132AG H100 AF GPU-Utilization Diagnostic r4/r5/r6

Artifacts:

```text
r4: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ag-h100-af-gpuutil-b1024a1-normal-unroll2-m1084-w270-r4-20260603/row_001_result.json
r5: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ag-h100-af-gpuutil-b1024a1-normal-unroll2-m1084-w270-r5-20260603/row_001_result.json
r6: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ag-h100-af-gpuutil-b1024a1-normal-unroll2-m1084-w270-r6-20260603/row_001_result.json
comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ag-h100-af-gpuutil-r4-r5-r6-comparison-20260603/comparison.json
combined AF+AG comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132af-ag-h100-af-gpuutil-r1-r6-comparison-20260603/comparison.json
```

Read:

```text
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
unroll2 specialized builder violations: []
stable_speed_claim_allowed: false
window: stability_1084_270
r4 wall/speed: 195.62403612300002s / 5674.231152771377
r5 wall/speed: 185.642903234s / 5979.30748045263
r6 wall/speed: 199.27535672300002s / 5570.262265509139
wall spread: 13.632453489000028s / 6.97%
sample gate: 121.378s / 109.782s / 123.823s, spread 11.57%
learner-batch build: 76.124s / 65.197s / 76.876s
builder group-loop: 75.411s / 64.437s / 76.134s, spread 15.51%
gpu sampler:
  samples 443 / 412 / 526
  mean util 18.44% / 18.69% / 15.54%
  max util 51% / 55% / 52%
  max memory used 38867 MiB on all rows
  max power 257.90W / 260.27W / 227.38W
  errors []
proof: requested=true, used=true, eligible_count=399, call_count=399,
  fallback_count=0, fallback_reason=none, impl=unroll2_specialized_v1,
  path=unroll2_specialized
combined AF+AG wall spread: 185.643s -> 234.042s, 23.55% of median
combined sample gate spread: 109.782s -> 146.505s, 29.12%
combined learner-batch build spread: 65.197s -> 90.698s, 32.80%
```

Decision:

- Diagnostic only. OPT-132AG breaks the simple monotonic AF slowdown pattern,
  and the GPU sampler does not support a simple power/thermal-throttle
  explanation: the slowest AG row had lower mean utilization and lower max
  power than the faster AG rows.
- Measurement stability still fails. This is superseded by OPT-132AH, where
  the generic-builder H100 comparison also drifted hard; do not claim speed
  or patch the builder again until diagnostic/runtime overhead is bounded.

## OPT-132AH H100 Generic GPU-Utilization Diagnostic r1/r2/r3

Artifacts:

```text
r1: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ah-h100-generic-gpuutil-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
r2: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ah-h100-generic-gpuutil-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
r3: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ah-h100-generic-gpuutil-b1024a1-normal-unroll2-m1084-w270-r3-20260603/row_001_result.json
comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ah-h100-generic-gpuutil-r1-r2-r3-comparison-20260603/comparison.json
```

Read:

```text
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
stable_speed_claim_allowed: false
window: stability_1084_270
builder path: generic
generic proof: requested=false, used=false, eligible_count=0, call_count=0,
  fallback_count=0, impl=none, path=generic
r1 wall/speed: 205.584135452s / 5399.3271297880265
r2 wall/speed: 243.805104229s / 4552.882530947301
r3 wall/speed: 289.171608001s / 3838.606451281211
wall spread: 83.58747254899998s / 34.28%
sample gate: 123.444s / 151.007s / 196.761s, spread 48.55%
learner-batch build: 75.790s / 92.226s / 123.781s, spread 52.04%
builder group-loop: 74.642s / 91.378s / 122.843s, spread 52.75%
gpu sampler:
  samples 445 / 539 / 583
  mean util 17.97% / 15.25% / 13.96%
  max util 48% / 44% / 43%
  max memory used 38867 MiB on all rows
  max power 264.47W / 222.48W / 240.14W
  errors []
```

Decision:

- Diagnostic only. OPT-132AH proves the exact same-work runtime swing also
  appears with the generic builder, so the remaining instability is not
  specific to OPT-132AF's specialized path.
- Slower rows had lower mean GPU utilization, max utilization, and the same max
  memory used. Max power was mixed, so this does not support a simple
  power/thermal-throttle explanation.
- Do not make another learner-batch builder speed patch until diagnostic or
  runtime overhead is bounded.

## OPT-132AI Local Runtime-Step-Only Diagnostic Flag

Local change:

```text
purpose: measure runtime-step envelope without CUDA-sync timing probes
speed claim: none
accepted-speed denominator change: none
new flag: --compact-profile-runtime-step-timing-diagnostics
```

What changed:

- `HybridObservationProfileConfig` now has
  `compact_profile_runtime_step_timing_diagnostics`.
- `--compact-profile-runtime-step-timing-diagnostics` is threaded through the
  local smoke CLI, Modal launcher command, Modal entrypoint config, result
  bundle/report fields, and source profile metadata.
- `--compact-profile-cuda-sync-timing-diagnostics` still implies runtime-step
  stats for backward compatibility with OPT-132Y through AH.
- Local and Modal collection fail closed if requested runtime-step stats do not
  project as active with positive count and finite nonnegative timing fields.

Validation:

```text
ruff: source profile, local smoke, Modal entrypoint/launcher, focused tests passed
pytest:
  tests/test_source_state_hybrid_observation_profile.py::test_runtime_step_timing_diagnostics_can_run_without_cuda_sync passed
  runtime-step Modal launcher/validator focused tests passed
  broader Modal launcher/report slice passed
  tests/test_compact_coach_speed_row_compare.py passed
```

Decision:

- Diagnostic tooling only. The next H100 overhead check can keep runtime-step
  envelope fields while removing CUDA-sync probes.

## OPT-132AJ H100 Generic Runtime-Step-Only GPU-Utilization Diagnostic r1/r2/r3

Artifacts:

```text
r1: artifacts/local/curvytron_compact_coach_speed_row_results/opt132aj-h100-generic-runtimeonly-gpuutil-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
r2: artifacts/local/curvytron_compact_coach_speed_row_results/opt132aj-h100-generic-runtimeonly-gpuutil-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
r3: artifacts/local/curvytron_compact_coach_speed_row_results/opt132aj-h100-generic-runtimeonly-gpuutil-b1024a1-normal-unroll2-m1084-w270-r3-20260603/row_001_result.json
r1/r2 comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132aj-h100-generic-runtimeonly-gpuutil-r1-r2-comparison-20260603/comparison.json
r1/r2/r3 comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132aj-h100-generic-runtimeonly-gpuutil-r1-r2-r3-comparison-20260603/comparison.json
```

Read:

```text
identity: exact
accepted-fast-path violations: []
runtime-step diagnostic violations: []
cuda-sync diagnostics: false
sync counts: 0
stable_speed_claim_allowed: false
window: stability_1084_270
builder path: generic
r1 wall/speed: 219.054914326s / 5067.295583919481
r2 wall/speed: 221.938201366s / 5001.4643408300135
r3 wall/speed: 198.94967915700002s / 5579.380699196992
wall spread: 22.988522208999967s / 10.49%
runtime-step sum: 219.050s / 221.932s / 198.945s, spread 10.49%
sample gate: 117.675s / 132.786s / 118.763s, spread 12.72%
learner-batch build: 63.964s / 73.104s / 64.857s, spread 14.09%
builder group-loop: 63.102s / 72.243s / 64.144s, spread 14.25%
gpu sampler:
  samples 470 / 482 / 444
  mean util 15.23% / 16.98% / 18.30%
  max util 48% / 45% / 45%
  max memory used 38867 MiB on all rows
  max power 226.81W / 245.75W / 243.50W
```

Decision:

- Diagnostic only. Removing CUDA-sync probes narrows the generic instability
  versus AH but does not solve exact-repeat stability.
- Runtime-step sum reconciles with wall, so the remaining swing is inside
  measured iterations.
- Do not make another learner-batch builder speed patch until the AJ
  observation/sample-gate runtime swing is explained or bounded.

## OPT-132AK Local Wall-Swing Attribution and Runtime-Cadence Diagnostics

Local change:

```text
purpose: explain same-work measured-iteration swing without CUDA-sync probes
speed claim: none
accepted-speed denominator change: none
```

What changed:

- `scripts/compare_compact_coach_speed_rows.py` now emits
  `exact_repeat_stability.wall_swing_attribution`, comparing the slowest and
  fastest exact rows and reporting per-field deltas as a percentage of the wall
  delta.
- Runtime-step diagnostics now summarize sample-gate active/inactive measured
  steps, early/mid/late measured thirds, and a bounded
  `compact_profile_runtime_step_top_slowest_records` list.
- Summary/compact payloads, local and remote Modal reports, and the comparator
  project the new flattened cadence fields.

Validation:

```text
ruff: source profile, smoke builder, Modal launcher/entrypoint, comparator, focused tests passed
pytest:
  source runtime-step-only test passed
  compact MuZero learner-gate source-profile test passed
  speed-row smoke runtime projection/validator tests passed
  comparator suite passed, 12 passed
```

Refresh artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132ak-aj-wall-swing-attribution-refresh-20260603/comparison.json`

AJ attribution read:

```text
fastest: r3, 198.950s
slowest: r2, 221.938s
wall delta: 22.989s
runtime-step sum delta: 22.987s, 99.995% of wall delta
primary accounted delta: 22.093s, 96.10%
sample gate delta: 14.022s, 61.00%
learner-batch build delta: 8.247s, 35.87%
builder group-loop delta: 8.099s, 35.23%
```

Decision:

- Diagnostic tooling only. It clarifies AJ but does not change the speed
  denominator.
- The next H100 read should use the cadence fields without CUDA-sync probes;
  OPT-132AL did that with GPU sampling disabled.

## OPT-132AL H100 Generic Runtime-Cadence No-Sampler Diagnostic r1/r2/r3

Artifacts:

```text
r1: artifacts/local/curvytron_compact_coach_speed_row_results/opt132al-h100-generic-runtimecadence-nosampler-b1024a1-normal-unroll2-m1084-w270-r1-20260603/row_001_result.json
r2: artifacts/local/curvytron_compact_coach_speed_row_results/opt132al-h100-generic-runtimecadence-nosampler-b1024a1-normal-unroll2-m1084-w270-r2-20260603/row_001_result.json
r3: artifacts/local/curvytron_compact_coach_speed_row_results/opt132al-h100-generic-runtimecadence-nosampler-b1024a1-normal-unroll2-m1084-w270-r3-20260603/row_001_result.json
comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132al-h100-generic-runtimecadence-nosampler-r1-r2-r3-comparison-20260603/comparison.json
function calls: fc-01KT7HJSM5GD8GE40GWZHSZXWW / fc-01KT7HJSNGCK7E8F2QAQ3GCRWE / fc-01KT7HJT5RRE94EFVRYCR9JKY7
```

Read:

```text
identity: exact
accepted-fast-path violations: []
runtime-step diagnostic violations: []
cuda-sync diagnostic violations: []
GPU sampling: false
builder path: generic
stable_speed_claim_allowed: false
r1 wall/speed: 168.661s / 6581.34
r2 wall/speed: 178.629s / 6214.07
r3 wall/speed: 207.419s / 5351.56
wall spread: 38.758s / 21.70%
runtime-step sum spread: 38.756s / 21.70%
sample gate: 97.710s / 102.538s / 121.130s, spread 22.84%
learner-batch build: 54.231s / 56.034s / 65.956s, spread 20.92%
builder group-loop: 53.566s / 55.349s / 65.163s
sample-gate active measured-step sum: 114.679s / 120.979s / 142.494s
late measured third: 73.166s / 75.718s / 96.470s
late sample gate: 48.364s / 50.301s / 64.010s, spread 31.11%
late builder group-loop: 23.702s / 24.450s / 30.956s
```

Wall-swing attribution:

```text
fastest: r1, slowest: r3
wall delta: 38.758s
runtime-step sum delta: 38.756s, 99.995% of wall delta
primary accounted delta: 37.384s, 96.45%
sample-gate active-step sum delta: 27.815s, 71.76%
sample gate delta: 23.420s, 60.42%
late measured-third delta: 23.304s, 60.13%
late sample-gate delta: 15.647s, 40.37%
learner-batch build delta: 11.725s, 30.25%
builder group-loop delta: 11.596s, 29.92%
```

Decision:

- Diagnostic only. Disabling the GPU sampler did not solve stability; AL
  actually returned a monotonic no-sampler slowdown.
- Superseded by OPT-132AN: the active-call fields show broad active
  sample-gate slowdown/elevation, not only late measured-window spikes. Do not
  make another builder speed patch until that cadence slowdown is explained or
  bounded.

## OPT-132AM Local Active Sample-Gate Cadence Diagnostics

Local change:

```text
purpose: distinguish broad late active sample-gate slowdown from late spikes
speed claim: none
accepted-speed denominator change: none
```

What changed:

- Runtime-step cadence stats now include chronological active sample-gate
  distributions by measured third for sample gate, sample-gate residual,
  builder group-loop, learner gate, observation, and primary residual.
- Bucket sums include sample-gate residual.
- Top-slowest runtime-step records include sample-gate residual and builder
  sync fields.
- Sample-gate per-call stats project slowest iteration and slowest measured
  iteration through local and remote reports.
- Comparator timing fields include the new active sample-gate p50/p95/max
  fields so future exact-repeat reports can surface the late bucket directly.

Validation:

```text
ruff: source profile, smoke builder, Modal launcher/entrypoint, comparator, focused tests passed
pytest:
  focused source-profile/smoke/comparator slice: 3 passed
  comparator suite: 12 passed
  Modal launcher/report plus source-profile slice: 5 passed
```

Sidecar read:

- AL artifact audit found active sample-gate counts stable at `135/135/135`,
  late active counts stable at `45/45/45`, deferred sample/learner disabled,
  replay/sample checksums and retained snapshot counts identical, and the same
  late sample-call indices slowing down.
- The next H100 evidence should be an AL-style exact generic no-sampler
  `1084/270` row with AM fields, then compare early/mid/late active
  sample-gate p50/p95/max. If still ambiguous, add a bounded per-sample-gate
  replay-state trace.

Decision:

- Local diagnostic tooling only. No speed claim.
- Do not patch builder speed code until the AM fields have been read on exact
  H100 or the late cadence branch is otherwise explained.

## OPT-132AF Local Guarded Unroll-2 Specialized Builder Patch

Local change:

```text
purpose: attack AE's largest named child while preserving generic learner-batch
  semantics
speed claim: none
accepted-speed denominator change: none
trigger: --compact-muzero-learner-batch-unroll2-specialized-builder
```

What changed:

- Added a default-off specialized unroll-2 helper for the fused resident grouped
  learner-batch builder. The generic unroll helper remains the fallback/oracle.
- Guarded the specialized path on learner unroll `2`, chain length `3`,
  terminal-window row-count metadata, and device replay rows.
- Added proof fields for requested, eligible count, used, call count, fallback
  count/reason, implementation, and unroll builder path.
- Projected those fields through source-profile sample-gate telemetry,
  speed-row summary/compact payload, local and remote Modal reports, the shared
  compact speed-row reducer, and the comparator.
- Tightened local/script validation so requested specialization fails closed
  unless call count is positive, fallback count is zero, fallback reason is
  `none`, impl is `unroll2_specialized_v1`, and path is
  `unroll2_specialized`.

Validation:

```text
ruff: passed for touched source/profile, speed-row, Modal, comparator, and tests
pytest:
  source-profile parity focused test passed
  speed-row proof/modal focused tests passed
  compact speed-row broader slice: 98 passed, 2 warnings
local smoke:
  opt132af-local-unroll2-specialized-builder-hardened-smoke-20260603
  ok=true
  requested=true, used=true
  eligible_count=13, call_count=13
  fallback_count=0, fallback_reason=none
  impl=unroll2_specialized_v1
  path=unroll2_specialized
  terminal_unroll_value_target_row_count=1
  training_wall_sec=1.1563019589957548
  steps_per_sec=7084.654606236878
```

Decision:

- Local proof only; superseded by the H100 r1/r2/r3 diagnostic above. No speed
  claim.
- Future OPT-132AF reruns must keep the same fail-closed proof fields already
  satisfied by H100 r1/r2/r3: requested=true, used=true, eligible_count>0,
  call_count>0, fallback_count=0, fallback_reason=`none`, impl=
  `unroll2_specialized_v1`, and path=`unroll2_specialized`.

## OPT-132AC Local Deep Builder Group-Loop Diagnostic Patch

Local change:

```text
purpose: split the OPT-132AB group-loop/unroll/terminal swing before speed code
speed claim: none
accepted-speed denominator change: none
trigger: --compact-profile-cuda-sync-timing-diagnostics
```

What changed:

- Terminal metadata now reports child timers for mask construction, tensor
  fallback/readback handling, validation, and final-observation handling.
- Unroll field construction now reports child timers for terminal-window hints,
  identity handling, field stacking, mask building, terminal value loading,
  mask application, and action stacking.
- The source profile maps all new builder totals into sample-gate telemetry and
  emits matching per-call stats.
- Speed-row summary/compact payloads, local and remote Modal reports, and the
  comparator project totals and per-call count/sum/min/max/p50/p95 fields.

Validation:

```text
ruff: passed for touched source/profile, speed-row, Modal, comparator, and tests
pytest: focused comparator, smoke, and source-profile tests
result: 12 passed, 2 warnings
```

H100 launch/read:

```text
runs that failed before a row:
  opt132ac-h100-builder-deep-child-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1
  opt132ac-h100-builder-deep-child-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1b
error: Modal RESOURCE_EXHAUSTED, server memory usage too high
report/result: none for either run; both predate structured launch-failure artifacts

completed rows:
  opt132ac-h100-builder-deep-child-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1c
  opt132ac-h100-builder-deep-child-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2
  opt132ac-h100-builder-deep-child-diagnostic-b1024a1-normal-unroll2-m1084-w270-r3
reports:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ac-h100-builder-deep-child-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1c/compact_coach_speed_row_modal_report.json
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ac-h100-builder-deep-child-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2/compact_coach_speed_row_modal_report.json
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ac-h100-builder-deep-child-diagnostic-b1024a1-normal-unroll2-m1084-w270-r3/compact_coach_speed_row_modal_report.json
comparison:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ac-h100-builder-deep-child-diagnostic-r1c-r2-r3-comparison-20260603/comparison.json
function calls:
  r1c fc-01KT6W3WTW60TWHSBKH64R97VG
  r2  fc-01KT6XXN4M5JW9YQ5CS5618WVD
  r3  fc-01KT6YG1E9TPM44YPA3BEZ4F5R
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
stable speed claim allowed: false
wall/speed:
  r1c 235.855s / 4706.36
  r2  174.009s / 6379.07
  r3  172.247s / 6444.33
spread:
  wall 63.608s, 36.55% of median
  runtime-step sum 63.607s, 36.55%
  sample gate 40.718s, 37.32%
  learner-batch build 23.229s, 34.23%
  builder group-loop 22.931s, 34.11%
  unroll fields 13.686s, 40.16%
  terminal metadata 4.433s, 27.48%
deep-child sums:
  terminal final observation 11.121s / 8.376s / 9.902s
  terminal tensor fallback 4.541s / 3.501s / 3.811s
  terminal mask 0.642s / 0.500s / 0.458s
  terminal validate 0.987s / 0.762s / 0.725s
  unroll terminal-window hint 3.486s / 2.769s / 3.250s
  unroll mask apply 3.878s / 2.563s / 2.727s
  unroll action stack 2.172s / 1.452s / 1.689s
  unroll mask build 13.321s / 9.635s / 10.302s
  unroll terminal value 11.582s / 7.427s / 8.237s
  unroll stack fields 10.919s / 7.603s / 8.013s
  builder cuda_sync 11.734s / 9.910s / 8.588s
```

Decision:

- Diagnostic only. The new timers add extra CUDA sync timing blocks under the
  diagnostic flag, so compare AC rows only against AC rows.
- Exact identity and proof fields held for r1c/r2/r3, but timing stability
  still failed. No speed claim.
- R2/R3 clustered while r1c was a slow outlier; explain or bound that branch
  before deciding whether to instrument deeper or patch builder code.

## OPT-132AC Modal Launch-Failure Artifact Patch

Local change:

```text
purpose: preserve evidence when Modal rejects launch before a FunctionCall exists
speed claim: none
accepted-speed denominator change: none
```

What changed:

- `_launch_remote()` now raises a structured launch exception for nonzero
  `modal run --detach` exits instead of exiting before artifacts can be written.
- The Modal entrypoint catches reachable `.spawn(config)` failures and prints a
  structured `spawn_failed` payload for the local launcher to convert into the
  same launch-failure artifact shape.
- The launcher writes `launch.json` and
  `compact_coach_speed_row_modal_report.json` for pre-FunctionCall launch
  failures.
- The modal report records `ok=false`, `failure_stage=launch`,
  `function_call_id=""`, launch return code, stdout/stderr tails, non-claims,
  accepted-fast-path diagnostic context, and
  `modal_launch_resource_exhausted` / `modal_launch_error_code` when Modal
  reports `RESOURCE_EXHAUSTED` or server memory pressure.
- A focused parity test keeps local and remote Modal per-call report prefix
  lists identical.

Validation:

```text
ruff: passed for launcher and smoke tests
pytest: focused launcher/projection/source-profile/comparator suite
result: 16 passed, 2 warnings
```

Decision:

- This does not turn the failed OPT-132AC r1/r1b attempts into rows; they remain
  empty-dir launch failures because they predate the patch.
- Future AC retries should leave a structured artifact even if Modal rejects
  launch before a FunctionCall exists.

## OPT-132X H100 Learner-Batch Timing Diagnostic R1

Artifacts:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132x-h100-learner-batch-timing-diagnostic-b1024a1-normal-unroll2-m1084-w270-r1/row_001_result.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132x-h100-learner-batch-timing-diagnostic-comparison-20260602/comparison.json`

Read:

```text
wall/speed: 181.348s / 6120.91 env steps/sec
identity versus prior r1/r2: exact
accepted-fast-path violations: []
bounded diagnostics: true
source profile payload embedded: false
resident replay snapshot mode: latest_frame_history
retained resident snapshot bytes: 9346744320
host fallback: 0
render-state copy steps: 0
terminal sample/target rows: 512/512
sample/learner updates: 135/135
sample gate: 107.107s
learner-batch build: 57.573s
```

New learner-batch detail:

```text
per-call count: 135
per-call p50/p95/max: 0.500s / 0.606s / 0.879s
builder group loop: 56.837s
unroll fields: 32.877s
terminal metadata: 13.292s
write output: 1.122s
order restore: 0.191s
metadata sync/readback: 0.035s
```

Three-row comparison:

```text
rows: prior r1, prior r2, diagnostic r1
wall range: 181.348s to 244.168s, 27.95% of median
sample gate range: 107.107s to 155.937s, 36.04% of median
learner-batch build range: 57.573s to 86.207s, 38.82% of median
stable_speed_claim_allowed: false
```

Decision:

- The new fields are visible and useful.
- The faster row is not a speed claim.
- Superseded by r2: the same-field r1/r2 pair is stable on the main buckets.

## OPT-132X H100 Learner-Batch Timing Diagnostic R2

Artifacts:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132x-h100-learner-batch-timing-diagnostic-b1024a1-normal-unroll2-m1084-w270-r2/row_001_result.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132x-h100-learner-batch-timing-diagnostic-r1r2-comparison-20260602/comparison.json`

Read:

```text
r1 wall/speed: 181.348s / 6120.91 env steps/sec
r2 wall/speed: 182.777s / 6073.07 env steps/sec
identity: exact
accepted-fast-path violations: []
resident replay snapshot mode: latest_frame_history
retained resident snapshot bytes: 9346744320 on both
host fallback: 0
render-state copy steps: 0
terminal sample/target rows: 512/512
sample/learner updates: 135/135
wall spread: 1.429s, 0.78% of median
sample gate spread: 2.294s, 2.17% of median
learner-batch build spread: 1.145s, 2.01% of median
```

Repeated learner-batch detail:

```text
learner-batch build: 57.573s / 56.428s
per-call p50: 0.500s / 0.474s
per-call p95: 0.606s / 0.573s
per-call max: 0.879s / 0.852s
slowest call: index 95, seed 20260624 on both
builder group loop: 56.837s / 55.674s
unroll fields: 32.877s / 32.057s
terminal metadata: 13.292s / 13.174s
metadata sync/readback: 0.035s / 0.037s
```

Decision:

- Same-field r1/r2 is stable on wall, sample gate, and learner-batch build.
- Still no speed claim: these are long-window diagnostics, not the OPT-104
  accepted denominator.
- Superseded by r3: the three-row packet failed stability.

## OPT-132X H100 Learner-Batch Timing Diagnostic R3

Artifacts:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132x-h100-learner-batch-timing-diagnostic-b1024a1-normal-unroll2-m1084-w270-r3/row_001_result.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132x-h100-learner-batch-timing-diagnostic-r1r2r3-comparison-20260602/comparison.json`

Read:

```text
r1 wall/speed: 181.348s / 6120.91 env steps/sec
r2 wall/speed: 182.777s / 6073.07 env steps/sec
r3 wall/speed: 154.574s / 7181.12 env steps/sec
identity: exact
accepted-fast-path violations: []
resident replay snapshot mode: latest_frame_history
retained resident snapshot bytes: 9346744320 on all three
host fallback: 0
render-state copy steps: 0
terminal sample/target rows: 512/512
sample/learner updates: 135/135
wall spread: 28.202s, 15.55% of median
sample gate spread: 20.428s, 19.49% of median
learner-batch build spread: 10.654s, 18.88% of median
```

R3 learner-batch detail:

```text
learner-batch build: 46.919s
per-call p50/p95/max: 0.385s / 0.475s / 0.733s
slowest call: index 117, seed 20260646
builder group loop: 46.358s
unroll fields: 26.821s
terminal metadata: 10.825s
metadata sync/readback: 0.031s
```

Decision:

- R3 kept exact identity but broke timing stability.
- Do not patch speed code yet.
- Add a diagnostic-only CUDA sync/event timing mode around sample gate,
  learner-batch builder children, and learner phases. If that stabilizes, then
  target learner-batch builder group-loop work.

## OPT-123: Earlier Useful Owner-Search Row

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt123-h100-owner-action-only-materialized-device-replay-samework-b1024a1-normal-unroll2-speed-20260602-r3/compact_coach_speed_row_modal_report.json`

Read:

```text
speed: 7679.53 env steps/sec
wall: 24.001s
parent wait: 9.219s
worker wall: 8.417s
root resolve: 5.299s
search: 2.755s
replay append: 8.983s
learner train: 8.593s
refresh: 1.333s
result bytes: 4444
request bytes: 32779
owner appends/train/update/refresh: 224/28/28/28
parent sample/learner: 0/0
terminal sample/target rows: 42/42
normal-death gate: true
final pending / policy lag: 0/0
```

Decision:

- OPT-123 was the first useful owner-search bucket map.
- It is progress, but it is rejected as the speed win and superseded by
  OPT-127 for current owner-route timing.

## OPT-125: Resident Replay Step Host-Copy Elision, Rejected

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt125-h100-resident-owner-replay-step-no-host-copy-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
speed: 6593.40 env steps/sec
wall: 27.955s
root resolve: 4.031s
search: 3.408s
replay append: 2.306s
learner train: 13.628s
actor wall: 11.895s
owner appends/train/update/refresh: 224/28/28/28
parent sample/learner: 0/0
normal-death gate: true
final pending / policy lag: 0/0
```

Decision:

- Rejected as a full-loop speed row.
- Useful clue: replay append had real copy waste, but cutting one copy was not
  enough because learner/train/action wait worsened.

## OPT-126: Replay Successor-Window Cache, Rejected

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt126-h100-replay-successor-window-cache-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
speed: 6591.11 env steps/sec
wall: 27.965s
parent wait: 11.310s
worker wall: 10.191s
root resolve: 6.662s
search: 3.116s
replay append: 4.215s
learner train: 10.564s
refresh: 1.477s
owner appends/train/update/refresh: 224/28/28/28
parent sample/learner: 0/0
terminal sample/target rows: 42/42
normal-death gate: true
final pending / policy lag: 0/0
GPU mean/max: 7.36% / 30%
```

Decision:

- Rejected as a full-loop speed row.
- The small replay-sampling cache is not the next path. Stop small cache edits
  unless a future row makes sampling setup the clear measured blocker again.

## OPT-127: Resident Owner Root/Replay Host-Copy Cut, Useful But Still Too Slow

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt127-h100-resident-root-replay-no-host-observation-copy-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Local validation:

```text
ruff passed for compact owner-search service and owner-search tests
focused resident root/replay tests: 2 passed, 2 warnings
```

H100 read:

```text
speed: 10371.96 env steps/sec
wall: 17.771s
parent wait: 4.616s
worker wall: 3.711s
root resolve: 0.740s
search: 2.641s
replay append: 0.301s
learner train: 7.324s
refresh: 0.980s
actor wall: 7.736s
observation: 1.890s
slab: 5.929s
GPU mean/max: 11.27% / 39%
owner appends/train/update/refresh: 224/28/28/28
parent sample/learner: 0/0
terminal sample/target rows: 42/42
normal-death gate: true
final pending / policy lag: 0/0
```

Decision:

- Useful improvement over OPT-123: `24.001s -> 17.771s` wall.
- Still below OPT-104: target is `14.5255s` wall.
- Root resolve and replay append are no longer the main explanation.
- OPT-128 later tested serial inline owner execution and lost. Use OPT-129 as
  the next task: direct root handoff plus restored background overlap.
- Caveat: the H100 report did not project the new
  `compact_owner_search_resident_root_bridge_host_observation_copied` boolean.
  The report field list is fixed locally for the next row.

## OPT-128: Serial Inline Owner Path, Rejected

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt128-h100-inline-owner-search-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Local validation:

```text
ruff passed for touched owner-search/speed-row files
py_compile passed for touched runtime scripts/modules
direct-root and inline proxy tests passed
owner-search service suite slice passed: 25 passed, 2 warnings
local inline smoke passed with direct-root proof
```

H100 read:

```text
speed: 9516.73 env steps/sec
wall: 19.368s
boundary: inline_owner_search_parent_slab_commit
direct root store: true
direct root publish/resolve: 225/225
parent wait: 0.001s
worker wall: 2.706s
root resolve: 0.025s
search: 2.296s
replay append: 0.278s
learner train: 4.903s
refresh: 1.019s
actor wall: 8.404s
observation: 2.127s
slab: 8.589s
GPU mean/max: 14.30% / 31%
owner appends/train/update/refresh: 224/28/28/28
parent sample/learner: 0/0
terminal sample/target rows: 42/42
normal-death gate: true
final pending / policy lag: 0/0
```

Decision:

- Rejected as a full-loop speed row.
- It proves direct in-process root handoff can preserve correctness and cut root
  resolve/parent wait.
- It also proves plain serial inline execution is the wrong shape: wall regressed
  versus OPT-127 `17.771s -> 19.368s`.
- Later OPT-129 threaded owner work also lost badly. Current task after OPT-130
  is OPT-131 direct hot-path work.

## OPT-129: Threaded Owner Path, Rejected

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt129-h100-threaded-owner-search-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
speed: 6051.61 env steps/sec
wall: 30.458s
boundary: threaded_owner_search_parent_slab_commit
direct root publish/resolve: 225/225
parent wait: 6.747s
worker wall: 5.343s
root resolve: 0.051s
search: 4.555s
replay append: 2.062s
learner train: 23.778s
refresh: 2.313s
actor wall: 17.914s
GPU mean/max: 7.72% / 27%
owner appends/train/update/refresh: 224/28/28/28
parent sample/learner: 0/0
terminal sample/target rows: 42/42
normal-death gate: true
final pending / policy lag: 0/0
```

Decision:

- Rejected as a full-loop speed row.
- Threaded owner work caused severe contention. Do not continue this lane.

## OPT-130: Direct Compact Torch Direct-Core, Rejected

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt130-h100-direct-compacttorch-directcore-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
speed: 11038.67 env steps/sec
wall: 16.698s
search service: compact_torch_search_service
initial inference mode: direct_core
actor wall: 7.363s
sample gate: 2.735s
slab/search: 2.826s
observation: 1.839s
learner gate: 1.397s
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
resident host fallback: 0
GPU mean/max: 12.27% / 36%
```

Decision:

- Rejected as a speed win versus OPT-104.
- It is faster than all current owner variants but still slower than OPT-104.
- The next task is OPT-131: choose a direct compact trainer hot-path target from
  this bucket map.

## OPT-131-A: Native Actor Direct Autoreset, Rejected

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt131a-h100-direct-autoreset-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
target bucket: OPT-130 actor/autoreset
OPT-130 actor wall: 7.363s
OPT-130 actor autoreset: 3.715s
change: native actor step_into uses compact direct autoreset instead of public
        autoreset batch materialization
local validation: focused terminal/autoreset tests passed
H100 speed: 9194.70 env steps/sec
H100 wall: 20.046s
direct autoreset count / rows: 170 / 1050
template-copy skipped count: 170
actor wall: 7.256s
sample gate: 3.461s
slab/search: 3.214s
observation: 2.346s
learner gate: 3.114s
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
resident host fallback: 0
```

Decision:

- Rejected as a full-loop speed win.
- The counters prove the direct autoreset hookup worked, but actor wall barely
  moved and the full row regressed.
- Do not keep chasing autoreset as the main target.
- Next proof is OPT-131-B: combine direct_core with the fused learner-batch path
  that OPT-104 already used.

## OPT-131-B: Direct-Core Plus Fused Learner Batch, Rejected

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt131b-h100-fused-learner-batch-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
status: complete
reason: OPT-104 used fused prebuilt learner batches; OPT-130 direct_core did not
speed: 10473.11 env steps/sec
wall: 17.599s
fused proof: true
learner-batch-only: true
resident grouped direct learner: true
learner gate prebuilt batch: true
sample batch build: 0.000s
learner batch build: 1.302s
actor wall: 7.018s
observation: 3.077s
sample gate: 2.558s
slab/search: 2.805s
learner gate: 1.518s
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
```

Decision:

- Rejected as a full-loop speed win.
- It proved the fused path fired, but direct autoreset was still forced on in
  this row. Run the clean recombination with direct autoreset parked.

## OPT-131-C: Direct-Core Plus Fused Learner Batch, Direct Autoreset Parked, Rejected

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt131c-h100-directcore-fused-no-direct-autoreset-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
status: complete
reason: clean recombination of direct_core with OPT-104 fused learner-batch path
speed: 9272.60 env steps/sec
wall: 19.878s
direct autoreset counters: 0
fused proof: true
learner-batch-only: true
resident grouped direct learner: true
sample batch build: 0.000s
learner batch build: 1.667s
actor wall: 9.016s
observation: 2.385s
sample gate: 2.900s
slab/search: 2.987s
learner gate: 1.925s
direct_core used/fallback: 180/0
initial-inference sync: 1.032s
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
```

Decision:

- Rejected as a full-loop speed win.
- The clean recombination still lost. The next test is the existing
  `host_final_sync_only` timing mode to see whether the direct-core
  initial-inference sync is real wall waste or just moved wait.

## OPT-131-D: Direct-Core Fused, Final-Sync-Only Timing, Rejected

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt131d-h100-directcore-fused-final-sync-only-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
status: complete
reason: OPT-131-C still had about 1.03s direct-core initial-inference sync
speed: 7625.92 env steps/sec
wall: 24.170s
initial-inference sync: 0.000s
search dispatch: 2.393s
actor wall: 11.762s
sample gate: 3.762s
learner gate: 2.468s
fused proof: true
direct autoreset counters: 0
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
```

Decision:

- Rejected as a full-loop speed win.
- Removing the direct-core phase sync did not produce a wall win. The loop got
  much slower, so the wait moved or the mode changed scheduling badly.
- Next run is a diagnostic control: current code, model_method plus fused
  learner batch. This is not the OPT-104 accepted shape; OPT-104 used
  direct_core plus fused learner batch.

## OPT-131-E: Model-Method Fused Control

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt131e-h100-modelmethod-fused-control-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
status: complete
reason: test whether model_method avoids the current direct_core regression
speed: 10076.28 env steps/sec
wall: 18.292s
initial inference mode: model_method
fused proof: true
learner-batch-only: true
resident grouped direct learner: true
sample batch build: 0.000s
learner batch build: 1.669s
actor wall: 7.410s
observation: 2.154s
sample gate: 3.237s
slab/search: 3.082s
learner gate: 1.832s
direct autoreset counters: 0
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
```

Decision:

- Rejected as a full-loop speed win.
- This was a useful diagnostic, but it was not the accepted OPT-104 shape.
  OPT-104 and OPT-109 used direct_core plus fused learner batches.
- Next run is OPT-132-A: direct_core plus fused learner batch with GPU
  utilization sampling off, to test whether newer measurement overhead is part
  of the current regression.

## OPT-132-A: Direct-Core Fused, GPU Sampler Off

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132a-h100-directcore-fused-no-gpu-sampler-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
status: complete
reason: OPT-104/109 used direct_core + fused and did not carry the newer GPU
sampler fields; recent slow rows did. Test whether the current measurement
wrapper is perturbing the accepted path.
speed: 9040.90 env steps/sec
wall: 20.387s
initial inference mode: direct_core
fused proof: true
learner-batch-only: true
resident grouped direct learner: true
sample batch build: 0.000s
learner batch build: 1.619s
actor wall: 9.531s
observation: 2.708s
sample gate: 2.805s
slab/search: 2.991s
learner gate: 1.674s
borrowed render state: false
render-state copy steps: 225
lean trainer step: false
direct autoreset counters: 0
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
```

Decision:

- Rejected as a full-loop speed win.
- GPU sampling was not the explanation.
- The row exposed the bigger mistake: recent direct rows dropped accepted-path
  actor/trainer flags. OPT-104/109 used borrowed single-actor render state.
  OPT-117/118 also used lean trainer step.

## OPT-132-B: Direct-Core Fused, Borrowed Render State, Lean Trainer Step

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132b-h100-directcore-fused-borrow-lean-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
status: complete
reason: restore the actual fast-path flags from OPT-104/109/117/118
speed: 11609.78 env steps/sec
wall: 15.876s
initial inference mode: direct_core
fused proof: true
borrowed render state: true
render-state copy steps: 0
lean trainer step: true
actor wall: 5.060s
observation: 2.494s
sample gate: 3.110s
slab/search: 2.982s
learner gate: 1.565s
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
```

Decision:

- Useful recovery but still rejected as a full-loop speed win.
- Restoring borrow+lean recovered about `4.51s` wall versus OPT-132-A, proving
  that dropped fast-path flags were the main source of the regression.
- Still slower than OPT-117/118 by about `1.2s`; run OPT-132-C with the
  OPT-118-style GPU sampler on to separate variance from another code/default
  regression.

## OPT-132-C: Direct-Core Fused, Borrowed Render State, Lean Trainer Step, GPU Sampler On

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132c-h100-directcore-fused-borrow-lean-gpusampler-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
status: complete
reason: repeat the OPT-118-style flag set against current code
speed: 9735.83 env steps/sec
wall: 18.932s
initial inference mode: direct_core
fused proof: true
borrowed render state: true
render-state copy steps: 0
lean trainer step: true
GPU sampling: true
actor wall: 6.112s
observation: 2.129s
sample gate: 4.084s
slab/search: 3.459s
learner gate: 2.177s
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
```

Decision:

- Rejected as a full-loop speed win.
- GPU sampling is not the rescue; this repeat was worse than OPT-132-B.
- OPT-132-B remains the useful recovery row. The next move is to prevent
  dropping the accepted fast-path flag bundle, then isolate the remaining
  current-code gap against OPT-117/118.

## OPT-132-G/H: Resident Stack Timing Split, Fast Row Not Repeat-Confirmed

OPT-132-G artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132g-h100-resident-stack-timing-split-preset-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

OPT-132-H artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132h-h100-resident-stack-timing-split-confirm-preset-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Local validation before H100 rows:

```text
ruff passed for touched source/report/evidence/test files
source-state suite: 106 passed, 2 warnings
focused launcher guard tests: 2 passed, 2 warnings
focused compact speed-row/evidence tests: 28 passed, 2 warnings
focused resident terminal/sample tests: 3 passed, 2 warnings
```

OPT-132-G read:

```text
speed: 13649.29 env steps/sec
wall: 13.504s
accepted preset: true
initial inference mode: direct_core
fused learner batch: true
borrowed render state: true
render-state copy steps: 0
lean trainer step: true
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
direct autoreset count: 0
actor wall: 4.073s
actor env runtime: 1.099s
observation: 1.984s
observation other: 1.558s
resident stack update: 0.029s
resident stack shift: 0.012s
resident autoreset stack reset: 0.031s
sample gate: 2.692s
slab/search: 2.769s
learner gate: 1.466s
```

OPT-132-H repeat read:

```text
speed: 11366.84 env steps/sec
wall: 16.216s
accepted preset: true
initial inference mode: direct_core
fused learner batch: true
borrowed render state: true
render-state copy steps: 0
lean trainer step: true
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
direct autoreset count: 0
actor wall: 4.924s
actor env runtime: 1.590s
observation: 2.495s
observation other: 2.059s
resident stack update: 0.034s
resident stack shift: 0.015s
resident autoreset stack reset: 0.034s
sample gate: 3.426s
slab/search: 2.892s
learner gate: 1.778s
```

Decision:

- OPT-132-G is a real accepted-proof row that beat OPT-104.
- OPT-132-H did not repeat it, so OPT-104 remains the accepted baseline.
- Do not claim the goal is closed from OPT-132-G alone.
- The resident stack split answered the immediate question: the clone-based
  stack shift is tiny, not the main wall gap.
- Park the ring-buffer resident stack unless a future row contradicts this.
- Local artifact comparison found no non-timing scalar drift on the report
  surface after ignoring run IDs and paths.
- Nested row results matched row seed `20260530`, sample seed `20260551`,
  learner seed `20260551`, refresh sample seed `20260533`, sample calls `22`,
  learner calls `22`, learner updates `22`, replay entries `180`, stored index
  rows `456602`, committed index rows `366548`, death rows `1050`, terminal
  rows `1050`, and terminal sample rows `167`.
- The report surface was still too weak for repeatability work because these
  seeds and work-shape fields were not all top-level fields.
- Code now threads repeatability fields through speed-row summary, Modal
  reports, and hash-bound evidence: row seed, sample seed base, last sample
  seed, last learner seed, last refresh sample seed, sample interval, sample
  batch, replay capacity, learner train steps, policy refresh interval,
  simulation count, actor trajectory checksums, terminal/autoreset/death
  checksums, sample-order checksums, and sample/learner counters.
- Next work is repeatability: explain why actor/env, observation-other, sample,
  and learner swung by much more than the resident stack under the same
  accepted proof fields.

## OPT-132-F: Sparse Resident Terminal Final Observation

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132f-h100-sparse-resident-final-observation-preset-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Evidence:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132f-h100-sparse-resident-final-observation-preset-samework-b1024a1-normal-unroll2-speed-20260602-r1/row_001_result.json.compact_coach_speed_row.evidence.json`

Read:

```text
status: complete
reason: accepted-fast-path preset row after sparse resident terminal final observation
speed: 12177.36 env steps/sec
wall: 15.136s
initial inference mode: direct_core
fused proof: true
borrowed render state: true
render-state copy steps: 0
render-state borrowed steps: 225
lean trainer step: true
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
direct autoreset count: 0
actor wall: 4.812s
actor autoreset: 3.071s
actor env runtime: 1.486s
observation: 2.357s
observation other: 1.988s
renderer render: 0.350s
sample gate: 3.019s
slab/search: 2.842s
learner gate: 1.469s
```

Decision:

- Rejected as the final full-loop speed win because it still loses to OPT-104:
  `12177.36` versus `12689.38` env steps/sec, wall `15.136s` versus
  `14.5255s`.
- Accepted as a useful same-work improvement: versus OPT-132-E it gained
  `+634.97` env steps/sec and recovered about `0.833s` wall.
- The sparse resident terminal final-observation change was correct locally and
  safe in the H100 preset row: accepted-fast-path proof fields stayed intact.
- Remaining targets versus OPT-117: observation-other about `+0.552s`, actor
  env runtime about `+0.372s`, learner about `+0.139s`.
- Sample and slab/search are not the next target in this row.

## OPT-132-E: Accepted Fast-Path Breakdown Row

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132e-h100-actor-observation-breakdown-preset-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Evidence:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132e-h100-actor-observation-breakdown-preset-samework-b1024a1-normal-unroll2-speed-20260602-r1/row_001_result.json.compact_coach_speed_row.evidence.json`

Read:

```text
status: complete
reason: accepted-fast-path preset row with actor/observation child timers
speed: 11542.39 env steps/sec
wall: 15.969s
initial inference mode: direct_core
fused proof: true
borrowed render state: true
render-state copy steps: 0
render-state borrowed steps: 225
lean trainer step: true
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
direct autoreset count: 0
actor wall: 5.032s
actor step: 4.945s
actor autoreset: 3.221s
actor env runtime: 1.548s
observation: 2.472s
observation other: 2.103s
renderer render: 0.350s
sample gate: 3.121s
slab/search: 3.018s
learner gate: 1.666s
```

Decision:

- Rejected as a full-loop speed win.
- Accepted as the current clean diagnosis row because fast-path proof passed and
  actor/observation child timers are present in report, result, and evidence.
- Compared with OPT-117, the remaining wall gap is about `+1.301s`.
- The largest measured target is observation-other: `2.103s` versus about
  `1.436s`, about `+0.667s`.
- Actor env runtime is next: `1.548s` versus `1.113s`, about `+0.435s`.
- Learner is third: `1.666s` versus `1.330s`, about `+0.336s`.
- Actor autoreset is not the main remaining explanation in this row:
  `3.221s` versus `3.173s`, about `+0.049s`.
- Renderer render is not the explanation: both rows are `0.350s`.
- Sample is still not the target because OPT-132-E sample is faster than
  OPT-117.

## OPT-132-D: Accepted Fast-Path Preset Row

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132d-h100-accepted-fast-path-preset-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json`

Read:

```text
status: complete
reason: prove the launcher preset carries the full accepted fast-path bundle
speed: 10538.13 env steps/sec
wall: 17.491s
initial inference mode: direct_core
fused proof: true
borrowed render state: true
render-state copy steps: 0
render-state borrowed steps: 225
lean trainer step: true
terminal sample/target rows: 167/167
normal-death gate: true
truncations: 0
host fallback: 0
direct autoreset count: 0
actor wall: 6.008s
observation: 2.299s
sample gate: 3.145s
slab/search: 3.238s
learner gate: 2.089s
```

Decision:

- Rejected as a full-loop speed win.
- Useful as the clean current-code regression row because the fast-path flags
  are present.
- The remaining regression is not sample gate. Compared with OPT-117, actor is
  about `+1.486s`, learner about `+0.759s`, observation about `+0.499s`, and
  slab/search about `+0.358s`; sample is faster.
- Next target: actor plus observation, then learner.
- The launcher now has a remote-result guard so preset rows fail if the returned
  result does not actually carry the accepted fast-path fields.

## OPT-119: Local Owner Replay/Learner Wiring Proof

Artifact:

`artifacts/local/tmp_owner_search_real_smoke/optimizer-owner-search-real-local-smoke-20260602c/row_001_result.json`

Read:

```text
hardware: local CPU tiny smoke
status: complete
owner replay append enabled: true
owner append entries staged/submitted: 4 / 4
owner learner updates: 2
owner train requests: 2
parent sample calls: 0
parent learner updates: 0
model-state bytes returned to parent: 0
worker owns replay/model state: true / true
search consumed owner learner update: true
```

Decision:

- This proves production speed-row wiring can run real owner replay append,
  owner learner train, owner-ref publication, and owner-side compact Torch search
  refresh with parent sample/learner gates disabled.
- It is not H100 speed evidence and not the final device-resident replay path.
  Replay append entries are host-materialized full compact entries for this
  blocker row. If the H100 row shows host staging is the blocker, move compact
  Torch two-phase replay retention fully into the owner.

## OPT-119: Tiny H100 Owner Replay/Learner Wiring Proof

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-owner-search-compacttorch-owner-replay-learner-smoke-20260602-r2-strict/row_001_result.json`

Read:

```text
hardware: H100
death mode: profile_no_death
shape: B2/A1, 4 measured, 1 warmup, unroll1
speed: 4.04 env steps/sec
wall:  1.9793s
owner append staged/submitted/appended: 4 / 4 / 4
owner pending appends: 0
owner train interval: 2
owner train requests: 2
owner learner updates: 2
parent sample calls: 0
parent learner updates: 0
root-observation bytes: 0
request/result CUDA tensors: 0 / 0
model-state bytes/returns/snapshots: 0 / 0 / 0
bridge: cuda:0, 65536 bytes, generation 5
parent wait: 0.0874s
worker wall: 0.0843s
worker replay append: 0.00008s
worker learner train: 0.0471s
worker search refresh: 0.0316s
GPU util mean/max: 0.36% / 6%
```

Decision:

- This proves the production owner replay/learner path runs on H100/CUDA with
  owner-owned replay/model state and parent sample/learner gates disabled.
- It is tiny `profile_no_death` wiring proof only. It is not speed evidence and
  it does not satisfy the normal-death terminal contract.
- The next proof is normal-death owner-search telemetry, then a small
  normal-death H100 owner replay/learner blocker row.

## OPT-119: Local Fixed-Shape Owner-Search Wiring Row

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt119-local-owner-search-slab-proxy-fixedshape-smoke-20260602-r3/compact_coach_speed_row_smoke_report.json`

Shape:

```text
local CPU
owner_search_slab_proxy
inner search: fixed_shape_search_owner
profile row, not same-shape H100
```

Read:

```text
ok: true
speed: 8.399 env steps/sec
owner_search_slab_proxy_requested: true
owner_search_inner_search_service_kind: fixed_shape_search_owner
owner_search_inner_search_service_impl: fixed_shape_batched_search_owner_profile_only_v0
owner_search_compact_torch_resident_root_bridge_ready: false
compact_owner_search_boundary_kind: worker_search_parent_slab_commit
root observation bytes sent: 0
request/result CUDA tensor count: 0/0
committed replay index rows: 16
stored replay index rows: 16
learner gate updates: 3
```

Decision:

- This proves the speed-row wiring can select owner-search, run through the
  full profile path, use two-phase compact replay flush, store replay rows, and
  update the learner.
- It is not speed evidence. It is local CPU and fixed-shape search.
- It is not compact Torch evidence. The compact Torch owner-search resident-root
  blocker was later closed locally by the interim bridge row below.

## OPT-119: Compact Torch Owner-Search Fail-Fast Row

Artifact:

`/private/tmp/curvyzero-owner-search-failfast/opt119-local-owner-search-compacttorch-failfast-20260602/compact_coach_speed_row_smoke_report.json`

Shape:

```text
local CPU
owner_search_slab_proxy
inner search: compact_torch_search_service
fail-fast guard row
```

Read:

```text
ok: false
problem: owner-search compact Torch resident root bridge is not implemented
owner_search_slab_proxy_requested: true
owner_search_inner_search_service_kind: compact_torch_search_service
owner_search_inner_search_service_impl: compact_torch_device_tree_fixed_shape_v0
owner_search_compact_torch_resident_root_bridge_ready: false
compact_torch_memory_format_applies_to_search_service: true
```

Decision:

- This is a guard, not a speed row.
- The builder now records the compact Torch owner-search blocker cleanly instead
  of producing only a traceback.
- Do not bypass this guard with host-zero roots or CUDA tensor IPC. The owner
  process must create or own the resident observation tensor.

## OPT-119: Compact Torch Owner-Search Interim Bridge Row

Artifact:

`/private/tmp/curvyzero-owner-search-resident-bridge/opt119-local-owner-search-compacttorch-resident-bridge-20260602-r6/compact_coach_speed_row_smoke_report.json`

Shape:

```text
local CPU
owner_search_slab_proxy
inner search: compact_torch_search_service
steps: 4 measured, 1 warmup
batch: 2
num simulations: 1
```

Read:

```text
ok: true
speed: 8.99 env steps/sec
owner_search_inner_search_service_impl: compact_torch_device_tree_fixed_shape_v0
owner_search_compact_torch_resident_root_bridge_ready: true
compact_owner_search_resident_root_bridge_ready: true
bridge kind: shared_memory_host_root_to_owner_resident_tensor_v1
bridge device: cpu
bridge bytes: 65536
bridge generation: 5
root observation bytes sent: 0
request/result CUDA tensor count: 0/0
model state bytes returned: 0
committed replay index rows: 16
stored replay index rows: 16
learner gate updates: 3
```

Decision:

- This fixes the clean fail-fast blocker with an honest interim bridge.
- It is not speed evidence: local CPU, tiny shape, and not H100.
- It is not the final zero-H2D render-state bridge. The row explicitly reports
  the host-root-to-owner-resident tensor copy.
- It is not fully device-resident replay. The current owner-search proxy returns
  host search arrays to the parent slab, and the parent slab uploads replay
  payload tensors.

## OPT-119: H100 Compact Torch Owner-Search Interim Bridge Row

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-owner-search-compacttorch-resident-bridge-smoke-20260602-r1/compact_coach_speed_row_modal_report.json`

Shape:

```text
H100
owner_search_slab_proxy
inner search: compact_torch_search_service
steps: 4 measured, 1 warmup
batch: 2
num simulations: 1
death mode: profile_no_death
```

Read:

```text
ok: true
speed: 7.21 env steps/sec
training wall: 1.1095s
bridge kind: shared_memory_host_root_to_owner_resident_tensor_v1
bridge device: cuda:0
bridge bytes: 65536
bridge generation: 5
root observation bytes sent: 0
request/result CUDA tensor count: 0/0
model state bytes returned: 0
committed replay index rows: 16
stored replay index rows: 16
learner gate updates: 3
owner parent wait: 0.0060s
owner worker wall: 0.0044s
H100 mean/max utilization: 0.76%/9%
worker owns replay state: false
worker owns model state: false
```

Decision:

- This proves the interim resident-root bridge runs on H100/CUDA without
  root-observation request bytes, CUDA tensors in process messages, host-zero
  roots, or parent model-state returns.
- It is not speed evidence: tiny shape, profile-no-death, and very low H100 use.
- It is not the final owner-owned trainer structure. In this row, the owner owns
  search/root resolution only; replay commit and learner updates still happen
  through the parent slab path.
- Next measurement requires H100 proof of the production owner replay/learner
  wiring that already passed locally: `worker_owns_replay_state=true`,
  `worker_owns_model_state=true`, owner learner updates positive, parent
  sample/learner calls zero, and final search consuming the final owner learner
  update.

## OPT-119: Scalar-Ref Same-Shape Row, Correct But Slow

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-scalarref-persistent-fused-sameshape-r2-autoreset-union-20260601/row_001_result.json`

Shape:

```text
H100
B1024/A1
180 measured, 45 warmup
normal death
compact Torch search, direct-core initial inference
sample interval 8
sample batch 512
replay capacity 4096
learner unroll 2
fused host-provider learner batch
persistent compact render-state buffer
scalar_ref_v1 append transport
```

Read:

```text
speed: 2245.36 env steps/sec
wall:  82.0892s
request bytes: 42.4MB
result bytes: 353.3MB
sample-learner wait: 55.1s
sample gate: 5.8216s
learner gate: 1.5534s
actor step wall: 17.7986s
primary residual: 50.6202s
H100 mean/max utilization: 5.1% / 27%
terminal/death rows: 1050
terminal sample rows: 167
terminal final observation before autoreset: true
normal-death gate: true
```

Decision:

- The row is same-shape and correct.
- It does not beat OPT-104. It is about `5.65x` slower on the accepted speed
  currency.
- Request bytes are no longer in the old GB class, so the immediate blocker is
  not the scalar-ref request path.
- H100 utilization is still low, so H200/B200 is not the next lever.
- The measured blocker is the Python child-process result/wait path: the parent
  waits `55.1s` and receives `353.3MB` of result/model-state payload.
- Next work is either to shrink/remove that result path or to park this route
  and use a persistent owner/fixed-buffer structure where model/search state
  does not bounce through Python results.

## OPT-119: Model-State Scheduler Blocker Row

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-scalarref-persistent-fused-byte-probe-r4-modelstate-scheduler-20260601/row_001_result.json`

Compared against:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-scalarref-persistent-fused-byte-probe-r3-evidence-host-provider-20260601/row_001_result.json`

Shape:

```text
H100
B128/A1
33 measured, 4 warmup
normal death
compact Torch search
sample interval 4
sample batch 64
replay capacity 512
learner unroll 2
fused host-provider learner batch
persistent compact render-state buffer
scalar_ref_v1 append transport
```

Read:

```text
r3 result bytes: 160.5MB
r4 result bytes: 128.5MB
r3 wait: 13.66s
r4 wait: 12.41s
r3 model-state return/omit: 5/3
r4 model-state return/omit: 4/4
r3 speed: 256.25 env steps/sec
r4 speed: 282.92 env steps/sec
r4 request bytes: 1.21MB
r4 terminal/death rows: 2
r4 terminal sample rows: 1
r4 normal-death gate: true
```

Decision:

- The scheduler fixes are real: they remove one duplicate model-state return
  on this small H100 row.
- The improvement is not large enough to make the Python child-process route
  the expected speed path.
- Do not run a same-shape H100 speed row for this exact route yet.
- Next implementation should remove full model/search state from Python result
  payloads through a versioned snapshot handle, persistent publish buffer, or
  colocated persistent owner.

## OPT-119: Snapshot-File Model-State Transport Rows

Local validation:

```text
uv run pytest tests/test_compact_owned_loop.py \
  tests/test_source_state_hybrid_observation_profile.py::test_compact_owned_loop_refreshes_separate_search_worker_after_muzero_update -q

28 passed, 2 warnings

uv run ruff check \
  src/curvyzero/training/compact_owned_loop.py \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  scripts/build_compact_coach_speed_row_smoke.py \
  scripts/run_compact_coach_speed_row_modal_smoke.py \
  src/curvyzero/infra/modal/compact_coach_speed_row.py \
  tests/test_compact_owned_loop.py \
  tests/test_source_state_hybrid_observation_profile.py

All checks passed.
```

H100 artifacts:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-scalarref-persistent-fused-byte-probe-r5-snapshot-file-20260601/row_001_result.json`

`artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-scalarref-persistent-fused-byte-probe-r6-snapshot-file-request-window-20260601/row_001_result.json`

Shape:

```text
H100
B128/A1
33 measured, 4 warmup
normal death
compact Torch search
sample interval 4
sample batch 64
replay capacity 512
learner unroll 2
fused host-provider learner batch
persistent compact render-state buffer
scalar_ref_v1 append transport
snapshot_file_v1 model-state transport
```

Read:

```text
r5 speed: 361.07 env steps/sec
r5 wall: 11.70s
r5 request bytes: 1.21MB
r5 ordinary result bytes: 132KB
r5 parent wait: 9.40s
r5 snapshot return/load count: 4/4
r5 snapshot publish/load bytes: 128.3MB / 128.3MB
r5 snapshot publish/load sec: 0.129s / 0.148s
r5 H100 mean/max utilization: 1.86% / 12%

r6 speed: 301.84 env steps/sec
r6 wall: 13.99s
r6 request bytes: 1.21MB
r6 ordinary result bytes: 132KB
r6 parent wait: 11.66s
r6 snapshot return/load count: 4/4
r6 snapshot publish/load bytes: 128.3MB / 128.3MB
r6 snapshot publish/load sec: 0.133s / 0.179s
r6 H100 mean/max utilization: 1.25% / 11%

r6 correctness: normal-death gate true, terminal/death rows 2,
terminal sample rows 1, final refresh update 8/8, append host observations 0,
resident snapshot transfer 0, provider missing history 0, materialized entries 32
```

What it proves:

- `snapshot_file_v1` can publish worker model state outside the ordinary Python
  result payload.
- Parent code can load, delete, and apply the snapshot.
- Local-process search refresh accepts the snapshot handle as returned worker
  state and consumes the latest learner update.
- The artifacts now expose model-state transport kind, snapshot return count,
  snapshot publish/load bytes, and snapshot publish/load seconds.

Decision:

- This is correctness and instrumentation proof, not a speed win.
- Ordinary result bytes fell, but parent wait stayed high. Snapshot write/load
  time was not the main cost.
- r6 did not reduce snapshot return count because final refresh forcing still
  asks extra pending jobs to publish model state.
- Local code now reports the missing timing buckets for the next row: worker
  job wall time, replay preparation, sample, learner, model-state
  prepare/fn/clone/digest, public-result conversion, and result pickle.
- The next fix is persistent ownership of replay, learner state, and model
  publication. Do not spend another same-shape H100 row on file-transport-only
  polishing.

## OPT-119: Worker-Timing Snapshot Row

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-scalarref-persistent-fused-byte-probe-r7-worker-timing-20260602/row_001_result.json`

Shape:

```text
H100
B128/A1
33 measured, 4 warmup
normal death
compact Torch search
sample interval 4
sample batch 64
replay capacity 512
learner unroll 2
fused host-provider learner batch
persistent compact render-state buffer
scalar_ref_v1 append transport
snapshot_file_v1 model-state transport
```

Read:

```text
speed: 370.09 env steps/sec
wall: 11.41s
parent wait: 9.23s
worker job wall: 3.76s
worker inner job wall: 3.04s
worker replay prepare: 0.71s
worker sample: 0.56s
worker learner: 2.48s
worker model-state prepare/fn/clone/digest: 0.326s / 0.002s / 0.055s / 0.143s
worker result public/pickle: 0.00007s / 0.00060s
ordinary result bytes: 138KB
snapshot publish/load bytes: 128.3MB / 128.3MB
snapshot publish/load sec: 0.125s / 0.155s
search refresh sec: 0.173s
H100 mean/max utilization: 1.85% / 11%
```

Correctness:

```text
submit/completed: 8/8
pending: 0
final drain: true
worker owns replay/model: true/true
full replay snapshot submit count: 0
request/result CUDA tensor count: 0/0
append host observation bytes: 0
append resident snapshot bytes: 0
normal-death gate: true
terminal/death rows: 2
terminal sample rows: 1
final search refresh update: 8/8
```

Decision:

- The row is valid evidence for the small blocker shape.
- It is not speed evidence against OPT-104.
- Result pickle, ordinary result size, snapshot write/load, and search refresh
  are too small to explain the slow wall time.
- The parent waits because one child-process worker owns sample/learner work
  while search still needs parent-visible model publication.
- Do not keep polishing `snapshot_file_v1`. The next structure must make
  search consume an owner-managed model publish handle or move search behind
  the same persistent owner boundary.

## OPT-119: Owner-Search Mock

Local evidence:

```text
uv run pytest tests/test_compact_split_transport_mock.py -q
2 passed

uv run pytest \
  tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_validates_local_process_owner_ref_model_transport \
  tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_rejects_local_process_without_model_state_apply -q
2 passed
```

Shape:

```text
parent/actor side sends root slot ids and replay append deltas
one process owner owns search state, replay state, learner model state,
and model publication
owner returns actions plus owner refs
no model state dict, snapshot payload, observation tensor, or CUDA tensor
crosses the boundary
```

Read from the test:

```text
steps: 8
search requests/results: 9/9
final refresh request id: last completed request id
actor steps while pending: >0
policy lag max: >0
pending at end: 0
final drain in wall: true
model-state bytes returned: 0
model-state return count: 0
root observation bytes sent: 0
CUDA tensor payload count: 0
owner-ref result count: search result count
worker owns replay/model/search: true/true/true
final model version: 4
final search model version: 4
search consumed final update: true
```

Decision:

- This is mock evidence only. It does not prove H100 speed.
- It is the first local proof of the aggressive structure sidecars recommended:
  search moves behind the same persistent owner as replay and learner, so the
  parent does not receive model state.
- Speed-row/report surfaces now expose owner-ref proof fields, and the local
  validator rejects owner-ref rows that return/apply parent model state, return
  snapshot state, omit the owner-ref count, or omit the final digest.
- The next real implementation should make this shape concrete enough for a
  small H100 blocker row, while preserving scalar-ref/provider correctness,
  normal death, fused learner batches, policy lag, and final drain.

## OPT-119: Owner-Search Production Slice

Local evidence:

```text
uv run pytest tests/test_compact_owner_search_service.py -q
6 passed

uv run ruff check \
  src/curvyzero/training/compact_owner_search_service.py \
  tests/test_compact_owner_search_service.py
All checks passed.
```

Shape:

```text
parent request carries root slot ids and replay append deltas
owner resolves root slot ids into CompactRootBatchV1 internally
owner runs search service and validates CompactSearchResultV1 identity
owner replay store ingests append deltas
owner learner publishes owner_ref_v1
owner-side search consumes owner_ref_v1 via refresh_model_owner_ref
ordinary result returns actions plus proof metadata
shared-memory root store lets the owner resolve slot ids from fixed buffers
without root observations in the request
slab proxy returns checked search arrays so CompactRolloutSlab can commit replay
rows through the existing path
```

Read:

```text
process worker PID distinct from parent: true
worker owns search/replay/model state: true/true/true
root_observation_bytes_sent: 0
model_state_return_count: 0
model_state_bytes: 0
model_state_snapshot_return_count: 0
request/result CUDA tensor count: 0/0
owner-ref digest reported: true
search consumed learner update: true
shared root-slot store sees changed slot contents across child-process requests: true
row-level terminal masks align to root slots: true
owner-search slab proxy drives next action: true
parent reconstructs CompactSearchResultV1 from worker arrays: true
parent slab replay commit explicitly labelled: worker_search_parent_slab_commit
replay rows committed by current slab path: true
search-result selected-action / visit-policy / root-value byte counters: present
lazy full-profile proxy initializes from first root batch: true
parent publish / submit / wait / wall timing fields: present
worker root-resolve / search / wall timing fields: present
profile top-level owner-search proof fields: present
local speed-row owner-search proof validation: present
local speed-row report owner-search proof fields: present
Modal H100 bundle owner-search proof pass-through: present
Modal caller report owner-search proof fields: present
owner-search request/result bytes and CUDA tensor counts: present
parent reconstructed CompactSearchResultV1 proof field: present
```

Decision:

- This is the first production-facing owner-search boundary, not a mock.
- The fixed shared root-slot provider applies the PufferLib-style lesson we need:
  stable buffers and small ids instead of per-request observation blobs.
- The slab proxy is honest progress: it moves search behind the owner worker and
  preserves replay commit through the existing slab. It does not yet prove
  worker-owned replay or H100 speed.
- It is still local only and not H100 speed evidence.
- Local and Modal speed-row/report surfaces now consume the proxy proof fields.
  The local validator rejects fake owner-search rows that omit zero-byte,
  checked-array, request/result, CUDA-count, parent reconstruction, replay-row,
  learner-update, or timing proof.
- Do not run a fake compact Torch owner-search row that swaps resident
  observations for host-zero roots. The interim resident-root bridge below closes
  the local blocker, but H100 proof is still required.

## OPT-119 Guardrail: Append Observation Payload Accounting

Change:

- `CompactOwnedLoopV1` now reports append-entry bytes, host observation bytes,
  resident replay snapshot count/bytes, compact-batch bytes, step-payload
  bytes, and render-state bytes for process sample+learner append transport.
- The speed-row validator rejects future `local_process` rows if host replay
  observations or resident replay snapshots still cross the process boundary.

Evidence:

```text
local code and tests only
ruff: passed on touched optimizer files
pytest: 44 focused compact-owned-loop and speed-row smoke tests passed
```

Decision:

- This is not speed evidence.
- It prevents a future row from hiding the rejected transfer under a new field
  name.
- The next valid row must drive host observation bytes and resident snapshot
  transfer to zero while preserving replay/learner/death counters and final
  drain.

## OPT-119 Local Code Proof: Scalar-Ref Replay Append Scaffold

Change:

- Added opt-in `scalar_ref_v1` replay append transport selection.
- `_CompactReplayRingV1.make_scalar_append_delta_entry()` copies scalar replay
  rows and action/reward/done fields but strips host observations, resident
  replay snapshots, and terminal final observation tensors.
- Scalar-ref append entries are marked as requiring an observation
  reconstruction provider. Sampling them fails closed before training.
- Profile/Modal/smoke summaries now expose the requested replay append
  transport kind.

Evidence:

```text
local code and tests only
ruff: passed on touched optimizer files
pytest: 113 compact-owned-loop/source-state tests passed
pytest: 29 speed-row smoke tests passed
```

Decision:

- This is not speed evidence.
- It proves the next append shape can stop carrying replay observations, but it
  cannot train until the learner side can reconstruct sampled observations,
  including sparse terminal final observations and unroll successor windows.

## OPT-119 Local Code Proof: Scalar-Ref Observation Provider Hook

Change:

- `_CompactReplayRingV1` now accepts a store-owned or per-sample observation
  provider.
- `sample_from_snapshot()` materializes scalar-ref entries through that provider
  before building the sample batch, and still fails closed if a provider is
  missing.
- Sample metadata and telemetry report provider use and materialized entry
  count.
- A focused local parity test proves a provider-backed scalar-ref terminal
  sample can match the durable host-observation sample for observation,
  next observation, action, action mask, reward, final reward, and done.

Evidence:

```text
local code and tests only
ruff: passed on touched optimizer files
pytest: 143 focused compact-owned-loop/source-state/speed-row smoke tests passed, 2 warnings
```

Decision:

- This is not speed evidence.
- This was not a trainable worker path by itself.
- Superseded by later local proofs: worker provider plumbing, bootstrap
  transport, and the renderer-backed provider now exist. The current missing
  proof is integrated correctness through the real speed-row path.

## OPT-119 Local Code Proof: Worker Provider Plumbing And Unroll2 Parity

Change:

- `CompactProcessSampleLearnerWorkerV1` now accepts a host-only observation
  provider factory.
- The process initializer constructs the provider once inside the child.
- The process worker materializes scalar-ref append entries before storing them
  in worker-owned replay and reports provider presence plus materialized-entry
  counts.
- Added a local process-worker test proving scalar-ref append transport can
  train with zero host-observation bytes and zero resident snapshot transfer
  when the worker owns a provider.
- Added scalar-ref plus provider unroll2 parity against durable resident replay
  for successor windows and unroll fields.

Evidence:

```text
local code and tests only
ruff: passed on compact_owned_loop and focused tests
pytest tests/test_compact_owned_loop.py -q: 18 passed, 2 warnings
pytest tests/test_source_state_hybrid_observation_profile.py -q: 98 passed, 2 warnings
pytest compact_owned_loop/source_state/speed_row_smoke focused suite: 145 passed, 2 warnings
```

Decision:

- This is not speed evidence.
- This proved worker-owned provider plumbing before the real renderer-backed
  provider existed.
- Later local proofs now cover row-sliced render-state facts, bootstrap
  history, and the first renderer-backed provider.

## OPT-119 Local Code Proof: Row-Sliced Render-State Facts

Change:

- Scalar-ref append entries now keep row-sliced render-state snapshots for the
  selected env rows.
- Host observations, resident replay snapshots, and terminal final observation
  tensors are still stripped from scalar-ref append entries.
- `CompactOwnedLoopV1` now counts render-state bytes separately from host
  observation bytes in process-worker append transport.

Evidence:

```text
local code and tests only
ruff: passed on touched optimizer files
pytest compact_owned_loop/source_state/speed_row_smoke focused suite: 146 passed, 2 warnings
```

Decision:

- This is not speed evidence.
- This is now consumed by the renderer-backed provider.
- It closes one prerequisite: the worker can receive compact render facts
  without hiding observation tensors under another field name.

## OPT-119 Local Code Proof: Provider Bootstrap Transport

Change:

- `CompactOwnedLoopV1` now queues scalar-ref provider bootstrap steps when the
  previous step is primed.
- The queue is bounded to the last four steps, matching the 4-frame observation
  stack.
- Bootstrap steps are stripped of host observations, resident replay snapshots,
  timestep objects, flat observations, target rewards, and final observation
  tensors.
- Process requests carry bootstrap steps separately from replay append entries.
- The process worker applies bootstrap to the observation provider before
  materializing scalar-ref append entries.
- Bootstrap step bytes, host observation bytes, resident snapshot bytes, and
  render-state bytes are reported separately from replay append bytes.

Evidence:

```text
local code and tests only
ruff: passed on compact_owned_loop and focused tests
pytest tests/test_compact_owned_loop.py -q: 19 passed, 2 warnings
pytest compact_owned_loop/source_state/speed_row_smoke focused suite: 147 passed, 2 warnings
```

Decision:

- This is not speed evidence.
- This closes the transport prerequisite: the worker/provider can receive
  bounded history-only render-state facts without adding replay rows, learner
  calls, or observation tensor transport.

## OPT-119 Local Code Proof: Renderer-Backed Scalar-Ref Provider

Change:

- Added `CompactReplayRendererBackedObservationProviderV1`.
- The provider consumes bootstrap history, renders row-sliced render-state
  facts, maintains the 4-frame stack, captures terminal final observations
  before autoreset, and resets stack rows from autoreset render-state facts.
- The `local_process` worker factory path now configures this provider when the
  replay append transport is `scalar_ref_v1`.
- The speed-row script now exposes
  `--compact-owned-loop-deferred-sample-learner-replay-append-transport-kind`
  and fails scalar-ref rows unless provider/bootstrap/materialization proof
  fields are present.
- The scalar-ref guard also rejects bootstrap resident snapshots, bootstrap
  replay rows, bootstrap learner calls, missing worker bootstrap, missing append
  render-state bytes, missing stack history, and materialized-entry mismatch.

Evidence:

```text
local code and tests only
ruff: passed on touched optimizer files
pytest tests/test_source_state_hybrid_observation_profile.py -q: 102 passed, 2 warnings
pytest tests/test_compact_owned_loop.py::test_compact_process_worker_uses_renderer_backed_scalar_ref_provider -q: 1 passed, 2 warnings
pytest tests/test_compact_coach_speed_row_smoke.py -q: 29 passed, 2 warnings
pytest focused source_state/compact_owned_loop/speed_row_smoke suite: 152 passed, 2 warnings
```

Decision:

- This is not speed evidence.
- This proves the provider can reconstruct locally and can be constructed in
  the process-worker path.
- The next proof is integrated correctness for a real scalar-ref speed-row
  invocation, then a small H100 correctness row.

## OPT-119 Local Integrated Proof: Scalar-Ref Speed-Row Runner

Change:

- The real local speed-row runner now exposes the nested compact-owned loop
  telemetry at top level, so scalar-ref provider/bootstrap/materialization
  proof fields cannot disappear before validation.
- The `local_process` worker receives `scalar_ref_v1` append entries plus
  bounded bootstrap render-state facts, constructs the renderer-backed provider
  in the child, materializes append entries before storing them in worker-owned
  replay, samples replay, trains the compact MuZero learner, and drains the
  final worker jobs.
- Host sample batches now preserve explicit next-target fields through concat.
- Provider materialization now marks reconstructed compact batches as
  host-backed, so the host learner path does not keep requiring resident replay
  snapshots after reconstruction.

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-scalarref-local-integrated-proof-20260601-r11-final-code/row_001_result.json
```

Read:

```text
local CPU correctness proof only
death mode: profile_no_death
search: device_target
learner unroll: 1
fused learner batch: false
env steps: 192
wall: 2.981102375s
worker kind: local_process
transport: scalar_ref_v1
submit/completed: 24/24
actor steps while pending: 23
request bytes: 95774795
append host observation bytes: 0
append resident snapshot count: 0
append render-state bytes: 88126464
provider present: true
provider bootstrap steps: 4
provider bootstrap render-state bytes: 7343616
missing stack history: 0
materialized entries / append entries: 24/24
trainer learner updates: 23
trainer sample batches: 24
```

Validation:

```text
ruff check scripts/build_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_smoke.py tests/test_source_state_hybrid_observation_profile.py: passed
pytest tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_surfaces_nested_loop_telemetry_for_scalar_ref_guard tests/test_source_state_hybrid_observation_profile.py::test_compact_replay_renderer_provider_captures_terminal_before_autoreset tests/test_source_state_hybrid_observation_profile.py::test_host_compact_sample_batch_preserves_explicit_next_targets_through_concat -q: 3 passed, 2 warnings
```

Decision:

- This is real integrated correctness progress.
- This is not speed evidence.
- This is not same-shape readiness because it is CPU, tiny,
  `profile_no_death`, `device_target`, unroll1, and non-fused.
- The next blocker is same-shape correctness: compact Torch search, CUDA
  learner, normal-death counters, unroll2 successor windows, policy-refresh
  metadata/digests, and either host unroll2 support or resident/fused
  scalar-ref compatibility.
- Watch request bytes. Observation bytes are zero, but render-state bytes are
  still large enough that the H100 row must report them honestly before any
  speed claim.

## OPT-119 Local Integrated Proof: Scalar-Ref Host Unroll2

Change:

- Host `SourceStateMultiplayerSampleBatchV0` now carries explicit unroll
  targets and terminal validity masks.
- The host compact replay sampler now builds unroll2 target windows instead of
  rejecting all non-resident samples with `num_unroll_steps > 1`.
- The speed-row summary now surfaces sample-gate unroll proof fields:
  explicit unroll targets, unroll target group count, unroll step count, and
  terminal-window support.

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-scalarref-local-unroll2-proof-20260601-r2-summary-fields/row_001_result.json
```

Read:

```text
local CPU correctness proof only
death mode: profile_no_death
search: device_target
learner unroll: 2
fused learner batch: false
env steps: 192
wall: 2.941655667s
worker kind: local_process
transport: scalar_ref_v1
submit/completed: 24/24
actor steps while pending: 23
request/result bytes: 95775348 / 470092
append host observation bytes: 0
append resident snapshot count: 0
append render-state bytes: 88126464
provider present: true
provider bootstrap steps: 4
provider bootstrap host-observation bytes: 0
missing stack history: 0
materialized entries / append entries: 24/24
explicit unroll targets: true
explicit unroll target groups: 11
sample-gate unroll steps: 2
terminal unroll windows supported: true
trainer learner updates: 22
trainer sample batches: 24
final drain included: true
```

Validation:

```text
ruff check scripts/build_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_smoke.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/multiplayer_source_state_target_rows.py tests/test_source_state_hybrid_observation_profile.py: passed
pytest tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_requires_and_emits_fused_learner_batch_proof tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_surfaces_nested_loop_telemetry_for_scalar_ref_guard -q: 2 passed, 2 warnings
pytest tests/test_source_state_hybrid_observation_profile.py::test_compact_replay_ring_host_scalar_ref_unroll2_builds_learner_batch tests/test_source_state_hybrid_observation_profile.py::test_compact_replay_ring_scalar_ref_provider_matches_durable_unroll2_sample -q: 2 passed, 2 warnings
```

Decision:

- This closes the host scalar-ref unroll2 blocker for the real local speed-row
  runner.
- This is still not speed evidence.
- This is still not same-shape readiness because it is CPU, tiny,
  `profile_no_death`, `device_target`, and non-fused.
- The next blocker is compact Torch search, CUDA/small-H100 correctness,
  normal-death counters, terminal final observations, and then resident/fused
  compatibility if required for a true OPT-104 speed row.

## OPT-119 Local Diagnostic: Compact Torch Scalar-Ref Host Unroll2

Change:

- The local-process model-state return cadence now uses completed learner
  updates plus pending jobs instead of request id alone. This avoids missing a
  learner-to-search refresh when replay unroll eligibility offsets request ids
  from actual learner updates.
- The compact Torch diagnostic passed after using a refresh-friendly tiny row.

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-scalarref-local-compacttorch-unroll2-diagnostic-20260601-r6-refreshfriendly/row_001_result.json
```

Read:

```text
local CPU blocker-finding diagnostic only
death mode: profile_no_death
search: compact_torch_search_service
initial inference: model_method
learner unroll: 2
fused learner batch: false
env steps: 88
wall: 2.706128417s
worker kind: local_process
transport: scalar_ref_v1
submit/completed: 3/3
request/result bytes: 38652350 / 67115
append host observation bytes: 0
append resident snapshot count: 0
append render-state bytes: 33047424
provider present: true
missing stack history: 0
materialized entries: 9
explicit unroll targets: true
sample-gate unroll steps: 2
refresh calls: 2
forced final refresh count: 3
last refresh update / learner updates: 3 / 3
post-refresh search metadata count: 5
post-refresh replay metadata count: 4
final drain included: true
```

Failed diagnostics:

```text
opt119-scalarref-local-compacttorch-unroll2-diagnostic-20260601-r2-interval2:
  direct_core local model path lacked callable model._representation/_prediction.

opt119-scalarref-local-compacttorch-unroll2-diagnostic-20260601-r3-modelmethod:
  local-process refresh required returned worker model state.
  Fixed by update-window-based model-state return cadence.

opt119-scalarref-local-compacttorch-unroll2-diagnostic-20260601-r4-modelstate-window
and r5-maxpending1:
  final refresh did not reach final learner update for that cadence.

opt119-scalarref-local-compacttorch-normal-unroll2-diagnostic-20260601-r1
and r2-b64:
  normal-death contract rejected the row because terminal_sample_row_count was 0.
```

Decision:

- Compact Torch plus scalar-ref host unroll2 is locally possible.
- This is not speed evidence and not H100/CUDA evidence.
- The final-refresh cadence must be chosen or fixed so the final learner update
  is consumed by search before any compact Torch H100 correctness row.
- Normal-death correctness still needs a terminal-producing row. The failed
  local normal-death diagnostics did not prove a scalar-ref terminal bug; they
  failed because no terminal samples were present.

## OPT-119 Local Proof: Scalar-Ref Terminal Sample Metadata

Change:

- Host scalar-ref sample concat now aggregates `terminal_sample_row_count`,
  `next_final_observation_row_count`, and host terminal-final-observation use
  across grouped sample batches.
- Sample telemetry now exposes host terminal-final-observation use.
- Terminal forcing was kept conservative: force a terminal row only when the
  random sample missed terminal evidence entirely. Do not replace a row that
  already contains terminal evidence through its unroll window.

Evidence:

```text
ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py tests/test_source_state_hybrid_observation_profile.py scripts/run_compact_coach_speed_row_modal_smoke.py: passed
pytest tests/test_source_state_hybrid_observation_profile.py -q: 104 passed, 2 warnings
```

Decision:

- This is not speed evidence.
- This proves the local scalar-ref host path can carry real terminal samples,
  terminal final observations, and terminal telemetry without changing valid
  resident terminal-window sampling semantics.
- The next required proof is the same fact in a terminal-producing normal-death
  H100 row.

## OPT-119 H100 Correctness Proof: Scalar-Ref Compact Torch CUDA Host Unroll2

Change:

- The Modal wrapper now exposes
  `--compact-owned-loop-deferred-sample-learner-replay-append-transport-kind`,
  so H100 rows can explicitly run `scalar_ref_v1` instead of accidentally
  using the old durable-entry transport.
- The host sample builder now converts compact index-row fields through
  `_small_array()`, so CUDA tensors are copied to CPU before NumPy conversion
  instead of failing with `can't convert cuda:0 device type tensor to numpy`.

Failed first attempt:

```text
opt119-h100-scalarref-compacttorch-profile-no-death-unroll2-r1-20260601
problem: TypeError: can't convert cuda:0 device type tensor to numpy
```

Passing artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-scalarref-compacttorch-profile-no-death-unroll2-r2-cuda-host-fix-20260601/row_001_result.json
```

Read:

```text
H100 correctness proof only
death mode: profile_no_death
search: compact_torch_search_service
initial inference: model_method
learner device: cuda
worker kind: local_process
transport: scalar_ref_v1
learner unroll: 2
fused learner batch: false
env steps: 88
wall: 12.958003603s
speed: 6.791169588 env steps/sec
submit/completed: 3/3
actor steps while pending: 2
worker CUDA: cuda:0
request/result CUDA tensor counts: 0/0
request/result bytes: 38653195 / 64219787
append host observation bytes: 0
append resident snapshot count/bytes: 0 / 0
append render-state bytes: 33047424
bootstrap host observation bytes: 0
bootstrap resident snapshot count: 0
provider present: true
missing stack history: 0
materialized entries: 9
explicit unroll targets: true
sample-gate unroll steps: 2
refresh calls: 2
last refresh update / learner updates: 3 / 3
search/replay metadata lag to final update: 0 / 0
final drain included: true
GPU: NVIDIA H100 80GB HBM3
GPU util mean/max: 0.14% / 1.0%
```

Validation:

```text
ruff check scripts/run_compact_coach_speed_row_modal_smoke.py src/curvyzero/training/source_state_hybrid_observation_profile.py: passed
pytest tests/test_source_state_hybrid_observation_profile.py -q: 104 passed, 2 warnings
```

Decision:

- This closes the small H100 CUDA scalar-ref compact Torch host-unroll2
  correctness blocker.
- This is not speed evidence.
- This is not same-shape readiness because it is tiny, `profile_no_death`, and
  non-fused.
- The next correctness blocker is terminal-producing normal death.
- PufferLib supports the next structural direction if payloads remain large:
  fixed/shared/pinned buffers and small slot references, not larger GPUs or a
  wholesale training-stack port.

## OPT-119 H100 Correctness Proof: Scalar-Ref Normal Death

Change:

- Scalar-ref append entries now carry terminal row indices derived from their
  scalar done/final-observation fields.
- Host and resident unroll builders now use a terminal-window hint so terminal
  padding does not fail row-identity checks.
- The normal-death contract now accepts scalar-ref host/provider terminal-final
  observation proof, not only resident-device final-observation proof.
- Contract failure messages now include the missing proof field.

Failed probes:

```text
r1/r2: hidden contract failure, then detailed as compact unroll player identity mismatch
r4: host final-observation proof surfaced, then contract wanted resident final proof only
r5/r6/r7/r9: remaining unroll identity/hint gaps
r8: contract wanted device replay terminal rows only
```

Passing artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-scalarref-compacttorch-normal-unroll2-terminal-r10-done-fallback-hint-20260601/row_001_result.json
```

Read:

```text
H100 correctness proof only
death mode: normal
search: compact_torch_search_service
initial inference: direct_core
learner device: cuda
worker kind: local_process
transport: scalar_ref_v1
learner unroll: 2
fused learner batch: false
env steps: 4224
wall: 19.801465356s
speed: 213.317546 env steps/sec
submit/completed: 8/8
actor steps while pending: 18
terminal/death rows: 2/2
terminal sample rows: 1
terminal final observations before autoreset: true
terminal final observation rows: 2
terminal unroll target rows: 1
terminal target mode: stock_terminal_no_bootstrap_return_discount_1.0
normal-death contract gate: true
append entries/materialized entries: 32/32
request/result CUDA tensor counts: 0/0
request/result bytes: 1072526418 / 160567474
append host observation bytes: 0
append resident snapshot count/bytes: 0 / 0
append render-state bytes: 998078104
bootstrap steps: 4
bootstrap host observation bytes: 0
bootstrap resident snapshot count: 0
bootstrap render-state bytes: 73436160
provider present: true
missing stack history: 0
learner updates/final refresh update: 8/8
search/replay metadata lag to final update: 0 / 1
final drain included: true
GPU util mean/max: 2.38% / 16.0%
```

Validation:

```text
ruff check touched optimizer files: passed
pytest tests/test_source_state_hybrid_observation_profile.py tests/test_compact_death_terminal_contract.py tests/test_compact_coach_speed_row_smoke.py -q: 152 passed, 2 warnings
```

Decision:

- This closes the scalar-ref normal-death correctness blocker.
- This is not speed evidence.
- The next blocker is structure: resident/fused scalar-ref compatibility if
  needed for fair OPT-104 shape, and fixed/shared/pinned buffers if
  render-state/request bytes remain large.

## OPT-119 Local Proof: Sample+Learner Worker Boundary

Change:

- `CompactOwnedLoopV1` can now submit replay sample plus learner work through
  an injected worker boundary.
- The existing same-process path is preserved as the default
  `in_process_thread` adapter.
- Reports now carry worker kind/resource, whether the worker resource is
  distinct from actor/search, record-step count, replay append count, actor
  steps while sample+learner work was pending, policy lag, request ids,
  snapshot versions, and final drain.

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-local-split-proof-fields-smoke-20260601/compact_coach_speed_row_smoke_report.json
```

Read:

```text
local only
worker kind: in_process_thread
resource distinct from actor/search: false
record steps: 6
appended replay entries: 6
deferred sample+learner submit/completed: 6/6
actor steps while sample+learner pending: 4
policy lag current/max: 0/2
last submitted/completed request id: 6/6
final deferred drain included in measured wall: true
```

Decision:

- This is a correctness/proof-surface step, not a speed win.
- It prevents future rows from calling the default thread adapter a real
  actor/search/replay/learner split.
- The next implementation step was a real worker adapter backed by a distinct
  process resource.

## OPT-119 Local Proof: Process Sample+Learner Worker

Change:

- Added a default-off `local_process` sample+learner worker.
- The process worker samples from a replay snapshot instead of pickling the
  live actor replay store.
- Reports now expose actor/search PID, worker PID, process scope, hardware
  distinctness, request-id closure, model-state apply count, and final drain.
- The validator fails local-process rows if they do not prove process
  distinctness, actor steps while pending, policy lag, request-id closure, and
  final drain.

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-local-process-sample-learner-worker-smoke-20260601/compact_coach_speed_row_smoke_report.json
```

Read:

```text
local CPU smoke only
worker kind: local_process
resource scope: process
resource distinct from actor/search: true
hardware resource distinct from actor/search: false
actor/search pid: 57964
last worker pid: 57969
deferred sample+learner submit/completed: 6/6
actor steps while sample+learner pending: 5
policy lag current/max: 0/2
model state apply count: 5
final deferred drain included in measured wall: true
```

Decision:

- This is an architecture/proof-surface step, not a speed win.
- It proves an OS-process split can run through the local speed-row path.
- It does not prove separate GPU/hardware use.
- The same-shape H100 attempt failed because the request still contained
  parent-owned CUDA tensors.
- Keep `local_process` as a CPU/process proof only.

## OPT-119 H100 Local Process: Rejected Transport

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-normal-death-local-process-sample-learner-compacttorch-r1-20260601/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-normal-death-local-process-spawn-sample-learner-compacttorch-r2-20260601/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-normal-death-local-process-spawn-cudaipc-sample-learner-compacttorch-r3-20260601/compact_coach_speed_row_modal_report.json
```

Read:

```text
all three rows: BrokenProcessPool
r2/r3 stderr: child process died while unpickling CUDA tensors
root cause: Python multiprocessing tried CUDA IPC for parent-owned replay/model tensors
```

Decision:

- Do not rerun this H100 lane unchanged.
- The next transport must send no CUDA tensors through Python multiprocessing.
- `local_process` now fails fast when a request contains CUDA tensor payloads.

## OPT-119 Host-Only Split Transport Mock

Change:

- Added a small process-service mock for the target dataflow.
- Actor/search appends replay entries and can keep stepping while work is
  pending.
- The worker owns replay/model state and receives only host-safe values.
- The report measures request closure, policy lag, final drain, transport
  bytes, distinct process identity, and CUDA payload count.

Artifact:

```text
artifacts/local/curvytron_compact_split_transport_mock_results/opt119-host-only-split-transport-mock-20260601/mock_report.json
```

Read:

```text
local mock only
transport: host_only_process_service_mock
steps: 12
submit/completed: 12/12
actor steps while pending: 11
policy lag max: 2
pending at end: 0
final drain included in wall: true
final drain sec: 0.0254
CUDA tensor payload count: 0
worker PID distinct from actor: true
worker owns replay/model state: true/true
request/result bytes: 5328/2208
wall: 0.2021s
```

Decision:

- This is not throughput evidence.
- It proves the intended ownership and accounting shape before the real H100
  row.
- The next ledger section shows the same host-only request/result shape wired
  into the real local-process compact replay/learner path.
- Before another H100 speed claim, the real row must also report transfer or
  encode/decode time, parent wait time, worker wall time, worker resource/GPU
  identity if remote, and proof that search consumed the returned worker
  update.

## OPT-119 Local Process Factory-Bootstrap Worker Smoke

Change:

- The actual `local_process` worker is now prepared once and owns learner/model
  state in the child process.
- For compact MuZero, worker bootstrap uses a host-side learner factory instead
  of passing the parent learner object.
- Per-job submit sends a host-side transport request plus replay snapshot, not
  the learner object.
- Speed-row proof fields now include request/result CUDA counts, request/result
  bytes, worker-owned model state, worker initialization count, and worker
  completed count.

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-local-process-factory-bootstrap-smoke-20260601/compact_coach_speed_row_smoke_report.json
```

Read:

```text
local CPU smoke only
worker kind: local_process
bootstrap source: factory
resource scope/start: process/spawn
submit/completed: 6/6
actor steps while pending: 5
policy lag max: 2
model-state apply count: 5
last model state applied: true
request host-only: true
request/result CUDA tensor count: 0/0
request/result bytes: 8290429/113854
worker owns model state: true
worker model initialized count: 1
worker completed count: 6
final drain included: true
```

Decision:

- This replaces the earlier CPU process proof as the current local process
  evidence.
- It is still not H100 throughput evidence.
- The next H100 proof used this factory bootstrap. It passed correctness but
  exposed a result-payload problem.

## OPT-119 Tiny H100 Proof: Factory Bootstrap Works, Payload Is Too Large

Change:

- The `local_process` worker bootstrapped from a host-side factory on H100.
- The child process owned CUDA learner/model state.
- Per-job request/result payloads stayed host-only.

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-factory-bootstrap-hostsample-interval2-tiny-20260601/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-factory-bootstrap-hostsample-interval2-tiny-20260601/remote_bundle.json
```

Read:

```text
tiny H100 correctness proof only
shape: B2/A1, 8 measured, 2 warmup, profile_no_death, device_target
worker kind: local_process
resource scope/start: process/spawn
worker CUDA device: cuda:0
bootstrap source: factory
submit/completed: 4/4
actor steps while pending: 5
policy lag max: 2
learner calls/updates: 4/4
model-state apply count: 4
request/result host-only: true/true
request/result CUDA tensor count: 0/0
snapshot host clone used: true
request/result bytes: 7215312/128391128
worker owns model state: true
worker model initialized count: 1
worker completed count: 4
final drain included: true
speed: 0.95 env steps/sec
```

Decision:

- The old CUDA IPC failure is solved for this tiny path.
- This is not speed evidence and not comparable to OPT-104.
- The current transport returns full model state for each completed job. A
  `128MB` result payload on a tiny row is not a viable expected speed path.
- Next: ordinary job results should return small telemetry and digest fields;
  model state should publish only when search refresh needs it or at final
  drain.

## OPT-119 Local Code Proof: Cadence-Based Model-State Return

Change:

- Added a `return_model_state` bit to sample+learner requests and process
  transport requests.
- The process worker can omit model state on ordinary jobs.
- The loop reports model-state return count, omitted count, last returned flag,
  and return interval.
- The speed-row builder sets the return interval from `policy_refresh_interval`.
- The profile forces model state for the last usable refresh window.
- Local-process search refresh now fails closed if it tries to refresh without
  returned worker model state.

Validation:

```text
ruff: passed
pytest focused: 134 passed, 2 warnings
```

Decision:

- This is local correctness evidence, not H100 throughput evidence.
- The next row should be a small H100 compact Torch normal-death correctness
  row. It must prove result bytes shrink, request/result CUDA tensor counts
  stay `0/0`, final refresh consumes the returned worker state, and final drain
  is included.

## OPT-119 Small H100 Proof: Cadence Works, Replay Snapshot Transport Is Too Large

Change:

- Compact Torch policy-refresh validation now treats async replay metadata
  honestly. Search must consume the final returned worker update. Replay rows
  may lag because the rollout slab commits replay for the previous search.
- Reports now include search/replay metadata update counts and lag to the final
  refreshed learner update.
- Launchers now reject compact Torch deferred sample+learner rows that cannot
  produce a post-refresh actor/search step.

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-cadence-model-state-compacttorch-normal-nofused-unroll2-r5-async-replay-lag-20260601/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-cadence-model-state-compacttorch-normal-nofused-unroll2-r5-async-replay-lag-20260601/remote_bundle.json
```

Read:

```text
small H100 correctness row only
B128/A1, 33 measured, 4 warmup, normal death, compact Torch, no fused batch
speed: 197.1053 env steps/sec
wall: 21.4302s
submit/completed: 8/8
actor steps while pending: 18
policy lag max: 2
worker kind: local_process
worker CUDA: cuda:0
model-state return/omit: 4/4
model-state apply count: 4
request/result host-only: true/true
request/result CUDA tensor count: 0/0
request bytes: 2750079669
result bytes: 128469062
search metadata update/final lag: 8/0
replay metadata update/final lag: 7/1
learner updates: 8
final drain included: true
terminal/death rows: 2
truncations: 0
```

Decision:

- The old full-model-state-every-result path is fixed enough for correctness.
- The row is not speed evidence and is intentionally not comparable to OPT-104.
- The next blocker is request transport. Sending a replay snapshot per job
  produced `2.75GB` of request bytes on a small row.
- Do not run the same-shape H100 speed row yet. Build worker-owned replay state
  or compact replay append/chunk transport first.

## OPT-119 Local Code Proof: Worker-Owned Replay Append Transport

Change:

- `_CompactReplayRingV1` can now clone one appended replay entry for transport
  and apply appended entries into another replay ring.
- `CompactProcessSampleLearnerWorkerV1` initializes a replay ring in the child
  process and keeps it across jobs.
- `CompactOwnedLoopV1` queues newly appended replay entries and sends those to
  the process worker instead of sending `snapshot_for_sample()` for every job.
- Reports now expose whether a full replay snapshot was sent, replay append
  entry/row counts, worker-owned replay, worker replay row counts, and worker
  replay eviction counts.
- Local-process speed-row validation now fails if the worker does not own replay
  or if a full replay snapshot is sent.

Validation:

```text
ruff: passed
pytest focused: 43 passed, 2 warnings
```

Decision:

- This is local correctness evidence, not H100 throughput evidence.
- The next row is a small H100 compact Torch normal-death correctness row for
  this request path.
- That row must prove request bytes now scale with appended replay chunks, not
  replay capacity, while keeping request/result CUDA tensor counts at `0/0`,
  applying returned worker model state to search, and including final drain.

## OPT-119 H100 Proof: Worker-Owned Replay Append Is Honest

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-worker-owned-replay-append-compacttorch-normal-nofused-unroll2-r5-20260601/compact_coach_speed_row_modal_report.json
```

Read:

```text
small H100 correctness row only
B128/A1, 33 measured, 4 warmup, normal death, compact Torch, no fused batch
speed: 196.29 env steps/sec
wall: 21.52s
submit/completed: 8/8
request/result CUDA tensor count: 0/0
worker owns replay/model: true/true
full replay snapshot sent: false
full replay snapshot submit count: 0
replay append entries/worker append count: 32/32
replay append rows/worker replay rows: 8188/8188
model-state return/omit/apply: 4/4/4
search/replay metadata lag: 0/1
final drain included: true
request/result bytes: 840488574/128472687
```

Decision:

- The full replay snapshot blocker is fixed.
- The row is still not speed evidence.
- Request bytes are smaller than `2.75GB`, but still large because append
  entries carry full step observation snapshots.

## OPT-119 Same-Shape H100: Local Process Replay Append Rejected

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-worker-owned-replay-append-sameshape-compacttorch-r1-20260601/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-worker-owned-replay-append-sameshape-compacttorch-r2-20260601/compact_coach_speed_row_modal_report.json
```

R1:

```text
failed quickly
problem: ReplayCompatibilityError: compact MuZero learner resident input observation must be on cuda
cause: host-only process request moved replay observations to CPU; fused learner batch required CUDA observations
fix: child process now moves received replay tensors to the child learner CUDA device before storing them
```

R2:

```text
same shape as OPT-104
B1024/A1, 180 measured, 45 warmup, normal death
sample interval 8, sample batch 512, replay capacity 4096
learner unroll 2, fused learner batch, refresh interval 4
speed: 1148.88 env steps/sec
wall: 160.43s
OPT-104 wall: 14.53s
submit/completed: 22/22
actor steps while pending: 149
policy lag max: 2
wait sec: 53.54
request/result bytes: 57772732239/225060976
request/result CUDA tensor count: 0/0
worker owns replay/model: true/true
full replay snapshot sent/count: false/0
replay append entries/worker append count: 176/176
replay append rows/worker replay rows: 358404/358404
model-state return/omit/apply: 7/15/7
search/replay metadata lag: 0/0
learner calls/updates: 22/22
sample calls: 22
terminal/death/truncated rows: 1050/1050/0
resident host fallback: 0
final refresh update: 22
final drain included: true
H100 util mean/max: 3.85%/40%
```

Decision:

- This local-process replay append lane is rejected as a speed path.
- Correctness is intact, so the speed failure is real.
- The bad cost is not full replay snapshots anymore. It is serializing replay
  observations through Python multiprocessing and then moving them back to the
  child GPU.
- Do not tune this exact lane. Next implementation must remove pickled replay
  observation transfer from the learner path.

## OPT-118: H100 Utilization Is Low

Change:

- compact Coach speed rows can sample `nvidia-smi` during the measured run;
- reports now expose GPU name, sample count, mean/max utilization, memory use,
  power, and sampler errors;
- future reports also count nonzero samples and samples above 50% and 80%.

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt118-h100-normal-death-current-sync-gpu-util-compacttorch-r1-20260601/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt118-h100-normal-death-current-sync-gpu-util-compacttorch-r2-20260601/row_001_result.json
```

Reads:

```text
r1 speed: 12795.45 env steps/sec
r1 wall:  14.4051s
r1 GPU samples: 47
r1 GPU util: mean 15.5%, max 48%
r1 sampler errors: 0

r2 speed: 12115.98 env steps/sec
r2 wall:  15.2130s
r2 GPU samples: 50
r2 GPU util: mean 14.7%, max 38%
r2 samples above 50% util: 0
r2 samples above 80% util: 0
r2 sampler errors: 0
```

R2 timing buckets:

```text
actor/game/autoreset: 4.6467s
replay sample: 3.7662s
compact rollout/search: 2.8232s
observation: 2.1180s
learner: 1.2645s
```

Decision:

- This is not a saturated H100.
- Do not run H200/B200 as the next speed attempt.
- Do not call multi-GPU a plan without a real actor/search/replay/learner
  split.
- R1 beat OPT-104 slightly, but R2 did not. Keep OPT-104 as the accepted
  baseline until a repeated row clearly beats it while preserving all work.
- Next speed work should remove host/control-flow cost or separate the real
  loop across resources.

## OPT-116 r1: Deferred Learner In Lean Row, Correct But Slower

Change:

- lean trainer owner can use the existing deferred learner path;
- deferred drains go through the trainer, so trainer counters and policy refs
  stay in sync with loop counters;
- speed-row payloads expose pending count, max pending, wait count, and wait
  time.

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt116-h100-normal-death-lean-deferred-learner-compacttorch-r1-20260601/row_001_result.json
```

Read:

```text
speed: 10976.10 env steps/sec
wall:  16.7928s
sample gate: 4.4421s
learner gate: 2.8876s
learner-batch build: 2.3653s
deferred submit/completed: 22/22
pending at end: false, count 0
max pending observed: 2
wait sec: 0.0643s
trainer learner updates: 22
loop learner updates: 22
sample calls: 22
final refresh update: 22
terminal/death rows: 1050
truncations: 0
resident host fallback: 0
```

Decision rule:

- The proof is valid.
- The speed is worse than OPT-104 and OPT-115 r2.
- Do not keep tuning deferred learner alone.
- Current measurement needed: stage replay sample plus learner off the actor
  path and run the same H100 shape.

## OPT-116 Step 2 Local Proof: Staged Sample Plus Learner

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt116-local-lean-deferred-sample-learner-smoke-20260601/compact_coach_speed_row_smoke_report.json
```

Read:

```text
local only
deferred sample+learner: true
submit/completed: 6/6
pending at end: false, count 0
max pending observed: 1
final drain included in wall time: true
sample calls: 6
learner calls: 4
learner updates: 4
```

Decision:

- The wiring and proof fields work locally.
- This is not a speed decision.
- Next measurement is the same-shape H100 row against OPT-104.

## OPT-116 Step 2 H100: Same-GPU Staging Rejected

Control artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt117-h100-normal-death-current-sync-control-compacttorch-r1-20260601/row_001_result.json
```

Control read:

```text
speed: 12566.17 env steps/sec
wall:  14.6680s
sample calls: 22
learner calls: 22
learner updates: 22
sample gate: 3.6076s
learner gate: 1.3303s
actor step: 4.5225s
observation: 1.8013s
compact rollout/search: 2.8809s
terminal/death rows: 1050
truncations: 0
resident host fallback: 0
```

Staged max-pending-1 artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt116-h100-normal-death-lean-deferred-sample-learner-compacttorch-r1-20260601/row_001_result.json
```

Staged max-pending-1 read:

```text
speed: 11340.59 env steps/sec
wall:  16.2531s
submit/completed: 22/22
pending at end: false, count 0
wait sec: 2.3230s
max pending observed: 1
sample calls: 22
learner updates: 22
final refresh update: 22
```

Staged max-pending-3 artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt117-h100-normal-death-lean-deferred-sample-learner-maxpending3-compacttorch-r1-20260601/row_001_result.json
```

Staged max-pending-3 read:

```text
speed: 11297.02 env steps/sec
wall:  16.3158s
submit/completed: 22/22
pending at end: false, count 0
wait sec: 1.8301s
max pending observed: 3
sample calls: 22
learner updates: 22
final refresh update: 22
```

Decision:

- Current sync code is close to OPT-104, so snapshot plumbing did not explain
  the staging loss.
- Same-GPU background staging is rejected as the next speed path.
- Backlog reduced explicit wait but increased actor/search contention.
- Stop extending this lane. Pick the next target from the sync-control costs.
- This ledger does not yet prove full H100 utilization. Hardware-scale claims
  require a same-shape utilization row.

## OPT-115: Useful Child Improvement, Not A Speed Win

Change:

- direct learner-batch builder keeps observations uint8 through grouped
  gather/scatter when possible;
- normalizes the final ordered observation tensor once;
- collects grouped chunks and restores sample order once per tensor.

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt115-h100-normal-death-grouped-order-restore-compacttorch-r1-20260601/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt115-h100-normal-death-grouped-order-restore-compacttorch-r2-20260601/row_001_result.json
```

Reads:

```text
r1 speed: 11033.99 env steps/sec
r1 wall:  16.7047s
r1 sample gate: 3.9848s
r1 learner-batch build: 1.8951s
r1 learner gate: 1.4207s

r2 speed: 11936.45 env steps/sec
r2 wall:  15.4418s
r2 sample gate: 3.7729s
r2 learner-batch build: 1.8092s
r2 learner gate: 1.9731s
```

Proof counters in both rows:

```text
B1024/A1
180 measured, 45 warmup
normal death
compact Torch search
direct-core initial inference
sample calls: 22
learner calls: 22
learner updates: 22
terminal/death rows: 1050
truncations: 0
terminal sample rows: 167
resident host fallback: 0
final refresh update: 22
calls_train_muzero: false
touches_live_runs: false
```

Decision:

- Keep the code.
- Do not call it a top-line speed win.
- Current target is replay sample plus learner staging.

## OPT-114: Single Fast Row, Failed Repeat

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt114-h100-normal-death-late-observation-normalize-compacttorch-r1-20260601/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt114-h100-normal-death-late-observation-normalize-compacttorch-r2-20260601/row_001_result.json
```

Reads:

```text
r1 speed: 13711.58 env steps/sec
r1 wall:  13.4426s
r1 learner-batch build: 1.7137s

r2 speed: 10202.39 env steps/sec
r2 wall:  18.0664s
r2 learner-batch build: 2.3261s
```

Decision:

- r1 was encouraging but not repeatable.
- OPT-114 cannot replace OPT-104.

## OPT-113 r2: Valid Timer Row

Artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt113-h100-normal-death-sample-learner-timers-compacttorch-r2-20260601/row_001_result.json`

Read:

```text
speed: 10929.03 env steps/sec
wall:  16.8652s
sample gate: 4.1432s
learner-batch build: 2.1486s
learner gate: 1.6991s
```

Decision:

- This row identified replay sample direct learner-batch construction as the
  first measured target.

## OPT-119 H100 Proof: Normal-Death Owner-Search

Artifacts:

```text
local:
artifacts/local/tmp_owner_search_normal_repro/opt119-local-owner-search-normal-unroll2-b128-sampleall-20260602-r5/row_001_result.json

H100:
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-owner-search-normal-unroll2-b128-sampleall-proof-20260602-r1/row_001_result.json
```

H100 read:

```text
death mode: normal
shape: B128/A1, 37 measured, 4 warmup
sample mode: sample all eligible replay rows, sample_batch_size=0
owner replay appends: 40
owner train requests: 5
owner learner updates: 5
parent sample calls: 0
parent learner updates: 0
root-observation bytes: 0
model-state bytes: 0
request/result CUDA tensors: 0/0
normal-death gate: true
terminal rows: 2
terminal sample rows: 4
terminal unroll target rows: 4
learner done count: 4
parent wait: 1.610s
worker wall: 1.591s
worker train: 1.552s
worker replay append: 0.0007s
worker search refresh: 0.0310s
H100 utilization mean/max: 6.46% / 100%
```

Decision:

- This closes the normal-death owner-search correctness blocker.
- This is not speed evidence because `sample_batch_size=0` samples all eligible
  replay rows and changes learner work.
- Bigger GPU, multi-GPU, and GPU mechanics are still not justified by this row:
  mean H100 utilization was only `6.46%`.

## OPT-119 H100 Rejection: Same-Work Owner-Search

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-owner-search-samework-b1024a1-normal-unroll2-speed-20260602-r1/row_001_result.json
```

Read:

```text
shape: H100, B1024/A1, normal death, 180 measured, 45 warmup
sample: interval 8, batch 512, replay capacity 4096, learner unroll 2
speed: 735.83 env steps/sec
wall: 250.493s
compact rollout slab: 238.702s
actor step: 8.963s
observation: 2.568s
request/result bytes: 134.6MB / 199.7KB
root bridge H2D bytes: 33.6MB
owner replay appends: 224
owner train requests: 28
owner learner updates: 28
parent sample calls / parent learner updates: 0 / 0
model-state bytes: 0
request/result CUDA tensors: 0/0
terminal sample rows / terminal target rows: 42 / 42
normal-death gate: true
H100 utilization mean/max: 1.90% / 24%
```

Decision:

- Correctness is good enough to trust the rejection.
- Speed is not close to OPT-104 (`12689.38` env steps/sec, `14.5255s` wall).
- The row exposes the next blocker: owner replay append still sends previous
  and current compact observation batches through the process request. At B1024,
  those two stacks explain the `134.6MB` request payload.
- Do not rerun this row unchanged.
- Do not move to H200/B200, multi-GPU, PufferLib port, or GPU mechanics from
  this evidence. H100 utilization is low and the failed row is dominated by
  data movement/process payload shape.
- Next accepted work: fixed-buffer/id owner replay append, then rerun the
  same-work row.

## OPT-119 Local Transport Proof: Index-Only Owner Replay Append

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-local-owner-search-indexonly-request-byte-smoke-20260602-r5/row_001_result.json
```

Read:

```text
shape: local CPU, B8/A1, 6 measured, 2 warmup
search: owner_search_slab_proxy, compact Torch inner search, sim1
request bytes: 5439
owner replay appends: 7
owner train requests: 2
owner learner updates: 2
parent sample calls / parent learner updates: 0 / 0
model-state bytes: 0
request/result CUDA tensors: 0/0
owner model ref returned: true
owner learner update consumed by search: true
report ok: true
```

Decision:

- This is the local proof that the owner-search replay append request no longer
  carries previous/current compact observation batches.
- It is not speed evidence and cannot be compared to OPT-104.
- The next accepted measurement is the same-work H100 owner-search rerun. It
  must show request bytes no longer in the `134.6MB` compact-batch class while
  preserving owner replay/learner correctness.

## OPT-119 H100 Index-Only Owner-Search Rerun

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt119-h100-owner-search-indexonly-samework-b1024a1-normal-unroll2-speed-20260602-r1/row_001_result.json
```

Read:

```text
shape: H100, B1024/A1, normal death, 180 measured, 45 warmup
sample: interval 8, batch 512, replay capacity 4096, learner unroll 2
speed: 4202.05 env steps/sec
wall: 43.864s
compact rollout slab: 31.520s
actor step: 9.412s
observation: 2.652s
request/result bytes: 138733 / 207278
old owner-search request bytes: 134610322
root bridge H2D bytes: 33554432
owner replay appends: 224
owner train requests: 28
owner learner updates: 28
parent sample calls / parent learner updates: 0 / 0
model-state bytes: 0
request/result CUDA tensors: 0/0
terminal sample rows / terminal target rows: 42 / 42
normal-death gate: true
H100 utilization mean/max: 5.39% / 32%
```

Decision:

- The replay append payload fix worked at H100/B1024 scale.
- The route improved from `735.83` to `4202.05` env steps/sec, but still loses
  to OPT-104 `12689.38` env steps/sec.
- The old `134.6MB` request payload is no longer the blocker.
- Next accepted work: explain and reduce the remaining `31.52s` compact
  rollout slab and surrounding actor/observation wall without changing work.

## OPT-120 H100 Owner-Search Totals Row

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt120-h100-owner-search-totals-samework-b1024a1-normal-unroll2-speed-20260602-r1/row_001_result.json
```

Read:

```text
shape: H100, B1024/A1, normal death, 180 measured, 45 warmup
sample: interval 8, batch 512, replay capacity 4096, learner unroll 2
speed: 3888.48 env steps/sec
wall: 47.402s
compact rollout slab: 33.523s
actor step: 9.582s
observation: 4.004s
request/result bytes: 138733 / 207278
owner replay appends: 224
owner train requests: 28
owner learner updates: 28
parent sample calls / parent learner updates: 0 / 0
model-state bytes: 0
request/result CUDA tensors: 0/0
normal-death gate: true
H100 utilization mean/max: 5.36% / 31%
slab totals exposed in summary/compact/report: true
search dispatch total: 32.827s
owner parent wait total: 31.344s
owner worker wall total: 21.646s
worker root resolve: 7.151s
worker replay append: 6.571s
worker learner train: 3.201s
worker search: 2.359s
worker search refresh: 0.875s
parent replay-row build/store: 0.162s / 0.096s
```

Decision:

- This row is not a speed win. It is slower than the prior index-only row and
  far below OPT-104.
- It closes the measurement gap: the remaining slab wall is mostly owner parent
  wait around synchronous owner work.
- Parent replay-row build/store is not the next target.
- This selected the OPT-121 implementation: split action-critical search from
  owner replay append, learner train, and search refresh maintenance. OPT-121
  proved that split and then exposed the remaining FIFO owner-lane blocker.

## OPT-121 H100 Deferred Owner Maintenance Row

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt121-h100-owner-search-deferred-maintenance-samework-b1024a1-normal-unroll2-speed-20260602-r2/row_001_result.json
```

Read:

```text
shape: H100, B1024/A1, normal death, 180 measured, 45 warmup
sample: interval 8, batch 512, replay capacity 4096, learner unroll 2
speed: 4658.42 env steps/sec
wall: 39.567s
owner replay appends / train requests / learner updates: 224 / 28 / 28
parent sample calls / parent learner updates: 0 / 0
final pending owner maintenance: 0
final policy lag: 0
normal-death gate: true
H100 utilization mean/max: 6.58% / 34%
compact rollout slab search dispatch: 24.233s
owner parent wait total: 22.520s
owner worker wall total: 9.397s
worker root resolve: 5.633s
worker search: 2.361s
worker replay append: 6.121s
worker learner train: 5.793s
worker search refresh: 1.056s
actor steps while maintenance pending: 51
parent replay-row build/store: 0.230s / 0.100s
```

Decision:

- Deferred maintenance is a real improvement over OPT-120
  (`3888.48 -> 4658.42` env steps/sec), and it preserves the same work.
- It still loses badly to OPT-104 (`12689.38` env steps/sec).
- The split did not eliminate owner wait because the owner is still one FIFO
  process lane. Maintenance jobs can still sit in front of the next search.
- Next accepted implementation: replace the `ProcessPoolExecutor(max_workers=1)`
  owner handoff with a persistent owner loop that prioritizes search requests
  over maintenance while still draining maintenance to zero before report.

## OPT-121 Local Persistent Priority Owner Loop Smoke

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt121-local-owner-search-priority-loop-smoke-20260602-r1/compact_coach_speed_row_smoke_report.json
```

Read:

```text
local CPU proof only: not H100 speed evidence
loop kind: persistent_priority_owner_loop_v1
action priority enabled: true
action requests: 16
maintenance requests: 5
actions while maintenance pending: 11
actions served before maintenance completed: 11
FIFO-blocked actions: 0
owner replay appends / train requests / learner updates: 15 / 3 / 3
search refresh updates: 3
final pending owner maintenance: 0
final inflight maintenance: false
final policy lag: 0
final drain sec: 0.777s
```

Decision:

- The local priority-loop implementation and report proof surface work.
- This does not prove speed. The required next evidence is the same-work H100
  priority-loop row against OPT-104.

## OPT-121 H100 Persistent Priority Owner Loop Row

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt121-h100-owner-search-priority-loop-samework-b1024a1-normal-unroll2-speed-20260602-r1/row_001_result.json
```

Read:

```text
shape: H100, B1024/A1, normal death, 180 measured, 45 warmup
sample: interval 8, batch 512, replay capacity 4096, learner unroll 2
speed: 5853.08 env steps/sec
wall: 31.491s
loop kind: persistent_priority_owner_loop_v1
action requests / maintenance requests: 225 / 173
actions while maintenance pending: 65
actions served before maintenance completed: 65
FIFO-blocked actions: 0
owner replay appends / train requests / learner updates: 224 / 28 / 28
parent sample calls / parent learner updates: 0 / 0
request/result bytes: 138733 / 187414
model-state bytes: 0
request/result CUDA tensors: 0 / 0
terminal sample rows / terminal target rows: 42 / 42
normal-death gate: true
final pending owner maintenance: 0
final policy lag: 0
final drain sec: 0.228
H100 utilization mean/max: 5.93% / 27%
owner parent wait total: 17.476s
owner worker wall total: 7.823s
worker root resolve: 4.418s
worker search: 2.515s
worker replay append: 4.890s
worker learner train: 7.725s
worker search refresh: 1.215s
```

Decision:

- Priority scheduling is real and improves the route:
  `4202.05 -> 4658.42 -> 5853.08` env steps/sec across index-only,
  deferred-FIFO, and priority-loop rows.
- It still does not beat OPT-104 (`12689.38` env steps/sec).
- The next implementation target is the remaining measured owner parent wait
  and owner worker work. Do not reopen append payload, scalar-ref, PufferLib,
  H200/B200, or GPU mechanics without a new row pointing there.

## OPT-122 Local Owner-Search IPC / Sparse Bridge / Coalescing Smoke

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt122-local-owner-search-ipc-sparse-coalesce-smoke-20260602-r1/compact_coach_speed_row_smoke_report.json
```

Read:

```text
local CPU proof only: not H100 speed evidence
payload transport: numpy_ndarray_ipc_v1
owner replay appends / train requests / learner updates / refresh: 15 / 3 / 3 / 3
parent sample calls / parent learner updates: 0 / 0
maintenance requests: 2
previous local priority-loop maintenance requests: 5
maintenance coalescing: train_boundary_or_final_drain_v1
coalesced skip count: 3
actions while maintenance pending: 11
actions served before maintenance completed: 11
FIFO-blocked actions: 0
final pending owner maintenance: 0
final policy lag: 0
representative B1024 tuple payload pickle bytes: 126420
representative B1024 ndarray payload pickle bytes: 66175
```

Decision:

- The local IPC, sparse terminal-final bridge, and train-boundary maintenance
  coalescing fixes are wired and proofed.
- This is not speed evidence. The next row must be the same-work H100
  owner-search row against OPT-104, preserving the full same-work counters and
  normal-death gates.

## OPT-122 H100 Owner-Search IPC / Sparse Bridge / Coalescing Row

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt122-h100-owner-search-ipc-sparse-coalesce-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json
```

Read:

```text
H100 speed: 6639.26 env steps/sec
wall: 27.762s
baseline OPT-104: 12689.38 env steps/sec, 14.5255s wall
owner appends / train requests / learner updates: 224 / 28 / 28
parent sample calls / parent learner calls: 0 / 0
request bytes: 138733
result bytes: 115213
model-state bytes: 0
request/result CUDA tensors: 0 / 0
terminal sample/target rows: 42 / 42
normal-death gate: true
final pending owner maintenance: 0
final owner policy lag: 0
maintenance requests: 25
coalesced skip count: 106
owner parent wait: 10.872s
owner worker wall: 9.671s
worker root resolve/search/replay/train/refresh: 6.233s / 2.963s / 6.484s / 8.904s / 1.487s
H100 mean/max utilization: 6.62% / 28%
sparse final rows: 5 rows, 163840 bytes, dense clone avoided 33554432 bytes
```

Decision:

- OPT-122 is a real improvement over OPT-121 (`5853.08 -> 6639.26` env
  steps/sec), but it still loses to OPT-104.
- Do not repeat IPC/sparse/coalescing as the active task. The next measured
  target is action-only owner results: the parent should receive selected
  actions and proof fields, while the owner keeps search results and
  materializes replay.

## OPT-123 Local Action-Only Owner Result Smoke

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt123-local-owner-action-only-materialized-replay-smoke-20260602-r2/compact_coach_speed_row_smoke_report.json
```

Read:

```text
local CPU proof only: not H100 speed evidence
action-only result: true
owner materializes replay: true
parent reconstructed search result: false
search_result_payload_bytes: 0
visit_policy_bytes / root_value_bytes: 0 / 0
result bytes: 334
transport: action_only_owner_cached_replay_v1
owner replay appends / train requests / learner updates / refresh: 15 / 3 / 3 / 3
committed parent index rows: 0
final pending owner maintenance: 0
final policy lag: 0
maintenance requests: 2
actions while maintenance pending: 11
actions served before maintenance completed: 11
FIFO-blocked actions: 0
```

Decision:

- OPT-123 local proof has the intended action-only shape: full search arrays no
  longer return to the parent on action responses, and replay/train still
  happen on the owner.
- The first OPT-123 H100 attempt failed the normal-death proof gate before
  speed reporting because `terminal_rows_verified` was missing. The code now
  materializes owner replay as device replay rows when cached roots are
  resident-device, and direct deferred `run()` fails closed instead of returning
  a synthetic full search result.
- Superseded by the passing H100 row below.

## OPT-123 H100 Action-Only Owner Result Row

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt123-h100-owner-action-only-materialized-device-replay-samework-b1024a1-normal-unroll2-speed-20260602-r3/compact_coach_speed_row_modal_report.json
```

Read:

```text
H100 speed: 7679.53 env steps/sec
wall: 24.001s
baseline OPT-104: 12689.38 env steps/sec, 14.5255s wall
owner appends / train requests / learner updates / refresh: 224 / 28 / 28 / 28
parent sample calls / parent learner calls: 0 / 0
request bytes: 32779
result bytes: 4444
search_result_payload_bytes: 0
selected-action bytes: 4076
visit-policy bytes / root-value bytes: 0 / 0
payload transport: action_only_owner_cached_replay_v1
parent reconstructed search result: false
owner materializes replay: true
model-state bytes: 0
request/result CUDA tensors: 0 / 0
terminal sample/target rows: 42 / 42
normal-death gate: true
final pending owner maintenance: 0
final owner policy lag: 0
maintenance requests / drains / coalesced skips: 22 / 22 / 69
owner parent wait: 9.219s
owner worker wall: 8.417s
worker root resolve/search/replay/train/refresh: 5.299s / 2.755s / 8.983s / 8.593s / 1.333s
compact rollout slab: 10.803s
actor step wall: 8.527s
observation: 2.308s
H100 mean/max utilization: 9.64% / 41%
```

Decision:

- OPT-123 is a real improvement over OPT-122 (`6639.26 -> 7679.53` env
  steps/sec), and action-only owner results worked.
- It still loses to OPT-104. Do not rerun OPT-123 unchanged.
- The next measured target is OPT-124: reduce owner parent wait/root resolve and
  owner replay append materialization. Learner train remains a large bucket too,
  but the first action-critical problem is that the parent still waits `9.219s`
  for owner action work.

## OPT-124 Local Inner Two-Phase Owner Proof

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt124-local-owner-inner-two-phase-device-replay-smoke-20260602-r2/compact_coach_speed_row_smoke_report.json
```

Read:

```text
local CPU proof only: not H100 speed evidence
action-only result: true
owner materializes replay: true
inner two-phase action step: true
inner deferred device replay payload: true
parent reconstructed search result: false
request/result bytes: 2513 / 334
search payload / visit-policy / root-value bytes: 0 / 0 / 0
owner replay appends / train requests / learner updates / refresh: 15 / 3 / 3 / 3
final pending owner maintenance: 0
final policy lag: 0
```

Decision:

- This proved the inner two-phase path could run locally.
- It did not prove speed. The same-work H100 row below rejected it.

## OPT-124 H100 Inner Two-Phase Owner Rejection Row

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt124-h100-owner-inner-two-phase-device-replay-samework-b1024a1-normal-unroll2-speed-20260602-r1/compact_coach_speed_row_modal_report.json
```

Read:

```text
H100 speed: 3884.97 env steps/sec
wall: 47.444s
baseline OPT-104: 12689.38 env steps/sec, 14.5255s wall
best useful owner row OPT-123: 7679.53 env steps/sec, 24.001s wall
inner two-phase action step: true
inner deferred device replay payload: true
action-only result: true
owner materializes replay: true
parent reconstructed search result: false
request/result bytes: 32779 / 4444
search payload / visit-policy / root-value bytes: 0 / 0 / 0
owner appends/train/updates/refresh: 224 / 28 / 28 / 28
parent sample/learner: 0 / 0
terminal sample/target rows: 42 / 42
normal-death gate: true
final pending owner maintenance: 0
final owner policy lag: 0
parent wait: 21.455s
worker wall: 19.870s
root resolve/search/replay/train/refresh: 15.801s / 3.327s / 28.825s / 14.267s / 1.931s
H100 mean/max utilization: 4.54% / 25%
```

Decision:

- Reject OPT-124 as a speed path.
- Keep inner two-phase owner replay default-off.
- Use OPT-123, not OPT-124, as the current bucket map.
- The next target is OPT-125: remove concrete owner root/replay construction
  waste, then run one same-work H100 row.

## OPT-125 Local Resident Replay Step Cleanup

Read:

```text
change: owner-search resident replay steps omit host observation copies
files: scripts/build_compact_coach_speed_row_smoke.py, tests/test_compact_coach_speed_row_smoke.py
validation: ruff passed
validation: resident replay step test passed
validation: owner-search service suite passed, 22 passed / 2 warnings
```

Decision:

- This is local correctness evidence only.
- The next speed evidence must be a same-work H100 row against OPT-123 and
  OPT-104.

## Older Context

- OPT-112 made the lean trainer path correct but slower:
  `11167.95` env steps/sec, `16.5044s` wall.
- OPT-111 was also slower:
  `10746.54` env steps/sec, `17.1516s` wall.
- OPT-103 was the last accepted improvement before OPT-104:
  `12398.24` env steps/sec, `14.8666s` wall.

## OPT-132-K/L Long-Window Stability Pair

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132k-h100-longwindow-stability-diagnostic-b1024a1-normal-unroll2-m724-w180-r3/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132l-h100-longwindow-stability-diagnostic-b1024a1-normal-unroll2-m724-w180-r4/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132kl-longwindow-stability-comparison-20260602/comparison.json
```

Read:

```text
window: stability_724_180
identity: exact
K speed/wall: 7365.79 env steps/sec, 100.651s
L speed/wall: 8297.18 env steps/sec, 89.353s
wall swing: 11.298s
sample gate swing: 5.448s
learner-batch build swing: 3.827s
actor wall swing: 3.057s
actor autoreset swing: 1.990s
learner gate swing: 1.114s
```

Decision:

- The long-window tooling is working.
- The work identity proof is strong enough to say K and L performed the same
  work.
- The measurement is still not stable enough for speed claims.
- Do not compare these rows to OPT-104 as a speed win; they are diagnostics
  unless a matched long-window baseline exists.
- Later `stability_1084_270` and `stability_1444_360` H100 pairs failed on
  memory. Later still, five exact `724/180` A/A rows also failed stability.
  The latest-frame patch later made `1084/270` fit, but timing stability still
  failed. Do not repeat unstable rows unchanged.

## OPT-132-M/N Failed 1444/360 H100 Diagnostics

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132m-h100-longwindow-stability-diagnostic-b1024a1-normal-unroll2-m1444-w360-r5/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132n-h100-longwindow-stability-diagnostic-b1024a1-normal-unroll2-m1444-w360-r6/compact_coach_speed_row_modal_report.json
```

Read:

```text
window: stability_1444_360
gpu utilization sampling: enabled
result: failed before valid row
problem: speed-row producer exited with 1
log detail: CUDA out of memory on at least one FunctionCall
memory detail: PyTorch allocated about 78.36 GiB on H100
```

Decision:

- `1444/360` was too large for the H100 memory budget in this setup.
- Do not retry `1444/360` unchanged.
- Later evidence superseded this memory-only read: latest-frame snapshots made
  `1084/270` fit, but the exact repeat still failed timing stability.

## OPT-132-O/P Failed 1084/270 H100 Diagnostics

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132o-h100-midwindow-stability-diagnostic-b1024a1-normal-unroll2-m1084-w270-r7/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132p-h100-midwindow-stability-diagnostic-b1024a1-normal-unroll2-m1084-w270-r8/compact_coach_speed_row_modal_report.json
```

Read:

```text
window: stability_1084_270
gpu utilization sampling: disabled
result: failed before valid row
problem: speed-row producer exited with 1
log detail: CUDA out of memory on P
memory detail: PyTorch allocated about 78.34 GiB on H100
```

Decision:

- Historical decision before latest-frame resident replay snapshots:
  `1084/270` was too large for the old H100 memory path.
- Superseded decision: `1084/270` now fits with latest-frame snapshots, but
  timing stability still fails. The active target is sample-gate
  learner-batch-build isolation.

## OPT-132-K/L/Q/R/S Five-Row 724/180 A/A Packet

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132klqrs-longwindow-stability-comparison-20260602/comparison.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132q-h100-longwindow-stability-diagnostic-b1024a1-normal-unroll2-m724-w180-r9/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132r-h100-longwindow-stability-diagnostic-b1024a1-normal-unroll2-m724-w180-r10/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132s-h100-longwindow-stability-diagnostic-b1024a1-normal-unroll2-m724-w180-r11/row_001_result.json
```

Read:

```text
window: stability_724_180
rows: K, L, Q, R, S
identity: exact
accepted-fast-path violations: 0
wall min/median/max: 89.353s / 100.816s / 137.297s
wall range: 47.945s, 47.56% of median
steps/sec range: 5399.78 -> 8297.18
sample gate range: 44.170s -> 77.233s
learner-batch build range: 25.023s -> 43.698s
actor wall range: 19.824s -> 28.402s
learner gate range: 3.271s -> 6.501s
```

Decision:

- This is a hard measurement/runtime-stability failure.
- The work identity proof is exact, so this is not hidden work drift.
- Stop repeating `724/180` unchanged.
- Do not patch normal speed code yet; the patch would be buried in noise.
- Superseded next-target note: the memory-bounded `1084/270` path now fits.
  The active target is direct sample-gate learner-batch-build isolation.

## OPT-132 Bounded Long Diagnostic Patch

Local change:

```text
purpose: make longer warmup/measured diagnostics runnable without profiler history growth
speed claim: none
accepted-speed denominator change: none
```

What changed:

- `CompactRolloutSlab` now has counters for committed row groups and committed
  rows independent of retained history.
- Long diagnostic rows can skip retaining the full committed-row history while
  still passing each committed row through the normal training path.
- Final profile summary now reads committed group/row counters directly instead
  of copying `committed_index_rows` just to take `len(...)`.
- Speed-row bounded diagnostics can omit the nested `source_profile_payload`.
- Accepted stability windows enable `--compact-profile-bounded-diagnostics`
  automatically.

Decision:

- Bounded diagnostics were necessary but not sufficient.
- The next `stability_1084_270` rerun still OOMed on H100, which led to the
  latest-frame resident replay snapshot patch.
- Superseded decision: `1084/270` now fits; timing stability is still open.

## OPT-132-U Bounded 1084/270 H100 Rerun

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132u-h100-bounded-midwindow-stability-diagnostic-b1024a1-normal-unroll2-m1084-w270-r13/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132u-h100-bounded-midwindow-stability-diagnostic-b1024a1-normal-unroll2-m1084-w270-r13/remote_bundle.json
```

Read:

```text
window: stability_1084_270
bounded diagnostics: true
result: failed before valid row
problem: CUDA out of memory
PyTorch allocated: about 78.34 GiB
process memory in use: about 79.14 GiB
free GPU memory at failure: about 32 MiB
producer_return_code: 1
producer_result_present: true
producer_report_present: true
```

Decision:

- The Modal wrapper fix worked: the remote bundle now preserves the inner
  failed producer result/report.
- Historical decision: the remaining blocker was live memory growth before the
  long row could finish, not just final result JSON size.
- Superseded decision: latest-frame resident replay snapshots made `1084/270`
  fit. The current blocker is timing stability on exact same-work repeats.

## OPT-132 Retained Resident Replay Memory Counters

Local change:

```text
purpose: expose direct replay-store resident snapshot retention
speed claim: none
accepted-speed denominator change: none
```

What changed:

- `_CompactReplayRingV1` now tracks unique retained resident replay snapshots
  and their primary tensor bytes as entries are appended and evicted.
- `CompactOwnedLoopV1.telemetry()` now reports:
  `compact_owned_loop_replay_store_retained_resident_snapshot_count` and
  `compact_owned_loop_replay_store_retained_resident_snapshot_bytes`.
- Standalone sample-ring summaries now report:
  `compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_count`
  and
  `compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_bytes`.

Decision:

- Later diagnostic rows showed resident replay snapshots were the relevant H100
  memory ceiling for `1084/270`, and latest-frame snapshots made that window fit.
- This was measurement-stability work, not speed evidence.

## OPT-132 Latest-Frame Resident Replay Snapshot Mode

Local change:

```text
purpose: cut step-scaled resident replay snapshot memory in bounded diagnostics
speed claim: none
accepted-speed denominator change: none
accepted 180/45 default: still full-stack resident replay snapshots
bounded diagnostics: latest-frame resident replay snapshots
```

What changed:

- Added `resident_replay_snapshot_mode` with default `full_stack`.
- Added latest-frame resident replay snapshots for bounded diagnostics.
- Normal steps clone one latest frame instead of the full 4-frame stack.
- Reset/autoreset steps rebuild frame history from the live stack so old
  episode frames do not leak after reset.
- Terminal final observations remain sparse rows.
- Replay sample and learner builders reconstruct sampled 4-frame stacks from
  either full-stack or latest-frame snapshots through the same helper path.

Validation:

```text
ruff: source profile, speed-row smoke builder, source-profile tests passed
pytest source profile: 110 passed, 2 warnings
pytest speed-row smoke + compact-owned replay-store slice: 48 passed, 2 warnings
focused parity: latest-frame snapshot matched full-stack terminal sample
focused reset barrier: latest-frame snapshot zeroed older reset channels
```

Decision:

- This is a memory-stability patch, not speed evidence.
- The H100 bridge row and `1084/270` r1/r2 proved the memory fit with
  latest-frame resident replay snapshots.
- Timing stability is still open: exact `1084/270` repeats failed the stability
  bar in sample gate and learner-batch build.

## OPT-132 Owner-Search Shared-Model H100 r12/r13/r14

Date: 2026-06-05

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-action-only-inlinelearner-refreshcadence-warmupgate-b1024a1-normal-unroll2-r12-20260605/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-action-only-inlinelearner-sharedmodelrefresh-warmupgate-b1024a1-normal-unroll2-r13-20260605/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-warmupgate-b1024a1-normal-unroll2-r14-20260605/row_001_result.json
```

Read:

```text
baseline: OPT-104, 12689.38 env steps/sec, 14.5255s wall

r12 cadence-only:
  speed: 8610.37 env steps/sec
  wall: 21.4067s
  expected train requests: 22
  refresh interval/request/skip/update: 4 / 6 / 16 / 22
  final update consumed: true
  decision: cadence proof passed, speed failed

r13 shared-model refresh, still host-cloning inline payload:
  speed: 9791.12 env steps/sec
  wall: 18.8252s
  expected train requests: 22
  refresh interval/request/skip/update: 1 / 22 / 0 / 22
  digest/state_dict: 0.0s / 0.0s
  host clone: 0.5190s
  final owner drain: 0.000044s
  decision: shared refresh removed digest/state_dict work, but host clone remained

r14 shared-model refresh plus no inline host payload clone:
  speed: 13497.30 env steps/sec
  wall: 13.6561s
  delta vs OPT-104: +6.36% speed, -0.869s wall
  expected train requests: 22
  owner learner updates: 22
  refresh interval/request/skip/update: 1 / 22 / 0 / 22
  final update consumed: true
  warmup replay entries suppressed: 44
  parent committed replay rows: 0
  search-result payload bytes: 0
  owner train wall/sample/update: 2.8637s / 1.8169s / 1.0323s
  digest/state_dict/host clone: 0.0s / 0.0s / 0.0s
  owner ref build: 0.000075s
  search refresh: 0.001553s
  final owner drain: 0.000042s
  policy lag max: 1
```

Decision:

- r14 is the first real owner-search H100 speed win over OPT-104.
- Do not promote from one row. Exact r15 repeat failed speed while preserving
  identity/proof. Longer r16 also failed speed and exposed owner train sample
  cost. r17 then showed the fused/tensor-native learner-batch flags do not
  compose with owner-search inline by CLI.
- The proof caveat is cadence identity: r14 uses `policy_refresh_interval=1`,
  while OPT-104 used interval `4`. Because r14 does more refreshes, not fewer,
  the row is still strong speed evidence. If a strict cadence match is required,
  run the same shared/no-clone path at interval `4`.
- Cadence-only r12 and shared-refresh-with-host-clone r13 are closed as failed
  speed paths. Do not reopen MPS/eager-append/same-process-async as P0 unless
  the r14 repeat fails and the measured fields point back there.

## OPT-132 Owner-Search Shared-Model r15/r16/r17 Followup

Date: 2026-06-05

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-warmupgate-b1024a1-normal-unroll2-r15-20260605/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-r14-r15-sharedmodel-nopayloadclone-repeat-comparison-20260605/comparison.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-warmupgate-b1024a1-normal-unroll2-m724-w180-r16-20260605/row_001_result.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-fusedtensor-warmupgate-b1024a1-normal-unroll2-m724-w180-r17-20260605/compact_coach_speed_row_modal_report.json
```

Read:

```text
r15 exact repeat:
  speed/wall: 10502.70 env steps/sec / 17.5498s
  identity: exact against r14
  stable_speed_claim_allowed: false
  wall spread r14/r15: 24.95% of median
  proof: train/update/refresh 22/22/22, parent replay 0, payload bytes 0

r16 longer same-mechanism row:
  window: 724 measured / 180 warmup
  speed/wall: 6491.80 env steps/sec / 114.202s
  expected train/update/refresh: 90 / 90 / 90
  owner train sample/update: 52.807s / 3.236s
  digest/state_dict/host clone: 0 / 0 / 0

r17 fast-batch composition attempt:
  result: failed before row creation
  problem: compact_owned_loop_fused_learner_batch requires compact_owned_loop_entrypoint
```

Decision:

- r14 is useful, but not stable.
- Longer rows show the remaining problem is not model-state digest/clone; it is
  owner-search train sampling / learner-batch materialization at steady state.
- The existing fused/tensor-native learner-batch path is gated behind the
  compact-owned-loop entrypoint and does not currently compose with the
  owner-search inline route. Next implementation should wire that path, or an
  equivalent owner-search-local batch path, before another promotion row.

## OPT-132 r29-r34 Vector RNG, Refresh-4, Threaded Background Maintenance

Date: 2026-06-05

Artifacts:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-inline-coalesced-directautoreset-vectorrng-fusedtensor-b1024a1-normal-unroll2-m724-w180-r29-20260605/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-inline-coalesced-directautoreset-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r30-20260605/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-inline-coalesced-directautoreset-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r31-repeat-20260605/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-threaded-coalesced-directautoreset-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r32-20260605/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-threaded-coalesced-directautoreset-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m724-w180-r33-repeat-20260605/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-threaded-coalesced-directautoreset-vectorrng-refresh4-fusedtensor-b1024a1-normal-unroll2-m1448-w360-r34-long-20260605/compact_coach_speed_row_modal_report.json
```

Read:

```text
baseline:
  OPT-104 validation-only: 12689.381637 env/s, 14.5255305s wall

code patch:
  vectorized seeded_source_math_random_history
  exact scalar parity checked
  focused reset/autoreset tests passed

r29 inline refresh1:
  speed/wall: 12057.81 env/s / 61.485s
  actor_autoreset: 2.321s
  result: reset fixed, cadence mismatch remained below baseline

r30 inline refresh4:
  speed/wall: 14230.53 env/s / 52.098s
  target wall at OPT-104 rate: 58.425s
  result: same-cadence win, not enough alone

r31 inline refresh4 repeat:
  speed/wall: 11794.65 env/s / 62.857s
  actor_autoreset: 2.406s
  slab: 42.728s
  result: inline owner/slab dispatch still unstable

r32 threaded refresh4:
  speed/wall: 14002.12 env/s / 52.947s
  slab: 21.881s
  actions served while maintenance pending: 608
  result: background maintenance reduces slab and beats baseline

r33 threaded refresh4 repeat:
  speed/wall: 13640.60 env/s / 54.351s
  slab: 21.166s
  actions served while maintenance pending: 543
  result: exact repeat stays above baseline

r34 threaded refresh4 long:
  window: 1448 measured / 360 warmup
  speed/wall: 13145.34 env/s / 112.797s
  baseline-equivalent wall: 116.850s
  actions served while maintenance pending: 1131
  result: longer row stays positive but margin is modest
```

Decision:

- Vectorized reset RNG is a real speed patch and must stay.
- Refresh interval must match OPT-104 interval `4` for the accepted comparison.
- Inline owner maintenance is unstable after the reset fix: r30 won, r31 failed.
- Threaded/background owner maintenance is the current best candidate: r32 and
  r33 repeated above baseline, and r34 stayed positive over a 2x-long window.
- This is not the finish. The speedup is only about `1.04x-1.10x`, so the next
  implementation must attack measured remaining surfaces rather than rerun the
  same candidate or add more attribution timers.

## OPT-132 Direct Owner-Search Slab Bypass Local Proof

Date: 2026-06-05

Artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-local-owner-search-slab-bypass-knownshape-smoke-r2-20260605/compact_coach_speed_row_smoke_report.json
```

Read:

```text
status: local wiring/proof only, not H100 speed evidence
speed/wall: 786.29 env/s / 0.1221s on tiny CPU known-shape smoke
requested: owner_search_threaded_proxy + owner_search_slab_bypass
bypass kind: owner_search_direct_transition_stepper_v1
compact_rollout_slab_bypassed: true
compact_rollout_slab_general_replay_row_builder_used: false
parent committed/stored rows: 0 / 0
bypass parent committed/stored proof rows: 0 / 0
action-only owner result: true
owner materializes replay: true
parent slab commits replay: false
search-result payload bytes: 0
owner replay appends / train requests / learner updates: 12 / 3 / 3
owner maintenance pending / inflight: 0 / false
action feedback verified / mismatches: true / 0
```

Implementation:

- Added `CompactOwnerSearchDirectStepperV1` and CLI flag
  `--owner-search-slab-bypass`.
- Direct stepper still returns the existing profile-manager step envelope, but
  bypasses the general slab replay-row builder and stages owner transition
  handles directly.
- Speed-row proof now requires missing bypass fields to fail closed; default
  missing `0`/`False` counters are not accepted.
- Speed-row evidence validation now permits owner-search owner-side prebuilt
  learner batches without requiring the parent fused-learner-batch config, but
  only when action-only replay is owner materialized and parent sample/learner
  gates are zero.

Validation:

```text
python -m py_compile src/curvyzero/training/compact_rollout_slab.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/compact_coach_speed_row.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row.py
uv run ruff check src/curvyzero/training/compact_rollout_slab.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/compact_coach_speed_row.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row.py
uv run pytest tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py -k owner_search -q
uv run pytest tests/test_compact_coach_speed_row.py::test_compact_coach_speed_row_allows_owner_search_prebuilt_batch_without_parent_fused tests/test_compact_coach_speed_row.py::test_compact_coach_speed_row_requires_tensor_native_replay_proof -q
```

Decision:

- This proves a real owner-graph/dataflow change locally.
- It does not yet justify an H100 speed claim by itself.
- Next implementation should batch/coalesce owner replay transition transport
  so logical transition count can exceed transport entry count, or move a
  larger fixed-buffer surface. Do not return to owner-worker-mode polish.

## OPT-132 Owner Deferred One-Simulation Replay Proof

Date: 2026-06-07

Status: local proof and report plumbing only; no H100 speed evidence.

Implementation:

- Owner service now aggregates inner deferred replay flush metadata from device
  replay payloads and fails closed on missing materialized-on-flush proof,
  identity mismatch, or nonzero model-refresh-crossed count.
- Direct transition-batch replay now reports requested-deferral proof fields
  under `compact_owner_search_direct_transition_batch_replay_*`.
- Speed-row proof rejects requested-deferral rows unless flush/materialization/
  identity/recurrent-call counts match transition count, final pending is `0`,
  model-refresh-crossed is `0`, and replay-payload D2H bytes are `0`.
- Local smoke and Modal wrappers now pass
  `--compact-torch-defer-one-simulation-replay-payload` into
  `CompactTorchCompileConfig` and preserve the request bit in evidence.

Validation:

```text
python -m py_compile src/curvyzero/training/compact_owner_search_service.py src/curvyzero/training/compact_coach_speed_row.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py scripts/build_compact_coach_compatibility_speed_row_refresh.py src/curvyzero/infra/modal/compact_coach_speed_row.py tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row.py tests/test_compact_coach_compatibility_speed_row_refresh.py
uv run ruff check src/curvyzero/training/compact_owner_search_service.py src/curvyzero/training/compact_coach_speed_row.py scripts/build_compact_coach_speed_row_smoke.py scripts/run_compact_coach_speed_row_modal_smoke.py scripts/build_compact_coach_compatibility_speed_row_refresh.py src/curvyzero/infra/modal/compact_coach_speed_row.py tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row.py tests/test_compact_coach_compatibility_speed_row_refresh.py
uv run pytest tests/test_compact_owner_search_service.py::test_owner_action_uses_inner_two_phase_device_replay_for_resident_roots tests/test_compact_owner_search_service.py::test_owner_inner_two_phase_deferred_replay_fails_closed_on_identity_cross tests/test_compact_owner_search_service.py::test_owner_search_service_routes_transition_batch_to_direct_replay_store tests/test_compact_owner_search_service.py::test_owner_search_service_drain_projects_direct_transition_batch_replay_metadata tests/test_compact_torch_search_service.py::test_one_simulation_deferred_replay_payload_moves_recurrent_to_flush tests/test_compact_torch_search_service.py::test_deferred_one_simulation_replay_flush_fails_after_model_identity_drift -q
uv run pytest tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_loaded_checkpoint_mode_emits_loaded_identity tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_backend_factory_selects_floor_decomposition_services tests/test_compact_coach_speed_row_smoke.py::test_direct_transition_batch_replay_store_strips_terminal_metadata_before_resident_append tests/test_compact_coach_speed_row.py::test_compact_coach_speed_row_evidence_binds_manifest_result_and_lifecycle tests/test_compact_coach_speed_row.py::test_compact_coach_speed_row_projection_fields_remain_nonproduction -q
git diff --check
```

Decision:

- This closes the local owner-loop proof gap for the default-off deferral lane.
- It does not prove speed and should not be launched on H100 before one
  end-to-end local flagged smoke shows report-level proof fields.

Local flagged smoke closeout:

```text
run_id: opt132-local-owner-deferred-one-sim-direct-transition-batch-smoke-20260607-r7
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132-local-owner-deferred-one-sim-direct-transition-batch-smoke-20260607-r7/compact_coach_speed_row_smoke_report.json
status: ok=true
direct transition count: 12
direct deferred flush/materialized/identity/recurrent: 12 / 12 / 12 / 12
direct model-refresh-crossed: 0
direct replay-payload D2H bytes: 0.0
direct pending final: 4
owner-inner deferred flush/materialized/identity/recurrent: 16 / 16 / 16 / 16
owner-inner model-refresh-crossed: 0
owner-inner pending final: 0
```

Interpretation:

- r1-r6 were useful fail-closed proof bugs, not speed evidence: inference
  tensors leaked into learner replay, digest strings polluted numeric telemetry
  aggregation, deferred handles could survive into model refresh, and the first
  validator treated direct current-action staging as a replay leak.
- r7 closes the local correctness gate for the flagged owner/deferred/direct
  transition-batch path.
- r7 is still CPU/local and tiny. It justified at most one H100 probe of this
  default-off path; that probe has now run and is recorded below as speed
  rejected. r7 itself was never a speedup claim or a `10x` narrative.

Projection preflight:

`opt132-local-owner-deferred-one-sim-direct-transition-batch-smoke-20260607-r8-projection`
reran the same local shape after adding source-profile projection of the
owner-inner deferred proof fields. It passed with `ok=true`: direct transition
count `12`, direct deferred flush/materialized/identity/recurrent
`12/12/12/12`, direct model-refresh-crossed `0`, replay-payload D2H bytes
`0.0`, owner-inner deferred flush/materialized/identity/recurrent
`16/16/16/16`, owner-inner model-refresh-crossed `0`, owner-inner pending
final `0`, and action-only owner payload/visit/root bytes `0/0/0`. This closes
the local report-projection preflight for the optional single H100 probe.

H100 one-probe closeout:

```text
run: opt132-h100-owner-deferred-one-sim-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607
status: proof clean, speed rejected
speed: 13691.98 env/s
wall: 54.1467s
baseline delta: 1.079x OPT-104 by env/s
current fastest columnar r2 delta: 0.864x by env/s
direct transitions: 724
columnar append used: true
direct deferred flush/materialized/identity/recurrent: 724 / 724 / 724 / 724
direct model-refresh-crossed: 0
direct replay-payload D2H bytes: 0.0
direct pending final: 2
owner-inner pending final: 0
normal-death promotion gate: true
terminal sample / target rows: 512 / 512
```

Interpretation:

- The deferral mechanics worked and failed closed on H100.
- It did not move the whole loop in the needed direction. Direct append grew
  from the columnar r2 reference `14.169s` to `22.377s`; deferred replay flush
  added `3.613s` and device replay payload flush added `4.381s`.
- Do not repeat this lane unless a later implementation changes where the
  deferred work runs or overlaps it with a different owner boundary. The next
  P0 returns to broader owner-resident root/search/parent-wait/learner-
  publication surfaces.

## OPT-132 Owner Async-Learner Overlap Probe

Date: 2026-06-07

Local preflight:

```text
run: opt132-local-owner-async-learner-columnar-directtable-smoke-20260607-r1
status: ok=true, local correctness only
direct transitions / batches / transport: 16 / 4 / 4
columnar append used: true
async enabled/kind: true / in_process_thread_v1
async submit/completed/pending: 4 / 4 / 0
async max pending observed: 2
actions while async learner pending: 9
async wait: 0.0545s
maintenance pending/failed: 0 / false
```

H100 closeout:

```text
run: opt132-h100-owner-asynclearner-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607
status: proof clean, speed rejected
speed: 12954.74 env/s
wall: 57.2282s
baseline delta: 1.021x OPT-104 by env/s
current fastest columnar r2 delta: 0.817x by env/s
direct transitions / batches / transport: 724 / 181 / 181
columnar append used: true
async enabled/kind/max pending: true / in_process_thread_v1 / 2
async submit/completed/pending: 90 / 90 / 0
async max pending observed: 2
actions while async learner pending: 510
async wait: 0.851s
async failed: false
normal-death promotion gate: true
terminal sample / target rows: 512 / 512
```

Interpretation:

- The mechanism changed a real owner boundary and proved overlap: actions were
  served while async learner work was pending.
- It still failed the speed criterion. Worker replay append rose to `21.246s`,
  learner train to `13.231s`, worker search to `14.049s`, parent wait to
  `19.095s`, and observation to `12.305s`, versus columnar r2's
  `14.219s`, `9.778s`, `13.295s`, `17.655s`, and `9.867s`.
- Do not repeat this in-process async learner lane unchanged. It is another
  proof-clean owner-boundary falsification, not the 2x path.

## OPT-132 Owner Root-Cache Slice Local Patch

Date: 2026-06-07

Local code/proof patch:

```text
scope:
  src/curvyzero/training/compact_owner_search_service.py
  tests/test_compact_owner_search_service.py
mechanism:
  deferred owner maintenance now slices the staged root-batch cache to replay
  record_indices, next_record_indices, and current actor_step when entries are
  transition-batch or index-entry shaped
fallback:
  opaque replay entries keep the previous full-cache snapshot behavior and
  increment compact_owner_search_owner_maintenance_root_cache_snapshot_full_fallback_count
proof counters:
  compact_owner_search_owner_maintenance_root_cache_snapshot_count
  compact_owner_search_owner_maintenance_root_cache_snapshot_full_entry_count
  compact_owner_search_owner_maintenance_root_cache_snapshot_retained_entry_count
  compact_owner_search_owner_maintenance_root_cache_snapshot_required_entry_count
  compact_owner_search_owner_maintenance_root_cache_snapshot_dropped_entry_count
  compact_owner_search_owner_maintenance_root_cache_snapshot_full_fallback_count
focused test:
  seeded owner cache keys 0,1,2,6; deferred transition batch references 0,1,2;
  staged direct append sees keys (0,1,2,5), proving unrelated key 6 is dropped
validation:
  uv run ruff check src/curvyzero/training/compact_owner_search_service.py tests/test_compact_owner_search_service.py
  uv run pytest tests/test_compact_owner_search_service.py -q -k direct_transition_batch
  uv run pytest tests/test_compact_owner_search_service.py -q
result:
  ruff passed
  focused pytest 1 passed, 46 deselected, 2 warnings
  full owner-search service pytest 47 passed, 2 warnings
```

Interpretation:

- This reduces a real Python-object fanout inside the deferred owner path and
  adds fail-closed counters for future rows.
- It is not a speed claim and is probably too small to explain the missing
  `2x` by itself.
- The next larger code lane should remove a full boundary, either by deriving
  replay transitions owner-locally or by returning action results through a
  fixed reusable slot instead of a per-step Python payload envelope.

## OPT-132 Fixed Action-Result Slot Local Patch

Date: 2026-06-07

Local code/proof patch:

```text
scope:
  src/curvyzero/training/compact_owner_search_service.py
  tests/test_compact_owner_search_service.py
mechanism:
  same-process inline/threaded owner proxies can allocate an action-result slot,
  send action_result_slot_id on the root-build action request, have the owner
  write the full action result into CompactOwnerActionResultSlotTableV1, and
  receive only a tiny slot stub through the worker result queue
process owner:
  rejected for this mode until a real shared-memory result table exists
proof counters:
  compact_owner_search_fixed_action_result_buffer_requested
  compact_owner_search_fixed_action_result_buffer_used
  compact_owner_search_fixed_action_result_buffer_slot_count
  compact_owner_search_fixed_action_result_buffer_acquire_count
  compact_owner_search_fixed_action_result_buffer_write_count
  compact_owner_search_fixed_action_result_buffer_read_count
  compact_owner_search_fixed_action_result_buffer_pending_slot_count
  compact_owner_search_fixed_action_result_buffer_wire_result_bytes
  compact_owner_search_fixed_action_result_buffer_full_result_bytes
focused proof:
  threaded direct-root build-request step uses the fixed slot; wire result bytes
  are positive and less than full result bytes; write/read counts are 1/1 and
  pending slot count returns to 0
validation:
  uv run ruff check src/curvyzero/training/compact_owner_search_service.py tests/test_compact_owner_search_service.py
  uv run pytest tests/test_compact_owner_search_service.py -q -k "root_build_request or direct_transition_batch"
  uv run pytest tests/test_compact_owner_search_service.py -q
result:
  ruff passed
  focused pytest 5 passed, 42 deselected, 2 warnings
  full owner-search service pytest 47 passed, 2 warnings
```

Interpretation:

- This is the smaller sidecar-recommended owner RPC-boundary patch. It attacks
  the per-step result envelope, not replay gather.
- At this point it was owner-service local proof only. The follow-up
  2026-06-08 section below closes launcher/speed-row plumbing and local smoke.
- If this path does not move local/report bytes or H100 parent wait, the next
  deeper cut is owner-local transition derivation that removes parent replay
  transport entries entirely.

## OPT-132 Fixed Action-Result Slot Speed-Row Proof

Date: 2026-06-08

Local speed-row/launcher proof is now closed; this is not H100 speed evidence.

```text
run:
  opt-fixed-action-result-slot-local-smoke-proof5-20260608
report:
  artifacts/local/curvytron_compact_coach_speed_row_results/
    opt-fixed-action-result-slot-local-smoke-proof5-20260608/
    compact_coach_speed_row_smoke_report.json
status:
  ok true, complete
tiny CPU speed:
  548.91 env/s, not comparable to OPT-104
fixed-slot proof:
  requested/used true/true
  slot count 4
  acquire/write/read 5/5/5
  pending slots 0
  wire/full result bytes 413/712
preserved owner proof:
  action-only true
  direct-root true
  owner owner_search_worker
  owner train/expected train 1/1
  tensor-native replay true
```

The route is intentionally fail-closed:

- Fixed action-result buffer now requires `--owner-search-defer-maintenance` in
  the local speed-row builder, local launcher path, and remote Modal producer.
- The fixed-slot proof requires
  `compact_owner_search_owner_defer_maintenance=true`.
- Owner-owned fused learner-batch proof now accepts the owner-owned training
  path without requiring parent
  `compact_rollout_slab_learner_gate_prebuilt_batch_used=true`, and instead
  requires owner learner timing/count proof.
- The proof-only lifecycle shim used here has
  `promotion_eligible=false`; it exists only to satisfy the local lifecycle
  smoke schema.

Validation:

```text
uv run ruff check ... touched owner/speed-row/modal/test files
uv run pytest tests/test_compact_coach_speed_row_smoke.py -q -k "owner_search_owned_fused or owner_search_fused_tensor_native or fixed_action_result"
uv run pytest tests/test_compact_owner_search_service.py -q
uv run pytest tests/test_compact_coach_speed_row_smoke.py -q
```

Result: ruff passed; focused speed-row slice `7 passed`; owner-search service
`47 passed`; full speed-row smoke `96 passed`.

Decision: one same-work H100 row is now the next decision gate. Compare against
OPT-104 (`12689.38 env/s`) and columnar r2 (`15852.67 env/s`). If parent
wait/result transport is speed-neutral, close fixed-slot as support and switch
to owner-local transition derivation or a broader fixed owner-buffer/root/search
patch.

## OPT-132 Fixed Action-Result Slot H100 Decision

Date: 2026-06-08

```text
run:
  opt132-h100-fixed-action-result-slot-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608
function_call_id:
  fc-01KTJETC7G7JQ6RMCNM8N5FNBC
report:
  artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-h100-fixed-action-result-slot-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608/
    compact_coach_speed_row_modal_report.json
status:
  proof clean, speed rejected as main lane
speed/wall:
  12794.42 env/s / 57.9453s
env steps:
  741376
baseline reads:
  vs OPT-104 12689.38 env/s: 1.008x
  vs columnar r2 15852.67 env/s: 0.807x
fixed-slot proof:
  requested/used true/true
  slot count 4
  acquire/write/read 904/904/904
  pending 0
  wire/full result bytes 414/4837
preserved proof:
  parent root builder calls 0
  parent compact root-batch objects sent 0
  parent replay rows 0/0
  action-only true
  terminal sample/target 512/512
  normal-death promotion gate true
  accepted/tensor-native violations []
```

Timing read:

```text
fixed-slot row:
  parent wait 19.035s
  compact slab 21.757s
  worker search 13.916s
  replay append 19.931s
  learner train 12.880s
  observation 13.259s
  actor step 20.088s
root-build-request r1:
  parent wait 26.151s
  compact slab 29.850s
  worker search 16.422s
  replay append 22.783s
  learner train 15.526s
  observation 13.196s
  actor step 19.101s
columnar r2:
  parent wait 17.655s
  compact slab 20.915s
  worker search 13.295s
  replay append 14.219s
  learner train 9.778s
  observation 9.867s
  actor step 14.055s
```

Interpretation:

- The fixed-slot mechanism is real and useful support: it shrank the full
  result envelope (`414` wire bytes versus `4837` full bytes) and improved the
  earlier root-build-request row (`11327.75 -> 12794.42 env/s`, parent wait
  `26.151s -> 19.035s`).
- It is not the speed path. It remains slower than the fastest columnar support
  stack, with replay append, learner train, observation, and actor step all
  materially worse than columnar r2.
- Close fixed-slot unchanged. The next implementation should remove a larger
  owner/local transition or fixed-buffer/root/search boundary, not polish
  result-slot bookkeeping.

## OPT-132 Manager Action-Dispatch Overlap Local Proof

Date: 2026-06-08

Status: local implementation/proof and launcher wiring only; no H100 speed
evidence yet.

What changed:

- `CompactOwnerSearchDirectStepperV1` now exposes `submit_step()` and
  `resolve_step()` around direct-root owner action search.
- `HybridBatchedObservationProfileManager.step()` can submit the owner action
  request, run real parent post-slab work, then resolve before returning the
  next joint action when `compact_owner_action_dispatch_step_overlap=true`.
- The overlapped parent work is the existing profile-loop work after the slab
  point: optional batched probe, scalar/no-scalar materialization, payload
  pickle, resident replay snapshot, render-state snapshot, and minimal/full
  step payload snapshot.
- Local speed-row and Modal launcher paths now thread
  `--compact-owner-action-dispatch-step-overlap` and require proof fields when
  requested.

Required proof before any speed read:

```text
compact_owner_action_dispatch_step_overlap_enabled true
compact_owner_action_dispatch_step_overlap_proof_passed true
compact_rollout_slab_action_dispatch_step_overlap_supported true
compact_rollout_slab_action_dispatch_step_overlap_used true
compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait true
compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper false
submit_count == resolve_count == measured iterations
pending_count 0
max_pending_count > 0
parent_work_sec > 0
compact_owner_search_action_dispatch_handle_result_wait_in_submit_count 0
```

Validation:

```text
uv run ruff check src/curvyzero/training/compact_rollout_slab.py \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  scripts/build_compact_coach_speed_row_smoke.py \
  scripts/run_compact_coach_speed_row_modal_smoke.py \
  src/curvyzero/infra/modal/compact_coach_speed_row.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_compact_owner_search_service.py \
  tests/test_compact_coach_speed_row_smoke.py
result: all checks passed

uv run pytest tests/test_source_state_hybrid_observation_profile.py -q \
  -k 'action_dispatch_step_overlap or owner_action_step_boundary_uses_direct_root_build_request'
result: 2 passed, 120 deselected

uv run pytest tests/test_compact_owner_search_service.py -q
result: 55 passed

uv run pytest tests/test_compact_search_replay_contract.py -q
result: 47 passed

uv run pytest tests/test_compact_coach_speed_row_smoke.py -q
result: 99 passed
```

Interpretation:

- This is the first owner wait/search dispatch slice where the profile loop
  actually places real parent work between submit and resolve.
- It is still not proof of speed. The next H100 row must carry the exact
  overlap proof; otherwise it is diagnostic/invalid.
- If the row is flat or negative versus columnar r2, do not repeat unchanged.
  Move to fixed owner mechanics-frame handles or learner publication/update
  tickets/refs.

## OPT-132 Overlap/Proxy Launch-Readiness Correction

Date: 2026-06-08

Status: local proof and launch/readout hardening only; no H100 speed evidence.

Correction:

- The owner/proxy transition-closure rung was local-clean, but the public
  speed-row CLI and Modal launch/readout stack did not expose
  `--owner-search-owner-proxy-transition-closure`. A guarded H100 overlap row
  could not honestly preserve proxy closure before this fix.
- The local speed-row CLI, local Modal launcher, remote Modal producer,
  config/result summaries, and proof/report allowlists now carry the proxy
  closure request bit and proof fields.
- The overlap proof now requires cumulative slab and owner sync-wrapper counts
  `0`, owner completed-at-submit count `0`, and owner wait-in-submit count `0`.
  These counters must be present on the raw profile payload before projection
  fills defaults, so omitted counters fail closed instead of becoming fake
  zeros.

Latest validation after hardening:

```text
uv run ruff check scripts/build_compact_coach_speed_row_smoke.py
result: all checks passed

uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_owner_action_dispatch_step_overlap_resolves_after_parent_payload \
  tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_owner_action_step_boundary_uses_direct_root_build_request -q
result: 2 passed

uv run pytest tests/test_compact_owner_search_service.py -q
result: 55 passed

uv run pytest tests/test_compact_search_replay_contract.py -q
result: 47 passed

uv run pytest tests/test_compact_coach_speed_row_smoke.py -q
result: 99 passed
```

Next H100 row, if launched, must use the columnar/direct-table support stack,
explicit `--owner-search-owner-proxy-transition-closure`, and explicit
`--compact-owner-action-dispatch-step-overlap`. Suggested run id:
`opt132-h100-action-dispatch-overlap-proxyclosure-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608`.

## OPT-132 Overlap/Proxy H100 r1 Pre-Row Failure

Date: 2026-06-08

Run id:
`opt132-h100-action-dispatch-overlap-proxyclosure-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608`

Status: failed before speed row; not speed evidence.

Failure:

```text
ReplayCompatibilityError:
  owner-proxy transition closure requires owner learning enabled
```

Interpretation:

- This exposed a real warmup interaction, not a speed result. The profile loop
  disables owner learning for warmup iterations. Proxy transition closure ran
  during that disabled period and aborted instead of suppressing warmup replay
  like the existing owner replay append path.
- Fix: owner/proxy closure now counts the closure request, drops pending
  warmup transition state, and returns zero timing when owner learning is
  disabled. This preserves the final warmup action-frame request path for the
  first measured transition while avoiding warmup replay append.

Validation after fix:

```text
uv run ruff check src/curvyzero/training/compact_owner_search_service.py \
  tests/test_compact_owner_search_service.py
result: all checks passed

uv run pytest tests/test_compact_owner_search_service.py -q \
  -k 'owner_proxy_transition_closure or learning_gate_suppresses_warmup_replay'
result: 4 passed, 52 deselected

uv run pytest tests/test_compact_owner_search_service.py -q
result: 56 passed

uv run pytest tests/test_compact_coach_speed_row_smoke.py -q \
  -k 'owner_search_slab_bypass or remote_modal_owner_search_config_projects_digest_deferral or fixed_action_result_buffer_without_deferred_maintenance or accepted_fast_path_preset_rejects_owner_search_overrides'
result: 8 passed, 91 deselected

uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_owner_action_dispatch_step_overlap_resolves_after_parent_payload \
  tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_owner_action_step_boundary_uses_direct_root_build_request -q
result: 2 passed
```

Next: relaunch as r2 with the same guarded flags. Treat r1 as invalid/pre-row,
not as a proof-clean or speed-failed lane.

## OPT-132 Overlap/Proxy H100 r2 Proof-Wrapper Failure

Date: 2026-06-08

Run id:
`opt132-h100-action-dispatch-overlap-proxyclosure-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r2-20260608`

Status: produced a profile payload but failed the proof wrapper; do not promote
as a normal speed row artifact.

Failure:

```text
ValueError:
  owner-search worker transition-batch count mismatch
```

What actually happened:

- The overlap proof passed: submit/resolve `904/904`, sync-wrapper count `0`,
  completed-at-submit `0`, wait-in-submit `0`, pending `0`, max pending `1`.
- Proxy closure proof fields were coherent: closed/transition `724/724`,
  batch/transport `181/181`, applied-action verification `724`, mismatch `0`,
  fallback `0/none`, digest verified.
- Direct/proxy/drained transition-batch counters matched `181/724/181`, but
  stale generic replay-append primary counters reported the last batch only:
  `1/4/1`. The proof helper incorrectly preferred any positive primary counter
  over the drained aggregate.

Fix:

- `_transition_batch_worker_counter()` now prefers the drained aggregate when
  present, and falls back to the primary counter only when no drained aggregate
  exists.
- Regression added for stale primary `1/4/1` with drained aggregate
  `2/8/2`; the existing negative test still fails when no drained aggregate is
  present.

Validation:

```text
uv run ruff check scripts/build_compact_coach_speed_row_smoke.py \
  tests/test_compact_coach_speed_row_smoke.py
result: all checks passed

uv run pytest tests/test_compact_coach_speed_row_smoke.py -q \
  -k 'owner_search_slab_bypass or remote_modal_owner_search_config_projects_digest_deferral or fixed_action_result_buffer_without_deferred_maintenance or accepted_fast_path_preset_rejects_owner_search_overrides'
result: 8 passed, 91 deselected
```

Next: relaunch as r3 with the same guarded flags so the normal report/evidence
artifacts are emitted under the corrected proof.

## OPT-132 Overlap/Proxy H100 r3 Valid Speed Read

Date: 2026-06-08

Run id:
`opt132-h100-action-dispatch-overlap-proxyclosure-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r3-20260608`

Status: valid same-work H100 row; proof clean; speed-rejected unchanged.

Speed:

```text
speed currency: compact_trainer_env_steps_per_sec
env steps collected: 741376
measured/warmup steps: 724 / 180
measured wall: 47.701594815s
speed: 15541.954160552945 env/s
vs OPT-104 baseline: 1.2248x
vs columnar/direct-table r2: 0.9804x
gap to 2x OPT-104: 9836.81 env/s
```

Proof:

```text
compact_owner_action_dispatch_step_overlap_proof_passed true
submit/resolve 904/904
sync_wrapper false
sync_wrapper_count 0
completed_at_submit_count 0
result_wait_in_submit_count 0
pending_count 0
max_pending_count 1

owner proxy transition closure requested true
owner proxy transition closure used true
closed/transition 724/724
batch/transport 181/181
fallback 0/none
pending 0
digest verified true
parent previous-transition closure count 0
applied-action mismatch count 0

direct transition replay used true
direct transition batches/transitions/transport 181/724/181
```

Timing compared with current best columnar/direct-table r2:

```text
columnar r2:
  speed/wall 15852.67 env/s / 46.7666s
  parent wait 17.655s
  worker search 13.295s
  worker replay append 14.219s
  learner train 9.778s
  search dispatch wall 18.945s
  slab 20.915s

overlap/proxy r3:
  speed/wall 15541.95 env/s / 47.7016s
  parent wait 19.985s
  worker search 13.193s
  worker replay append 19.009s
  learner train 11.235s
  search dispatch wall 21.025s
  slab 21.945s
```

Decision:

- This was a real proof-clean row, not a wrapper failure.
- It did not beat the current best support stack and remains far below the
  near-target `2x` row.
- Do not repeat overlap+proxy unchanged. Preserve the proof counters as guards
  for broader owner-boundary work.
- The next implementation target is the owner-resident mechanics/root step-
  frame handle ring, unless a local learner-publication ticket patch is made
  disjointly and proves a larger owner graph change.

## Local Slot-Backed Mechanics Step-Frame Proof

Date: 2026-06-08

Status: local proof only; not H100 speed evidence.

What changed:

```text
manager publishes fixed owner mechanics frame slots
schema curvyzero_compact_owner_mechanics_step_frame_slot/v1
handle schema curvyzero_compact_owner_mechanics_step_frame_handle/v1
direct root request reads slot arrays
legacy step-view builder avoided
parent compact-batch builder avoided
parent root request from-batch helper avoided
parent dense-action reconstruction avoided
root sidecar array bytes/field count 0/0
```

Validation:

```text
ruff touched files: passed
tests/test_source_state_hybrid_observation_profile.py
  -k owner_action_step_boundary or owner_mechanics_step_frame or owner_action_dispatch_step_overlap
  5 passed
tests/test_compact_owner_search_service.py
  -k direct_root or owner_proxy_transition_closure or action_dispatch_step_overlap
  11 passed
tests/test_compact_coach_speed_row_smoke.py
  -k owner_search_slab_bypass or accepted_fast_path_preset_rejects_owner_search_overrides or remote_modal_owner_search_config_projects_digest_deferral or owner_search_direct_root_build_request_proof_fails_closed or modal_speed_row_report_projects_owner_search_slab_proxy_fields
  7 passed
```

Decision:

- Preserve the slot/root proof as a guard.
- Do not launch H100 for this rung alone.
- Next H100-eligible code must move a broader owner graph surface than this
  local slot/root request proof.

## OPT-132 Selected-Maintained Direct-Prebuilt Falsifier

Date: 2026-06-09

Local benchmark:

```text
artifact: /tmp/selected_maintained_replay_benchmark.json
status: required_pass true
shape: records 724, sample rows 512, sampled groups 420, CPU
current replay-ring sample-build median: 0.0559639375s
selected-maintained sample-build median: 0.0167085210s
local surface speedup: 3.349x
impl/source: selected_maintained_record_table_gather_v1 / selected_maintained_record_table_v1
```

H100 row:

```text
run: opt132-h100-selected-maintained-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260609
currency: H100 speed row, proof clean, speed rejected
shape: B1024/A1, normal death, 724 measured, 180 warmup, sample batch 512, unroll 2
report:
  artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-h100-selected-maintained-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260609/
    compact_coach_speed_row_modal_report.json
function call: fc-01KTPC72T36E5EJ5VE5SZ5SY9S
```

Speed read:

```text
env/s: 12028.547859804483
wall: 61.634705090000004
env steps: 741376
vs columnar r2: 0.7588x
vs OPT-104: 0.9479x
```

Proof and timing read:

```text
accepted/tensor-native/cache/unroll violations: [] / [] / [] / []
impl/source: selected_maintained_record_table_gather_v1 / selected_maintained_record_table_v1
fast metadata requested/used/groups: true / true / 352
table rows/reused/missing: 512 / 352 / 0
tensor-native table concat/gather: 0.003224s / 0.073162s
sample gate: 22.5805s
learner gate: 7.0617s
owner learner train: 31.7680s
worker search total: 15.9839s
direct/ring append: 23.8506s / 20.5126s
```

Decision:

- This falsifies the selected-maintained path as a default H100 speed lane.
- The local CPU win inverted on H100 because the selected path does many
  per-record selected gathers across hundreds of groups; that is the wrong GPU
  execution shape.
- Code now keeps selected-maintained gather explicit/default-off behind
  `compact_muzero_learner_batch_tensor_native_replay_selected_maintained_gather`.
- Do not repeat this row unchanged. The next replay/sample attempt must be a
  fused/batched selected gather or part of a broader fixed owner-buffer/root/
  search/learner-publication patch.

## OPT-132 Production Fixed-SoA Handle Consumption Local Proof

Date: 2026-06-09

Status: local contract proof only; not H100 speed evidence.

Code:

```text
scripts/build_compact_coach_speed_row_smoke.py
  owner_search_fixed_soa_replay now requests:
    compact_replay_fixed_soa_learner_batch_handle_ring=true
    compact_replay_fixed_soa_learner_batch_handle_ring_requested=true

src/curvyzero/training/compact_owned_loop.py
  learner boundary records:
    compact_owned_loop_learner_resident_batch_handle_requested
    compact_owned_loop_learner_resident_batch_handle_consumed
    compact_owned_loop_learner_resident_batch_handle_materialized_parent_fallback_count
    schema/handle/snapshot/checksum/sample-row/target-row/fallback details
```

Proof:

```text
fixed-SoA handle-ring metadata reaches owner-search replay store: true
owned-loop learner consumed resident handle after train_on_learner_batch: true
green-path fallback count: 0
fallback-path consumed flag: false
fallback-path materialized-parent fallback count: 1
real ring fixed-SoA handle-ring partial-terminal proof: preserved
production_speed_claim: false
```

Validation:

```text
uv run ruff check src/curvyzero/training/compact_owned_loop.py tests/test_compact_owned_loop.py scripts/build_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_smoke.py
  passed

uv run pytest tests/test_compact_coach_speed_row_smoke.py::test_owner_search_replay_store_metadata_receives_fused_tensor_native_flags tests/test_compact_owned_loop.py tests/test_source_state_hybrid_observation_profile.py::test_compact_replay_ring_samples_partial_terminal_entry_without_successor -q
  28 passed
```

Decision: no H100 for this rung. It makes the existing fixed-SoA
row/window path production-facing and fail-closed at the learner boundary, but
the handle is still resolved inline around a materialized learner batch. Next
gate is corrected local whole-loop timing with resident-handle consumed true
and materialized-parent fallback zero.

## OPT-132 Production Owner-Search Fixed-SoA Handle Whole-Loop Local Gate

Date: 2026-06-09

Status: local contract gate passed; not H100 speed evidence.

What was wrong:

```text
tiny steps=6 probe:
  expected train requests came from raw steps // train interval
  actual train requests came from submitted replay entries // train interval
  result: expected 2, actual 1

warmup=0, steps=64, transition-batch=4:
  replayable submitted entries are 60, not raw 64
  correct train requests are 15

opt107 envelope:
  terminal rows existed, but random learner sampling could hit a terminal window
  without sampling the terminal row itself
  normal-death contract requires a real terminal sample row
```

Code changes:

```text
scripts/build_compact_coach_speed_row_smoke.py
  _owner_search_expected_train_request_count()
    warmup=0 uses steps-1 replayable transitions
    warmup>0 can include the warmup tail transition
    slab-bypass transition batching drops the final unflushed partial batch
  owner-search proxy construction now passes that expected count

src/curvyzero/training/source_state_hybrid_observation_profile.py
  terminal sample forcing now checks actual terminal-row presence, not only
  terminal-window presence
```

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_smoke.py
  passed

uv run pytest tests/test_compact_coach_speed_row_smoke.py::test_owner_search_expected_train_count_tracks_replayable_transition_batches tests/test_compact_coach_speed_row_smoke.py::test_speed_row_smoke_projects_and_validates_owner_search_slab_proxy_fields tests/test_compact_coach_speed_row_smoke.py::test_direct_transition_batch_replay_store_strips_terminal_metadata_before_resident_append -q
  4 passed
```

Passing local rows:

```text
run: opt132-local-production-fixed-soa-handle-normaldeath-b128s64-r2-20260609
artifact: /private/tmp/curvy_fixed_soa_handle_timing/opt132-local-production-fixed-soa-handle-normaldeath-b128s64-r2-20260609/row_001_result.json
shape: B128/A1 steps64 warmup0 normal sample16 interval4 replay256 unroll2 cpu
wall/env-s: 20.3908s / 401.75 env/s
replay entries/direct transitions: 60 / 60
expected train / actual train: 15 / 15
resident handle consumed: true
resident/materialized-parent fallback: 0 / 0
normal-death gate: true

run: opt132-local-production-fixed-soa-handle-opt107-envelope-r2-20260609
artifact: /private/tmp/curvy_fixed_soa_handle_timing/opt132-local-production-fixed-soa-handle-opt107-envelope-r2-20260609/row_001_result.json
shape: B128/A1 steps64 warmup16 normal sample512 interval8 replay4096 refresh4 unroll2 cpu
wall/env-s: 31.8960s / 256.83 env/s
replay entries/direct transitions: 64 / 64
expected train / actual train: 8 / 8
model refresh requests / skips: 2 / 6
resident handle consumed: true
resident/materialized-parent fallback: 0 / 0
terminal sample rows / terminal target rows: 1 / 2
normal-death proof death rows: 8
normal-death gate: true
```

Decision:

- The local production owner-search fixed-SoA handle path is unblocked.
- The failures were not whole-loop speed variance; they were one cadence
  accounting bug and one terminal-proof sampling bug.
- This is still not a speed win. H100 should be reserved for a candidate that
  plausibly removes/overlaps a broad measured surface, or for one explicitly
  labeled support row if the next implementation cannot be locally projected.
