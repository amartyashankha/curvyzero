# Current State

Date: 2026-06-09

Read `goal.md` first. This is the short fact sheet.

## Plain Truth

- Current speed state changed on 2026-06-05 after r29-r34. We now have a
  modest, explained H100 speedup over OPT-104, but not the near-target `2x`
  win.
- Accepted baseline remains
  `opt104-h100-normal-death-learner-validation-only-compacttorch-r1-20260531`:
  `12689.381637 env steps/sec`, `14.5255305s` wall, B1024/A1, normal death,
  measured/warmup `180/45`, sample interval `8`, sample batch `512`, replay
  capacity `4096`, unroll `2`, train steps `1`, refresh interval `4`.
- New code speed patch: `src/curvyzero/env/vector_source_random.py` now
  vectorizes `seeded_source_math_random_history` with exact scalar parity.
  Microbench for `1024 x 64` random tape was about `95x` faster, and focused
  reset/autoreset tests pass.
- r29 proved the reset RNG patch hit the measured row: refresh-1 inline row
  improved actor autoreset to `2.321s` and ran `12057.81 env/s`, but remained
  below OPT-104 because it still used refresh interval `1`.
- r30 fixed the cadence mismatch and beat OPT-104 once:
  `14230.53 env/s`, `52.098s` wall over `741376` env steps, refresh interval
  `4`, proof clean. r31 exact inline repeat regressed to `11794.65 env/s` with
  actor autoreset still low (`2.406s`) but slab/owner dispatch back high
  (`42.728s`), so the remaining instability was not reset RNG.
- r32 switched to the existing threaded/background owner maintenance path at
  the same refresh-4 shape and beat OPT-104: `14002.12 env/s`, `52.947s` wall,
  `608` actions served while maintenance was pending, slab `21.881s`.
- r33 repeated the threaded/background candidate above baseline:
  `13640.60 env/s`, `54.351s` wall, `543` actions served while maintenance was
  pending, slab `21.166s`.
- r34 answered the longer-run variance question: `1448/360` threaded/background
  row ran `13145.34 env/s`, `112.797s` wall over `1482752` env steps versus
  baseline-equivalent `116.850s`. The margin is modest (`1.036x`) but positive.
- The fixed transition-batch slab-bypass increment was tried on the longer
  `724/180` H100 window. It proved real transport reduction (`724` logical
  replay transitions through `181` transport entries, parent rows `0/0`,
  legacy transition entries `0`, pending `0`, action mismatches `0`) but was
  speed-neutral: r2 ran `13618.91 env/s`, `54.437s`, essentially tied with r33
  and below r32. This falsifies "transport request count alone" as the next
  `2x` lever.
- The follow-up ring-batched append/cache-refresh patch did move the measured
  owner replay surface. It keeps the same transition-batch transport
  (`724 -> 181`) but appends each fixed batch into `_CompactReplayRingV1` as a
  batch and de-duplicates learner-ready/tensor-native cache refreshes. Local
  unit proof shows four sequential transitions build maintained tensor-native
  tables once for each eligible record. H100 r3/r4 repeated above r2:
  r3 `14394.77 env/s`, `51.503s`, replay append `20.204s`; r4
  `14160.27 env/s`, `52.356s`, replay append `18.636s`. Average r3/r4:
  `14277.52 env/s`, `51.930s`, replay append `19.420s`, versus r2
  `13618.91 env/s`, `54.437s`, replay append `31.926s`.
- The direct record-table builder then moved the H100 row again:
  `opt132-h100-direct-table-builder-b1024a1-normal-unroll2-m724-w180-r3-20260607`
  ran `15681.59 env/s`, `47.2768s` wall over `741376` env steps. It is the
  fastest maintained-table H100 decision input so far, about `1.24x` OPT-104 by
  env/s, but it is not repeat-proven and not close to `2x`. Nested owner learner
  telemetry proves `direct_record_table_v1`, direct build used `true`, table
  build `0.637s` for `9770` rows, worker replay append `13.879s`, ring append
  `9.848s`, and remaining `ring_entry_objects=724`.
- The first columnar-append launch (`...columnar...r1`) was not a meaningful
  speed row. The accepted-fast-path preset overwrote owner-search/direct-replay
  flags after parse and again in the remote launch path, so the row used
  `compact_torch_search_service` instead of `owner_search_threaded_proxy` and
  failed the returned-result preset guard. A launcher guard now rejects this
  invalid flag combination.
- The corrected no-preset columnar append row is the current fastest single
  H100 input:
  `opt132-h100-columnar-append-direct-table-b1024a1-normal-unroll2-m724-w180-r2-20260607`
  ran `15852.67 env/s`, `46.7666s` wall over `741376` env steps, about
  `1.25x` OPT-104 and only about `1.01x` faster than direct-table r3. Proof:
  owner-search threaded proxy, slab bypass, direct transition replay, columnar
  append used, `724` columnar slots, `ring_entry_objects=0`, direct prebuilt
  sample path used, direct group objects `0`, tensor-native
  `maintained_unroll2_table_gather_v1` from `maintained_record_table_v1`,
  table build `0.810s`, reused/missing records `719/0`. Timing still leaves
  worker replay append `14.219s`, direct append `14.169s`, ring append
  `11.523s`, index-row build `1.875s`, worker search `13.295s`, learner train
  `9.778s`, and parent wait `17.655s`.
- The lazy selected-table experiment is rejected as the next path:
  `opt132-h100-lazy-selected-table-columnar-b1024a1-normal-unroll2-m724-w180-r2-20260607`
  passed proof but slowed to `11179.26 env/s`, `66.3171s` wall. It cut append
  (`14.169s -> 10.224s`) and ring append (`11.523s -> 7.101s`) but moved the
  cost into owner train/sample (`9.778s -> 48.696s`, owner train sample
  `2.777s -> 41.640s`) and reintroduced `352` direct group objects. Keep lazy
  selected default-off unless a later design also removes group objects and the
  learner/sample blowup.
- The selected-maintained direct-prebuilt experiment is also rejected as a
  default H100 path. Local CPU benchmark
  `/tmp/selected_maintained_replay_benchmark.json` passed proof and made the
  selected replay sample-build surface `3.35x` faster
  (`0.0167s` vs `0.0560s`) for `512` sampled rows over `724` records, but the
  H100 row
  `opt132-h100-selected-maintained-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260609`
  was proof-clean and speed-bad: `12028.55 env/s`, `61.6347s`, only `0.759x`
  columnar r2 and `0.948x` OPT-104. It used
  `selected_maintained_record_table_gather_v1` from
  `selected_maintained_record_table_v1`, fast metadata `true`, selected groups
  `352`, fallback `0`, but sample gate rose to `22.581s`, owner learner train
  to `31.768s`, direct append to `23.851s`, and ring append to `20.513s`.
  Interpretation: per-record selected GPU gathers are many tiny
  Python/kernel-launch operations; local CPU timing inverted on H100. The code
  now keeps this path explicit/default-off behind
  `compact_muzero_learner_batch_tensor_native_replay_selected_maintained_gather`;
  production default returns to maintained-table concat plus single gather.
- The fixed-action-tape harness now reports a corrected local whole-loop
  denominator, not just env-step/autoreset timing. The owner-slot fixture behind
  `--run-owner-slot-ceiling` is closed-local support: mechanics/root/action
  feedback goes through fixed owner mechanics slots, root requests come from
  slots instead of batch helpers, `HybridCompactBatch` object count is `0`,
  root observation copy bytes are `0`, the previous transition now stages
  through `stage_replay_append_entries()`, and the owner-slot service validates
  replay-payload handles/digests before the local fixed replay-slot/sample-
  handle shim appends/sample-checks owner-selected rows. The same staged owner
  rows now drain into `_CompactReplayRingV1`, emit owner-context device replay
  index rows, sample unroll-1 through the resident ring path with
  `device_replay_index_rows_sample=true`, and build a resident device learner
  unroll-2 batch on a nonterminal fixture. Latest smoke
  `/private/tmp/curvy_owner_slot_device_rows_unroll2_b2_m3_w1_20260609.json`
  passed with replay appends/rows `3/12`, stage transport/transitions `3/3`,
  stage cache hit/miss/release/pending `3/0/3/1`, stage drained
  records/device rows `3/12`, replay-ring append calls/records/rows `3/3/12`,
  resident device sample batch `true`, learner unroll-2 built/rows `true/8`,
  learner shapes action/reward/value/policy `[8,2]/[8,2]/[8,3]/[8,3,3]`,
  host fallback `false`, parent replay objects / replay object entries /
  selected groups `0/0/0`, sample handles create/resolve/inline/pending
  `3/3/3/0`, sample rows/targets `8/8`, and `production_speed_claim=false`.
  A new owner-action-context bridge helper,
  `build_compact_replay_index_rows_v1_from_owner_action_context_payload()`,
  now builds real `CompactReplayIndexRowsV1` without full parent root
  batches/`HybridCompactBatch` and matches the trusted root-batch builder in
  replay-contract tests; its device sibling now builds
  `CompactDeviceReplayIndexRowsV1` from the same owner context. This is not
  H100 speed evidence. The active local gate is now fixed resident row/window
  slots or handle-ring sampling, then sample/feed production owner
  replay/sample/learner-batch handles without materialized-parent fallback.
- Proof status for r29-r34: tensor-native fallback `none`, direct autoreset
  rows equal terminal rows, terminal sample/target rows present and equal,
  accepted-fast-path violations `[]`, unroll2 violations `[]`.
- The next small materialization cut is now implemented locally:
  `_CompactReplayRingV1.sample_from_snapshot` can use maintained unroll-2
  tensor-native replay tables to build the prebuilt learner batch without
  constructing per-sampled-group resident sample objects. Proof fields now
  surface through local sampling, speed-row summaries, Modal result-bundle
  guards, and owner-search sample telemetry:
  `compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used`,
  fallback count/reason, and direct group-object count. Local focused tests
  pass. This is not H100 speed evidence yet.
- Vocabulary correction: most recent "failures" are speed failures or
  falsified lanes, not correctness failures. Correctness failure means proof
  broke or the row aborted. Speed failure means proof stayed closed but the
  whole-loop wall target was not met. Falsified lane means a mechanism was
  tested and did not have enough headroom.
- Interpretation: the active blocker is no longer "prove any H100 speedup",
  "turn on tensor-native owner sampling", "transport count alone", or "direct
  table building is too tiny to matter." Columnar append proves that removing
  `_CompactReplayRingEntry` alone is too small: the best row is still only
  about `1.25x` OPT-104, and the remaining wall lives across owner replay
  append/cache/table maintenance, owner search, learner train/sample, and
  parent wait. Fixed SoA/locality and the one-simulation deferral probe have now
  falsified the obvious replay-only next steps. The next code target must move
  a broader owner boundary: owner-resident root/search dispatch, parent-wait
  overlap, learner publication/update cadence, mechanics/observation ownership,
  or a fixed owner-buffer design that removes object/materialization surfaces
  without moving the cost into learner train/sample.
- Reorientation correction after the latest review: the next blocker is not
  "run longer to beat variance" and not another replay-only layout. The owner
  can cache previous root/search/action and can derive next reward/done from a
  complete next root, but terminal flavor and final rewards were still
  parent-only transition facts. Local patch now makes mechanics outcome
  sidecars explicit on `CompactRootBuildRequestV1` and `CompactRootBatchV1`
  (`terminated`, `truncated`, `final_reward_map`) and adds a fail-closed helper
  that derives the replay transition outcome tuple from a next root. Tests:
  full `tests/test_compact_search_replay_contract.py` passes (`47 passed`);
  focused owner direct-root/transition tests pass (`2 passed`). This became the
  support layer for the owner-local H100 decision row below.
- Owner-local transition derivation is now implemented, proof-gated, and read on
  H100. The code path is real: derived transition batches carry only record
  links, replay handles/digests, and applied-action count/checksum; the owner
  derives reward/done/terminal/final-observation facts from cached root
  sidecars. The launch history matters:
  `...owner-local-transition-derivation...r1` used the wrong checkpoint and
  failed checkpoint verification before a row; r2 timed out during local Modal
  detach and left an empty artifact directory; r3 ran remotely but failed the
  proof guard because stale generic transition-batch fields still reported the
  legacy kind; the proof normalizer was fixed and the stale-generic regression
  was added to `tests/test_compact_coach_speed_row_smoke.py` (`98 passed`).
  Valid H100 r4
  `opt132-h100-owner-local-transition-derivation-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r4-20260608`
  passed proof but is speed-rejected as the main lane: `13265.51 env/s`,
  `55.8875s` wall over `741376` env steps, `1.045x` OPT-104 but only `0.837x`
  columnar r2. Proof stayed closed: derived schema/kind through generic
  transition-batch telemetry, `724` cache hits, `0` misses, `724` checksum
  verifications, `0` mismatches, no fallback/pending/drop, parent outcome array
  transport `0/0`, direct replay and columnar append used, tensor-native replay
  used, GPU sampling disabled, and normal-death terminal contract satisfied.
  Decision: preserve owner-local derivation as support; do not repeat unchanged.
  Move to the broader fixed owner-buffer/root/search/mechanics/learner-publication
  patch.
- Local fixed-action-tape owner-buffer probes are now refreshed as architecture
  constraint evidence, not H100 speed evidence. Mechanics-only B1024/m724/w180
  passed parity but the fixed-buffer direct loop was slower than the current
  compact profile path: `0.843x`, `1.3959s` versus `1.1774s`. Rendered
  search-feedback slab/replay B128/m180/w45 also passed proof, including
  closed-loop slab replay, `224` replay appends, `224` sample-gate calls, and
  no root-observation copy, but only showed a toy `1.129x` measured mechanics
  delta while CPU-oracle render time was `218.223s`. Artifacts:
  `artifacts/local/vector_fixed_action_tape_results/owner_buffer_mechanics_b1024_m724_w180_20260608.json`
  and
  `artifacts/local/vector_fixed_action_tape_results/owner_buffer_render_slab_b128_m180_w45_20260608.json`.
  Decision: do not implement a naive fixed-buffer env loop as the speed plan.
  Use these probes only as parity/proof support while moving the actual owner
  boundary across action, step, root, replay, and learner-publication state.
- First guarded owner action-step boundary slice is now local-clean in the
  profile loop. `HybridObservationProfileConfig.compact_owner_action_step_boundary`
  is default-off and requires `compact_rollout_slab`, search-feedback action
  mode, and `materialize_scalar_timestep=False`. When enabled, the run loop owns
  the cached next joint action, applies it through `manager.step()`, verifies
  the step payload carried exactly the applied action, requires a slab step, and
  verifies the next action for the following step. Result fields now expose
  enabled/proof/step/seeded/feedback/verified/next counts, last applied and next
  action checksums, last action source, and failure reason. Local validation:
  ruff passed for `source_state_hybrid_observation_profile.py` and the profile
  tests; focused boundary tests passed (`2 passed`); neighboring compact-slab
  regression passed (`1 passed`). This is proof-only and not H100 speed
  evidence.
- The owner action-step boundary is now also bound to the real direct-root
  threaded owner path locally. New focused test
  `test_hybrid_profile_owner_action_step_boundary_uses_direct_root_build_request`
  monkeypatches the legacy parent `build_compact_root_batch_v1` symbol to raise,
  then runs the profile loop through `CompactOwnerSearchDirectStepperV1` plus
  `CompactLazyThreadedOwnerSearchSlabProxyV1` with direct root-build request,
  resident-root view, host-observation stub, and fixed action-result buffer.
  It proves the parent root builder is avoided (`0` calls), owner build is used
  for all `3` root requests, observation bytes sent are `0`, H2D/D2H are `0`,
  search payload bytes are `0`, parent replay rows are `0/0`, and the previous
  owner-selected action checksum becomes the next `manager.step()` action while
  the newest search result is cached for the following step. Validation:
  focused boundary tests `3 passed`; ruff passed. This closes the local support
  gate; it is still not H100 speed evidence.
- The first owner mechanics step-view rung is now local-clean. New default-off
  `HybridObservationProfileConfig.compact_owner_mechanics_step_boundary`
  requires owner action-step feedback, resident observations, direct
  root-build request, and resident host-observation stub. In that mode
  `HybridBatchedObservationProfileManager.step()` builds a borrowed
  `CompactOwnerMechanicsStepViewV1` after mechanics/resident observation update
  and before `_make_compact_batch`, passes it to
  `CompactOwnerSearchDirectStepperV1`, and deliberately exposes
  `observation=None` so accidental host-observation consumers fail locally.
  The direct-root boundary test now monkeypatches both legacy parent
  `build_compact_root_batch_v1` and parent `_make_compact_batch` to raise, then
  proves the owner path still passes. Telemetry reports measured-step
  `compact_owner_mechanics_step_boundary_count`, parent compact-batch builder
  call/object counts `0`, step-view object count, and host observation/final
  observation bytes `0`. Validation: expanded owner-boundary/guard packet
  `6 passed`, and ruff passed for touched modules.
  This is the first upstream boundary rung, not the final owner-owned
  step-frame ring and not H100 speed evidence.
- Owner-published dense next-action is now local-clean on the direct-root
  fixed-slot path. `CompactSearchActionStepV1` carries optional
  `dense_joint_action`; the owner derives it from validated root/search
  sidecars when an action-result slot is active, and
  `CompactOwnerSearchDirectStepperV1` validates and applies that dense action
  instead of reconstructing it from parent compact sidecars. The direct-root
  boundary test now monkeypatches `_selected_joint_action_from_compact_sidecars`
  to raise and still passes, proving owner dense-action present/used,
  parent assembly avoided, checksum/bytes/digest closed, mismatch `0`, and
  parent dense-action reconstruction count `0`. Validation: focused
  owner-boundary dense-action packet `5 passed`, direct-root owner-service
  packet `4 passed`, and ruff passed for touched modules. This removes one more
  parent action-assembly surface; it is still not H100 speed evidence.
- Pending compact-batch/mechanics-step-view and root-batch sidecar storage is
  now avoided in the direct owner stepper hot state. `_PendingCompactSearchV1`
  keeps optional `compact_batch` and `root_batch`, and
  `CompactOwnerSearchDirectStepperV1` now stores both as `None` for pending
  direct-owner searches while preserving action-step identity handles and dense
  next-action proof. New proof fields show pending compact-batch sidecar stored
  `false`, storage avoided `true`, store count `0`, avoided count positive;
  pending root-batch sidecar stored `false`, storage avoided `true`, store
  count `0`, avoided count positive; action-step identity handle stored
  `true`, identity-handle count positive; and pending root-build request stored
  `false`. Validation: focused direct-root tests `2 passed`,
  owner-step-frame packet `8 passed`, full owner-search service suite
  `50 passed`, and ruff passed. This closes two parent pending-state storage
  surfaces locally; it is not H100 speed evidence and not the whole owner step
  frame.
- Direct-root return root-batch sidecar materialization is now also removed
  locally. In `direct_root_build_request=True`, `CompactRolloutSlabStepV1`
  returns `root_batch=None`; `_slab_telemetry()` derives root counts and
  resident host-stub metadata from `CompactRootBuildRequestV1`; and
  owner-local transition derivation validates mechanics outcome sidecars
  directly from the root-build request instead of building a parent
  request-derived root sidecar. New proof fields show return root-batch sidecar
  stored `false`, storage avoided `true`, and build count `0`. A focused
  direct-root + owner-local derivation test monkeypatches both the parent root
  builder and `_root_batch_sidecar_from_build_request()` to raise and still
  stages a derived transition batch. Validation: request outcome contract
  packet `2 passed`, direct-root owner packet `3 passed`, combined owner/profile
  packet `6 passed`, full owner-search service suite `50 passed`, and ruff
  passed. This is support-only. It fixes a proof/speed mismatch in the direct
  path.
- Owner/proxy previous-transition closure is now local-clean as the next
  support rung. `CompactRootBuildRequestV1` carries validated `joint_action`
  so the proxy can verify the owner-selected previous action was actually
  applied. Default-off `owner_proxy_transition_closure=True` makes
  `CompactOwnerSearchDirectStepperV1` call
  `stage_owner_proxy_transition_from_root_build_request()` instead of
  `_stage_previous_derived_transition`; the proxy caches the previous action
  frame, validates applied actions from the current root-build request, builds
  the existing derived transition-batch schema, and stages it before owner
  request/train scheduling. Focused proof monkeypatches the parent previous
  closure, derived flush, and commit helpers to raise and still appends a
  derived transition batch through the real direct-root threaded proxy. The
  mismatch test corrupts the applied action and fails closed before replay
  append. Proof fields show parent previous-transition closure count `0`,
  proxy closure closed count `2`, batch/transition count `1/2`, applied-action
  verification `2`, mismatch `0`, fallback `0/none`, and parent applied-action
  validation count `0`. Validation: focused proxy-closure packet `4 passed`,
  full owner-search service suite `52 passed`, replay-contract suite
  `47 passed`, and ruff passed. This is still local support, not H100 speed
  evidence; H100 remains refused for this rung alone. Launch/readout plumbing
  is now fixed so this support gate can actually travel with the guarded H100
  overlap row: `--owner-search-owner-proxy-transition-closure` is exposed in
  the local speed-row CLI, local Modal launcher, remote Modal producer,
  config/result summaries, and proof/report allowlists. The proof requires the
  requested bit, proxy source `owner_proxy_cached_state_v1`, closed/transition
  counts matching transport entries, digest verification, zero parent previous
  closure and parent applied-action validation, zero mismatch/fallback/pending,
  and positive proxy action-frame/application counts.
- Owner action dispatch handles are now local-clean as the first direct attack
  on the synchronous parent wait/search dispatch boundary. The threaded
  direct-root proxy now exposes
  `submit_action_step_from_root_build_request()` and
  `resolve_action_step_handle()`, with the old
  `run_action_step_from_root_build_request()` kept as a sync compatibility
  wrapper. Local proof monkeypatches `worker.result()` to raise during submit;
  submit still returns a `CompactOwnerActionDispatchHandleV1`, leaves the fixed
  action-result slot unread, and resolves only when `resolve_action_step_handle`
  is called. The direct stepper/proxy-closure path also no longer stores the
  parent action-step identity handle in `_pending` when proxy closure owns the
  previous action frame. New proof fields project dispatch submit/resolve/
  pending/wait-in-submit counts, sync-wrapper status, proxy action-frame
  ownership, and pending action-step storage avoidance through slab/profile
  telemetry. The stricter proof now also reports cumulative sync-wrapper count
  and completed-at-submit count; rows fail closed if overlap used the sync
  wrapper, if the owner handle was already complete at submit, if submit waited
  for `worker.result()`, or if those counters are omitted. Validation:
  action-dispatch/direct-root/proxy packet `5 passed`, profile boundary packet
  `5 passed`, full owner-search service suite `55 passed`, replay-contract
  suite `47 passed`, full speed-row smoke `99 passed`, and ruff passed. This is a
  support gate, not an H100 claim by itself. The follow-up profile-loop overlap
  slice below now places real parent work between submit and resolve, but that
  later slice still needs same-work H100 evidence before any speed claim.
- Resident terminal final-observation host allocation is now elided in the
  resident-observation/no-scalar-timestep mechanics path. When sparse resident
  device final rows already own terminal observations,
  `HybridBatchedObservationProfileManager.step()` no longer allocates a dense
  host `final_observation = zeros_like(observation)` only to discard it before
  the resident compact batch/root handoff. New timing fields expose
  `resident_final_host_observation_dense_elided_count`, bytes, and row count.
  Focused resident terminal plus boundary tests passed (`4 passed`); ruff
  passed. This is a small mechanics/observation ownership support cut, not a
  standalone H100 speed lane.
- Resident-root-view proof is now local-clean for direct-root threaded owner
  mode: `opt132-local-owner-resident-root-view-threaded-directtransition-smoke-20260607-r5`
  proves resident-root-view required/proved true, direct root publish/resolve
  `15/15`, H2D/D2H `0.0/0.0`, host fallback false, action-only true, parent
  rows `0/0`, payload bytes `0/0/0`, and closed final owner maintenance. This
  is a fail-closed ownership gate, not H100 speed evidence; parent root-batch
  construction still exists.
- Resident host-observation stub proof is now local-clean for that same
  direct-root threaded owner path:
  `opt132-local-owner-resident-root-hoststub-threaded-directtransition-smoke-20260607-r1`
  proves stub requested/stubbed true, kind `zero_stride_shape_only_v1`,
  materialized bytes `0`, logical bytes `262144` last step / `3932160` total,
  resident-root-view proof preserved, H2D/D2H `0.0/0.0`, host fallback false,
  action-only true, parent rows `0/0`, search payload bytes `0`, and closed
  final owner maintenance. This is still only a local gate: it removes parent
  host-observation materialization in the resident proof path, but the parent
  root-batch builder call remains.
- Direct root build-request core hook is now local-clean, default-off, and
  parent-builder-avoidance is proven in unit and local speed-row form. The code adds
  `CompactRootBuildRequestV1`, owner-side
  `build_compact_root_batch_v1_from_request()`, direct-root store
  `publish_root_build_request()`,
  `CompactOwnerSearchSlabProxyV1.run_action_step_from_root_build_request()`,
  lazy direct-proxy routing, and
  `CompactOwnerSearchDirectStepperV1(direct_root_build_request=True)`. Focused
  tests include a monkeypatch that makes the legacy parent
  `compact_rollout_slab.build_compact_root_batch_v1` raise; both a fake request
  hook and the real `CompactLazyThreadedOwnerSearchSlabProxyV1` path still
  complete through `run_action_step_from_root_build_request`. Full
  owner-search service suite passes (`45 passed`). The speed-row
  CLI/report guard is now wired and locally closed as proof-only: the local smoke/Modal/report
  path threads `--owner-search-direct-root-build-request`, passes
  `direct_root_build_request=True` into the direct stepper, projects the nested
  build-request fields, and fails closed on parent builder calls, owner-build
  count mismatch, parent root-batch objects sent, observation bytes in the
  request, missing resident handle, or lost resident-root/stub/action-only
  gates. Full speed-row smoke module passes (`87 passed`). Local smoke
  `opt132-local-owner-rootbuildrequest-threaded-directtransition-smoke-20260607-r3`
  passed with schema/kind `curvyzero_compact_root_build_request/v1` /
  `resident_root_view_build_request_v1`, publish/resolve/owner-build
  `15/15/15`, parent builder used/calls `false/0`, parent build sec `0.0`,
  request observation bytes `0`, resident handle/proof true, H2D/D2H
  `0.0/0.0`, stub materialized bytes `0`, parent rows `0/0`, search payload
  bytes `0`, direct replay batches/transitions/transport `3/12/3`, action
  mismatches `0`, pending maintenance/policy lag `0/0`, and owner pid/bytes
  reported. H100
  `opt132-h100-rootbuildrequest-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607`
  kept that proof closed at scale but was speed-rejected: `11327.75 env/s`,
  `65.4477s`, `0.893x` OPT-104 and `0.715x` columnar r2. The proof-clean row
  removed parent root build (`0.0s`, parent builder calls `0`) but regressed
  the whole loop: parent wait `26.151s`, replay append `22.783s`, learner
  train `15.526s`, worker search `16.422s`, observation `13.196s`, and slab
  `29.850s`. Decision: preserve root-build request as a support gate; do not
  repeat unchanged.
- Local support patch after this read: columnar append now exposes internal
  telemetry through `_CompactReplayRingV1.columnar_append_telemetry_snapshot()`
  and speed-row proof fields: columnar record count, entry-view/step-view object
  counts, and append/cache timing splits. This is observability/support only;
  it does not change the hot loop by itself.

## 2026-06-08 Step-Frame Reorientation

- The current blocker is now sharper: not variance, not proof failure, and not
  "run longer." We are parent-step-frame limited. The parent still owns the
  outer loop, `manager.step()`, compact scalar/root sidecars,
  `_make_compact_batch`, direct-stepper root request/identity/action assembly,
  and synchronous owner wait.
- Three bounded sidecars converged on the same model. The mature pattern we are
  still failing to copy is a single owner-owned step frame: actors/mechanics
  write fixed slots; search/replay/sample/learner consume slot/generation/
  digest handles; Python coordinates epochs and proof.
- Replay/sample work remains useful but cannot be the whole answer. The fastest
  columnar r2 replay/sample ceiling projection is `24903.25 env/s`
  (`1.9625x` OPT-104), still about `0.558s` short of `2x` for the same
  `741376` steps.
- Local support patch landed: `compact_owner_minimal_step_payload_snapshot`
  is default-off and requires `compact_owner_action_step_boundary`. It keeps
  only summary/action payload keys in the guarded owner-boundary step snapshot
  and records measured-step counts, full payload bytes/key count elided, and
  retained key count. The direct-root/threaded boundary proof still passes
  with parent root builder `0`, resident-root/stub proof, search payload bytes
  `0`, parent rows `0/0`, and clean action checksums. Validation:
  focused source-state tests `4 passed`; ruff passed.
- This patch is support only. The next real code target is an owner-resident
  mechanics/root/search step-frame boundary before `_make_compact_batch`, not
  another H100 row for minimal payload, root-build-request, fixed-slot,
  owner-local derivation, or replay-only layout.
  counts, prepare/register/append-store/retain/evict/candidate/cache-refresh/
  cache-rebuild/total timings. Focused ruff and pytest pass. This is not speed
  evidence; it is the measurement surface for the fixed-SoA cut.
- Fixed SoA/direct-gather replay is now implemented locally behind
  `--owner-search-fixed-soa-replay` / `compact_replay_fixed_soa_unroll2_buffer_requested`.
  `_CompactReplayRingV1` owns a default-off fixed SoA unroll-2 buffer with
  `append_fixed_soa_columnar_records()`, and learner-batch-only unroll-2
  sampling can emit `fixed_soa_direct_gather_v1` from `fixed_soa_columns_v1`
  without maintained table concat or learner-ready/table-entry objects. The
  owner-search direct transition-batch sidecar, local speed-row proof, Modal
  launcher/producer, and tensor-native validation now thread the fixed-SoA flag
  and fail closed on nonzero fixed-SoA object/table counters. Local parity test
  compares every learner tensor against maintained replay for a mixed live +
  partial-terminal replay case, and focused ruff/pytest passed before the H100
  fixed-SoA rows below.
- Fixed SoA H100 evidence now exists. The row-level successor/terminal
  correctness bugs were real and are fixed: r5 cut append-time successor
  indexing from r4's `191.93s` to `10.76s`, r6/r7/r8 stayed proof-clean, and
  r8 is the best exact fixed-SoA row so far:
  `opt132-h100-fixed-soa-rowlevel-selectedgroups...r8` ran `11616.45 env/s`,
  `63.82s` measured wall, sample aggregate `43.14s`, last gather `0.2247s`,
  terminal samples `4`, selected records about `373` for a `512` row sample,
  record count `2155`, and table rows `1460766`. This is a correctness win and
  a large recovery from the broken fixed-SoA rows, but it still loses to
  OPT-104 `12689.38 env/s`. r9 removed contiguity copies and regressed to
  `8491.45 env/s`, so that copy-trim was reverted.
- Interpretation: this is not a "longer run" or variance blocker. The exact
  fixed-SoA path still scatters a 512-row learner sample across hundreds of
  record groups, so the builder pays many small per-record observation/unroll
  gathers. Fixed SoA removed table/object materialization, but did not yet give
  the learner a flat/global row layout.
- Local support patch after the fixed-SoA H100 read: an explicitly labeled
  `compact_replay_fixed_soa_locality_sample_group_size` probe can sample
  fixed-SoA rows in group-local chunks. Values greater than `1` mark
  `fixed_soa_locality_sample_semantic_drift=true`; they are causal speed probes
  for row locality, not exact replay-sampling promotion rows. Focused
  pycompile, ruff, and pytest pass.
- H100 locality result is now in. The first launch failed because the Modal app
  had not threaded `--owner-search-fixed-soa-locality-sample-group-size`; that
  plumbing was fixed. The valid r2 row
  `opt132-h100-fixed-soa-locality-g8-rowlevel-selectedgroups-b1024a1-normal-unroll2-m724-w180-r2-20260607`
  passed proof but slowed to `10428.59 env/s`, `71.0907s` wall. The causal
  read is useful: locality collapsed selected records from r8's about `373` to
  `62` for a real `512` row learner batch, and owner train sample fell
  `43.14s -> 32.03s`, so fragmentation is real. But whole-loop speed worsened
  because append/search/parent-wait/learner-update surfaces still dominate:
  replay append `17.24s`, worker search `16.68s`, parent wait `21.62s`,
  learner update `6.52s`, owner train wall `40.30s`. Decision: do not keep the
  main lane on fixed-SoA gather/locality tweaks alone. Any exact flat/global
  row layout must be justified as part of a broader owner-buffer layout, not as
  the next hoped-for 2x by itself.
- H100 root-build-request evidence reaches the same conclusion for root
  ownership alone: proof-clean boundary movement is not automatically speed.
  The next code target must move a larger graph or hot-data surface across
  owner search dispatch/parent wait, mechanics/observation ownership, learner
  publication/update, or a fixed owner-buffer design that prevents cost from
  reappearing in replay append, search, learner train, or parent wait.
- Local reporting cleanup after the r2 read: top-level result/evidence/report
  projections now carry fixed-SoA requested/used/object counters, record count,
  selected-record count, table-row count, and fixed-SoA locality fields. This
  is not speed code; it prevents future fixed-SoA rows from requiring nested
  learner-telemetry archaeology. Validation passed: pycompile, ruff, and the
  focused fixed-SoA tensor-native smoke/evidence tests (`3 passed`).
- Local ceiling/proof instrument now exists for the whole-owner-buffer replay
  question. `compact_whole_owner_buffer_replay_ceiling_*` fields are computed
  from already-measured owner-search replay append, owner train sample, parent
  wait, worker search, learner update, wall, and env-step counts. They are
  explicitly projection-only: `production_speed_claim=false`,
  `touches_live_training=false`, `speed_currency=local_projection_no_speed`,
  `h100_validation_status=not_run`, `promotion_eligible=false`, and evidence
  validation rejects claim/currency drift. Local report and Modal result-bundle
  surfaces preserve the fields. Validation passed: pycompile, ruff, and focused
  pytest (`4 passed`). Manual read using known H100 timings says the fastest
  columnar r2 stack would project to about `24903 env/s` (`1.96x` OPT-104) if
  replay append plus owner-train sample vanished under the conservative
  parent-wait-bounded model, still about `0.56s` short of the 2x target for the
  `741376` env-step window. Fixed-SoA locality r2 projects only about
  `14986 env/s` because non-replay surfaces dominate.
- The manual ceiling read is now reproducible as an artifact, not archaeology.
  `scripts/compare_compact_coach_speed_rows.py` derives
  `whole_owner_buffer_replay_ceiling` for old H100 rows from existing timing
  fields and writes a `whole_owner_buffer_replay_ceiling_rank`. Review artifact:
  `artifacts/local/curvytron_compact_coach_speed_row_results/opt132-whole-owner-buffer-ceiling-review-20260607/comparison.json`.
  Rank: columnar r2 projects `24903.25 env/s` (`1.9625x`, still `0.558s`
  short of 2x), direct-table r3 projects `24138.94 env/s` (`1.9023x`, `1.500s`
  short), fixed-SoA locality r2 projects `14986.22 env/s` (`1.1810x`, `20.258s`
  short). The fastest row already has action-only owner result transport
  enabled (`search_result_payload_bytes=0`, visit/root bytes `0`, parent
  reconstruction false, owner materializes replay true, replay handle present,
  inner two-phase/deferred replay true), so the next non-replay surface is
  owner-resident root/mechanics/search dispatch and learner-publication overlap,
  not another action-only or replay-layout-only H100 run.
- Local action-critical search proof is now closed through the owner loop for
  the default-off one-simulation deferral lane.
  `CompactTorchCompileConfig.defer_one_simulation_replay_payload` only affects
  `run_action_step()` for `num_simulations=1` with zero root noise: action
  selection uses the initial policy logits during the parent-wait phase, while
  recurrent replay-payload materialization moves to `flush_*_replay_payload()`.
  The owner service now preflushes cached deferred replay handles before model
  refresh, aggregates owner-inner flush proof, and fails closed on missing
  materialized-on-flush proof, identity drift, refresh crossing, nonzero
  owner-final pending handles, or replay-payload D2H bytes. Local/Modal speed
  row launchers carry `--compact-torch-defer-one-simulation-replay-payload`.
  Local flagged smoke
  `opt132-local-owner-deferred-one-sim-direct-transition-batch-smoke-20260607-r7`
  passed with `ok=true`: direct transition count `12`, direct
  flush/materialized/identity/recurrent `12/12/12/12`, direct
  model-refresh-crossed `0`, replay-payload D2H `0.0`, owner-inner
  flush/materialized/identity/recurrent `16/16/16/16`, owner-inner crossed
  `0`, and owner-inner pending final `0`.
- Interpretation: the r7 lane was a local correctness gate and permitted at
  most one same-work H100 probe of the default-off deferral path. That probe has
  now run and is speed-rejected. Replay/sample removal alone still projects
  just under `2x`, so the next primary implementation has to attack
  owner-resident root/search/parent-wait/mechanics/observation or a broader
  fixed owner-buffer boundary.
- H100 one-simulation deferral result is now in:
  `opt132-h100-owner-deferred-one-sim-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607`
  passed proof but failed the speed test. It ran `13691.98 env/s`, `54.1467s`
  wall, with direct transition count `724`, columnar append used, direct
  flush/materialized/identity/recurrent `724/724/724/724`,
  model-refresh-crossed `0`, replay-payload D2H `0.0`, owner-inner final
  pending `0`, and normal-death proof closed. This is only `1.079x` OPT-104 and
  slower than columnar r2's `15852.67 env/s`. Direct append grew to `22.377s`;
  deferred flush/device flush were `3.613s/4.381s`. Decision: do not repeat this
  deferral lane unchanged. It proves the mechanics can be made correct, but not
  that moving one-simulation recurrent replay materialization to owner flush is
  a speed path.
- H100 in-process async learner overlap result is now in:
  `opt132-h100-owner-asynclearner-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607`
  passed proof but failed speed. It ran `12954.74 env/s`, `57.2282s` wall, with
  async worker enabled, submit/completed/pending `90/90/0`, max pending `2`,
  actions while async learner pending `510`, async wait `0.851s`, failed
  `false`, direct transitions `724`, columnar append used, action-only result
  true, and normal-death proof closed. This is only `1.021x` OPT-104 and much
  slower than columnar r2. Decision: do not repeat in-process async learner
  overlap unchanged. The next target is owner-root/root-view ownership or
  another deeper fixed-buffer handoff, not threading the existing learner call.
- Fixed action-result slot speed-row plumbing is now locally closed. The local
  proof row
  `opt-fixed-action-result-slot-local-smoke-proof5-20260608` passed with
  `ok=true`: fixed-slot requested/used `true/true`, slot count `4`,
  acquire/write/read `5/5/5`, pending `0`, wire/full result bytes `413/712`,
  action-only `true`, direct-root `true`, owner `owner_search_worker`, owner
  train/expected train `1/1`, and tensor-native replay `true`. The tiny CPU
  speed (`548.91 env/s`) is not speed evidence. The important proof lesson is
  that this direct-root fixed-slot path must require deferred owner maintenance
  and owner-owned unroll-2/tensor-native learner-batch proof.
- The fixed-slot H100 decision row has now run:
  `opt132-h100-fixed-action-result-slot-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608`
  passed proof but is rejected as the main speed lane. It ran `12794.42 env/s`,
  `57.9453s` wall over `741376` env steps, barely over OPT-104 (`1.008x`) and
  far below columnar r2 (`0.807x`). The mechanism was real: fixed-slot
  requested/used `true/true`, acquire/write/read `904/904/904`, pending `0`,
  wire/full result bytes `414/4837`, parent root builder calls `0`, parent root
  objects `0`, action-only true, normal-death gate true, accepted/tensor-native
  violations `[]`. It improved the root-build-request row (`11327.75 ->
  12794.42 env/s`, parent wait `26.151s -> 19.035s`) but did not beat the
  fastest support stack. Decision: preserve fixed-slot as support. Owner-local
  transition derivation has now also been speed-rejected as a standalone lane, so
  move to the broader fixed owner-buffer/root/search/mechanics/learner-publication
  patch.

## Historical Context Before r29-r34

- First owner-search H100 win exists, but it did not repeat: r14
  `opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-warmupgate-b1024a1-normal-unroll2-r14-20260605`
  ran at `13497.30 env steps/sec`, `13.6561s` wall versus OPT-104
  `12689.38 env steps/sec`, `14.5255s` wall.
- Exact r15 repeat of the same path failed speed: `10502.70 env steps/sec`,
  `17.5498s` wall. Work identity matched exactly and proof stayed closed, but
  `stable_speed_claim_allowed=false`.
- This is a real first win, not a mock/local/profile-only result. It used the
  compact Torch owner-search inline proxy, warmup replay/training gate,
  shared-model-state refresh, and no inline host payload clone.
- It is not promotion-grade stable speedup. OPT-104 remains the accepted
  baseline. The active blocker is now repeat instability on the r14 mechanism,
  not missing proof and not a new model-refresh implementation.
- r14/r15 comparison: exact identity, wall spread `24.95%` of median. Largest
  moving buckets were compact rollout slab `+2.128s`, actor step `+1.047s`,
  observation `+0.681s`, actor autoreset `+0.514s`, and env runtime `+0.488s`.
  Owner train also moved (`2.864s -> 4.075s`), but the comparator points to a
  broad actor/observation/slab runtime swing rather than a count/proof drift.
- Longer r16 (`724/180`) did not rescue the speed claim. It ran
  `6491.80 env steps/sec`, `114.202s` wall with proof still clean. It exposed
  steady-state owner train sample cost: `52.807s` sample versus only `3.236s`
  learner update across `90` train requests. Digest/state_dict/host clone stayed
  `0/0/0`.
- r17 tried to combine the r14 shared/no-clone owner-search route with the
  fused/learner-ready/tensor-native/specialized learner-batch flags. It failed
  before row creation with
  `compact_owned_loop_fused_learner_batch requires compact_owned_loop_entrypoint`.
  Current concrete blocker: the fast learner-batch path is not wired into the
  owner-search inline route.
- The stale pre-r14 blocker was "move learner resource/work shape." That is
  now superseded as the P0 path. Cadence-only refresh and same-process async
  learner work were falsified; the winning mechanism is cheap same-object
  model refresh plus removal of inline host payload clone.
- The compact-owned loop exists end to end: actor steps, search, replay/sample,
  learner updates, and policy refresh all run in H100 speed rows.
- Game mechanics still run on CPU.
- Search and learner run on GPU.
- H100 utilization has been low, so bigger GPUs, multi-GPU, and GPU mechanics
  are not the next speed move. H200/B200 can be used only as explicit memory
  headroom diagnostics, not as replacements for the H100 acceptance target.
- The accepted benchmark route is the direct compact trainer path. The active
  work is architecture feasibility around env/search/replay/sample/learner
  ownership.
- Owner-search process/inline/threaded variants are parked beyond the measured
  action-only threaded/background falsifier.
- Resident observation stack shift is not the next target; the new timer shows
  it is only about `0.012s` to `0.015s`.
- Integration plus repeatability diagnosis is now P0 for the r14 path. The next
  useful implementation is to make owner-search train sampling use the fast
  learner-ready/tensor-native batch path or a smaller equivalent owner-side
  path, then rerun the longer row. Measurement stability remains the promotion
  gate after the first H100 win failed repeat. The accepted 180/45 row
  window produced
  fast and slow same-work rows, and the `724/180` diagnostic also swung hard.
  After the latest-frame resident replay snapshot patch, bounded `1084/270`
  now fits on H100, but exact rows are not the exploration tool.
- Maintained tensor-native replay table state now has local real-loop proof.
  `_CompactReplayRingV1` maintains per-record learner tables when the
  default-off tensor-native replay flag is requested, and the real sample-gate
  path gathers from `maintained_record_table_v1`. The 32x8/128-row CPU proof
  passed with reused records `32`, missing records `0`, current ring median
  `0.004377667s`, real tensor-native median `0.000867458s`, and local speedup
  `5.047x`. Focused missing-table fallback, eviction/rebuild, local
  smoke/projection, parent learner-ready cache dependency, standalone remote
  validation, Modal producer/launcher, and evidence tests pass. OPT-132BD then
  passed H100 proof but was slow, and OPT-132BF cut sample gate to `4.116s`
  while full-loop speed remained only `5362.68 env steps/sec`. Replay/sample is
  closed as the current P0 surface; owner-search deferred-maintenance
  action-only replay ownership is the selected candidate and has passed local
  real-entrypoint proof. The hardened local smoke now closes the observable
  cadence/sample/final-drain/train-timing/action-only-handle/action-feedback
  gates plus explicit maintenance work-item vs replay-entry drain accounting.
  Timing split and threaded/background proof now show the remaining owner
  learner-update/final-drain tail was dominant. Nested learner telemetry then
  exposed first-call LightZero import/setup overhead outside compact MuZero
  learner timing, and import prewarm drops local train wall/update/final drain
  from about `0.865s/0.855s/0.790s` to `0.072s/0.057s/0.043s`. Cadence-8
  drops train requests to `1` and final drain to `0.019s`, confirming the tail
  is now cadence-sensitive real learner work. Compact Torch owner-search inner
  two-phase device replay is now wired and proved. Post-prewarm normal-death
  scale rows preserve the proof but fail speed shape: slab/threaded local rows
  run about `25 env steps/sec`, with owner train/learner update about
  `14-15s` and final drain about `12-13s`. Mock-fast normal-death ceiling then
  preserves proof and runs `214.06 env steps/sec` locally with final drain
  `0.0026s`. Local MPS placement then preserved proof but slowed the whole row
  to `20.81 env steps/sec`; eager append pre-drain preserved proof and fired
  `7` bounded drains, but the normal-death scale row slowed to
  `24.09 env steps/sec` with learner update `15.12s` and final drain
  `12.67s`. The default-off in-process async learner worker then proved
  actions can continue while learner futures are pending, but the clean scale
  row still ran only `24.47 env steps/sec` with async submit/completed/pending
  `6/6/0`, action-while-async-pending `47`, async wait `14.17s`, learner
  update `14.88s`, and final drain `12.82s`. Raising same-process async max
  pending to `6` only reached `25.37 env steps/sec`, still below the CPU scale
  row. Decision: scheduler polish, placement alone, and same-process async
  overlap/queue depth are exhausted as speed paths. This pre-r14 conclusion is
  now refined: the winning H100 mechanism was not another async/resource split,
  but eliminating local shared-model state transport and inline host payload
  cloning while preserving owner-search/search/replay mechanics.
- The `724/180` A/A packet has now been run five times. All rows matched exact
  work identity, but wall still ranged `89.353s -> 137.297s`, so measurement
  stability is not solved.
- Historical: OPT-132-U preserved the inner failed producer result/report and
  proved the pre-latest-frame bounded `1084/270` path was a real CUDA OOM:
  PyTorch had about `78.34 GiB` allocated and only about `32 MiB` was free.
  That memory-fit blocker is now superseded for `1084/270`.
- Replay-store retained resident-snapshot counters now exist.
- Bounded diagnostics now use latest-frame resident replay snapshots instead of
  cloning the full 4-frame resident observation stack every step. Accepted
  `180/45` rows still use full-stack snapshots. This made `1084/270` fit on
  H100, but it is memory-stability work, not speed evidence.
- Earlier H100 diagnostic set: three `1084/270` rows with learner-batch
  sub-timers passed and match exact identity, accepted-fast-path violations
  `[]`, latest-frame snapshot mode, and retained resident snapshot bytes
  `9346744320`. R3 broke timing stability: wall spread `15.55%`, sample gate
  spread `19.49%`, learner-batch build spread `18.88%`.
- Diagnostic-only CUDA sync timing is now plumbed through the local smoke
  script, Modal launcher/producer, hybrid profile config, sample gate,
  learner-batch builder, learner gate, summary/report/bundle artifacts,
  comparison artifacts, and focused local tests. OPT-132Y r1/r2/r3 completed
  with exact identity and sync diagnostic violations `[]`, but r3 failed timing
  stability hard. This is diagnostic evidence, not speed evidence.
- Runtime-step envelope stats are now also plumbed for CUDA-sync diagnostic
  rows. They summarize measured-step wall count/sum/min/max/p50/p95 and the
  slowest step's actor, observation, slab, sample, learner, policy-refresh, and
  residual context. OPT-132Z used this to show the instability lives inside
  measured loop iterations rather than final drain/untracked wall.
- OPT-132Z r1/r2 used those runtime-step fields. Identity matched exactly and
  proof violations stayed `[]`, but timing failed again: wall spread `32.50%`,
  runtime-step sum spread `32.50%`, sample gate spread `42.67%`, and
  learner-batch build spread `44.14%`. Runtime-step p50 barely moved while
  p95/max jumped, and the slowest step was sample-gate dominated. The slowdown
  is inside measured loop iterations and concentrated on sample-gate cadence
  steps.
- A diagnostic-only per-call sample-gate child distribution patch is now local:
  sample-gate total, candidate, RNG, and residual per-call stats are emitted as
  nested dicts and flattened count/sum/min/max/p50/p95 fields for reports and
  comparisons. This is not a speed patch.
- OPT-132AA r1/r2 read those sample-gate per-call fields on H100. Identity
  matched exactly against each other and OPT-132Z, proof violations stayed
  `[]`, and the AA pair was diagnostic-only: wall `213.543s -> 224.408s`
  (`4.96%` spread), sample gate `128.039s -> 134.063s` (`4.60%`), and
  learner-batch build `76.980s -> 79.225s` (`2.88%`). Sample-gate per-call
  p50/p95/max was broadly high in both rows (`1.032/1.630/1.723s` and
  `1.061/1.674/1.783s`); learner-batch-build per-call p50/p95/max was
  `0.652/0.804/1.094s` and `0.666/0.802/1.100s`. This points at broad
  builder-centered sample-gate cost rather than a few pathological calls.
- Builder-child per-call attribution is now wired locally for learner-batch
  builder group-loop, terminal-metadata, unroll-fields, write-output,
  order-restore, finalize-outputs, metadata-sync, metadata-build, and
  builder-cuda-sync timers. Summary/compact payloads, local and remote Modal
  reports, and comparison artifacts expose nested stats and flattened
  count/sum/min/max/p50/p95 fields.
- OPT-132AB r1/r2 read those builder-child fields on H100. Identity matched
  exactly and proof violations stayed `[]`, but timing failed hard again:
  wall `271.062s -> 170.468s` (`45.57%` spread), sample gate
  `180.198s -> 102.544s` (`54.93%`), learner-batch build
  `105.658s -> 59.377s` (`56.09%`), and builder group-loop
  `104.758s -> 58.756s` (`56.27%`). Unroll fields were the largest visible
  child (`50.816s -> 29.511s`), followed by terminal metadata
  (`23.072s -> 12.377s`). This is diagnostic-only and does not permit a speed
  claim.
- OPT-132AC deep builder group-loop attribution is now wired locally and has
  H100 evidence. It splits terminal metadata into mask, tensor fallback,
  validation, and final-observation timers; it splits unroll work into
  terminal-window hint, identity, stack fields, mask build, terminal value, mask
  apply, and action stack timers. Source profile, speed-row summaries/compact
  payloads, local/remote Modal reports, and comparator fields project totals
  plus per-call stats. Focused ruff and pytest passed. The first two H100 launch
  attempts failed before row creation with Modal `RESOURCE_EXHAUSTED`, then
  r1c/r2/r3 completed with exact identity and violations `[]`. Timing still
  failed: wall `235.855s / 174.009s / 172.247s` (`36.55%`), sample gate
  `147.161s / 106.443s / 109.114s` (`37.32%`), learner-batch build
  `87.449s / 64.220s / 67.871s` (`34.23%`), builder group-loop
  `86.514s / 63.583s / 67.224s` (`34.11%`), unroll fields
  `45.615s / 31.928s / 34.076s` (`40.16%`), and terminal metadata
  `18.970s / 14.537s / 16.130s` (`27.48%`). R2/R3 clustered; r1c was the
  slow outlier. No speed claim.
- The launcher now preserves structured artifacts for future pre-FunctionCall
  Modal launch failures, and the Modal entrypoint prints structured
  `spawn_failed` payloads for reachable remote spawn failures. Local and remote
  per-call prefix parity is covered by a focused test. This is measurement
  hygiene only.
- OPT-132AD r1/r2/r3 is now historical context, not the latest read. Exact
  identity held and accepted-fast-path/sync diagnostic violations were `[]`,
  but timing still failed: wall `204.520s / 231.874s / 182.574s` (`24.11%`),
  sample gate `122.980s / 145.023s / 107.539s` (`30.48%`), learner-batch
  build `72.702s / 88.877s / 63.929s` (`34.32%`), sample-gate builder
  group-loop `71.909s / 88.019s / 63.218s` (`34.49%`), and unroll fields
  `36.310s / 44.992s / 31.871s` (`36.13%`). No speed claim.
- OPT-132AE is the prior H100 diagnostic read. It
  adds builder group-loop accounted/residual totals and per-call stats, where
  accounted is terminal metadata + terminal-window hint + unroll fields +
  write output. Projection covers source profile, speed-row summary/compact
  payload, local/remote Modal reports, and the comparator. Validation passed:
  ruff, focused pytest `4 passed`, and broader speed-row/report slice
  `63 passed`. AE r1/r2 exact H100 identity held with violations `[]`; wall was
  `271.920s -> 284.561s` (`4.54%`), group-loop
  `113.971s -> 119.853s`, accounted child work `95.872s -> 101.300s`, and
  residual `18.099s -> 18.553s`. Both rows are slow; no speed claim.
- Historical OPT-132AF was a local speed patch, but it is no longer the active
  target or latest diagnostic read. It adds a default-off specialized
  unroll-2 builder only for fused resident grouped learner batches, guarded by
  unroll count, exact chain length, device replay rows, and terminal-window
  metadata. It preserves generic-builder learner-batch tensors in local parity
  tests and emits requested/eligible/used/call/fallback/reason/impl/path proof fields
  through the source profile, speed-row summaries, Modal reports, shared
  reducer, and comparator. Validation passed locally. The hardened local proof
  smoke `opt132af-local-unroll2-specialized-builder-hardened-smoke-20260603`
  completed `ok=true` with requested=true, used=true, eligible/call_count `13`,
  fallback_count `0`, fallback_reason `none`, impl `unroll2_specialized_v1`,
  and path `unroll2_specialized`. H100 r1/r2/r3 also completed cleanly:
  accepted-fast-path, CUDA-sync, and unroll2 violations were `[]`; exact
  identity held; specialized eligible/call_count was `399` in all rows. Wall
  was `211.683s -> 225.020s -> 234.042s` (`9.94%`), so AF is rejected as a
  stable speed claim under exact `1084/270` repeatability.
- OPT-132AG is now the prior AF-path H100 diagnostic read. It reran the same AF path with
  GPU utilization sampling enabled at `0.5s`. R4/R5/R6 kept exact identity,
  accepted-fast-path/CUDA-sync/unroll2 violations `[]`, and the same specialized
  proof counts. The monotonic AF slowdown did not repeat: wall was
  `195.624s / 185.643s / 199.275s`, but timing still failed the bar with wall
  spread `6.97%`, sample gate `11.57%`, and builder group-loop `15.51%`. GPU
  sampler fields worked (`412` to `526` samples, mean util `15.54%` to
  `18.69%`, max power `227.38W` to `260.27W`) and do not support a simple
  thermal/power-throttle explanation. The combined AF+AG six-row packet spans
  `185.643s -> 234.042s` wall (`23.55%` of median), so measurement stability
  remains P0.
- OPT-132AH is now the prior sync-probed generic H100 diagnostic read. It
  reran exact `1084/270` with GPU utilization sampling but without the unroll-2
  specialization flag, so the builder path was generic. R1/R2/R3 kept exact
  identity and accepted-fast-path/
  CUDA-sync violations `[]`; generic proof fields showed requested=false,
  used=false, eligible/call_count `0`, fallback_count `0`, impl=`none`, and
  path=`generic`. Timing failed harder than AG: wall
  `205.584s / 243.805s / 289.172s` (`34.28%`), sample gate
  `123.444s / 151.007s / 196.761s` (`48.55%`), learner-batch build
  `75.790s / 92.226s / 123.781s` (`52.04%`), and builder group-loop
  `74.642s / 91.378s / 122.843s` (`52.75%`). Mean GPU utilization decreased
  as rows slowed (`17.97% / 15.25% / 13.96%`), max memory used stayed
  `38867 MiB`, and max power did not form a simple throttle story. This
  separates the remaining instability from the AF specialization.
- OPT-132AI is now local diagnostic tooling. It adds
  `--compact-profile-runtime-step-timing-diagnostics`, decoupling measured-step
  envelope stats from `--compact-profile-cuda-sync-timing-diagnostics`. The
  CUDA-sync flag still implies runtime-step stats for existing rows. Local and
  Modal launch/collection paths now thread the new flag and fail closed if
  requested runtime-step stats are missing. Validation passed: ruff, focused
  source-profile runtime-only test, Modal launcher/validator/report tests, and
  comparator tests.
- OPT-132AJ is now the prior GPU-sampled runtime-step-only generic H100 read. It reran exact generic
  `1084/270` with GPU utilization sampling and
  `--compact-profile-runtime-step-timing-diagnostics`, but without CUDA-sync
  probes. R1/R2/R3 kept exact identity, accepted-fast-path/runtime-step
  violations `[]`, CUDA-sync diagnostics false, and generic builder proof.
  Wall was `219.055s / 221.938s / 198.950s` (`10.49%`), and runtime-step sum
  matched it (`10.49%`). Sample gate still spread `12.72%`, learner-batch
  build `14.09%`, and builder group-loop `14.25%`; observation also moved
  broadly. Removing CUDA-sync probes narrowed the AH generic packet but did not
  make exact work stable.
- OPT-132AK is local diagnostic tooling. It adds same-work comparator
  slowest-vs-fastest wall-swing attribution and runtime-step cadence summaries:
  sample-gate active/inactive measured-step buckets, early/mid/late measured
  thirds, and a bounded top-slowest-step list. Focused ruff and pytest passed.
  The refreshed AJ attribution comparison shows r2 was slowest and r3 fastest;
  runtime-step sum explains essentially all wall delta, sample gate about
  `61.00%`, learner-batch build `35.87%`, and builder group-loop `35.23%`.
- OPT-132AM is prior local diagnostic tooling. It extends the AK cadence
  surface with chronological active sample-gate distributions for each measured
  third, adds sample-gate residual bucket sums, includes sample-gate residual
  and builder-sync fields in top-slowest runtime-step records, and projects
  slowest per-call iteration/measured-iteration through local and Modal
  reports. OPT-132AN used these fields to distinguish broad active-call
  slowdown from a few late spikes. Local only; no speed claim.
- OPT-132AL is the prior H100 diagnostic read. It reran exact generic
  `1084/270` with runtime-step cadence diagnostics and CUDA-sync probes off,
  but disabled GPU sampling. R1/R2/R3 kept exact identity,
  accepted-fast-path/runtime-step/CUDA-sync violations `[]`, generic builder
  proof, and no GPU sampler. Timing still failed and slowed monotonically:
  wall `168.661s / 178.629s / 207.419s` (`21.70%`), runtime-step sum
  `21.70%`, sample gate `97.710s / 102.538s / 121.130s` (`22.84%`),
  learner-batch build `20.92%`, and builder group-loop
  `53.566s / 55.349s / 65.163s`. Cadence attribution points at the late
  measured third: late-step sum `73.166s -> 96.470s`, late sample gate
  `48.364s -> 64.010s`, and late builder group-loop `23.702s -> 30.956s`.
  Disabling the sampler did not solve stability; no speed claim.
- OPT-132AN is the prior active-cadence H100 diagnostic read. It reran exact generic
  `1084/270` with the AM active-call fields, CUDA-sync probes off, and GPU
  sampling disabled. R1/R2 kept exact identity and violations `[]`, but timing
  failed again: wall `206.746s / 166.095s` (`21.806%`), runtime-step sum
  matched wall, sample gate `125.169s -> 93.384s`, learner-batch build
  `68.851s -> 49.643s`, and builder group-loop `68.101s -> 48.923s`.
  Active sample-gate p50 rose early/mid/late in both rows and r1 was elevated
  across all thirds versus r2. Diagnostic only; no speed claim.
- OPT-132AO is prior local diagnostic tooling. It adds bounded
  per-sample-gate call trace records with call index, measured iteration,
  measured third, sample seed/rows/checksums, stored/eligible/excluded/evicted
  replay counts, replay capacity, and per-call sample/builder/residual timing.
  Source profile, speed-row summary/compact payload, local/remote Modal
  reports, and focused tests preserve the trace. Local only; no speed claim.
- OPT-132AO is prior H100 diagnostic evidence. R1/R2 used those trace
  fields on exact generic `1084/270` rows with CUDA-sync probes off and GPU
  sampling disabled. Identity and violations stayed clean, but timing failed:
  wall `221.118s / 254.261s` (`13.944%`), sample gate
  `132.338s -> 163.614s`, learner-batch build `72.215s -> 91.981s`, and
  builder group-loop `71.350s -> 91.030s`. The trace state matched at all
  `135` call indices, including sample seeds/checksums, stored/eligible/
  excluded/evicted replay counts, and replay capacity. Diagnostic only; no
  speed claim.
- OPT-132AP is local diagnostic tooling and prior H100
  diagnostic read. It extends the AO per-sample-gate trace with allocator/
  runtime state and deeper builder child timings: CUDA memory allocated/
  reserved/peak counters, allocator retry/OOM counters, learner-batch-build
  boundary deltas, Python GC generation counts, process max-RSS raw state,
  sample-gate CUDA-sync timing, and builder children for terminal metadata,
  unroll, order-restore, finalize, metadata-sync, and metadata-build work.
  Focused ruff and `17` focused tests passed locally. AP r1/r2/r3 then ran
  exact generic no-sampler `1084/270` H100 rows with identity exact and
  violations `[]`, but timing still failed: wall
  `219.446s / 205.371s / 153.752s` (`31.99%` of median), sample gate
  `130.199s / 120.714s / 86.639s`, learner-batch build
  `70.732s / 65.983s / 46.881s`, and builder group-loop
  `69.882s / 65.174s / 46.350s`. Trace identity and CUDA allocator/memory
  counters matched exactly across all `135` calls; Python GC counts and RSS
  varied. No speed claim.
- OPT-132AQ is the prior GC-stats H100 diagnostic read. It adds actual
  `gc.get_stats()` collection/collected/uncollectable counters to the AP
  runtime trace. AQ r1/r2 exact generic no-sampler `1084/270` rows kept
  identity exact and violations `[]`, but wall still failed stability:
  `218.320s / 163.265s` (`28.856%`), sample gate `131.651s -> 95.558s`,
  learner-batch build `72.267s -> 51.322s`, and builder group-loop
  `71.432s -> 50.716s`. GC collection totals were nearly identical
  (`7232/659/28` versus `7234/655/28` for gen0/gen1/gen2), and `100` calls
  with identical collection deltas still carried `22.6695s` of sample-gate
  delta. Actual Python GC collections do not explain the swing. No speed claim.
- OPT-132AR is prior H100 diagnostic evidence. It adds process and
  current-thread CPU-time before/after/delta fields for the full sample gate
  and learner-batch-build slice. Focused ruff passed and the AP/AQ validation
  slice passed (`17` tests, `2` warnings). AR r1/r2 exact generic no-sampler
  `1084/270` rows kept identity exact and violations `[]`, but wall still
  failed stability: `189.915s / 217.364s` (`13.479%`), sample gate
  `111.765s -> 130.614s`, learner-batch build `60.527s -> 71.857s`, and
  builder group-loop `59.815s -> 71.015s`. The sample-gate wall delta
  `18.848s` was backed by `17.33s / 17.18s` process/thread CPU-time deltas;
  learner-build wall delta `11.331s` was backed by `10.44s / 10.39s`
  process/thread CPU-time deltas. The remaining swing is CPU-time-backed, not
  mostly off-CPU waiting. No speed claim.
- OPT-132AS is prior H100 diagnostic evidence. It adds
  `resource.getrusage` before/after/delta fields for the full sample gate and
  learner-batch-build slice: process/thread user CPU, system CPU, minor/major
  page faults, and voluntary/involuntary context switches. Focused ruff passed
  and the AP/AQ/AR validation slice passed (`17` tests, `2` warnings). AS
  r1/r2 exact generic no-sampler `1084/270` rows kept identity exact and
  violations `[]`, but wall still failed stability: `211.257s / 182.483s`
  (`14.616%`), sample gate `126.934s -> 107.934s`, learner-batch build
  `70.377s -> 58.933s`, and builder group-loop `69.602s -> 58.206s`. The
  sample-gate r1-minus-r2 resource delta was process user/system CPU
  `16.17s / 0.49s`, thread user/system `15.83s / 0.58s`; learner-build process
  user/system was `9.31s / 0.55s`. Page faults and context-switch deltas were
  zero. No speed claim; the remaining swing is mostly user CPU.
- OPT-132AT is prior local diagnostic tooling. It adds
  `--compact-profile-cpu-perf-stat-diagnostics`, wrapping the remote producer
  in `perf stat -x,`, capturing perf stdout/stderr artifacts, parsing
  task-clock/cycles/ref-cycles/instructions/branch/cache/LLC/dTLB/page-fault/
  context-switch/CPU-migration counters into
  `compact_profile_cpu_perf_stat_*` fields, and projecting them through row,
  Modal report, and comparison artifacts. Validation passed: ruff, targeted
  perf-stat tests (`4` tests, `2` warnings), and broader smoke/compare slice
  (`20` tests, `2` warnings). Local only; no speed claim.
- OPT-132AT's first H100 attempt spawned `fc-01KT7YGKDM2Y35NCG6C204TYMY` and
  failed before a speed row because `perf` was not found in the remote image.
  The structured report has `compact_profile_cpu_perf_stat_available=false`
  and return code `127`. OPT-132AU is the narrow retry: install `linux-perf`
  only in `speed_row_image` and rerun once.
- OPT-132AU is the latest local diagnostic tooling. It adds `linux-perf` only
  to the compact speed-row Modal image, leaving the shared LightZero image
  untouched. Ruff passed and two focused Modal perf tests passed (`2` warnings).
- OPT-132AU H100 spawned `fc-01KT7YWEDCC60CMGTFY6388QBE`; `/usr/bin/perf` was
  available, but `perf stat` failed before the producer ran because
  `sys_perf_event_open()` returned `19 (No such device)` for `task-clock`.
  Return code `255`, parsed event count `0`, no speed row. External perf
  counters are unavailable in this Modal container.
- OPT-132AV is a prior H100 diagnostic read. It adds in-process
  process/thread CPU-time deltas for learner-batch builder child phases after
  the external perf route failed. Local validation passed: ruff, focused
  profile/smoke/comparator tests, and broader smoke/compare pytest (`78`
  tests, `2` warnings). AV r1/r2 exact generic no-sampler `1084/270` rows kept
  identity exact and report violations `[]`, but wall still failed stability:
  `190.101s / 220.991s` (`15.03%`), sample gate `114.928s -> 135.240s`,
  learner-batch build `66.229s -> 82.661s`, and builder group-loop
  `65.505s -> 81.792s`. Group-loop process CPU moved `62.48s -> 77.36s`
  (`+14.88s`), about `91%` of the group-loop wall delta. Child process CPU
  split: accounted `+10.60s`, residual `+4.28s`, terminal metadata `+6.37s`
  with terminal final-observation `+2.67s`, and unroll fields `+2.76s`. No
  speed claim; AW is the follow-up split below those buckets.
- OPT-132AW is a prior completed H100 diagnostic read. It splits AV's target
  further with group-loop prepare and terminal-value bookkeeping wall/CPU
  fields, terminal final-observation presence/select-current/gather wall/CPU
  fields, and branch/storage proof counters for terminal final-observation
  work. Local validation passed: ruff and focused source-profile/smoke/
  comparator pytest (`189` tests, `2` warnings). R1/R2 exact generic
  no-sampler `1084/270` kept exact identity and report violations `[]`, but
  wall still failed the repeat bar: `250.183s / 232.521s` (`7.318%` spread).
  Sample gate moved `157.724s -> 146.441s`, learner-batch build
  `94.924s -> 88.854s`, and builder group-loop `94.002s -> 87.987s`.
  Group-loop process CPU was `89.13s / 83.17s`; accounted/residual CPU was
  `74.68s / 69.10s` and `14.45s / 14.07s`. Terminal metadata CPU was
  `24.01s / 21.75s`, terminal final-observation CPU `14.78s / 13.35s`,
  final-gather CPU `6.84s / 5.95s`, and unroll-fields CPU
  `38.18s / 34.98s`. Final-observation branch/storage proof was identical:
  groups `399`, index fast path `0`, fallback `399`, final-row sum/max
  `512/4`, sparse storage `399`, and sparse-row sum/max `3102/16`. No speed
  claim.
- OPT-132AX is a prior completed H100 diagnostic read. It replaces the
  grouped learner fallback terminal-final-observation materialization with a
  validate-only resident final-observation coverage helper. The materializing
  helper remains in sample paths that write `next_observation`; AX changes only
  the grouped learner path where the returned tensor was discarded. New proof:
  `terminal_final_observation_validate_only_count` and
  `terminal_final_observation_materialized_count`; new timing:
  `terminal_metadata_final_observation_validate` wall/per-call/process/thread
  CPU fields. The comparator treats terminal-final-observation proof counters
  as optional identity fields. Local validation passed: ruff and focused
  source-profile/smoke/comparator pytest (`190` tests, `2` warnings). AX r1/r2
  exact generic no-sampler `1084/270` kept exact identity, no missing optional
  proof fields, and report violations `[]`, but timing failed badly: wall
  `168.280s / 262.830s` (`43.864%`), sample gate
  `98.894s / 173.517s`, learner-batch build `56.376s / 102.112s`, and builder
  group-loop `55.699s / 101.183s`. Terminal-final-observation proof was exact:
  groups/fallback/validate-only `399/399/399`, materialized `0`, final-row
  sum/max `512/4`, sparse storage `399`, sparse-row sum/max `3102/16`.
  Select-current and gather wall/CPU were `0`; final validate CPU was
  `3.33s / 5.76s`. No speed claim.
- OPT-132AY is a prior completed H100 diagnostic read. It adds
  process/thread CPU-time deltas to the existing unroll sub-timers inside
  `unroll_fields`: identity, stack fields, mask build, terminal value, mask
  apply, and action stack. These subfields are nested attribution and are not
  added to group-loop accounted CPU. Local validation passed: ruff; focused
  source-profile/smoke/comparator pytest (`190` tests, `2` warnings); and a
  tiny local fused unroll-2 smoke emitted nonzero nested CPU fields. AY r1/r2
  exact generic no-sampler `1084/270` kept exact identity and stable speed
  claim false, but wall still failed: `237.752s / 183.386s` (`25.819%`
  spread). Sample gate was `148.271s / 110.841s`, learner-batch build
  `91.035s / 65.376s`, and builder group-loop `89.834s / 64.694s`.
  Group-loop process CPU was `85.30s / 62.04s`; accounted/residual CPU was
  `76.22s / 55.78s` and `9.08s / 6.26s`; terminal metadata CPU was
  `17.77s / 12.02s`; unroll-fields CPU was `43.86s / 34.14s`. The unroll
  sub-CPU split was identity `0.39s / 0.23s`, stack fields `7.76s / 6.29s`,
  mask build `10.08s / 8.47s`, terminal value `8.51s / 6.55s`, mask apply
  `3.25s / 2.29s`, and action stack `1.97s / 1.31s`. No speed claim.
- OPT-132AZ is now the latest completed H100 diagnostic read. It adds nested
  prepare, terminal-metadata, final-observation, and unroll residual/accounted
  CPU attribution plus unroll-row-index and unroll-builder-selection timers.
  Local validation passed: ruff; focused source-profile/smoke/comparator
  pytest (`190` tests, `2` warnings); and a tiny local fused unroll-2 smoke
  emitted nonzero AZ fields. AZ r1/r2 exact generic no-sampler `1084/270` kept
  exact identity and stable speed claim false, but wall still failed:
  `217.391s / 180.892s` (`18.329%`). Sample gate was
  `136.622s / 109.260s`, learner-batch build `85.001s / 65.049s`, and builder
  group-loop `84.177s / 64.278s`. Group-loop process CPU was
  `80.37s / 61.70s`; unroll-fields CPU `42.13s / 32.24s`;
  unroll accounted/residual CPU `30.29s / 24.21s` and `11.84s / 8.03s`;
  terminal metadata CPU `15.58s / 11.44s`; prepare CPU
  `10.44s / 8.33s`; terminal metadata residual `2.54s / 1.90s`.
  No speed claim. The sidecar step-back audits agree that pure attribution has
  diminishing returns. Next action is no longer another timer or a direct H100
  rerun; the selected owner-search deferred-maintenance action-only replay
  candidate has been implemented/proved locally, real-learner scale failed,
  and mock-fast scale passed. The active implementation target is real learner
  update placement/overlap. Learner-ready replay already removed the
  sample-gate surface; fixed-buffer env/mechanics stays a bounded fallback lane
  unless the owner-search learner-placement path is falsified.

Accepted baseline:

```text
OPT-104
12689.38 env steps/sec
14.5255s wall
H100, B1024/A1, normal death
180 measured, 45 warmup
sample interval 8, batch 512, replay capacity 4096
learner unroll 2, refresh interval 4
```

Latest accepted-fast-path preset rows:

```text
OPT-132-F
12177.36 env steps/sec
15.136s wall
direct_core: true
fused learner batch: true
borrowed render state: true
render-state copy steps: 0
render-state borrowed steps: 225
lean trainer step: true
terminal sample/target rows: 167/167
normal death: true
truncations: 0
host fallback: 0
direct autoreset count: 0
decision: rejected for speed

OPT-132-G
13649.29 env steps/sec
13.504s wall
accepted proof fields passed
decision: useful fast row, not accepted as new baseline because OPT-132-H lost

OPT-132-H
11366.84 env steps/sec
16.216s wall
accepted proof fields passed
decision: failed repeat confirmation

OPT-132-I
13308.59 env steps/sec
13.8497s wall
accepted proof fields pass under the corrected signed-checksum guard
decision: useful fast row, still not enough to close because OPT-132-H was slow

OPT-132-J
10057.98 env steps/sec
18.3258s wall
accepted proof fields passed with corrected guard
decision: clean slow repeat; same trajectory/sample-order checksums as I
```

Best recent recovery:

```text
OPT-132-B
11609.78 env steps/sec
15.876s wall
same fast-path flags
decision: useful recovery, still too slow
```

Useful controls:

```text
OPT-109: 12527.10 env steps/sec, 14.714s wall
OPT-117: 12566.17 env steps/sec, 14.668s wall
OPT-118 r1: 12795.45 env steps/sec, 14.405s wall
```

## Main Diagnosis

We were not stuck because the loop was impossible. We were stuck because the
comparison process was weak.

Recent slow rows dropped known fast-path flags, then we chased the resulting
slowdown as if it were a new architecture problem. Restoring the full bundle
recovered a lot of time, but did not fully reproduce the old fast rows.

The accepted-fast-path launcher preset now sets the full bundle. It also checks
the actual remote result and fails if the returned row does not match the
preset.

## Current Bottleneck

The active bottleneck is real owner learner-update placement/cost inside
owner-search maintenance.

History: OPT-132AZ was the last useful H100 builder diagnostic. It showed exact
same-work identity but unstable broad user-CPU learner-batch builder work. That
lane pushed the architecture reset toward fixed-buffer ownership, then
OPT-132BD/BF/BK closed replay/sample/slab as the current P0 surface.

Current truth: owner-search action-only replay ownership is selected and locally
proved. Real normal-death scale preserves the proof but collapses to about
`25 env steps/sec` locally because train/update/final-drain spend about
`14-15s/14-15s/12-13s`. The mock-fast normal-death ceiling preserves the same
owner-search proof and runs `214.06 env steps/sec` locally with final drain
`0.0026s`. Therefore owner-search/search/replay mechanics are not the local
scale blocker once neural update is removed.

Local placement probe: explicit MPS support now exists, but the MPS
normal-death scale row slowed to `20.81 env steps/sec`. It cut learner update
to `8.08s` but introduced enough sample/build, replay append, digest, and drain
cost that placement alone is falsified locally.

Next move: change the real learner resource/work shape while preserving
normal-death, zero-parent-row, zero-payload, action-feedback, owner-drain,
cadence, submitted/owner/refreshed update closure, positive in-row policy-lag
proof when overlap is claimed, and final zero lag. Same-process async overlap
has already failed as a speed path, so the next change must use a genuinely
different resource/owner boundary, a faster learner update path, or an honest
learner-work-shape change. No H100 row is justified until local real-learner
scale shape is plausible. Fixed-buffer env/mechanics stays parked unless
owner-search learner resource/work-shape work is falsified.

```text
AC r1c wall/speed: 235.855s / 4706.36 env steps/sec
AC r2 wall/speed: 174.009s / 6379.07 env steps/sec
AC r3 wall/speed: 172.247s / 6444.33 env steps/sec
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
wall spread: 63.608s, 36.55% of median
sample gate spread: 40.718s, 37.32% of median
learner-batch build spread: 23.229s, 34.23% of median
builder group-loop spread: 22.931s, 34.11% of median
unroll-fields spread: 13.686s, 40.16% of median
terminal-metadata spread: 4.433s, 27.48% of median
comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ac-h100-builder-deep-child-diagnostic-r1c-r2-r3-comparison-20260603/comparison.json
```

The prior local field set is OPT-132AE and it has now been read on exact H100
`1084/270` r1/r2 repeats. The new fields showed the builder group-loop is
mostly named child work, not residual: unroll fields are the largest absolute
child, terminal metadata is next, and residual is secondary. OPT-132AF then
targeted that unroll builder path under exact learner-batch identity/proof
gates and was rejected as a stable speed claim. OPT-132AG showed the AF
slowdown was not monotonic under GPU sampling, and OPT-132AH showed the generic
builder can drift too. Superseded by AZ, BD/BF, and the architecture reset:
learner-ready resident replay / unroll-2 cache is closed for the current P0.
That pre-r14 target of real learner update placement/overlap is now superseded,
but r15 failed the exact repeat. The active target is r14/r15 repeat-instability
diagnosis after shared-model refresh plus no inline host payload clone produced
one fast H100 row and one slow exact row.

The prior CUDA-sync packet is still useful context:

```text
sync r1 wall/speed: 196.699s / 5643.23 env steps/sec
sync r2 wall/speed: 194.540s / 5705.85 env steps/sec
sync r3 wall/speed: 255.499s / 4344.51 env steps/sec
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
wall spread: 60.959s, 30.99% of median
sample gate spread: 36.790s, 30.47% of median
learner-batch build spread: 19.588s, 28.01% of median
actor wall spread: 11.223s, 31.66% of median
observation spread: 8.268s, 62.14% of median
observation-other spread: 8.016s, 75.95% of median
builder cuda_sync spread: 2.754s, 39.85% of median
sync counts: sample 12/12/12, builder 11977/11977/11977, learner 28/28/28
comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132y-h100-cuda-sync-timing-diagnostic-r1r2r3-comparison-20260602/comparison.json
```

Use OPT-132-G and OPT-132-H as the accepted-window context pair.

The same accepted preset swung hard:

```text
wall:        G 13.504s vs H 16.216s (+2.712s)
actor:       G 4.073s  vs H 4.924s  (+0.851s)
observation: G 1.984s  vs H 2.495s  (+0.512s)
sample:      G 2.692s  vs H 3.426s  (+0.734s)
learner:     G 1.466s  vs H 1.778s  (+0.312s)
slab/search: G 2.769s  vs H 2.892s  (+0.124s)
```

Resident stack timers now show:

```text
G resident stack update: 0.029s
G resident stack shift:  0.012s
H resident stack update: 0.034s
H resident stack shift:  0.015s
```

Do not build the ring-buffer resident stack now. It would optimize a tiny
measured child while actor/env, observation-other, sample, and learner are
moving by much larger amounts.

Current tooling/fix status:

- OPT-132-E: report/result/evidence expose actor/observation child timers.
- OPT-132-F: resident terminal final observations store sparse terminal rows
  instead of dense full-stack clones. This improved wall by about `0.833s`
  versus OPT-132-E.
- OPT-132-G/H: report/result/evidence expose resident observation stack timing.
  OPT-132-G beat OPT-104, but OPT-132-H did not confirm the win.
- Long-window diagnostic support: the accepted fast-path launcher now supports
  `--compact-owned-accepted-fast-path-step-window stability_724_180` and
  `stability_1084_270`. `stability_1444_360` also exists. Earlier `1084/270`
  rows OOMed, but latest-frame resident replay snapshots now make `1084/270`
  fit on H100. The comparator labels these as diagnostics and will not let them
  replace OPT-104 without a matched long-window baseline.
- OPT-132-K/L/Q/R/S long-window packet: exact same work, same `724/180`
  diagnostic window, no accepted-fast-path violations, but wall ranged
  `89.353s -> 137.297s`. Sample gate ranged `44.170s -> 77.233s`;
  learner-batch build ranged `25.023s -> 43.698s`.
- Bounded long-window diagnostic mode: the slab now counts committed replay
  rows without retaining the full committed-row history, diagnostic rows can omit
  the nested source profile payload, and latest-frame resident replay snapshots
  make `1084/270` fit on H100.
- Learner-batch timing diagnostic: three `1084/270` diagnostics with sub-timers
  completed. Learner-batch build ranged `46.919s -> 57.573s` across 135 calls;
  the largest internals were group loop, unroll fields, and terminal metadata.
  OPT-132Y added CUDA-sync timing. R1/R2/R3 kept exact sync counts but builder
  cuda_sync wait moved `5.520s -> 8.274s`, which is visible but too small to
  explain the `60.959s` wall range by itself.
- Runtime-step envelope diagnostic: local code now records measured-step wall
  stats when CUDA-sync diagnostics are requested and projects
  `compact_profile_runtime_step_*` through local/Modal summaries and the
  comparator. This is completed historical tooling; the active next row is a
  real learner placement/update row, not another runtime-envelope diagnostic.
- OPT-132Z runtime-envelope r1/r2: exact identity held, but r2 slowed hard.
  Wall moved `197.292s -> 273.865s`; runtime-step sum moved
  `197.288s -> 273.860s`; sample gate moved `118.072s -> 182.127s`;
  learner-batch build moved `68.491s -> 107.283s`. Runtime p50 was
  `0.063s -> 0.073s`, p95 `1.150s -> 1.784s`, max `1.786s -> 2.717s`.
  The follow-on per-call diagnostics were completed and superseded by the
  owner-search learner-tail blocker.
- Sample-gate per-call distribution patch: source profile now emits
  `compact_rollout_slab_sample_gate_per_call_stats` plus candidate/RNG/residual
  siblings; speed-row summaries/reports/comparator expose flattened
  count/sum/min/max/p50/p95 fields. Focused ruff and pytest passed.
- OPT-132AA sample-gate per-call r1/r2: exact identity held and violations
  stayed `[]`. The AA pair showed sample gate `128.039s -> 134.063s`,
  learner-batch build `76.980s -> 79.225s`, sample-gate per-call p50/p95/max
  `1.032/1.630/1.723s -> 1.061/1.674/1.783s`, and learner-batch-build
  per-call p50/p95/max `0.652/0.804/1.094s -> 0.666/0.802/1.100s`.
  Comparison: `artifacts/local/curvytron_compact_coach_speed_row_results/opt132aa-h100-sample-gate-per-call-diagnostic-r1r2-comparison-20260603/comparison.json`.
- Builder-child per-call attribution patch: source profile now emits
  `compact_rollout_slab_sample_gate_learner_batch_builder_*_per_call_stats`
  for group-loop, terminal-metadata, unroll-fields, write-output,
  order-restore, finalize-outputs, metadata-sync, metadata-build, and
  builder-cuda-sync child timers. Reports flatten count/sum/min/max/p50/p95 and
  the comparator includes sum/min/max/p50/p95 for timing ranges. Focused ruff
  and pytest passed.
- OPT-132AB builder-child per-call r1/r2: exact identity held and violations
  stayed `[]`, but stable speed claim stayed false. Wall spread was `45.57%`,
  sample gate `54.93%`, learner-batch build `56.09%`, and builder group-loop
  `56.27%`. The visible child swing is dominated by unroll fields, then
  terminal metadata; builder CUDA sync scales with the build but is smaller.
  Comparison: `artifacts/local/curvytron_compact_coach_speed_row_results/opt132ab-h100-builder-child-per-call-diagnostic-r1r2-comparison-20260603/comparison.json`.

Historical AX follow-up was completed and superseded by AY/AZ and then BD/BF.
Current concrete target: remove the next owner-search object/materialization
surface after ring-batched replay append/cache refresh repeated positive on
H100. Try early tensor-native prebuilt-batch sampling before per-group
sample-object construction; if flat, move to true fixed SoA/ring-buffer owner
storage. The proof has truthful parent-commit metadata, zero parent
committed/stored rows, owner replay/train/update/drain counts, and final owner
drain in measured wall.

The comparator now carries per-row GPU utilization context plus numeric
utilization/memory/power timing fields. The refreshed AF r1/r2/r3 comparison
kept exact identity and `stable_speed_claim_allowed=false`; GPU sampling was
disabled in those rows, so the hardware block is zero/null. OPT-132AG then
proved those fields on H100 and showed the slow row did not coincide with high
power or high mean GPU utilization. OPT-132AH then proved the generic-builder
control also drifts. OPT-132AI adds the local flag needed to run runtime-step
diagnostics without CUDA-sync probes, and OPT-132AJ shows that removing sync
probes narrows but does not solve the instability. OPT-132AK adds wall-swing
attribution and runtime-step cadence fields. OPT-132AL disables GPU sampling
and still fails stability; OPT-132AN then uses the AM active-call fields and
shows the slowdown is broad across active sample-gate calls, not only
spike-driven. OPT-132AO adds and reads the per-call replay-state trace; the
state is identical across the AO r1/r2 pair, so replay/sample-shape drift at
that trace level is not the cause. OPT-132AP reads below that exposed state and
finds CUDA allocator/memory state exact across r1/r2/r3 while Python GC/RSS
fields vary. OPT-132AQ shows actual Python GC collections do not explain the
swing; OPT-132AR shows the remaining swing is CPU-time-backed rather than
mostly off-CPU wait; OPT-132AS shows the resource split is mostly user CPU,
not system CPU, page faults, or context switches; OPT-132AV splits the moving
builder CPU into terminal metadata/final-observation, unroll fields, and
residual group-loop buckets; OPT-132AW splits final-observation and residual
one level deeper, reads those fields on H100, and proves the
final-observation branch/storage mix is identical across repeats.

```text
compact_profile_cuda_sync_timing_diagnostics
compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics
compact_rollout_slab_sample_gate_cuda_sync_timing_enabled
compact_rollout_slab_sample_gate_cuda_sync_count
compact_rollout_slab_sample_gate_cuda_sync_sec
compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics
compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled
compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count
compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec
compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics
compact_rollout_slab_learner_gate_cuda_sync_timing_enabled
compact_rollout_slab_learner_gate_cuda_sync_count
compact_rollout_slab_learner_gate_cuda_sync_sec
compact_muzero_learner_batch_unroll2_specialized_builder
compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested
compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count
compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used
compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count
compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count
compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason
compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl
compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path
```

Do not claim speed from OPT-132AF. Its exact H100 proof repeated, but the
comparison script kept `stable_speed_claim_allowed=false`.

Latest guard correction:

- Repeatability checksums can be signed. The accepted-fast-path validator now
  requires checksum fields to be present and nonzero, while counters must be
  positive.
- The OPT-132-I artifact was wrongly rejected by the previous positive-checksum
  rule. With the corrected rule it has no accepted-preset violations.
- OPT-132-J passed the corrected guard and matched OPT-132-I trajectory and
  sample-order checksums, but was slow. That points to runtime instability on
  the same work rather than hidden data drift.

Local artifact comparison so far:

- G and H matched the accepted proof fields.
- G and H matched row seed, sample seed, learner seed, sample calls, learner
  calls, learner updates, replay entries, stored/committed index rows, death
  rows, terminal rows, and terminal sample rows in nested result data.
- I and J matched trajectory and sample-order checksum fields while wall time
  swung by about `4.476s`.
- Top-level reports did not expose enough of those repeatability fields, so the
  current code now threads them into summary and Modal reports for the next row.
- Hash-bound evidence now also binds seed/work-shape fields, actor trajectory
  checksums, terminal/autoreset/death checksums, sample-order checksums, and
  sample/learner counters.
- Latest local owner-dispatch update: the direct-root owner stepper now has
  `submit_step()` / `resolve_step()`, and the hybrid profile loop can put real
  parent post-slab work between them with
  `compact_owner_action_dispatch_step_overlap`. The local proof verifies
  submit/resolve counts, pending `0`, max pending `1`, no sync wrapper,
  cumulative slab/owner sync-wrapper counts `0`, completed-at-submit count `0`,
  submit wait-in-submit count `0`, submit-no-wait, and resolve after parent
  payload/snapshot work. The speed-row producer, local Modal launcher, and
  remote Modal producer now carry `--compact-owner-action-dispatch-step-overlap`
  and `--owner-search-owner-proxy-transition-closure`; the guard fails if the
  proof is missing, default-projected instead of raw-reported, sync-wrapper
  based, already complete at submit, or waited in submit. Validation passed:
  full owner-search service `55 passed`, replay-contract `47 passed`, full
  speed-row smoke `99 passed`, focused source-state overlap/direct-root tests
  `2 passed`, and ruff. This was local proof before the H100 read below.
- The guarded H100 overlap/proxy row has now been read and is speed-rejected
  unchanged, not correctness-rejected. r1 failed before a row because warmup
  owner learning disabled proxy transition closure; that was fixed. r2 produced
  a coherent payload but failed the wrapper on stale primary transition-batch
  counters; that was fixed. Valid r3
  `opt132-h100-action-dispatch-overlap-proxyclosure-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r3-20260608`
  completed `ok=true` at `15541.95 env/s`, `47.7016s` wall over `741376` env
  steps, `1.225x` OPT-104 but only `0.980x` the current best columnar/direct-
  table support row. Proof stayed closed: overlap submit/resolve `904/904`,
  sync-wrapper count `0`, completed-at-submit `0`, wait-in-submit `0`, max
  pending `1`; proxy closure used with closed/transition `724/724`,
  batch/transport `181/181`, fallback `0/none`, pending `0`, digest verified,
  parent previous-transition closure `0`, and action mismatches `0`. Decision:
  preserve overlap/proxy as proof support; do not repeat unchanged. The P0 code
  path is now fixed owner mechanics/root-frame handles or learner publication
  tickets, starting with the owner-resident step-frame boundary.
- First owner mechanics step-frame handle rung is now implemented locally. The
  mechanics step view now publishes a fixed ring slot/generation/digest handle
  (`curvyzero_compact_owner_mechanics_step_frame_handle/v1`, ring slots `4`);
  the direct owner stepper verifies the digest before building/submitting the
  root request and returns consumed/digest-verified proof through slab telemetry,
  source-profile result fields, local speed-row projection, Modal allowlists,
  and compact speed-row evidence summaries. This is still support/local proof,
  not H100 speed evidence: the view object still exists, but the boundary now
  has a stable handle identity that future patches can make the actual
  transport. Validation so far: ruff passed across touched code/scripts/tests,
  focused source-state handle packet `2 passed`, source-state owner-boundary
  packet `7 passed`, owner-search direct-root/proxy packet `11 passed`,
  speed-row focused smoke `5 passed`, full source-state suite `123 passed`,
  full owner-search service suite `56 passed`, and full speed-row smoke
  `99 passed`.

## Active Work

OPT-132 remains active:

```text
build from ring-batched owner replay append/cache refresh
preserve columnar/direct-table as the fastest single support stack
preserve owner-local derivation, root-build-request, resident-root/stub, and
fixed-slot as support gates, not standalone speed lanes
preserve compact_owner_action_step_boundary as a proof guard; it is locally
validated and speed-row projected, but not H100 speed evidence
preserve compact_owner_action_dispatch_step_overlap as proof support; the
guarded H100 row was proof-clean but slower than columnar r2, so do not repeat
unchanged
implement a broader fixed owner-buffer/root/search/mechanics/learner-publication
patch that moves parent wait, search/root dispatch, mechanics/observation, or
learner publication/update across a real ownership boundary
current local rung: owner mechanics step-frame handle ring is implemented and
proof-projected; next rung should replace richer parent view/object handoff with
the handle-owned fixed frame data or pair it with learner publication tickets
before any remote row claiming the boundary, require direct-root/action-only
binding plus zero parent rows/search payload/root-builder fallback/host fallback,
clean action feedback, clean owner-local derivation counters, real terminal
proof, and if overlap is requested: submit/resolve counts equal measured
iterations, pending 0, max pending positive, parent-work overlap positive,
submit wait-in-submit 0, and sync-wrapper false
keep accepted-fast-path flags locked
make one measured change at a time
preserve fail-closed proof fields
do not run same-work H100 unless owner hot data movement changes or a repeat
control is explicitly needed
```

## 2026-06-08 Slot-Backed Mechanics Frame Update

The handle-identity rung has been extended into a real fixed-slot local proof.
`HybridBatchedObservationProfileManager.step()` now publishes a
`curvyzero_compact_owner_mechanics_step_frame_slot/v1` entry in a four-slot
ring and passes that slot-backed frame to the direct owner stepper. The direct
root path builds `CompactRootBuildRequestV1` from the slot arrays instead of
calling the parent `compact_root_build_request_v1_from_batch()` helper.

What is proved locally:

- parent `_make_compact_owner_mechanics_step_view`, `_make_compact_batch`,
  `compact_root_build_request_v1_from_batch`, parent root batch builder, and
  parent dense-action reconstruction can be monkeypatched to raise while the
  owner-boundary profile path still passes;
- slot/generation/digest handle proof is preserved, consumed, and verified;
- legacy step-view object count is `0`, parent step-frame build count is `0`,
  root request from-batch helper used is `false`, and root sidecar array bytes
  and field count are `0`;
- speed-row, Modal report, and compact speed-row evidence projection now carry
  the slot/root-request proof fields, and the direct-root proof guard fails
  closed on missing/dirty mechanics-slot fields.

Validation:

```text
ruff touched files: passed
source-state owner-boundary packet: 5 passed
owner-search direct-root/proxy packet: 11 passed
speed-row smoke owner/direct-root/projection packet: 7 passed
```

This is still not H100 speed evidence. It is the first local patch in the
correct ownership direction: fixed owner frame slots and small handles instead
of rich parent view/root request construction. Next speed-relevant work must
either move more mechanics/root/search state behind this slot owner, or move
learner publication/update state behind owner tickets/refs. Do not launch H100
for this slot-rung alone.

## 2026-06-08 Slot Generation Guard Update

The slot-backed frame proof now fails closed on stale or reused mechanics-frame
identity before replay/search side effects. `CompactOwnerMechanicsStepFrameSlotV1`
carries a live `slot_generation` view, `_owner_mechanics_step_frame_handle_metadata_v1()`
checks metadata slot/generation, handle batch/player shape, generation modulo
slot, live slot generation, and digest, and
`CompactOwnerSearchDirectStepperV1` tracks consumed generations per stepper.

Important ordering fix: direct-step root metadata and owner-frame validation now
happen before `_commit_previous_transition()` can stage replay, before proxy
closure can run, and before direct-root action search can submit. A regression
test plants a pending sentinel, mutates the ring generation, and requires the
stale frame to fail with the pending object, record index, replay append count,
proxy closure count, submit count, and pending dispatch unchanged.

Validation:

```text
ruff touched files: passed
focused owner-frame/direct-root packet: 3 passed
source-state owner-boundary packet: 6 passed
owner-search direct-root/proxy/overlap packet: 11 passed
```

This is still local proof, not speed evidence. It makes the slot-frame rung
honest enough to be used as support in the next broader owner transaction.

## 2026-06-08 Root Action Context Update

The direct owner dispatch path no longer keeps the full `CompactRootBuildRequestV1`
in pending parent/proxy state. A new `CompactRootActionContextV1` strips the
root request down to action-critical fixed arrays: active root ids, env rows,
players, policy env ids, active legal masks, shape, stub, and scalar metadata.
`CompactOwnerSearchDirectStepperV1` and `CompactOwnerSearchSlabProxyV1` now
retain that context across submit/resolve and build/validate action steps from
it, not from the rich root request.

Why this matters: the prior proof field
`compact_owner_search_pending_root_build_request_stored=false` was too
optimistic for async dispatch because the pending object still held the full
request. This patch makes the field true in substance: pending dispatches have
no `root_build_request` slot, while public metadata reports
`compact_owner_search_pending_root_action_context_stored` or
`compact_owner_search_action_dispatch_pending_root_action_context_stored`.

Validation:

```text
ruff touched files: passed
owner-search direct-root/action-dispatch packet: 13 passed
source-state owner-boundary packet: 6 passed
full owner-search service suite: 56 passed
full source-state hybrid observation profile suite: 124 passed
```

This is still a local support rung, not the final owner root/search transaction
API and not H100 speed evidence. The next larger patch should start the
transaction from the step-frame slot/handle inside the owner/proxy so parent
code does not build the root request at all.

## 2026-06-08 Owner Root/Search Transaction Update

The direct owner dispatch path now has the first real slot-started root/search
transaction. `CompactOwnerSearchDirectStepperV1.submit_step()` uses
`submit_owner_root_search_transaction_from_step_frame_slot()` when the mechanics
step-frame ring is present and the proxy supports it, so the slab no longer
calls `_root_build_request_from_owner_step_frame_slot_v1()` in that path.
`CompactOwnerSearchSlabProxyV1` builds the `CompactRootBuildRequestV1` from the
slot/handle, optionally closes the proxy transition from that owner-side request,
submits the existing owner action dispatch, and returns only the action dispatch
handle plus `CompactRootActionContextV1`.

What is now fail-closed locally:

```text
parent step-frame root-request builder can be monkeypatched to raise
parent root request from-batch helper can be monkeypatched to raise
parent compact-batch builder and legacy step-view builder can be monkeypatched to raise
direct pending dispatch stores no CompactRootBuildRequestV1
transaction pending dispatch stores no compact_batch in the new transaction path
transaction parent root-request build count is 0
transaction parent compact-batch stored is false
owner/proxy root-request build and root-store publish counts are positive
slot generation/digest and action identity are verified
```

Validation so far:

```text
ruff touched code/scripts/tests: passed
focused transaction/source boundary packet: 4 passed
focused owner-search direct-root/action-dispatch packet: 13 passed
full owner-search service suite: 56 passed
full source-state hybrid observation profile suite: 125 passed
focused speed-row proof/report packet: 2 passed
```

This is still local support, not H100 speed evidence. It removes parent
root-request construction for the slot path, but the parent still keeps
`CompactRootActionContextV1` for action validation and still coordinates the
per-step submit/resolve. The next owner-graph patch must either make the
root/action context itself owner-resident with a smaller action handle, or move
a disjoint measured surface such as learner-publication/update tickets/refs.

## 2026-06-08 Owner Root-Action Context Handle Update

The transaction path now keeps the action/root validation context behind an
owner/proxy handle instead of storing `CompactRootActionContextV1` in parent
pending transaction state. `CompactOwnerRootActionContextHandleV1` carries the
schema, context id, transaction id, dispatch id, root counts, active-root
counts, and digest. The proxy stores the context in an owner table, resolves it
by handle, verifies the digest, and releases it. The direct stepper's
transaction path now fails closed if the handle is missing, malformed, or
missing digest/root-count proof.

What is now fail-closed locally:

```text
slot-started transaction cannot fall back to parent root_action_context
direct pending transaction has root_action_context is None
proxy pending action dispatch stores a handle, not the context
owner store/resolve/release counts balance
owner pending context count returns to 0
context digest is verified
parent action-context validation count is 0
owner action-context validation count is positive/count-matched
transaction parent root-action-context stored false
transaction parent root-action-context store/array/field counts 0
speed-row and Modal proof guards require the handle fields
final idle proxy metadata no longer erases per-step handle proof
```

Validation:

```text
ruff touched code/scripts/tests: passed
source transaction/profile packet: 3 passed
source owner-boundary packet: 5 passed
owner-search direct-root/action-dispatch packet: 2 passed
full owner-search service suite: 56 passed
speed-row owner-search/report/proof packet: 3 passed
```

This is still local support, not H100 speed evidence. Sidecar critique agreed:
root-action-context handle cleanup alone is not a credible `2x` lever because
the loop still coordinates per-step submit/resolve and still has large
replay/sample/learner/search/wait surfaces. The next speed-worthy local gate
should move a broader owner surface: an owner replay/sample/learner-batch ring,
a coarse owner actor/search transaction tranche, or a disjoint learner
publication/version-ticket path. Do not launch H100 for this handle rung alone.

## Parked

- Owner process/inline/threaded boundary variants beyond the measured
  action-only threaded/background falsifier.
- Legacy OPT-124 inner two-phase owner replay as a standalone lane; the selected
  owner-search path already wires/proves compact Torch inner two-phase device
  replay.
- More small replay-sampling caches.
- More tiny owner-process copy/proof edits.
- Direct autoreset as a speed win.
- GPU utilization sampler in accepted speed rows; labeled diagnostics may opt in.
- Blind GPU mechanics rewrite. GPU-resident mechanics is allowed only as a
  toy-ceiling experiment with fixed-action semantics proof.
- H200/B200 or multi-GPU as speed baselines. H200/B200 are allowed only as
  explicit memory-headroom diagnostics.
- Blind PufferLib port. PufferLib/EnvPool/Sample Factory/Isaac-style systems
  are active references for fixed-buffer/vectorized env architecture.
- Scalar-ref/local-process polishing.
- Snapshot-file transport.
- Compile default.
- Blind MCTX/backend swap. MCTX/JAX-style search is an active reference for
  batched fixed-shape search architecture.
- More result-payload-format tweaks.

## 2026-06-07 Owner Deferred Replay Proof Update

Local owner proof now exists for the default-off compact Torch
`defer_one_simulation_replay_payload` lane. This is not H100 speed evidence.

What changed:

- `CompactOwnerSearchServiceV1` now records owner-inner device replay flush
  telemetry from replay payload metadata.
- Deferred one-simulation replay fails closed at the owner boundary if the
  flushed payload was not materialized on flush, reports identity mismatch, or
  reports a model-refresh-crossed count.
- Direct transition-batch speed-row replay now exposes matching
  `compact_owner_search_direct_transition_batch_replay_*` flush counters and
  rejects rows that request the deferral flag without final pending `0`,
  refresh-crossed `0`, identity-match count equal to transition count, and no
  replay-payload D2H bytes.
- Smoke/Modal launch plumbing now carries
  `--compact-torch-defer-one-simulation-replay-payload` into
  `CompactTorchCompileConfig`, while preserving it as a request bit in speed-row
  evidence/config summaries.

Validation passed:

```text
python -m py_compile ... owner service / speed-row / modal / evidence files
uv run ruff check ... owner service / speed-row / modal / evidence files
uv run pytest owner deferred replay + compact Torch deferred replay slice: 6 passed
uv run pytest speed-row proof/evidence/direct-store slice: 6 passed
git diff --check
```

Follow-up validation closed the stale smoke fixture: the full
`tests/test_compact_coach_speed_row_smoke.py` suite now passes `90 passed` after
adding the new owner-search request bit to the expected config surface.

Latest read: local flagged smoke passed. Run
`opt132-local-owner-deferred-one-sim-direct-transition-batch-smoke-20260607-r7`
completed with `ok=true`; direct transition count `12`, direct deferred
flush/materialized/identity/recurrent counts `12/12/12/12`, direct
model-refresh-crossed `0`, replay-payload D2H bytes `0.0`, owner-inner
deferred flush/materialized/identity/recurrent counts `16/16/16/16`,
owner-inner model-refresh-crossed `0`, and owner-inner pending final `0`.

Next read: the bounded H100 probe has now run and is speed-rejected despite
clean proof. Keep the deferral mechanics as support only. The active path is
the next owner-resident root/mechanics/search/parent-wait/learner-publication
or broader fixed-buffer surface.
