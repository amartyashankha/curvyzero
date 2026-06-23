# Optimizer Follow-Ups

Date: 2026-06-09

This file is the active queue plus closed anchors that prevent old lanes from
reactivating.

## 2026-06-09 Production Fixed-SoA Handle Consumption Follow-Up

Status: closed-local as a production-facing contract proof, not speed evidence.

The speed-row owner-search fixed-SoA replay metadata now requests the existing
fixed-SoA learner-batch handle ring, and `CompactOwnedLoopV1` records whether
the learner actually consumed a resident batch handle. The proof fails closed:
if handle resolution falls back, the learner may still train, but
`compact_owned_loop_learner_resident_batch_handle_consumed` is false and the
materialized-parent fallback count/reason are surfaced.

Validation:

```text
uv run ruff check src/curvyzero/training/compact_owned_loop.py tests/test_compact_owned_loop.py scripts/build_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row_smoke.py
  passed

uv run pytest tests/test_compact_coach_speed_row_smoke.py::test_owner_search_replay_store_metadata_receives_fused_tensor_native_flags tests/test_compact_owned_loop.py tests/test_source_state_hybrid_observation_profile.py::test_compact_replay_ring_samples_partial_terminal_entry_without_successor -q
  28 passed
```

Decision: do not launch H100 for this rung. It proves the production-facing
fixed-SoA path now asks for handle-ring sampling and that the owned loop records
learner consumption/fallback counters. Next gate is corrected local whole-loop
timing with these counters clean; only then consider a same-work H100 row.

## 2026-06-09 Owner-Slot Ceiling Follow-Up

Status: closed-local as a boundary fixture, not speed evidence.

The fixed-action-tape harness now has a corrected local whole-loop denominator
and a default-off `--run-owner-slot-ceiling` proof. This closes the narrow
mechanics/root/action feedback boundary, owner-context device replay rows,
real compact replay-ring append/sample metadata, and the first resident device
learner unroll-2 batch on a nonterminal owner-slot fixture:

```text
mechanics frame slot writes/generation/digest verified
root request built from slot, not from batch
HybridCompactBatch object count 0
owner-selected action result write/read balanced
next mechanics step consumes the owner-selected action
root observation copy bytes 0
stage_replay_append_entries transport/transitions 3/3
stage replay device builds / rows 3/12
stage replay drained records / index rows 3/12
real replay-ring append calls / records / rows 3/3/12
real replay-ring sample device row metadata true
real replay-ring resident device sample true
real replay-ring sample source compact_replay_ring_resident_sample_gate
real replay-ring observation-provider fallback 0
learner unroll-2 built true
learner unroll-2 rows/target rows 8/8
learner action/reward/value/policy shapes [8,2] / [8,2] / [8,3] / [8,3,3]
learner unroll-2 host fallback false
parent replay objects / replay object entries / selected groups 0/0/0
```

Decision: do not launch H100 for this rung. It is a contract fixture that
keeps the next patch honest. The first owner-action-context production helper,
`build_compact_replay_index_rows_v1_from_owner_action_context_payload()`, now
builds real `CompactReplayIndexRowsV1` without a full parent root batch and
matches the trusted root-batch builder in replay-contract tests. Its device
sibling,
`build_compact_device_replay_index_rows_v1_from_owner_action_context_payload()`,
builds packed tensor rows from the same owner context with host fallback
forbidden. The owner-slot fixture now drains those rows into
`_CompactReplayRingV1`, samples with device-row metadata, and builds a
resident device learner unroll-2 batch. The next active follow-up is the
broader Puffer-style fixed-buffer production handle rung:

```text
production sample returns an owner-issued batch handle with lifetime/generation/digest proof
fixed resident row/window slots or a handle ring feed the sampler directly
learner-owned unroll-2 targets are consumed from that resident handle
learner consumes that handle or reports explicit materialized-parent fallback
parent replay/sample objects 0
selected-group/tiny-gather loop 0
terminal roots do not fake samples
normal-death/action/tensor-native gates preserved
```

The fixed-SoA learner-batch handle-ring work remains support only. It is useful
for vocabulary/proof projection, but if it only wraps an already-built parent
batch inline, it is not a speed lane.

## Active Update After Owner-Boundary Reorientation

| Priority | Lane/Owner | Follow-Up | Status | Closing Evidence | Source/Last Evidence |
| --- | --- | --- | --- | --- | --- |
| P0 | columnar append | Preserve corrected columnar direct append as the current fastest single H100 input, but do not promote it. | support / do-not-repeat unchanged | Corrected no-preset H100 `opt132-h100-columnar-append-direct-table-b1024a1-normal-unroll2-m724-w180-r2-20260607` ran `15852.67 env/s`, `46.7666s`, about `1.25x` OPT-104 and about `1.01x` faster than direct-table r3. Proof: owner-search threaded proxy, slab bypass, direct transition replay, columnar append used, `724` slots, `ring_entry_objects=0`, direct prebuilt sample used, direct group objects `0`, maintained tensor-native table source, fallback `0`. | Not repeat-proven and not `2x`; do not promote. It proves ring-entry removal alone is too small. Preserve as support for fixed-buffer work. |
| P0 | direct table-builder | Preserve direct record-table building; exact-repeat only for a speed claim or contradiction. | support / do-not-repeat unchanged | `opt132-h100-direct-table-builder-b1024a1-normal-unroll2-m724-w180-r3-20260607` ran `15681.59 env/s`, `47.2768s`, about `1.24x` OPT-104. Nested owner learner telemetry proves `direct_record_table_v1`, direct build used `true`, table build `0.637s` for `9770` rows, worker replay append `13.879s`, ring append `9.848s`, and remaining `ring_entry_objects=724`. | Superseded slightly by columnar append, but still useful. Exact-repeat only for a speed claim or contradiction. |
| P0 | lazy selected table | Keep lazy selected direct table build default-off. | closed/rejected | H100 `opt132-h100-lazy-selected-table-columnar-b1024a1-normal-unroll2-m724-w180-r2-20260607` passed proof but slowed to `11179.26 env/s`, `66.3171s`. It cut append/ring append (`14.169s/11.523s -> 10.224s/7.101s`) but moved cost into owner train/sample (`9.778s -> 48.696s`, owner train sample `2.777s -> 41.640s`) and reintroduced `352` direct group objects. | Do not chase this lane unless a new design avoids sampled record-group objects and learner/sample blowup. |
| P0 | selected maintained gather | Keep selected-maintained direct-prebuilt gather as explicit/default-off experiment only. | closed/rejected | Local CPU benchmark passed proof and showed a `3.35x` sample-build surface win (`0.0167s` vs `0.0560s`) for `512` sampled rows over `724` records, but H100 `opt132-h100-selected-maintained-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260609` was proof-clean and slower: `12028.55 env/s`, `61.6347s`, `0.759x` columnar r2 and `0.948x` OPT-104. It used selected-maintained impl/source, selected groups `352`, fallback `0`, but sample gate rose to `22.581s`, owner learner train to `31.768s`, direct append to `23.851s`, and ring append to `20.513s`. | The CPU/GPU inversion means per-record selected GPU gathers are the wrong shape. Default restored to maintained concat plus single gather; selected path now requires `compact_muzero_learner_batch_tensor_native_replay_selected_maintained_gather`. Next replay/sample work must be fused/batched selected gather or broader fixed owner-buffer work, not many tiny selected groups. |
| P0 | direct-prebuilt sample | Keep the local direct tensor-native prebuilt sample path as proof support, not as the whole strategy. | closed/support | Local and columnar H100 proof skip per-sampled-group resident sample object construction when maintained unroll-2 tensor-native tables fully cover the sample universe. Source-state, speed-row, Modal result-bundle, and owner-search telemetry proof fields expose direct prebuilt requested/eligible/used, fallback count/reason, direct group-object count, and group-object build skipped. | Use these fields inside fixed-buffer rows. The latest blocker is broader owner replay/table/sample materialization, not missing direct-prebuilt proof. |
| P0 | inline-background owner | Do not keep polishing `owner_search_inline_background_proxy` as the next headroom path. | closed / speed-failed-long | Local proof passed and short H100 r36/r37 repeated above OPT-104, but r38-long fell below OPT-104 and below threaded r34-long. This answered the variance question: it is a useful discriminator, not a stable path to 2x. | r36 `14906.20`, r37 `13679.66`, r38-long `12361.15`; r38 `0.974x` OPT-104 and `0.940x` threaded r34-long. |
| P0 | candidate preservation | Preserve vectorized reset RNG + refresh-4 + threaded/background owner maintenance as the current best candidate shape. | active | Any next patch must keep normal-death proof, tensor fallback `none`, terminal sample/target rows present and equal, accepted-fast-path violations `[]`, unroll2 violations `[]`, final owner maintenance clean, and background maintenance overlap if threaded. | r32 `14002.12 env/s`, r33 repeat `13640.60`, r34 long `13145.34`; all above OPT-104 `12689.38`. |
| P0 | one-simulation deferral | Keep the owner-loop deferral proof as support, but close the speed lane unchanged. | closed / proof clean, H100 speed rejected | Local flagged smoke `opt132-local-owner-deferred-one-sim-direct-transition-batch-smoke-20260607-r7` passed with direct transition count `12`, direct flush/materialized/identity/recurrent `12/12/12/12`, refresh-crossed `0`, replay D2H `0.0`, owner-inner counts `16/16/16/16`, and owner-inner pending final `0`. The one H100 probe `opt132-h100-owner-deferred-one-sim-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607` also passed proof with `724/724/724/724`, crossed `0`, D2H `0.0`, owner-inner final pending `0`, but ran only `13691.98 env/s` / `54.1467s`. | Do not repeat this lane unchanged. It moved work into append/flush (`22.377s` direct append; `3.613s/4.381s` deferred/device flush) and regressed versus columnar r2 `15852.67 env/s`. |
| P0 | async learner overlap | Keep in-process async owner learner as proof support, but close the current-stack speed lane unchanged. | closed / proof clean, H100 speed rejected | Local smoke `opt132-local-owner-async-learner-columnar-directtable-smoke-20260607-r1` passed with async submit/completed/pending `4/4/0` and actions while async pending `9`. H100 `opt132-h100-owner-asynclearner-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607` passed proof with async worker enabled, submit/completed/pending `90/90/0`, max pending `2`, actions while async pending `510`, wait `0.851s`, failed `false`, direct transitions `724`, columnar append used, and normal-death proof intact. Speed was `12954.74 env/s`, `57.2282s`. | Do not repeat in-process async unchanged. It regressed versus columnar r2 `15852.67 env/s`; worker replay append, learner train, search, parent wait, and observation all moved the wrong way. |
| P0 | resident root view | Keep the direct-root resident-root-view proof as a fail-closed gate, not a speed claim. | closed / local proof clean | Implemented `--owner-search-require-resident-root-view` for direct-root owner proxies. Shared-memory owner mode fails closed for this proof. Local smoke `opt132-local-owner-resident-root-view-threaded-directtransition-smoke-20260607-r5` passed with resident-root-view required/proved true, kind `direct_root_batch_resident_handle_v1`, direct root publish/resolve `15/15`, H2D/D2H `0.0/0.0`, host fallback false, action-only true, parent rows `0/0`, search payload/visit/root-value bytes `0/0/0`, transition entries/batches/transport `12/3/3`, and final policy lag/pending/failed `0/0/false`. | This proves owner consumption of the source resident root handle. It does not remove parent root-batch construction or prove H100 speed; next work must move a larger parent/root/search/mechanics/learner-publication surface. |
| P0 | resident root host-observation stub | Preserve the new direct-root resident host-observation stub as a local fail-closed proof gate only. | closed / local proof clean | Implemented `--owner-search-resident-root-host-observation-stub` for direct-root owner-search slab bypass with required resident-root view. Local smoke `opt132-local-owner-resident-root-hoststub-threaded-directtransition-smoke-20260607-r1` passed with stub requested/stubbed true, kind `zero_stride_shape_only_v1`, materialized bytes `0`, logical bytes `262144` in the last step and `3932160` total, resident-root-view required/proved true, H2D/D2H `0.0/0.0`, host fallback false, action-only true, parent rows `0/0`, search payload bytes `0`, transition entries/batches/transport `12/3/3`, and final policy lag/pending/failed `0/0/false`. | This removes parent host-observation materialization for the direct resident-root proof path, but still calls the parent root-batch builder. Do not launch H100 for this alone; the next patch is a true root-view/build-request handoff or combined owner-buffer/root/search/learner-publication/mechanics surface. |
| P0 | direct root build request | Preserve as support; do not repeat unchanged. | closed / proof clean, H100 speed rejected | Added `CompactRootBuildRequestV1`, owner-side `build_compact_root_batch_v1_from_request()`, direct-root store `publish_root_build_request()`, `CompactOwnerSearchSlabProxyV1.run_action_step_from_root_build_request()`, lazy direct-proxy routing, and `CompactOwnerSearchDirectStepperV1(direct_root_build_request=True)`. Full owner-search service suite passes (`45 passed`). Local smoke `opt132-local-owner-rootbuildrequest-threaded-directtransition-smoke-20260607-r3` passed with publish/resolve/owner-build `15/15/15`, parent builder used/calls `false/0`, request observation bytes `0`, parent rows/search payload bytes `0/0`, direct replay `3/12/3`, and clean maintenance/policy lag. H100 `opt132-h100-rootbuildrequest-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607` preserved proof at scale: publish/resolve/owner-build `904/904/904`, parent builder used/calls `false/0`, root build sec `0.0`, request observation bytes `0`, parent rows/search payload bytes `0/0`, direct replay `181/724/181`, action mismatches `0`, normal-death proof true, maintenance/policy lag `0/0`. Speed was rejected: `11327.75 env/s`, `65.4477s`, `0.893x` OPT-104 and `0.715x` columnar r2. | Root-build request removes the parent root builder but does not improve the whole loop alone; parent wait, replay append, learner train, worker search, observation, and slab all regressed versus columnar r2. Keep the proof gate as support for larger owner-resident designs. |
| P0 | owner-local transition derivation | Preserve as support; close as the main speed lane. | closed / proof clean, H100 speed rejected versus columnar r2 | Default-off `--owner-search-owner-local-transition-derivation` is implemented and H100-proven. r1 used the wrong checkpoint and failed before a row; r2 timed out during local Modal detach; r3 ran remotely but failed a stale generic transition-batch proof guard, which is now fixed and regression-tested; valid r4 `opt132-h100-owner-local-transition-derivation-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r4-20260608` passed proof. It used derived schema/kind, direct replay, columnar append, tensor-native replay, and normal-death proof; derived batches/transitions/transport `181/724/181`, cache hits/misses `724/0`, checksum verified/mismatch `724/0`, fallback/pending/drop `0/0/0`, parent outcome bytes/fields `0/0`, GPU sampling false. Speed was `13265.51 env/s`, `55.8875s`: `1.045x` OPT-104 but only `0.837x` columnar r2. | Do not repeat owner-local derivation unchanged. Keep its counters and derived-outcome mechanics as support inside the broader fixed owner-buffer/root/search/mechanics/learner-publication patch. |
| P0 | owner/proxy transition closure | Preserve as launch-plumbed support; do not launch H100 for this rung alone. | closed-local / support | Default-off `owner_proxy_transition_closure=True` makes `CompactOwnerSearchDirectStepperV1` call proxy `stage_owner_proxy_transition_from_root_build_request()` instead of parent `_stage_previous_derived_transition`. `CompactRootBuildRequestV1` now carries validated `joint_action`; `CompactOwnerSearchSlabProxyV1` caches previous action frames, validates applied actions from the current request, builds the existing derived transition-batch schema, and stages it before owner request/train scheduling. Focused proof monkeypatches parent previous-transition closure, derived flush, and commit helpers to raise while the real direct-root threaded proxy still appends a derived transition batch; mismatch proof corrupts applied action and fails closed. The public speed-row CLI, local Modal launcher, remote Modal producer, result summaries, and proof/report allowlists now expose `--owner-search-owner-proxy-transition-closure`, so the guarded overlap H100 row can honestly preserve this support gate. Validation: focused proxy-closure packet `4 passed`, full owner-search service `55 passed`, replay-contract suite `47 passed`, full speed-row smoke `99 passed`, ruff passed. | This closes parent closure authority for the guarded direct-root path, not the speed goal. Keep proof fields: proxy request bit true, source `owner_proxy_cached_state_v1`, parent previous-transition closure count `0`, proxy closure closed count positive and matched to transitions/transport, digest verified, applied-action verification positive, mismatch `0`, fallback `0/none`, pending `0`, parent applied-action validation `0`, parent rows `0/0`, search payload bytes `0`. Next work must move owner wait/search dispatch, learner publication/update, mechanics/observation, or fixed owner-buffer handles. |
| P0 | fixed-action-tape owner-buffer toy | Preserve as parity/proof support only; reject naive fixed-buffer direct env step as the speed lane. | closed-local / proof clean, speed-shape not enough | Refreshed probes passed. Mechanics-only B1024/m724/w180 artifact `owner_buffer_mechanics_b1024_m724_w180_20260608.json` had proof true but fixed direct was slower than compact profile: `0.843x`, `1.3959s` versus `1.1774s`. Rendered search-feedback slab/replay B128/m180/w45 artifact `owner_buffer_render_slab_b128_m180_w45_20260608.json` had proof true, slab replay proof true, closed-loop feedback true, `224` replay appends, `224` sample gates, no root-observation copy, and toy `1.129x`, but CPU-oracle render took `218.223s` and there is no learner/GPU/H100 claim. | Use the toy only as a contract fixture for the broader owner-owned action+step/root/replay path. Do not launch H100 or rewrite the hot loop around `_fixed_buffer_direct_step` alone. |
| P0 | ring-batched replay append | Preserve ring-batched owner replay append/cache refresh below direct table-builder, but do not stop there. | support / do-not-repeat unchanged | H100 r3/r4 kept the same `724 -> 181` transition-batch transport but batched ring append/cache refresh. Replay append fell from r2 `31.926s` to r3/r4 average `19.420s`; speed rose from r2 `13618.91 env/s` to r3/r4 average `14277.52 env/s`. The direct table-builder row superseded it as fastest single evidence but depends on this stack. | r3 `14394.77 env/s`, `51.503s`, replay append `20.204s`; r4 `14160.27 env/s`, `52.356s`, replay append `18.636s`; tensor-native reused/missing `719/0`, pending `0`, action mismatch `0`. |
| P0 | next headroom | Implement the next broader owner-boundary patch, not another replay-only, thread-overlap, root-build-request repeat, or proof-only tweak. | active | Transport batching alone was neutral; ring-batched append was useful; direct table-building was useful; columnar append removed `_CompactReplayRingEntry` but only gained about `1%` over direct-table r3; fixed-SoA exact/locality, one-simulation deferral, in-process async learner overlap, and root-build-request alone were proof-clean but speed-rejected. The next patch must attack owner search dispatch/parent wait, mechanics/observation ownership, learner publication/update, or a fixed owner-buffer design that removes object/materialization surfaces without moving cost into learner train/sample. Use root-build-request/resident-root/stub gates as support, not as the whole speed lane. | Mature systems use fixed buffers and small handles. Kill criterion: a local proof must show a real owner graph or hot-data boundary changed, with no hidden parent rows, no search payload bytes, unchanged terminal/action checksum proof, resident-root-view, host-stub, and parent-build-avoided proofs preserved where claimed, and explicit counters for the moved/removed root/search/parent-wait/mechanics/observation traffic. |
| P0 | maintained tensor-native replay table handle | Preserve as a default-off local support rung; do not treat it as speed evidence. | closed-local / support | `compact_muzero_learner_batch_tensor_native_replay_maintained_table_handle` makes the replay snapshot carry an aligned maintained-table handle and lets the maintained tensor-native gather consume it. Proof counters now surface requested/used/record/missing/row counts through source telemetry, compact speed rows, Modal bundles, build smoke, and launcher proof fields. Validation passed: ruff, focused handle/projection packet `3 passed`, full source-state `127 passed`, full speed-row smoke `100 passed`, benchmark tests `4 passed`. | This is not the selected-maintained path and does not justify H100 alone. It creates the next authority hook. The next replay/sample slice should turn this into an owner-issued fixed-SoA or learner-batch handle ring with lifetime/watermark proof, or be superseded by a broader owner transaction/learner-publication patch. |
| P0 | owner-resident step-frame boundary | Preserve the slot-backed frame/root-search transaction as support for the owner replay/sample/learner-handle path. | closed-local / support | Main-thread read plus sidecars converged that Curvy still violates the mature fixed-buffer pattern: too much per-step cadence still crosses parent Python. Local slices now bypass parent compact-batch materialization, parent dense-action reconstruction, pending compact/root sidecars, returned root sidecars, parent previous-transition closure, parent pending action-step storage in proxy-closure mode, and parent step-frame root-request construction for the ring-backed transaction path. The guarded overlap/proxy H100 r3 read was proof-clean but slower than columnar r2 (`15541.95 env/s`, `47.7016s`, `0.980x` columnar r2), so replay-only and shallow overlap rungs remain support. The latest local rung makes the proxy build `CompactRootBuildRequestV1` from the `curvyzero_compact_owner_mechanics_step_frame_slot/v1` ring; tests can poison the slab `_root_build_request_from_owner_step_frame_slot_v1()` helper while the owner-boundary path passes. Speed-row/Modal/evidence projection now carries transaction counters for parent build/store `0`, owner build/publish positive, slot generation/digest verified, and action identity verified. | Do not launch H100 for this rung. The next active implementation is production owner replay/sample/learner-batch handles, using these slot/root/action proofs as guards. |
| P0 | owner action dispatch profile-loop overlap | Preserve as proof support; close the current H100 speed lane. | closed / proof clean, H100 speed rejected | Local implementation places existing profile-loop parent work between owner action submit and resolve. `compact_owner_action_dispatch_step_overlap` requires action-step boundary, compact slab, direct root-build request, and direct stepper submit/resolve. The valid H100 row `opt132-h100-action-dispatch-overlap-proxyclosure-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r3-20260608` completed `ok=true`, proof-clean: submit/resolve `904/904`, sync-wrapper count `0`, completed-at-submit `0`, wait-in-submit `0`, pending `0`, max pending `1`; proxy closure closed/transition `724/724`, batch/transport `181/181`, fallback `0/none`, pending `0`, digest verified. Speed was `15541.95 env/s`, `47.7016s`, `1.225x` OPT-104 but only `0.980x` columnar r2. | Do not repeat overlap+proxy unchanged. Keep its raw fail-closed counters as guards in future owner-boundary rows. The next speed attempt must change a broader owner surface: fixed owner mechanics/root-frame handles, learner publication/update tickets/refs, or another local proof that removes real parent hot-loop data movement. |
| P0 | owner mechanics step-view boundary | Preserve the direct-root mechanics step-view boundary as local support only. | closed-local / support | `compact_owner_mechanics_step_boundary` is default-off and requires owner action-step feedback, resident observations, direct root-build request, and resident host-observation stub. `HybridBatchedObservationProfileManager.step()` builds `CompactOwnerMechanicsStepViewV1` after mechanics/resident observation update and before `_make_compact_batch`, with `observation=None` and borrowed root/mechanics sidecars. Focused proof monkeypatches parent `_make_compact_batch` and legacy parent root build to raise while the owner path passes. Validation: expanded owner-boundary/guard packet `6 passed`; ruff passed. | This closes one parent compact-batch materialization surface in the guarded path, not the speed goal. Keep it as a guard inside broader owner step-frame/fixed-buffer work; reverse if a legitimate direct-root owner path really needs full parent `HybridCompactBatch` materialization. |
| P0 | owner-published dense next action | Preserve the direct-root fixed-slot dense-action publication as local support only. | closed-local / support | `CompactSearchActionStepV1` now carries optional `dense_joint_action`. Owner direct-root fixed-slot results derive the dense `(batch, player)` action from validated owner root/search sidecars, and `CompactOwnerSearchDirectStepperV1` validates checksum/digest/legality/active-selected identity before applying it. Focused tests prove parent dense-action reconstruction can be monkeypatched to raise while the owner path passes with owner dense action present/used, parent assembly avoided, fallback `0/none`, mismatch `0`, bytes/checksum/digest set, and parent reconstruction count `0`. Validation: dense owner-boundary packet `5 passed`, direct-root owner-service packet `4 passed`, ruff passed. | This closes one parent action-assembly surface, not the whole speed problem. Keep it as a guard inside the broader owner step-frame/fixed-buffer patch; reverse if any legitimate direct-root fixed-slot path cannot supply a dense action without weakening action/legal/terminal proof. |
| P0 | direct owner pending/returned root handles | Preserve direct-stepper pending compact/root sidecar avoidance and direct-root returned root-sidecar avoidance as local support only. | closed-local / support | `_PendingCompactSearchV1.compact_batch` and `root_batch` are optional and `CompactOwnerSearchDirectStepperV1` stores both as `None` in `_pending`, retaining action-step identity handles instead. Direct-root `CompactRolloutSlabStepV1` now also returns `root_batch=None`; `_slab_telemetry()` reads `CompactRootBuildRequestV1`; and owner-local transition validation reads mechanics outcome sidecars directly from the request. Proof fields show pending compact-batch sidecar stored `false`, storage avoided `true`, store count `0`, avoided count positive; pending root-batch sidecar stored `false`, storage avoided `true`, store count `0`, avoided count positive; returned root-batch sidecar stored `false`, storage avoided `true`, build count `0`; action-step identity handle stored `true`; root-build request stored `false`. Validation: request-outcome contract packet `2 passed`, direct-root owner packet `3 passed`, combined owner/profile packet `6 passed`, full owner-search service `50 passed`, ruff passed. | This closes parent pending-state storage and returned-step root materialization surfaces; it is not H100 speed evidence. Keep as a guard inside broader owner step-frame/fixed-buffer work; reverse if a legitimate direct-owner replay/transition path requires compact-batch or root-batch sidecars in pending hot state or a materialized returned root batch in direct-root mode. |
| P0 | minimal owner step payload snapshot | Preserve as support-only proof hardening. | closed-local / support | `compact_owner_minimal_step_payload_snapshot` is default-off and requires `compact_owner_action_step_boundary`. It keeps only summary/action payload keys, counts full compact payload bytes and key count elided, and is now carried by the direct-root/threaded boundary test. Validation: focused source-state tests `4 passed`; ruff passed. | This is not a speed row and not the owner-resident step-frame ring. Reverse if omitted payload keys are needed by a legitimate owner-boundary replay/sample path. |
| P0 | guarded owner action-step boundary | Preserve the new profile-loop boundary proof and use it as a direct-root/action-only speed-row guard before any H100 speed claim. | closed-local / proof support only | Default-off `compact_owner_action_step_boundary` now requires compact slab search-feedback and no scalar timestep materialization. It owns cached next action in the profile loop, verifies `manager.step()` payload action equals the applied action, requires the slab next action, and reports strict proof counters/checksums/failure reason. The speed-row producer and Modal wrappers now project the same proof fields and fail closed for direct-root/action-only owner rows when the proof is missing or inconsistent. A real local direct-root/threaded proof now also passes: `test_hybrid_profile_owner_action_step_boundary_uses_direct_root_build_request` monkeypatches the legacy parent root-batch builder to raise, then proves `CompactOwnerSearchDirectStepperV1` + `CompactLazyThreadedOwnerSearchSlabProxyV1` uses root-build-request handoff, owner build, resident-root view, host-stub/no-observation transport, zero search payload bytes, zero parent committed/stored rows, and closed action-feedback checksums across `3` iterations. Validation passed: focused boundary tests `3 passed`; ruff. | Do not launch H100 for this boundary alone. Use the guard in the broader owner-buffer/root/search/mechanics/learner-publication patch, rejecting missing proof, speed overclaims, parent replay/search payload/root-builder fallback, host fallback, action mismatch, owner-local derivation fallback, pending derived transitions, and fake terminal samples. |
| P0 | resident terminal final-observation host allocation | Preserve the new resident sparse-final host allocation elision as mechanics/observation support only. | closed-local / support | In resident-observation search with scalar timestep materialization disabled, `HybridBatchedObservationProfileManager.step()` no longer builds a dense host `final_observation = zeros_like(observation)` for terminal rows. Resident sparse device final rows already own the terminal observation sidecar. New telemetry exposes `resident_final_host_observation_dense_elided_count`, `..._bytes`, and `..._row_count`. Focused resident terminal test proves sparse final rows remain present and the dense host allocation is elided; boundary/direct-root tests still pass. Validation: focused source-state tests `4 passed`; ruff passed. | This is not a speed claim or H100 lane alone. Preserve it inside broader mechanics/observation/root-owner work; reverse if scalar timestep materialization, host final observations, or resident final-row replay semantics require dense host final observations in this mode. |
| P0 | fixed SoA slot-candidate support | Preserve the fixed-SoA slot-candidate path as proof support, but close it as a standalone speed lane. | closed / proof clean, H100 speed rejected | Local parity/source-profile tests and tiny smoke `opt132-local-fixed-soa-slot-digestdefer-smoke-20260607-r3` passed. Labeled H100 support row `opt132-h100-slotcandidate-support-b1024a1-normal-unroll2-m724-w180-r1-20260607` also passed proof: normal-death gate true, accepted/tensor-native/cache/unroll violations `[]`, fixed SoA requested/used true, fallback `0/none`, tensor-native replay used, direct transitions/batches/transport `724/181/181`, columnar append false, search payload bytes `0`, fixed-SoA object/table counts `0`, and terminal rows present. Speed was rejected: `10348.87 env/s`, `71.638s`, `0.816x` OPT-104 and `0.653x` columnar r2. | Do not repeat fixed-SoA exact/locality/slot-candidate unchanged. The path proved the data structure can run, but successor-index/sample/train work moved enough wall to lose. Keep it only as support inside a broader fixed owner-buffer/root/search/learner-publication patch. |
| P0 | owner root-cache slicing | Preserve the local deferred-maintenance root-cache slice as support and proof hardening, not as the speed lane. | closed-local / support | `CompactOwnerSearchServiceV1.run_action` now slices staged maintenance root caches to transition-batch/index-entry `record_indices`, `next_record_indices`, and current actor step when introspectable; opaque entries fall back to full snapshots and count that fallback. Metadata and owner-sample telemetry expose snapshot count, full entries, retained entries, required entries, dropped entries, and full-fallback count. Local validation passed: ruff, focused direct-transition pytest, and full owner-search service pytest (`47 passed`). | This removes avoidable object fanout but is not enough to explain the missing `2x`. Keep the counters in future rows. Owner-local derivation and fixed action-result slot are now both support-only; the next real implementation is the broader fixed owner-buffer/root/search/mechanics/learner-publication patch. |
| P0 | fixed action-result slot | Preserve as support; close as the main speed lane. | closed / proof clean, H100 speed rejected versus columnar r2 | `CompactOwnerActionResultSlotTableV1` supports default-off inline/threaded direct-root owner proxies: root-build action requests carry a slot id, the owner writes the full action result to the slot, and the worker result queue returns a tiny slot stub. Local proof row `opt-fixed-action-result-slot-local-smoke-proof5-20260608` passed, and H100 `opt132-h100-fixed-action-result-slot-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608` kept proof closed: fixed-slot requested/used `true/true`, slot count `4`, acquire/write/read `904/904/904`, pending `0`, wire/full result bytes `414/4837`, parent builder calls/root objects `0/0`, action-only true, terminal sample/target `512/512`, accepted/tensor-native violations `[]`. Speed was `12794.42 env/s`, `57.9453s`: slightly over OPT-104 but only `0.807x` columnar r2. | Do not repeat fixed-slot unchanged. It improved root-build-request parent wait (`26.151s -> 19.035s`) but did not beat the fastest support stack. Owner-local derivation is now also closed as standalone; next work is broader fixed owner-buffer/root/search/mechanics/learner-publication movement. |
| P0 | inline owner | Treat inline owner maintenance as unstable unless a new inline executor patch proves otherwise. | parked unless code changes | r30 won once, r31 exact repeat failed with reset still fixed and slab back to `42.728s`; this points to inline owner/slab dispatch variance. | r30 `14230.53 env/s`; r31 `11794.65 env/s`. |
| P0 | reset RNG | Keep vectorized seeded random tape generation. | closed-speed-patch / preserve | Exact scalar parity passed; focused reset/autoreset tests passed; H100 actor autoreset dropped from about `13-14s` to about `2s` in inline rows. | `src/curvyzero/env/vector_source_random.py`; r29/r30/r31. |

## 2026-06-08 Owner Slot Guard Closure

The stale-generation/reuse follow-up for the slot-backed mechanics frame is now
closed-local support. The code validates the slot metadata, generation modulo
slot, live `slot_generation`, handle shape, and digest before any direct-step
replay commit, proxy closure, root action submit, or pending-dispatch mutation.
`CompactOwnerSearchDirectStepperV1` also tracks consumed generations so a
previously valid frame cannot be replayed through the same stepper.

Validation:

```text
ruff touched files: passed
focused owner-frame/direct-root packet: 3 passed
source-state owner-boundary packet: 6 passed
owner-search direct-root/proxy/overlap packet: 11 passed
```

Follow-up status: do not repeat this guard as a separate task. Use it as a
support invariant for the next broader owner root/search transaction or a
disjoint learner-publication ticket/ref patch.

## 2026-06-08 Root Action Context Closure

The pending root-build-request storage follow-up is now closed-local support.
`CompactRootActionContextV1` keeps only action-critical root sidecars after
owner submit, and async direct dispatch now stores that context instead of
`CompactRootBuildRequestV1` in both the direct stepper and the owner-search
proxy pending dispatch. Tests assert the pending dataclasses have no
`root_build_request` slot and public metadata reports root-action context
storage instead.

Follow-up status: this fixes a proof/data-ownership leak. It does not yet make
the owner/proxy open the root/search transaction directly from the step-frame
slot. The next transaction patch must bypass parent root-request construction,
not just pending storage.

## 2026-06-08 Owner Root/Search Transaction Closure

The first slot-started owner/proxy root-search transaction follow-up is now
closed-local support. `CompactOwnerSearchDirectStepperV1` uses
`submit_owner_root_search_transaction_from_step_frame_slot()` for ring-backed
mechanics frames, and `CompactOwnerSearchSlabProxyV1` builds the
`CompactRootBuildRequestV1` from the slot/handle before publishing roots and
submitting the action dispatch.

Fail-closed proof now includes:

```text
slab _root_build_request_from_owner_step_frame_slot_v1 can be monkeypatched to raise
compact_root_build_request_v1_from_batch can be monkeypatched to raise
pending direct transaction stores no compact_batch in the transaction path
pending direct/proxy dispatches have no root_build_request slot
transaction parent root-request build count 0
transaction parent root-request stored false
transaction parent compact-batch stored false
transaction owner root-request build/root-store publish counts positive
transaction slot generation/digest/action identity verified
```

Validation:

```text
ruff touched code/scripts/tests: passed
focused transaction/source boundary packet: 4 passed
focused owner-search direct-root/action-dispatch packet: 13 passed
full owner-search service suite: 56 passed
full source-state hybrid observation profile suite: 125 passed
focused speed-row proof/report packet: 2 passed
```

Follow-up status: do not repeat this as a speed lane or launch H100 for it
alone. The remaining parent-owned transaction surface is the
`CompactRootActionContextV1` validation context and submit/resolve coordination;
the next useful patch must move that context/handle deeper into the owner or
switch to the disjoint learner-publication/update ticket/ref surface.

## 2026-06-08 Owner Root-Action Context Handle Closure

The root/action validation context follow-up is now closed-local support. The
slot-started transaction path now requires an owner/proxy
`CompactOwnerRootActionContextHandleV1`; the direct stepper no longer accepts a
parent `root_action_context` fallback in that transaction path. The proxy stores
the action-critical context in an owner table, resolves it by handle, verifies
the digest, releases it, and reports balanced store/resolve/release counters.

Fail-closed proof now includes:

```text
transaction pending direct dispatch root_action_context is None
proxy pending action dispatch root_action_context stored false
handle schema/id/transaction/dispatch/root counts/digest reported
owner context store/resolve/release counts positive and balanced
owner pending context count 0 after resolve
owner digest verified true
parent action-context validation count 0
owner action-context validation count positive
transaction parent root-action-context stored false
transaction parent root-action-context store/array/field counts 0
speed-row and Modal proof/report fields carry the context-handle proof
```

Validation:

```text
ruff touched code/scripts/tests: passed
source transaction/profile packet: 3 passed
source owner-boundary packet: 5 passed
owner-search focused action-dispatch/transaction packet: 2 passed
full owner-search service suite: 56 passed
speed-row owner-search/report/proof packet: 3 passed
```

Follow-up status: closed as proof/ownership support. Do not launch H100 or call
this a speed result. The next P0 must move a broader measured surface: fixed
owner replay/sample/learner-batch ring, a coarse owner actor/search transaction
tranche, or learner publication/version tickets. Root-action-context handle
cleanup alone is not enough headroom.

## Historical Queue Before r29-r34

| Priority | Lane/Owner | Follow-Up | Status | Closing Evidence | Source/Last Evidence |
| --- | --- | --- | --- | --- | --- |
| P0 | owner-search/sample-batch | Wire fast learner-ready/tensor-native batch path, or an equivalent owner-search-local path, into owner-search train sampling. | historical / superseded by r29-r34 candidate | Owner-search shared/no-clone route must keep r14 proof fields while reducing long-run train sample/materialization cost. The next proof should show the fast path requested/used with no fallback, or explicitly explain why owner-search needs a separate implementation. | r14 first win: `13497.30 env steps/sec`, `13.6561s`; r15 exact repeat failed speed at `10502.70`, `17.5498s`; r16 longer row `6491.80`, `114.202s`, owner train sample/update `52.807s/3.236s`; r17 composition failed before row because `compact_owned_loop_fused_learner_batch requires compact_owned_loop_entrypoint`. This is no longer the active first move after r32/r33/r34. |
| P0 | prototype/replay | Maintain candidate / offset / sampleable-row index state. | closed-surface / speed-failed | `_CompactReplayRingV1.sample_from_snapshot` no longer rebuilds the expensive sample universe on every sample call, and H100 sample gate falls to a non-dominant surface while terminal-only synthetic candidates and tensor-native proof fields remain fail-closed. | OPT-132BF passed proof and cut sample gate from `74.005s` to `4.116s`, but full-loop speed was still only `5362.68 env steps/sec` (`0.42x` OPT-104). |
| P0 | prototype/replay | Promote maintained tensor-native replay table state. | closed-proof / speed-failed | Actual accepted-preset same-work row returns table source/reused/missing rows plus checksum, sample-order, cache, fallback, implementation, parent learner-ready cache, and host-fallback proof fields before any speed claim. | OPT-132BD passed: violations `[]`, table source `maintained_record_table_v1`, reused records `1080`, missing records `0`, fallback count `0`; speed was `0.31x` OPT-104, so table gather proof alone is not enough. |
| P0 | architecture | Keep the architecture reset map and toy feasibility matrix current while implementing the ownership-selected candidate. | active | Written map covers env state/reset/death/observation/action, search, replay append/storage, sample, learner batch, learner update, proof/reporting, and every CPU/GPU plus Python/native boundary. Toy matrix covers fixed-buffer/vectorized env stepping, tensor-native replay sampling, learner-ready unroll-2 targets, batch-level unroll gather, and batched search. | `goal.md`, `ARCHITECTURE_RESET_2026-06-04.md`, `PRIMARY_RESIDUAL_OWNERSHIP_2026-06-04.md`; Cicero/Hubble sidecars: fixed buffers should be the API, and owner-search should own replay materialization when it already owns cached root/search state. |
| P0 | toy/env | Extend the fixed-action-tape env/outer-loop mechanics ceiling. | closed-local-proof / superseded-by-owner-search-proof | Local proof passes with compact-profile and fixed-buffer-direct full-state, per-step trajectory, output, action/timing-tape, expected-death, terminal/autoreset, rendered-observation equality, fixed-shape search/root metadata equality, and closed search-feedback slab/replay/sample equality. The primary-residual ownership map has since selected and locally proved the owner-search action-only replay path, so fixed-action tape is now constraint evidence, not the next lane. | OPT-132BG local B1024/180/45 artifact: compact `0.3586s`, direct `0.2179s`, measured new deaths `0`, terminal/autoreset rows `0`. OPT-132BH local B1024/1/0 terminal artifact: compact `0.2292s`, direct `0.1728s`, terminal rows `1024`, autoreset rows `1024`, wall death rows `1024`, zero-observation stub. OPT-132BI local rendered-observation artifacts passed: B32/3/1 borderless rendered rows `288`, nonzero obs `1179648`; B4/1/0 wall-death rendered rows `8`, terminal/autoreset rows `4`, zero_observation_stub false. OPT-132BJ local fixed-shape search/root artifacts passed: B32/3/1 rendered rows `288`, search roots `288`, active roots `192`, selected actions `192`, zero CTree/tolist/per-sim D2H; B4/1/0 terminal rendered rows `8`, terminal/autoreset rows `4`, search roots `8`, active roots `0`, selected actions `0`. OPT-132BK local closed slab proof passed: `slab_step_count=4`, roots `48`, selected actions `36`, committed index rows `28`, replay appends `3`, sample gates `3`, sample batch size `8`, `observation_materialized=false`, `slab_replay_failure_reasons=[]`, previous staged-action matches `3`, mismatches `0`, feedback differs from fixed tape `3`, compact/direct replay metadata match. |
| P0 | toy/slab | Close the next honest slab/replay/sample gate with search-feedback actions. | closed-local-proof | Deterministic search-selected actions become the next joint actions for both compact and direct toy envs; slab action-check, replay append count, replay-row checksum, row-index checksum, sample seed/order checksum, sample batch checksums, compact/direct equality, no-full-history-retention, failure-reason, and staged-action match/mismatch proof fields all pass. | OPT-132BK artifact `artifacts/local/curvytron_compact_coach_speed_row_results/opt132bk-local-fixed-action-closed-slab-replay-sample-toy-b4-m3-w1-20260604.json`; focused tests `test_fixed_action_tape_slab_replay_sample_proof_closes_search_feedback_loop` and `test_fixed_action_tape_slab_replay_terminal_roots_do_not_fake_samples`. |
| P0 | ownership | Promote the owner-search deferred-maintenance action-only replay path as the selected primary-residual candidate. | historical / superseded by r32-r34 and ring-batch | Real local profile smoke proves `committed_index_rows is None`, parent committed/stored row counts are `0/0`, parent reconstructed search result is false, search-result payload bytes are zero, owner materializes replay, final owner replay counts replace stale per-action counts, owner maintenance drains inside measured wall, and pending/inflight/failed close cleanly. Hardened local smoke now also fails closed for requested cadence/train steps, owner sample telemetry, zero-batch sample semantics, finite final drain, train timing split fields, action-only/two-phase handle fields, action-feedback checksum/mismatch proof, explicit maintenance work-item vs replay-entry drain semantics, compact Torch inner two-phase device replay, submitted-update policy-lag proof, eager append pre-drain proof, and same-process async learner proof. The r14 H100 path then won by sharing the inline learner/search model state and eliminating inline host payload clone; later r32-r34 and r3/r4 ring-batch rows superseded it as the active speed candidate. | r14 row path `artifacts/local/curvytron_compact_coach_speed_row_results/opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-warmupgate-b1024a1-normal-unroll2-r14-20260605/row_001_result.json`; ring-batch r3/r4 artifacts in the measurement ledger. |
| P0 | owner-proof | Make owner-search action-only proof H100-ready before scale. | closed-local-proof | Already-observable gates now pass locally and fail closed: requested-vs-actual cadence, owner sample telemetry, zero-batch sample/target equality, finite/nonnegative final drain timing, train timing split fields, required action-only/two-phase handle fields, action-feedback checksums, explicit staged-work/drained-work/drained-replay-entry/drained-append accounting, and submitted-update policy-lag overlap proof. The temporary action-only request==entry shortcut is removed. | `opt132-local-owner-action-only-explicit-drain-work-entry-proof-20260604`: append requests/submitted/appended `15/15/15`, staged/drained work `15/15`, drained entries/appends `15/15`, pending/inflight/failed `0/false/false`; lag-proof row `opt132-local-owner-action-only-inner2-slab-lagproof-steps16-cadence8-b512-nodeath-r1-20260604`: submitted/owner/refreshed updates `2/2/2`, policy lag current/max `0/2`, actions while policy-lagged `15`; focused tests and ruff pass. |
| P0 | owner-timing | Change the real owner learner-update resource/work shape. | superseded-by-r14 / keep-as-falsified-lanes | Nested learner timing showed compact MuZero learner work was only `0.055s` while the owner update wrapper reported `0.855s`; moving LightZero learner-function imports into learner construction removed the measured first-call import/setup cost. Cadence-8 confirmed the post-prewarm tail follows train request count. The real normal-death scale rows then failed speed shape: slab/threaded local scale both spend about `14-15s` in train/learner update and about `12-13s` in final drain. Mock-fast ceiling passed at `214.06 env steps/sec` with final drain `0.0026s`. Local MPS placement, eager append, and same-process async learner overlap all failed. R14 supersedes this as the active P0 by avoiding model-state transport and host payload clone in the inline path. | Keep these as guardrails against reopening failed lanes: CPU local `25.78/25.09`, MPS `20.81`, eager append `24.09`, async max1 `24.47`, async max6 `25.37`; r14 H100 shared/no-clone `13497.30`, final drain `0.000042s`, digest/state_dict/host_clone `0/0/0`. |
| P0 | proof/plumbing | Keep OPT-132BA as a proof/plumbing lane, not the strategy. | closed | Local learner-ready unroll-2 cache proof passed; accepted-preset smoke emits cache requested/available/eligible/used/call/fallback/path proof fields before any H100-only contradiction row. | OPT-132BD H100 proof passed learner-ready/tensor-native cache fields; remaining failure is speed architecture. |
| P0 | measurement | Run exact `1084/270` only after owner-search scale has plausible speed shape. | gated | Same code/flags/shape/seeds/proof fields repeat with wall spread `<= 5%` and major bucket spread `<= 10%`; no H100 row is launched just to add attribution, replay proof, or learner-tail confirmation. | `goal.md` H100 Gate |
| P0 | docs/control | Keep `goal.md` as a compass, not a ledger. | active | `goal.md` stays short enough to answer: goal, current truth, why we floundered, active world model, next three gates, H100 gate. Long run history stays in the measurement ledger/reset docs; stale "next" lists are archived or reduced instead of reactivated. | 2026-06-04 rewrite cut the file down to the active decision and added this guard after sidecar critique. |
| P0 | evidence | Keep evidence honest. | open | Summary, Modal report, hash-bound evidence, and comparison artifacts all carry the same repeatability/proof fields; validation fails on mismatch. | speed-row report/comparator guards |
| P1 | research | Keep blind ports parked, but study mature systems. | active-pattern-known | No active task defaults to GPU mechanics, H200/B200, multi-GPU, compile-default, or sample-builder polish. PufferLib/EnvPool/Sample Factory/Isaac-style env systems and MCTX/JAX-style search stay bounded architecture research and toy-ceiling lanes. | Erdos/Cicero sidecars confirmed the mature pattern is ownership by fixed-shape data surface: env/vector actors -> batched search/inference -> contiguous replay/trajectory tables -> learner gather/update. Trap to avoid: framework or GPU-resident theater while rich Python objects remain the hot-loop handoff. |

## 2026-06-09 Production Fixed-SoA Handle Whole-Loop Followups

Main-thread closure:

- Added owner-search expected train-count accounting based on replayable
  submitted transitions, including warmup-tail handling and transition-batch
  truncation.
- Fixed terminal-row sampling so normal-death proof forces an actual terminal
  sample row when terminal rows are available, not merely any terminal window.
- Passed two local production owner-search fixed-SoA handle rows:
  `opt132-local-production-fixed-soa-handle-normaldeath-b128s64-r2-20260609`
  and `opt132-local-production-fixed-soa-handle-opt107-envelope-r2-20260609`.

Do next:

- Treat the production fixed-SoA handle path as locally contract-clean.
- Do not spend more local cycles proving the same cadence, terminal-row, or
  resident-handle facts.
- Next implementation must target a broader Amdahl surface: owner actor/search
  transaction wait, learner publication/version tickets, or mechanics/
  observation fixed-buffer ownership.
- Use H100 only for a candidate plausibly clearing the `2x` wall gap, or for
  one explicitly labeled support row that decides the broader lane.

Stop-list:

- Do not repeat unchanged fixed-SoA locality, slot-candidate, or handle-ring
  rows.
- Do not treat local CPU env/s as speed evidence.
- Do not edit `goal.md` for this support closure.

## 2026-06-09 Owner-Manager Boundary Followup

New guard:

- Added a production-manager boundary test that poisons parent compact-batch,
  root-batch, root-request, dense-action, and parent transition-closure
  materialization while running the compact owner action/mechanics path through
  owner replay append and owner train.
- The first failing version was informative, not a speed failure: the test used
  a transition-batch capacity of four with only two closed transitions, so the
  fixed batch correctly did not flush. The passing guard uses capacity two and
  proves owner replay/train submission without parent materialization.

Do next:

- Treat replay/train ownership through the manager as a contract proof, not a
  new speed lane.
- The broader speed lane must move the schedule: submit owner search earlier,
  overlap owner transition/maintenance with real parent work, or move
  mechanics/observation into an owner-buffer pipeline. Replay-only ownership is
  already too small by Amdahl.
