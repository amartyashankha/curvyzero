# Optimizer Task Board

Date: 2026-06-09

This file only tracks live work. History lives in the ledger and artifacts.

## Active

| ID | Priority | Task | Closing Evidence |
| --- | --- | --- | --- |
| OPT-132-OWNER-BATCH-HANDLES | P0 / active next implementation | Promote the local owner-slot device-row/unroll-2 proof into production fixed resident replay/sample/learner-batch handles. The local fixture now closes mechanics/root/action feedback, stages previous transitions through `stage_replay_append_entries()`, validates replay-payload handles/digests, materializes owner-action-context host rows and device replay index rows, drains staged owner records into `_CompactReplayRingV1`, samples unroll-1 through the resident ring path with device-row metadata, and builds a resident device learner unroll-2 batch on a nonterminal owner-slot fixture. The production speed-row path now requests fixed-SoA learner-batch handle-ring sampling when fixed-SoA replay is requested, and the compact-owned learner boundary records whether the resident handle was consumed or fell back to materialized-parent training. | Local fixture evidence: `/private/tmp/curvy_owner_slot_device_rows_unroll2_b2_m3_w1_20260609.json` passed with owner-slot failures `[]`, stage device builds/rows `3/12`, real replay-ring append calls/records/rows `3/3/12`, device row sample `true`, resident device sample `true`, learner unroll-2 built `true`, learner rows `8`, action/reward/value/policy shapes `[8,2]` / `[8,2]` / `[8,3]` / `[8,3,3]`, host fallback `false`, parent replay objects / replay object entries / selected groups `0/0/0`, and production speed claim `false`. Production-facing handle proof: speed-row metadata requests fixed-SoA handle ring; owned-loop green path records resident-handle consumed with fallback `0`; fallback path records consumed `false` and fallback `1`; real ring handle path remains green. Validation: ruff passed; replay-contract plus benchmark tests `57 passed`; source-state hybrid selector `1 passed`; combined production-handle validation `28 passed`. Closing evidence requires corrected local whole-loop timing with resident-handle consumed true and materialized-parent fallback zero before any H100 launch. |
| OPT-132-OWNER-STEP-FRAME | Closed-local / support | Mechanics/action/pending/returned-root/closure/action-dispatch rungs are proof support only. The guarded overlap/proxy H100 read already ran and was speed-rejected versus columnar r2, so do not repeat overlap unchanged or treat it as the next active task. | H100 overlap/proxy r3 was proof-clean but slower than columnar r2 (`15541.95 env/s`, `47.7016s`, `0.980x` columnar r2). Preserve its fail-closed counters as support for production owner handles. |
| OPT-132-OWNER-BOUNDARY | Closed / support | Owner-local transition derivation completed its H100 decision row and is support-only. Replay-only, fixed-SoA/locality, one-simulation deferral, in-process async learner, root-build-request alone, fixed action-result slot, overlap alone, and owner-local derivation alone are all support/rejected as standalone paths. | Owner-local r4 passed proof but ran only `13265.51 env/s`, `55.8875s`: `1.045x` OPT-104 and `0.837x` columnar r2. Keep the counters inside broader owner-buffer work. |
| OPT-132-MINIMAL-STEP-PAYLOAD | Closed-local / support | Preserve default-off minimal step payload snapshot only as a guard/support cut under the owner action-step boundary. It is not the final owner step-frame ring and not an H100 lane. | `compact_owner_minimal_step_payload_snapshot` now requires `compact_owner_action_step_boundary`, keeps only summary/action payload keys, and reports measured-step count, full payload bytes/key count elided, retained key count, and snapshot timing. Direct-root/threaded boundary test passes with this flag while preserving parent builder `0`, resident-root/stub proof, search payload bytes `0`, parent rows `0/0`, and action checksums. Focused tests `4 passed`; ruff passed. |
| OPT-132-OWNER-ACTION-STEP-BOUNDARY | Closed-local / support | `compact_owner_action_step_boundary` is now bound to the direct-root/action-only owner proof path. It is a guard for the broader owner-boundary rewrite, not a speed lane by itself. | Local profile proof shows no scalar timestep, search-feedback slab mode, `enabled=true`, `proof_passed=true`, counts equal total iterations, seeded count `1`, feedback count `total-1`, failure reason `none`, and clean action checksums. Speed-row proof rejects missing proof, parent replay rows, search payload bytes, parent root-builder fallback, host fallback, action mismatch, owner-local fallback/pending transitions, and fake terminal samples. New direct-root threaded proof `test_hybrid_profile_owner_action_step_boundary_uses_direct_root_build_request` monkeypatches legacy parent root-batch building to raise and passes with root-build-request owner handoff, resident-root/stub proof, parent builder `0`, search payload bytes `0`, parent rows `0/0`, and action-feedback checksums. Focused boundary tests `3 passed`; ruff passed. |
| OPT-132-FIXED-SLOT | Closed / support | Preserve fixed action-result slot as a support mechanism only. | Local proof and H100 proof passed. H100 `opt132-h100-fixed-action-result-slot-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608` ran `12794.42 env/s`, with fixed-slot `904/904/904`, pending `0`, wire/full `414/4837`. It improved root-build-request but lost to columnar r2; do not repeat unchanged. |
| OPT-132-ONE-SIM-DEFER | Closed / support | Preserve the default-off one-simulation deferral mechanics, but do not spend more H100 rows on it unchanged. | Local r7/r8 passed and the H100 probe kept proof closed, but speed was rejected: `13691.98 env/s`, `54.1467s`, below columnar r2 `15852.67 env/s`; direct append rose to `22.377s` with deferred/device flush `3.613s/4.381s`. |
| OPT-132-ASYNC-LEARNER | Closed / support | Preserve the in-process async learner worker as a proof/diagnostic component only. | Local and H100 probes proved async submit/completed/pending closure, but the H100 row was speed-rejected: `12954.74 env/s`, `57.2282s`, slower than columnar r2. Do not repeat unchanged. |
| OPT-132-FIXBUF | Support / implementation component | Fixed owner-buffer storage remains useful only as part of broader owner-boundary work. | Fixed-SoA exact/locality and whole-owner-buffer projection show replay/sample changes alone are not enough. Use fixed buffers when they also remove parent wait, search/root dispatch, learner-publication/update, mechanics/observation, or other measured non-replay surfaces. |
| OPT-132-CANDIDATE-SHAPE | Preserve | Keep the fastest known support stack intact while testing any new patch. | Preserve vectorized reset RNG, refresh interval `4`, owner-search threaded/background maintenance, slab bypass, transition-batch direct replay, ring-batched append/cache refresh, direct table/columnar support where requested, fused learner batch, learner-ready unroll-2 cache, tensor-native replay, borrowed render state, direct core, and normal death proof. |
| OPT-132-SUPPORT-LEDGER | Support / not active tasks | Keep direct-table, columnar, ring-batch, direct-prebuilt, fixed-SoA exact/locality, and inline-background results as evidence, not active P0s. | These lanes are either support or rejected in the ledger: columnar r2 is fastest single input (`15852.67 env/s`) but only about `1.25x`; fixed-SoA exact r8 is proof-clean but below OPT-104; locality proves fragmentation but regresses; lazy selected and inline-background are rejected as main lanes. |

## Current Facts

- Baseline: OPT-104, H100, `12689.38` env steps/sec, `14.5255s` wall.
- Current fastest single same-window candidate is threaded/background owner
  maintenance plus vectorized reset RNG, transition-batch slab bypass,
  ring-batched owner replay append/cache refresh, tensor-native maintained
  replay, direct record-table building, and columnar direct append:
  `15852.67` env steps/sec, `46.7666s` wall. This is real progress over
  OPT-104, but it is not repeated and still not the near-target `2x` or `10x`.
- Owner-local transition derivation is proof-clean but speed-rejected as a
  standalone lane. Valid H100 r4 ran `13265.51 env/s`, `55.8875s`, with derived
  schema/kind, `724/0` cache hits/misses, `724/0` checksum verified/mismatch,
  parent outcome bytes/fields `0/0`, and normal-death/tensor-native gates
  closed. It is `1.045x` OPT-104 but only `0.837x` columnar r2.
- Exact fixed SoA is not the speed win yet. The best proof-clean exact row is r8
  `11616.45 env/s` versus OPT-104 `12689.38`; the evidence points to learner
  batch scatter (`~373` selected records for a `512` row sample), not variance
  or the need for a longer unchanged run.
- The H100 locality probe confirms scatter but rejects locality alone:
  `10428.59 env/s`, `71.0907s` wall, selected records `62`, semantic drift
  true. Use it as causal evidence only.
- The whole-owner-buffer replay ceiling instrument is local and guarded, and
  the compare script now derives it for existing H100 artifacts. The generated
  ceiling review projects the fastest columnar/direct-table r2 timings to
  `24903.25 env/s` (`1.9625x` OPT-104), still `0.558s` short of `2x` for
  `741376` env steps. Replay layout alone therefore needs either real overlap
  proof or a paired attack on owner search/root dispatch, parent wait, learner
  update/publication, or another measured surface.
- The one permitted H100 deferral probe ran and lost speed despite clean proof:
  `opt132-h100-owner-deferred-one-sim-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607`
  produced `13691.98 env/s`, `54.1467s`, direct
  flush/materialized/identity/recurrent `724/724/724/724`, crossed `0`, D2H
  `0.0`, owner-inner pending final `0`. Do not repeat this lane unchanged.
- The in-process async learner overlap probe also ran and lost speed despite
  clean proof:
  `opt132-h100-owner-asynclearner-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607`
  produced `12954.74 env/s`, `57.2282s`; async submit/completed/pending
  `90/90/0`, max pending `2`, actions while async pending `510`, failed
  `false`. Do not repeat this lane unchanged.
- Local resident-root-view proof for direct-root threaded owner mode is now
  implemented and smoke-tested:
  `opt132-local-owner-resident-root-view-threaded-directtransition-smoke-20260607-r5`
  passed with resident-root-view required/proved true, kind
  `direct_root_batch_resident_handle_v1`, direct root publish/resolve `15/15`,
  H2D/D2H `0.0/0.0`, host fallback false, action-only true, parent rows
  `0/0`, search payload/visit/root-value bytes `0/0/0`, transition
  entries/batches/transport `12/3/3`, and final policy lag/pending/failed
  `0/0/false`. This is local proof only; it does not yet remove parent
  root-batch construction or prove H100 speed.
- Local resident host-observation stub proof is also implemented for the same
  direct-root threaded owner path:
  `opt132-local-owner-resident-root-hoststub-threaded-directtransition-smoke-20260607-r1`
  passed with stub requested/stubbed true, kind
  `zero_stride_shape_only_v1`, materialized bytes `0`, logical bytes `262144`
  in the last step and `3932160` total, resident-root-view required/proved
  true, H2D/D2H `0.0/0.0`, host fallback false, action-only true, parent rows
  `0/0`, search payload bytes `0`, transition entries/batches/transport
  `12/3/3`, and final policy lag/pending/failed `0/0/false`. This is still a
  local proof gate, not H100 speed evidence: parent root-batch construction is
  still called, but it no longer materializes the resident host observation
  payload in this gated path.
- Local direct-root build-request proof is now implemented and locally
  speed-row-guarded:
  focused tests prove `CompactOwnerSearchDirectStepperV1(direct_root_build_request=True)`
  can complete when the legacy parent `build_compact_root_batch_v1` symbol is
  monkeypatched to raise. The real local threaded owner proxy also completes
  through `run_action_step_from_root_build_request` under the same parent
  builder monkeypatch. The owner-search service suite passes (`45 passed`).
  Local smoke
  `opt132-local-owner-rootbuildrequest-threaded-directtransition-smoke-20260607-r3`
  passed with publish/resolve/owner-build `15/15/15`, parent builder
  used/calls `false/0`, parent build sec `0.0`, request observation bytes `0`,
  resident-root/stub/action-only gates preserved, parent rows `0/0`, search
  payload bytes `0`, direct replay batches/transitions/transport `3/12/3`,
  action mismatches `0`, and final maintenance/policy lag `0/0`. This is local
  proof only. H100
  `opt132-h100-rootbuildrequest-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607`
  also passed proof but failed speed at `11327.75 env/s`, `65.4477s`
  (`0.893x` OPT-104, `0.715x` columnar r2). Do not repeat root-build-request
  unchanged.
- New local implementation `owner_search_inline_background_proxy` is a tactical
  discriminator, not a proven speed win: it keeps inline direct action while
  draining owner maintenance on a background thread. H100 answered it: r36/r37
  were positive short rows, but r38-long failed speed, so stop this
  worker-family polish.
- The next implementation target is the owner-resident root/mechanics/search
  dispatch and fixed-buffer ownership boundary. Exact-repeat the direct
  table-builder or columnar row only when making a speed claim or isolating a
  suspected regression. Candidate alternatives still come from remaining
  measured surfaces: actor/autoreset/observation, owner dispatch residual,
  replay append/materialization, search batching, or learner
  publication/update cadence. Do not rerun the same candidate unchanged as the
  next move.
- The direct owner-search slab-bypass branch now has a local coalesced
  transition-batch proof: `12` logical replay transitions crossed the
  parent/owner boundary as `3` transport entries with batch size `4`, parent
  replay rows `0/0`, pending replay `0`, and action-feedback mismatches `0`.
  H100 r2 confirmed the same invariant at `724 -> 181` transport entries, but
  whole-row speed stayed flat at `13618.91 env/s`. This proves transport count
  alone is not the 2x blocker.
- Direct table-building is the latest post-r2 patch that moved the target
  surface on H100. It reduced whole-row wall to `47.2768s`, with nested owner
  learner proof `direct_record_table_v1`, table build `0.637s`, worker replay
  append `13.879s`, ring append `9.848s`, and `ring_entry_objects=724`.
  Ring-batched append/cache refresh remains the support layer underneath it:
  replay append previously fell from r2 `31.926s` to r3/r4 average `19.420s`.
- Columnar append is the latest valid H100 increment. It removes
  `_CompactReplayRingEntry` objects (`0`) and keeps direct prebuilt sample group
  objects at `0`, but whole-row speed only moves to `15852.67 env/s`. The
  active target is therefore not a ring-entry-only counter; it is a fixed-buffer
  replay/table path that avoids append records/views, cache/table maintenance
  churn, and sample-side group-object rebuilds without moving cost into learner
  train.
- Lazy selected direct table build is rejected for now: it improved append
  timing but regressed whole-row wall to `66.3171s` and reintroduced `352`
  direct group objects.
- Inline refresh4 is parked as unstable after r31. Reset RNG itself is
  preserved; actor autoreset stayed low in r31 while slab regressed.
- r14 beat the baseline once on H100 with the owner-search shared/no-clone path:
  `13497.30` env steps/sec, `13.6561s` wall. r15 exact repeat was slow:
  `10502.70` env steps/sec, `17.5498s`. This is a first win, not a stable
  baseline.
- r16 longer same-mechanism row was slow and exposed steady-state owner train
  sample cost: `52.807s` sample vs `3.236s` update across `90` train requests.
- r17 failed before row creation because the existing fused/tensor-native
  learner-batch flags require `compact_owned_loop_entrypoint`; they are not
  currently wired into owner-search inline.
- r14 keeps train/sample/update work at `22` and refreshes more often
  (`policy_refresh_interval=1`, `22` refreshes, `0` skips). If a strict
  cadence objection matters, run the same shared/no-clone path at interval `4`.
- The live task has moved beyond post-direct-table-builder materialization
  removal and beyond root-build-request, fixed-slot, and owner-local derivation
  as standalone lanes. Fixed SoA/ring-buffer replay storage remains support
  evidence, but exact/locality rows showed it is not the standalone 2x path. The
  active implementation target is a broader owner root/search/parent-wait/
  mechanics/observation/learner-publication or fixed owner-buffer patch. The
  pre-r14 "change learner resource/work shape" and r14 repeat lanes are
  historical unless new evidence points back there.
- OPT-132-G beat the baseline once: `13649.29` env steps/sec, `13.504s`
  wall, accepted proof passed.
- OPT-132-H immediately failed to confirm it: `11366.84` env steps/sec,
  `16.216s` wall, accepted proof passed.
- OPT-132-I beat the baseline again: `13308.59` env steps/sec, `13.8497s`
  wall. It validates under the corrected signed-checksum guard.
- OPT-132-J passed the corrected guard but was slow: `10057.98` env steps/sec,
  `18.3258s` wall. Its trajectory and sample-order checksums match OPT-132-I.
- The speed goal is therefore not closed yet. OPT-104 remains the accepted
  baseline until the win repeats without the OPT-132-H/J-style slow swing.
- OPT-132-F carried the accepted fast-path preset and still lost:
  `12177.36` env steps/sec, `15.136s` wall.
- OPT-132-F improved wall by about `0.833s` versus OPT-132-E by storing
  resident terminal final observations sparsely.
- OPT-132-B is the best recent recovery: `11609.78` env steps/sec,
  `15.876s` wall.
- Restoring borrowed render state and lean trainer step recovered a large
  regression, but did not reproduce OPT-104/109/117/118.
- The accepted-fast-path launcher now validates the returned remote result and
  fails if the actual row does not match the preset.
- The repeatability guard now treats checksums as signed: required checksums
  must be nonzero; counters must be positive.
- Speed-row reports/evidence now expose actor/observation child timers.
- Speed-row reports/evidence now expose resident observation stack timing.
- Resident stack shift is tiny: `0.012s` in OPT-132-G and `0.015s` in
  OPT-132-H. Do not build the ring-buffer stack now.
- Historical pre-r29 target was wiring the fast learner-ready/tensor-native
  batch path or equivalent owner-side sample path into owner-search. The active
  target is now the r32/r33/r34 threaded/background candidate plus the next
  measured headroom patch.
  OPT-132BD proved maintained
  tensor-native replay on H100
  (`maintained_record_table_v1`, reused `1080`, missing `0`, fallback `0`) but
  was slow: `3884.88 env steps/sec`, `285.727s` wall. OPT-132BF then proved
  maintained sample-universe sampling and cut sample gate from `74.005s` to
  `4.116s`, but full-loop speed was still only `5362.68 env steps/sec`.
  Remaining costs are outside sample gate: primary residual `122.679s`, actor
  step wall `41.852s`, actor/autoreset `29.656s`, search dispatch `14.248s`,
  observation `11.423s`, env runtime `10.000s`.
- OPT-132BK local closed-loop slab/replay/sample proof passes with rendered
  observations, deterministic slab-selected action feedback, replay appends
  `3`, committed index rows `28`, sample gates `3`, sample batch size `8`,
  `observation_materialized=false`, no retained committed index rows in the
  slab, and compact/direct metadata equality. This closes the toy slab gate
  locally; it is not H100 speed evidence.
- Owner-search action-only local real-entrypoint proof passes:
  `action_only=true`, parent committed/stored rows `0/0`, zero search payload
  bytes, replay entries/submitted/append `15/15/15`, train requests `3`,
  learner updates `3`, drain requests `2`, drained `15`, pending/inflight/
  failed `0/false/false`, final owner drain inside measured wall. Local speed
  `70.93 env steps/sec` is not speed evidence. Later proof-hardening closed
  the permissive fields; the remaining blocker is real owner learner-update
  placement/cost.
- Hardened owner-search action-only local smoke also passes:
  `opt132-local-owner-action-only-proof-hardened-fields-20260604` keeps
  action-only `true`, parent rows `0/0`, zero search payload bytes, sample
  batch/requested/sample/target `0/0/160/160`, replay submitted/requested/
  append `15/15/15`, train/update `3/3`, train wall `0.856s`, learner update
  `0.845s`, final drain `0.778s`, and final drain inside measured wall. This
  closes the cadence/sample/final-drain/handle gate locally, not the speed
  problem.
- Action-feedback owner-search local smoke also passes:
  `opt132-local-owner-action-only-action-feedback-proof-20260604` verifies
  selected search actions become applied/replay actions with transitions/actions/
  mismatches `15/240/0` and expected/applied/replay checksums
  `6120/6120/6120`. Train wall remains `0.825s`, learner update `0.814s`, and
  final drain `0.761s`, so proof improved but speed did not.
- Nested learner timing and import prewarm changed the local bottleneck:
  `opt132-local-owner-action-only-learner-nested-aggregate-proof-20260604`
  showed train wall/update/final drain `0.865s/0.855s/0.790s`, but aggregated
  compact MuZero learner time only `0.055s`. After moving LightZero
  learner-function imports into `CompactMuZeroLearnerEdgeV1` construction,
  `opt132-local-owner-action-only-learner-import-prewarm-proof-20260604`
  drops train wall/update/final drain to `0.072s/0.057s/0.043s`, with nested
  MuZero `0.057s`. This is local bottleneck speedup, not H100 evidence.
  Cadence-8 falsifier `opt132-local-owner-action-only-prewarm-cadence8-
  falsifier-20260604` drops train requests to `1` and train wall/update/final
  drain to `0.035s/0.030s/0.019s`, confirming the post-prewarm tail now follows
  train cadence.
- Explicit drain-work/entry owner-search local smoke also passes:
  `opt132-local-owner-action-only-explicit-drain-work-entry-proof-20260604`
  removes the stale action-only append-request/submitted-entry equality
  shortcut and proves append requests/submitted/appended `15/15/15`,
  maintenance staged/drained work items `15/15`, drained replay entries/appends
  `15/15`, pending/inflight/failed `0/false/false`, parent rows `0`, payload
  bytes `0`, action feedback true, train wall `0.077s`, and final drain
  `0.047s`.
- Inner two-phase owner-search proof also passes:
  `opt132-local-owner-action-only-inner-two-phase-explicit-drain-proof-20260604`
  proves compact Torch owner-search uses the inner two-phase/device replay
  path, with parent rows `0`, payload bytes `0`, train wall/update/final drain
  `0.065s/0.054s/0.037s`.
- Post-prewarm normal-death owner-search scale fails speed shape:
  `opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-20260604`
  preserves proof with parent rows `0`, payload bytes `0`, staged/drained work
  `55/55`, drained replay entries/appends `55/55`, but speed is only
  `25.78 env steps/sec` local and train/update/final drain are
  `14.23s/14.07s/11.97s`.
  `opt132-local-owner-action-only-inner2-threaded-scale48-cadence8-b512-normal-20260604`
  is the same story: speed `25.09`, train/update/final drain
  `14.94s/14.77s/13.00s`. Proof improved; speed did not.
- Mock-fast normal-death owner-search ceiling passes:
  `opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-mockfast-r3-20260604`
  keeps normal-death contract proof true, carries owner learner
  value-valid/done/truncated counters, parent rows `0`, payload bytes `0`,
  staged/drained work `55/55`, drained replay entries/appends `55/55`, and
  runs `214.06 env steps/sec` locally. Train wall/update/final drain are
  `0.091s/0.0s/0.0026s`. Decision: owner-search/search/replay mechanics are
  not the local scale blocker after neural update is removed.
- Local MPS learner placement is not enough:
  `opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-mps-r1-20260604`
  preserves normal-death/owner-search proof but slows to `20.81 env steps/sec`
  versus CPU `25.78`. MPS cuts learner update to `8.08s`, but sample/build,
  replay append, digest, and final-drain costs keep the whole row slow. Decision:
  placement alone is falsified locally. This old conclusion is superseded by
  r14; do not implement a learner owner or overlap boundary unless the r14
  repeat fails and points back to that surface.
- Owner-search submitted-update lag proof now passes:
  `opt132-local-owner-action-only-inner2-slab-lagproof-steps16-cadence8-b512-nodeath-r1-20260604`
  proves submitted/owner/refreshed learner updates `2/2/2`, final policy lag
  `0`, max lag `2`, actions while policy-lagged `15`, and final pending/inflight
  `0/false`. Future async rows must preserve this shape.
- Eager append pre-drain is active but speed-failed:
  `opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-eagerappend-r1-20260604`
  preserves proof with eager append drains `7`, coalesced skips `0`,
  staged/drained work `55/55`, submitted/owner/refreshed learner updates
  `6/6/6`, policy lag current/max `0/6`, and actions while lagged `47`, but
  speed is only `24.09 env steps/sec`; train/update/final drain are
  `15.31s/15.12s/12.67s`. Decision: stop scheduler polish as the P0 move.
- Owner maintenance timing split and nested learner aggregation now explain the
  coarse train bucket:
  aggregate local proof `opt132-local-owner-action-only-train-split-aggregate-
  proof-20260604` maps worker learner train `0.874s` to train wall `0.874s`,
  learner update `0.861s`, sample `0.012s`, and negligible model digest/state/
  ref publication. Threaded/background owner-search proof passes and improves
  short local wall to `0.861s`, but final drain remains `0.798s`; scale-48
  serves 51 actions while maintenance is pending and still ends with final
  drain `1.312s` and learner update `1.410s` across 13 train calls. Nested
  aggregation then showed most local update wall was first-call import/setup,
  and import prewarm drops the short local final drain to `0.043s`. The
  post-prewarm normal-death scale rows now prove the remaining local tail is
  real learner/update placement, and the mock-fast ceiling confirms
  owner-search/search/replay mechanics are fast enough locally without neural
  update. Next speed work is moving, overlapping, or isolating the real learner
  update.
- The five-row `724/180` packet failed stability despite exact work identity.
- The latest `1084/270` r1/r2 pair now fits on H100 with latest-frame resident
  replay snapshots, but timing still failed: wall spread `8.28%`, sample gate
  spread `14.04%`, learner-batch build spread `15.57%`.
- Three `1084/270` rows with learner-batch sub-timers passed with exact
  identity, but r3 broke timing stability: wall `15.55%`, sample gate `19.49%`,
  learner-batch build `18.88%`. They are diagnostic only, not speed evidence.
- OPT-132Y r1/r2/r3 ran exact `1084/270` with
  `--compact-profile-cuda-sync-timing-diagnostics`. Identity matched exactly,
  accepted-fast-path violations were `[]`, and sync diagnostic violations were
  `[]`. R3 was a broad slow repeat: wall spread `30.99%`, sample gate `30.47%`,
  learner-batch build `28.01%`, actor wall `31.66%`, observation `62.14%`,
  and builder cuda_sync `39.85%`. Historical pre-AF gate: no speed code was
  allowed until the broad runtime slowdown was bounded.
- Superseded P0 gate: the AF through AZ diagnostics bounded the swing enough
  to stop pure attribution. BA's guarded unroll-2 cache is a useful proof lane,
  but it is not the whole architecture answer.
- Runtime-step envelope diagnostics are now wired locally for CUDA-sync timing
  rows. Historical Z/AJ gates used them; future exact rows are promotion or
  contradiction rows after toy proof, not the next exploration step.
- OPT-132Z r1/r2 inspected the runtime envelope. Exact identity held, but wall
  spread was `32.50%`; runtime-step sum spread matched wall; sample gate spread
  was `42.67%`; learner-batch build spread was `44.14%`. Runtime p50 barely
  moved while p95/max moved hard, and the slowest steps were sample-gate
  dominated.
- Sample-gate per-call child distribution stats are now wired locally and
  validated.
- OPT-132AA r1/r2 read those per-call fields on H100 with exact identity and
  violations `[]`. The AA pair had wall spread `4.96%`, sample gate spread
  `4.60%`, and learner-batch build spread `2.88%`. Sample-gate per-call
  p50/p95/max was high in both rows (`1.032/1.630/1.723s` and
  `1.061/1.674/1.783s`); learner-batch-build per-call p50/p95/max was
  `0.652/0.804/1.094s` and `0.666/0.802/1.100s`. Historical next target was
  builder-child per-call attribution; AB completed it.
- Builder-child per-call attribution is now wired locally for learner-batch
  builder group-loop, terminal-metadata, unroll-fields, write-output,
  order-restore, finalize-outputs, metadata-sync, metadata-build, and
  builder-cuda-sync timers. Focused ruff and pytest passed.
- OPT-132AB r1/r2 read those builder-child fields on H100. Exact identity held
  and proof violations stayed `[]`, but timing failed hard: wall spread
  `45.57%`, sample gate `54.93%`, learner-batch build `56.09%`, and builder
  group-loop `56.27%`. The visible child swing is dominated by unroll fields,
  then terminal metadata. No speed claim; no blind builder patch.
- OPT-132AC deep builder group-loop attribution is wired locally, validated, and
  read on H100. It splits terminal metadata into mask/tensor-fallback/validate/
  final-observation timers and unroll work into terminal-window-hint/identity/
  stack-fields/mask-build/terminal-value/mask-apply/action-stack timers, with
  totals and per-call stats projected through reports and comparison. The first
  two H100 launches failed before row creation with Modal `RESOURCE_EXHAUSTED`,
  then r1c/r2/r3 completed with exact identity and violations `[]`. Timing
  still failed: wall `36.55%`, sample gate `37.32%`, learner-batch build
  `34.23%`, builder group-loop `34.11%`, unroll fields `40.16%`, and terminal
  metadata `27.48%`. R2/R3 clustered while r1c was slow. No speed claim; no
  blind builder patch.
- OPT-132AD runtime phase/residual attribution completed r1/r2/r3 on H100 with
  exact identity and violations `[]`, but still failed stability: wall
  `24.11%`, sample gate `30.48%`, learner-batch build `34.32%`,
  sample-gate builder group-loop `34.49%`, and unroll fields `36.13%`.
  Diagnostic only; no speed claim.
- OPT-132AE builder group-loop accounted/residual attribution is wired,
  validated, and read on H100. It adds group-loop accounted/residual totals and
  per-call stats, where accounted is terminal metadata + terminal-window hint +
  unroll fields + write output. Ruff, focused pytest `4 passed`, and broader
  speed-row/report slice `63 passed`. AE r1/r2 exact H100 identity held with
  violations `[]`: wall `271.920s -> 284.561s` (`4.54%`), group-loop
  `113.971s -> 119.853s`, accounted child work `95.872s -> 101.300s`, residual
  `18.099s -> 18.553s`. Both rows are slow; unroll fields are the largest
  absolute child and terminal metadata is next. Diagnostic only; no speed claim.
- OPT-132AF guarded unroll-2 specialization is now wired locally and read on
  H100 r1/r2/r3. It is limited to fused resident grouped learner batches,
  preserves learner-batch tensor parity locally, and emits requested/eligible/
  used/call/fallback/reason/impl/path proof fields. H100 r1/r2/r3 exact identity
  held with accepted-fast-path, CUDA-sync, and unroll2 violations `[]`; proof
  used eligible/call_count `399`, fallback_count `0`, fallback_reason `none`,
  impl `unroll2_specialized_v1`, and path `unroll2_specialized`. Wall slowed
  `211.683s -> 225.020s -> 234.042s` (`9.94%`), sample gate spread `12.99%`,
  and builder group-loop spread `14.03%`. AF is rejected as a stable speed
  claim; its "explain before patch" gate is superseded by AZ's structural
  builder-surface decision.
- OPT-132AG reran AF with GPU sampling. Exact identity and clean proof held,
  but stability still failed and the monotonic AF slowdown did not repeat.
- OPT-132AH reran exact `1084/270` with GPU sampling and the generic builder.
  Exact identity held, but timing failed harder: wall spread `34.28%`, sample
  gate `48.55%`, learner-batch build `52.04%`, and builder group-loop
  `52.75%`. This separates the remaining instability from AF specialization.
- OPT-132AI locally decouples runtime-step envelope diagnostics from CUDA-sync
  timing diagnostics. Historical note: AJ later used runtime-step stats without
  sync probes; this is not a current H100 action.
- OPT-132AJ used that runtime-step-only path on H100. Exact identity held and
  CUDA-sync probes were off, but timing still failed: wall spread `10.49%`,
  runtime-step sum `10.49%`, sample gate `12.72%`, learner-batch build
  `14.09%`, and builder group-loop `14.25%`. Diagnostic only.
- OPT-132AK adds local comparator wall-swing attribution and runtime-step
  cadence summaries: sample-gate active/inactive measured-step buckets,
  early/mid/late measured thirds, and bounded top-slowest-step records. The
  refreshed AJ attribution shows r2's slow-fast wall delta is mostly measured
  runtime, with sample gate about `61.00%` and builder group-loop `35.23%` of
  the delta. Tooling only; no speed claim.
- OPT-132AM added local chronological active sample-gate distributions by
  measured third, sample-gate residual bucket sums, enriched top-slowest
  runtime-step records, and slowest per-call iteration projection. Tooling
  only; no speed claim.
- OPT-132AL used the AK cadence fields on H100 with GPU sampling disabled and
  CUDA-sync probes off. Exact identity held with violations `[]`, but timing
  still failed: wall/runtime-step sum `21.70%`, sample gate `22.84%`,
  learner-batch build `20.92%`, and late measured-third sample gate
  `31.11%`. Diagnostic only.
- OPT-132AN used the AM active-call fields on H100 with GPU sampling disabled
  and CUDA-sync probes off. Exact identity held with violations `[]`, but
  timing failed again: wall/runtime-step `21.806%`, sample gate
  `125.169s -> 93.384s`, learner-batch build `68.851s -> 49.643s`, and
  builder group-loop `68.101s -> 48.923s`. Active sample-gate p50/p95 rose
  early/mid/late in both rows and r1 was elevated across all thirds versus r2.
  Diagnostic only; later AO through AZ closed enough of this attribution lane
  to stop pure timing work and move to architecture/toy proof.
- OPT-132AO adds bounded local per-sample-gate trace records with call index,
  measured iteration, measured third, sample seeds/checksums, replay stored/
  eligible/excluded/evicted counts, replay capacity, and per-call timing.
  Tooling only; no speed claim.
- OPT-132AO also read those trace records on H100 r1/r2. Exact identity held
  and violations stayed `[]`, but timing failed: wall/runtime-step `13.944%`,
  sample gate `21.136%`, learner-batch build `24.076%`, and builder group-loop
  about `59.38%` of the slow-fast wall delta. The trace state matched at all
  `135` sample-gate calls, including replay counts and sample checksums.
  Diagnostic only; the next question is timing variance on identical exposed
  replay/sample state.
- OPT-132AP adds per-sample-gate allocator/runtime trace fields below AO's
  exposed replay/sample state and has now been read on H100. AP r1/r2/r3 exact
  generic no-sampler `1084/270` rows kept identity exact and violations `[]`,
  but wall still spread `31.99%` of median
  (`219.446s / 205.371s / 153.752s`). Exposed replay/sample identity and every
  CUDA allocator/memory counter matched exactly across all `135` calls; Python
  GC counts and process RSS varied. Diagnostic only; no speed claim.
- OPT-132AQ added per-sample-gate `gc.get_stats()` collection counters below
  AP and has now been read on H100. AQ r1/r2 kept exact identity and
  violations `[]`, but timing still failed: wall `218.320s / 163.265s`
  (`28.856%`), sample gate `131.651s -> 95.558s`, learner-batch build
  `72.267s -> 51.322s`, and builder group-loop `71.432s -> 50.716s`. Actual
  GC collection totals were nearly identical (`7232/659/28` versus
  `7234/655/28`), and `100` same-GC-delta calls still carried `22.6695s` of
  sample-gate delta. Diagnostic only; no speed claim.
- OPT-132AR adds local process/current-thread CPU-time trace fields for the
  full sample gate and learner-batch-build slice and has now been read on H100.
  AR r1/r2 kept exact identity and violations `[]`, but timing still failed:
  wall `189.915s / 217.364s` (`13.479%`), sample gate
  `111.765s -> 130.614s`, learner-batch build `60.527s -> 71.857s`, and
  builder group-loop `59.815s -> 71.015s`. About `92%` of the sample-gate and
  learner-build wall deltas were backed by process/thread CPU-time deltas, so
  the swing is CPU-time-backed rather than mostly off-CPU wait. Diagnostic
  only; no speed claim.
- OPT-132AS adds `resource.getrusage` trace fields for process/thread
  user/system CPU, page faults, and context switches around sample gate and
  learner-batch build, and has now been read on H100. R1/R2 exact generic
  no-sampler `1084/270` rows kept identity exact and violations `[]`, but
  timing still failed: wall `211.257s / 182.483s` (`14.616%`), sample gate
  `126.934s -> 107.934s`, learner-batch build `70.377s -> 58.933s`, and
  builder group-loop `69.602s -> 58.206s`. The r1-minus-r2 resource split is
  mostly user CPU: sample-gate process user/system `16.17s / 0.49s`; page
  faults and context-switch deltas were zero. Diagnostic only; no speed claim.
- OPT-132AT local CPU perf-stat diagnostics are wired and validated. The new
  `--compact-profile-cpu-perf-stat-diagnostics` flag wraps the remote producer
  in `perf stat -x,`, captures perf stdout/stderr artifacts, parses task-clock,
  cycles/ref-cycles, instructions, branch/cache/LLC/dTLB events, page faults,
  context switches, and CPU migrations into report/result/comparison fields,
  and fails closed if perf is unavailable or denied. This is historical
  diagnostic tooling only; AU closed the external perf-counter lane in the
  current Modal container.
- OPT-132AT H100 attempt proved the current remote speed-row image had no
  `perf` binary: `function_call_id=fc-01KT7YGKDM2Y35NCG6C204TYMY`, problem
  `perf stat diagnostic requested but perf was not found`, return code `127`.
  OPT-132AU added `linux-perf` to `speed_row_image` and completed the bounded
  availability retry.
- OPT-132AU proved external perf counters are unavailable in the current Modal
  container: `/usr/bin/perf` exists, but `sys_perf_event_open()` returns
  `19 (No such device)` for `task-clock`; return code `255`; no speed row.
  Do not retry perf unchanged; OPT-132AV took the in-process follow-up path.
- OPT-132AV added and read in-process builder-child process/thread CPU-time
  deltas on H100. R1/R2 exact generic no-sampler `1084/270` rows kept exact
  identity and report violations `[]`, but stability failed: wall
  `190.101s / 220.991s` (`15.03%`), sample gate `114.928s -> 135.240s`,
  learner-batch build `66.229s -> 82.661s`, and builder group-loop
  `65.505s -> 81.792s`. Group-loop process CPU moved `+14.88s`; terminal
  metadata moved `+6.37s`, terminal final-observation `+2.67s`, residual
  `+4.28s`, and unroll fields `+2.76s`. No speed claim.
- OPT-132AW added local final-observation/residual split diagnostics:
  group-loop prepare, terminal-value bookkeeping, terminal final-observation
  presence/select-current/gather, and final-observation branch/storage proof
  now project through trace/report/comparison surfaces. Ruff passed and focused
  source-profile/smoke/comparator pytest passed (`189` tests, `2` warnings).
- OPT-132AW H100 r1/r2 exact generic no-sampler `1084/270` kept exact identity
  and report violations `[]`, but wall still failed repeatability:
  `250.183s / 232.521s` (`7.318%`). Final-observation branch/storage proof was
  identical: groups `399`, index-fast-path `0`, fallback `399`, final-row
  sum/max `512/4`, sparse storage `399`, sparse-row sum/max `3102/16`.
  Residual CPU was mostly stable (`14.45s / 14.07s`), while unroll-fields CPU
  (`38.18s / 34.98s`) and final-observation gather CPU (`6.84s / 5.95s`) still
  moved. No speed claim.
- OPT-132AX local cleanup replaces the grouped learner discarded
  final-observation materialization with validate-only resident
  final-observation coverage. It leaves sample-path materialization intact,
  adds validate-only/materialized proof counters, adds
  `terminal_metadata_final_observation_validate` timing/CPU/per-call fields,
  and makes terminal-final-observation proof counters optional comparator
  identity fields. Ruff passed and focused source-profile/smoke/comparator
  pytest passed (`190` tests, `2` warnings). H100 r1/r2 then kept exact
  identity and proof but failed timing badly: wall `168.280s / 262.830s`
  (`43.864%`). Validate-only/fallback counts were `399/399`, materialized
  `0`, gather/select-current wall/CPU `0`. No speed claim; historical next
  target was unroll-fields and broader builder group-loop CPU movement.
  Superseded by AZ and the tensor-native replay prototype.
- OPT-132AY added nested process/thread CPU-time fields for the existing
  unroll sub-timers: identity, stack fields, mask build, terminal value, mask
  apply, and action stack. These are attribution fields under `unroll_fields`,
  not new group-loop accounted children. Local validation passed: ruff,
  focused source-profile/smoke/comparator pytest (`190` tests, `2` warnings),
  and a tiny local fused unroll-2 smoke emitted nonzero nested CPU fields.
  H100 r1/r2 exact generic no-sampler `1084/270` kept exact identity and stable
  speed claim false, but wall still failed: `237.752s / 183.386s`
  (`25.819%`). Group-loop process CPU was `85.30s / 62.04s`, unroll-fields
  CPU `43.86s / 34.14s`, and unroll sub-CPU sum `31.96s / 25.14s`. Largest
  unroll subchildren were mask build `10.08s / 8.47s`, terminal value
  `8.51s / 6.55s`, and stack fields `7.76s / 6.29s`. No speed claim.
  Historical attribution target was remaining unroll residual, group-loop
  prepare, terminal metadata, and broad sample-gate CPU; AZ and the
  architecture reset supersede that as active work.
- OPT-132AZ is now the latest H100 read. It split the remaining broad builder
  CPU into group-loop prepare snapshot/index/observation, terminal metadata
  accounted/residual/final-observation storage, and unroll accounted/residual
  plus builder-select/row-index prep. Local ruff and focused pytest passed
  (`190` tests, `2` warnings); the local smoke emitted the new fields. H100
  r1/r2 exact generic no-sampler `1084/270` kept exact identity and stable
  speed claim false, but wall still failed: `217.391s / 180.892s`
  (`18.329%`). Sample gate was `136.622s / 109.260s`, learner-batch build
  `85.001s / 65.049s`, builder group-loop `84.177s / 64.278s`, group-loop
  process CPU `80.37s / 61.70s`, unroll-fields CPU `42.13s / 32.24s`, and
  terminal metadata CPU `15.58s / 11.44s`. Decision: stop adding pure
  attribution timers; choose the next patch through the architecture map and
  toy-ceiling matrix. Builder reshaping is one candidate lane.
- The Modal speed-row launcher now preserves structured artifacts for future
  pre-FunctionCall launch failures: `launch.json` plus
  `compact_coach_speed_row_modal_report.json`, `ok=false`,
  `failure_stage=launch`, and `modal_launch_resource_exhausted` when applicable.
  The Modal entrypoint also prints structured `spawn_failed` JSON for reachable
  remote spawn failures, and local/remote per-call prefix parity is tested. This
  is measurement hygiene, not speed evidence.
- Long-window rows are diagnostics unless a matched long-window baseline is
  also produced; OPT-104 remains the accepted baseline.

## Parked Unless Architecture Proof Reopens Them

- Owner process/inline/threaded boundary variants beyond the measured
  owner-search action-only threaded/background overlap falsifier.
- Rerunning older owner rows unchanged.
- Legacy OPT-124 inner two-phase owner replay as a standalone lane; compact
  Torch owner-search inner two-phase device replay is now wired/proved only as
  part of the selected action-only ownership path.
- More small replay/cache/copy edits.
- Direct autoreset as a speed win.
- GPU utilization sampler in accepted speed rows; labeled diagnostics may opt in.
- Blind GPU mechanics rewrite. GPU-resident mechanics is allowed only as a
  toy-ceiling experiment with fixed-action semantics proof.
- H200/B200 or multi-GPU as speed baselines. H200/B200 are allowed only as
  explicit memory-headroom diagnostics.
- Blind PufferLib port. PufferLib/EnvPool/Sample Factory/Isaac-style systems
  are active references for fixed-buffer/vectorized env architecture.
- Scalar-ref/local-process/snapshot transport.
- Compile default.
- Blind MCTX/backend swap. MCTX/JAX-style search is an active reference for
  batched fixed-shape search architecture.
- Result-payload-format tweaks.

## 2026-06-07 Immediate Board

- P0 complete: owner-boundary deferred one-simulation replay proof. Code now
  carries the compact Torch default-off flag through local/Modal speed-row
  construction, records direct and owner-inner flush-time proof fields, and
  fails closed on non-materialized deferred payloads, model identity drift,
  refresh crossing, nonzero final pending deferred payloads, or hidden replay
  D2H bytes under the requested proof.
- P0 complete: local flagged speed-row smoke
  `opt132-local-owner-deferred-one-sim-direct-transition-batch-smoke-20260607-r7`
  passed with `ok=true`, direct proof counts `12/12/12/12`, crossed refresh
  `0`, replay D2H `0.0`, and owner-inner pending final `0`.
- P0 complete: the one same-work H100 deferral probe ran and was proof-clean
  but speed-rejected: `13691.98 env/s`, `54.1467s`, below columnar r2
  `15852.67 env/s`. Do not repeat this lane unchanged.
- P1 guard: the deferral patch is proof plumbing plus a falsified H100
  speed probe, not speed currency.
- P1 architecture lane: choose the next non-replay owner/root/search surface
  with a toy ceiling or small patch. Candidate surfaces are owner-resident root
  construction, mechanics/search dispatch, parent wait overlap, and learner
  publication/update cadence. Replay/sample-only polishing is parked unless a
  fresh measured row reopens it.
