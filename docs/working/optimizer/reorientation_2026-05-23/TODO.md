# Reorientation TODO

Date: 2026-06-07

Use `goal.md` and `TASK_BOARD.md` for active work. This file only keeps the
next concrete checklist.

## Current Active

- [x] Re-read repo-root `goal.md` and active operating docs. Current target is
  still same-work H100 whole-loop speed, near target `2x`; local proof, mocks,
  and projections are only decision inputs.
- [x] Close the owner-loop proof for default-off one-simulation replay
  deferral. Local r7 passed with direct counts `12/12/12/12`, owner-inner
  counts `16/16/16/16`, refresh-crossed `0`, replay D2H `0.0`, and owner-inner
  pending final `0`.
- [x] Update the active docs so the deferral lane is no longer listed as
  owner-proof-pending and fixed-SoA gather/locality is no longer listed as the
  main P0.
- [x] Decide the next speed-currency move. The one permitted H100 deferral
  probe ran and was proof-clean but speed-rejected:
  `13691.98 env/s`, `54.1467s`, below columnar r2 `15852.67 env/s`.
- [x] Add the H100 probe run-registry row before launch and close it with the
  result after collection.
- [x] Close the in-process async learner overlap probe as proof-clean but
  speed-rejected. H100 r1 produced `12954.74 env/s`, `57.2282s`, with async
  submit/completed/pending `90/90/0`, max pending `2`, and failed `false`;
  do not repeat unchanged.
- [x] Implement the local direct-root resident-root-view proof gate. Local
  smoke `opt132-local-owner-resident-root-view-threaded-directtransition-smoke-20260607-r5`
  passed with resident-root-view required/proved true, H2D/D2H `0.0/0.0`,
  host fallback false, direct root publish/resolve `15/15`, action-only true,
  parent rows `0/0`, payload bytes `0/0/0`, and final owner maintenance closed.
- [x] Implement the local direct-root resident host-observation stub proof
  gate. Local smoke
  `opt132-local-owner-resident-root-hoststub-threaded-directtransition-smoke-20260607-r1`
  passed with stub requested/stubbed true, kind `zero_stride_shape_only_v1`,
  materialized bytes `0`, logical bytes `262144` last step / `3932160` total,
  resident-root-view proof preserved, H2D/D2H `0.0/0.0`, host fallback false,
  parent rows `0/0`, search payload bytes `0`, and final owner maintenance
  closed. This is proof-only; parent root-batch construction still exists.
- [x] Implement the local direct-root build-request core hook. Added
  `CompactRootBuildRequestV1`, owner-side root-batch construction from a
  request, direct-root store request publishing, and
  `CompactOwnerSearchDirectStepperV1(direct_root_build_request=True)`. Focused
  tests prove the stepper completes when the legacy parent
  `build_compact_root_batch_v1` symbol is monkeypatched to raise.
- [x] Wire the root-build-request hook through the real direct threaded owner
  proxy locally. `CompactOwnerSearchSlabProxyV1.run_action_step_from_root_build_request()`
  plus lazy direct-proxy routing now lets
  `CompactLazyThreadedOwnerSearchSlabProxyV1` consume the request. Focused
  test proves the real threaded path completes while the parent builder symbol
  raises, and the full owner-search service suite passes (`45 passed`). This is
  local proof, not H100 speed evidence.
- [x] Wire the root-build-request hook through the speed-row proof guard.
  Local smoke/Modal/report plumbing now threads
  `--owner-search-direct-root-build-request`, passes
  `direct_root_build_request=True` into `CompactOwnerSearchDirectStepperV1`,
  projects the new nested proof fields, and fails closed on missing/nonzero
  parent builder calls, publish/resolve/owner-build mismatches, parent root
  batch objects sent, observation bytes in the request, missing resident
  handle, or lost resident-root/stub/action-only/parent-row gates. Validation:
  `tests/test_compact_coach_speed_row_smoke.py` full module `87 passed`.
- [x] Run the local root-build-request speed-row smoke with resident-root view
  and host-observation stub enabled. Local smoke
  `opt132-local-owner-rootbuildrequest-threaded-directtransition-smoke-20260607-r3`
  passed with build-request schema/kind
  `curvyzero_compact_root_build_request/v1` /
  `resident_root_view_build_request_v1`, publish/resolve/owner-build
  `15/15/15`, parent builder used/calls `false/0`, parent build sec `0.0`,
  request observation bytes `0`, resident handle/proof true, H2D/D2H
  `0.0/0.0`, stub materialized bytes `0`, parent rows `0/0`, search payload
  bytes `0`, direct replay batches/transitions/transport `3/12/3`, action
  mismatches `0`, pending maintenance/policy lag `0/0`, and owner pid/bytes
  reported. This is proof currency, not H100 speed evidence.
- [x] Run the H100 same-work decision row for direct root build-request on top
  of the current columnar/direct-table owner-search stack. H100
  `opt132-h100-rootbuildrequest-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607`
  was proof-clean but speed-rejected: `11327.75 env/s`, `65.4477s`, only
  `0.893x` OPT-104 and `0.715x` columnar r2. It proved parent builder
  used/calls `false/0`, root build sec `0.0`, publish/resolve/owner-build
  `904/904/904`, request observation bytes `0`, parent rows/search payload
  bytes `0/0`, direct replay `181/724/181`, action mismatches `0`, and clean
  maintenance/policy lag. Do not repeat unchanged.
- [ ] Implement locally: move a larger owner graph/hot data surface beyond
  proof metadata: owner search dispatch/parent wait, mechanics/observation
  ownership, replay/table maintenance as part of a broader fixed buffer, or
  learner publication/update handoff. Start from the H100 regression buckets:
  parent wait `26.151s`, replay append `22.783s`, learner train `15.526s`,
  worker search `16.422s`, observation `13.196s`, slab `29.850s`. Do not add
  pure timers, replay-only gather tweaks, or another root-build-request repeat.

## Historical Checklist Before r29-r34

- [x] Wire accepted-preset local proof projection for maintained tensor-native
  replay. Decision: local smoke builder, Modal producer, outer Modal launcher,
  and speed-row evidence validation now preserve and fail closed on table
  source/reused/missing rows, fallback count/reason, implementation name,
  checksum/sample-order/cache context, parent learner-ready cache dependency,
  and host-fallback proof fields. Focused tensor-native tests passed, the
  compact speed-row suite passed, and the mechanics/proof combined slice passed
  (`214 passed`).
- [x] Run the maintained tensor-native replay H100 proof row. Decision:
  OPT-132BD (`1084/270`, B1024/A1, normal death, learner-ready cache on,
  maintained tensor-native replay on, specialized builder off) passed
  fail-closed proof with violations `[]`, table source
  `maintained_record_table_v1`, reused records `1080`, missing records `0`,
  and fallback count `0`.
- [x] Treat OPT-132BD as a speed failure, not success. It produced
  `3884.88 env steps/sec` and `285.727s` wall, about `0.31x` OPT-104. The next
  implementation target is maintained sample candidate / offset / sampleable
  row state, because sample gate still cost `74.005s`: candidate scan
  `28.734s`, RNG `6.479s`, residual `35.897s`, while tensor-native
  concat/gather was only `0.0127s` / `0.00033s`.
- [x] Implement maintained candidate/index sampling for `_CompactReplayRingV1`:
  avoid rebuilding candidate entries, successor windows, group counts, offsets,
  terminal flat rows, and massive no-replacement RNG surfaces every sample.
  Preserve terminal-only synthetic candidates, tensor-native proof fields, and
  grouped fallback equality.
  Current state: first local layer is implemented for unroll-2 accepted-row
  sampling. It snapshots `maintained_sample_universe_v1`, reuses successor
  windows/group counts/terminal local rows, reports candidate-universe proof
  fields, and preserves partial-terminal/mixed-candidate tensor-native equality.
  Local benchmark now shows real tensor-native replay sample/build median
  `0.000501s` vs current ring `0.004726s`, a `9.44x` CPU proof.
- [x] Run the maintained sample-universe H100 row. Decision: OPT-132BF passed
  proof and improved speed to `5362.68 env steps/sec`, `206.989s` wall.
  Sample gate fell to `4.116s` from OPT-132BD's `74.005s`; candidate scan fell
  to `0.332s`, RNG to `0.774s`, residual to `0.535s`. Still no speed claim:
  it is only `0.42x` OPT-104.
- [x] Move to the env/loop mechanics branch. Decision: first local
  fixed-action tape proof is implemented in
  `scripts/benchmark_vector_fixed_action_tape.py` and passes B1024/180/45 with
  compact-profile and fixed-buffer-direct full-state, per-step trajectory,
  output, action/timing-tape, and expected-death equality. Latest local toy
  wall: compact `0.3586s`, fixed direct `0.2179s`, `1.65x`. This is not
  full-loop speed evidence because observation is a zero stub, measured-window
  new deaths are `0` after warmup, and the fixture has no terminal/autoreset
  row, search, slab commit, replay append, sample, learner, or policy refresh.
- [x] Extend the env/loop mechanics branch through closed slab/replay/sample
  handoff. Decision: OPT-132BK locally closes deterministic search-feedback
  action flow, slab commit, replay append/index rows, replay-ring sample, and
  sample-batch checksum proof while preserving compact/direct equality.
  Maintained sampling closed the sample-gate surface but did not close the
  speed goal. Remaining H100 costs are outside
  sample gate: primary residual `122.679s`, actor step wall `41.852s`,
  actor/autoreset `29.656s`, search dispatch `14.248s`, observation
  `11.423s`, env runtime `10.000s`. Terminal/autoreset coverage is now
  locally proved by OPT-132BH: B1024/1/0 wall-death fixture, terminal rows
  `1024`, autoreset rows `1024`, matching post-autoreset state. Rendered
  observation coverage is now locally proved by OPT-132BI with
  `zero_observation_stub=false`, schema/hash/shape proof, nonzero observation
  content, and terminal/autoreset equality preserved. Fixed-shape search/root
  handoff is now locally proved by OPT-132BJ with rendered observations,
  active-root/action identity checks, deterministic digests, and zero
  CTree/tolist/per-sim D2H. Slab/replay/sample handoff is now locally proved by
  OPT-132BK with `slab_step_count=4`, committed index rows `28`, replay appends
  `3`, sample gates `3`, and no slab-retained committed index rows.
- [x] Map the OPT-132BF primary residual to concrete ownership surfaces and
  pick one implementation candidate that removes a whole handoff. Decision:
  select owner-search deferred maintenance with action-only results and
  owner-materialized replay. Closed by local real-entrypoint owner-search proof:
  parent committed/stored rows are `0/0`, owner replay/sample/learner
  maintenance is visible, and parent-commit metadata is truthful. Later
  normal-death scale preserved proof but failed speed shape. Mock-fast scale
  preserved proof and was fast locally, so the next implementation is real
  learner placement/update/overlap before any H100 gate.
- [x] Fix the first owner-search proof-field lie. Decision: action-only
  owner-materialized replay now reports
  `compact_owner_search_parent_slab_commits_replay=false`; the owner-search
  service tests and speed-row smoke proof both cover this shape.
- [x] Produce a real-profile local proof for the selected owner-search path.
  Decision: local smoke
  `opt132-local-owner-action-only-profile-proof-20260604` passed with action-only
  result, zero search payload bytes, parent committed/stored rows `0/0`, owner
  replay entries/submitted/append `15/15/15`, train requests `3`, learner
  updates `3`, owner drain requests `2`, drained `15`, pending/inflight/failed
  `0/false/false`, and final owner drain inside measured wall. Local speed
  `70.93 env steps/sec` is proof only, not speed evidence.
- [x] Close the pre-r14 owner learner resource/work-shape lane as superseded.
  Decision: local MPS placement failed as a speed path (`20.81 env/s` vs CPU
  `25.78`), eager append pre-drain failed (`24.09 env/s`, learner update
  `15.12s`, final drain `12.67s`), same-process async learner overlap failed
  (`24.47 env/s`, async submit/completed/pending `6/6/0`, actions while async
  pending `47`, async wait `14.17s`, final drain `12.82s`), and max-pending
  `6` only reached `25.37 env/s` with final drain `12.31s`. The r14 H100 row
  then beat OPT-104 by eliminating shared-model transport and inline host
  payload clone in the inline owner-search path. Do not implement a learner
  owner/overlap boundary as P0 unless the r14 repeat fails and points back to
  that surface.
- [x] Repeat r14 exactly and run the first promotion audit. Decision: r15 kept
  exact identity/proof but failed speed at `10502.70 env/s`, `17.5498s`.
  Stable speed claim remains false.
- [x] Try a longer same-mechanism row. Decision: r16 `724/180` kept proof but
  ran `6491.80 env/s`, `114.202s`, with owner train sample/update
  `52.807s/3.236s`.
- [x] Try composing r14 shared/no-clone with existing fused/tensor-native batch
  flags. Decision: r17 failed before row creation because
  `compact_owned_loop_fused_learner_batch requires compact_owned_loop_entrypoint`.
- [x] Close the fast learner-ready/tensor-native owner-search train-sampling
  item as superseded by the r29-r34 candidate decision. It remains useful
  historical context, but it is not the active first move after threaded/
  background refresh-4 rows repeated above OPT-104.
- [x] Answer strict policy-refresh cadence parity for the current candidate.
  Decision: r30/r31/r32/r33/r34 used `policy_refresh_interval=4`; threaded
  r32/r33 repeated above OPT-104 and r34 stayed positive over a long row.
- [x] Run eager append pre-drain as the scheduler-polish falsifier. Decision:
  `opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-eagerappend-r1-20260604`
  preserved proof with eager append drains `7`, coalesced skips `0`,
  staged/drained work `55/55`, submitted/owner/refreshed updates `6/6/6`,
  policy lag current/max `0/6`, and actions while lagged `47`, but speed was
  `24.09 env steps/sec`, worse than the CPU baseline. Scheduler polish is
  exhausted as the P0 path.
- [x] Close owner-search H100-readiness proof gaps as preservation guardrails:
  action-feedback checksum/mismatch proof, requested-vs-actual cadence checks,
  owner sample telemetry requirements, request-vs-entry drain counts,
  finite/nonnegative final-drain timing, and required action-only/two-phase
  handle fields. Added submitted-update lag proof:
  `opt132-local-owner-action-only-inner2-slab-lagproof-steps16-cadence8-b512-nodeath-r1-20260604`
  has submitted/owner/refreshed updates `2/2/2`, policy lag current/max `0/2`,
  and actions while policy-lagged `15`.
- [x] Split owner maintenance timing: separate sample, learner update, model
  digest, model state dict/ref publication, refresh, drain, and residual so
  `compact_owner_search_worker_learner_train_sec` is actionable.
- [x] Run the threaded/background owner-search action-only local falsifier. It
  passed proof and showed overlap mechanics exist, but the owner learner-update
  tail remains dominant: scale-48 served 51 actions while maintenance was
  pending and still ended with final drain `1.312s`, learner update `1.410s`,
  and train count `13`.
- [ ] Keep the OPT-132 architecture reset map current while implementing:
  env state layout, reset/autoreset, terminal/death identity, observation
  buffers, action handoff, search, replay, sample, learner, proof ownership,
  and every CPU/GPU plus Python/native boundary.
- [ ] Keep sidecar reads bounded to implementation falsifiers: primary-residual
  ownership, resident/compact data movement, terminal/autoreset replay
  semantics, and mature fixed-shape env/search/replay patterns. Do not reopen
  table maintenance or sample-universe design unless BK-style proof fields
  regress.
- [x] Build `scripts/benchmark_compact_tensor_native_unroll2_replay.py`:
  compare current `_CompactReplayRingV1.sample_from_snapshot`, resident grouped
  builder with learner-ready unroll-2 targets, and a toy flat tensor-native
  `CompactMuZeroLearnerBatchV1` path with row/tensor equality proof. Decision:
  local 128x16/512-row CPU proof passed; flat gather median was `0.000054791s`
  versus current ring sample/build `0.016738708s`, a `305.501x` toy ceiling.
- [x] Prototype the default-off tensor-native replay / learner-ready unroll-2
  path inside the real loop while preserving benchmark proof fields: sample
  order, row/window checksums, terminal/death identity, host fallback false,
  and cache proof. Decision: `sample_from_snapshot` now has a guarded
  tensor-native gather branch; local proof passed and exposed per-sample table
  prepack as the next ownership problem, superseded by the maintained-table
  proof below.
- [x] Install maintained tensor-native replay table state. Decision:
  `_CompactReplayRingV1` now maintains default-off
  per-record learner table entries when the tensor-native replay flag is
  requested, snapshots table state, rebuilds it on metadata enable, invalidates
  it on evict/stale successor windows, and samples by concatenating maintained
  record tables plus row-index gather. Local proof passes with source
  `maintained_record_table_v1`, reused records `32`, missing records `0`, and
  real path `5.047x` versus current on the 32x8/128-row CPU benchmark.
- [x] Add first maintained-table fail-closed fallback proof. Decision:
  focused test now breaks the table snapshot while tensor-native replay is
  requested and proves fallback to the grouped builder preserves tensor equality
  while reporting `used=false`, `fallback_count=1`, and fallback reason
  `tensor-native replay missing maintained table`.
- [x] Add maintained-table eviction/rebuild proof. Decision: focused test fills
  a table-backed ring, appends a new record to evict record `0`, proves the
  maintained table keys match remaining records `{1,2,3,4,5,6}`, samples from
  the post-eviction snapshot, and matches the grouped builder with
  `maintained_record_table_v1`.
- [ ] Keep env-system feasibility as a bounded candidate lane only. Do not
  extend mechanics toys unless the primary-residual ownership map or the
  owner-search candidate falsification selects that handoff. Compare against
  PufferLib, EnvPool, Sample Factory, and Isaac-style architecture patterns
  before choosing any port.
  Current smallest proof exists in `scripts/benchmark_vector_fixed_action_tape.py`
  using `source_borderless_wrap_skips_destination_body_then_next_frame_kills.json`.
  It repeats a deterministic source-action tape across batches, compares current
  mapping-shaped CPU stepping with a fixed-buffer direct runtime path, and
  requires state/body/death/counter/checksum equality before reporting speed.
  Terminal/autoreset proof is now covered by
  `source_normal_wall_death_step.json`; rendered-observation proof is now
  covered by OPT-132BI; fixed-shape search/root proof is now covered by
  OPT-132BJ; closed search-feedback slab/replay/sample proof is now covered by
  OPT-132BK. Primary residual is mapped and owner-search is selected/proved
  locally; env-system feasibility stays parked unless owner-search scale/
  hardening falsifies the path.
- [x] Resolve accepted-preset cache/proof plumbing. Decision: OPT-132BB/BC
  exposed missing local/remote validator surfaces; OPT-132BD passed the remote
  maintained replay proof. Remaining work is speed architecture, not cache
  proof plumbing.
- [ ] Do not launch another pure attribution H100 diagnostic. The next H100 row
  must test a concrete architecture hypothesis with fail-closed proof fields.

## Historical Checklist

- [x] Guard the accepted-fast-path preset against remote-result flag drift.
- [x] Reject OPT-132-G as a new baseline because OPT-132-H did not repeat it.
- [x] Patch top-level report fields for seed/work-shape repeatability.
- [x] Patch hash-bound evidence fields for seed/work shape, actor trajectory,
  terminal/autoreset/death checksums, sample-order checksums, and
  sample/learner counters.
- [x] Fix the accepted-fast-path repeatability guard so signed checksums are
  allowed when nonzero and counters remain positive.
- [x] Revalidate the OPT-132-I H100 artifact under the corrected guard.
- [x] Run one more accepted-fast-path same-work H100 repeat with the corrected
  guard and no new feature flags: OPT-132-J.
- [x] Compare G/H/I/J using the new top-level fields and record the runtime
  swing in `opt132-ghij-runtime-comparison-20260602/comparison.json`.
- [x] Add explicit long-window stability diagnostic support to the accepted
  fast-path launcher and comparator.
- [x] Tighten the comparator so a stable speed claim requires at least one clean
  exact candidate repeat and no accepted-fast-path violations.
- [x] Run at least two long-window same-work H100 stability diagnostics with
  `--compact-owned-accepted-fast-path-step-window stability_724_180`.
- [x] Compare the long-window diagnostics and decide whether measurement is
  stable enough to optimize actor/autoreset, sample build/RNG, or learner
  backward. Decision: not stable enough yet; exact same work still moved by
  `11.298s`.
- [x] Try a longer `stability_1444_360` same-work pair. Decision: failed on
  H100 memory, not useful as the next default diagnostic.
- [x] Try a middle `stability_1084_270` same-work pair. Decision: also failed
  on H100 memory.
- [x] Run repeated `stability_724_180` A/A rows. Decision: five exact-identity
  rows still ranged `47.945s` wall, so measurement is not stable.
- [x] Build memory-bounded long-window diagnostic mode. It stops retaining the
  slab's full committed-row history and omits the nested source profile payload
  for diagnostic rows.
- [x] Run the first `stability_1084_270` bounded diagnostic. Decision:
  OPT-132-U still OOMed on H100, with PyTorch around `78.34 GiB` allocated.
- [x] Add replay-store retained resident-snapshot counters so the next row can
  identify step-scaled resident replay memory instead of hiding it.
- [x] Add latest-frame resident replay snapshot mode for bounded diagnostics.
  It stores one latest frame per normal step, rebuilds full frame history only
  on reset steps, and keeps terminal final observations sparse.
- [x] Surface bounded diagnostic labels, resident replay snapshot mode, and
  retained resident-snapshot count/byte fields in speed-row summary, compact
  payload, and smoke report.
- [x] Run a bounded `724/180` H100 bridge diagnostic with latest-frame resident
  replay snapshots active. Decision: passed as a report/memory proof row only;
  no speed claim.
- [x] Run the bounded `1084/270` H100 fit diagnostic with latest-frame resident
  replay snapshots active. Decision: r1 passed without OOM; memory fit is no
  longer the immediate blocker.
- [x] Repeat the exact same `1084/270` row and compare measurement stability.
  Decision: memory fit passed but timing stability still failed. Wall spread was
  `8.28%` of median; sample gate spread was `14.04%`; learner-batch build
  spread was `15.57%`.
- [x] Add sample-gate learner-batch-build diagnostics. Decision: the resident
  grouped learner-batch builder now reports internal sub-timers, and the profile
  records per-call learner-batch-build timing stats so the next row can separate
  uniform drift from a few slow sample calls.
- [x] Run exact `1084/270` H100 diagnostic with the new builder sub-timers and
  per-call stats visible. Decision: first diagnostic passed and exposed
  builder internals, but the three-row timing spread got worse. It is still
  diagnostic evidence only, not a speed claim.
- [x] Repeat exact `1084/270` H100 diagnostic under the same new field set.
  Decision: r1/r2 matched exact identity and were stable on the main buckets:
  wall `0.78%`, sample gate `2.17%`, learner-batch build `2.01%`.
- [x] Run exact `1084/270` r3 under the same new field set. Decision: r3 kept
  exact identity but broke timing stability. Three-row wall spread was `15.55%`,
  sample gate `19.49%`, and learner-batch build `18.88%`.
- [x] Add diagnostic-only CUDA sync timing around sample gate, learner-batch
  builder children, and learner phases. Decision: local smoke, Modal launcher,
  Modal producer, comparison artifacts, and focused tests now fail closed for
  requested sync diagnostics.
- [x] Run exact `1084/270` r1/r2 with sync timing. Decision: identity and sync
  counts matched exactly, violations were `[]`, and wall/sample/build spreads
  improved to `1.10%`, `3.48%`, and `3.61%`; still diagnostic only because
  observation/observation-other/sample-rng and builder cuda_sync exceed or brush
  the major-bucket bar.
- [x] Run exact OPT-132Y r3 with the same `1084/270` sync timing flags and
  compare r1/r2/r3. Decision: exact identity and sync counts held, but timing
  stability failed hard. Wall spread was `30.99%`, sample gate `30.47%`,
  learner-batch build `28.01%`, actor wall `31.66%`, observation `62.14%`, and
  builder cuda_sync `39.85%`.
- [x] Add diagnostic-only runtime-step envelope stats for CUDA-sync diagnostic
  rows. Decision: source profile, speed-row summary/compact payload, Modal
  reports, and comparison artifacts now carry `compact_profile_runtime_step_*`
  fields. Focused ruff and pytest passed. This is diagnostic-only and does not
  make a speed claim.
- [x] Run OPT-132Z r1/r2 exact `1084/270` with runtime-step envelope fields.
  Decision: exact identity held, but timing failed hard again. Wall and
  runtime-step sum both spread `32.50%`; sample gate spread `42.67%`; runtime
  p50 barely moved while p95/max jumped; slowest steps were sample-gate
  dominated.
- [x] Add diagnostic-only sample-gate per-call child distribution stats.
  Decision: source profile, speed-row summary/compact payload, Modal reports,
  and comparison artifacts now carry sample-gate total/candidate/RNG/residual
  per-call count/sum/min/max/p50/p95 fields. Focused ruff and pytest passed.
- [x] Run/read OPT-132AA r1/r2 exact `1084/270` with sample-gate per-call
  fields. Decision: exact identity held and violations stayed `[]`. AA r1/r2
  wall spread was `4.96%`; sample gate spread `4.60%`; learner-batch build
  spread `2.88%`. Sample-gate per-call p50/p95/max was
  `1.032/1.630/1.723s -> 1.061/1.674/1.783s`; learner-batch-build per-call
  p50/p95/max was `0.652/0.804/1.094s -> 0.666/0.802/1.100s`.
- [x] Add learner-batch builder child per-call attribution locally. Decision:
  source profile, speed-row summary/compact payload, local/remote Modal
  reports, and comparison artifacts now carry builder child per-call stats for
  group-loop, terminal-metadata, unroll-fields, write-output, order-restore,
  finalize-outputs, metadata-sync, metadata-build, and builder-cuda-sync timers.
  Focused ruff and pytest passed.
- [x] Run/read exact `1084/270` on H100 with builder-child per-call
  attribution as the historical pre-AF gate. Decision: OPT-132AB
  r1/r2 matched exact identity and violations stayed `[]`, but timing failed
  hard. Wall spread was `45.57%`, sample gate `54.93%`, learner-batch build
  `56.09%`, builder group-loop `56.27%`, and unroll fields `53.05%`. No speed
  claim.
- [x] Add diagnostic-only deep builder group-loop attribution locally. Decision:
  OPT-132AC now projects terminal metadata mask/tensor-fallback/validate/final-
  observation timers and unroll terminal-window-hint/identity/stack-fields/
  mask-build/terminal-value/mask-apply/action-stack timers through source
  profile, speed-row reports, Modal reports, comparator fields, and focused
  tests. This is diagnostic-only.
- [x] Run/read OPT-132AC exact `1084/270` r1c/r2 on H100. Decision: after the
  first two launch attempts failed before row creation with Modal
  `RESOURCE_EXHAUSTED`, r1c/r2 completed with exact identity and violations
  `[]`. Timing still failed hard: wall spread `30.18%`, sample gate `32.11%`,
  learner-batch build `30.63%`, builder group-loop `30.56%`, unroll fields
  `35.30%`, and terminal metadata `26.46%`. No speed claim.
- [x] Read OPT-132AC exact `1084/270` r3 and compare against r1c/r2 before any
  builder speed patch. Decision: r3 completed cleanly and clustered with r2,
  but the three-row comparison still failed stability: wall `36.55%`, sample
  gate `37.32%`, learner-batch build `34.23%`, builder group-loop `34.11%`,
  unroll fields `40.16%`, and terminal metadata `27.48%`. No speed claim.
- [x] Explain the OPT-132AC r1c slow-outlier branch before any builder speed
  patch. Decision: superseded by OPT-132AD through OPT-132AL. Later exact
  rows showed the swing recurs without a single persistent AC-only outlier,
  without AF specialization, without CUDA-sync probes, and without GPU
  utilization sampling. The current open branch is late measured-window
  sample-gate cadence slowdown.
- [x] Add OPT-132AD runtime phase/residual attribution locally. Decision: the
  existing CUDA-sync diagnostic flag now emits measured-step distributions for
  actor env runtime, actor autoreset, sample-gate residual, sample-gate sync,
  sample-gate builder group-loop, and sample-gate builder sync, alongside the
  existing runtime phase fields. Source profile, speed-row summary/compact
  payload, local/remote Modal reports, and comparator fields project them.
  Ruff passed; focused source-profile/smoke/comparator pytest passed; broader
  compact speed-row slice passed. Diagnostic only.
- [x] Run/read exact H100 `1084/270` OPT-132AD r1/r2/r3 repeats. Decision:
  identity exact, accepted-fast-path violations `[]`, sync diagnostic
  violations `[]`, but stability still failed: wall `24.11%`, sample gate
  `30.48%`, learner-batch build `34.32%`, sample-gate builder group-loop
  `34.49%`, and unroll fields `36.13%`. No speed claim.
- [x] Add OPT-132AE builder group-loop accounted/residual attribution locally.
  Decision: source profile, speed-row summary/compact payload, local/remote
  Modal reports, and comparator now project group-loop accounted/residual totals
  and per-call stats. Accounted is terminal metadata + terminal-window hint +
  unroll fields + write output. Ruff passed; focused pytest `4 passed`; broader
  speed-row/report slice `63 passed`. Diagnostic only.
- [x] Run/read exact H100 `1084/270` OPT-132AE r1/r2 repeats. Decision:
  identity exact, accepted-fast-path violations `[]`, sync diagnostic
  violations `[]`, wall `271.920s -> 284.561s` (`4.54%`), group-loop
  `113.971s -> 119.853s`, accounted child work `95.872s -> 101.300s`, residual
  `18.099s -> 18.553s`. Both rows are slow; no speed claim.
- [x] Prototype only a guarded/default-off unroll-2 builder specialization if it
  preserves exact learner-batch tensors, checksums, terminal masks, and proof
  fields. Decision: OPT-132AF adds a default-off unroll-2 specialized builder
  only for fused resident grouped learner batches with exact guard/proof
  fields. Ruff passed; focused source-profile parity, speed-row smoke,
  comparator, and full smoke validation passed. Local only; no speed claim.
- [x] Run local OPT-132AF proof smoke with the specialized flag. Decision:
  `opt132af-local-unroll2-specialized-builder-hardened-smoke-20260603` completed
  `ok=true` with requested=true, used=true, eligible/call_count `13`,
  fallback_count `0`, fallback_reason `none`, impl `unroll2_specialized_v1`,
  path `unroll2_specialized`, and terminal target rows still present. Local
  only; no speed claim.
- [x] Run OPT-132AF H100 r1/r2 with accepted-fast-path proof fields intact.
  Decision: r1/r2 completed `ok=true`, exact identity held, accepted-fast-path
  violations `[]`, CUDA-sync violations `[]`, unroll2 violations `[]`, and both
  rows proved requested=true, used=true, eligible_count/call_count `399`,
  fallback_count=0, fallback_reason=`none`, impl=`unroll2_specialized_v1`, and
  path=`unroll2_specialized`. Wall was `211.683s -> 225.020s` (`6.11%`), so
  no speed claim.
- [x] Finish OPT-132AF H100 r3 and compare r1/r2/r3. Decision: r3 completed
  cleanly with the same proof counts and exact identity. The three-row
  comparison kept `stable_speed_claim_allowed=false`; wall slowed
  `211.683s -> 225.020s -> 234.042s` (`9.94%` spread), sample gate spread
  `12.99%`, learner-batch build `14.06%`, and builder group-loop `14.03%`.
  OPT-132AF is rejected as a stable speed claim under exact `1084/270`.
- [x] Preserve structured artifacts for future pre-FunctionCall Modal launch
  failures. Decision: the launcher now writes `launch.json` and
  `compact_coach_speed_row_modal_report.json` with `failure_stage=launch`,
  resource-exhaustion classification, proof-context fields, and non-claims.
  The remote Modal entrypoint prints structured `spawn_failed` payloads for
  reachable spawn failures, and local/remote per-call prefix parity is tested.
  Focused ruff and pytest passed. Existing OPT-132AC r1/r1b attempts predate
  this patch and remain empty-dir launch failures.
- [x] Measure or reject OPT-132AF under exact H100 `1084/270` proof before any
  additional learner-batch builder speed patch. Decision: rejected as a stable
  speed claim; proof path is intact, timing is not stable.
- [x] Add comparator hardware context before the next AF slowdown diagnostic.
  Decision: `scripts/compare_compact_coach_speed_rows.py` now projects a
  per-row `gpu_utilization` block and numeric timing/range fields for GPU
  sample count, utilization, memory, and power. Focused ruff and comparator
  pytest passed; a refreshed AF r1/r2/r3 comparison kept exact identity and
  `stable_speed_claim_allowed=false`.
- [x] Explain the OPT-132 AF through AZ same-work runtime swing enough to choose
  a direction. Decision: OPT-132AZ showed the remaining movement is broad
  sample-gate learner-batch builder user CPU. This rejects more pure attribution
  and makes learner-ready replay or batch-level unroll gather candidate lanes;
  the next active action is the architecture dataflow map and toy matrix.
- [x] Run OPT-132AG exact AF `1084/270` diagnostic rows with GPU utilization
  sampling enabled, then compare them with the hardware context fields.
  Decision: r4/r5/r6 kept exact identity and clean proof, and the monotonic AF
  slowdown did not repeat, but stability still failed: wall spread `6.97%`,
  sample gate `11.57%`, and builder group-loop `15.51%`. The slowest AG row
  had lower mean GPU utilization and lower max power than the faster rows, so
  the sampler does not support a simple power/throttle explanation. Diagnostic
  only, no speed claim.
- [x] Run or otherwise produce an exact `1084/270` generic-builder GPU-sampled
  control packet to separate AF specialization from the remaining runtime
  swing. Decision: OPT-132AH r1/r2/r3 kept exact identity with the generic
  builder, but drifted hard: wall `205.584s / 243.805s / 289.172s`
  (`34.28%`), sample gate `48.55%`, learner-batch build `52.04%`, and builder
  group-loop `52.75%`. The remaining runtime swing is not AF-specific.
- [x] Bound diagnostic/runtime overhead on exact `1084/270` before any further
  builder speed patch. Decision: CUDA-sync probes, runtime-step diagnostics,
  and GPU utilization sampling have each been separated enough to show none is
  the sole cause of the underlying generic/AF runtime swing.
- [x] Add a runtime-step envelope flag that does not require CUDA-sync timing.
  Decision: OPT-132AI adds
  `--compact-profile-runtime-step-timing-diagnostics`; CUDA-sync diagnostics
  still imply runtime-step stats, but runtime-step stats can now be requested
  alone. Local/Modal collection fails closed if requested runtime-step stats are
  missing. Focused validation passed.
- [x] Run the exact `1084/270` runtime-step-only H100 overhead check that
  separates AJ from AH-style sync probes. Decision: OPT-132AJ r1/r2/r3
  kept exact identity with CUDA-sync probes off, but timing still failed:
  wall/runtime-step sum `10.49%`, sample gate `12.72%`, learner-batch build
  `14.09%`, and builder group-loop `14.25%`. Removing CUDA-sync probes narrows
  generic drift versus AH but does not solve stability.
- [x] Add same-work wall-swing attribution and runtime-step cadence diagnostics.
  Decision: OPT-132AK adds comparator slowest-vs-fastest attribution plus
  runtime-step active/inactive, early/mid/late, and top-slowest-record fields.
  The AJ refresh showed runtime-step sum explains `99.995%` of the r2-r3 wall
  delta, sample gate `61.00%`, learner-batch build `35.87%`, and builder
  group-loop `35.23%`. Local validation passed.
- [x] Run exact generic `1084/270` runtime-cadence H100 rows with GPU sampling
  disabled. Decision: OPT-132AL r1/r2/r3 kept exact identity and violations
  `[]`, with CUDA-sync and GPU sampling both false, but stability still failed:
  wall/runtime-step `21.70%`, sample gate `22.84%`, learner-batch build
  `20.92%`, late runtime-step sum `30.78%`, and late sample-gate `31.11%`.
  No speed claim.
- [x] Add chronological active sample-gate cadence diagnostics after OPT-132AL.
  Decision: OPT-132AM projects active sample-gate phase distributions by
  measured third, sample-gate residual bucket sums, enriched top-slowest
  runtime-step records, and slowest per-call iteration/measured-iteration.
  Local validation passed; no speed claim.
- [x] Run an OPT-132AL-style exact generic `1084/270` no-sampler H100 row with
  OPT-132AM fields and compare early/mid/late active sample-gate p50/p95/max.
  Decision: OPT-132AN r1/r2 kept exact identity and violations `[]`, but
  timing failed again: wall/runtime-step `21.806%`, sample gate
  `125.169s -> 93.384s`, learner-batch build `68.851s -> 49.643s`, and
  builder group-loop `68.101s -> 48.923s`. Active sample-gate p50/p95 rose
  early/mid/late in both rows and r1 was elevated across all thirds versus r2,
  so the slowdown is broad active-call elevation, not only late spikes.
- [x] Add bounded per-sample-gate replay-state trace records after OPT-132AN.
  Decision: OPT-132AO projects
  `compact_rollout_slab_sample_gate_call_trace_records` with call index,
  measured iteration/third, sample seeds/checksums, replay stored/eligible/
  excluded/evicted counts, replay capacity, and per-call timing through source
  profile, speed-row summary/compact payload, local/remote Modal reports, and
  focused tests. Local only; no speed claim.
- [x] Run an OPT-132AN-style exact generic `1084/270` no-sampler H100 row with
  OPT-132AO trace records and compare per-call replay state against active
  sample-gate timing. Decision: AO r1/r2 kept exact identity and violations
  `[]`, but timing still failed: wall/runtime-step `13.944%`, sample gate
  `132.338s -> 163.614s`, learner-batch build `72.215s -> 91.981s`, and
  builder group-loop `71.350s -> 91.030s`. Trace state mismatch count was
  `0/135` across call index, measured iteration, sample seeds/checksums,
  stored/eligible/excluded/evicted replay counts, and replay capacity.
- [x] Add allocator/runtime trace fields below OPT-132AO exposed replay/sample
  state. Decision: OPT-132AP extends each bounded sample-gate trace record with
  CUDA memory allocated/reserved/peak counters, allocator retry/OOM counters,
  learner-batch-build memory deltas, Python GC generation counts, process
  max-RSS raw state, sample-gate CUDA-sync timing, and deeper builder child
  timings. Focused ruff and `17` focused tests passed. Local only; no speed
  claim.
- [x] Run/read OPT-132AP exact generic `1084/270` no-sampler H100 rows and
  compare allocator/runtime plus deeper builder fields by sample-gate call
  index. Decision: AP r1/r2/r3 kept exact identity and violations `[]`, but
  timing still failed: wall `219.446s / 205.371s / 153.752s` (`31.99%` of
  median), sample gate `130.199s / 120.714s / 86.639s`, learner-batch build
  `70.732s / 65.983s / 46.881s`, and builder group-loop
  `69.882s / 65.174s / 46.350s`. Replay/sample trace identity and every CUDA
  allocator/memory counter matched exactly across all `135` calls; Python GC
  counts and RSS varied. No speed claim.
- [x] Add actual Python GC collection counters below OPT-132AP. Decision:
  OPT-132AQ adds `gc.get_stats()` collection/collected/uncollectable
  before/after/delta fields for gen0/gen1/gen2 to each runtime trace record.
  Focused ruff and `17` focused tests passed. Local only; no speed claim.
- [x] Run/read OPT-132AQ exact generic `1084/270` no-sampler H100 rows and
  compare real GC collection counters plus AP timing fields by sample-gate call
  index. Decision: AQ r1/r2 kept exact identity and violations `[]`, but wall
  still spread `28.856%` of median. Actual GC collection totals were nearly
  identical (`7232/659/28` versus `7234/655/28`), and `100` calls with
  identical GC collection deltas still carried `22.6695s` of sample-gate
  delta. Python GC collections do not explain the swing.
- [x] Add process/thread CPU-time trace fields below OPT-132AQ. Decision:
  OPT-132AR adds process and current-thread CPU-time before/after/delta fields
  for both the full sample gate and learner-batch-build slice. Focused ruff
  passed and the AP/AQ validation slice passed (`17` tests, `2` warnings).
  Local only; no speed claim.
- [x] Run/read OPT-132AR exact generic `1084/270` no-sampler H100 rows and
  compare process/thread CPU-time deltas against sample-gate and builder wall
  deltas by sample-gate call index. Decision: AR r1/r2 kept exact identity and
  violations `[]`, but wall spread `13.479%`. Sample-gate wall delta
  `18.848s` was backed by `17.33s / 17.18s` process/thread CPU-time deltas;
  learner-build wall delta `11.331s` was backed by `10.44s / 10.39s`
  process/thread CPU-time deltas. The remaining swing is CPU-time-backed, not
  mostly off-CPU wait.
- [x] Explain why identical trace state consumes different CPU time before any
  new speed patch. Decision: AS through AZ narrowed the swing to broad
  CPU-backed sample-gate learner-batch builder work. The architecture reset
  supersedes more pure attribution; use toy ceilings and prototype whole
  ownership surfaces instead.
- [x] Add resource-usage split below OPT-132AR. Decision: OPT-132AS adds
  process/thread `resource.getrusage` before/after/delta fields for
  user/system CPU, minor/major page faults, and voluntary/involuntary context
  switches around the full sample gate and learner-batch-build slice. Focused
  ruff and the AP/AQ/AR validation slice passed (`17` tests, `2` warnings).
  Local only; no speed claim.
- [x] Run/read OPT-132AS exact generic `1084/270` no-sampler H100 rows and
  compare resource usage deltas against AR CPU-time and child wall deltas by
  sample-gate call index. Decision: AS r1/r2 kept exact identity and
  violations `[]`, but timing still failed. The sample-gate r1-minus-r2 delta
  was `18.999s`; process user/system CPU deltas were `16.17s / 0.49s`;
  page-fault and context-switch deltas were zero. The swing is mostly user CPU.
- [x] Explain the remaining identical-trace-state sample-gate timing swing
  enough to choose a builder direction. Decision: AS through AZ narrowed it to
  CPU-backed sample-gate learner-batch builder work. The guarded learner-ready
  resident replay / unroll-2 cache is a prior proof lane. Later BD/BF/BK and
  owner-search proof superseded that path: current work is real learner
  resource/work-shape change in the selected owner-search candidate.
- [x] Add external CPU perf-stat diagnostic plumbing below OPT-132AS. Decision:
  OPT-132AT adds `--compact-profile-cpu-perf-stat-diagnostics`, wraps the
  remote speed-row producer in `perf stat -x,`, captures stdout/stderr
  artifacts, parses task-clock/cycles/ref-cycles/instructions/branch/cache/
  LLC/dTLB/page-fault/context-switch/CPU-migration counters into
  `compact_profile_cpu_perf_stat_*` fields, and projects them through the
  result/report/comparison surfaces. Validation passed: ruff, targeted
  perf-stat tests (`4` tests, `2` warnings), and broader smoke/compare slice
  (`20` tests, `2` warnings). Local only; no speed claim.
- [x] Run/read OPT-132AT exact generic `1084/270` no-sampler H100 row. Decision:
  the remote H100 function spawned (`fc-01KT7YGKDM2Y35NCG6C204TYMY`) but failed
  before a speed row because `perf` was not found in the image. Structured
  fields show `compact_profile_cpu_perf_stat_available=false` and return code
  `127`. No speed claim.
- [x] Add `linux-perf` to the compact speed-row Modal image only. Decision:
  OPT-132AU keeps the perf diagnostic scoped to `speed_row_image` and leaves the
  shared LightZero image untouched. Ruff and two focused smoke tests passed.
- [x] Run/read OPT-132AU exact generic `1084/270` no-sampler H100 row. Decision:
  `/usr/bin/perf` was available, but `perf stat` failed before the producer ran
  because `sys_perf_event_open()` returned `19 (No such device)` for
  `task-clock`. Return code `255`, parse line count `0`, no speed row.
- [x] Add the next in-process diagnostic below AS/AU. Decision: OPT-132AV adds
  process/thread CPU-time delta fields for learner-batch builder child phases
  and projects them through source profile trace records, speed-row summary/
  compact payload, local/remote Modal reports, and comparison timings. Ruff
  passed; focused profile/smoke/comparator tests passed; broader smoke/compare
  slice passed (`78` tests, `2` warnings). Local only; H100 read is active.
- [x] Run/read OPT-132AV exact generic `1084/270` no-sampler H100 rows and use
  the builder-child CPU deltas to split AS's user-CPU-backed sample-gate/build
  swing by child phase. Decision: r1/r2 kept exact identity and report
  violations `[]`, but stability failed: wall spread `15.03%`, sample gate
  `20.312s`, learner-batch build `16.432s`, builder group-loop `16.287s`.
  Group-loop process CPU rose `14.88s`, with accounted child CPU `+10.60s`,
  residual `+4.28s`, terminal metadata `+6.37s`, and unroll fields `+2.76s`.
  No speed claim.
- [x] Add the next local split under OPT-132AV's terminal/final-observation and
  residual buckets. Decision: OPT-132AW adds group-loop prepare and
  terminal-value bookkeeping wall/CPU fields, splits terminal final-observation
  into presence/select-current/gather wall/CPU fields, and adds branch/storage
  proof counters for terminal final-observation work. Ruff passed; focused
  source-profile/smoke/comparator pytest passed (`189` tests, `2` warnings).
  Local only; superseded by the AW H100 read.
- [x] Run/read OPT-132AW exact generic `1084/270` no-sampler H100 rows and use
  the branch/storage proof plus deeper CPU split to explain AV's ambiguous
  buckets. Decision: r1/r2 kept exact identity and report violations `[]`, but
  wall still failed repeatability (`250.183s / 232.521s`, `7.318%`). The
  final-observation branch/storage proof was identical (`399` groups,
  `0` index-fast-path, `399` fallback, final-row sum/max `512/4`, sparse
  storage `399`, sparse-row sum/max `3102/16`). Residual CPU was mostly stable
  (`14.45s / 14.07s`), while unroll-fields CPU (`38.18s / 34.98s`) and
  final-observation gather CPU (`6.84s / 5.95s`) still moved. No speed claim.
- [x] Explain/remove identical terminal-final-observation fallback gather work.
  Decision: OPT-132AX removed grouped learner select-current/gather
  materialization and H100 proved `validate_only_count == fallback_count`,
  `materialized_count == 0`, and gather/select-current wall/CPU `0` in r1/r2.
  This did not solve stability.
- [x] Add local validate-only terminal-final-observation coverage for the
  grouped learner path. Decision: OPT-132AX adds
  `_validate_resident_final_observation_for_rows`, keeps sample-path
  materialization unchanged, adds `terminal_metadata_final_observation_validate`
  timing/CPU fields, validate-only/materialized proof counters, and optional
  comparator identity for terminal-final-observation proof counters. Ruff
  passed; focused source-profile/smoke/comparator pytest passed (`190` tests,
  `2` warnings).
- [x] Run/read OPT-132AX exact generic `1084/270` no-sampler H100 r1/r2.
  Decision: exact identity held and proof was exact, but wall spread was
  `43.864%` (`168.280s / 262.830s`); no speed claim.
- [x] Split the still-moving unroll-fields CPU and broader builder group-loop
  CPU after AX. Decision: OPT-132AY added nested process/thread CPU fields for
  unroll identity, stack fields, mask build, terminal value, mask apply, and
  action stack; r1/r2 exact identity held, but stability still failed.
- [x] Close the OPT-132AY remaining-builder-CPU attribution question enough to
  choose a direction. Decision: OPT-132AZ split the remaining broad CPU surface
  and showed exact same-work instability is still broad builder user CPU, not a
  single unmeasured final-observation/unroll bookkeeping bucket.
- [x] Superseded by OPT-132BD/BF/BK: tensor-native replay/sample proof passed
  but speed failed. Builder reshaping remains only an ownership-map-selected
  candidate, not active work.
  Historical evidence: OPT-132AZ exact no-sampler rows kept identity exact but
  wall still failed (`217.391s / 180.892s`, `18.329%`). Builder group-loop
  process CPU moved `80.37s / 61.70s`; unroll-fields CPU moved
  `42.13s / 32.24s`; terminal metadata CPU moved `15.58s / 11.44s`; prepare
  CPU moved `10.44s / 8.33s`. Local toy proof showed learner-ready replay plus
  flat row-index gather can remove a surface in isolation; BD/BF then proved the
  replay/sample lane was not sufficient. Do not reopen it unless the ownership
  map explicitly selects table-maintenance mechanics in the real loop.
- [x] Bound the broad OPT-132Y r3 common runtime slowdown enough to choose the
  next diagnostic target. Decision: OPT-132Z put the swing inside measured
  sample-gate cadence steps, and OPT-132AA showed broad per-call sample-gate
  cost centered on stable high learner-batch-build p50/p95. OPT-132AB then put
  the builder swing inside group-loop work, but still no speed claim and no
  blind builder patch.

Historical OPT-126 through OPT-131 items are complete and superseded by
OPT-132 repeatability work. Do not use them as current instructions.

## Decision Rule

- Only after an architecture candidate has local/toy proof and exact H100
  same-work repeat evidence, if a repeated accepted-fast-path H100 row beats
  OPT-104 and repeats under the accepted 180/45 denominator, close the speed
  goal.
- If a long-window diagnostic row is fast, do not close the speed goal unless a
  matched long-window baseline also exists.
- If repeatability fields differ, fix the source of drift before optimizing.
- If repeatability fields match but `1084/270` timings swing, treat the largest
  moving bucket as evidence about architecture pressure, not as an automatic
  instruction to add another timer. OPT-132AZ shows broad user-CPU
  learner-batch builder work under exact identity; followups also put env/search
  fixed-buffer architecture back in scope as toy/falsifier lanes.
- If long-window repeatability is stable, target the largest stable surface only
  after the dataflow map says that surface is the right ownership boundary.
- `724/180` already failed stability. Do not repeat it unchanged. Use the
  completed OPT-132AZ `1084/270` r1/r2 no-sampler remaining-builder-CPU packet
  as the current timing-instability evidence. Do not add another diagnostic
  unless it is required to prove a structural architecture candidate.

## Do Not Reopen

- Legacy OPT-124 inner two-phase owner replay as a standalone lane; the selected
  owner-search path already wires/proves compact Torch inner two-phase device
  replay.
- OPT-123 rerun without a code change.
- More small replay-sampling caches.
- More tiny owner-process copy/proof edits.
- More owner process/inline/threaded boundary variants.
- Blind GPU mechanics ports, bigger-GPU denominator changes, multi-GPU,
  scalar-ref, snapshot-file transport, compile default, or the OPT-128 serial
  inline owner path. PufferLib/EnvPool/Sample Factory/Isaac-style systems and
  MCTX/JAX-style search are allowed only as bounded architecture research and
  toy-ceiling experiments before any port.
