# Optimizer Operating Patterns

Date: 2026-06-08

Repo-root `goal.md` is the compass. This file is the behavior contract.

## Start Here

- Accepted baseline: OPT-104, H100, `12689.38` env steps/sec, `14.5255s`
  wall.
- Current best single H100 decision input: vectorized reset RNG plus
  `policy_refresh_interval=4`, `owner_search_threaded_proxy` background
  maintenance, transition-batch slab bypass, ring-batched replay append/cache
  refresh, tensor-native maintained replay, direct record-table building, and
  columnar direct append. The corrected columnar row ran `15852.67` env
  steps/sec and `46.7666s` wall, about `1.25x` OPT-104. It is not
  repeat-proven and not a `2x` or `10x` win.
- Local owner-loop proof for default-off one-simulation replay deferral is
  complete, and the one H100 probe has run. It passed proof but was
  speed-rejected: `13691.98 env/s`, `54.1467s`, slower than columnar r2
  `15852.67 env/s`.
- In-process async learner overlap has also run on H100 and was
  speed-rejected: `12954.74 env/s`, `57.2282s`, slower than columnar r2.
- Local direct-root resident-root-view proof is implemented and smoke-tested.
  It proves threaded direct-root owner mode can consume the source resident
  root handle with zero H2D/D2H and no host fallback, while preserving
  action-only/owner-replay/parent-row/maintenance closure. It is not speed
  evidence and does not yet remove parent root-batch construction.
- Local direct-root resident host-observation stub proof is also implemented
  and smoke-tested. It proves the same path can carry only a zero-stride
  shape-only host observation stub while search consumes the resident root
  handle: materialized host-observation bytes `0`, logical bytes `262144` in
  the last step and `3932160` total, resident-root-view proof preserved. This
  is a support gate, not a speed claim; the parent root-batch builder is still
  called.
- Local root-build-request proof is now implemented and speed-row-guarded for
  the real threaded owner proxy. Smoke
  `opt132-local-owner-rootbuildrequest-threaded-directtransition-smoke-20260607-r3`
  passed with publish/resolve/owner-build `15/15/15`, parent builder
  used/calls `false/0`, request observation bytes `0`, resident-root/stub/
  action-only gates preserved, parent rows `0/0`, search payload bytes `0`,
  direct replay `3/12/3`, and clean maintenance/policy lag. This is local
  proof, not speed evidence.
- H100 root-build-request evidence is proof-clean but speed-rejected:
  `11327.75 env/s`, `65.4477s`, below both OPT-104 and columnar r2. Parent
  root-build work is gone (`0.0s`, builder calls `0`), but parent wait, replay
  append, learner train, worker search, observation, and slab wall all got
  worse. Do not repeat unchanged.
- Owner-local transition derivation is proof-clean but speed-rejected as a
  standalone lane. Valid H100 r4 ran `13265.51 env/s`, `55.8875s`, with parent
  outcome transport `0/0`, cache hits/misses `724/0`, checksum
  verified/mismatch `724/0`, and fallback/pending/drop `0/0/0`. Preserve the
  counters as support; do not repeat unchanged.
- Fixed-action-tape owner-buffer probes are support-only. Mechanics-only
  B1024/m724/w180 was slower (`0.843x`); rendered slab/replay B128/m180/w45
  was a small toy win (`1.129x`) but has no learner/GPU/H100 claim.
- The profile-loop `compact_owner_action_step_boundary` slice has landed
  locally and is now speed-row-guarded. It owns the cached next action, verifies
  `manager.step()` applied the same action, requires the slab next action, and
  projects strict counters/failure reason. It is proof-only.
- The direct-root/action-only binding for `compact_owner_action_step_boundary`
  is now locally closed. Focused test
  `test_hybrid_profile_owner_action_step_boundary_uses_direct_root_build_request`
  forces the legacy parent root builder to raise, then proves the boundary
  through the real threaded direct-root build-request owner path with parent
  builder `0`, owner build used, resident-root/stub proof, zero host/search
  payload transport, parent rows `0/0`, and action-feedback checksums. It is
  still proof-only.
- The first mechanics/action/pending-state upstream step-frame rungs are now
  local-clean but still proof-only. `compact_owner_mechanics_step_boundary`
  bypasses parent `_make_compact_batch` with a borrowed
  `CompactOwnerMechanicsStepViewV1`, the direct-root fixed-slot path publishes
  and validates owner-assembled dense next actions so parent dense sidecar
  reconstruction is avoided, and the direct owner stepper avoids storing
  parent compact-batch/mechanics-step-view or root-batch sidecars in pending
  hot state while retaining action-step identity handles. The direct-root
  returned step also no longer materializes a parent request-derived root
  sidecar; telemetry and owner-local transition outcome validation read the
  root-build request instead. Owner/proxy transition closure now also lets the
  proxy close previous transitions from cached action frames and current
  root-build `joint_action`, while parent previous-transition stage/flush/
  commit helpers can be monkeypatched to raise. Owner action dispatch handles
  split direct-root action search into submit/resolve; submit returns without
  `worker.result()` or fixed-slot reads. The follow-up profile-loop overlap
  slice below now uses those handles across real parent post-slab work, but
  this mechanics/action/pending-state paragraph remains support-only by itself.
  In proxy-closure mode, the direct stepper no longer stores the parent
  action-step identity handle in `_pending`; the proxy action frame is the
  previous-action authority. Focused tests prove parent
  compact-batch/root builders, parent dense-action reconstruction, pending
  compact-batch/root-batch sidecar storage, returned root-sidecar
  construction, parent previous-transition closure, and dispatch submit waiting
  can be rejected while the direct-root owner path passes. Do not launch H100
  for these rungs alone.
- The next owner mechanics frame rung is now slot-backed locally. The manager
  writes `curvyzero_compact_owner_mechanics_step_frame_slot/v1` arrays in a
  four-slot ring, publishes slot/generation/digest handle proof, and the direct
  owner root request builder reads those slot arrays instead of the parent
  `compact_root_build_request_v1_from_batch()` helper. Local proof poisons the
  legacy mechanics step-view builder, parent compact-batch builder, parent
  root request helper, and parent dense-action reconstruction. Speed-row,
  Modal, and evidence summaries project the new fields and fail closed on
  missing/dirty mechanics-slot proof. This is still not H100 speed evidence;
  it is the guard for the next broader root/search transaction or learner-
  publication ticket/ref patch.
- The first manager-level action-dispatch overlap slice is now local-clean and
  launcher-wired. `compact_owner_action_dispatch_step_overlap` makes the
  hybrid profile loop submit direct-root owner action search, run existing
  parent post-slab payload/snapshot work, then resolve before returning the next
  joint action. Proof must show enabled/proof_passed, supported/used,
  submit-no-wait true, sync-wrapper false, cumulative slab/owner sync-wrapper
  counts `0`, completed-at-submit count `0`, submit/resolve counts equal
  measured iterations, pending `0`, max pending positive, parent-work overlap
  positive, and wait-in-submit `0`. These fields must be present on the raw
  profile payload before proof projection/defaulting. This is not speed
  evidence until a same-work H100 row proves it.
- Resident terminal final-observation host allocation is now elided in the
  resident-observation/no-scalar-timestep path. Sparse resident device final
  rows remain the terminal sidecar, while dense host `final_observation`
  allocation is skipped and counted. This is mechanics/observation support, not
  speed evidence.
- Immediate active task: build from the owner-slot device-row/unroll-2 proof
  into production fixed resident row/window slots or handle-ring sampling.
  Keep the action-step, mechanics step-view, owner-published dense-action,
  pending identity-handle, returned root-sidecar-avoidance, owner/proxy
  transition-closure, and action-dispatch handle rungs only as fail-closed
  guards. H100 rows must reject missing boundary proof, parent replay rows,
  search payload bytes, parent compact-batch/root-builder fallback, parent
  dense-action reconstruction, pending compact-batch/root-batch sidecar
  storage, returned root-batch sidecar construction, parent previous-transition
  closure, dispatch wait-in-submit or sync-wrapper-only overlap claims, host
  observation/outcome transport, action mismatch, owner-local fallback/pending
  transitions, terminal/final sidecar drift, fake terminal samples, and any
  learner batch that silently materializes parent rows before any speed claim.
- The guarded overlap/proxy H100 read is closed as support: proof was clean
  but the row was slower than columnar r2 (`15541.95 env/s`, `0.980x`
  columnar). Do not rerun overlap unchanged.
- Owner-slot whole-loop fixtures are local boundary guards, not speed evidence.
  The latest nonterminal fixture closes mechanics/root/action, device replay
  index rows, real ring append/sample metadata, and a resident device learner
  unroll-2 batch. Production H100 remains gated on fixed resident row/window
  slots or handle-ring sampling that changes the production owner denominator.
- Treat lazy selected direct table build as rejected/default-off: it reduced
  append time but slowed the whole row to `11179.26 env/s` by moving work into
  owner train/sample and reintroducing `352` direct group objects.
- Do not launch owner-search direct-replay rows with
  `--compact-owned-accepted-fast-path-preset`; that preset overwrites
  owner-search flags. Spell the accepted shape manually for these rows.
- Do not spend another turn trying to prove reset RNG, cadence-only refresh,
  MPS placement, eager append pre-drain, or same-process async learner overlap
  as primary speed paths. They are answered enough for the current decision.

## Historical Start Before r29-r34

- First owner-search H100 win, not stable: r14
  `opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-warmupgate-b1024a1-normal-unroll2-r14-20260605`
  ran `13497.30 env steps/sec`, `13.6561s` wall using shared-model refresh and
  no inline host payload clone. r15 exact repeat failed speed at `10502.70`,
  `17.5498s`. Treat r14 as a useful fast row, not a promoted baseline.
- Latest clean preset rows: OPT-132-G beat OPT-104, `13649.29` env steps/sec
  and `13.504s` wall; OPT-132-H failed the repeat, `11366.84` env steps/sec
  and `16.216s` wall; OPT-132-I beat OPT-104 again, `13308.59` env steps/sec
  and `13.8497s` wall under the corrected signed-checksum guard; OPT-132-J
  passed the same guard but ran slow, `10057.98` env steps/sec and `18.3258s`
  wall.
- Best recent improvement: OPT-132-F recovered about `0.833s` wall versus
  OPT-132-E by storing resident terminal final observations sparsely.
- Active task: wire/prove fast owner-search train sampling and learner-batch
  materialization, then rerun longer.
  Maintained tensor-native replay and maintained sample-universe sampling
  passed proof and cut sample gate; OPT-132BK locally closed the
  search-feedback slab/replay/sample toy gate; owner-search then removed parent
  replay/search row transport locally. Proof hardening, compact Torch inner
  two-phase device replay, explicit drain semantics, real-learner scale, and
  mock-fast ceiling, MPS placement, policy-lag proof, eager append
  pre-drain, and in-process async learner overlap are all answered locally.
  Real-learner scale fails at about
  `24-25 env steps/sec`; mock-fast scale passes at `214.06 env steps/sec`.
  MPS placement, eager append, and in-process async learner overlap all
  preserve proof and fail whole-row speed. R14 supersedes the old learner
  resource/work-shape target by winning once on H100 through shared-model
  refresh and no inline host payload clone. But r15 failed the repeat and r16
  longer row exposed owner train sample cost (`52.807s` sample vs `3.236s`
  update). r17 showed the existing fused/tensor-native learner-batch flags are
  gated behind compact-owned-loop entrypoint and do not compose with
  owner-search inline. Full-loop H100 speed has a first fast row, but stable
  speedup is still unproven.
  Same-work H100 remains a promotion gate, not an exploration loop.
- Current target: architecture feasibility across env/search/replay/sample/
  learner ownership. The five-row `724/180` A/A
  packet failed hard. Latest-frame resident replay snapshots fixed the
  immediate `1084/270` memory fit, but exact r1/r2 timing still failed
  stability. Three `1084/270` rows with learner-batch sub-timers passed with
  exact identity, but r3 broke timing stability. Three OPT-132Y CUDA-sync rows
  also passed exact identity and sync proof, but r3 slowed broadly. Runtime-step
  envelope stats then showed OPT-132Z r1/r2 failed inside measured loop
  iterations, concentrated on sample-gate cadence steps. Sample-gate per-call
  child distribution stats then showed OPT-132AA r1/r2 had broadly high
  sample-gate p50/p95 and stable high learner-batch-build p50/p95. OPT-132AB
  r1/r2 read builder-child per-call attribution on H100: exact identity held,
  but stability failed inside builder group-loop work, especially unroll fields
  and terminal metadata. OPT-132AC then added deeper unroll and
  terminal-metadata child timers and read them on H100 after two pre-row Modal
  `RESOURCE_EXHAUSTED` launches: r1c/r2/r3 matched exact identity and
  violations `[]`, but still failed timing stability. OPT-132AE showed builder
  group-loop cost is mostly named child work, not residual. OPT-132AF guarded
  unroll-2 specialization kept exact r1/r2/r3 identity and clean proof, but
  wall spread was `9.94%` and major sample/builder buckets exceeded `10%`; AF
  is rejected as a stable speed claim under exact `1084/270`. OPT-132AG then
  reran the same AF path with GPU sampling, kept exact r4/r5/r6 identity and
  clean proof, broke the monotonic slowdown pattern, but still failed
  stability with wall spread `6.97%`, sample gate `11.57%`, and builder
  group-loop `15.51%`. OPT-132AH then read exact `1084/270`
  with GPU sampling and the generic builder kept identity but drifted harder,
  with wall spread `34.28%`, sample gate `48.55%`, and builder group-loop
  `52.75%`. The sampler does not support a simple power/throttle explanation,
  and the remaining swing is not AF-specific. OPT-132AI then let runtime-step
  envelope diagnostics run without CUDA-sync timing probes. OPT-132AJ used that
  path on H100; removing sync probes narrowed the generic packet versus AH, but
  exact runtime-step-only rows still failed stability with wall `10.49%`,
  sample gate `12.72%`, and builder group-loop `14.25%`. OPT-132AK then added
  same-work wall-swing attribution and runtime-step cadence fields. OPT-132AL
  used them with GPU sampling disabled; exact generic rows still failed with
  wall/runtime-step `21.70%`, sample gate `22.84%`, and late sample-gate
  `31.11%`, so the sampler is not the simple cause. OPT-132AM then added
  chronological active sample-gate distributions by measured third. OPT-132AN
  read those fields on H100 and showed active sample-gate p50/p95 rises
  early/mid/late inside each row and is elevated across r1 versus r2, so the
  branch is broad active-call slowdown, not only a few late spikes. OPT-132AO
  then read bounded per-call replay-state trace records on H100 and showed the
  exposed replay/sample state is identical across r1/r2 while timing still
  swings. OPT-132AP then read below that state with per-call allocator/runtime
  snapshots plus deeper builder child timings; AP r1/r2/r3 still failed
  stability, but every CUDA allocator/memory counter matched exactly across all
  `135` calls. OPT-132AQ then added actual Python GC collection counters; AQ
  r1/r2 still failed stability, and the collection counters did not explain
  the swing. OPT-132AR then added process/thread CPU-time attribution; AR
  r1/r2 still failed stability, but about `92%` of the sample-gate and
  learner-build wall deltas were backed by CPU time. OPT-132AS then read the
  resource-usage split on H100: r1/r2 still failed stability, and the delta was
  mostly user CPU rather than system CPU, page faults, or context switches.
  OPT-132AZ is the latest read in that chain: exact identity still held, but
  wall spread remained `18.329%` and the moving work was broad builder user
  CPU, with group-loop process CPU `80.37s / 61.70s`, unroll-fields CPU
  `42.13s / 32.24s`, terminal metadata CPU `15.58s / 11.44s`, and prepare CPU
  `10.44s / 8.33s`.
- Resident stack shift is too small to chase now: about `0.012s` to `0.015s`.
- H100 utilization is low. Bigger GPUs and multi-GPU are parked as speed moves.
  GPU mechanics is not a blind port, but a GPU-resident mechanics toy is allowed
  as an architecture ceiling. H200/B200 are allowed only as explicit
  memory-headroom diagnostics.

## Read Order

1. repo-root `goal.md`
2. `docs/working/optimizer/reorientation_2026-05-23/ARCHITECTURE_RESET_2026-06-04.md`
3. `docs/working/optimizer/reorientation_2026-05-23/ORCHESTRATION.md`
4. `docs/working/optimizer/reorientation_2026-05-23/TASK_BOARD.md`
5. `docs/working/optimizer/reorientation_2026-05-23/TODO.md`
6. `docs/working/optimizer/reorientation_2026-05-23/FOLLOWUPS.md`
7. `docs/working/optimizer/reorientation_2026-05-23/SUBAGENT_DELEGATION.md`
8. `docs/working/optimizer/reorientation_2026-05-23/CURRENT_STATE.md`
9. `docs/working/optimizer/reorientation_2026-05-23/NEXT_MOVES.md`
10. `docs/working/optimizer/reorientation_2026-05-23/MEASUREMENT_LEDGER.md`

Everything else is archive unless a claim is copied into those files.

## Speed Rules

- Same-work H100 rows decide speed.
- Local tests prove correctness only.
- Profile-only rows prove contracts only.
- Tiny, local, profile-only, short-window, or one-off fast rows are not speed
  wins.
- A row is comparable to OPT-104 only if it carries the accepted-fast-path
  proof fields.
- If flags differ, label the row diagnostic and do not compare it as an
  accepted speed row.
- If an overlap row reports sync-wrapper true, submit wait-in-submit nonzero,
  pending nonzero, submit/resolve counts below measured iterations, or parent
  overlap work zero, reject the row before reading speed.
- Do not add diagnostics to accepted speed rows unless measuring diagnostic
  overhead is the point.
- Do not rerun a losing row unchanged.
- After two losing H100 rows in one lane, park the lane unless a new measurement
  reopens it.
- Fast rows do not become the new baseline while a same-preset repeat still
  shows an unexplained slow swing.
- Repeatability checksums are allowed to be signed. Required checksums must be
  present and nonzero; counters must be positive.
- Long-window rows are stability diagnostics unless a matched long-window
  baseline is also produced. Do not use them to replace OPT-104 by accident.

Accepted-fast-path proof fields:

```text
B1024/A1
normal death
180 measured / 45 warmup for accepted-speed rows
724 / 180, 1084 / 270, or 1444 / 360 only for labeled stability diagnostics
sample interval 8
sample batch 512
replay capacity 4096
learner unroll 2
refresh interval 4
compact_torch_search_service
direct_core
fused learner batch
prebuilt learner batch used
borrowed single-actor render state
render-state copy steps 0
if OPT-132AF is requested:
  compact_muzero_learner_batch_unroll2_specialized_builder true
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested true
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count > 0
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used true
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count > 0
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count 0
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason none
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl unroll2_specialized_v1
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path unroll2_specialized
lean trainer step
terminal sample/target rows 167/167
long diagnostic terminal sample/target rows positive and equal
normal-death gate true
truncations 0
host fallback 0
direct autoreset count 0
```

Long-window diagnostic decision rule:

```text
Use stability_724_180 only as a bounded bridge/proof row after memory patches.
Use stability_1084_270 as the long-window repeatability test.
Do not rerun any failed or unstable window unchanged.
```

Historical memory failures: `stability_1084_270` and `stability_1444_360` OOMed
before bounded diagnostics, and the first bounded `1084/270` rerun OOMed before
latest-frame resident replay snapshots. Current state: `1084/270` now fits on
H100. Do not rerun old failed rows or `724/180` unchanged. That historical
blocker pushed the architecture reset; the active blocker is now real owner
learner-update placement/cost.

Bounded long-window diagnostic mode:

```text
--compact-profile-bounded-diagnostics
```

The accepted stability windows enable this automatically. It keeps scalar proof
fields but avoids retaining the slab's full committed-row history and omits the
full nested source profile payload.

That patch was not enough by itself. The latest-frame resident replay snapshot
patch solved the immediate `1084/270` fit problem. OPT-132AZ bounded the
remaining movement to broad sample-gate learner-batch builder user CPU. BD/BF
then proved maintained tensor-native replay plus maintained sample-universe
sampling, cutting sample gate to `4.116s` but leaving full-loop speed at
`5362.68 env steps/sec`. Replay/sample is closed as the current P0 surface;
OPT-132BK closes the local slab/replay/sample handoff proof; owner-search
action-only replay ownership is the selected candidate and has passed local
real-entrypoint, proof-hardening, real-learner scale, and mock-fast ceiling
proofs. The active H100 path now adds transition-batch transport plus
ring-batched owner replay append/cache refresh. The next active work is cutting
the next object/materialization surface while preserving the selected
candidate's proof.

Latest memory-stability patch: bounded diagnostics now use latest-frame
resident replay snapshots. Accepted `180/45` speed rows still use full-stack
resident replay snapshots. The H100 `1084/270` diagnostic now fits, so OOM is
not the active blocker. The active blocker is remaining owner/replay/sample
materialization and hot-loop object traffic, not another missing timer.

Latest timing-stability read: OPT-132AZ r1/r2 exact generic no-sampler
remaining-builder-CPU diagnostics matched identity and proof fields, but timing
still failed. Wall/runtime-step spread was `18.329%` of median, sample gate
moved `136.622s / 109.260s`, learner-batch build `85.001s / 65.049s`, and
builder group-loop `84.177s / 64.278s`. GPU sampling and CUDA-sync probes were
both off. AP already showed exposed per-call trace state and every CUDA
allocator/memory counter matched exactly; AQ ruled out Python GC collections;
AR/AS showed the remaining sample-gate/build delta is user-CPU-backed; AZ
showed that CPU is broad builder work. Stop adding pure attribution timers. The
current target is the selected owner-search handoff: move or overlap the real
owner learner-update/final-drain tail while preserving BK-style action-feedback,
normal-death, zero-parent-row, and replay/sample proof fields. Keep the
env/search/replay/sample/learner map current while doing this.
Latest local timing diagnostic tooling: OPT-132AZ added the remaining
builder-CPU split and projected it through source profile, speed-row summary,
Modal report, and comparison surfaces. OPT-132BD/BF then closed replay/sample
as the current P0 surface. OPT-132BI/BJ then closed rendered observation and
fixed-shape search/root locally. OPT-132BK closes slab/replay/sample locally.
Next local work is real owner learner-update resource/work-shape change. Local
MPS placement, eager append pre-drain, and in-process async learner overlap all
preserved proof but slowed or failed to improve the whole row, so another
device-placement probe, maintenance-scheduling patch, or same-process future is
not the main plan. Owner-search action-only proof hardening is no longer the
active lane except as a guardrail for that learner-boundary change. Do not
return to builder-surface changes unless this selected owner-search candidate
is falsified.
OPT-132AT's first H100 attempt found the image-level blocker:
`perf` was not on PATH (`compact_profile_cpu_perf_stat_available=false`,
return code `127`). OPT-132AU installed `linux-perf` and proved the
container-level blocker: `/usr/bin/perf` is present, but
`sys_perf_event_open()` returns `19 (No such device)` for `task-clock`.
External perf counters are unavailable here; do not retry perf unchanged.
Historical
path: OPT-132Y r1/r2/r3 first showed a broad CUDA-sync-row slowdown; OPT-132Z
reconciled wall with runtime-step sum; OPT-132AA showed broad sample-gate
per-call cost; OPT-132AB/AC localized the builder child movement to
group-loop/unroll/terminal work before later rows showed the branch was not
AF-specific.
OPT-132AC r1c/r2/r3 exposed deeper children on H100 and still failed stability:
wall `36.55%`, sample gate `37.32%`, learner-batch build `34.23%`, builder
group-loop `34.11%`, unroll fields `40.16%`, and terminal metadata `27.48%`.
R2/R3 clustered while r1c was slow.
OPT-132AD then added runtime phase/residual distributions for actor
env/autoreset, sample-gate residual/sync, and builder group-loop/sync. AD
r1/r2/r3 held exact identity but still failed stability: wall `24.11%`, sample
gate `30.48%`, learner-batch build `34.32%`, and sample-gate builder
group-loop `34.49%`. OPT-132AE then formalized the builder group-loop
accounted/residual split. AE r1/r2 held exact identity; both rows were slow,
with group-loop `113.971s -> 119.853s`, accounted child work
`95.872s -> 101.300s`, and residual `18.099s -> 18.553s`. Compare AC rows only
against AC rows, AD rows only against AD rows, and AE rows only against AE rows
because the diagnostic field/sync shape changed.

Pre-FunctionCall Modal launch failures are not rows. The launcher now writes
structured `launch.json` and modal-report artifacts for those failures; use them
to record launch/resource state, not timing. Reachable remote spawn failures
print structured `spawn_failed` payloads, and local/remote per-call prefix
parity is tested to avoid AC report drift.

Required proof fields for every bounded `1084/270` stability row:

```text
compact_profile_bounded_diagnostics true
source_profile_payload_embedded false
resident_replay_snapshot_mode latest_frame_history
retained resident-snapshot count and bytes visible in summary/report
accepted-fast-path violations 0
host fallback 0
render-state copy steps 0
```

## Implementation Rules

Before writing code, state:

```text
baseline row
target bucket
expected wall-time win
proof fields that must remain unchanged
```

Do not make a small patch unless it can plausibly move the current wall gap.
If the fix is structural, make the structural fix instead of nibbling.

Do not update many docs after every small row. Keep `goal.md`,
`CURRENT_STATE.md`, `TASK_BOARD.md`, `NEXT_MOVES.md`, and the ledger current.

Do not do another broad reorientation pass unless it ends in one of these:

```text
a dataflow map or toy-ceiling artifact
a code change that targets a chosen architecture surface
a same-work H100 row after local/toy proof
a comparison artifact from existing rows
a short edit to goal.md that changes the next action
```

If the answer is only "read more docs," stop and use the current `goal.md`
status check instead.

## Sidecars

- Use bounded sidecars for substantial optimizer work.
- Reuse existing sidecars when the thread limit is full.
- Give each sidecar one concrete question.
- Keep at least one sidecar aimed at critique of the active path and proof
  standard when the work is substantial.
- Keep implementing locally while sidecars run.
- Do not wait on sidecars unless the next local action depends on them.
- Sidecars can critique the active path. They cannot create a new lane without
  measurement evidence.

## External Patterns

Use external systems as checks, not as ports.

- AlphaZero/MuZero-style systems separate actors, search/inference, replay, and
  learner with clear handoff and model refresh.
- PufferLib-style systems emphasize fixed buffers, vectorized envs, and small
  data movement.
- The lesson here is fixed resident data and clean batching first, not a GPU
  mechanics rewrite or a framework port.

## Current Ownership Rule

As of 2026-06-09, the active blocker is production replay/sample/learner-batch
ownership, not another local proof that device replay rows can be built.

- The fastest support row already removed several narrow transports, but still
  leaves parent wait, worker search, replay append, learner train, and parent
  mechanics/root sidecar work.
- The guarded action-dispatch overlap plus owner/proxy transition closure H100
  read is closed unchanged: r3 was proof-clean at `15541.95 env/s`, slower than
  columnar/direct-table r2 `15852.67 env/s`. Do not repeat that stack unless a
  new owner-boundary implementation changes the hot data movement.
- Replay/sample removal alone projects to `1.9625x`, just short of `2x`, so
  replay-only work is support unless paired with a non-replay boundary move.
- The next real patch must make production sampling look like a mature fixed
  buffer system: owner-issued fixed row/window slots or handles, lifetime/
  generation/digest proof, learner consumption of the handle, and an explicit
  materialized-parent fallback counter.
- The first step-frame handle rung is support, not victory: a
  slot/generation/digest handle is now published and owner-verified, but the
  rich mechanics view object still exists. Do not run H100 for handle identity
  alone; the next code must move actual replay/sample/learner batch data or
  learner publication state behind fixed handles.
- Slot-backed owner-frame proofs must fail before side effects. The current
  guard validates slot/generation/digest/live generation before replay commit,
  proxy closure, or search submit; future owner-boundary rungs should keep this
  ordering and use pending-sentinel style tests when a stale handle could
  otherwise mutate owner or parent state.
- Any proof field that says a parent object is not stored must have a private
  state test against the actual pending dataclass or queue. The root-action
  context patch fixed a false-clean root-build-request storage claim by making
  pending dispatch objects physically lack a `root_build_request` field.
- Final idle owner/proxy metadata is not the same thing as per-step proof. When
  closing/draining a service, preserve per-step handle identity fields before
  merging idle final-state defaults such as "no handle currently active."
- A slot-started transaction must fail closed instead of using compatibility
  fallbacks. The root-action-context handle rung now rejects missing/malformed
  handles and forbids falling back to a parent `CompactRootActionContextV1`.
- When there are multiple pending layers, name the layer. The owner
  root/search transaction patch made `compact_batch` absent from the direct
  stepper's transaction pending object, while still acknowledging that
  earlier rungs still had separate action-context storage surfaces. Do not
  generalize a clean counter from one pending object to the whole loop.
- Root/action context handle cleanup is support, not a speed lane. The next
  H100 gate needs a broader measured owner-surface change: fixed owner
  replay/sample/learner-batch, a coarse owner actor/search transaction tranche,
  or learner publication/version tickets.
- `compact_owner_minimal_step_payload_snapshot` is support only. It may be used
  to prove parent snapshot thinning under `compact_owner_action_step_boundary`,
  but it is not a speed lane and must not trigger H100 by itself.
- A replay flat maintained table or batched columnar append can be a paired
  lane after the owner step-frame proof starts, but not the sole main task.

## Latest Diagnostic Pattern

- External `perf` is unavailable in the current Modal container: OPT-132AU
  installed `/usr/bin/perf`, but `perf_event_open` returned ENODEV for
  `task-clock`. Do not retry external perf unchanged.
- In-process CPU-time/resource diagnostics have done their job for the active
  `1084/270` stability problem.
- OPT-132AV showed the moving builder group-loop bucket is mostly process CPU:
  `+14.88s` CPU against `+16.287s` wall.
- OPT-132AW is the prior H100 read on that target: final-observation
  branch/storage proof was identical across exact repeats and group-loop
  residual CPU was mostly stable after the deeper split. That target was
  superseded by AX/AZ and then BD/BF/BK; the live work is real owner learner
  update placement/overlap before spending H100.
- OPT-132AX is the pattern for acting on a diagnostic and then staying honest:
  it cut only the discarded grouped-learner materialization, kept sample-path
  materialization unchanged, added proof counters, and made emitted
  terminal-final-observation proof exact when present. H100 proved the cleanup
  worked but stability still failed hard, so do not keep polishing
  final-observation materialization or unroll subfields. AX/AY/AZ/BD/BF
  supersede builder-child work as the active lane.
- OPT-132AY is prior evidence on that target: the unroll sub-CPU split
  projected under exact identity, but stability still failed.
- OPT-132AZ is the current H100 read: it split the remaining builder CPU and
  still found broad exact-work user-CPU instability. Treat learner-batch builder
  reshaping as one candidate lane inside the architecture reset, not the whole
  strategy and not another pure attribution timer.

## Parked Lanes

- Owner process/inline/threaded boundary variants; the selected owner-search
  action-only threaded/background overlap falsifier has already been measured
  and still left the learner-update tail dominant.
- OPT-124 inner two-phase owner replay.
- More tiny replay/cache/copy/proof edits.
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
- Result-payload polishing.
