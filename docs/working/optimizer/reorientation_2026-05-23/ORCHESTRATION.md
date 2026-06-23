# Optimizer Orchestration

Date: 2026-06-09

This file says how to move without drifting.

## Order

1. Read repo-root `goal.md`.
2. Confirm OPT-104 and the latest valid H100 measurement.
3. Work the one active task in `TASK_BOARD.md`.
4. Use `SUBAGENT_DELEGATION.md` to choose bounded sidecar questions while the
   main thread keeps implementing.
5. Run local correctness checks.
6. Prove architecture feasibility locally or with toy ceilings before speed
   patches. Use measurement stability as a promotion gate, not exploration.
7. Record sidecar outcomes in `FOLLOWUPS.md`, and record decisions in the
   ledger plus active docs.

## Current Measurement

```text
baseline: OPT-104 validation-only, 12689.381637 env steps/sec, 14.5255305s wall
selected support stack: vectorized reset RNG + refresh interval 4 + owner_search_threaded_proxy background maintenance + slab bypass + transition-batch direct replay + ring-batched append/cache refresh + direct table/columnar replay support
r32/r33/r34: repeated modest wins over OPT-104, not 2x
direct-table r3: 15681.59 env/s, 47.2768s wall, about 1.24x OPT-104
columnar r2: 15852.67 env/s, 46.7666s wall, about 1.25x OPT-104, fastest single input, not repeat-proven
fixed-SoA exact r8: 11616.45 env/s, below OPT-104; scatter across about 373 selected records per 512-row sample
fixed-SoA locality r2: 10428.59 env/s; selected records collapsed to 62 but whole row regressed
whole-owner-buffer ceiling review: fastest columnar r2 projection 24903.25 env/s, 1.9625x OPT-104, still 0.558s short of 2x
one-simulation deferral local r7: ok=true, direct proof 12/12/12/12, owner-inner proof 16/16/16/16, crossed 0, D2H 0.0, owner-inner pending final 0
one-simulation deferral H100 r1: proof clean, speed rejected at 13691.98 env/s, 54.1467s wall; slower than columnar r2
async learner overlap H100 r1: proof clean, speed rejected at 12954.74 env/s, 57.2282s wall; slower than columnar r2
local resident-root-view r5: proof clean for direct-root threaded owner mode; required/proved true, H2D/D2H 0.0/0.0, host fallback false, direct root publish/resolve 15/15, parent rows 0/0
local resident-root host-observation stub r1: proof clean for the same direct-root threaded owner path; stub requested/stubbed true, kind zero_stride_shape_only_v1, materialized bytes 0, logical bytes 262144 last step / 3932160 total, resident-root-view proof preserved, parent rows 0/0
local root-build-request r3: proof clean for parent-builder avoidance through the real threaded owner proxy and speed-row report guard; publish/resolve/owner-build 15/15/15, parent builder false/0, request observation bytes 0, resident/stub/action-only gates preserved, parent rows 0/0, search payload bytes 0, direct replay 3/12/3
H100 root-build-request r1: proof clean, speed rejected at 11327.75 env/s, 65.4477s wall; parent builder false/0 and root build 0.0s, but parent wait/replay append/learner train/search/observation/slab regressed versus columnar r2
owner-local transition derivation r4: proof clean, speed rejected at 13265.51 env/s, 55.8875s wall; parent outcome bytes/fields 0/0, cache hits/misses 724/0, checksum verified/mismatch 724/0, fallback/pending/drop 0/0/0
fixed-action-tape owner-buffer probes: proof clean but support-only; mechanics-only 0.843x local profile, rendered slab/replay 1.129x toy with no learner/GPU/H100 claim
compact_owner_action_step_boundary: local profile proof landed and speed-row guard wired; enabled/proof/counts/failure fields are projected and fail closed for direct-root/action-only owner rows
selected-maintained gather H100: proof clean but speed-rejected at 12028.55 env/s, 0.759x columnar r2; per-record selected gathers are not the main lane
guarded overlap/proxy H100 r3: proof clean but speed-rejected versus columnar r2 at 15541.95 env/s, 47.7016s, 0.980x columnar r2
owner-slot local whole-loop fixture: corrected local denominator includes action source, observation, owner-slot work, and residual; nonterminal B2/M3 smoke passed with stage device builds/rows 3/12, ring append calls/records/rows 3/3/12, device-row sample true, resident device sample true, learner unroll-2 built true with 8 rows, host fallback false, parent replay objects 0, selected groups 0, production_speed_claim false
production fixed-SoA handle-consumption proof: owner_search_fixed_soa_replay now requests learner-batch handle-ring sampling; compact-owned learner boundary records resident-handle consumed true only after train_on_learner_batch, and exposes materialized-parent fallback count/reason
status: real progress, not 2x; active task is corrected local whole-loop timing for the production handle path, with resident-handle consumed true and fallback zero as guards, and no standalone H100 lane
```

## Historical Measurement Before r29-r34

```text
baseline: OPT-104, 12689.38 env steps/sec, 14.5255s wall
first owner-search H100 win: r14 shared-model/no-payload-clone, 13497.30 env steps/sec, 13.6561s wall
r14 status: real first win, repeatability/promotion audit pending
r14 caveat: policy_refresh_interval=1 vs OPT-104 interval=4; train/sample/update count remains 22 and refresh work is more frequent, not less frequent
fast row: OPT-132-G, 13649.29 env steps/sec, 13.504s wall
failed repeat: OPT-132-H, 11366.84 env steps/sec, 16.216s wall
latest clean slow repeat: OPT-132-J, 10057.98 env steps/sec, 18.3258s wall
long diagnostic: 1084/270 fits on H100 but failed timing stability
latest sync diagnostic: OPT-132Y r1/r2/r3 CUDA-sync timing exact identity, stability failed
latest runtime read: OPT-132Z r1/r2 exact identity, stability failed inside sample-gate cadence steps
latest per-call read: OPT-132AA r1/r2 exact identity, sample-gate cost broad across calls
latest builder-child read: OPT-132AB r1/r2 exact identity, stability failed inside builder group-loop
latest deep builder read: OPT-132AC r1c/r2/r3 exact identity, stability failed inside builder group-loop
latest runtime phase read: OPT-132AD r1/r2/r3 exact identity, stability failed inside sample-gate builder group-loop
prior builder loop accounting read: OPT-132AE r1/r2 exact identity, group-loop mostly named child work
prior guarded builder read: OPT-132AF r1/r2/r3 exact identity, clean unroll2 proof, stability failed monotonically
prior GPU-sampled guarded read: OPT-132AG r4/r5/r6 exact identity, clean proof, stability still failed
prior GPU-sampled generic read: OPT-132AH r1/r2/r3 exact identity, generic path, stability failed harder
prior local diagnostic tooling: OPT-132AM chronological active sample-gate cadence fields
prior local diagnostic tooling: OPT-132AO bounded per-sample-gate replay-state trace records
prior local diagnostic tooling: OPT-132AP per-sample-gate allocator/runtime trace fields
prior local diagnostic tooling: OPT-132AQ per-sample-gate Python GC collection counters
prior local diagnostic tooling: OPT-132AR per-sample-gate process/thread CPU-time counters
prior local diagnostic tooling: OPT-132AS per-sample-gate resource-usage counters
prior local diagnostic tooling: OPT-132AT CPU perf-stat wrapper/reporting; first H100 attempt found no perf binary
prior diagnostic failure: OPT-132AU perf installed but perf_event unavailable
prior local diagnostic tooling: OPT-132AW final-observation/residual split and branch/storage proof
latest H100 read: OPT-132AZ r1/r2 no-sampler remaining-builder-CPU generic exact identity, stability failed, broad builder CPU visible
latest local architecture proof: maintained tensor-native replay table real sample-gate path, source maintained_record_table_v1, 5.047x CPU median vs current; focused fallback and eviction proofs pass
latest H100 proof row: OPT-132BD accepted-preset `1084/270` maintained tensor-native replay passed fail-closed proof with violations `[]`, table source `maintained_record_table_v1`, reused records `1080`, missing records `0`, fallback count `0`; speed failed at `3884.88` env steps/sec and `285.727s` wall
latest H100 sample-universe row: OPT-132BF accepted-preset maintained sample-universe/tensor-native replay passed fail-closed proof, improved speed to `5362.68` env steps/sec, and collapsed sample gate from `74.005s` to `4.116s`; still only `0.42x` OPT-104
latest local search/root proof: OPT-132BJ fixed-action toy passes rendered observations plus fixed-shape root/action/search identity checks; local proof only, not H100 speed evidence
latest local slab/replay/sample proof: OPT-132BK closed search-feedback toy passes deterministic slab-selected action feedback, replay append, index-row, replay-ring, and sample-gate checks; local proof only, not H100 speed evidence
latest owner-search H100 proof: r14 preserves zero parent replay rows and zero search payload bytes, keeps train/sample/update closure at `22`, refreshes search `22` times with `0` skips, consumes the final train update, suppresses `44` warmup replay entries, and collapses final owner drain to `0.000042s`
historical active task: repeat and audit the r14 shared-model/no-payload-clone path. Do not reopen cadence-only refresh, MPS placement, eager append, same-process async learner futures, or old learner-resource split as P0 unless the r14 repeat fails and points there.
```

OPT-132-G and OPT-132-H both have the accepted fast-path flags:

```text
direct_core: true
fused learner batch: true
borrowed render state: true
render-state copy steps: 0
lean trainer step: true
terminal sample/target rows: 167/167
normal death: true
truncations: 0
host fallback: 0
direct autoreset count: 0
```

The speed goal is closer but still open because r14 is a single row. Do not
call r14 the new baseline until it repeats and the cadence/proof audit passes.

## Shape To Copy

Efficient AlphaZero/MuZero-like systems keep roles clear:

```text
actors -> batched search/inference -> compact replay -> learner -> refresh
```

For this repo:

- CPU actor side runs CurvyTron mechanics.
- GPU search side runs compact Torch inference/search.
- GPU learner side runs compact MuZero updates.
- Replay/sample and metadata are mixed and must stay fixed, compact, and
  resident where possible.

Do not infer "move game mechanics to GPU" from this shape. First map the env
state/observation/action/replay boundary and use fixed-action toy ceilings to
prove whether vectorized CPU, compiled CPU, or GPU-resident mechanics can remove
a whole surface while preserving semantics.

## Now

Next action:

1. Preserve the fastest candidate support shape: vectorized reset RNG, refresh
   interval `4`, `owner_search_threaded_proxy`, deferred/background owner
   maintenance, slab bypass, transition-batch direct replay, ring-batched
   append/cache refresh, tensor-native replay, borrowed render state, direct
   core, and normal death.
2. Do not rerun fixed-SoA exact/locality, direct-table, columnar, or threaded
   candidate rows unchanged. Each has already answered a narrower question.
3. The H100 deferral decision is closed: the row was proof-clean and
   speed-rejected. Do not repeat it unchanged.
4. Preserve the local resident-root-view and resident host-observation stub
   proof fields in direct-root owner mode when claimed, but do not treat them
   as speed evidence.
5. No H100 row for owner-slot, overlap, or handle-consumption support rungs.
   The next H100 gate requires corrected local whole-loop timing showing the
   production handle path moved a real owner surface while resident-handle
   consumed is true and materialized-parent fallback is zero.
6. The next patch must change owner graph or hot data movement. Pure timers,
   replay-only gather/locality tweaks, worker-mode polish, result-payload
   formatting, and unchanged overlap reruns are parked.
7. Keep repo-root `goal.md` rare/stable. Put row details and sidecar outcomes
   in `MEASUREMENT_LEDGER.md`, `FINDINGS_LOG.md`, `ACTIVE_RUN_REGISTRY.md`,
   `FOLLOWUPS.md`, and `SUBAGENT_DELEGATION.md`.

## Historical Now Before r29-r34

Next action:

1. Run one exact r14 repeat with the same flags, then compare against OPT-104
   and r14. Promotion-grade status requires the repeat to stay above OPT-104
   with exact owner-search proof fields and no new validation gaps.
2. If strict same-work cadence is challenged, run the same shared-model/no-clone
   path at `policy_refresh_interval=4`. Cadence-only without shared/no-clone is
   already falsified and should not be reopened.
3. Treat owner-search H100-readiness fields as preservation gates:
   action-feedback checksum/mismatch proof, arg-aware cadence checks, owner
   sample telemetry gates, request-vs-entry drain counts, finite final-drain
   timing, required action-only/two-phase handle fields, split owner
   train/ref-publication timing, submitted/owner/refreshed update closure, and
   positive in-row policy-lag evidence if overlap is claimed.
4. Treat shared-model-state refresh and no inline host payload clone as the
   selected r14 mechanism. Document `shared_model_version_token` versus real
   model digest so the proof is honest about same-object refresh.
5. Treat the five-row `724/180` packet as a hard stability failure:
   `89.353s -> 137.297s` wall with exact identity.
6. Treat OPT-132AZ `1084/270` no-sampler remaining-builder-CPU r1/r2 as the
   current evidence: exact identity held, but timing stability failed with wall
   spread `18.329%` of median and broad user-CPU movement under sample-gate
   learner-batch builder work.
7. Do not claim stable H100/full-loop speedup from a single row.
8. Do not rerun the same no-sampler cadence packet unchanged as a speed-code
   gate.
9. Treat OPT-132AA r1/r2 as the sample-gate per-call read: sample-gate p50/p95
   stayed broadly high and learner-batch-build p50/p95 stayed high/stable.
10. Stability bar: wall spread `<= 5%` of median and major-bucket spread
   `<= 10%` of median, with exact identity and no accepted-fast-path
   violations.
11. Treat OPT-132AB r1/r2 as the latest builder-child read: exact identity held,
   but wall spread was `45.57%`, sample gate `54.93%`, learner-batch build
   `56.09%`, and builder group-loop `56.27%`. No speed claim.
12. Treat OPT-132AC r1c/r2/r3 as the latest H100 deep builder read: exact
   identity and proof held, but wall spread was `36.55%`, sample gate
   `37.32%`, learner-batch build `34.23%`, and builder group-loop `34.11%`.
   R2/R3 clustered while r1c was slow. No speed claim.
13. If Modal rejects launch before a FunctionCall exists, use the structured
   launch-failure report or remote `spawn_failed` payload as non-speed evidence
   and do not count it as an H100 row.
11. Treat OPT-132AD r1/r2/r3 as the runtime phase read: wall spread `24.11%`,
   sample-gate builder group-loop spread `34.49%`, no speed claim.
12. Treat OPT-132AE r1/r2 as prior attribution evidence: both rows were slow,
   and the group-loop was mostly named child work rather than residual.
13. Treat OPT-132AF r1/r2/r3 as the rejected speed-patch proof packet: proof
   held, but the monotonic wall/sample/builder slowdown rejected the stable
   speed claim.
14. Treat OPT-132AG r4/r5/r6 as the prior exact AF H100 diagnostic read: proof
   held and GPU sampler fields worked, but stability still failed and the
   sampler did not show a simple power/throttle explanation.
15. Treat OPT-132AH r1/r2/r3 as the prior exact generic H100 diagnostic read: the
   generic builder also drifted badly, so the remaining same-work swing is not
   AF-specific.
16. Treat OPT-132AJ r1/r2/r3 as the prior exact runtime-step-only diagnostic read:
   runtime-step-only generic rows narrowed but did not solve timing stability.
17. Treat OPT-132AK as prior local diagnostic tooling that adds same-work
   wall-swing attribution plus runtime-step cadence summaries.
18. Treat OPT-132AM as prior local diagnostic tooling: chronological
   active sample-gate distributions by measured third plus slowest per-call
   iteration projection.
19. Treat OPT-132AL r1/r2/r3 as prior no-sampler H100 evidence: sampler
   removal did not solve stability and late cadence carried much of the swing.
20. Treat OPT-132AN r1/r2 as prior H100 evidence: active
   sample-gate p50/p95 rises early/mid/late inside each row and r1 is elevated
   across all thirds versus r2, so the branch is broad active-call slowdown,
   not only a few late spikes.
21. Treat OPT-132AO as prior exact H100
   read: bounded per-call trace records are present on r1/r2, and all exposed
   trace state matched across `135` calls despite a `33.142s` wall swing.
22. Treat OPT-132AP as prior allocator-exact H100 read: AP extends the AO trace
   with allocator/runtime snapshots and deeper builder child timings. AP
   r1/r2/r3 kept exact identity, but wall spread was `31.99%`; exposed
   replay/sample state and every CUDA allocator/memory counter matched exactly
   across `135` calls.
23. Treat OPT-132AQ as prior GC-stats H100 read: AQ adds real
   `gc.get_stats()` collection/collected/uncollectable counters. R1/R2 kept
   exact identity, but wall spread was `28.856%`; GC collection totals were
   nearly identical and same-GC-delta calls still carried `22.6695s` of
   sample-gate delta.
24. Treat OPT-132AR as prior H100 read: it adds process/current-
   thread CPU-time deltas for sample gate and learner-batch build. R1/R2 kept
   exact identity but wall spread was `13.479%`; about `92%` of sample-gate and
   learner-build wall deltas were backed by CPU-time deltas.
25. Treat OPT-132AS as prior H100 read: it adds process/thread
   resource-usage deltas for user/system CPU, page faults, and context
   switches around sample gate and learner-batch build. R1/R2 kept exact
   identity but wall spread was `14.616%`; sample-gate delta was mostly user
   CPU (`16.17s` process user CPU versus `0.49s` system CPU), with zero
   page-fault/context-switch deltas.
26. Treat OPT-132AT as prior local diagnostic tooling: it wraps the remote
   speed-row producer in `perf stat -x,`, parses CPU counter fields into the
   row/report/comparison surfaces, and fails closed if perf is unavailable or
   denied. First H100 attempt failed before a speed row because `perf` was not
   found in the image.
27. Treat OPT-132AU as closing the external perf-counter attempt: `perf` is
   installed, but `sys_perf_event_open()` returns ENODEV for `task-clock`
   before the producer runs.
28. Treat OPT-132AV as prior completed H100 read: it adds in-process
   builder-child process/thread CPU-time deltas. R1/R2 exact generic
   no-sampler `1084/270` kept exact identity and report violations `[]`, but
   wall spread was `15.03%`. Sample gate moved `114.928s -> 135.240s`,
   learner-batch build `66.229s -> 82.661s`, and builder group-loop
   `65.505s -> 81.792s`. Group-loop process CPU moved `+14.88s`; terminal
   metadata `+6.37s`, terminal final-observation `+2.67s`, residual `+4.28s`,
   and unroll fields `+2.76s`. No speed claim.
29. Treat OPT-132AW as prior completed H100 read: it splits group-loop
   prepare and terminal-value bookkeeping, terminal final-observation
   presence/select-current/gather, and adds final-observation branch/storage
   proof. R1/R2 exact generic no-sampler `1084/270` kept exact identity and
   report violations `[]`, but wall spread was `7.318%`: `250.183s /
   232.521s`. Sample gate was `157.724s / 146.441s`, learner-batch build
   `94.924s / 88.854s`, builder group-loop `94.002s / 87.987s`.
   Final-observation proof was identical: groups `399`, index fast path `0`,
   fallback `399`, final-row sum/max `512/4`, sparse storage `399`, sparse-row
   sum/max `3102/16`. Group-loop residual CPU was mostly stable after the new
   accounting (`14.45s / 14.07s`), while unroll-fields CPU
   `38.18s / 34.98s` and final-observation gather CPU `6.84s / 5.95s` still
   moved. No speed claim.
30. Treat OPT-132AX as prior completed H100 read: the grouped learner fallback no
   longer selects current observations or materializes final-next observations
   when that tensor is discarded. It validates resident final-observation mask
   and dense/sparse storage coverage instead, preserves sample-path
   materialization, adds validate-only/materialized proof counters, and makes
   terminal-final-observation proof counters optional identity fields in the
   comparator. Ruff passed and focused tests passed (`190` tests, `2`
   warnings). R1/R2 exact identity held, proof was exact, validate-only count
   was `399/399`, materialized count `0/0`, select-current/gather wall and CPU
   `0`, but wall spread was `43.864%`: `168.280s / 262.830s`. No speed claim.
31. Treat OPT-132AY as prior completed H100 read: it adds process/thread CPU
   deltas for unroll identity, stack fields, mask build, terminal value, mask
   apply, and action stack under the existing `unroll_fields` bucket. R1/R2
   exact identity held and stable speed claim stayed false, but wall spread was
   `25.819%`: `237.752s / 183.386s`. Unroll-fields CPU split from
   `43.86s / 34.14s` into stack `7.76s / 6.29s`, mask build
   `10.08s / 8.47s`, terminal value `8.51s / 6.55s`, mask apply
   `3.25s / 2.29s`, action stack `1.97s / 1.31s`, and identity
   `0.39s / 0.23s`. No speed claim.
32. Treat OPT-132AZ as latest completed H100 read: it splits remaining
   prepare, terminal-metadata, final-observation, and unroll residual/accounted
   CPU. R1/R2 exact identity held, but wall spread was `18.329%`:
   `217.391s / 180.892s`. Sample gate was `136.622s / 109.260s`,
   learner-batch build `85.001s / 65.049s`, builder group-loop
   `84.177s / 64.278s`, group-loop process CPU `80.37s / 61.70s`,
   unroll-fields CPU `42.13s / 32.24s`, and terminal metadata CPU
   `15.58s / 11.44s`. No speed claim.
33. Do not add another pure attribution timer. Accepted-preset proof projection
   for the maintained tensor-native replay table path is wired locally and
   tested, including the parent learner-ready cache proof. Keep the guarded
   learner-ready cache as a proof/plumbing lane, not the whole strategy.
34. Repeat exact `1084/270` only after owner-search scale has a plausible speed
   shape with fail-closed proof. The real-learner scale rows failed that bar,
   while mock-fast scale passed, so the next action is real learner placement/
   update work, not a pure replay, projection, attribution-only,
   mock-ceiling-repeat, or learner-tail-confirmation H100 row.

## Do Not Do

- Do not reopen owner-boundary variants, except the selected owner-search
  action-only threaded/background overlap falsifier.
- Do not chase the resident stack ring-buffer; G/H measured stack shift at only
  about `0.012s` to `0.015s`.
- Do not add GPU sampling to accepted speed rows.
- Do not start blind GPU mechanics, multi-GPU, compile-default, scalar-ref,
  local-process, snapshot-file, or result-payload work. PufferLib/EnvPool/
  Sample Factory/Isaac-style env systems and MCTX/JAX-style search are allowed
  only as bounded research and toy-ceiling experiments until a dataflow map
  chooses that lane.
- Do not use H200/B200 as speed baselines. They are allowed only as explicit
  memory-headroom diagnostics.
