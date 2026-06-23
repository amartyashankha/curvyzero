# Optimizer Sidecars

Date: 2026-06-09

Sidecars are critics and scouts. They are not the main work.

## Pattern

- Use two to four bounded sidecars for substantial optimizer work.
- Reuse existing sidecars when the thread limit is reached.
- Give each sidecar one question and one requested output shape.
- Keep the main thread on the next measurement, local check, or implementation
  while sidecars run.
- Do not wait unless the next local action depends on the answer.
- Use follow-ups only when new evidence changes the next step.
- Copy useful conclusions into working docs such as
  `ARCHITECTURE_RESET_2026-06-04.md`, `FOLLOWUPS.md`, `TASK_BOARD.md`,
  `NEXT_MOVES.md`, `MEASUREMENT_LEDGER.md`, or `FINDINGS_LOG.md`.
  Edit `goal.md` only when the north-star goal, baseline, selected
  architecture, active blocker, or decision gates genuinely change.

## Recent Delegations

- 2026-06-09 / Erdos and Rawls followups for the production fixed row/window
  handle gate. Erdos found the smallest useful code move: `owner_search_fixed_soa_replay`
  already reaches fixed-SoA append/sample code, but the speed-row metadata did
  not request the fixed-SoA learner-batch handle ring. Main thread wired those
  metadata flags and added `CompactOwnedLoopV1` learner-boundary counters that
  mark a resident batch handle consumed only after `train_on_learner_batch`
  runs with handle-ring metadata and zero fallback. Rawls identified where to
  record the proof and warned against stale owner-local-derivation,
  mechanics-step-frame, or selected-gather standalone wording. Validation:
  ruff passed; combined speed-row/owned-loop/real-ring packet `28 passed`.
  Decision: local contract proof only, no H100. Next task is
  corrected local whole-loop timing with resident-handle consumed/fallback
  counters clean.
- 2026-06-09 / Franklin, Leibniz, Parfit, and Euclid followups after the user
  asked for a hard reorientation and no more stale goal churn. Franklin
  recommended the owner-context device replay row builder; Leibniz recommended
  the nonterminal resident learner unroll-2 proof; Parfit/Euclid audited stale
  active docs. Main thread landed
  `build_compact_device_replay_index_rows_v1_from_owner_action_context_payload()`,
  wired the owner-slot service to emit device replay index rows, drained them
  through the real compact replay ring, and proved resident device unroll-2 on
  `scenarios/environment/source_kinematics_straight_multistep.json`. Evidence:
  `/private/tmp/curvy_owner_slot_device_rows_unroll2_b2_m3_w1_20260609.json`
  passed with stage device builds/rows `3/12`, ring append calls/records/rows
  `3/3/12`, device-row sample true, resident device sample true, learner
  unroll-2 built/rows `true/8`, host fallback false, parent replay objects /
  replay object entries / selected groups `0/0/0`, and production speed claim
  `false`; ruff passed; replay-contract plus benchmark tests `57 passed`;
  source-state hybrid selector `1 passed`. Decision: no `goal.md` edit and no
  H100 for this local proof. Next sidecar/useful task is production fixed
  resident row/window slots or handle-ring sampling, with lifetime/generation/
  digest proof and explicit materialized-parent fallback counters.
- 2026-06-09 / Russell, Faraday, and Chandrasekhar followups after the user
  requested a high-level Puffer/Isaac-style reset plus Amdahl audit. Russell
  confirmed the mature pattern: mechanics, search, replay, sample, and learner
  sit on fixed resident/shared buffers and exchange small handles/signals; this
  maps to Curvy as mechanics slots -> owner root/action request -> selected
  action -> staged owner transition -> replay/sample/learner handle. Faraday
  audited the in-progress ring-drain code and found the first failing contract:
  `_CompactReplayRingV1.sample()` entered the resident host-index path but the
  columnar step view lacked `joint_action`, action mask, reward, and done
  sidecars. Main thread chose the bounded green rung first: extend
  `CompactReplayColumnarAppendRecordV1` with optional sidecars, populate them
  from owner transition facts, drain staged owner rows into the real replay
  ring, and sample unroll-1 from `compact_replay_ring_resident_sample_gate`.
  Chandrasekhar reconstructed the Amdahl truth: columnar r2 is real `1.249x`
  (`15852.67 env/s`) but still needs about `17.55s` removed/overlapped for
  `2x`; replay alone cannot do that. Evidence:
  `/private/tmp/curvy_owner_slot_replay_ring_b2_m2_w1_20260609.json` passed
  with ring append calls/records/rows `2/2/10`, ring sample calls/rows `2/8`,
  observation-provider fallback `0`, failures `[]`, production speed claim
  `false`; ruff passed; replay-contract plus benchmark tests `56 passed`.
  Decision: no H100 for this rung. Superseded by the later device-row/unroll-2
  local proof above; the remaining useful task is production fixed resident
  row/window slots or handle-ring sampling with explicit materialized-parent
  fallback counters.
- 2026-06-09 / Rawls and Laplace followups after the user requested a broader
  PufferLib-style reorientation: Rawls confirmed the mature pattern is fixed
  buffers plus handles/signals, with mechanics/search/replay/sample/learner
  owned by resident/shared data surfaces and parent Python only coordinating.
  Laplace identified the missing bridge: the owner-slot path proved
  mechanics/root/action but still bypassed production replay because the real
  `stage_replay_append_entries()` hook was not driving `_CompactReplayRingV1`.
  Main thread responded with two local code moves: the owner-slot service now
  caches action payloads by replay handle, stages previous transitions through
  `stage_replay_append_entries()`, validates digests, releases staged payloads,
  and proves stage/sample counters in
  `/private/tmp/curvy_owner_slot_stage_replay_b2_m2_w1_20260609.json`; and
  `build_compact_replay_index_rows_v1_from_owner_action_context_payload()`
  now builds real `CompactReplayIndexRowsV1` from owner action context plus
  replay payload, matching the trusted root-batch builder in replay-contract
  tests. Validation: ruff plus replay-contract/benchmark packet `56 passed`.
  Decision: no H100 yet. Superseded by the later device-row/unroll-2 proof
  above; the remaining implementation is production fixed resident row/window
  slots or handle-ring sampling.
- 2026-06-09 / Huygens and Harvey followups after the Amdahl reset and
  owner-slot whole-loop correction: Harvey audited the planning docs and found
  stale overlap-H100 language in `TASK_BOARD.md`, `SUBAGENT_DELEGATION.md`,
  `ORCHESTRATION.md`, `OPERATING_PATTERNS.md`, `NEXT_MOVES.md`, and
  `CURRENT_STATE.md`. Huygens audited the code path and identified the true
  replay boundary: `CompactOwnerSearchDirectStepperV1` does not expose
  `committed_index_rows`; production replay/sample proof should use
  `stage_replay_append_entries()` / replay-payload handles, then drain rows
  into a replay ring and sample through real learner-batch handle metadata.
  Main thread first landed the local owner-slot fixture extension:
  corrected whole-loop timing, mechanics/root/action proof, local fixed
  replay-slot append, inline sample-handle checksums, and then the stage-hook
  lifecycle proof. Decision: local fixture is support. The next implementation
  follow-up is the real owner replay/sample/learner-batch handle path, not
  another overlap or root-build-request H100.
- 2026-06-08 / Tesla, Banach, and Herschel followups after the dispatch-handle
  local proof: main thread asked for the next non-random step before another
  H100 row. Tesla recommended the minimal honest manager-loop overlap: split
  the direct stepper into `submit_step()` / `resolve_step()`, submit before
  existing parent post-slab work, resolve before return, and prove no sync
  wrapper or wait-in-submit. Banach added fail-closed guard requirements:
  reject duplicate pending submits, pending close, stale action checksums, and
  fixed-slot reads before resolve. Herschel mapped the mature-system pattern
  back to fixed owner handles: if the overlap slice is too small on H100, the
  next work should be owner mechanics-frame handles or learner publication
  tickets/refs, not another local support rung. Main thread implemented the
  Tesla slice: `CompactOwnerSearchDirectStepperV1` now has
  `submit_step()`/`resolve_step()`, `HybridBatchedObservationProfileManager`
  uses them around real parent payload/snapshot work under
  `compact_owner_action_dispatch_step_overlap`, and the speed-row/Modal paths
  thread `--compact-owner-action-dispatch-step-overlap` with proof gates.
  Validation: focused source-state overlap tests passed, full owner-search
  service `55 passed`, replay-contract suite `47 passed`, full speed-row smoke
  `99 passed`, and ruff. Superseded 2026-06-09: the guarded H100 read is closed
  as support after the overlap/proxy row was flat against columnar r2. Do not
  rerun this overlap stack unchanged; move to production fixed replay/sample/
  learner-batch handles.
- 2026-06-08 / Tesla, Banach, and Herschel followups during the owner dispatch
  reorientation: main thread kept coding while sidecars audited the three live
  surfaces. Tesla identified the smallest honest owner wait/search dispatch
  cut: split direct-root owner action into submit/resolve handles, and do not
  claim speed until profile-loop work actually sits between them. Banach found
  learner publication cadence should become owner ticket/ref proof, not another
  unchanged async learner H100 lane. Herschel found the next mechanics/
  observation boundary should be owner mechanics-frame handles written by the
  existing actor mechanics path, not a private fixed env loop. Main thread
  implemented the Tesla slice locally: `CompactOwnerSearchSlabProxyV1` now has
  `submit_action_step_from_root_build_request()` and
  `resolve_action_step_handle()` around `CompactOwnerActionDispatchHandleV1`;
  the sync method is a wrapper; lazy threaded direct-root proxy forwards the
  split; proxy-closure mode avoids parent pending action-step identity storage.
  Validation: action-dispatch/direct-root/proxy packet (`5 passed`), profile
  boundary packet (`5 passed`), full owner-search service (`54 passed`),
  replay-contract suite (`47 passed`), and ruff. Decision: local support only;
  next code must either use submit/resolve across real profile-loop parent
  work, build fixed mechanics-frame handles, or move learner publication to
  owner tickets/refs.
- 2026-06-08 / Tesla and Banach followups during the rootless direct-root
  support rung: fresh spawns were blocked by the thread limit again, so the
  main thread reused existing sidecars and kept the code path local. Tesla's
  audit identified the exact proof/speed mismatch: direct-root mode avoided
  the parent root builder for owner search, but still built a parent
  request-derived root sidecar for returned-step telemetry; if that helper was
  monkeypatched to raise, the old path would fail. Banach agreed the true next
  gate is still owner/proxy previous-transition closure with applied-action
  verification, not merely moving code inside the parent direct stepper. Main
  thread implemented the prerequisite support cut: direct-root slab steps now
  return `root_batch=None`, telemetry reads `CompactRootBuildRequestV1`, and
  owner-local transition validation derives outcome sidecars from the request.
  Validation: request-outcome contract packet (`2 passed`), direct-root owner
  packet (`3 passed`), combined owner/profile packet (`6 passed`), full
  owner-search service (`50 passed`), and ruff. Decision: rootless direct-root
  return is closed-local support. The follow-up owner/proxy closure rung is now
  also closed-local support; next work must move a broader owner dispatch,
  learner-publication, mechanics/observation, or fixed-buffer boundary.
- 2026-06-08 / Tesla, Banach, and Mill followups during the pending-handle
  rung: fresh spawns were blocked by the thread limit, so the main thread reused
  live sidecars while keeping the code path local. Banach's architecture audit
  says the next local gate should prove fixed owner step-frame handles around
  pending root/action/transition state plus parent-wait/search-dispatch
  removal; learner cadence alone is not the first move. Mill's docs audit says
  `goal.md` should stay stable and the live docs should record the latest
  mechanics step-view, dense-action, and pending-handle rungs as support only.
  Main thread implemented pending compact-batch/mechanics-step-view and
  root-batch sidecar avoidance in the direct owner stepper and validated
  focused direct-root tests (`2 passed`), the
  owner-step-frame packet (`8 passed`), full owner-search service (`50 passed`),
  and ruff. Decision: pending identity handles are closed-local support; next
  work must move synchronous owner wait/search dispatch, pending transition/
  root ownership beyond parent handles, learner publication, or fixed
  owner-buffer handles.
- 2026-06-08 / Tesla, Mill, and Kant followups during the step-frame
  reorientation: thread limit blocked fresh spawns, so existing sidecars were
  reused while the main thread kept implementing. Mill's world-model audit says
  there is no proven fundamental blocker; the blocker is architectural parent
  ownership of hot-loop state, not variance. It also says replay/sample removal
  alone projects only `1.9625x`, so H100 for narrow proof rungs should stay
  refused. Kant's external-pattern audit mapped AlphaZero/MuZero/EfficientZero,
  PufferLib, EnvPool, Sample Factory, Isaac-style systems, and MCTX to the same
  pattern: mechanics/search/replay/learner each own fixed buffers and parent
  Python coordinates coarse epochs/proof only. Main thread left `goal.md`
  untouched, tightened the owner-published dense-action rung, and validated the
  focused owner-boundary packet (`5 passed`), direct-root owner-service packet
  (`4 passed`), and ruff. Decision: dense action publication and mechanics
  step-view are closed-local support; next work must move pending root/
  transition state, owner wait/search dispatch, learner publication, or fixed
  owner-buffer handles deeper into owner ownership.
- 2026-06-08 / Kepler, Bacon, and Sagan during the post-boundary reorientation:
  Kepler audited the smallest next broader owner-boundary implementation; Bacon
  ran an Amdahl/hot-path audit; Sagan audited stale lanes and follow-up hygiene.
  Kepler's completed audit says the current loop still has parent Python
  coordinating hot cadence: the action-step boundary proves owner-selected
  actions are applied through `manager.step()`, but mechanics/observation still
  live in the hybrid manager and the direct stepper still receives a
  parent-created compact batch/request object. Kepler ranked next patch
  candidates as boundary-bound direct owner transaction, core direct
  transition-batch replay ingestion, then owner maintenance epoch coalescing.
  Bacon's completed audit says the blocker is
  insufficient architectural movement, not variance or proven impossibility:
  columnar r2 is real progress at `15852.67 env/s` (`46.7666s`, `1.25x`
  OPT-104), replay/sample removal projects to only `24903.25 env/s`
  (`1.9625x`, `29.770s`), still `0.558s` short of `2x`, so the missing wall
  must come from parent wait/search dispatch, actor/mechanics/observation, or
  learner publication/update cadence. Sagan's completed audit says most recent
  failures were speed failures, not correctness failures, and unchanged H100
  repeats of proof-clean losers should be refused. Main thread kept `goal.md`
  untouched, added the real direct-root/threaded boundary proof
  `test_hybrid_profile_owner_action_step_boundary_uses_direct_root_build_request`,
  added resident terminal dense-host final-observation elision in the
  resident/no-scalar path, and validated focused source-state tests
  (`4 passed`) plus ruff. Decision: action-step boundary and resident final
  host elision are closed-local support; next implementation must move a
  broader measured owner surface.
- 2026-06-08 / Hooke the 2nd and Dewey the 2nd during the guarded
  action-step boundary binding: Hooke audited the code path and recommended
  the smallest next gate: wire `compact_owner_action_step_boundary` into the
  compact Coach speed-row proof/report surface and require it for direct-root/
  action-only owner rows before any H100 launch. Dewey audited active docs and
  flagged stale operating/orchestration/task-board language. Main thread wired
  the default-off flag through the local producer, Modal wrapper, remote Modal
  producer, evidence summary, proof projection, and fail-closed direct-root/
  action-only guard. Focused validation passed (`7 passed`) plus ruff. Decision:
  this boundary is proof-only support; the next launchable code must move a
  broader owner-buffer/root/search/mechanics/learner-publication surface.
- 2026-06-08 / Zeno the 2nd and Carver the 2nd followups during the fixed-slot
  proof cleanup: new sidecar spawns were unavailable because the thread limit
  was already full, so the existing sidecars were reused. Codeflow audit agreed
  that fixed action-result buffers should require deferred owner maintenance
  for the current direct-root path; simply forcing `submit_action()` everywhere
  would be wrong without a synchronous maintenance-drain design. Retrospective
  audit agreed that the month produced real but modest progress, not a `2x`
  close: columnar r2 is the fastest single H100 input at `15852.67 env/s`
  (`~1.25x` OPT-104), with many proof-clean lanes speed-rejected. Main thread
  added the deferred-maintenance guards, repaired owner-owned fused
  learner-batch proof so it no longer requires parent prebuilt learner batches,
  closed local fixed-slot speed-row proof
  `opt-fixed-action-result-slot-local-smoke-proof5-20260608`, and reran
  validation (`7 passed` focused speed-row slice, `47 passed` owner-search
  service, `96 passed` full speed-row smoke). H100
  `opt132-h100-fixed-action-result-slot-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608`
  then passed proof but failed as the main lane: `12794.42 env/s`, only
  `0.807x` columnar r2. Decision: keep fixed-slot as support because it
  improved root-build-request, but move next to owner-local transition
  derivation or broader fixed owner-buffer work.
- 2026-06-07 / Zeno and Carver followups during the owner-boundary reset:
  Zeno recommended the deeper architecture patch: let owner-search derive and
  append replay transitions from owner-cached previous/current root/search/action
  state, so parent no longer stages replay transport entries back into the
  owner. Carver audited the current hot path and recommended the smaller next
  RPC-boundary patch: a same-process fixed action-result slot buffer so the
  parent receives a tiny slot handle instead of a per-step Python result
  envelope. Main thread implemented the narrower support patch found during the
  code audit: deferred owner maintenance now slices staged root-batch cache
  snapshots to referenced replay record indices/current actor step, with
  fallback and retained/full/dropped counters. Main also implemented Carver's
  smaller RPC-boundary patch locally: same-process inline/threaded direct-root
  owner proxies can return a tiny action-result slot stub while the full result
  stays in `CompactOwnerActionResultSlotTableV1`. Validation passed with ruff
  and the full owner-search service suite (`47 passed`). Decision: keep cache
  slicing as support; next real speed gate is fixed-slot launcher/local
  speed-row smoke, then H100 only if proof and local bytes/parent-wait evidence
  justify it. If that does not move the loop, switch to owner-local transition
  derivation, not another replay-only/fixed-SoA tweak.
- 2026-06-07 / Euler, Hilbert, Volta, and Dalton followups during the root-build-
  request reorientation: Euler audited the active docs and flagged stale claims
  that CLI/report proof was pending or fixed-SoA gather was still the main
  lane. Hilbert supplied the exact local root-build-request speed-row command
  and fail-closed proof fields. Volta ranked the next implementation surfaces:
  first root/search ownership and parent wait, then owner-resident replay plus
  learner-ready tables, then learner publication by owner-ref. Dalton compared
  AlphaZero/MuZero/EfficientZero, MCTX, EnvPool/Sample Factory/PufferLib, and
  Isaac-style systems and extracted the pattern: mechanics has one owner,
  search sits behind a fixed root/result ABI, replay owns target semantics,
  learner owns weights, and resident data crossings must be explicit. Main
  thread fixed the root-build-request action metadata bug, added regression
  coverage, closed local smoke
  `opt132-local-owner-rootbuildrequest-threaded-directtransition-smoke-20260607-r3`,
  launched H100
  `opt132-h100-rootbuildrequest-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607`,
  and updated active docs. Decision: root-build-request is proof-clean but
  speed-rejected (`11327.75 env/s`); preserve it as support and move to larger
  owner-resident buffer/search/learner-publication work.
- 2026-06-07 / Lovelace and Poincare followups during the latest
  reorientation: Lovelace audited the root/dataflow path and found the direct
  root owner-search path still calls `build_compact_root_batch_v1` in the
  parent (`CompactOwnerSearchDirectStepperV1.step`), so the existing
  resident-root proof only proves owner consumption of the resident handle, not
  removal of parent root-batch construction. Poincare synthesized the H100
  residuals and confirmed the fastest columnar r2 row still spends material
  wall in replay append, worker search, parent wait, observation, learner
  train/sample, and learner update; conservative replay/sample removal alone
  projects `24903.25 env/s` (`1.9625x`), still short of `2x`. Main thread then
  implemented the default-off resident host-observation stub gate for
  direct-root owner-search: local smoke
  `opt132-local-owner-resident-root-hoststub-threaded-directtransition-smoke-20260607-r1`
  proves zero materialized host-observation bytes while preserving
  resident-root/action-only/parent-row/maintenance proof. Decision: useful
  local gate, not speed evidence; the next real patch is true root-view/build
  request ownership or a combined owner-buffer/root/search/learner-publication/
  mechanics surface.
- 2026-06-07 / Maxwell and Boole followups after the latest user
  reorientation: Maxwell gave the blunt retrospective. Real progress exists
  but is modest: columnar r2 is the fastest single H100 input at `15852.67
  env/s` (`~1.25x`), while fixed-SoA exact/locality, one-simulation deferral,
  and in-process async learner overlap are all proof-clean or diagnostic but
  speed-rejected. Variance happened earlier, but it is no longer the main
  explanation for missing `2x`; unchanged longer runs are not the next move.
  Boole audited the root/data path and identified direct-root owner modes as
  the correct first resident-root-view target. Main thread implemented the
  default-off `owner_search_require_resident_root_view` proof for direct root
  stores/proxies, made shared-memory owner mode fail closed for this proof,
  projected the new fields, and passed local proof smoke
  `opt132-local-owner-resident-root-view-threaded-directtransition-smoke-20260607-r5`.
  Decision: resident-root-view proof is a gate, not speed evidence; next patch
  must remove parent root-batch construction, owner search dispatch/parent
  wait, mechanics/observation ownership, or learner publication/update.
- 2026-06-07 / Noether, Avicenna, and Peirce followups after the generated
  ceiling review: Noether confirmed old H100 artifacts lack native
  `compact_whole_owner_buffer_replay_ceiling_*` keys but the compare script is
  the correct backfill surface and should rank old rows from existing timing
  fields. Peirce supplied the fail-closed proof list: projection fields stay
  nonproduction, cannot rewrite denominator currency or reported SPS, and next
  owner-buffer rows must preserve parent/search/replay/terminal counts. Avicenna
  identified parent wait/result transport as the next non-replay surface, then
  the main thread checked the fastest columnar r2 row and found action-only
  transport is already active (`payload/visit/root bytes=0`, owner materializes
  replay, replay handle present, parent reconstruction false, inner two-phase
  deferred replay true). Decision: do not spend another H100 row on action-only
  or replay-only work; target owner-resident root/mechanics/search dispatch,
  parent wait, and learner-publication overlap.
- 2026-06-07 / Peirce followup on default-off one-simulation replay deferral:
  Peirce identified model identity as the semantic blocker. Payload parity is
  enough to land default-off code only with negative refresh-cross tests. Main
  thread implemented action-time model identity, flush-time identity comparison
  before recurrent materialization, refresh guards for pending deferred handles,
  identity-drift failure, pending-count telemetry, and slab/profile projection
  of action-phase guard fields. Validation passed: pycompile, ruff, and
  compact Torch search-service tests (`42 passed`). Decision: owner-search
  integration proof is required before any H100 launch.
- 2026-06-07 / Noether, Avicenna, and Peirce followups during the
  whole-owner-buffer ceiling patch: Noether identified the smallest
  non-hot-loop hook as a derived projection beside
  `_speed_timing_projection_fields()` in
  `scripts/build_compact_coach_speed_row_smoke.py`, using existing owner-search
  parent wait, replay append, search, and owner-train sample fields. Avicenna
  argued the mature-system placement should eventually live in replay-store
  telemetry carried through owner-service telemetry, but agreed the immediate
  decision instrument should preserve real mechanics/search/replay proof and
  estimate flat-row append/sample beside it. Peirce required the projection to
  fail closed as non-production: projection-only, no production speed claim, no
  live-training touch, local projection currency, H100 validation not run, and
  promotion ineligible. Main thread implemented the projection/report/evidence
  guard locally and validated pycompile, ruff, and focused pytest (`4 passed`).
- 2026-06-07 / Noether, Avicenna, and Peirce followups after the H100
  locality r2 read and projection cleanup: all three agreed that
  fixed-SoA gather/locality alone is no longer the main lane. Noether's
  concrete next option is to promote the fastest exact columnar/direct-table
  path only with fixed-SoA parity and strict repeat H100 kill criteria: two
  rows at least `15%` above OPT-104, owner sample at least `20%` better than
  r8, and no replay append or parent-wait blowup. Avicenna's mature-system
  mapping says the real next question is a whole-owner-buffer ceiling: keep
  mechanics/search proof real, but replace replay append plus sample with
  owner-resident flat row storage/prebuilt gather, optionally mock learner
  update separately, and ask whether that ceiling can reach `2x` before another
  production rewrite. Peirce's gate list: local toy correctness, local toy
  speed, row algebra, fragmentation visibility, no-fake-speed proof fields,
  matched variance/longer-run rules, H100 kill criteria, and promotion order
  local correctness -> local speed -> CPU smoke/report -> short H100 -> longer
  paired H100. Main-thread decision: do the ceiling/toy proof before another
  gather-only or flat-row production edit.
- 2026-06-07 / Noether, Avicenna, and Peirce followups during the fixed-SoA
  R8/R9 step-back: fresh subagent spawn attempts hit the thread limit, so the
  existing sidecars were reused. Main-thread read: R8 is proof-clean but below
  OPT-104 (`11616.45` vs `12689.38`) and touches about `373` record groups for
  `512` sampled rows; R9 copy-trim regressed and was reverted. Main thread did
  not wait on sidecars before acting: it added an explicitly semantic-drift
  fixed-SoA locality sampler probe plus launcher plumbing and tests, then
  updated working docs. H100 r2 then judged the probe: selected-record
  fragmentation is real (`62` records for a `512` row learner batch), but the
  whole row slowed to `10428.59 env/s`. Returned reads: Avicenna reaffirmed the mature
  pattern as fixed buffers, small handles, coarse sync, and no rich Python
  objects in hot handoffs; Noether ranked semantic-preserving grouped execution
  first, locality sampler second, and flat global row layout third, with R8
  already covering the selected-group execution part; Peirce requested
  additional guards before flat rows, especially variable-width groups,
  duplicate/replacement sample order, eviction invalidation, ambiguous
  env/player keys, sparse final observation, and row-count proof identities.
  Main thread immediately added the `group_size=1` off-parity guard for the
  locality sampler. Decision after r2: do not green-light flat/global rows as a
  standalone gather fix; only revisit them inside a broader owner-buffer design
  that also attacks append/search/parent-wait/learner surfaces.
- 2026-06-07 / Noether, Peirce, and Avicenna during the fixed-SoA
  implementation pass: Noether audited parity requirements for replacing the
  maintained unroll-2 replay path, including target tensors, masks, terminal
  final-observation semantics, and no-bootstrap terminal values. Peirce
  specified the fail-closed owner-search proof shape: fixed-SoA requested/used
  flags, zero columnar/table/group-object materialization, direct transition
  replay, slab bypass, fused learner batch, learner-ready unroll-2 cache,
  tensor-native replay, and learner unroll 2. Avicenna checked mature-system
  patterns across AlphaZero/MuZero/EfficientZero, PufferLib, Isaac-style
  tensor systems, Sample Factory, and MCTX/JAX: the useful common pattern is
  fixed resident buffers plus handle-based ownership boundaries, not a blind
  framework port. Main thread implemented the local fixed-SoA/direct-gather
  replay slice and threaded it through owner-search and Modal. The next gate is
  H100 same-work speed evidence with the fixed-SoA proof fields, not another
  `goal.md` rewrite.
- 2026-06-07 / Singer, Newton, and Planck during columnar/fixed-buffer
  reorientation: Singer found the invalid r1 cause: the accepted-fast-path
  preset overwrote owner-search/direct-replay flags after parse and in the
  remote launch path, so future owner-search direct-replay rows must omit the
  preset and spell the accepted shape manually. Newton found that columnar
  append removes `_CompactReplayRingEntry` but still materializes
  `CompactReplayColumnarAppendRecordV1`, `_CompactReplayColumnarEntryViewV1`,
  step views, per-record table entries, and table concat paths. Planck mapped
  the minimal mature-system next step: a fixed SoA unroll-2 replay buffer with
  direct slot writes and `fixed_soa_direct_gather_v1` learner-batch gather,
  plus proof for zero entry-view/table-entry/group objects and zero table
  concat. Main H100 results matched the critique: columnar r2 reached
  `15852.67 env/s` but only about `1%` over direct-table r3, while lazy selected
  regressed to `11179.26 env/s` by moving append savings into sample/learner.
- 2026-06-07 / Linnaeus and Lorentz followups after user reorientation:
  Linnaeus audited the last several weeks/months and clarified the language:
  most rows "failed" as speed claims, not correctness. Real H100 progress is
  reset RNG, threaded/background owner maintenance, and ring-batched append;
  the blocker is headroom, not proof or variance alone. Lorentz mapped the
  mature-system pattern across AlphaZero/MuZero/EfficientZero, PufferLib,
  EnvPool, Sample Factory, Isaac-style tensor buffers, and MCTX/JAX: fixed
  buffers and small handles across env/search/replay/learner boundaries. They
  initially allowed one direct-prebuilt verification row, but the main-thread
  timing read now deprioritizes that standalone row: the skipped
  group-object/builder surface is tiny against r3/r4 replay append. Fixed SoA
  owner replay is the next implementation.
- 2026-06-07 / Ampere and Hubble followups during ring-batch work: Hubble
  audited the docs and confirmed `goal.md` should stay untouched while stale
  lower-doc "active next" language is cleaned in working docs. Ampere audited
  the owner-search tensor-native learner-batch path and found the next smallest
  meaningful patch after replay append: short-circuit
  `_CompactReplayRingV1.sample_from_snapshot` to return maintained
  tensor-native prebuilt learner batches before per-group sample-object
  construction. Main thread implemented the non-overlapping replay append/cache
  refresh batch patch, ran local proof, and measured H100 r3/r4 positive:
  average `14277.52 env/s`, `51.930s`, replay append `19.420s`.
- 2026-06-05 / Ampere, Hypatia, Hubble followups during transition-batch work:
  reused existing agents because the thread limit blocked new spawns. Ampere
  argued that wrapping old per-step replay append entries would be fake and
  that the batch contract should live in `compact_rollout_slab.py` with owner
  materialization in `compact_owner_search_service.py`. Hypatia required a
  fail-closed proof that logical replay transitions exceed transport entries,
  with no legacy fallback, no overflow, schema/kind/digest proof, and exact
  action-feedback/replay counts. Main integrated this locally: batch size `4`
  local smoke now proves `12` logical transitions via `3` transport entries.
  H100 r2 later proved the same invariant at `724 -> 181`, but whole-row speed
  stayed flat against r32/r33. Use future sidecars to design the larger
  fixed-buffer/prebuilt-batch boundary, not to ask whether this wrapper has
  been tried.
- 2026-06-05 / Ampere, Hubble, Hypatia after inline-background r38: reused
  existing agents while the main thread implemented locally. Hubble said the
  mature-system pattern points to fixed-buffer slab bypass before more worker
  variants. Hypatia flagged that bypass proof must fail closed on missing
  fields, not default missing counters to zero. Ampere recommended the next
  increment after direct stepping should batch/coalesce owner replay transition
  transport so logical transitions can exceed transport entries. Main
  integrated Hypatia's fail-closed proof critique and landed the first
  direct-stepper bypass proof; the follow-up above integrated the
  transition-batch increment.
- 2026-06-05 / Ampere and Hubble reuse after r29-r34: thread limit prevented
  new agents, so existing sidecars were reused. Ampere was asked to audit
  stale docs and where rare `goal.md` versus working-doc updates belong. Hubble
  was asked to audit the next code target after threaded/background maintenance
  produced repeated modest H100 wins. Main thread continued doc consolidation
  and validation rather than waiting.
- 2026-06-05 / Ampere and Hubble earlier followups: r28 async-owner learner
  proved learner wait was not the blocker; Hubble synthesized mature-system
  patterns from AlphaZero/EfficientZero/PufferLib/Isaac/Sample Factory. Main
  integrated the result as: fixed buffers, explicit owner boundaries, coarse
  refresh, background maintenance, and no blind framework port.
- 2026-06-05 / Hume and Beauvoir reuse: resumed agents initially returned
  pre-r14 advice, which confirmed the stale-doc risk. They were redirected to
  audit the r14 world model, but the main thread kept the critical path local.
  Lesson: when sidecars return old "learner boundary" advice, check whether
  they saw r14 before using it.
- 2026-06-04 / Ampere: audited the `goal.md` cleanup and active docs. Finding:
  `goal.md` should stay a durable compass; churn and row evidence belong in the
  working docs.
- 2026-06-04 / Ohm: audited owner-search async learner behavior. Finding:
  same-process async and higher queue depth can show actions while futures are
  pending, but the measured full-row wall still pays the learner/final-drain
  tail. The next speed-relevant boundary is a distinct learner resource/work
  shape, not another local future.
- 2026-06-04 / Hubble: active bounded followup on the smallest real owner
  learner worker/resource boundary: picklability, replay ownership, model
  publication, final drain, and proof fields. Main thread should continue local
  verification and non-overlapping implementation while this runs.

## Current Questions

OPT-132 is active. Sidecars should support the current decision and the next
owner-boundary patch, not resurrect fixed-SoA gather/locality or old attribution
lanes.

Current state:

- Columnar/direct-table owner-search is the fastest single H100 support input:
  `15852.67 env/s`, about `1.25x` OPT-104, not repeat-proven and not `2x`.
- Exact fixed-SoA and locality were measured. They prove fragmentation but are
  not the speed lane by themselves.
- Whole-owner-buffer replay/sample removal projects close to, but still short
  of, `2x` on the fastest row.
- The fastest row already has action-only owner transport and zero
  search/visit/root payload bytes.
- Default-off one-simulation replay deferral is locally proved through the
  owner loop, and its one H100 probe has already run. r7 passed locally with
  direct proof counts `12/12/12/12`, owner-inner proof counts `16/16/16/16`,
  crossed `0`, replay D2H `0.0`, and owner-inner pending final `0`. The H100
  row kept proof closed but was speed-rejected at `13691.98 env/s`,
  `54.1467s`, below columnar r2 `15852.67 env/s`.

Current sidecar lanes:

```text
P0 decision audit:
  The H100 deferral decision is closed. Do not launch duplicate deferral rows
  unless a new implementation changes the owner boundary. Sidecars should now
  audit the next local owner-boundary patch choice: root/search dispatch,
  parent-wait overlap, learner publication/update cadence, mechanics/
  observation ownership, or broader fixed-buffer handoff.

P0 next implementation audit:
  If not launching the probe, identify the smallest patch that changes owner
  graph or hot data movement: owner-resident root/mechanics/search dispatch,
  parent wait, replay/table maintenance as part of a broader fixed buffer, or
  learner publication/update overlap.

P0 proof gaps:
  Treat owner-search proof fields as preservation guardrails. Do not let
  background maintenance or threaded ownership hide pending work, failed
  maintenance, parent replay rows, search payload bytes, replay D2H bytes,
  tensor fallback, terminal proof regressions, or deferred handles crossing
  model refresh.

P0 variance audit:
  If a future H100 probe regresses, decide whether a measured bucket moved
  because of the patch or because known H100 timing variance reappeared. Do not
  ask for a longer run unless the first row changes a real bucket and preserves
  proof. The completed deferral probe moved a real bucket in the wrong
  direction: direct append rose to `22.377s` with deferred/device flush
  `3.613s/4.381s`, so it is a speed-rejected lane, not an A/A variance puzzle.

P0 mature-system patterns:
  What should Curvy copy from AlphaZero/MuZero/EfficientZero, PufferLib,
  EnvPool, Sample Factory, Isaac Gym/Lab, and MCTX/JAX-style search without
  starting a blind framework port?

P0 proof/plumbing:
  Keep accepted-fast-path proof honest while owner-search action-only mode
  removes parent replay/search row materialization. Replay proof fields remain
  required when replay is touched, but parent row transport must stay zero.

P0 docs consistency:
  Active docs should name one current P0 decision and demote direct-table,
  columnar, ring-batch, fixed-SoA exact/locality, and inline-background rows to
  support or rejected lanes. Any line that says deferral owner proof is still
  pending, fixed-SoA gather is the main lane, or r14/r32 repeats are the next
  action is stale unless new evidence reopens it.
```

Historical anchor rows:

```text
OPT-104 target: 12689.38 steps/sec, 14.5255s wall
OPT-132 r29 inline vector-RNG refresh1: 12057.81 steps/sec, 61.485s wall
OPT-132 r30 inline vector-RNG refresh4: 14230.53 steps/sec, 52.098s wall
OPT-132 r31 inline vector-RNG refresh4 repeat: 11794.65 steps/sec, 62.857s wall
OPT-132 r32 threaded vector-RNG refresh4: 14002.12 steps/sec, 52.947s wall
OPT-132 r33 threaded vector-RNG refresh4 repeat: 13640.60 steps/sec, 54.351s wall
OPT-132 r34 threaded vector-RNG refresh4 long: 13145.34 steps/sec, 112.797s wall
OPT-132 r14 first owner-search H100 win: 13497.30 steps/sec, 13.6561s wall; historical, repeat failed
OPT-132-G fast row: 13649.29 steps/sec, 13.504s wall
OPT-132-H failed repeat: 11366.84 steps/sec, 16.216s wall
OPT-132-I fast row: 13308.59 steps/sec, 13.8497s wall
OPT-132-J failed repeat: 10057.98 steps/sec, 18.3258s wall
OPT-132-K/L/Q/R/S long-window A/A: exact identity, wall range 47.945s
OPT-132X 1084/270 new-field r1/r2/r3: exact identity, wall spread 28.202s
OPT-132Y 1084/270 CUDA-sync r1/r2/r3: exact identity, wall spread 60.959s
OPT-132Z 1084/270 runtime-envelope r1/r2: exact identity, wall spread 76.574s
OPT-132AA 1084/270 sample-gate per-call r1/r2: exact identity, wall spread 10.865s
OPT-132AB 1084/270 builder-child r1/r2: exact identity, wall spread 100.593s
OPT-132AC 1084/270 deep builder r1c/r2/r3: exact identity, wall spread 63.608s; r2/r3 cluster, r1c slow
OPT-132AD 1084/270 runtime phase/residual r1/r2/r3: exact identity, wall spread 49.300s; sample-gate builder group-loop spread 24.801s
OPT-132AE 1084/270 builder loop accounted/residual r1/r2: exact identity, wall spread 12.641s; group-loop 113.971s -> 119.853s; accounted 95.872s -> 101.300s; residual 18.099s -> 18.553s
OPT-132AF 1084/270 guarded unroll-2 r1/r2/r3: exact identity, clean proof, wall spread 22.360s; sample-gate builder group-loop spread 11.952s; rejected as stable speed claim
OPT-132AG 1084/270 guarded unroll-2 GPU-sampled r4/r5/r6: exact identity, clean proof, wall spread 13.632s; monotonic AF slowdown did not repeat
OPT-132AH 1084/270 generic GPU-sampled r1/r2/r3: exact identity, generic path, wall spread 83.587s; sample-gate builder group-loop spread 48.201s; remaining swing is not AF-specific
OPT-132AI local: runtime-step envelope diagnostics can now run without CUDA-sync timing probes
OPT-132AJ 1084/270 generic runtime-step-only GPU-sampled r1/r2/r3: exact identity, CUDA-sync false, wall spread 22.989s; runtime-step sum reconciles wall; instability narrowed but remains
OPT-132AK local: comparator wall-swing attribution plus runtime-step cadence summaries
OPT-132AL 1084/270 generic runtime-cadence no-sampler r1/r2/r3: exact identity, CUDA-sync false, GPU sampling false, wall spread 38.758s; late sample-gate cadence carries much of the swing
OPT-132AM local: chronological active sample-gate distributions by measured third, sample-gate residual bucket sums, enriched top-slowest runtime-step records, and slowest per-call iteration projection
OPT-132AN 1084/270 generic active-cadence no-sampler r1/r2: exact identity, CUDA-sync false, GPU sampling false, wall spread 40.651s; active sample-gate p50/p95 rises early/mid/late in both rows and r1 is elevated across thirds
OPT-132AO local: bounded per-sample-gate replay-state trace records with call index, measured iteration/third, sample seeds/checksums, replay stored/eligible/excluded/evicted counts, replay capacity, and per-call timing
OPT-132AO 1084/270 generic replay-trace no-sampler r1/r2: exact identity, CUDA-sync false, GPU sampling false, wall spread 33.142s; trace state mismatch count 0/135, but sample gate still moved 132.338s -> 163.614s
OPT-132AP local: AO trace includes allocator/runtime snapshots, learner-batch-build memory deltas, Python GC/max-RSS state, sample-gate CUDA-sync timing, and deeper builder child timings
OPT-132AP 1084/270 generic allocator/runtime no-sampler r1/r2/r3: exact identity, CUDA-sync false, GPU sampling false, wall spread 65.694s; CUDA allocator/memory counters match exactly across all 135 calls, but Python GC/RSS fields vary
OPT-132AQ local: AP trace now includes `gc.get_stats()` collection/collected/uncollectable counters by generation
OPT-132AQ 1084/270 generic GC-stats no-sampler r1/r2: exact identity, CUDA-sync false, GPU sampling false, wall spread 55.055s; real GC collection totals are nearly identical and same-GC-delta calls still carry 22.6695s sample-gate delta
OPT-132AR local: AQ trace now includes process/current-thread CPU-time deltas for sample gate and learner-batch build
OPT-132AR 1084/270 generic CPU-time no-sampler r1/r2: exact identity, CUDA-sync false, GPU sampling false, wall spread 27.449s; sample-gate wall delta 18.848s is backed by 17.33s/17.18s process/thread CPU deltas; learner-build wall delta 11.331s is backed by 10.44s/10.39s process/thread CPU deltas
OPT-132AS local: AR trace now includes process/thread `getrusage` user/system CPU, page fault, and context-switch deltas for sample gate and learner-batch build
OPT-132AS 1084/270 generic resource-usage no-sampler r1/r2: exact identity, CUDA-sync false, GPU sampling false, wall spread 28.774s; sample-gate wall delta 18.999s is mostly process/thread user CPU (16.17s/15.83s), with low system CPU (0.49s/0.58s) and zero page-fault/context-switch deltas
OPT-132AT local: remote speed-row producer can now be wrapped in `perf stat -x,` with parsed task-clock/cycles/ref-cycles/instructions/branch/cache/LLC/dTLB/page-fault/context-switch/CPU-migration fields in row, report, and comparison artifacts
OPT-132AT H100 attempt: spawned fc-01KT7YGKDM2Y35NCG6C204TYMY but failed before a speed row because perf was not found; compact_profile_cpu_perf_stat_available=false, returncode=127
OPT-132AU local: speed_row_image now installs linux-perf for the perf-stat retry
OPT-132AU H100 attempt: spawned fc-01KT7YWEDCC60CMGTFY6388QBE; /usr/bin/perf exists but sys_perf_event_open returned 19 (No such device) for task-clock, returncode=255, parsed events=0, no speed row
OPT-132AZ 1084/270 generic remaining-builder-CPU no-sampler r1/r2: exact identity, CUDA-sync false, GPU sampling false, wall spread 36.500s (18.329%); group-loop process CPU 80.37s / 61.70s; unroll-fields CPU 42.13s / 32.24s; terminal metadata CPU 15.58s / 11.44s; decision is architecture reset and toy proof before any direct replacement/H100 row
sample-gate per-call distributions: compact_rollout_slab_sample_gate_*_per_call_stats
```

Current G/H deltas:

```text
wall:        +2.712s
actor:       +0.851s
observation: +0.512s
sample:      +0.734s
learner:     +0.312s
slab/search: +0.124s
```

Resident stack read:

```text
G resident stack shift: 0.012s
H resident stack shift: 0.015s
```

## Recent Delegations

- OPT-132AF Pauli: recommended a separate default-off unroll-2 helper, fused
  resident grouped-only routing, exact guards, generic oracle fallback, and
  additive proof metadata.
- OPT-132AF Avicenna: recommended generic-vs-specialized parity tests,
  requested/eligible/used/call/fallback/reason/impl/path proof, comparator
  fail-closed checks, and H100 r1/r2 before speed claims.
- OPT-132AF Euclid/Lorentz/Nash audit: proof projection is wired through the
  main chain, but `impl` and `fallback_reason` needed hard gates and the Modal
  collector needed a hard OPT-132AF proof failure. Main added those gates and
  reran ruff plus the compact speed-row suite.
- OPT-132AJ artifact audit: AJ's wall swing reconciled with runtime-step sum;
  r2's slow-fast delta was mostly sample/builder, while r1/r3 showed
  observation/learner-gate movement. Recommended a no-sampler exact generic
  repeat before another speed patch.
- OPT-132AJ instrumentation audit: recommended runtime-step cadence summaries
  in `source_state_hybrid_observation_profile.py`, not another builder timer:
  sample-gate active/inactive buckets, early/mid/late measured thirds, and
  bounded top-slowest-step records. Main implemented this as OPT-132AK and ran
  OPT-132AL on H100.
- OPT-132AL artifact audit: AL exact rows had stable sample-gate active counts
  `135/135/135`, deferred sample/learner disabled, identical replay/sample
  checksums and retained snapshot counts, and the same late sample-call indices
  slowed down. Recommended comparing by sample-gate call index and adding a
  bounded replay-state trace only if simpler chronology fields remain
  ambiguous.
- OPT-132AL instrumentation audit: recommended per-sample-gate call state with
  measured iteration, replay/ring state, retained snapshot counts/bytes, and
  optional allocator memory. Main implemented the low-overhead first step as
  OPT-132AM: chronological active-call distributions and slowest-call
  iteration projection.
- OPT-132AN artifact audit: active sample-gate call distributions rose
  early/mid/late in both rows and were elevated across r1 versus r2; wall swing
  attribution put `78.19%` of the slow-fast wall delta on sample gate and
  `47.18%` on builder group-loop. Main treated this as broad active-call
  slowdown, not only late spikes.
- OPT-132AO projection audit: main added bounded per-call replay-state trace
  records and projected them through source profile, speed-row summary/compact
  payload, local/remote Modal reports, plus focused tests.
- OPT-132AO H100 trace audit: r1/r2 preserved exact identity and clean
  violations; every exposed trace state field matched across `135` calls, but
  wall still spread `33.142s` and sample gate accounted for `94.37%` of that
  delta. Recommended next questions are allocator/cache/runtime timing or
  lower-level builder timing on identical replay/sample state.
- OPT-132AP sidecars: one code audit recommended per-call CUDA allocator
  before/after/delta counters, retry/OOM counters, and deeper builder child
  fields; one artifact audit noted AO's broad p50/p95 builder shift and
  recommended allocator/runtime snapshots plus GC/RSS state. Main implemented
  the local AP trace fields and read AP r1/r2/r3 on H100. CUDA allocator state
  was exact across all calls while GC/RSS fields varied; main implemented AQ
  `gc.get_stats()` counters.
- OPT-132AQ artifact audit: AQ r1/r2 kept exact identity but wall still spread
  `28.856%`; actual GC collection totals were nearly identical, and `100`
  calls with identical collection deltas still carried `22.6695s` of
  sample-gate delta. Main implemented AR process/thread CPU-time counters.
- OPT-132AR artifact audit: AR r1/r2 kept exact identity but wall still spread
  `13.479%`; about `92%` of sample-gate and learner-build wall deltas were
  backed by process/thread CPU-time deltas. Next sidecar should explain why
  identical trace state consumes different CPU time.
- OPT-132AS design audit: recommended low-overhead `getrusage` deltas for
  user/system CPU, page faults, and context switches around sample gate and
  learner-batch build before any builder speed patch.
- OPT-132AS implementation pass: main added the resource-usage fields locally,
  projected them through the existing runtime trace telemetry, and validated
  with ruff plus the focused source-profile/speed-row/Modal/comparator slice
  (`17` tests, `2` warnings). H100 r1/r2 then showed the remaining spread is
  mostly user CPU, not system CPU, page faults, or context switches.
- OPT-132AT implementation pass: main added external CPU perf-stat launch
  plumbing, parser/report/comparison projection, structured perf failure
  behavior, and focused tests. The intended H100 read should distinguish more
  retired instructions from worse IPC/cache/TLB/branch behavior or lower
  effective CPU frequency.
- OPT-132AV H100 read: main added in-process builder-child CPU-time deltas
  after OPT-132AU proved external perf counters unavailable. AV r1/r2 exact
  generic no-sampler `1084/270` kept exact identity and report violations `[]`,
  but wall spread was `15.03%`. Group-loop process CPU moved `+14.88s`;
  terminal metadata moved `+6.37s`, terminal final-observation `+2.67s`,
  residual `+4.28s`, and unroll fields `+2.76s`. No speed claim.
- OPT-132AW local diagnostic: main split AV's target one level deeper with
  group-loop prepare and terminal-value bookkeeping CPU fields, terminal
  final-observation presence/select-current/gather CPU fields, and
  final-observation branch/storage proof counters. Lorentz audited the
  final-observation path; Euclid audited group-loop residual. H100 r1/r2 then
  kept exact identity, failed wall repeatability (`7.318%`), and proved the
  final-observation branch/storage mix identical across repeats.
- OPT-132AX sidecar audit: Lorentz confirmed `_resident_final_next_observation_for_rows`
  is used by sample paths but discarded only in the grouped learner fallback
  path, so validate-only is semantically scoped there. Euclid recommended
  validate-only/materialized proof counters and optional comparator identity
  for terminal-final-observation proof fields. Main implemented both locally;
  H100 r1/r2 kept exact identity and proof, removed gather/select-current
  materialization, and still failed timing badly (`43.864%` wall spread).
- OPT-132AY sidecar audit: Lorentz recommended nested CPU children for the
  existing unroll wall sub-timers and warned not to add them to group-loop
  accounted CPU. Euclid identified the projection lists and tests that needed
  the new CPU family as timing fields, not identity fields. Main implemented
  the split; H100 r1/r2 kept exact identity, split unroll-fields CPU, and still
  failed stability (`25.819%` wall spread).
- OPT-132AZ sidecar/step-back audit: sidecars agreed the diagnostic loop had
  reached diminishing returns. AZ split remaining builder CPU and still found
  broad exact-work user-CPU movement. Current reset: stop pure attribution,
  treat learner-ready replay or batch-level unroll gather as candidate lanes,
  and choose through the dataflow map plus toy ceilings.
- OPT-132BF Cicero pattern audit: mature systems point to fixed buffers as the
  API: mechanics owns state/action/reward/done/reset buffers, search owns
  resident batched root/inference/action outputs, replay owns learner-ready
  sampleable tensors, and parent Python coordinates epochs/proof instead of
  per-row transport. Stop treating BF as "keep shaving sample gate"; use
  owner-search action-only replay ownership as the current concrete candidate.
- OPT-132BF Kierkegaard code map: primary residual is computed in
  `source_state_hybrid_observation_profile.py` as outer step wall minus named
  actor/observation/slab/sample/learner/policy timers. Actor/autoreset flows
  through `HybridBatchedObservationProfileManager.step`,
  `InProcessHybridCurvyTronActor.step_into`, and
  `VectorMultiplayerEnv.step_compact_profile`; env runtime flows to
  `vector_runtime.step_many`; search dispatch starts in `CompactRolloutSlab.step`
  and `CompactTorchSearchServiceV1.run_action_step`. Historical recommendation
  was learner-update placement/overlap; current order after ring-batch r3/r4 is
  fixed SoA/ring-buffer owner storage. Direct-prebuilt sample proof is support,
  not a standalone H100 lane.

Use sidecars for these questions now:

- Owner replay/sample boundary audit: how should `stage_replay_append_entries()`
  cache/resolve action payload handles, build replay index rows, and expose
  replay-ring/sample-handle proof without parent replay objects?
- Learner-batch handle audit: where should an owner-issued sample handle become
  a learner-consumable batch handle, and what metadata proves lifetime,
  generation, row/window checksums, fallback `0`, and pending `0`?
- Corrected denominator audit: does the next local artifact measure the whole
  owner surface, including observation/root/search/replay/sample/learner
  handoffs, or did it slip back into env-step-only timing?
- Proof preservation audit: will the next patch keep terminal proof,
  tensor-native fallback `none`, accepted/unroll2 violations `[]`, owner
  maintenance closure, action-feedback/replay ownership, parent dense-action
  reconstruction `0`, parent replay/sample objects `0`, selected groups `0`,
  and fake terminal samples `0`?
- H100 readiness audit: does the patch move a production owner graph or
  hot-data boundary, or is it another local support/proof rung that should not
  launch?
- Active-doc audit: rerun only after new evidence changes the replay/sample/
  learner-handle decision, not after every local support proof.

Historical sidecar questions about OPT-132AB through OPT-132AZ attribution
rows are parked. They can be reopened only if a new architecture toy or proof
row produces a specific contradiction that needs one of those old diagnostics.

Do not ask sidecars for broad reorientation. Do not reopen old owner-boundary
lanes, overlap reruns, root-build-request repeats, GPU-sampler, bigger-GPU,
compile-default, memory-fit, or result-payload lanes without a dataflow/toy/
measurement reason. GPU mechanics, PufferLib/EnvPool/Sample Factory/
Isaac-style systems, and MCTX/JAX-style search are allowed as bounded
architecture research and toy-ceiling lanes, not blind ports.

## External Sidecar

Use an external-pattern sidecar when the dataflow map or toy matrix needs a
mature-system comparison. Keep it bounded to one architecture question.

Ask one narrow question:

```text
Where do AlphaZero/MuZero/EfficientZero, PufferLib, EnvPool, Sample Factory,
Isaac-style systems, and MCTX/JAX-style search put environment mechanics,
search, replay, learner work, model snapshots, and bulk buffers?
```

The expected lesson is fixed buffers, batched services, and small references
first. Do not use external research as an excuse to port frameworks, move game
mechanics to GPU, buy bigger GPUs, or start multi-GPU before a toy ceiling and
dataflow map justify that lane.

## 2026-06-07 Followup Closure

- Avicenna checked the owner deferred replay proof surface. Useful result:
  action-phase proof was already mostly present, but flush-time proof needed to
  expose direct transition-batch flush count, materialized-on-flush count,
  recurrent call count, model identity match, refresh-crossed count, final
  pending deferred payload count, flush seconds, and replay D2H bytes. Main
  implemented those fields and fail-closed checks.
- Noether mapped the code path and found the launcher gap: the default-off
  `CompactTorchCompileConfig.defer_one_simulation_replay_payload` flag existed
  below the owner path but was not wired through the speed-row owner-search
  construction. Main wired the flag through local and Modal speed-row launchers,
  evidence summaries, and compatibility refresh plumbing.
- Peirce focused the semantic tests. Main added owner-level positive proof and
  negative identity/refresh-crossing tests, plus direct transition-batch proof
  assertions in the speed-row smoke tests.
- Noether also ran the bounded mature-systems audit. The common pattern across
  AlphaZero/MuZero/EfficientZero, PufferLib, EnvPool, Sample Factory,
  Isaac-style systems, and MCTX/JAX-style search is fixed buffers and batched
  owner services with small handles across boundaries. That supports the
  current order of operations: prove the owner boundary locally, then move the
  next non-replay owner/root/search surface, rather than blind-porting a
  framework or spending more rows on replay-only edits.

Status: integrated. No open sidecar blocker for the current local flagged smoke.

## 2026-06-08 Step-Frame Reorientation Sidecars

Spawned three bounded read-only sidecars while the main thread inspected and
patched locally.

- Parfit audited replay/sample/table maintenance. Finding: direct transition
  replay, columnar append, direct table, direct-prebuilt sample, and fixed SoA
  already exist, but columnar append still builds entry/step view objects and
  maintained direct tables still concatenate per-record batches before gather.
  Best replay support candidate is a maintained flat unroll-2 table or batched
  columnar append, with fail-closed table/version/window/object counters.
- Anscombe audited owner/root/search/mechanics boundaries. Finding: direct-root
  build-request avoids parent root-batch construction, but the parent still
  owns `manager.step`, mechanics/autoreset/observation, `_make_compact_batch`,
  root sidecar derivation, direct-stepper request/identity/action assembly,
  and synchronous owner wait. Best next local patch starts inside
  `HybridBatchedObservationProfileManager.step` before `_make_compact_batch`.
- Gauss audited mature-system patterns. Finding: the missing pattern is a
  single owner-owned step frame passed by slot/generation/digest handles from
  action application through root/search/replay/sample/learner publication.
  PufferLib/EnvPool/Sample Factory/Isaac-style systems and
  AlphaZero/MuZero/EfficientZero-style systems all point to fixed buffers and
  coarse role handoffs, not another narrow payload tweak or blind framework
  port.

Main-thread integration:

- Landed default-off `compact_owner_minimal_step_payload_snapshot` as
  support-only proof hardening under `compact_owner_action_step_boundary`.
  Focused source-state tests passed (`4 passed`) and ruff passed.
- Updated working docs to make owner-resident step-frame boundary the next
  real implementation target and to refuse H100 for the minimal payload support
  flag alone.

No open sidecar blocker remains. The next sidecar should be spawned only when
the owner step-frame patch has a concrete design or code slice to audit.

## 2026-06-08 Owner/Proxy Closure Sidecars

Followed up with existing sidecars while the main thread implemented locally.

- Tesla audited the smallest honest code move and called out the key proof:
  direct stepper must stop calling `_stage_previous_derived_transition`; proxy
  cached state must close previous transitions before owner request/train
  scheduling; dense/applied actions must be verified; final parent flush must
  fail closed unless a proxy final-closure API exists.
- Banach audited circular-import and replay-batch risks. Finding: do not import
  owner/proxy helpers back into slab, do not reuse the stepper private derived
  helper from owner service, and add fail-closed tests for parent closure bypass,
  applied-action mismatch, missing caches/sidecars, and final flush.
- Herschel audited docs hygiene. Finding: record what authority moved and what
  is explicitly not claimed; keep `goal.md` stable; mark the rung as local
  support and refuse H100 unchanged.

Main-thread integration:

- Added `joint_action` to `CompactRootBuildRequestV1` with shape validation.
- Added default-off `owner_proxy_transition_closure=True`.
- Added `CompactOwnerSearchSlabProxyV1.stage_owner_proxy_transition_from_root_build_request()`
  to close previous transitions from proxy cached action frames, verify applied
  action from the current root-build request, build the existing derived
  transition-batch schema, and stage it before owner request/train scheduling.
- Hardened final parent flush to fail closed in this mode.
- Added focused tests for parent stage/flush/commit bypass and applied-action
  mismatch rejection.

Validation:

```text
focused proxy-closure packet: 4 passed
tests/test_compact_owner_search_service.py: 52 passed
tests/test_compact_search_replay_contract.py: 47 passed
ruff on touched code/tests: passed
```

Decision: owner/proxy previous-transition closure is closed-local support, not
H100 speed evidence. Next work should move owner wait/search dispatch, learner
publication/update cadence, mechanics/observation ownership, or fixed
owner-buffer handles.

## 2026-06-08 Overlap/Proxy Readout and Step-Frame Handle Followups

Reused existing sidecars after the valid guarded H100 row completed:

- Tesla judged the r3 speed read: close overlap/proxy as speed-rejected because
  it was proof-clean but only `0.980x` columnar r2, with max pending `1` and no
  useful pipeline depth. P0 remains owner/proxy step-frame fixed slots; reject
  patches that only move code inside `CompactOwnerSearchDirectStepperV1` while
  parent still owns frame state.
- Herschel specified the next handle-ring implementation: fixed owner step-frame
  slots with slot/generation/digest handles, static owner row-major metadata,
  owner-derived `target_reward`/`done_root`/`active_root_mask`, and a first
  local test that monkeypatches `_make_compact_owner_mechanics_step_view`,
  `_make_compact_batch`, `compact_root_build_request_v1_from_batch`, parent
  dense-action reconstruction, and returned root sidecar builders to raise.
- Banach specified proof risks: missing critical fields must fail instead of
  default-projecting zeros; stale generic fields must not mask owner telemetry;
  warmup/measured counters must be explicit; speed currency must remain
  measured `compact_trainer_env_steps_per_sec`; fake overlap must fail on
  sync-wrapper/completed-at-submit/wait-in-submit/max-pending counters.

Main-thread integration:

- Recorded valid r3 in the measurement ledger:
  `15541.95 env/s`, `47.7016s`, `1.225x` OPT-104, `0.980x` columnar r2,
  overlap/proxy proof clean.
- Closed overlap/proxy unchanged and promoted owner-resident step-frame handles
  as the active code target.
- Added `CompactOwnerMechanicsStepFrameHandleV1` with schema
  `curvyzero_compact_owner_mechanics_step_frame_handle/v1`, four ring slots,
  slot/generation/digest identity, manager publish proof, direct-owner digest
  verification, and source-profile/speed-row/Modal/evidence projection.
- Added stale metadata digest rejection so the handle proof fails closed.

Status: integrated. No sidecar blocker remains. The first handle-identity rung
is local support only; the next patch must move actual frame data or learner
publication state behind fixed handles. Do not launch H100 for handle identity
alone.

## 2026-06-08 Slot-Backed Step-Frame Followups

Reused existing sidecars again because the agent thread limit was reached.

- Herschel audited the slot-backed root-request construction contract:
  derive `target_reward`, `done_root`, `active_root_mask`, and static
  policy/root metadata from the slot; validate applied `joint_action`; keep
  borrowed/fixed-array proof distinct from parent materialization; reject stale
  generation or digest drift; and do not overclaim `root_build_request_sec` as
  removed work.
- Banach specified fail-closed proof fields: slot schema, handle schema,
  ring-used, publish/consume/digest verification, resident handle presence,
  stale/digest mismatch counters, parent compact-batch/root request helper
  call counts `0`, and direct root-request sidecar bytes/fields `0/0`.
- Tesla audited speed-row/Modal projection and found the right P0 was not more
  projection alone, but raw fail-closed validation so missing zero-valued proof
  fields cannot be defaulted to clean zeros.

Main-thread integration:

- Added `CompactOwnerMechanicsStepFrameSlotV1` and a four-slot mechanics frame
  ring in `source_state_hybrid_observation_profile.py`.
- Replaced the direct-root mechanics boundary transport with slot-backed
  frame publication and a direct root-request builder in
  `compact_rollout_slab.py`.
- Kept the existing digest handle proof and added top-level proof for
  `compact_owner_step_frame_root_build_request_used`, from-batch helper
  bypass, root sidecar bytes/field count `0/0`, slot write count, and parent
  step-frame build count `0`.
- Updated local speed-row extraction, Modal report allowlists, compact
  speed-row evidence summary, and fail-closed direct-root proof guards.
- Added tests that poison `_make_compact_owner_mechanics_step_view`,
  `_make_compact_batch`, `compact_root_build_request_v1_from_batch`, and parent
  dense-action reconstruction while the owner path passes; missing/dirty slot
  proof now fails the speed-row guard.

Validation:

```text
ruff touched files: passed
source-state owner-boundary packet: 5 passed
owner-search direct-root/proxy packet: 11 passed
speed-row owner/direct-root/projection packet: 7 passed
```

Status: integrated as local support. Do not launch H100 for this rung alone.
Next sidecar-worthy questions are stale slot generation/reuse protection and a
larger root/search transaction or learner-publication ticket/ref patch.

## 2026-06-08 Slot Generation/Reused Frame Guard Followups

Reused existing sidecars while the main thread implemented the guard.

- Banach audited `CompactOwnerSearchDirectStepperV1` and found the concrete
  ordering hole: stale slot metadata was validated after
  `_commit_previous_transition()` in `submit_step()` and the sync fallback path,
  so a bad frame could have staged replay or mutated pending state before the
  stale-frame proof fired.
- Herschel mapped the broader mature-system pattern back to the repo: mechanics,
  search, replay, and learner should converge on fixed buffers plus small
  handles; the next real speed patch should make the owner consume the step
  frame through a longer-lived root/search transaction or move learner
  publication to tickets/refs.
- Einstein summarized the month-level failure mode: progress was real but
  modest; the flailing came from treating proof rungs and single rows as the
  map instead of returning to the owner graph. No fundamental impossibility has
  been proven; the active blocker is still insufficient owner-boundary movement.

Main-thread integration:

- Added live `slot_generation` tracking to the four-slot mechanics frame ring.
- Tightened `_owner_mechanics_step_frame_handle_metadata_v1()` to validate
  metadata slot/generation, generation modulo slot, handle shape, live slot
  generation, and digest.
- Moved direct-step root metadata/owner-frame validation before replay commit,
  proxy closure, and search submit.
- Added per-stepper consumed-generation tracking in
  `CompactOwnerSearchDirectStepperV1`.
- Strengthened the stale-generation test with a pending sentinel to prove the
  stale frame fails before replay append, proxy closure, submit, record-index
  movement, pending-dispatch creation, or pending mutation.

Validation:

```text
ruff touched files: passed
focused owner-frame/direct-root packet: 3 passed
source-state owner-boundary packet: 6 passed
owner-search direct-root/proxy/overlap packet: 11 passed
```

Status: integrated as local support. The next sidecar-worthy question is no
longer stale/reuse protection; it is the concrete design of the longer-lived
owner root/search transaction, with learner-publication tickets/refs as the
disjoint alternate.

## 2026-06-08 Root Action Context Followups

Sidecars audited the next owner-boundary choice while the main thread inspected
the pending transaction code.

- Herschel found the true next transaction API should start from the
  step-frame slot/handle inside the owner/proxy and bypass parent root-request
  construction. It should eventually have explicit owner root/search
  transaction counters and tests that monkeypatch both parent root-request
  builders to raise.
- Banach audited the disjoint learner-publication ticket/ref alternative. It is
  viable but likely lower leverage because learner refresh cadence is less hot
  than per-step parent wait/search/root dispatch.
- Einstein specified test requirements: forbid parent compact/root sidecars,
  parent dense-action reconstruction, parent previous-transition closure, and
  any pending full request storage when the transaction claims owner ownership.

Main-thread integration:

- Added `CompactRootActionContextV1` plus
  `compact_root_action_context_v1_from_request()`.
- Changed async direct-stepper pending state to store root-action context
  instead of `CompactRootBuildRequestV1`.
- Changed `CompactOwnerSearchSlabProxyV1` action-dispatch pending state to
  store root-action context instead of `CompactRootBuildRequestV1`.
- Made action-step construction/validation work from the context.
- Added public proof fields and tests asserting pending dispatches have no
  `root_build_request` slot while retaining action-critical validation.

Status: integrated as local support. This fixes a false-clean proof field and
shrinks pending transaction state, but it is not the final owner root/search
transaction. The next implementation should add the owner/proxy API that begins
from the slot-backed step-frame handle and avoids parent root-request
construction altogether.

## 2026-06-08 Owner Root/Search Transaction Followups

Reused existing sidecars while the main thread implemented the slot-started
transaction.

- Herschel mapped the smallest correct API: a proxy-owned transaction should
  start from the mechanics step-frame slot/handle, validate slot/generation/
  digest, build the root request inside the proxy, preserve proxy transition
  closure, and return an action dispatch handle plus the action-critical
  validation context.
- Banach specified fail-closed tests: monkeypatch
  `_root_build_request_from_owner_step_frame_slot_v1`,
  `compact_root_build_request_v1_from_batch`, parent root-batch builders, and
  dense-action reconstruction to raise; assert transaction parent root-request
  build/store counts are zero and owner build/publish counts are positive.
- Einstein gave the kill criterion: if the patch still calls the slab
  step-frame root-request builder, stores `compact_batch` in transaction
  pending state, or submits a parent-built request object, reject it as another
  support-only proof patch.

Main-thread integration:

- Added `CompactOwnerRootSearchTransactionDispatchV1` and
  `submit_owner_root_search_transaction_from_step_frame_slot()` on the
  owner-search proxy/lazy proxy path.
- Moved slot-backed root-request construction from the slab helper into
  `compact_owner_search_service.py` for the transaction path.
- Changed the direct stepper to use the transaction for ring-backed frames
  when the proxy supports it, including owner/proxy transition closure.
- Avoided storing `compact_batch` in the direct pending dispatch for the new
  transaction path.
- Added source-profile, speed-row summary, Modal report, and local smoke proof
  fields for the transaction counters.
- Strengthened tests so the slab step-frame request builder can be
  monkeypatched to raise while the owner-boundary path passes.

Status: integrated as local support. The next sidecar-worthy question is not
"how do we build the request from the slot?" anymore. It is whether to move
`CompactRootActionContextV1` and action validation behind a smaller owner
handle, or to switch to the disjoint learner-publication/update ticket/ref
surface if that is the faster measured owner-graph move.

## 2026-06-08 Root-Action Context Handle Followups

Reused existing sidecars while the main thread closed the owner-handle path.

- Banach audited the patch and found the important false-proof risk: a
  slot-started transaction must not fall back to a parent
  `CompactRootActionContextV1`, and proof must require handle schema/id/
  transaction/dispatch/root counts/digest plus balanced owner store/resolve/
  release counts.
- Herschel mapped the mature-system pattern back to the repo: this rung is
  still a support boundary. The speed-worthy next work is fixed owner
  replay/sample/learner-batch ownership first, coarse owner actor/search
  transaction tranches second, and learner publication/version tickets third.
- Tesla challenged the speed thesis: root-action-context storage cleanup alone
  is not a credible H100 launch; the remaining headroom likely requires
  replay/sample plus another non-replay owner surface. H100 is justified only
  after local proof removes or overlaps a measured wall bucket beyond this
  handle cleanup.

Main-thread integration:

- Added `CompactOwnerRootActionContextHandleV1` and an owner/proxy context
  table with digest verification and release counters.
- Changed owner action dispatch pending state to store only the handle.
- Changed direct transaction pending state to store no parent
  `root_action_context` and to fail closed if the handle is missing or malformed.
- Preserved per-step proof when final idle proxy metadata is merged into the
  profile result.
- Added source-profile, speed-row, Modal, and smoke-guard fields for handle
  schema/id/transaction/dispatch/root counts/digest, owner store/resolve/release
  counts, parent validation `0`, owner validation positive, and transaction
  parent root-action-context store/bytes/fields `0`.

Status: integrated as local support. Do not launch H100 for this rung alone.
Next sidecar-worthy implementation question is the broader owner
replay/sample/learner-batch ring versus coarse owner actor/search tranche versus
learner publication/version tickets.

## 2026-06-09 Maintained Replay Table Handle Followups

Reused existing sidecars while the main thread landed the default-off
maintained tensor-native replay table-handle rung.

- Tesla recommended the stronger next replay/sample slice: use fixed-SoA or
  learner-batch handle rings and prove selected-maintained per-record gather
  counters stay zero. The current maintained-table handle is a smaller support
  rung, not the final fixed-SoA handle ring.
- Mill mapped authority boundaries: replay/sample/learner-batch handles should
  be owner-issued capabilities with lifetime/watermark proof, not parent
  sidecars renamed as handles. Preserve action, transition, root, cache,
  ownership, and learner proof counters.
- Kant recommended not touching `goal.md`, the measurement ledger, or archived
  operating docs for this support rung; update only active working docs after
  real code lands.

Main-thread integration:

- Added default-off
  `compact_muzero_learner_batch_tensor_native_replay_maintained_table_handle`.
- Added `_CompactReplayTensorNativeUnroll2TableHandleV1` carrying schema,
  snapshot version, aligned record ids, table entries, learner-ready targets,
  source record pairs/windows, offsets, total rows, and missing-record count.
- Threaded the handle through `_CompactReplayRingV1.snapshot_for_sample()` and
  the maintained tensor-native replay builder.
- Added source telemetry, compact speed-row, Modal, build-smoke, and launcher
  proof/report fields for handle requested/used/record/missing/row counts.
- Added focused source and speed-row smoke tests; broader validation passed:
  source-state `127 passed`, compact speed-row smoke `100 passed`, tensor-native
  benchmark tests `4 passed`, and ruff passed for touched files.

Status: integrated as local support. Do not launch H100 for this rung alone.
Next replay/sample work should either turn this authority hook into an
owner-issued fixed-SoA or learner-batch handle ring with lifetime/watermark
proof, or be superseded by a broader owner transaction/learner-publication
patch that moves a measured wall surface.

## 2026-06-09 Reorientation Sidecars

Used three sidecars to rebuild the world model before continuing the
production fixed-SoA handle path.

- Hume synthesized mature-system patterns from PufferLib, EnvPool, Sample
  Factory, Isaac-style env stacks, AlphaZero/MuZero, EfficientZero, and
  MCTX-style search: mechanics, search, replay, and learner boundaries should
  be fixed-shape buffers or small handles; Python coordinates work but should
  not carry scalar object payloads through the hot loop.
- Meitner audited the Amdahl shape: current best support row is about `1.25x`,
  and `2x` versus OPT-104 needs removing or overlapping about `17.55s` from
  the current best wall. Replay/sample alone is near a `1.96x` ceiling, so the
  next patch must move at least one broader boundary.
- Lorentz recovered the local normal-death production shapes and selected the
  opt107-style owner-search fixed-SoA envelope as the right local contract gate
  before any H100 speed claim.

Main-thread integration:

- Fixed owner-search train-count expectation to track replayable submitted
  transition batches instead of raw step count.
- Fixed terminal sampling to force an actual terminal sample row when terminal
  rows are available.
- Passed local production fixed-SoA handle rows for the small normal-death gate
  and the opt107 envelope, with resident handle consumed, fallback counts zero,
  train cadence matched, and normal-death terminal proof satisfied.

Status: local support is clean enough. Do not repeat this lane as speed work.
Next sidecar-worthy question is which broader boundary to delete first:
actor/search transaction wait, learner publication/version tickets, or
mechanics/observation fixed-buffer ownership.
