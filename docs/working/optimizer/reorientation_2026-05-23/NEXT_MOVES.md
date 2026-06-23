# Next Moves

Date: 2026-06-09

Read `goal.md` first. This file is only the immediate action list.

## 2026-06-09 Amdahl Reset

See `AMDAHL_WORLD_MODEL_2026-06-09.md`.

The main lane is no longer gather/layout micro-patches. The fastest support row
still has broad critical-path surfaces: parent wait, replay append/materialize,
search, and learner train/publication. A standalone replay/sample patch cannot
credibly produce `2x` under Amdahl unless it also removes or overlaps another
large boundary.

The local benchmark denominator has now been corrected. The fixed-action-tape
harness keeps the old env-step/autoreset timer, but also reports a whole-loop
local wall that includes action source, observation, search, slab/replay/sample
work, owner-slot work, named-surface total, and residual. Use this whole-loop
view for owner-buffer ceiling decisions; do not use env-step-only timing to
justify H100.

First owner-buffer ceiling slice:

```text
flag:
  --run-owner-slot-ceiling

status:
  local proof landed
  no H100 claim
  no production speed claim

what it proves:
  mechanics result -> fixed owner mechanics step-frame slot
  slot handle -> direct owner root/action step
  owner-selected next joint action -> next mechanics step
  previous transition stages through stage_replay_append_entries
  owner-slot service validates replay-payload handles/digests
  staged owner replay rows drain into _CompactReplayRingV1
  owner-context replay rows are emitted as device replay index rows
  real compact replay ring appends columnar resident records
  real compact replay ring samples unroll-1 from resident snapshots with
    device_replay_index_rows_sample=true and host fallback false
  nonterminal owner-slot fixture builds a resident device learner unroll-2
    batch from the real ring
  local fixed replay slots append owner-selected rows
  local sample handles create/resolve inline with checksums
  generation/digest/action-feedback proof closes
  parent batch/root helper use is zero
  root-observation copy bytes are zero
  parent replay objects / selected-group objects are zero
  owner-action-context replay-index row builder matches the trusted root-batch
    builder without requiring a full root batch or HybridCompactBatch

what it does not prove yet:
  corrected local whole-loop timing does not yet show this production surface
    moved enough to justify H100
  fixed resident row/window slots and fixed-SoA handle-ring sampling are still
    support/default-off, not a promoted speed path
  local smoke uses CPU torch tensors, not an H100-resident search/replay graph
  CPU-oracle observation still dominates the tiny local toy denominator
```

Next main gate:

```text
promote the owner-slot replay/sample shim into production owner handles:
  mechanics/root/action plus local replay/sample handles close locally
  owner-context replay-index rows can now be built from handles
  local replay now has real-ring unroll-1 device-row sample proof
  local learner batch now has resident device unroll-2 proof
  production fixed-SoA replay now requests handle-ring sampling
  compact-owned learner boundary records resident handle consumed/fallback
  parent replay/sample objects and selected-group loops stay zero

then measure with corrected local whole-loop denominator:
  if the moved surface is not large enough locally, do not launch H100
  if it is large enough, run one same-work H100 candidate against columnar r2

then productionize only if H100 beats the fastest support row:
  mechanics writes fixed slots
  search consumes root handles and returns action handles
  replay owns row ids/windows
  learner consumes batch handles
  parent sees counters/proof/drain only
```

Puffer/Isaac-style contract to copy locally:

```text
one batch owner owns fixed arrays for:
  mechanics state/action/reward/done/masks
  root/search sidecars
  replay rows and successor windows
  learner-window handles

hot loop exchanges only:
  slot id
  generation
  digest/checksum
  row/window handle

hard counters before any H100:
  replay entry/view objects 0
  full replay snapshot 0
  materialized-parent fallback 0
  host observation fallback 0
  per-record selected gather storm 0
  learner consumes resident batch handle true
```

The fixed-SoA learner-batch handle-ring work is support only. Finish or park it
only if it strengthens fail-closed proof for the broader owner ring. Do not run
a standalone H100 row for it.

Latest owner-slot local smoke:

```text
/private/tmp/curvy_owner_slot_device_rows_unroll2_b2_m3_w1_20260609.json
status: pass
owner-slot proof: pass, failures []
fixed_vs_compact_whole_loop_speedup: 1.080x local toy, not speed evidence
replay slot appends/rows: 3 / 12
stage replay transport/transitions: 3 / 3
stage replay cache hit/miss/release/pending: 3 / 0 / 3 / 1
stage replay drained records / device index rows: 3 / 12
real replay-ring append calls/records/rows: 3 / 3 / 12
real replay-ring sample device rows: true
resident device sample batch: true
learner unroll-2 built/rows: true / 8
learner unroll-2 shapes: action [8,2], reward [8,2], value [8,3], policy [8,3,3]
learner unroll-2 host fallback: false
real replay-ring observation provider fallback count: 0
parent replay objects / replay object entries / selected groups: 0 / 0 / 0
sample handles create/resolve/inline/pending: 3 / 3 / 3 / 0
stage sample handles create/resolve/inline/pending: 3 / 3 / 3 / 0
sample rows/targets: 8 / 8
production speed claim: false
```

Current validation anchor:

```text
uv run ruff check scripts/benchmark_vector_fixed_action_tape.py tests/test_benchmark_vector_fixed_action_tape.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/compact_policy_row_bridge.py tests/test_compact_search_replay_contract.py
  passed

uv run pytest tests/test_compact_search_replay_contract.py tests/test_benchmark_vector_fixed_action_tape.py -q
  57 passed

uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_fixed_soa_samples_row_level_successors_for_coalesced_transition_batch -q
  1 passed

uv run pytest tests/test_source_state_hybrid_observation_profile.py -k "columnar or fixed_soa_locality" -q
  1 passed
```

## Now

Current immediate next move:

```text
build the next owner replay/sample/learner handle rung, not another H100 row:
  run corrected local whole-loop timing with production fixed-SoA handle-ring requested
  require compact-owned learner resident-handle consumed true
  require materialized-parent fallback count 0
  keep parent replay/sample objects and selected-group/tiny-gather loops at 0 where claimed
  preserve normal-death/action/terminal/tensor-native gates

launch gate:
  no H100 until the corrected local whole-loop denominator shows a measured
  owner surface moved beyond the toy proof
```

Latest selected-maintained direct-prebuilt result:

```text
local benchmark:
  /tmp/selected_maintained_replay_benchmark.json
  proof: required_pass true
  selected replay sample-build: 0.0167s median
  current replay-ring sample-build: 0.0560s median
  local surface speedup: 3.35x

H100 falsifier:
  run: opt132-h100-selected-maintained-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260609
  status: proof clean, speed rejected
  speed/wall: 12028.55 env/s / 61.6347s
  vs columnar r2: 0.759x
  vs OPT-104: 0.948x
  impl/source: selected_maintained_record_table_gather_v1 / selected_maintained_record_table_v1
  fast metadata: requested true, used true, selected groups 352
  tensor-native fallback: 0 / none
  sample gate: 22.581s
  owner learner train: 31.768s
  direct/ring append: 23.851s / 20.513s

decision:
  selected maintained gather is now explicit/default-off
  do not repeat this row unchanged
  do not pursue more per-record selected GPU gathers as the main lane
```

Immediate next move: build a fused/batched selected gather only if it removes
the per-record GPU gather/Python loop shape, or skip directly to the broader
owner transaction/learner-publication/fixed-buffer patch. The next useful
local proof must move a measured owner surface without reintroducing hundreds
of tiny selected groups.

OPT-132 is still active. Do not rerun exact fixed-SoA unchanged as a "longer
run" variance check. The exact fixed-SoA path is proof-clean but below OPT-104:
r8 ran `11616.45 env/s` versus baseline `12689.38`, with about `373` selected
record groups for a `512` row learner sample.

The locality probe has now answered the immediate fixed-SoA question:

```text
probe:
  run: opt132-h100-fixed-soa-locality-g8-rowlevel-selectedgroups-b1024a1-normal-unroll2-m724-w180-r2-20260607
  status: proof clean, semantic drift explicitly marked
  speed/wall: 10428.59 env/s / 71.0907s
  selected records: 62 for a real 512-row learner batch
  owner train sample: 32.03s
  learner update: 6.52s
  replay append: 17.24s
  worker search: 16.68s
  parent wait: 21.62s

interpretation:
  selected-record fragmentation is real
  locality alone is not the 2x/10x lever
  stop spending the main lane on fixed-SoA gather tweaks by themselves
```

The slot-candidate support row also answered its H100 question:

```text
run: opt132-h100-slotcandidate-support-b1024a1-normal-unroll2-m724-w180-r1-20260607
status: proof clean, speed rejected
speed/wall: 10348.87 env/s / 71.638s
vs OPT-104: 0.816x
vs columnar r2: 0.653x
proof: normal-death gate true; accepted/tensor-native/cache/unroll violations []
fixed SoA: requested/used true, fallback 0/none, object/table counts 0
direct replay: 724 transitions -> 181 batches -> 181 transport entries
columnar append: false
selected records/table rows: 373 / 1460766
fixed-SoA total/sec: 4.083s, mostly successor-index 4.080s

interpretation:
  slot-candidate fixed SoA is proof support, not the standalone speed lane
  do not rerun fixed-SoA exact/locality/slot-candidate unchanged
```

Immediate next move: return to the fastest exact candidate stack and attack the
larger owner-buffer surfaces that still cross Python/object-heavy boundaries.
Use the columnar/direct-table stack as the current fastest same-window support
row (`15852.67 env/s`) and preserve fixed-SoA locality only as a diagnostic
input. The next code patch must change one of these surfaces: replay
append/materialization, owner search/root batching, parent wait, learner
update/publication cadence, or mechanics/observation ownership. A flat/global
row layout is only worth building if it is part of that broader fixed owner
buffer design.

Owner-local transition derivation decision:

```text
owner-local transition derivation:
  status:
    valid H100 decision row is proof-clean but speed-rejected as the main lane
  patch:
    CompactRootBuildRequestV1 and CompactRootBatchV1 now carry explicit
    terminated/truncated/final_reward_map sidecars
    compact_transition_outcome_v1_from_next_root_batch() derives
    next_reward/next_done/next_terminated/next_truncated/
    next_final_reward_map/next_final_observation_row_mask from the next root
    and raises if mechanics sidecars are missing
    direct root-build sidecar path preserves those fields
    CompactOwnerSearchDirectStepperV1 can stage derived transition batches
    carrying only record links, replay handles/digests, and applied-action
    count/checksum
    owner direct replay sidecar derives next reward/done/terminal/final facts
    from cached current root, with cache/checksum/fallback/pending proof
    speed-row and Modal paths thread
    --owner-search-owner-local-transition-derivation
  validation:
    uv run pytest tests/test_compact_search_replay_contract.py
      47 passed
    uv run pytest tests/test_compact_owner_search_service.py
      48 passed
    uv run pytest tests/test_compact_coach_speed_row_smoke.py
      97 passed
  guard:
    local proof now requires parent transition outcome array transport
    bytes/field-count 0/0, derivation counts matching replay/action-feedback
    counts, cache misses/fallbacks/action mismatches zero, pending/drop zero,
    and derived schema/kind through generic transition-batch telemetry
  H100 launch history:
    r1 failed before row because the wrong checkpoint was supplied
    r2 timed out during local Modal detach and left an empty artifact directory
    r3 ran remotely but failed proof because stale generic transition-batch
      fields reported legacy fixed-coalesced schema/kind; guard fix added a
      regression test for this exact stale-generic shape
    r4 passed proof:
      run opt132-h100-owner-local-transition-derivation-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r4-20260608
      speed/wall 13265.51 env/s / 55.8875s over 741376 env steps
      vs OPT-104 1.045x
      vs columnar r2 0.837x
      derived batches/transitions/transport 181/724/181
      cache hits/misses 724/0
      checksum verified/mismatch 724/0
      fallback/pending/drop 0/0/0
      parent outcome array bytes/fields 0/0
      normal-death terminal contract true
      GPU utilization sampling false
  decision:
    preserve owner-local derivation as support; do not repeat unchanged
    move to broader fixed owner-buffer/root/search/mechanics/learner-publication
```

Current concrete next rung:

```text
broader fixed owner-buffer/root/search/mechanics/learner-publication patch:
  purpose:
    stop moving isolated proof-clean surfaces and move a larger owner boundary
  copy the mature pattern:
    fixed buffers + slot/generation/digest handles
    owner-resident mechanics/root/search/replay state
    learner consumes fixed-shape batches and publishes small model/version refs
  target one or more measured surfaces:
    parent wait / owner search dispatch
    mechanics/observation ownership
    replay append plus sample table without moving cost into learner train
    learner publication/update cadence
  guard:
    no parent rows or search payload bytes
    normal-death/action/terminal/tensor-native gates preserved
    support gates preserved when claimed: owner-local derivation,
      root-build-request/resident-root/stub, fixed-slot, direct replay/columnar
  local toy constraint:
    fixed-action-tape probes passed parity/proof but are not the speed lane
    mechanics-only B1024/m724/w180: fixed direct 0.843x compact profile
    rendered slab/replay B128/m180/w45: fixed direct 1.129x toy delta, proof
      closed, CPU-oracle render 218.223s, no learner/GPU/H100 claim
    therefore the first implementation must combine ownership boundaries rather
      than replacing manager.step with a private direct env call
  first local slice now landed:
    config flag compact_owner_action_step_boundary default-off
    profile loop owns cached next action and verifies manager.step payload action
    local proof counters pass; no H100 claim
    speed-row producer/Modal wrappers now project the proof and reject missing
      direct-root/action-only boundary proof
  direct-root local gate now landed:
    focused test test_hybrid_profile_owner_action_step_boundary_uses_direct_root_build_request
      monkeypatches the legacy parent root-batch builder to raise and proves the
      boundary through CompactOwnerSearchDirectStepperV1 plus
      CompactLazyThreadedOwnerSearchSlabProxyV1
    required local proof: parent replay rows 0, search payload bytes 0, parent
      root builder 0, root-build-request handoff/owner-build true,
      resident-root/stub proof true, host H2D/D2H 0, and action checksums
      prove previous owner-selected action became the next manager.step action
    validation: focused boundary tests 3 passed; ruff passed
  mechanics/observation support cut now landed:
    resident-observation/no-scalar terminal rows skip dense host final_observation
      allocation because sparse resident device final rows already own the sidecar
    telemetry counts elided dense host bytes/rows
    validation: focused source-state tests 4 passed; ruff passed
  first owner mechanics step-view rung now landed:
    config flag compact_owner_mechanics_step_boundary default-off
    requires compact_owner_action_step_boundary, resident_observation_search,
      direct_root_build_request, and resident_root_host_observation_stub
    manager.step now builds CompactOwnerMechanicsStepViewV1 after mechanics/
      resident observation update and before _make_compact_batch
    the view deliberately sends observation=None and carries only borrowed
      root/mechanics sidecars plus resident handle
    focused direct-root test monkeypatches both parent _make_compact_batch and
      legacy parent build_compact_root_batch_v1 to raise and still passes
    proof counters: parent compact-batch builder/object count 0, parent root
      builder 0, host observation/final-observation bytes 0, step-view count
      positive, action feedback checksums preserved
    validation: expanded owner-boundary/guard packet 6 passed, ruff passed
  owner-published dense next-action rung now landed:
    CompactSearchActionStepV1 carries optional dense_joint_action
    owner direct-root fixed-slot action results derive the dense (batch, player)
      action from validated owner root/search sidecars
    the direct owner stepper validates checksum/digest/legality/active selected
      actions and applies the dense action without parent sidecar reconstruction
    focused direct-root test monkeypatches parent dense-action reconstruction to
      raise and still passes
    proof counters: owner dense action present/used true, parent assembly
      avoided true, fallback 0/none, mismatch 0, bytes/checksum/digest nonzero,
      parent dense-action reconstruction count 0
    validation: owner-boundary dense-action packet 5 passed, direct-root
      owner-service packet 4 passed, ruff passed
  pending compact/root/action identity handle rung now landed:
    direct owner stepper no longer stores parent compact-batch/mechanics-step-
      view or root-batch sidecars in _pending
    pending state keeps action-step identity handles and dense next-action proof
    proof counters: pending compact-batch sidecar stored false, storage avoided
      true, store count 0, avoided count positive; pending root-batch sidecar
      stored false, storage avoided true, store count 0, avoided count
      positive; action identity handle stored true, root-build request stored
      false
    validation: direct-root focused tests 2 passed, owner-step-frame packet
      8 passed, full owner-search service 50 passed, ruff passed
  direct-root returned root-sidecar rung now landed:
    direct_root_build_request=True now returns CompactRolloutSlabStepV1 with
      root_batch=None instead of materializing a parent request-derived root
      sidecar for the returned slab step
    slab telemetry reads root count and resident host-stub metadata from
      CompactRootBuildRequestV1
    owner-local transition derivation validates next-root mechanics outcome
      sidecars from the root-build request, not a parent-built root sidecar
    focused direct-root owner-local test monkeypatches both the parent root
      builder and _root_batch_sidecar_from_build_request to raise and still
      stages a derived transition batch
    proof counters: return root-batch sidecar stored false, storage avoided
      true, build count 0; root batch build sec 0; root count preserved
    validation: request-outcome contract packet 2 passed, direct-root owner
      packet 3 passed, combined owner/profile packet 6 passed, full
      owner-search service 50 passed, ruff passed
  owner/proxy previous-transition closure rung now landed:
    CompactRootBuildRequestV1 carries validated joint_action so the proxy can
      verify the previous owner-selected action was applied
    owner_proxy_transition_closure=True makes the direct stepper call
      stage_owner_proxy_transition_from_root_build_request instead of
      _stage_previous_derived_transition
    CompactOwnerSearchSlabProxyV1 caches previous action frames, validates
      current applied actions, builds the existing derived transition-batch
      schema, and stages it before owner request/train scheduling
    focused proof monkeypatches parent previous-transition closure, derived
      flush, and commit helpers to raise while the real direct-root threaded
      proxy still appends a derived transition batch
    mismatch proof corrupts applied action and fails closed before replay append
    proof counters: parent previous-transition closure count 0, proxy closure
      closed count 2, batch/transition count 1/2, applied-action verification
      2, mismatch 0, fallback 0/none, parent applied-action validation 0
    validation: focused proxy-closure packet 4 passed, full owner-search
      service 52 passed, replay-contract suite 47 passed, ruff passed
    launch plumbing correction: the public speed-row CLI, local Modal launcher,
      and remote Modal producer now expose
      --owner-search-owner-proxy-transition-closure and project/report its
      proof fields. Earlier proxy-closure support was local-only; H100 commands
      could not honestly preserve this gate before this correction.
  owner action dispatch-handle split now landed:
    CompactOwnerSearchSlabProxyV1 now exposes submit/resolve around direct-root
      owner action search:
      submit_action_step_from_root_build_request returns
      CompactOwnerActionDispatchHandleV1 without calling worker.result
      resolve_action_step_handle waits, reads the fixed action-result slot, and
      builds the CompactSearchActionStepV1
    run_action_step_from_root_build_request remains a sync compatibility wrapper
      over the handle path and advertises sync-wrapper proof fields
    lazy threaded direct-root proxy forwards submit/resolve
    proxy-closure mode no longer stores the parent action-step identity handle
      in the direct stepper pending state; the proxy cached action frame is the
      previous-action authority
    focused proof monkeypatches worker.result to raise during submit, proves the
      fixed action-result slot read count stays 0 until resolve, then resolves
      once and checks dispatch submit/resolve/pending/wait-in-submit counters
    proof fields: action-dispatch handle used true, submit-no-wait true,
      submit/resolve counts, pending count, max pending count, result-wait-in-
      submit count 0, sync-wrapper flag/count, completed-at-submit count 0,
      pending action-step storage avoided in proxy closure, proxy action-frame
      pending/store count
    validation: action-dispatch/direct-root/proxy packet 5 passed, profile
      boundary packet 5 passed, full owner-search service 55 passed,
      replay-contract suite 47 passed, full speed-row smoke 99 passed, ruff
      passed
  profile-loop action-dispatch overlap now landed:
    CompactOwnerSearchDirectStepperV1 exposes submit_step/resolve_step around
      the direct-root owner action request
    HybridBatchedObservationProfileManager.step can submit the slab owner action
      request, run the existing parent post-slab work, then resolve before
      returning the next joint action
    the overlap window currently covers real parent work already present in the
      loop: optional batched probe, scalar/no-scalar materialization, payload
      pickle, resident replay snapshot, render state snapshot, and minimal/full
      step payload snapshot
    speed-row producer, local Modal launcher, and remote Modal producer carry
      --compact-owner-action-dispatch-step-overlap and require the proof when
      requested
    proof fields: enabled/proof_passed, supported/used, submit-no-wait true,
      sync-wrapper false, cumulative slab/owner sync-wrapper counts 0,
      completed-at-submit count 0, submit_count == resolve_count == total
      iterations, pending_count 0, max_pending_count positive,
      parent_work_sec positive, wait-in-submit count 0
    speed-row proof guard now requires these fields on the raw profile payload
      before default/projection normalization, so omitted wait/sync/completion
      counters fail closed instead of becoming fake zeros.
    validation: focused source-state overlap tests passed, full owner-search
      service 55 passed, replay-contract suite 47 passed, full speed-row smoke
      99 passed, ruff passed
  next implementation gate:
    one same-work H100 candidate is now allowed only if it preserves the
      columnar/direct-table support stack plus direct-root/action-only/fixed-
      slot/proxy-closure gates and requests both
      --owner-search-owner-proxy-transition-closure and
      --compact-owner-action-dispatch-step-overlap
    the H100 row must prove the overlap was real: no sync wrapper, submit wait
      in submit 0, completed-at-submit 0, submit/resolve counts equal measured
      iterations, pending 0, max pending positive, and overlapped parent work
      positive
    suggested run id:
      opt132-h100-action-dispatch-overlap-proxyclosure-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608
    if the H100 delta is flat or negative, do not repeat unchanged; go directly
      to fixed owner mechanics-frame handles or learner publication/update
      tickets/refs
    H100 launch remains refused for mechanics step-view, dense-action,
      pending/returned root sidecar, owner/proxy closure, action-dispatch
      handle, or profile-loop overlap proof rungs when used alone or with a
      broken flag stack
```

The parent-builder-avoiding root-build-request gate is now locally closed:

```text
run:
  opt132-local-owner-rootbuildrequest-threaded-directtransition-smoke-20260607-r3
status:
  ok=true, local proof only
proof:
  schema/kind curvyzero_compact_root_build_request/v1 / resident_root_view_build_request_v1
  publish/resolve/owner-build 15/15/15
  parent builder used/calls false/0
  parent build sec 0.0
  request observation bytes 0
  resident-root/stub/action-only gates preserved
  parent rows 0/0
  search payload bytes 0
  direct replay batches/transitions/transport 3/12/3
  action mismatches 0
  maintenance/policy lag 0/0
decision:
  H100 decision row ran and was killed for speed:
  opt132-h100-rootbuildrequest-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607
  speed/wall 11327.75 env/s / 65.4477s
  proof stayed clean, including publish/resolve/owner-build 904/904/904 and
  parent builder false/0
  keep root-build-request as support and move to the larger owner
  buffer/search/learner-publication patch
```

The local ceiling instrument for that test now exists:

```text
fields:
  compact_whole_owner_buffer_replay_ceiling_*

status:
  local/report/evidence/Modal projection surfaces implemented
  projection-only guard fails closed on production-speed/currency drift
  compare script now backfills the same ceiling for old H100 artifacts
  no hot-loop behavior changed

artifact:
  artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-whole-owner-buffer-ceiling-review-20260607/comparison.json

generated H100 read:
  columnar r2 projected replay/sample removal:
    24903.25 env/s, 1.9625x OPT-104, 29.770s wall, 0.558s short of 2x
  direct-table r3 projected replay/sample removal:
    24138.94 env/s, 1.9023x OPT-104, 30.713s wall, 1.500s short of 2x
  fixed-SoA locality r2 projection:
    14986.22 env/s, 1.1810x OPT-104, 49.471s wall, 20.258s short of 2x
```

The fastest columnar row already has the action-only owner boundary active:
owner materializes replay, parent reconstruction false, parent slab commits
false, search payload / visit policy / root value bytes all `0`, replay payload
handle present, and inner two-phase deferred device replay true. Do not spend
the next H100 slot on "make it action-only"; that boundary is already spent in
the current fastest input. The next implementation target is the measured
owner-search/root/mechanics/parent-wait surface or learner-publication overlap.

The latest local support smoke is explicitly not a speed row:
`opt132-local-fixed-soa-slot-digestdefer-smoke-20260607-r3` passed as tiny
`profile_no_death` wiring proof for fixed-SoA slot-candidate replay plus
owner-ref digest deferral. It proves the flags compose through the speed-row
entrypoint with fixed SoA requested/used, slot-candidate learner path true,
zero fixed-SoA object/table leakage, columnar append false, action-only payload
bytes `0`, and digest deferral true with learner-side digest time `0.0s`.
Do not run this combination as a standalone speed row unless it is explicitly
labeled ablation/support and preserves normal-death/action/replay/terminal
proof gates.

The fastest exact same-window support candidate before fixed-SoA remains the
columnar/direct-table stack at `15852.67 env/s`, but that path is not the
architecture finish and was not repeated. The current architecture question is
whether a broader owner-buffer rewrite can remove replay/sample wall and at
least another roughly half-second from search/parent-wait/learner/publication
without moving cost elsewhere.

Local support patch after the latest owner-boundary reset:

```text
patch 1:
  deferred owner maintenance root-cache snapshots are now sliced to replay
  record_indices, next_record_indices, and current actor_step when the replay
  entries are introspectable
proof:
  counters expose snapshot count, full entries, retained entries, required
  entries, dropped entries, and full-fallback count in service metadata and
  owner sample telemetry
validation:
  ruff passed
  focused direct-transition pytest passed
  full owner-search service pytest passed: 47 passed
decision:
  support only; do not treat this as speed evidence
```

Local fixed action-result slot proof:

```text
patch 2:
  same-process inline/threaded direct-root owner proxies can return a tiny
  action-result slot stub through the worker result queue while the full owner
  action result stays in CompactOwnerActionResultSlotTableV1
proof:
  requested/used, slot count, acquire/write/read, pending slots, wire result
  bytes, and full result bytes are projected into action metadata
validation:
  ruff passed
  focused root-build/direct-transition pytest passed
  full owner-search service pytest passed: 47 passed
  speed-row launcher/proof plumbing focused pytest passed: 7 passed
  full speed-row smoke pytest passed: 96 passed
local speed-row proof:
  run opt-fixed-action-result-slot-local-smoke-proof5-20260608
  ok true, status complete
  tiny CPU proof speed 548.91 env/s, not a speed claim
  fixed slot requested/used true/true
  slot count 4, acquire/write/read 5/5/5, pending 0
  wire/full result bytes 413/712
  action-only true, direct-root true, owner owner_search_worker
  owner train/expected train 1/1, tensor-native replay true
H100 decision row:
  run opt132-h100-fixed-action-result-slot-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260608
  proof clean, speed rejected as main lane
  speed/wall 12794.42 env/s / 57.9453s
  vs OPT-104 1.008x by env/s
  vs columnar r2 0.807x by env/s
  fixed slot requested/used true/true
  slot count 4, acquire/write/read 904/904/904, pending 0
  wire/full result bytes 414/4837
  parent wait 19.035s, down from root-build-request 26.151s but above columnar r2 17.655s
  replay append 19.931s, learner train 12.880s, observation 13.259s
decision:
  keep fixed-slot as support because it improved the root-build-request row, but
  close it as the main speed lane because it regressed hard versus columnar r2
fallback:
  owner-local transition derivation has also been run and speed-rejected as a
  standalone lane. Switch to the broader fixed owner-buffer/root/search/
  mechanics/learner-publication patch. Do not spend the next step on fixed-slot,
  replay-only, owner-local-only, or fixed-SoA tweaks.
```

The one permitted H100 deferral probe has now run:

```text
run:
  opt132-h100-owner-deferred-one-sim-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607
status:
  proof clean, speed rejected
speed/wall:
  13691.98 env/s / 54.1467s
proof:
  direct transition count 724
  columnar append used true
  direct flush/materialized/identity/recurrent 724/724/724/724
  model-refresh-crossed 0
  replay-payload D2H 0.0
  owner-inner pending final 0
speed read:
  beats OPT-104 only modestly at 1.079x
  regresses versus columnar r2 at 15852.67 env/s
  direct append rose to 22.377s with deferred/device flush 3.613s/4.381s
decision:
  do not repeat this lane unchanged
  return to broader owner-boundary work
```

The in-process async learner overlap probe has also run:

```text
run:
  opt132-h100-owner-asynclearner-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607
status:
  proof clean, speed rejected
speed/wall:
  12954.74 env/s / 57.2282s
proof:
  async submit/completed/pending 90/90/0
  max pending observed 2
  actions while async pending 510
  failed false
decision:
  do not repeat in-process async learner overlap unchanged
  keep overlap as an architecture idea only if learner publication/update
  ownership changes, not as this same thread tweak
```

Local owner-resident root-view proof now exists for the direct-root threaded
owner path:

```text
run:
  opt132-local-owner-resident-root-view-threaded-directtransition-smoke-20260607-r5
status:
  local proof clean; not speed evidence
proof:
  resident root view required/proved true
  kind direct_root_batch_resident_handle_v1
  direct root publish/resolve 15/15
  resident-root-view H2D/D2H 0.0/0.0
  host fallback allowed false
  action-only true, owner materializes replay true
  parent committed/stored rows 0/0
  search payload/visit/root-value bytes 0/0/0
  transition entries/batches/transport entries 12/3/3
  policy lag/pending maintenance/failed 0/0/false
decision:
  this proves the owner can consume the source resident root handle in
  direct-root threaded mode; it does not yet prove H100 speed or remove parent
  root-batch construction
```

Local resident host-observation stub proof now exists on top of that gate:

```text
run:
  opt132-local-owner-resident-root-hoststub-threaded-directtransition-smoke-20260607-r1
status:
  local proof clean; not speed evidence
proof:
  resident host-observation stub requested/stubbed true
  stub kind zero_stride_shape_only_v1
  materialized bytes 0
  logical bytes 262144 last step / 3932160 total
  resident-root-view required/proved true
  H2D/D2H 0.0/0.0
  host fallback false
  action-only true, owner materializes replay true
  parent committed/stored rows 0/0
  search payload bytes 0
  transition entries/batches/transport entries 12/3/3
  policy lag/pending maintenance/failed 0/0/false
decision:
  this removes parent host-observation materialization in the direct-root
  resident proof path; it still calls the parent root-batch builder, so it is
  not a standalone H100 row
```

```text
baseline:
  OPT-104 validation-only, 12689.381637 env/s, 14.5255305s wall

current candidate shape:
  vectorized seeded reset RNG
  policy_refresh_interval=4
  owner_search_threaded_proxy
  owner-search deferred maintenance
  background maintenance overlap
  fused learner batch
  learner-ready unroll-2 cache
  tensor-native replay
  borrowed single-actor render state
  compact Torch direct_core
  normal death

r32:
  speed/wall: 14002.12 env/s / 52.947s
  actions while maintenance pending: 608
  slab: 21.881s

r33 repeat:
  speed/wall: 13640.60 env/s / 54.351s
  actions while maintenance pending: 543
  slab: 21.166s

r34 long:
  window: 1448 / 360
  speed/wall: 13145.34 env/s / 112.797s
  baseline-equivalent wall: 116.850s
  actions while maintenance pending: 1131

r2 transition-batch only:
  speed/wall: 13618.91 env/s / 54.437s
  logical/transport replay transitions: 724 / 181
  worker replay append: 31.926s

r3/r4 ring-batched replay append:
  speed/wall r3: 14394.77 env/s / 51.503s
  speed/wall r4: 14160.27 env/s / 52.356s
  average: 14277.52 env/s / 51.930s
  worker replay append average: 19.420s

local direct-prebuilt sample patch:
  status: implemented locally, no standalone H100 row planned
  mechanism: tensor-native maintained unroll-2 tables can feed the prebuilt
    learner batch without constructing per-sampled-group resident sample
    objects
  proof fields: direct prebuilt requested/eligible/used, fallback count/reason,
    direct group-object count, group-object build skipped
  local gate: focused ruff and tensor-native/source-state/speed-row/modal proof
    tests pass
  decision: keep as a fail-closed support path, but do not spend a standalone
    H100 row on it; r3/r4 timing shows the skipped group-object surface is tiny
    next to replay append/materialization.
```

Do next, in order:

1. Stop fixed-SoA gather/locality as the main lane. Keep the code default-off
   as a causal diagnostic. Do not build exact flat/global row layout unless the
   implementation also removes broader object/materialization surfaces.
2. Treat the generated whole-owner-buffer ceiling review as the current
   decision artifact. Replay append plus owner train sample removal projects
   close to, but still below, `2x` on the fastest row. Do not launch a
   replay-layout-only H100 row unless it also changes a measured non-replay
   surface or proves real overlap.
3. Use the resident-root-view and resident host-observation stub local proofs
   as gates, not as speed claims. The next code target must remove a larger
   measured surface: parent root-batch construction, owner search
   dispatch/parent wait, mechanics/observation ownership, or learner
   publication/update. Preserve the new
   `compact_owner_search_resident_root_view_*` and
   `compact_rollout_slab_resident_host_observation_stub_*` proof fields
   whenever direct root mode claims resident ownership.
4. Close the default-off compact Torch deferral lane as proof-clean but
   speed-rejected. Local r7/r8 proved the owner-loop mechanics; the one allowed
   H100 probe above kept the proof closed but regressed versus columnar r2. Do
   not run repeats unless a later patch changes the owner boundary or overlaps
   the deferred work differently.
5. Stop owner-worker-mode polish. `owner_search_inline_background_proxy` has
   been tested: r36/r37 short rows were positive, but r38-long failed below
   OPT-104 and below threaded r34-long.
6. Treat ring-batched replay append as the current incremental best, not the
   final architecture. It moved replay append from r2 `31.926s` to r3/r4
   average `19.420s` and full speed to r3/r4 average `14277.52 env/s`; this is
   real but still only about `1.13x` OPT-104, not `2x`.
7. Do not run a standalone H100 verification of the direct-prebuilt sample
   patch. It skips per-sampled-group resident sample objects, while r3/r4 show
   replay append/materialization is still about `19.420s` and the direct
   group-object/builder surface is measured in milliseconds. The expected
   whole-row effect is too small to justify another remote row.
8. Keep fixed SoA/ring-buffer replay storage as a support component only when
   it is part of the broader owner-resident buffer/root design. Bypass
   transition-batch expansion, index-entry wrappers,
   `previous_step/current_step` objects, and `_CompactReplayRingEntry` object
   storage, but do not treat those counters alone as the speed target.
9. Preserve reset/autoreset parity, owner maintenance closure,
   terminal proof, tensor-native fallback `none`, and action-feedback/replay
   ownership.
10. Run local toy/fixture proof first; use H100 only when the owner graph or hot
   data movement has changed.
11. Keep r14-r17, inline r30/r31, threaded r32/r33/r34, and
   inline-background r36/r37/r38 as historical clues, not the active next
   move.

## Historical Now Before r29-r34

OPT-132 is still active, and the task changed again.

The r14 repeatability gate has now run and failed. The immediate next move is
to diagnose the r14/r15 exact-repeat timing swing, not to claim promotion.

```text
r14:
  artifact: artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-warmupgate-b1024a1-normal-unroll2-r14-20260605/row_001_result.json
  speed: 13497.30 env steps/sec
  wall: 13.6561s
  baseline: OPT-104, 12689.38 env steps/sec, 14.5255s
  delta: +6.36% speed, -0.869s wall
  route: owner_search_inline_proxy + compact_torch_search_service
  mechanism: shared_model_state_v1 refresh + no inline host payload clone
  train/sample/update requests: 22
  refresh interval/request/skip/update: 1 / 22 / 0 / 22
  final update consumed: true
  warmup suppressed replay entries: 44
  parent committed replay rows: 0
  search-result payload bytes: 0
  final owner drain: 0.000042s
  status: first real H100 win, failed exact repeat

r15 exact repeat:
  artifact: artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-warmupgate-b1024a1-normal-unroll2-r15-20260605/row_001_result.json
  speed: 10502.70 env steps/sec
  wall: 17.5498s
  proof: exact identity, accepted-fast-path violations []
  train/sample/update requests: 22
  refresh interval/request/skip/update: 1 / 22 / 0 / 22
  parent committed replay rows: 0
  search-result payload bytes: 0
  final owner drain: 0.000044s
  status: speed repeat failed

r14/r15 comparison:
  artifact: artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-r14-r15-sharedmodel-nopayloadclone-repeat-comparison-20260605/comparison.json
  identity: exact
  stable_speed_claim_allowed: false
  wall spread: 13.6561s -> 17.5498s, 24.95% of median
  largest timing ranges:
    compact rollout slab: +2.128s
    actor step wall: +1.047s
    observation: +0.681s
    observation other: +0.665s
    actor autoreset: +0.514s
    actor env runtime: +0.488s

r16 longer same-mechanism row:
  artifact: artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-warmupgate-b1024a1-normal-unroll2-m724-w180-r16-20260605/row_001_result.json
  speed: 6491.80 env steps/sec
  wall: 114.202s
  expected train/update/refresh: 90 / 90 / 90
  owner train sample/update: 52.807s / 3.236s
  digest/state_dict/host clone: 0 / 0 / 0
  status: longer run did not rescue speed; steady-state sample path dominates

r17 fast-batch composition attempt:
  artifact: artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132-h100-owner-action-only-inlinelearner-sharedmodel-nopayloadclone-fusedtensor-warmupgate-b1024a1-normal-unroll2-m724-w180-r17-20260605/compact_coach_speed_row_modal_report.json
  status: failed before row creation
  problem: compact_owned_loop_fused_learner_batch requires compact_owned_loop_entrypoint
  meaning: fused/tensor-native learner-batch path is not currently wired into
    owner-search inline route
```

Do next, in order:

1. Treat r14 as a useful fast row and r15 as the failed exact repeat. No stable
   speed claim.
2. Wire or prove the equivalent of the fast learner-ready/tensor-native batch
   path inside owner-search train sampling. r16 shows long-run owner train
   sample dominates; r17 shows the existing fast flags do not compose by CLI.
3. After that integration, rerun a longer `724/180` row before another
   promotion claim.
4. Keep documenting `shared_model_state_v1` as a same-object version-token
   refresh, not a model-state digest transfer. The row is honest because the
   learner/search model object is shared in the inline path; it would be fake
   only if reported as a distinct-worker digest handoff.

The launcher preset and remote-result guard now exist. The old flag-drift
problem should fail closed instead of producing a misleading accepted row.

The latest accepted-fast-path preset pair is mixed:

```text
OPT-132-G:
  speed: 13649.29 env steps/sec
  wall: 13.504s
  direct_core + fused + borrowed render state + lean trainer
  render-state copy steps: 0
  render-state borrowed steps: 225
  terminal sample/target rows: 167/167
  normal death: true
  truncations: 0
  host fallback: 0
  direct autoreset count: 0
  result: beat OPT-104, but not repeat-confirmed

OPT-132-H:
  speed: 11366.84 env steps/sec
  wall: 16.216s
  same accepted proof fields
  result: failed repeat confirmation

OPT-132-I:
  speed: 13308.59 env steps/sec
  wall: 13.8497s
  same accepted proof fields under the corrected signed-checksum guard
  result: beat OPT-104, but not enough to close while OPT-132-H remains slow

OPT-132-J:
  speed: 10057.98 env steps/sec
  wall: 18.3258s
  same accepted proof fields under the corrected signed-checksum guard
  result: clean slow repeat; same trajectory/sample-order checksums as I
```

Prior useful row:

```text
OPT-132-F:
  speed: 12177.36 env steps/sec
  wall: 15.136s
  useful sparse resident terminal final-observation improvement
```

Target controls:

```text
OPT-104: 12689.38 env steps/sec, 14.5255s wall
OPT-109: 12527.10 env steps/sec, 14.714s wall
OPT-117: 12566.17 env steps/sec, 14.668s wall
OPT-118 r1: 12795.45 env steps/sec, 14.405s wall
```

## Next Architecture Target

Owner-search deferred-maintenance action-only replay ownership is the selected
OPT-132BF primary-residual candidate. The already-observable proof gates now
fail closed locally, including action-feedback checksums and explicit
maintenance work-item vs replay-entry drain accounting and compact Torch inner
two-phase device replay. The post-prewarm normal-death scale/cadence read has
now answered: proof survives, speed shape fails. Slab and threaded scale rows
both run about `25 env steps/sec` locally because owner train/learner update
absorbs about `14-15s` and final drain absorbs about `12-13s`. The mock-fast
ceiling row has now separated the surfaces: with neural update removed, the
same ownership proof runs `214.06 env steps/sec` locally and final drain drops
to `0.0026s`. The next read is not H100 and not another attribution row; it is
a real-learner resource/work-shape implementation followed by the same local
normal-death scale row. Local MPS placement, eager append pre-drain, and
in-process async learner overlap have now preserved proof and failed whole-row
speed, so the next patch must change learner ownership/resource/work shape
rather than device placement, maintenance scheduling, or same-process futures.
Maintained replay sampling is no longer the active P0:
OPT-132BD proved tensor-native replay on H100, and OPT-132BF proved the
maintained sample-universe layer. The sample gate is fixed enough to stop
squeezing it; the full loop is still slow.

```text
OPT-132BD:
  speed: 3884.88 env steps/sec
  wall: 285.727s
  proof: accepted-fast-path violations [], learner-ready violations [],
    tensor-native violations []
  tensor-native: used true, impl maintained_unroll2_table_gather_v1,
    table source maintained_record_table_v1, reused records 1080,
    missing records 0, fallback count 0
  sample gate: 74.005s total, 135 calls, p50 0.553s, p95 1.125s
  candidate scan: 28.734s total, p50 0.201s
  RNG: 6.479s total, p50 0.049s
  residual: 35.897s total, p50 0.250s
  tensor concat/gather: 0.0127s / 0.00033s
  decision: proof plumbing was solved; speed blocker at that point was
    per-sample rebuilding of candidate/index/RNG surfaces, not tensor-native
    learner batch gather
```

The maintained candidate / offset / sampleable-row state artifact is now in the
tree and has been read on H100. Do not add another pure attribution row.

Local first layer:

```text
OPT-132BE local:
  feature: maintained_sample_universe_v1 for unroll-2 sample candidates
  proof: partial-terminal and mixed-candidate tensor-native equality pass
  benchmark: records 32, rows/record 8, sample rows 128, CPU
  current ring sample/build median: 0.004726s
  real tensor-native replay sample/build median: 0.000501s
  local speedup vs current: 9.44x
  result: local proof was good enough to justify the H100 proof row
```

H100 result:

```text
OPT-132BF:
  speed: 5362.68 env steps/sec
  wall: 206.989s
  proof: accepted-fast-path violations [], learner-ready violations [],
    tensor-native violations []
  sample gate: 4.116s total, p50 0.030s, p95 0.055s
  candidate scan: 0.332s total, p50 0.0024s
  RNG: 0.774s total, p50 0.0058s
  residual: 0.535s total, p50 0.0039s
  decision: maintained sample universe fixed the sample-gate bottleneck but
    did not fix full-loop speed
```

Latest local architecture artifact:

```text
OPT-132BK:
  artifact: artifacts/local/curvytron_compact_coach_speed_row_results/
    opt132bk-local-fixed-action-closed-slab-replay-sample-toy-b4-m3-w1-20260604.json
  status: pass
  proof: deterministic slab-selected action feedback, replay append, index
    rows, replay-ring sample, sample-batch checksums, compact/direct equality
  rows: slab roots 48, selected actions 36, committed index rows 28
  replay/sample: replay appends 3, sample gates 3, sample batch size 8
  limit: local architecture proof only; not compact Torch/MCTS/learner/H100
```

Superseded pre-r14 artifact target: real-learner resource/work-shape proof.
Explicit local MPS
placement preserved proof but slowed the whole row, eager append pre-drain
preserved proof but slowed the normal-death scale row to `24.09 env steps/sec`
with learner update `15.12s` and final drain `12.67s`, and in-process async
learner overlap preserved async proof but still ran only `24.47 env steps/sec`.
Increasing same-process async max pending to `6` only reached
`25.37 env steps/sec`. This is no longer the next artifact after r14; keep it
as evidence that placement/scheduler/same-process async lanes were exhausted.
Historical next artifact at this point was an exact r14 repeat, then an
optional strict interval-4 shared/no-clone cadence check. That lane is now
superseded by the r32/r33/r34 threaded/background candidate and the next
headroom patch.
The remaining measured H100 surface is not replay gather anymore: primary
residual `122.679s`, actor step wall `41.852s`, actor/autoreset `29.656s`,
search dispatch `14.248s`, observation `11.423s`, env runtime `10.000s`. The
fixed-action toy now has zero-stub proof,
terminal/autoreset proof, rendered-observation proof through OPT-132BI, and
fixed-shape search/root proof through OPT-132BJ, and closed slab/replay/sample
proof through OPT-132BK. The selected owner-search deferred-maintenance
action-only replay path now has a local real-entrypoint smoke:
`opt132-local-owner-action-only-profile-proof-20260604`. It proves parent
committed/stored rows `0/0`, zero search payload bytes, owner replay
entries/submitted/append `15/15/15`, owner train/update/drain counts, and final
owner drain inside measured wall. Local speed `70.93 env steps/sec` is not speed
evidence. The hardened smoke
`opt132-local-owner-action-only-proof-hardened-fields-20260604` also proves
requested cadence/train steps, owner sample telemetry, zero-batch sample
semantics, finite final drain, train timing split fields, and required
action-only/two-phase handle fields. The fresh action-feedback smoke
`opt132-local-owner-action-only-action-feedback-proof-20260604` adds
transitions/actions/mismatches `15/240/0` and expected/applied/replay checksums
`6120/6120/6120`. Nested learner timing aggregation is now done, and
`opt132-local-owner-action-only-learner-import-prewarm-proof-20260604` drops
train wall/update/final drain to `0.072s/0.057s/0.043s`. Explicit drain proof
`opt132-local-owner-action-only-explicit-drain-work-entry-proof-20260604`
passes with append requests/submitted/appended `15/15/15`, staged/drained work
items `15/15`, drained replay entries/appends `15/15`, pending/inflight/failed
`0/false/false`, parent rows `0`, payload bytes `0`, action feedback true, and
final drain `0.047s`. Inner two-phase owner-search proof
`opt132-local-owner-action-only-inner-two-phase-explicit-drain-proof-20260604`
passes with inner two-phase/device replay true, parent rows `0`, payload bytes
`0`, train wall/update/final drain `0.065s/0.054s/0.037s`. Normal-death scale
then fails speed shape despite preserved proof:
`opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-20260604`
reports speed `25.78`, train/update/final drain `14.23s/14.07s/11.97s`;
`opt132-local-owner-action-only-inner2-threaded-scale48-cadence8-b512-normal-20260604`
reports speed `25.09`, train/update/final drain `14.94s/14.77s/13.00s`.
Mock-fast ceiling
`opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-mockfast-r3-20260604`
passes the normal-death contract without relaxing it, carries owner learner
value-valid/done/truncated counters, keeps parent rows `0`, payload bytes `0`,
staged/drained work `55/55`, drained replay entries/appends `55/55`, and runs
`214.06 env steps/sec` locally with train wall `0.091s`, learner update `0.0s`,
and final drain `0.0026s`. Decision: owner-search/search/replay mechanics are
not the local scale blocker once neural update is removed. MPS placement then
failed at `20.81 env steps/sec`; eager append pre-drain then failed at
`24.09 env steps/sec` despite eager drains `7`, policy lag current/max `0/6`,
and submitted/owner/refreshed updates `6/6/6`; in-process async learner overlap
then failed at `24.47 env steps/sec` despite async submit/completed/pending
`6/6/0` and actions while async pending `47`. R14 supersedes this as the active
P0 by winning on H100 through shared-model refresh and no inline payload clone.

Latest exact `1084/270` evidence:

```text
rows: OPT-132AZ r1/r2
identity: exact
stable speed claim: false
wall/speed: 217.391s / 5106.07 and 180.892s / 6136.36
wall spread: 36.500s, 18.329% of median
sample gate: 136.622s / 109.260s
learner-batch build: 85.001s / 65.049s
builder group-loop: 84.177s / 64.278s
group-loop process CPU: 80.37s / 61.70s
unroll-fields process CPU: 42.13s / 32.24s
unroll accounted/residual CPU: 30.29s / 24.21s and 11.84s / 8.03s
terminal metadata CPU: 15.58s / 11.44s
prepare CPU: 10.44s / 8.33s
decision: no speed claim; stop pure attribution. Builder-surface reshaping was
  the next replay candidate, but BF has since closed the sample-gate surface.
  Later owner-search work selected the action-only path, transition batching,
  and then ring-batched replay append/cache refresh. The active artifact is now
  the next materialization cut after that ring-batch patch.
```

OPT-132AZ bounded the remaining same-work swing enough to stop pure
attribution. The toy and real sample-gate path then made learner-ready
tensor-native replay the active lane, and BD/BF answered it: proof passed,
sample gate collapsed, full-loop speed did not. BA remains useful proof
plumbing, not the whole plan. Do not run another H100 row just to project
replay fields.

The first long-window diagnostic pair has been run with the same fast-path
flags:

```text
--compact-owned-accepted-fast-path-preset
--compact-owned-accepted-fast-path-step-window stability_724_180
```

First comparison:

```text
K: 7365.79 env steps/sec, 100.651s wall
L: 8297.18 env steps/sec, 89.353s wall
identity: exact
wall swing: 11.298s
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132kl-longwindow-stability-comparison-20260602/comparison.json
```

Repeated A/A packet:

```text
rows: K/L/Q/R/S
identity: exact
accepted-fast-path violations: 0
wall: min 89.353s, median 100.816s, max 137.297s
wall range: 47.945s
sample gate range: 33.063s
learner-batch build range: 18.675s
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132klqrs-longwindow-stability-comparison-20260602/comparison.json
```

These long-window rows are diagnostics, not replacements for OPT-104, unless a
matched long-window baseline is also produced.

Use OPT-132-G/H/I/J only as short-window background. The current measurement
failure is the exact `1084/270` long-window path below. The older G/H shape was:

```text
wall:        G 13.504s vs H 16.216s (+2.712s)
actor:       G 4.073s  vs H 4.924s  (+0.851s)
observation: G 1.984s  vs H 2.495s  (+0.512s)
sample:      G 2.692s  vs H 3.426s  (+0.734s)
learner:     G 1.466s  vs H 1.778s  (+0.312s)
slab/search: G 2.769s  vs H 2.892s  (+0.124s)
```

Do not call this a stable speed win yet. The `724/180` longer window still
swung after five exact-identity rows. `1084/270` and `1444/360` were too large
for H100 before the latest-frame resident replay snapshot patch. After that
patch, `1084/270` now fits on H100, but the exact r1/r2 repeat still failed the
timing stability bar.

Latest same-field `1084/270` evidence:

```text
diag r1 wall/speed: 181.348s / 6120.91 env steps/sec
diag r2 wall/speed: 182.777s / 6073.07 env steps/sec
diag r3 wall/speed: 154.574s / 7181.12 env steps/sec
identity: exact
accepted-fast-path violations: []
resident replay snapshot mode: latest_frame_history
retained resident snapshot bytes: 9346744320 on every row
same-field wall spread: 28.202s, 15.55% of median
same-field sample gate spread: 20.428s, 19.49% of median
same-field learner-batch build spread: 10.654s, 18.88% of median
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132x-h100-learner-batch-timing-diagnostic-r1r2r3-comparison-20260602/comparison.json
decision: r3 broke stability; superseded by OPT-132Y CUDA-sync r1/r2/r3 below
```

Latest CUDA-sync `1084/270` evidence:

```text
rows: OPT-132Y r1/r2/r3
identity: exact, including sync flags and counts
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
r1 wall/speed: 196.699s / 5643.23 env steps/sec
r2 wall/speed: 194.540s / 5705.85 env steps/sec
r3 wall/speed: 255.499s / 4344.51 env steps/sec
wall spread: 60.959s, 30.99% of median
sample gate spread: 36.790s, 30.47% of median
learner-batch build spread: 19.588s, 28.01% of median
actor wall spread: 11.223s, 31.66% of median
observation spread: 8.268s, 62.14% of median
observation-other spread: 8.016s, 75.95% of median
builder cuda_sync: 6.910s / 5.520s / 8.274s, count 11977/11977/11977
sample cuda_sync: 0.030s / 0.028s / 0.051s, count 12/12/12
learner cuda_sync: 0.845s / 0.831s / 0.611s, count 28/28/28
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132y-h100-cuda-sync-timing-diagnostic-r1r2r3-comparison-20260602/comparison.json
decision: exact identity held, but r3 failed stability broadly. Sync wait is
visible but too small to explain the whole wall swing.
```

Latest runtime-envelope `1084/270` evidence:

```text
rows: OPT-132Z r1/r2
identity: exact, including runtime-step diagnostics and count 1084
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
r1 wall/speed: 197.292s / 5626.27 env steps/sec
r2 wall/speed: 273.865s / 4053.14 env steps/sec
wall spread: 76.574s, 32.50% of median
runtime-step sum spread: 76.573s, 32.50% of median
sample gate spread: 64.055s, 42.67% of median
learner-batch build spread: 38.792s, 44.14% of median
runtime p50: 0.063s / 0.073s
runtime p95: 1.150s / 1.784s
runtime max: 1.786s / 2.717s
slowest steps: sample-gate dominated
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132z-h100-runtime-step-envelope-diagnostic-r1r2-comparison-20260603/comparison.json
decision: exact identity held, but r2 failed stability. The wall swing is
inside measured loop iterations and concentrated on sample-gate cadence steps,
not final drain/untracked wall.
```

Local diagnostic now added:

```text
purpose: explain sample-gate learner-batch-build timing, not speed
new fields: learner-batch builder sub-timers
new stats: per-call learner-batch-build count/sum/min/max/p50/p95 and slowest-call proof fields
new flag: --compact-profile-cuda-sync-timing-diagnostics
new sync fields: sample-gate, learner-batch-builder, and learner-gate cuda_sync timing/counts
new runtime envelope: compact_profile_runtime_step_* measured-step wall stats
  plus slowest-step actor/observation/slab/sample/learner/policy/residual context
new sample-call distributions: compact_rollout_slab_sample_gate_*_per_call_stats
  plus flattened total/candidate/rng/residual count/sum/min/max/p50/p95 fields
validation: ruff passed; focused source-profile, speed-row smoke, Modal guard,
and comparator tests passed
latest evidence: OPT-132AA exact 1084/270 r1/r2 read those fields on H100;
do not patch learner-batch builder speed code blindly
```

Latest local builder-child attribution patch:

```text
purpose: explain which learner-batch builder child drives broad sample-gate cost
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
validation: ruff passed; focused source-profile, speed-row smoke, Modal guard,
and comparator tests passed
latest evidence: OPT-132AB exact 1084/270 r1/r2 read those fields on H100;
do not patch learner-batch builder speed code blindly
superseded by: OPT-132AC deep unroll/terminal child timers below
```

Latest builder-child per-call diagnostic read:

```text
rows: OPT-132AB r1/r2
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
stable speed claim allowed: false
wall/speed:
  r1 271.062s / 4095.07
  r2 170.468s / 6511.57
spread:
  wall 100.593s, 45.57% of median
  sample gate 77.654s, 54.93%
  learner-batch build 46.281s, 56.09%
  builder group-loop 46.002s, 56.27%
  unroll fields 21.305s, 53.05%
builder-child sums:
  group-loop 104.758s -> 58.756s
  unroll fields 50.816s -> 29.511s
  terminal metadata 23.072s -> 12.377s
  write output 7.243s -> 4.326s
  builder cuda_sync 9.988s -> 5.930s
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ab-h100-builder-child-per-call-diagnostic-r1r2-comparison-20260603/comparison.json
decision: exact identity held, but timing stability failed hard. The swing is
inside builder group-loop work. Unroll fields are the largest visible child,
then terminal metadata; builder CUDA sync is visible but smaller.
```

Latest deep builder-group diagnostic patch:

```text
label: OPT-132AC
purpose: split the OPT-132AB group-loop/unroll/terminal swing before speed code
new terminal metadata children:
  mask, tensor fallback, validate, final observation
new unroll children:
  terminal-window hint, identity, stack fields, mask build, terminal value,
  mask apply, action stack
new stats: totals plus matching per-call count/sum/min/max/p50/p95 fields
projection: source profile, speed-row summary/compact payload, local and remote
  Modal reports, comparator timing fields
validation: ruff passed; focused source-profile, speed-row smoke, Modal guard,
  and comparator tests passed
H100 launches: r1 and r1b failed before row creation with Modal RESOURCE_EXHAUSTED;
  r1c, r2, and r3 completed as exact repeats
launcher follow-up: future pre-FunctionCall Modal launch failures write
  structured `launch.json` plus `compact_coach_speed_row_modal_report.json`
  with `failure_stage=launch` and `modal_launch_resource_exhausted`; reachable
  remote spawn failures print structured `spawn_failed` payloads; local/remote
  per-call prefix parity is now tested
latest evidence: OPT-132AD r1/r2/r3 exact `1084/270` H100 read below
latest evidence: OPT-132AE r1/r2 exact `1084/270` H100 read below
historical next evidence: if patching, target the unroll builder path only
  under exact learner-batch identity/proof gates; residual is not the main
  absolute cost. Superseded by OPT-132AZ and the tensor-native replay toy.
```

Latest runtime phase/residual diagnostic patch:

```text
label: OPT-132AD
purpose: split OPT-132AC's r1c slow branch across broad runtime phases and
  sample/builder residual buckets
new runtime phase distributions:
  actor_env_runtime, actor_autoreset, sample_gate_residual,
  sample_gate_cuda_sync, sample_gate_builder_group_loop,
  sample_gate_builder_cuda_sync
projection: source profile, speed-row summary/compact payload, local and remote
  Modal reports, comparator timing fields
validation:
  ruff passed for touched source/profile, speed-row, Modal, comparator, tests
  focused source-profile/smoke/comparator pytest: 3 passed, 2 warnings
  broader compact speed-row slice: 16 passed, 2 warnings
claim: diagnostic only, no speed claim
H100 evidence:
  r1 fc-01KT70ANVD3N80TX0JXNN88RJP, wall/speed 204.520s / 5427.42
  r2 fc-01KT72GZ503M27W7RPXRJDJ07H, wall/speed 231.874s / 4787.14
  r3 fc-01KT73847VC31GWYJ4KXBTZFVK, wall/speed 182.574s / 6079.81
comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ad-h100-runtime-phase-residual-diagnostic-r1-r2-r3-comparison-20260603/comparison.json
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
spread:
  wall 24.11%
  sample gate 30.48%
  learner-batch build 34.32%
  sample-gate builder group-loop 34.49%
  unroll fields 36.13%
claim: diagnostic only, no speed claim
next evidence: superseded by OPT-132AE's formal accounted/residual split
```

Prior builder group-loop accounted/residual diagnostic:

```text
label: OPT-132AE
purpose: formalize builder group-loop named child work versus residual before
  any speed patch
new fields:
  compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec
  compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec
plus matching per-call stats and flat report/comparator projections
accounted formula: terminal_metadata + unroll_terminal_window_hint +
  unroll_fields + write_output
validation:
  ruff passed for touched source/profile, speed-row, Modal, comparator, tests
  focused pytest: 4 passed, 2 warnings
  broader speed-row/report slice: 63 passed, 2 warnings
claim: diagnostic only, no speed claim
H100 evidence:
  r1 fc-01KT74133RFR458NBFCFWX0C3Q, wall/speed 271.920s / 4082.14
  r2 fc-01KT74CKC3DAHGG3J2DVD94QGC, wall/speed 284.561s / 3900.80
comparison: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ae-h100-builder-loop-accounted-residual-diagnostic-r1-r2-comparison-20260603/comparison.json
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
spread:
  wall 4.54%
  sample gate 4.51%
  learner-batch build 5.14%
  sample-gate builder group-loop 5.03%
builder group-loop split:
  group-loop 113.971s -> 119.853s
  accounted child work 95.872s -> 101.300s
  residual 18.099s -> 18.553s
named child totals:
  unroll fields 57.127s -> 61.190s
  terminal metadata 26.666s -> 27.764s
  terminal-window hint 5.174s -> 5.415s
  write output 6.905s -> 6.931s
next evidence: superseded locally by OPT-132AF guarded unroll-2 specialization;
  measure only when requested/eligible/used/call/fallback/reason/impl/path
  proof fields pass
```

Latest local guarded unroll-2 specialization:

```text
label: OPT-132AF
purpose: reduce the largest AE named child without changing learner-batch
  tensors or the accepted denominator
scope: default-off, fused resident grouped learner-batch builder only
guards:
  learner unroll steps == 2
  unroll chain length == 3
  terminal-window row-count hint present
  all rows are device replay rows
proof fields:
  compact_muzero_learner_batch_unroll2_specialized_builder
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl
  compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path
validation:
  ruff passed for source/profile, speed-row, Modal, comparator, tests
  focused source-profile parity test passed
  focused speed-row proof tests passed
  broader compact speed-row slice passed: 98 passed
local smoke:
  opt132af-local-unroll2-specialized-builder-hardened-smoke-20260603
  ok=true
  requested=true, used=true
  eligible_count=13, call_count=13
  fallback_count=0, fallback_reason=none
  impl=unroll2_specialized_v1
  path=unroll2_specialized
  training_wall_sec=1.1563019589957548
  steps_per_sec=7084.654606236878
claim: local proof only, superseded by H100 r1/r2/r3 diagnostic
```

Latest H100 guarded unroll-2 specialization read:

```text
label: OPT-132AF r1/r2/r3
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
unroll2 specialized builder violations: []
proof: requested=true, used=true, eligible_count=399, call_count=399,
  fallback_count=0, fallback_reason=none, impl=unroll2_specialized_v1,
  path=unroll2_specialized
wall/speed:
  r1 211.682755496s / 5243.771498528963
  r2 225.02032744299999s / 4932.958780273657
  r3 234.04234284400002s / 4742.799898990401
spread:
  wall 22.35958734800002s / 9.93669665406737%
  sample gate 128.43333603900004s -> 139.11173540300007s -> 146.50472299799998s
  learner-batch build 78.60774218799986s -> 85.98325656999997s -> 90.698261082s
  builder group-loop 77.85183318599981s -> 85.16751857000008s -> 89.80348819599989s
claim: diagnostic only, stable_speed_claim_allowed=false; AF rejected as a
  stable speed claim
historical note: the old next step was to explain the monotonic same-work
  slowdown before another learner-batch builder speed patch. Superseded by
  OPT-132AZ and the architecture reset.
```

Latest deep builder-child diagnostic read:

```text
rows: OPT-132AC r1c/r2/r3
identity: exact
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
stable speed claim allowed: false
wall/speed:
  r1c 235.855s / 4706.36
  r2 174.009s / 6379.07
  r3 172.247s / 6444.33
spread:
  wall 63.608s, 36.55% of median
  sample gate 40.718s, 37.32%
  learner-batch build 23.229s, 34.23%
  builder group-loop 22.931s, 34.11%
  unroll fields 13.686s, 40.16%
  terminal metadata 4.433s, 27.48%
deep-child sums:
  unroll fields 45.615s / 31.928s / 34.076s
  terminal metadata 18.970s / 14.537s / 16.130s
  terminal final observation 11.121s / 8.376s / 9.902s
  terminal tensor fallback 4.541s / 3.501s / 3.811s
  unroll mask build 13.321s / 9.635s / 10.302s
  unroll terminal value 11.582s / 7.427s / 8.237s
  unroll stack fields 10.919s / 7.603s / 8.013s
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132ac-h100-builder-deep-child-diagnostic-r1c-r2-r3-comparison-20260603/comparison.json
decision: exact identity held, but timing stability failed. R2/R3 cluster while
r1c is the slow outlier. Explain that branch before instrumenting further or
targeting the now-visible deep-child builders.
```

Latest sample-gate per-call diagnostic read:

```text
rows: OPT-132AA r1/r2
identity: exact against each other and OPT-132Z
accepted-fast-path violations: []
cuda-sync diagnostic violations: []
wall/speed: 213.543s / 5198.09, 224.408s / 4946.42
AA wall spread: 10.865s, 4.96% of median
AA sample-gate spread: 6.024s, 4.60% of median
AA learner-batch-build spread: 2.246s, 2.88% of median
sample-gate per-call p50/p95/max:
  r1 1.032s / 1.630s / 1.723s
  r2 1.061s / 1.674s / 1.783s
learner-batch-build per-call p50/p95/max:
  r1 0.652s / 0.804s / 1.094s
  r2 0.666s / 0.802s / 1.100s
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132aa-h100-sample-gate-per-call-diagnostic-r1r2-comparison-20260603/comparison.json
combined artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132aa-h100-sample-gate-per-call-diagnostic-zz-aa-comparison-20260603/comparison.json
decision: sample-gate cost is broad across calls and centered on builder work,
not explained by a few pathological calls. Diagnostic only, no speed claim.
```

First diagnostic read:

```text
learner-batch build: 46.919s to 57.573s across 135 calls
per-call p50/p95/max: about 0.385-0.500s / 0.475-0.606s / 0.733-0.879s
builder group loop: 46.358s to 56.837s
unroll fields: 26.821s to 32.877s
terminal metadata: 10.825s to 13.292s
metadata sync/readback: 0.035s
decision: sub-timers are visible, but host timing is still unstable
```

OPT-132-I exposed a guard bug, not a new performance blocker:
checksums can be negative. Required checksums should be nonzero; counters should
be positive. That has been fixed locally and the OPT-132-I artifact validates
cleanly under the corrected rule.

OPT-132-J then passed the corrected guard and still ran slow. That means the
next problem is same-work runtime instability, not proof-field drift.

Current resident stack read:

```text
G resident stack update: 0.029s
G resident stack shift:  0.012s
H resident stack update: 0.034s
H resident stack shift:  0.015s
```

The resident stack is not the big target. Park the ring-buffer stack.

Next steps:

```text
1. Do not claim speedup.
2. Do not rerun tiny timing rows.
3. Keep proof fields threaded through every diagnostic: seeds, work shape,
   actor trajectory, terminal/autoreset/death, sample order, replay/index rows,
   resident replay snapshot mode/bytes, sample/learner counters, and the new
   cuda_sync timing/count fields plus runtime-step envelope fields.
4. Treat OPT-132Y r3 as a broad same-work runtime slowdown, not a learner-batch
   builder-only signal.
5. Treat OPT-132Z r1/r2 as proof that the slowdown is inside measured loop
   iterations and concentrated on sample-gate cadence steps.
6. Treat OPT-132AA r1/r2 as proof that sample-gate cost is broad per call and
   learner-batch builder work is the stable heavy center.
7. Treat OPT-132AB r1/r2 as proof that the broad builder swing is inside the
   group loop, with unroll fields and terminal metadata as the largest visible
   children.
8. Treat OPT-132AC r1c/r2/r3 as proof that the deeper child timers project on
   H100 and the exact same-work AC field set still fails stability.
9. Treat OPT-132AD r1/r2/r3 as proof the instability remains centered on
   measured sample-gate builder group-loop work, not proof-field drift.
10. Treat OPT-132AF r1/r2/r3 as the rejected speed-patch proof packet:
   proof passed, but stable speed claim is rejected.
11. Treat OPT-132AG r4/r5/r6 as the prior exact AF-path read: same AF path,
   GPU utilization sampling enabled, exact identity, clean proof, wall
   `195.624s / 185.643s / 199.275s`, and stability still failed. The monotonic
   AF slowdown did not repeat, and the slowest AG row had lower mean GPU util
   and lower max power than the faster AG rows.
12. Treat OPT-132AH r1/r2/r3 as the prior exact generic H100 read: generic builder,
   GPU utilization sampling enabled, exact identity, wall
   `205.584s / 243.805s / 289.172s`, sample gate
   `123.444s / 151.007s / 196.761s`, and builder group-loop
   `74.642s / 91.378s / 122.843s`. Stability failed harder than AG, and mean
   GPU utilization decreased as rows slowed.
13. OPT-132AZ has bounded the remaining same-work runtime swing enough to stop
   attribution: broad user-CPU learner-batch builder work under exact identity.
   This was superseded by the architecture reset, fixed-action/slab local
   proofs, owner-search action-only proof, real-learner scale failure, and
   mock-fast ceiling pass. The useful next target is real learner
   placement/overlap.
14. Treat OPT-132AJ r1/r2/r3 as the prior exact runtime-step-only generic read:
   generic rows, CUDA-sync probes off, exact identity, wall spread `10.49%`,
   sample gate `12.72%`, learner-batch build `14.09%`, and builder group-loop
   `14.25%`. Runtime-step sum reconciles with wall. Removing CUDA-sync probes
   narrows but does not solve the instability.
15. Treat OPT-132AK as local diagnostic tooling: comparator wall-swing
   attribution plus runtime-step active/inactive and early/mid/late cadence
   summaries. The refreshed AJ comparison says r2's slow-fast wall delta is
   mostly measured runtime, sample gate, and builder group-loop.
16. Treat OPT-132AM as prior local diagnostic tooling: chronological active
   sample-gate distributions by measured third, sample-gate residual bucket
   sums, enriched top-slowest runtime-step records, and slowest per-call
   iteration projection. Tooling only; no speed claim.
17. Treat OPT-132AL r1/r2/r3 as prior no-sampler H100 evidence: exact
   identity held, wall/runtime-step spread `21.70%`, and late sample-gate
   cadence carried much of the swing. Disabling the sampler did not solve
   stability.
18. Treat OPT-132AN r1/r2 as prior exact H100 evidence: no GPU sampler,
   CUDA-sync probes off, exact identity, wall/runtime-step spread `21.806%`,
   sample gate `125.169s -> 93.384s`, learner-batch build
   `68.851s -> 49.643s`, and builder group-loop `68.101s -> 48.923s`. Active
   sample-gate p50/p95 rises early/mid/late inside each row and r1 is elevated
   across all thirds versus r2.
19. Treat OPT-132AO as prior exact H100 read:
   bounded per-call replay-state trace records projected through source
   profile, speed-row summary/compact payload, local and remote Modal reports.
   AO r1/r2 kept exact identity and violations `[]`, but wall/runtime-step
   spread `13.944%`; sample gate moved `132.338s -> 163.614s`; all exposed
   trace state matched at all `135` call indices.
20. Treat OPT-132AP as the prior allocator-exact H100 read: AP extends each
   sample-gate call trace with CUDA allocator counters, Python GC/max-RSS
   runtime state, learner-batch-build memory deltas, sample-gate CUDA-sync
   timing, and deeper builder child fields. AP r1/r2/r3 kept exact identity
   and violations `[]`, but wall spread was `31.99%` of median
   (`219.446s / 205.371s / 153.752s`). Exposed replay/sample trace identity
   and every CUDA allocator/memory counter matched exactly across all `135`
   calls; Python GC counts and RSS varied.
21. Treat OPT-132AQ as a prior completed exact H100 read: AQ adds actual
   Python GC collection/collected/uncollectable counters. R1/R2 kept exact
   identity and violations `[]`, but wall spread was `28.856%`; GC collection
   totals were nearly identical and `100` same-GC-delta calls still carried
   `22.6695s` of sample-gate delta. GC collections do not explain the swing.
22. Treat OPT-132AR as prior exact H100 read: AR adds
   process/thread CPU-time deltas. R1/R2 kept exact identity and violations
   `[]`, but wall spread was `13.479%`; about `92%` of both sample-gate and
   learner-build wall deltas were backed by process/thread CPU time.
23. Treat OPT-132AS as a prior completed exact H100 read: AS adds
   process/thread `getrusage` user/system CPU, page-fault, and context-switch
   deltas. R1/R2 kept exact identity and violations `[]`, but wall spread was
   `14.616%`; the sample-gate delta was mostly user CPU and not system CPU,
   faults, or context switches.
24. Superseded by OPT-132BF/BI/BJ, the architecture reset, and the later
   owner-search ring-batch rows: do not add another pure attribution diagnostic
   or replay-hardening lane. Maintained tensor-native replay/sample-universe is
   closed as proof; OPT-132BK closes local slab/replay/sample handoff; active
   work is the next owner-search materialization cut after ring-batched replay
   append/cache refresh.
```

Latest resource-usage diagnostic patch:

```text
label: OPT-132AS
purpose: split OPT-132AR's CPU-time-backed sample-gate/build swing into
  user CPU, system CPU, page faults, and context switches
new fields: per-sample-gate and learner-batch-build process/thread
  getrusage before/after/delta counters
fields include: user_cpu_time_ns, system_cpu_time_ns, minor/major page faults,
  voluntary/involuntary context switches
validation: ruff passed; focused source-profile, speed-row smoke, Modal guard,
  and comparator slice passed: 17 tests, 2 warnings
claim: diagnostic only, no speed claim
H100 evidence: AS r1/r2 exact generic no-sampler `1084/270` rows read the
  fields and still failed timing stability
historical note: explain the mostly-user-CPU sample-gate/build delta
  before any speed patch. Superseded by OPT-132AZ and the architecture reset.
```

Latest CPU perf-stat diagnostic patch:

```text
label: OPT-132AT
purpose: split AS's mostly-user-CPU swing into lower-level CPU evidence:
  retired instructions, cycles/ref-cycles, IPC, cache/LLC/TLB/branch behavior,
  page faults, context switches, CPU migrations, or effective CPU frequency
new flag: --compact-profile-cpu-perf-stat-diagnostics
remote behavior: wraps the speed-row producer in perf stat -x, and writes
  cpu_perf_stat_stdout.txt / cpu_perf_stat_stderr.txt beside the row artifacts
projected through: remote result summary/compact payload, Modal report,
  comparison identity/timing fields
validation: ruff passed; targeted perf-stat tests passed: 4 tests, 2 warnings;
  broader smoke/compare slice passed: 20 tests, 2 warnings
claim: diagnostic only, no speed claim
historical next evidence: run/read one exact generic no-sampler 1084/270 H100
  row, or preserve the structured perf-unavailable/perf-denied failure. This
  was completed by OPT-132AU; do not reopen perf unchanged.
```

OPT-132AT H100 result:

```text
function: fc-01KT7YGKDM2Y35NCG6C204TYMY
result: failed before speed row
problem: perf stat diagnostic requested but perf was not found
compact_profile_cpu_perf_stat_available: false
compact_profile_cpu_perf_stat_returncode: 127
decision: not a speed row; the bounded perf availability retry was completed
  by OPT-132AU and should not be reopened unchanged
```

OPT-132AU immediate retry:

```text
change: speed_row_image now installs linux-perf
validation: ruff passed; focused Modal perf tests passed: 2 tests, 2 warnings
run: exact generic no-sampler 1084/270 with
  --compact-profile-runtime-step-timing-diagnostics
  --compact-profile-cpu-perf-stat-diagnostics
expected outcomes:
  counters available, or structured perf-denied/unsupported-event failure
claim: diagnostic only, no speed claim
```

OPT-132AU H100 result:

```text
function: fc-01KT7YWEDCC60CMGTFY6388QBE
result: failed before speed row
perf binary: /usr/bin/perf
problem: perf stat diagnostic failed before speed-row result
stderr: sys_perf_event_open() returned 19 (No such device) for task-clock
compact_profile_cpu_perf_stat_available: true
compact_profile_cpu_perf_stat_returncode: 255
parsed events: 0
decision: external perf counters are unavailable in this Modal container
next evidence: completed historical in-process child-phase resource/CPU
diagnostic below AS; not an active next move
```

OPT-132AV H100 result:

```text
rows: exact generic no-sampler 1084/270 r1/r2
functions: fc-01KT7ZWHYPG3342MWGDP7TFSHB / fc-01KT809AAW2TY3DAEHPR8TV1B2
identity: exact
report violations: []
wall/speed:
  r1 190.101s / 5839.10
  r2 220.991s / 5022.90
spread:
  wall 30.890s, 15.03% of median
  sample gate 20.312s
  learner-batch build 16.432s
  builder group-loop 16.287s
builder process CPU delta:
  group-loop +14.88s
  accounted +10.60s
  residual +4.28s
  terminal metadata +6.37s
  terminal final-observation +2.67s
  unroll fields +2.76s
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132av-h100-generic-builder-child-cputime-nosampler-r1-r2-comparison-20260603/comparison.json
decision: exact work still does not repeat. No speed claim. Later diagnostics
  explained enough to stop this lane; active next evidence is real learner
  placement/overlap.
```

OPT-132AW H100 final-observation/residual split:

```text
local split:
  group_loop_prepare
  group_loop_terminal_value_bookkeeping
  terminal_metadata_final_observation_presence
  terminal_metadata_final_observation_select_current
  terminal_metadata_final_observation_gather
proof counters:
  terminal_final_observation_group_count
  terminal_final_observation_index_fast_path_count
  terminal_final_observation_fallback_count
  terminal_final_observation_final_row_count_sum/max
  terminal_final_observation_dense/sparse/missing_storage_count
  terminal_final_observation_sparse_row_count_sum/max
validation:
  ruff passed
  focused source-profile/smoke/comparator pytest: 189 passed, 2 warnings
rows:
  r1 250.183s / 4436.81 env steps/sec
  r2 232.521s / 4773.82 env steps/sec
spread:
  wall 17.662s, 7.318% of median
  sample gate 157.724s / 146.441s
  learner-batch build 94.924s / 88.854s
  builder group-loop 94.002s / 87.987s
builder process CPU:
  group-loop 89.13s / 83.17s
  accounted 74.68s / 69.10s
  residual 14.45s / 14.07s
  terminal metadata 24.01s / 21.75s
  terminal final-observation 14.78s / 13.35s
  final gather 6.84s / 5.95s
  unroll fields 38.18s / 34.98s
proof:
  groups 399 / 399
  index-fast-path 0 / 0
  fallback 399 / 399
  final-row sum/max 512/4 in both rows
  sparse storage 399 / 399
  sparse-row sum/max 3102/16 in both rows
artifact:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132aw-h100-generic-finalobs-residual-split-nosampler-r1-r2-comparison-20260604/comparison.json
decision:
  exact work still does not repeat. No speed claim. Historical next evidence
  was final-observation gather work; superseded by AX/AZ and the architecture
  reset. Learner-ready replay remains one candidate lane, not the live plan by
  itself.
```

OPT-132AX H100 validate-only cleanup:

```text
local change:
  grouped learner terminal-final-observation fallback validates resident
  final-observation coverage instead of materializing a discarded
  final-next-observation tensor.
unchanged:
  sample paths still materialize final observations because they write
  next_observation.
new proof:
  terminal_final_observation_validate_only_count
  terminal_final_observation_materialized_count
new timing:
  terminal_metadata_final_observation_validate wall/per-call/process/thread CPU
comparator:
  terminal-final-observation proof counters are optional identity fields.
validation:
  ruff passed
  focused source-profile/smoke/comparator pytest: 190 passed, 2 warnings
rows:
  r1 168.280s / 6596.25 env steps/sec
  r2 262.830s / 4223.32 env steps/sec
spread:
  wall 94.550s, 43.864% of median
  sample gate 98.894s / 173.517s
  learner-batch build 56.376s / 102.112s
  builder group-loop 55.699s / 101.183s
proof:
  groups/fallback/validate-only 399/399/399 in both rows
  materialized 0 / 0
  final-row sum/max 512/4 in both rows
  sparse storage 399 / 399
  sparse-row sum/max 3102/16 in both rows
timing:
  select-current wall/CPU 0 / 0 in both rows
  gather wall/CPU 0 / 0 in both rows
  validate wall 4.312s / 7.057s
  validate process CPU 3.33s / 5.76s
  unroll-fields process CPU 22.70s / 42.99s
artifact:
  artifacts/local/curvytron_compact_coach_speed_row_results/opt132ax-h100-generic-finalobs-validate-only-nosampler-r1-r2-comparison-20260604/comparison.json
decision:
  cleanup worked but stability failed. No speed claim. Historical next target
  was unroll-fields CPU and broader builder group-loop CPU; superseded by
  OPT-132AZ and the architecture reset.
```

Latest resource-usage H100 read before AV:

```text
rows: OPT-132AS r1/r2
identity: exact
accepted-fast-path violations: []
runtime-step diagnostic violations: []
cuda-sync diagnostic violations: []
CUDA-sync diagnostics: false
GPU sampling: false
stable speed claim allowed: false
wall/speed:
  r1 211.257s / 5254.34
  r2 182.483s / 6082.84
spread:
  wall 28.774s, 14.616% of median
  sample gate 18.999s, 16.179% of median
  learner-batch build 11.444s, 17.701% of median
  builder group-loop 11.396s
resource split:
  sample-gate process user/system delta 16.17s / 0.49s
  sample-gate thread user/system delta 15.83s / 0.58s
  learner-build process user/system delta 9.31s / 0.55s
  page-fault and context-switch deltas 0
artifact: artifacts/local/curvytron_compact_coach_speed_row_results/opt132as-h100-generic-rusage-trace-nosampler-r1-r2-comparison-20260603/comparison.json
decision: exact work still does not repeat. The remaining swing is mostly user
  CPU, not system CPU, page faults, or context switches. No speed claim.
```

The G/H/I/J comparison now has a machine-built artifact:

`artifacts/local/curvytron_compact_coach_speed_row_results/opt132-ghij-runtime-comparison-20260602/comparison.json`

Verdict: `stable_speed_claim_allowed=false`. The exact clean comparison is
I -> J. Largest regressions are sample gate `+1.852s`, actor `+1.397s`,
actor autoreset `+1.104s`, learner-batch build `+1.038s`, and learner
`+0.654s`.

## Still Parked

- Owner process/inline/threaded boundary variants beyond the measured
  owner-search action-only threaded/background falsifier.
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
- Result-payload-format tweaks outside bounded diagnostics.

## Immediate Gate After Owner Deferred Replay Proof

Status on 2026-06-07:

- The compact Torch service-level deferral proof exists.
- The owner boundary now records and fails closed on deferred replay flush
  proof.
- The speed-row direct transition-batch replay proof now rejects requested
  deferral rows unless device flush count, deferred flush count,
  materialized-on-flush count, identity-match count, recurrent-call count,
  final pending count, refresh-crossed count, and replay D2H bytes are all
  consistent.
- The local/Modal launch flag is wired.
- Local flagged speed-row smoke
  `opt132-local-owner-deferred-one-sim-direct-transition-batch-smoke-20260607-r7`
  passed with report-level proof: direct transition count `12`, direct
  flush/materialized/identity/recurrent counts `12/12/12/12`,
  model-refresh-crossed `0`, replay-payload D2H bytes `0.0`, and owner-inner
  pending final `0`.
- The one H100 probe
  `opt132-h100-owner-deferred-one-sim-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r1-20260607`
  passed proof but failed speed: `13691.98 env/s`, `54.1467s`, below columnar
  r2 `15852.67 env/s`. Direct append rose to `22.377s` with deferred/device
  flush `3.613s/4.381s`.

Do next:

1. Do not repeat the flagged owner-search/direct-transition-batch/compact-
   Torch-deferral path unchanged. It is proof support, not the speed lane.
2. Keep the architecture target pointed at the non-replay surfaces
   still needed for `2x`: owner-resident root/mechanics/search dispatch,
   parent wait, and learner publication/update overlap.
3. Start the next architectural patch as a local owner-boundary or toy-ceiling
   experiment, not another attribution timer or replay-only tweak.

## 2026-06-08 Guarded Overlap/Proxy Decision

The one guarded H100 overlap/proxy read is complete.

```text
run:
  opt132-h100-action-dispatch-overlap-proxyclosure-columnar-directtable-b1024a1-normal-unroll2-m724-w180-r3-20260608
status:
  ok=true, proof clean, speed-rejected unchanged
speed/wall:
  15541.95 env/s / 47.7016s over 741376 env steps
comparison:
  1.225x OPT-104 baseline
  0.980x current best columnar/direct-table support row
proof:
  overlap submit/resolve 904/904
  sync-wrapper count 0
  completed-at-submit 0
  wait-in-submit 0
  max pending 1
  proxy closure closed/transition 724/724
  proxy closure batch/transport 181/181
  proxy fallback/pending 0/0
  proxy digest verified true
decision:
  preserve overlap/proxy as proof support
  do not repeat unchanged
  next implementation is owner-resident step-frame/root/search handles or
    learner publication tickets, with the step-frame boundary first
```

## 2026-06-08 Next Implementation Order

The next implementation is not another long run, not root-build-request
again, and not replay-only polish. Build the first local owner-resident
step-frame boundary.

First local slice:

```text
entry point:
  HybridBatchedObservationProfileManager.step()

boundary:
  immediately after mechanics/resident observation update
  before _make_compact_batch

new handoff shape:
  fixed/view/slot reference carrying compact scalar buffers,
  resident observation handle, joint action, batch/player metadata,
  terminal/final sidecars, generation, and digest/checksum proof

owner side:
  build root/search request from the step-frame view
  assemble dense next action on owner side
  stage owner-local transition facts from the same frame refs

parent side:
  keep only dense next action plus telemetry/proof
  keep parent wait measured as the remaining synchronous residual
```

Required local proof before H100:

```text
compact_owner_mechanics_boundary_used true
parent compact-batch builder/object count 0
parent root builder calls 0
host observation/outcome bytes sent 0
owner root build/action assembly counts positive
action feedback mismatch 0
terminal/final sidecar checksums preserved
owner-local transition fallback/pending/drop 0
parent replay rows 0/0
normal-death and tensor-native gates preserved
```

Support patch landed:

```text
compact_owner_minimal_step_payload_snapshot:
  status: closed-local / support only
  guard: requires compact_owner_action_step_boundary
  proof: counts full payload bytes/key count elided and retained keys
  validation: focused source-state tests 4 passed; ruff passed
```

Do not run H100 for the minimal payload flag alone. It is only a small guard
that makes one parent snapshot copy measurable while the real step-frame
boundary is being built.

First step-frame handle rung landed:

```text
compact_owner_mechanics_step_frame_handle:
  status: closed-local / support only
  schema: curvyzero_compact_owner_mechanics_step_frame_handle/v1
  ring slots: 4
  proof:
    manager publishes slot/generation/digest for each owner mechanics frame
    direct owner stepper verifies the digest before root request submission
    telemetry reports published/consumed true, publish/consume counts,
      slot id, generation, digest, digest_verified, owner_digest_verified,
      and resident observation handle presence
    stale metadata digest fails closed locally
  surfaces:
    source profile result fields
    speed-row proof projection
    local/Modal report allowlists
    compact speed-row evidence summary
  validation so far:
    ruff passed across touched code/scripts/tests
    focused source-state handle packet 2 passed
    source-state owner-boundary packet 7 passed
    owner-search direct-root/proxy packet 11 passed
    speed-row focused smoke 5 passed
    full source-state suite 123 passed
    full owner-search service suite 56 passed
    full speed-row smoke suite 99 passed
```

Slot-backed mechanics frame/root-request rung landed after the handle identity
slice:

```text
compact_owner_mechanics_step_frame_slot:
  status: closed-local / support only
  schema: curvyzero_compact_owner_mechanics_step_frame_slot/v1
  proof:
    manager writes fixed ring slot arrays for action_mask/reward/done/
      policy rows/final/terminal/autoreset/joint_action/mechanics sidecars
    direct owner root request builder reads those slot arrays
    legacy CompactOwnerMechanicsStepViewV1 builder can be monkeypatched to
      raise
    parent _make_compact_batch can be monkeypatched to raise
    compact_root_build_request_v1_from_batch can be monkeypatched to raise
    parent dense-action reconstruction can be monkeypatched to raise
    root request from-batch helper used false
    root sidecar array bytes/field count 0/0
    step-view object count 0
    parent step-frame build count 0
  projection:
    source profile result, speed-row local builder, Modal report bundles, and
      compact speed-row evidence summary all carry the new slot/root fields
    direct-root proof guard fails closed on missing/dirty mechanics-slot fields
  validation:
    ruff touched files passed
    source-state owner-boundary packet 5 passed
    owner-search direct-root/proxy packet 11 passed
    speed-row owner/direct-root/projection packet 7 passed
```

Do next:

1. Do not launch H100 for the handle rung alone. It is identity/proof support.
2. Do not launch H100 for the slot-backed mechanics frame alone. It is a
   necessary local ownership rung, not a proven wall-time win.
3. Next code target: either make the owner consume the slot frame through a
   longer-lived root/search transaction so parent wait and search dispatch move
   materially, or implement learner-publication tickets/refs if that patch is
   disjoint and can remove parent update/refresh cadence work.
4. The stale-generation/reuse guard is now closed-local support. The current
   patch validates slot metadata, generation modulo slot, live slot generation,
   handle shape, digest, and per-stepper consumed generations before replay
   commit/proxy closure/search submit. The strengthened regression keeps a
   pending sentinel in place and proves stale slot reuse fails without changing
   pending state, record index, replay append/proxy/submit counters, or pending
   dispatch. Validation: ruff passed; focused owner-frame/direct-root packet
   `3 passed`; source-state owner-boundary packet `6 passed`; owner-search
   direct-root/proxy/overlap packet `11 passed`.
5. Next code target is now broader than the stale guard: either make the owner
   consume the slot frame through a longer-lived root/search transaction that
   moves parent wait/search dispatch materially, or implement learner-publication
   tickets/refs if that patch is disjoint and removes parent update/refresh
   cadence. Do not launch H100 until one of those owner-graph surfaces changes.
6. The root-action-context prerequisite is landed locally: async direct dispatch
   pending state keeps `CompactRootActionContextV1` instead of the full
   `CompactRootBuildRequestV1`, and tests assert both the direct stepper and
   proxy pending dispatches have no `root_build_request` slot.
7. The first owner/proxy root-search transaction rung is now landed locally:
   ring-backed step frames call
   `submit_owner_root_search_transaction_from_step_frame_slot()`, the proxy
   builds the root request from the slot, and tests poison the slab
   `_root_build_request_from_owner_step_frame_slot_v1()` helper while the
   boundary still passes. The new transaction path also avoids storing
   `compact_batch` in the direct pending dispatch. This is still support, not
   speed evidence, because parent still stores `CompactRootActionContextV1` for
   action validation and still coordinates submit/resolve.
8. Next code target: either make the action/root validation context
   owner-resident behind a smaller action-result handle, or implement the
   disjoint learner-publication/update ticket/ref patch. Do not launch H100
   until one of those owner-graph surfaces changes and the local proof remains
   fail-closed.

9. The action/root validation context handle rung is now closed-local support.
   The slot-started transaction path requires
   `CompactOwnerRootActionContextHandleV1`, validates handle id/root counts/
   digest before finishing, stores no parent `CompactRootActionContextV1` in
   pending transaction state, and speed-row/Modal guards require owner
   store/resolve/release, digest verification, parent-validation `0`, owner
   validation positive, and transaction parent context store/bytes/fields `0`.
   Validation: ruff passed; source transaction/profile packet `3 passed`;
   source owner-boundary packet `5 passed`; owner-search focused packet
   `2 passed`; full owner-search service `56 passed`; speed-row focused
   owner-search/report packet `3 passed`.
10. Do not launch H100 for the root-action-context handle rung. Sidecars
    agreed it is ownership/proof cleanup, not enough whole-loop wall movement.
    Next implementation priority is broader: owner replay/sample/learner-batch
    fixed ring first, coarse owner actor/search transaction tranche second, or
    learner publication/version tickets if that disjoint patch is smaller and
    removes measured parent update/refresh transport. The local H100 gate is:
    one of those patches must move a measured surface beyond this support rung
    while preserving action/death/replay proof and zero parent hot-loop
    transport.
11. The maintained tensor-native replay table-handle rung is now local-clean
    and default-off. `compact_muzero_learner_batch_tensor_native_replay_maintained_table_handle`
    makes `_CompactReplayRingV1.snapshot_for_sample()` publish a fixed
    unroll-2 table handle with aligned record ids, offsets, learner-ready
    targets, and maintained table entries. The maintained tensor-native gather
    can consume that handle and reports requested/used/record/missing/row
    counters through source telemetry, speed-row summaries, Modal bundles, and
    smoke guards. Validation: ruff passed for touched source/reporting/tests;
    focused handle/projection packet `3 passed`; full source-state profile
    `127 passed`; full compact speed-row smoke `100 passed`; tensor-native
    benchmark tests `4 passed`. This is support, not H100 evidence: it uses the
    existing maintained gather and deliberately avoids the selected-maintained
    per-record gather path. Next replay/sample implementation should be an
    owner-issued fixed-SoA or learner-batch handle ring with explicit lifetime/
    watermark proof, or a broader owner transaction/learner-publication slice.
