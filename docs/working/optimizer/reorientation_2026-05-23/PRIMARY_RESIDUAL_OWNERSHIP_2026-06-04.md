# OPT-132BF Primary Residual Ownership Map

Date: 2026-06-05

This file records the ownership map that led to the r14 speed win. It is now
historical context plus proof guardrails; the active next move lives in
`TASK_BOARD.md` and is r14 repeat/promotion audit.

## Decision

Select the owner-search deferred-maintenance candidate. As of r14, it is a
first single-row H100 speed path, but not yet a stable promoted baseline.

The existing owner-search `run_action_step` plus `owner_defer_maintenance` path
has now been promoted far enough to prove the intended ownership shape locally.
The parent publishes root references, the owner returns selected actions plus a
replay handle, and the owner materializes replay rows during maintenance. The
parent must not rebuild the full search result or committed replay rows in this
mode.

Why this is the right cut:

```text
It removes a whole hot-loop handoff:
parent search-result payload -> parent replay-row construction -> parent
CompactOwnedLoopV1.record_step(previous_step, current_step, index_rows).
```

This is closer to AlphaZero/MuZero, PufferLib/EnvPool/Sample Factory, and
Isaac-style ownership patterns than another replay/sample micro-optimization:
the service that owns the search state also owns the replay materialization
needed by that search.

## Evidence

OPT-132BF H100 remaining surfaces:

```text
primary residual: 122.679s
actor step wall: 41.852s
actor/autoreset: 29.656s
compact rollout slab: 19.748s
search dispatch: 14.248s
observation: 11.423s
env runtime: 10.000s
sample gate after fix: 4.116s
```

The sample gate was a real problem, but it is no longer the active P0 surface.
OPT-132BD proved maintained tensor-native replay and OPT-132BF proved the
maintained sample universe. Full-loop speed still failed at `0.42x` OPT-104.

Latest owner-search scale evidence:

```text
opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-20260604
  proof: parent rows 0, payload bytes 0, inner two-phase true,
    device replay deferred true, staged/drained work 55/55,
    drained replay entries/appends 55/55
  speed: 25.78 env steps/sec local
  train/update/final drain: 14.23s / 14.07s / 11.97s

opt132-local-owner-action-only-inner2-threaded-scale48-cadence8-b512-normal-20260604
  proof: same zero-parent-row / zero-payload / drain closure
  speed: 25.09 env steps/sec local
  train/update/final drain: 14.94s / 14.77s / 13.00s
```

Mock-fast ceiling evidence:

```text
opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-mockfast-r3-20260604
  proof: normal-death contract gate true, owner learner telemetry carries
    value-valid/done/truncated counters, parent rows 0, payload bytes 0,
    inner two-phase true, device replay deferred true, staged/drained work
    55/55, drained replay entries/appends 55/55
  speed: 214.06 env steps/sec local
  train/update/final drain: 0.091s / 0.000s / 0.0026s
```

Pre-r14 conclusion: proof was meaningful, real-learner speed shape failed
locally, and mock-fast ceiling was fast. Owner-search/search/replay mechanics
were not the local scale blocker once neural update was removed. Explicit local
MPS placement preserved proof but slowed the whole row to `20.81 env
steps/sec`, and eager append pre-drain preserved proof but slowed the
normal-death scale row to `24.09 env steps/sec` with learner update `15.12s`
and final drain `12.67s`. Placement alone and maintenance scheduling are both
falsified locally.

Post-r14 refinement: the winning H100 mechanism was shared inline model refresh
plus no inline host payload clone, not another learner-resource split. r14 ran
`13497.30 env steps/sec`, `13.6561s` wall, with train/update closure `22/22`,
refresh request/skip/update `22/0/22`, zero parent replay rows, zero search
payload bytes, and final owner drain `0.000042s`. Next gate is repeat and
cadence/proof audit.

## Primary Residual Definition

`scripts/build_compact_coach_speed_row_smoke.py` computes the speed-row primary
residual as:

```text
training wall - (
  actor_step_wall_sec
  + observation_sec
  + compact_rollout_slab_sec
  + compact_rollout_slab_sample_gate_sec
  + compact_rollout_slab_learner_gate_sec
  + compact_rollout_slab_policy_refresh_after_learner_gate_sec
)
```

So the primary residual is not one kernel. It is the parent-loop remainder:
manager/profile work, compact-batch construction, checksums, reporting,
deferred bookkeeping, Python object construction, and any hot-loop handoff not
covered by the named actor/observation/slab/sample/learner/refresh buckets.

## Current Handoffs

Main speed projection:

- `scripts/build_compact_coach_speed_row_smoke.py::_speed_timing_projection_fields`
  owns the speed-row primary residual formula.

Profile manager:

- The outer measured loop still selects or copies `actions` before
  `manager.step(actions)`, then applies `slab_step.next_joint_action` back into
  the next loop action buffer. This action assembly/copy path is part of the
  ownership surface, not a separate strategy.
- `HybridBatchedObservationProfileManager.step` owns the per-iteration actor,
  observation, compact batch, slab, and proof/reporting surface.
- `InProcessHybridCurvyTronActor.step` returns rich payload objects.
- `InProcessHybridCurvyTronActor.step_into` already writes actor results into
  parent-owned compact buffers. This is useful but does not remove the later
  slab/search/replay handoff by itself.
- `_make_compact_batch` copies many scalar arrays into `HybridCompactBatch`.
  `copy_observation=False` avoids one observation copy, but scalar/replay/search
  rows still become a rich object boundary.

Slab/search/replay:

- `CompactRolloutSlab.step` calls `run_action_step` when the search service
  supports it.
- Shared owner-search root publishing still copies root sidecar arrays into the
  owner boundary. The long-term shape is fixed owner/actor buffers for root
  sidecars, not fresh rich dataclasses per step.
- `CompactRolloutSlab._commit_previous` already has the decisive branch: if the
  action step metadata says `compact_owner_search_owner_materializes_replay`,
  it stages a transition-only owner replay entry and returns
  `committed_index_rows=None`.
- The current profile loop still feeds parent committed rows into
  `CompactOwnedLoopV1.record_step` when they exist. In the selected mode, those
  parent rows should intentionally not exist; owner-search maintenance becomes
  the replay/sample/learner proof source.
- `CompactOwnerSearchServiceV1._materialize_owner_transition_entries` rebuilds
  replay/index rows from cached root/search state inside the owner, including
  the resident/device two-phase replay path.

Compact-owned loop:

- `CompactOwnedLoopV1.record_step` still appends replay from
  `previous_step/current_step/index_rows` and queues append entries from those
  same parent objects. That path is a rejected next candidate unless owner-search
  proof fails, because it preserves the handoff being removed.

## Selected Candidate

Promote owner-search action-only replay ownership.

Required behavior:

```text
search service path: run_action_step
owner mode: owner_defer_maintenance=true
parent result: selected actions plus replay handle
parent search payload bytes: 0
parent committed_index_rows: None
parent reconstructed search result: false
owner replay append: true
owner learner train/update: true when sample/train cadence fires
owner maintenance drain: counted and fail-closed
```

Proof fields that must be true or nonzero before a speed row:

```text
compact_owner_search_action_only_result=true
compact_owner_search_owner_materializes_replay=true
compact_owner_search_parent_slab_commits_replay=false
compact_owner_search_parent_reconstructed_search_result=false
compact_owner_search_search_result_payload_bytes=0
compact_owner_search_owner_replay_append_submitted_entry_count > 0
compact_owner_search_owner_replay_append_count > 0
compact_owner_search_owner_train_request_count > 0 when training is due
compact_owner_search_owner_learner_update_count > 0 when training is due
compact_owner_search_owner_maintenance_drained_count > 0
compact_owner_search_owner_maintenance_staged_work_item_count > 0
compact_owner_search_owner_maintenance_drained_work_item_count equals staged work items
compact_owner_search_owner_maintenance_drained_replay_append_entry_count equals submitted entries
compact_owner_search_owner_maintenance_drained_replay_append_count equals owner appended rows
compact_owner_search_owner_maintenance_pending_work_count=0 after final drain
compact_owner_search_owner_maintenance_final_drain_in_measured_sec=true
compact_owner_search_owner_maintenance_final_drain_sec finite and >= 0
compact_owner_search_owner_sample_telemetry requires next targets
compact_owner_search_owner_train_timing_aggregate_count equals train requests
compact_owner_search_action_feedback_verified=true in action-only mode
compact_owner_search_action_feedback_mismatch_count=0 in action-only mode
compact_owner_search_expected_joint_action_checksum equals applied and replay
compact_rollout_slab_committed_index_row_count=0 in action-only mode
compact_rollout_slab_stored_index_row_count=0 in action-only mode
compact_owner_search_worker_owns_replay_state=true
compact_owner_search_worker_owns_model_state=true
compact_owner_search_replay_payload_handle_present=true in action-only mode
```

BK-style invariants to preserve:

```text
search-selected actions are applied as the next joint actions
slab action-check failures are empty
replay append/order checksums are deterministic
sample-order and sample-batch checksums are deterministic when sample fires
compact/direct metadata equality remains local-proofable where the toy applies
host fallback stays zero
accepted-fast-path violations stay empty
```

## Implementation Work

1. Done: fix owner-search metadata so action-only owner-materialized replay reports
   `compact_owner_search_parent_slab_commits_replay=false`.
2. Done: add focused tests for the deferred owner-search slab proxy and
   speed-row smoke proof:
   `committed_index_rows is None`, zero search-result payload bytes, owner
   replay append, owner learner update, maintenance drain, and truthful parent
   commit metadata.
3. Done: wire profile/speed-row evidence so owner-maintenance counts close the
   loop when parent committed rows are intentionally absent. The profile path
   now proves zero parent committed/stored rows for action-only owner replay,
   final owner replay counts, and final owner drain inside measured wall time.
4. Done: run a small local/profile row through the real entrypoint. Same-work
   H100 still comes later and must satisfy the normal proof gate.
5. Done: wire compact Torch owner-search inner two-phase device replay into the
   production owner-search builders and gate it in the speed-row proof.
6. Done: run post-prewarm normal-death scale/cadence falsifiers. They preserve
   proof but fail speed shape because the real owner learner update dominates.
7. Done: finish the mock-fast owner learner ceiling row's normal-death contract.
   The ceiling is fast locally, so do not reject owner-search/search/replay
   mechanics yet.
8. Done: add submitted-update policy-lag proof for future overlap rows:
   submitted/owner/refreshed learner updates close exactly, policy lag finishes
   at zero, and actions while policy-lagged are visible.
9. Done: run eager append pre-drain. It fires bounded append drains and closes
   proof, but does not speed the whole row.
10. Done: run in-process async learner overlap. It proves actions can run while
    learner futures are pending and closes async submit/completed/pending proof,
    but still does not speed the whole row.
11. Next: implement a real learner resource/work-shape boundary so the
    normal-death scale row approaches the mock-fast ceiling while preserving
    proof.

Local proof artifact:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/opt132-local-owner-action-only-profile-proof-20260604/compact_coach_speed_row_smoke_report.json
```

Local proof facts:

```text
compact_owner_search_action_only_result=true
compact_owner_search_owner_materializes_replay=true
compact_owner_search_parent_slab_commits_replay=false
compact_owner_search_parent_reconstructed_search_result=false
compact_owner_search_search_result_payload_bytes=0
compact_owner_search_search_result_payload_transport_kind=action_only_owner_cached_replay_v1
compact_rollout_slab_committed_index_row_count=0
compact_rollout_slab_stored_index_row_count=0
compact_owner_search_replay_append_entry_count=15
compact_owner_search_owner_replay_append_submitted_entry_count=15
compact_owner_search_owner_replay_append_count=15
compact_owner_search_owner_train_request_count=3
compact_owner_search_owner_learner_update_count=3
compact_owner_search_owner_maintenance_drain_request_count=2
compact_owner_search_owner_maintenance_drained_count=15
compact_owner_search_owner_maintenance_pending_work_count=0
compact_owner_search_owner_maintenance_inflight=false
compact_owner_search_owner_maintenance_failed=false
compact_owner_search_owner_maintenance_final_drain_in_measured_sec=true
compact_trainer_env_steps_per_sec=70.93 local CPU proof only
```

Validation so far:

```text
uv run ruff check src/curvyzero/training/compact_owner_search_service.py src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/build_compact_coach_speed_row_smoke.py tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py tests/test_source_state_hybrid_observation_profile.py
uv run pytest tests/test_compact_owner_search_service.py tests/test_compact_coach_speed_row_smoke.py tests/test_source_state_hybrid_observation_profile.py::test_hybrid_profile_projects_action_only_owner_search_deferred_maintenance -q
uv run python scripts/build_compact_coach_speed_row_smoke.py --run-id opt132-local-owner-action-only-profile-proof-20260604 --unified-lifecycle-report artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json --search-service-kind owner_search_slab_proxy --owner-search-inner-search-service-kind compact_torch_search_service --owner-search-defer-maintenance --batch-size 8 --actor-count 1 --steps 12 --warmup-steps 4 --sample-batch-size 0 --sample-interval 4 --replay-pair-capacity 64 --learner-train-steps 1 --learner-num-unroll-steps 2 --policy-refresh-interval 4 --learner-device cpu --num-simulations 1
uv run pytest tests/test_compact_coach_speed_row_smoke.py tests/test_compact_owner_search_service.py tests/test_source_state_hybrid_observation_profile.py -q
uv run python scripts/build_compact_coach_speed_row_smoke.py --run-id opt132-local-owner-action-only-proof-hardened-fields-20260604 --unified-lifecycle-report artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json --search-service-kind owner_search_slab_proxy --owner-search-inner-search-service-kind compact_torch_search_service --owner-search-defer-maintenance --batch-size 8 --actor-count 1 --steps 12 --warmup-steps 4 --sample-batch-size 0 --sample-interval 4 --replay-pair-capacity 64 --learner-train-steps 1 --learner-num-unroll-steps 2 --policy-refresh-interval 4 --learner-device cpu --num-simulations 1
```

Hardened local proof facts:

```text
artifact=artifacts/local/curvytron_compact_coach_speed_row_results/opt132-local-owner-action-only-proof-hardened-fields-20260604/compact_coach_speed_row_smoke_report.json
sample_batch/requested/sample/target=0/0/160/160
replay submitted/requested/append=15/15/15
train_request/update=3/3
train_wall=0.856s
learner_update=0.845s
final_drain=0.778s
```

Action-feedback local proof facts:

```text
artifact=artifacts/local/curvytron_compact_coach_speed_row_results/opt132-local-owner-action-only-action-feedback-proof-20260604/compact_coach_speed_row_smoke_report.json
action_feedback_verified=true
transition/action/mismatch=15/240/0
expected/applied/replay_action_checksum=6120/6120/6120
train_wall=0.825s
learner_update=0.814s
final_drain=0.761s
```

Next speed-model audit:

```text
final drain is deferred maintenance, not only learner update:
  _append_replay + _train_and_publish + _refresh_search_from_owner_ref
current local timing says train/update dominates:
  train wall 0.825s, learner update 0.814s, final drain 0.761s
nested aggregate result:
  train wall/update/final drain 0.865s/0.855s/0.790s, but nested compact
  MuZero learner time 0.055s
import-prewarm result:
  moving LightZero learner-function imports into CompactMuZeroLearnerEdgeV1
  construction drops train wall/update/final drain to 0.072s/0.057s/0.043s
cadence-8 falsifier:
  train requests drop 3 -> 1 and train wall/update/final drain become
  0.035s/0.030s/0.019s, with nested compact MuZero learner time 0.030s
explicit drain-work/entry proof:
  `opt132-local-owner-action-only-explicit-drain-work-entry-proof-20260604`
  passes append requests/submitted/appended 15/15/15, staged/drained work items
  15/15, drained replay entries/appends 15/15, pending/inflight/failed
  0/false/false, parent rows 0, payload bytes 0, action feedback true, train
  wall 0.077s, and final drain 0.047s
inner two-phase proof:
  `opt132-local-owner-action-only-inner-two-phase-explicit-drain-proof-20260604`
  passes with compact Torch owner-search inner two-phase/device replay true,
  parent rows 0, payload bytes 0, train wall 0.065s, learner update 0.054s,
  and final drain 0.037s
normal-death scale result:
  slab and threaded scale rows preserve ownership proof but fail speed shape:
  speed about 25 env steps/sec, train/learner update about 14-15s, final drain
  about 12-13s
mock-fast ceiling:
  `opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-mockfast-r3-20260604`
  passes normal-death proof, carries learner terminal counters, runs
  214.06 env steps/sec locally, and drops final drain to 0.0026s
eager append falsifier:
  `opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-eagerappend-r1-20260604`
  fires eager append drains 7 and preserves update/lag/drain proof, but runs
  only 24.09 env steps/sec with learner update 15.12s and final drain 12.67s
async learner falsifier:
  `opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-asynclearner-r2-20260604`
  proves same-process async learner worker submit/completed/pending 6/6/0 and
  actions while async pending 47, but runs only 24.47 env steps/sec with async
  wait 14.17s, learner update 14.88s, and final drain 12.82s
async learner max-pending falsifier:
  `opt132-local-owner-action-only-inner2-slab-scale48-cadence8-b512-normal-asynclearner-max6-r1-20260604`
  proves the same-process async queue can observe max pending 5 and still close
  submit/completed/pending 6/6/0, but runs only 25.37 env steps/sec with async
  wait 13.56s, learner update 14.33s, and final drain 12.31s; queue depth is
  not a speed path
next experiment:
  change real learner resource/work shape, then rerun the same normal-death
  scale proof
```

## Rejected Next Moves

Do not continue pure attribution rows. They explain the old loop without
changing it.

Do not make "more tensor-native replay/sample builder" the next move. BD/BF
proved that lane and speed-failed it.

Do not lead with GPU mechanics or a framework port. PufferLib, EnvPool, Sample
Factory, and Isaac-style systems are constraints on ownership and fixed buffers,
not permission to bypass the current search/replay handoff.
