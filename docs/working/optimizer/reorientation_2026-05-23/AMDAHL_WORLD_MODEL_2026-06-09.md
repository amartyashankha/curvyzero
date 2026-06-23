# Amdahl World Model

Date: 2026-06-09

This is a reset note, not a run ledger. `goal.md` remains the durable compass.

## Current Correction

The local fixed-action-tape harness now reports a real local whole-loop
denominator in addition to the old env-step/autoreset timer. The previous
env-step-only read was not enough to reason about Puffer-style ownership
because render/root/search/slab/replay/sample work could sit outside the main
speed number. New local fields include `whole_loop_wall_sec`,
`action_source_wall_sec`, `observation_wall_sec`, `search_wall_sec`,
`slab_replay_wall_sec`, `owner_slot_wall_sec`, named-surface total, residual,
and whole-loop rows/sec.

The first owner-buffer ceiling slice has landed locally behind
`--run-owner-slot-ceiling`. It proves this local chain:

```text
mechanics result
-> fixed owner mechanics step-frame slot
-> direct owner root/action step consumes the slot handle
-> action result metadata publishes the next joint action
-> next mechanics step consumes the owner-selected action
-> CompactOwnerSearchDirectStepperV1 stages the previous transition through
   search_service.stage_replay_append_entries()
-> the owner-slot service validates replay-payload handles/digests and drains
   staged previous-transition payloads
-> staged owner rows become CompactReplayColumnarAppendRecordV1 records
-> owner-context payloads become CompactDeviceReplayIndexRowsV1 device rows
-> _CompactReplayRingV1.append_columnar_entries() owns the append
-> _CompactReplayRingV1.sample() builds a resident unroll-1 sample batch from
   device replay rows with host fallback disabled
-> a nonterminal fixture builds a resident device learner unroll-2 batch from
   the real ring
-> local fixed replay-slot shim appends owner-selected rows
-> local sample handle is created/resolved with row/action/reward/done checksums
```

The proof is intentionally fail-closed. It requires slot writes, generation
checks, digest checks, root requests from slot, zero root requests from the
batch helper, zero `HybridCompactBatch` objects, action result write/read
counts, feedback action matches, zero root-observation copy bytes, no hidden
CTree/tolist calls, positive replay-slot/sample-handle counters for active
roots, and zero fake samples for terminal-only roots. It also keeps parent
replay objects, replay object entries, and selected-group objects at zero in
this local fixture.

The first production bridge helper has also landed:
`build_compact_replay_index_rows_v1_from_owner_action_context_payload()` builds
real `CompactReplayIndexRowsV1` from `CompactRootActionContextV1`,
`CompactSearchActionStepV1`, and replay payload facts without requiring a full
`CompactRootBatchV1` or `HybridCompactBatch`. A replay-contract parity test
proves those rows match the trusted root-batch builder for the same synthetic
transition.

Small local smoke result:

```text
command:
  uv run python scripts/benchmark_vector_fixed_action_tape.py
    --scenario scenarios/environment/source_kinematics_straight_multistep.json
    --batch-size 2 --warmup-steps 1 --measured-steps 3
    --body-capacity 8 --render-observation --run-owner-slot-ceiling

proof:
  owner_slot_ceiling_failure_reasons: []
  step_count: 4
  mechanics_slot_write_count: 4
  mechanics_slot_generation_verified_count: 4
  mechanics_slot_digest_verified_count: 4
  root_request_from_slot_count: 4
  root_request_from_batch_count: 0
  hybrid_compact_batch_object_count: 0
  action_result_write/read_count: 4/4
  prev_next_joint_action_mismatch_count: 0
  root_observation_copy_bytes: 0
  replay_slot_append_count/rows: 3/12
  stage_replay_transport/transition_count: 3/3
  stage_replay_payload_cache_hit/miss/release/pending: 3/0/3/1
  stage_replay_drained_record_count: 3
  stage_replay_device_index_rows_build_count/rows: 3/12
  replay_ring_append_call/record_count: 3/3
  replay_ring_stored_index_row_count: 12
  replay_ring_sample_device_replay_index_rows_sample: true
  replay_ring_resident_device_sample_batch: true
  learner_unroll2_built/rows: true/8
  learner_unroll2_shapes: action [8,2], reward [8,2], value [8,3], policy [8,3,3]
  learner_unroll2_host_fallback_allowed: false
  replay_ring_observation_provider_used_count: 0
  parent_replay_object_count: 0
  replay_slot_object_entry_count: 0
  selected_group_object_count: 0
  sample_gate_calls: 3
  sample_handle_create/resolve/inline/pending: 3/3/3/0
  stage_sample_handle_create/resolve/inline/pending: 3/3/3/0
  sample_row/target_count: 8/8
```

Do not read this as speed evidence. In the tiny CPU-oracle toy, observation
dominates the local denominator and the owner-slot/replay-ring surface is
small. The value is that the benchmark can now stop lying to us about the
denominator and can fail closed on real replay-ring append/sample movement
before H100. It still uses CPU torch resident shims and a local fixture; the
next stronger proof is fixed resident row/window slots or handle-ring sampling
inside the production owner graph, not another local proof of device rows.

## What Went Wrong

The recent work kept proving support rungs and then letting each rung pull the
next task into its own neighborhood. That produced real local correctness
work, but too much of it was gather/layout/proof plumbing after H100 rows had
already shown that standalone replay/sample gather work was not the main 2x
lane.

The failure was not variance. Variance explains repeat noise after an
architecture is plausible. The current blocker is structural: too much of the
whole loop still crosses Python-owned object/materialization boundaries.

## Amdahl Read

The fastest current support row is still the columnar/direct-table stack:

```text
run: opt132-h100-columnar-append-direct-table-b1024a1-normal-unroll2-m724-w180-r2-20260607
speed: 15852.67 env/s
wall: 46.7666s
reported surfaces:
  replay append: 14.219s
  direct append: 14.169s
  ring append: 11.523s
  search: 13.295s
  learner train: 9.778s
  parent wait: 17.655s
```

These timers overlap and should not be summed, but they are enough for the
decision:

```text
delete replay append alone: about 46.77 / (46.77 - 14.22) = 1.44x max
delete search alone: about 46.77 / (46.77 - 13.30) = 1.40x max
delete learner train alone: about 46.77 / (46.77 - 9.78) = 1.26x max
delete parent wait alone: about 46.77 / (46.77 - 17.66) = 1.61x max
```

A `2x` row needs roughly half the wall removed or overlapped. A `10x` row needs
almost the entire hot loop to become a different fixed-buffer pipeline. That
means another local gather micro-patch is not a credible main move unless it
also removes a larger owner boundary.

## Puffer-Style Pattern

PufferLib 4.0 is the extreme version of the fixed-buffer pattern:

```text
one contiguous allocation for tensor memory
no hot-loop tensor creation or reallocation
CUDA graph capture after warmup on stable memory addresses
environment chunks owned by rollout workers / streams
C environments write observations, actions, rewards, terminals in contiguous buffers
Python launches, configures, logs, and inspects; it does not own hot data
```

Sample Factory and EnvPool rhyme with this:

```text
components are rollout/inference/batcher/learner
components communicate by signals plus shared-buffer ids
observations/trajectories are written to shared buffers, not serialized objects
double buffering overlaps env work with inference/learner waits
```

CurvyTron is not currently close to that. It has useful proof/support pieces:
resident observations, owner-search action-only proof, direct transition
batches, columnar append, tensor-native learner batches, fixed-SoA diagnostics,
and handle vocabulary. But the measured loop still builds and moves too much
through Python object surfaces.

## Actual Boundaries

Current rough ownership:

```text
mechanics/env:
  Python/vector runtime plus compact rollout slab; reset RNG vectorized but not
  a static native env buffer engine.

search:
  owner-search service/proxy owns important proof and selected actions, but
  parent wait/search dispatch are still large.

replay:
  owner can materialize replay and append batches, but append/sample still pass
  through record objects, selected groups, and Python-owned metadata surfaces.

learner:
  consumes prebuilt tensor-native batches when paths hit, but the owner train
  boundary still pays sample/train/publication cadence costs.

parent Python:
  should coordinate only, but still sees too much hot-loop timing as wait,
  materialization, and proof/report extraction.
```

## Next Big Gate

Stop standalone fixed-SoA/locality/selected-gather work as the main lane.

The next credible implementation gate is the next Puffer-style owner-buffer
rung, not an H100 repeat of the current owner-slot proof:

1. Promote the local owner-slot replay/sample shim into production-owned
   replay/sample/learner-batch handles. Mechanics/root/action, the local
   `stage_replay_append_entries()` lifecycle, and the owner-action-context
   host/device replay-index row builders are now closed locally. Staged owner
   rows drain into `_CompactReplayRingV1`, sample resident device rows with
   host fallback false, and build a resident device learner unroll-2 batch on a
   nonterminal local fixture. The production fixed-SoA path now requests
   learner-batch handle-ring sampling, and the compact-owned learner boundary
   records resident-handle consumption only after learner train. The next proof
   must show corrected local whole-loop timing moved a real production owner
   surface, with resident-handle consumed true and materialized-parent fallback
   zero.
2. Measure the local ceiling with the corrected whole-loop denominator. The
   local proof must show moved surface large enough to plausibly beat columnar
   r2 before an H100 launch is worth doing.
3. If the ceiling is strong, implement the production owner ring: mechanics
   writes fixed root/transition/replay slots; search consumes root handles and
   returns action handles; replay owns row ids/windows; learner consumes batch
   handles. Parent sees counters and drain proof, not rows.
4. If the ceiling is weak, stop promising 10x from buffer work and move the
   target to the measured limiting boundary, likely search batching or learner
   publication/update.

The existing fixed-SoA learner-batch handle-ring patch is only support. Finish
or park it only if it strengthens fail-closed proof for the broader owner ring.
Do not launch H100 for it alone.

## Validation Anchor

Latest local validation after the owner-slot/whole-loop correction:

```text
uv run ruff check scripts/benchmark_vector_fixed_action_tape.py tests/test_benchmark_vector_fixed_action_tape.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/compact_policy_row_bridge.py tests/test_compact_search_replay_contract.py
  passed

uv run pytest tests/test_benchmark_vector_fixed_action_tape.py tests/test_compact_search_replay_contract.py -q
  57 passed

uv run pytest tests/test_source_state_hybrid_observation_profile.py::test_fixed_soa_samples_row_level_successors_for_coalesced_transition_batch -q
  1 passed

uv run python scripts/benchmark_vector_fixed_action_tape.py --scenario scenarios/environment/source_kinematics_straight_multistep.json --batch-size 2 --warmup-steps 1 --measured-steps 3 --body-capacity 8 --render-observation --run-owner-slot-ceiling --output /private/tmp/curvy_owner_slot_device_rows_unroll2_b2_m3_w1_20260609.json
  status pass; owner-slot failures []; stage device builds/rows 3/12; ring append calls/records/rows 3/3/12; device row sample true; resident device sample true; learner unroll-2 built/rows true/8; host fallback false

uv run pytest tests/test_compact_coach_speed_row_smoke.py::test_owner_search_replay_store_metadata_receives_fused_tensor_native_flags tests/test_compact_owned_loop.py tests/test_source_state_hybrid_observation_profile.py::test_compact_replay_ring_samples_partial_terminal_entry_without_successor -q
  28 passed
```
