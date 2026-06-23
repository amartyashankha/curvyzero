# Next Compact Service Synthesis, 2026-05-22

Status: optimizer working memory. No live Coach training run, checkpoint, eval,
GIF, tournament artifact, or Modal volume was touched.

## Plain State

We have stopped treating raw rendering as the main problem.

Newest durable common-telemetry rerun:

```text
H100, B512/A16, sim16, 80 measured steps, 20 warmup, compact replay proof on:
  mock_search_service:       14,970.0 steps/sec
  service_tax_probe:         11,854.6 steps/sec
  direct_ctree_gpu_latent:    5,965.1 steps/sec
```

Ratios:

```text
mock / direct:        2.51x
service_tax / direct: 1.99x
mock / service_tax:   1.26x
```

Durable result dirs:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/rerun_current_compact_direct_20260523
artifacts/local/curvytron_hybrid_observation_profile_results/rerun_current_compact_mock_20260523
artifacts/local/curvytron_hybrid_observation_profile_results/rerun_current_compact_service_tax_20260523
```

The fresh profile-only service comparison says:

```text
H100, B512/A16, 60 measured steps, 15 warmup, compact replay proof on:
  mock_search_service:       17,711.9 steps/sec
  service_tax_probe:         12,461.6 steps/sec
  direct_ctree_gpu_latent:    7,155.7 steps/sec
```

Plain meaning:

```text
mock_search_service is fake search.
service_tax_probe pays real model calls but no real CTree.
direct_ctree_gpu_latent is real LightZero CTree MCTS.
```

Ratios:

```text
mock / direct:        2.48x
service_tax / direct: 1.74x
mock / service_tax:   1.42x
```

The older 60/15 row and the newer 80/20 durable row agree on the direction:
there is real headroom in the search/dataflow boundary, but not a full 10x by
itself.

## Current Dataflow

The hot loop is:

```text
CPU selected actions
-> CPU CurvyTron env step
-> compact render/search sidecars
-> GPU latest frame / stack
-> compact roots and legal masks
-> search
-> CPU selected actions
-> compact replay payloads
```

The selected action readback is small and unavoidable while the env is
CPU-owned. The large and wasteful parts are:

- host visual stack and root observation movement;
- render/search-input handoff;
- scalar LightZero object fanout;
- CPU/list CTree boundary;
- per-simulation model output readback/listification;
- replay/RND materialization before a compact row contract proves it is safe.

## Amdahl Read

On the latest refresh-on compact rows, deleting observation refresh was only a
`1.5-1.6x` ceiling. Raw GPU draw itself was under 1% of wall time.

On the fresh direct CTree sidecar row, the search boundary was large:

```text
total measured:                         8.586s
batched_stack_probe_wall:               5.423s
lightzero_mcts_arrays_boundary_total:   5.048s
lightzero_mcts_arrays_boundary_search:  3.941s
model total:                            1.290s
direct boundary non-model:              3.758s
ctree traverse + backprop:              1.037s
root prepare:                           0.494s
observation / renderer stack update:    1.263s
actor step wall:                        1.550s
compact replay proof:                   0.174s
```

So the current optimizer target is not one leaf timer. It is the ownership
boundary:

```text
compact roots -> model/search -> compact action/visit/value -> replay rows
```

## Decision

The next serious implementation lane is:

```text
CompactRootBatchV1
-> CompactSearchServiceV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

Stock LightZero scalar objects should remain as validation/debug edges, not the
hot-path owner for this lane.

## What Must Happen First

Before any speed path becomes trainer-facing, we need a small closed-loop parity
proof.

Minimum proof:

```text
for the same record k:
  search consumes observation/root k
  selected_action[k] is the action used to step env k -> k+1
  visit_policy[k] and root_value[k] attach to the same root
  reward/done/final_observation come from record k+1
  materialized compact replay rows equal the current target-row oracle
```

This must cover non-prefix roots and non-identity ids. Terminal/final
observation and RND reward shaping should be added as soon as the small proof is
stable. The 2026-05-23 action-feedback telemetry now proves that search-selected
joint actions drive the next env step in the compact replay proof; RND and
player perspective are still the remaining promotion gates.

2026-05-23 update:

```text
The compact replay proof now also verifies identity and RND latest-frame
attachment:
  compact_service_replay_identity_feedback_verified
  compact_service_replay_rnd_latest_verified
  compact_service_replay_compact_root_row_checksum
  compact_service_replay_player_checksum
  compact_service_replay_policy_env_id_checksum
  compact_service_replay_rnd_latest_checksum

The next combined proof now exists. It proves the selected compact roots and
player sidecars line up with RND latest-frame extraction, then pushes
materialized compact replay observations through the actual
`CurvyRNDRewardModel` collect/train/estimate path. It also covers a terminal row
and verifies that the terminal replay row uses the final observation rather than
the latest live observation.

What is still not proven:

```text
The learner sampler edge. Before a compact service speed path becomes
trainer-facing, a multi-record closed-loop proof should show that the faster
path and the trusted immediate replay path feed the same learner-facing samples.
```
```

## Active Delegation

- Beauvoir: add the first compact replay parity test in
  `tests/test_compact_search_replay_contract.py`.
- Huygens: map the clean source location and minimum shape for
  `CompactSearchServiceV1`.

## Current Implementation Update

Completed:

```text
tests/test_compact_search_replay_contract.py
  test_closed_compact_loop_index_rows_materialize_same_as_immediate_rows

src/curvyzero/training/compact_search_service.py
  CompactSearchServiceV1 Protocol

src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
  _LightZeroCollectForwardCompactSearchService adapter
  _LightZeroArrayCeilingCompactSearchService adapter

tests/test_compact_search_replay_contract.py
  test_compact_search_service_v1_protocol_runs_fake_service_to_index_rows

tests/test_source_state_batched_observation_boundary_profile.py
  test_lightzero_compact_search_service_adapter_preserves_root_identity
  test_lightzero_array_ceiling_compact_search_service_adapter_preserves_identity
```

What this proves:

```text
The compact service boundary can consume CompactRootBatchV1, return
CompactSearchResultV1, drive compact replay index rows, and materialize those
rows back to the same target rows as the immediate compact target-row path.

The first profile-owned direct CTree adapter can now be called through the
CompactSearchServiceV1 shape and preserves root_index, env_row, player,
policy_env_id, selected_action, visit_policy, and root_value.

The array-ceiling adapter gives mock/service-tax style probes the same boundary
shape, so the three comparator families no longer need separate conceptual
interfaces.

The shared helper `compact_search_result_v1_from_arrays` now validates arrays
from an already-run probe. This is important because it lets the profile path
convert direct/mock/service-tax arrays into `CompactSearchResultV1` without
running search a second time.
```

Validation:

```text
uv run ruff check src/curvyzero/training/compact_search_service.py tests/test_compact_search_replay_contract.py
uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py -k 'closed_compact_loop_index_rows or compact_search_service_v1_protocol'
uv run python -m py_compile src/curvyzero/training/compact_search_service.py
uv run ruff check src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py src/curvyzero/training/compact_search_service.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_compact_search_replay_contract.py
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py -k compact_search_service_adapter
uv run python -m py_compile src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py src/curvyzero/training/compact_search_service.py
uv run ruff check src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py tests/test_source_state_batched_observation_boundary_profile.py
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py -k 'compact_search_service_adapter'
uv run python -m py_compile src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py
```

Result:

```text
ruff passed
pytest compact replay: 2 passed, 7 deselected
pytest boundary adapter: 1 passed, 105 deselected
pytest boundary adapters after array-ceiling adapter: 2 passed, 105 deselected
pytest compact replay full file after array-helper test: 10 passed
pytest boundary profile full file after no-double-run test: 108 passed
pytest hybrid observation profile full file after proof-helper refactor: 35 passed
py_compile passed
```

## 2026-05-23 Dataflow Wave

The current adapter state is useful, but it is not the final hot-path wiring.
The direct CTree and array-ceiling adapters call their wrapped probe. If the
profile loop called an adapter after already running the probe, it would measure
two searches and lie to us.

So the next code move is not "just call the adapter everywhere." The next code
move is either:

```text
probe produces arrays once
-> arrays convert to CompactSearchResultV1
-> replay/telemetry consume that result
```

or:

```text
CompactSearchServiceV1 owns the one probe call
-> profile loop consumes only the service result
```

Both are acceptable. Double-running search is not.

Fresh validation after the adapter work:

```text
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py
=> 108 passed

uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py
=> 10 passed

uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py
=> 35 passed
```

New parallel critique wave:

```text
subagent_full_iteration_dataflow_critique_20260523.md
subagent_sync_budget_and_experiments_20260523.md
subagent_architecture_designs_2x_5x_10x_20260523.md
subagent_optimizer_validation_gate_audit_20260523.md
```

Sidecar synthesis:

```text
The current LightZero service-comparator lane is not fully GPU-resident. It can
use a GPU-backed JAX renderer, but the comparator often returns frames to a host
stack and then sends that stack to Torch/LightZero. Do not mix its steps/sec
with older resident MCTX profile rows as if they were the same currency.

Actions, masks, visits, and root values are tiny. The bad sync shape is tiny
payloads inside the per-simulation loop: CPU CTree traverse, GPU recurrent
model, GPU-to-CPU model output, Python listification, CPU CTree backprop. That
is the suspicious wall in direct LightZero CTree GPU-latent rows.

2x class: make the compact service real and replay-valid.
5x class: compact service plus fixed-shape search plus compact env/replay
ownership.
10x class: service/device-resident architecture with many roots in flight.

The first closed-loop action-feedback checksum is now explicit: actual
search-selected actions must match the next env step's applied joint action.
Promotion is still blocked until the same proof is extended through RND and
player-perspective rows.

Mock/service-tax array-ceiling probes now also have a `run_compact_batch` path.
That path runs the probe once, validates the produced arrays into
`CompactSearchResultV1`, and emits the common `compact_service_*` telemetry
family.
```

## Next Implementation Order

1. Add the closed compact-loop replay parity test. Done.
2. Define the minimal `CompactSearchServiceV1` boundary in the compact training
   contract layer.
3. Wrap the current `direct_ctree_gpu_latent` profile backend behind that
   boundary without changing semantics. First adapter skeleton is done.
4. Wrap mock/service-tax array-ceiling probes behind the same boundary. First
   adapter skeleton is done.
5. Add and use `compact_search_result_v1_from_arrays` so existing probe arrays
   can become `CompactSearchResultV1` without double-running search. Done.
6. Add explicit action-feedback verification telemetry to the compact replay
   proof. Done.
7. Make mock/service-tax array-ceiling compact batches report through the
   compact service telemetry family without a second probe call. Done.
8. Next: run the fresh direct/mock/service-tax profile rows again and compare
   them under the common compact service telemetry fields.
9. Put one real fixed-shape search backend behind the same API.
10. Only after parity gates pass, decide whether to connect the compact service
   lane to a trainer-facing path.

## Kill Rules

Stop a lane quickly if:

- it cannot produce replay rows matching the trusted target-row oracle;
- it hides fallbacks or scalar materialization inside the speed number;
- it improves only a local timer but not total steps/sec or roots/sec;
- it cannot beat direct CTree by at least a clear class margin in the same
  denominator;
- it changes player perspective, terminal final observation, legal masks, or
  RND inputs.

## Short Recommendation

Build the compact service boundary and its parity proof. Do not spend the next
main patch on renderer-only work. Do not promote direct CTree, MCTX, mock, or
service-tax profile rows as Coach training speed until the compact replay/RND
contract proves the rows are semantically attached to the same facts as the
trusted path.
