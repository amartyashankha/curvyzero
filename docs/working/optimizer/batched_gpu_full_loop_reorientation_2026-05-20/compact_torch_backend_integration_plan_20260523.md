# Compact Torch Backend Integration Plan, 2026-05-23

Status: optimizer working plan. This is not Coach launch advice.

## Plain Goal

Make one real candidate backend behind `CompactSearchServiceV1` that can be
compared against direct LightZero CTree on the same compact closed-loop profile.

The goal is not to change the MuZero algorithm. The goal is to remove the slow
Python/list/CTree boundary and keep the search work in fixed-shape arrays long
enough to measure whether the architecture is worth pursuing.

## Current Ground Truth

Already green:

```text
CompactRootBatchV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
-> materialized target rows
-> repo learner sample batch
-> opt-in stock LightZero target/sample hooks
```

Also green:

```text
real direct CTree compact service
-> selected actions drive the next hybrid env step
-> compact deferred rows match trusted immediate rows
```

New helper:

```text
src/curvyzero/training/compact_torch_search_service.py
```

This helper is only a safe staging area. It is profile-only and not trainer
ready.

## Next Implementation Ladder

1. Keep direct CTree as the semantic oracle.
2. Keep service-tax/mock rows as speed ceilings.
3. Build a fixed-shape Torch candidate that returns exactly:

```text
selected_action
visit_policy
root_value
optional raw counts / predicted diagnostics
```

4. Validate it through `validate_compact_search_result_v1(...)`.
5. Run the same closed-loop replay proof shape as direct CTree.
6. Only then run H100 speed rows.

## First Candidate Shape

The first candidate should stay behind a profile-only service class, roughly:

```text
CompactRootBatchV1
-> torch tensor observation/mask on device
-> model.initial_inference(...)
-> fixed-shape Torch tree arrays
-> model.recurrent_inference(...) once per sim
-> selected_action / visit_policy / root_value
-> CompactSearchResultV1
```

Do not reuse `_LightZeroArrayCeilingStackProbe.run(...)` as the final service
shape if it forces grid reshaping, public output materialization, or unrelated
telemetry into the hot path. It is useful donor code and an oracle for small
tests.

Concrete integration map from the code read:

```text
Added CompactTorchSearchServiceV1 in:
  src/curvyzero/training/compact_torch_search_service.py

Implement:
  CompactSearchServiceV1.run(root_batch)

Inside run:
  active = np.flatnonzero(root_batch.active_root_mask)
  select root_batch.observation/legal_mask at active roots
  perform exactly one model/search pass
  call compact_search_result_v1_from_arrays(...)
```

Useful donor code from the profile module:

```text
_run_dense_torch_mcts
_network_output_field
_policy_inverse_scalar_value
_recurrent_action_input
_sync_torch_device_if_cuda
```

Do not leave the real backend as a `_LightZeroArrayCeilingStackProbe` wrapper.
The direct CTree adapter may keep using `probe.run(...)` because it is the
oracle adapter. The candidate backend should own search directly so there is no
double-run and no hidden public-output materialization.

Local implementation status:

```text
CompactTorchSearchServiceV1 now exists.
It selects active roots, runs one policy model/search pass, and validates into
CompactSearchResultV1.
The profile surface now has an explicit compact_torch_search_service mode that
calls this service directly instead of wrapping _LightZeroArrayCeilingStackProbe.run.
```

Remote profile status:

```text
H100 B512/A16/sim16, compact replay proof on:
  direct_ctree_gpu_latent:      4,966 steps/sec, probe 5.702s
  service_tax_probe:           5,853 steps/sec, probe 2.857s
  compact_torch_search_service 5,140 steps/sec, probe 5.867s

Timing-split compact_torch_search_service:
  5,575 steps/sec, probe 5.098s
  initial model: 0.271s
  tree/recurrent loop: 4.250s

No-noise pair:
  direct_ctree_gpu_latent:      3,955 steps/sec, probe 6.944s
  compact_torch_search_service 5,704 steps/sec, probe 5.077s
```

Interpretation:

```text
The service boundary is real and remote replay proof passes. The current eager
Torch tree body is not the large speedup. Its hot slice is the Python/Torch
tree plus recurrent loop. The next attempt should either compile/fuse that body
for real or switch to an array-native search sidecar; do not keep polishing this
exact eager implementation unless a smaller trust gate needs it.
```

Input-mode caveat:

```text
compact_torch_search_service currently consumes CompactRootBatchV1 host uint8
observations directly. The profile/grid tooling now rejects other input modes
for this backend so a row cannot be mislabeled as pinned, float32, or resident
reuse.
```

## Required Fast-But-Wrong Gates

Before any speed row matters:

```text
active-root order: non-prefix roots stay in compact active order
legal masks: illegal actions get zero mass; single legal action is exact
backup math: reward, discount, root value, and visits match direct CTree toy rows
root noise: noise-off exact first; noise-on seeded later
latent slots: fake recurrent model proves chosen action changes the next logits
fresh input: same shape, different observations change model-dependent outputs
cache key: shape, sim count, dtype, device, model identity, and action count
timing: cold compile, warm search, recurrent calls, sync, assembly, readback
```

## Kill Conditions

Kill or demote the backend if:

```text
it cannot pass replay/RND/player identity gates
it wins sim16 but regresses sim32 on the same denominator
it falls back silently to eager/CPU/list paths
it only wins by omitting required readback into CompactSearchResultV1
it needs all actions legal for correctness rather than only for an optional compile path
```

## Next Local Steps

- Add more local helper guard tests for compile signatures and masks. Done:
  `tests/test_compact_torch_search_service.py` now covers signature drift,
  non-binary masks, inactive-root legality, active-root order, legal-mask
  visit mass, and same-shape fresh-observation behavior. Focused validation:
  `9 passed`.
- Add a candidate-vs-direct toy parity test before any trainer-facing claim.
- Add a profile-only service class only after the tests define the contract.
  Done for the first local candidate. It is wired to Modal profile modes, but is
  still profile-only.
- Add a closed-loop Torch service smoke. Done in
  `tests/test_source_state_batched_observation_boundary_profile.py`:
  service-selected actions drive the next env step and compact replay index
  rows materialize to the trusted immediate rows.
- Add profile-mode wiring. Done:
  `LIGHTZERO_ARRAY_CEILING_MODE_COMPACT_TORCH_SEARCH_SERVICE` calls
  `CompactTorchSearchServiceV1.run(root_batch)` directly and stores compact
  search arrays/results for the existing proof path.
- Add phase-honest timing. Done: the compact Torch mode now reports tensor
  prepare, initial inference, tree/recurrent loop, and readback instead of
  hiding everything inside one opaque search bucket.
- Add input-mode honesty. Done: compact Torch service rows fail closed unless
  `lightzero_array_ceiling_input_mode=host_uint8`.
- Run the existing minimal gate:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k 'compact_service or direct_ctree or dense_torch or single_legal or biased_logits'
```

## What Coach Should Do

Nothing changes for Coach yet. Coach should continue using the stock LightZero
lane. This backend plan is optimizer-only until it passes the same replay,
RND, player-identity, sampler, and closed-loop smoke gates.
