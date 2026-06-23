# Compact Service Next Smoke Plan, 2026-05-23

Status: working plan. This is optimizer-only; do not use it as Coach launch
advice.

Update: the first local version of this smoke is implemented in
`tests/test_source_state_batched_observation_boundary_profile.py` as
`test_real_direct_ctree_compact_service_drives_next_step_and_matches_rows`.
Focused validation passed.

## Current Truth

The compact service path now has these local gates:

- compact search result identity and legality checks
- compact replay index row materialization
- terminal final-observation handling
- RND latest-frame/reward attachment
- repo learner-facing sample batch parity
- stock LightZero target-hook parity
- stock LightZero public `MuZeroGameBuffer.sample(...)` parity

The remaining trust gap is not the sampler edge. The remaining trust gap is a
matched stock-vs-candidate smoke with a real compact backend.

The first local real-backend smoke now exists. The remaining gap is the same
idea on the full candidate backend once the new compiled/fused Torch service is
implemented.

## Next Smoke

Use the smallest real backend that already exists:

```text
_LightZeroCollectForwardStackProbe
arrays_boundary=True
arrays_boundary_impl=direct_ctree or direct_ctree_gpu_latent
run_compact_batch_with_replay_chunk(...)
```

Compare it against the trusted immediate row path on the same tiny closed-loop
record:

- selected actions
- visit policy / target policy
- root value
- env row
- player
- policy env id
- reward
- done / terminal / final observation
- RND latest-frame slice
- public sample rows if a buffer can be built

Implemented focused validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k 'real_direct_ctree_compact_service_drives_next_step_and_matches_rows'

1 passed
```

## Guardrails

- This is still profile-only.
- Skip cleanly if local `lzero`/CTree hooks are unavailable.
- Do not call Modal or live Coach runs.
- Do not compare against MCTX toy search as if it were stock LightZero.
- Do not promote dense Torch from a speed row alone; it still needs the same
  identity, replay, RND, and sampler gates.

## Why This Is Next

The service-tax profile says the direct CTree boundary is expensive. The public
sampler test says compact rows can feed LightZero sampling. The missing bridge
is one real compact backend producing those rows in a matched closed-loop smoke.
That is the next thing that makes any speed claim trustworthy.

## Backend Direction

The next real backend should be a new compiled/fused fixed-shape Torch compact
search service behind `CompactSearchServiceV1`.

Do not promote the existing eager `dense_torch_mcts` probe. It is useful donor
logic, but the same-denominator rows already showed sim32 measured regression.
The new service should keep fixed shapes, preallocate tree tensors, avoid
per-simulation CPU materialization, and return only:

```text
selected_action
visit_policy
root_value
optional debug payloads
```

through `validate_compact_search_result_v1(...)`.

Use these as comparators:

- direct LightZero CTree: semantic oracle
- service-tax/mock rows: speed ceilings
- MCTX/JAX sidecar: architecture signal only, not LightZero parity

2026-05-23 local helper status:

```text
src/curvyzero/training/compact_torch_search_service.py
tests/test_compact_torch_search_service.py
```

This now contains the first profile-only `CompactTorchSearchServiceV1`
candidate. It selects active roots, runs one model/search pass, and validates
into `CompactSearchResultV1`. It also records fixed-shape compile eligibility,
profile-only labels, and small select/backup helper tests. It is not wired into
the trainer or Modal profile launcher, not a LightZero CTree replacement, and
not a speed claim.

Initial keep/kill gate:

```text
H100 B512/A16 sim16 and sim32, compact replay proof on.
Keep only if it beats direct_ctree_gpu_latent by about 1.5x end-to-end or 2x on
the search boundary while passing identity/replay/RND/sampler gates.
```

## Fast But Wrong Gates

Before any compiled/fused Torch backend gets an H100 profile row that we trust,
it must avoid these failure modes:

- active-root compaction drift: output rows must stay in
  `np.flatnonzero(active_root_mask)` order.
- legal-mask bugs: illegal actions get zero visit mass, and single-legal-action
  rows stay exact.
- search semantics drift: PUCT, reward, discount, and backup values must match
  direct CTree on small fixed toy cases.
- root-noise mismatch: noise-off rows should be exact against direct CTree;
  noise-on rows must be seeded and legal.
- recurrent action/latent corruption: action shape and latent slots must be
  checked with a fake recurrent model.
- stale observation reuse: two same-shape calls with different observations
  must produce different model-dependent outputs.
- compile-cache misuse: shape, sim count, dtype, device, model identity, and
  action count must be part of the cache signature or rejected.
- timing lies: every profile must include cold compile, warm steady state,
  recurrent/model calls, explicit CUDA sync boundaries, output assembly, and
  readback needed for `CompactSearchResultV1`.
