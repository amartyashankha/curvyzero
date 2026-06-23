# CompactSearchServiceV1 API Location, 2026-05-22

Status: docs-only bounded explorer note. I did not touch Coach training,
Modal volumes, checkpoints, evals, GIFs, tournaments, or source code.

## 1. Location

Put the interface in a new training contract module:

```text
src/curvyzero/training/compact_search_service.py
```

Keep the existing array schemas and validators in
`src/curvyzero/training/compact_policy_row_bridge.py` for the first patch, and
have the service module import:

```text
CompactRootBatchV1
CompactSearchResultV1
validate_compact_search_result_v1
```

Why this is the cleanest split:

- `compact_policy_row_bridge.py` already owns the root/result/replay dataclasses
  and the hard semantic checks. Reusing those avoids a second contract surface.
- A new module avoids making `compact_policy_row_bridge.py` even more mixed:
  policy-row V0 bridge, root/result V1 validation, replay chunks, index rows,
  and future backend protocols should not all keep accumulating in one file.
- `training` is the right dependency direction. Modal/profile code can import a
  training contract, but the contract must not import Modal, LightZero, JAX,
  torch backends, checkpoints, or Coach trainer code.
- Future backends can implement the same protocol from their own modules:
  `direct_ctree_gpu_latent` can stay profile/LightZero-owned, while mock,
  service-tax, and MCTX/JAX adapters can share the same boundary.

Do not put the interface in
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.
That file is a profile harness and backend zoo; making it the contract owner
would force non-Modal backends to depend on profile-only machinery.

## 2. Minimal API Now

Smallest useful skeleton:

```python
from typing import Protocol, runtime_checkable

from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1


@runtime_checkable
class CompactSearchServiceV1(Protocol):
    search_impl: str
    num_simulations: int

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        ...
```

Required behavior:

- `run` consumes only `CompactRootBatchV1`.
- `run` returns only `CompactSearchResultV1`.
- Implementations must return rows in active-root order, with
  `root_index == flatnonzero(root_batch.active_root_mask)`, unless a later
  contract explicitly permits another explainable order.
- Implementations must call `validate_compact_search_result_v1` or produce an
  object that passes the same checks before crossing the boundary.
- Backend-specific timing, model identity, fallback counts, logical/actual eval
  counts, and semantics should live in `CompactSearchResultV1.metadata`.

The replay edge stays outside the service:

```text
CompactRootBatchV1
-> CompactSearchServiceV1.run(...)
-> CompactSearchResultV1
-> build_compact_replay_index_rows_v1_from_search_result(...)
```

That keeps search ownership separate from the record-k to record-k+1 replay
attachment rules.

## 3. Do Not Include Yet

Do not include these in the shared interface yet:

- `HybridCompactBatch`; it is an input builder detail, not the search boundary.
- `HybridBatchedStackProbeResult` or profile telemetry schemas.
- LightZero policy/model objects, CTree roots, torch tensors, JAX devices, Modal
  handles, checkpoint ids, or volume paths.
- Replay chunk/index-row construction inside `run`.
- Public LightZero collect-output dicts or scalar timesteps.
- Async queues, resident service lifecycle, batching windows, weight refresh,
  model registry, or actor RPC semantics.
- RND reward shaping beyond carrying the already present root sidecars and
  result metadata.
- A common config dataclass until two real backends need the same constructor
  fields.

## 4. First Backend And Smallest Test

First backend to wrap: `direct_ctree_gpu_latent`.

It is the current real-search comparator and already emits the necessary
compact arrays through `_LightZeroCollectForwardStackProbe._last_direct_mcts_arrays`.
The first adapter should be a thin profile-owned wrapper that:

```text
CompactRootBatchV1
-> existing direct CTree GPU-latent search body
-> validate_compact_search_result_v1(...)
-> CompactSearchResultV1
```

Smallest semantic-drift test:

1. Use the existing deterministic fake-policy/FakeMCTS pattern from
   `tests/test_source_state_batched_observation_boundary_profile.py`.
2. Build a `HybridCompactBatch`, then a `CompactRootBatchV1`.
3. Run the old direct compact path and the new `CompactSearchServiceV1`
   adapter on the same roots.
4. Assert exact equality for `root_index`, `env_row`, `player`,
   `policy_env_id`, `selected_action`, `visit_policy`, and `root_value`.
5. Feed both results through
   `build_compact_replay_index_rows_v1_from_search_result(...)` and
   `materialize_compact_target_rows_from_index_rows_v1(...)`; the materialized
   rows must match the current target-row oracle.

This test should include non-prefix active roots as soon as possible. The
existing compact replay contract already covers that shape, and it is the best
cheap guard against "same action array, wrong row/player" drift.

## 5. Risky Smells

- The same compact facts are currently represented as dataclasses in
  `compact_policy_row_bridge.py` and as raw side-channel dicts named
  `_last_direct_mcts_arrays` / `_last_compact_search_arrays` in the profile
  probes. That is the main drift risk.
- Active-root filtering is still done inside profile consumers from
  `observation, action_mask`, then later revalidated against
  `CompactRootBatchV1`. The service should consume the root batch directly.
- Compact batch validation is duplicated between `_compact_batch_inputs(...)`
  and `build_compact_root_batch_v1(...)`.
- `_maybe_run_compact_service_replay_proof(...)` reconstructs search identity
  from telemetry strings and raw mappings. It is useful, but stringly typed.
- Mutable "last result" fields make future async or resident services risky.
- `direct_ctree_gpu_latent` still pays CPU CTree/list/D2H/listification costs.
  Wrapping it proves the boundary; it is not itself the speedup.
- `CompactRootBatchV1` copies observations by default. Hot profile wrappers
  should opt into `copy_observation=False` only when lifetime is clearly safe.
