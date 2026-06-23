# Compact Slab Patch Plan, 2026-05-23

Status: docs-only implementation-design sidecar. I inspected the local profile
code and tests. No Coach launch code, live runs, trainer defaults, Modal
Volumes, checkpoints, evals, or tournaments should be touched by this slice.

## Recommendation

Add the smallest possible profile-only owner:

```text
HybridBatchedObservationProfileManager.step
-> _make_compact_batch(...)
-> CompactRolloutSlab.step(compact_batch)
-> CompactRootBatchV1
-> CompactSearchServiceV1.run(...)
-> CompactSearchResultV1
-> selected_action[B,P] for the next env tick
-> CompactReplayIndexRowsV1 when the following compact batch arrives
-> materialize_lightzero_scalar_timestep only if the old scalar debug edge is enabled
```

This keeps the current contract chain intact:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> CompactSearchServiceV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

It should not rewrite Coach. It should not try to replace the stock LightZero
training lane. It should only make the existing profile loop name the compact
state/search/replay owner that the newer profiles are implicitly asking for.

## Exact Patch Surface

### New file

`src/curvyzero/training/compact_rollout_slab.py`

Purpose: own one profile-only compact rollout stream. It is deliberately a thin
owner over existing contracts, not a new replay format.

Skeleton:

```python
"""Profile-only compact rollout slab for search-service dataflow probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.training.compact_policy_row_bridge import CompactReplayIndexRowsV1
from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_replay_index_rows_v1_from_search_result,
)
from curvyzero.training.compact_policy_row_bridge import build_compact_root_batch_v1
from curvyzero.training.compact_search_service import CompactSearchServiceV1
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.source_state_hybrid_observation_profile import HybridCompactBatch


@dataclass(frozen=True, slots=True)
class CompactRolloutSlabStepV1:
    schema_id: str
    record_index: int
    compact_batch: HybridCompactBatch
    root_batch: CompactRootBatchV1
    search_result: CompactSearchResultV1
    next_joint_action: np.ndarray
    committed_index_rows: CompactReplayIndexRowsV1 | None
    telemetry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PendingCompactSearchV1:
    record_index: int
    compact_batch: HybridCompactBatch
    root_batch: CompactRootBatchV1
    search_result: CompactSearchResultV1
    next_joint_action: np.ndarray


class CompactRolloutSlab:
    """Profile-only owner for compact root/search/action/replay index flow."""

    profile_only = True
    calls_train_muzero = False
    trainer_defaults_changed = False
    touches_live_runs = False

    def __init__(
        self,
        *,
        batch_size: int,
        player_count: int,
        search_service: CompactSearchServiceV1,
        search_lane: str,
        policy_source: str,
        copy_root_observation: bool = False,
    ) -> None:
        self.batch_size = int(batch_size)
        self.player_count = int(player_count)
        self.search_service = search_service
        self.search_lane = str(search_lane)
        self.policy_source = str(policy_source)
        self.copy_root_observation = bool(copy_root_observation)
        self._record_index = 0
        self._pending: _PendingCompactSearchV1 | None = None

    def step(self, compact_batch: HybridCompactBatch) -> CompactRolloutSlabStepV1:
        """Commit the previous search into index rows, then search current roots."""

        self._validate_batch_shape(compact_batch)
        committed = self._commit_previous(compact_batch)
        root_batch = build_compact_root_batch_v1(
            compact_batch,
            search_lane=self.search_lane,
            metadata={"compact_rollout_slab": True},
            copy_observation=self.copy_root_observation,
        )
        search_result = self.search_service.run(root_batch)
        next_joint_action = selected_joint_action_from_search_result(
            root_batch,
            search_result,
            batch_size=self.batch_size,
            player_count=self.player_count,
        )
        record_index = self._record_index
        self._pending = _PendingCompactSearchV1(
            record_index=record_index,
            compact_batch=compact_batch,
            root_batch=root_batch,
            search_result=search_result,
            next_joint_action=next_joint_action,
        )
        self._record_index += 1
        return CompactRolloutSlabStepV1(
            schema_id="curvyzero_compact_rollout_slab_step/v1",
            record_index=record_index,
            compact_batch=compact_batch,
            root_batch=root_batch,
            search_result=search_result,
            next_joint_action=next_joint_action,
            committed_index_rows=committed,
            telemetry=_slab_telemetry(root_batch, search_result, committed),
        )

    def _commit_previous(
        self,
        next_batch: HybridCompactBatch,
    ) -> CompactReplayIndexRowsV1 | None:
        pending = self._pending
        if pending is None:
            return None
        if not np.array_equal(pending.next_joint_action, next_batch.joint_action):
            raise ReplayCompatibilityError(
                "compact slab next batch did not apply staged selected actions"
            )
        return build_compact_replay_index_rows_v1_from_search_result(
            pending.compact_batch,
            pending.root_batch,
            pending.search_result,
            record_index=pending.record_index,
            next_joint_action=next_batch.joint_action,
            next_reward=next_batch.reward,
            next_done=next_batch.done,
            next_terminated=next_batch.done,
            next_truncated=np.zeros_like(next_batch.done, dtype=np.bool_),
            next_final_reward_map=next_batch.reward,
            next_final_observation_row_mask=next_batch.final_observation_row_mask,
            policy_source=self.policy_source,
            metadata={"compact_rollout_slab": True},
        )

    def _validate_batch_shape(self, batch: HybridCompactBatch) -> None:
        observation = np.asarray(batch.observation)
        expected = (self.batch_size, self.player_count)
        if observation.shape[:2] != expected:
            raise ReplayCompatibilityError("compact slab batch shape mismatch")
        if np.asarray(batch.joint_action).shape != expected:
            raise ReplayCompatibilityError("compact slab joint_action shape mismatch")


def selected_joint_action_from_search_result(
    root_batch: CompactRootBatchV1,
    search_result: CompactSearchResultV1,
    *,
    batch_size: int,
    player_count: int,
    inactive_action: int = 0,
) -> np.ndarray:
    """Map active-root search actions back to a dense [B,P] action buffer."""

    joint_action = np.full(
        (int(batch_size), int(player_count)),
        int(inactive_action),
        dtype=np.int16,
    )
    root_index = np.asarray(search_result.root_index, dtype=np.int64)
    env_row = np.asarray(search_result.env_row, dtype=np.int64)
    player = np.asarray(search_result.player, dtype=np.int64)
    selected = np.asarray(search_result.selected_action, dtype=np.int16)
    legal_mask = np.asarray(root_batch.legal_mask, dtype=np.bool_)
    for output_row, compact_root in enumerate(root_index):
        action = int(selected[output_row])
        if action < 0 or action >= ACTION_COUNT:
            raise ReplayCompatibilityError("compact slab selected action out of range")
        if not bool(legal_mask[int(compact_root), action]):
            raise ReplayCompatibilityError("compact slab selected action is illegal")
        joint_action[int(env_row[output_row]), int(player[output_row])] = np.int16(action)
    return joint_action
```

The implementation can add the tiny `_slab_telemetry(...)` helper in the same
file. Keep it boring: root count, active root count, selected-action checksum,
index-row count, schema ids, and `observation_copied`.

Important import-cycle rule: `source_state_hybrid_observation_profile.py` should
not import this new module at top level. The new module may import
`HybridCompactBatch` because `compact_policy_row_bridge.py` already does. The
manager should accept the slab as an injected object and call it by duck type.

### Existing file edit

`src/curvyzero/training/source_state_hybrid_observation_profile.py`

Add one optional profile-only owner to `HybridBatchedObservationProfileManager`
and `run_hybrid_observation_profile`.

Manager skeleton:

```python
class HybridBatchedObservationProfileManager:
    def __init__(
        self,
        config: HybridObservationProfileConfig,
        *,
        observation_renderer: SourceStateBatchedObservationRenderer | None = None,
        batched_stack_probe: HybridBatchedStackProbe | None = None,
        compact_rollout_slab: Any | None = None,
    ) -> None:
        ...
        self.compact_rollout_slab = compact_rollout_slab
```

Step insertion point: after `_make_compact_batch(...)` has built
`compact_batch`, and before:

```python
if bool(self.config.materialize_scalar_timestep):
    timestep, flat_obs, target_reward = materialize_lightzero_scalar_timestep(...)
```

Patch shape:

```python
compact_batch: HybridCompactBatch | None = None
if self.batched_stack_probe is not None or self.compact_rollout_slab is not None:
    started = time.perf_counter()
    compact_batch = _make_compact_batch(...)
    timings["compact_batch_build_sec"] = _elapsed(started)

compact_slab_step = None
compact_slab_telemetry: dict[str, Any] = {}
if self.compact_rollout_slab is not None:
    if compact_batch is None:
        raise ValueError("compact_rollout_slab requires compact_batch")
    started = time.perf_counter()
    compact_slab_step = self.compact_rollout_slab.step(compact_batch)
    timings["compact_rollout_slab_sec"] = _elapsed(started)
    compact_slab_telemetry = dict(getattr(compact_slab_step, "telemetry", {}))

if self.batched_stack_probe is not None:
    probe_result = _run_batched_stack_probe(self.batched_stack_probe, compact_batch)
```

Return surface: add only enough fields for tests and profile aggregation:

```python
@dataclass(frozen=True, slots=True)
class HybridObservationProfileStep:
    ...
    compact_slab_step: Any | None = None
    compact_slab_telemetry: dict[str, Any] | None = None
```

Timing fields: add `compact_rollout_slab_sec` to
`HYBRID_OBSERVATION_TIMING_FIELDS`.

Profile harness: allow `materialize_scalar_timestep=False` when either
`batched_stack_probe` or `compact_rollout_slab` is present.

```python
if (
    batched_stack_probe is None
    and compact_rollout_slab is None
    and not bool(config.materialize_scalar_timestep)
):
    raise ValueError(
        "materialize_scalar_timestep=False requires batched_stack_probe or "
        "compact_rollout_slab"
    )
```

In the loop, prefer the slab's staged action for the next iteration:

```python
next_compact_slab_action: np.ndarray | None = None
...
if next_compact_slab_action is not None:
    actions = next_compact_slab_action.astype(np.int16, copy=True)
else:
    actions = rng.integers(...)
step = manager.step(actions)
if step.compact_slab_step is not None:
    next_compact_slab_action = step.compact_slab_step.next_joint_action
```

This mirrors the existing compact replay proof loop, but moves ownership into a
named object instead of leaving it as helper-function glue.

### Optional boundary-profile edit

`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`

Do not move the big direct CTree code. Just add a slab profile mode or a small
local builder that wraps the existing services:

```python
from curvyzero.training.compact_rollout_slab import CompactRolloutSlab

slab = CompactRolloutSlab(
    batch_size=batch_size,
    player_count=PLAYER_COUNT,
    search_service=_LightZeroArrayCeilingCompactSearchService(probe)
    if array_ceiling
    else _LightZeroCollectForwardCompactSearchService(probe),
    search_lane=mode_or_impl,
    policy_source="profile_compact_rollout_slab",
)
```

Keep this opt-in and profile-only. No Coach config should import it.

## Tests To Add

### New unit file

`tests/test_compact_rollout_slab.py`

Add a fake `CompactSearchServiceV1` and use existing synthetic compact-batch
helpers from `tests/test_compact_search_replay_contract.py` where practical.

1. `test_compact_rollout_slab_maps_search_result_to_dense_joint_action`

Proves `selected_action` maps through `root_index/env_row/player` into `[B,P]`,
including non-prefix active roots and inactive roots filled with `0`.

2. `test_compact_rollout_slab_preserves_contract_chain`

Asserts the step returns:

```text
HybridCompactBatch
CompactRootBatchV1
CompactSearchResultV1
committed_index_rows is None on first step
metadata schema ids are v1
observation_copied is False
```

3. `test_compact_rollout_slab_second_step_commits_index_rows`

Run slab step 0, feed `step0.next_joint_action` into a second compact batch,
run slab step 1, and assert `step1.committed_index_rows` is
`CompactReplayIndexRowsV1` with `observation_materialized=False` and
`next_observation_materialized=False`.

4. `test_compact_rollout_slab_rejects_unapplied_selected_action`

Change one action in the next compact batch and assert it fails before replay
rows become sample-visible.

5. `test_compact_rollout_slab_terminal_final_observation_mask_survives_commit`

Use a terminal next batch and assert the committed index rows carry
`next_final_observation_row=True` for the terminal searched row.

### Existing hybrid manager tests

`tests/test_source_state_hybrid_observation_profile.py`

1. `test_hybrid_manager_runs_compact_rollout_slab_before_scalar_materialization`

Use a fake slab that records receipt of `HybridCompactBatch`; configure
`materialize_scalar_timestep=False`; assert `materialized_timestep_count == 0`,
`compact_batch_build_sec > 0`, and `compact_rollout_slab_sec > 0`.

2. `test_hybrid_profile_accepts_slab_as_pre_scalar_consumer`

Same guard as the existing skip-scalar test, but with `compact_rollout_slab`
instead of `batched_stack_probe`.

3. `test_hybrid_profile_compact_slab_action_feedback_drives_next_step`

Use a fake search service returning deterministic legal actions. Run two steps
through `run_hybrid_observation_profile(..., compact_rollout_slab=slab)`.
Assert step 1 applied the staged action and the slab committed one
`CompactReplayIndexRowsV1`.

### Boundary/profile canary

`tests/test_source_state_batched_observation_boundary_profile.py`

Add a no-LightZero canary first:

```text
test_array_ceiling_service_tax_compact_rollout_slab_drives_next_step_and_index_rows
```

Then optionally add a LightZero-gated direct canary parallel to the existing
`test_real_direct_ctree_compact_service_drives_next_step_and_matches_rows`:

```text
test_real_direct_ctree_compact_rollout_slab_drives_next_step_and_matches_rows
```

The second one should be skipped unless `lzero`, `ding`, and `torch` are
installed. It should not run remote jobs.

## Why This Is The Smallest Useful Slice

- It does not touch Coach.
- It does not replace `CompactRootBatchV1`, `CompactSearchResultV1`, or
  `CompactReplayIndexRowsV1`.
- It keeps stock LightZero scalar timestep materialization as an optional edge.
- It reuses existing direct CTree, service-tax, mock, and compact Torch service
  adapters through `CompactSearchServiceV1`.
- It names the state owner needed by the next profile row without claiming a
  trainer integration.

## Kill Criteria

Immediate semantic kill:

- Any local parity test above fails.
- The slab imports Modal, Coach, `train_muzero`, tournament, eval, checkpoint,
  or live-run code.
- The manager must import the slab implementation at top level and creates an
  import cycle with `compact_policy_row_bridge.py`.
- `selected_action` can be illegal, out of range, or fail to match the next
  `joint_action`.
- `CompactReplayIndexRowsV1` materializes observations during collection.
- Terminal rows lose final-observation-before-autoreset identity.

Profile kill on the same H100 denominator:

```text
B512/A16, steps >= 80, warmup >= 20,
materialize_scalar_timestep=False,
compact replay/index proof on,
host_uint8 or host_uint8_pinned input mode, same model/search settings.
```

Stop polishing the slab patch if:

- `mock_search_service` with the slab is more than 10 percent slower than the
  current no-slab mock profile, because the owner added overhead instead of
  removing object churn.
- `service_tax_probe` with the slab is more than 10 percent slower than the
  current no-slab service-tax profile, because the next architecture slice did
  not preserve the measured headroom.
- A real backend behind `CompactSearchServiceV1` still cannot beat
  `direct_ctree_gpu_latent` by a 2x-class margin at both sim16 and sim32 after
  scalar materialization and replay materialization stay out of the measured hot
  path. In that case keep the tests, but stop treating this slab as the 5x/10x
  answer.

## Final Read

The slab is not the breakthrough by itself. It is the smallest patch that makes
the next breakthrough measurable: one profile-only owner around compact
state/root/search/action/replay rows, inserted exactly before scalar LightZero
timestep materialization, with the existing contracts still doing the safety
work.
