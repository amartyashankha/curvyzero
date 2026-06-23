# Compact Training Metrics Lineage Contract

Date: 2026-05-30

Status: local checkpoint-level contract implemented.

## Decision

The compact-owned route now has a fail-closed metrics-lineage contract:

```text
curvyzero_compact_training_metrics_lineage/v1
```

This is not a speed claim and not promotion evidence by itself. It proves that
a compact-owned checkpoint can carry structured training metrics only when the
metrics are bound to learner telemetry, trainer resume counters, compact replay
state, compact-owned loop counters, search provenance, policy/model refs, and
non-claims.

## What It Proves

The contract requires:

- finite compact MuZero learner loss, policy loss, value loss, reward loss, and
  grad norm;
- positive learner sample rows and train steps;
- resident/device replay sample telemetry;
- positive trainer `train_step`, `learner_update_count`, and
  `sample_batch_count`;
- compact replay-store metadata owned by the compact-owned loop;
- replay policy/model refs matching the checkpoint resume refs;
- loop sample/learner counters with positive sample rows and updates;
- search provenance: search impl, root-batch schema, search-result schema,
  replay-payload schema, active roots, simulations, and replay-payload digest;
- explicit `calls_train_muzero=false`, `touches_live_runs=false`,
  `promotion_claim=false`, and `training_speed_claim=false`;
- non-empty evidence refs.

Arbitrary nonempty metrics remain partial evidence only. A metrics dict such as
`{"loss": 1.25}` does not pass `training_metrics_lineage`.

## Code

```text
src/curvyzero/training/compact_training_metrics_lineage.py
src/curvyzero/training/compact_trainer_checkpoint.py
src/curvyzero/training/compact_owned_trainer.py
src/curvyzero/training/compact_policy_row_bridge.py
src/curvyzero/training/compact_search_service.py
src/curvyzero/training/compact_torch_search_service.py
src/curvyzero/training/source_state_hybrid_observation_profile.py
```

The checkpoint builder accepts a validated lineage contract and flips only:

```text
training_metrics_lineage=true
compact_coach_compatibility_gate_training_metrics_lineage=true
```

It still leaves `policy_refresh_handoff=false`, promotion false, no Coach speed
claim, no stock `train_muzero` call, and no live-run touch.

## Validation

```text
uv run ruff check src/curvyzero/training/compact_training_metrics_lineage.py src/curvyzero/training/compact_trainer_checkpoint.py src/curvyzero/training/compact_owned_trainer.py src/curvyzero/training/compact_policy_row_bridge.py src/curvyzero/training/compact_search_service.py src/curvyzero/training/compact_torch_search_service.py src/curvyzero/training/source_state_hybrid_observation_profile.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py
uv run pytest tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py -q
uv run pytest tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_search_replay_contract.py tests/test_source_state_hybrid_observation_profile.py -q
```

Result:

```text
ruff passed
16 passed, 2 warnings
138 passed, 2 warnings
```

## Remaining Gates

- Policy-refresh handoff is still open. Existing refs are not proof that a
  distinct search worker consumed refreshed weights.
- The exact normal-death checkpoint chain still needs either a refreshed
  eval/GIF/tournament-load artifact or a structured evidence bundle binding
  older eval evidence to the current checkpoint.
- Promotion remains false.
