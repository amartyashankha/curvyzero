# Compact Policy-Refresh Handoff Contract

Date: 2026-05-30

Status: local checkpoint-level contract landed. Not promotion. Not Coach speed.

## Purpose

The compact-owned trainer must prove that learner-updated model state reaches a
distinct search worker before the search worker produces roots, actions, replay
rows, learner samples, and checkpoint resume metadata.

The contract is:

```text
curvyzero_compact_policy_refresh_handoff/v1
```

## What Landed

- `src/curvyzero/training/compact_policy_refresh_handoff.py`
  validates learner/search model digests, policy/model refs, learner update
  count, distinct model object ids, search-worker refresh count, cache clearing,
  root/action/replay/sample row stamps, evidence refs, and non-claims.
- `CompactTorchSearchServiceV1.refresh_model_state(...)` loads learner weights
  into the search-worker model, verifies the expected digest, clears compiled
  helper/model caches, and exposes
  `curvyzero_compact_policy_refresh_search_worker_state/v1`.
- `CompactTorchSearchServiceV1` stamps search/action/replay payload metadata
  after refresh.
- `CompactRolloutSlab.step(...)` now stamps root batches from a validated
  refreshed search-worker state instead of leaving roots unstamped.
- `compact_policy_row_bridge.py` propagates refresh metadata into host and
  device replay-index rows.
- `_CompactReplayRingV1.sample(...)` aggregates refresh metadata into
  learner-facing sample metadata.
- `compact_trainer_checkpoint.py` attaches the validated handoff contract,
  flips only `policy_refresh_handoff=true`, and adds Coach evidence.
- `CompactOwnedTrainerV1.checkpoint(...)` and `save_checkpoint(...)` can carry
  the contract through the public compact-owned checkpoint path.

## Required Row Stamp

Every positive root/action/replay/sample proof must agree on:

```text
policy_version_ref
model_version_ref
policy_source
compact_policy_refresh_handoff_state_schema_id
compact_policy_refresh_model_state_digest
compact_policy_refresh_learner_update_count
compact_policy_refresh_search_worker_refreshed=true
compact_policy_refresh_count
```

## What It Proves

- A learner update changed model state.
- A distinct search-worker model loaded exactly that state digest.
- Search-worker caches were cleared after refresh.
- Root/action/replay/sample metadata carry matching policy/model/source refs,
  digest, update count, refresh marker, and refresh count.
- A compact checkpoint can carry the handoff contract and set
  `compact_coach_compatibility_gate_policy_refresh_handoff=true`.

## What It Does Not Prove

- No stock `train_muzero` call.
- No live-run or Coach resume claim.
- No training speed claim.
- No promotion claim.
- No distributed/async staleness bound across multiple remote workers.
- This policy-refresh artifact itself does not include refreshed current-chain
  eval/GIF/tournament-load evidence. Later unified lifecycle/current-chain
  reports supply that separate evidence.

## Validation

```text
uv run ruff check src/curvyzero/training/compact_rollout_slab.py tests/test_compact_search_replay_contract.py src/curvyzero/training/compact_policy_refresh_handoff.py src/curvyzero/training/compact_torch_search_service.py src/curvyzero/training/compact_policy_row_bridge.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/compact_trainer_checkpoint.py src/curvyzero/training/compact_owned_trainer.py tests/test_compact_policy_refresh_handoff.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_torch_search_service.py tests/test_compact_coach_compatibility.py
uv run pytest tests/test_compact_policy_refresh_handoff.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_torch_search_service.py tests/test_compact_search_replay_contract.py tests/test_compact_coach_compatibility.py tests/test_source_state_hybrid_observation_profile.py -q
```

Result:

```text
ruff passed
182 passed, 2 warnings
```

## Remaining Promotion Blocker

The local current-chain evidence contract now exists separately in
`COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_2026-05-30.md`.

The active remaining blocker is a real current-chain eval/GIF/tournament-load
artifact: a LightZero-shaped compact-owned checkpoint chain must refresh stock
export, strict-load, tournament-loader, one-game GIF, standalone eval, and the
current-chain evidence bundle.
