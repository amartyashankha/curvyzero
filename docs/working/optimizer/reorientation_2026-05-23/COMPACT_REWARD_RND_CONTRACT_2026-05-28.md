# Compact Reward/RND Contract

Date: 2026-05-28

Status: landed as a fail-closed compact-owned lifecycle gate.

## Decision

The first compact-owned trainer candidate is explicitly no-RND.

```text
reward_variant=survival_plus_bonus_no_outcome
reward_target_effect=extrinsic_reward_only
exploration_bonus_mode=none
rnd_enabled=false
```

This does not say RND is bad or unnecessary. It says compact-owned training does
not yet own the RND update, reward-target mutation, checkpoint, or restore
lifecycle. Stock reward-model/RND plumbing exists elsewhere, and compact latest
frame extraction is tested, but that is not enough for a compact-owned promotion
claim.

## Implementation

Added:

```text
src/curvyzero/training/compact_reward_rnd_contract.py
tests/test_compact_reward_rnd_contract.py
```

Threaded into:

```text
src/curvyzero/training/compact_owned_trainer.py
src/curvyzero/training/compact_trainer_checkpoint.py
tests/test_compact_owned_trainer.py
tests/test_compact_trainer_checkpoint.py
tests/test_compact_coach_compatibility.py
```

The compact trainer/checkpoint metadata now carries
`curvyzero_compact_reward_rnd_contract/v1`, including reward schema id/hash,
reward policy, exploration-bonus config hash, and RND non-claims. The builder
rejects RND-enabled exploration configs and rejects `rnd_state_dict` until an
RND-enabled compact-owned contract exists.

## Compatibility Refresh

Durable local report:

```text
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-reward-rnd-contract-20260528/compatibility_report.json
```

Passed required gates:

- trainer entrypoint;
- checkpoint save/load;
- resume metadata;
- eval/GIF/tournament load;
- reward/RND contract.

Still missing:

- death/terminal contract;
- policy refresh handoff;
- training metrics lineage.

Next selected gate:

```text
death_terminal_contract
```

## Validation

```text
uv run ruff check src/curvyzero/training/compact_reward_rnd_contract.py src/curvyzero/training/compact_coach_compatibility.py src/curvyzero/training/compact_owned_loop.py src/curvyzero/training/compact_owned_trainer.py src/curvyzero/training/compact_trainer_checkpoint.py src/curvyzero/training/compact_stock_checkpoint_export.py tests/test_compact_reward_rnd_contract.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py
uv run pytest tests/test_compact_reward_rnd_contract.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py tests/test_compact_owned_trainer.py tests/test_compact_trainer_checkpoint.py tests/test_compact_stock_checkpoint_export.py -q
```

Result:

```text
ruff passed
40 passed, 2 warnings
```

## Non-Claims

- no positive-RND claim;
- no intrinsic-reward target mutation claim;
- no RND checkpoint/resume claim;
- no stock `train_muzero` call;
- no promotion claim;
- no Coach speed claim.
