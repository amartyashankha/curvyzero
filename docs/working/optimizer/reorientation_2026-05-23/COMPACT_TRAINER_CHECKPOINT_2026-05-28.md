# Compact Trainer Checkpoint Envelope

Date: 2026-05-28

Status: local checkpoint/resume envelope implemented.

## Decision

The compact-owned route now has a local trainer checkpoint envelope. It is
compact-native and resume-facing. It is not a stock LightZero checkpoint and it
does not claim eval/tournament loadability yet.

New code:

```text
src/curvyzero/training/compact_owned_trainer.py
src/curvyzero/training/compact_trainer_checkpoint.py
```

New tests:

```text
tests/test_compact_owned_trainer.py
tests/test_compact_trainer_checkpoint.py
```

## What The Envelope Saves

Checkpoint schema:

```text
curvyzero_compact_trainer_checkpoint/v1
```

Trainer schema:

```text
curvyzero_compact_owned_trainer/v1
```

The checkpoint contains:

- model state dict;
- optimizer state dict;
- compact replay-store state object, not only replay metadata;
- resume state: trainer id, train step, learner update count, sample batch
  count, policy/model refs, policy source, loop counters;
- compact-owned loop runtime state, including `_previous_step`, so the first
  post-resume `record_step` does not warmup-drop a transition;
- trainer config and metrics;
- optional scheduler/RND state fields;
- fail-closed compatibility metadata.

The compact-owned trainer wrapper can:

- run the real `CompactMuZeroLearnerEdgeV1` on a sample batch;
- wrap `CompactOwnedLoopV1.record_step(...)`;
- reject stale/duplicate `record_index` values after resume;
- advance policy/model version refs when learner updates happen;
- update the owned loop's policy version ref after learner updates;
- save checkpoints carrying loop runtime state and replay contents.

## Non-Claims

The checkpoint explicitly records:

```text
calls_train_muzero=false
touches_live_runs=false
promotion_claim=false
stock_eval_tournament_loadable=false
stock_eval_tournament_load_status=adapter_missing
compact_eval_adapter_required=true
```

This is not a `.pth.tar` mapping that the current stock eval/tournament loader
can consume as-is. That gap is now handled by a separate derived export artifact
documented in `COMPACT_STOCK_EXPORT_2026-05-28.md`; the compact-native
checkpoint remains the resume artifact.

## Validation

```text
uv run ruff check src/curvyzero/training/compact_trainer_checkpoint.py src/curvyzero/training/compact_owned_trainer.py src/curvyzero/training/compact_owned_loop.py tests/test_compact_trainer_checkpoint.py tests/test_compact_owned_trainer.py
uv run pytest tests/test_compact_trainer_checkpoint.py tests/test_compact_owned_trainer.py -q
```

Result:

```text
ruff passed
7 passed, 2 warnings
```

## Read

This closes the first checkpoint/resume-facing local gate. The compact-owned
candidate now has an artifact shape that preserves model, optimizer, replay,
loop runtime, counters, and lineage.

The verified-export attachment policy for the derived export has now landed as a
sibling evidence bundle. The verifier harness exists, a stock-shaped compact
export strict-load smoke passed, the tournament policy-loader path consumed the
same export plus sidecar, and the bundle hashes those reports together without
mutating the base export. The first one-game/GIF plus standalone eval smoke also
passed for that bundle, and the reward/RND compatibility refresh selected
`death_terminal_contract` as the next missing lifecycle gate. Later 2026-05-30
artifacts closed death/terminal, policy-refresh, metrics, current-chain
eval/GIF, and unified lifecycle locally. Assignment refresh and live Coach
canaries remain blocked by explicit `coach_speed_row` evidence and promotion
review, not by raw throughput.

## 2026-05-30 Profile-Result Normal-Death Checkpoint Smoke

`scripts/build_compact_owned_normal_death_checkpoint_smoke.py` can now consume a
real hybrid profile result via `--profile-result`. When the JSON contains a
runner envelope, it uses the nested `compact` payload; otherwise it accepts the
payload directly.

Durable refresh:

```text
artifacts/local/curvytron_compact_owned_normal_death_checkpoint_results/optimizer-compact-owned-normal-death-checkpoint-from-profile-20260530/normal_death_checkpoint_smoke_report.json
```

Source profile result:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-normal-death-compact-owned-profile-20260530/row_001_result.json
```

The smoke saves and loads a compact-owned checkpoint, preserves the
profile-result evidence ref, and reports:

```text
ok=true
death_terminal_contract=true
compact_coach_compatibility_gate_death_terminal_contract=true
compact_coach_compatibility_promotion_eligible=false
```

Missing required gates for that exact report chain remain
`eval_gif_tournament_load`, `policy_refresh_handoff`, and
`training_metrics_lineage`. This is a checkpoint compatibility proof, not a
stock `train_muzero` call, not live-run safety, and not a speed claim.

## 2026-05-30 Metrics-Lineage Attachment

The checkpoint format now has a fail-closed structured metrics-lineage
attachment:

```text
curvyzero_compact_training_metrics_lineage/v1
```

`build_compact_trainer_checkpoint_v1(...)` can flip
`training_metrics_lineage=true` only when the contract validates finite compact
MuZero learner losses, trainer resume counters, compact replay-store metadata,
compact-owned loop counters, search provenance, matching policy/model refs,
evidence refs, and non-claims. Arbitrary metrics still only record partial
metrics-present evidence. This local contract does not retroactively change the
normal-death checkpoint smoke above; that smoke remains exact-chain evidence for
death/terminal only.

## 2026-05-30 Policy-Refresh Handoff Attachment

The checkpoint format now also has a fail-closed structured policy-refresh
attachment:

```text
curvyzero_compact_policy_refresh_handoff/v1
```

`build_compact_trainer_checkpoint_v1(...)` can flip
`policy_refresh_handoff=true` only when the contract validates learner/search
model digest equality, distinct learner/search model object ids, refreshed
search-worker state, cache clearing, matching policy/model/source refs,
root/action/replay/sample row stamps, evidence refs, and non-claims.

`CompactOwnedTrainerV1.checkpoint(...)` and `save_checkpoint(...)` now accept
the contract and preserve it through the public compact-owned checkpoint path.
`CompactRolloutSlab.step(...)` stamps root batches from refreshed search-worker
state, so root metadata is no longer a test-only manual placeholder.

This local contract does not retroactively change the normal-death checkpoint
smoke above. That smoke predates both local metrics and policy-refresh
attachments and remains exact-chain death/terminal evidence only.
