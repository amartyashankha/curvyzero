# Coach Compatibility Artifact Selection

Date: 2026-05-28

Status: selected, locally encoded, refreshed through unified local lifecycle,
and refreshed again after accepted threshold H100 speed-row evidence.

Latest read: the local lifecycle gate set is complete on one LightZero-shaped
compact checkpoint candidate, and the accepted threshold H100 non-profile
`curvyzero_compact_coach_speed_row_evidence/v1` artifact now exists with
loaded-checkpoint model identity:
`artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-h100-b1024a16-threshold-20260530/compact_coach_speed_row_modal_report.json`.
The speed-row compatibility refresh is:
`artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530/compatibility_report.json`.
It sets `coach_speed_row=true`, has no missing current local compatibility
gates/evidence, and computes `promotion_eligible=true` under local
compatibility metadata. A bare
`compact_coach_speed_row:` string, profile-only row, or support-only model
identity still does not close the gate. The next blocker is not compatibility
metadata. OPT-060 now supplies the hash-bound readiness bundle and OPT-061
supplies the original sufficiency decision. OPT-068 later supplies the larger
32x2048 matched-quality packet, but the refreshed sufficiency review accepts it
only for `larger_32x2048_matched_quality_manual_review_not_promotion`;
promotion still requires a separate promotion-policy/code decision.

## Decision

Historical decision after the matched denominator pair: the next artifact was
not another speed row. It was a fail-closed Coach-compatibility attestation for
the compact-owned candidate. As of the later unified lifecycle smoke, the next
artifact has changed: capture explicit `coach_speed_row` evidence for that
candidate without turning lifecycle evidence into a speed claim.

Selected route:

```text
compact-owned trainer candidate
calls_train_muzero=false until that is literally false no longer
promotion_claim=false; the post-compatibility readiness bundle and sufficiency
review now exist, but they are manual-review/no-promotion artifacts
```

Rejected as the immediate route:

```text
stock train_muzero bridge
```

Reason: entering stock `train_muzero` honestly would require replacing or
bridging LightZero collector, collect policy/search, replay buffer/sample, and
learner/checkpoint plumbing without rebuilding the scalar object hot path we
are trying to avoid. That may still happen later, but doing it now would invite
another half-compact, half-stock comparison with muddy speed currency.

## New Local Contract

Added:

```text
src/curvyzero/training/compact_coach_compatibility.py
tests/test_compact_coach_compatibility.py
```

The contract schema is:

```text
curvyzero_compact_coach_compatibility/v1
```

The current compact-owned profile loop now carries this report through
`CompactOwnedLoopV1` metadata. It records useful support gates, but keeps
promotion closed.

Current support gates recorded as present:

- matched denominator;
- split profile loop entrypoint;
- durable replay-store snapshot;
- public stock sample diff;
- terminal N-step targets;
- compact MuZero learner edge.

Initial profile-only required promotion gates recorded as missing:

- non-profile trainer entrypoint;
- checkpoint save/load;
- resume metadata;
- eval/GIF/tournament load;
- reward/RND contract;
- death/terminal contract for training settings;
- policy refresh handoff;
- training metrics lineage.

The report rejects `promotion_claim=true` when local compatibility is incomplete
and reports the missing gate/evidence. After the accepted H100 speed-row refresh,
local compatibility can be complete and `promotion_eligible=true`; that still is
not a promotion claim. The generic builder now rejects `promotion_claim=true`
with a `post_compatibility_promotion_readiness_required` blocker until the
separate readiness contract exists; direct `CompactCoachCompatibilityReportV1`
construction with `promotion_claim=true` fails too. It also rejects
`calls_train_muzero=true` on the compact-owned route.

The readiness decision is recorded in:

```text
docs/working/optimizer/reorientation_2026-05-23/COMPACT_PROMOTION_READINESS_GATE_2026-05-30.md
```

## Validation

```text
uv run ruff check src/curvyzero/training/compact_coach_compatibility.py src/curvyzero/training/compact_owned_loop.py tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py
uv run pytest tests/test_compact_coach_compatibility.py tests/test_compact_owned_loop.py -q
```

Result:

```text
ruff passed
10 passed, 2 warnings
```

Current promotion-readiness guard validation:

```text
uv run ruff check src/curvyzero/training/compact_coach_compatibility.py tests/test_compact_coach_compatibility.py
uv run pytest tests/test_compact_coach_compatibility.py -q
```

Result:

```text
ruff passed
18 passed
```

## Read

This closes the selection part of OPT-027: the compact path stays
compact-owned, and the next work is promotion semantics, not raw speed.

The first missing gate selected here was the checkpoint/resume-facing compact
trainer envelope. That has now landed, followed by a separate derived
stock-shaped export artifact. The compact-native checkpoint says
`adapter_missing`; the derived export says `strict_stock_model_load_not_run`.

The verifier harness for this proof now exists, a local stock-shaped compact
export strict-load smoke passed, the tournament policy-loader smoke passed for
the same durable export, and a sibling evidence bundle now binds the export,
sidecar, verifier report, and loader report without mutating the base export.
The first one-game/GIF plus standalone eval smoke also passed:

```text
exported model state under "model"
-> sidecar, verification report, loader report, and evidence-bundle refs
-> stock eval/tournament gameplay/render/eval path
-> first action/result succeeds
-> standalone eval strict-load/env-reset/policy-action succeeds
```

Historical reward/RND refreshed report:

```text
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-reward-rnd-contract-20260528/compatibility_report.json
```

Passed required gates:

- trainer entrypoint;
- checkpoint save/load;
- resume metadata;
- eval/GIF/tournament load;
- reward/RND contract.

At that snapshot, still missing:

- death/terminal contract;
- policy refresh handoff;
- training metrics lineage.

At that snapshot, the selected next gate was `death_terminal_contract`.

Death/terminal audit report:

```text
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-death-terminal-contract-20260528/compatibility_report.json
```

Read it as the historical blocker report, not the current passed gate. It records partial
profile/no-death terminal N-step support and keeps:

```text
compact_coach_compatibility_gate_death_terminal_contract=false
selected_next_missing_gate=death_terminal_contract
compact_death_terminal_contract_blocker=normal_collision_death_not_proven
```

Reward/RND stance:

- `reward_variant=survival_plus_bonus_no_outcome`;
- `reward_target_effect=extrinsic_reward_only`;
- `exploration_bonus_mode=none`;
- `rnd_enabled=false`;
- no positive-RND, intrinsic-reward, or RND checkpoint/resume claim.

Likely file targets:

- `src/curvyzero/training/compact_owned_trainer.py`;
- `src/curvyzero/training/compact_trainer_checkpoint.py`;
- `src/curvyzero/training/compact_reward_rnd_contract.py`;
- `src/curvyzero/training/compact_death_terminal_contract.py`;
- eventual durable replay-store extraction from
  `source_state_hybrid_observation_profile.py`;
- stock eval/tournament loader boundaries only as reference/load surfaces, not
  as hidden compact injection.

Likely tests:

- `tests/test_compact_owned_trainer.py`;
- `tests/test_compact_trainer_checkpoint.py`;
- `tests/test_compact_reward_rnd_contract.py`;
- terminal/autoreset/replay/learner target tests around the existing compact
  search/replay and compact MuZero suites.

## 2026-05-30 Normal-Death Refresh

The death/terminal gate now has a positive compact-owned evidence chain:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-normal-death-compact-owned-profile-20260530/row_001_result.json
artifacts/local/curvytron_compact_owned_normal_death_checkpoint_results/optimizer-compact-owned-normal-death-checkpoint-from-profile-20260530/normal_death_checkpoint_smoke_report.json
```

The checkpoint smoke consumes the actual profile result, so
`death_terminal_contract=true` and
`compact_coach_compatibility_gate_death_terminal_contract=true` are no longer
handbuilt metadata. Promotion remains false. The exact chain still reports
missing:

```text
eval_gif_tournament_load
policy_refresh_handoff
training_metrics_lineage
```

At the lane level, historical eval/GIF/tournament-load evidence exists for the
earlier bundled stock-shaped export, but current Coach evidence now requires a
`compact_current_chain_eval_gif_tournament_load:` ref. A local
checkpoint-level metrics-lineage contract now exists in
`COMPACT_TRAINING_METRICS_LINEAGE_2026-05-30.md`; the exact normal-death smoke
did not rerun with that contract attached.

## 2026-05-30 Policy-Refresh Refresh

The policy-refresh handoff gate now has local fail-closed checkpoint evidence:

```text
docs/working/optimizer/reorientation_2026-05-23/COMPACT_POLICY_REFRESH_HANDOFF_2026-05-30.md
```

The contract can flip `policy_refresh_handoff=true` only when a distinct search
worker refreshes to the learner model digest, clears caches, and stamps matching
policy/model refs, source, digest, learner update count, refresh marker, and
refresh count through roots, actions, replay rows, samples, and checkpoint
metadata. It preserves:

```text
calls_train_muzero=false
touches_live_runs=false
promotion_claim=false
training_speed_claim=false
```

The current-chain eval/GIF/tournament-load binding contract and real local
smoke now exist:

```text
docs/working/optimizer/reorientation_2026-05-23/COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_2026-05-30.md
artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-current-chain-eval-gif-tournament-smoke-20260530/current_chain_eval_gif_tournament_smoke_report.json
artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-current-chain-eval-gif-tournament-smoke-20260530/compatibility_report.json
```

It rejects arbitrary `eval_gif_tournament_load` evidence strings and
strict-load/loader-only overclaims. The smoke uses a LightZero-shaped compact
checkpoint, refreshes stock export, strict-load, tournament-loader,
one-game GIF/frames, standalone eval, and writes the current-chain evidence
bundle.

The later unified local lifecycle smoke now binds reward/RND, normal death,
policy refresh, training metrics, and current-chain eval/GIF to the same
LightZero-shaped compact checkpoint candidate:

```text
artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json
```

The remaining promotion-evidence blocker is no longer lifecycle unification.
The first local compact Coach speed-row producer smoke exists at:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-smoke-20260530/compact_coach_speed_row_smoke_report.json
```

Read it as support evidence for the speed-row producer and validator only. The
accepted H100 threshold row and compatibility refresh are now:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-h100-b1024a16-threshold-20260530/compact_coach_speed_row_modal_report.json
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530/compatibility_report.json
```

The refresh closes `coach_speed_row=true` structurally and computes
`promotion_eligible=true`, but it keeps `promotion_claim=false`. The unified
smoke plus threshold row prove local compatibility metadata, not stock resume,
live-run safety, rating quality, long-horizon learning quality, or actual
promotion.
