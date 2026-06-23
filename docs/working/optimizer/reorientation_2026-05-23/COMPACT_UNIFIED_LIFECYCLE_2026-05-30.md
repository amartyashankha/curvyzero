# Compact Unified Lifecycle Smoke

Date: 2026-05-30

Status: local lifecycle artifact passed; base report remains speed-blocked, and
the later threshold H100 speed-row refresh closes that gate separately.
Role: bind compact-owned lifecycle evidence to one LightZero-shaped checkpoint
candidate without claiming Coach speed.
Authority: artifact-specific record; `CURRENT_STATE.md` and `goal.md` remain
the current operating truth.

## Artifact

```text
artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/current_chain_eval_gif_tournament_smoke_report.json
artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json
artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/iteration_0.pth.tar.current_chain_eval_gif_tournament_load.evidence.json
```

Source normal-death profile:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-normal-death-compact-owned-profile-20260530/row_001_result.json
```

## What Passed

The smoke binds these gates to one LightZero-shaped compact checkpoint/report
chain:

- trainer entrypoint;
- checkpoint save/load;
- resume metadata;
- reward/RND contract;
- normal death/terminal contract;
- policy-refresh handoff;
- training-metrics lineage;
- current-chain eval/GIF/tournament load.

Headline:

```text
ok=true
lifecycle_gates_complete=true
missing_required_gates=[coach_speed_row]
missing_required_evidence=[coach_speed_row]
promotion_eligible=false
promotion_claim=false
```

## Non-Claims

This is local lifecycle evidence only. It is not:

- a Coach speed row;
- stock `train_muzero` execution;
- stock resume proof;
- live-run safety proof;
- rating or promotion-quality proof.

Caveat: the metrics/search provenance uses a tiny local compact learner update
and local resident/search stamps. That is enough for the lifecycle smoke, but
not enough by itself for the required Coach speed-row gate.

## Speed-Row Follow-Up

The structured speed-row evidence contract now exists:

```text
docs/working/optimizer/reorientation_2026-05-23/COMPACT_COACH_SPEED_ROW_EVIDENCE_2026-05-30.md
```

The closing evidence must validate under
`curvyzero_compact_coach_speed_row_evidence/v1`; a profile-only row or a bare
`compact_coach_speed_row:` string is not enough.

Update: a local CPU producer smoke now proves the sibling speed-row artifact
shape and hardened evidence checks:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-smoke-20260530/compact_coach_speed_row_smoke_report.json
```

It is not the final speed gate. The accepted H100 threshold row is now:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-h100-b1024a16-threshold-20260530/compact_coach_speed_row_modal_report.json
```

The compatibility refresh that attaches it is:

```text
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530/compatibility_report.json
```

That refresh sets `coach_speed_row=true` and computes
`promotion_eligible=true`, while keeping `promotion_claim=false`.
