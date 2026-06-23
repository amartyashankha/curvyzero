# CZ26 Analysis Workspace - 2026-05-18

This folder is the current analysis workspace for `cz26-full-20260517a`.

Start here when analyzing the batch. Older `r18fresh` docs are evidence, but
this folder is the live workspace for turning the 136-run `cz26` batch into
clear findings.

## Current Question

What did `cz26-full-20260517a` teach us about:

- reward outcome strength;
- action noise;
- leaderboard-opponent immortality;
- opponent slot recipes;
- pure opponent controls;
- tournament feedback quality?

## Source Context

The rationale for launching this batch is summarized in:

```text
docs/working/training/r18fresh_postmortem_2026-05-16/CZ26_BATCH_RATIONALE_REORIENTATION_2026-05-18.md
```

The active manifest is:

```text
artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.json
```

Batch shape:

```text
Grid A: 96 rows = 4 recipes * 4 reward alphas * 3 noise settings * 2 immortality settings
Grid B: 40 rows = 10 recipes * out50 * 2 noise settings * 2 immortality settings
Total: 136 rows
```

## Workspace Files

- `ORCHESTRATION.md`: tactical lane plan, delegation map, and current state.
- `TODO.md`: concrete task list.
- `ANALYSIS_PLAN.md`: metric definitions and projection plan.
- `DATA_SOURCES.md`: local and Modal data inventory.
- `TOOLING_INVENTORY.md`: existing tools and tooling gaps.
- `GLOSSARY.md`: plain-language definitions for all shorthand.
- `DEPTH_STANDARD.md`: checklist for analysis quality based on the stronger
  prior r18fresh analysis.
- `DEEP_ANALYSIS.md`: current plain-language synthesis from the generated
  second-pass report.
- `MATCHED_GRID_ANALYSIS.md`: matched one-knob comparisons and exact-horizon
  readouts.
- `SLOT_RECIPE_DEEP_DIVE.md`: opponent recipe analysis.
- `TREND_ANALYSIS.md`: first/best/latest and retention analysis.
- `REWARD_BREAKDOWN_ANALYSIS.md`: reward component and outcome-residual read.
- `GRID_REFINEMENT_REVIEW.md`: current implications and next questions.
- `FINDINGS.md`: durable findings as they become real.

Generated local outputs belong in:

```text
artifacts/local/cz26_analysis_2026-05-18/
```

Current core generated reports:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_joined_analysis.md
artifacts/local/cz26_analysis_2026-05-18/cz26_matched_contrasts.md
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
```

## Operating Rules

- Keep reward, survival, and tournament performance separate.
- Do matched comparisons before latest-only comparisons.
- Collapse one axis at a time, then inspect interactions.
- Treat Grid B pure controls as controls inside the 136-run batch.
- Do not call an operational proof a learning-quality proof.
- Update this workspace before and after every substantial analysis pass.
