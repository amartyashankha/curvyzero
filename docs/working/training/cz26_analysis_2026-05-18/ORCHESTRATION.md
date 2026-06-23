# Orchestration

Last updated: 2026-05-18 11:28 EDT.

This is the tactical map for the `cz26-full-20260517a` analysis. Durable working
habits remain in:

```text
docs/working/training/r18fresh_postmortem_2026-05-16/OPERATING_PATTERNS.md
```

## Current Objective

Build a reliable analysis workbench before drawing conclusions, then run the
first compact outcome-data pull. The analysis must answer what happened across
reward, survival, and tournament performance, and then project the 136-run
tensor one axis at a time.

## Current Parallel Lanes

| Lane | Owner | Question | Expected Output | Integration Target |
| --- | --- | --- | --- | --- |
| tooling inventory | Raman | What scripts already exist for eval curves, run status, tournament status, and lineage? | Candidate tools and gaps. | `TOOLING_INVENTORY.md` |
| data inventory | Avicenna | What local/remote data exists for cz26 and r18fresh? | Source table and missing pulls. | `DATA_SOURCES.md` |
| metric design | Rawls | How should we project the batch across axes fairly? | Tables, plots, and comparison rules. | `ANALYSIS_PLAN.md` |
| meta-doc patterns | Leibniz | Which operating/orchestration docs should we reuse? | Patterns and source files. | `ORCHESTRATION.md` |

All four initial lanes have reported. Their findings are merged into the
workspace. Keep any follow-up delegation bounded to specific questions.

## Main-Thread Responsibilities

- Keep this workspace coherent.
- Build or choose the analysis tooling.
- Pull only the data needed for the next analysis pass.
- Merge subagent results into the docs.
- Run the first concrete summary only after data/tool contracts are clear.

## Current Tool State

First local analyzer:

```text
scripts/analyze_curvytron_cz26_grid.py
```

Smoke output:

```text
artifacts/local/cz26_analysis_2026-05-18/manifest_only_analysis.json
artifacts/local/cz26_analysis_2026-05-18/manifest_only_analysis.md
```

The compact Modal pulls have completed. Current snapshots:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_eval_status_latest.json
artifacts/local/cz26_analysis_2026-05-18/cz26_rating_latest.json
artifacts/local/cz26_analysis_2026-05-18/cz26_rating_progress.json
artifacts/local/cz26_analysis_2026-05-18/cz26_live_loop_status_latest.json
```

Current joined outputs:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_joined_analysis.json
artifacts/local/cz26_analysis_2026-05-18/cz26_joined_analysis.md
```

Second-pass analysis tooling now exists:

```text
scripts/analyze_curvytron_cz26_deep_report.py
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.json
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
```

It separates all-row common horizons, exact 300k sensitivity, learned-only
tournament metrics, reward residuals, tournament exposure, and action collapse.

## First Analysis Pass

The first pass should produce a row-level table with one row per training run:

```text
run_id
grid
reward_tag / reward_alpha
noise_tag
leaderboard_immortal_tag / probability
recipe_code / slot counts
train status
latest iteration
checkpoint count
survival first / matched endpoint / best / latest
reward first / matched endpoint / best / latest
best tournament rank
latest tournament rank
top10/top30/top100 presence
games / distinct opponents
missingness flags
```

Then derive axis projections from that row-level table.

## Open Questions

- Why do many rows peak and then regress?
- Which top tournament rows are real strength versus sparse exposure?
- Can reward drops be aligned with opponent-refresh/export generations?
- Does `n20` help matchup diversity even though `n10` is cleaner on fixed eval?
- Which recipe changes should be isolated next: blank mass, wall mass, rank
  concentration, or rank diversity?

## Parked On Purpose

Lineage/tournament-stitching tools are not the active lane. Other evidence says
the run outputs are trustworthy enough to start the experiment analysis. Return
to lineage only if the compact status/eval/rating snapshots show contradictions.
