# r18fresh Postmortem Workspace

Created: 2026-05-16

This directory is the current workspace for the r18fresh postmortem and
next-batch planning. It is a case study and evidence folder. The general
trainer/checkpoint/tournament feedback-loop contracts now live in
`docs/working/training/curvytron_feedback_loop`.

The old planning folder at `docs/working/training/leaderboard_to_training_2026-05-13` is archive/audit trail material. Treat it as historical context, not the live source of truth, except for explicitly linked current-pipeline docs.

## Current Docs

- `OPERATING_PATTERNS.md`: Durable work habits and proof standards.
- `ORCHESTRATION.md`: Current lanes, owners, follow-ups, and next actions.
- `ASSET_REGISTRY.md`: Current apps, arena/rating ids, run prefixes, and
  preserve/cleanup status.
- `CURRENT_LAUNCH_DEFAULTS.md`: Current next-batch launch defaults. This is the
  quick guardrail against stale H100/batch32/body-circles settings.
- `CZ26_CURRENT_TRUTH_2026-05-17.md`: Short live cz26 operator truth:
  current ids, validated path, remaining gates, and the note that
  `round-*` names are tournament game-batch ids, not training rounds.
- `WHAT_THE_HELL_IS_GOING_ON_HANDOFF_2026-05-17.md`: Plain-language handoff for
  the current cz26 stall/recovery state, current live IDs, proven facts,
  missing proof, and next commands.
- `TOURNAMENT_STABILITY_HANDOFF_2026-05-17.md`: Deeper handoff for why skipped
  or obsolete tournament batches can still leave live Modal workers behind, plus
  permanent fix options.
- `NAMING_CONVENTIONS.md`: Current short names for Grid A/Grid B/canary rows,
  reward/noise/immortality tags, and slot recipe codes. Use this before
  generating new manifests.
- `TASK_BOARD.md`: Short tactical lane table with owners, status, next actions,
  and artifact predicates.
- `ANALYSIS_METHOD.md`: Fair comparison rules for r18fresh.
- `MATCHED_GRID_ANALYSIS.md`: Matched 0..240k eval AUC, best, retention, and
  per-run table.
- `TREND_ANALYSIS.md`: High-level and granular trend readout across survival,
  own reward, tournament signal, and checkpoint throughput.
- `DETAILED_RUN_STOCKTAKE.md`: Per-run survival/tournament stocktake with best
  tournament checkpoint, latest checkpoint rank, and knob-level readout.
- `REWARD_BREAKDOWN_ANALYSIS.md`: Own-reward and reward-component analysis by
  variant.
- `SLOT_RECIPE_DEEP_DIVE.md`: Exact historical slot recipes, matched slot-only
  comparisons, and next slot-axis candidates.
- `REWARD_AXIS_NEXT.md`: Next reward-axis clarification: outcome coefficient
  between no-outcome and current plus-outcome.
- `H100_L4_OPTIMIZER_HANDOFF.md`: Current hardware/render-path handoff for the
  optimizer, including what r18fresh actually ran and the fresh L4 vs H100
  current-surface profile.
- `SIGNALS.md`: Signal inventory and observability map.
- `OBSERVABILITY_PLAN.md`: Required lineage tables, game-batch cards, export
  ledgers, operator readouts, and next-batch proof gates.
- `NEXT_BATCH_SEEDING.md`: How to preserve top policies for the next batch.
- `NEXT_BATCH_DESIGN.md`: Current next-batch matrix worldview and critiques.
- `GRID_REFINEMENT_REVIEW.md`: Latest Grid A/Grid B recommendation after
  slot-recipe critique, including pure blank/wall/rank1 controls in Grid B.
- `TOP10_RAW_REFS_auto-r000032-g22-555c999b.txt`: Exact raw top-10 refs from
  the public/trainer snapshot read on 2026-05-16.
- `TESTING_AND_GAPS.md`: Current local coverage, fast test commands, and gaps.
- `TODO.md`: Active work queue and decisions to make.
- `EXPERIMENT_LOG.md`: Chronological notes from analysis and operator decisions.
- `FINDINGS.md`: Evidence-backed observations from per-run and tournament review.

## Ground Rules

- Do not physically move old planning files as part of this archive reset.
- Prefer concise entries with dates, run IDs, commands, and artifact paths.
- Separate evidence from interpretation; promote settled conclusions into `FINDINGS.md`.
- If a new claim changes how we would launch the next batch, update `ORCHESTRATION.md`, `TODO.md`, and either `FINDINGS.md` or `EXPERIMENT_LOG.md` in the same pass.
- Runtime truth comes from deployed status/control tooling and local test
  results. Modal Volume artifacts are backing evidence for debugging, not the
  first operator interface.
