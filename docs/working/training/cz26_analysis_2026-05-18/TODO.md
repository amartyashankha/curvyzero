# TODO

## Setup

- [x] Create current analysis workspace.
- [x] Merge subagent tooling/data/metric/meta findings.
- [x] Decide the canonical local artifact directory for generated analysis.
- [x] Build or select a row-level analysis tool.
- [x] Record exact commands before running heavyweight Modal pulls.

## Data Pulls

- [x] Locate local cz26 run-status/eval-status snapshots.
- [x] If missing, pull compact cz26 run status for the 136 manifest. Pulled:
  `artifacts/local/cz26_analysis_2026-05-18/cz26_eval_status_latest.json`.
- [x] Pull or locate cz26 eval curves with `mean_survival` and
  `mean_training_reward`.
- [x] Pull or locate current cz26 tournament ratings and active leaderboard.
- [ ] Pull or locate battle summaries for tournament duration and games/opponent
  counts.
- [ ] Pull trainer/export consumption proof only if needed for interpretation.

## Tooling

- [ ] Verify `scripts/analyze_curvytron_eval_curves.py` works on cz26 snapshots.
- [ ] Check whether `curvytron_live_loop_control.py` can export compact JSON
  enough for analysis.
- [x] Build `scripts/analyze_curvytron_cz26_grid.py` or equivalent if no tool
  joins manifest + eval + tournament cleanly.
- [x] Tool output should be JSON and Markdown summary, not only prose.
- [x] Smoke-run the analyzer on manifest-only data.
- [x] Add second-pass deep report generator for reward components, exact
  horizons, tournament exposure, action collapse, and per-run rows.
- [ ] Add a small fixture/test if a new analysis tool is introduced.

## Analysis Passes

- [x] Pass 0: completeness and missingness by grid/axis.
- [x] Pass 1a: survival projection, first read.
- [x] Pass 1b: matched pairwise contrasts, first read.
- [x] Pass 1c: completed-only and 300k sensitivity contrast tables.
- [x] Pass 2a: reward projection, first read.
- [x] Pass 2b: reward component breakdown if available.
- [x] Pass 3a: first tournament rank/top-band read from latest completed rating.
- [x] Pass 3b: tournament exposure/games/opponents validation.
- [x] Pass 4a: Grid A one-axis projections, first read.
- [x] Pass 4b: Grid A matched contrasts, first read.
- [ ] Pass 5: Grid A interaction slices.
- [x] Pass 6a: Grid B slot recipe and pure-control first read.
- [x] Pass 6b: Grid B matched contrasts, first read.
- [x] Pass 6c: Grid B low-coverage sensitivity table.
- [ ] Pass 7: tournament duration over time and whether the pool strengthened.
- [x] Pass 8: synthesize what is real, confounded, or still unknown.
- [ ] Pass 9: interaction slices for the most important candidates after
  reviewing the current synthesis.

## Reporting

- [x] Keep `FINDINGS.md` and `DEEP_ANALYSIS.md` updated after completed pass.
- [x] Add r18fresh-style focused notes:
  - `MATCHED_GRID_ANALYSIS.md`;
  - `SLOT_RECIPE_DEEP_DIVE.md`;
  - `TREND_ANALYSIS.md`;
  - `REWARD_BREAKDOWN_ANALYSIS.md`;
  - `GRID_REFINEMENT_REVIEW.md`.
- [ ] Present short user-facing summaries in plain language.
- [ ] Preserve exact commands and artifact paths.
- [ ] Do not collapse reward, survival, and tournament into one fake score.
