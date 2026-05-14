# CurvyTron 212-Run Overview And Next-Run Recommendations

Date: 2026-05-13  
Status: consolidated read from survival curves, fair comparisons, and latest-212 leaderboard.

## Executive Read

The 212-run cohort gives two different kinds of evidence:

- **Survival eval**: did the policy learn to live longer?
- **Leaderboard / Elo**: does the checkpoint beat other checkpoints in actual
  head-to-head games?

Both matter. Survival tells us whether the objective is working. Leaderboard
tells us which checkpoints are useful champions/opponents. They do not always
select the same rows.

Current best read:

- Survival improved strongly. Median first-to-latest eval gain is about `+61.5`
  steps overall; survival-family rows gained about `+91.1`.
- Latest-212 leaderboard is real evidence: `212` checkpoints, all-pairs,
  `246,026` games, `provisional=false`, top rows with `211` distinct opponents.
- Leaderboard top ranks are survival-heavy, but the #1 checkpoint is a `mix2`
  `repH` row, not a pure survival row.
- Fast render looked better in latest snapshots mostly because fast rows were
  more mature. Same-checkpoint render comparisons are basically tied.
- `sim8 / batch32` should remain the default. `batch64` looks bad. `sim16`
  is not earning its cost. `collector64` is plausible only as a small probe.
- For survival stochasticity, `medium` is the best leaderboard center; `light`
  is the clean survival-gain point estimate; `heavy` has top-tail value but
  should not dominate.
- For mix repeats, `repH` has top-tail/leaderboard upside; `repM` is the safer
  anchor; keep both.

## Terms

| Term | Meaning |
| --- | --- |
| `survival` family | Stock train rows focused on survival-plus-bonus/no-outcome style learning. |
| `mix2` / `mix3` | Opponent-mixture training families. Names encode recipe, render, sim, collectors, batch, repeat, checkpoint cadence, copy/seed. |
| `rep0` | Repeat baseline / lowest repeat setting in the mix matrix. |
| `repM` | Medium repeat / medium stochasticity proxy in mix rows. Stable anchor. |
| `repH` | High repeat / high stochasticity proxy in mix rows. Higher-upside, higher-variance probe. |
| `fast` / `browser` | Render/input fidelity lanes. Browser is full-fidelity-ish; fast is cheaper approximation. |
| `s8/c32/b32` | `num_simulations=8`, `collector_env_num=32`, `batch_size=32`. |
| `leaderboard` | Tournament Elo/ranking from checkpoint-vs-checkpoint games. First death loses; simultaneous death/timeout draws. |

## Evidence Sources

Core inputs:

```text
artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json
artifacts/local/curvytron_status_snapshots/analysis_20260513e/row_projections.csv
artifacts/local/curvytron_status_snapshots/analysis_20260513e/fair_pair_comparisons.csv
artifacts/local/curvytron_status_snapshots/analysis_20260513e/matched_knob_summary_mix3.csv
```

Local tooling added for this pass:

```text
scripts/analyze_curvytron_212_recommendations.py
```

Generated outputs:

```text
artifacts/local/curvytron_status_snapshots/analysis_20260513e/clean_report/clean_report.md
artifacts/local/curvytron_status_snapshots/analysis_20260513e/clean_report/clean_report.json
artifacts/local/curvytron_status_snapshots/analysis_20260513e/clean_report/family_summary.csv
artifacts/local/curvytron_status_snapshots/analysis_20260513e/clean_report/checkpoint_matched_render_pairs.csv
artifacts/local/curvytron_status_snapshots/analysis_20260513e/clean_report/matched_compute_knobs.csv
```

Leaderboard outputs:

```text
artifacts/local/curvytron_status_snapshots/analysis_20260513e/leaderboard_latest212/rating_standings_all_enriched.csv
artifacts/local/curvytron_status_snapshots/analysis_20260513e/leaderboard_latest212/leaderboard_top50_knobs.csv
artifacts/local/curvytron_status_snapshots/analysis_20260513e/leaderboard_latest212/leaderboard_knob_summary.json
```

Known caveat: older fixed-path status can undercount checkpoints if LightZero
wrote into timestamped `lightzero_exp_*` directories. Avoid pruning or declaring
stalls from fixed-path status alone.

## Data Quality

| Family | Rows | Median latest iter | Main caveat |
| --- | ---: | ---: | --- |
| `survival` | 33 | about `240k` | Most mature; strongest survival evidence. |
| `mix2` | 52 | about `145k` | Noisier; some stale/low-maturity tails. |
| `mix3` | 126 | about `90k` | Broadest but least mature. |

Do not compare raw latest eval medians across families without accounting for
maturity. Survival rows had much more training time.

## Survival Learning Signal

Metric: `eval_latest_mean_steps - eval_first_mean_steps`.

| Family | N | % improved | Median gain | P25 gain | P75 gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| `survival` | 33 | 100.0% | +91.1 | +82.6 | +99.4 |
| `mix3` | 126 | 96.8% | +58.4 | +46.4 | +71.3 |
| `mix2` | 51 | 86.3% | +48.9 | +12.1 | +66.6 |
| all usable rows | 211 | 94.8% | +61.5 | +45.8 | +80.9 |

Curve-tooling cross-check:

```text
202 curves with valid first-to-latest delta
200 / 202 improved
median delta +62.9
132 / 197 had positive late slope
```

Interpretation:

- The survival objective is working in the broad sense.
- The median gain is large, not marginal.
- Not every row is still improving late; some flatten or wobble after early
  gains.
- Survival gains alone should not be used to choose champions.

## Leaderboard / Head-To-Head Signal

The latest-212 leaderboard is the best current head-to-head evidence.

```text
tournament = arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513-145153
rating_run = elo-latest212-allpairs-gpp11-gifs3-20260513-145153
rows = 212
provisional = false
pairs = 22,366
games = 246,026
games per top row = 2,321
distinct opponents per top row = 211
```

### Top-20 Composition

| Slice | Family mix | Render | Survival stochasticity | Mix settings |
| --- | --- | --- | --- | --- |
| Top 10 | 6 `survival`, 2 `mix2`, 2 `mix3` | 6 fast, 4 browser | 2 medium, 2 heavy, 2 light | all mix rows `sim8/batch32`; 3 `repH`, 1 `repM`; 1 `collector64` |
| Top 20 | 12 `survival`, 5 `mix2`, 3 `mix3` | 11 fast, 9 browser | 6 medium, 3 heavy, 2 light, 1 steady | 7/8 mix rows `sim8`; all 8 mix rows `batch32`; 4 `repH`, 4 `repM` |
| Top 50 | 26 `survival`, 13 `mix2`, 10 `mix3`, 1 other | 29 fast, 21 browser | 11 medium, 8 heavy, 4 light, 3 steady | 22/23 mix rows `sim8`; all 23 mix rows `batch32`; 10 `repH`, 8 `repM`, 5 `rep0` |

### Top Rows

| Rank | Family | Recipe / lane | Knobs | Rating | Win rate |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `mix2` | `r50-blank25-scr25` | fast, `repH`, `sim8/c32/b32`, `iteration_170000` | 2344.4 | 0.742 |
| 2 | `survival` | blank fast medium base | `iteration_270000` | 2271.6 | 0.720 |
| 3 | `survival` | blank fast heavy base | `iteration_255000` | 2223.6 | 0.708 |
| 4 | `mix3` | `r50-blank50` | fast, `repH`, `sim8/c64/b32`, `iteration_80000` | 2223.6 | 0.705 |
| 5 | `survival` | passive fast light base | `iteration_255000` | 2220.0 | 0.705 |

### Leaderboard Interpretation

If optimizing for actual head-to-head strength:

- `survival` family is the strongest overall family by leaderboard distribution.
  Median rank: `survival` 25, `mix2` 92.5, `mix3` 136.5.
- The best individual checkpoint is `mix2`, so do not ignore mix rows.
- Blank/no-op survival rows dominate the survival block, but passive rows can be
  strong opponent candidates.
- Survival `medium` is the best leaderboard center by median rank.
- Mix `repH` has high-upside top-tail evidence; `repM` remains the safer anchor.
- `sim8` and `batch32` are strongly supported by the top mix rows.
- `collector64` has plausible upside but should remain a probe.

## Render: Fast vs Browser

Latest-checkpoint comparison is confounded by maturity:

| Pair group | Pairs | Browser better | Browser worse | Median browser-fast eval | Median browser-fast iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| mix render pairs | 59 | 25 | 34 | -4.0 | -10k |
| survival strata | 14 | 5 | 9 | -6.1 | -26.25k |

Same-checkpoint mix comparison:

| Slice | Pairs | Browser better | Browser worse | Median browser-fast delta |
| --- | ---: | ---: | ---: | ---: |
| all common checkpoints | 58 | 27 | 28 | 0.0 |
| common >= 10k | 55 | 27 | 27 | 0.0 |
| common >= 50k | 47 | 23 | 23 | 0.0 |
| common >= 100k | 20 | 9 | 10 | -0.44 |
| common >= 150k | 5 | 4 | 1 | +3.38 |

Conclusion: do not claim fast is a better learning surface. It mostly had a
maturity advantage in this snapshot. If browser is now faster operationally,
use browser more freely, but preserve matched fast twins for key cells.

## Stochasticity And Repeat

### Survival Stochasticity

| Level | N | Median survival gain | Median leaderboard rank | Read |
| --- | ---: | ---: | ---: | --- |
| `medium` | 12 | +88.6 | 21 | Best leaderboard center; strong default. |
| `light` | 5 | +97.5 | 28 | Best survival-gain point estimate; small N. |
| `heavy` | 12 | +88.1 | 38 | Top-tail value; more variance/staleness risk. |
| `steady` | 4 | +82.6 | 26.5 | Useful control, not the main setting. |

Conclusion: use `medium` and `light` as the main survival settings. Keep
`heavy` as a meaningful but bounded stress/top-tail lane.

### Mix Repeat Proxy

| Repeat | N | Median survival gain | Median leaderboard rank | Read |
| --- | ---: | ---: | ---: | --- |
| `repM` | 76 | +53.3 | 120.5 | Stable anchor. |
| `repH` | 66 | +58.6 | 126.0 | High-upside; owns #1 row and many top mix rows. |
| `rep0` | 35 | +58.4 | 124.5 | Control/baseline; less compelling than `repM`/`repH`. |

Conclusion: there **is** signal for high repeat. It is top-heavy rather than
uniformly better. Use `repM` as anchor and give `repH` serious budget.

## Compute Knobs

Matched one-knob contrasts from mix3:

| Knob | Metric | Contrasts | Median high-low delta | Positive | Negative | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `batch64` vs `batch32` | latest eval | 20 | -17.1 | 5 | 15 | Bad |
| `batch64` vs `batch32` | latest iter | 20 | -10k | 4 | 15 | Slower |
| `sim16` vs `sim8` | latest eval | 20 | -12.8 | 7 | 12 | Weak negative |
| `sim16` vs `sim8` | latest iter | 20 | -10k | 3 | 13 | Slower |
| `collector64` vs `collector32` | latest eval | 20 | +1.9 | 11 | 9 | Tiny/mixed |
| `collector64` vs `collector32` | latest iter | 20 | +10k | 12 | 5 | Possible progress help |

Conclusion:

- Default `sim8`.
- Default `batch32`.
- `batch64` should not be scaled.
- `sim16` should remain a small sentinel only.
- `collector64` is plausible enough for a small probe, especially if throughput
  is the question.

## Recommended Next Matrix

### Goal A: Improve Survival Robustly

Use this as the core if the next run is primarily about making the objective work
better.

| Block | Settings | Why |
| --- | --- | --- |
| Core survival medium | blank/no-op, `medium`, `sim8/c32/b32`, paired render | Best leaderboard center with strong survival gain. |
| Core survival light | blank/no-op, `light`, `sim8/c32/b32`, paired render | Best survival-gain point estimate. |
| Control survival steady | blank/no-op, `steady`, `sim8/c32/b32`, paired render | Keeps baseline/control pressure. |
| Heavy stress | blank/no-op, `heavy`, `sim8/c32/b32`, paired render | Top-tail/variance probe; do not dominate. |

### Goal B: Top Leaderboard / Find Champions

Use this if the next run is more about producing strong tournament policies.

| Block | Settings | Why |
| --- | --- | --- |
| Survival medium repeats | blank/no-op, `medium`, paired render | Strong median leaderboard rank. |
| Survival heavy top-tail | blank/no-op, `heavy`, bounded copies | Top-3 includes heavy; useful champion probe. |
| Mix blank/high repeat | blank-containing mix recipes, `repH`, `sim8/batch32` | Rank 1 and multiple top mix rows use this shape. |
| Mix medium repeat anchor | blank-containing mix recipes, `repM`, `sim8/batch32` | Stabilizes high-repeat variance. |
| Collector64 probe | selected survival/mix rows, `collector64`, `batch32` | Top-5 includes a `collector64` mix row; signal is plausible but not broad. |

### Goal C: Generate Frozen Opponents

Use leaderboard as the selector here, not raw survival.

Candidate opponent sources:

- top survival medium/blank checkpoints;
- top survival heavy/blank checkpoints;
- top mix2 `r50-blank25-scr25 repH`;
- top mix3 `r50-blank50 repH`;
- a small passive survival checkpoint set as dirty/control opponents.

Do not sample from live Elo during training. Publish an immutable assignment
snapshot.

## Concrete Settings To Prefer

Preferred defaults:

```text
reward/objective: survival-plus-bonus/no-outcome
main opponent surface: blank/no-op survival plus selected blank-containing mixes
render: paired browser/fast for important cells
sim: 8
batch: 32
collector: 32 default, 64 probe
survival stochasticity: medium + light core, heavy bounded
mix repeat: repM anchor + repH upside, rep0 control
checkpoint cadence: keep current cadence unless operational constraints force change
```

Avoid scaling:

```text
batch64
wide sim16
projection@200k-selected rows without maturity/health guard
single-render conclusions
fixed-path-only stale/prune decisions
```

## Open Risks

- Seat/color asymmetry in leaderboard games may affect rankings.
- Some comparisons still mix maturity levels.
- `mix3` is broad but younger than survival.
- Projection@200k is not a ranking metric.
- Broad checkpoint discovery should be used for any refresh.

## Next Tooling

1. Add a leaderboard/survival crosswalk report:
   - rank, run id, knobs, survival latest/best, survival delta, rating, games,
     distinct opponents.
2. Export checkpoint-matched render rows as a stable CSV table.
3. Add maturity/health guards to promotion candidate generation.
4. Add top-k candidate snapshots for frozen-opponent assignment.
5. Add seat-swap or seat-bias diagnostics for leaderboard finalists.
