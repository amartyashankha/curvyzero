# Analysis Plan

## Goal

Analyze `cz26-full-20260517a` without getting fooled by incomplete runs,
different latest iterations, noisy rewards, or tournament exposure imbalance.

## Batch Structure

Grid A:

```text
4 recipes * 4 reward alphas * 3 noise settings * 2 immortality settings = 96
```

Grid B:

```text
10 recipes * out50 * 2 noise settings * 2 immortality settings = 40
```

## Recipe Glossary

Each recipe is a 64-slot opponent bag. The collector repeats that bag across
the larger collector wave and shuffles assignments.

| Recipe | 64-slot makeup | Plain meaning |
| --- | --- | --- |
| `b100` | 64 blank | only blank immortal opponent seats |
| `w100` | 64 wall | only wall-avoider opponent seats |
| `r1` | 64 rank1 | only current rank-1 leaderboard opponent seats |
| `b50r1` | 32 blank, 32 rank1 | half blank, half rank-1 |
| `b25w25r1` | 16 blank, 16 wall, 32 rank1 | quarter blank, quarter wall, half rank-1 |
| `b30w05r1` | 19 blank, 3 wall, 42 rank1 | more blank, small wall, rank-1 heavy |
| `b20w05r1` | 13 blank, 3 wall, 48 rank1 | baseline mixed recipe |
| `b10w05r1` | 7 blank, 3 wall, 54 rank1 | less blank than baseline, more rank-1 |
| `b20w10r1` | 13 blank, 6 wall, 45 rank1 | baseline blank, more wall |
| `b20w05top2` | 13 blank, 3 wall, 32 rank1, 16 rank2 | baseline blank/wall, split leaderboard between top 2 |
| `b20w05lad4` | 13 blank, 3 wall, 19 rank1, 13 rank2, 10 rank3, 6 rank4 | ladder over top 4 with small wall |
| `b20w20lad4s` | 13 blank, 13 wall, 19 rank1, 13 rank2, 3 rank3, 3 rank4 | ladder over top 4 with heavy wall |

`blank` and `wall` are hard-coded opponents. Leaderboard seats can be mortal or
immortal depending on the `imm0`/`imm10` axis. `imm10` means leaderboard seats
are made immortal with 10% probability; it does not meaningfully affect pure
blank-only or wall-only recipes.

## Signals

### Reward Progression

Use as a diagnostic, not the main score.

Important rules:

- Compare own reward only within compatible reward definitions.
- Reward can fall if tournament-fed opponents get stronger.
- For outcome reward, inspect components or residuals when available.

Candidate row metrics:

- first reward;
- matched-endpoint reward;
- best reward and best iteration;
- latest reward;
- reward retention from best to latest;
- reward around assignment/export refresh boundaries if available.

### Survival Progression

Use as the cleanest cross-reward learning signal.

Candidate row metrics:

- first survival;
- matched-endpoint survival;
- survival AUC over common checkpoint grid;
- best survival and best iteration;
- latest survival;
- retention from best to latest;
- slope early/mid/late;
- missingness/completion flags.

### Tournament Performance

Use as the head-to-head policy-quality signal.

Candidate row metrics:

- best tournament rank reached by any checkpoint in the run;
- best Elo/rating;
- latest-for-run tournament rank;
- top10/top30/top100 checkpoint counts;
- deduped top-band presence by run and by setting;
- games played and distinct opponents;
- tournament survival/duration if available;
- rating stability/missingness flags.

## Projection Method

Every projection should start from one joined table:

```text
manifest row
+ training/eval curve summary
+ tournament checkpoint summary
+ health/missingness flags
```

Every aggregate should report:

```text
n_expected
n_present_with_eval
n_present_with_tournament
common_horizon
missing_rows
```

The goal is to avoid pretending an aggregate is clean when half the rows are
missing, unrated, or stopped early.

### Grid A

Grid A is fully crossed, so every axis can be collapsed while averaging over
the other axes.

Primary projections:

1. Reward alpha:
   - compare `out0`, `out33`, `out67`, `out100`;
   - aggregate over recipe, noise, and immortality.
2. Noise:
   - compare `n0`, `n10`, `n20`;
   - aggregate over reward, recipe, and immortality.
3. Leaderboard immortality:
   - compare `imm0`, `imm10`;
   - aggregate over reward, noise, and recipe.
4. Recipe:
   - compare `b20w05r1`, `b10w05r1`, `b20w10r1`, `b20w05top2`;
   - aggregate over reward, noise, and immortality.

Secondary interactions:

- reward alpha x recipe;
- reward alpha x noise;
- recipe x immortality;
- noise x immortality;
- anchor recipe `b20w05r1` versus perturbations.

Matched contrast blocks:

| Effect | Matching block | Contrasts |
| --- | --- | --- |
| Reward alpha | recipe x noise x immortality | `out33 - out0`, `out67 - out33`, `out100 - out67`, `out100 - out0`, alpha slope |
| Noise | recipe x reward x immortality | `n10 - n0`, `n20 - n10`, `n20 - n0` |
| Immortality | recipe x reward x noise | `imm10 - imm0` |
| Recipe | reward x noise x immortality | `b10w05r1 - b20w05r1`, `b20w10r1 - b20w05r1`, `b20w05top2 - b20w05r1` |

### Grid B

Grid B is slot-focused. Reward is fixed to `out50`, so the main question is
opponent population.

Primary projections:

1. Recipe:
   - compare all 10 recipes;
   - explicitly separate pure controls `b100`, `w100`, `r1`.
2. Noise:
   - compare `n0`, `n10`.
3. Leaderboard immortality:
   - compare `imm0`, `imm10`.

Secondary interactions:

- recipe x immortality;
- recipe x noise;
- pure controls versus mixed recipes;
- anchor `b20w05r1` versus `b30w05r1`, `b50r1`, `b25w25r1`,
  `b20w05top2`, and ladder variants.

Matched contrast blocks:

| Effect | Matching block | Contrasts |
| --- | --- | --- |
| Recipe/control | noise x immortality | every recipe versus `b100`, `w100`, `r1`, plus recipe rank within each block |
| Immortality negative control | `b100`/`w100` x noise | `imm10 - imm0`; should be near noise because pure hard-coded recipes have no leaderboard slots |
| Noise | recipe x immortality | `n10 - n0` |
| Grid A/B bridge | shared recipes `b20w05r1`, `b20w05top2`, shared `n0/n10`, shared `imm0/imm10` | compare relative effects near Grid A `out33/out67` to Grid B `out50`; do not merge absolute scores |

## Fairness Rules

- Use matched checkpoints first.
- If not all rows reached the same checkpoint, report the matched endpoint and
  row inclusion count.
- Latest-only is operational, not primary evidence.
- Tournament rank must be reported with games/opponents because exposure is
  uneven.
- Do not average across Grid A and Grid B unless the axis meaning is identical.
- Do not compare reward values across incompatible reward definitions.
- Report uncertainty and missingness beside every aggregate.

Use three labeled readouts when rows are incomplete:

| Readout | Rule |
| --- | --- |
| all-row common | highest exact iteration common to all rows in that projection |
| pairwise matched | each contrast uses the minimum available horizon for the rows in that contrast |
| finishers only | rows reaching the final horizon; biased sensitivity check only |

Known sensitivity rows to isolate if they still look incomplete or failed:

```text
cz26a-r013-out67-n0-imm0-b20w05r1
cz26b-r028-out50-n10-imm10-b20w05r1
```

## Minimum Output Tables

1. Row-level summary.
2. Grid A reward-alpha projection.
3. Grid A noise projection.
4. Grid A immortality projection.
5. Grid A recipe projection.
6. Grid B recipe projection.
7. Grid B pure controls versus mixed recipes.
8. Tournament top-band table by run and setting.
9. Missingness/completion table.

## Useful Plots Later

- survival curve small multiples by recipe/reward;
- reward curve small multiples within reward family;
- best rank versus best survival scatter;
- tournament duration over rating batch index;
- retention histogram by setting.
- matched-effect forest plots for reward/noise/immortality/recipe;
- Grid A reward x noise heatmaps per recipe/immortality;
- Grid B recipe rank strip plots by noise/immortality;
- survival AUC versus tournament best Elo scatter;
- retention versus tournament best Elo;
- coverage heatmaps by grid axes.

## Explicitly Later

Lineage tables are useful but not the first analysis lane. For this pass,
assume the core run/tournament data is trustworthy enough and focus on the
experiment signals. Return to lineage only if the data has contradictions that
cannot be resolved from status/eval/rating snapshots.
