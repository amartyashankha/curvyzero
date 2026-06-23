# Slot Recipe Deep Dive

Captured: 2026-05-16.

This note focuses only on the opponent-slot population axis in the completed
`r18fresh` batch. The broader reward, survival, and tournament analysis already
lives in `TREND_ANALYSIS.md`, `MATCHED_GRID_ANALYSIS.md`,
`REWARD_BREAKDOWN_ANALYSIS.md`, and `DETAILED_RUN_STOCKTAKE.md`.

The historical batch was:

```text
3 reward variants * 3 slot recipes * 2 action-noise settings = 18 runs
```

All 18 runs shared H100 CPU40, `collector_env_num=256`, `batch_size=32`,
`num_simulations=8`, `save_ckpt_after_iter=10000`, `max_train_iter=300000`,
`source_max_steps=1048576`, `random_per_episode` learner seat, and
`browser_lines + simple_symbols` observation.

## Exact Historical Recipes

| Short name | Manifest id | Exact slot makeup | Explicit immortal mass |
| --- | --- | --- | ---: |
| `r2/r1` | `blank10-wall10-rank2_25-rank1_55` | 10% blank immortal, 10% wall-avoidant immortal, 25% rank2 mortal, 55% rank1 mortal | 20% |
| `ladder` | `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5` | 10% blank immortal, 10% wall-avoidant immortal, 10% rank4 mortal, 15% rank3 mortal, 20% rank2 mortal, 30% rank1 mortal, 5% rank1 immortal | 25% |
| `b20/r1imm` | `blank20-wall5-rank1_70-rank1imm5` | 20% blank immortal, 5% wall-avoidant immortal, 70% rank1 mortal, 5% rank1 immortal | 30% |

Important caveat: rank slots were refreshable tournament-derived assignment
slots. The labels say what rank the slot intended to consume, not a single fixed
checkpoint identity for the whole run.

## Aggregate Readout

| Recipe | Runs | Survival AUC | Survival at 240k | Latest Survival | Best Survival | Top10 Rows | Top30 Rows | Top100 Rows | Best Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r2/r1` | 6 | 161.3 | 173.1 | 162.5 | 230.9 | 0 | 1 | 20 | 18 |
| `ladder` | 6 | 154.2 | 156.0 | 156.4 | 225.8 | 1 | 12 | 38 | 4 |
| `b20/r1imm` | 6 | 198.5 | 239.6 | 240.2 | 297.2 | 9 | 17 | 42 | 1 |

Plain read:

- `b20/r1imm` is the strongest survival recipe by every aggregate survival
  view: matched AUC, 240k, latest, and best checkpoint.
- `b20/r1imm` also has the strongest tournament top band in the latest
  `round-000035` snapshot.
- `ladder` is weak on survival but still puts many checkpoints high in the
  tournament. That means it may create matchup-useful policies that the fixed
  survival eval does not fully reward.
- `r2/r1` is the weakest tournament recipe and only beats ladder on survival
  aggregate.

## Same Reward And Noise Comparisons

This table changes only the slot recipe inside each reward/noise pair. Deltas
are measured against `r2/r1` for that same reward and noise setting.

| Reward | Noise | `r2/r1` AUC | Ladder Delta AUC | B20 Delta AUC | `r2/r1` 240k | Ladder Delta 240k | B20 Delta 240k | Best Rank `r2/r1` | Best Rank Ladder | Best Rank B20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse | clean | 150.6 | -25.7 | +50.3 | 151.1 | -24.5 | +134.1 | 92 | 27 | 24 |
| sparse | so10 | 183.2 | -28.4 | +21.7 | 187.2 | -48.1 | +108.8 | 90 | 48 | 30 |
| no_out | clean | 153.3 | +10.7 | -6.5 | 159.0 | +16.5 | +7.1 | 32 | 68 | 14 |
| no_out | so10 | 157.5 | +1.3 | +49.7 | 171.6 | -3.4 | +38.6 | 18 | 59 | 60 |
| plus_out | clean | 152.1 | +0.4 | +90.7 | 169.9 | +4.2 | +77.6 | 34 | 4 | 57 |
| plus_out | so10 | 171.0 | -0.9 | +17.5 | 199.9 | -47.4 | +32.5 | 54 | 13 | 1 |

Plain read:

- B20 is the strongest survival move in four of six matched comparisons, and
  it is dramatically stronger in the sparse and plus-outcome slices.
- B20 is not a clean law. In `no_out + clean`, ladder slightly beats it on
  matched survival AUC, while B20 still has better tournament best rank.
- Ladder almost never improves survival relative to `r2/r1`, except weakly in
  no-outcome. Its case is tournament robustness, not fixed-eval survival.
- `plus_out + b20 + so10` is the tournament champion setting, while
  `plus_out + b20 + clean` has better matched survival AUC. This is why the
  next batch should keep clean, p10, and p20.

## What We Can Compare From Existing Data

The existing 18-run batch gives only three recipes, so it cannot isolate every
cause. It can still answer a few comparisons.

### Ladder vs R2/R1

This is the cleanest historical slot comparison because both recipes used:

```text
10% blank immortal
10% wall-avoidant immortal
```

What changed:

```text
r2/r1:
  25% rank2
  55% rank1

ladder:
  10% rank4
  15% rank3
  20% rank2
  30% rank1
  5% rank1 immortal
```

Read: ladder adds rank diversity and a little rank1 immortality while reducing
rank1 concentration. Survival got worse, but tournament placement improved.
This is real evidence that the ladder can create matchup-useful policies even
when fixed eval survival is weaker.

### B20 vs R2/R1

What changed:

```text
blank: 10% -> 20%
wall-avoidant immortal: 10% -> 5%
rank2: 25% -> 0%
rank1 total: 55% -> 75%
explicit immortal mass: 20% -> 30%
```

Read: B20 was much better for survival. But this comparison is confounded: more
blank, less wall, no rank2, more rank1, and more immortality all changed at
once.

Here "less wall" only means the hard-coded wall-avoidant opponent was sampled
5% of episodes instead of 10%. It does not mean walls in the game changed.

### B20 vs Ladder

What changed:

```text
blank: 10% -> 20%
wall-avoidant immortal: 10% -> 5%
rank4/rank3/rank2 ladder mass: 45% -> 0%
rank1 total: 35% -> 75%
explicit immortal mass: 25% -> 30%
```

Read: B20 wins survival and current top-band tournament count. Ladder still has
some unusually good tournament placements, so the broad-rank curriculum may be
doing something different from pure survival training.

## What The Slot Axis Probably Means

The B20 recipe probably helped survival because it gave more exposure to an
immortal blank/no-op board while keeping most non-hardcoded opponent mass on
the current rank-1 checkpoint. That is a simpler curriculum than a ladder of
moving targets, but still includes enough head-to-head pressure to avoid pure
solo training.

That is not proof that "20% blank" alone caused the gain. B20 changed several
things at once:

- blank increased from 10% to 20%;
- wall avoider decreased from 10% to 5%;
- leaderboard slots became almost entirely rank 1;
- explicit immortal mass rose to 30%;
- refresh timing may have changed actual checkpoint opponents behind the slot
  labels.

The ladder result is the caution flag. It looked bad on fixed survival but good
enough in tournament placement to deserve a deeper read. A plausible
interpretation is that it teaches policies to handle a wider range of opponent
styles, which can help in head-to-head tournament ranking even if the fixed eval
survival score is lower.

## Refresh Cadence Thought

Opponent refresh cadence is a separate side lane. In r18fresh, leaderboard
slots refreshed during training. That can be useful because the trainer keeps
seeing stronger tournament-selected opponents, but it can also make the target
move too often.

Current question to test later:

```text
Should leaderboard opponents refresh only when the learner drops a checkpoint?
```

Why this may be cleaner:

- the policy trains against one assignment for a longer block;
- every opponent change lines up with a durable learner checkpoint boundary;
- reward drops after refresh are easier to interpret as "new stronger
  opponents arrived";
- assignment lineage is easier to audit.

This is not a conclusion. It is an experiment/control axis to keep in view.

## Two-Grid Experiment Shape Under Discussion

There are two distinct experiment shapes under discussion.

### Grid A: Broad Crossed Matrix

This is the full cross around several dimensions:

```text
slot recipes * 4 reward alphas * 3 action-noise settings * 2 leaderboard-immortal settings
```

Use these reward outcome alphas:

```text
0.0, 0.33, 0.67, 1.0
```

All reward variants keep survival and bonus reward turned on; alpha controls
only the terminal outcome term.

If six slot recipes are crossed, this is:

```text
6 * 4 * 3 * 2 = 144 runs
```

### Grid B: Slot-Focused Grid

This grid would pin the reward closer to the current best guess and spend the
budget on slot structure.

Current fixed reward idea:

```text
survival reward on
bonus reward on
terminal outcome alpha = 0.5
```

Noise candidates for this grid:

```text
clean
straight_override_p10_repeat_p10
```

Leaderboard immortality candidates matter here and should remain explicit. The
point is to study slot populations and immortality, not to over-read the reward
axis.

Candidate knobs for Grid B:

- many slot recipes;
- at least two leaderboard-immortal probabilities;
- maybe clean and p10 noise;
- maybe refresh cadence as a later control: continuous/2k refresh vs
  checkpoint-boundary refresh.

## Current Candidate Recipes From User Discussion

These recipes are candidate experiment settings, not launch decisions.

Planning-language note: keep recipe names human-readable as percentages, but
author the implementation as a 64-slot opponent bag. The current collector wave
has 256 environments, so repeat the 64-slot bag four times and shuffle
deterministically. Keep learner `batch_size=64` unchanged. A learner mini-batch
is sampled from replay, so exact per-gradient recipe proportions are not
guaranteed unless we later add stratified replay sampling.

### User-proposed recipes

| Candidate | Exact makeup | What it probes |
| --- | --- | --- |
| `blank100` | 100% blank immortal | Pure blank/no-op source. Useful as a diagnostic control. |
| `wall100` | 100% wall-avoidant immortal | Pure wall-avoidant source. Useful as a diagnostic control. |
| `rank1_100` | 100% rank1 leaderboard checkpoint | Pure current-best leaderboard pressure. Useful as a diagnostic control. |
| `blank50-rank1_50` | 50% blank immortal, 50% rank1 leaderboard | Very solo-heavy curriculum with only the current best leaderboard opponent as pressure. |
| `blank25-wall25-rank1_50` | 25% blank immortal, 25% wall-avoidant immortal, 50% rank1 leaderboard | Same 50% hard-coded immortal mass as `blank50-rank1_50`, but half of it is moving wall-avoidant pressure. |
| `blank20-wall20-rankspread60` | 20% blank immortal, 20% wall-avoidant immortal, 60% leaderboard spread | More hard-coded immortal pressure plus leaderboard diversity. |

Arithmetic note for `blank20-wall20-rankspread60`: the first phrasing
`30% rank1 + 20% rank2 + 10% rank3 + 10% rank4` sums to 70% leaderboard mass,
so with 20% blank and 20% wall it totals 110%. Clean corrected versions:

| Corrected option | Exact leaderboard split | Comment |
| --- | --- | --- |
| `blank20-wall20-r1_30-r2_20-r3_5-r4_5` | 30% rank1, 20% rank2, 5% rank3, 5% rank4 | Keeps the user's rank1/rank2 weights and gives small rank3/rank4 exposure. |
| `blank15-wall15-r1_30-r2_20-r3_10-r4_10` | 30% rank1, 20% rank2, 10% rank3, 10% rank4 | Keeps the user's leaderboard weights, but lowers blank/wall to 15% each. |
| `blank20-wall20-rankspread_3_2_1_1` | rank weights proportional to 3:2:1:1 over 60%, approximately 25.7/17.1/8.6/8.6 | Preserves proportions, but introduces awkward fractional recipe weights. |

The cleanest integer version is probably
`blank20-wall20-r1_30-r2_20-r3_5-r4_5` if the priority is keeping 20% blank
and 20% wall.

## Grid A Candidate Slot Set

Grid A is the broad cross:

```text
slot recipe
* alpha in {0.0, 0.33, 0.67, 1.0}
* noise in {clean, p10, p20}
* leaderboard_immortal_probability in {0.0, 0.10}
```

This is 24 runs per slot recipe.

Current recommended Grid A slot recipes:

| Candidate | Exact makeup | Why it is useful |
| --- | --- | --- |
| `blank20-wall5-rank1_75` | 20% blank immortal, 5% wall immortal, 75% rank1 leaderboard | Clean version of historical B20 winner with leaderboard immortality moved out to its own axis. |
| `blank10-wall5-rank1_85` | 10% blank immortal, 5% wall immortal, 85% rank1 leaderboard | Tests whether 20% blank was necessary. |
| `blank20-wall10-rank1_70` | 20% blank immortal, 10% wall immortal, 70% rank1 leaderboard | Tests whether 5% wall was important versus 10% wall. |
| `blank20-wall5-rank2_25-rank1_50` | 20% blank immortal, 5% wall immortal, 25% rank2, 50% rank1 | Tests rank1-heavy versus top-2 diversity while holding blank/wall fixed. |

If all four are crossed, Grid A is:

```text
4 * 24 = 96 runs
```

The ladder recipe is not added to Grid A in this design. It belongs in Grid B.
The pure controls and high-hardcoded recipes also belong in Grid B. They are
useful diagnostics, but they are not as clean for Grid A's broad
production-like cross.

## Grid B Slot-Focused Candidate Shape

Grid B spends more budget on slot structure and less on reward/noise.

Current fixed-ish settings under discussion:

```text
reward:
  survival on
  bonus on
  terminal outcome alpha = 0.5

noise:
  clean and p10 are the main candidates

leaderboard immortality:
  at least p0 and p10
  maybe a higher sentinel later, but do not silently add it to the main cross
```

Recommended Grid B recipe set:

```text
blank100
wall100
rank1_100
blank50-rank1_50
blank25-wall25-rank1_50
blank20-wall20-rank1_30-rank2_20-rank3_5-rank4_5
blank20-wall5-rank1_75
blank30-wall5-rank1_65
blank20-wall5-rank2_25-rank1_50
blank20-wall5-ladder75-rank1_30-rank2_20-rank3_15-rank4_10
```

With fixed reward alpha around 0.5, clean/p10 noise, and p0/p10 leaderboard
immortality, this is:

```text
10 recipes * 2 noise settings * 2 immortal settings = 40 runs
```

The clean interpretation goal for Grid B is to answer:

- How much blank practice is helpful before it becomes too solo-heavy?
- Does the wall-avoidant hard-coded opponent help or distract?
- Is rank diversity responsible for ladder's tournament placements?
- How much leaderboard-opponent immortality helps when blank/wall are already
  immortal?
- Does checkpoint-boundary opponent refresh make the learning curves easier to
  interpret than fixed-interval refresh?

## Critique Of Current Recipe Ideas

- `blank100`, `wall100`, and `rank1_100` are pure diagnostic controls. They
  should be first-class Grid B recipes, but they should not be mistaken for
  likely production curricula.
- `blank50-rank1_50` is high-signal but risky. It may boost survival mechanics
  while undertraining head-to-head interaction.
- `blank25-wall25-rank1_50` is useful because it keeps 50% hard-coded immortal
  mass like `blank50-rank1_50`, but swaps half the blank exposure for moving
  obstacle pressure.
- `blank20-wall20-rankspread60` is useful but should be corrected to sum to
  100%. It mixes two changes at once: more wall pressure and more rank
  diversity.
- `leaderboard_immortal_p10` is a clean axis, but total immortal exposure will
  vary a lot by recipe because blank and wall are always immortal. For example,
  `blank50-rank1_50` with p10 has 55% expected immortal opponents, while
  `blank20-wall5-rank1_75` with p10 has 32.5%.
- Bonus reward should stay on for these grids. A bonus reward bump is a separate
  side question because r18fresh barely collected bonuses; increasing the
  bonus scale may not matter unless policies encounter bonuses often enough.

## Current Interpretation In Plain Language

The r18fresh slot result says:

- More blank-board practice plus mostly rank1 opponents was best for fixed eval
  survival.
- Ladder/rank diversity was not best for fixed eval survival, but it put a
  surprising number of checkpoints high in tournament rank.
- Therefore there are probably two different useful skills:
  - surviving for a long time in the fixed eval setup;
  - beating many different policy styles in the tournament.

The next experiment should preserve that distinction. Survival, tournament
rank, and reward are separate readouts:

- survival says whether games last longer;
- tournament rank says whether checkpoints beat other checkpoints;
- own reward is mostly a diagnostic, because stronger refreshed opponents can
  naturally make reward drop even if the system is working.

The most important unanswered slot questions are:

1. How much blank-board practice is useful?
2. How much wall-avoidant hard-coded pressure is useful?
3. Does rank1-heavy training beat mixed top-2 training?
4. Does ladder/rank diversity create tournament robustness even if survival is
   weaker?
5. Does making leaderboard opponents immortal some of the time improve learning,
   or does it make the curriculum too harsh?
6. Should opponent assignments refresh at fixed train-iteration intervals or
   only when the learner writes a checkpoint?

## Next Slot-Axis Candidates

Candidate slot recipes:

| Candidate | Exact intended makeup | Question answered |
| --- | --- | --- |
| `blank20-wall5-rank1_75` | 20% blank immortal, 5% wall immortal, 75% rank1 leaderboard | Clean version of the historical winner, with leaderboard immortality moved to the separate probability axis. |
| `blank10-wall5-rank1_85` | 10% blank immortal, 5% wall immortal, 85% rank1 leaderboard | Was the B20 gain mostly extra blank exposure? |
| `blank20-wall10-rank1_70` | 20% blank immortal, 10% wall immortal, 70% rank1 leaderboard | Was reducing wall exposure from 10% to 5% important? |
| `blank20-wall5-rank2_25-rank1_50` | 20% blank immortal, 5% wall immortal, 25% rank2, 50% rank1 | Was rank1-heavy better than mixed top-2 once blank/wall are held near the winner? |
| `blank20-wall5-ladder` | 20% blank immortal, 5% wall immortal, 10% rank4, 15% rank3, 20% rank2, 30% rank1 | Does ladder still help tournament robustness when blank/wall match the winner? |
| `blank30-wall5-rank1_65` | 30% blank immortal, 5% wall immortal, 65% rank1 leaderboard | Is more solo/blank pressure better, or does it weaken head-to-head learning? |

This file is not choosing how many slot recipes to run. It records the candidate
recipe meanings so the user can choose the scope.

## Proof Requirements

- Log actual assignment SHA and selected opponent kind; recipe labels alone are
  not enough.
- Record both the human-readable intended percentages and final 64-slot counts
  in the manifest artifact.
- Repeat the 64-slot bag four times across the current 256 collector
  environments, then deterministically shuffle.
- Do not add stratified replay sampling now.
- Keep leaderboard-opponent immortality as a probability flag applied after
  selecting the policy. Do not encode it as a separate policy identity.
- Log hard-coded immortal mass, realized leaderboard immortal mass, and total
  realized immortal mass for every row.
- Compare survival, own reward, and tournament rank separately. They measured
  different things in r18fresh.
- Preserve best-so-far checkpoints, not only latest checkpoints.
