# Next Batch Design

Created: 2026-05-16

## Current Worldview

The next batch should explore reward strength, action stochasticity,
leaderboard-opponent immortality, and slot population. The 24-run matrix is one
sub-matrix per slot recipe. If we choose several slot recipes, we deliberately
cross beyond 24 runs.

Current naming convention: use `NAMING_CONVENTIONS.md`. New visible batch names
are `cz26a` for Grid A, `cz26b` for Grid B, and `cz26c` for canaries. New run
IDs should look like `cz26a-r017-out33-n10-imm0-b20w05r1`: batch, row, reward
alpha, action noise, leaderboard immortality, and slot recipe. Exact slot
counts and checkpoint refs stay in manifest fields.

Current broad launch defaults are in `CURRENT_LAUNCH_DEFAULTS.md`:
`gpu-l4-t4-cpu40`, `collector_env_num=256`, `n_episode=256`, `batch_size=64`,
`num_simulations=8`, `browser_lines + simple_symbols + cpu_oracle`. These are
defaults, not another experiment axis. H100 and batch32 are explicit ablations or
sentinels now.

Seeding default: every new training run in Grid A, Grid B, and canary lanes
starts from the single rank-1 checkpoint from the previous overnight r18fresh
leaderboard snapshot. The pinned ref is documented in
`NEXT_BATCH_SEEDING.md`. Top-10 lists are audit/opponent material, not mixed
initial-policy seeds for the next launch.

## Main Axes

### Reward outcome coefficient

Use four outcome strengths:

```text
alpha in {0.0, 0.33, 0.67, 1.0}
```

All four keep survival and bonus reward. `alpha = 0.0` is no outcome reward.
`alpha = 1.0` is full plus-outcome, but it must be implemented so total episode
return cannot go negative.

Implementation status: `reward_outcome_alpha` is now a plain manifest/trainer
knob. Grid A uses `out0`, `out33`, `out67`, and `out100`; Grid B uses `out50`.
All of these stay in the same reward family: survival reward on, bonus reward
on, terminal outcome coefficient varied.

### Stochasticity

Use three action-noise settings:

```text
clean
straight_override_p10_repeat_p10
straight_override_p20_repeat_p20
```

Evidence from r18fresh says `p10` helped on average, but not uniformly. It may
regularize action choices, but it can also muddy credit assignment. The `p20`
arm is exploratory and should not be assumed better. Existing `so10` is not one
scalar knob; it is 10% straight-action override plus 10% held-action repeat.

### Leaderboard-opponent immortality

Keep hard-coded blank and wall-avoidant opponents immortal. For leaderboard
checkpoint opponents, make immortality an explicit probability instead of a
separate ad hoc slot name.

Candidate settings:

```text
leaderboard_immortal_p0
leaderboard_immortal_p10
```

For the cleaned-up version of the current strong recipe, about 20% blank and 5%
wall are always immortal. With `imm10`, about 10% of the leaderboard slots are
also made immortal. Log the realized hard-coded, leaderboard, and total
immortal slot counts because the total differs by recipe.

## Matrix Shapes Under Discussion

Full per-recipe matrix:

```text
4 reward alphas * 3 stochasticity settings * 2 immortal settings = 24 runs
```

This is the literal full cross inside one slot recipe. If slot population is an
axis, multiply this by the recipe count:

```text
4 slot recipes -> 96 runs
5 slot recipes -> 120 runs
6 slot recipes -> 144 runs
```

The older 18-run cut-down ideas are not the current discussion. Current
discussion keeps 24 as the per-recipe sub-matrix and then considers crossing
several slot recipes.

There is also a second, slot-focused grid under discussion:

```text
reward alpha fixed around 0.5
survival reward on
bonus reward on
noise maybe clean and p10
multiple leaderboard-immortal probabilities
many slot populations
```

That second grid asks a different question from the broad 24-per-recipe matrix:
which opponent population actually creates useful learning and tournament
strength?

## Current Recommendation For Discussion

This is not a final launch decision. It is the current clearest experiment
shape based on the r18fresh evidence and the user's latest direction.

Latest refinement: see `GRID_REFINEMENT_REVIEW.md`. Current split:

- Grid A: compact production-like mixed recipes, fixed at 96 runs.
- Grid B: slot-focused recipe exploration, fixed at 40 runs. Grid B is
  where pure blank, pure wall-avoidant, pure rank1, and user-proposed coarse
  mixtures belong.
- Side controls: useful one-off checks that should not silently bloat either
  main grid.

### Grid A: Broad Robustness Grid

Question: do the best-looking slot recipes stay good across reward outcome
strength, action noise, and leaderboard-opponent immortality?

Use:

```text
reward outcome alpha: 0.0, 0.33, 0.67, 1.0
noise: clean, p10, p20
leaderboard opponent immortal probability: 0.0, 0.10
```

That is:

```text
4 * 3 * 2 = 24 runs per slot recipe
```

Recommended Grid A slot recipes to discuss:

| Code | Exact makeup | Why include |
| --- | --- | --- |
| `b20w05r1` | 20% blank immortal, 5% wall immortal, 75% rank1 leaderboard | Clean version of the historical survival winner. |
| `b10w05r1` | 10% blank immortal, 5% wall immortal, 85% rank1 leaderboard | Tests whether 20% blank was actually important. |
| `b20w10r1` | 20% blank immortal, 10% wall immortal, 70% rank1 leaderboard | Tests whether lowering wall from 10% to 5% mattered. |
| `b20w05top2` | 20% blank immortal, 5% wall immortal, 50% rank1, 25% rank2 | Tests rank1-heavy versus top-2 diversity while holding blank/wall fixed. |

These four recipes would make Grid A:

```text
4 * 24 = 96 runs
```

The ladder recipe is not added to Grid A. It belongs in Grid B. The pure
recipes and coarse stress recipes also belong in Grid B because they diagnose
what each opponent source teaches by itself.

### Grid B: Slot-Focused Grid

Question: which opponent population actually creates useful learning?

Use fewer non-slot settings:

```text
reward:
  survival reward on
  bonus reward on
  terminal outcome alpha around 0.5

noise:
  clean
  p10

leaderboard opponent immortal probability:
  0.0
  0.10
```

Then spend the budget on slot recipes. Recommended Grid B recipes:

| Code | Exact makeup | Why include |
| --- | --- | --- |
| `b100` | 100% blank immortal | Pure blank source control. |
| `w100` | 100% wall-avoidant immortal | Pure wall-avoidant source control. |
| `r1` | 100% rank1 leaderboard | Pure current-best leaderboard pressure control. |
| `b50r1` | 50% blank immortal, 50% rank1 leaderboard | User's solo-heavy mixture. |
| `b25w25r1` | 25% blank immortal, 25% wall immortal, 50% rank1 leaderboard | User's hard-coded-heavy mixture. |
| `b20w20lad4s` | 20% blank immortal, 20% wall immortal, 30% rank1, 20% rank2, 5% rank3, 5% rank4 | User's complex recipe, corrected to sum to 100%. |
| `b20w05r1` | 20% blank immortal, 5% wall immortal, 75% rank1 leaderboard | Anchor. |
| `b30w05r1` | 30% blank immortal, 5% wall immortal, 65% rank1 leaderboard | Higher blank dose without going all the way to 50%. |
| `b20w05top2` | 20% blank immortal, 5% wall immortal, 50% rank1, 25% rank2 | Top-2 diversity with anchor blank/wall. |
| `b20w05lad4` | 20% blank immortal, 5% wall immortal, 30% rank1, 20% rank2, 15% rank3, 10% rank4 | Ladder/tournament-robustness test. |

That makes Grid B:

```text
10 slot recipes * 2 noise settings * 2 leaderboard-immortal settings = 40 runs
```

Grid B can also test refresh cadence later:

```text
current interval refresh
checkpoint-boundary refresh
```

Refresh cadence stays a side lane until the core slot recipes are chosen. The
current accepted default is `2000` learner iterations. Checkpoint-boundary
refresh is a later control idea, not the current launch requirement.

Current implementation status: `scripts/build_curvytron_next_batch_manifest.py`
emits this design as `cz26c` canary, 96-row `cz26a` Grid A, 40-row `cz26b` Grid
B, or a 136-row full manifest. The builder publishes the scheduler config so
the tournament leaderboard can automatically flow back into trainer opponent
assignments.

## Recipe Evidence

r18fresh recipe survival aggregate:

| Recipe | Survival AUC | Survival 240k | Survival Latest | Survival Best |
| --- | ---: | ---: | ---: | ---: |
| rank2/rank1 | 161.0 | 173.1 | 162.5 | 230.9 |
| ladder | 154.3 | 156.0 | 156.4 | 225.8 |
| blank20/rank1-heavy | 199.1 | 239.6 | 240.2 | 297.2 |

Tournament aggregate:

| Recipe | Avg Top100 Rows | Top10 Rows | Top30 Rows | Best Rank |
| --- | ---: | ---: | ---: | ---: |
| rank2/rank1 | 3.3 | 0 | 1 | 18 |
| ladder | 6.3 | 1 | 12 | 4 |
| blank20/rank1-heavy | 7.0 | 9 | 17 | 1 |

The evidence strongly says the `blank20/rank1-heavy` recipe was best for
survival. It does not prove blank 20% was the cause. That recipe changed
several things at once:

- blank canvas went from 10% to 20%.
- hard-coded wall-avoidant opponent went from 10% to 5%.
- leaderboard slots became mostly rank 1 instead of laddered ranks.
- total explicit immortal exposure increased.
- scratch-bootstrap and refresh timing may have made the actual early opponent
  population differ from the recipe labels.

## Slot Recipe Axis

The slot recipe is now a real fourth axis. See `SLOT_RECIPE_DEEP_DIVE.md` for
the exact historical slot analysis.

Use percentages for human-readable recipe names, but author the actual opponent
recipe as a 64-slot bag. For the current 256 collector environments, repeat
that 64-slot bag four times and shuffle deterministically. Keep learner
`batch_size=64` unchanged; the learner samples from replay, so exact
per-gradient recipe proportions are not guaranteed without a future stratified
replay feature.

The historical winner was `blank20-wall5-rank1_70-rank1imm5`, but the next
version should move leaderboard immortality into the separate probability axis.
The current short code is `b20w05r1`:

```text
b20w05r1:
  20% blank immortal
  5% wall-avoidant immortal
  75% rank1 leaderboard checkpoint
```

Candidate slot recipes to cross:

| Code | Exact intended makeup | Question answered |
| --- | --- | --- |
| `b20w05r1` | 20% blank immortal, 5% wall immortal, 75% rank1 leaderboard | Clean version of the historical winner. |
| `b10w05r1` | 10% blank immortal, 5% wall immortal, 85% rank1 leaderboard | Was the B20 gain mostly extra blank exposure? |
| `b20w10r1` | 20% blank immortal, 10% wall immortal, 70% rank1 leaderboard | Was reducing wall exposure from 10% to 5% important? |
| `b20w05top2` | 20% blank immortal, 5% wall immortal, 50% rank1, 25% rank2 | Was rank1-heavy better than mixed top-2 once blank/wall are held near the winner? |
| `b20w05lad4` | 20% blank immortal, 5% wall immortal, 30% rank1, 20% rank2, 15% rank3, 10% rank4 | Does ladder still help tournament robustness when blank/wall match the winner? |
| `b30w05r1` | 30% blank immortal, 5% wall immortal, 65% rank1 leaderboard | Is more solo/blank pressure better, or does it weaken head-to-head learning? |

Log actual assignment SHAs, refresh pointer generations, provider load results,
and selected opponent kinds. Otherwise the recipe axis will explain recipe
labels rather than the opponents actually seen by the trainer.

Current recommended Grid A slot set from user discussion and critique:

| Code | Exact makeup | Why it is useful |
| --- | --- | --- |
| `b20w05r1` | 20% blank immortal, 5% wall immortal, 75% rank1 leaderboard | Clean historical-winner anchor. |
| `b10w05r1` | 10% blank immortal, 5% wall immortal, 85% rank1 leaderboard | Tests whether the historical B20 gain was mostly blank exposure. |
| `b20w10r1` | 20% blank immortal, 10% wall immortal, 70% rank1 leaderboard | Tests whether 5% wall was important versus 10% wall. |
| `b20w05top2` | 20% blank immortal, 5% wall immortal, 50% rank1, 25% rank2 | Tests rank1 concentration versus top-2 diversity. |

The originally proposed complex recipe with 20% blank, 20% wall, and
30/20/10/10 leaderboard weights sums to 110%; the corrected 20/20 version uses
30/20/5/5 for ranks 1/2/3/4. That corrected complex recipe now belongs in
Grid B, not Grid A.

Current recommended Grid B slot set:

| Code | Exact makeup | Why it is useful |
| --- | --- | --- |
| `b100` | 100% blank immortal | Pure blank source control. |
| `w100` | 100% wall immortal | Pure wall-avoidant source control. |
| `r1` | 100% rank1 leaderboard | Pure top-leaderboard source control. |
| `b50r1` | 50% blank immortal, 50% rank1 leaderboard | Extreme solo-heavy pressure test. |
| `b25w25r1` | 25% blank immortal, 25% wall immortal, 50% rank1 leaderboard | Same 50% hard-coded mass as blank50, but half moving wall pressure. |
| `b20w20lad4s` | 20% blank immortal, 20% wall immortal, 30% rank1, 20% rank2, 5% rank3, 5% rank4 | Corrected user complex recipe. |
| `b20w05r1` | 20% blank immortal, 5% wall immortal, 75% rank1 leaderboard | Anchor for comparison. |
| `b30w05r1` | 30% blank immortal, 5% wall immortal, 65% rank1 leaderboard | Higher blank dose without jumping to 50%. |
| `b20w05top2` | 20% blank immortal, 5% wall immortal, 50% rank1, 25% rank2 | Top-2 diversity with anchor blank/wall. |
| `b20w05lad4` | 20% blank immortal, 5% wall immortal, 30% rank1, 20% rank2, 15% rank3, 10% rank4 | Ladder/tournament-robustness test. |

## Opponent Refresh Cadence

Current side-lane question: should leaderboard opponent assignments refresh only
when the learner drops a checkpoint?

This might make training easier to interpret: each learner checkpoint would
cover a coherent opponent assignment block, and reward/survival changes after a
refresh could be tied to a specific new opponent set. It may also reduce
thrashing from constantly changing opponents. This is only a hypothesis to
track, not a settled decision.

## Missing Knobs To Remember

- Checkpoint selection: latest checkpoint is not enough; preserve tournament and
  heldout best-so-far checkpoints.
- Role randomization: trainer and tournament should not bake in one fixed seat.
- Observation contract: every checkpoint must carry the observation type it was
  trained with.
- Refresh cadence: leaderboard slots need proof they actually consume updated
  tournament winners.
- Runtime proof: the next launch needs an end-to-end proof that checkpoints move
  trainer -> subscriber -> tournament -> leaderboard -> trainer opponents.
