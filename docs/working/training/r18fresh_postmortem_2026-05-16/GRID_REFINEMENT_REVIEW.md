# Grid Refinement Review

Created: 2026-05-16.

This note is the current synthesis after the user's latest slot-curriculum
intuition and two waves of critique. It does not launch anything. It records
the current recommendation and the reasons behind it.

Naming convention: new current-code manifests should use `NAMING_CONVENTIONS.md`.
The old long names remain useful evidence labels, but new operator-facing rows
should use short codes like `b20w05r1`, `out33`, `n10`, and `imm10`.

## Plain Glossary

- **Blank opponent**: a hard-coded no-op / empty-board opponent. It should be
  immortal because it is not a learned policy.
- **Wall-avoidant opponent**: a hard-coded opponent that mostly avoids walls.
  It should also be immortal. This is an opponent behavior, not a change to the
  game walls.
- **Leaderboard opponent**: a checkpoint selected from the tournament
  leaderboard, such as rank 1 or rank 2.
- **Leaderboard immortal probability**: after selecting a leaderboard
  checkpoint, this probability makes that checkpoint opponent immortal for that
  sampled environment slot. This is separate from which policy was selected.
- **Pure control**: a deliberately extreme recipe such as `b100`, `w100`, or
  `r1`. These explain what each source teaches by itself.
- **Mixed curriculum**: a recipe that combines blank, wall-avoidant, and/or
  leaderboard opponents. These are closer to what we actually want to train.

## Current Split

The pure source checks are important, but they answer a different question from
the broad robustness grid.

- **Grid A**: compact production-like mixed recipes crossed with reward alpha,
  action noise, and leaderboard-opponent immortality. Fixed shape: 96 runs.
- **Grid B**: slot-focused exploration. This is where we directly test pure
  blank, pure wall-avoidant, pure rank1, user-proposed coarse mixtures, and
  evidence-based mixed recipes. Fixed shape: 40 runs.
- **Side controls**: useful one-off checks that should not silently bloat
  either main grid.

Grid B is not optional background. It is the main experiment for understanding
which opponent population actually helps.

## Count Contract

Author opponent recipes as a 64-slot bag.

The current collector wave has 256 environments. Fill it by repeating the
64-slot bag four times and then deterministically shuffling the 256 assignments.

Keep learner `batch_size=64` unchanged. That learner batch is sampled from the
replay buffer, so it is not guaranteed to contain exactly the same proportions
as the latest 64-slot recipe bag. The 64-slot bag controls what data we feed
into the buffer; it does not guarantee exact proportions in every gradient
update.

Do not add stratified replay sampling now. If we later want every learner
mini-batch to have exact opponent proportions, that is a separate feature.

Manifest rule:

```text
recipe intent: human-readable percentages
recipe implementation: exact 64-slot counts
collector assignment: repeat 64-slot bag 4x -> shuffle -> 256 envs
learner batch: unchanged batch_size=64 sampled from replay
```

## Ten Current Slot Suggestions And Critique

| # | Recipe | Current role | Critique |
| ---: | --- | --- | --- |
| 1 | `b100` | Grid B diagnostic | Clean solo-survival control; not a full curriculum because it removes all opponent pressure. |
| 2 | `w100` | Grid B diagnostic | Tests the wall-avoidant source alone; likely unnatural as a real curriculum. |
| 3 | `r1` | Grid B diagnostic | Clean all-leaderboard pressure control; may be too narrow and harsh without blank/wall scaffolding. |
| 4 | `b50r1` | Grid B core | Tests whether much more blank practice helps; risks starving head-to-head learning. |
| 5 | `b25w25r1` | Grid B core | Tests hard-coded-heavy pressure with blank and wall split evenly; confounds blank amount and wall amount. |
| 6 | `b20w20lad4s` | Grid B core/stress | Corrected version of the user's complex recipe; useful, but changes wall pressure and rank diversity at once. |
| 7 | `b20w05r1` | Grid A + Grid B anchor | Clean version of the historical winner with leaderboard immortality moved to its own axis. |
| 8 | `b10w05r1` | Grid A core | Tests whether 20% blank was really needed. |
| 9 | `b20w10r1` | Grid A core | Tests whether the historical winner benefited from lowering wall-avoidant exposure to 5%. |
| 10 | `b20w05top2` | Grid A + Grid B core | Tests rank1-heavy pressure versus top-2 diversity while holding blank and wall fixed. |

Important correction: the earlier phrasing "20% blank, 20% wall, 30% rank1,
20% rank2, 10% rank3, 10% rank4" sums to 110%. The corrected version keeps 20%
blank and 20% wall, then uses:

```text
30% rank1
20% rank2
5% rank3
5% rank4
```

## Ten More Useful Variants

These are not all recommended for the next launch. They are useful variants to
keep in view.

| # | Variant | Why it exists |
| ---: | --- | --- |
| 1 | `b30w05r1` | Tests whether more blank than the anchor improves retention. |
| 2 | `b20w05lad4` | Tests ladder diversity while matching the anchor's 20% blank and 5% wall. |
| 3 | `b20w05lad4h` | More rank1-heavy ladder; less extreme than full broad diversity. |
| 4 | `b20r1` | Tests whether wall-avoidant exposure is needed at all. |
| 5 | `b20w15r1` | Smooth wall-dose variant between 10% and 20%. |
| 6 | `w05r1` | Tests whether blank scaffolding is needed when wall remains present. |
| 7 | `b40w05r1` | Intermediate between blank30 and blank50. |
| 8 | `b20w05top2s` | Softer top-2 diversity than 25% rank2. |
| 9 | `b20w05lad3` | Ladder-lite without rank4. |
| 10 | `b20w05r1-rfckpt` | Same population, different opponent-refresh timing; keep as a later side control. |

## Recommended Grid A

Grid A asks:

```text
Do production-like mixed recipes stay good across reward outcome strength,
action noise, and leaderboard-opponent immortality?
```

Use four recipes for the first pass:

| Code | 64-slot bag | Approximate makeup | Question |
| --- | --- | --- | --- |
| `b20w05r1` | blank 13, wall 3, rank1 48 | 20.3% blank, 4.7% wall, 75.0% rank1 | Anchor: cleaned historical winner. |
| `b10w05r1` | blank 7, wall 3, rank1 54 | 10.9% blank, 4.7% wall, 84.4% rank1 | Was extra blank the cause? |
| `b20w10r1` | blank 13, wall 6, rank1 45 | 20.3% blank, 9.4% wall, 70.3% rank1 | Was lower wall exposure the cause? |
| `b20w05top2` | blank 13, wall 3, rank2 16, rank1 32 | 20.3% blank, 4.7% wall, 25.0% rank2, 50.0% rank1 | Was rank1 concentration the cause? |

Cross each recipe with:

```text
reward outcome alpha: 0.0, 0.33, 0.67, 1.0
noise: clean, p10, p20
leaderboard immortal probability: 0.0, 0.10
```

Size:

```text
4 recipes * 4 alphas * 3 noise settings * 2 immortal settings = 96 runs
```

The ladder recipe is not added to Grid A in this design. It belongs in Grid B.

## Recommended Grid B

Grid B asks:

```text
Which opponent population is actually useful?
```

This grid should follow the user's concrete direction closely: test pure
sources, coarse mixtures, and evidence-based mixed recipes directly.

Fixed non-slot settings:

```text
reward: survival on, bonus on, outcome alpha around 0.5
noise: clean and p10
leaderboard immortal probability: 0.0 and 0.10
```

Recommended ten recipes:

| Code | 64-slot bag | Approximate makeup | Why include |
| --- | --- | --- | --- |
| `b100` | blank 64 | 100% blank | Pure blank source control. |
| `w100` | wall 64 | 100% wall-avoidant | Pure wall-avoidant source control. |
| `r1` | rank1 64 | 100% rank1 | Pure current-best leaderboard pressure control. |
| `b50r1` | blank 32, rank1 32 | 50% blank, 50% rank1 | User's solo-heavy mixture. |
| `b25w25r1` | blank 16, wall 16, rank1 32 | 25% blank, 25% wall, 50% rank1 | User's hard-coded-heavy mixture with wall pressure. |
| `b20w20lad4s` | blank 13, wall 13, rank1 19, rank2 13, rank3 3, rank4 3 | 20.3% blank, 20.3% wall, 29.7% rank1, 20.3% rank2, 4.7% rank3, 4.7% rank4 | User's complex recipe, corrected to sum to 100%. |
| `b20w05r1` | blank 13, wall 3, rank1 48 | 20.3% blank, 4.7% wall, 75.0% rank1 | Anchor for comparison. |
| `b30w05r1` | blank 19, wall 3, rank1 42 | 29.7% blank, 4.7% wall, 65.6% rank1 | Higher blank dose without going all the way to 50%. |
| `b20w05top2` | blank 13, wall 3, rank2 16, rank1 32 | 20.3% blank, 4.7% wall, 25.0% rank2, 50.0% rank1 | Top-2 diversity with anchor blank/wall. |
| `b20w05lad4` | blank 13, wall 3, rank1 19, rank2 13, rank3 10, rank4 6 | 20.3% blank, 4.7% wall, 29.7% rank1, 20.3% rank2, 15.6% rank3, 9.4% rank4 | Tests the ladder/tournament-robustness signal. |

Size:

```text
10 recipes * 2 noise settings * 2 immortal settings = 40 runs
```

That is the fixed Grid B. It is not merely a side lane; it is the direct answer
to the slot-configuration question.

## Side Controls To Keep Out Of The Main Cross

These are useful later or as one-off checks, but they are not central enough to
cross with everything immediately:

- `b20r1`: isolates whether the wall-avoidant opponent helps
  at all.
- `w05r1`: isolates whether blank scaffolding helps at all.
- `b20w15r1`: smoother wall-dose curve if wall pressure looks
  useful.
- checkpoint-boundary opponent refresh versus fixed-interval refresh:
  important, but it changes the learning environment over time.
- bonus-scale bump: keep bonus on, but do not silently change its scale.

## How To Judge The Runs

Do not over-read own reward. If the tournament feeds stronger opponents back
into training, reward can drop while the system is getting harder. Use reward
as a diagnostic, not the main success metric.

Judge each recipe with:

1. Matched survival over checkpoint time:
   - common-iteration AUC;
   - matched endpoint survival;
   - best-so-far survival;
   - retention from best to latest.
2. Tournament strength:
   - best rank per run;
   - top-10 / top-30 / top-100 presence;
   - deduped by run or setting so sibling checkpoints do not dominate;
   - number of games played and opponent coverage.
3. Diagnostic reward breakdown:
   - survival component;
   - bonus component;
   - outcome component;
   - whether drops line up with opponent refresh.

## Implementation Requirements Before Launch

- Recipe definitions must be structured data, not parsed from recipe names.
- Pure controls must be first-class recipes.
- Pure `r1` must fail clearly if no rank1 leaderboard checkpoint is available;
  `b100` and `w100` must not require leaderboard state.
- `leaderboard_immortal_probability` must stay separate from policy identity.
- The manifest must record intended percentages and realized 64-slot counts.
- The collector assignment must repeat the 64-slot bag four times for the
  current 256-env collection wave, then deterministically shuffle.
- Learner `batch_size=64` must stay unchanged; exact per-gradient recipe
  proportions are not guaranteed without future stratified replay sampling.
- The realized total immortal mass must be logged, because p10 means different
  total immortal exposure for different recipes.
- Assignment logs must include recipe id, component list, assignment SHA,
  refresh index, selected opponent kind, selected checkpoint/rank, and final
  immortal flag.
- The final launch artifact must say exactly which grid a row belongs to:
  `grid_a`, `grid_b`, or `side_control`.

Current code status: `scripts/build_curvytron_next_batch_manifest.py` is the
current `cz26` builder for this plan. It emits the locked 96-row Grid A, the
locked 40-row Grid B, and a one-row canary. The old
`scripts/build_curvytron_tonight18_manifest.py` is historical evidence only and
must not be used for the next launch.

Current validation status: the builder contract tests prove row counts, concise
names, Grid B `out50`, canary `cz26c`, shared rank-1 initial checkpoint seeding,
assignment pointer coverage, and leaderboard immortality as a separate slot
flag. Submitter dry-run tests prove the same manifest publishes assignment
artifacts, refresh pointers, and the training-candidate scheduler config.

Reward-alpha implementation status: `reward_outcome_alpha` now exists in the
source-state visual survival env and Modal trainer config. The terminal outcome
term is:

```text
sparse_outcome * reward_outcome_alpha * accumulated_non_outcome_training_reward
```

So `alpha=0.0` is survival+bonus only, `alpha=1.0` can cancel a loss back to
zero but should not make total episode return negative, and intermediate alphas
interpolate cleanly. Targeted tests currently cover env behavior and Modal
config plumbing.

Current seed contract: every new Grid A/Grid B/canary trainer starts from the
old overnight r18fresh leaderboard rank-1 checkpoint. Do not mix raw top-10 or
deduped top-10 as initial policy seeds for this next launch.

Current refresh contract: keep periodic opponent refresh at `2000` learner
iterations for now. Checkpoint-boundary refresh remains a later side-control
idea.
