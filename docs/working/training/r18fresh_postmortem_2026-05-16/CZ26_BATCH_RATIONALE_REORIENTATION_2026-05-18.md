# CZ26 Batch Rationale Reorientation - 2026-05-18

This is the current plain-language reconstruction of why we launched
`cz26-full-20260517a`, what evidence it came from, and what was still uncertain.

## Short Answer

We launched `cz26-full-20260517a` because the prior 18-run `r18fresh` batch
showed real learning signal, but the winning settings were confounded.

The prior batch told us:

- policies often improved in the middle of training, then regressed later;
- the tournament was useful because it preserved strong mid-run checkpoints;
- the strongest survival and tournament region involved:
  - plus-outcome reward;
  - the `b20/r1imm` opponent recipe;
  - some action stochasticity;
  - a mid-run checkpoint from r018 as the best shared seed.

It did not tell us which part of that recipe caused the improvement. `cz26`
was built to separate those causes.

## Plain Glossary

- **Grid A**: the broad 96-run matrix. It tests several mostly production-like
  opponent mixtures across reward strength, action noise, and checkpoint-opponent
  immortality.
- **Grid B**: the 40-run slot-focused matrix. It tests what kind of opponent
  population helps: pure blank, pure wall-avoider, pure rank-1 leaderboard
  opponent, and several mixtures.
- **Slot recipe**: the bag of opponent types sampled by the trainer. Example:
  `b20w05r1` means roughly 20% blank immortal opponents, 5% wall-avoidant
  immortal opponents, and the rest rank-1 leaderboard opponents.
- **Blank opponent**: a hard-coded empty/no-op opponent. It is immortal.
- **Wall-avoidant opponent**: a hard-coded opponent that mostly avoids walls.
  It is immortal.
- **Leaderboard opponent**: a checkpoint pulled from the tournament ranking.
  Its policy identity is separate from whether this sampled copy is immortal.
- **Leaderboard immortality**: a probability that a sampled leaderboard
  checkpoint opponent is made immortal for that episode.
- **Reward alpha**: how much terminal win/loss outcome matters, on top of
  survival and bonus reward. `out0` is no terminal outcome pressure; `out100`
  is full terminal outcome pressure.
- **Noise**: extra action stochasticity during training. `n10` means 10%
  straight-action override plus 10% held-action repeat.

## The Three Signals We Were Looking At

The prior analysis deliberately kept three signals separate:

1. **Training reward progression**
   - Useful as a diagnostic.
   - Hard to use as the main score because tournament-fed opponents can get
     stronger over time, which can make reward fall even when the curriculum is
     getting harder in a useful way.
   - Own reward is only directly comparable within the same reward definition.

2. **Survival eval**
   - Cleaner for comparing learning progress across checkpoints.
   - Prior matched analysis used common checkpoints from `0..240k`.
   - This showed real improvement, but weak retention: many best checkpoints
     were intermediate, not latest.

3. **Tournament performance**
   - The actual head-to-head signal.
   - It moderately tracked survival, but was not the same thing as survival.
   - It did not track own reward well, which is expected if the tournament is
     measuring policy strength rather than the scalar training objective.

## What r18fresh Actually Showed

The 18-run `r18fresh` batch was:

```text
3 reward variants * 3 opponent recipes * 2 noise settings = 18 runs
```

Main matched survival result:

- mean survival rose from about `160` at iteration `0` to about `190` at
  matched `240k`;
- mean per-run best survival was about `251`;
- latest mean survival was only about `186`.

Plain read: there was learning, but many runs did not retain their best policy.

### Reward Evidence

Prior aggregate read:

- `plus_out` had the best survival AUC and strongest tournament top-band signal.
- `sparse` was competitive at `240k`, but latest checkpoints regressed harder.
- `no_out` was easy to read because own reward tracked survival, but it looked
  weaker as a primary arm.

Important uncertainty:

- The old plus-outcome scale could make late losses too punitive. The next
  batch therefore turned reward outcome into an explicit alpha axis instead of
  just using one old `plus_out` setting.

### Slot Recipe Evidence

Prior recipes:

| Short name | Historical recipe | Plain meaning |
| --- | --- | --- |
| `r2/r1` | `blank10-wall10-rank2_25-rank1_55` | 10% blank, 10% wall, mostly rank1/rank2 leaderboard |
| `ladder` | `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5` | 10% blank, 10% wall, spread across ranks 1-4, with 5% immortal rank1 |
| `b20/r1imm` | `blank20-wall5-rank1_70-rank1imm5` | 20% blank, 5% wall, mostly rank1, with 5% immortal rank1 |

Prior aggregate read:

- `b20/r1imm` clearly won survival: best AUC, best `240k`, best latest, best
  per-run best.
- It also produced the tournament champion region.
- `ladder` was weak on fixed survival but surprisingly strong in tournament
  placement. That is why ladder-style diversity stayed alive in Grid B.

Important uncertainty:

`b20/r1imm` changed many things at once:

- blank exposure went from 10% to 20%;
- wall-avoidant exposure went from 10% to 5%;
- rank2/rank3/rank4 exposure mostly disappeared;
- rank1 concentration increased;
- total immortal pressure increased.

So the historical data did not prove that "20% blank" alone caused the gain.
That is the main reason Grid A and Grid B exist.

### Noise Evidence

Prior aggregate read:

- `so10` helped on average by survival AUC and tournament placement.
- It was not uniformly better in every matched pair.

Why cz26 tested `n0`, `n10`, and `n20`:

- `n10` looked promising;
- `n20` was exploratory;
- clean was still needed as the control because noise can muddy credit
  assignment.

## Why Grid A Exists

Grid A asks:

```text
Do production-like mixed opponent recipes keep working across reward strength,
noise, and leaderboard-opponent immortality?
```

Exact shape:

```text
4 slot recipes * 4 reward alphas * 3 noise settings * 2 immortality settings = 96 runs
```

Grid A recipes:

| Code | 64-slot bag | Why included |
| --- | --- | --- |
| `b20w05r1` | blank 13, wall 3, rank1 48 | Cleaned historical winner anchor. |
| `b10w05r1` | blank 7, wall 3, rank1 54 | Tests whether extra blank exposure mattered. |
| `b20w10r1` | blank 13, wall 6, rank1 45 | Tests whether lower wall exposure mattered. |
| `b20w05top2` | blank 13, wall 3, rank1 32, rank2 16 | Tests rank1 concentration versus top-2 diversity. |

Grid A axes:

- reward alpha: `out0`, `out33`, `out67`, `out100`;
- noise: `n0`, `n10`, `n20`;
- leaderboard immortality: `imm0`, `imm10`.

## Why Grid B Exists

Grid B asks:

```text
Which opponent population is actually useful?
```

Exact shape:

```text
10 slot recipes * 1 reward alpha * 2 noise settings * 2 immortality settings = 40 runs
```

Grid B fixes reward at `out50`, then spends the budget on slot populations.

Grid B recipes:

| Code | 64-slot bag | Why included |
| --- | --- | --- |
| `b100` | blank 64 | Pure blank control. |
| `w100` | wall 64 | Pure wall-avoidant control. |
| `r1` | rank1 64 | Pure leaderboard rank1 pressure control. |
| `b50r1` | blank 32, rank1 32 | User-proposed heavy blank plus rank1 mixture. |
| `b25w25r1` | blank 16, wall 16, rank1 32 | Hard-coded-heavy mixture split across blank and wall. |
| `b20w20lad4s` | blank 13, wall 13, rank1 19, rank2 13, rank3 3, rank4 3 | More wall plus broader ladder exposure. |
| `b20w05r1` | blank 13, wall 3, rank1 48 | Anchor from Grid A. |
| `b30w05r1` | blank 19, wall 3, rank1 42 | More blank than anchor without going to 50%. |
| `b20w05top2` | blank 13, wall 3, rank1 32, rank2 16 | Top-2 diversity while holding blank/wall fixed. |
| `b20w05lad4` | blank 13, wall 3, rank1 19, rank2 13, rank3 10, rank4 6 | Ladder diversity with anchor blank/wall. |

The pure controls are inside Grid B. They are not a separate third grid in the
136-run manifest.

## Seeding Decision

Every Grid A and Grid B learner starts from the same old r18fresh rank-1
checkpoint:

```text
curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423
iteration_180000.pth.tar
```

Why:

- the prior top 10 was dominated by sibling checkpoints from the same run;
- using one shared seed avoids mixing seed quality into every other axis;
- the experiment is trying to compare reward/noise/slot settings, not compare
  different starting checkpoints.

The raw top-10 remained useful as audit/opponent material, but not as mixed
initial seeds for this launch.

## What Actually Launched

Manifest:

```text
artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.json
```

Actual row count:

```text
96 Grid A rows
40 Grid B rows
136 total rows
```

Actual reward counts:

```text
out0:   24
out33:  24
out50:  40
out67:  24
out100: 24
```

Actual noise counts:

```text
n0:  52
n10: 52
n20: 32
```

Actual leaderboard-immortality counts:

```text
imm0:  68
imm10: 68
```

Actual shared training defaults:

```text
collector_env_num = 256
n_episode = 256
batch_size = 64
num_simulations = 8
max_train_iter = 300000
max_env_step = 30000000
save_ckpt_after_iter = 10000
opponent_assignment_refresh_interval_train_iter = 2000
learner_seat_mode = random_per_episode
policy surface = browser_lines + simple_symbols + cpu_oracle
```

## Controls And Canaries

Controls inside the 136-run batch:

- `b100`;
- `w100`;
- `r1`.

These are Grid B rows.

The 136-run manifest has no separate `control` grid and no canary rows. If a
doc says "control" without context, read it carefully:

- `control:` can mean a Modal control-volume path;
- "control script" can mean orchestration tooling;
- "pure controls" means the Grid B experimental rows above.

Canary outside the 136-run batch:

- `cz26c-e2e-20260516a`;
- one row;
- shorter run;
- checkpoint every 100 learner iterations;
- used for wiring/proof, not for learning conclusions.

Older controls:

- the own-latest/static control runs from 2026-05-16 are historical side
  evidence;
- they are not part of the 136-run `cz26-full-20260517a` manifest.

## What Happened After Launch

The experiment rationale and the infrastructure problems are separate.

What launched correctly:

- the 136 trainer rows were submitted/spawned;
- assignment artifacts and refresh pointers were written;
- all rows used the same pinned rank-1 initial checkpoint.

What went wrong operationally:

- a live tournament intake manifest briefly used giant `all_pairs` scheduling;
- that created a huge game batch and stalled the pipeline;
- later fixes moved live scheduling back to bounded adaptive work;
- later fixes also prevented repeated same-pool rerates and premature skipping
  of useful batches.

What was proven later:

- at least one feedback pass worked end to end:
  trainer checkpoints -> intake -> tournament rating -> trainer-facing export
  -> trainers loading assignments;
- newer batches were later observed producing bounded tournament work.

What remained weak or needed continued proof:

- fully automatic repeated publish/export/consume cycles without manual repair;
- complete per-checkpoint lineage across written -> discovered -> scheduled ->
  rated -> exported -> trainer-loaded;
- browser visual validation;
- tournament GIF generation, which was later found to be off because a
  `save_gif=false` scale-probe setting leaked into live config.

## What The Previous Analysis Did Not Prove

The previous analysis was useful, but it did not answer everything.

Reward gaps:

- reward drops were not tied cleanly to exact opponent-refresh events;
- value loss, value support saturation, policy entropy, and action mix were not
  deeply analyzed;
- plus-outcome residuals were inferred, not always directly logged as a clean
  component.

Survival gaps:

- survival was mostly fixed-eval survival, not broken down by exact opponent
  type;
- retention failure was measured but not causally explained;
- wall-clock and trainer-progress comparisons were partial.

Tournament gaps:

- tournament rank was confounded by how many games and which opponents a
  checkpoint had seen;
- raw top-band rows were sibling-heavy, especially from r018;
- matchup-specific weaknesses were not deeply analyzed;
- rating stability was imperfect, so top-rank order could still move.

This is why the next analysis should use matched comparisons and collapse one
axis at a time, instead of reading one headline leaderboard number.

## How To Analyze This Batch Next

Use all three signals, and keep them separate:

1. Reward progression:
   - compare only within the same reward definition;
   - expect reward to sometimes fall if tournament opponents get stronger;
   - inspect reward components when available.

2. Survival progression:
   - use matched checkpoints where possible;
   - report AUC, matched endpoint, best checkpoint, latest checkpoint, and
     retention from best to latest;
   - when rows differ in progress, avoid naive latest-only comparisons.

3. Tournament performance:
   - track best rank per run and per setting;
   - track top-10/top-30/top-100 presence;
   - dedupe by run/setting so one sibling-heavy run does not dominate;
   - track tournament game duration over time.

For knob-level analysis, collapse the tensor one axis at a time:

- reward alpha while holding recipe/noise/immortality balanced;
- noise while holding reward/recipe/immortality balanced;
- leaderboard immortality while holding reward/noise/recipe balanced;
- slot recipe while holding reward/noise/immortality balanced.

## Current Honest Interpretation

`cz26` was a reasonable follow-up to r18fresh. It was not based on a clean proof
that every knob was right. It was based on a promising but confounded region:

```text
plus-outcome-ish reward
+ b20/rank1-heavy opponent population
+ mild action stochasticity
+ tournament-selected shared seed
```

Grid A tests whether that region is robust across reward/noise/immortality.
Grid B tests what opponent population is actually doing the work.

The main analytical risk is still over-reading one metric. Reward, survival,
and tournament strength are related but not interchangeable.
