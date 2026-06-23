# Detailed Run Stocktake

Captured: 2026-05-16.

Sources:

- Eval curves: `/tmp/r18fresh_eval_status.json`, pulled from
  `lightzero_curvytron_run_status --output eval-json`.
- Tournament ratings: `round-000035` from
  `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`.
- Manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json`.

## Glossary

- `sparse`: `sparse_outcome`.
- `no_out`: `survival_plus_bonus_no_outcome`.
- `plus_out`: `survival_plus_bonus_plus_outcome`.
- `r2/r1`: `blank10-wall10-rank2_25-rank1_55`.
- `ladder`: `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5`.
- `b20/r1imm`: `blank20-wall5-rank1_70-rank1imm5`.
- `so10`: `straight_override_p10_repeat_p10`.
- `best eval`: best survival eval on the matched 0..240k grid.
- `best tournament`: best current rank among all checkpoints from that run.
- `latest tournament`: the row marked `latest_for_run=true` in the current
  tournament ratings, if present.

## Knobs Actually Used

These are historical facts about the completed r18fresh batch, not launch
defaults for the next broad lane. Current broad launch defaults live in
`CURRENT_LAUNCH_DEFAULTS.md` and `src/curvyzero/contracts/curvytron.py`.

Common across the completed r18fresh 18-run batch:

- Historical compute lane: `gpu-h100-cpu40`.
- Historical size knobs: `collector_env_num=256`, `num_simulations=8`,
  `batch_size=32`.
- `save_ckpt_after_iter=10000`.
- Learner seat mode: `random_per_episode`.
- Observation surface: `browser_lines + simple_symbols`.
- Three reward variants x three opponent recipes x clean/so10 stochasticity.

## Current Tournament State

`round-000035` has `573` rated rows and `100` active rows. It is still
`stable=false`, with `max_abs_delta=87.80`.

The current rank 1 is:

```text
run: curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423
checkpoint: iteration_260000
rank: 1
rating: 1663.5
games: 1470
distinct opponents: 64
```

The stale public/trainer snapshot remains `auto-r000032-g22-555c999b`, so the
public seed snapshot has not caught up to this round yet.

## Per-Run Survival And Tournament Table

| Row | Reward | Recipe | Noise | Eval Shape | Best Eval | 240k | Latest Eval | Best Tournament | Latest Tournament | Top100 Rows |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| r001 | sparse | r2/r1 | clean | decline by 240k | 228 @ 180k | 151 | 118 @ 308k | #92 @ 0k | #310 @ 308k | 1 |
| r002 | sparse | r2/r1 | so10 | mid-run peak then regression | 242 @ 120k | 187 | 152 @ 306k | #90 @ 10k | #459 @ 306k | 1 |
| r003 | sparse | ladder | clean | flat/noisy | 198 @ 120k | 127 | 119 @ 301k | #27 @ 40k | #484 @ 301k | 6 |
| r004 | sparse | ladder | so10 | decline by 240k | 208 @ 160k | 139 | 157 @ 304k | #48 @ 110k | #291 @ 304k | 3 |
| r005 | sparse | b20/r1imm | clean | steady/moderate growth | 285 @ 240k | 285 | 183 @ 303k | #24 @ 0k | #120 @ 303k | 7 |
| r006 | sparse | b20/r1imm | so10 | strong sustained to 240k | 305 @ 210k | 296 | 245 @ 310k | #30 @ 230k | #162 @ 310k | 7 |
| r007 | no_out | r2/r1 | clean | mid-run peak then regression | 202 @ 230k | 159 | 166 @ 307k | #32 @ 200k | #473 @ 307k | 1 |
| r008 | no_out | r2/r1 | so10 | mid-run peak then regression | 251 @ 160k | 172 | 177 @ 300k | #18 @ 300k | #192 @ 300k | 8 |
| r009 | no_out | ladder | clean | decline by 240k | 244 @ 120k | 176 | 154 @ 308k | #68 @ 10k | #487 @ 308k | 1 |
| r010 | no_out | ladder | so10 | mid-run peak then regression | 231 @ 50k | 168 | 147 @ 308k | #59 @ 190k | #105 @ 308k | 4 |
| r011 | no_out | b20/r1imm | clean | flat/noisy, late recovery | 222 @ 130k | 166 | 244 @ 280k | #14 @ 270k | #55 @ 280k | 5 |
| r012 | no_out | b20/r1imm | so10 | flat/noisy | 264 @ 80k | 210 | 203 @ 303k | #60 @ 140k | #252 @ 303k | 3 |
| r013 | plus_out | r2/r1 | clean | mid-run peak then regression | 213 @ 150k | 170 | 183 @ 306k | #34 @ 170k | #336 @ 306k | 2 |
| r014 | plus_out | r2/r1 | so10 | flat/noisy | 214 @ 140k | 200 | 179 @ 300k | #54 @ 290k | #244 @ 300k | 7 |
| r015 | plus_out | ladder | clean | steady/moderate growth | 185 @ 80k | 174 | 175 @ 301k | #4 @ 150k | #100 @ 301k | 8 |
| r016 | plus_out | ladder | so10 | peaked, dipped, late recovery | 249 @ 140k | 152 | 187 @ 290k | #13 @ 180k | #21 @ 300k | 16 |
| r017 | plus_out | b20/r1imm | clean | mid-run peak then regression | 298 @ 140k | 248 | 276 @ 290k | #57 @ 130k | #424 @ 290k | 4 |
| r018 | plus_out | b20/r1imm | so10 | peaked, dipped, late recovery | 287 @ 190k | 232 | 290 @ 270k | #1 @ 260k | #6 @ 270k | 16 |

## Knob-Level Readout

### Reward Variant

| Reward | Avg AUC | Avg Top100 Rows | Top10 Rows | Top30 Rows | Best Rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse | 169.9 | 4.2 | 0 | 3 | 24 |
| no_out | 164.6 | 3.7 | 0 | 3 | 14 |
| plus_out | 179.5 | 8.8 | 10 | 24 | 1 |

Survival says sparse and plus-outcome are both real. Tournament says
plus-outcome is much stronger head-to-head in this pool. No-outcome does not
look like a primary arm.

### Opponent Recipe

| Recipe | Avg AUC | Avg Top100 Rows | Top10 Rows | Top30 Rows | Best Rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| r2/r1 | 161.3 | 3.3 | 0 | 1 | 18 |
| ladder | 154.2 | 6.3 | 1 | 12 | 4 |
| b20/r1imm | 198.5 | 7.0 | 9 | 17 | 1 |

Survival strongly favors `b20/r1imm`. Tournament also favors it, but the ladder
has some tournament strength despite weak matched survival. The simple `r2/r1`
recipe is weakest overall.

### Stochasticity

| Noise | Avg AUC | Avg Top100 Rows | Top10 Rows | Top30 Rows | Best Rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 165.3 | 3.9 | 1 | 6 | 4 |
| so10 | 177.3 | 7.2 | 9 | 24 | 1 |

`so10` helped in this batch by both survival AUC and tournament placement. It
should still be treated carefully because it changes the action process and can
muddy credit assignment.

## Main Patterns

1. The batch did learn, but often did not retain.
   Many runs have a mid-run best checkpoint and a worse latest checkpoint.

2. Tournament ranking is doing useful work.
   It surfaces mid-run checkpoints that the latest-checkpoint view would miss.

3. Survival and tournament agree on the best broad region.
   `plus_out + b20/r1imm + so10` produced the current tournament champion and
   strong survival.

4. Survival and tournament are not identical.
   Sparse can survive well at matched eval, especially with `b20/r1imm`, but it
   is not winning the head-to-head tournament top band.

5. Raw top 10 is too concentrated.
   r018 owns most of the current top band. The next batch should preserve the
   champion but probably dedupe seeds by run or setting.

6. Latest checkpoint is the wrong default promotion unit.
   Use best-so-far, tournament-selected, or heldout-selected checkpoints.

7. Tournament rank is not the same as own reward.
   In the current joined checkpoint-level read, rating correlates moderately
   with survival, but weakly or negatively with own reward. For plus-outcome,
   the tournament top rows often still have negative scaled-outcome residuals.

## Questions For Deeper Analysis

- Why did the public/trainer leaderboard publish remain at generation 22 while
  rating latest advanced to round 35?
- Does r018 dominate because of true policy strength, easier matchup exposure,
  or both?
- For each run, does tournament rank track eval survival, or does head-to-head
  strength favor different behaviors?
- For initial policy weights, the next launch is locked to one champion seed:
  the old-overnight rank-1 checkpoint. Diverse challengers can still be used
  as explicit opponent/audit material.
- How should own-reward drops trigger best-checkpoint preservation or branching?
- Is plus-outcome's scaled terminal outcome too volatile for stable learning, or
  is it the useful pressure that produced tournament strength?
