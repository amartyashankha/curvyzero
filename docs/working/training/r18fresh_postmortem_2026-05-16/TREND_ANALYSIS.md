# Trend Analysis

Captured: 2026-05-16.

This file answers the current analysis question: what actually happened over
time in the 18-run `r18fresh` batch, using matched checkpoints and the repo's
existing eval-curve tooling.

## Sources And Tooling

- Raw status pull: `/tmp/r18fresh_eval_status.json`.
- Cleaned status JSON: `/tmp/r18fresh_eval_status_clean.json`.
- Existing curve scorer:

```bash
uv run python scripts/analyze_curvytron_eval_curves.py \
  /tmp/r18fresh_eval_status_clean.json \
  --metric mean_survival,mean_training_reward \
  --format json \
  --output /tmp/r18fresh_eval_curve_scores.json
```

The raw status file contains Modal startup/shutdown text around the JSON array,
so the repo tool needs the cleaned JSON slice. The data content is unchanged.

The manifest source of truth for axes is:

```text
artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json
```

Use manifest fields for axes. Do not infer `clean`/`so10` only from run ID text;
some run IDs shorten the noisy label.

## Exact Opponent Recipes

These are the slot weights tested in the 18-run matrix. Each episode samples
one slot from the configured recipe.

Important caveat: the original `r18fresh` assignment bank was built with
`scratch_bootstrap=true`. In that manifest, the rank-labelled slots are
scratch-bootstrap placeholders implemented as proactive wall-avoidant policies
until the control-plane refresh pointer supplies updated assignment contents.
So the recipe names describe the intended slot layout, while the actual policy
behind a rank slot can change after assignment refresh.

### Recipe A: Simple Top-Heavy

Manifest id: `blank10-wall10-rank2_25-rank1_55`.

| Slot | Weight | Policy Kind | Immortal | Notes |
| --- | ---: | --- | --- | --- |
| blank | 10% | fixed straight | yes | blank-canvas no-op runtime |
| wall avoider | 10% | proactive wall-avoidant | yes | safe margin 20 |
| rank 2 | 25% | rank-2 slot | no | scratch placeholder at bootstrap; refreshable |
| rank 1 | 55% | rank-1 slot | no | scratch placeholder at bootstrap; refreshable |

### Recipe B: Ladder

Manifest id:
`blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5`.

| Slot | Weight | Policy Kind | Immortal | Notes |
| --- | ---: | --- | --- | --- |
| blank | 10% | fixed straight | yes | blank-canvas no-op runtime |
| wall avoider | 10% | proactive wall-avoidant | yes | safe margin 20 |
| rank 4 | 10% | rank-4 slot | no | scratch placeholder at bootstrap; refreshable |
| rank 3 | 15% | rank-3 slot | no | scratch placeholder at bootstrap; refreshable |
| rank 2 | 20% | rank-2 slot | no | scratch placeholder at bootstrap; refreshable |
| rank 1 | 30% | rank-1 slot | no | scratch placeholder at bootstrap; refreshable |
| rank 1 immortal | 5% | rank-1 slot | yes | scratch placeholder at bootstrap; refreshable |

### Recipe C: Blank20/Rank1

Manifest id: `blank20-wall5-rank1_70-rank1imm5`.

| Slot | Weight | Policy Kind | Immortal | Notes |
| --- | ---: | --- | --- | --- |
| blank | 20% | fixed straight | yes | blank-canvas no-op runtime |
| wall avoider | 5% | proactive wall-avoidant | yes | safe margin 20 |
| rank 1 | 70% | rank-1 slot | no | scratch placeholder at bootstrap; refreshable |
| rank 1 immortal | 5% | rank-1 slot | yes | scratch placeholder at bootstrap; refreshable |

## Glossary

- `survival`: mean eval steps survived.
- `own reward`: `mean_training_reward`, using that run's reward function. This
  is only directly comparable within the same reward variant.
- `sparse`: terminal win/loss only. A win is positive, a loss is negative, and
  living longer does not directly add reward.
- `no_out`: survival plus bonus pickup reward, with terminal win/loss recorded
  as telemetry but not added to the training reward.
- `plus_out`: survival plus bonus pickup reward plus terminal win/loss. The
  terminal result is scaled by episode source step count, so a late loss can
  subtract roughly the survival length.
- `r2/r1`: opponent mixture with 10% blank canvas, 10% immortal wall avoider,
  25% rank-2 frozen checkpoint, and 55% rank-1 frozen checkpoint.
- `ladder`: opponent mixture with 10% blank canvas, 10% immortal wall avoider,
  10% rank-4 checkpoint, 15% rank-3, 20% rank-2, 30% rank-1, and 5% immortal
  rank-1.
- `b20/r1imm`: opponent mixture with 20% blank canvas, 5% immortal wall
  avoider, 70% rank-1 frozen checkpoint, and 5% immortal rank-1.
- `clean`: no extra action noise.
- `so10`: 10% straight-action override plus 10% held-action repeat; this is the
  stochastic/noisy action setting.
- `AUC`: average value over matched checkpoints `0..240k` in `10k` steps. It is
  not a calculus integral here; it is the matched-grid average.
- `latest`: the latest eval checkpoint in the status pull, often around
  `270k..310k`, not the same iteration for every run.
- `best`: best checkpoint seen anywhere in that run's eval curve.
- `speed`: checkpoint-production proxy from first/last checkpoint mtimes and
  latest iteration.

All 18 runs used the same `batch_size=32`, `collector_env_num=256`, and
`save_ckpt_after_iter=10000`, so matched iteration is a fair first comparison
inside this batch. If a later batch changes batch size, compare by learner
updates, environment frames, and wall-clock speed separately.

## High-Level Survival Trend

Across all 18 runs on the common `0..240k` grid:

| Iteration | Mean Survival |
| ---: | ---: |
| 0 | 160.2 |
| 50k | 167.6 |
| 100k | 171.5 |
| 150k | 178.4 |
| 200k | 173.7 |
| 240k | 189.6 |
| latest eval | 186.4 |
| per-run best mean | 251.3 |

Read: the batch did learn some survival. It did not keep most of the gains.
The average run passed through a much better intermediate checkpoint than its
latest checkpoint.

Retention counts:

- Latest survival is within 90% of the run's best survival in `3/18` runs.
- Latest own reward is within 10% of the run's best own reward in `1/18` runs.
- The common pattern is therefore not "no learning"; it is "learning appears,
  then often regresses."

## Reward Variant Trend

| Reward Variant | Runs | Survival AUC | Survival 240k | Survival Latest | Survival Best | Iter/hr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sparse` | 6 | 170.2 | 197.6 | 162.2 | 260.4 | 37,898 |
| `no_out` | 6 | 164.6 | 175.1 | 181.7 | 239.2 | 29,949 |
| `plus_out` | 6 | 179.6 | 196.0 | 215.1 | 254.3 | 26,644 |

Read:

- `plus_out` has the best integrated survival and best latest survival.
- `sparse` is competitive at exactly `240k`, but its latest checkpoints regress
  harder.
- `no_out` is easiest to read because its own reward tracks survival almost
  exactly, but it is weaker on average than `plus_out` in this batch.
- The faster variants were not automatically better. `sparse` produced more
  iterations per hour, while `plus_out` produced the strongest tournament band.

## Opponent Recipe Trend

| Opponent Recipe | Runs | Survival AUC | Survival 240k | Survival Latest | Survival Best | Iter/hr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r2/r1` | 6 | 161.0 | 173.1 | 162.5 | 230.9 | 32,315 |
| `ladder` | 6 | 154.3 | 156.0 | 156.4 | 225.8 | 31,176 |
| `b20/r1imm` | 6 | 199.1 | 239.6 | 240.2 | 297.2 | 31,001 |

Read: the clearest survival signal is the `b20/r1imm` recipe. It wins on
matched-grid average, `240k`, latest, and best survival. This is the strongest
single knob-level effect in the batch.

## Noise Trend

| Noise Mode | Runs | Survival AUC | Survival 240k | Survival Latest | Survival Best | Iter/hr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | 9 | 165.7 | 183.9 | 179.7 | 244.2 | 30,750 |
| `so10` | 9 | 177.2 | 195.2 | 193.1 | 258.4 | 32,244 |

Read: `so10` helped on average in this batch. It was not uniformly better in
every matched pair, but the aggregate direction is positive.

## One-Knob Noise Comparisons

These compare `clean -> so10` while holding reward and opponent recipe fixed.

| Reward | Recipe | Delta AUC | Delta 240k | Delta Latest | Delta Iter/hr |
| --- | --- | ---: | ---: | ---: | ---: |
| `sparse` | `r2/r1` | +31.0 | +36.1 | +33.5 | -2,786 |
| `sparse` | `ladder` | +29.6 | +12.5 | +38.5 | +9,041 |
| `sparse` | `b20/r1imm` | +2.5 | +10.8 | +62.8 | +2,052 |
| `no_out` | `r2/r1` | +3.4 | +12.6 | +10.5 | +7,779 |
| `no_out` | `ladder` | -6.2 | -7.2 | -6.6 | +1,196 |
| `no_out` | `b20/r1imm` | +59.6 | +44.1 | -40.5 | +1,720 |
| `plus_out` | `r2/r1` | +19.5 | +30.0 | -3.6 | -3,031 |
| `plus_out` | `ladder` | +17.6 | -21.6 | +12.4 | -393 |
| `plus_out` | `b20/r1imm` | -53.0 | -15.1 | +13.6 | -2,135 |

Read: noise is mostly helpful, but not a law. The most important exception is
`plus_out + b20/r1imm`: the noisy run wins latest survival and tournament rank,
while the clean run has better matched AUC and `240k` survival.

## Own Reward Trend

Own reward is not one global metric:

- In `sparse`, own reward is a sparse outcome scalar. It is small, noisy, and
  does not track survival.
- In `no_out`, own reward is almost survival. Reward and survival move together.
- In `plus_out`, own reward is survival plus bonus plus scaled terminal
  outcome. The outcome term is volatile and can erase survival gains.

Observed reward retention:

- `sparse`: latest reward near best in `0/6`, latest survival near best in
  `0/6`.
- `no_out`: latest reward near best in `1/6`, latest survival near best in
  `1/6`.
- `plus_out`: latest reward near best in `0/6`, latest survival near best in
  `2/6`.

The reward story is therefore sharper than the survival story: most runs had a
better intermediate checkpoint under their own scalar reward too.

## Tournament Trend

Current joined tournament snapshot:

- Rating latest: `round-000035`.
- Rated rows: `573`.
- Active trainer-facing rows: `100`.
- Stability flag: `stable=false`, `max_abs_delta=87.80`.
- Joined rating rows to eval checkpoints: `572/573`.

Tournament rating correlation:

| Scope | Rows | Survival Corr | Own Reward Corr |
| --- | ---: | ---: | ---: |
| all matched rating rows | 572 | 0.431 | -0.009 |
| active top-100 rows | 99 | 0.302 | -0.114 |

Read:

- Tournament rank is moderately related to survival, but it is not just
  survival.
- Tournament rank is not generally aligned with own reward.
- This is expected if the tournament is measuring head-to-head strength rather
  than the training reward scalar.

Tournament game duration moved upward over the rating rounds:

| Measure | Value |
| --- | ---: |
| Round 0 weighted mean duration | 131.16 physical steps |
| Round 35 weighted mean duration | 162.20 physical steps |
| Change | +31.04 steps |
| Relative change | +23.7% |
| First 5 rounds mean | 131.59 |
| Last 5 rounds mean | 159.78 |
| Round-index correlation | 0.945 |

Read: the tournament pool itself is trending toward longer games, even though
many individual training runs regress at latest. That is consistent with the
tournament preserving better intermediate checkpoints and promoting them into
the active pool.

The current top band is dominated by `plus_out`, especially r018:

```text
run: curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423
best tournament checkpoint: iteration_260000
current rank: #1
latest eval checkpoint in that run: iteration_270000
latest eval survival: 290.0
```

That top checkpoint is not simply the highest own-reward checkpoint in its run.
Tournament, survival, and own reward are separate views of policy quality.

## Own-Latest Control Lane

There is a separate no-tournament/own-latest control manifest:

```text
artifacts/local/curvytron_tonight18_manifests/curvy-ownlatest-staticmix-20260516b/curvy-ownlatest-staticmix-20260516b.json
```

Only three rows were launched from it in
`curvy-ownlatest-staticmix-20260516b.selected3.launch.json`. All three are
clean `survival_plus_bonus_no_outcome` rows with
`own_checkpoint_opponent_refresh_enabled=true`.

| Row | Recipe | Eval Points | Latest Iter | First S | Latest S | Best S | Best Iter |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r007 | `r2/r1` | 5 | 40k | 200.9 | 169.5 | 200.9 | 0k |
| r009 | `ladder` | 5 | 40k | 116.0 | 134.2 | 183.2 | 30k |
| r011 | `b20/r1imm` | 8 | 70k | 155.0 | 148.0 | 202.6 | 40k |

Matched against the original r18fresh clean/no-outcome rows at the same early
iteration:

| Recipe | Control Latest Iter | Control Latest S | Control Best S | r18 S Same Iter | r18 Best <= Same Iter | r18 240k | r18 Latest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r2/r1` | 40k | 169.5 | 200.9 | 129.0 | 148.2 | 159.0 | 166.1 |
| `ladder` | 40k | 134.2 | 183.2 | 130.4 | 206.6 | 175.5 | 153.6 |
| `b20/r1imm` | 70k | 148.0 | 202.6 | 118.5 | 159.0 | 166.1 | 243.8 |

Read: this control is still early, so it is not an apples-to-apples final
comparison. It does show the same basic shape: a useful intermediate checkpoint
followed by a worse latest checkpoint. That means the retention problem is not
obviously caused only by tournament feedback.

## Per-Run Shape Summary

| Row | Reward | Recipe | Noise | S0 | S120k | S240k | Latest S | Best S | Best Iter | Latest Reward | Best Reward |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r001 | sparse | r2/r1 | clean | 179.4 | 154.9 | 151.1 | 118.1 | 227.5 | 180k | -0.6 | 0.0 |
| r002 | sparse | r2/r1 | so10 | 127.4 | 241.8 | 187.2 | 151.6 | 241.8 | 120k | -0.5 | -0.2 |
| r003 | sparse | ladder | clean | 134.6 | 198.2 | 126.6 | 118.6 | 198.2 | 120k | -0.8 | -0.2 |
| r004 | sparse | ladder | so10 | 166.1 | 132.1 | 139.1 | 157.1 | 207.6 | 160k | -0.5 | 0.0 |
| r005 | sparse | b20/r1imm | clean | 207.8 | 218.1 | 285.2 | 182.6 | 346.1 | 260k | -1.0 | -0.5 |
| r006 | sparse | b20/r1imm | so10 | 131.0 | 201.5 | 296.0 | 245.4 | 341.4 | 290k | -0.8 | -0.2 |
| r007 | no_out | r2/r1 | clean | 130.9 | 140.0 | 159.0 | 166.1 | 201.9 | 230k | 165.2 | 201.4 |
| r008 | no_out | r2/r1 | so10 | 92.1 | 142.6 | 171.6 | 176.6 | 251.2 | 160k | 176.0 | 250.6 |
| r009 | no_out | ladder | clean | 197.9 | 243.5 | 175.5 | 153.6 | 243.5 | 120k | 152.6 | 242.6 |
| r010 | no_out | ladder | so10 | 149.9 | 192.5 | 168.2 | 147.0 | 230.9 | 50k | 146.1 | 230.1 |
| r011 | no_out | b20/r1imm | clean | 159.0 | 112.1 | 166.1 | 243.8 | 243.8 | 280k | 242.8 | 242.8 |
| r012 | no_out | b20/r1imm | so10 | 195.5 | 240.2 | 210.2 | 203.2 | 264.0 | 80k | 202.2 | 263.0 |
| r013 | plus_out | r2/r1 | clean | 125.6 | 203.5 | 169.9 | 183.0 | 213.1 | 300k | -1.0 | 134.2 |
| r014 | plus_out | r2/r1 | so10 | 164.0 | 167.1 | 199.9 | 179.4 | 249.6 | 290k | 97.0 | 236.1 |
| r015 | plus_out | ladder | clean | 117.5 | 177.9 | 174.1 | 174.8 | 225.4 | 280k | -1.0 | 95.4 |
| r016 | plus_out | ladder | so10 | 179.2 | 159.6 | 152.5 | 187.1 | 249.4 | 140k | 70.0 | 84.2 |
| r017 | plus_out | b20/r1imm | clean | 228.1 | 262.4 | 247.5 | 276.4 | 298.1 | 140k | 96.6 | 184.2 |
| r018 | plus_out | b20/r1imm | so10 | 198.1 | 192.9 | 232.4 | 290.0 | 290.0 | 270k | 80.2 | 125.4 |

## Bottom Line

The high-level trend is not "nothing worked" and not "everything worked." The
batch produced real intermediate improvements, especially in the `b20/r1imm`
recipe and in the `plus_out` tournament top band. The failure mode is retention:
most latest checkpoints are worse than the best checkpoint that run already
visited, and own reward usually shows the same collapse.

The most robust signal is:

```text
opponent recipe: b20/r1imm
reward region: plus_out for tournament strength, sparse as a survival diagnostic
best single run: r018, plus_out + b20/r1imm + so10
main failure: latest-checkpoint regression after a better intermediate policy
```
