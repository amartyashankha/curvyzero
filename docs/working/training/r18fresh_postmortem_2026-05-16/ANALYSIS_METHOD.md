# Analysis Method

The goal is not to find a single flattering number. The goal is to understand
which settings produced useful policies, where they regressed, and which
policies should seed the next batch.

## Fair Comparison Axes

Primary axis:

- Matched eval checkpoint iteration. All 18 runs have exact evals through
  `iteration_240000`, so compare common grid points first.

Secondary axes:

- AUC over common eval grid.
- Best eval survival up to the common endpoint.
- Retention from best-so-far to the common endpoint.
- Checkpoint production count and elapsed time.
- Tournament rank/Elo with games and distinct-opponent counts.

Use latest checkpoint only as an operational signal. Latest is not a fair
primary quality comparison because runs reached different latest iterations and
different exposure in the tournament.

## Signals To Keep Separate

- Trainer/eval survival.
- Trainer/eval `mean_training_reward`, compared only within the same reward
  variant.
- Tournament game duration.
- Tournament head-to-head rank/Elo.
- Best rank ever reached.
- Latest-for-run rank.
- Checkpoint production rate.
- Assignment refresh and trainer consumption proof.

## Current Matched Eval Readout

Matched eval survival at `iteration_240000`:

| Slice | Rows | Mean Survival |
| --- | ---: | ---: |
| All | 18 | 189.6 |
| `sparse_outcome` | 6 | 197.6 |
| `survival_plus_bonus_no_outcome` | 6 | 175.1 |
| `survival_plus_bonus_plus_outcome` | 6 | 196.0 |
| `blank10-wall10-rank2_25-rank1_55` | 6 | 173.1 |
| ladder recipe | 6 | 156.0 |
| `blank20-wall5-rank1_70-rank1imm5` | 6 | 239.6 |
| clean | 9 | 183.9 |
| `so10rep10` | 9 | 195.3 |

Common-grid all-run means:

| Eval Iteration | Mean Survival |
| ---: | ---: |
| 0 | 160.2 |
| 50k | 167.6 |
| 100k | 171.5 |
| 150k | 178.4 |
| 200k | 173.7 |
| 240k | 189.6 |

## Interpretation Rules

- The `blank20-wall5-rank1_70-rank1imm5` recipe is the strongest matched
  survival slice so far.
- `sparse_outcome` and `survival_plus_bonus_plus_outcome` are roughly tied on
  matched survival at 240k, but tournament top ranks favor plus-outcome.
- `survival_plus_bonus_no_outcome` trails on matched survival.
- Ladder is weak on matched survival despite some tournament placements.
- `so10rep10` helped matched survival in this batch, but it muddies credit
  assignment and should remain paired with clean controls.

## Reward Metric Semantics

The eval status field is `mean_training_reward`, not a generic reward. It is the
scalar return under that run's training reward variant.

- Do not compare `mean_training_reward` across reward variants.
- For `sparse_outcome`, reward is sparse ego outcome; survival and bonus
  components are zero.
- For `survival_plus_bonus_no_outcome`, reward is survival plus bonus. Outcome
  is telemetry only.
- For `survival_plus_bonus_plus_outcome`, reward is survival plus bonus plus a
  scaled terminal outcome. `mean_reward_components` only exposes survival and
  bonus, so infer the terminal outcome contribution from the residual:
  `mean_training_reward - survival - bonus`.

## Missing Or Confounded Data

- Trainer progress does not cleanly expose actual env-step survival for every
  running row.
- Wall-clock comparisons are partial and unbalanced.
- Tournament ranks are confounded by game/opponent exposure, especially for
  latest checkpoints.
- Tournament duration data exists in battle summaries but must be analyzed
  separately from Elo/rank.
