# Reward Breakdown Analysis

Captured: 2026-05-16.

Source: `/tmp/r18fresh_eval_status.json`, pulled from
`lightzero_curvytron_run_status --output eval-json`.

## Metric Semantics

- `mean_training_reward` is the eval env's scalar return for the run's own
  reward variant.
- Compare `mean_training_reward` only within the same reward variant.
- Use survival and tournament rank for cross-variant comparisons.
- `mean_reward_components` exposes only `survival` and `bonus`.
- For `survival_plus_bonus_plus_outcome`, the scaled terminal outcome term is
  not listed as a component. Infer it as:

```text
outcome_residual = mean_training_reward - survival_component - bonus_component
```

## Reward Variants

- `sparse_outcome`: own reward is only sparse ego outcome. Survival and bonus
  components are zero by design.
- `survival_plus_bonus_no_outcome`: own reward is survival plus same-step bonus.
  Terminal outcome is telemetry only.
- `survival_plus_bonus_plus_outcome`: own reward is survival plus same-step bonus
  plus scaled terminal outcome. Loss can wipe out survival reward.

## Aggregate Own-Reward Readout

| Variant | Reward AUC | Reward 240k | Latest Reward | Survival AUC | Bonus AUC | Reward/Survival AUC Correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sparse_outcome` | -0.61 | -0.67 | -0.69 | 169.9 | 0.000 | -0.68 |
| `survival_plus_bonus_no_outcome` | 163.77 | 174.29 | 180.83 | 164.6 | 0.023 | 1.00 |
| `survival_plus_bonus_plus_outcome` | 31.94 | 43.85 | 56.98 | 179.5 | 0.009 | 0.70 |

Read:

- Sparse reward is tiny, noisy, and bounded. It does not track survival.
- No-outcome reward is basically survival. That variant is easy to read but
  weaker in tournament.
- Plus-outcome reward is not the same as survival because scaled terminal
  outcome can be strongly negative. It has the best tournament signal, but its
  reward curve is volatile.
- Bonus pickups were almost absent. Bonus reward is not a meaningful
  differentiator in this batch.

## Per-Run Own Reward

| Row | Variant | Best Reward | Latest Reward | Reward Drop | Best Survival | Latest Survival | Survival Drop | Latest Outcome Residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r001 | sparse | 0.0 @ 40k | -0.6 @ 308k | -0.6 | 227.5 @ 180k | 118.1 @ 308k | -109.4 | -0.6 |
| r002 | sparse | -0.2 @ 10k | -0.5 @ 306k | -0.2 | 241.8 @ 120k | 151.6 @ 306k | -90.1 | -0.5 |
| r003 | sparse | -0.2 @ 40k | -0.8 @ 301k | -0.5 | 198.2 @ 120k | 118.6 @ 301k | -79.6 | -0.8 |
| r004 | sparse | 0.0 @ 0k | -0.5 @ 304k | -0.5 | 207.6 @ 160k | 157.1 @ 304k | -50.5 | -0.5 |
| r005 | sparse | -0.5 @ 60k | -1.0 @ 303k | -0.5 | 346.1 @ 260k | 182.6 @ 303k | -163.5 | -1.0 |
| r006 | sparse | -0.2 @ 60k | -0.8 @ 310k | -0.5 | 341.4 @ 290k | 245.4 @ 310k | -96.0 | -0.8 |
| r007 | no_out | 201.4 @ 230k | 165.2 @ 307k | -36.1 | 201.9 @ 230k | 166.1 @ 307k | -35.8 | 0.0 |
| r008 | no_out | 250.6 @ 160k | 176.0 @ 300k | -74.6 | 251.2 @ 160k | 176.6 @ 300k | -74.6 | 0.0 |
| r009 | no_out | 242.6 @ 120k | 152.6 @ 308k | -90.0 | 243.5 @ 120k | 153.6 @ 308k | -89.9 | 0.0 |
| r010 | no_out | 230.1 @ 50k | 146.1 @ 308k | -84.0 | 230.9 @ 50k | 147.0 @ 308k | -83.9 | 0.0 |
| r011 | no_out | 242.8 @ 280k | 242.8 @ 280k | 0.0 | 243.8 @ 280k | 243.8 @ 280k | 0.0 | 0.0 |
| r012 | no_out | 263.0 @ 80k | 202.2 @ 303k | -60.8 | 264.0 @ 80k | 203.2 @ 303k | -60.8 | 0.0 |
| r013 | plus_out | 134.2 @ 300k | -1.0 @ 306k | -135.2 | 213.1 @ 300k | 183.0 @ 306k | -30.1 | -183.0 |
| r014 | plus_out | 236.1 @ 290k | 97.0 @ 300k | -139.1 | 249.6 @ 290k | 179.4 @ 300k | -70.2 | -81.6 |
| r015 | plus_out | 95.4 @ 270k | -1.0 @ 301k | -96.4 | 225.4 @ 280k | 174.8 @ 301k | -50.6 | -174.8 |
| r016 | plus_out | 84.2 @ 270k | 70.0 @ 290k | -14.2 | 249.4 @ 140k | 187.1 @ 290k | -62.2 | -116.6 |
| r017 | plus_out | 184.2 @ 70k | 96.6 @ 290k | -87.6 | 298.1 @ 140k | 276.4 @ 290k | -21.8 | -178.9 |
| r018 | plus_out | 125.4 @ 140k | 80.2 @ 270k | -45.1 | 290.0 @ 270k | 290.0 @ 270k | 0.0 | -209.0 |

## Retention Counts

- Latest reward is exactly the run's best reward in `1/18` runs.
- Latest reward is within 10% of best in `1/18` runs.
- Latest survival is exactly the run's best survival in `2/18` runs.
- Latest survival is within 90% of best in `3/18` runs.
- For sparse: latest reward near best `0/6`, latest survival near best `0/6`.
- For no-outcome: latest reward near best `1/6`, latest survival near best
  `1/6`.
- For plus-outcome: latest reward near best `0/6`, latest survival near best
  `2/6`.

## Interpretation

The retention problem is even clearer with own reward than with survival:
almost every run's latest checkpoint is worse than its own best checkpoint. The
exception is r011 for no-outcome reward/survival, and r018 for survival only.

For next-batch design, use:

- survival as the cross-run/cross-reward progress metric;
- own reward as a within-variant sanity check;
- tournament rank as the actual head-to-head strength metric;
- residual outcome term for plus-outcome to detect whether the policy is
  surviving but still losing.

## Tournament Alignment Check

Joined current `round-000035` rating rows to eval checkpoints by
`run_id + iteration`. This matched `572/573` rating rows.

Rating correlation with eval metrics:

| Scope | Rows | Survival Corr | Own Reward Corr | Outcome Residual Corr |
| --- | ---: | ---: | ---: | ---: |
| all matched rows | 572 | 0.431 | -0.009 | -0.232 |
| active top-100 rows | 99 | 0.302 | -0.114 | -0.443 |

By reward variant, rating correlation with survival/reward:

| Variant | Rows | Active | Survival Corr | Own Reward Corr |
| --- | ---: | ---: | ---: | ---: |
| sparse | 193 | 25 | 0.429 | 0.227 |
| no_out | 192 | 22 | 0.449 | 0.450 |
| plus_out | 187 | 52 | 0.402 | 0.066 |

Read:

- Tournament rating is moderately aligned with survival.
- Tournament rating is not generally aligned with own reward, especially for
  plus-outcome.
- For plus-outcome, high tournament rows can still have negative outcome
  residuals. The tournament is judging head-to-head strength, not the training
  reward scalar.
- r018's top tournament checkpoint is not simply the checkpoint with highest
  eval survival or highest own reward. That means next selection should keep
  multiple signals: tournament champion, best survival, and best own reward.
