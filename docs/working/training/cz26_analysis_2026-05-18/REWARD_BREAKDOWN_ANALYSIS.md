# Reward Breakdown Analysis

Captured: 2026-05-18.

Source:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
```

## Reward Semantics

The eval artifact exposes:

- `mean_training_reward`;
- `mean_reward_components.survival`;
- `mean_reward_components.bonus`;
- `mean_bonus_reward`;
- `mean_bonus_pickup_count`.

It does not expose a named numeric `outcome` component. We infer it as:

```text
outcome_residual = mean_training_reward - survival_component - bonus_component
```

Read: outcome residual is a diagnostic. It is probably the terminal win/loss
contribution, but the artifact does not name it directly.

## Grid A Reward At 170k

| Reward | Rows | Survival | Training reward | Outcome residual | Collapsed rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `out0` | 24 | 175.9 | 175.1 | 0.0 | 1 |
| `out33` | 24 | 157.2 | 147.5 | -9.0 | 0 |
| `out67` | 24 | 211.8 | 146.8 | -64.4 | 0 |
| `out100` | 24 | 189.2 | 110.1 | -78.4 | 0 |

Read: at 170k, `out67` has the strongest survival but a large negative outcome
residual. That means it survives longer in eval but still often loses under the
outcome-shaped reward.

## Grid A Reward At 300k

| Reward | Rows | Survival | Training reward | Outcome residual | Collapsed rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `out0` | 24 | 160.7 | 159.8 | 0.0 | 1 |
| `out33` | 23 | 153.2 | 122.0 | -30.3 | 0 |
| `out67` | 22 | 148.1 | 85.5 | -61.8 | 1 |
| `out100` | 24 | 157.7 | 93.7 | -63.2 | 3 |

Read: by 300k, `out0` has the best survival and reward. The outcome variants
still have negative residuals. This does not prove "no outcome reward is best"
because tournament rank points partly the other way, but it shows outcome
settings were less stable by the endpoint.

## Bonus Reward

Bonus reward is available, but it is tiny in this batch. It does not explain
the major differences between settings.

Read: keep bonus in the reward definition, but do not treat bonus pickup as the
main explanatory signal for CZ26.

## What This Means

- `out0` reward is easiest to interpret because it mostly tracks survival.
- `out67` can create strong mid-run survival but poor endpoint retention.
- `out100` has tournament signal but poor reward/survival stability.
- `out33` is weak in this batch.
- Training reward must not be averaged across reward alphas as if it were one
  shared metric.

## Next Reward Questions

- Are outcome variants losing because the terminal penalty is too strong, or
  because they face stronger tournament opponents?
- Are the best outcome-variant checkpoints actually better head-to-head even
  when their fixed eval reward is lower?
- Should future reward analysis compare checkpoints against fixed opponent
  sets, separate from adaptive tournament opponents?
