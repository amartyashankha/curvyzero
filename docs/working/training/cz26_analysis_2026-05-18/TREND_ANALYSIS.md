# Trend Analysis

Captured: 2026-05-18.

This note asks: did runs improve over time, and did they keep the improvement?

## Batch-Level Shape

| Grid | First survival | Best survival | Latest survival | Latest / best |
| --- | ---: | ---: | ---: | ---: |
| Grid A | 199.0 | 276.3 | 157.5 | 0.60 |
| Grid B | 199.9 | 336.2 | 167.9 | 0.55 |

Read: the batch found better intermediate checkpoints, then many runs lost the
gain by the latest checkpoint. This is a retention failure, not a pure
no-learning failure.

## Reward Trend Caveat

Training reward is harder to read than survival:

- the tournament can feed stronger opponents back into training;
- different reward alphas are not one shared scale;
- outcome reward is inferred as a residual, not a separately serialized field.

Read: falling reward can be expected if opponents get stronger. Survival and
tournament rank are better cross-setting signals.

## Grid A Trend By Reward

| Reward | First survival | Best survival | Latest survival | Latest / best | Best learned rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `out0` | 192.8 | 264.3 | 165.8 | 0.65 | 40 |
| `out33` | 182.6 | 248.9 | 153.1 | 0.63 | 68 |
| `out67` | 223.3 | 308.0 | 149.9 | 0.53 | 34 |
| `out100` | 197.5 | 284.1 | 161.3 | 0.58 | 56 |

Read: `out67` has the highest best survival but the weakest retention.
`out0` has the best latest survival. This supports the idea that outcome reward
can produce strong middle checkpoints but may be less stable.

## Grid A Trend By Noise

| Noise | First survival | Best survival | Latest survival | Latest / best | Best learned rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `n0` | 202.1 | 258.2 | 153.1 | 0.62 | 58 |
| `n10` | 211.4 | 290.3 | 171.0 | 0.61 | 40 |
| `n20` | 183.6 | 280.4 | 148.6 | 0.56 | 34 |

Read: `n10` has the best latest survival. `n20` reaches better tournament rank
but has worse retention.

## Grid B Trend By Recipe

| Recipe | Best survival | Latest survival | Latest / best | Best learned rank |
| --- | ---: | ---: | ---: | ---: |
| `b100` | 492.9 | 160.2 | 0.37 | 78 |
| `b20w05r1` | 288.3 | 225.3 | 0.79 | 98 |
| `b25w25r1` | 279.0 | 224.4 | 0.76 | 65 |
| `b50r1` | 466.1 | 161.2 | 0.39 | 116 |
| `r1` | 197.9 | 128.3 | 0.67 | 479 |

Read: high best survival is not enough. `b100` and `b50r1` peak very high but
lose most of that gain. `b20w05r1` and `b25w25r1` retain more.

## Tournament Trend Caveat

The top learned CZ26 checkpoint reached rank 34, but most top learned
checkpoints have sparse exposure:

- rank 34: 6 battles;
- rank 40: 6 battles;
- rank 56: 2 battles;
- rank 58: 1 battle;
- rank 60: 2 battles.

Read: tournament rank says some learned checkpoints are promising. It does not
yet support precise fine ordering between those checkpoints.

## Current Trend Interpretation

- The training process can produce useful checkpoints.
- The latest checkpoint is often not the best checkpoint.
- Future analysis should track best-so-far, not only latest.
- Future training/tournament loops should preserve promising intermediate
  checkpoints rather than assuming the final checkpoint is best.
