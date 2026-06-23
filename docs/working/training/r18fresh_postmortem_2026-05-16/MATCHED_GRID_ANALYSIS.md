# Matched-Grid Analysis

Captured: 2026-05-16.

Source: `lightzero_curvytron_run_status --output eval-json` for all 18
`curvy-r18fresh-allv2-20260516a` runs.

Method:

- Use common eval checkpoints from `iteration_0` through `iteration_240000`.
- AUC is normalized trapezoidal mean survival over the 10k-spaced common grid.
- `best` is best eval survival on the common grid through 240k.
- `drop` is `at240 - best`; negative means the run has already regressed from
  its own best by 240k.
- `latest` is reported separately and is not used as the fair comparison axis.

## Aggregate By Reward

| Reward | Rows | First | At 240k | AUC | Best | Drop | Latest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sparse_outcome` | 6 | 157.7 | 197.6 | 169.9 | 244.2 | -46.6 | 162.2 |
| `survival_plus_bonus_no_outcome` | 6 | 154.2 | 175.1 | 164.6 | 235.5 | -60.4 | 181.7 |
| `survival_plus_bonus_plus_outcome` | 6 | 168.8 | 196.0 | 179.5 | 241.1 | -45.0 | 215.1 |

Read: sparse and plus-outcome are close at exactly 240k. Plus-outcome has the
best AUC and latest retention. No-outcome trails on AUC and has the worst common
grid regression.

## Aggregate By Opponent Recipe

| Recipe | Rows | First | At 240k | AUC | Best | Drop | Latest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blank10-wall10-rank2_25-rank1_55` | 6 | 136.6 | 173.1 | 161.3 | 224.9 | -51.8 | 162.5 |
| `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5` | 6 | 157.5 | 156.0 | 154.2 | 219.1 | -63.1 | 156.4 |
| `blank20-wall5-rank1_70-rank1imm5` | 6 | 186.6 | 239.6 | 198.5 | 276.7 | -37.1 | 240.2 |

Read: `blank20-wall5-rank1_70-rank1imm5` is the strongest recipe by every fair
matched survival metric here.

## Aggregate By Stochasticity

| Stochasticity | Rows | First | At 240k | AUC | Best | Drop | Latest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 9 | 164.5 | 183.9 | 165.3 | 230.4 | -46.5 | 179.7 |
| `straight_override_p10_repeat_p10` | 9 | 155.9 | 195.2 | 177.3 | 250.1 | -54.8 | 193.1 |

Read: the stochastic lane did better on AUC and 240k/latest survival here, but
also had worse common-grid drop. Treat it as paired intervention evidence, not a
free win.

## Per-Run Table

| Row | Reward | Recipe | Noise | First | At 240k | AUC | Best | Best Iter | Drop | Latest Iter | Latest |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r001 | sparse | rank2/rank1 | clean | 179.4 | 151.1 | 150.6 | 227.5 | 180000 | -76.4 | 308600 | 118.1 |
| r002 | sparse | rank2/rank1 | so10rep10 | 127.4 | 187.2 | 183.2 | 241.8 | 120000 | -54.5 | 306847 | 151.6 |
| r003 | sparse | ladder | clean | 134.6 | 126.6 | 124.9 | 198.2 | 120000 | -71.6 | 301469 | 118.6 |
| r004 | sparse | ladder | so10rep10 | 166.1 | 139.1 | 154.8 | 207.6 | 160000 | -68.5 | 304439 | 157.1 |
| r005 | sparse | blank20/wall5/rank1 | clean | 207.8 | 285.2 | 200.9 | 285.2 | 240000 | 0.0 | 303909 | 182.6 |
| r006 | sparse | blank20/wall5/rank1 | so10rep10 | 131.0 | 296.0 | 204.9 | 304.6 | 210000 | -8.6 | 310893 | 245.4 |
| r007 | no-outcome | rank2/rank1 | clean | 130.9 | 159.0 | 153.3 | 201.9 | 230000 | -42.9 | 307021 | 166.1 |
| r008 | no-outcome | rank2/rank1 | so10rep10 | 92.1 | 171.6 | 157.5 | 251.2 | 160000 | -79.6 | 300865 | 176.6 |
| r009 | no-outcome | ladder | clean | 197.9 | 175.5 | 164.0 | 243.5 | 120000 | -68.0 | 308259 | 153.6 |
| r010 | no-outcome | ladder | so10rep10 | 149.9 | 168.2 | 158.7 | 230.9 | 50000 | -62.6 | 308448 | 147.0 |
| r011 | no-outcome | blank20/wall5/rank1 | clean | 159.0 | 166.1 | 146.8 | 221.8 | 130000 | -55.6 | 280000 | 243.8 |
| r012 | no-outcome | blank20/wall5/rank1 | so10rep10 | 195.5 | 210.2 | 207.2 | 264.0 | 80000 | -53.8 | 303563 | 203.2 |
| r013 | plus-outcome | rank2/rank1 | clean | 125.6 | 169.9 | 152.1 | 212.6 | 150000 | -42.8 | 306004 | 183.0 |
| r014 | plus-outcome | rank2/rank1 | so10rep10 | 164.0 | 199.9 | 171.0 | 214.5 | 140000 | -14.6 | 300000 | 179.4 |
| r015 | plus-outcome | ladder | clean | 117.5 | 174.1 | 152.5 | 185.1 | 80000 | -11.0 | 301850 | 174.8 |
| r016 | plus-outcome | ladder | so10rep10 | 179.2 | 152.5 | 170.0 | 249.4 | 140000 | -96.9 | 290000 | 187.1 |
| r017 | plus-outcome | blank20/wall5/rank1 | clean | 228.1 | 247.5 | 242.8 | 298.1 | 140000 | -50.6 | 290000 | 276.4 |
| r018 | plus-outcome | blank20/wall5/rank1 | so10rep10 | 198.1 | 232.4 | 188.5 | 286.6 | 190000 | -54.2 | 270000 | 290.0 |

## Implications

- Next batch should not use latest-only promotion.
- Preserve mid-run tournament winners and heldout/eval best checkpoints.
- Expand around plus-outcome and `blank20-wall5-rank1_70-rank1imm5`.
- Keep sparse as a real diagnostic arm because it is competitive on matched
  survival, even though the tournament top band currently favors plus-outcome.
- Do not treat no-outcome as a primary arm unless a different metric justifies
  it.
