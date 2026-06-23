# Matched Grid Analysis

Captured: 2026-05-18.

Sources:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.json
```

Method:

- Compare one setting at a time while holding the other settings fixed.
- Use exact eval checkpoints where possible: 30k, 170k, and 300k.
- Use learned tournament rank only after excluding `iteration 0`.
- Lower tournament rank is better; higher survival is better.

## Coverage

| Grid | Rows | Completed | Running | Rows at 30k | Rows at 170k | Rows at 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grid A | 96 | 92 | 4 | 96 | 96 | 93 |
| Grid B | 40 | 40 | 0 | 40 | 39 | 38 |

Read: Grid A is nearly complete at 300k. Grid B has two low-coverage rows, so
30k is the only fully covered exact horizon; 300k is a useful sensitivity read.

## Grid A By Reward

| Comparison | 30k survival | 170k survival | 300k survival | Learned tournament rank |
| --- | ---: | ---: | ---: | ---: |
| `out33 - out0` | -18.5 | -18.8 | -7.9 | +108.2 |
| `out67 - out0` | +10.4 | +35.9 | -14.5 | -39.9 |
| `out100 - out0` | -7.9 | +13.3 | -3.0 | -111.8 |
| `out67 - out33` | +29.0 | +54.7 | -1.2 | -148.1 |
| `out100 - out67` | -18.3 | -22.7 | +7.5 | -71.9 |

Read: `out67` is the clearest mid-run survival winner, but it does not hold at
300k. `out100` is not best on survival but has the best tournament-rank delta.
`out33` is weak.

## Grid A By Noise

| Comparison | 30k survival | 170k survival | 300k survival | Learned tournament rank |
| --- | ---: | ---: | ---: | ---: |
| `n10 - n0` | +18.4 | +4.3 | +28.8 | +63.1 |
| `n20 - n0` | +17.0 | -1.1 | +22.5 | -27.5 |
| `n20 - n10` | -1.4 | -5.4 | -12.9 | -90.6 |

Read: `n10` is the cleanest fixed-eval survival improvement. `n20` looks
better in tournament rank but worse than `n10` on survival. That means noise is
not a simple "more is better" knob.

## Grid A By Leaderboard Immortality

| Comparison | 30k survival | 170k survival | 300k survival | Learned tournament rank |
| --- | ---: | ---: | ---: | ---: |
| `imm10 - imm0` | +5.1 | -16.6 | +0.2 | +172.8 |

Read: Grid A does not support 10% leaderboard immortality. The tournament-rank
delta is especially bad for `imm10`.

## Grid A Lifecycle By Recipe

| Recipe | Rows | First survival | Best survival | Latest survival | Best learned rank | Top-100 learned runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `b10w05r1` | 24 | 186.9 | 271.3 | 170.4 | 40 | 5 |
| `b20w05r1` | 24 | 204.3 | 281.3 | 150.2 | 34 | 3 |
| `b20w05top2` | 24 | 191.5 | 278.3 | 156.1 | 60 | 5 |
| `b20w10r1` | 24 | 213.4 | 274.4 | 153.5 | 56 | 5 |

Read: recipe evidence is split. `b20w05r1` has the best single learned rank,
`b10w05r1` has the best latest survival, and `b20w05top2` has strong average
tournament placement in the full generated report.

## Grid B Lifecycle By Recipe

| Recipe | Rows | First survival | Best survival | Latest survival | Best learned rank | Top-100 learned runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `b100` | 4 | 189.4 | 492.9 | 160.2 | 78 | 1 |
| `b20w05lad4` | 4 | 244.3 | 354.2 | 144.3 | 87 | 2 |
| `b20w05r1` | 4 | 199.7 | 288.2 | 225.3 | 98 | 1 |
| `b25w25r1` | 4 | 150.0 | 279.0 | 224.4 | 65 | 1 |
| `b30w05r1` | 4 | 165.1 | 321.8 | 174.8 | 83 | 1 |
| `r1` | 4 | 170.7 | 197.9 | 128.3 | 479 | 0 |
| `w100` | 4 | 226.5 | 342.5 | 132.7 | 112 | 0 |

Read: pure `r1` is weak. Mixed recipes and hard-coded controls both beat it,
but some of the high-survival hard-coded-control rows regress hard by latest.

## Current Interpretation

- The batch has useful intermediate checkpoints.
- Latest checkpoints are often worse than the best checkpoint.
- Tournament rank helps, but fine ordering is weak because many top learned
  checkpoints have only 1-6 battles.
- Use exact-horizon matched comparisons first, then use lifecycle and
  tournament rank as supporting context.
