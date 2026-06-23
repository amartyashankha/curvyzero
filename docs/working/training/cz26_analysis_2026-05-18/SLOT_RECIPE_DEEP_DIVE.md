# Slot Recipe Deep Dive

Captured: 2026-05-18.

This note focuses only on opponent recipes. The broad reward/noise/immortality
analysis is in `MATCHED_GRID_ANALYSIS.md`.

## Exact Recipes

Each recipe is a 64-slot opponent bag. Blank and wall seats are hard-coded
immortal opponents. Rank seats are tournament leaderboard checkpoints.

| Recipe | 64-slot makeup | Plain meaning |
| --- | --- | --- |
| `b100` | 64 blank | Pure blank hard-coded control |
| `w100` | 64 wall | Pure wall-avoider hard-coded control |
| `r1` | 64 rank1 | Pure current rank-1 leaderboard opponent |
| `b50r1` | 32 blank, 32 rank1 | Half blank, half rank1 |
| `b25w25r1` | 16 blank, 16 wall, 32 rank1 | Blank/wall/rank1 split |
| `b30w05r1` | 19 blank, 3 wall, 42 rank1 | More blank, small wall, rank1 heavy |
| `b20w05r1` | 13 blank, 3 wall, 48 rank1 | Baseline mixed recipe |
| `b10w05r1` | 7 blank, 3 wall, 54 rank1 | Less blank than baseline |
| `b20w10r1` | 13 blank, 6 wall, 45 rank1 | More wall than baseline |
| `b20w05top2` | 13 blank, 3 wall, 32 rank1, 16 rank2 | Split leaderboard across top 2 |
| `b20w05lad4` | 13 blank, 3 wall, 19 rank1, 13 rank2, 10 rank3, 6 rank4 | Ladder over top 4 |
| `b20w20lad4s` | 13 blank, 13 wall, 19 rank1, 13 rank2, 3 rank3, 3 rank4 | Ladder with heavy wall |

Read: the recipes deliberately mix solo-like survival pressure, wall avoidance,
and tournament opponent pressure. Leaderboard immortality is a separate flag;
it does not change what policy the rank slot points to.

## Grid A Recipes

Grid A crossed four production-like recipes with reward/noise/immortality.

| Recipe | Best survival | Latest survival | Retention | Best learned rank | Top-100 learned runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b10w05r1` | 271.3 | 170.4 | 0.64 | 40 | 5 |
| `b20w05r1` | 281.3 | 150.2 | 0.57 | 34 | 3 |
| `b20w05top2` | 278.3 | 156.1 | 0.58 | 60 | 5 |
| `b20w10r1` | 274.4 | 153.5 | 0.60 | 56 | 5 |

Read: Grid A does not pick one recipe cleanly. `b10w05r1` keeps the best latest
survival, `b20w05r1` reaches the best learned rank, and `b20w05top2` has useful
tournament signal.

## Grid B Recipes

Grid B is the slot-focused grid. It fixes reward near the middle setting and
tests broader recipe families.

| Recipe | Best survival | Latest survival | Retention | Best learned rank | Top-100 learned runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b100` | 492.9 | 160.2 | 0.37 | 78 | 1 |
| `b20w05lad4` | 354.2 | 144.3 | 0.42 | 87 | 2 |
| `b20w05r1` | 288.3 | 225.3 | 0.79 | 98 | 1 |
| `b20w05top2` | 260.3 | 163.1 | 0.63 | 155 | 0 |
| `b20w20lad4s` | 359.6 | 164.8 | 0.50 | 109 | 0 |
| `b25w25r1` | 279.0 | 224.4 | 0.76 | 65 | 1 |
| `b30w05r1` | 321.8 | 174.8 | 0.54 | 83 | 1 |
| `b50r1` | 466.1 | 161.2 | 0.39 | 116 | 0 |
| `r1` | 197.9 | 128.3 | 0.67 | 479 | 0 |
| `w100` | 342.5 | 132.7 | 0.40 | 112 | 0 |

Read: pure rank1 is clearly weak. Pure blank and half blank can create huge
best-survival spikes, but they regress hard. The most interesting retention
recipes are `b20w05r1` and `b25w25r1`.

## What We Can Infer

- Some hard-coded opponent mass is useful. Pure tournament rank1 pressure is
  not enough.
- Too much hard-coded control can create high survival peaks that do not last.
- A small amount of wall pressure is probably useful, but the data does not
  isolate wall percentage cleanly.
- The ladder recipes may create matchup-useful policies, but they do not yet
  dominate survival.

## What Is Still Confounded

- Blank percentage, wall percentage, rank concentration, and rank diversity
  often change together.
- Tournament rank has sparse exposure for top learned checkpoints.
- Latest survival is not the same thing as best checkpoint quality.

## Next Questions

- Does `b25w25r1` keep high latest survival because of the added wall mass or
  because it reduces rank1 pressure?
- Does `b20w05lad4` produce broader matchup skills that fixed survival eval
  misses?
- Should future recipes preserve `b20w05r1` and `b25w25r1` as anchors while
  testing controlled blank/wall/rank-diversity changes?
