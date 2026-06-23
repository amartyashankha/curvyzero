# CZ26 Deep Analysis

This is the current plain-language synthesis. The full generated tables are in:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.json
```

## Sources

- Manifest: `cz26-full-20260517a`, 136 runs.
- Eval/status snapshot: `cz26_eval_status_latest.json`.
- Tournament rating snapshot: `cz26_rating_latest.json`, latest completed
  rating snapshot `round-000049`.
- Joined first-pass report: `cz26_joined_analysis.json`.

No live Modal state was changed for this analysis pass.

## Plain Glossary

- `Grid A`: 96-run broad grid. It crosses reward outcome strength, action
  noise, leaderboard-opponent immortality, and four opponent recipes.
- `Grid B`: 40-run slot-focused grid. Reward is fixed near the middle setting;
  it mainly tests opponent recipes.
- `survival`: average eval game length. Higher is better.
- `training reward`: scalar reward from the eval/training code. Compare only
  within the same reward definition.
- `outcome residual`: inferred win/loss term: training reward minus survival
  component minus bonus component.
- `learned tournament rank`: tournament rank after ignoring `iteration 0`
  seed checkpoints. Lower is better.
- `rating exposure`: how many tournament games/battles support a rank.
- `exact horizon`: exact checkpoint iteration used for a table, such as 30k,
  170k, or 300k.

## Coverage

| Grid | Rows | Completed | Running | 30k rows | 170k rows | 300k rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grid A | 96 | 92 | 4 | 96 | 96 | 93 |
| Grid B | 40 | 40 | 0 | 40 | 39 | 38 |

Read: all 136 runs have eval curves and tournament rows. Grid B has two
low-coverage rows, so the all-row fair Grid B horizon is only 30k. The 300k
Grid B read is still useful, but it excludes two rows.

## Main Shape

The batch did not simply fail from the start. Most rows found a better
intermediate checkpoint and then regressed by the latest checkpoint.

| Grid | First survival | Best survival | Latest survival | Latest / best |
| --- | ---: | ---: | ---: | ---: |
| Grid A | 199.0 | 276.3 | 157.5 | 0.60 |
| Grid B | 199.9 | 336.2 | 167.9 | 0.55 |

Read: the central problem is retention. The system often creates a useful
checkpoint in the middle, but the final checkpoint is often worse.

## Reward Axis

Grid A reward settings:

- `out0`: survival plus bonus, no terminal outcome reward.
- `out33`: small outcome reward.
- `out67`: medium outcome reward.
- `out100`: full outcome reward.

Matched survival deltas against `out0`:

| Comparison | 30k survival | 170k survival | 300k survival | Learned tournament rank |
| --- | ---: | ---: | ---: | ---: |
| `out33 - out0` | -18.5 | -18.8 | -7.9 | +108.2 |
| `out67 - out0` | +10.4 | +35.9 | -14.5 | -39.9 |
| `out100 - out0` | -7.9 | +13.3 | -3.0 | -111.8 |

Read: `out67` looked best around 170k, but not at 300k. `out100` has the best
learned tournament-rank delta versus `out0`, but that rank evidence is sparse.
`out33` is weak in this batch. This does not prove outcome reward is bad; it
proves the outcome axis changes the time shape and needs survival plus
tournament evidence together.

Reward component read:

- `out0` has outcome residual near zero, as expected.
- `out33`, `out67`, `out100`, and `out50` have nonzero residuals.
- Outcome residual is often negative, because losing subtracts from the return.
- Bonus reward is tiny in this batch and is not explaining the major
  differences.

## Noise Axis

Grid A matched survival deltas:

| Comparison | 30k survival | 170k survival | 300k survival | Learned tournament rank |
| --- | ---: | ---: | ---: | ---: |
| `n10 - n0` | +18.4 | +4.3 | +28.8 | +63.1 |
| `n20 - n0` | +17.0 | -1.1 | +22.5 | -27.5 |
| `n20 - n10` | -1.4 | -5.4 | -12.9 | -90.6 |

Read: `n10` is the cleanest eval-survival improvement. `n20` looks worse than
`n10` on survival but better in tournament top-band/rank evidence. The noise
knob is not monotonic, and the tournament evidence is too sparse for exact
ordering.

## Leaderboard Immortality

`imm10` means leaderboard-derived opponents are made immortal 10% of the time.
Blank and wall opponents are already hard-coded immortal.

Grid A lifecycle:

| Setting | Best survival | Latest survival | Best learned rank | Top-100 learned runs |
| --- | ---: | ---: | ---: | ---: |
| `imm0` | 285.1 | 151.9 | 34 | 12 |
| `imm10` | 267.5 | 163.2 | 56 | 6 |

Read: Grid A does not support `imm10`. It slightly improves latest survival,
but it is worse on best survival and much worse on learned tournament rank.

Grid B is more mixed:

| Setting | Best survival | Latest survival | Best learned rank | Top-100 learned runs |
| --- | ---: | ---: | ---: | ---: |
| `imm0` | 329.8 | 160.3 | 65 | 2 |
| `imm10` | 342.7 | 175.5 | 78 | 4 |

Read: Grid B has a small survival/retention signal for `imm10`, but best
learned rank is still better for `imm0`. This is not a clean win either way.

## Recipe Axis

Exact recipe counts are in `ANALYSIS_PLAN.md` and in the generated report. The
important plain meanings:

- `b20w05r1`: 13 blank, 3 wall, 48 rank1. Baseline mixed recipe.
- `b10w05r1`: less blank, same wall, more rank1.
- `b20w10r1`: same blank, more wall.
- `b20w05top2`: same blank/wall, split leaderboard between rank1 and rank2.
- `r1`: pure rank1 leaderboard opponent.
- `b100` / `w100`: pure hard-coded controls.

Grid A recipe read:

| Recipe | Best survival | Latest survival | Best learned rank | Top-100 learned runs |
| --- | ---: | ---: | ---: | ---: |
| `b10w05r1` | 271.3 | 170.4 | 40 | 5 |
| `b20w05r1` | 281.3 | 150.2 | 34 | 3 |
| `b20w05top2` | 278.3 | 156.1 | 60 | 5 |
| `b20w10r1` | 274.4 | 153.5 | 56 | 5 |

Read: no single Grid A recipe wins every signal. `b20w05r1` has the best
individual learned rank, `b10w05r1` has the best latest survival, and
`b20w05top2` has the best mean learned rank. This is exactly why we should not
pick recipes from one table.

Grid B recipe read:

| Recipe | Best survival | Latest survival | Best learned rank | Top-100 learned runs |
| --- | ---: | ---: | ---: | ---: |
| `b100` | 492.9 | 160.2 | 78 | 1 |
| `b20w05lad4` | 354.2 | 144.3 | 87 | 2 |
| `b20w05r1` | 288.3 | 225.3 | 98 | 1 |
| `b25w25r1` | 279.0 | 224.4 | 65 | 1 |
| `b30w05r1` | 321.8 | 174.8 | 83 | 1 |
| `r1` | 197.9 | 128.3 | 479 | 0 |

Read: pure rank1 is weak. Some hard-coded or mixed recipes create high
survival peaks, but many have poor retention. `b20w05r1` and `b25w25r1` look
interesting because their latest survival is still high. `b20w05lad4` has good
tournament placement but weaker latest survival.

## Tournament Read

Every CZ26 run appears in the tournament snapshot, and every run has learned
checkpoints represented. But the fine ordering is not reliable yet.

Top learned CZ26 rows:

| Rank | Run | Best-ranked checkpoint games | Battles |
| ---: | --- | ---: | ---: |
| 34 | `cz26a-r017-out67-n20-imm0-b20w05r1` | 126 | 6 |
| 40 | `cz26a-r027-out0-n10-imm0-b10w05r1` | 126 | 6 |
| 56 | `cz26a-r072-out100-n20-imm10-b20w10r1` | 42 | 2 |
| 58 | `cz26a-r043-out100-n0-imm0-b10w05r1` | 21 | 1 |
| 60 | `cz26a-r096-out100-n20-imm10-b20w05top2` | 42 | 2 |

Read: these are promising learned checkpoints, but many ranks are based on only
one or two battles. Treat rank 34 versus rank 60 as directional, not precise.

## Action Collapse

- Latest checkpoint action-collapse count: 8 runs.
- Any full-checkpoint action-collapse during training: 63 runs.
- Any per-row action-collapse count at latest eval: 69 runs.

Read: action collapse is common enough to track, but it is not the whole story.
Some high tournament rows also show low latest survival or collapse. This is
another reason to keep best checkpoint, latest checkpoint, and tournament rank
separate.

## Current Conclusions

1. The batch created useful intermediate policies.
2. Latest checkpoints often regressed.
3. Raw tournament top rows must ignore `iteration 0`; otherwise the shared
   starting checkpoint pollutes the story.
4. Tournament ordering is still too sparse for fine ranking.
5. `out67` and `out100` are not dead; they show useful mid-run or tournament
   signals.
6. `out33` looks weak.
7. `n10` is the cleanest eval-survival noise setting; `n20` has some
   tournament signal but worse survival stability.
8. Grid A does not support `imm10`; Grid B is mixed but not a clean endorsement.
9. Pure rank1 opponents are weak in Grid B.
10. Mixed recipes with blank/wall support are still the most interesting slot
    lane.

## Questions To Dig Next

1. Why do many rows peak and then regress? Is this due to opponent refresh,
   training instability, action collapse, or tournament feedback getting harder?
2. Are the best tournament checkpoints usually early because they are actually
   stronger, or because they have sparse lucky exposure?
3. Does survival against hard-coded opponents improve while survival against
   tournament opponents degrades?
4. Would selecting or preserving best checkpoints, instead of latest
   checkpoints, improve the feedback loop?
5. Does `n20` produce matchup diversity that helps tournament rank despite
   weaker fixed eval survival?
6. Which Grid B recipes keep high latest survival after removing low-exposure
   tournament noise?
