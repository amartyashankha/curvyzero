# r18fresh Learning Readout - 2026-05-16

Scope: original tournament-attached `r18fresh` 18-run batch,
`curvy-r18fresh-allv2-20260516a`, with live feedback tournament
`curvy-r18fresh-live-bounded-dsf1-20260516b` /
`elo-r18fresh-live-bounded-dsf1-20260516b`.

Short answer: the mechanical loop worked, and the batch did learn mid-run. It
also regressed late. Every row found a checkpoint better than its own
iteration-0 survival eval, but only `10/18` latest checkpoints stayed above
their first eval and only `4/18` latest checkpoints were within 90% of their own
best.

## Bottom Line

- Progress: yes. Mean survival moved from first `159.9` to best `246.0`
  (`+86.1`), and best improved in `18/18` rows.
- Regression: latest mean was only `175.4`, so the batch gave back `70.6` steps
  on average from each row's best checkpoint. Latest-vs-first was only `+15.5`.
- Shape: best checkpoints are usually mid-run, while latest checkpoints are
  often worse. This is not a dead training/tournament loop; it is a retention
  and stability problem.
- Strongest latest survival slice: `survival_plus_bonus_plus_outcome`
  (`+35.6` latest-vs-first, `5/6` latest up, `3/6` near best).
- Strongest opponent-recipe slice by latest survival delta:
  `blank20-wall5-rank1_70-rank1imm5` (`+35.8` latest-vs-first), with high best
  but still a large latest-vs-best drop.
- Strongest control/noise slice by latest survival delta:
  `straight_override_p10_repeat_p10` (`+30.6` latest-vs-first), though it still
  dropped `69.0` from best. Clean rows were basically flat at latest (`+0.5`).
- Main suspects: late instability from the live opponent curriculum, dense
  reward/value support saturation, large collect chunks with small batch,
  shallow `num_simulations=8`, and intermittent action-collapse histories. The
  mechanics/reward audit did not find wrong-player bonus credit, wrong-seat
  reward ownership, or tournament seat-observation mismatch.

## Aggregate Survival

Source for this table: `/tmp/r18fresh-status-digested.json`, produced from the
current read-only eval/status pull. A parallel later note in
`survival_stagnation_investigation_2026-05-16.md` reported a slightly fresher
`160.2 / 247.0 / 181.6` first/best/latest read with `12/18` latest improved;
the diagnosis is unchanged.

| Group | Rows | First | Best | Latest | Best-first | Latest-first | Latest-best | Latest up | Near best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All rows | 18 | 159.9 | 246.0 | 175.4 | +86.1 | +15.5 | -70.6 | 10/18 | 4/18 |

## By Reward Variant

| Reward variant | Rows | First | Best | Latest | Best-first | Latest-first | Latest-best | Latest up | Near best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sparse_outcome` | 6 | 157.7 | 260.4 | 156.4 | +102.7 | -1.3 | -104.1 | 2/6 | 0/6 |
| `survival_plus_bonus_no_outcome` | 6 | 154.2 | 235.5 | 166.4 | +81.3 | +12.2 | -69.1 | 3/6 | 1/6 |
| `survival_plus_bonus_plus_outcome` | 6 | 167.7 | 242.0 | 203.3 | +74.4 | +35.6 | -38.7 | 5/6 | 3/6 |

Read: sparse produced the biggest mid-run lift but worst late retention.
`survival_plus_bonus_plus_outcome` was least bad at latest, despite the dense
support-saturation concern.

## By Opponent Recipe

| Opponent recipe | Rows | First | Best | Latest | Best-first | Latest-first | Latest-best | Latest up | Near best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blank10-wall10-rank2_25-rank1_55` | 6 | 135.5 | 225.9 | 160.6 | +90.4 | +25.2 | -65.2 | 5/6 | 2/6 |
| `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5` | 6 | 157.5 | 219.1 | 143.1 | +61.6 | -14.4 | -76.0 | 1/6 | 1/6 |
| `blank20-wall5-rank1_70-rank1imm5` | 6 | 186.6 | 293.0 | 222.4 | +106.4 | +35.8 | -70.6 | 4/6 | 1/6 |

Read: `blank20-wall5-rank1_70-rank1imm5` produced the strongest best and latest
means, but still regressed materially. The multi-rank ladder with
`rank4/rank3/rank2/rank1/rank1imm` is the weakest latest survival slice.

## By Noise/Control

| Noise/control | Rows | First | Best | Latest | Best-first | Latest-first | Latest-best | Latest up | Near best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | 9 | 164.5 | 237.2 | 165.0 | +72.7 | +0.5 | -72.2 | 4/9 | 3/9 |
| `straight_override_p10_repeat_p10` | 9 | 155.2 | 254.8 | 185.8 | +99.6 | +30.6 | -69.0 | 6/9 | 1/9 |

Read: stochastic action override/repeat did not obviously hurt this batch's
latest survival. It may have regularized exploration, but it also makes action
credit assignment less clean, so this is a clue, not a recommendation.

## Per Row Survival

`Latest-best` is negative when the row regressed from its own best checkpoint.

| Run slice | Reward | Recipe | Noise | Eval pts | Ckpts | First | Best @ iter | Latest @ iter | Best-first | Latest-first | Latest-best | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sparse / r2_25-r1_55 / clean` | sparse | `r2_25-r1_55` | clean | 32 | 64 | 179.4 | 227.5 @ 180000 | 118.1 @ 308600 | +48.1 | -61.3 | -109.4 | completed |
| `sparse / r2_25-r1_55 / so10rep10` | sparse | `r2_25-r1_55` | so10rep10 | 31 | 31 | 127.4 | 241.8 @ 120000 | 129.3 @ 300000 | +114.4 | +1.9 | -112.5 | running |
| `sparse / ladder / clean` | sparse | ladder | clean | 30 | 30 | 134.6 | 198.3 @ 120000 | 105.8 @ 290000 | +63.6 | -28.9 | -92.5 | running |
| `sparse / ladder / so10rep10` | sparse | ladder | so10rep10 | 32 | 64 | 166.1 | 207.6 @ 160000 | 157.1 @ 304439 | +41.5 | -9.0 | -50.5 | completed |
| `sparse / r1_70+imm / clean` | sparse | `r1_70+imm` | clean | 32 | 64 | 207.8 | 346.1 @ 260000 | 182.6 @ 303909 | +138.4 | -25.1 | -163.5 | completed |
| `sparse / r1_70+imm / so10rep10` | sparse | `r1_70+imm` | so10rep10 | 33 | 66 | 131.0 | 341.4 @ 290000 | 245.4 @ 310893 | +210.4 | +114.4 | -96.0 | completed |
| `survbonusnoout / r2_25-r1_55 / clean` | no-outcome | `r2_25-r1_55` | clean | 29 | 29 | 130.9 | 201.9 @ 230000 | 196.1 @ 280000 | +71.0 | +65.3 | -5.8 | running |
| `survbonusnoout / r2_25-r1_55 / so10rep10` | no-outcome | `r2_25-r1_55` | so10rep10 | 32 | 64 | 92.1 | 251.3 @ 160000 | 176.6 @ 300865 | +159.1 | +84.5 | -74.6 | completed |
| `survbonusnoout / ladder / clean` | no-outcome | ladder | clean | 26 | 27 | 197.9 | 243.5 @ 120000 | 116.8 @ 250000 | +45.6 | -81.1 | -126.8 | running |
| `survbonusnoout / ladder / so10rep10` | no-outcome | ladder | so10rep10 | 29 | 29 | 149.9 | 230.9 @ 50000 | 144.6 @ 280000 | +81.0 | -5.3 | -86.3 | running |
| `survbonusnoout / r1_70+imm / clean` | no-outcome | `r1_70+imm` | clean | 23 | 24 | 159.0 | 221.8 @ 130000 | 154.3 @ 220000 | +62.8 | -4.8 | -67.5 | running |
| `survbonusnoout / r1_70+imm / so10rep10` | no-outcome | `r1_70+imm` | so10rep10 | 25 | 26 | 195.5 | 264.0 @ 80000 | 210.3 @ 240000 | +68.5 | +14.8 | -53.8 | running |
| `survbonusout / r2_25-r1_55 / clean` | plus-outcome | `r2_25-r1_55` | clean | 27 | 28 | 125.6 | 212.6 @ 150000 | 141.6 @ 260000 | +87.0 | +16.0 | -71.0 | running |
| `survbonusout / r2_25-r1_55 / so10rep10` | plus-outcome | `r2_25-r1_55` | so10rep10 | 25 | 26 | 157.4 | 220.3 @ 230000 | 202.1 @ 240000 | +62.9 | +44.8 | -18.1 | running |
| `survbonusout / ladder / clean` | plus-outcome | ladder | clean | 25 | 26 | 117.5 | 185.1 @ 80000 | 174.1 @ 240000 | +67.6 | +56.6 | -11.0 | running |
| `survbonusout / ladder / so10rep10` | plus-outcome | ladder | so10rep10 | 26 | 26 | 179.3 | 249.4 @ 140000 | 160.3 @ 250000 | +70.1 | -19.0 | -89.1 | running |
| `survbonusout / r1_70+imm / clean` | plus-outcome | `r1_70+imm` | clean | 24 | 25 | 228.1 | 298.1 @ 140000 | 295.6 @ 230000 | +70.0 | +67.5 | -2.5 | running |
| `survbonusout / r1_70+imm / so10rep10` | plus-outcome | `r1_70+imm` | so10rep10 | 23 | 23 | 198.1 | 286.6 @ 190000 | 246.1 @ 220000 | +88.5 | +48.0 | -40.5 | running |

## Tournament Facts

- Nonzero checkpoints did make it into the tournament. The corrected bounded
  live arena was seeded from the same 18 run ids, initially found `287`
  checkpoints, and then kept advancing.
- Current copied bounded latest artifact:
  `artifacts/local/curvytron_no_tournament_control_20260516/source/r18fresh_bounded_latest.json`.
  It is `round-000023`, `stable=false`, with `511` rated rows/checkpoints,
  `493` nonzero rows, `18` iteration-zero rows, `18` latest-for-run rows,
  max rated iteration `310893`, `300` rated pairs, `6,300` games,
  `decision_source_frames=1`, and `0` failed rating rows.
- Current docs record the earlier progression:
  `round-000002` had `304` rated rows through max iteration `220000`;
  `round-000008` had `365` rated rows through max iteration `260000`;
  `round-000011` had `398` rated rows through max iteration `290000`.
  The local copied latest therefore extends the same story to about `511` rated
  rows / `493` nonzero rows.
- Top tournament placements favor mid-run checkpoints, not latest checkpoints.
  In the local `round-000023` artifact, top 10 is `10/10` non-latest nonzero
  checkpoints with iterations `40000..210000`; top 30 is `28/30` non-latest
  nonzero, `2/30` iteration-zero, and `0/30` latest-for-run.
- Top-30 tournament composition: `15` plus-outcome, `12` sparse, `3`
  no-outcome; `19` clean and `11` so10rep10; `15` ladder recipe, `10`
  `r1_70+imm`, `5` `r2_25-r1_55`.

## Mechanical Loop Read

The mechanical loop worked:

- tournament-ranked checkpoints were published into training-candidate
  assignments;
- generation 9 and generation 10 proofs had all `18/18` trainers consuming the
  tournament-derived assignment shas;
- gen9/gen10 env-tail checks showed about `177k` target-sha rows each,
  `89,934` then `93,427` frozen-checkpoint provider-ok rows, and `0`
  provider-load false rows;
- scheduled generation 12 later had all still-running `15/15` trainers consume
  the fresh assignment shas, again with `0` provider-load false rows.

So the failure mode is not "checkpoints never reached tournament" or
"trainers never consumed tournament output." The failure mode is "the learner
often discovers a better policy and then does not retain it."

## What We Learned

1. r18fresh was making progress, but progress was not monotonic. Best checkpoints
   are meaningfully better than first checkpoints across all rows.
2. Latest checkpoints are the wrong promotion target unless they pass a
   retention gate. Best/mid-run checkpoint selection is likely better than
   latest-for-run selection for both survival and tournament placement.
3. The plus-outcome reward slice is the strongest latest-survival candidate, but
   the support-scale audit makes dense reward/value target saturation a serious
   suspect rather than a settled win.
4. The multi-rank ladder recipe looks suspect for survival retention. It appears
   in top tournament placements, especially sparse rows, but its survival
   latest-vs-first aggregate is negative.
5. Clean control did not protect against late regression. The stochastic lane
   had better latest survival in this read, but the action override/repeat
   mechanism remains a credit-assignment caveat.
6. The next useful control is not another proof that the loop can refresh. It is
   a retention-focused static or own-latest control with learner metrics,
   smaller/cleaner reward scales, and promotion based on best-or-heldout
   checkpoints instead of latest-only checkpoints.

