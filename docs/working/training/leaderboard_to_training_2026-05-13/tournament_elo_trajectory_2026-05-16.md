# Active Tournament Elo Trajectory, 2026-05-16

Scope: read-only inspection of active CurvyTron tournament
`curvy-r18fresh-live-bounded-dsf1-20260516b` / rating
`elo-r18fresh-live-bounded-dsf1-20260516b`. Inspected durable Modal Volume
artifacts through `latest.json` at `round-000030`; `round-000031` existed but
only had `input.json` and `progress.json` with no completed ratings yet.

## Plain Answers

1. Absolute/mean Elo is anchored, not expected to rise over time. In
   `batch_elo_v1`, new rows start at `initial_rating=1500.0`; each rated pair
   applies `delta_a` to one checkpoint and `-delta_a` to the other. Across
   completed rounds, the all-row mean stayed effectively exactly `1500.0`
   (`1499.9999999999986` to `1500.000000000003`). Rising top Elo means the
   pool's rating spread is widening, not that the whole population has an
   absolute score that should climb.
2. The tournament source-of-truth is being maintained: completed snapshots
   advanced from `287` rows / max iteration `210000` to `554` rows / max
   iteration `310893`; active/provisional/retired moved from `0/287/0` to
   `100/0/454`; nonzero checkpoint rows moved from `269` to `536`. The top
   band became almost entirely nonzero mid-run checkpoints, not latest
   checkpoints.
3. The rank-1 Elo series mostly trends up but is not monotonic and leadership
   changes. Current rank 1 did genuinely climb inside the tournament from
   `1504.0` at `round-000013` to `1635.3` at `round-000030`, with rank
   `196 -> 1`; however that is still relative Elo under an adaptive, changing
   opponent pool. It is not an absolute learning curve.
4. Evidence shows modest relative winners, not a clean breakthrough. The latest
   top policy is nonzero and well-tested (`2562` games, `118` distinct
   opponents), which is useful as a training opponent. But latest is
   `stable=false`, `max_abs_delta=41.3`, top-10 is `9/10` mid-run, and the
   leaders are cohort-relative. Treat tournament maintenance as healthy and
   learning quality as still mixed.

## Completed Round Trajectory

`A/P/R` is active/provisional/retired. `T10 nz/latest/mid` counts nonzero,
latest-for-run, and nonzero-not-latest rows among the top 10.

| Round | Rows | A/P/R | Nonzero | Max iter | Top1 | Top5 avg | Top10 avg | Spread | Max delta | Top1 policy | T10 nz/latest/mid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r000000 | 287 | 0/287/0 | 269 | 210000 | 1516.8 | 1512.7 | 1511.1 | 38.0 | 21.2 | 80k mid/prov | 9/1/8 |
| r000001 | 296 | 0/296/0 | 278 | 210000 | 1521.8 | 1517.5 | 1514.8 | 48.7 | 16.5 | 80k mid/prov | 8/0/8 |
| r000002 | 304 | 0/304/0 | 286 | 220000 | 1522.8 | 1521.6 | 1519.7 | 58.1 | 12.7 | 100k mid/prov | 8/1/7 |
| r000003 | 316 | 0/316/0 | 298 | 230000 | 1526.6 | 1524.2 | 1521.9 | 74.1 | 12.3 | 20k mid/prov | 7/0/7 |
| r000004 | 319 | 0/319/0 | 301 | 230000 | 1540.6 | 1529.2 | 1525.8 | 91.7 | 14.0 | 20k mid/prov | 10/0/10 |
| r000005 | 330 | 0/330/0 | 312 | 240000 | 1541.1 | 1532.5 | 1528.3 | 95.4 | 12.8 | 20k mid/prov | 10/0/10 |
| r000006 | 338 | 0/338/0 | 320 | 250000 | 1545.5 | 1534.9 | 1530.4 | 100.5 | 12.9 | 20k mid/prov | 10/0/10 |
| r000007 | 349 | 24/319/6 | 331 | 260000 | 1538.9 | 1534.1 | 1530.4 | 96.6 | 10.2 | 20k mid/active | 10/0/10 |
| r000008 | 365 | 25/335/5 | 347 | 260000 | 1560.3 | 1551.2 | 1544.7 | 121.2 | 24.9 | 40k mid/active | 9/0/9 |
| r000009 | 373 | 24/343/6 | 355 | 270000 | 1562.5 | 1556.4 | 1550.6 | 128.1 | 19.3 | 40k mid/active | 9/0/9 |
| r000010 | 387 | 24/357/6 | 369 | 280000 | 1575.8 | 1561.0 | 1553.2 | 144.6 | 14.9 | 120k mid/active | 9/0/9 |
| r000011 | 398 | 27/357/14 | 380 | 290000 | 1573.4 | 1563.2 | 1554.0 | 145.5 | 20.5 | 40k mid/active | 9/0/9 |
| r000012 | 404 | 47/306/51 | 386 | 290000 | 1563.4 | 1557.2 | 1554.2 | 138.1 | 27.9 | 120k mid/active | 8/0/8 |
| r000013 | 442 | 59/303/80 | 424 | 310893 | 1567.2 | 1561.3 | 1557.1 | 146.4 | 16.3 | 150k mid/active | 10/0/10 |
| r000014 | 449 | 78/261/110 | 431 | 310893 | 1569.6 | 1563.1 | 1557.8 | 152.3 | 18.6 | 40k mid/active | 8/0/8 |
| r000015 | 454 | 80/193/181 | 436 | 310893 | 1576.4 | 1566.5 | 1558.6 | 161.4 | 12.9 | 150k mid/active | 9/0/9 |
| r000016 | 468 | 78/159/231 | 450 | 310893 | 1577.8 | 1565.6 | 1558.3 | 162.0 | 16.7 | 150k mid/active | 10/0/10 |
| r000017 | 473 | 73/161/239 | 455 | 310893 | 1580.1 | 1566.4 | 1560.3 | 164.3 | 13.8 | 120k mid/active | 9/0/9 |
| r000018 | 483 | 71/167/245 | 465 | 310893 | 1582.2 | 1566.9 | 1559.5 | 166.4 | 13.9 | 120k mid/active | 10/0/10 |
| r000019 | 486 | 72/165/249 | 468 | 310893 | 1579.6 | 1569.7 | 1561.3 | 163.8 | 10.9 | 120k mid/active | 10/0/10 |
| r000020 | 495 | 69/174/252 | 477 | 310893 | 1574.4 | 1568.1 | 1560.4 | 158.6 | 12.6 | 120k mid/active | 10/0/10 |
| r000021 | 500 | 67/170/263 | 482 | 310893 | 1574.4 | 1567.2 | 1561.2 | 158.6 | 14.0 | 40k mid/active | 10/0/10 |
| r000022 | 506 | 63/175/268 | 488 | 310893 | 1581.8 | 1568.5 | 1562.0 | 166.0 | 13.3 | 40k mid/active | 10/0/10 |
| r000023 | 511 | 72/132/307 | 493 | 310893 | 1578.0 | 1568.2 | 1563.4 | 162.2 | 15.5 | 40k mid/active | 10/0/10 |
| r000024 | 520 | 95/66/359 | 502 | 310893 | 1578.4 | 1567.6 | 1563.0 | 162.6 | 14.7 | 40k mid/active | 9/0/9 |
| r000025 | 525 | 96/42/387 | 507 | 310893 | 1580.6 | 1570.6 | 1564.7 | 164.8 | 24.2 | 40k mid/active | 10/0/10 |
| r000026 | 530 | 97/24/409 | 512 | 310893 | 1587.7 | 1577.0 | 1568.5 | 171.9 | 22.3 | 180k mid/active | 10/0/10 |
| r000027 | 537 | 100/0/437 | 519 | 310893 | 1596.2 | 1582.2 | 1572.0 | 180.3 | 63.3 | 180k mid/active | 10/0/10 |
| r000028 | 543 | 100/0/443 | 525 | 310893 | 1606.6 | 1587.1 | 1577.1 | 190.8 | 45.4 | 180k mid/active | 10/0/10 |
| r000029 | 549 | 100/0/449 | 531 | 310893 | 1627.5 | 1601.4 | 1586.3 | 211.7 | 108.5 | 180k mid/active | 10/1/9 |
| r000030 | 554 | 100/0/454 | 536 | 310893 | 1635.3 | 1607.6 | 1590.7 | 219.5 | 41.3 | 180k mid/active | 10/1/9 |

## Rank-1 Behavior

The rank-1 score series was:

`1516.8, 1521.8, 1522.8, 1526.6, 1540.6, 1541.1, 1545.5, 1538.9, 1560.3, 1562.5, 1575.8, 1573.4, 1563.4, 1567.2, 1569.6, 1576.4, 1577.8, 1580.1, 1582.2, 1579.6, 1574.4, 1574.4, 1581.8, 1578.0, 1578.4, 1580.6, 1587.7, 1596.2, 1606.6, 1627.5, 1635.3`.

That is an upward upper-tail trend with visible dips and leader churn. There
were `7` unique rank-1 checkpoint IDs. The current top checkpoint is:

`ckpt-432-train-lightzero_exp-ckpt-iteration_180000-0ed114de`

`training/lightzero-curvytron-visual-survival/curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/attempts/try-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/train/lightzero_exp/ckpt/iteration_180000.pth.tar`

Its own rating trajectory after entering the pool:

| Round | Rating | Rank | Status | Games | Opponents | Delta |
|---|---:|---:|---|---:|---:|---:|
| r000013 | 1504.0 | 196 | provisional | 21 | 1 | +4.0 |
| r000014 | 1509.8 | 140 | provisional | 42 | 2 | +5.9 |
| r000015 | 1512.6 | 107 | provisional | 63 | 3 | +2.8 |
| r000016 | 1515.2 | 92 | provisional | 105 | 5 | +2.6 |
| r000017 | 1526.2 | 55 | provisional | 147 | 7 | +11.0 |
| r000018 | 1529.5 | 53 | provisional | 189 | 9 | +3.2 |
| r000019 | 1533.1 | 46 | provisional | 231 | 11 | +3.6 |
| r000020 | 1544.7 | 16 | provisional | 273 | 13 | +11.6 |
| r000021 | 1554.3 | 8 | provisional | 315 | 15 | +9.6 |
| r000022 | 1561.7 | 4 | provisional | 357 | 17 | +7.4 |
| r000023 | 1564.2 | 4 | provisional | 399 | 19 | +2.5 |
| r000024 | 1568.3 | 2 | active | 420 | 20 | +4.1 |
| r000025 | 1579.4 | 2 | active | 504 | 24 | +11.1 |
| r000026 | 1587.7 | 1 | active | 588 | 28 | +8.3 |
| r000027 | 1596.2 | 1 | active | 987 | 47 | +8.5 |
| r000028 | 1606.6 | 1 | active | 1491 | 71 | +10.4 |
| r000029 | 1627.5 | 1 | active | 2016 | 94 | +20.9 |
| r000030 | 1635.3 | 1 | active | 2562 | 118 | +7.8 |

Interpretation: this specific checkpoint has a real tournament-relative climb,
but it is still an Elo estimate inside a moving, adaptively scheduled pool.
The top policy is a nonzero mid-run checkpoint, not the latest checkpoint for
its run.

## Latest Top 10

| Rank | Rating | Status | Games | Opponents | Iter | Latest? | Notes |
|---:|---:|---|---:|---:|---:|---|---|
| 1 | 1635.3 | active | 2562 | 118 | 180000 | false | current top, mid-run |
| 2 | 1609.5 | active | 945 | 41 | 200000 | false | mid-run |
| 3 | 1607.2 | active | 1092 | 49 | 190000 | false | mid-run |
| 4 | 1605.1 | active | 588 | 27 | 250000 | true | only latest in top 10 |
| 5 | 1580.7 | active | 1596 | 71 | 150000 | false | mid-run |
| 6 | 1579.1 | active | 903 | 40 | 240000 | false | mid-run |
| 7 | 1574.6 | active | 861 | 39 | 150000 | false | mid-run |
| 8 | 1574.3 | active | 1071 | 47 | 220000 | false | mid-run |
| 9 | 1570.8 | active | 987 | 46 | 150000 | false | mid-run |
| 10 | 1570.0 | active | 840 | 39 | 210000 | false | mid-run |

## Evidence Quality

This is good evidence that the tournament store, active-pool retirement, and
training-candidate source-of-truth machinery are alive. It is weaker evidence
for learning quality. The latest snapshot has a broader spread and a clear
relative top band, but `stable=false`; the last completed round still moved by
`41.3` Elo, and `round-000029` moved by `108.5`. The top band being mostly
mid-run rather than latest also matches the separate learning-health readout:
some runs discover better checkpoints and later regress.

Bottom line: use the top rows as useful tournament-selected opponents, but do
not call this a clear policy breakthrough from Elo alone.

## Fields And Artifacts Used

Primary durable artifacts:

- `tournaments/curvytron/curvy-r18fresh-live-bounded-dsf1-20260516b/ratings/elo-r18fresh-live-bounded-dsf1-20260516b/latest.json`
- `tournaments/curvytron/curvy-r18fresh-live-bounded-dsf1-20260516b/ratings/elo-r18fresh-live-bounded-dsf1-20260516b/config.json`
- `tournaments/curvytron/curvy-r18fresh-live-bounded-dsf1-20260516b/ratings/elo-r18fresh-live-bounded-dsf1-20260516b/progress.json`
- `tournaments/curvytron/curvy-r18fresh-live-bounded-dsf1-20260516b/ratings/elo-r18fresh-live-bounded-dsf1-20260516b/rounds/round-000000..round-000030/ratings.json`
- `tournaments/curvytron/curvy-r18fresh-live-bounded-dsf1-20260516b/ratings/elo-r18fresh-live-bounded-dsf1-20260516b/rounds/round-000031/progress.json`

Fields used:

- Snapshot: `round_id`, `ratings`, `checkpoint_count`, `max_abs_delta`,
  `stable`, `pair_count`, `game_count`, `rating_spec`.
- Rating rows: `rank`, `rating`, `previous_rating`, `last_round_delta`,
  `status`, `games`, `distinct_opponents`, `iteration`, `latest_for_run`,
  `checkpoint_id`, `checkpoint_ref`, `run_id`.
- Rating config/spec: `initial_rating`, `base_k`, `k_min`, `k_max`,
  `delta_clamp`, `active_pool_limit`, `placement_min_games`,
  `placement_min_opponents`, `pair_selection`, `pairs_per_round`,
  `games_per_pair`, `decision_source_frames`, `policy_mode`,
  `seat_order_mode`.

Code checked for Elo mechanics:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`
  - `elo_expected_score`
  - `elo_k_for_games`
  - `rating_snapshot_from_pair_results`
  - `_base_rating_rows`
- `src/curvyzero/tournament/curvytron/contracts.py`
- `src/curvyzero/contracts/curvytron.py`

## Reproduction Commands

```bash
T=curvy-r18fresh-live-bounded-dsf1-20260516b
R=elo-r18fresh-live-bounded-dsf1-20260516b
TVOL=curvyzero-curvytron-tournaments-v2
BASE=tournaments/curvytron/$T/ratings/$R

uv run --extra modal modal volume ls "$TVOL" "$BASE"
uv run --extra modal modal volume ls "$TVOL" "$BASE/rounds"
uv run --extra modal modal volume get "$TVOL" "$BASE/latest.json" - \
  | jq '{round_id, checkpoint_count, max_abs_delta, stable, rating_spec}'
uv run --extra modal modal volume get "$TVOL" "$BASE/rounds/round-000030/ratings.json" - \
  | jq '.ratings[:10] | map({rank, rating, status, games, distinct_opponents, iteration, latest_for_run, checkpoint_id, checkpoint_ref})'
uv run --extra modal modal volume get "$TVOL" "$BASE/rounds/round-000031/progress.json" - \
  | jq '{round_id, status, pair_count, game_count, completed_pair_count, completed_game_count, ratings_written, updated_at}'
```

The per-round table was generated by looping over
`$BASE/rounds/round-*/ratings.json`, sorting `.ratings` by descending
`rating`, and aggregating counts from `status`, `iteration`, and
`latest_for_run`.
