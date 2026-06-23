# Experiment Log

Last updated: 2026-05-16.

This is append-only for scheduler research.

## Results Index

| Experiment | Status | Biggest Finding | Changed Decision |
| --- | --- | --- | --- |
| EXP-001 standalone toy schedulers | complete-first-pass | Top-heavy placement can protect strong new rows but spends more work; random coverage is strong in simple worlds; toy all-pairs is too expensive even at modest sizes. | Need current-code multi-wave coverage probe before choosing V1. |
| EXP-002 current-code one-wave adaptive probe | complete-first-pass | `adaptive_v0` stayed at 300 pairs, but 100 established + 500 new left 200 new rows untouched in the first wave. | Need time-to-first-evidence and time-to-20-opponents metrics. |
| EXP-003 current-code multi-wave coverage probe | complete-first-pass | With 100 established + 500 new, eight 300-pair waves gave every new row first evidence but weakest new rows only four distinct opponents. | Burst intake needs a coverage plan beyond a flat 300-pair wave. |
| EXP-004 online-loop toy regressions | complete-local | Fake game outcomes now prove weak entrants can be placed then retired, clone/draw swarms keep exactly top 100 active, and 1000 old + 50 new scheduling stays budgeted/top-100-only. | Keep these as the local gate before Modal scale experiments. |
| EXP-005 Modal fanout comparison | complete-small-remote | `games_per_shard=1` completed the same 210-game five-checkpoint probe cleanly and better matches speed-of-one-game than `games_per_shard=21`. | Default should be game-level fanout; solve Volume pressure directly. |
| EXP-006 18-checkpoint game fanout | complete-remote | 18 latest checkpoints ran 153 battles / 3,213 games with one Modal call per game and wrote ratings. | The game worker path scales at this size; completed progress must read `latest.json` first instead of scanning all game summaries. |

## 2026-05-16 - Research Phase Opened

Reason:

- More than 100 runs/checkpoints makes all-pairs too expensive as the default.
- We need to understand bounded online scheduling before changing production
  behavior.

Known prior evidence from earlier red-team notes:

- 424 all-new checkpoints with a 20-opponent placement target can expand a
  212-pair request into 4,240 pairs.
- 900 new plus 100 established checkpoints can make the best established policy
  receive every placement game if placement has no anchor cap.
- Same-run-heavy pools can spend most games inside one lineage.
- Top-band bias can starve lower rows once placement is satisfied.
- Scalar Elo can miss non-transitive counter-policies if the schedule misses
  key edges.

Next experiments:

- Run a small standalone simulation for 100 established plus 20 new policies.
- Run a burst simulation for 100 established plus 500 new policies.
- Compare all-pairs oracle, random stratified, top-anchor placement,
  spread-anchor placement, and binary ladder placement.

## EXP-001 - Standalone Toy Scheduler First Pass

Hypothesis:

- Cheap bounded schedulers can recover useful top pools with far fewer games
  than all-pairs, but each has a different failure mode.

World:

- Synthetic hidden strengths.
- Cases: `100 + 20`, `100 + 500`, strong new batch, non-transitive styles.

Schedulers:

- all-pairs oracle.
- random coverage.
- single-best-anchor.
- top-anchor.
- spread-anchor.
- binary ladder.

Result:

- All-pairs is too expensive: `100 + 500` is 179,700 pairs and 3,773,700 games
  at 21 games per pair.
- Random coverage is a strong baseline in simple worlds.
- Top-anchor placement finds strong new policies quickly, but spends more work
  and needs anchor caps.
- Spread-anchor and binary-ladder need more careful implementation before they
  beat random coverage in the burst case.
- The toy harness is useful for direction, not product claims.

Follow-ups:

- FU-001, FU-002, FU-005.

## EXP-002 - Current-Code One-Wave Adaptive Probe

Hypothesis:

- Current `adaptive_v0` should keep work bounded under burst intake.

World:

- Current scheduler helper only. No games run.
- `100 established + 20 new`.
- `100 established + 500 new`.
- `0 established + 424 all new`.

Result:

- `100 + 20`: requested 300 pairs, scheduled 300, all 20 new rows touched.
- `100 + 500`: requested 300 pairs, scheduled 300, 200 new rows had zero
  appearances in the first wave.
- `424 all new`: requested 300 pairs, scheduled 300, all rows touched at least
  once because new rows can cover each other.

Interpretation:

- The current code is no longer doing the old placement explosion in these
  probes.
- The remaining problem is multi-wave coverage and top-100 false drops, not
  first-wave pair-budget explosion.

Follow-ups:

- FU-001, FU-005.

## EXP-003 - Current-Code Multi-Wave Coverage Probe

Hypothesis:

- Current `adaptive_v0` can eventually cover large new-checkpoint bursts, but we
  need to know how many waves it takes.

World:

- Current scheduler helper only. No games run.
- Existing rows were synthetic active rows.
- New rows were synthetic provisional rows.
- Pair budget was 300 pairs per wave.
- Placement target was 20 distinct opponents.

Results:

`100 established + 20 new`:

- Wave 1: all new rows touched, each got 15 distinct opponents.
- Wave 2: weakest new rows reached 22 distinct opponents and could become
  active under the synthetic status rule.

`100 established + 500 new`:

- Wave 1: 200 new rows had zero games.
- Wave 2: every new row had at least one game.
- Wave 8: weakest new rows had only 4 distinct opponents, p10 was 4, best new
  rows had 5.
- Expected wave count to reach 20 distinct opponents for every new row is about
  34 waves at 300 pairs per wave.
- Placement alone is roughly `500 * 20 = 10,000` pairs, or about 210,000 games
  at 21 games per pair.
- The scheduler stayed bounded at 300 pairs per wave and kept choosing
  placement pairs.

`0 established + 424 all new`:

- Wave 1 touched every row.
- Wave 8: weakest rows had 10 distinct opponents, p10 was 10, best rows had 16.
- No rows became active because the synthetic status rule required 20 opponents
  and 420 games.

Interpretation:

- The old unbounded placement explosion is not present in this probe.
- The current flat 300-pair wave is too slow for a large `100 + 500` burst if we
  need timely 20-opponent evidence.
- Established-first placement usually gives new-checkpoint evidence to only one
  new row per pair. New-vs-new placement would cover two new rows per pair, but
  it must be mixed with established anchors so ratings connect to the trusted
  pool.
- When all rows are new, new-vs-new placement gives faster broad coverage than
  when every new row must also find established evidence.
- A burst-aware scheduler should report time-to-first-evidence and
  time-to-placement-target before starting a large run.

Follow-ups:

- FU-001: decide whether live intake needs a burst-specific pair budget.
- FU-005: test false top-100 drops under delayed coverage.
- FU-006: turn this into a pass/fail gate.

## EXP-004 - Online-Loop Toy Regressions

Hypothesis:

- Before launching more Modal work, we need a cheap local proof that scheduler,
  fake outcomes, Elo reduction, top-100 retirement, and public leaderboard
  status agree.

World:

- Pure Python tests. No CurvyTron games, no Modal workers, no Volume writes.
- Synthetic checkpoint refs use timestamped `lightzero_exp*` paths so the tests
  do not teach stale fixed-path habits.

Cases:

- `100 established + 10 weak new`: new rows receive placement games, lose them,
  fall below the top-100 active pool, and are excluded from the next schedule.
- `120 clone/draw swarm`: all-pairs draw outcomes produce zero Elo movement,
  deterministic ranking, 100 active rows, and 20 retired rows.
- `1000 established + 50 new`: one adaptive wave schedules 1000 unique pairs,
  touches all 50 new rows, and does not schedule old rank-101+ rows.

Result:

- Added `tests/test_curvytron_tournament_online_simulation.py`.
- Local validation:
  - `uv run pytest tests/test_curvytron_tournament_online_simulation.py -q`
    passed: `3 passed in 11.94s`.
  - `uv run ruff check tests/test_curvytron_tournament_online_simulation.py`
    passed.
  - Existing scheduler/shard focused suite passed:
    `9 passed in 1.54s`.

Interpretation:

- The local scheduler/rating/top-100 contract is no longer only a doc claim.
- The current adaptive scheduler can keep a 1050-row roster bounded and focused
  on top-100 plus new entrants.
- This still does not prove Modal throughput, checkpoint loading, or real
  policy quality. It is the gate before those tests.

Follow-ups:

- Run a deployed shard fanout probe with `save_gif=false`.
- Add preflight output for expected shard calls, worker warm-pool settings, and
  Volume commit/reload risk before any large run.
- Keep testing burst coverage: flat 300-pair waves are still too slow for
  timely 20-opponent evidence when hundreds of new checkpoints arrive.

## EXP-005 - Modal Fanout Comparison

Hypothesis:

- The default should make games embarrassingly parallel. If each battle has 21
  games, those 21 games should be allowed to run at the same time instead of
  serially inside one shard worker.

Runs:

- `curvy-scale-probe-5latest-nogif-20260516a`
  - five latest `curvy-r18fresh` checkpoints;
  - 10 all-pairs battles;
  - 21 games per battle;
  - `games_per_shard=21`;
  - `save_gif=false`;
  - expected 10 worker calls.

- `curvy-scale-probe-5latest-gamefanout-20260516a`
  - same checkpoint count and game count;
  - `games_per_shard=1`;
  - `save_gif=false`;
  - expected 210 worker calls.

Result:

- Both completed cleanly:
  - 210/210 games;
  - 10/10 pairs;
  - 10 rated pairs;
  - ratings written;
  - no failed games.
- The game-fanout run matched the product goal better because every game can be
  a separate Modal call.

Decision:

- Set `DEFAULT_GAMES_PER_SHARD=1`.
- Treat larger shards as an explicit cost-saving knob, not the default.
- Next bottleneck to fix is Volume churn:
  - each game worker reloads checkpoint Volume;
  - each game worker commits tournament Volume;
  - progress/provisional readers need to count game-summary-only rounds cleanly;
  - commit errors should be surfaced in compact results.

## EXP-006 - 18-Checkpoint Game Fanout

Hypothesis:

- The real default shape should still work when we move from five checkpoints to
  the current small live set.

Run:

- `curvy-scale-probe-18latest-gamefanout-20260516a`
- rating run `elo-scale-probe-18latest-gamefanout-20260516a`
- 18 latest `curvy-r18fresh` checkpoints discovered from
  `train/lightzero_exp*/ckpt/iteration_*.pth.tar`;
- `pair_selection=adaptive_v0`;
- 153 battles;
- 21 games per battle;
- `games_per_shard=1`;
- `save_gif=false`;
- `max_steps=512`;
- `num_simulations=1`.

Result:

- 3,213 / 3,213 games completed.
- 153 / 153 battles completed and rated.
- Ratings were written.
- Failed game count was 0.
- `stable=false` because this was only one fresh round and Elo moved a lot.
  That is not a plumbing failure.

Fix made after the run:

- Completed progress now reads the small `latest.json` snapshot first, so a
  finished game-level round does not scan thousands of game summary files just
  to say it is complete.
- Live progress now counts game-summary-only rounds when `games_per_shard=1`.
- Compact game results now keep checkpoint reload, commit, and summary rewrite
  errors instead of hiding them.

Validation:

- `uv run pytest ...` focused tournament/scheduler/progress gate:
  19 tests passed.
- `uv run ruff check ...` passed.
- `uv run python -m py_compile ...` passed.
- Remote progress read for the completed 18-checkpoint run returned
  `count_basis=latest_snapshot`, `3213/3213`, `phase=ratings_written`.

Next scale gate:

- Run a larger no-GIF game-level fanout with completed-progress polling.
- Watch game-worker `commit_seconds`, `checkpoint_reload_seconds`, and any
  surfaced commit/reload errors.
- If Volume commits dominate, fix the commit pattern directly instead of hiding
  the issue behind large shards.
