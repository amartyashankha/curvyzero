# Restart18 Nonzero Source Rerate Gate Audit, 2026-05-15

Independent audit verdict for the optional leaderboard-derived top-slot source:
round 6 changed the decision. The 96-row rerate is coverage-mature and
failure-free, but it is still not publishable as a trusted ranked source because
the latest completed snapshot is `stable=false`; worse, the round-6 stability
delta rose again after round 5. This does not block bootstrap training with
curated exact checkpoint refs.

Inspected local code:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`
- `src/curvyzero/training/opponent_leaderboard.py`
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- `tests/test_curvytron_checkpoint_tournament.py`
- `scripts/prepare_curvytron_restart_source_refs.py`

Inspected Modal Volume artifacts:

- `tournaments/curvytron/curvy-restart18-source-rerate-nonzero-20260515a/ratings/elo-restart18-source-rerate-nonzero-20260515a/latest.json`
- `tournaments/curvytron/curvy-restart18-source-rerate-nonzero-20260515a/ratings/elo-restart18-source-rerate-nonzero-20260515a/rounds/round-000003/ratings.json`
- `tournaments/curvytron/curvy-restart18-source-rerate-nonzero-20260515a/ratings/elo-restart18-source-rerate-nonzero-20260515a/rounds/round-000005/ratings.json`
- `tournaments/curvytron/curvy-restart18-source-rerate-nonzero-20260515a/ratings/elo-restart18-source-rerate-nonzero-20260515a/rounds/round-000006/input.json`

Current latest snapshot:

- latest completed round: `round-000006`
- latest Modal call/app: `fc-01KRPR02D72FZ27G5KED70GFBC` /
  `ap-EqV1pzucLCW8fZjMA3FEqM`
- rows: `96`
- pairs/games: `300` pairs / `6300` games
- failures: `0`
- stability: `stable=false`, `max_abs_delta=25.199213332028748`
- active rows: `96`
- provisional rows: `0`
- coverage: `693-2226` games, `30-92` distinct opponents
- rating context: `context_hash=3e1af9183db39818`,
  `roster_hash=d2563608441af000`

Exact active-row gates:

- Tournament rating row status is `active` only when both gates pass:
  `games >= placement_min_games` and
  `distinct_opponents >= min(placement_min_opponents, roster_size - 1)`.
- This run's rating spec sets `placement_min_games=420`,
  `placement_min_opponents=20`, `games_per_pair=21`,
  `active_pool_limit=100`, and `stop_max_delta=10.0`.
- The training-facing public leaderboard builder has its own default active
  policy: `active_min_valid_games=300`, `active_min_distinct_opponents=20`,
  `max_failure_rate=0.02`, and `max_active_rank=100`. For this source, the
  tournament status gate is currently stricter on games (`420`), while both
  require `20` opponents.

Round 3 / round 4 / round 5 / round 6 audit:

- Round 3 completed as a continuation, not a restart:
  `previous_round_id=round-000002`, `continue_from_latest=true`, same
  `context_hash=3e1af9183db39818`, same
  `roster_hash=d2563608441af000`.
- Round 3 cleared coverage for all `96` rows but did not clear stability:
  `stable=false`, `max_abs_delta=39.7420779825474`.
- Round 4 input is healthy and preserves the evidence chain:
  `previous_round_id=round-000003`, same `context_hash=3e1af9183db39818`,
  same `roster_hash=d2563608441af000`, full `96`-checkpoint roster, and
  `300` pairs / `6300` planned games.
- Round 4 completed cleanly with `0` failures and all rows active, but still
  failed the publish stability gate: `stable=false`,
  `max_abs_delta=17.371056613899057`.
- Round 5 ran as the same-context continuation with sampled logs showing
  balanced randomized seats, `max_steps=1048576`, and real game completions.
- Round 5 completed cleanly with `0` failures and all rows active, but still
  failed the publish stability gate: `stable=false`,
  `max_abs_delta=15.636412948237727`.
- Round 6 input was healthy and preserved the evidence chain:
  `previous_round_id=round-000005`, same `context_hash=3e1af9183db39818`,
  same `roster_hash=d2563608441af000`, full `96`-checkpoint roster, and
  `300` pairs / `6300` planned games.
- Round 6 completed cleanly with `0` failures and all rows active, but the
  stability delta worsened: `stable=false`,
  `max_abs_delta=25.199213332028748`.
- The biggest round-6 mover was
  `ckpt-079-train-lightzero_exp-ckpt-iteration_240000-a391d866`, which rose
  from rank 21 to rank 7. It played `11` scheduled pairs / `231` games in
  round 6 with `146` wins, `82` losses, `3` draws, and no failures. Its
  schedule was mostly `random_bridge` exposure (`8` of `11` pairs), so the next
  action is to check whether adaptive scheduling is producing high-leverage
  max-delta swings rather than true source convergence.

Recommendation for leaderboard-derived top slots:

- Do not publish or materialize trusted ranked top slots from this rerate.
- Do not block bootstrap training on this rerate.
- Bootstrap training may launch from curated exact checkpoint refs, immortal
  hard-coded/blank pressure, and live tournament intake.
- Before another blind continuation, inspect high movers, their pair outcomes,
  prior battle counts, and scheduler reasons. If the same pattern explains the
  instability, run a targeted confirmation diagnostic or adjust scheduling
  before spending more rounds.
- Do not publish or materialize leaderboard-derived top slots from this rerate
  until the latest completed snapshot is both `stable=true` and
  coverage-mature.
- Do not lower K, raise `stop_max_delta`, switch to all-pairs, or publish
  `stable=false` just to move faster.

Publishable source definition:

1. Latest snapshot is non-provisional and `stable=true`
   (`max_abs_delta <= 10.0` for the current spec).
2. The selected source rows are coverage-mature: for this rating run, active
   rows have at least `420` valid games and `20` distinct opponents, with no
   failure-rate surprise.
3. Publish uses explicit expected guards for round id/index, rating context
   hash, roster hash, and rating snapshot SHA.
4. The resulting public leaderboard has active rows and is then materialized via
   stable-slot assignment into the `control` target volume before restart18
   launch.
