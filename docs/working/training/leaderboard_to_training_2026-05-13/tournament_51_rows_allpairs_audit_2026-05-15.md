# Tournament 51-Row / All-Pairs Audit - 2026-05-15

Finding: the live leaderboard can show only 51 ranked checkpoints while newer
checkpoint refs have landed because the standings API reads the last completed
`latest.json`. Main-thread evidence says current
`curvy-v2refresh18p-live-20260514b / elo-v2refresh18p-live-20260514b`
`latest.json` is still `round-000004`. The later `round-000005` has a much
larger input pool but is still running, so it has not replaced `latest.json`.

The large round shape is explained by scheduler policy, not by missing
checkpoints:

- `round-000005` with 193 checkpoints and unordered non-self `all_pairs`
  produces `193 * 192 / 2 = 18,528` pairs.
- With `games_per_pair=21`, that is `389,088` games.
- A local source reproduction with a synthetic 51-row latest plus 193 desired
  refs produced exactly `18,528` pairs under `all_pairs`.
- The same state under `adaptive_v0` with `pairs_per_round=300` produced `300`
  pairs / `6,300` games.

Code refs:

- `curvyzero_checkpoint_tournament.py::_read_best_rating_snapshot_for_run`
  returns final `latest.json` first whenever it exists, even when live
  provisional is allowed.
- `curvytron_checkpoint_tournament.py::build_rating_round_pair_specs` applies
  `_schedulable_rating_checkpoints`, then for `all_pairs` enumerates the whole
  schedulable pool unless `pairs_per_round` is set.
- `_schedulable_rating_checkpoints` keeps top active rows up to
  `active_pool_limit`, but also includes every provisional/new checkpoint. With
  51 completed rows plus roughly 142 new refs, the schedulable pool is still 193.
- Intake defaults currently default to `pair_selection=all_pairs`; adaptive must
  be explicitly configured with `pairs_per_round`.

Verdict: expected behavior for the current configured policy, but operationally
wrong for a live growing pool. Active-pool limiting did not cap the new refs, and
`all_pairs` expanded the whole schedulable pool quadratically.

Safe plan:

1. Do not wait for the giant `round-000005` unless compute burn is acceptable.
2. Stop or supersede the running Modal app/call for this rating round.
3. Start a fresh rating run id, or disable the current intake and re-seed it,
   with `pair_selection=adaptive_v0`, explicit `pairs_per_round` such as 300,
   `active_pool_limit=100`, `games_per_pair=21`, one-frame cadence, and the
   validated render contract.
4. Use startup probe/estimate before spawning; expected shape should be bounded
   by `pairs_per_round * games_per_pair`, not `N choose 2`.
5. Add a regression test: with a 51-row previous latest and 193 desired refs,
   intake continuation configured as live/adaptive schedules 300 pairs, while
   `all_pairs` is only allowed when explicitly requested as a stress run.
