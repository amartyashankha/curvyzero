# Top-100 One-Frame Tournament Lane - 2026-05-14

## Current Breadcrumbs

Current launch gate, 2026-05-14 03:14 ET:

- tournament_id: `curvy-oneframe-top100-gate-20260514a`
- rating_run_id: `elo-oneframe-top100-gate-20260514a`
- launch app: `ap-CyLcpfOewtS1bkJwfJ5OtH`
- rating function call id: `fc-01KRJNC1YY3QDQ57V4N27A6G5P`
- source refs: first 100 exact checkpoint refs from
  `/private/tmp/curvy-exact212-refs.csv`, written to
  `/private/tmp/curvy-top100-refs.csv`
- settings: one-frame, `adaptive_v0`, 1000 pairs, 21 games per pair, GIFs on,
  5 GIF samples per pair, `placement_min_opponents=20`,
  `placement_min_games=420`, `active_pool_limit=100`
- purpose: practical launch gate for the H100 18-run batch, not a full
  exhaustive tournament and not the old exact-212 stress lane.
- first monitor: started pairs advanced to 159/1000, estimated seen games around
  3339/21000, and the detached app had hundreds of active Modal tasks. This is
  a live tournament, not the earlier dead-launch shape.
- later monitor: started pairs advanced to 253/1000, estimated seen games around
  5313/21000, with game logs showing `ok=true` game results. Pair summaries had
  not reduced yet at that checkpoint, so continue monitoring until `latest.json`
  is written.
- 03:27 ET monitor: started pairs advanced to 365/1000, estimated seen games
  around 7665/21000. Pair summaries were still zero; current interpretation is
  that pair workers are waiting for all 21 game workers in each pair before
  writing pair summaries. Keep watching for reduction rather than launching off
  provisional rows.
- Critique note: zero completed pairs is not itself a failure while individual
  game workers are still writing `ok=true` summaries. It becomes a red flag only
  if exact summary progress shows no completed games, or if the detached app
  loses workers and `latest.json` never appears.
- 03:38 ET monitor: started pairs advanced to 474/1000, estimated seen games
  around 9954/21000. `latest.json` was still absent. Continue waiting; this is
  about half done, not a dead gate.

Do not block the H100 launch on exact-212 unless this practical gate fails to
produce active top rows.

Status as of 2026-05-14 after the Modal launch-lifetime finding:

- First batch launched and completed one bounded rating round.
- Do not launch a duplicate tournament for these IDs.
- Second batch was submitted and a continuation spawn was attempted for the same
  IDs, but non-detached `modal run` was not a safe parent for the background
  child game workers.
- Pause further injection/launch attempts until the continuation is relaunched
  with `modal run --detach`, through a deployed path that keeps work alive, or
  with a parent command that waits for the child work to finish.

## IDs

- tournament_id: `arena-oneframe-top100-plus-latest-20260514a`
- rating_run_id: `elo-oneframe-top100-plus-latest-20260514a`
- rating function call id from seed/spawn: `fc-01KRJFVHQQAPQ9YSNS5DM7RQJN`
- second-wave continuation call id: `fc-01KRJGB2BXTY7D5AD6PDW9RZAB`

## Historical Ranked Source

- selected historical ranked source: `curvytron-latest212-smoke-20260513`
- source snapshot: `latest212-smoke-20260513`
- durable source ref:
  `tournaments/curvytron/leaderboards/curvytron-latest212-smoke-20260513/snapshots/latest212-smoke-20260513.json`
- verification: it has 212 active checkpoint rows; newer visible one-frame public leaderboards had fewer than 100 active rows, so they were not enough for the top-100 seed.

## Candidate Refs

- first-batch candidate count: 100
- uniqueness: 100 unique checkpoint refs
- missing refs on training Volume during seed validation: 0
- local candidate list during this lane:
  `/private/tmp/curvy_top100_refs_lines.txt`
- local comma-separated CLI candidate list:
  `/private/tmp/curvy_top100_refs.txt`
- durable intake manifest:
  `tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a/intake/elo-oneframe-top100-plus-latest-20260514a/config.json`

## Commands Used

Historical note: these commands are a record of what happened, not a safe
template for background continuation. Any future command that spawns child
tournament workers and should keep running after the command returns must add
`modal run --detach` or wait for child completion.

Verify historical ranked sources:

```text
modal volume ls curvyzero-curvytron-tournaments /tournaments/curvytron/leaderboards
modal volume get curvyzero-curvytron-tournaments /tournaments/curvytron/leaderboards/curvytron-latest212-smoke-20260513/latest.json /private/tmp/curvy-latest212-leaderboard.json --force
```

Seed validation, no rating spawn:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode intake-seed \
  --tournament-id arena-oneframe-top100-plus-latest-20260514a \
  --rating-run-id elo-oneframe-top100-plus-latest-20260514a \
  --checkpoint-refs "$(cat /private/tmp/curvy_top100_refs.txt)" \
  --round-count 1 \
  --continue-from-latest \
  --pair-selection adaptive_v0 \
  --pairs-per-round 100 \
  --games-per-pair 3 \
  --games-per-shard 1 \
  --max-steps 8000 \
  --num-simulations 8 \
  --decision-source-frames 1 \
  --decision-ms 16.666666666666668 \
  --source-physics-step-ms 16.666666666666668 \
  --policy-mode eval \
  --active-pool-limit 100
```

First-batch rating spawn:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode intake-seed \
  --tournament-id arena-oneframe-top100-plus-latest-20260514a \
  --rating-run-id elo-oneframe-top100-plus-latest-20260514a \
  --checkpoint-refs "$(cat /private/tmp/curvy_top100_refs.txt)" \
  --round-count 1 \
  --continue-from-latest \
  --pair-selection adaptive_v0 \
  --pairs-per-round 100 \
  --games-per-pair 3 \
  --games-per-shard 1 \
  --max-steps 8000 \
  --num-simulations 8 \
  --decision-source-frames 1 \
  --decision-ms 16.666666666666668 \
  --source-physics-step-ms 16.666666666666668 \
  --policy-mode eval \
  --active-pool-limit 100 \
  --intake-spawn-rating
```

## Current Evidence

- The seed validation wrote a 100-checkpoint manifest with `missing_count=0`.
- The rating spawn returned `rating_call_id=fc-01KRJFVHQQAPQ9YSNS5DM7RQJN`.
- The tournament Volume has rating artifacts started:
  - `ratings/elo-oneframe-top100-plus-latest-20260514a/config.json`
  - `ratings/elo-oneframe-top100-plus-latest-20260514a/progress.json`
  - `ratings/elo-oneframe-top100-plus-latest-20260514a/provisional_latest.json`
  - `ratings/elo-oneframe-top100-plus-latest-20260514a/rounds`
- Targeted Volume reads after round 0 showed:
  - config checkpoints: 100
  - `decision_source_frames=1`
  - `decision_ms=16.666666666666668`
  - `source_physics_step_ms=16.666666666666668`
  - round 0: 100 pairs, 300 games, 300 valid games, 0 failures, 100 rated pairs
  - latest ratings: 100 rows, all provisional

## Second Batch Update

Second-batch candidate source:

- run prefix: `curvy-survive-bonus`
- selection: latest checkpoint by Volume mtime
- command output:
  `/private/tmp/recent_curvy_survive_bonus_latest100_discover.out`
- candidate list:
  `/private/tmp/recent_curvy_survive_bonus_latest100_refs_lines.txt`
- deduped candidate list:
  `/private/tmp/recent_curvy_survive_bonus_latest100_dedup_refs_lines.txt`
- discovery result: 100 found, 0 missing, 100 unique
- overlap with top-100 seed refs: 0

Second-wave submit:

- accepted 100 refs
- enqueued 100 events
- output:
  `/private/tmp/top100_second_wave_submit.out`

First drain after submit:

- did not spawn a continuation
- reason: `spawn_skipped_reason=rating_run_claim_exists`
- it reported `rating_checkpoint_count=200`
- output:
  `/private/tmp/top100_second_wave_drain.out`

Operator repair drain:

- used `--round-count 2`, `--intake-allow-rating-overrides`, and
  `--intake-claim-stale-after-seconds 0`
- repaired stale/fresh claim for the 200-ref pool
- `event_count=100`
- `rating_checkpoint_count=200`
- spawned continuation call id: `fc-01KRJGB2BXTY7D5AD6PDW9RZAB`
- output:
  `/private/tmp/top100_second_wave_drain_round2.out`

Important caveat:

- The durable intake manifest has `checkpoint_refs` / `checkpoint_count` at 100,
  while `seen_checkpoint_refs` and `queued_checkpoint_refs` grew to 200. Treat
  this as a truncated/base-manifest display issue and use `seen_checkpoint_refs`
  plus rating config for continuation pool size.
- Immediately after the repair spawn, `rounds/` still listed only
  `round-000000`; round-1 creation needs a follow-up Volume check.
- Later evidence showed this was a launch-lifetime problem, not proof that the
  workers were healthy. Round input/progress can be written before child game
  workers finish; verify `latest.json` advances and game summaries exist.

## Risks / Watch Items

- A broad Modal progress poll became noisy because older active manifests also emitted drain/provisional logs.
- One background intake drain tick showed `ValueError: gif_sample_strategy must be one of ('first_n', 'evenly_spaced')`; this appears tied to an active manifest/default path, not yet proven to block the top-100 rating.
- Another background tick showed `modal.exception.ConflictError: The app is stopped or disabled` after the seed/spawn local app stopped.
- Logs for the failed continuation shape showed `RemoteError`,
  `KeyboardInterrupt`, and `Runner terminated`; the Volume had empty game dirs
  and no completed summaries. Treat this as failed child work until durable
  summaries prove otherwise.
- Before second-batch injection, verify the top-100 rating config/progress/latest directly from the durable Volume and confirm the 100-player pool, one-frame settings, game starts/completions, and no top-100-specific failure.

## Recommendation

Pause further second-wave work until the 200-ref continuation is inspected.
For the next clean operator move, prefer a bigger base continuation round first
over more recent-checkpoint injection: the base top-100 rows are still
provisional after only 3 games / about 3 opponents each.

## Top-100 Gate For Tonight18

Current intended gate:

- `tournament_id=curvy-oneframe-top100-gate-20260514a`
- `rating_run_id=elo-oneframe-top100-gate-20260514a`
- purpose: produce active top rows for the 18 H100 training manifest
- shape: 100 checkpoints, one-frame source decisions, 1000 scheduled pairs,
  21 games per pair

Progress check at 2026-05-14 03:49 ET:

- `latest.json` did not exist yet.
- status was `running`.
- 621 of 1000 pairs had started.
- lightweight progress estimated 13,041 of 21,000 games had been seen.
- no final ratings should be consumed until `latest.json` exists and rows are
  active, not provisional.

Debug update, 2026-05-14:

- Sampled battle directories contain real `games/game-*/summary.json` files and
  GIFs.
- With `games_per_shard=1`, this run writes per-game summaries, not
  `shards/*/summary.json`.
- The old cheap progress path could show `completed_game_count=0` because it
  only counted shard-summary files. That was an observability bug.
- The current tree adds an explicit `--progress-count-game-summaries`
  diagnostic switch that counts per-game summary files without loading their
  payloads.
- Do not enable that as a routine top100/web refresh. A live top100 diagnostic
  showed broad file counting can still be slow enough to become its own problem.
- The reducer path itself scans `battles/*/games/*/summary.json`, so the reduce
  logic is looking in the right place.
- `latest.json` is absent because the rating parent has not reduced the full
  1000-pair / 21,000-game map yet. Do not run `--allow-partial-reduce` for the
  top100 launch gate.

Operator conclusion:

- Top100 is too large and opaque as the immediate launch gate.
- Keep it running/debugging if capacity permits, but use the top10 fallback as
  the smaller honest loop for near-term trainer validation.
- Future tournament gates should either use fewer pairs, shard summaries with a
  progress path that matches the output mode, explicit bounded diagnostics, or
  an incremental reduce design. Do not infer failure from cheap progress alone.
