# Checkpoint Tournament Elo Follow-Up, 2026-05-13

## Future Goal

After the V0 battle runner works, build a small rating loop over checkpoints.
The loop chooses pairs, runs enough games per pair in parallel, updates ratings,
and repeats until the ladder is stable enough to guide coach decisions.

The reason this matters: the coach will soon have too many checkpoints to judge
by hand. Raw GIFs help explain behavior, but a rating loop gives a quick map of
which checkpoints are actually improving.

2026-05-13 pivot: the next useful version should handle every useful checkpoint
from every run, not just the latest checkpoint from each run. That makes
adaptive scheduling the main work. All-pairs remains useful as a stress test or
audit, but it cannot be the steady-state system.

Scale grounding: this is not just a 50-run toy. If we ever do all-pairs over
300 checkpoints, unordered no-self all-pairs is 44,850 battles. With 50 games
per battle, that is 2,242,500 games. But if we include every checkpoint from
every run, the pool gets much larger than 300. The system must choose a bounded
set of useful battles, fan out at shard/game scale, and reduce at battle scale.

Modal autoscaling caveat: at this scale, fan-out can outpace warm containers.
Some work may queue, start late, or time out. The rating lane needs retry and
backoff patterns around idempotent shard work, plus cheap progress and resumable
reduce, before we trust very large all-pairs runs.

Current retry rule: low-level game and game-shard workers use Modal retries with
backoff. Rating parent/orchestration functions do not retry as a whole, because
that can duplicate a large fan-out.

## Real Dataset Target

Use the coach's current batch of about 50 CurvyTron runs as the first serious
rating dataset.

First pass:

- identify the 50 runs from the coach docs/manifests
- take the latest checkpoint from each run
- run checkpoint-vs-checkpoint battles
- use 50 to 100 games per battle
- update ratings from the battle summaries

Current concrete source:

- matrix prefix: `survivaldiag-v1b-20260513h`
- row count: `50`
- manifest:
  `artifacts/local/curvytron_survivaldiag_manifests/survivaldiag-v1b-20260513h.rows.jsonl`
- run root on `curvyzero-runs`:
  `training/lightzero-curvytron-visual-survival/<run_id>`
- row IDs are `001` through `050`

Checkpoint path caveat:

- The stable mirror path
  `checkpoints/lightzero_resume_state/iteration_<n>.resume_state.pkl` is a
  resume sidecar, not the model checkpoint used by the tournament runner.
- The real weight checkpoints seen in the v1b rows are under
  `attempts/<attempt_id>/train/lightzero_exp/ckpt/iteration_<n>.pth.tar`.
- For the first 50-run tournament picker, scan each run for the highest
  `iteration_<n>.pth.tar` under its active attempt `lightzero_exp/ckpt`.

Second pass:

- include more than the latest checkpoint, possibly all useful checkpoints
- use the same rating loop to see which training phase/checkpoint is strongest
- keep battle summaries immutable so ratings can be recomputed

Tiny remote rating smoke is green and the website can read rating snapshots.
The latest-checkpoint picker is in place. The full 50-run rating job has now
been launched and completed through that picker.

## Implemented First Version

Files:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- `tests/test_curvytron_checkpoint_tournament.py`

The first rating version is batch Elo:

- one rating round builds pair specs
- rating work should flatten to a game-level map, because the one-game Modal
  function is the high-parallel unit
- all game summaries remain immutable
- the round writes `input.json`, `results.json`, `ratings.json`, and a slim
  `latest.json`
- large sharded rounds should reduce from shard tallies, not from one compact
  game row per game in the parent
- the website reads rating snapshots, it does not compute ratings in the request
  path
- the website should use a battle index or other small listing artifact so
  rating pages stay fast
- `--mode discover` scans current training runs and returns the latest real
  LightZero `iteration_<n>.pth.tar` checkpoint per run
- `--mode rating` can use `--run-id-prefix survivaldiag-v1b-20260513h` instead
  of hand-pasted checkpoint refs
- discovery should fail clearly if expected rows or checkpoints are missing

Local validation:

```text
uv run python -m py_compile src/curvyzero/tournament/curvytron_checkpoint_tournament.py src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py tests/test_curvytron_checkpoint_tournament.py
uv run pytest tests/test_curvytron_checkpoint_tournament.py
```

Result:

```text
25 passed, 1 skipped
```

Remote smoke:

```text
arena-rating-smoke-v1b-20260513a / elo-smoke
```

Result:

- two real v1b checkpoints loaded
- one game completed
- battle summary written
- `ratings/elo-smoke/latest.json` written
- website APIs returned rating rows and battle rows
- sample GIF was `704 x 704`

Prefix-discovery smoke:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode discover \
  --run-id-prefix survivaldiag-v1b-20260513h \
  --max-runs 3
```

Result: found 3 of 3 latest real checkpoints, with no missing rows.

Integrated prefix-rating smoke:

```text
arena-rating-prefix-smoke-20260513a / elo-smoke
```

Result:

- `--mode rating` used `--run-id-prefix survivaldiag-v1b-20260513h`
- discovery found 2 of 2 checkpoints
- one game completed
- rating snapshot landed
- website rating APIs returned the snapshot
- sample GIF was `704 x 704`

50-checkpoint probe:

```text
arena-rating-v1b-50probe-20260513a / elo-probe
```

Result:

- discovery found all 50 expected checkpoints
- `pairs_per_round=5`, `games_per_pair=2`
- 10 games ran through the global one-game map
- 5 battles were indexed
- website API showed 50 rating rows and 5 indexed battles
- GIFs were off

Full detached run started:

```text
arena-rating-v1b-full50-gpp50-20260513a / elo-full-gpp50
function_call_id: fc-01KRGDJS0T10E2PZV9BJ8K1ETF
```

Plan:

- 50 checkpoints
- 1,225 unordered pairs
- 50 games per pair
- 61,250 independent game calls
- GIFs off
- expected artifact count around 62,479 files

Full detached run result:

- parent app stopped after finishing the rating snapshot
- `latest.json` exists
- `ratings`: 50 rows
- `pair_count`: 1,225
- `rated_pair_count`: 1,225
- `game_count`: 61,250
- `battle_index.json`: 1,225 battle rows
- website `/api/rating-standings` returns 50 rows
- website `/api/battles` uses `battle_index` and returns total 1,225
- `latest.json` is slim: it does not include `pair_rating_results`
- current top row at first check:
  `ckpt-032-...`, rating `1764.0`, games `2450`, wins `1601`,
  losses `776`, draws `73`

Important scale observation:

- The one-game fan-out reached the full 61,250-game round.
- The first exact progress scanner was too slow because it tried to read every
  game summary.
- Default `--mode progress` now does a cheap battle-directory pass and writes
  `ratings/<rating_run_id>/progress.json`.
- If `latest.json` exists for the same round, progress is marked complete.
- Exact summary reads are still available with `--progress-read-summaries`, and
  recovery/recompute is available with `--mode reduce`.
- This is good enough for 50 checkpoints. For 200-300 checkpoints, the main
  expected costs are policy reload per game and one Volume commit per game.
- Sharded game workers have now been added locally to reduce that cost. With
  `--games-per-shard N`, one worker runs N games for one pair, reuses the two
  loaded policies, and commits once.
- Reuse is eval-mode only. Collect-mode shards deliberately reload policies per
  game so stochastic policy seeds do not silently change.
- No-GIF rating games now skip raw RGB rendering and keep `frame_count: 0`.

## Current Recommendation

Use batch Elo first, not game-by-game Elo. Keep the deployment as one Modal app.
Inside it, use one global game map for small runs and one global shard map for
larger runs. Rating rounds still collect game summaries into pair and rating
results.

The runtime target is roughly the speed of one shard. That is not exact because
Modal has cold starts, autoscaling, checkpoint loading, Volume commits, and
final aggregation. But the code should fan out all independent shards in one
global map so it does not add avoidable serial pair waits.

1. Snapshot ratings at the start of a rating round.
2. Choose pairs from that snapshot.
3. Run games in parallel.
4. Apply one batch update after battle summaries are complete.

This avoids async order bias when many Modal jobs finish in random order.

## Simple Formula

Expected score:

`E_A = 1 / (1 + 10^((R_B - R_A) / 400))`

Observed score:

`S_A = (wins_A + 0.5 * draws) / valid_games`

Update:

`R_A += K_pair * (S_A - E_A)`

Default:

- initial rating: `1500`
- `K_pair = 32 * sqrt(valid_games / 50)`
- clamp `K_pair` to `[16, 64]`
- clamp one pair delta to `+/-80`

## Pairing Policy

Use a mix:

- 60% near-rating pairs
- 25% uncertain/new checkpoint probes
- 15% anchor repeat pairs

For a new checkpoint, test against:

- best current checkpoint
- median checkpoint
- weaker checkpoint
- nearest neighbor after an early estimate

## Draws And Failures

- Draw is `0.5`.
- Timeout with no death is draw for rating v0.
- Infrastructure failure is invalid and excluded.
- Checkpoint/policy failure should be a loss for that checkpoint if clearly
  caused by the checkpoint runner, otherwise invalid.
- Do not rate a pair unless enough games are valid, e.g. at least 80%.

## Trust Rule

Battle results are immutable. Rating snapshots are disposable.

That means we should always be able to recompute ratings from stored battle
results.

## Validation Plan

- Estimate cost, file count, and runtime for the 50-run run before launching it.
  The estimate must respect `pairs_per_round` for probes.
- Run a remote smoke with rating GIFs off by default and a small explicit GIF
  sample enabled.
- Confirm missing expected checkpoints produce a clear failure.
- Confirm `latest.json` is small and website API reads do not scan all game
  artifacts.
- Build fake deterministic-strength bots or a fake battle simulator.
- Verify recovered ranking matches hidden strength.
- Verify random completion order gives the same batch rating snapshot.
- Inject invalid games and confirm they do not pollute ratings.
- Add a strong new checkpoint and check the pairing policy finds it quickly.

## V1 Data Model Sketch

Keep this derived and simple:

- `ratings/<rating_run_id>/config.json`
- `ratings/<rating_run_id>/rounds/round-000000/input.json`
- `ratings/<rating_run_id>/rounds/round-000000/results.json`
- `ratings/<rating_run_id>/rounds/round-000000/ratings.json`
- `ratings/<rating_run_id>/latest.json` as a slim pointer and standings summary
- `ratings/<rating_run_id>/progress.json` as a small progress/status artifact
- `ratings/<rating_run_id>/rounds/round-000000/progress.json` for the same
  progress payload at round scope

Each round points to immutable tournament battle refs. Ratings can be rebuilt if
the formula changes.

## Runbook

Discover latest checkpoints:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode discover --run-id-prefix survivaldiag-v1b-20260513h --max-runs 50
```

Estimate:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode estimate --run-id-prefix survivaldiag-v1b-20260513h --max-runs 50 --expected-checkpoint-count 50 --games-per-pair 50 --games-per-shard 10
```

Launch detached rating:

```text
uv run --extra modal python -B -m modal run --detach -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode rating --tournament-id <tournament-id> --rating-run-id <rating-run-id> --run-id-prefix survivaldiag-v1b-20260513h --max-runs 50 --expected-checkpoint-count 50 --round-count 1 --games-per-pair 50 --games-per-shard 10
```

Refresh cheap progress:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode progress --tournament-id <tournament-id> --rating-run-id <rating-run-id>
```

Refresh exact progress, slower:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode progress --tournament-id <tournament-id> --rating-run-id <rating-run-id> --progress-read-summaries
```

Recover ratings from summaries if needed:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode reduce --tournament-id <tournament-id> --rating-run-id <rating-run-id> --wait
```

## V1 Stop Rules

Start with boring rules:

- stop after a fixed round count, or
- stop when top-10 order changes little for several rounds, or
- stop when all checkpoints have at least a minimum number of valid games.

Do not overfit the stop rule before the raw battle runner is proven.

## V1 Website Ideas

- standings table with rating, games, uncertainty proxy, latest checkpoint time
- battle drill-down for a selected checkpoint
- "show me surprising results" list
- sample GIF per battle when explicitly requested, not every GIF at once
- battle index reads for fast listing

## Not For V0

- Full Glicko or TrueSkill.
- Global Bradley-Terry fitting.
- Fancy website ranking math.
- Full 50-run launch before estimate and smoke.

Those are useful later, but the first useful thing is reliable pair battle data.

## Shard Follow-Up

What changed locally:

- Pair specs carry `games_per_shard` and `reuse_policies_per_shard`.
- Rating specs carry the same fields.
- `build_game_shard_specs_for_pair(...)` chunks the existing game specs without
  changing game ids, seeds, scoring, or artifact paths.
- The Modal app maps either one game per worker or one shard per worker.
- The shard worker reuses the two policy objects for every game in that shard
  and commits the tournament Volume once after the shard.
- Shard outputs are deduped by battle/game id before pair summaries are built.
- The CLI estimate reports one game for `--mode game`; pair/tournament/rating
  estimates report the requested `games_per_pair`.
- `games_per_shard=0` is rejected consistently.

Validation so far:

- Python compile passed for the helper, Modal app, and focused tests.
- Focused local tests passed: `25 passed, 1 skipped`.
- Deployed Modal app passed a shard rating smoke:
  `arena-rating-shard-smoke-20260513b / elo-shard-smoke`.
- Smoke used 2 checkpoints, 1 pair, 4 games, `games_per_shard=2`, and produced
  two shard calls.
- Shard workers reported `policy_reuse: true`.
- Game summaries showed `policy_loads[*].preloaded: true`.
- No-GIF game summaries showed `frame_count: 0`.
- Website `/api/rating-progress` and `/api/battles` served the smoke result.
- `--mode reduce --wait` rebuilt the same tiny rating from committed summaries.
- Deployed Modal app also completed a full 50-checkpoint sharded rating run:
  `arena-rating-v1b-full50-gpp50-shard10-20260513a / elo-full-gpp50-shard10`.
- Full sharded run found 50 checkpoints, rated 1,225 pairs, ran 61,250 games,
  and used 6,125 shard calls by plan.
- Progress was refreshed after the final UI fix and now reports
  `completed_game_count=61250`, `completed_pair_count=1225`, and
  `completion_fraction=1.0` without exact summary scanning.
- Top row at first API check was
  `ckpt-032-train-lightzero_exp_260513_075512-ckpt-ite-ba98e3ff`, rating
  `1761.12`, with `2450` games.

Recommended first scale settings:

- 50 checkpoints: `--games-per-pair 50 --games-per-shard 10`
- 200 checkpoints: start with `--games-per-pair 50 --games-per-shard 10`
- 300 checkpoints: estimate first; start with `--games-per-shard 10`, then try
  `25` only after a medium smoke is healthy

Scale estimates:

- 200 checkpoints means 19,900 unordered no-self pairs.
- At 50 games per pair, that is 995,000 games.
- With `--games-per-shard 10`, that is about 99,500 worker calls.
- 300 checkpoints means 44,850 pairs and 2,242,500 games.
- With `--games-per-shard 10`, that is about 224,250 worker calls.

Open critique:

- Sharding cuts calls, policy loads, and commits, but it also means one slow
  shard delays N game results. Keep N small at first.
- Sharding does not by itself reduce the number of immutable game summary files.
  Full 200-300 checkpoint round robins still write many files unless a later
  explicit lean-artifact mode changes that.
- Rating progress still estimates cheaply from battle directories unless exact
  summary scanning is requested.
- The rating parent now has the lighter reducer path: shard workers can return
  tallies instead of compact game lists, and the parent writes lean battle
  summaries without `games`.
- For large all-pairs ratings, use the lean tally reducer. Parent aggregation
  should scale with battle count, not game count.

Lean reducer validation:

- local compile passed for helper, Modal app, and tests
- focused local tests passed: `30 passed, 1 skipped`
- tally-only pair summaries produce the same batch-Elo result as game-list
  summaries for wins, draws, failures, and min-valid checks
- the shard-tally summarizer writes battle summaries with no `games`, preserves
  tally counts, and produces progress from those tallies
- deployed remote smoke passed after one quick bug fix:
  `arena-rating-lean-shard-smoke-20260513b / elo-lean-shard-smoke`
- that smoke used 2 shard workers for 4 games and wrote lean `results.json`
  without `pair_results`
- website review APIs served rankings and checkpoint detail from `latest.json`
  plus `battle_index.json`
