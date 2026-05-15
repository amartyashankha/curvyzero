# Top-100 Tournament Stress Lane

Date: 2026-05-14

## Plain Goal

Start one real one-frame tournament with about 100 checkpoint players. This is a
stress test for tournament intake, scheduling, game fanout, rating continuation,
and leaderboard publication.

This is not a trainer-refresh proof. A running trainer still needs the separate
refresh lane before promoted checkpoints can change future collection inside the
same attempt.

It is also not a launch-lifetime proof unless spawned child game/rating workers
survive after the launch command returns. For fire-and-return work, use
`modal run --detach` or a deployed path that keeps the work alive; otherwise the
parent must wait for the child work to finish.

## Two-Step Plan

| Step | What to do | Success condition |
| --- | --- | --- |
| 1. Base tournament | Take the top 100 active checkpoint refs from `curvytron-latest212-smoke-20260513` and rerate them under current one-frame settings. | Tournament sees 100 checkpoint players, starts games, writes rating progress/latest, and shows no obvious worker failure. |
| 2. New checkpoint injection | After step 1 is healthy, add the latest checkpoint from about 100 current trainer runs. | Intake admits new refs, schedules placement games against the active pool, and continues ratings without losing old history. |

## Source Rule

Use this old leaderboard only as seed material:

```text
tournaments/curvytron/leaderboards/curvytron-latest212-smoke-20260513/snapshots/latest212-smoke-20260513.json
```

Do not treat it as current one-frame truth. The new tournament must rerate the
chosen checkpoint refs under current one-frame settings.

## Active Agents

| Agent | Role | Instructions |
| --- | --- | --- |
| Cicero | Operator | Find the seed refs, start the top-100 tournament if safe, and leave exact IDs/artifacts. |
| Sagan | Shadow and critic | Check Cicero's artifacts/live state, avoid duplicate launches, and take over only if no top-100 tournament is actually running. |

## Naming

Prefer simple IDs if unused:

- tournament: `arena-oneframe-top100-20260514a`
- rating: `elo-oneframe-top100-20260514a`

If those are already used, suffix with `b`, `c`, etc. Record the exact IDs here
before moving to step 2.

## Current Status

- Cicero already launched an equivalent base top-100 tournament job. Do not
  launch `arena-oneframe-top100-20260514a` as a duplicate.
- Live IDs:
  - tournament: `arena-oneframe-top100-plus-latest-20260514a`
  - rating: `elo-oneframe-top100-plus-latest-20260514a`
- Source leaderboard:
  `curvytron-latest212-smoke-20260513`, snapshot
  `latest212-smoke-20260513`.
- Candidate count: 100 checkpoint refs.
- Verification: the rating config's checkpoint-ref set exactly matches the top
  100 active refs from the source leaderboard. Order differs after rating
  normalization, but no source refs are missing and no extra refs were added.
- Despite the `plus-latest` name, no second-wave recent-training checkpoints
  are visible in this run. The observed pool is the base top-100 source set.

## Launch Evidence

The tournament Volume has:

```text
tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a/tournament.json
tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a/ratings/elo-oneframe-top100-plus-latest-20260514a/config.json
tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a/ratings/elo-oneframe-top100-plus-latest-20260514a/latest.json
tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a/ratings/elo-oneframe-top100-plus-latest-20260514a/progress.json
tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a/ratings/elo-oneframe-top100-plus-latest-20260514a/results.json
```

Observed settings and results:

```text
checkpoint_count=100
pair_selection=adaptive_v0
pairs_per_round=100
pair_count=100
games_per_pair=3
game_count=300
rated_pair_count=100
decision_source_frames=1
decision_ms=16.666666666666668
source_physics_step_ms=16.666666666666668
max_steps=8000
num_simulations=8
round_count_completed=1
rating status=complete
queue_len=0
```

Battle index check:

```text
battle rows=100
ok rows=100
failed rows=0
completed games=300
failure_count=0
```

Rating row status:

```text
active=0
provisional=100
min_games_per_checkpoint=3
max_games_per_checkpoint=9
min_distinct_opponents=1
max_distinct_opponents=3
```

Interpretation:

- Healthy for the first stress-test question: scheduling, game fanout, worker
  completion, and rating output all happened for 100 checkpoint players under
  one-frame settings.
- Not healthy for public leaderboard publication or training assignment use:
  all rows are still provisional because this pass only gave each checkpoint a
  small amount of evidence.
- Do not inject the second-wave recent checkpoints until the main thread
  accepts that this base proof is enough or requests another base round.

## Commands Used For Verification

```text
uv run --extra modal modal volume ls --json curvyzero-curvytron-tournaments tournaments/curvytron
uv run --extra modal modal volume ls --json curvyzero-curvytron-tournaments tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a
uv run --extra modal modal volume ls --json curvyzero-curvytron-tournaments tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a/ratings
uv run --extra modal modal volume get --force curvyzero-curvytron-tournaments tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a/ratings/elo-oneframe-top100-plus-latest-20260514a/latest.json /private/tmp/curvy_top100_20260514/rating_latest.json
uv run --extra modal modal volume get --force curvyzero-curvytron-tournaments tournaments/curvytron/leaderboards/curvytron-latest212-smoke-20260513/snapshots/latest212-smoke-20260513.json /private/tmp/curvy_top100_20260514/latest212_snapshot.json
uv run --extra modal modal queue len curvyzero-curvytron-checkpoint-events-v0 --partition q:arena-oneframe-top100-pl:elo-oneframe-top:50e03cbecc
uv run --extra modal modal dict get curvyzero-curvytron-checkpoint-intake-v0 manifest:arena-oneframe-top100-plus-latest-20260514a:elo-oneframe-top100-plus-latest-20260514a
```

## Risks And Bugs

- The Volume intake manifest
  `tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a/intake/elo-oneframe-top100-plus-latest-20260514a/config.json`
  downloaded as invalid/truncated JSON. The Modal Dict manifest for the same
  `(tournament_id, rating_run_id)` is readable and current, and the rating
  config/latest artifacts are valid. Smallest fix recommendation: make intake
  manifest Volume writes atomic, for example write a temporary JSON file and
  rename/move it into place after the write is complete, then avoid concurrent
  status/tick writers touching the same config file.
- Non-detached `modal run` can kill background child game/rating workers when
  the local entrypoint/app stops. We saw `RemoteError`, `KeyboardInterrupt`, and
  `Runner terminated`, with empty game dirs and no completed summaries. Future
  continuation attempts must use `modal run --detach`, a deployed function that
  keeps work alive correctly, or wait for child completion.
- This run is a throughput and one-frame rerating stress proof, not policy
  strength evidence.
- This run does not prove that running trainers refresh their slots.
- Slots still mean trainer assignment entries, not tournament players.
