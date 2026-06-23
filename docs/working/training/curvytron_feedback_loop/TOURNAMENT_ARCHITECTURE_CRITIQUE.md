# Tournament Architecture Critique

Last updated: 2026-05-16.

## Plain Problem

The tournament should feel simple:

```text
choose pairs
-> run games in parallel
-> reduce results
-> update ratings
-> publish leaderboard
```

If games are independent, the slow part should be roughly one game plus Modal
autoscale overhead. When it feels unstable, it usually means the system is
mixing concerns or paying hidden costs around the game workers.

## Current Shape

Current game execution already uses Modal fanout:

- `curvytron_tournament_game` runs one game.
- `curvytron_tournament_game_shard` runs one shard of games and can reuse
  loaded policies.
- `_run_game_work_map` uses `.map(..., order_outputs=False)`.

That means the basic parallel primitive exists.

Recent cleanup:

- removed arbitrary `max_containers` caps from game worker functions;
- changed the default rating/tournament path back to true game-level fanout by
  setting `DEFAULT_GAMES_PER_SHARD=1`;
- added explicit aggressive warm-pool settings for the game worker functions:
  `min_containers`, `buffer_containers`, and `scaledown_window`.

What is still messy:

- every game/shard worker reloads checkpoint Volume;
- every game/shard worker commits tournament Volume;
- scheduling and reduction live in large Modal orchestration files;
- the preflight readout does not yet clearly say expected workers, waves,
  games, warm-pool state, or bottleneck.

## What Warm Containers Fix

Warm containers fix cold-start and queueing latency.

For bursty tournaments, the hot function should probably have a warm pool during
the rating window:

- `min_containers`: keep some workers ready even before work arrives.
- `buffer_containers`: keep extra workers ready while work is active.

This applies mainly to `curvytron_tournament_game` for speed-of-one-game
execution. `curvytron_tournament_game_shard` remains useful when we explicitly
choose fewer Modal calls and fewer commits, but it serializes games inside the
worker and is therefore not the default speed path.

## What Warm Containers Do Not Fix

Warm pools do not fix:

- scheduler waves that are too small;
- top-100 false drops;
- too many Volume commits;
- slow Volume reloads;
- writing too many tiny JSON/GIF artifacts;
- reducing results serially in one large parent;
- unclear state when a run was detached or killed;
- a website reading too much raw Volume state.

So yes, warm workers are likely necessary for smooth scale. They are not the
whole architecture.

## Cleaner Architecture

Keep four layers separate.

1. Intake
   - finds checkpoints;
   - records immutable refs;
   - does not decide pair details.

2. Scheduler
   - reads current rating state;
   - writes a bounded wave plan;
   - reports expected pairs, games, workers, waves, and coverage;
   - does not run games.

3. Game Workers
   - one shard is one independent Modal input;
   - warm pool can be high during active rating;
   - no scheduler decisions;
   - write compact shard summaries.

4. Reducer / Publisher
   - reads completed shard summaries;
   - updates pair history and ratings;
   - writes latest/provisional/public snapshots;
   - publishes trainer-facing leaderboard only through evidence gates.

## Warm Pool Recommendation

Do not cap game workers unless there is a specific reason. An arbitrary
`max_containers` on the game lane is an invisible throughput ceiling.

The desired speed path is one game per worker. The protection against chaos is
explicit preflight, no-GIF scale defaults, warm pools, and better Volume
write/read contracts, not a hidden cap or silent serial shard.

Do keep caps on singleton/control functions where duplicated state writes would
be dangerous, such as scheduled pollers, drains, reducers, and website refresh
jobs.

Better game-worker default:

- no fixed `max_containers` on game workers;
- aggressive `min_containers` so the lane does not start from zero;
- aggressive `buffer_containers` so bursts do not wait on cold starts;
- a long enough `scaledown_window` to keep workers warm across rating waves;
- shard-first defaults so scale goes through `curvytron_tournament_game_shard`.

Concrete candidate:

```text
curvytron_tournament_game:
  max_containers: unset
  min_containers: 100
  buffer_containers: 400
  scaledown_window: 20 minutes

curvytron_tournament_game_shard:
  max_containers: unset
  min_containers: 500
  buffer_containers: 500
  scaledown_window: 20 minutes
```

The exact values should be measured and can be raised dynamically before known
large tournaments. The point is to avoid cold-starting the whole tournament
every wave.

Modal also supports `Function.update_autoscaler()`, so a controller or operator
can raise shard `min_containers` / `buffer_containers` before a known burst and
lower them afterward without changing scheduler behavior.

## Immediate Critique

The architecture has been unstable because we have been debugging three
different problems as if they were one:

- scheduling quality: which games should exist;
- execution throughput: how fast Modal runs those games;
- artifact visibility: how the website and reducers read written results.

These need separate readouts. Otherwise every symptom looks like "the
tournament is broken."

## Bottleneck Map

The main bottlenecks are now clearer.

1. Too many Modal calls
   - Old default: `games_per_shard=1`.
   - That meant 21 Modal calls for a 21-game battle.
   - We briefly changed this to `21`, but measured it against the product goal.
   - Current default is back to `games_per_shard=1` so every game can fan out.

2. Too many Volume commits
   - Game-level fanout makes every game worker commit.
   - This is now the main engineering cost to fix.
   - Do not hide it by serializing games inside shards unless we explicitly pick
     a cost-saving mode.
   - Further cleanup should make commit/reload failures visible and move more
     durable writing toward reducer/index summaries where possible.

3. Cold starts and queueing
   - Game workers now have no hidden `max_containers` cap.
   - Warm pools are aggressive for the shard lane.
   - This should reduce first-wave latency.

4. Blocking parent reduction
   - Rating rounds still call `.map(...)` and wait for all shard workers before
     reducing.
   - One slow worker can delay the whole round.
   - That is acceptable for batch Elo, but the progress path must make this
     visible.

5. Broad live scans
   - Some progress/provisional paths scan historical shard summaries.
   - That gets worse as the tournament grows.
   - These should move to round-local manifests or direct expected shard refs.

6. Large state payloads
   - Some parent functions pass full pair history and scheduler state between
     rounds.
   - Better pattern: pass refs, load from Volume once, write compact summaries.

## What This Means For All-Pairs

All-pairs among the top 100 is large but plausible with game-level fanout:

```text
100 checkpoints -> 4,950 pairs
21 games per pair -> 103,950 games
games_per_shard=1 -> 103,950 game calls
```

That is a lot of calls, but it matches the desired shape: all games are
independent and Modal should autoscale the CPU worker lane. If Volume commits
become the limiter, fix commit/reload patterns directly instead of silently
serializing games.

All-pairs across every checkpoint from every run is still a scheduler problem.
All-pairs top-100 can be an audit or refinement mode. The online service still
needs bounded placement for new entrants.

## Next Simple Tests

1. Preflight only:
   - print pair count, game count, shard count, expected worker calls,
     scheduler seconds, and estimated coverage.

2. Worker fanout only:
   - run synthetic/no-op or tiny games through `curvytron_tournament_game_shard`
     with no game-worker `max_containers`;
   - compare cold, moderate warm pool, and aggressive warm pool.
   - first real command should be a detached no-GIF rating probe, not the older
     generic `mode=tournament` path, because rating mode returns compact shard
     tallies to the parent.

3. Volume overhead only:
   - measure checkpoint reload seconds and commit seconds from worker timing;
   - decide whether commits can move from per-shard to reducer-only for some
     paths.

4. Full tiny tournament:
   - use the chosen scheduler;
   - run enough shards to prove autoscaling;
   - verify reducer and leaderboard without website involvement.

5. Top-100 all-pairs smoke:
   - 100 lightweight refs or mock policies if available;
   - `games_per_shard=1`;
   - no GIFs;
   - confirm shard count, timing, commits, reduction, and latest snapshot.

## Latest Local Scale Evidence

Local tests now cover scheduler/rating behavior before Modal:

- `100 established + 10 weak new`: weak entrants get placement games, lose, are
  retired below the top-100 active pool, and are absent from the next schedule.
- `120 clone/draw swarm`: all draw outcomes produce no Elo movement and exactly
  100 public active rows.
- `1000 established + 50 new`: one 1000-pair wave schedules unique pairs,
  touches all new rows, and avoids old rank-101+ rows.

Fast one-wave timing on this machine:

```text
100 established + 20 new, budget 300: 0.516s
100 established + 500 new, budget 300: 2.507s
100 established + 500 new, budget 1000: 7.781s
0 established + 424 new, budget 300: 1.660s
```

This means pure scheduling is not the first bottleneck. Modal throughput, Volume
reload/commit time, broad progress scans, and reducer barriers are the next
things to measure.

## Fanout Probe Result

On 2026-05-16, two no-GIF five-checkpoint probes were run:

- `curvy-scale-probe-5latest-nogif-20260516a`
  - `games_per_shard=21`
  - 10 worker calls, 210 games.
  - Completed cleanly, but games inside each pair were serial-ish.

- `curvy-scale-probe-5latest-gamefanout-20260516a`
  - `games_per_shard=1`
  - 210 worker calls, 210 games.
  - Completed cleanly and matched the speed goal better.

Conclusion: default rating speed should be game-level fanout. Shards remain an
explicit cost/commit-pressure knob, not the default.
