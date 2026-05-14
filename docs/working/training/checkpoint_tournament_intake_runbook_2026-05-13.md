# Checkpoint Tournament Intake Runbook, 2026-05-13

## Current Contract

V0 is a guarded batch launcher, not a true online Elo service yet.

Safe flow:

1. Deploy the tournament app.
2. Seed a manifest from a clear checkpoint cutoff.
3. Enqueue the manifest checkpoints.
4. Drain once into a unique `tournament_id` and `rating_run_id`.
5. Monitor progress and website output.
6. Treat later checkpoints as pending for the next wave.

Do not spawn into an existing rating run unless there is a real continuation
contract. The current rating loop freezes its roster when it starts.

## Durable States

- `discovered`: checkpoint ref appeared in a scan or explicit manifest.
- `queued`: checkpoint ref has a Queue wakeup event and is recorded in the
  manifest `queued_checkpoint_refs`.
- `drained`: a drain consumed wakeup events for a manifest.
- `claimed`: one drain holds the rating-run claim in Modal Dict.
- `scheduled`: the rating loop wrote round input and pair specs.
- `rated`: a final `latest.json` snapshot exists.
- `blocked`: an existing rating run, stale claim, reload failure, or missing
  checkpoint prevents safe launch.

## Important Guards

- Drain now checks existing rating output and claims the rating run before
  consuming Queue events.
- Adaptive V0 now schedules low-coverage checkpoints first, so a newly injected
  checkpoint must get at least one placement battle before normal near-rating
  scheduling spends the budget.
- Checkpoint discovery must scan `train/lightzero_exp*/ckpt/iteration_*.pth.tar`.
  Do not assume `train/lightzero_exp/ckpt` is the whole checkpoint set.
- Human labels must include run identity and iteration; duplicate visible labels
  are disambiguated.

## Latest Evidence

Expanded probe:

`arena-curvytron-top20runs-allckpts-placement-gpp21-gifs5-step8000-20260513a`
/
`elo-top20runs-allckpts-placement-gpp21-gifs5-step8000-20260513a`

Result:

- 424 checkpoints
- 212 placement battles
- 4452 games
- 0 failed games
- 0 zero-game checkpoints
- 5 GIF samples for checked battle
- GIF endpoint served `image/gif`

## Next Design Work

- Add true online continuation: load previous `latest.json`, start the next
  round index, preserve old ratings, and schedule newly discovered checkpoints
  as placement/provisional players.
- Add duplicate-event and queue-loss repair.
- Add stale-claim detection and operator reset.
- Keep score Elo in `policy_mode=eval`. Use collect mode only for separately
  named diagnostic runs.
- Preserve training/env parity metadata in checkpoint payloads so portable
  checkpoints do not silently fall back to tournament defaults.
