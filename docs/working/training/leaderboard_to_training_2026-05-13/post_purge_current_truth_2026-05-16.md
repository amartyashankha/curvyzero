# Post-Purge Current Truth: 2026-05-16

This note records the operator truth after the CurvyZero volume purge. It
supersedes earlier launch notes in this folder that refer to already-deleted
checkpoint refs or pre-purge Modal objects.

## What Changed

- 2026-05-16 00:54 EDT update: CurvyZero v2 apps were stopped again, the
  CurvyZero v2 storage lane was deleted again, and only empty v2 storage was
  recreated. Any training/tournament work launched before this timestamp is now
  invalid as live state because its backing volumes/dicts/queue were wiped.
- CurvyZero v2 apps were stopped before the purge.
- Old/non-v2 CurvyZero Modal Volumes, Dicts, and Queues were purged.
- v2 CurvyZero Modal Volumes, Dicts, and Queues were also purged.
- The active v2 Volumes were recreated as Modal VolumeFS v2 objects.
- CurvyZero v2 apps were redeployed after the 00:54 purge and are now the only
  CurvyZero apps that should be used for this lane:
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`,
  `curvyzero-checkpoint-tournament-v2`, and
  `curvyzero-curvytron-gif-browser-v2`.
- As of the latest verification, the only CurvyZero Volumes are:
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`.
- As of the latest verification, the only CurvyZero Dicts are:
  `curvyzero-curvytron-checkpoint-intake-v2` and
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- As of the latest verification, the only CurvyZero Queue is:
  `curvyzero-curvytron-checkpoint-events-v2`.

## Stale Launch State

- The old checkpoint-ref bootstrap manifest is stale.
- The stale manifest cannot be used as the next full real-lane launch input,
  because the checkpoint refs it depended on were deleted during the purge.
- Any pre-purge manifest, assignment, control pointer, live-watch state,
  tournament state, Dict pointer, or Queue event should be treated as historical
  unless it is regenerated against the recreated v2 Volumes.
- Do not copy launch conclusions from old/non-v2 storage or from pre-purge v2
  storage into the current lane without a fresh existence audit on the recreated
  v2 Volumes.

## Current Plan

04:03 EDT reality check: the post-purge large 18-run lane is not clean proof.
It started correctly and produced useful stress data, but the tournament
artifact tree is now polluted by overlapping rounds from older deploys. The
current `latest.json` for `curvy-r18fresh-live-20260516a` /
`elo-r18fresh-live-20260516a` is `round-000008` with `98` checkpoints,
`99,813` games, `58` failed games, `stable=false`, and
`max_abs_delta=318.85214603560865`. Root progress still points at an old
running `round-000010`, and `round-000011`/`round-000012` also exist. The next
honest proof should use a fresh tournament/rating id with current deployed code,
or explicitly purge/repair this dirty lane first.

Latest correction, 2026-05-16: the active tournament is not capped at
51 forever and all-pairs is not disabled. The visible uneven game counts are a
round-age effect: old checkpoints have participated in more completed rounds;
newer checkpoints have fewer historical games. The real bug is that one
continuation round created input/progress artifacts but did not write final
ratings/latest, so durable standings lagged behind intake. The tournament code
now has local-tested recovery for this: resume completed rounds, reduce
existing round inputs from game/shard summaries, let intake spawn the reducer
automatically, and stop progress from counting empty battle directories as
completed games.

1. Treat all previous remote CurvyZero runs/tournaments as dead historical
   state.
2. Redeploy the CurvyZero v2 apps against the fresh empty v2 storage lane.
   Done at about 00:56 EDT.
3. Build a fresh scratch bootstrap `real18` manifest, using inputs that are
   valid in the recreated v2 storage namespace. Done:
   `artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json`.
4. Launch the full real lane first; use any toy lane only as a separate
   validation path while the real lane is running. Done: submitter wrote `3`
   control assignments, `3` refresh pointers, and spawned `18` trainers plus
   `18` pollers.
5. Attach the tournament watcher by run id. Done with detached app
   `ap-AgqvazcT1csEtAqFbqS5Lg`, tournament
   `curvy-r18fresh-live-20260516a`, rating
   `elo-r18fresh-live-20260516a`. First seed found `0` checkpoints and missed
   all `18` run ids because no checkpoint files existed yet; that was expected
   startup timing and is now resolved. The scan manifest is active and run-id
   based.
6. Early live proof after launch: all `18` trainers wrote `iteration_0`
   checkpoints. Intake status sees `18/18` refs, queue length `0`, and the
   checkpoint sidecars show the current policy observation surface
   `browser_lines + simple_symbols`, `cpu_oracle`, one-frame timing, and
   `random_per_episode` learner seats. Rating `round-000000` completed for the
   first `2` checkpoints (`1` pair, `21` games, `0` failed games,
   `ratings_written=true`). Rating `round-000001` is running from the
   `18`-checkpoint pool (`136` pairs, `2856` games planned).
7. Follow-up checks after the first sleep: trainer progress is real past
   startup. At about 01:11 EDT, run status showed multiple rows at
   `iteration_10000` with `2` checkpoints. At about 01:12 EDT, intake status
   saw `24` refs, including `iteration_10000` refs, and queue length `0`.
   `round-000003` was still running from the earlier `18`-checkpoint pool
   (`153` all-pairs, `3213` games); the next proof is that the following
   continuation uses the newer `24+` ref pool.
8. Continuation proof: logs show `round-000003` completed at about 01:12 EDT
   with `153` rated pairs, `3213` games, and `stable=false`. The service then
   continued automatically. At about 01:18 EDT, `round-000005` was running with
   `27` checkpoints, `351` pairs, `7371` games planned, and game logs include
   `iteration_10000` checkpoint ids. That proves later trainer checkpoints are
   being discovered and scheduled into tournament games.
9. Later truth: the same lane became dirty because old tournament workers
   overlapped. Use the fresh-code fixes plus a clean id or explicit artifact
   purge for the next proof; do not keep treating this dirty tree as production
   validation.

## Operator Reminder

The current lane is post-purge all-v2. Regenerate refs, assignments, pointers,
and tournament inputs from the recreated v2 objects before launching. The
pre-purge checkpoint-ref bootstrap manifest is documentation history only.
The `curvy-r18scratch-*` run launched before 00:54 EDT was also purged and is
not a live batch anymore.

## Current Implementation Truth

- The real 18-run manifest builder has an explicit `--scratch-bootstrap` mode.
- In scratch mode, training rows start from random initialization, not from a
  deleted or mutable checkpoint ref.
- Scratch mode still creates rank-shaped opponent slots. Those slots begin as
  hardcoded placeholders so the run can start with empty v2 storage.
- The training-candidate refresh controller can now replace rank-tagged
  placeholders with real frozen leaderboard checkpoints after the tournament has
  produced an active leaderboard.
- `opponent_immortal` is now independent of the policy kind. Hardcoded policies
  can be mortal or immortal; blank-canvas noop remains explicitly immortal.
- Focused local verification passed:
  manifest/submission/mixture/controller tests: 80 passing.
- GIF browser current truth: the browser defaults to current run ids with
  prefix `curvy-r18fresh-*`; old rows are archived by default; the UI action is
  `Archive`; and the deployed image installs `numpy` to avoid the previous
  import-time crash.
- Tournament GIF current truth: newly generated tournament GIFs default to
  `800` fps with `1ms` minimum frame duration. This changes future GIF artifacts
  only; old files keep their embedded playback timing until regenerated.
