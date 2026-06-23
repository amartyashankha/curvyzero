# Cleanup Lane - 2026-05-15

This page is the current cleanup truth. Do not use older chat messages or old
arena names as authority.

## Current Decision

- 2026-05-15 latest override completed: recreated the exact v2 Curvy
  storage/control objects, then redeployed the current apps against them. This
  supersedes the earlier "do not rerun purge" note for the exact v2 objects
  only.
- Current exact v2 objects already recreated and in use:
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`,
  `curvyzero-curvytron-control-v2`,
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-opponent-leaderboard-live-v2`, and
  `curvyzero-curvytron-checkpoint-events-v2`.
- Continue preserving non-v2 historical objects unless the operator explicitly
  asks for a non-v2 purge.
- Old non-v2 Curvy trainer/tournament deployments were stopped after the v2 app
  redeploy so the active app list points at the v2 lane.
- Treat the v2 real18 training/tournament state as invalid evidence for the
  next real launch.
- Do not launch anything new from this cleanup lane.
- 2026-05-15 update: a fresh empty `curvyzero-runs-v2` Modal Volume was
  recreated for the next clean runs lane.
- The shared CurvyTron runs-volume default now points to `curvyzero-runs-v2` in
  `src/curvyzero/contracts/curvytron.py`.
- Keep the old `curvyzero-runs` volume read-only for now because old
  checkpoints/results are still referenced by historical docs and audits. New
  runs should not write there.
- The purge commands below are historical. Do not rerun them against
  `curvyzero-runs-v2` after the 12:28 EDT recreation.
- No wildcard deletes.

## 2026-05-15 Live App Cleanup

- Current necessary Curvy apps:
  `curvyzero-checkpoint-tournament-v2`,
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`, and
  `curvyzero-curvytron-gif-browser-v2`.
- Latest detached tournament work:
  `ap-EqV1pzucLCW8fZjMA3FEqM`, round-6 same-context continuation for
  `curvy-restart18-source-rerate-nonzero-20260515a` /
  `elo-restart18-source-rerate-nonzero-20260515a`.
  Round 6 completed cleanly with `300` pairs / `6300` games, `0` failures,
  `96` active rows, and the same context/roster hashes, but stayed
  `stable=false` with `max_abs_delta=25.199213332028748`. It is stopped now;
  keep it as evidence, not active work.
- Stopped stale detached app `ap-ROvrtOp1TPQUYFEcB1OGDh` after verifying from
  logs that it belonged to old 100-ref diagnostic round-4 work
  (`elo-restart18-source-rerate-20260515a`) and had already written its final
  snapshot. That lane is rejected as a restart source, so keeping leftover tasks
  alive was just noise.
- Round-2 app `ap-0HzVT85O8UHt0rgUdMVyRg` is stopped after completing.
- Round-3 app `ap-91EOlo30iDhxlwDqVgJhYw` is stopped after completing.
- Round-4 app `ap-j21sPzVU0Ow0OS6RUZSXV0` is stopped after completing.
- Round-5 app `ap-9mpXrA6OsyLGY85WAz7KsM` is stopped after completing.
- Round-6 app `ap-EqV1pzucLCW8fZjMA3FEqM` is stopped after completing.

## Purge Result

Completed and verified around 2026-05-15 10:50 EDT.

Deleted exact v2 storage/control objects:

- Volume `curvyzero-runs-v2`
- Volume `curvyzero-curvytron-tournaments-v2`
- Volume `curvyzero-curvytron-control-v2`
- Dict `curvyzero-curvytron-checkpoint-intake-v2`
- Dict `curvyzero-curvytron-opponent-leaderboard-live-v2`
- Queue `curvyzero-curvytron-checkpoint-events-v2`

Verification after deletion:

- `modal volume list --json` no longer shows the three v2 Curvy volumes.
- `modal dict list --json` no longer shows the two v2 Curvy dicts.
- `modal queue list --json` no longer shows the v2 checkpoint-events queue.
- Non-v2 Curvy storage remains present: `curvyzero-runs`,
  `curvyzero-curvytron-tournaments`, and `curvyzero-curvytron-control`.

## Superseded Recheck

This was checked after the seat/slot cleanup and before the all-v2 redeploy.
It is preserved as history only.

- Superseded: the v2 trainer, tournament, and GIF browser apps have since been
  redeployed, and the active source rerate is running in the all-v2 lane.
- Non-v2 Curvy volumes remain present and should be preserved unless a future
  plan explicitly chooses a full namespace reset:
  `curvyzero-runs`, `curvyzero-curvytron-tournaments`,
  `curvyzero-curvytron-control`.
- Non-v2 Curvy coordination objects remain present:
  `curvyzero-curvytron-checkpoint-intake-v0`,
  `curvyzero-curvytron-opponent-leaderboard-live`, and
  `curvyzero-curvytron-checkpoint-events-v0`.
- The non-v2 checkpoint event queue has `7` partitions and `938` total old
  items. This is not active compute, but it is a restart foot gun if we reuse
  an old tournament/rating id. Fresh tournament/rating ids create a fresh queue
  partition.

No new app was launched by this lane.

## Latest Modal Inventory Before Purge

Checked around 2026-05-15 10:32 EDT.

Apps:

- `curvyzero-lightzero-curvytron-visual-survival-train` deployed, 0 tasks.
- `curvyzero-lightzero-curvytron-visual-survival-eval` deployed, 0 tasks.
- `curvyzero-lightzero-curvytron-run-status` deployed, 0 tasks.
- `curvyzero-lightzero-curvytron-visual-survival-train-v2` stopped.
- `curvyzero-checkpoint-tournament-v2` stopped.
- `curvyzero-curvytron-gif-browser-v2` stopped.

V2 purge candidates, now deleted:

- Volume `curvyzero-runs-v2`
- Volume `curvyzero-curvytron-tournaments-v2`
- Volume `curvyzero-curvytron-control-v2`
- Dict `curvyzero-curvytron-checkpoint-intake-v2`
- Dict `curvyzero-curvytron-opponent-leaderboard-live-v2`
- Queue `curvyzero-curvytron-checkpoint-events-v2` with 136 queued events at
  the last read

Non-v2 objects to keep for the clean path:

- Volume `curvyzero-runs`
- Volume `curvyzero-curvytron-tournaments`
- Volume `curvyzero-curvytron-control`
- Dict `curvyzero-curvytron-checkpoint-intake-v0`
- Dict `curvyzero-curvytron-opponent-leaderboard-live`
- Queue `curvyzero-curvytron-checkpoint-events-v0`

## Exact Purge Commands Used

```bash
modal volume delete --yes --allow-missing curvyzero-runs-v2
modal volume delete --yes --allow-missing curvyzero-curvytron-tournaments-v2
modal volume delete --yes --allow-missing curvyzero-curvytron-control-v2
modal dict delete --yes --allow-missing curvyzero-curvytron-checkpoint-intake-v2
modal dict delete --yes --allow-missing curvyzero-curvytron-opponent-leaderboard-live-v2
modal queue delete --yes --allow-missing curvyzero-curvytron-checkpoint-events-v2
```

## Cleanup Rules

- Delete exact v2 objects only.
- Do not delete non-v2 storage from this lane.
- Do not rely on app names alone to decide what evidence is valid; use the
  documented tournament id, rating id, manifest id, and storage namespace.
- If a dashboard looks confusing, fix the dashboard marker/default separately
  instead of preserving stale arenas just because they are visible.
