# CurvyTron Arena Cleanup - 2026-05-14

## Keep Set

- Do not delete or hide the active final arena.
- Prior keep arena in this note: `curvy-oneframe-visual-exact212-final-20260514a`
- Superseded contaminated arena: `curvy-oneframe-visual-exact212-final-20260514b`
- Next intended keep arena once the main thread relaunches it: `curvy-oneframe-visual-exact212-final-20260514c`
- Next intended rating run once relaunched: `elo-oneframe-visual-exact212-final-20260514c`
- Keep directory: `tournaments/curvytron/leaderboards`

## Hard Warning

This note was updated after a cleanup collision. Treat `curvy-oneframe-visual-exact212-final-20260514b` as trash: it was contaminated and stopped, and it is not the final arena.

Do not run cleanup that deletes or hides `curvy-oneframe-visual-exact212-final-20260514c` after it exists. If `20260514c` is not present yet, do not guess. Wait for the main thread to relaunch it or confirm the exact active final arena name.

Do not use broad wildcard deletion. Do not delete `leaderboards`.

## Actions Taken

- Listed tournament visibility through the built-in visibility mode.
- Initial state had 28 visible arenas.
- Ran visibility cleanup:
  - action: `hide_except`
  - keep: `curvy-oneframe-visual-exact212-final-20260514a`
  - dry run: false
- Result: 27 stale visible arenas were hidden; only the keep arena remained visible.
- Deleted 32 explicitly named stale arena directories with `modal volume rm --recursive`.
- Did not use broad wildcard deletion.
- Did not delete `leaderboards`.
- Did not delete the keep arena.

## Important Finding

Some stale arena directories reappeared after deletion with fresh timestamps. That means live or pending tournament tasks in the deployed `curvyzero-checkpoint-tournament` app are still writing old arena outputs.

The deployed tournament app had hundreds of live/pending tasks at inspection time, so stopping the whole app would also risk interrupting the current real tournament. I did not stop the deployed app.

## Current State At 2026-05-14T07:02:25Z

Historical snapshot from before the correction:

- Current visible arena count: 1
- Current visible arena: `curvy-oneframe-visual-exact212-final-20260514a`
- Safe hide action completed: yes
- Full deletion completed: no, because stale writers recreated some roots while cleanup was running.

Correction after this snapshot: `curvy-oneframe-visual-exact212-final-20260514b` is trash, and `curvy-oneframe-visual-exact212-final-20260514c` is the next intended keep target once it exists.

Historical keep roots from the old snapshot:

- `curvy-oneframe-visual-exact212-final-20260514a` - old keep target from this note's earlier snapshot; verify with the main thread before treating it as active
- `leaderboards` - keep

Current intended keep root after relaunch:

- `curvy-oneframe-visual-exact212-final-20260514c` - preserve after it exists

Exact stale arena dirs still seen by the visibility list and/or volume listing:

- `arena-closed-loop-smoke-20260513a` - stale, recreated by writer
- `arena-closed-loop-smoke-20260513b` - stale, recreated by writer
- `arena-closed-loop-smoke-20260514-v2b` - stale, recreated by writer
- `arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513d` - stale, recreated by writer
- `arena-full-loop-oneframe-20260514a` - stale, recreated by writer
- `arena-inspector-loop-oneframe-smoke-20260514a` - stale, recreated by writer
- `arena-inspector-loop-oneframe-smoke-20260514b` - stale, recreated by writer
- `arena-live-loop-smoke-20260514-a` - stale, recreated by writer
- `arena-live-loop-smoke-20260514-b` - stale, recreated by writer
- `arena-oneframe-top100-plus-latest-20260514a` - stale, recreated by writer
- `arena-oneframe-top100-plus-latest212-20260514b` - stale, recreated by writer
- `arena-oneframe-top100-plus-latest212-coachtest-20260514c` - stale, recreated by writer
- `curvy-oneframe-visual-exact212-20260514c` - stale, recreated by writer; not the final arena because it is missing `final`
- `curvy-oneframe-visual-exact212-final-20260514b` - trash; contaminated and stopped
- `curvy-oneframe-visual-main-20260514a` - stale, recreated by writer
- `curvy-oneframe-visual-main-20260514b` - stale, recreated by writer

The inspector proof dirs listed in the first pass were deleted and were not present in the latest volume listing.

## Follow-Up Runbook

1. First verify the active final arena name from the main thread. Expected next keep target:

   - `curvy-oneframe-visual-exact212-final-20260514c`
   - `elo-oneframe-visual-exact212-final-20260514c`

2. If `20260514c` exists and is the active final arena, keep browser cleanup scoped to that exact arena:

   ```bash
   uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
     --mode visibility \
     --visibility-action hide_except \
     --visibility-keep-tournament-ids curvy-oneframe-visual-exact212-final-20260514c \
     --no-visibility-dry-run
   ```

3. Do not run this if `20260514c` has not been relaunched yet. In that case, do nothing except recheck/list.

4. Later, retry exact-path deletion for stale roots only, after verifying none of these are the active final arena. These are future cleanup commands only; they were not run as part of this correction:

   ```bash
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-closed-loop-smoke-20260513a
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-closed-loop-smoke-20260513b
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-closed-loop-smoke-20260514-v2b
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513d
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-full-loop-oneframe-20260514a
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-inspector-loop-oneframe-smoke-20260514a
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-inspector-loop-oneframe-smoke-20260514b
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-live-loop-smoke-20260514-a
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-live-loop-smoke-20260514-b
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-oneframe-top100-plus-latest-20260514a
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-oneframe-top100-plus-latest212-20260514b
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/arena-oneframe-top100-plus-latest212-coachtest-20260514c
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-visual-exact212-20260514c
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-visual-exact212-final-20260514b
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-visual-main-20260514a
   modal volume rm --recursive curvyzero-curvytron-tournaments tournaments/curvytron/curvy-oneframe-visual-main-20260514b
   ```

5. Never include `curvy-oneframe-visual-exact212-final-20260514c` in deletion commands once it exists.

6. Recheck:

   ```bash
   modal volume ls curvyzero-curvytron-tournaments tournaments/curvytron
   uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
     --mode visibility \
     --visibility-action list
   ```

7. If stale roots keep reappearing and cleanup becomes urgent, the operator choice is to stop/redeploy the tournament app and then relaunch the real arena. Do that only intentionally, because it may interrupt the current real tournament.

## Future Cleanup Agent Handoff

Your job is to keep the tournament browser clean without harming the active final arena.

- Do not run destructive commands until you verify the active final arena.
- Treat `curvy-oneframe-visual-exact212-final-20260514b` as trash.
- Preserve `curvy-oneframe-visual-exact212-final-20260514c` after it exists.
- Preserve `tournaments/curvytron/leaderboards`.
- Prefer `visibility hide_except` before deletion. It is the reversible cleanup step.
- Delete only exact stale paths, never wildcard groups.
- If stale roots reappear, assume an active writer is recreating them and leave the lane resumable instead of fighting it.
