# Checkpoint Tournament To-Do, 2026-05-13

## Now

- Keep docs current.
- Run local tests after edits.
- Redeploy the Modal app after code changes.
- Find two current checkpoint refs.
- Run one tiny real pair smoke.
- Open or curl the tournament browser.

## Soon

- Add checkpoint discovery helpers for "latest N checkpoints from recent runs."
- Add a tournament dry-run estimator.
- Add a repair/backfill command for pair and tournament aggregates.
- Add a simple standings table to the tournament website.
- Decide default policy mode for score tournaments: greedy eval or collect.

## Elo Phase

- Add batch Elo from battle summaries.
- Add rating snapshots under a separate `ratings/` namespace.
- Add synthetic simulator tests for Elo recovery.
- Add website view for rating standings and battle drill-down.

## Do Not Do Yet

- Do not build a complex tournament format.
- Do not add a new Modal app per battle.
- Do not make training depend on tournament code.
- Do not save every GIF for a huge tournament unless explicitly asked.
