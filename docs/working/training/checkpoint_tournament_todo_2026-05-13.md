# Checkpoint Tournament To-Do, 2026-05-13

## Now

- Keep docs current.
- Current pivot: stop treating latest-only all-pairs as the main path. The new
  lane is all-checkpoint adaptive Elo. See
  `checkpoint_tournament_orchestration_2026-05-13.md`.
- Keep the active-thread ledger current:
  `checkpoint_tournament_active_threads_2026-05-13.md`.
- Keep the refactor plan current:
  `checkpoint_tournament_refactor_plan_2026-05-13.md`.
- Keep an active failure-mode critique doc:
  `checkpoint_tournament_architecture_critique_2026-05-13.md`.
- Build the scheduler in research-first order: design, pure helper, tests,
  artifact wiring, tiny smoke, then larger launch.
- First pure helper patch landed: `adaptive_v0` pair selection, schedule
  metadata pass-through, canonical `pair_key`, `pool_hash`, and pair-history
  helper. Focused tests passed after discovery tests were added:
  `78 passed, 10 skipped`.
- Remote discovery smoke passed with timestamped `lightzero_exp_*` checkpoint
  refs visible in the returned rows.
- Keep the main thread on orchestration. Rating research, code architecture,
  Modal ops, website scale, and docs synthesis are parallel sub-agent lanes.
- Run local tests after edits.
- Redeploy the Modal app after code changes.
- Use meaningful arena names that include the important contract: latest count,
  score/eval, games per pair, max step cap, and observation-match intent.
- For adaptive runs, include the player set and scheduler in the name, for
  example `arena-curvytron-allckpt-adaptive-elo-gpp21-step8000-YYYYMMDDa`.

## Current Next Gates

- Land refactor Cut 1: name repeated contract strings and artifact paths without
  changing behavior.
- Rerun focused tests and compile checks.
- Run a tiny adaptive rating smoke with explicit checkpoint refs.
- Record the smoke result in the validation and active-thread docs.
- Keep website/per-checkpoint index work as a follow-up before large runs.

## Recent Historical Notes

- Older latest-212 all-pairs runs were stress tests, not the main path now.
- Even battle sizes are invalid for new tournament specs.
- Website dropdown, progress, and drilldown fixes landed earlier and should stay
  covered by focused tests.

## Soon

- Add checkpoint discovery helpers for "latest N checkpoints from recent runs."
- Add a tournament dry-run estimator. A basic `--mode estimate` exists; decide
  whether it is enough for operator use.
- Add a repair/backfill command for pair and tournament aggregates.
- Watch whether the provisional loop cadence is frequent enough for long
  tournaments. It currently refreshes about once per minute.
- Decided default policy mode for score tournaments: greedy/eval for official
  score, collect mode only as a separately labeled diagnostic or visual sample.
- Add an explicit tournament evaluation contract so observation/environment
  settings cannot drift silently.
- First contract patch is done: policy loading recovers model env/reward variant
  metadata and records the effective values. Still add parity tests for the
  actual observation tensor path before calling the tournament contract final.
- Decide seat fairness for real Elo. Current latest-212 runs are unordered
  no-self pairs with one fixed seat order. That is fast and simple, but not
  seat-neutral.
- Add a GIF preview/downsample lane only if lazy loading plus byte caching is
  still not enough.
- Start the adaptive Elo design: new checkpoint enters, plays a small but useful
  set of games against anchors/current neighbors, then old important matchups
  are replayed when rankings are uncertain or stale.
- Modal rating rounds now write scheduler-state and pair-history artifacts, and
  the parent rating loop passes them into the next round.
- Add a tiny adaptive remote smoke with explicit checkpoint refs before any
  larger all-checkpoint run.
- Add a synthetic scheduler/rating simulator before trusting a large adaptive
  run. It should prove that a bounded number of scheduled pairs can recover a
  rough hidden strength order.
- Discovery now has `checkpoint_selection=latest|all|iteration`. Default
  remains `latest`; adaptive all-checkpoint runs use `all`.
- Add a shard-summary reduce path for repair/reduce. Current normal shard flow
  can reduce from shard tallies, but manual reduce still falls back toward game
  summary scanning.
- Add or plan per-checkpoint battle indexes before very large adaptive runs.
  The website should not list a million battles and filter them in the request
  handler.
- Promote orphan path literals into helpers before adding more features:
  `provisional_latest.json`, rating-run `results.json`, and `pair_spec.json`
  should have clear ownership and helper refs.

## Elo Phase

- Batch Elo from battle summaries exists for final snapshots.
- Provisional Elo exists as a website bridge from `provisional_latest.json`.
- Add synthetic simulator tests for Elo recovery.
- Add tests for background provisional snapshots once that writer exists.

## Do Not Do Yet

- Do not build a complex tournament format.
- Do not add a new Modal app per battle.
- Do not make training depend on tournament code.
- Do not save every GIF for a huge tournament unless explicitly asked.
