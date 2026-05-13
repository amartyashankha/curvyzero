# Checkpoint Tournament To-Do, 2026-05-13

## Now

- Keep docs current.
- Current pivot: stop treating latest-only all-pairs as the main path. The new
  lane is all-checkpoint adaptive Elo. See
  `checkpoint_tournament_orchestration_2026-05-13.md`.
- Keep an active failure-mode critique doc:
  `checkpoint_tournament_architecture_critique_2026-05-13.md`.
- Build the scheduler in research-first order: design, pure helper, tests, CLI
  hook, tiny smoke, then larger launch.
- Keep the main thread on orchestration. Rating research, code architecture,
  Modal ops, website scale, and docs synthesis are parallel sub-agent lanes.
- Run local tests after edits.
- Redeploy the Modal app after code changes.
- Verify the tournament browser loads rankings at three states: starting,
  running with `provisional_latest.json`, and complete with `latest.json`.
- Purged the old latest-212 tournament artifact
  `arena-curvytron-latest212-allpairs-gpp10-gifs3-20260513a`.
- Rediscovered the latest checkpoint for every preserved CurvyTron run
  immediately before relaunch: 212 found, 0 missing.
- Stopped and purged the even-game latest-212 relaunch. Even battle sizes are
  now invalid for new tournament specs.
- Relaunched the latest-212 all-pairs run detached with 11 games per pair, 11
  games per shard, and 3 GIF samples per pair:
  `arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513c`.
- Automatic provisional-rating loop is now spawned by `curvytron_rating_loop`.
- Website progress now merges the small provisional snapshot so the progress
  strip can show live rated pairs without scanning shard summaries.
- Open or curl the tournament browser and confirm rankings, battle drilldown,
  and GIF samples. Done after the checkpoint-drilldown fallback patch:
  rankings returned 212 rows, checkpoint drilldown returned 8 rows from 211
  scheduled battles in 1.228 seconds, battle detail returned 11 games and 3 GIF
  samples in 0.157 seconds, and the first GIF was `GIF89a`, `704x704`.
- Test dropdown behavior. Done: the deployed page now attaches change handlers
  to tournament and rating selects, updates the URL immediately, and clears
  stale checkpoint/battle state.
- Website refresh contract is now explicit: page reload forces one Volume
  refresh; auto-progress polling reads only small committed artifacts and does
  not scan shard summaries.
- Latest timestamped run progress is complete, but its final `latest.json` was
  still missing after the detached reduce spawn. Recheck reduce output before
  treating that arena as final.
- Battle drilldown sorting is now a first-class website requirement: opponent
  rank, average steps, and failures must be sortable without a page reload.
- New tournament defaults should use 21 games per pair unless an operator passes
  a smaller odd number intentionally.
- New tournament defaults should use a long score-game cap. Current default is
  8,000 decision steps, because shorter caps can turn survival games into fake
  timeout draws.
- Before the next full latest-212 launch, recheck the observation/runtime
  contract: current two-seat training and tournament scoring should both use
  `VectorMultiplayerEnv` plus per-seat `SourceStateGray64Stack4`, with each
  checkpoint's render/model contract recovered from metadata.
- Use meaningful arena names that include the important contract: latest count,
  score/eval, games per pair, max step cap, and observation-match intent.
- For adaptive runs, include the player set and scheduler in the name, for
  example `arena-curvytron-allckpt-adaptive-elo-gpp21-step8000-YYYYMMDDa`.

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
- Add `schedule_reason` to adaptive pair specs so the website and docs can
  explain why a battle exists.
- Add `pair_key` and `pool_hash` support before large adaptive runs. Pair
  history must be keyed by canonical checkpoint pair, not battle id.
- Add a synthetic scheduler/rating simulator before trusting a large adaptive
  run. It should prove that a bounded number of scheduled pairs can recover a
  rough hidden strength order.
- Add `checkpoint_selection=latest|all|iteration` to discovery. Default remains
  `latest`; adaptive all-checkpoint runs use `all`.
- Add a shard-summary reduce path for repair/reduce. Current normal shard flow
  can reduce from shard tallies, but manual reduce still falls back toward game
  summary scanning.
- Add or plan per-checkpoint battle indexes before very large adaptive runs.
  The website should not list a million battles and filter them in the request
  handler.

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
