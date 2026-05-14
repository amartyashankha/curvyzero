# Checkpoint Tournament Domain Refactor Critique

Date: 2026-05-13

Scope: CurvyTron checkpoint tournament domain module only. No source code was
changed in this pass.

## Facts

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py` is carrying
  several roles at once: contracts, specs, scoring, scheduler, reducer, policy
  loading, game runtime, and artifact writes.
- The module is not truly pure domain today because it imports
  `curvyzero.infra.modal.run_management` for ID cleaning, ref validation, path
  resolution, and JSON/artifact writes.
- The safest architectural goal is not a new framework. It is small domain
  modules with the existing file left as a compatibility facade.
- Live tournament artifacts and the website depend on current schema IDs,
  ref paths, field names, and schedule/rating metadata. Those contracts should
  move before they are changed.
- The all-checkpoint adaptive Elo target increases pressure on the scheduler and
  reducer, so those should become independently testable before more policy is
  added.

## Proposed Domain Boundaries

Use this package shape behind the existing facade:

```text
curvyzero/tournament/curvytron/
  contracts.py
  specs.py
  scoring.py
  scheduler.py
  reducer.py
  game_runtime.py
```

`contracts.py`

- Schema IDs, artifact filenames, defaults, selection/reason constants.
- Safe ID helpers, checkpoint ID generation, pair key generation.
- Tournament/rating/battle/game ref builders.
- `validate_tournament_artifact_ref`.
- Initially it may still depend on `run_management`; removing that dependency is
  a later cleanup.

`specs.py`

- `normalize_checkpoint_spec`, `normalize_checkpoint_specs`.
- Pair/game/shard spec normalization and builders.
- GIF sampling helpers.
- Tournament plan estimates.
- Rating spec normalization can live here at first because it defines scheduler
  and reducer inputs.

`scoring.py`

- `score_from_info`.
- Game tally merge helpers.
- Pair summaries, compact game result shape, standings.
- This module should know schemas and specs, but not Modal, Volume, FastAPI, or
  policy loading.

`scheduler.py`

- Adaptive and legacy pair-slot selection.
- Schedule reason constants if not centralized in `contracts.py`.
- Scheduler state validation against pool hash.
- It should return pair slots and schedule metadata, not write artifacts or run
  games.

`reducer.py`

- Elo math.
- Rating result from pair summary.
- Batch rating snapshot from pair results.
- Pair history accumulation.
- Active/provisional status rules.
- No Volume writes; it should return plain dict artifacts for callers to persist.

`game_runtime.py`

- Checkpoint payload inspection.
- Policy loading.
- Runtime source-frame/decision settings.
- `_policy_action`.
- `run_checkpoint_game` and failure summary.
- GIF/frame writing can remain here for now because it is coupled to the game
  loop and existing tests.

## What Should Move First

1. Move `contracts.py` first.
   Keep the old module re-exporting every moved symbol. This is low behavior
   risk and gives later cuts a stable place for schema/ref constants.

2. Move `scoring.py` next.
   It is mostly pure and already has compact tests. Keep `summarize_pair_results`
   behavior byte-for-byte compatible.

3. Move `specs.py`.
   This touches many callers, so do it after contracts/scoring. Keep
   `build_rating_round_pair_specs` callable from the old facade even if its
   internals start delegating to scheduler.

4. Move `reducer.py`.
   The reducer is central to all-checkpoint Elo. It should be pure and should not
   write pair history/latest/scheduler files itself.

5. Move `scheduler.py`.
   Do this after reducer/specs are stable. Adaptive scheduling is new and likely
   to change; isolating it makes iteration safer.

## What Must Not Move Yet

- Do not move `run_checkpoint_game` first. It couples LightZero policy loading,
  observation construction, source-frame timing, score extraction, frame/GIF
  capture, and artifact writes.
- Do not rewrite checkpoint payload/model-contract loading during the structural
  refactor. That path protects the tournament/training observation contract.
- Do not change schema IDs, artifact filenames, ref layout, pair IDs, checkpoint
  IDs, or rating formula version as part of a move-only refactor.
- Do not split Modal orchestration at the same time as the domain cuts. The Modal
  app can keep importing the facade until the domain package is boring.

## Main Risks

- Import cycles: `specs`, `scheduler`, and `reducer` all touch rating specs and
  checkpoint rows. Keep `contracts` dependency-only; avoid `scheduler -> specs`
  if possible.
- Hidden infra dependency: `run_management` use inside the domain module can make
  "pure" tests depend on Modal-adjacent utilities. Accept this temporarily; do
  not replace it while moving modules.
- Live artifact compatibility: website rankings, battle drilldown, GIF/meta
  endpoints, and reducers consume the existing JSON field shapes.
- Observation-contract regression: moving runtime code can silently change what
  policies see. Runtime should be the final cut, guarded by direct tests.
- Working tree collision: other agents are actively changing scheduler,
  discovery, and docs. Use re-export moves and avoid sweeping format churn.

## Validation By Cut

After `contracts.py`:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py -k 'tournament_artifact_ref_validation or slim_rating_snapshot or modal_browser_lists_rating_runs'
```

After `scoring.py`:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py -k 'score_from_info or pair_summary_and_standings or shard_tally_pair_summary or schedule_metadata_survives'
```

After `specs.py`:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py -k 'build_pair_specs or games_per_pair or build_game_specs or source_timing_settings or build_game_shard_specs or gif_sampling or pair_spec_rejects'
```

After `reducer.py`:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py -k 'rating_snapshot or pair_history or rating_round_outputs_write_pair_history'
```

After `scheduler.py`:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py -k 'adaptive_v0 or rating_pair_specs_carry_shard_settings'
```

Before any `game_runtime.py` move:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py -k 'policy_loader or source_frame_runtime_settings or checkpoint_game_uses_per_seat_policy_modes or render_contract'
```

Final compatibility pass after each cut:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py
```

## Practical Rule

Every cut should be a move plus re-export, with no semantic changes. If a cut
requires behavior changes, stop and land the behavior under the current module
first, then move it after tests are green.

## Next Smallest Cut After Cleanup Cut 1

Recommendation: extract a small dependency-light `contracts.py` before adding
per-checkpoint indexes.

Reason:

- Per-checkpoint indexes are a new artifact contract and website/read-path
  behavior. They are valuable, but they should be built after the shared artifact
  names, schema IDs, selection constants, and reason constants have one stable
  domain home.
- A constants-only contract cut is a mechanical move and re-export. It should not
  change scheduling, reduction, website reads, or Modal execution.
- Do not try to make all ref builders pure in this cut. Current ref builders and
  ID helpers still depend on `run_management`. Move those in a second contracts
  cut or after replacing that dependency deliberately.

Create:

```text
src/curvyzero/tournament/curvytron/__init__.py
src/curvyzero/tournament/curvytron/contracts.py
```

Move first:

- `TOURNAMENT_TASK_ID`
- `TOURNAMENT_BASE_REF`
- `TOURNAMENT_RUN_MARKER_FILENAME`
- `CHECKPOINT_SELECTION_LATEST`
- `CHECKPOINT_SELECTION_ALL`
- `CHECKPOINT_SELECTION_ITERATION`
- `CHECKPOINT_SELECTION_CHOICES`
- `CHECKPOINT_SCAN_GLOB`
- `CHECKPOINT_EXP_CKPT_DIR_GLOB`
- `CHECKPOINT_WEIGHT_FILENAME_GLOB`
- all `*_SCHEMA_ID` constants
- `RATING_FORMULA_VERSION`
- `POLICY_MODE_*`
- `DEFAULT_*` constants
- `RATING_PAIR_SELECTION_*`
- `RATING_PAIR_SELECTION_CHOICES`
- `SCHEDULE_REASON_*`
- `SCHEDULE_REASON_CHOICES`
- `ARTIFACT_*_FILENAME`
- `ALLOWED_TOURNAMENT_ARTIFACT_FILENAMES`
- `TournamentRefError`
- optionally `_validate_games_per_pair`, if the team accepts it as a tiny
  contract invariant rather than a specs helper

Leave in the facade for now:

- `_safe_id`
- `_slug`
- `_short_hash`
- `checkpoint_id_from_ref`
- `rating_pool_hash`
- `rating_context_hash`
- `rating_pair_key`
- all `*_ref(...)` functions
- `validate_tournament_artifact_ref`
- `parse_checkpoint_refs`
- `_to_plain`
- `exception_payload`

Wrapper/re-export plan:

- In `curvytron_checkpoint_tournament.py`, import the moved names explicitly from
  `curvyzero.tournament.curvytron.contracts`.
- Keep all existing public names available on `curvytron_checkpoint_tournament`
  so Modal/web/tests continue using `arena.DEFAULT_GAMES_PER_PAIR`,
  `arena.RATING_SNAPSHOT_SCHEMA_ID`, and similar.
- Do not change imports in the Modal file yet. Let the facade absorb the move.

Validation for this exact cut:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py -k 'games_per_pair or default_games_per_pair or tournament_artifact_ref_validation or checkpoint_discovery or adaptive_v0 or rating_snapshot'
```

Then run:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py
```

Follow-up contracts cut:

- Move ID/ref helpers only after the constants cut is green.
- Either accept `contracts.py` depending on `run_management`, or first add a tiny
  local relative-ref/clean-id adapter with tests proving identical behavior.

Per-checkpoint index timing:

- Add after contracts cut A and after the artifact ref helpers are stable.
- It should be its own artifact-contract change with tests for index write,
  fallback read, stale index enrichment, and website drilldown speed.
