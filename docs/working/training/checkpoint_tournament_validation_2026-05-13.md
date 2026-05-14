# Checkpoint Tournament Validation, 2026-05-13

## Read First: Current Canary Verdict

- Active canary:
  `arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513d` /
  `elo-top20-furthest-intake-gifs5-20260513d`.
- Verdict: complete and healthy for this visual/intake gate.
- Completion: `420/420` games, `20/20` battles, `failed_game_count=0`,
  `status=complete`.
- Rankings: endpoint returned 20 rows for the canary rating run.
- Battle detail: sample battle returned `sample_gif_count=5`.
- GIF serving: `/gif?ref=...` returned HTTP 200, `Content-Type: image/gif`,
  `704x704`, 15 frames.
- Interpretation: this proves the small top-20 intake canary can complete,
  reduce, expose rankings, expose battle detail samples, and serve GIFs. It does
  not prove large all-checkpoint scale or fully paged battle/game detail.

## Latest Local Gate

- Focused tournament tests passed after scheduler coverage and label fixes:
  `tests/test_curvytron_checkpoint_tournament.py` passed, with optional tests
  skipped.
- Pure scheduler probe for the expanded roster passed: 424 checkpoint refs,
  requested placement budget 212, scheduled slots 212, played checkpoints 424,
  unplayed checkpoints 0.
- Sample labels now include run identity plus iteration, for example
  `blank-browser-heavy-collect64-r298 i300773`.
- Deployed expanded probe completed:
  `arena-curvytron-top20runs-allckpts-placement-gpp21-gifs5-step8000-20260513a`
  / `elo-top20runs-allckpts-placement-gpp21-gifs5-step8000-20260513a`.
  Progress reported `completed_game_count=4452`, `game_count=4452`,
  `pair_count=212`, `status=complete`, `phase=ratings_written`.
- Website/API validation after completion: `/api/review/rankings` returned 424
  rows, zero rows had `games=0`, min/max games were both 21, checkpoint drilldown
  used `source=checkpoint_battle_index`, battle detail returned five GIF
  samples, and `/gif` served HTTP 200 `image/gif`.

## What Must Be True

- Pair specs are deterministic.
- Game specs use stable IDs and seeds.
- Score extraction matches the env terminal info.
- Artifact refs cannot escape the tournament namespace.
- A remote game can load two real checkpoints.
- A remote game can write a summary and GIF.
- The browser can list tournaments and serve GIF/JSON files.

## Adaptive Elo Must Also Prove

- Scheduler output is deterministic from pool, snapshot, round index, and seed.
- Scheduler output is bounded by the requested battle budget.
- Scheduler does not build all possible pairs for huge pools.
- Each adaptive pair has a clear schedule reason.
- New checkpoints get enough distinct opponents before active status.
- Replay battles create new battle refs and do not overwrite old summaries.
- Ratings can still be recomputed from immutable battle summaries.
- Website reads rankings/progress from small snapshot/index files, not all game
  summaries.

## Local Tests

Run:

`uv run pytest tests/test_curvytron_checkpoint_tournament.py`

This proves the pure helpers and local browser listing. It does not prove
checkpoint loading.

Latest focused run:

`PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_curvytron_checkpoint_tournament.py -q`

Result:

`78 passed, 10 skipped`

Latest cleanup Cut 1 checks:

```text
PYTHONDONTWRITEBYTECODE=1 uv run python -m py_compile \
  src/curvyzero/tournament/curvytron_checkpoint_tournament.py \
  src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_tournament.py

PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_curvytron_checkpoint_tournament.py -q

git diff --check -- \
  src/curvyzero/tournament/curvytron_checkpoint_tournament.py \
  src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_tournament.py \
  docs/working/training/checkpoint_tournament_active_threads_2026-05-13.md \
  docs/working/training/checkpoint_tournament_refactor_plan_2026-05-13.md \
  docs/working/training/checkpoint_tournament_todo_2026-05-13.md \
  docs/working/training/checkpoint_tournament_orchestration_2026-05-13.md \
  docs/working/training/checkpoint_tournament_architecture_critique_2026-05-13.md \
  docs/working/training/curvytron_inspector_operating_loop_2026-05-11.md
```

Result before the context-hash guard:

- compile passed;
- focused test result stayed `78 passed, 10 skipped`;
- diff whitespace check passed.

What this proves:

- naming artifact filenames, schedule reasons, rating pair-selection values,
  checkpoint selection values, and orphan rating refs did not break local
  tournament helper behavior.

What this does not prove:

- remote Modal app import after deploy;
- a real adaptive rating smoke;
- website behavior at large scale.

Latest context-hash guard checks:

```text
PYTHONDONTWRITEBYTECODE=1 uv run python -m py_compile \
  src/curvyzero/tournament/curvytron_checkpoint_tournament.py \
  src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_tournament.py

PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_curvytron_checkpoint_tournament.py -q

git diff --check -- \
  src/curvyzero/tournament/curvytron_checkpoint_tournament.py \
  src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_tournament.py \
  docs/working/training
```

First context-hash result:

- compile passed;
- focused test result: `82 passed, 10 skipped`;
- diff whitespace check passed.

Coverage added:

- `rating_context_hash` changes when evaluator settings change but not when the
  roster expands.
- `pair_history.json` with a context hash can carry forward across roster
  expansion.
- pair history rejects changed evaluator context.
- adaptive scheduler state rejects changed evaluator context.
- pair history and scheduler state artifacts record `context_hash`.

Follow-up result after Pauli validation review:

- compile passed;
- focused test result: `84 passed, 10 skipped`;
- diff whitespace check passed.

Extra coverage added:

- previous rating snapshots reject changed evaluator context;
- helper refs for `pair_spec.json`, `provisional_latest.json`, and run-level
  `results.json` validate through the tournament artifact ref guard;
- GIF and game-shard settings do not affect `rating_context_hash`.

Per-checkpoint battle index result:

- compile passed;
- focused test result: `85 passed, 10 skipped`;
- diff whitespace check passed.

Coverage added:

- global battle-index writes also publish per-checkpoint battle-index files;
- checkpoint drilldown reads `checkpoint_battle_index` first;
- provisional battle-index artifacts use the same per-checkpoint read path.

Coverage added in this run:

- `adaptive_v0` requires a pair budget.
- adaptive pair specs are budgeted, unique, and tagged with schedule metadata.
- schedule metadata survives pair summary creation.
- pair history accumulates by canonical pair key across seat order.
- pair history rejects pool-hash mismatch.
- discovery can select latest, exact iteration, or all checkpoints while
  scanning timestamped `lightzero_exp_*` directories.
- Modal rating output helpers write `pair_history.json` and
  `scheduler_state.json`, and snapshots point at those refs.

Checkpoint discovery regression to keep:

- discovery scans `train/lightzero_exp*/ckpt/iteration_*.pth.tar`;
- timestamped DI-engine dirs can beat stale fixed `lightzero_exp/ckpt`;
- resume sidecars and empty checkpoint files do not enter tournament refs;
- `checkpoint_selection=all` can return more checkpoints than selected runs, so
  `max_runs` is not treated as the expected checkpoint count in that mode.

Remote discovery smoke:

```text
uv run --extra modal python -B -m modal run \
  -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode discover \
  --run-id-prefix survivaldiag-v1b-20260513h \
  --max-runs 3 \
  --checkpoint-selection latest
```

Result:

- `found_count=3`
- `missing_count=0`
- `checkpoint_scan_glob=train/lightzero_exp*/ckpt/iteration_*.pth.tar`
- returned refs included timestamped `lightzero_exp_260513_*` dirs, proving the
  tournament path is not limited to stale fixed `lightzero_exp/ckpt`.

Remote adaptive rating smoke:

```text
uv run --extra modal python -B -m modal run \
  -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode rating \
  --tournament-id arena-curvytron-adaptive-v0-context-smoke-20260513a \
  --rating-run-id elo-adaptive-v0-context-smoke \
  --checkpoint-refs <3 refs from survivaldiag-v1b-20260513h discovery> \
  --round-count 1 \
  --pair-selection adaptive_v0 \
  --pairs-per-round 2 \
  --games-per-pair 3 \
  --games-per-shard 3 \
  --max-steps 8 \
  --num-simulations 1 \
  --wait
```

Result:

- completed ok;
- two shard workers ran, one per scheduled pair;
- `pair_count=2`;
- `game_count=6`;
- `rated_pair_count=2`;
- all six tiny games hit the 8-step cap and were draws;
- `latest.json` contains `pool_hash`, `roster_hash`, and `context_hash`;
- rating refs were written for `pair_history.json` and `scheduler_state.json`;
- no GIFs were requested.

Smoke caveat:

- This was a plumbing/context smoke with a deliberately tiny `max_steps=8`.
  The all-draw result says nothing useful about policy strength.

## Modal Smoke

Run one pair with one game:

`uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode pair --tournament-id arena-smoke-YYYYMMDD --checkpoint-refs <ref-a>,<ref-b> --games-per-pair 1 --max-steps 16 --num-simulations 1 --wait`

Success means:

- command returns `ok: true` or a clear failure summary
- `tournaments/curvytron/<id>/show_in_tournament_browser.flag` exists in
  `curvyzero-curvytron-tournaments`
- `tournaments/curvytron/<id>/battles/.../battle.json` exists in
  `curvyzero-curvytron-tournaments`
- `tournaments/curvytron/<id>/battles/.../games/game-000000/summary.json`
  exists in `curvyzero-curvytron-tournaments`
- `game.gif` exists when GIF saving is enabled
- game summary has `frame_size: 704`
- game summary has `artifacts.gif.pixel_size: [704, 704]`

## Website Smoke

Open the deployed tournament browser.

Check:

- tournament appears in the dropdown/list
- battle card appears
- JSON link opens battle JSON
- GIF link opens the sample GIF

The `/api/tournaments` endpoint should answer quickly even when no tournament
exists. If it hangs, first suspect slow web-container import or a Volume listing
path that scans too much.

Observed fix on 2026-05-13: after removing the trainer import, the tournament
image needed an explicit `fastapi` dependency for the ASGI browser.

The website reads from `curvyzero-curvytron-tournaments`, not `curvyzero-runs`.

## Failure Handling

If a game fails, it should still write a failure summary. That is useful because
checkpoint loading failures are product evidence, not just infrastructure noise.

## Latest Smoke, 2026-05-13

Command shape:

`uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode tournament --tournament-id arena-v2volume-704-smoke-20260513a --checkpoint-refs <iteration_0>,<iteration_7> --games-per-pair 1 --max-steps 8 --num-simulations 1 --wait`

Result:

- completed ok
- checkpoint volume: `curvyzero-runs`
- artifact volume: `curvyzero-curvytron-tournaments`
- tournament id: `arena-v2volume-704-smoke-20260513a`
- website `/api/tournaments` lists this tournament from the new artifact volume
- website `/api/battles` lists one battle
- game summary reports `frame_size: 704`
- game summary reports `artifacts.gif.pixel_size: [704, 704]`
- downloaded GIF header reports `GIF image data, version 89a, 704 x 704`

Smoke caveat:

- It was a tiny 8-step timeout smoke, not a meaningful policy comparison.
- It proves plumbing, storage split, website read path, and GIF size.

## Local Cleanup Validation, 2026-05-13

After the refactor cleanup pass:

- compile check passed for the tournament helper, contracts module, Modal app,
  and focused tests;
- facade compatibility check passed for representative `arena.*` contract names;
- checkpoint-index focused tests passed:
  `9 passed, 88 deselected` before the roster guard and
  `9 passed, 91 deselected` after it;
- full focused tournament test file passed:
  `90 passed, 10 skipped`;
- `git diff --check` passed for the touched code/tests/docs.

What this proves:

- the old public `arena.*` surface still works after moving pure contract names
  to `curvyzero.tournament.curvytron.contracts`;
- per-checkpoint battle indexes carry their own refs;
- checkpoint index reads filter stale wrong rows;
- checkpoint drilldown no longer scans live shard summaries when a
  checkpoint-specific index exists.
- roster expansion still works, but previous pair history, scheduler state, and
  previous snapshots reject the case where an explicit checkpoint id is reused
  for a different checkpoint ref.

What this does not prove:

- a large adaptive tournament is ready;
- the website is fully paged for very large battle/game/GIF detail;
- policy observation parity is fully audited for every historical checkpoint.

## Adaptive Contract Smoke, 2026-05-13

Command shape:

`uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode rating --tournament-id arena-curvytron-cleanup-contracts-adaptive-smoke-20260513a --rating-run-id elo-adaptive-contract-smoke --checkpoint-refs <3 discovered refs> --round-count 1 --pair-selection adaptive_v0 --pairs-per-round 2 --games-per-pair 3 --games-per-shard 3 --max-steps 64 --num-simulations 1 --wait`

Run:

- Modal app run:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-LMJY8yvBC8fzWvBiizvOFz`
- tournament id:
  `arena-curvytron-cleanup-contracts-adaptive-smoke-20260513a`
- rating run id:
  `elo-adaptive-contract-smoke`

Result:

- `pair_count=2`;
- `game_count=6`;
- `rated_pair_count=2`;
- `max_abs_delta=8.0`;
- no game failures;
- latest snapshot includes `checkpoint_roster`;
- pair history and scheduler state refs were written;
- GIFs were off, as intended for a score smoke.

What this proves:

- the extracted contracts import correctly inside Modal;
- adaptive scheduling still launches game shards;
- shard summaries reduce into pair history, scheduler state, and latest ratings;
- the roster-identity field survives the remote path.

What this does not prove:

- the ranking is meaningful, because this was only 3 checkpoints and 6 games;
- large all-checkpoint scale;
- website speed at large battle/GIF detail.

## Website Deploy Smoke, 2026-05-13

Deployment:

- Modal deployment:
  `https://modal.com/apps/modal-labs/shankha-dev/deployed/curvyzero-checkpoint-tournament`
- Browser endpoint:
  `https://modal-labs-shankha-dev--curvyzero-checkpoint-tournament--03b893.modal.run`

Checks:

- `/api/tournaments?limit=5` returned quickly and listed
  `arena-curvytron-cleanup-contracts-adaptive-smoke-20260513a` first.
- `/` returned HTTP 200 with a 16,633 byte HTML page.
- `/api/review/rankings` for the cleanup smoke returned three ranking rows.
- `/api/rating-progress` for the cleanup smoke returned `status=complete`,
  `completed_game_count=6`, and `volume_reload_error=null`.
- `/api/review/checkpoint` for the top row returned
  `source=checkpoint_battle_index`, one battle row, schedule metadata, and
  tally fields.

What this proves:

- the deployed app includes the new contract module;
- the website normal checkpoint-drilldown path uses the per-checkpoint index;
- the deployed API can read the post-cleanup smoke artifacts.

What this does not prove:

- browser interaction with every dropdown/sort path;
- large-list paging and GIF-heavy battle detail speed.

## Scale Probe Later

Before 300 checkpoints:

- run 4 checkpoints, unordered pairs, 2 games each
- run 10 checkpoints, unordered pairs, 1 game each
- measure total wall time, failures, and file count
- decide whether one-game-per-input is still good or needs game shards

## Intake And GIF Canary, 2026-05-13

Focused local tests:

- `uv run pytest tests/test_curvytron_checkpoint_tournament.py -q`
- Latest result after V0 intake hardening: `95 passed, 10 skipped`.

What the new test protects:

- explicit checkpoint refs passed into intake discovery produce a non-empty
  discovery payload;
- five GIF sample settings survive the intake manifest into rating pair specs;
- Queue partition keys stay under Modal's 64-character limit.
- queued refs are tracked separately from discovered/seen refs;
- existing rating-run output is detected before drain spawns work.

Remote checks:

- Preserved-run discovery found `212/212` runs.
- Top-20 canary estimate returned `checkpoint_count=20`, `pair_count=20`,
  `game_count=420`, `gif_count=100`, and `gif_per_pair=5`.
- Deployed intake seed for
  `arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513d` returned
  `seed_checkpoint_count=20` and `seed_enqueued_count=20`.
- Deployed intake drain returned `drain_event_count=20`,
  `drain_checkpoint_count=20`, and rating call
  `fc-01KRHN3X4F6V04BQTC4HV51701`.
- Website/API progress for the canary showed `pair_count=20`,
  `game_count=420`, and phase moved from `game_map_started` to
  `games_running`.

Earlier follow-up checklist, now resolved below:

- at least one completed battle exposes `sample_gif_count >= 5`;
- `/gif?ref=<sample>` returns HTTP 200 and `Content-Type: image/gif`;
- rankings/checkpoint/battle drilldown work for the same canary run after reduce.

Completed follow-up:

- The canary finished: `420/420` games, `20/20` battles, `failed_game_count=0`,
  `phase=reduced`, `status=complete`.
- Rankings endpoint returned 20 rows for the canary rating run.
- Battle detail for
  `rate-elo-top20-furt-r000000-pair-000000-ckpt-000-train-l-vs-ckpt-005-train-l-7bec704bf3`
  returned `sample_gif_count=5` with games `0, 5, 10, 15, 20`.
- First GIF served from `/gif?ref=...` with HTTP 200,
  `content-type=image/gif`, `704x704`, 15 frames, and 8071 bytes.
- Battle settings recorded `frame_size=704`, `save_gif=true`,
  `gif_sample_games_per_pair=5`, `gif_trail_render_mode=browser_lines`, and
  `natural_bonus_spawn=true`.

Cleanup/scale reading:

- The immediate cleanup phase is website/detail-index hardening for GIF-heavy
  battle detail, not a broad tournament refactor.
- Keep Modal Dict/Queue as the intake coordination layer, with Volume artifacts
  as the durable source of truth.
- The next scale gate should deliberately pick one path: a larger latest-only
  adaptive placement probe, or the all-checkpoint online Elo intake. Either path
  needs bounded website/detail reads before it becomes large.

## Intake Idempotency Tests To Add

- Queue failure recovery: if enqueue fails after discovery, the checkpoint must
  remain pending or be re-enqueued on the next tick.
- Duplicate tick dedupe: two scans of the same new checkpoint should create one
  durable event/generation.
- Large seed coalescing: 212 queued checkpoint events should lead to one rating
  placement spawn, not multiple full rating loops for the same manifest.
- Existing rating guard: repeated drains for the same `rating_run_id` should
  no-op unless an operator explicitly overrides.
- Online continuation: a new checkpoint after `latest.json` exists must either
  start the next round from the previous snapshot or fail loudly as unsupported.

## Deployed Intake V0 Queue Smoke, 2026-05-13

- Smoke id:
  `arena-curvytron-intake-v0-queue-smoke-20260513a` /
  `elo-intake-v0-queue-smoke`.
- Seed used two explicit checkpoint refs, `active=false`, and
  `enqueue_existing=true`.
- Seed result: `seed_checkpoint_count=2`, `seed_enqueued_count=2`,
  `queued_checkpoint_count=2`, and the first event had stable
  `event_id=93c86f53a7ce5fbc`.
- Manual drain result: `drain_event_count=2`, `drain_rating_call_id=""`.
- The smoke marker was hidden after the check, so it should not clutter the
  website.
