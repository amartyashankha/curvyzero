# Checkpoint Tournament Modal Refactor: Hilbert Lane

Date: 2026-05-13
Scope: `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
Intent: keep one Modal app and one deploy target, but separate Modal binding from plain tournament logic.

## Current Shape

`curvyzero_checkpoint_tournament.py` is now about 6,009 lines and contains:

- Modal app identity, image, Volumes, retry constants, and mount helpers.
- Volume commit/reload helpers plus web reload throttling and in-process cache.
- checkpoint discovery over the training Volume.
- tournament marker, manifest, visibility, rating config, battle indexes, progress, reducer, and provisional artifacts.
- Modal functions for discovery, game/shard workers, pair/full tournament, rating round/loop/progress/provisional/reduce, browser, visibility, and local CLI.
- review payload builders, battle/rating readers, live shard scans, HTML rendering, FastAPI routes, GIF/meta endpoints, and page/API cache behavior.

This is workable for deploy simplicity but risky for maintenance: every website tweak imports the whole Modal app; every reducer change sits beside FastAPI route code; and tests reach private helpers inside a deployed app module.

## Modal Docs Assumptions Checked

References checked on 2026-05-13:

- Modal Volumes guide: https://modal.com/docs/guide/volumes
- Modal Queues guide: https://modal.com/docs/guide/queues
- Modal Dicts guide: https://modal.com/docs/guide/dicts
- Modal web endpoints guide: https://modal.com/docs/guide/webhooks
- Modal scaling guide: https://modal.com/docs/guide/scale

Operational implications for this lane:

- Volume changes need explicit `commit()` to become visible elsewhere and `reload()` to see other containers' commits.
- A Volume cannot be reloaded while files on that Volume are open in the same container; web routes must read file bytes/json inside short-lived context managers before reloading again.
- During reload, the initiating container's Volume view can appear empty; routes should not issue concurrent reloads around active reads.
- Volumes v2 are the right shape for many distinct shard/game files and concurrent distinct-file writes, but same-file writes remain last-writer-wins and should be single-writer.
- Volume v2 has no total file-count limit, but a single directory has a documented high cap and filesystem traversal can still be expensive. Index artifacts matter.
- Queue is not durable enough for tournament correctness. It is for active communication, not final game/reducer records.
- Dict is useful for small metadata, leases, heartbeats, or compact progress cache, but not as the authoritative artifact store. Dict reads/writes are network operations and values have size/expiry constraints.
- Modal web apps can use `@modal.asgi_app()` with `@modal.concurrent`, but the current `max_inputs=1` avoids in-container reload/cache races at the cost of more containers.
- `.map()` is the right V0 primitive for bounded adaptive shards, but each invocation has concurrency/input limits, so adaptive scheduling should cap work per round rather than queue all-pairs work.

## Boundary Rule

Only the Modal app module may import `modal` or reference Modal function objects.

Plain modules may import:

- `curvyzero.tournament.curvytron_checkpoint_tournament as arena`
- `curvyzero.infra.modal.run_management as runs`
- standard library types and file/path helpers

Plain modules must not import:

- `curvyzero.infra.modal.curvyzero_checkpoint_tournament`
- Modal app/function/Volume objects

This avoids circular imports and preserves the existing deploy command.

## Proposed Module Boundaries

### 1. Thin Modal Entry Point

Keep:

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`

Owns:

- `APP_NAME`, image, Volumes, app object, mount constants.
- Modal retry/autoscaler settings.
- `@app.function`, `@modal.asgi_app`, and `@app.local_entrypoint`.
- `.map`, `.remote`, `.spawn`, and call-id capture.
- conversion between Modal Volume objects and plain `Path` mounts.
- final commit/reload calls around each remote job.

Target size after cuts: mostly decorators plus small wrappers.

### 2. Volume Ops And Cache Helpers

Create:

- `src/curvyzero/infra/modal/volume_ops.py`

Move:

- `_commit_volume`
- `_reload_volume`
- `_web_reload_volume`
- `_web_cache_get`
- `_web_cache_set`
- `_read_cached_file_bytes`

Recommended shape:

- `commit_volume(volume, attempts=5, initial_delay=2.0, backoff=2.0, max_delay=30.0, jitter=True) -> str | None`
- `reload_volume(volume, attempts=4, initial_delay=1.0, backoff=2.0, max_delay=15.0, jitter=True) -> str | None`
- `WebVolumeReloader` holding last reload timestamp and cache clear callback.
- `TtlCache` for simple in-process route caches.

Risk notes:

- Do not set the web reload timestamp before a successful reload unless intentionally throttling reload errors.
- Do not reload while a route still has open file handles.
- Keep GIF bytes cached longer than JSON/progress. Current TTLs are directionally sensible: progress 5s, battle/provisional 30s, GIF bytes 300s, browser GIF cache max-age 1 day.

Smoke tests:

- `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "web_reload_volume or cached_live_rating_progress or dynamic_web_headers"`
- Add a small fake-volume test for retry/backoff once helpers move.

### 3. Checkpoint Discovery

Create:

- `src/curvyzero/tournament/curvytron_checkpoint_discovery.py`

Move:

- `_checkpoint_iteration_from_path`
- `_run_ids_from_prefix`
- `_sort_discovery_rows_by_latest_checkpoint`
- `_attempt_roots_for_run`
- `_checkpoint_candidate_rows_for_run`
- `_discover_checkpoint_refs`
- `_discover_latest_checkpoint_refs`
- `_assert_checkpoint_count`

Modal wrapper remains:

- reload checkpoint Volume
- call discovery module
- return payload

Smoke tests:

- `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "discover or checkpoint_count or checkpoint_refs"`

### 4. Tournament Artifact Store

Create:

- `src/curvyzero/tournament/curvytron_tournament_artifacts.py`

Move:

- `_read_json`
- `_path_for_ref`
- `_write_tournament_marker_at`
- `_write_tournament_manifest`
- `_write_rating_config`
- `_list_tournament_visibility_rows`
- `_update_tournament_visibility`
- `_compact_battle_index_row`
- `_write_battle_index`
- `_slim_rating_snapshot`
- `_slim_provisional_rating_snapshot`
- `_write_provisional_rating_artifacts`
- `_read_rating_round_input`
- `_iter_rating_game_summaries`
- `_rating_round_progress_payload`
- `_pending_rating_progress`
- `_write_rating_progress`
- `_rating_scheduler_state_payload`
- `_write_rating_scheduler_state`
- `_previous_rating_snapshot`
- `_write_rating_round_outputs`
- `_reduce_rating_round_from_summaries`
- `_read_battle_shard_summaries`
- `_summarize_live_pair_from_shards`
- `_summarize_pair_results_from_shard_tallies`
- `_rating_progress_from_pair_results`
- `_read_rating_snapshot`
- `_read_rating_snapshot_for_run`
- `_read_rating_progress`
- `_read_live_rating_progress`
- `_read_rating_config_for_run`
- `_rating_provisional_latest_ref`
- `_read_provisional_rating_snapshot_for_run`
- `_merge_progress_with_provisional_snapshot`
- `_live_pair_results_from_shard_summaries`
- `_build_provisional_rating_snapshot_for_run`
- `_read_best_rating_snapshot_for_run`
- `_battle_index_from_pair_results`

Risk notes:

- All artifacts should remain single-writer by path. Shard workers may write distinct shard/game summary paths concurrently; parent reducers write progress/latest/index paths.
- Avoid writing the same `progress.json` from multiple long-lived loops unless there is a clear winner rule.
- For adaptive Elo, prefer shard-summary reduction over scanning every game summary.
- Keep small committed artifacts (`progress.json`, `latest.json`, `provisional_latest.json`, `battle_index.json`) as the web source of truth.

Smoke tests:

- `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "rating_progress or reduce_rating_round or provisional or battle_index or scheduler_state or round_outputs"`

### 5. Rating Orchestration Helpers

Create:

- `src/curvyzero/tournament/curvytron_rating_orchestration.py`

Move:

- `_build_game_work_specs`
- `_flatten_game_results_from_shards`
- `_dedupe_shard_results`
- `_first_ref_from_shards`
- `_refs_from_shards`

Keep in Modal module:

- actual `.map()` calls, because those reference Modal function objects.
- actual `.remote()` and `.spawn()` calls.

Optional plain helpers:

- `build_rating_round_input_payload`
- `build_work_summary`
- `summarize_round_from_shard_results`
- `build_rating_loop_manifest`

Risk notes:

- Parent fan-in still matters. V0 adaptive Elo should bound `pairs_per_round` and use one shard per pair (`games_per_shard == games_per_pair`).
- Do not add Queue unless `.map()` limits become the dominant blocker. If added later, Queue should carry coarse shard specs only, never the durable game result.

Smoke tests:

- `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "game_work_specs or shard_tallies or rating_round_outputs"`

### 6. Review Payloads

Create:

- `src/curvyzero/tournament/curvytron_tournament_review.py`

Move:

- `_list_tournaments`
- `_battle_checkpoint_ids`
- `_battle_matches_checkpoint`
- `_list_battle_index`
- `_list_battles`
- `_list_rating_runs`
- `_list_rating_latest_runs`
- `_default_rating_run_id`
- `_safe_int_or_none`
- `_checkpoint_live_shard_battles`
- `_enrich_battle_row_from_live_shards`
- `_default_tournament_id`
- `_rating_row_by_checkpoint`
- `_rating_rows`
- `_rating_rank_by_checkpoint`
- `_battle_player_for_checkpoint`
- `_battle_opponent_for_checkpoint`
- `_checkpoint_battle_sort_key`
- `_sort_checkpoint_battle_rows`
- `_wins_for_checkpoint`
- `_review_battle_row`
- `_review_rankings_payload`
- `_review_checkpoint_payload`
- `_review_battle_payload`
- `_read_tournament_json_ref`
- `_compact_review_game`
- `_read_battle_summary`
- `_read_game_summary_refs`
- `_sample_gif_refs`

Risk notes:

- This module should prefer indexes and small artifacts first, then targeted battle/shard reads, then broad scans as last resort.
- It should not reload Volumes itself. Routes or Modal wrappers decide when to reload.

Smoke tests:

- `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "modal_browser or list_battles or review_rankings or review_checkpoint or review_battle"`

### 7. Web Routes And Rendering

Create:

- `src/curvyzero/tournament/curvytron_tournament_web.py`

Move:

- `_href`
- `_page_href`
- `_battle_href`
- `_friendly_progress_label`
- `_short_battle_label`
- `_render_battle_detail_section`
- `_render_page`
- `_render_battle_page`
- `_build_fastapi_app`

Recommended factory signature:

```python
def build_fastapi_app(
    *,
    volume: object,
    mount: Path,
    reload_volume_for_web: Callable[..., str | None],
    read_cached_file_bytes: Callable[..., bytes],
    headers: Mapping[str, str],
    cache_settings: WebCacheSettings,
) -> FastAPI:
    ...
```

Risk notes:

- Keep `DYNAMIC_HEADERS` on all HTML/API responses.
- Keep immutable browser caching only for GIF paths that are content-addressed by tournament artifact ref.
- Do not let the web module import the Modal app module. Inject reload/cache functions instead.
- If `@modal.concurrent(max_inputs=1)` is raised later, protect `_WEB_CACHE` and reload gates with locks, and make reload impossible while streaming/reading file bytes.

Smoke tests:

- `uv run pytest tests/test_curvytron_checkpoint_tournament.py -k "render_page or render_battle_page or dynamic_web_headers or api"`
- `modal serve src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py` for a manual route smoke after the route factory cut.

## Cut Order

1. Move volume/cache helpers to `volume_ops.py`.
   - Low dependency count.
   - Keeps Modal wrapper behavior visible.
   - First chance to fix commit/reload retry behavior.

2. Move artifact store helpers.
   - Highest correctness value.
   - Enables reducer/progress tests without importing the Modal app.
   - Keep compatibility aliases in the Modal module for one PR if needed.

3. Move review payload helpers.
   - Separates web data reads from HTML.
   - Enables targeted tests around stale indexes, live shard fallback, and battle details.

4. Move renderers and FastAPI app factory.
   - Larger text move, but lower logic risk after review helpers are gone.
   - Keep `curvytron_tournament_browser` in the Modal module as a two-line wrapper.

5. Move checkpoint discovery.
   - Mostly path walking; straightforward after artifact/review references are settled.

6. Move orchestration helpers, leaving Modal `.map/.remote/.spawn` in the entry point.
   - Do this after adaptive Elo scheduler decisions stabilize, because this area is changing.

7. Remove temporary re-export aliases from the Modal module.
   - Update tests to import plain modules directly.
   - Keep only Modal wrapper integration tests importing the Modal app module.

## V0 Queue/Dict Decision

Keep V0 adaptive Elo Volume-only for durable records.

Use neither Queue nor Dict for correctness-critical artifacts:

- not game summaries
- not shard summaries
- not battle summaries
- not progress/latest/rating snapshots
- not GIF refs

Possible later Dict uses:

- active rating-loop lease
- active provisional-loop lease
- run heartbeat and call ids
- cancellation flag
- compact latest-progress cache

Possible later Queue uses:

- event-driven coarse shard work dispatch if `.map()` limits become painful
- reducer notification messages after shard commit

Do not use Queue for per-game events or durable results. The Volume remains the record.

## Final Smoke Matrix

After each cut:

```bash
uv run pytest tests/test_curvytron_checkpoint_tournament.py -q
uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py -q
python -m compileall -q src/curvyzero/infra/modal src/curvyzero/tournament
```

Before deploy:

```bash
python - <<'PY'
from curvyzero.infra.modal import curvyzero_checkpoint_tournament as app
print(app.APP_NAME)
print(callable(app.curvytron_tournament_browser))
PY
```

Manual Modal smoke:

```bash
modal serve src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py
```

Then open:

- `/api/tournaments?fresh=true`
- `/api/rating-progress?fresh=true`
- `/api/rating-standings?fresh=true`
- `/api/review/battle?fresh=true&battle_id=<known-battle>`
- `/gif?ref=<known-gif-ref>`

Deploy smoke:

```bash
modal deploy src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py
```

The deploy target should not change during this refactor.

## 2026-05-13 Checkpoint Subscriber/Intake Addendum

The user now wants an online checkpoint intake lane for the longer-term
all-checkpoint adaptive system. Keep this separate from the immediate
top-20/latest visual tournament.

Modal docs check:

- Queue is a good active communication primitive, but it is cleared after 24
  hours from the last put and should not be treated as persistent storage:
  https://modal.com/docs/guide/queues
- Dict is persisted and concurrently accessible, but reads/writes are network
  calls and should stay compact/primitive:
  https://modal.com/docs/guide/dicts
- Volume is still the durable artifact store, with explicit commit/reload
  semantics and concurrent same-file write risks:
  https://modal.com/docs/guide/volumes

Recommended intake shape:

- Queue event: primitive JSON only, for example `run_id`, `attempt_id`,
  `checkpoint_ref`, `iteration`, checkpoint file mtime/size if cheap, and
  observed env/reward/evaluator contract fields.
- Queue partition: by run or matrix so one noisy run cannot starve all intake.
- Dict keys: `seen_checkpoint:<checkpoint_id>`, `latest_by_run:<run_id>`,
  `subscriber_lease:<name>`, and `source_watermark:<source>`.
- Subscriber loop: drain queue in small batches, reload/check the checkpoint
  path, dedupe through Dict, merge a small pending-pool artifact on the Volume,
  commit, and let the adaptive scheduler consume bounded waves.
- Reconciliation: periodically scan the checkpoint Volume directly. This repairs
  missed Queue events and keeps Volume artifacts authoritative.

Do not use Queue for durable game results, battle summaries, rating snapshots,
or GIF refs. It can wake the scheduler up; it cannot be the scheduler's memory.
