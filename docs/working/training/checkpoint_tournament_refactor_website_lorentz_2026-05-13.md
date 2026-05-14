# Checkpoint Tournament Website Refactor Notes

Date: 2026-05-13
Lane: website modularity and artifact scale
Scope: `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`

## Product Shape

Keep the website as one page:

1. Live progress and data freshness.
2. Rankings, final or provisional.
3. Click a policy/checkpoint to see paged battles.
4. Click a battle to see a compact summary, GIF samples, and paged games.

The page must stay useful during a long-running adaptive Elo run. It should load fast, preserve state during refresh, and never do a full tournament scan because a user clicked a row.

## Clean Layers

V0 should split behavior by responsibility while keeping Modal route wiring in the existing module.

1. Route layer
   - Keep `_build_fastapi_app(volume)` in `curvyzero_checkpoint_tournament.py`.
   - Route handlers should parse query params, optionally reload the volume, call one payload builder, and return HTML or JSON.
   - Avoid rendering logic and artifact traversal in route functions.

2. Payload/API layer
   - Move or isolate:
     - `_review_rankings_payload`
     - `_review_checkpoint_payload`
     - `_review_battle_payload`
     - `_review_battle_row`
     - future `_recent_battles_payload`
   - Payload builders should return plain dicts with paging metadata, freshness metadata, and source metadata.
   - Payload builders should read bounded artifacts only, except named fallback paths that tests explicitly cover.

3. Artifact/index layer
   - Keep tournament execution writes near existing execution code for now.
   - Add small web-facing artifacts:
     - final/latest ratings: existing `latest.json`
     - running ratings: existing `provisional_latest.json`
     - progress: existing `progress.json`
     - global recent battles: small `recent_battles.json` or progress-derived rows
     - per-checkpoint battle indexes
     - optional battle game index/pages

4. Render layer
   - Keep HTML render pure.
   - Split `_render_page` into section functions:
     - header/pickers
     - progress
     - rankings
     - recent battles
     - checkpoint detail
     - checkpoint battles
     - battle detail
   - Rendering functions should not read the filesystem or mutate cache.

5. Assets layer
   - Extract inline CSS and JS into constants first:
     - `TOURNAMENT_BROWSER_CSS`
     - `TOURNAMENT_BROWSER_JS`
     - `TOURNAMENT_BATTLE_PAGE_CSS`
   - Later move them to a browser assets module. Do not introduce bundling in V0.

6. Cache layer
   - Isolate:
     - `_web_reload_volume`
     - `_web_cache_get`
     - `_web_cache_set`
     - `_read_cached_file_bytes`
   - Add cache key conventions and total cache limits before adding more cached payloads.

## API Contracts

All dynamic JSON payloads should include:

- `selected_tournament_id`
- `rating_run_id` when relevant
- `source`
- `updated_at`
- `age_seconds` or `is_stale`
- `volume_reload_error`
- `limit`
- `offset`
- `total`
- `has_older`
- `has_newer`

### `GET /api/rating-progress`

Purpose: tiny poll target for progress and freshness.

Source:

- `progress.json`
- merged with `provisional_latest.json` metadata when present

Must not:

- scan all battle dirs
- scan all game summaries
- force volume reload unless `fresh=true`

Key fields:

- `status`
- `phase`
- `pair_count`
- `started_pair_count`
- `completed_pair_count`
- `partial_pair_count`
- `game_count`
- `completed_game_count`
- `estimated_seen_game_count`
- `recent_started_pairs` or `recent_battles_ref`
- `updated_at`
- `is_stale`

### `GET /api/rating-standings`

Purpose: paged rankings.

Source order:

1. final `latest.json`
2. running `provisional_latest.json`

Must not:

- build live provisional rankings by scanning shard summaries in request path
- hide provisional rankings just because final latest is absent

Key fields:

- `provisional`
- `ratings_ref`
- `completed_game_count`
- `total_game_count`
- `completed_pair_count`
- `total_pair_count`
- paged `rows`

UI copy:

- Final: `Rankings`
- Provisional: `Live rankings from completed games`

### `GET /api/recent-battles`

Purpose: useful landing state while no checkpoint is selected.

Source:

- tiny recent battles artifact, or bounded `progress.recent_started_pairs`

Rows should include:

- `battle_id`
- `pair_index`
- `checkpoint_labels`
- `seen_game_count`
- `expected_game_count`
- `complete`
- `schedule_reason`
- `updated_at`
- `first_gif_ref` if already known

Must not:

- scan global battle index
- scan battle dirs

### `GET /api/review/checkpoint`

Purpose: paged/sorted battles for one checkpoint.

Source:

- per-checkpoint battle index

Query params:

- `checkpoint_id`
- `rating_run_id`
- `limit`
- `offset`
- `sort=opponent_rank|updated|avg_steps|failures|games`
- `direction=asc|desc`

Rows should include:

- `battle_id`
- opponent checkpoint id/label
- opponent rank if known
- W-L-D from the selected checkpoint perspective
- completed/failure counts
- average steps
- `first_gif_ref`
- `sample_gif_refs`
- `summary_ref`
- `pair_key`
- `schedule_reason`
- `updated_at`

Must not:

- call `_list_battle_index(... limit=1_000_000)`
- filter the whole global battle index on every click
- enrich each row by scanning live shards unless a small bounded fallback is explicitly requested

### `GET /api/review/battle`

Purpose: battle header, GIF samples, and paged game rows.

Query params:

- `battle_id`
- `gif_sample_limit`
- `game_limit`
- `game_offset`

Payload:

- battle summary/tally
- players
- schedule reason
- `pair_key` for internal linking only
- `sample_gifs`
- paged `games`
- game paging metadata

Must not:

- read every game summary before returning the battle header
- scan global battle index if direct `battle.json` or battle dir exists
- return huge embedded game lists by default

### `GET /gif`

Purpose: serve immutable GIF bytes.

Keep:

- ETag
- browser cache headers
- server byte cache

Add later:

- preview/downsample route or still thumbnail
- bounded total byte cache
- graceful broken GIF placeholder contract in HTML/JS

## Cache And Reload Plan

1. Volume reload
   - Page and API polling should not force reload by default.
   - `fresh=true` should be user-initiated only.
   - Keep throttling for ordinary page reloads and progress polling.

2. Payload cache
   - Progress: short TTL, currently appropriate.
   - Battle detail: short TTL is fine, but key must include `game_limit` and `game_offset` once detail is paged.
   - Rankings: cache by `ratings_ref` or file stat, not only by run id.
   - Checkpoint battles: cache per checkpoint, sort, direction, limit, offset, index file stat.

3. GIF byte cache
   - Current per-item max is useful.
   - Add total byte/item cap before increasing sample counts or preview use.

4. Cache invalidation
   - Clearing all web cache on volume reload is acceptable for V0.
   - Prefer file-stat keys for immutable-ish artifacts so stale cache does not present old data as fresh.

## Paging Plan

1. Rankings
   - Existing route already has `limit` and `offset`.
   - Initial HTML should use a small first page, not max rows.

2. Checkpoint battles
   - Require per-checkpoint indexes.
   - Sort server-side, then page.
   - Browser sort may still rearrange the visible page, but UI should not imply it sorted the full result set unless server sort was used.

3. Battle games
   - Add `game_limit` and `game_offset`.
   - Render only first page on battle click.
   - Add next/previous controls for games.
   - GIF samples should come from `sample_gif_refs` before full game expansion.

4. Recent battles
   - Fixed small limit, probably 25 or 50.
   - No deep paging needed for V0; link to checkpoint/battle drilldown for details.

## UI-State Contract

1. Dropdowns
   - Changing tournament clears:
     - `rating_run_id` unless explicitly reselected
     - `checkpoint_id`
     - `battle_id`
     - `offset`
     - hash
   - Changing rating clears:
     - `checkpoint_id`
     - `battle_id`
     - battle/checkpoint offsets
     - hash

2. Selection
   - Selected checkpoint and battle are URL state.
   - Auto-refresh must not erase selected row styling.
   - If selected battle disappears from the current page, show a small selected-battle panel rather than jumping away.

3. Scroll
   - Auto-refresh should patch rows in place.
   - Preserve:
     - page scroll
     - rankings scroll container
     - battles scroll container
     - current sort and pagination

4. Freshness
   - Every panel should show data age or last update.
   - Stale progress should not look live.
   - If provisional rankings are old, label them as old provisional data.

5. Provisional rankings
   - Use plain copy:
     - `Live rankings from completed games`
     - `Not final`
   - Show completed/total games beside the label.
   - Do not let coach-facing copy imply final ranking quality until final latest exists.

6. Schedule reason
   - Show schedule reason as plain coach copy:
     - `placement`: `new/unknown checkpoint`
     - `near_rating`: `close matchup`
     - `uncertain`: `needs more evidence`
   - Do not show raw `pair_key`.
   - Do not show `schedule_priority` as importance or confidence.

7. Broken GIFs
   - Broken GIF card becomes `GIF unavailable`.
   - JSON/game links remain visible.
   - Missing GIF should not make battle detail look failed if game summaries exist.

## Failure Modes To Test

1. Stale data presented as fresh
   - Progress and ranking payloads include freshness metadata.
   - Render labels stale progress/provisional snapshots clearly.

2. Click path scans too much
   - Checkpoint click does not call global battle index with huge limit.
   - Battle click does not scan all games unless a paged fallback is explicitly requested.

3. Dropdown state leak
   - Tournament/rating change clears stale checkpoint/battle state and offsets.

4. Partial rankings hidden
   - Provisional rows render when no final latest exists.

5. Broken GIF samples
   - Missing GIF refs render unavailable state, not broken cards or empty battle detail.

6. Scroll/state jumps during auto-refresh
   - JS preserves selected checkpoint, selected battle, sort, paging, and scroll containers while replacing rows.

7. Coach misreads provisional rankings
   - Render copy differentiates provisional from final.
   - Completed/total evidence is visible near the ranking title.

8. Adaptive reason overclaim
   - UI says why a battle was queued, not that it is the best or most important battle.

## Tests

### Payload Tests

- `test_rating_standings_reads_provisional_latest_without_live_scan`
- `test_rating_standings_paginates_rows`
- `test_recent_battles_payload_reads_small_artifact_or_progress_only`
- `test_checkpoint_payload_uses_per_checkpoint_index`
- `test_checkpoint_payload_server_sorts_then_pages`
- `test_checkpoint_payload_does_not_read_global_million_row_index`
- `test_battle_payload_returns_header_without_all_games`
- `test_battle_payload_pages_games`
- `test_battle_payload_prefers_sample_gif_refs_for_samples`
- `test_battle_payload_cache_key_includes_game_page`

### Render Tests

- `test_render_page_fast_shell_with_progress_rankings_recent`
- `test_render_progress_panel_marks_stale_data`
- `test_render_rankings_panel_labels_provisional_not_final`
- `test_render_checkpoint_battles_shows_server_sort_state`
- `test_render_battle_detail_shows_gif_unavailable`
- `test_render_schedule_reason_uses_plain_copy`
- `test_render_does_not_show_raw_pair_key`
- `test_dropdown_js_clears_checkpoint_battle_offsets`

### Route Tests

- `test_index_uses_default_limit_not_max_limit`
- `test_auto_refresh_routes_do_not_force_volume_reload`
- `test_fresh_true_is_explicit_only`
- `test_api_payloads_include_freshness_and_source`
- `test_dynamic_routes_keep_no_store_headers`

### Cache Tests

- `test_progress_cache_short_ttl`
- `test_checkpoint_battle_cache_key_includes_index_stat_sort_page`
- `test_battle_detail_cache_key_includes_game_limit_offset`
- `test_gif_byte_cache_uses_etag_and_item_limit`
- `test_volume_reload_clears_payload_cache`

### JS Contract Tests

- Rendered JS contains polling for progress, standings, and recent/checkpoint data.
- Rendered JS does not add `fresh=true` automatically.
- Rendered JS has a single in-flight request guard per polling stream.
- Rendered JS preserves scroll state before row replacement.

## Minimal Refactor Order

1. Extract CSS/JS constants with exact output preserved.
2. Split `_render_page` into pure section renderers.
3. Add freshness/source metadata to existing payloads.
4. Add `game_limit` and `game_offset` to battle payloads and route.
5. Add per-checkpoint battle index writer/reader.
6. Switch checkpoint payload to per-checkpoint index.
7. Add recent battles payload/route.
8. Add auto-refresh for standings and recent/checkpoint panels.
9. Add cache key tightening and total GIF byte cache cap.

Each step should land with tests before the next step. The risky behavioral change is step 6; everything before it can mostly be structure and contract hardening.

## Smallest Per-Checkpoint Battle Index

Context after adaptive smoke: the next website scale blocker is policy/checkpoint click. Today `_review_checkpoint_payload` calls `_list_battle_index(... limit=1_000_000, checkpoint_id=...)`, then sorts and slices the filtered rows. `_list_battle_index` reads the global `battle_index.json`, filters every row by checkpoint, sorts every matching row, and returns a page. That is fine for the current tiny run and becomes the wrong click-path shape for many checkpoints or repeated adaptive rounds.

### Path Layout

Add one helper in the tournament artifact module:

- `checkpoint_battle_index_ref(tournament_id, checkpoint_id)`

Use a path under the existing tournament root:

```text
tournaments/curvytron/{tournament_id}/checkpoint_battle_indexes/{checkpoint_id}.json
```

This is intentionally not nested by rating run for V0. Battle rows already carry `rating_run_id`, `round_id`, and `round_index`; one checkpoint page can show all known battle evidence for that checkpoint across adaptive rounds. If the website later needs per-rating-run isolation for old visible tournaments, add an optional second helper:

```text
tournaments/curvytron/{tournament_id}/ratings/{rating_run_id}/checkpoint_battle_indexes/{checkpoint_id}.json
```

V0 should avoid that unless a concrete collision appears.

### Payload Shape

Schema id:

```text
curvyzero_curvytron_checkpoint_battle_index/v0
```

Top-level fields:

- `schema_id`
- `tournament_id`
- `checkpoint_id`
- `updated_at`
- `updated_ts`
- `total`
- `rows`

Rows should be compact but complete enough for the checkpoint battle table without live enrichment:

- `tournament_id`
- `rating_run_id`
- `round_id`
- `round_index`
- `battle_id`
- `pair_index`
- `pair_key`
- `schedule_reason`
- `schedule`
- `checkpoint_id`
- `opponent_checkpoint_id`
- `opponent_label`
- `opponent_seat`
- `players`
- `checkpoint_ids`
- `tally`
- `ok`
- `summary_ref`
- `first_gif_ref`
- `sample_gif_refs`
- `shard_summary_refs`
- `shard_summary_ref_count`
- `updated_at`
- `updated_ts`

Do not store rank in this artifact. Rank is snapshot-dependent and should be joined from the currently selected `latest` or `provisional_latest` during payload building.

### Write Timing

Smallest write path:

1. Keep `_write_battle_index(tournament_id, pair_results, mount=...)` as the single entry point.
2. After it merges and writes global `battle_index.json`, also fan out the just-merged global rows into per-checkpoint index files.
3. For each row, derive checkpoint-specific rows for both players.
4. Merge with the existing per-checkpoint index, keyed by `battle_id`.
5. Sort each checkpoint index by `updated_ts desc, battle_id asc`.
6. Write only checkpoint files touched by the incoming `pair_results`.

This keeps all current callers working:

- final rating artifacts already call `_write_battle_index`
- provisional rating artifacts already call `_write_battle_index`
- completed tournament/battle reducers that call `_write_battle_index` automatically publish checkpoint indexes

Avoid a separate background migration in V0. If an old tournament lacks per-checkpoint indexes, use the fallback below.

### Read Fallback

Add:

- `_list_checkpoint_battle_index(mount, tournament_id, checkpoint_id, limit, offset, sort="updated", direction="desc")`

Read order:

1. Try the per-checkpoint index.
2. If present, sort/page that small file and return `source="checkpoint_battle_index"`.
3. If missing, fall back to existing `_list_battle_index(... checkpoint_id=...)`, but cap the fallback to the current behavior only as a compatibility bridge.
4. If global index is missing, keep existing live-shard / `live_pair_results` fallback.

Then switch `_review_checkpoint_payload` to call `_list_checkpoint_battle_index` first.

Important V0 constraint: if source is `checkpoint_battle_index`, do not call `_enrich_battle_row_from_live_shards` per row. The index row must already include `tally`, `first_gif_ref`, `sample_gif_refs`, and refs needed for the table.

### Sort/Paging

Server-side sort options for V0:

- `updated`
- `avg_steps`
- `failures`
- `games`

`opponent_rank` can wait unless rank is joined before sorting. If implemented, build `rank_by_checkpoint` from the selected snapshot, annotate the small checkpoint rows, sort, then page.

Return standard paging:

- `total`
- `limit`
- `offset`
- `has_older`
- `has_newer`
- `sort`
- `direction`
- `source`

### Minimal Function Changes

Likely functions to touch:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`
  - add `checkpoint_battle_index_ref`
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
  - extend `_compact_battle_index_row` to preserve `pair_key`, `schedule_reason`, and `schedule`
  - add a checkpoint-specific row helper
  - extend `_write_battle_index` to write touched checkpoint indexes
  - add `_list_checkpoint_battle_index`
  - change `_review_checkpoint_payload` to use `_list_checkpoint_battle_index`

No route shape needs to change for the first pass.

### Tests

Writer tests:

- `test_write_battle_index_writes_per_checkpoint_indexes`
- `test_checkpoint_battle_index_has_one_row_per_checkpoint_per_battle`
- `test_checkpoint_battle_index_merges_existing_rows_by_battle_id`
- `test_checkpoint_battle_index_preserves_schedule_reason_pair_key_and_gif_refs`
- `test_checkpoint_battle_index_writes_only_touched_checkpoint_files`

Reader tests:

- `test_list_checkpoint_battle_index_pages_without_reading_global_index`
- `test_list_checkpoint_battle_index_sorts_updated_desc_by_default`
- `test_list_checkpoint_battle_index_sorts_failures_and_avg_steps`
- `test_list_checkpoint_battle_index_falls_back_to_global_index_when_missing`
- `test_list_checkpoint_battle_index_reports_source`

Review payload tests:

- `test_review_checkpoint_payload_uses_checkpoint_index_first`
- `test_review_checkpoint_payload_does_not_call_global_million_row_index_when_checkpoint_index_exists`
- `test_review_checkpoint_payload_does_not_enrich_checkpoint_index_rows_from_live_shards`
- `test_review_checkpoint_payload_joins_current_opponent_rank_after_read`
- `test_review_checkpoint_payload_keeps_existing_live_fallback_for_old_tournaments`

UI/API regression tests:

- `test_api_review_checkpoint_returns_sort_direction_and_source`
- `test_render_checkpoint_battles_does_not_show_raw_pair_key`
- `test_render_checkpoint_battles_shows_plain_schedule_reason`
- `test_click_checkpoint_path_is_bounded_with_many_global_rows`

## Current Per-Checkpoint Index Review

Current local code writes checkpoint battle indexes at:

```text
tournaments/curvytron/{tournament_id}/checkpoints/{checkpoint_id}/battle_index.json
```

That path is good for V0. It sits under the tournament, matches the policy drilldown mental model, and avoids introducing rating-run nesting before the website proves it needs it.

### Is It Enough For Fast Policy -> Battle Drilldown?

Mostly yes, with one important cleanup.

The read path now checks the checkpoint index first inside `_list_battle_index` when `checkpoint_id` is present. If the file exists, the website no longer reads and filters the global `battle_index.json` for that checkpoint. This fixes the largest click-path scale problem: selecting a policy should be bounded by that policy's row count instead of the whole tournament's battle count.

Remaining issue: `_review_checkpoint_payload` still asks `_list_battle_index` for `limit=1_000_000`, then sorts/slices after joining rank data. With a per-checkpoint file this is no longer a whole-tournament scan, so it is acceptable for the immediate adaptive smoke. It can still become expensive for a single heavily sampled checkpoint after many adaptive rounds.

More urgent: `_review_checkpoint_payload` currently includes `checkpoint_battle_index` in the branch that calls `_enrich_battle_row_from_live_shards` on page rows. That keeps a per-row artifact read in the click path. For V0, checkpoint index rows should already have `tally`, `first_gif_ref`, `sample_gif_refs`, and refs needed by the table; they should not need live enrichment on click.

### Next Smallest Website Cleanup

1. Remove `checkpoint_battle_index` from the live-shard enrichment branch.
   - Keep enrichment for legacy `battle_index`, `checkpoint_round_input`, and `live_shard_tallies`.
   - This is the smallest direct click-path speedup.

2. Make the checkpoint index row self-sufficient.
   - Preserve `pair_key`, `schedule_reason`, and `schedule` in `_compact_battle_index_row`.
   - Preserve enough GIF fields for the table: `first_gif_ref`, `sample_gif_refs`, `summary_ref`.
   - Avoid showing raw `pair_key` in UI.

3. Add a dedicated source assertion.
   - When `_review_checkpoint_payload` returns `source="checkpoint_battle_index"`, test that `_enrich_battle_row_from_live_shards` was not called.

4. Keep the current fallback.
   - If checkpoint index is missing, fall back to global battle index/live shard paths for old tournaments.
   - Do not remove compatibility yet.

5. Defer bigger paging/sorting.
   - Do not redesign routes yet.
   - Later, add `sort`/`direction` params and pass `limit`/`offset` into a checkpoint-specific reader after rank-join sorting is settled.

### Focused Tests To Add Next

- `test_review_checkpoint_payload_checkpoint_index_does_not_live_enrich`
- `test_checkpoint_battle_index_rows_preserve_schedule_reason_pair_key`
- `test_checkpoint_battle_index_rows_preserve_gif_sample_refs`
- `test_review_checkpoint_payload_uses_checkpoint_index_before_global_index`
- `test_review_checkpoint_payload_falls_back_to_global_index_when_checkpoint_index_missing`
- `test_policy_click_with_large_global_index_reads_only_checkpoint_index`

### Not Needed Yet

- Do not split checkpoint indexes by rating run yet.
- Do not add a separate checkpoint-specific route yet.
- Do not replace the one-page flow.
- Do not build a client-side data grid. The current table is fine once the server click path is bounded.

## Main-Thread Implementation Note

The first per-checkpoint index cut landed with the same behavior but a different
path from the draft above:

```text
tournaments/curvytron/<tournament_id>/checkpoints/<checkpoint_id>/battle_index.json
```

Reason: this keeps the final filename as `battle_index.json`, which already
passes the tournament artifact ref validator. A path like
`checkpoint_battle_indexes/<checkpoint_id>.json` would require loosening the
filename allow-list or adding a new allowed dynamic filename pattern.

Current implemented contract:

- `_write_battle_index(...)` writes the global index and per-checkpoint indexes.
- `_list_battle_index(..., checkpoint_id=...)` reads the per-checkpoint index
  first and returns `source="checkpoint_battle_index"` when present.
- Existing global-index and live-shard fallbacks stay for old artifacts.
- Focused tests pass with `85 passed, 10 skipped`.
