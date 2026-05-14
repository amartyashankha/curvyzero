# Checkpoint Tournament Website Performance Plan

Date: 2026-05-13
Scope: performance exploration only. No production code changes.

## Files Read

- Tournament website and Modal app:
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- Older training GIF browser:
  `src/curvyzero/infra/modal/curvytron_gif_browser.py`
- Supporting artifact contracts and tests:
  `src/curvyzero/tournament/curvytron/contracts.py`,
  `tests/test_curvytron_checkpoint_tournament.py`,
  `tests/test_curvytron_gif_browser.py`

## Current Shape

The tournament browser already has several good pieces:

- dynamic pages use no-store headers;
- page and progress Volume reloads are throttled;
- GIF responses use ETag, immutable browser cache headers, and an in-process byte cache;
- progress polling no longer forces a Volume reload by default;
- per-checkpoint battle indexes exist, and the checkpoint-index path avoids per-row live-shard enrichment.

The remaining problem is that several route payloads are still built by doing too much synchronous filesystem and JSON work at request time, while the browser only refreshes the progress panel. This makes the site feel slow, empty, or stale during live runs with many battles and GIFs.

## Top Bottlenecks

1. The main page is a serial mega-route.
   `/` lists tournaments, lists rating runs, reads snapshots and progress, optionally loads checkpoint battles, optionally loads battle detail, then renders all HTML. A checkpoint+selected-battle URL can trigger repeated listing, index reads, sorting, game expansion, and GIF sample rendering before the first byte is useful.

2. Checkpoint drilldown is bounded by the checkpoint index only when that file exists, but it still asks for up to `1_000_000` rows, sorts all matching rows, then slices. If a per-checkpoint index is missing, fallback can read/filter the global battle index.

3. Battle detail expands too much.
   `_review_battle_payload` reads all game summaries it can find through embedded games, game summary refs, shard summaries, or a games directory scan. There is no `game_limit` / `game_offset`, so battle pages grow with games per battle.

4. Progress is cheap-ish but can be stale.
   `/api/rating-progress` now reads `progress.json` and merges `provisional_latest.json`, but it still resolves latest via tournament/rating-run listing on every poll. It depends on a background progress refresh spawn for running runs. When progress becomes complete before final rankings are visible, the JS reloads once and stores a session key, so it can remain empty after that one unlucky reload.

5. GIF handling is mostly right, but full GIFs are still heavy UI assets.
   Battle detail can place many full GIFs on the page. Tournament GIF URLs do not include the older browser's `v=<mtime>-<size>` cache-busting query param, and the byte cache has per-item limits but no total byte/item cap.

## Older GIF Browser Patterns To Copy

- Use explicit `fresh=1` for operator refresh, not for normal polling.
- Add a tiny head/token endpoint before fetching large rows. The old browser polls `/api/head`, compares a head token, then fetches `/api/summaries` only when something changed.
- Version GIF URLs with mtime and size in the query string, while still serving ETag + immutable cache headers.
- Page selected-run results by stopping after `offset + limit + 1` matches, and return `total_rows_exact=false` instead of counting the world.
- Use short TTL listing caches keyed by route inputs, clone cached mutable values, and clear listing caches on successful Volume reload.
- Prefer known artifact globs and marker files over recursive scans.

## Simple Plan

### Landed First Cut

- Battle detail accepts `game_limit` and `game_offset`.
- Battle detail payloads report total game count, returned row count, and
  previous/next paging flags.
- Summary-level `sample_gif_refs` are preferred before game-row GIF fallback.
- Older battle summaries that lack `sample_gif_refs` recover GIF samples by
  reading a bounded candidate set of `game_summary_refs`.
- Normal checkpoint/battle click paths no longer request `limit=1_000_000`.
- Focused tournament website tests and compile checks passed locally.

Remaining issue: this is still a request-time JSON reader, not the final
artifact-index design. It is a safer bounded path, not the end state.

## Next Plan

1. Make small artifacts the web source of truth.
   Keep `progress.json`, `latest.json`, `provisional_latest.json`, and checkpoint battle indexes. Add or formalize `recent_battles.json` for a fixed-size live sample, and add a compact per-battle `games_index.json` or paged game-index artifacts. Battle detail should read a header, samples, and one game page without scanning all game directories.

2. Tighten payload caches.
   Cache `list_tournaments` and `list_rating_runs` by directory/stat tokens plus a short TTL. Cache rankings by rating snapshot/provisional file stat. Cache checkpoint battle pages by checkpoint-index stat, selected snapshot stat, sort, limit, and offset. Cache battle header separately from game pages. Add a total-size LRU cap for GIF/JSON byte cache.

3. Page everything that can grow.
   Initial HTML should use smaller defaults: rankings around 100 rows, checkpoint battles around 50-100 rows, battle games around 50 rows, and GIF samples around 3-5. Remove `limit=1_000_000` from request paths except as a compatibility fallback guarded by tests and source metadata.

4. Lazy-load panels from APIs.
   Let the first HTML render as a fast shell with progress and current selection. Fetch rankings, checkpoint battles, recent battles, and battle detail as separate JSON panels. Preserve selected rows, scroll positions, and client sort state. Add token/head endpoints for progress, rankings, checkpoint battles, and battle detail so polling can skip unchanged payloads.

5. Fix stale UI behavior.
   Do not rely on one full-page reload after `progress.status === "complete"`. Keep polling a standings/rankings token until `ratings_ref` or a non-empty provisional snapshot is visible, then patch the rankings panel. Expose `updated_at`, `updated_ts`, `age_seconds`, `source`, and `is_stale` in every dynamic payload so old Modal Volume views are labeled as old instead of looking fresh.

6. Keep GIFs cheap.
   Add version query params to tournament GIF links using stat mtime/size. Keep lazy loading and immutable headers. Consider still thumbnails or low-frame previews for battle tables/details, with full GIFs opened on demand. Avoid force-reloading the whole Volume on every GIF miss unless explicitly requested.

7. Keep Volume reloads boring.
   Keep normal polling reload-free. Keep `fresh=true` explicit. Borrow the older browser's non-blocking reload lock and soft handling for "open files preventing reload" if browser concurrency rises above one. Track last reload attempt and last successful reload separately so a failed reload does not silently age the view without clear metadata.

## Avoid

- Whole-tournament scans during normal page loads or row clicks.
- Recomputing provisional ratings from shard summaries in web requests.
- Full game expansion before rendering a battle header.
- Client-side sort controls that imply the full result set was sorted when only the visible page was sorted.
- Full-page reload loops as the only way rankings appear.
- More full-size GIFs on the first viewport.

## Suggested Test Targets

- Progress API reads only committed progress/provisional artifacts in the normal path.
- Checkpoint route does not call global battle index when a checkpoint index exists.
- Checkpoint route does not request `1_000_000` rows for the normal page.
- Battle detail can return header + sample GIFs without reading every game summary.
- Battle detail supports `game_limit` and `game_offset`.
- GIF links include a version token derived from stat mtime/size.
- Auto-refresh keeps retrying rankings after complete progress until ratings are visible.
- All dynamic payloads include freshness/source metadata.
