# Checkpoint Tournament Website UX Red Team

Date: 2026-05-13
Scope: operator UX and request-shape critique for the CurvyTron tournament dashboard.

Inspected:

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- `src/curvyzero/infra/modal/curvytron_gif_browser.py`
- `src/curvyzero/tournament/curvytron/contracts.py`
- `tests/test_curvytron_checkpoint_tournament.py`
- `tests/test_curvytron_gif_browser.py`
- `docs/working/training/checkpoint_tournament_refactor_website_lorentz_2026-05-13.md`
- related checkpoint tournament critique/scheduler docs

No production code was edited.

## Operator Read

The current website has the right product shape: one page with progress,
ratings, policy drilldown, battle drilldown, and GIF samples. The risk is that
the page still behaves like a small artifact browser in several click paths.
During a live tournament, the operator needs fast first paint, clear freshness,
bounded clicks, and enough auto-refresh to trust that the page is alive.

The strongest existing pattern to copy is the older GIF browser:

- default page is small (`DEFAULT_LIMIT = 8`);
- GIF cards use `loading="lazy"`, fixed image dimensions, and only the first
  preview is eager/high priority;
- filter/navigation actions disable controls and show a loading state;
- auto-refresh is opt-in/limited to offset 0 and polls a tiny `/api/head`
  token before fetching the first page.

The tournament page has pieces of this, but not the same contract discipline
yet.

## First Load

Current risks:

- `/` defaults `limit` to `MAX_LIMIT` for the initial HTML. That can render up
  to 500 ranking/battle rows before the operator has asked for depth.
- The initial page reads tournaments, rating runs, best rating snapshot, and
  live progress. Live progress is cheap-ish (`pair_only=True`), but it still
  derives current progress rather than only reading the last tiny progress
  artifact.
- Only the progress panel auto-updates. If provisional standings or recent
  battles change, the operator sees a moving progress bar beside stale tables
  until reload.

Desired contract:

- First paint should be a shell plus progress, top 50 standings, and either a
  small recent-battles list or "waiting for first battles".
- Initial HTML should not request `MAX_LIMIT` by default.
- Every panel should show source and age: final latest, provisional latest,
  progress artifact, checkpoint index, battle summary, or fallback scan.
- If data is older than a small threshold, the panel should say stale rather
  than merely showing an old timestamp.

## Refresh

Current risks:

- Progress polling is good in spirit: one in-flight guard, 10s visible cadence,
  60s hidden/backoff, and no automatic `fresh=true`.
- When progress reaches complete and no ratings are visible, JS forces one full
  reload with `fresh=true`. This is reasonable, but it is currently the only
  standings refresh path.
- `/api/rating-standings` and `/api/review/rankings` exist, but the rendered
  page does not use them for lightweight table updates.

Desired contract:

- Poll `/api/rating-progress` every 10-15s visible and 60s hidden.
- Poll standings only when the progress payload's rating/provisional ref or
  updated timestamp changes.
- Poll recent/checkpoint battle lists only on offset 0, and only when selected
  tournament/rating/checkpoint still matches.
- Preserve scroll, selected checkpoint, selected battle, sort, and paging while
  replacing rows.

## Selecting Tournament Or Rating Run

Current state:

- Tournament/rating changes clear `checkpoint_id`, `battle_id`, `fresh`, and
  hash. Controls are disabled before navigation. This is good.

Remaining gaps:

- There is no visible loading state beyond disabled controls.
- The page does not distinguish "rating run has no final rankings yet" from
  "rating run is broken" unless progress happens to be readable.

Desired contract:

- Picker changes should show a small "loading selected run" state.
- If only `provisional_latest.json` exists, render live rankings immediately
  with "not final" and completed/total evidence.
- If neither progress nor standings exist, show "no artifacts visible yet" with
  last Volume refresh result.

## Clicking A Policy

Current risks:

- `_review_checkpoint_payload` asks `_list_battle_index(... limit=1_000_000,
  offset=0, checkpoint_id=...)`, then sorts and slices locally.
- A per-checkpoint index now exists and `_list_battle_index` will prefer it, but
  the caller still advertises and caches the dangerous request shape. If the
  checkpoint index is missing, the fallback reads the global index at huge
  limit and can then enrich rows from live shards.
- The API route default for checkpoint review is still `MAX_LIMIT`.
- Browser sorting only sorts the visible set. That is fine if clearly local,
  but misleading if the operator expects all battles to be sorted.

Desired contract:

- Policy click should call a checkpoint-index reader with explicit
  `limit=50`, `offset=0`, `sort`, and `direction`.
- If the per-checkpoint index is missing, return a bounded compatibility
  fallback and label the source as degraded.
- Do not enrich each visible checkpoint row from live shards unless explicitly
  requested; the checkpoint index row should carry table-ready tally/GIF refs.
- Add next/previous controls for checkpoint battles.
- Show schedule reason as coach copy, not raw priority or pair key.

## Clicking A Battle

Current risks:

- `_review_battle_payload` has no `game_limit` or `game_offset`.
- `_read_game_summary_refs` reads embedded games, all listed
  `game_summary_refs`, all shard games, and then scans `games/*/summary.json`
  if counts do not look complete.
- GIF samples are selected after the full game list is read. That defeats the
  intended "header and samples first" feel on large battles.
- If the battle dir is not found, the payload may scan the global battle index
  with `limit=1_000_000` to find one row.

Desired contract:

- Battle click returns header plus `sample_gif_refs` first, then first page of
  games (`game_limit=50`, `game_offset=0`).
- Prefer `battle.json` and direct battle dir. Avoid global index scans for a
  known `battle_id` unless there is no direct artifact.
- Cache key must include game paging once added.
- Game rows need next/previous paging; GIF sample count should stay small.
- Broken or missing GIFs should render "GIF unavailable" while JSON/game rows
  remain usable.

## Loading GIFs

Current strengths:

- Tournament GIF route validates refs, uses ETag/cache headers, immutable
  browser caching, and a per-item byte cache.
- GIF images in battle detail use `loading="lazy"` and `decoding="async"`.

Current risks:

- All sample GIF cards use `src` immediately once the battle detail HTML is
  rendered. With many large GIFs, the network can become the bottleneck even if
  image loading is lazy.
- There is no total byte cap on the server GIF cache, only a per-item max.
- Missing GIFs return 404 and the card remains an image/link failure rather
  than a deliberate unavailable state.
- There are no still thumbnails or downsampled previews, so opening a battle can
  imply multiple full GIF transfers.

Desired contract:

- Render only a small sample set by default, ideally 3-5 GIFs.
- Consider still thumbnails or preview GIFs for battle lists; load full GIF only
  when the operator opens it or expands samples.
- Add JS `error` handling that swaps broken images to "GIF unavailable".
- Add total byte/item count limits to `_WEB_CACHE` before increasing samples.

## API Payload Shape

The existing routes mostly expose the right objects, but they are inconsistent
as operator contracts.

Needed on every dynamic payload:

- selected tournament id;
- selected rating run id when relevant;
- source artifact/ref;
- updated timestamp and age/stale flag;
- limit, offset, total, has_older, has_newer for lists;
- volume reload error;
- degraded/fallback source label when scanning or live derivation was used.

Small payload targets:

- `/api/rating-progress`: only progress/freshness/recent battle token.
- `/api/rating-standings`: paged top rows plus snapshot ref.
- `/api/recent-battles`: fixed small list from progress or tiny artifact.
- `/api/review/checkpoint`: paged checkpoint-index rows.
- `/api/review/battle`: header, GIF samples, paged game rows.

## Priority Fix Queue

1. Change initial/index and review defaults from `MAX_LIMIT` to a small first
   page, probably 50.
2. Add `game_limit`/`game_offset` to battle payload and route.
3. Make checkpoint payload call a per-checkpoint reader with server sort/page,
   not `limit=1_000_000`.
4. Add freshness/source/age metadata to all rendered panels and API payloads.
5. Add standings/recent-battle light polling keyed by progress/snapshot tokens.
6. Add visible loading/disabled states for policy and battle clicks.
7. Add broken-GIF UI fallback plus total GIF byte cache cap.
8. Port the older GIF browser's tiny-head-token refresh pattern for recent
   battles and GIF samples.

## Tests To Add

- First load renders bounded rows and does not request `MAX_LIMIT`.
- Provisional-only fixture renders live rankings with "not final" copy.
- Progress poll does not force `fresh=true`.
- Standings update only fetches when progress/snapshot token changes.
- Checkpoint click uses checkpoint index and bounded limit.
- Missing checkpoint index reports degraded bounded fallback.
- Battle payload returns header before reading all game summaries.
- Battle payload pages games and cache key includes game page.
- Battle detail prefers `sample_gif_refs` for samples.
- Missing GIF renders an unavailable state without hiding JSON links.
- Auto-refresh preserves scroll, selected rows, sort, and pagination.
