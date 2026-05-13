# CurvyTron Checkpoint Tournament, 2026-05-13

## Grounding Prompt

The user wants basic tooling for checkpoint tournaments before a larger batch of
CurvyTron runs lands. The core unit is simple: given two checkpoints, run N
games, score who dies first, and save artifacts. Everything should run on Modal.
There should be exactly one tournament Modal app with several functions, not a
new app per battle. The lowest-level function should be a small CPU function
that runs one game to completion and can autoscale hard. Higher layers should
run many games for one pair and many pairs for a whole tournament. Battles
should also be able to produce rich-render GIF artifacts. A separate tournament
website should exist, even if the first version is basic.

## North Star

Give the coach and inspector a quick way to compare checkpoints against each
other after a large training batch. The result should answer a plain question:
when checkpoint A plays checkpoint B, who tends to die first?

## Minimal Product

- One reusable game runner: checkpoint A vs checkpoint B, one seeded game.
- One battle runner: checkpoint A vs checkpoint B, many games in parallel.
- One tournament runner: many checkpoint pairs, many battle jobs in parallel.
- Simple scoring: player who dies first loses; other player wins; simultaneous
  death or timeout is a draw.
- Optional rich GIF per game, off by default for huge tournaments.
- Volume artifacts with JSON summaries and GIFs.
- A basic Modal-hosted website to list tournaments and battles.

## Current Design

- Keep this lane separate from training and live checkpoint GIF plumbing.
- Use the same source-state turn-commit visual env used by checkpoint GIFs.
- Load two LightZero checkpoints into two policies.
- At each game step, ask both seat policies for actions and step the shared
  two-player env once.
- Capture human RGB frames from the env when requested.
- Read checkpoints from `curvyzero-runs`.
- Store tournament artifacts in the separate v2 Modal Volume
  `curvyzero-curvytron-tournaments` under `tournaments/curvytron/...`.
- Default GIFs use the full 704 by 704 rich RGB source-state canvas, not the
  64 by 64 grayscale model input.

## Operating Pattern

The main thread should stay focused on orchestration:

1. Re-read the product goal.
2. Check the current code path.
3. Write down evidence, gaps, and hypotheses.
4. Delegate narrow research or validation work when it can run in parallel.
5. Implement the smallest useful thing.
6. Test it locally.
7. Run a remote smoke when the change touches Modal.
8. Update these docs before stopping.

Stopping point rule: do not stop at "the code compiles" if the actual product
goal needs a remote job, a browser, or a real artifact.

## User

Primary user: the coach. They need to compare many checkpoints from a large
training batch and see whether one checkpoint actually beats another in the real
game.

Secondary user: the inspector. They need GIFs and simple battle summaries to
spot weird behavior quickly.

The first UI should not try to explain everything. It should answer:

- Which tournament am I looking at?
- Which checkpoints played?
- Who won more?
- Can I open a sample GIF and JSON?

## Scale Constraint

Treat all-pairs as the default future shape, not a special case. For 200
checkpoints, unordered no-self all-pairs is 19,900 battles. For 300 checkpoints,
it is 44,850 battles. At 50 games per battle, that is up to 2,242,500 games.

The parent process should not gather one object per game for those runs. Shard
workers may run multiple games, but the parent should reduce from shard tallies
into one summary per battle. The review website should read ratings and battle
indexes, not scan game summaries.

Modal autoscaling caveat: high fan-out is necessary, but it is not magic.
Containers may not appear immediately, some shards may sit in queues, and slow
starts can cause timeouts. The large-run pattern needs retries, backoff,
idempotent artifact paths, cheap progress, and resumable reduce.

## Open Questions

- How high Modal autoscaling should be pushed for 300x300x50 scale.
- What retry/backoff settings are safest when Modal autoscale lags under very
  large fan-out.
- Whether round-robin should include both A-vs-B and B-vs-A separately.
- How many GIFs to save for very large tournaments; likely not every game.
- Whether the website should rank checkpoints by Elo/Bradley-Terry later.

## Critique

The first version should not over-model tournament formats. A round-robin pair
matrix plus raw win/loss/draw counts is enough. More ranking math can be added
after the basic runner is proven.

## 2026-05-13 Inspector Update

- The review website should stay one-page:
  rankings at the top, selected checkpoint battles below, selected battle games
  and GIF samples below that.
- Battle detail now reads shard summary files first. This avoids scanning every
  per-game directory on page load.
- Tournament GIF samples default to evenly spaced games within a battle. For a
  12-game battle with 5 samples, the saved samples are games `0, 3, 6, 8, 11`.
- Large rating runs should keep GIFs off unless the run is explicitly a visual
  sample run.
- Old tournament browser clutter is hidden by removing
  `show_in_tournament_browser.flag`. This does not delete the actual tournament
  artifacts.
- Keep latest-checkpoint-per-run as the first real 200+ checkpoint tournament
  target. Historical checkpoints are useful later as anchors, but including
  every checkpoint immediately explodes the pair budget.
- 2026-05-13 16:00 UTC browser cleanup:
  only `arena-rating-211-latest-random-20260513b` should be visible in the
  tournament dropdown. Older arena artifacts remain on disk, but their
  `show_in_tournament_browser.flag` files are hidden.
- The tournament website must show running progress from `progress.json` before
  ratings exist. An empty page for a running arena is a product bug.
- The browser now polls `/api/rating-progress` every 10 seconds for the
  selected arena/rating run. It does not force Volume reloads and does not
  reload the page.
- The first 211-checkpoint sampled rating run completed:
  3,000 random pairs, 60,000 games, 211 rating rows, no GIF samples by design.
- That no-GIF run was purged because the website needs battle GIFs to be useful.
  The replacement live arena is
  `arena-rating-211-latest-random-gifs-20260513a`, with 3 GIF samples per
  20-game battle.
- Current visual-review rule: large rating runs should still avoid every-game
  GIF capture, but they must save a small representative GIF sample per battle
  when the run is meant to be inspected through the tournament website.
- The first and second website hierarchy levels must be bounded scroll panels.
  Do not let rankings or battles expand the whole page.
- Final check for the replacement arena:
  the website can now show rankings, checkpoint battles, and battle GIF samples
  for `arena-rating-211-latest-random-gifs-20260513a`.
