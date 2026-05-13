# Checkpoint Tournament Goal And Plan, 2026-05-13

## Goal

Build a small CurvyTron checkpoint tournament lane. It should compare policies
from saved checkpoints by making them play the real two-player game against each
other.

The first score is simple: who dies first.

The next score is a small Elo ladder. That is not the first thing to trust. The
first thing to trust is the raw battle runner.

## Product Shape

- One Modal app for the whole tournament lane.
- One game function is the high-parallel unit for exact one-game fan-out.
- One game-shard function is the scale unit for larger runs. It runs several
  seeded games for one checkpoint pair, reuses the two loaded policies, and
  commits once for the shard.
- Pair, tournament, and rating functions are orchestration layers over
  game-level `.map(...)` work.
- One function that runs many pairs for a tournament.
- One basic website that lists tournaments, battles, scores, and sample GIFs.
- Checkpoints are read from the existing `curvyzero-runs` Volume.
- Tournament artifacts live in the separate v2 Modal Volume
  `curvyzero-curvytron-tournaments` under
  `tournaments/curvytron/...`.
- The target runtime shape is simple: a big tournament should run close to the
  speed of one shard, plus Modal autoscale and aggregation overhead. If a design
  makes pair orchestration the bottleneck, it is the wrong shape.
- The real target is all-pairs at 200-300 checkpoints. At 300 checkpoints,
  unordered no-self all-pairs is 44,850 battles. At 50 games per battle that is
  2,242,500 games. Any parent process that holds one row per game is already the
  wrong shape for that target.
- Modal autoscaling is not instant. Big fan-outs can queue while containers warm
  up, and queue delay plus cold starts can push shards into timeouts. Huge runs
  need idempotent shard refs, retries/backoff, cheap progress checks, and a
  resumable reducer instead of assuming every shard appears immediately.
- Low-level game and game-shard workers now use Modal retries with backoff.
  Parent rating/orchestration functions do not, because retrying the parent can
  duplicate large fan-outs. Retry the idempotent lowest-level unit first.

## Product Contract

- A game has exactly two checkpoint policies and one shared two-player env.
- The default score is from env terminal info: first dead player loses.
- A battle is N independent seeded games for the same two checkpoints.
- A shard is a small batch of games for one battle. It is an implementation
  detail, not a scoring concept.
- A tournament is a set of checkpoint pairs and battle jobs.
- Game summaries are immutable facts.
- Battle summaries and standings can be recomputed from game summaries.
- Rating/Elo snapshots are derived state and can be thrown away.

## Implementation Plan

1. Add pure tournament helpers under `src/curvyzero/tournament/`. Done.
2. Add one Modal app module under `src/curvyzero/infra/modal/`. Done.
3. Reuse existing checkpoint loading and LightZero policy action code where it is
   safer than writing our own. Done for V0.
4. Use `VectorMultiplayerEnv` directly for real checkpoint-vs-checkpoint games.
   Done.
5. Use the existing source-state render path for model observations and GIFs.
   Done.
6. Add focused tests that do not need Modal or a real checkpoint. Done.
7. Deploy the Modal app. Done once; redeploy after every code change.
8. Run a tiny remote game/pair with real checkpoint refs. Done.
9. Check the website can list the tournament and serve GIF/JSON artifacts.
   Done.
10. Add Elo loop after raw battle artifacts are reliable. Done for the first
    batch-Elo version.
11. Add sharded game workers for larger checkpoint sets. Done and remote-smoked.

## Next Remote Smoke

Use two existing checkpoint refs and run the sharded path:

`uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode rating --tournament-id arena-rating-shard-smoke-YYYYMMDD --rating-run-id elo-shard-smoke --run-id-prefix survivaldiag-v1b-20260513h --max-runs 2 --expected-checkpoint-count 2 --round-count 1 --games-per-pair 4 --games-per-shard 2 --max-steps 8 --num-simulations 1 --wait`

Expected result:

- one tournament marker
- one battle summary
- four game summaries
- two game-shard calls in the work summary
- one GIF if `--save-gif true`
- website lists the tournament after refresh/reload

## Operating Notes

- Keep this separate from training and live GIF plumbing.
- Do not create a Modal app per battle.
- Do not write shared aggregate files from game workers.
- Game workers write only their own immutable game summary and optional GIF.
- Rating rounds should flatten work to the game level instead of nesting a
  large pair map around many inner game maps.
- Pair and tournament functions write aggregate summaries.
- The checkpoint volume is mounted read-only by the tournament game worker.
- The artifact volume is mounted read/write and is the only volume committed by
  tournament workers.
- Use Modal `.map(..., order_outputs=False)` for parallel work when waiting for
  all child results is fine.
- For rating and full tournaments, flatten work to one global game map when
  possible. The one-game function is the high-parallel unit.
- For larger tournaments, use `--games-per-shard N`. Sharding keeps pair work
  parallel while cutting Modal calls, eval-mode policy loads, and Volume commits
  by about `N`.
- Keep `reuse_policies_per_shard` on by default. If someone explicitly wants
  each game to reload policy objects, they can turn it off.
- Policy reuse is enabled only for eval-mode shards. Collect-mode shards fall
  back to per-game policy loading so per-game stochastic seeds are not silently
  changed.
- Rating games with GIFs and frame dumps off do not render or keep raw RGB
  frame lists.
- Use `.spawn()` from the local entrypoint so long jobs can run detached.
- Avoid Volume reloads in hot request paths unless the user asks for freshness.
- Rating GIFs should default to off. Save only a small explicit sample for
  human inspection.
- Keep `ratings/<rating_run_id>/latest.json` slim enough for fast website
  reads. Put bulky round details in round files.
- Keep a battle index or other small listing artifact so the website does not
  scan every game file on each request.
- Keep `ratings/<rating_run_id>/progress.json` as a small progress artifact.
  The website reads it directly. It must not scan all game summaries on page
  load.
- For sharded rating runs, reduce shard tallies into pair summaries. Do not
  carry every compact game row through the parent just to compute Elo.
- Keep parent aggregation at pair scale where possible: one row per battle is
  acceptable for all-pairs, one row per game is not.
- Use `--mode progress` for a cheap progress refresh. By default it counts
  visible battle directories. Use `--progress-read-summaries` only when exact
  game-summary counts are worth the slower scan.
- Use `--mode reduce` to rebuild `battle.json`, `ratings.json`, and
  `latest.json` from immutable game summaries if the parent reducer fails.
- Discovery should fail clearly when expected checkpoint rows are missing,
  instead of silently rating a partial dataset.

## Done For V0

- A user can pass two checkpoint refs and run N games.
- The pair summary says seat 0 wins, seat 1 wins, draws, failures.
- At least one GIF can be saved for human inspection.
- The website can list tournament rows and link to JSON/GIFs.
- A user can run `--mode rating` to launch one or more batch-Elo rounds.
- Rating snapshots live under
  `tournaments/curvytron/<tournament_id>/ratings/<rating_run_id>/...`.
- The website exposes `/api/ratings` and `/api/rating-standings`.
- The website exposes `/api/rating-progress`.

## Not Done Yet

- Website is basic. It is useful but not the final tournament dashboard.
- Huge 300x300x50 round robin may need sharding to avoid too many tiny commits
  and too many files.
- The 50-run batch was launched and completed from this lane. The next scale
  target is hundreds of checkpoints, where sharding and fewer per-game commits
  may matter.
- The first remote shard smoke passed after deployment.

## Remote Smoke Result

2026-05-13 smoke:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode rating \
  --tournament-id arena-rating-smoke-v1b-20260513a \
  --rating-run-id elo-smoke \
  --checkpoint-refs <v1b-row-001-iteration-20000>,<v1b-row-002-iteration-20000> \
  --round-count 1 \
  --games-per-pair 1 \
  --max-steps 8 \
  --num-simulations 1 \
  --wait
```

Result:

- one game completed
- score was a draw by timeout at 8 steps
- one battle summary was written
- `ratings/elo-smoke/latest.json` was written
- website `/api/ratings`, `/api/rating-standings`, and `/api/battles` returned
  the new tournament data
- downloaded GIF was `704 x 704`

Important checkpoint path note:

- The preserved mirror under `checkpoints/lightzero_resume_state` contains
  resume sidecars such as `iteration_20000.resume_state.pkl`.
- The actual LightZero weight files used by the tournament runner are currently
  under
  `attempts/<attempt_id>/train/lightzero_exp/ckpt/iteration_<n>.pth.tar`.
- The Modal entrypoint can discover these refs with `--mode discover` and
  `--run-id-prefix survivaldiag-v1b-20260513h`.
- An integrated prefix smoke,
  `arena-rating-prefix-smoke-20260513a / elo-smoke`, found 2 of 2 checkpoints,
  ran one rated battle, served ratings through the website API, and produced a
  `704 x 704` GIF.

## Full 50-Run Result

2026-05-13 full rating:

```text
arena-rating-v1b-full50-gpp50-20260513a / elo-full-gpp50
```

Result:

- discovery found 50 latest real LightZero checkpoints from
  `survivaldiag-v1b-20260513h`
- 1,225 unordered checkpoint pairs
- 50 games per pair
- 61,250 games total
- `ratings/elo-full-gpp50/latest.json` written
- 50 rating rows
- 1,225 rated pairs
- battle index total: 1,225
- website `/api/rating-standings` and `/api/battles` read the result
- `latest.json` remains slim and does not include `pair_rating_results`

Scale critique:

- The 50-run game fan-out worked, but that completed run paid one policy load
  per seat per game and one Volume commit per game.
- The exact progress scan over every game summary is slow at this scale.
  Default progress now uses battle directories as a fast estimate and marks the
  result complete when `latest.json` exists for the same round.
- For larger tournaments, use the new shard worker first. A separate sharded
  reducer is still possible later if final aggregation becomes the bottleneck.
- The sharded rating path now has a lean reducer mode: shard workers can return
  only `tally`, `first_gif_ref`, and counts, and the parent writes battle
  summaries without `games`. This keeps Elo and the website working while
  avoiding game-row fan-in.

## Sharded Scale Plan

New local contract:

- `--games-per-shard 1` keeps the old one-game-per-worker behavior.
- `--games-per-shard N` groups N games for the same checkpoint pair into one
  worker call.
- The shard worker loads the two policies once, runs each seeded game, writes
  the same immutable per-game summaries, then commits once.
- If the shard is collect-mode, policy reuse is disabled and the worker reloads
  policies per game.
- Pair summaries, battle index, standings, and Elo snapshots still read the same
  game summaries. Sharding should not change scores.

Scale math:

- 200 checkpoints, unordered no self-pairs, 50 games per pair:
  19,900 pairs and 995,000 games.
- With `--games-per-shard 10`, that is about 99,500 shard calls instead of
  995,000 game calls.
- 300 checkpoints, unordered no self-pairs, 50 games per pair:
  44,850 pairs and 2,242,500 games.
- With `--games-per-shard 10`, that is about 224,250 shard calls.

Current recommendation:

- Use `--games-per-shard 10` for the first 200-checkpoint run.
- Increase to 25 only after the shard smoke and a medium run look healthy.
- Keep GIFs off for rating runs unless the run is explicitly for visual samples.
- Keep `result_detail_mode=shard_tallies` for large rating runs. Use full
  `games` in battle summaries only for small debug runs or explicit visual
  samples.

## Shard Remote Smoke Result

2026-05-13 deployed smoke:

```text
arena-rating-shard-smoke-20260513b / elo-shard-smoke
```

Command shape:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode rating \
  --tournament-id arena-rating-shard-smoke-20260513b \
  --rating-run-id elo-shard-smoke \
  --run-id-prefix survivaldiag-v1b-20260513h \
  --max-runs 2 \
  --expected-checkpoint-count 2 \
  --round-count 1 \
  --games-per-pair 4 \
  --games-per-shard 2 \
  --max-steps 8 \
  --num-simulations 1 \
  --wait
```

Result:

- discovery found 2 of 2 checkpoints
- one pair, four games
- two shard calls
- `work_summary.work_kind` was `shard`
- `work_summary.work_count` was `2`
- shard workers reported `policy_reuse: true`
- game summaries showed `policy_loads[*].preloaded: true`
- no-GIF game summaries showed `frame_count: 0`
- website `/api/rating-progress` and `/api/battles` served the result
- `--mode reduce --wait` rebuilt the rating from committed summaries

## Full 50-Checkpoint Shard Result

2026-05-13 deployed run:

```text
arena-rating-v1b-full50-gpp50-shard10-20260513a / elo-full-gpp50-shard10
```

Command shape:

```text
uv run --extra modal python -B -m modal run --detach -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode rating \
  --tournament-id arena-rating-v1b-full50-gpp50-shard10-20260513a \
  --rating-run-id elo-full-gpp50-shard10 \
  --run-id-prefix survivaldiag-v1b-20260513h \
  --max-runs 50 \
  --expected-checkpoint-count 50 \
  --round-count 1 \
  --games-per-pair 50 \
  --games-per-shard 10
```

Result:

- discovery found 50 of 50 checkpoints
- 1,225 unordered pairs
- 61,250 games
- 6,125 shard calls estimated
- status complete
- 1,225 rated pairs
- top checkpoint at first API check:
  `ckpt-032-train-lightzero_exp_260513_075512-ckpt-ite-ba98e3ff`
- top rating: `1761.12`
- top row games: `2450`
- top row wins/losses/draws: `1595 / 779 / 76`
- website `/api/rating-standings` and `/api/battles` served the result
- progress was refreshed so cheap progress shows `61250 / 61250` games and
  `1225 / 1225` pairs complete without exact summary scanning

## Lean Shard Reducer Update

2026-05-13 local update:

- shard workers now include a per-shard tally in their result
- the rating parent can ask shard workers not to return compact game rows
- the rating parent merges shard tallies into battle summaries
- those battle summaries omit `games` and carry `result_detail_mode:
  shard_tally`
- `ratings/<rating_run_id>/rounds/<round_id>/results.json` can omit
  `pair_results` for the lean path and keep `pair_rating_results` plus
  `pair_summary_refs`
- focused local tests passed: `30 passed, 1 skipped`

This is the important scale change for all-pairs tournaments. It does not
change scoring. It changes how much data the parent has to hold while reducing
results.

Remote validation:

- First remote smoke found a bug after shard work completed: the lean branch
  still printed `len(game_results)`, but `game_results` only exists in the old
  game-list path. Fixed by printing the branch-independent `game_count`.
- Redeployed and reran:
  `arena-rating-lean-shard-smoke-20260513b / elo-lean-shard-smoke`.
- Discovery found 2 checkpoints.
- Work summary: `work_kind=shard`, `work_count=2`, `games_per_shard=2`,
  `parent_result_mode=shard_tallies`.
- Result: one rated pair, four games, no failures, latest rating snapshot
  written.
- Website/API validation passed:
  `/api/review/rankings` returned 2 ranking rows.
- Checkpoint detail used `source=battle_index` and returned the expected one
  battle row.
- Round `results.json` had `result_detail_mode=shard_tallies`,
  `pair_result_count=1`, `pair_rating_results=1`, and no `pair_results` field.

## Review Website And GIF Drilldown Update

2026-05-13 update:

- The review website keeps one page.
- Level 1 is rankings.
- Level 2 is battles for the selected checkpoint.
- Level 3 is games and GIF samples for the selected battle.
- The selected battle is carried in the URL as `battle_id`, so links are
  copyable.
- The battle detail API reads `battle.json`, then `shard_summary_refs`, and only
  falls back to scanning `games/*/summary.json` when the summary refs are
  missing.
- Shard workers write `battles/<battle_id>/shards/<shard_id>/summary.json` with
  compact game rows, tally, and GIF refs.
- Battle summaries carry `shard_summary_refs`, `shard_summary_ref_count`,
  `game_summary_ref_count`, and `sample_gif_refs`.
- The website can show up to 10 sample GIFs for a battle.
- The worker should not save GIFs for every game by default. Use explicit
  visual sample runs with `--save-gif --gif-sample-games-per-pair N`.
- The default sample strategy is `evenly_spaced`, not first-N.

Validated smoke:

```text
arena-rating-review-gif-smoke-20260513a / elo-review-gif-smoke
```

Command shape:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode rating \
  --tournament-id arena-rating-review-gif-smoke-20260513a \
  --rating-run-id elo-review-gif-smoke \
  --run-id-prefix curvy-survive-bonus \
  --max-runs 4 \
  --expected-checkpoint-count 4 \
  --round-count 1 \
  --games-per-pair 12 \
  --games-per-shard 4 \
  --max-steps 96 \
  --num-simulations 1 \
  --save-gif \
  --gif-sample-games-per-pair 5 \
  --gif-sample-strategy evenly_spaced \
  --wait
```

Result:

- discovery found 4 of 4 checkpoints
- 6 unordered battles
- 72 games
- 18 shard calls
- 30 GIFs planned
- parent result mode was `shard_tallies`
- deployed website `/api/review/rankings` returned 4 rows
- selected checkpoint detail returned 3 battles from `battle_index`
- selected battle detail returned 12 games from `shard_summary_refs`
- sample GIF indexes were `0, 3, 6, 8, 11`
- first sample GIF served as `image/gif` and started with `GIF89a`

Cleanup:

- Added `--mode visibility` for tournament browser marker cleanup.
- `hide_except` removes only `show_in_tournament_browser.flag`; it does not
  delete tournament artifacts.
- Browser markers were hidden for old smokes. The visible tournaments after
  cleanup should be:
  `arena-rating-review-gif-smoke-20260513a` and
  `arena-rating-v1b-full50-gpp50-shard10-20260513a`.

## First 211-Checkpoint Rating Launch

2026-05-13 launch:

```text
arena-rating-211-latest-random-20260513a / elo-latest-rand3000-gpp20
```

Command shape:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode rating \
  --tournament-id arena-rating-211-latest-random-20260513a \
  --rating-run-id elo-latest-rand3000-gpp20 \
  --run-id-prefix curvy-survive-bonus \
  --max-runs 211 \
  --expected-checkpoint-count 211 \
  --round-count 1 \
  --games-per-pair 20 \
  --games-per-shard 10 \
  --pairs-per-round 3000 \
  --pair-selection random \
  --gif-sample-games-per-pair 0 \
  --seed 211013
```

Plan estimate:

- discovery found 211 of 211 latest checkpoints
- pair candidates: 22,155 unordered no-self pairs
- sampled pairs this round: 3,000
- games: 60,000
- shard calls: 6,000
- GIFs: off
- function call id: `fc-01KRGZMWF3XB2SGNCH1HPNXWNZ`

This is the first real scale run for the current `curvy-survive-bonus` pool. It
is intentionally sampled, not all-pairs. All-pairs can wait until the sampled
Elo path and progress/reduce path have stayed healthy at this size.

Follow-up:

- The first non-detached launch used tournament id
  `arena-rating-211-latest-random-20260513a` and function call
  `fc-01KRGZMWF3XB2SGNCH1HPNXWNZ`, but an immediate progress check still saw no
  round input.
- Progress now handles that startup state as `waiting_for_round_input` instead
  of throwing.
- Relaunched with `modal run --detach` as
  `arena-rating-211-latest-random-20260513b`.
- Detached function call id:
  `fc-01KRGZW0HS1EGB8E36Q8SC0STE`.
- Progress check at 2026-05-13 15:39 UTC:
  `status=running`, `pair_count=3000`, `game_count=60000`,
  `started_pair_count=46`, `estimated_seen_game_count=920`, and
  `estimated_completion_fraction=0.0153` using the cheap battle-directory count.

Website cleanup at 2026-05-13 16:00 UTC:

- Deployed the tournament browser with running-progress support.
- The browser now lists rating runs from `progress.json` even before
  `latest.json` exists, so a large running arena does not look empty.
- The selected page shows a small progress panel with status, phase, started
  pairs, and estimated games seen.
- Visibility cleanup hid every older tournament browser marker except
  `arena-rating-211-latest-random-20260513b`.
- External validation:
  `/api/tournaments?fresh=true` returned only
  `arena-rating-211-latest-random-20260513b`; the selected page returned 200
  and contained the progress panel; `/api/rating-progress` returned
  `status=running`, `phase=games_running`, `started_pair_count=46`,
  `pair_count=3000`, `estimated_seen_game_count=920`, and `game_count=60000`.

Completion validation at 2026-05-13 16:10 UTC:

- Redeployed again with lightweight in-page progress polling. The page checks
  `/api/rating-progress` every 10 seconds without forcing a Volume reload or a
  full page reload.
- `/api/rating-progress` for
  `arena-rating-211-latest-random-20260513b / elo-latest-rand3000-gpp20`
  reported `status=complete`, `phase=reduced`, `started_pair_count=3000`,
  `pair_count=3000`, `estimated_seen_game_count=60000`, and
  `game_count=60000`.
- `/api/rating-standings` reported 211 rating rows.
- Top row at validation time:
  `ckpt-037-train-lightzero_exp_260513_110607-ckpt-ite-a7ce4d60`,
  rating `1597.65`, games `580`, W-L-D `378-185-17`.
- Checkpoint battle detail loaded through `shard_summary_refs`, with 20 games
  found for the sampled battle.
- This run intentionally saved no GIF samples (`gif_sample_games_per_pair=0`),
  so battle sample GIF count is expected to be 0.

Correction at 2026-05-13 16:05-16:10 EDT:

- The no-GIF 211-checkpoint arena was a bad product artifact for the review
  website. It was purged from the tournament artifact Volume, including the
  failed startup attempt `arena-rating-211-latest-random-20260513a` and the
  completed no-GIF attempt `arena-rating-211-latest-random-20260513b`.
- Relaunched the same sampled rating shape as
  `arena-rating-211-latest-random-gifs-20260513a /
  elo-latest-rand3000-gpp20-gifs3`.
- Launch used `--save-gif --gif-sample-games-per-pair 3
  --gif-sample-strategy evenly_spaced`, so each 20-game battle should save
  games `0`, `10`, and `19` as GIF samples.
- Plan estimate: 211 checkpoints, 3,000 sampled pairs, 60,000 games, 6,000
  shard calls, 9,000 GIFs.
- Function call id: `fc-01KRH1ARDXAWE2PC0HNF2HJDTN`.
- Visibility cleanup now keeps only
  `arena-rating-211-latest-random-gifs-20260513a` visible.
- Direct artifact validation found
  `games/game-000000/game.gif`, `games/game-000010/game.gif`, and
  `games/game-000019/game.gif` for battle pair `000000`.
- The deployed `/gif` endpoint served the first sample as `image/gif` with
  `GIF89a`.
- Website layout patch: rankings and battles are scrollable panels with sticky
  headers, and battle links jump to `#battle-detail`.
- Completion validation:
  `/api/ratings` reported the GIF-enabled run complete with 211 checkpoints,
  3,000 rated pairs, and 60,000 games. A top-checkpoint drilldown loaded a
  battle from `shard_summary_refs`, found 20 games, and exposed 3 GIF samples
  at game indexes `0`, `10`, and `19`.

Render-contract rerun at 2026-05-13 16:55-17:05 UTC:

- The first 50-checkpoint source-render launch was not detached, so the local
  entrypoint stopped before `curvytron_rating_loop` wrote `config.json` or
  round input. Symptom: progress stayed at `waiting_for_round_input`.
- Correct rule: long rating/tournament runs use `modal run --detach` unless
  they are tiny `--wait` smokes.
- A two-checkpoint detached smoke completed with two games and two GIFs:
  `arena-source-render-detach-smoke-20260513b /
  elo-source-render-detach-smoke`.
- Pulled `game-000000` from that smoke. Summary showed `frame_size=704`,
  `gif_trail_render_mode=browser_lines`, `decision_source_frames=12`, and mixed
  policy observation modes: seat 0 `body_circles_fast`, seat 1 `browser_lines`.
  The last GIF frame showed dense curved trails and a visible bonus.
- Relaunched the 50-checkpoint source-render arena with `modal run --detach` as
  `arena-v1b-latest50-source-render-20260513b /
  elo-latest50-gpp20-gifs3`.
- Plan estimate: 50 checkpoints, 1,225 pairs, 24,500 games, 2,450 shard calls,
  and 3,675 GIF samples.
- Detached rating-loop call id: `fc-01KRH4BZ9QRZRJR7SVE571GRVW`.
- Startup validation found `config.json`, round input, battle folders, and the
  first battle's `game-000000/game.gif`.
- Website progress now computes cheap live progress from started battle folders
  for running ratings instead of showing a stale startup progress file.
- Completion validation: `/api/rating-progress` reported `status=complete`,
  `phase=reduced`, `pair_count=1225`, `game_count=24500`,
  `ok_game_count=24500`, and `failed_game_count=0`.
- `/api/rating-standings` returned 50 ranking rows. Top row at validation time:
  `ckpt-041-train-lightzero_exp_260513_075034-ckpt-ite-3dfce01e`, rating
  `1529.35`, games `980`, W-L-D `515-457-8`.
- Drilldown validation for that checkpoint returned 49 battles. A battle detail
  returned 20 games and 3 GIF samples at `game-000000`, `game-000010`, and
  `game-000019`.
- The deployed `/gif` endpoint served a sample as `image/gif` with `GIF89a`,
  `704x704`, cached for one day.
- The detached smoke arena was deleted after validation. The tournament volume
  now keeps only `arena-v1b-latest50-source-render-20260513b`.
