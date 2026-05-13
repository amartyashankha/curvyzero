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

## 2026-05-13 Live Website Patch

- Do not hide rankings until the whole tournament is done. The website now
  reads final `latest.json` if it exists; otherwise it reads the small
  `provisional_latest.json` file written by the background provisional writer.
- Provisional rankings are a bridge. They are shown as live/updating and are
  not written to `latest.json`.
- Slow battle clicks were mainly caused by direct battle pages falling back to
  broad battle-index scans. Direct battle detail now checks `battle.json`, then
  the battle folder, before any wide index scan.
- Web reads now use short in-container caches. Volume reloads clear the caches,
  and reloads are throttled even when Modal reports an open-file/busy error.
- Browser auto-refresh does not force Modal volume reloads. Forced reloads are
  too slow on large tournament volumes and should be reserved for explicit
  operator actions or background artifact refresh jobs.
- Page reloads do force a Volume refresh, because the user expects a browser
  reload to show the latest committed state. Progress polling stays light and
  reads small committed artifacts only.
- GIF cards lazy-load the full GIFs. A separate downsample/preview artifact is
  still an optional performance lane if full GIF bandwidth remains too high.
- Provisional rankings are built by a background Modal function that writes
  `ratings/<run>/provisional_latest.json`. The website reads that small file;
  it should not scan all shard summaries on a click or page load.
- Correctness audit remains open: make the tournament evaluation contract
  explicit so observation mode, environment settings, scoring mode, and
  checkpoint identity cannot drift silently from training.

## 2026-05-13 Latest-212 Relaunch

- Preserved-run manifest:
  `artifacts/local/curvytron_pruning/curvytron_prune_preserve_20260513c.json`.
- Relaunch source set: latest checkpoint from each preserved CurvyTron run.
  Preflight found 212 checkpoints and 0 missing checkpoints.
- Old tournament artifact was deleted from `curvyzero-curvytron-tournaments`:
  `tournaments/curvytron/arena-curvytron-latest212-allpairs-gpp10-gifs3-20260513a`.
- New detached rating tournament:
  `arena-curvytron-latest212-allpairs-gpp10-gifs3-20260513b`.
- Rating run:
  `elo-latest212-allpairs-gpp10-gifs3`.
- Plan size: 22,366 unordered no-self pairs, 223,660 games, 10 games per pair,
  3 GIF samples per pair.
- First progress check saw the round input and early battle directories:
  1,100 pairs started, 0 completed pairs counted yet. That means the launcher
  and artifact path are alive, but results were not yet ready.
- The website loads the new tournament URL and sees its progress file. Standings
  stay empty until `latest.json` or `provisional_latest.json` has rating rows.
- One detached provisional writer was started for this run. Longer term, this
  needs a cadence/trigger so live ratings refresh every few minutes while a big
  tournament is still running.

## 2026-05-13 Odd-Battle Relaunch

- Even `games_per_pair` is now rejected for new pair, rating, and estimate
  specs. Default battle size is 11.
- Reason: a battle should not end with an avoidable tied game count. This does
  not solve draw games inside CurvyTron, but it avoids the obvious even-count
  footgun.
- The even latest-212 run
  `arena-curvytron-latest212-allpairs-gpp10-gifs3-20260513b` was stopped and
  deleted.
- New current run:
  `arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513c /
  elo-latest212-allpairs-gpp11-gifs3`.
- Preflight found 212 latest checkpoints and 0 missing checkpoints.
- Plan size: 22,366 unordered no-self pairs, 246,026 games, 11 games per pair,
  11 games per shard, and 67,098 GIF samples.
- Shard logs confirm the corrected run is active: sampled rows show
  `game_count=11`, `ok=true`, and `failure_count=0`.
- The rating loop now spawns a background provisional-rating loop. The website
  reads `provisional_latest.json` for live standings while the full reducer is
  still running.
- Website validation after deploy:
  `/api/rating-standings` returned 212 provisional rows from
  `live_shard_summaries`; `/api/rating-progress` merged the provisional counts
  and showed 26 completed pairs / 286 completed games at the first check.
- Provisional-loop log confirmed a write for the current run:
  `rating_count=212`, `completed_pair_count=26`, `completed_game_count=286`.

## 2026-05-13 Timestamped Latest-212 Launch

- Source manifest stayed
  `artifacts/local/curvytron_pruning/curvytron_prune_preserve_20260513c.json`;
  it contains 212 preserved run IDs. `/tmp/curvy-preserved-212-run-ids.txt`
  parsed as 211 comma-separated IDs, so the manifest was used instead.
- Focused checkpoint tournament tests passed before launch:
  `55 passed, 1 skipped`.
- Modal estimate found 212 latest checkpoints and 0 missing checkpoints.
- New detached run:
  `arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513-145153 /
  elo-latest212-allpairs-gpp11-gifs3-20260513-145153`.
- Correct detached app/call:
  `ap-pwmiNrqOksG4dsu4EB9JUz`,
  `fc-01KRHB5MC4H5GCB6D9E3ZSYGPH`.
- Plan size matches the odd-battle contract: 22,366 unordered no-self pairs,
  246,026 games, 11 games per pair, 11 games per shard, and 67,098 GIF samples.
- First launch attempt omitted Modal CLI `--detach`; it returned a call id but
  the app stopped with 0 tasks. The corrected launch used `modal run --detach`.
- First live progress probe after the corrected launch reported `status=running`,
  `phase=games_running`, `pair_count=22366`, `game_count=246026`,
  `started_pair_count=19`, and `estimated_seen_game_count=209`.
- The prior current run
  `arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513c` was not purged or
  stopped.

## 2026-05-13 Checkpoint Drilldown Fix

- Symptom: rankings were visible for the latest-212 tournament, but clicking a
  checkpoint could return `source=battle_index_missing` and no battle rows.
- The first fix made the page usable by reading live shard summaries, but it
  was still too slow. A real checkpoint click took 14-21 seconds because it read
  all 211 shard summaries for that checkpoint before returning one page.
- Final fix: checkpoint drilldown now builds the full opponent list from the
  small rating round input, sorts it by opponent rank, pages it, and only then
  reads shard summaries for the visible rows. This makes the click proportional
  to page size, not to the full opponent count.
- Battle detail can also synthesize detail from live shard summaries when
  `battle.json` is not ready yet. That keeps GIF samples visible while the full
  reducer is still running.
- Deployed website validation against
  `arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513c /
  elo-latest212-allpairs-gpp11-gifs3`:
  - standings: 0.566 seconds, 212 rows, `source=live_shard_summaries`
  - checkpoint drilldown: 1.228 seconds, 8 visible rows from 211 scheduled
    battles, `source=checkpoint_round_input`
  - battle detail: 0.157 seconds, 11 games, 3 GIF samples
  - first GIF: `GIF89a`, `704x704`, 10,441 bytes
- The same website also sees the timestamped fresh run
  `arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513-145153 /
  elo-latest212-allpairs-gpp11-gifs3-20260513-145153` with 212 provisional
  rating rows.
- Dropdown fix: tournament and rating selects now navigate immediately. Changing
  tournament clears stale `rating_run_id`, `checkpoint_id`, `battle_id`, and
  `fresh`; changing rating clears stale checkpoint/battle/fresh state.
- Deployed dropdown validation used the real served HTML and a small JavaScript
  harness. It confirmed the change handlers rewrite the URL correctly and
  disable the picker during navigation.
- Direct page validation confirmed both current latest-212 arenas are selectable
  from URL:
  - `arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513c`
  - `arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513-145153`

## 2026-05-13 Action-Selection Recommendation

- Official score tournaments should use one action-selection contract:
  `policy_mode=eval`. That means each checkpoint is scored by the deterministic
  LightZero eval path after MCTS search, with no tournament temperature sampling
  and no epsilon exploration.
- Do not mix eval and collect games in the same Elo/rating pool. If collect-mode
  behavior is worth measuring, run it as a separate diagnostic tournament or a
  separately labeled visual sample set.
- Training-style collect mode is useful for inspection, not for the main score.
  It adds root search noise and samples from the search visit-count distribution
  using temperature. Epsilon only matters when the LightZero policy config enables
  eps-greedy exploration in collect; the upstream MuZero default leaves that off.
- Human GIFs should prefer the same action mode as the score when they are meant
  to explain a rated battle. A second collect-mode GIF is useful only when the
  question is "what did training exploration look like?", and it should be labeled
  separately from the rated result.
- Practical default: keep fair score runs greedy/eval, odd games per pair, fixed
  seeds and environment settings, and record `policy_mode`, temperature, epsilon,
  and simulation count in every artifact. Use collect/temperature only for
  explicitly named diagnostics.
- Evidence: local tournament code defaults normalized pair/rating specs to
  `policy_mode=eval`, calls `policy.eval_mode` for score actions, and records the
  selected mode in summaries. Upstream LightZero MuZero documents collect as
  sampling during collection and eval as choosing the argmax after MCTS search:
  `https://github.com/opendilab/LightZero/blob/main/lzero/policy/muzero.py`.

## 2026-05-13 Tournament Correctness Audit

- Current training defaults to `source_state_fixed_opponent`; the trainer module
  explicitly labels `source_state_turn_commit` as profile/plumbing only and blocks
  train mode for it.
- The tournament game loop uses the same underlying `VectorMultiplayerEnv`,
  two players, source-frame timing, action space of 3, normal death mode, and
  natural bonus spawn by default. It also records decision timing, source ticks,
  policy mode, and per-seat trail render mode in game summaries.
- The policy input shape matches the current LightZero conv model contract:
  float32 `(4, 64, 64)` plus an action mask. The tournament uses eval mode by
  default, which is the right scoring mode.
- The policy observation path is not the exact training wrapper path. Training
  `source_state_fixed_opponent` returns the ego player 0 stack from
  `CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv`; tournament games build
  per-seat stacks with `SourceStateGray64Stack4` from the current-policy smoke
  helper. That helper is player-perspective and uses its own schema id.
- Biggest concrete risk: tournament policy loading builds the LightZero policy
  with `env_variant=source_state_turn_commit` unless each checkpoint spec carries
  `model_env_variant`. Raw discovered checkpoint refs currently carry no
  `model_env_variant`, so the loader may compile a turn-commit surface for
  checkpoints trained on `source_state_fixed_opponent`. Shape compatibility can
  hide this drift.
- Second concrete risk: training observations have `to_play=-1`; tournament
  calls policy eval with `to_play=0` or `1`. If the non-board LightZero policy
  ignores `to_play`, this is harmless; if not, rankings can shift by seat.
- Third risk: tournament recovers checkpoint trail render mode and decision
  timing from metadata, but not the full env contract. Runs with non-default
  `natural_bonus_spawn`, opponent runtime/death mode, policy action repeat, or
  unusual reward/model variants can be evaluated under normal tournament knobs
  unless the spec carries explicit fields.
- Recommended small patch: during checkpoint discovery/loading, read
  `run.json`, `attempt.json`, or `command.json` for `env_variant`,
  `reward_variant`, `source_state_trail_render_mode`, decision settings, and
  relevant env knobs; pass `model_env_variant`/`model_reward_variant` into the
  policy builder and record the loaded contract in every battle/game summary.
- Implemented first guard: tournament policy loading now recovers
  `model_env_variant` and `model_reward_variant` from the checkpoint payload,
  attempt `command.json`, attempt metadata, or run metadata when the checkpoint
  spec does not provide them. The effective values are passed into the LightZero
  policy builder and recorded in `policy_loads`.
- Recommended guard: fail or loudly mark a tournament when a checkpoint trained
  on `source_state_fixed_opponent` is compiled as `source_state_turn_commit`.
  Do not silently rely on identical tensor shapes.
- Recommended tests: add an observation parity test comparing the training
  fixed-opponent wrapper's player 0 reset/step stack against the tournament
  `VectorMultiplayerEnv + SourceStateGray64Stack4` seat 0 path for the same
  seed/actions/trail mode; add a policy-load test that raw discovered checkpoint
  refs recover `model_env_variant`; add a `to_play` compatibility test or force
  tournament eval observations to the same `to_play=-1` non-board value.

## 2026-05-13 Website Refresh And Progress Fix

- The website had two slow paths:
  - progress endpoints scanned the live battle/shard tree;
  - standings/checkpoint pages could build a live provisional rating snapshot by
    scanning shard summaries when `latest.json` was missing.
- Both scans are now out of the normal web path. The website reads small
  committed artifacts: `progress.json`, `latest.json`, `provisional_latest.json`,
  `battle_index.json`, `battle.json`, and shard refs only for visible battle
  detail rows.
- A normal page reload forces one Modal Volume refresh and clears the in-process
  cache. Auto-refresh remains light: it polls `/api/rating-progress` and does
  not walk the tournament tree.
- Validation after deploy against
  `arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513-145153 /
  elo-latest212-allpairs-gpp11-gifs3-20260513-145153`:
  - page load: 2.885 seconds, `Cache-Control: no-store, max-age=0`
  - progress: 2.684 seconds, `complete`, `all_games_seen`,
    22366/22366 pairs and 246026/246026 games
  - standings: 5.519 seconds with forced fresh reload, 212 rows from the
    provisional file
  - checkpoint drilldown: 1.190 seconds, 211 battles, opponent ranks sorted
  - battle detail: 3 GIF samples for an 11-game battle
- A detached reduce was spawned for the timestamped run to write final
  `latest.json`: `fc-01KRHDG0AD3ZKBNWTJP44EGS2N`. At the last check,
  `latest.json` was not written yet, but `progress.json` showed all games seen.
- Design note: `modal.Dict` is a good future live cache for compact progress,
  but the Volume remains the durable source of truth. `modal.Queue` should only
  be added if we build an event-driven reducer; it should not be the durable
  record.

## 2026-05-13 Battle Table Sorting And Game Count Default

- The checkpoint drilldown table now exposes client-side sort toggles for
  opponent rank, average physical steps, and failure count. This sort only
  reorders rows already on the page, so it does not touch the Modal Volume or
  slow down review.
- The default tournament battle size is now 21 games per pair. Odd battle sizes
  remain required so a finished pair cannot end in a pure even-game split unless
  the game outcomes themselves draw.
- The default tournament game cap is now 8,000 decision steps. A timeout still
  counts as a draw, so the cap must be high enough that real death outcomes are
  not hidden by artificial truncation. With the usual 12 source frames per
  decision, this is up to 96,000 source physics ticks.
- Normal website clicks no longer force a Modal Volume reload every time.
  Ordinary page loads use a 30-second reload throttle, progress polling uses a
  60-second throttle, and `fresh=true` still forces a reload when an operator
  explicitly wants one.
- Next scheduling direction: move from one-off all-pairs runs toward an
  adaptive Elo lane. New policies should enter the pool, play enough games
  against existing policies to get a useful rating, and occasionally replay old
  matchups when the evidence is stale, noisy, or important for the current rank
  boundary.
- Deployed validation:
  - local focused suite: `68 passed, 10 skipped`
  - deployed page reload with selected checkpoint showed 211 battle rows and all
    three sort controls;
  - browser click test using Chrome confirmed average steps asc/desc, opponent
    rank asc/desc, and the failures toggle state;
  - `/api/review/checkpoint` returned rows from `source=battle_index` in about
    1.2-1.6 seconds for sampled top checkpoints.

## 2026-05-13 Final Fresh Latest-212 Plan

- Goal: launch a fresh score tournament from the latest checkpoint of every
  preserved CurvyTron run, after checking that tournament policy inputs match the
  current training observation path.
- User concern: recent tournament games look too short. That could be real weak
  play, but it could also mean the tournament is feeding the wrong observation,
  using the wrong environment knobs, or truncating games too early.
- Current contract check:
  - canonical two-seat training uses `VectorMultiplayerEnv` plus
    `SourceStateGray64Stack4`;
  - tournament score games also use `VectorMultiplayerEnv` plus per-seat
    `SourceStateGray64Stack4`;
  - tournament policy loading recovers each checkpoint's model env/reward
    contract and policy trail render mode from checkpoint/run metadata when the
    spec does not provide it;
  - tournament score mode remains greedy/eval, not collect-mode exploration;
  - tournament games now default to 21 games per pair and an 8,000-decision-step
    cap.
- Naming rule for new arenas: names must describe the actual contract. Use names
  like `arena-curvytron-latest212-score-eval-gpp21-max8000-obsmatch-YYYYMMDDa`
  and `elo-latest212-score-eval-gpp21-max8000-obsmatch-YYYYMMDDa`.
- Launch choice under review: use `games_per_shard=21` for the full all-pairs
  run. This keeps one shard per battle, reuses loaded policies across the 21
  games, and avoids hundreds of thousands of separate Modal calls. It is still
  parallel across all 22,366 pairs.
- GIF caution: with an 8,000-step cap, full 704x704 GIFs can become huge if the
  policies survive. Prefer the first fresh score run without massive GIF
  sampling; launch smaller visual diagnostics separately if we need to inspect
  long games.
