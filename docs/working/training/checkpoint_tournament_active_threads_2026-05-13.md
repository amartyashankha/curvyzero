# Checkpoint Tournament Active Threads, 2026-05-13

## Purpose

Keep the active plays visible. This is the short ledger; detailed thinking lives
in the goal, orchestration, scheduling, critique, and validation docs.

## Read First: Current Status

- Orchestration source of truth:
  `checkpoint_tournament_orchestration_2026-05-13.md` now separates main-thread
  ownership, subagent lanes, scheduler state, website sanity, future public
  leaderboard readiness, and the checkpoint discovery footgun.
- Active canary:
  `arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513d` /
  `elo-top20-furthest-intake-gifs5-20260513d`.
- Status: complete. It ran `420/420` games across `20/20` adaptive battles with
  `failed_game_count=0`, but this is not enough. Some policies have zero games
  because the batch scheduled only 20 battles for 20 policies. Treat this as a
  plumbing/GIF canary, not as useful Elo coverage.
- Visual inspection path is proven for the canary: battle detail reported
  `sample_gif_count=5`, and `/gif?ref=...` returned HTTP 200,
  `Content-Type: image/gif`, `704x704`, 15 frames.
- Intake architecture for this lane: Modal Dict/Queue handles coordination,
  dedupe, leases, and notifications; Volume artifacts remain the durable source
  of truth for manifests, scheduler state, battle summaries, rankings, and GIF
  refs.
- Hidden zero-checkpoint attempts: `20260513b`, `20260513c`, and the older
  `20260513a` attempt. The active intake manifest is the successful
  `20260513d` canary.
- Current active fix: adaptive V0 now has a hard low-coverage placement pass.
  If a checkpoint has zero games or zero distinct opponents, it must get a
  scheduled placement battle before the scheduler spends budget on normal
  near-rating or random bridge games.
- Current scale probe target: use the 20 furthest training runs and inject all
  discovered checkpoints from those runs. Latest discovery found 424 checkpoint
  refs. With `pairs_per_round=212`, pure scheduling covers all 424 checkpoints
  in one placement wave.
- Expanded probe launched and completed:
  `arena-curvytron-top20runs-allckpts-placement-gpp21-gifs5-step8000-20260513a`
  / `elo-top20runs-allckpts-placement-gpp21-gifs5-step8000-20260513a`.
  It ran 212 placement battles, 4452 games, and finished with
  `failed_game_count=0`.
- Website/API validation for the expanded probe: rankings returned 424 rows,
  zero checkpoints had zero games, min/max games per checkpoint were both 21,
  checkpoint drilldown returned a battle from the per-checkpoint index, battle
  detail returned `sample_gif_count=5`, and `/gif` served `image/gif`.
- Product reason for this whole lane: top-rated checkpoints become candidates
  for frozen opponents in future self-play training. The tournament must evolve
  toward an ongoing intake/rating process, not a one-shot scoreboard.
- Live progress fix in flight: the website was reading the persisted
  `progress.json` written at round start, so live rounds could show `0` games
  until the whole map reduced. The web progress reader now uses committed shard
  summaries for running rounds and keeps the short web cache.
- Scheduler update in flight: placement/low-coverage still runs first, then
  extra near-rating budget is smoothly biased toward high-ranked policies,
  especially the top 10 and top 20. This is not a hard cutoff; uncertain/random
  phases and low-evidence placement remain alive.
- Scheduler red-team follow-up landed locally: scalar-only
  `distinct_opponents` is no longer trusted for adaptive placement. A row needs
  real `opponent_ids` or pair history to count as covered.
- Rating status guard landed locally: `active` now requires the placement
  evidence target, not the old weak `300 games / 5 opponents` shortcut.
- Website performance side lane is documented in
  `checkpoint_tournament_website_performance_plan_2026-05-13.md` and
  `checkpoint_tournament_website_ux_redteam_2026-05-13.md`. Main lesson: use
  small indexed artifacts, paged API payloads, lazy GIFs, and light refresh
  tokens instead of request-time scans.

## Next Cleanup Phase

- Keep the single Modal app entrypoint and stable public facade.
- Prefer a small battle-detail/GIF sample index or website paging/index cut
  next. This is the practical cleanup needed before larger GIF-heavy runs.
- Only extract another pure helper if the boundary is obvious and focused tests
  stay green.

## Next Scale Gate

- Next scale decision: run the expanded top-20-runs all-checkpoint probe after
  deploy. It is still a probe, not the final online Elo system. This probe is
  now complete and should be used as evidence for the next online-intake design.
- For this probe, keep all 424 discovered checkpoints because each selected run
  has about 20 checkpoints already. No downsampling is needed yet.
- Schedule 21 games per battle, five GIF samples per battle, max 8000 steps,
  and enough placement pairs that no checkpoint has zero games.
- Before either path grows, confirm battle-detail/read paths stay bounded and
  avoid request-time scans over huge game/GIF lists.

## Current Main Target

Build all-checkpoint adaptive Elo for CurvyTron.

Plain version: every useful checkpoint can enter the pool, but we do not run
all-pairs. New checkpoints get placement games, strong ones move up, uncertain
or important matchups get more games, and the coach can watch progress on the
website.

Product reason: the highest-rated policies become candidates for frozen
opponents in future self-play training. This must become an ongoing background
process, not a one-off report.

## Active Threads

| Thread | Owner | Status | Next Gate |
| --- | --- | --- | --- |
| Orchestration | main | active | keep this ledger and orchestration doc current |
| Rating research | Gibbs / Arendt | first pass done | turn critique items into tests and scheduler rules |
| Code architecture | Pauli / main | helper and artifact wiring landed | adaptive remote smoke |
| Checkpoint discovery | Arendt / Pauli / main | footgun guard landed | use `checkpoint_selection=all` in adaptive runbook |
| Modal ops | Hilbert | first pass done | add commit/reload retry only if needed; keep Volume as truth |
| Website scale | Lorentz / main | per-checkpoint index contract tightened | later add paging for game rows and battle details |
| Validation | main | focused tests passing | run focused tests after each patch; smoke remote only after wiring |
| Product/coach view | main + critique agents | active | ensure rankings show status, evidence, and freshness |
| Refactor architecture | Arendt / Mill / Hilbert / Lorentz / main | Cut 1 landed; wave 2 mostly returned | fold critiques into next small cut |

## Current Evidence

- Pure `adaptive_v0` scheduler helpers exist and are tested.
- Pair specs can carry `pair_key`, `schedule_reason`, and `schedule`.
- Pair history is keyed by canonical sorted checkpoint ids.
- Adaptive scheduler tests now cover two new contracts: placement happens before
  top-band bias, and top-band bias increases leader appearances without killing
  uncertain/random phases.
- Live website progress tests now guard the stale-zero regression by writing a
  stale `progress.json` with zero completed games and expecting live shard
  progress instead.
- Progress refresh now keeps the normal website path cheap: it uses shard
  summaries when available and pair-directory estimates otherwise. Exact
  per-game summary counting exists as an explicit diagnostic path, not the
  default web refresh, because scanning tens of thousands of game files is too
  slow for normal operation.
- Rating rounds now write `pair_history.json` and `scheduler_state.json`.
- Discovery scans `train/lightzero_exp*/ckpt/iteration_*.pth.tar`.
- Discovery supports `latest`, `iteration`, and `all`.
- Focused local test result after contract extraction, checkpoint-index
  tightening, and roster-identity guards: `90 passed, 10 skipped`.
- Cleanup Cut 1 compile and `git diff --check` passed.
- Context-hash guard compile and `git diff --check` passed.
- Remote discovery smoke passed: `found_count=3`, `missing_count=0`, and two
  of the three returned refs were under timestamped `lightzero_exp_*` dirs.
- Remote adaptive rating smoke passed:
  `arena-curvytron-adaptive-v0-context-smoke-20260513a`,
  `pair_selection=adaptive_v0`, `pairs_per_round=2`, `games_per_pair=3`,
  `games_per_shard=3`, `max_steps=8`, `game_count=6`, `rated_pair_count=2`.
- Post-cleanup adaptive Modal smoke passed:
  `arena-curvytron-cleanup-contracts-adaptive-smoke-20260513a`,
  `pair_selection=adaptive_v0`, `pairs_per_round=2`, `games_per_pair=3`,
  `games_per_shard=3`, `max_steps=64`, `game_count=6`, `rated_pair_count=2`,
  `max_abs_delta=8.0`, and `checkpoint_roster` was present in the latest
  snapshot.
- Deployed website smoke passed. The deployed endpoint loaded tournaments,
  rankings, progress, and checkpoint drilldown for the post-cleanup smoke.
  Checkpoint drilldown returned `source=checkpoint_battle_index`.
- Refactor critique lane returned a consistent answer: keep the single Modal
  app entrypoint, keep public wrappers stable, and first promote implicit
  strings/paths into named contracts before moving code.
- The two largest risk files remain
  `src/curvyzero/tournament/curvytron_checkpoint_tournament.py` and
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`.
- Cleanup wave 2 is now delegated:
  domain boundaries, artifact contracts, validation, Modal ops, website, and
  scheduler/Elo critique are running in parallel.
- Returned critique docs now cover domain, contracts, Modal ops, website,
  scheduler/Elo, and validation.
- Strongest new finding: `pool_hash` alone was too blunt. A separate
  evaluator/rating context hash is now implemented locally and tested.
- Previous rating snapshots also reject changed evaluator context.
- Named helper refs for pair spec, provisional latest, and run-level rating
  results are covered by tests.
- Website checkpoint drilldown now prefers per-checkpoint battle indexes instead
  of filtering the whole global battle index.
- The pure contract names and artifact ref helpers now live in
  `src/curvyzero/tournament/curvytron/contracts.py`. The old
  `curvytron_checkpoint_tournament.py` module still re-exports them, so Modal
  and tests keep using the stable `arena.*` surface.
- Per-checkpoint battle indexes now carry their own `ref`, preserve scheduling
  metadata, defensively filter stale rows, and do not trigger live shard scans
  on normal checkpoint drilldown.
- Rating artifacts now carry `checkpoint_roster`, keyed by checkpoint id. This
  lets roster expansion reuse old evidence while rejecting the dangerous case
  where an explicit checkpoint id is reused for a different checkpoint ref.

## Current Not Done

- Website has per-checkpoint battle indexes for checkpoint drilldown. It still
  needs more paging/index work for very large game/GIF detail and probably a
  lighter battle-detail artifact for GIF samples.
- Tiny adaptive rating smoke has run. It was a plumbing smoke only, not a policy
  strength claim.
- Cleanup refactor Cut 1 landed locally. It names artifact filenames, rating
  pair-selection names, schedule reasons, checkpoint discovery names, and orphan
  rating refs without changing tournament behavior.
- Cleanup refactor Cut 1A landed locally. It moved the pure tournament contract
  names/ref helpers into `curvytron/contracts.py` while preserving the facade.
- Next cleanup guard should be small: either add a battle-detail/GIF sample
  index, or extract another pure helper only if the boundary is obvious.
- No large all-checkpoint adaptive run should launch yet.

## Keep Spinning

- Keep critique lanes alive when design changes.
- Keep code changes small and test them in parallel.
- Keep docs updated immediately after evidence changes.
- Do not let the trainer footgun consume this lane; record it and protect
  tournament discovery, but leave trainer repair to the coach/optimizer lane.
- Refactor only in safe slices. Each cut should preserve the Modal app entrypoint
  and keep focused tests green.
- First cleanup slice should be boring: name contract strings and paths only.
  Do not move `run_checkpoint_game` or website route registration yet.

## 2026-05-13 Visual And Intake Addendum

- The latest post-cleanup adaptive smoke
  `arena-curvytron-cleanup-contracts-adaptive-smoke-20260513a` had GIFs off on
  purpose. Treat it as a plumbing/context smoke, not as proof that the website
  can inspect GIF-producing tournament battles.
- Immediate user-facing target: a top-20/latest-checkpoint visual tournament.
  Use the latest discovered checkpoint for each selected run, keep the pool
  small enough for full pair coverage, and request at least 5 GIF samples per
  battle. This is an inspection/canary tournament, not the long-term
  all-checkpoint rating lane.
- Longer-term target: all-checkpoint adaptive intake. New checkpoint candidates
  should enter a bounded scheduler as they appear, get placement/anchor/nearby
  matches, and become durable only through Volume artifacts. Modal Dict/Queue can
  coordinate subscriber intake, leases, and notifications, but Volume artifacts
  remain the source of truth.

## 2026-05-13 Intake Evidence Update

- Bug found and fixed: `intake-seed` accepted explicit `--checkpoint-refs` but
  did not copy them into the remote scan spec. That produced a zero-checkpoint
  manifest and therefore no GIFs. Regression now covers explicit checkpoint refs.
- Current canary roster source: preserved 212-run manifest
  `artifacts/local/curvytron_pruning/curvytron_prune_preserve_20260513c.json`.
  Discovery found `212/212`; top-20 selection is sorted by highest latest
  checkpoint iteration, then checkpoint mtime.
- Current live visual canary:
  `arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513d` /
  `elo-top20-furthest-intake-gifs5-20260513d`.
- Canary plan estimate: 20 checkpoints, 20 adaptive battles, 21 games per
  battle, 5 GIF samples per battle, 420 games total, 100 GIFs total.
- Intake seed wrote 20 checkpoint refs into the named Dict/Volume manifest and
  enqueued 20 Queue events. Deployed drain consumed all 20 events and spawned
  rating call `fc-01KRHN3X4F6V04BQTC4HV51701`.
- Important Modal launch footgun: spawning rating from a local ephemeral
  `modal run` app did not prove the deployed background path. For operator
  testing, call the deployed functions by name or rely on deployed scheduled
  functions, so spawned work belongs to the deployed tournament app.
- Deployed app now includes `curvytron_checkpoint_intake_drain_tick`, a scheduled
  Queue drain. It only spawns rating work when it actually consumes events, so
  empty active manifests do not create duplicate runs.
- Broken zero-checkpoint intake attempts were hidden from the website and removed
  from the active intake key list. The only active intake manifest after cleanup
  is the successful `20260513d` canary.
- Website critique: the canary proves the happy path, but scale is not UI-safe
  yet. Battle detail can still read too many summaries, paging is stronger in
  the API than the HTML, and reload/error states need clearer user-facing
  handling.
- Architecture critique: keep the single Modal app stable, but the next
  extraction should be pure discovery/intake contracts first. Do not move
  `run_checkpoint_game`, policy loading, GIF writing, or public Modal wrappers
  during the next cleanup cut.
- Intake safety update: scheduled Queue drain now has a guard against spawning
  into an already-existing rating run unless an operator explicitly overrides it.
  This prevents repeated Queue ticks from overwriting the same round artifacts.
- Intake critique warning: V0 is still not full online Elo. Queue loss can still
  be permanent if a tick marks refs seen before enqueue succeeds, duplicate tick
  events are not fully deduped, and online continuation still needs a real
  next-round/lease contract. Do not scale this as a self-running all-checkpoint
  service until those gates are fixed.
- V0 drain hardening: scheduled drain now requests at least the full current
  manifest size, so a 212-ref seed is not split only because the old default
  drained 100 events at a time.
- V0 intake hardening after critique: manifests now track `queued_checkpoint_refs`
  separately from `seen_checkpoint_refs`, Queue events carry stable `event_id`
  values, and drains claim a rating run in Modal Dict before spawning. This does
  not finish online Elo, but it removes the most obvious duplicate-spawn and
  "seen but not queued" traps.
- Deployed V0 queue smoke passed with two explicit checkpoint refs:
  `seed_enqueued_count=2`, `queued_checkpoint_count=2`, `drain_event_count=2`,
  and no rating run was spawned. The smoke marker was hidden afterwards.
- Canary completion evidence: `420/420` games and `20/20` battles completed with
  `failed_game_count=0`.
- Battle detail evidence: one completed battle returned `sample_gif_count=5`;
  its first sampled `/gif` response was HTTP 200 `image/gif`, `704x704`, 15
  frames.

## 2026-05-13 Leaderboard Readiness Update

- Treat the current top-20 visual/intake canary as operational evidence only:
  it scheduled 21 games per battle, but only one distinct opponent per
  checkpoint.
- Public leaderboard readiness now requires breadth: target at least 20
  distinct opponents per checkpoint before active status, using bounded
  scheduler rounds instead of full N^2 all-pairs.
- The leaderboard should become a durable public surface that training loops can
  later consume for frozen-opponent sampling. Keep status/evidence fields
  visible so training does not sample from immature one-opponent ratings unless
  explicitly allowed.
- Continue discovering latest checkpoints with
  `train/lightzero_exp*/ckpt/iteration_*.pth.tar`; timestamped
  `lightzero_exp_*` dirs remain mandatory discovery inputs.
