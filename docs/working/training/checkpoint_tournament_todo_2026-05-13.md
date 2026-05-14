# Checkpoint Tournament To-Do, 2026-05-13

## Read First: Current State And Next Move

- Use `checkpoint_tournament_orchestration_2026-05-13.md` as the concise
  planning/delegation/operating-pattern doc before choosing the next launch.
- Current active canary:
  `arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513d` /
  `elo-top20-furthest-intake-gifs5-20260513d`.
- Done: the canary completed `420/420` games and `20/20` battles with
  `failed_game_count=0`.
- Done: battle detail returned `sample_gif_count=5`, and `/gif?ref=...` served
  HTTP 200 `image/gif`, `704x704`, 15 frames.
- Not done: the canary is not meaningful Elo coverage. Some policies have zero
  games because only 20 battles were scheduled for 20 policies.
- Done: broken zero-checkpoint attempts `20260513b`, `20260513c`, and the older
  `20260513a` attempt were hidden; `20260513d` is the active manifest.
- Next cleanup phase: battle-detail/GIF sample index or website paging/index
  work, keeping the existing Modal app facade stable.
- Next scale gate: expanded roster probe. Use the 20 furthest runs and inject
  all checkpoints discovered for those runs. Latest saved discovery found 424
  refs; pure adaptive scheduling covers all of them with 212 placement battles.
- Done: expanded roster probe completed with 424 checkpoints, 212 battles, 4452
  games, no failed games, no zero-game checkpoints, and five GIF samples on a
  checked battle.
- Zero-game policies are a scheduler/product bug. Every newly injected or
  low-coverage checkpoint must get at least one battle before it can appear as a
  ranked policy.

## Now

- Keep docs current.
- Current pivot: stop treating latest-only all-pairs as the main path. The new
  lane is all-checkpoint adaptive Elo. See
  `checkpoint_tournament_orchestration_2026-05-13.md`.
- Keep the active-thread ledger current:
  `checkpoint_tournament_active_threads_2026-05-13.md`.
- Keep the refactor plan current:
  `checkpoint_tournament_refactor_plan_2026-05-13.md`.
- Keep an active failure-mode critique doc:
  `checkpoint_tournament_architecture_critique_2026-05-13.md`.
- Build the scheduler in research-first order: design, pure helper, tests,
  artifact wiring, tiny smoke, then larger launch.
- New immediate scheduler requirement: every checkpoint in a new batch should
  get at least some games. A bounded random/placement mode is acceptable for now;
  true adaptive online Elo can come next.
- Scheduler heuristic after placement coverage: bias extra battles smoothly
  toward higher Elo/rank policies, especially top-10/top-20, because public
  leaderboard and training consumers care most about reliable top policies.
  Preserve a floor for low-Elo policies until they have enough games and
  distinct opponents.
- Do not let top-band bias hide weak coverage. The scheduler contract is:
  placement first, then top-band polish, with uncertain and random bridge phases
  still present.
- Website progress must update during a live round from committed shard
  summaries. The persisted `progress.json` can be stale between round start and
  reduce.
- Website progress must handle runs launched with `games_per_shard=1` without
  scanning every game file in the request or normal background refresh. Use
  pair-directory estimates now; build a small indexed progress artifact next.
- Next website cleanup should follow the new performance docs: page checkpoint
  and battle lists, lazy-load GIFs, use tiny freshness tokens, and remove
  `limit=1_000_000` from normal click paths.
- Done locally for the first website cut: battle detail has `game_limit` /
  `game_offset`, summary GIF samples are preferred, and normal checkpoint/battle
  click paths no longer request `limit=1_000_000`.
- Scheduler coverage should never trust scalar-only `distinct_opponents`; use
  `opponent_ids` and pair history. Rating `active` status should require the
  configured placement evidence target.
- Done locally for the first scheduler guardrail: placement cannot expand a
  round to the full evidence deficit. It is capped to the effective round budget
  with a first-touch floor for absurdly small requested budgets.
- Public leaderboard lane: keep it separate from the website. Current
  tournaments are not automatically the public leaderboard. Future training
  should consume frozen opponent assignment snapshots derived from durable
  leaderboard snapshots.
- High-fidelity online validation lane: design and run a remove/reintroduce
  test for the current top policy. Do not delete the checkpoint. Define exactly
  what purge means before touching state.
- For mixed old+new online batches, placement coverage is attached to the
  low-coverage checkpoint itself. Established checkpoints can be used as anchors,
  but they must not count as covering the new checkpoint unless the new
  checkpoint actually appears in the pair.
- First pure helper patch landed: `adaptive_v0` pair selection, schedule
  metadata pass-through, canonical `pair_key`, `pool_hash`, and pair-history
  helper. Focused tests passed after discovery tests were added:
  `78 passed, 10 skipped`.
- Remote discovery smoke passed with timestamped `lightzero_exp_*` checkpoint
  refs visible in the returned rows.
- Keep the main thread on orchestration. Rating research, code architecture,
  Modal ops, website scale, and docs synthesis are parallel sub-agent lanes.
- Run local tests after edits.
- Redeploy the Modal app after code changes.
- Use meaningful arena names that include the important contract: latest count,
  score/eval, games per pair, max step cap, and observation-match intent.
- For adaptive runs, include the player set and scheduler in the name, for
  example `arena-curvytron-allckpt-adaptive-elo-gpp21-step8000-YYYYMMDDa`.
- Readability rule: checkpoint labels shown to humans should include the run
  identity and iteration. Raw `iteration_*.pth.tar` labels are not enough for
  all-checkpoint rosters.
- Intake V0 rule: use it as a guarded batch launcher. A true online service
  still needs continuation from `latest.json`, stale-claim repair, and duplicate
  event repair.

## Current Next Gates

- Keep cleanup wave 2 critiques folded into the refactor plan and architecture
  critique doc.
- Local compile, focused tests, and diff check passed for cleanup Cut 1, the
  context-hash guard, contract extraction, checkpoint-index tightening, and
  roster-identity guard. Latest focused result: `90 passed, 10 skipped`.
- `rating_context_hash` landed locally. It lets roster expansion reuse old pair
  evidence while rejecting changed evaluator context.
- `checkpoint_roster` landed locally. It rejects the dangerous case where the
  same checkpoint id is reused for a different checkpoint ref.
- Previous rating snapshots also reject changed evaluator context.
- Tiny adaptive rating smoke with explicit checkpoint refs passed and is
  recorded in the validation and active-thread docs.
- Post-cleanup adaptive rating smoke with explicit checkpoint refs also passed:
  `arena-curvytron-cleanup-contracts-adaptive-smoke-20260513a`.
- Per-checkpoint battle indexes landed locally and were tightened. Checkpoint
  drilldown no longer needs to filter the whole global battle index or scan live
  shard summaries in the normal path.
- Contract extraction landed locally. `curvytron_checkpoint_tournament.py`
  remains the facade; pure contract names/ref helpers now live under
  `curvyzero.tournament.curvytron.contracts`.
- Next remote step should be website/detail-index work or a modest adaptive
  scale probe, not a large all-checkpoint run yet.
- Keep website paging and battle-detail artifact work as a follow-up before
  large runs.

## Recent Historical Notes

- Older latest-212 all-pairs runs were stress tests, not the main path now.
- Even battle sizes are invalid for new tournament specs.
- Website dropdown, progress, and drilldown fixes landed earlier and should stay
  covered by focused tests.

## Soon

- Add checkpoint discovery helpers for "latest N checkpoints from recent runs."
- Add a tournament dry-run estimator. A basic `--mode estimate` exists; decide
  whether it is enough for operator use.
- Add a repair/backfill command for pair and tournament aggregates.
- Watch whether the provisional loop cadence is frequent enough for long
  tournaments. It currently refreshes about once per minute.
- Decided default policy mode for score tournaments: greedy/eval for official
  score, collect mode only as a separately labeled diagnostic or visual sample.
- Add an explicit tournament evaluation contract so observation/environment
  settings cannot drift silently.
- First contract patch is done: policy loading recovers model env/reward variant
  metadata and records the effective values. Still add parity tests for the
  actual observation tensor path before calling the tournament contract final.
- Decide seat fairness for real Elo. Current latest-212 runs are unordered
  no-self pairs with one fixed seat order. That is fast and simple, but not
  seat-neutral.
- Add a GIF preview/downsample lane only if lazy loading plus byte caching is
  still not enough.
- Start the adaptive Elo design: new checkpoint enters, plays a small but useful
  set of games against anchors/current neighbors, then old important matchups
  are replayed when rankings are uncertain or stale.
- Modal rating rounds now write scheduler-state and pair-history artifacts, and
  the parent rating loop passes them into the next round.
- Add a tiny adaptive remote smoke with explicit checkpoint refs before any
  larger all-checkpoint run.
- Add a synthetic scheduler/rating simulator before trusting a large adaptive
  run. It should prove that a bounded number of scheduled pairs can recover a
  rough hidden strength order.
- Discovery now has `checkpoint_selection=latest|all|iteration`. Default
  remains `latest`; adaptive all-checkpoint runs use `all`.
- Add a shard-summary reduce path for repair/reduce. Current normal shard flow
  can reduce from shard tallies, but manual reduce still falls back toward game
  summary scanning.
- Add battle-detail paging/indexes before very large adaptive runs. The website
  should not list a million battles or read shard/game files in request
  handlers.
- Promote orphan path literals into helpers before adding more features:
  `provisional_latest.json`, rating-run `results.json`, and `pair_spec.json`
  should have clear ownership and helper refs.

## Elo Phase

- Batch Elo from battle summaries exists for final snapshots.
- Provisional Elo exists as a website bridge from `provisional_latest.json`.
- Add synthetic simulator tests for Elo recovery.
- Add tests for background provisional snapshots once that writer exists.
- Add pure public leaderboard snapshot/pointer validators and a publisher from
  verified rating snapshots.
- Add a selector that creates immutable opponent assignment snapshots from one
  public leaderboard snapshot.
- Wire the trainer to an explicit `--opponent-assignment-ref` later. Do not let
  `train_muzero`, hooks, env reset, or env step read Modal Dict, leaderboard
  latest pointers, or selector state.
- Add an operator command or repair tool for safe leaderboard/tournament member
  retirement and reintroduction. It must preserve checkpoint files and should
  be idempotent.

## Do Not Do Yet

- Do not build a complex tournament format.
- Do not add a new Modal app per battle.
- Do not make training depend on tournament code.
- Do not save every GIF for a huge tournament unless explicitly asked.

## 2026-05-13 Visual Tournament And Intake Next

- Do not describe the last adaptive smoke as a GIF test. It deliberately ran
  with GIFs off.
- Build a top-20 roster from the latest trusted ranking/readout, then rediscover
  the literal latest checkpoint for each selected run using
  `train/lightzero_exp*/ckpt/iteration_*.pth.tar`.
- Launch a small visual tournament over those latest checkpoints. For 20
  checkpoints, full unordered pairs is 190 battles. Use
  `games_per_pair >= 5` and `gif_sample_games_per_pair >= 5` so every battle can
  expose at least 5 GIF samples. Keep this separate from official large score
  waves.
- Modal intake design: use a named Queue for short-lived checkpoint-candidate
  notifications, a named Dict for dedupe/latest-seen/lease state, and Volume
  artifacts for durable checkpoint pool, scheduler state, battle summaries,
  rankings, and GIF refs.
- The intake path must be idempotent. Duplicate checkpoint events should be
  harmless, queue loss should be repaired by periodic Volume scanning, and
  roster/context hashes must still guard rating reuse.

## Immediate Follow-Up After Intake Canary

- Done: confirmed the live canary writes five GIF samples per completed battle
  and the website battle detail returns those samples.
- Done: confirmed the active canary completed all planned work:
  `420/420` games, `20/20` battles, `failed_game_count=0`.
- Done: confirmed `/gif?ref=...` serves sampled GIFs as HTTP 200 `image/gif`,
  `704x704`, 15 frames.
- Done: hidden the broken zero-checkpoint intake attempts:
  `arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513b`,
  `arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513c`, and
  `arena-curvytron-top20-latest-adaptive-gif5-gpp21-step8000-20260513a`.
- Done for V0: deployed a scheduled Queue drain that spawns rating work only
  after consuming Queue events.
- The 20-checkpoint canary now has visible GIFs and rankings from the website;
  do not scale to all 212 latest checkpoints until the next website/detail-index
  cleanup gate is chosen.
- For the eventual all-checkpoint lane, use `checkpoint_selection=all` or an
  explicit manifest built from `train/lightzero_exp*/ckpt/iteration_*.pth.tar`;
  never assume `train/lightzero_exp/ckpt` is the whole set.
- Next: scale from the 20-checkpoint canary to a larger/latest-212 plan only
  after choosing whether the next step is all-latest adaptive placement or the
  all-checkpoints online Elo intake.

## Intake Idempotency Gates Before Online Elo

- Fix enqueue durability: a checkpoint should not become permanently `seen`
  until the corresponding event is recoverable from Queue or durable pending
  state.
- Add event ids or manifest generations so duplicate ticks do not create
  duplicate scheduling work.
- Add a real lease/claim for rating runs so two drains cannot write the same
  `rating_run_id` artifacts.
- Decide online continuation: either reject appending to an existing rating run
  loudly, or load `latest.json` and start the next round instead of resetting
  round zero.
- Validate explicit checkpoint refs against the training Volume before trusting
  them for large manifests.

Partially addressed in V0:

- Queue events now carry stable `event_id` values.
- Drain now claims a rating run in Modal Dict before spawning.
- Scheduled drain coalesces at least the current manifest size in one pass.
- `queued_checkpoint_refs` is tracked separately from `seen_checkpoint_refs`.

## 2026-05-13 Scheduler Breadth Correction

- Do not promote the current 20-checkpoint visual canary as a rating-quality
  result. It ran 21 games per scheduled battle, but each checkpoint only saw one
  distinct opponent.
- Add a scheduler gate: a checkpoint needs at least 20 distinct opponents before
  it can be considered leaderboard-active. Until then, label it provisional and
  keep scheduling placement, anchor, nearby-rating, and bridge opponents.
- Keep the bounded scheduler goal. The target is broad evidence without full
  N^2 all-pairs coverage.
- Public leaderboard is a future training contract: training loops may later
  sample frozen opponents from the leaderboard, so the leaderboard must publish
  status, games, distinct opponents, freshness, and discovery provenance, not
  just rank/Elo.
- Latest checkpoint discovery remains broad:
  `train/lightzero_exp*/ckpt/iteration_*.pth.tar`. Any manifest built from
  `train/lightzero_exp/ckpt` alone must be treated as stale until rebuilt.
