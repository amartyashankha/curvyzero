# Checkpoint Tournament Orchestration, 2026-05-13

## Current North Star

Build a CurvyTron checkpoint rating lane that can absorb every useful checkpoint
without needing every checkpoint to play every other checkpoint.

The coach needs a live map of which policies are improving. The website should
show rankings, progress, battles, and a few GIF samples. The rating system should
keep running as new checkpoints arrive.

## Current Pivot

The latest-only all-pairs stress run is no longer the main target.

The new target is an online/adaptive Elo system:

- include every checkpoint that is worth rating, not only the latest checkpoint
  from each run;
- do not schedule all-pairs when the player pool gets large;
- place new checkpoints with a small slate of useful opponents;
- keep replaying useful old matchups when ratings are close, stale, noisy, or
  important for the top ranks;
- keep all game and battle artifacts immutable so ratings can be recomputed.

## Main Thread Job

The main thread should stay boring and clear:

1. Re-read this doc, the todo doc, and the scheduling research doc.
2. Re-read the architecture critique doc.
3. Check what changed in code before editing.
4. Delegate research and critique lanes before implementation.
5. Make small code/doc changes only after the target is clear.
6. Run focused tests and a small Modal smoke before larger launches.
7. Update docs with evidence, decisions, and remaining gaps.
8. Decide the next batch of work from the latest evidence.

If a requested launch no longer matches the current target, stop and rewrite the
target first.

## Sub-Agent Plan

Use sub-agents for work that can run in parallel and does not block the next main
thread step.

### Rating Research Lane

Difficulty: extra high.

Owner: Gibbs now, plus Arendt as backup.

Questions:

- How do Elo, Glicko, TrueSkill, Swiss, and active pairwise-comparison systems
  place new players?
- How should CurvyTron choose opponents without all-pairs?
- How many games per battle are useful?
- When should old battles be replayed?

Output:

- append short conclusions to
  `checkpoint_tournament_scheduling_research_2026-05-13.md`;
- return a V0 scheduler recommendation and V1 upgrades.

### Code Architecture Lane

Difficulty: high.

Owner: Pauli.

Questions:

- Where should adaptive pair selection plug into
  `build_rating_round_pair_specs(...)`?
- What fields should pair specs carry, such as `schedule_reason`?
- How can the reducer keep ratings available while work is still running?
- What tests prove the scheduler is deterministic and bounded?

Output:

- minimal code-change plan;
- risks if all checkpoints from all runs are included.

### Modal Ops Lane

Difficulty: high.

Owner: Hilbert.

Questions:

- How far can shard fan-out be pushed before cold starts, queueing, or timeouts
  dominate?
- Where do retries and backoff belong?
- Should Modal Dict or Queue help with progress, or is the current Volume
  artifact path enough for now?
- How do we avoid expensive Volume reloads and open-file reload errors?

Output:

- launch/runbook advice;
- simple retry/progress pattern;
- warnings before any very large launch.

Current recommendation from the ops lane:

- For adaptive Elo production, use one shard per selected pair:
  `games_per_shard == games_per_pair`.
- Keep GIFs off in large score waves. Run GIF canaries or sample jobs
  separately.
- Keep shard outputs compact and reduce at battle scale.
- Add retry/backoff around commit/reload helpers if open-file or transient
  freshness issues keep appearing.
- Treat Modal Dict/Queue as optional coordination helpers, not the durable source
  of truth. Durable truth stays in Volume artifacts.

### Website Scale Lane

Difficulty: high.

Owner: Lorentz.

Questions:

- How should the site show a huge player set without a giant wall of rows?
- How should it show partial rankings from the start?
- How should policy -> battles -> GIF samples stay fast?
- What should auto-refresh check, and how often?

Output:

- one-page UI plan;
- needed API/cache changes;
- smoke checks for reload, dropdowns, sorting, and GIF drilldown.

Current recommendation from the website lane:

- Initial HTML should be cheap: progress, first ranking page, and recent battles.
- Policy click should read a small per-checkpoint battle index or a paged server
  endpoint, not a giant full battle index.
- Battle click should return summary and GIF samples first; detailed game rows
  should be paged.
- Auto-refresh should poll small progress/ranking artifacts and keep scroll and
  selection stable.
- Large runs need published web artifacts, not request-time shard/game scans.

### Docs And Synthesis Lane

Difficulty: medium to high.

Owner: Mill.

Questions:

- Are the docs aligned with the current target?
- Are old launch commands clearly marked as historical?
- Are decisions and evidence easy for the coach to read later?

Output:

- doc section proposals;
- stale or contradictory notes to clean up.

### Architecture Critique Lane

Difficulty: extra high.

Owner: main thread plus reused sub-agents while the agent limit is full.

Questions:

- What could silently be wrong?
- What could become too slow?
- What could mislead the coach?
- What could be correct for a smoke test but wrong at all-checkpoint scale?

Output:

- keep `checkpoint_tournament_architecture_critique_2026-05-13.md` current;
- turn critique items into tests or explicit non-goals.

## Current Known Code Shape

- The rating helper currently supports `pair_selection` values `all_pairs` and
  `random`.
- `build_rating_round_pair_specs(...)` currently builds the full candidate pair
  list before selection. That is fine for hundreds, but it is the wrong shape
  for every checkpoint from every run.
- The sharded runner is already useful: `games_per_shard=21` can run one whole
  21-game battle per worker and reuse loaded policies.
- The current rating loop writes immutable battle summaries and derived rating
  snapshots.
- Website performance depends on small index/snapshot files. It should not scan
  every game summary in request paths.
- `normalize_pair_spec(...)` may drop unknown fields, so adaptive metadata needs
  an explicit `schedule` field or a separate round-input sidecar.

## V0 Adaptive Scheduler Shape

Keep batch Elo. Add adaptive pair selection above it.

The first scheduler should be simple:

- `placement`: new checkpoints play anchors, lineage neighbors, median players,
  and a few random established players.
- `near_rating`: active checkpoints play opponents with nearby current ratings.
- `uncertain`: checkpoints with few games, few opponents, high rating movement,
  or many failures get more matches.
- `bridge`: seeded random cross-band matches keep the graph connected.
- `replay`: old close/noisy/stale matchups get refreshed.

Each scheduled pair should carry `schedule_reason`.

Minimal code shape:

- add `adaptive_v0` as a third `pair_selection` value;
- require `pairs_per_round` for `adaptive_v0`;
- add additive args to `build_rating_round_pair_specs(...)`:
  `scheduler_state` and `pair_history`;
- add pure helper `select_adaptive_v0_pair_slots(...)`;
- keep existing `all_pairs` and `random` behavior unchanged;
- pass through schedule metadata in `normalize_pair_spec(...)`;
- add pair history keyed by canonical sorted checkpoint ids, not battle id;
- store a pool hash in history/scheduler state so old artifacts cannot silently
  mix with a new checkpoint pool.

The first adaptive selector must not build all candidate pairs. It should use:

- sorted ratings and small neighbor windows for near-rating pairs;
- low-coverage rows for placement/uncertain pairs;
- fixed anchors and rating quantiles;
- capped seeded rejection sampling for random bridges;
- existing pair history rows for replay candidates.

The first implementation should avoid true game-by-game online updates. Use
bounded batch-online waves instead:

1. Snapshot ratings at the start of the wave.
2. Pick at most `wave_battle_budget` pairs.
3. Run all 21-game battles in parallel shards.
4. Reduce to battle summaries.
5. Write the next rating snapshot and scheduler state.
6. Repeat.

This keeps Modal completion order from changing the rating truth.

## Pool And History

The adaptive lane needs state above a single round:

- pool rows: checkpoint ref, checkpoint id, run id, iteration, lineage key,
  observation/env contract, status, first seen time;
- pair history: stable sorted pair key, battle count, game count, last round,
  last score, failures, draws, latest battle refs;
- scheduler state: round index, seed, active anchors, budget settings, and
  current quotas.

Do not let long runs dominate just because they saved many checkpoints. The pool
can include all checkpoints, while the scheduler enforces per-run or per-lineage
budget caps.

## Acceptance Checks For V0

- The scheduler never creates more pairs than the requested budget.
- The scheduler is deterministic for the same seed and snapshot.
- Every new checkpoint gets a minimum number of distinct opponents before being
  marked active.
- The graph has random bridge pairs so rating islands do not form.
- Historical checkpoints from one run cannot consume the whole round budget.
- Replayed pairs create new immutable battle refs and update pair history.
- Existing `all_pairs` and `random` modes still produce the same pair order and
  same seeded behavior as before.
- Pair history rejects pool-hash mismatch instead of mixing old and new pools.
- The website can show partial rankings before all games complete.
- Website checkpoint and battle drilldowns are paged or indexed. Clicking a row
  should not scan every battle or every game.
- A small synthetic simulator recovers the rough ordering of fake strengths.
- A small remote smoke runs with GIFs off and writes a readable snapshot.

## Research Evidence So Far

- Glicko-2 tracks uncertainty directly and moves ratings more when results are
  inconsistent with the current estimate: https://www.glicko.net/glicko/glicko2.html
- TrueSkill matchmaking uses match quality and points out that close opponents
  are useful, but new-vs-established matches can also teach the system a lot:
  https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
- Swiss systems pair players with similar scores and avoid repeat pairings in a
  fixed event: https://handbook.fide.com/chapter/C0401Till2026

For CurvyTron, this means: prefer close matches most of the time, but do not
only do close matches. New checkpoints need anchors and random bridges too.

## Immediate Next Steps

1. Finish the research/design pass.
2. Add an adaptive pair-selection spec to the docs.
3. Add a small pure scheduler helper and tests.
4. Add CLI flags only after the helper contract is stable.
5. Smoke a tiny adaptive run with explicit checkpoint refs.
6. Then decide whether to run a larger all-checkpoint adaptive job.

## Guardrails

- Do not launch another full all-pairs job unless the user explicitly asks for a
  stress test.
- Use meaningful names that include player set, scheduler, games per pair, and
  max step cap.
- Keep official score tournaments in eval/greedy mode unless the coach asks for
  a separate collect-mode diagnostic.
- Keep GIFs off for large rating jobs unless there is a clear sample cap.
- Keep training separate. This lane reads checkpoints; it does not touch the
  trainer.
