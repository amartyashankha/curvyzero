# Checkpoint Tournament Scheduling Research, 2026-05-13

## Purpose

Design a practical adaptive scheduler for CurvyTron checkpoint ratings as new
checkpoints and policies arrive over time. The immediate goal is not a perfect
rating theory. It is a small scheduler that avoids quadratic all-pairs runs for
200-300 checkpoints, keeps Modal artifacts lean, and still gives coach-useful
rankings.

## 2026-05-13 Pivot

The target is now bigger than "latest checkpoint from each run."

We want to rate every useful checkpoint from every run. That means all-pairs is
not the normal plan anymore. It can still be used as a stress test or audit, but
the permanent lane should be online/adaptive:

- new checkpoints enter as provisional players;
- they play a bounded placement slate;
- later rounds spend games on nearby ratings, uncertain players, anchors,
  random bridges, and stale or noisy rematches;
- rankings should be visible while games are still running;
- ratings must be recomputable from immutable battle summaries.

The orchestration doc for this pivot is
`checkpoint_tournament_orchestration_2026-05-13.md`.

## Fresh Research Notes

- Glicko-2 is useful as a design reference because it treats rating confidence
  as a first-class value. The V0 can fake this with games played, distinct
  opponents, last rating movement, failure rate, and provisional status before
  we implement full rating deviation.
- TrueSkill is useful as a scheduling reference because it talks about match
  quality. Close matches are usually good, but a new checkpoint can also learn a
  lot from playing a strong established checkpoint.
- Swiss pairing is a useful tournament analogy: pair nearby performers and avoid
  pointless repeats. CurvyTron is different because we run continuous offline
  rating, so some repeats are good when a battle is stale, close, or noisy.

## Research Notes

Common ladder systems do three things we should copy:

- They track confidence separately from rating. Lichess uses Glicko-2 and marks
  ratings provisional when rating deviation is high; Chess.com also describes
  rating deviation as confidence in the rating. We can approximate this in Elo
  V0 with games, distinct opponents, recency, and recent delta.
- They prefer balanced matches. TrueSkill frames matchmaking around match
  quality and notes the tradeoff between waiting time and tighter matches. For
  offline Modal scheduling, waiting time is irrelevant, so near-rating pairs are
  cheap high-signal work.
- They keep rating pools local. Lichess explicitly warns that rating numbers are
  only meaningful inside the same pool. For us, env variant, reward variant,
  policy mode, seat policy, and checkpoint cohort define the pool.

Tournament systems add useful constraints:

- Swiss systems pair players with similar scores and avoid repeat pairings; this
  is a good analogy for repeated rating rounds.
- FIDE-style Elo uses high development coefficient for new players until enough
  games are played, then lower K. Our current `k_reference_games` path already
  points in this direction.
- The European Go Federation computes tournament updates from pre-tournament
  ratings, not game-by-game changed ratings. That supports our batch-update
  choice and avoids arbitrary Modal completion-order bias.

Active evaluation literature says the same thing in ML language: all-pairs is
quadratic, while actively choosing pairwise comparisons can identify strong
systems with far fewer comparisons. Treat that as V1 guidance, not a dependency.

## Scheduling Patterns

Random subset of all pairs:

- Good as a sanity baseline and to keep the comparison graph connected.
- Bad as the only strategy: wastes games on obvious mismatches once rough
  ratings exist.
- Use deterministic seeded sampling, stratified by rating band, and enforce
  minimum distinct opponents per checkpoint.

Provisional placement matches:

- Every new checkpoint starts provisional.
- Give it a first slate against anchors and nearby lineage checkpoints before it
  enters normal near-rating scheduling.
- Suggested first slate: previous checkpoint from same run, current best, current
  median, current lower-quartile, one fixed baseline if available, and 3-6
  random established checkpoints.
- Use both seat orders for placement if seat bias is still unquantified.

Anchors:

- Keep a small frozen anchor set in every long-lived rating pool.
- Include hand-designed baselines only if they are legal under the same env and
  policy interface; otherwise use frozen historical checkpoints.
- Useful anchor types: oldest known baseline, last known good checkpoint, current
  champion, median stable checkpoint, collapse/failure sentinel.
- Monitor anchor rating movement. If anchors drift together, the pool is moving;
  if one anchor moves strangely, suspect variance or a changed game contract.

Near-rating matches:

- Best default once a checkpoint has placement results.
- Pairs with expected score near 0.5 produce more ranking information than
  crushing mismatches.
- Avoid only-near scheduling forever, because it can split the graph into local
  islands. Mix in anchors and random bridges.

Uncertainty-driven scheduling:

- V0 uncertainty proxy: provisional flag, low valid games, low distinct
  opponents, high last-round delta, stale/old rating, or high failure rate.
- Schedule uncertain checkpoints more often, especially against established
  non-provisional opponents near their current rating.
- Do not overreact to one noisy pair; schedule in pair batches and update ratings
  after the batch.

Exploration/exploitation mix:

- Exploitation: near-rating pairs and top-vs-near-top matches.
- Exploration: random cross-band bridges, anchors, and occasional rematches for
  variance estimates.
- Suggested V0 mix per round after placement:
  - 60% near-rating pairs
  - 20% provisional/uncertain checkpoint pairs
  - 10% anchors
  - 10% seeded random bridges

Avoiding drift:

- Never mix incompatible pools: eval vs collect, env variant, reward variant,
  two-seat contract, seat-order policy, and major evaluator changes should start
  a new rating pool or carry explicit bridge runs.
- Recompute from immutable pair summaries when possible; do not let online update
  order define the truth.
- Keep anchors and overlap checkpoints between cohorts.
- Publish rating rows with `provisional`/`active` status and minimum evidence
  thresholds, not only rank.

## Recommended V0

Keep batch Elo. Add adaptive scheduling as pair selection, not as a new rating
algorithm.

Important implementation rule: for very large pools, do not build every possible
pair and then sample. Generate the needed candidates from rating bands, anchors,
lineage neighbors, random bridges, and replay queues directly.

1. Maintain per-checkpoint scheduling metadata derived from latest ratings:
   rating, games, battles, rated battles, distinct opponents, last round delta,
   failure count, provisional/active status.
2. Maintain pair history by stable checkpoint-pair key:
   battle count, total games, last round, last result, failures, draws, and
   latest battle refs.
3. New checkpoint enters with initial rating from its nearest predecessor if the
   lineage is known; otherwise use pool median or 1500.
4. Run placement batches first. Target 8-12 opponents, 10-50 games per opponent,
   both seats if needed.
5. After placement, select normal rounds with the 60/20/10/10 mix above.
6. Enforce graph health:
   minimum 5 distinct opponents before active status,
   at least one anchor match per checkpoint family,
   at least one random bridge per rating band per round.
7. Keep batch reduction: build pair specs, run Modal shards, reduce once, write a
   slim latest snapshot.
8. Stop or pause when all non-provisional checkpoints have enough games and
   `max_abs_delta` stays below the threshold for two consecutive rounds.

V0 output should remain simple:

- `input.json`: scheduled pair specs plus `schedule_reason`.
- `progress.json`: counts by schedule reason, provisional count, active count.
- `ratings.json`/`latest.json`: ratings plus provisional/active status.
- Optional compact `schedule_summary.json`: no per-game details.

## Recommended V1

Move from Elo-with-proxies toward explicit uncertainty.

- Add Glicko-style rating deviation or a small custom uncertainty estimate.
- Score candidate pairs by expected information:
  closeness to 50/50, participant uncertainty, anchor coverage, graph
  connectivity, and exploration quota.
- Use TrueSkill-like match quality as a pair ranking feature, not necessarily as
  the rating algorithm.
- Consider a top-K active-evaluation mode when the question is "which checkpoint
  is best?" rather than "rank everything accurately."
- Model draw/timeouts if they become common enough to bias Elo.
- Add retrospective recompute tools so V1 can rerun ratings from all immutable
  pair tallies.

## Practical Defaults

- Placement opponents: 8 established checkpoints.
- Games per placement pair: use odd battle sizes for current product runs. Start
  with 11 or 21, then top up close pairs to 51 if the result matters.
- Normal round budget: fixed pair count, not all-pairs. Start with
  `min(4 * checkpoint_count, 2000)` pairs per round for 200-300 checkpoint pools.
- Active status: at least 300 valid games, at least 5 distinct opponents, and
  not in the top recent-delta bucket.
- Anchor share: 10% of scheduled pairs, capped so anchors do not dominate.
- Random bridge share: 10%, stratified by rating decile.

## 211-Run First Real Job Addendum

Assume cleanup leaves 211 valid latest checkpoints. The full unordered no-self
pool is `211 * 210 / 2 = 22,155` pairs. Do not make the first real post-cleanup
job all-pairs unless the goal is specifically a burn-in stress test:

- all-pairs, 21 games/pair: 465,255 games
- all-pairs, 51 games/pair: 1,130,005 games
- with `games_per_shard` equal to `games_per_pair`, those are 22,155 shard calls

Recommended first launch: a conservative random subset that the current
Modal/Elo system can explain easily.

- Pair selection: seeded random subset of 2,000 unordered pairs.
- Games: 21 games per pair.
- Sharding: `games_per_shard=21`.
- Estimated work: 42,000 games, 2,000 shard calls, about 19 pair appearances per
  checkpoint on average, about 379 game participations per checkpoint on
  average.
- Artifacts: GIFs off, frames off, lean rating artifacts preferred.
- Why this is conservative: it is smaller than the completed 50-checkpoint
  all-pairs/gpp50 run in game count, while touching the whole 211-checkpoint
  pool with a connected random graph.
- Validation target: no missing checkpoints, no systemic failures, all
  checkpoints get at least 8 distinct opponents after the run. If a few do not,
  run a small top-up round.

If the team wants an even safer first step, use 1,055 random pairs, 21
games/pair, `games_per_shard=21`: 22,155 games and 1,055 shard calls. That is a
good operational smoke, but it is less useful as the first real ladder because
the average checkpoint only sees about 10 opponents.

More aggressive adaptive/provisional first plan:

1. Evidence-floor round:
   - Target 8 distinct opponent slots per checkpoint.
   - Use a regular-ish seeded graph, biased to include lineage neighbors and a
     small anchor set.
   - Approximate pair count if balanced: `211 * 8 / 2 = 844`.
   - At 21 games/pair and `games_per_shard=21`: 17,724 games, 844 shard calls.
2. Adaptive expansion round:
   - Add 1,500 pairs after the first reduction.
   - Mix: 60% near-rating, 20% provisional/high-delta, 10% anchors, 10% random
     bridges.
   - At 21 games/pair and `games_per_shard=21`: 31,500 games, 1,500 shard calls.
3. Combined aggressive first ladder:
   - About 2,344 pairs, 49,224 games, 2,344 shard calls.
   - If anchor placement is implemented as every checkpoint playing many fixed
     anchors, pair count may rise toward 2,500-3,000; at 21 games/pair this is
     still 52,500-63,000 games and 2,500-3,000 shard calls.

Use the conservative random subset if adaptive scheduling is not implemented or
if the cleanup may still contain questionable checkpoints. Use the aggressive
plan if we trust checkpoint discovery and already have a small compatible anchor
set.

## Latest-212 All-Pairs Stress Run

2026-05-13 user direction was to try the simple full matrix after cleanup:
latest checkpoint from every preserved run, all unordered pairs, and no stale
tournament artifacts visible.

Chosen launch shape after correction:

- Checkpoints: 212 latest refs from the preserved prune manifest.
- Pair selection: all unordered no-self pairs.
- Pairs: 22,366.
- Games per pair: 11.
- Games per shard: 11.
- Total games: 246,026.
- GIF samples: 3 per pair, 67,098 total.

Why this shape:

- It matches the user's direct "all latest checkpoints" request.
- It keeps one shard per battle, which avoids 223,660 tiny Modal calls.
- It keeps the three-sample GIF pattern already used by the website and smoke
  tests: games 0, middle, and latest equivalent for an 11-game battle
  when using `evenly_spaced`.
- It is still a stress run, not the long-term scheduler. The longer-term lane
  should use placement, anchors, random bridges, and near-rating pairs instead
  of repeating all-pairs at every scale.
- The first `gpp10` attempt was stopped and purged because even battle sizes are
  now invalid for new runs.

GIF sampling for the 211-run first job:

- Main rating job: `save_gif=false`.
- Do not enable one GIF per pair on a 2,000-pair job unless the scheduler has a
  global GIF cap; that would create 2,000 GIFs for little rating value.
- Before the main job, run a tiny GIF canary: 5-10 handpicked pairs, 1 GIF per
  pair, both seats if seat bias is under review.
- After the rating reduction, run targeted GIF sampling:
  - 10 top near-rating pairs
  - 10 anchor or placement pairs
  - 10 largest upsets
  - 10 high-failure or strange-draw pairs
  - optional 10 random bridge pairs
- Practical cap: 30-50 GIFs total for the first real 211-run ladder.

## Checkpoint Sampling For The 200+ Run Tournament

Immediate recommendation: use the latest real checkpoint per run only for the
first 200+ run rating job. Do not include every checkpoint from every run yet.

Why latest-only first:

- It answers the immediate coach question: which current runs are best now?
- It keeps one rating row per run, so the first ladder is readable.
- It avoids overweighting long or chatty runs that happened to save more
  checkpoints.
- It works with current prefix discovery, which finds the highest
  `iteration_<n>.pth.tar` for each run.

Multiple checkpoints per run are useful, but they should be a second layer, not
the first ladder. If we add two checkpoints for every one of 211 runs, the pool
becomes 422 players and full all-pairs jumps from 22,155 to 88,831 pairs. Three
per run becomes 633 players and 200,028 pairs. Even with random pair sampling,
that makes the first ranking harder to explain.

Minimal useful multi-checkpoint sample:

- Start with latest checkpoints for all valid runs.
- Add 25-50 extra historical checkpoints total, not per run.
- Pick extras from runs we actually want to understand:
  - previous checkpoint from likely top runs
  - mid-training checkpoint from a few long-running lineages
  - known collapse/failure checkpoints as sentinels
  - one or two handpicked old anchors from prior trusted tournaments
- Keep the combined player pool near 236-261 checkpoints for the first mixed
  run. At 3,000 sampled pairs, that still gives about 23-25 opponents per
  checkpoint on average.
- Do not add more than one non-latest checkpoint from the same run in the first
  mixed pool unless debugging that run's training trajectory is the primary
  question.

Handling runs still producing checkpoints:

- Freeze the tournament pool at launch time. A run that creates a newer
  checkpoint ten minutes later should not mutate this tournament.
- Use concrete checkpoint refs in the rating spec after discovery; the refs point
  at specific `iteration_<n>.pth.tar` files.
- If a run is still warming up and has no real `.pth.tar`, skip it or delay the
  launch. Do not include resume sidecars.
- Record the cutoff time, run prefix, expected count, and discovery found/missing
  count in the launch note.
- Treat checkpoints produced after the cutoff as new provisional entrants in the
  next rating round or next tournament.

Interaction with later adaptive Elo:

- The latest-only 200+ tournament becomes the baseline rating pool.
- New later checkpoints enter as provisional players with placement matches
  against anchors, their own run predecessor, and nearby-rated established
  checkpoints.
- Historical checkpoints should mostly act as anchors or lineage diagnostics,
  not compete equally with every current latest checkpoint forever.
- Adaptive scheduling should cap per-run lineage representation so one training
  run cannot dominate the pair budget.
- When multiple checkpoints from one run are included, make sure they get at
  least one direct lineage comparison, but keep most games against outside runs.

Concrete first aggressive-but-sane launch over about 200 checkpoints:

1. Discovery smoke:

```text
uv run --extra modal python -B -m modal run --detach -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode discover \
  --run-id-prefix <RUN_PREFIX> \
  --max-runs 211
```

Check that `found_count` is the expected cleaned run count and `missing_count` is
zero. If the cleaned count is not exactly 211, use the actual expected count in
the next commands.

2. Estimate:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode estimate \
  --run-id-prefix <RUN_PREFIX> \
  --max-runs 211 \
  --expected-checkpoint-count 211 \
  --games-per-pair 21 \
  --games-per-shard 21 \
  --pairs-per-round 3000 \
  --pair-selection random \
  --gif-sample-games-per-pair 0 \
  --seed 211013
```

Important launch rule: long rating/tournament jobs must use Modal
`run --detach` when `--wait` is not used. Without it, the local entrypoint can
finish before the spawned rating loop gets a container, leaving only a stale
`waiting_for_round_input` progress file.

Expected shape for 211 checkpoints:

- sampled pairs: 3,000
- games: 63,000
- shard calls at `games_per_shard=21`: 3,000
- average opponents per checkpoint: about 28
- average game participations per checkpoint: about 569

This is aggressive because it is a real ladder-size run, but sane because it is
still far below 465,255 games for all-pairs at 21 games/pair.

3. Launch:

```text
uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode rating \
  --tournament-id arena-rating-200plus-latest-random-20260513a \
  --rating-run-id elo-latest-rand3000-gpp21 \
  --run-id-prefix <RUN_PREFIX> \
  --max-runs 211 \
  --expected-checkpoint-count 211 \
  --games-per-pair 21 \
  --games-per-shard 21 \
  --pairs-per-round 3000 \
  --pair-selection random \
  --gif-sample-games-per-pair 0 \
  --seed 211013
```

If cleanup confidence is lower, reduce to `--pairs-per-round 2000` for 42,000
games and 2,000 shard calls. If the first 2,000-pair run is clean but too sparse,
run a second random bridge/top-up round rather than restarting with all-pairs.

For a first mixed latest-plus-history pool, do not use prefix discovery alone.
Build an explicit comma-separated `--checkpoint-refs` list containing all latest
refs plus the 25-50 selected historical refs, then use the same random-pair
parameters. Keep `--pairs-per-round` at 3,000-3,500 and keep GIFs off.

## 2026-05-13 Research Refresh

- Elo-like systems do not require every pair to play every other pair. They need
  a connected enough comparison graph. Use all-pairs as a clean baseline or
  stress test, not as the permanent continuous scheduler.
- TrueSkill-style matchmaking points toward similarly rated opponents when
  possible. Those games are usually more informative than repeated obvious
  mismatches.
- Glicko-style systems track uncertainty separately from rating. Our V0 can
  approximate this with games played, distinct opponents, last-round rating
  movement, and provisional status before adding full Glicko math.
- Swiss-style tournaments are a useful scheduling analogy: pair near current
  score/rating, avoid repeats, and get useful rankings with far fewer than all
  possible games.
- Caveat: a research pass noted that seat-balanced batches can be statistically
  cleaner than odd battle sizes if seat bias exists. Current product rule is
  still odd battle sizes, but keep seat-bias checks on the evaluation-contract
  list.

## Open Questions

- How much seat bias remains? If it is non-trivial, scheduling should treat
  seat-swapped games as part of the standard pair batch.
- Should predecessor lineage matter strongly, or should every new checkpoint use
  pool median until proven otherwise?
- What is the right active threshold for coach decisions: top checkpoint only,
  top 5, or full ordering?
- Are CurvyTron outcomes transitive enough for one-dimensional Elo, or do we need
  matchup-cluster diagnostics?
- How many anchors should be fixed forever versus refreshed by cohort?
- Should failed/collapsed checkpoints stay in the pool as sentinels or be moved
  to a diagnostic-only pool?

## Adaptive Elo Follow-Up

- Default battle size should be 21 games per pair for the next generation of
  runs. Keep the odd-count rule.
- Default score-game cap should be long enough that timeouts are rare. Current
  code default is 8,000 decision steps; if timeout draws remain common, treat
  that as a real scheduling/correctness signal to investigate, not as a reason
  to silently lower the cap.
- The permanent system should not rely on full all-pairs runs. Treat all-pairs
  as a clean stress test and occasional audit. The steady-state scheduler should
  add new checkpoints, give them enough games against the existing pool to land
  roughly correctly, then spend most new games on uncertain or important rank
  boundaries.
- Simple V1 scheduler idea:
  - every new checkpoint plays a few fixed anchors across the rating range;
  - then it plays nearby ratings after the first provisional update;
  - replay older existing pairs when they are stale, close in rating, or have a
    high failure/noise rate;
  - keep random exploration pairs so the graph stays connected and surprises
    can surface.
- Do not make this complex before the website and current batch tournament path
  are boringly reliable.

## Sources

- Lichess FAQ, Glicko-2, provisional ratings, and rating pool cautions:
  https://lichess.org/faq
- Lichess rating systems overview:
  https://lichess.org/page/rating-systems
- Chess.com rating deviation explanation:
  https://support.chess.com/en/articles/8566476-how-do-ratings-work-on-chess-com
- Microsoft TrueSkill overview and match-quality notes:
  https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
- TrueSkill paper summary:
  https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system-2/
- FIDE Swiss basic rules:
  https://handbook.fide.com/chapter/C0401Till2026
- FIDE rating regulations, initial rating and K-factor:
  https://doc.fide.com/docs/DOC/3FC2023/FC3_2023_25.pdf
- European Go Federation rating system:
  https://europeangodatabase.eu/EGD/EGF_rating_system.php
- Active Evaluation pairwise-comparison research:
  https://arxiv.org/abs/2203.06063
- Dueling-bandit Elo scheduling research:
  https://arxiv.org/abs/2201.04480

## Current Design Note, 2026-05-13

Research basis:

- TrueSkill and Glicko both say rating confidence matters, not just rating.
  For V0, keep batch Elo and use simple confidence proxies: valid games,
  distinct opponents, last-round delta, failure rate, age, and provisional
  status.
- FIDE Swiss rules are the best tournament analogy: pair similar scores,
  avoid repeats early, keep pairings explainable, and declare the schedule
  before a round starts.
- Active Evaluation shows the same ML shape: all-pairs grows as `k choose 2`,
  while actively chosen pairwise comparisons can find strong systems with far
  fewer comparisons.
- Modal wants independent inputs. The scheduler should make a static round
  manifest, then let workers run shards without sharing mutable rating state.

### 1. Placing New Checkpoints

Every new checkpoint starts `provisional`.

Initial rating:

- Use the previous checkpoint from the same run if lineage is known.
- Otherwise use the active pool median.
- If there is no active pool yet, use `1500`.

Placement slate:

- Previous checkpoint from the same run.
- Current champion.
- Current median active checkpoint.
- Current lower-quartile active checkpoint.
- One stable low baseline or collapse sentinel.
- Three to six seeded random active checkpoints.

Use 8-12 opponents for placement. Use 21 games per opponent by default. Keep the
checkpoint provisional until it has at least 300 valid games and at least five
distinct opponents. If seat bias is not proven small, schedule a seat-swapped
companion battle for placement pairs and report seat-order effects separately.

### 2. Choosing Opponents Over Time

Do not make all-pairs the steady-state scheduler. Use fixed-size rounds.

Suggested post-placement mix:

- 60% near-rating pairs. These are high-signal because expected score is near
  0.5.
- 20% provisional or uncertain checkpoints. Uncertain means low games, low
  distinct opponents, high last-round delta, high failure rate, or stale rating.
- 10% anchor pairs. Anchors keep cohorts comparable.
- 10% random bridges across rating bands. Bridges keep the graph connected and
  catch surprises.

Candidate pair rules:

- Prefer opponents with similar rating unless the pair is an anchor or bridge.
- Avoid repeats until both checkpoints meet the distinct-opponent floor.
- Do not let one long run dominate the budget with many checkpoints from the
  same lineage.
- Keep rating pools separate for eval/collect mode, env variant, reward variant,
  major evaluator changes, and seat-policy changes.
- Write `schedule_reason` on each pair: `placement`, `near_rating`,
  `uncertain`, `anchor`, `random_bridge`, or `replay`.

### 3. Games Per Battle

Keep 21 games per battle as the default.

Why:

- The current code already defaults to 21 and rejects even battle sizes.
- Odd counts avoid a pure win/loss half split. Draws can still create score
  ties, so this is not a complete tie solution.
- 21 is a useful scheduler signal without making every pair expensive. With no
  draws, a true 60% player wins the majority of an 11-game battle about 75% of
  the time, a 21-game battle about 83% of the time, and a 51-game battle about
  93% of the time.
- At a 50/50 matchup, the raw win-rate standard error is about 0.109 for 21
  games, so one battle is not final truth. It is enough to guide the next round.

Top up important close pairs to 51 or 101 games when the result changes a
promotion, top-10 ordering, or anchor interpretation. Do not spend 51 games on
obvious blowouts unless auditing.

### 4. Replaying Existing Battles

Never overwrite an old battle. Replay means a new battle id, new seeds, and a
link to the earlier pair.

Replay when:

- The pair is close and rank-important.
- The old result is stale after many new rounds.
- Either checkpoint has high uncertainty or a large recent rating delta.
- The battle had many failures, invalid games, or unexpected draws/timeouts.
- An anchor moves strangely and needs confirmation.
- A new evaluator, env contract, or seat policy needs a bridge audit.

Do not replay stable, old blowouts just because they are old. Keep old immutable
summaries so ratings can be recomputed with or without replay batches.

### 5. Keeping Modal Parallel

Keep the round embarrassingly parallel:

- Scheduler writes one `input.json` with static `pair_specs`.
- Each pair expands to independent games or shards.
- Prefer `games_per_shard=21` for a 21-game battle so one Modal shard can reuse
  loaded policies and commit once.
- Workers write only unique battle/game/shard paths.
- Workers do not update ratings.
- One reducer reads committed summaries and writes `results.json`,
  `ratings.json`, `latest.json`, and `progress.json`.
- Keep GIFs and frames off for rating rounds. Run tiny GIF canaries or targeted
  post-rating GIF samples.
- If a round exceeds Modal map/input limits, split the same round manifest into
  deterministic chunks and reduce only after all chunks finish.

This matches the current Modal shape: independent `map(..., order_outputs=False)`
work, sharded tallies, and a single reducer. It also avoids Volume last-write
conflicts by giving each worker unique files.

### 6. Files To Update Later

Code:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`
  - Add adaptive pair selection choices and scheduler fields.
  - Add `schedule_reason`, anchor ids, replay metadata, and deterministic
    candidate scoring.
  - Keep batch Elo reduction.
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
  - Add CLI fields for adaptive scheduling, anchors, placement mode, replay
    mode, and schedule-summary artifacts.
  - Keep the worker path static and parallel.
- `tests/test_curvytron_checkpoint_tournament.py`
  - Cover deterministic adaptive selection, placement slates, no-repeat floors,
    schedule reasons, replay metadata, graph health, and stable reduction from
    immutable summaries.

Docs:

- `docs/working/training/checkpoint_tournament_scheduling_research_2026-05-13.md`
  - This note was appended here.
- `docs/working/training/checkpoint_tournament_elo_followup_2026-05-13.md`
  - Add operator defaults and launch examples after implementation.
- `docs/working/training/checkpoint_tournament_todo_2026-05-13.md`
  - Add implementation tasks and validation gates.
- `docs/working/training/checkpoint_tournament_validation_2026-05-13.md`
  - Add scheduler smoke tests, replay-reduce tests, and Modal chunking checks.

Source links used in this refresh:

- TrueSkill overview and match quality:
  https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
- TrueSkill paper:
  https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system-2/
- Glicko-2 rating, RD, volatility, and rating periods:
  https://www.glicko.net/glicko/glicko2.pdf
- FIDE Swiss rules applied from 1 February 2026:
  https://doc.fide.com/docs/DOC/2025_3FC/CM3-202517.pdf
- FIDE rating K-factor rules:
  https://handbook.fide.com/chapter/B022024
- Active Evaluation:
  https://aclanthology.org/2022.acl-long.600/
- Modal scale guide:
  https://modal.com/docs/guide/scale
- Modal Volume consistency guide:
  https://modal.com/docs/guide/volumes
