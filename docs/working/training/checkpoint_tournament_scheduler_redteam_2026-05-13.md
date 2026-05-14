# Checkpoint Tournament Scheduler Red Team, 2026-05-13

## Scope

Critique the current adaptive Elo checkpoint scheduler before it becomes the
steady-state CurvyTron tournament lane.

Inspected:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- `tests/test_curvytron_checkpoint_tournament.py`
- current tournament scheduling/orchestration/critique docs under
  `docs/working/training/`

No production code was edited.

## Current Shape

The intended shape is right for V0: static batch waves, immutable Modal battle
artifacts, batch Elo reduction, pair history, scheduler state, and schedule
reasons.

The dangerous part is that the scheduler is now part of the measurement
instrument. It is not a neutral round robin. If the website or future frozen
opponent sampler shows only Elo/rank, the adaptive schedule can create confident
but narrow evidence.

## Local Simulations Run

These probes called the existing scheduler/rating helpers directly.

| Probe | Result |
| --- | --- |
| 424 all-new checkpoints, `pairs_per_round=212`, default 20 placement opponents | scheduled 4,240 placement pairs, 89,040 games; every checkpoint got 20 appearances |
| 900 new + 100 established, one placement opponent | scheduled 900 placement pairs; exactly one established checkpoint got all 900 appearances |
| 80 checkpoints from one lineage + 20 singleton families, no placement deficits | 100-pair wave had 66 same-family pairs |
| 100 established checkpoints, 40-pair top-biased wave | top 20 received 57.5% of appearances; 56 checkpoints got zero appearances |
| 60 established checkpoints over 5 rounds, fixed ratings | prior-pair slots by round: 0, 1, 5, 5, 7; repeats begin while many pairs remain unplayed |
| active-status threshold check | one checkpoint became `active` with 315 games but only 5 distinct opponents |
| 3-policy rock-paper-scissors Elo | full cycle returns all equal; missing one edge crowns A at 1510.369 and buries C at 1489.631 |

## Findings

### 1. Placement Can Blow Past The Requested Budget

`adaptive_v0` treats coverage as a hard floor and expands beyond
`pairs_per_round` when the placement deficit is large. That matches "coverage
first", but it can surprise operations.

Concrete failure: a requested 212-pair wave over 424 new checkpoints becomes
4,240 pairs at the default 20-opponent placement target.

Recommendation: keep coverage-first, but report and cap expansion explicitly:
`requested_pairs`, `coverage_floor_pairs`, `scheduled_pairs`,
`budget_expansion_factor`, and a hard operator max.

Test: 424 and 5,000 all-new synthetic rosters must either stay under a declared
max expansion or fail loudly before Modal work is spawned.

### 2. Early Top Checkpoints Can Become Placement Sinks

Placement candidates prefer high-rated established opponents and there is no
appearance cap in the placement pass. In the 900-new simulation, every new
checkpoint played the same top established checkpoint.

This creates three problems:

- one lucky early leader attracts too much compute;
- every new policy gets the same first opponent, so placement coverage looks
wide in games but narrow in opponent profile;
- the top checkpoint's rating can become a referendum on new-checkpoint quality
instead of its own strength.

Recommendation: add a placement appearance cap per established checkpoint and
spread new entrants across several anchors/rating bands.

Test: in a 900-new/100-established roster, no established checkpoint should
receive more than a configured share of placement appearances.

### 3. Same-Run Lineages Can Dominate Evidence

The scheduler has checkpoint ids and refs, but no explicit run/lineage/family
metadata in the selection rules. With 80 checkpoints from one lineage and 20
other families, a 100-pair no-placement wave scheduled 66% same-family pairs.

Lineage comparisons are useful, but they are not enough to trust a leaderboard.
Many checkpoints from one run can make the graph look dense while mostly
measuring one training trajectory against itself.

Recommendation: track `run_id`/`lineage_id`, cap same-lineage share, and require
outside-lineage opponents before a row can become trusted.

Test: a synthetic pool where one run contributes 80% of checkpoints must not
spend most of the adaptive budget inside that run unless explicitly requested.

### 4. Top-Band Bias Can Starve The Rest Once Placement Is Satisfied

The top-band boost is smooth, not a hard cutoff, which is good. But once
placement deficits are zero, a small wave can still concentrate evidence heavily
near the top. In the 100-policy/40-pair probe, top 20 policies received 57.5%
of all appearances and 56 policies were absent.

That may be acceptable for a "find the best" wave, but not for a general
leaderboard maintenance wave.

Recommendation: label wave intent: `leaderboard_breadth`, `top_k_refinement`,
or `audit`. Breadth waves need appearance floors; top-k waves need explicit
warnings that lower rows were not refreshed.

Test: top-band waves should publish appearance distribution by rank bucket and
zero-appearance count.

### 5. Active Status Is Too Weak For The Current Product Goal

Code currently marks rows active at `games >= 300` and
`distinct_opponents >= 5`. Current docs say the public/frozen-opponent future
needs about 20 distinct opponents before leaderboard-active trust.

The simulation produced an active row with 315 games and only 5 opponents. That
is too narrow for a public row or training consumer.

Recommendation: align status with the current gate. At minimum, publish
`status_reasons` so a 5-opponent row cannot look equivalent to a 20-opponent
row.

Test: a checkpoint with 315 games against 5 opponents remains provisional or
`audit_needed`; a 20-opponent row can become active if other gates pass.

### 6. Pair History Penalizes Repeats But Does Not Prevent Them

Repeat pair priority is reduced, but repeats are still legal as soon as the next
round starts. In the 60-policy probe, repeats appeared by round 1 and grew to 7
repeat slots by round 4.

Some repeats are useful, especially close or stale pairs. The failure mode is
spending repeat budget before coverage, bridges, and audits are healthy.

Recommendation: make repeats a named quota with a stale/close/noisy reason.
Until coverage is met, disallow repeats except explicit audits.

Test: over N rounds, assert repeat share stays below a configured quota while
any checkpoint is under the distinct-opponent floor.

### 7. Scalar Elo Can Hide Non-Transitive Matchups

The RPS probe is the smallest example:

- A beats B, B beats C, C beats A: batch Elo returns all equal.
- If the schedule misses C beats A, A becomes the leader and C becomes last.

CurvyTron policies can easily be non-transitive: wall avoiders, aggressive
cutoff policies, bonus-seekers, and collapse sentinels may form matchup cycles.

Recommendation: add a cycle/non-transitivity report beside Elo. For top rows,
run mandatory cross-style audit pairs, not only near-rating pairs.

Test: inject a three-cluster RPS outcome model and assert the report flags the
cycle instead of presenting a clean scalar ranking.

### 8. Weak Policies May Never Find Their Level

Strong-first placement is useful for detecting good new checkpoints, but weak
new checkpoints can get crushed by top anchors and remain poorly localized. They
need lower anchors or nearby weak policies after the first update.

Recommendation: placement should include a spread: top anchor, median anchor,
low anchor, random bridge, and then near-rating after a provisional update.

Test: synthetic weak entrant should get at least one low/median opponent before
being deprioritized.

### 9. Seat Bias Is Not First-Class In Pair History

Odd battle sizes avoid even-count W/L ties, but they do not cancel seat
advantage. Pair history is keyed by unordered checkpoint pair and mostly carries
checkpoint-level wins. Seat-specific evidence can be lost as the schedule ages.

Recommendation: keep seat-specific counts in pair history and publish global
seat win-rate by evaluator context. Use mirrored battles for top/close/audit
pairs until seat bias is measured.

Test: inject a fixed seat-0 advantage and verify the summary flags it.

### 10. Draws And Timeouts Can Make Bad Games Look Stable

Timeout draws count as 0.5 in Elo. A draw can mean equal skill, both agents are
good, both are broken, or max steps are too low. The current status gate does
not include draw/timeout rate.

Recommendation: separate normal draws from timeout draws in row evidence and
schedule high-timeout pairs for GIF/audit review.

Test: synthetic high-timeout rows become `audit_needed`, not `stable active`.

### 11. Online Insertion Can Stall Or Reset

The intake drain skips spawning when a rating run already has output unless
explicitly allowed. If forced to spawn, the intake default for
`continue_from_latest` is false, so a later wave can carry pair history but not
carry the latest Elo snapshot unless the operator remembered the flag.

That is risky for continuous checkpoint injection.

Recommendation: for an ongoing rating run, make `continue_from_latest=true` the
safe default, or use a new explicit wave id plus a documented import path. New
events should not be silently drained into no spawned work.

Test: seed A/B, finish a rating, inject C, drain, and assert the next schedule
uses A/B's latest ratings and gives C placement games.

### 12. Stale Elo Can Look Settled

The `stable` flag is based on max rating delta. A narrow graph can have low
delta because it keeps replaying local pairs or because no hard bridge games
are scheduled. Low delta is not the same as enough evidence.

Recommendation: stable should require graph health: minimum distinct opponents,
outside-lineage opponents, bridge coverage, low failure rate, and no high-draw
audit flags.

Test: a two-island graph with low deltas remains provisional/stale until bridge
pairs connect it.

## Suggested Test/Simulation Set

- Budget expansion: all-new 424 and 5,000 checkpoint rosters.
- Placement sink: many new checkpoints plus a small established pool.
- Same-lineage cap: one long run contributes most checkpoints.
- Top-band skew: measure appearance share by rank bucket.
- Coverage starvation: ensure zero-appearance count is acceptable for the wave
  type.
- Repeat quota: multi-round schedules cannot repeat early while coverage is low.
- Online insertion: A/B existing ladder plus newly injected C.
- Active-status threshold: 5 opponents is not trusted if the target is 20.
- Seat-bias injection: fixed seat advantage must be detected.
- Timeout/draw injection: high timeout rows become audit targets.
- Non-transitive clusters: RPS cycle is flagged.
- Held-out random audit: compare adaptive top ranks against random audit pairs.

## Near-Term Recommendation

Keep `adaptive_v0`, but make the next iteration brutally explicit:

- hard operator max for coverage expansion;
- placement appearance caps;
- lineage metadata and outside-lineage gates;
- active threshold aligned with 20 distinct opponents;
- bridge/repeat/top-k quotas in scheduler state;
- row evidence fields for status reasons, draw/timeout rate, seat rate, and
  freshness;
- a small held-out random audit every few waves.

That keeps the scheduler simple while making it harder for a lucky early leader,
a long checkpoint lineage, or a sparse comparison graph to masquerade as a
trusted leaderboard.
