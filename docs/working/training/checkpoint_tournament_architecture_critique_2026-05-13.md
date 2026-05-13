# Checkpoint Tournament Architecture Critique, 2026-05-13

## Purpose

Keep poking holes in the tournament system before we scale it.

The full flow is now:

1. discover checkpoints;
2. choose adaptive pairs;
3. run Modal game shards;
4. reduce battle results;
5. update Elo;
6. write artifacts;
7. show rankings, battles, and GIFs on the website;
8. help the coach decide what is actually improving.

This is complex enough that every change needs a failure-mode pass.

## Current Critique Lanes

- Rating correctness: selection bias, non-transitive matchups, seat bias,
  draw/timeouts, lineage overweighting, and provisional status.
- Code architecture: bounded scheduler, explicit schedule metadata, pair
  history, recompute path, and no regressions to `all_pairs`/`random`.
- Modal operations: autoscaling, retries, timeouts, commit/reload behavior, and
  whether Queue/Dict help.
- Website: freshness, paging, click latency, partial rankings, GIF drilldown,
  and coach confusion.
- Product: whether rankings and GIFs actually answer the coach's question.

## Early Failure Modes

### Scheduler Accidentally Becomes All-Pairs

Risk: high.

Why it matters: including every checkpoint from every run can make the player
pool huge. Building every possible pair before sampling can become the bottleneck
before Modal even starts running games.

Simple mitigation: `adaptive_v0` should generate a bounded set directly from
rating bands, anchors, random bridges, and replay queues.

Test: a synthetic 5,000-checkpoint pool schedules a small budget quickly and
never materializes `N * (N - 1) / 2` candidates.

### Schedule Metadata Gets Dropped

Risk: medium.

Why it matters: if the coach sees a battle, we need to know why it exists:
placement, near-rating, uncertain, anchor, bridge, or replay.

Simple mitigation: add an explicit `schedule` field to pair specs, or store
schedule rows in round input and summary sidecars.

Test: build an adaptive round and assert every pair has a schedule reason in the
round input and battle summary.

### Ratings Look Precise But Are Not

Risk: high.

Why it matters: early adaptive rankings can mislead the coach if provisional
players look as trustworthy as well-tested players.

Simple mitigation: every rating row should show status and evidence:
provisional/active, games, battles, distinct opponents, failures, and last-round
delta.

Test: a new checkpoint stays provisional until it crosses the evidence floor.

### Selection Bias Freezes The Ladder

Risk: high.

Why it matters: if we only play nearby ratings, rating islands can form and a
strong new checkpoint may not get enough bridge games.

Simple mitigation: reserve budget for random bridges and anchors in every round.

Test: each rating band has at least one bridge route to another band after a
few rounds.

### Long Runs Dominate The Pool

Risk: high.

Why it matters: if one training run saved many checkpoints, it can consume the
round budget and make the ladder mostly compare a single lineage to itself.

Simple mitigation: enforce per-run or per-lineage scheduling caps. Prefer
cross-run opponents unless debugging a specific lineage.

Test: with one run contributing 80% of checkpoints, its pair share is capped.

### Seat Bias Pollutes Elo

Risk: unknown, possibly high.

Why it matters: fixed seat order can make one checkpoint look stronger because
of start position or action timing, not policy quality.

Simple mitigation: stable hash seat assignment for normal pairs; seat-swapped
companion battles for placement, anchors, and close top-rank battles until bias
is measured.

Test: run mirrored pairs and report seat win-rate gap.

### Draws And Timeouts Hide Bad Games

Risk: medium.

Why it matters: if games regularly hit the 8,000-step cap, Elo sees draws while
the real issue may be no-death behavior or a broken setup.

Simple mitigation: expose timeout/draw rate per checkpoint and per battle.

Test: synthetic and real smokes show timeout count separately from normal draws.

### Reducer Cannot Recompute Large Runs

Risk: high.

Why it matters: if the reducer fails after shards finish, we must rebuild from
committed summaries without scanning every game forever.

Simple mitigation: normal and repair reducers should prefer shard summaries and
pair tallies, then fall back to game summaries only for legacy cases.

Test: delete `latest.json` and rebuild from shard summary files only.

### Website Shows Stale Data As Fresh

Risk: high.

Why it matters: the coach may trust a ranking that is behind the running job.

Simple mitigation: show `updated_at`, phase, provisional/final status, and
explicit stale markers. Auto-refresh only small progress/ranking artifacts.

Test: stale progress fixture renders a clear stale label.

### Website Hides Partial Rankings

Risk: high.

Why it matters: the coach needs to watch the system while games are still
running. If `provisional_latest.json` exists but the page waits for final
`latest.json`, the site looks broken.

Simple mitigation: render live rankings from provisional snapshots, label them
as live, and show completed/total games.

Test: provisional-only fixture renders ranking rows with live/provisional copy.

### Website Clicks Scan Too Much

Risk: high.

Why it matters: policy or battle clicks can feel broken if they scan all battle
or game summaries.

Simple mitigation: publish per-checkpoint battle indexes and page battle detail.

Test: checkpoint click uses an index/paged endpoint and stays fast with a large
fake battle index.

### All-Checkpoint Discovery Still Returns Latest-Only

Risk: high.

Why it matters: the new product target says every useful checkpoint can enter
the pool. Current discovery only returns the latest checkpoint per run.

Simple mitigation: add `checkpoint_selection=latest|all|iteration`, with
`latest` preserving current behavior.

Test: a run with several real `iteration_*.pth.tar` files returns all of them
only when `checkpoint_selection=all`.

### Pool Mismatch Mixes Old Artifacts

Risk: high.

Why it matters: pair history from one checkpoint pool can corrupt a different
pool if IDs or run sets changed.

Simple mitigation: store and validate a deterministic pool hash in scheduler
state, pair history, and rating snapshots.

Test: pair history with the wrong pool hash raises instead of being reused.

### GIFs Distract From The Score Run

Risk: medium.

Why it matters: saving many GIFs can inflate artifacts and slow inspection,
while score tournaments do not need GIFs for every battle.

Simple mitigation: keep GIFs off for rating waves. Use separate sample/canary
GIF jobs.

Test: large rating estimate reports zero GIFs unless an explicit cap is set.

## Modal Notes

From Modal docs checked on 2026-05-13:

- Volumes need commits before other containers can see writes, and reloads are
  needed in already-running readers.
- Volume reload can fail when files are open.
- Volumes v2 are the right durable artifact store here, but concurrent writes to
  the same file should be avoided.
- Queues are useful for communication between active functions, but they are not
  durable storage.
- Dict/Queue can help with leases, heartbeats, or work queues later. They should
  not replace Volume artifacts as truth in V0.

Sources:

- https://modal.com/docs/guide/volumes
- https://modal.com/docs/guide/queues
- https://modal.com/docs/reference/modal.Queue
- https://modal.com/docs/examples/dicts_and_queues

## Current Recommendation

Do not build the giant all-checkpoint system in one jump.

Next safe implementation step:

1. Add pure adaptive scheduler helpers and tests.
2. Add explicit schedule metadata.
3. Add bounded all-checkpoint discovery.
4. Add a tiny adaptive remote smoke with GIFs off.
5. Add website/index changes before any large adaptive run.

Keep every lane small enough that failures are easy to explain.
