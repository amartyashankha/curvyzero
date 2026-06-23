# Candidate Schedulers

Last updated: 2026-05-16.

## Current Synthesis

Do not throw away `adaptive_v0`. It is already close to the right shape:

- bounded pair budget;
- placement before refinement;
- near-rating work;
- uncertainty/high-delta work;
- random bridges;
- top-100 active pool;
- batch Elo reduction.

The gap is burst behavior and evidence trust. In a large burst, the current
flat 300-pair placement wave is bounded but slow. With 100 established and 500
new rows, it takes about 34 waves to give every new row 20 opponents. That is
too slow if we want new checkpoints to become useful quickly.

## Product Contract

Expose this to callers:

- submit checkpoint refs or watched run ids;
- status: `provisional`, `active`, `audit_needed`, `retired`;
- rating and uncertainty/conservative score;
- evidence: games, distinct opponents, outside-run opponents;
- scheduler readouts: pair budget, scheduled pairs, reason counts, coverage,
  anchor concentration, graph health;
- eligibility: public leaderboard and trainer-assignment eligibility.

Do not expose this as normal product API:

- exact quota percentages;
- anchor rank bands;
- random seed formulas;
- repeat freshness formulas;
- top-band boost constants;
- Glicko/RD hyperparameters;
- internal pair-selection mode names.

The service should provide a steering wheel, not engine-room controls.

## V1 Candidate

Keep the `adaptive_v0` spine, but split placement into two modes.

### Normal Intake

Use when new rows are small relative to the active pool.

- diversified established anchors;
- near-rating/top-band refinement;
- random bridges;
- uncertainty/high-delta rows;
- limited repeats for stale, close, or noisy pairs.

### Burst Intake

Use when many new rows arrive together.

- established anchors connect new rows to the trusted pool;
- new-vs-new placement gives broad coverage faster;
- a provisional grace pool prevents early strong-policy false drops;
- no row can retire before evidence gates are met;
- publish time-to-first-evidence and time-to-placement-target estimates before
  Modal work starts.

## Initial Quotas To Test

These are test candidates, not commitments.

For a normal 300-pair wave:

- 35% placement and undercovered rows;
- 30% near-rating/top-k refinement;
- 15% uncertainty/high-delta rows;
- 15% stratified random bridges and cross-lineage audits;
- 5% deliberate repeats.

For burst intake:

- 40% new-vs-established anchors;
- 40% new-vs-new coverage;
- 10% top/boundary refinement;
- 10% random bridges/audits.

The burst split is meant to avoid the `500 new * 20 opponents = 10,000`
established-first pair burden. It still needs simulation.

## Evidence Gates

A row should not be called trusted active unless it has:

- enough games;
- enough distinct opponents;
- enough outside-run opponents;
- acceptable failure rate;
- acceptable draw/timeout rate;
- current evaluator context;
- no incompatible observation metadata;
- no unresolved audit flag.

The trainer may still use provisional candidates for exploration if explicitly
requested, but the public leaderboard should clearly label them.

## Designs To Reject Unless Simulations Disagree

Pure all-pairs:

- Too expensive as the default.
- Keep only for small audits and oracle baselines.

Single best-anchor placement:

- Too narrow.
- A lucky or stale leader becomes a placement sink.

Pure binary ladder:

- Efficient in clean transitive worlds.
- Brittle under noise and non-transitive policies.

Pure random:

- Good baseline and bias detector.
- Wastes games once rough ratings exist.

Pure top-k/bandit:

- Useful for promotion selection.
- Too narrow for maintaining the main public leaderboard.

## Next Proof

Run a simulation where:

- 100 established rows exist;
- 500 new rows arrive;
- many new rows truly belong in the top 100;
- only top 100 can become trainer-facing active;
- compare current flat `adaptive_v0`, random coverage, burst split, and
  uncertainty/top-k refinement;
- measure false drops, deep false drops, promotion latency, and games.

