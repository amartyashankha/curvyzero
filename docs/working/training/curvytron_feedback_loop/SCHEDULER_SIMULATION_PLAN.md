# Scheduler Simulation Plan

Last updated: 2026-05-16.

## Purpose

Use local simulations before changing production scheduler behavior. We want to
know which scheduler gives a trustworthy top pool with far fewer games than
all-pairs.

## Model

Each synthetic checkpoint should have:

- `checkpoint_id`
- `run_id`
- `iteration`
- `arrival_round`
- hidden strength
- style tag
- optional lineage drift
- optional timeout or collapse risk

Game outcome should start simple:

```text
P(A beats B) = sigmoid((strength_A - strength_B) / scale)
```

Then add harder worlds:

- noisy games
- style cycles, like A beats B, B beats C, C beats A
- one run contributing many checkpoints
- batches of many new checkpoints
- strong new policies arriving together
- weak policies that time out or collapse
- seat bias if not mirrored

## Schedulers To Compare

1. `all_pairs_oracle`
   - Expensive upper bound.
   - Used to know the "truth" for small worlds.

2. `random_stratified`
   - Fixed budget.
   - Random pairs across rating bands.
   - Good baseline.

3. `current_adaptive_v0`
   - Current code behavior where possible.
   - Placement first, then near-rating/top-biased work.

4. `top_anchor_placement`
   - New policies first play strong policies.
   - Tests the user's idea and the placement-sink risk.

5. `spread_anchor_placement`
   - New policies play top, upper, median, lower, same-run neighbor, and random
     bridge opponents.
   - Tries to locate weak and strong policies without overusing one anchor.

6. `binary_ladder_placement`
   - New policies probe a rank ladder and then refine around the likely band.
   - This is the "do better than chess because games are cheap" idea.

7. `uncertainty_match`
   - Prefer players with uncertain ratings and pairs near 50/50.
   - Could be implemented later with Glicko-style rating deviation.

8. `top_pool_refinement`
   - Spend extra work near the top 10/top 20/top 100 boundary.
   - Useful because the trainer mostly cares about strong opponents.

## Metrics

Ranking quality:

- top-10 recall against oracle
- top-20 recall against oracle
- top-100 recall against oracle
- false drops from true top 100
- rank error for the top band
- held-out pair prediction error

Evidence health:

- games per checkpoint
- distinct opponents per checkpoint
- outside-run opponents per checkpoint
- repeat-pair share
- zero-appearance count per wave
- graph connected components
- top-anchor appearance concentration

Operational cost:

- pairs scheduled
- games scheduled
- budget expansion factor
- waves needed
- maximum work in a single wave
- expected Modal worker fanout

Failure signals:

- one anchor gets too many games
- new policies only play old top 100 and never each other
- strong new batch gets truncated too early
- lower rows never get enough evidence
- non-transitive counter-policy is missed
- stable Elo appears before graph coverage is healthy

## First Toy Experiments

Run all schedulers on:

1. `clean_100_plus_20`
   - 100 established, 20 new.
   - Pure hidden strength.

2. `burst_100_plus_500`
   - 100 established, 500 new.
   - Tests top-100 truncation and placement explosion.

3. `strong_new_batch`
   - 100 established, 100 new, many new are truly top 100.
   - Tests whether new policies need to play each other.

4. `lineage_heavy`
   - 80% of policies from one run.
   - Tests same-run evidence pollution.

5. `nontransitive_styles`
   - Three style clusters with cyclic advantages.
   - Tests whether scalar Elo misses leader counters.

6. `noisy_boundary`
   - Many policies clustered near ranks 70-130.
   - Tests top-100 false drops.

7. `timeout_world`
   - Some policies survive long or timeout without being strong.
   - Tests draw/timeout status flags.

## Initial Acceptance Bar

A scheduler is not ready if:

- true top-100 recall is bad under a reasonable budget;
- any checkpoint reaches trusted status with too few distinct opponents;
- one opponent receives most placement games;
- many strong new policies are dropped before playing each other;
- non-transitive leader counters are invisible;
- the scheduler cannot explain why it picked a pair.

## Experiment Card Template

Use this format in `EXPERIMENT_LOG.md` for every run:

```text
Experiment ID:
Hypothesis:
World:
Scheduler configs:
Seeds:
Budget:
Oracle:
Primary metrics:
Failure metrics:
Result:
Follow-up IDs:
Docs/code changed:
```

## Current Candidate Gates

For a scheduler to beat the current baseline, it should pass these rough gates
in local simulation:

- In a `100 established + 500 new` burst, every new row gets at least one game
  within a small number of waves, and the readout says how many waves.
- In a `100 established + 500 new` burst, every new row reaches the placement
  opponent target within an explicit wave cap, or the service marks the burst as
  still under placement rather than publishing trusted rankings.
- Report `new_vs_new`, `new_vs_established`, and `established_vs_established`
  pair shares. Burst mode probably needs both new-vs-new coverage and
  established anchor linkage.
- True top-100 false drops are low after the placement window.
- No single established anchor receives a large fraction of placement games.
- New policies play some other new policies when a large strong batch arrives.
- Top-20 recall improves over random coverage at the same game budget, or the
  scheduler explains why it is spending budget on breadth first.
- Non-transitive leader counters are tested by explicit audit pairs, not left to
  chance.

## Burst Readouts Required Before Modal Work

For any large intake batch, the scheduler should be able to say:

- requested pair budget;
- placement target;
- estimated waves to first evidence for all new rows;
- estimated waves to placement target for all new rows;
- expected total placement pairs and games;
- expected new-vs-new and new-vs-established share;
- whether any row can be retired before meeting evidence gates.

## Stronger Gates From Arendt

For an online top-100 tournament, evaluate against an oracle ranking and track:

- `FalseDrop@100`: true top-100 checkpoints dropped by the service.
- `DeepFalseDrop`: any dropped checkpoint whose true oracle rank is 50 or better.
- `DropRegret`: how much better a dropped checkpoint is than the true rank-100
  cutoff.
- `NewStrongRecall@100`: recall among new checkpoints that truly belong in the
  top 100.
- `PromotionLatency`: waves until a true top-100 newcomer is retained.
- `StyleRecallGap`: recall gap across non-transitive style clusters.
- `RunBurstBias`: same-run game share and retained top-100 run concentration.

Initial medium-budget bar:

- Clean transitive world: top-100 recall and precision near 99%.
- Strong new batches: top-50 newcomers are never dropped.
- Top-100 boundary world: false drops are rare and shallow.
- Non-transitive world: recall stays high across style clusters.
- Same-run burst: same-run evidence does not dominate unless unavoidable.
