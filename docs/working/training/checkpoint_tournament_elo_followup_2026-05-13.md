# Checkpoint Tournament Elo Follow-Up, 2026-05-13

## Future Goal

After the V0 battle runner works, build a small rating loop over checkpoints.
The loop chooses pairs, runs enough games per pair in parallel, updates ratings,
and repeats until the ladder is stable enough to guide coach decisions.

The reason this matters: the coach will soon have too many checkpoints to judge
by hand. Raw GIFs help explain behavior, but a rating loop gives a quick map of
which checkpoints are actually improving.

## Current Recommendation

Use batch Elo first, not game-by-game Elo.

1. Snapshot ratings at the start of a rating round.
2. Choose pairs from that snapshot.
3. Run battles in parallel.
4. Apply one batch update after battle summaries are complete.

This avoids async order bias when many Modal jobs finish in random order.

## Simple Formula

Expected score:

`E_A = 1 / (1 + 10^((R_B - R_A) / 400))`

Observed score:

`S_A = (wins_A + 0.5 * draws) / valid_games`

Update:

`R_A += K_pair * (S_A - E_A)`

Default:

- initial rating: `1500`
- `K_pair = 32 * sqrt(valid_games / 50)`
- clamp `K_pair` to `[16, 64]`
- clamp one pair delta to `+/-80`

## Pairing Policy

Use a mix:

- 60% near-rating pairs
- 25% uncertain/new checkpoint probes
- 15% anchor repeat pairs

For a new checkpoint, test against:

- best current checkpoint
- median checkpoint
- weaker checkpoint
- nearest neighbor after an early estimate

## Draws And Failures

- Draw is `0.5`.
- Timeout with no death is draw for rating v0.
- Infrastructure failure is invalid and excluded.
- Checkpoint/policy failure should be a loss for that checkpoint if clearly
  caused by the checkpoint runner, otherwise invalid.
- Do not rate a pair unless enough games are valid, e.g. at least 80%.

## Trust Rule

Battle results are immutable. Rating snapshots are disposable.

That means we should always be able to recompute ratings from stored battle
results.

## Validation Plan

- Build fake deterministic-strength bots or a fake battle simulator.
- Verify recovered ranking matches hidden strength.
- Verify random completion order gives the same batch rating snapshot.
- Inject invalid games and confirm they do not pollute ratings.
- Add a strong new checkpoint and check the pairing policy finds it quickly.

## V1 Data Model Sketch

Keep this derived and simple:

- `ratings/<rating_run_id>/config.json`
- `ratings/<rating_run_id>/rounds/round-000000/input.json`
- `ratings/<rating_run_id>/rounds/round-000000/results.json`
- `ratings/<rating_run_id>/rounds/round-000000/ratings.json`

Each round points to immutable tournament battle refs. Ratings can be rebuilt if
the formula changes.

## V1 Stop Rules

Start with boring rules:

- stop after a fixed round count, or
- stop when top-10 order changes little for several rounds, or
- stop when all checkpoints have at least a minimum number of valid games.

Do not overfit the stop rule before the raw battle runner is proven.

## V1 Website Ideas

- standings table with rating, games, uncertainty proxy, latest checkpoint time
- battle drill-down for a selected checkpoint
- "show me surprising results" list
- sample GIF per battle, not every GIF at once

## Not For V0

- Full Glicko or TrueSkill.
- Global Bradley-Terry fitting.
- Fancy website ranking math.

Those are useful later, but the first useful thing is reliable pair battle data.
