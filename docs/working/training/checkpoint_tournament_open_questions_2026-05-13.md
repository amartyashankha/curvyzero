# Checkpoint Tournament Open Questions, 2026-05-13

## V0 Questions

- Should default games use greedy eval or training-style collect?
  - Current V0 default: greedy eval for cleaner deterministic policy comparison.
  - Every summary records the mode so this can be changed later.
- Should a pair run both seat orders?
  - Current V0 default: unordered pairs, one seat order.
  - Add ordered pairs when seat bias matters.
- How many GIFs should be saved?
  - Current V0 can save GIFs, but large tournaments should probably sample them.
- Should timeout count as draw?
  - Current V0: yes.

## Hypotheses

- H1: One game per Modal input is the cleanest first autoscaling unit.
  - Evidence: user explicitly wants the lowest unit to be one CPU game.
  - Risk: too many small Volume commits at very large scale.
- H2: Battle summaries should be written by the pair function, not game workers.
  - Evidence: Modal Volume docs warn against concurrent writes to the same file.
  - Risk: if a pair function dies after games finish, summaries may need a
    repair/backfill command later.
- H3: Greedy eval is a good default for tournament score, but collect mode is
  useful for diagnosing training-style behavior.
  - Evidence: GIF work already showed greedy behavior can hide stochastic
    training behavior.
  - Risk: mixed policy modes could confuse Elo unless the mode is part of the
    tournament identity.
- H4: The first Elo loop should use batch updates.
  - Evidence: Modal game jobs finish in arbitrary order; game-by-game Elo would
    add order bias.

## Scale Questions

- How many Modal inputs can we push before orchestration overhead dominates?
- Should one game be one Modal input, or should one input run a small shard of
  games?
- How many files can the tournament namespace hold before Volume listing gets
  slow?
- Should completed game summaries be compacted into pair shards after a run?

## Elo Follow-Up Questions

- Use plain Elo, Glicko-style uncertainty, or a simpler custom score with
  uncertainty?
- How many games per sampled pair are enough before an update?
- How do draws/timeouts affect the update?
- What stopping rule means the ladder is stable enough?
- How should we choose pairs: random, near-rating, uncertain, or mixed?

## Validation Questions

- Can we run a tiny tournament with two known checkpoints and get non-empty
  summaries and GIFs?
- Can we run a fake/dummy scoring test without LightZero installed?
- Can the website load quickly when there are many battles?
- Can the tournament rerun safely without corrupting old artifacts?

## Gaps To Close Before A Big Tournament

- Find a reliable way to select checkpoints from the freshest runs.
- Decide whether a tournament should include both seat orders by default.
- Decide GIF sampling. Saving one GIF per game is nice for smoke and bad for
  huge runs.
- Add a repair/backfill command that scans game summaries and rewrites battle
  and tournament summaries.
- Add a dry-run estimator: checkpoint count, pair count, game count, expected
  file count, and rough cost/time warning.
