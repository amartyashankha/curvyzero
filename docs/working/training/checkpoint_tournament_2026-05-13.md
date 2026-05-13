# CurvyTron Checkpoint Tournament, 2026-05-13

## Grounding Prompt

The user wants basic tooling for checkpoint tournaments before a larger batch of
CurvyTron runs lands. The core unit is simple: given two checkpoints, run N
games, score who dies first, and save artifacts. Everything should run on Modal.
There should be exactly one tournament Modal app with several functions, not a
new app per battle. The lowest-level function should be a small CPU function
that runs one game to completion and can autoscale hard. Higher layers should
run many games for one pair and many pairs for a whole tournament. Battles
should also be able to produce rich-render GIF artifacts. A separate tournament
website should exist, even if the first version is basic.

## North Star

Give the coach and inspector a quick way to compare checkpoints against each
other after a large training batch. The result should answer a plain question:
when checkpoint A plays checkpoint B, who tends to die first?

## Minimal Product

- One reusable game runner: checkpoint A vs checkpoint B, one seeded game.
- One battle runner: checkpoint A vs checkpoint B, many games in parallel.
- One tournament runner: many checkpoint pairs, many battle jobs in parallel.
- Simple scoring: player who dies first loses; other player wins; simultaneous
  death or timeout is a draw.
- Optional rich GIF per game, off by default for huge tournaments.
- Volume artifacts with JSON summaries and GIFs.
- A basic Modal-hosted website to list tournaments and battles.

## Current Design

- Keep this lane separate from training and live checkpoint GIF plumbing.
- Use the same source-state turn-commit visual env used by checkpoint GIFs.
- Load two LightZero checkpoints into two policies.
- At each game step, ask both seat policies for actions and step the shared
  two-player env once.
- Capture human RGB frames from the env when requested.
- Store artifacts under `tournaments/curvytron/...`.

## Operating Pattern

The main thread should stay focused on orchestration:

1. Re-read the product goal.
2. Check the current code path.
3. Write down evidence, gaps, and hypotheses.
4. Delegate narrow research or validation work when it can run in parallel.
5. Implement the smallest useful thing.
6. Test it locally.
7. Run a remote smoke when the change touches Modal.
8. Update these docs before stopping.

Stopping point rule: do not stop at "the code compiles" if the actual product
goal needs a remote job, a browser, or a real artifact.

## User

Primary user: the coach. They need to compare many checkpoints from a large
training batch and see whether one checkpoint actually beats another in the real
game.

Secondary user: the inspector. They need GIFs and simple battle summaries to
spot weird behavior quickly.

The first UI should not try to explain everything. It should answer:

- Which tournament am I looking at?
- Which checkpoints played?
- Who won more?
- Can I open a sample GIF and JSON?

## Open Questions

- How high Modal autoscaling should be pushed for 300x300x50 scale.
- Whether round-robin should include both A-vs-B and B-vs-A separately.
- How many GIFs to save for very large tournaments; likely not every game.
- Whether the website should rank checkpoints by Elo/Bradley-Terry later.

## Critique

The first version should not over-model tournament formats. A round-robin pair
matrix plus raw win/loss/draw counts is enough. More ranking math can be added
after the basic runner is proven.
