# LightZero Stock Loop Contract

Purpose: define what "we are using LightZero" means for CurvyTron training.

## Trusted Stock Loop

A run is on the stock LightZero MuZero loop only if it calls
`lzero.entry.train_muzero` and lets LightZero own:

- collector / env manager;
- policy `collect_mode`;
- MCTS/search;
- `GameSegment` creation;
- `MuZeroGameBuffer` replay and target sampling;
- learner lifecycle;
- checkpoint saving;
- optional stock evaluator.

Using `MuZeroPolicy.collect_mode.forward` and `MuZeroPolicy.learn_mode.forward`
directly is not the same thing.

## Env Contract

The env should make one LightZero transition mean one real transition in the
problem being trained.

For a CurvyTron stock-control env, the run must state:

- observation shape and stack source;
- action meaning;
- reward meaning;
- done meaning;
- `to_play` meaning;
- opponent source, if any;
- whether the row is single-ego, centralized joint-action, or something else.

## Claim Labels

- `stock fixed/frozen opponent`: one learner-controlled ego player; opponent is
  owned by env or loaded from a frozen checkpoint.
- `stock recent frozen opponent`: same stock shape, but the frozen opponent is
  periodically chosen from recent checkpoints. This is useful if labeled
  honestly; it is not exact live same-current-policy self-play.
- `stock centralized joint-action`: one scalar action controls both players.
- `turn-commit profile`: fake pending/commit scalar steps, not trainable today.
- `custom two-seat`: current-policy joint action collection outside stock
  LightZero; not a trusted learning claim unless native target/replay parity is
  proven.
