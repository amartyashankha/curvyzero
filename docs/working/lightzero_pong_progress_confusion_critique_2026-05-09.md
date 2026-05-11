# LightZero Pong Progress Confusion Critique - 2026-05-09

Scope: official Atari Pong LightZero lane. This note reads the current working
docs and Modal wrappers only. No pytest was run. No training code was edited.

## Biggest Simple Explanation

The progress story is confusing because we are mixing evidence types.

The cleanest comparison is same run, same eval contract, `iteration_0` versus a
later normal `iteration_*.pth.tar` checkpoint. A bad live eval of the active
`32768` run's `iteration_0` is only the starting line. It should not be compared
as a learning result against the completed `8192` run's final checkpoint.
That comparison is not a regression test.

The second confusion is the scoreboard. Pong return is important, but "still
loses" can hide useful progress if the policy loses later, gives up fewer
points, or starts earning a point. Current summaries include some survival-like
fields, but most reported rows hit the fixed `512` eval cap, so `Steps=512` does
not by itself prove longer survival. Read reward timing and counts beside the
return.

Score every official Atari Pong row in this order: same-run baseline, stock
return, manual/raw return, steps survived, first/fewer negative rewards,
positive rewards, action collapse, placement, and manifest ref. For custom
dummy Pong, use true score plus survival mean/median/p90/std, shaped
loss-delay telemetry, action distribution, opponent/reset split, and manifest
ref. Do not mix either lane with CurvyTron claims.

## Current Read

- We are probably comparing the wrong checkpoints when we line up `8192` final
  against `32768 iteration_0`. Compare `32768 iteration_0` to a later `32768`
  periodic or final checkpoint instead.
- We are partly measuring survival, but not loudly enough. The eval manifest
  records `steps_survived`, nonzero reward count, positive reward count, and
  reward step indices. Promote those in the readout. Ask: did the first
  negative reward move later? Did negative rewards get farther apart? Did total
  negative rewards drop, even if the policy still lost?
- CPU/GPU evals are not yet comparable as learning evidence. On the same
  untrained `32768 iteration_0`, CPU and GPU both strict-loaded and both lost,
  but their action traces differed. Treat that as a parity warning, not a
  progress claim.
- The current `32768` run is probably too early to judge until a later normal
  checkpoint exists and is evaluated.
- We are still in `faithful-short`, not exact reproduction. The exact installed
  package budget is `200000` env steps; the completed clean run used `8192` and
  the active one uses `32768`.
- Lane labels matter: official Atari LightZero is a control/reproduction lane,
  custom dummy Pong is a bridge/debug lane, and CurvyTron is the repo-native
  simultaneous-game target. A signal in one lane is not a solved result in the
  others.

## Top Risks

1. Wrong checkpoint comparison: comparing one run's final checkpoint to another
   run's initial checkpoint.
2. Return-only reading: missing progress where the policy still loses but loses
   later or gives up fewer points.
3. CPU/GPU parity overclaim: treating different action traces on an initial
   checkpoint as learning evidence.
4. Too-early judgment: calling the active `32768` run bad before a later normal
   checkpoint is evaluated.
5. Reproduction overclaim: calling `faithful-short` an exact LightZero Pong
   reproduction.

## Current Next Actions

1. Poll for the next normal `iteration_*.pth.tar` from the active `32768` run.
   Do not use `ckpt_best` for quality unless its state proves it is real.
2. Eval `32768 iteration_0` and the later `32768` checkpoint with the same
   strict no-fallback stock-ish contract.
3. Report each row as: stock return, manual return, first negative reward step,
   negative reward count, positive reward count, reward step list, action
   collapse, and CPU/GPU placement.
4. Treat "same return but fewer/later negative rewards" as possible early
   progress, not as solved Pong.
5. Keep labels plain: `installed LightZero 0.2.0 faithful-short` until the full
   `200000` installed-package budget is actually run.
6. For custom dummy Pong, inspect target sidecars and support-scale proof
   fields before any longer same-config run.
