# Coach Current State - 2026-05-10

## 2026-05-11 Reset

- North Star: prove MuZero training signal through exact replications first,
  then return to CurvyTron.
- LightZero remains a serious replication/control lane. It is not dismissed as
  a reference just because our custom paths are unresolved.
- CurvyTron training is paused except for environment/interface work owned by
  other agents.
- Survival-time signal still matters for Pong and CurvyTron evals. The current
  priority is framework replication, not another CurvyTron scale run.
- No fixed-seed overfitting. Training and eval should use randomized starts with
  reproducible seed lists; fixed seeds are panels to compare against, not targets
  to optimize.
- The custom two-seat CurvyTron collector remains diagnostic only until we know
  whether it bypassed core LightZero collector/GameBuffer/target behavior that
  would matter for learning claims.

## Current Read

- Pong control evidence is useful only if it is tied to exact framework
  replication and survival-time evals over reproducible random seed panels.
- CurvyTron reward/eval remains steps survived. Keep timestep and decision
  interval labels attached to any survival comparison.
- Existing CurvyTron runs are not enough to claim a robust MuZero signal. Some
  runs updated weights, but the best evidence is mixed, setup-dependent, or
  matched-opponent only.
- Before launching more CurvyTron training, establish whether stock or
  near-stock LightZero MuZero can learn the known controls in our container.

## Active To-Do

- Replicate known controls first: stock LightZero Atari Pong MuZero, then a
  LightZero board-game MuZero control, before treating custom CurvyTron results
  as framework evidence.
- Answer the main open questions:
  did we ever prove stock learning;
  did the custom collector bypass core LightZero pieces;
  what controls are already running;
  what controls are still needed.
- Keep CurvyTron env/interface work moving only where it supports later
  framework integration: reset randomization, player perspective, reward/eval
  contracts, action masks, and artifact labeling.
- Preserve the observability contract: reports should tie each run to timestep,
  checkpoints, survival curve, action mix, seed list, death reasons, and artifact
  refs. See `curvytron_observability_inspector_brief_2026-05-11.md`.
