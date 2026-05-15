# Caveats And Footguns

## Checkpoint Discovery

- Never use only `train/lightzero_exp/ckpt`.
- Always scan `train/lightzero_exp*/ckpt/iteration_*.pth.tar`.
- Timestamped LightZero experiment dirs can hide real progress from fixed-path
  readers.

## Cadence

- Trusted train lane now uses one source frame per policy action.
- Old tournaments may use legacy multi-frame context.
- Rating context hash includes cadence fields, but only if spec is correct.
- Do not mix one-frame and legacy tournaments into one public leaderboard.

## Modal Dict And Queue

- Dict is cache/coordination, not truth.
- Dict entries can expire.
- Queue events are not a durable ledger.
- Volume snapshots and assignment files are reproducibility truth.

## Leaderboard Evidence

- Small canaries are plumbing evidence, not rating-quality evidence.
- Active leaderboard rows need enough distinct opponents.
- Seat/color asymmetry is still a risk.
- Head-to-head strength is not the same as survival improvement.

## Projection Metrics

- `projection@200k` is triage only.
- It over-rewards immature rows with steep early slopes.
- Do not use it for promotion without maturity and health guards.

## Scripted / Invincible Opponents

- Public manifests and slot recipes should use `opponent_immortal`; the
  lower-level `opponent_death_mode` is derived runtime plumbing at the env
  boundary.
- Invincibility as an episode modifier is not the same as an invincible policy.
- Scripted policies need explicit identity if included in tournaments.
- Blank/no-op can be a training opponent without being a leaderboard player.
- Current tournament/rating code assumes every player has a checkpoint ref.
  Scripted policies are not first-class leaderboard players today.
- `passive` and `scripted` are recipe/manifest labels, not separate env enums.
- `blank_canvas_noop` entries must use `fixed_straight`; frozen checkpoint
  mixture entries require normal runtime.
- A fractional "make opponent invincible sometimes" design is currently best
  represented by mixture weights or a future overlay, not by a leaderboard row.

## Optimizer Settings

- Speed improvements are orthogonal only if observation, cadence, reward, and
  evaluator semantics are unchanged.
- Faster browser render may change operational defaults, not necessarily policy
  quality conclusions.

## Stochastic / Repeat Knobs

- `repH` is top-tail, not uniformly better.
- `repM` is safer and more stable.
- Survival `medium` is leaderboard-favored; survival `light` has best
  survival-gain point estimate; `heavy` is a bounded stress lane.

## Trainer Boundary

- Do not let `train_muzero` read tournament state.
- Do not poll live Elo inside training.
- Use immutable assignment snapshots at launch/resume/explicit refresh
  boundaries.
