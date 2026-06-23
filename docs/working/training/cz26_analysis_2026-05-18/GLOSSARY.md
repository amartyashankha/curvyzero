# Glossary

Use these words exactly. If a new shorthand appears, add it here before using
it in findings.

## Batch

- `CZ26`: the current 136-run training batch.
- `Grid A`: 96 runs. It changes four things: reward outcome strength, action
  noise, leaderboard-opponent immortality, and opponent recipe.
- `Grid B`: 40 runs. It mostly changes the opponent recipe. Reward is fixed to
  the middle outcome setting.

## Runs And Checkpoints

- `run`: one training job.
- `checkpoint`: a saved policy from a run.
- `iteration`: training progress number attached to a checkpoint. `iteration 0`
  is the shared starting policy and should not count as learned progress.
- `learned checkpoint`: any checkpoint with `iteration > 0`.
- `latest checkpoint`: the newest checkpoint for a run.
- `best checkpoint`: the checkpoint that scored best on one metric. This is not
  always the latest checkpoint.

## Opponents

- `slot`: one opponent assignment in the 64-slot opponent bag.
- `opponent recipe`: the mix of opponent slots used while collecting training
  games.
- `blank`: a hard-coded empty/blank opponent. This is an immortal hard-coded
  opponent seat.
- `wall`: a hard-coded wall-avoider opponent. This is an immortal hard-coded
  opponent seat.
- `rank1`, `rank2`, etc.: a policy pulled from the tournament leaderboard at
  that rank.
- `b20w05r1`: recipe shorthand. Example: about 20% blank, 5% wall, and the rest
  rank-1 leaderboard opponent. The exact counts are in `ANALYSIS_PLAN.md`.

## Immortality

- `imm0`: leaderboard opponents are not randomly made immortal.
- `imm10`: leaderboard opponents are made immortal 10% of the time.
- `immortal`: the opponent cannot die. This is separate from what policy it is.

## Noise

- `n0`: no extra action noise.
- `n10`: 10% action noise.
- `n20`: 20% action noise.

## Reward

- `out0`: no outcome reward. Mostly survival plus bonus reward.
- `out33`: small outcome reward.
- `out67`: medium outcome reward.
- `out100`: full outcome reward.
- `out50`: middle outcome reward used by Grid B.
- `outcome reward`: reward from winning or losing the game.
- `survival reward`: reward for staying alive longer.
- `bonus reward`: reward from picking up bonuses.

## Metrics

- `survival`: average game length in eval games. Bigger is better.
- `training reward`: the reward value reported by the trainer/eval. Compare it
  only within the same reward definition.
- `outcome residual`: inferred outcome reward, computed as training reward
  minus survival component minus bonus component. This is not serialized as a
  named component in the eval artifact.
- `tournament rank`: leaderboard position. Smaller is better.
- `learned tournament rank`: tournament rank after ignoring `iteration 0`
  checkpoints.
- `rating exposure`: how much tournament evidence a checkpoint has. Count this
  with games and battles; a rank based on one battle is weak evidence.
- `battle`: one tournament matchup between two checkpoints. Each battle usually
  contains multiple games.
- `game`: one played CurvyTron game inside a battle.
- `top 100`: a checkpoint reached rank 100 or better. Since lower rank is
  better, rank 34 is inside top 100.
- `matched comparison`: compare two settings only when the other settings are
  the same.
- `common horizon`: the latest training iteration that every row in a group has
  reached. It keeps comparisons fair when some rows have less data.
- `exact horizon`: a table that uses only an exact checkpoint iteration, such
  as 30k, 170k, or 300k.
- `projection`: collapse one axis, such as reward or recipe, and average over
  the other axes. This is useful but less strong than a matched comparison.
- `retention`: latest score divided by best score. Low retention means the run
  peaked and then regressed.
- `action collapse`: the eval action summary says the policy mostly chose one
  action. It is a warning sign, not by itself a full diagnosis.
- `row action collapse count`: count of individual eval rows that looked
  action-collapsed. This is weaker than full checkpoint action collapse.
