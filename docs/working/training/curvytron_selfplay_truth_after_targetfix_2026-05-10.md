# CurvyTron Self-Play Truth After Target Fix - 2026-05-10

No pytest was run.

## Short Truth Audit

True current-policy two-seat, today:

- The bounded CurvyTron two-seat LightZero smoke runs one live
  `MuZeroPolicy` object for both player seats before `env.step`.
- It renders `float32[B,P,4,64,64]` source-state gray64 observations, maps live
  player slots into policy rows, evaluates each active seat through that same
  policy object, builds `joint_action[B,P]`, and steps
  `VectorMultiplayerEnv` externally.
- It records replay rows for both seats with `iteration`, `env_row_id`,
  `player_id`, `decision_index`, observation, legal mask, action, policy
  weights, root value, reward, and done.
- After the target/batch fixes, the two-seat learner adapter can use all sampled
  rows from the current collection iteration and can build discounted survival
  value targets from the replay metadata.
- With `allow_optimizer_step=True`, `MuZeroPolicy.learn_mode.forward` mutates
  the same policy object used by later bounded collection iterations, then
  checkpoints that object.

Still not full self-play:

- It does not call LightZero `train_muzero`.
- It does not use LightZero's normal collector, GameBuffer, replay priority, or
  distributed actor/learner weight-refresh machinery.
- It is a local bounded smoke/scale lane around a single process policy object,
  not an online production trainer.
- Each seat is evaluated as an independent one-row LightZero call with
  `to_play=-1`; there is no mature multiplayer search contract or explicit
  simultaneous-game solution concept.
- The main CurvyTron `train_muzero` lane remains single-ego versus fixed or
  frozen opponents, and should not be called current-policy self-play.

Are both seats trained into the same policy?

Yes, in the bounded two-seat smoke/scale lane. Replay rows from player 0 and
player 1 are sampled into one learner batch, and the update is applied to the
same shared `MuZeroPolicy` object. There are not separate per-seat policies.
Seat identity exists through observation and replay metadata, not through
separate network heads.

What mature full self-play would need:

- A real two-seat collector/trainer contract, or an upstream-compatible custom
  collector, that owns both seats before every simultaneous `env.step`.
- Explicit actor weight revisions, refresh cadence, checkpoint lineage, and
  replay tagging.
- A replay buffer that accumulates and samples across iterations, not only the
  current bounded collection chunk.
- Clear multiplayer search semantics: searched ego plus policy-sampled
  opponents, joint-action search, or another documented approximation.
- Promotion/evaluation gates across fixed baselines, frozen checkpoints,
  same-run parents, held-out seeds, and seat swaps before claiming learning.

Gate verdict:

This bounded loop is enough as a first learning-signal gate. It proves the
essential mechanical boundary after the target/batch fixes: one current policy
controls both seats, both seats enter the same learner update, discounted
survival targets are available, and checkpoints can be evaluated. It is not
enough to claim mature full self-play or learning quality by itself.
