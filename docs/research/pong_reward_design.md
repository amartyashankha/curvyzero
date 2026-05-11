# Pong Reward Design

Status: Updated for self-play toy training
Date: 2026-05-09

Scope: dummy Pong toy training, with notes for later CurvyTron-style
self-play.

## Short Answer

Keep native score as the environment and eval reward.

For dummy Pong v0, reward score changes:

```text
reward_for_player_i_at_step_t =
  +1.0 if player_i scores on this step
  -1.0 if the opponent scores on this step
   0.0 otherwise
```

For the first Pong self-play trainer, use a separate shaped training target
that gives partial credit for losing later. This is not the environment reward
and not the scoreboard metric.

```text
survival_fraction = episode_steps / max_steps

if ego wins:
    shaped_return = +1.0
elif ego loses:
    shaped_return = -1.0 + 0.5 * survival_fraction
else:
    shaped_return = 0.0
```

This keeps wins clearly best, makes fast losses clearly bad, and gives the
learner a smoother signal when it is still weak.

The true eval metric should stay win rate and score margin against fixed
opponents on heldout seeds. If we later add shaped training reward, eval should
still ignore the shaping.

Recent Pong note: the current high-level metrics shifted because we ran
separate probes with different targets, replay/data, policy classes, feature
modes, and Modal wrappers. The env reward did not change. Default `track_ball`
is now a survival/tie floor in the current default geometry, because exact
search found no normal-reset scoring path against it within the cap. That is
not an optimality proof for all Pong. `lagged_track_ball_1` is a one-tick-lag
version that tracks the previous ball row; it is the current scoreable target.
For learned-vs-`track_ball`, report wins, mean survival steps, truncation rate,
and shaped proxy together. Do not turn the readout back into only a win
fraction.

Variance note: variance is not part of the environment reward, but it may be
useful for early exploration. If two checkpoints have similar mean score and
mean survival, a small bounded bonus for higher survival variance, higher p90
survival, rare wins, or rare long rallies can help keep promising policies
alive. This is a checkpoint-selection or replay-prioritization trick, not a
license to reward random behavior forever.

## What The Sources Say

AlphaZero-style board-game training is mostly outcome reward. In the AlphaZero
paper, each self-play game is scored at the end: loss is `-1`, draw is `0`, and
win is `+1`. The value head learns expected outcome, not a hand-shaped position
score. The paper also says AlphaGo Zero optimized win probability, while
AlphaZero optimized expected outcome so draws and other outcomes can be handled.
Source: Silver et al.,
["Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
Algorithm"](https://arxiv.org/pdf/1712.01815), especially the training
description around game outcome.

MuZero keeps the AlphaZero setup for board games, but it also supports
intermediate environment rewards. In the MuZero paper, the learned model predicts
policy, value, and immediate reward. For board games, the authors state that
discount is `1` and there are no intermediate rewards. For Atari, MuZero learns
from observed game rewards and n-step returns. Source: Schrittwieser et al.,
["Mastering Atari, Go, Chess and Shogi by Planning with a Learned
Model"](https://arxiv.org/pdf/1911.08265).

Atari Pong's native reward is score. ALE's Pong docs say the player gets points
when the ball passes the opponent's paddle and loses points when the ball passes
their own paddle. The multi-agent Pong docs use `+1` for scoring and `-1` for
the opponent. Sources: [ALE Pong](https://ale.farama.org/environments/pong/)
and [ALE multi-agent Pong](https://ale.farama.org/multi-agent-environments/pong/).

DeepMind's earlier Atari DQN work also used pixels plus game score rather than
hand-shaped Pong features. Its Pong value example describes `-1` for losing the
ball and `+1` when about to score. Source: Mnih et al.,
["Human-level control through deep reinforcement learning"](https://www.nature.com/articles/nature14236).

Reward shaping can help learning, but it can also change what the agent tries
to do. The classic safe-shaping result is potential-based shaping: extra reward
must have a specific form if we want to preserve the optimal policy in an MDP.
Source: Ng, Harada, and Russell,
["Policy Invariance Under Reward Transformations"](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

DeepMind's specification-gaming note is the practical warning: agents can find
ways to satisfy the literal reward while missing the intended outcome. Source:
Krakovna et al.,
["Specification gaming: the flip side of AI ingenuity"](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity).

## Direct Answers

### Do AlphaZero and MuZero usually use outcome, per-step, or shaped reward?

For board games, AlphaZero and MuZero use game outcome, not dense shaped
rewards. The learner gets many training targets because every position in a
self-play game is trained toward the final outcome, but the reward itself is not
distance, material, territory guess, or "good move" shaping.

For Atari, MuZero uses the game's observed rewards. These can arrive during the
episode, such as scoring in Pong. That is still native game reward, not a
custom reward for being near the ball.

### For Pong, what should we reward?

Use score changes for environment and eval.

Do not start with final match win/loss only if the match contains several
points. Per-point score is a cleaner and less sparse signal. It is still the
real game objective.

Do not report rally length as the reward or eval metric. For the first toy
self-play trainer, it is acceptable to use final episode length as part of a
separate training target for losing games, because that gives a weak learner a
direction before it can win. Keep that weight small and watch for stalling.

Do not reward paddle hits in v0. A paddle hit is often useful, but it is not the
goal. A policy can learn to value safe, easy returns over winning shots.

Do not reward distance to the ball in v0. It can teach the paddle to chase the
ball in ways that look good moment by moment but are not always good for
positioning or scoring.

Current implementation note: the old loss-delay lookahead target is a
diagnostic and did not create useful non-`track_ball` labels. The current plan
is simpler: use self-play episodes and the shaped episode return above.

### What is dangerous about rewarding both players for surviving longer?

It changes the game. Pong is competitive. If both players get positive reward
for each extra tick, the easiest shared behavior can become "keep the rally
alive" instead of "score."

This creates several failures:

- Both players can get high training reward from long rallies even if neither
  improves at winning.
- Agents may learn to stall, serve badly, or avoid finishing points.
- Timeouts become attractive.
- The game stops being cleanly zero-sum, so self-play value targets become
  harder to read.
- Training curves can look better while heldout win rate stays flat or gets
  worse.

ALE's multi-agent Pong docs are a useful warning here: they add a serve timer to
prevent indefinite stalling, and note that this makes the environment no longer
purely zero-sum.

## Recommended V0 Plan

Keep the environment reward score-delta only:

```text
score_delta_i = score_i_after_step - score_i_before_step
score_delta_opp = score_opp_after_step - score_opp_before_step

reward_i = score_delta_i - score_delta_opp
```

For normal one-point Pong events, this gives:

```text
ego scores:       +1
opponent scores:  -1
no score event:    0
```

If the toy game ends after one point, this is the same as terminal win/loss. If
the toy game ends after several points, it gives useful per-point learning
signal without adding fake objectives.

Keep these logs separate from reward:

- rally length
- paddle hits
- distance from paddle center to ball at crossing time
- score margin
- win/loss
- timeout count

These are diagnostics. Episode length can also feed the shaped training return,
but it should not become the scoreboard.

## Training Target Shaping

The shaped training return for the first self-play loop is:

```text
win:      +1.0
loss:     -1.0 + 0.5 * episode_steps / max_steps
timeout:   0.0
```

Do not combine many shaping terms. Do not tune hit reward, distance reward,
alignment reward, time penalty, and win reward all at once. That will make
failures hard to explain.

Promotion should still use real outcomes. If shaped return rises because the
policy learns to stall, lower the weight or add a timeout penalty.

## Variance And Exploration

The user correction is important: variance is not only a chart. It can help
keep exploration alive while the learner is weak.

Use it carefully:

- Do not add variance to `PongEnv.step()` rewards.
- Do log per-checkpoint survival standard deviation, p90 survival, max
  survival, rare win count, and shaped-return standard deviation.
- During early training only, use a small bonus when selecting which checkpoint
  to keep training from:

```text
selection_score =
  mean_shaped_return
  + 0.05 * survival_std
  + 0.05 * (p90_survival_fraction - mean_survival_fraction)
```

Clamp or remove this bonus once wins start moving. A policy that only creates
noise, stalls forever, or gets worse against `random_uniform` must not be
promoted because it has high variance.

Replay can also use this idea without changing the game reward: sample rare
long rallies, rare scoring events, off-center paddle contacts, and surprising
losses more often so the learner sees them. Keep this separate from the real
scoreboard.

## CurvyTron Link

For CurvyTron-style games, keep the same separation. Eval uses the real
competitive outcome. Training can use small shaping only when it is logged
separately and checked against stalling or passive timeout farming.

The true eval metric should stay outcome-based:

- Pong: heldout win rate, score margin, and timeout rate against fixed
  opponents.
- CurvyTron: paired-seat win/loss/draw or rank payoff, terminal causes, and
  timeout rate against fixed opponents.

Training reward may be shaped later, but eval should answer the real question:
does the policy win more under the actual game rules?

## What Not To Overcomplicate

Do not build a reward cocktail yet.

Do not add a learned reward model.

Do not add per-opponent reward terms.

Do not use shaping to hide a bad observation or broken action mapping.

First prove the toy can learn something from score-delta reward, with clean
heldout eval. Then add exactly one shaped term only if the baseline failure says
we need it.
