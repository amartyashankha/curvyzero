# CurvyTron Survival Reward Design - 2026-05-11

## Short Answer

Clarified rule: survival length is always an eval/telemetry metric. Trainer
reward is separate. The first CurvyTron reward comparison should test two
labeled trainer reward variants, with neither treated as settled:

- `sparse_outcome`: game outcome reward only.
- `dense_survival_plus_outcome`: sparse outcome plus a dense survival helper
  for longer horizons.

When using the dense helper, log trainer reward, sparse outcome, and survival
length separately so a long loss is not mistaken for a real outcome
improvement.

Eval/progress survival is episode length. A reward variant or target profile
stored with a model/checkpoint only helps reconstruct the checkpoint's model and
target shape; it does not define the eval score.

Keep this on the stock LightZero `train_muzero` path. The current CurvyTron
patch surface should be env/reward hooks plus eval telemetry, not hidden trainer
changes. Modal smokes for both first reward variants passed on the stock
`train_muzero` path, and the omitted reward default now resolves to
`sparse_outcome`.

Do not make `+1 per survived step` plus `+episode_length` winner bonus the
default objective. It is simple, but it makes both agents share a large survival
reward and makes reward scale grow with horizon. A long loss can beat a short
win under that objective:

```text
1/step for both, +T winner bonus
win at T=40  -> winner return 80
loss at T=100 -> loser return 100
```

That is the wrong default for "survive longer than the opponent." Keep the
first comparison to the two explicit variants above. If the dense helper is
used, bound or normalize it so horizon length does not silently dominate the
game outcome.

## Stale Claims To Reject

- Do not call `source_state_turn_commit` trainable self-play. Its pending
  scalar row has no physics advance and can receive value credit from the later
  commit row.
- Do not call `source_state_joint_action` true two-seat self-play. It is a
  centralized controller over both players with one scalar reward.
- Do not apply per-player winner/loser shaping to a one-scalar centralized
  control wrapper unless the wrapper first grows a real per-player target
  surface.
- Do not promote checkpoints on survival length or dense helper return alone;
  trainer reward, sparse outcome, and survival length must stay separate.

## Current Repo Status

The reward surface is split across paths:

- `source_state_fixed_opponent`: the native LightZero source-state visual env
  plays `player_0` against a fixed-straight `player_1`. It returns scalar
  survival reward to the ego player after each physical source step, `1.0` if
  alive and `0.0` if dead. Terminal source outcome metadata is present in
  `source_terminal_reward_map`, but it is not the scalar training reward.
- `source_state_turn_commit`: the native LightZero turn-commit env alternates
  scalar player turns over a simultaneous source env. `player_0`'s pending
  action receives reward `0.0` because the physical env has not advanced yet;
  `player_1`'s commit advances the physical env and receives survival reward
  for `player_1`. This is useful plumbing telemetry, not trusted simultaneous
  reward credit.
- `source_state_joint_action`: the stock-LightZero control candidate uses one
  scalar action for the joint action `(player_0, player_1)`, advances one real
  source tick, and returns one diagnostic scalar: `+1` if both players are alive
  after that tick, else `0`. This is centralized control, not true competitive
  self-play, and it is not a per-player or zero-sum reward. Do not apply
  two-player terminal winner/loser return shaping here unless the wrapper grows
  an honest per-player target surface.
- Two-seat trainer smoke: the local two-seat replay rows carry per-player
  survival reward from `alive_after`, controlled by `alive_reward` and
  `dead_reward`. Its per-player target construction now uses the
  `terminal_winner_keeps_survival_loser_zero/v0` return schema: survival can
  accumulate during the episode, then a decisive terminal winner keeps its
  accumulated shaped return and the loser target is zeroed. Draws/truncations
  leave shaped survival returns unmodified. The shared learner adapter now
  forces `gamma=1.0` for this schema, so the simple "50-step game gives winner
  50, loser 0" example is literally true for this local target path. The loser
  is zeroed across its whole episode trajectory after the decisive terminal
  winner is known, not only at the terminal transition. Stock LightZero reward
  streams and unlabeled schemas still use the configured policy discount, with
  `0.997` as the default/fallback.
- Public vector trainer observation path: the 1v1 no-bonus trainer reward is
  already sparse zero-sum terminal payoff: survivor `+1/-1`, same-frame draw
  `0/0`, and timeout/truncation `0/0`.

So the clean reward exists in the lower trainer contract, but the current
LightZero source-state path being re-centered on stock LightZero is still using
survival scalar rewards. Keep that label loud.

## Discounting Decision

For the named `terminal_winner_keeps_survival_loser_zero/v0` shaped survival
return, use `gamma=1.0` if the intended target is "number of survived decision
steps, winner keeps that count, loser gets zero." That is the only setting under
which the simple example is literally true:

```text
50-step decisive game, alive_reward=1
gamma=1.000 -> winner target at episode start 50.000, loser 0
gamma=0.997 -> winner target at episode start 46.495, loser 0
gamma=0.990 -> winner target at episode start 39.499, loser 0
```

Using LightZero's configured `discount_factor` is acceptable only when it is an
explicit part of the experiment contract. Do not silently inherit an Atari-style
default such as `0.997` and still describe results as raw survived-step return.
If the run is stock LightZero, the scalar reward stream and native target builder
use the configured discount, so set `policy.discount_factor=1.0` for a finite
episodic survival-count objective, or label the run as discounted survival.

For long CurvyTron horizons, the hard part is reward scale, not discount theory.
With `gamma=1`, the value target range is bounded by
`alive_reward * max_decisions`; with `alive_reward=1` and thousands of decisions,
that can exceed LightZero support assumptions. Prefer one of these before
lowering gamma just to hide scale:

- normalize `alive_reward`, e.g. `1 / max_decisions`, if the target should be in
  `[0, 1]`;
- keep sparse terminal outcome as the training reward and log survival as
  telemetry;
- widen/support-scale the value head deliberately and log clipping rates.

A tuned `gamma < 1` is a real ablation, not the default. It caps effective
horizon at about `1 / (1 - gamma)` and deliberately values near-term survival
more than late survival. That may stabilize targets, but it changes the question
from "how many steps did I survive?" to "how much discounted survival mass did I
collect?" For example, `gamma=0.997` caps infinite all-alive return near `333.33`
with `alive_reward=1`, no matter how long the episode lasts.

Apply loser-zeroing at the trajectory return level after the decisive terminal
winner is known. Equivalently, mask all loser rewards in that decisive episode
before discounting. Do not merely zero the loser's terminal reward: that leaks
early survival credit into losing-state value targets. Winner returns should be
computed from the original survival reward sequence; draw and truncation returns
should keep their documented unmodified shaped-return policy.

## Unresolved Integration Questions

- Which scalar should stock LightZero optimize first: `sparse_outcome` or
  `dense_survival_plus_outcome`? Treat this as the first experiment, not a
  settled answer. Do not mix either under the old survival schema id.
- How should turn-commit credit be represented when LightZero exposes only one
  scalar per `env.step`? The current pending/commit asymmetry can teach seat and
  order artifacts.
- Should timeout be neutral, bad for both, or only an eval reject? Treat this as
  a reward variant knob, not an implicit convention.
- How do bonuses and source lifecycle scoring map into a 1v1 training payoff?
  The first stock LightZero patch should probably keep no-bonus/outcome-only
  semantics, then reintroduce source scoring explicitly.
- Which metric is allowed to promote checkpoints? Sparse outcome should stay
  primary until the eval policy is explicitly changed; survival length can
  diagnose and possibly break ties inside an equivalent sparse-outcome band.

## Knobs To Expose

Expose reward selection as explicit run metadata and config. Minimum useful
knobs:

- `reward_variant`: first-run values are `sparse_outcome` and
  `dense_survival_plus_outcome`. Older names such as
  `sparse_zero_sum_outcome`, `outcome_plus_bounded_alive_aux`,
  `sparse_outcome_plus_anti_stall`, and `survival_diagnostic` should be mapped
  or retired before comparing runs.
- `alive_aux_epsilon`: total episode budget for the dense survival helper;
  require `0.0 <= epsilon <= 0.25` for first
  `dense_survival_plus_outcome` runs.
- `anti_stall_epsilon`: total episode budget for both-alive time cost; require
  `0.0 <= epsilon <= 0.10` for the first runs.
- `max_decisions` or derived `max_ticks / decision_ms`: used to normalize any
  per-step auxiliary so reward scale does not grow with horizon.
- `return_discount`: start at `1.0` for finite survival-count targets; require a
  separate schema/variant label for `return_discount < 1.0`.
- `timeout_reward`: start at `0.0`; log timeout separately and reject timeout
  farming in eval.
- `same_tick_draw_reward`: start at `0.0`.
- `reward_schema_id` and `reward_schema_hash`: change whenever the scalar
  reward stream changes.

## Metrics That Must Stay Separate

Every run should log four separate families, not one blended return:

- Scalar training reward: the exact value passed to LightZero.
- Sparse outcome reward: win/loss/draw/timeout payoff under the clean
  zero-sum contract, even when not used for training.
- Survival length diagnostics: decisions survived, physical time survived,
  win/loss conditioned episode length, timeout rate, and p90/max episode
  length.
- Terminal facts: winner ids, loser ids, same-tick deaths, terminal reason,
  death cause/hit owner when available, and seat-swapped rates.

For learner debugging, also log reward/value target min, max, mean, std, and
support clipping rate separately for scalar training reward and sparse outcome
return.

Target/support tuning belongs in training config, not in the reward definition.
If a run changes value/reward support size, support scale, discount, or td
steps to fit a reward stream, label that as a separate ablation and report it
next to the reward variant.

Eval is separate from this. Survival eval should score episode length and
terminal facts. A checkpoint eval may still need the checkpoint's
`model_reward_variant` to rebuild the LightZero support/model shape, but that is
load plumbing, not the progress metric.
Likewise, any checkpoint-side reward variant or target profile is load-shape
metadata, not an eval-score contract.

Eval cadence rule: do not run CurvyZero checkpoint survival eval on every
checkpoint by default. Keep background checkpoint eval disabled for long
training runs, then score a small selected checkpoint set with the standalone
eval harness after the run or at sparse milestones. LightZero's stock evaluator
is a separate in-training signal controlled by `lightzero_eval_freq`.

Current run evidence, 2026-05-11:

- Modal smokes passed for both `sparse_outcome` and
  `dense_survival_plus_outcome` under
  `curvytron-reward-variant-smoke-20260511`.
- Omitting `--reward-variant` now resolves to `sparse_outcome`, verified by
  `curvytron-reward-default-smoke-20260511`.
- A launch canary after the stop-cap fix wrote volume artifacts under
  `curvytron-reward-stopcap-smoke-wait2-20260511`.
- The active comparison pair is
  `curvytron-reward-compare-sparse-sim16-waitlong-20260511` and
  `curvytron-reward-compare-dense-sim16-waitlong-20260511`.

## Evidence

AlphaZero trains value against game outcome: `-1` loss, `0` draw, `+1` win, not
a hand-shaped living score. MuZero adds a reward head, but it predicts the
observed environment reward stream plus value and policy targets; changing the
reward stream changes the objective, not just the logging.

Multi-agent reward structure is behavior structure. Shared rewards promote
cooperation; zero-sum rewards promote direct competition. Pong studies that vary
the reward scheme show the same game can become competitive or cooperative, and
cooperative Pong agents learn to keep the ball alive and delay serving when play
can only make things worse.

Potential-based shaping is the clean theoretical exception, but a flat living
bonus is not obviously `gamma * Phi(s') - Phi(s)`. Multi-agent extensions exist,
but they still require a real potential-style construction. "Alive this step"
does not get policy-invariance for free.

## Skeptical Read On Survival Plus Outcome

Let `T` be the decisive episode length and assume `alive_reward=1`.

- Sparse terminal win/loss optimizes the right first question: did this seat win
  the round? With `gamma=1`, a win is a win regardless of whether it took 40 or
  400 decisions. With `gamma<1`, the objective changes: faster wins are worth
  more and delayed losses are less bad. That can be a deliberate speed bias, but
  it is not neutral.
- Per-step survival only mostly optimizes "do not end the game." In a two-player
  death game, both players collect almost the same positive reward for a long
  shared survival period, so the competitive signal is weak and late losses can
  look good in aggregate logs.
- Winner-keeps-survival/loser-zero fixes the worst logging bug, because a
  decisive loser no longer keeps a positive training return. It still says that
  a long win is better than a short win. In symmetric self-play, both agents can
  prefer states with larger expected `P(win) * T`, which can reward delaying
  contact instead of improving win probability.
- Winner survival plus `+T` and loser `-T` is not a principled fix. If the loser
  also received `T` survival, `-T` just cancels the loser's survival while the
  winner becomes `2T`. If the penalty is larger than survival, an inevitable
  loser may prefer a shorter loss. Either way, the target scale is tied to
  horizon and curriculum knobs.
- A flat living bonus is not potential-based shaping unless a real potential is
  written down and checked. Potential-based shaping is safe because the shaping
  rewards telescope to a boundary term. A reward whose total is `T` does not
  telescope away when episode length depends on the agents' actions.

## Reward Variants To Test

Concrete verdict for the proposed variants:

| Variant | Reward | Verdict |
| --- | --- | --- |
| A | `sparse_outcome` | First trainer reward candidate: game outcome reward only. Not yet settled. |
| B | `dense_survival_plus_outcome` | First trainer reward candidate: sparse outcome plus dense survival helper for longer horizons. Not yet settled. |
| C | Winner keeps survival, loser zero | Useful smoke/debug target, but not the real game payoff and can still overvalue long games. |
| D | `+T` winner / `-T` loser terminal shaping | Reject as a default: horizon-dependent scale, no policy-invariance guarantee, and easy support/value-scale trouble. |
| E | Loser keeps partial survival | Reject for competitive training: it explicitly rewards losing later and can mask flat win rate. |

### 1. `sparse_outcome`

Game outcome trainer reward only:

```text
both alive:              0, 0
ego survives, opp dies: +1, -1
ego dies, opp survives: -1, +1
same-tick draw:          0, 0
time-limit truncation:   0, 0  # logged separately
```

This matches the game objective, keeps reward scale stable, and makes self-play
comparisons cleaner. It may be slow to explore, so it should be compared
against the dense-helper variant rather than assumed to be settled.

### 2. `dense_survival_plus_outcome`

Sparse outcome plus a dense survival helper for longer horizons:

```text
per decision while alive: +epsilon / max_decisions
terminal:                +1 / -1 / 0
epsilon:                 0.05 to 0.25
```

No episode-length winner bonus. Prefer a bounded or normalized helper so value
scale is not controlled by `max_ticks`. This is a trainer reward variant, while
survival length remains eval/telemetry. It can still encourage stalling, so log
terminal causes and timeout rate.

### 3. Sparse Outcome Plus Anti-Stall Time Cost

Use if policies learn to circle forever or timeout farm:

```text
per decision while both alive: -epsilon / max_decisions for both
terminal:                     +1 / -1 / 0
epsilon:                      0.02 to 0.10
```

This asks for decisive wins, not just survival. It can over-encourage risky
contact or suicide if too large, so keep it tiny and compare terminal causes.

## Risks And Footguns

- Shared living reward changes the game from competitive toward cooperative
  survival.
- Length-scaled winner bonuses change value scale across `decision_ms`,
  `max_ticks`, and curriculum settings.
- If unnormalized returns exceed MuZero support or local scalar assumptions,
  reward/value heads can spend capacity on scale rather than ranking actions.
- Survival shaping can make "lose later" look like progress while win rate is
  flat.
- Time-limit truncations must not look like successful survival unless the eval
  metric says so explicitly.
- Reward changes must get new reward schema ids; old and new replay should not
  be mixed silently.

## Metrics To Log

- reward schema id, `epsilon`, `max_decisions`, `decision_ms`, `max_ticks`
- true sparse outcome return even for shaped runs
- shaped return separately from true return
- win/loss/draw/truncation rates, seat-swapped
- survival decisions and physical time: mean, median, p90, max
- terminal reason: wall, own trail, opponent trail/body/head, draw, timeout
- winner id, loser id, same-tick death flag
- action histogram, top-action fraction, per-seat entropy
- episode length conditioned on win, loss, draw, timeout
- value/reward target min, max, mean, std, and support clipping rate
- policy-vs-baseline panels against random, straight, wall-avoid, and earlier
  checkpoints

## Recommendation

Do the next stock-LightZero reward work in this order:

1. Add metadata-only reward accounting to the source-state LightZero runs:
   current scalar reward, sparse outcome reward, reward variant id, epsilon
   fields, timeout/draw convention, and terminal facts.
2. Add `sparse_outcome` without changing observation or collection topology.
3. Add `dense_survival_plus_outcome` as a separate labeled trainer reward
   variant for longer horizons.
4. Compare both first-run variants with the same eval scorecard. Do not claim
   either variant is settled from design alone.
5. Keep target/support changes explicit as training-config ablations, not
   hidden reward shaping.
6. Add anti-stall cost only if timeout farming appears in sparse or dense-helper
   runs.

Keep survival length as telemetry throughout. Do not test the proposed
unbounded `+1/step + T winner bonus` as a serious candidate unless the goal is
explicitly to study the failure mode.

## Sources

- AlphaZero paper: https://arxiv.org/abs/1712.01815
- MuZero paper: https://arxiv.org/abs/1911.08265
- AlphaGo Zero Nature page: https://www.nature.com/articles/nature24270
- Multi-agent reward taxonomy survey:
  https://link.springer.com/article/10.1007/s10462-021-09996-w
- Multiagent Pong reward-structure study:
  https://doi.org/10.1371/journal.pone.0172395
- Potential-based reward shaping:
  https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html
- Multi-agent potential-based shaping:
  https://pure.york.ac.uk/portal/en/publications/theoretical-considerations-of-potential-based-reward-shaping-for-/
- ALE multi-agent Pong reward/timer docs:
  https://ale.farama.org/multi-agent-environments/pong/
- Time limits in RL:
  https://proceedings.mlr.press/v80/pardo18a.html
- LightZero MuZero policy/source docs:
  https://www.aidoczh.com/lightzero/_modules/lzero/policy/muzero.html
- Reward misspecification example:
  https://openai.com/index/faulty-reward-functions/
