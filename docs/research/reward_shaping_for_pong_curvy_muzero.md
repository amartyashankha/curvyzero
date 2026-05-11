# Reward Shaping For Pong, Curvy, And MuZero

Status: Recommendation for dummy Pong v0; CurvyTron note updated after reward
variant clarification
Date: 2026-05-09

## Short Answer

The coach rule holds, with one tightening:

```text
env.step reward:
  +1  ego scores / wins
  -1  opponent scores / ego loses
   0  otherwise
```

Do not silently add positive per-step survival reward inside `env.step()`. For
MuZero or LightZero MuZero, any survival helper in the reward head is a separate
trainer reward variant, not just logging. Use survival length as telemetry,
checkpoint-selection tie-breaker, and curriculum control signal. If a run
trains on a shaped episode target or dense helper, label it as
objective-changing and do not use survival length alone for promotion.

Simple decision:

| Use | Survival time? | Rule |
| --- | --- | --- |
| Environment reward | No | Keep game payoff sparse and true. |
| MuZero reward target | Variant only | Reward head predicts the chosen trainer reward stream. |
| MuZero value target | Usually no | Use true return; shaped value is a labeled ablation only. |
| Eval telemetry | Yes | Report survival, truncation, loss delay, and variance. |
| Checkpoint tie-breaker | Yes, bounded | Only after true win/score is tied, with anti-stall rejects. |
| Curriculum signal | Yes | Use it to choose opponents, starts, caps, or seeds, not to redefine reward. |

## Why The Rule Holds

AlphaGo Zero and AlphaZero keep the task target close to the game: self-play
policy improvement plus a value target for the game winner, with no hand-built
survival objective beyond the rules. MuZero adds a learned reward model, but the
model predicts reward, policy, and value needed for planning; in Atari this
means the observed game reward stream, not a custom "stay alive" replacement.
LightZero's MuZero learner similarly has explicit `target_reward` and
`target_value` surfaces, so changing the reward target is a real objective
change, not just logging.

Potential-based reward shaping is the main theoretical escape hatch. A shaping
reward of the form

```text
F(s, s') = gamma * Phi(s') - Phi(s)
```

can preserve optimal policies when the discount, potential, and terminal
conventions are fixed. A flat living bonus or "lose later is less bad" episode
target does not obviously have that form. It pays for extra time, so it can
change preferences between scoring now, risking a score, or extending the
point.

Pong-like games make the risk concrete. ALE/PettingZoo multi-agent Pong gives
+1 to the scorer and -1 to the opponent, and adds a serve timer specifically to
prevent indefinite stalling. A survival bonus creates the same kind of pressure
from our side: it can pay both agents for longer rallies, passive play,
timeout farming, or policies that look better on shaped return while true score
does not improve.

## Critique Of The Current Coach Rule

The rule is right, but "use shaped return as telemetry and maybe temporary
selection/tie-breaker" needs guardrails:

- As telemetry, survival is strongly useful. Early sparse-reward Pong may show
  loss-delay improvement before win rate moves.
- As a tie-breaker, survival is safe only within a true-score equivalence band.
  A checkpoint with worse win rate or score margin should not win because it
  survives longer.
- As a temporary training target, survival/loss-delay is acceptable only for
  crude baselines or debugging runs, not as the default MuZero target. Such runs
  should carry a tag like `temporary_shaped_target`.
- As a curriculum signal, survival is often the best use: pick weaker
  opponents, shorter horizons, reset states, or seed buckets where score
  pressure appears, then evaluate on the unshaped game.

The main failure mode is narrative drift. Once shaped return appears in replay
or summaries, it is easy to start calling it progress. Every run should keep
`score_return`, `shaped_loss_delay`, `timeout_rate`, and selection metric
separate.

## Recommended V0 Plan

Use four separate quantities:

| Quantity | Formula | Where to use |
| --- | --- | --- |
| Environment reward | `+1` ego score, `-1` opponent score, `0` otherwise | `env.step()`, replay rewards, MuZero reward head, n-step true returns |
| True eval metrics | win rate, score margin, timeout/truncation rate | dashboards, run summaries, improvement claims |
| Loss-delay diagnostic | `+1` win, `-1 + 0.5 * survival_fraction` loss, `0` timeout | telemetry, plots, early-progress readout |
| Curriculum state | survival buckets, loss length, terminal cause, opponent id | opponent ladder, seed selection, reset-state choice, horizon/cap changes |

Current diagnostic formula:

```text
survival_fraction = episode_steps / max_steps

if ego wins:
    shaped_loss_delay = +1.0
elif ego loses:
    shaped_loss_delay = -1.0 + 0.5 * survival_fraction
else:  # timeout
    shaped_loss_delay = 0.0
```

Checkpoint selection should be deliberately boring:

```text
primary_selection = mean_true_score_margin or win_rate
eligible_tie_band = no material drop in primary_selection
tie_breaker = mean_shaped_loss_delay or p90 survival

reject if timeout_rate rises materially
reject if score margin or win rate worsens
reject if survival increases only by avoiding scoring pressure
reject if action entropy collapses into passive/noop behavior
```

For the next dummy Pong runs:

1. Keep `env.step()` reward sparse and unchanged.
2. Train LightZero/MuZero reward targets on the environment reward only.
3. Report survival/loss-delay in every scorecard, including variance and
   truncation rate.
4. Use survival to build curriculum buckets: fast losses, long losses, wins,
   timeouts, and opponent type.
5. Promote only by true heldout score/win metrics; use survival only to choose
   which tied candidate deserves the next run.

## Carry Forward To CurvyTron

CurvyTron should inherit the same hierarchy:

- Survival length: always eval/telemetry, never the trainer reward by itself.
- Trainer reward: first runs should compare two labeled variants,
  `sparse_outcome` and `dense_survival_plus_outcome`.
- Sparse outcome: terminal competitive payoff, such as win/loss/tie or centered
  rank payoff for more than two players.
- Telemetry: survival ticks, crash cause, distance-to-death probes, pressure
  events, timeout rate, and rank/placement.
- Tie-breaker: survival only among candidates with statistically similar true
  payoff.
- Curriculum: use survival and crash causes to pick easier starts, weaker
  opponents, shorter horizons, or targeted reset states.

Do not silently reward "not dying" in the CurvyTron env. The
`dense_survival_plus_outcome` variant may include a dense survival helper for
longer horizons, but it must have its own reward schema id and scorecard. Do
not claim `sparse_outcome` or `dense_survival_plus_outcome` is settled before
the first CurvyTron comparison.

## Open Checks

- Does true sparse reward plus prioritized replay of rare scoring events beat
  the old temporary loss-delay target?
- What timeout-rate increase should disqualify a checkpoint?
- Should one-point Pong use score margin first, then switch to win rate once
  matches contain multiple points?
- Can we build a tiny anti-stall eval where endless rallies are clearly worse
  than scoring?
- For CurvyTron, what curriculum buckets best predict later true win/rank
  improvement: survival ticks, crash cause, opponent proximity, or pressure
  events?

## Sources

- Silver et al., 2017, "Mastering the game of Go without human knowledge":
  https://www.nature.com/articles/nature24270
- Silver et al., 2017, "Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm": https://arxiv.org/abs/1712.01815
- Schrittwieser et al., 2020, "Mastering Atari, Go, Chess and Shogi by Planning
  with a Learned Model": https://arxiv.org/abs/1911.08265 and
  https://www.nature.com/articles/s41586-020-03051-4
- LightZero MuZero policy source, showing `target_reward`, `target_value`, and
  predicted reward/value training surfaces:
  https://opendilab.github.io/LightZero/_modules/lzero/policy/gumbel_muzero.html
- Ng, Harada, and Russell, 1999, "Policy Invariance Under Reward
  Transformations": https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html
- Lu, Schwartz, and Givigi, 2011, "Policy Invariance under Reward
  Transformations for General-Sum Stochastic Games":
  https://auld.aaai.org/Library/JAIR/Vol41/jair41-012.php
- ALE multi-agent Pong documentation:
  https://ale.farama.org/multi-agent-environments/pong/
- OpenAI, 2016, "Faulty reward functions in the wild":
  https://openai.com/index/faulty-reward-functions
