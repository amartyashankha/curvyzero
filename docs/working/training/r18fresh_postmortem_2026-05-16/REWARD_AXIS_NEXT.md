# Reward Axis Next

Created: 2026-05-16

## Plain Definition

The next reward knob should be the terminal outcome coefficient.

Use one formula in spirit:

```text
training_return = survival_reward + bonus_reward + alpha * episode_length * outcome
```

where:

- `survival_reward`: dense reward for staying alive.
- `bonus_reward`: same-step reward for catching bonuses.
- `outcome`: `+1` for win, `-1` for loss, `0` for draw/truncation.
- `episode_length`: source-step count for the finished game.
- `alpha`: the new knob.

Implementation correction: do not let the losing outcome term make the total
episode return negative. The safer concrete contract is:

```text
base_return = survival_reward + bonus_reward
outcome_term = alpha * survival_reward * outcome
training_return = base_return + outcome_term
```

This keeps loss returns nonnegative for `alpha <= 1`. The terminal timestep may
still contain a negative adjustment because it subtracts from survival already
earned, but the episode-level return must not go below zero.

Current names:

- `no_out`: `alpha = 0`. Outcome is telemetry only.
- old `plus_out`: `alpha = 1` against `episode_length`; this can make loser
  episode returns slightly negative because of terminal-step conventions.
- new `plus_out`: `alpha = 1` against accumulated survival reward; loser return
  floors at bonus/nonnegative survival credit.

## Why This Matters

For a 30-step game and no bonus, approximate returns are:

| alpha | loser | winner | readout |
| --- | ---: | ---: | --- |
| `0.0` | `30` | `30` | pure survival, win/loss ignored |
| `0.25` | `22.5` | `37.5` | gentle outcome pressure |
| `0.5` | `15` | `45` | balanced outcome pressure |
| `1.0` | `0` | `60` | full nonnegative plus-outcome scale |

The old code has an off-by-one-style terminal convention because the terminal
dead step gets `0` alive reward. The new contract should use accumulated
survival reward, not raw source tick count, so the loser does not go negative.

## Equivalence Check

`0/60` and `15/45` are equivalent only in a fixed-length, fixed-terminal toy
comparison if you ignore neural learning dynamics: `15/45 = 0.5 * (0/60) + 15`.

They are not equivalent in this project because game length is policy-dependent.
A constant that depends on episode length changes the objective. It rewards or
penalizes survival differently as policies alter how long games last.

Also, MuZero-style neural training is sensitive to reward scale even when an
ideal tabular objective would preserve the same optimal policy. Changing alpha
changes target scale, value support pressure, and loss balance.

## Theory Notes

- Positive scaling of all rewards preserves the ideal optimal policy, but changes
  optimization dynamics.
- Adding a constant reward is safe only if every policy receives the same total
  added return.
- In variable-length survival games, per-step constants are not policy-neutral
  because better or worse policies change episode length.
- Potential-based shaping is the classic safe shaping form; our survival reward
  is intentionally not just neutral shaping because survival is part of the
  desired behavior.

Primary references checked:

- Ng, Harada, Russell 1999, policy-invariant reward shaping.
- Sutton and Barto, Reinforcement Learning: An Introduction, episodic versus
  continuing return behavior.
- Lu, Schwartz, Givigi 2014, potential-based shaping for stochastic games.
- MuZero paper, reward/value support and learned reward/value dynamics.

## Next Experiment

Do not add another vague reward variant name first. Add an explicit coefficient
field and launch a clean axis:

```text
alpha in {0.0, 0.25, 0.5, 1.0}
```

Keep the bonus reward on. Use the strongest observed opponent recipe as the
main lane, with clean and stochastic-opponent variants if capacity allows.

Track these separately:

- Eval survival.
- Own training reward.
- Inferred terminal outcome residual.
- Tournament rank.
- Tournament game duration.
- Whether latest checkpoints retain strength or only mid-run checkpoints peak.
