# CurvyTron Two-Seat Reward Contract - 2026-05-12

## Current Trainer Reward

Active path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay
```

The collector writes one reward float per per-seat replay row. That is the
reward LightZero sees through the replay sample and learner target code.

Default formula:

```text
dense_survival_helper = 0.01 if the seat is alive after the step else 0.0
terminal_outcome = env_sparse_outcome * 0.01 * episode_step_count if done else 0.0
training_reward = dense_survival_helper + terminal_outcome
return_target_discount = 1.0
```

`env_sparse_outcome` is the env terminal payoff for that player: `+1` for the
winner, `-1` for the loser, `0` for draw/truncation/nonterminal.

## Why This Shape

Pong is not the right reward-shaping comparison. The useful LightZero contract
is simpler: environments emit one reward per transition, and replay/learner code
uses that sequence. The CurvyTron adapter should stay close to that path.

The small survival helper gives earlier learning signal. The terminal outcome is
scaled by episode length so the helper cannot dominate just because games get
long. A 100-step loss gives about `99 * 0.01 - 100 * 0.01 = -0.01`; a 100-step
win gives about `100 * 0.01 + 100 * 0.01 = 2.0`.

## Logging Rule

Do not collapse the components in reports. Each replay row should keep:

- `reward`: final training reward consumed by replay/learner.
- `dense_survival_helper_reward`
- `sparse_outcome_reward`
- `terminal_outcome_reward`
- `episode_step_count`
- `return_target_discount`

Eval survival length is still telemetry, not proof by itself.
