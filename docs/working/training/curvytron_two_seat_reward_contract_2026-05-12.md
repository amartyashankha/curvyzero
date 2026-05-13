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
bonus_pickup_helper = 0.05 * bonuses caught by that player on this exact step
terminal_outcome = env_sparse_outcome * 0.01 * episode_step_count if done else 0.0
training_reward = dense_survival_helper + bonus_pickup_helper + terminal_outcome
return_target_discount = 1.0
```

`env_sparse_outcome` is the env terminal payoff for that player: `+1` for the
winner, `-1` for the loser, `0` for draw/truncation/nonterminal.
The bonus pickup helper is immediate. It is not delayed to the end of the game,
and progress sums are only logs.

## Why This Shape

Pong is not the right reward-shaping comparison. The useful LightZero contract
is simpler: environments emit one reward per transition, and replay/learner code
uses that sequence. The CurvyTron adapter should stay close to that path.

The small survival helper gives earlier learning signal. The terminal outcome is
scaled by episode length so the helper cannot dominate just because games get
long. A 100-step loss gives about `99 * 0.01 - 100 * 0.01 = -0.01`; a 100-step
win gives about `100 * 0.01 + 100 * 0.01 = 2.0`.

Bonus pickup reward is a small exploration reward. It credits the player who
catches a bonus on that exact policy step, even if the bonus type later turns
out situationally bad. The default `+0.05` is equal to five alive-helper steps.

## Logging Rule

Do not collapse the components in reports. Each replay row should keep:

- `reward`: final training reward consumed by replay/learner.
- `dense_survival_helper_reward`
- `bonus_pickup_count`
- `bonus_pickup_reward`
- `sparse_outcome_reward`
- `terminal_outcome_reward`
- `episode_step_count`
- `return_target_discount`
- terminal metadata for auditing: `terminal_winner`, `terminal_reason_name`,
  `death_count`, `death_player`, and `death_cause_name`

Eval survival length is still telemetry, not proof by itself.

## Death Signal Audit

Fresh policy-decision rows propagate death into training:

1. The vector runtime marks wall, own-trail, opponent-trail, and body deaths in
   `alive`, `death_cause`, and terminal lifecycle metadata.
2. `VectorMultiplayerEnv` emits the public sparse outcome: winner `+1`, loser
   `-1`, draw/truncation `0`.
3. The two-seat trainer writes one replay row per fresh current-policy seat
   decision. That row stores the shaped reward components plus death/terminal
   metadata.
4. `_sample_replay_batch` copies `reward` into `reward_batch` and the full row
   history into `return_context_reward_batch`.
5. `_learn_mode_batches` sends immediate rewards as `target_reward` and builds
   discounted per-player `target_value` from the replay context.

Death cause is audit metadata only right now. Wall death and trail death do not
have different reward values; both become the same win/loss/draw outcome plus
the alive/dead helper.

Important fence: policy no-op skip is not a training feature yet. If a skipped
physical tick kills a player, that tick has no fresh policy row, so its terminal
reward would not be a learner target. The trainer now refuses real optimizer
training when `policy_action_repeat_*` can create skipped policy ticks. Keep
`policy_action_repeat_min=1`, `policy_action_repeat_max=1`, and
`policy_action_repeat_extra_probability=0` for learning runs until skipped-tick
reward aggregation is implemented.
