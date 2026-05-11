# LightZero Two-Seat Reward Target Quick Check - 2026-05-10

Scope: `src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py`
and the two-seat smoke path in
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`.

## Current Target Semantics

The two-seat smoke records one replay row per active ego seat with:

- `observation`: stacked visual observation, currently `[4, 64, 64]`.
- `next_observation`: next stacked visual observation for the same env/player.
- `action`: selected LightZero action id.
- `action_weights`: search visit distribution or one-hot fallback, shape `[3]`.
- `root_value`: search root value if available.
- `reward`: survival reward for that env/player after the step.
- `done`: env-row done flag.

`_sample_replay_batch(...)` flattens those rows into arrays:

- `observation_batch`: `[N, 4, 64, 64]`.
- `next_observation_batch`: `[N, 4, 64, 64]`.
- `action_batch`: `[N]`.
- `reward_batch`: `[N]`.
- `done_batch`: `[N]`.
- `policy_batch`: `[N, 3]`.

`_learn_mode_batches(...)` then builds LightZero targets from the flat sample.
With the current profile config `num_unroll_steps == 1`:

- `target_reward[row] == [reward_batch[row]]`.
- `target_value[row] == [0.0, reward_batch[row]]`.
- `target_policy[row]` repeats the policy target for root plus one unroll step.

So an alive row with reward `1.0` trains immediate reward as `1.0`, and trains
the value target as `[0.0, 1.0]`. It does not train the value head to predict
the remaining survival lifetime.

## Toy 20-Step Alive Trajectory

For 20 consecutive alive rewards of `1.0`, the current adapter emits the same
target rows at every timestep:

```text
current target_reward rows: [1.0]
current target_value rows:  [0.0, 1.0]
```

An undiscounted survival-return value target would instead be:

```text
[20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0,
 10.0,  9.0,  8.0,  7.0,  6.0,  5.0,  4.0,  3.0,  2.0,  1.0]
```

The pure helper
`toy_alive_survival_target_diagnostic(steps=20, discount=1.0)` now returns both
views as structured data.

## Read

The immediate reward target is present, but the survival-return value signal is
weak or missing in this adapter. This is especially important for two-seat
CurvyTron because the replay sample is flat ego rows, not grouped trajectory
segments. A correct value-target patch should first carry enough metadata to
group rows by `(iteration, env_row_id, player_id)` and order by
`decision_index`, then compute discounted returns with terminal masks.

Recommended next patch: extend the two-seat sample with `iteration`,
`env_row_id`, `player_id`, `decision_index`, and `done_batch`; add a small
`build_discounted_survival_value_targets(...)` helper that computes per-seat
discounted returns over contiguous rows; wire `_learn_mode_batches(...)` to use
that value target only when the sample provides the metadata needed to prove the
ordering.
