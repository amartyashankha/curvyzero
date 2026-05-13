# Stock Train-MuZero Vs Two-Seat Profile Plan

Date: 2026-05-12

Audience: Optimizer first, Coach second.

## Plain Goal

Check whether the stock LightZero MuZero training loop is obviously faster than
our custom two-seat current-policy loop.

This is a speed/setup check, not a learning claim.

## Terms

`stock train_muzero` means:

```text
lzero.entry.train_muzero
-> LightZero collector
-> LightZero GameSegment / replay buffer
-> LightZero learner loop
-> LightZero checkpoints
```

`custom two-seat` means:

```text
custom CurvyZero collector chooses both seats
-> CurvyTron advances once with the joint action
-> custom replay rows
-> custom target arrays
-> MuZeroPolicy.learn_mode.forward
```

The custom path uses LightZero policy/search/model pieces, but it does not call
`train_muzero` and it does not use LightZero's normal collector or replay buffer.

## Honest Path Labels

`source_state_fixed_opponent`:

- Calls stock `train_muzero`.
- One LightZero action controls player 0.
- The env supplies player 1 using a fixed straight policy today.
- This is the cleanest stock-loop CurvyTron speed control.
- It is not live current-policy self-play.

`source_state_joint_action`:

- Calls stock `train_muzero`.
- One scalar action maps to both players' actions.
- One env step is one real CurvyTron physical tick.
- It is centralized single-agent control, not competitive self-play.

`two-seat-selfplay`:

- Does not call stock `train_muzero`.
- One live LightZero policy chooses both seats from the same pre-step state.
- It preserves the simultaneous-action shape.
- It is still a custom trainer/profiler path until it feeds native LightZero
  replay or has a stronger target contract.

`frozen checkpoint opponent`:

- Useful future lane: learner plays against a named frozen policy snapshot.
- Current source-state stock wrapper path in the Modal launcher rejects frozen
  checkpoint opponents for `source_state_fixed_opponent`.
- The immediate quick profile should therefore use fixed-straight and label it
  correctly.

## Matched Profile Results

The first two runs used stock `train_muzero`. The third run used the custom
two-seat path with roughly the same number of policy roots / seat decisions and
the same number of learner updates. This is still not a learning comparison.

```text
B64
num_simulations=8
about 0.8k-1.0k policy roots / seat decisions
4 learner updates
background eval/GIF off
```

| Profile | Calls stock `train_muzero` | Wall | Policy roots / seat decisions | Learner updates | Search calls | Rows per search call | Search time | Learner time | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `source_state_fixed_opponent` | yes | `21.689s` | `818` | `4` | `27` | `30.30` | `1.035s` | `1.920s` | fixed-straight opponent, stock loop control |
| `source_state_joint_action` | yes | `19.261s` | `929` | `4` | `35` | `26.54` | `0.993s` | `1.890s` | centralized 9-action joint controller, stock loop control |
| `two-seat-selfplay` matched wait run | no | `19.674s` | `1024` | `4` | `8` | `128.00` | `2.056s` | `1.683s` | simultaneous current-policy collection, custom replay/target path |

Custom matched attempt:

```text
run_id=opt-custom-two-seat-matched-wait-b64-sim8-20260512
attempt_id=custom-twoseat-matched-wait-b64-sim8-20260512
physical CurvyTron ticks collected=8
policy_search_row_count=1024
iteration_wall_before_progress_sec=5.893
collect_total_sec=3.296
visual_stack_update_sec=0.240
observation_noise_sec + replay_observation_noise_sec=0.604
progress_commit_sec=4.410
```

Important unit warning: custom `total_steps_collected=8` means physical
CurvyTron ticks. It is not comparable to stock `env_steps_collected` as a raw
`steps/s` number. The custom run did `64 envs * 2 seats * 8 ticks = 1024`
policy/search rows.

Do not compare these directly to the old custom profile:

```text
opt-replaycache-fullloop-fast-b64-sim8-20260512
B64, sim8
12 iterations
768 collect steps
98304 policy search rows
48 learner updates
elapsed=234.900509s
```

That old run did much more work than the stock profiles above. It is useful
background, but it is not the matched custom comparison.

## Current Read

Stock `train_muzero` is healthy for CurvyTron controls. That matters. It proves
the stock LightZero collector/replay/learner path can run our visual CurvyTron
wrappers.

But the matched speed result does not say "stock is obviously faster." For this
tiny matched profile, the custom two-seat path is in the same wall-clock range
and batches many more seat decisions per search call.

That does not make custom two-seat a trusted learning path. It only says the
speed panic was the wrong conclusion. The real custom-path risk is target/replay
semantics, not obvious throughput collapse.

Frozen checkpoint opponents are not currently wired through the clean
source-state stock path. The current clean stock opponent path is
`source_state_fixed_opponent` with fixed-straight player 1. Do not claim a
stock frozen-opponent comparison until the launcher/env gate is changed and
profiled.

## Decision Rule

Use stock `source_state_fixed_opponent` and `source_state_joint_action` as
controls and sanity checks. They are useful, but neither is true simultaneous
current-policy self-play.

Keep custom `two-seat-selfplay` as the simultaneous current-policy profile path
for now. Do not call it a trusted Coach learning path until replay/target
semantics are fixed or parity-tested against native LightZero targets.

If source-state stock rendering dominates the stock profile, add a small,
explicit render-mode knob later. Do not silently compare browser-lines stock
against fast-direct custom as if the trainer loop alone caused the difference.
