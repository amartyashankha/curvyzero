# Next Gates

Purpose: define what must pass before another large CurvyTron learning run.

## Gate 1: Stock Fixed/Frozen Control

Run a small-to-medium `source_state_fixed_opponent` stock `train_muzero` curve.

Must show:

- `called_train_muzero=true`;
- stock collector/GameBuffer used;
- `to_play` contract stated;
- checkpoints saved;
- held-out survival and outcome curve against the same opponent source;
- reset randomness recorded.

This does not prove live self-play. It proves the visual CurvyTron env and
stock LightZero loop can learn anything at all.

## Gate 1B: Recent Frozen Opponent Route

If true same-tick live self-play remains architecturally expensive, test a
recent-frozen-opponent route while staying on stock `train_muzero`.

Current proof: waited CPU and GPU Modal canaries succeeded with the stock
trainer:

```text
run_id=stock-frozen-canary-source-state-s304-20260512
attempt_id=trainmuzero-frozen-denseiter32-tiny-wait-cpu-s304
entrypoint=lzero.entry.train_muzero
called_train_muzero=true
env_variant=source_state_fixed_opponent
opponent_policy_kind=frozen_lightzero_checkpoint
opponent_provider_load_ok=true
opponent_provider_load_strict=true

run_id=stock-frozen-gpu-base-canary-source-state-s304-20260512b
attempt_id=waited-gpu-base-single-env-frozen-ckpt-canary-20260512b
entrypoint=lzero.entry.train_muzero
called_train_muzero=true
env_manager_type=base
gpu=NVIDIA L4
opponent_provider_load_ok=true
opponent_provider_load_strict=true
```

This proves the stock frozen-opponent lane can call LightZero `train_muzero`
and strictly load a real checkpoint opponent on CPU and GPU. It does not prove
learning yet. Earlier `s301` and `s302` attempts failed and are superseded by
`s304`.

Must show:

- opponent checkpoint source and refresh rule;
- no learner access from inside `env.step`;
- stock `GameSegment` / `MuZeroGameBuffer` still used;
- eval against more than the training opponent;
- survival and sparse outcome curves reported separately.

This is a practical self-play-adjacent route, not exact same-current-policy
self-play.

## Gate 2: Stock Joint-Action Control

Run `source_state_joint_action` through stock `train_muzero`.

Must show:

- one LightZero action maps to one real physical tick;
- no fake pending rows;
- reward is strong enough to detect learning;
- survival and outcome improve across checkpoints.

This is centralized control, not competitive self-play.

## Gate 3: Native Replay Bridge

Before scaling true current-policy two-seat play again:

- convert per-seat physical-tick rows into native-compatible `GameSegment`
  objects;
- push those segments through `MuZeroGameBuffer`;
- compare sampled targets to hand-computed targets on a tiny known trajectory;
- only then run a small learning curve.

Concrete first test:

```text
2 seats, 3 physical ticks
joint actions: [(2, 0), (1, 2), (0, 1)]
seat 0 rewards: [1, 2, 4] -> expected values [7, 6, 4]
seat 1 rewards: [1, 0, -2] -> expected values [-1, -2, -2]
```

This test should import-skip `lzero` locally if LightZero is not installed, but
it must not fake a native replay pass.

Current local status:

- pure projection test passes: two seat-local trajectories, no pending rows,
  expected values `[7, 6, 4]` and `[-1, -2, -2]`;
- local bridge config now includes LightZero 0.2.0 `support_scale` metadata;
- local pytest passes the pure/config checks and import-skips the native
  LightZero half because `lzero` is not installed locally;
- Modal/LightZero runtime passes the native target assertion:
  `python -m pytest -q tests/test_curvytron_two_seat_native_replay_bridge.py::test_native_lightzero_segments_push_and_tick0_targets_when_api_allows`
  returned `1 passed in 14.18s`.

This clears the tiny parity gate. It does not yet wire the bridge into the old
custom two-seat trainer.

## Do Not Scale Yet

Do not scale `--mode two-seat-selfplay` as a learning proof unless one of these
is true:

- it calls stock `train_muzero`;
- it feeds native `GameSegment` / `MuZeroGameBuffer` in the actual training
  loop;
- its repo-owned target builder has parity tests against LightZero targets.
