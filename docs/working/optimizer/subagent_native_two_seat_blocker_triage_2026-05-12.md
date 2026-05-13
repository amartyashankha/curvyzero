# Native Two-Seat Blocker Triage - 2026-05-12

Scope: read-only triage for Optimizer. No code was edited.

## Short Answer

Optimizer should stay on the trusted stock frozen-opponent path:

```text
stock LightZero train_muzero
env_variant=source_state_fixed_opponent
opponent_policy_kind=frozen_lightzero_checkpoint
```

The native two-seat bridge is relevant later, but it is not the active
Optimizer lane.

## 1. Native Two-Seat Route

The route is a replay/target bridge for true simultaneous two-seat play, not a
revival of the old custom `--mode two-seat-selfplay` learner.

Intended shape:

1. collect both seat actions from the same pre-tick state;
2. step CurvyTron once with the joint action;
3. project that physical tick stream into two seat-local trajectories;
4. encode each seat trajectory as a native LightZero `GameSegment`;
5. push those segments through `MuZeroGameBuffer`;
6. let LightZero compute sampled reward/value/policy targets;
7. prove parity on tiny known traces before scaling.

The first toy trace is two seats and three physical ticks:

```text
joint actions: [(2, 0), (1, 2), (0, 1)]
seat 0 rewards: [1, 2, 4] -> values [7, 6, 4]
seat 1 rewards: [1, 0, -2] -> values [-1, -2, -2]
```

Important distinction: one physical tick becomes one transition per seat-local
trajectory. It should not store fake pending rows, and it should not store a
joint action as a seat-local action.

## 2. `support_scale` Blocker

Simple version: LightZero's MuZero target code needs to know the scalar-support
head width for reward and value. In this local bridge test that width is derived
from:

```text
2 * config.model.support_scale + 1
```

If `support_scale` is missing from the tiny bridge config, the native target
parity test cannot even build the dummy zero target model correctly. That makes
the test fail before it can answer the real question: "Do native sampled targets
match the hand-computed reward/value/policy targets?"

Current workspace read: this specific blocker appears fixed. The bridge config
now sets `support_scale=10`, `reward_support_range=(-10, 11, 1)`, and
`value_support_range=(-10, 11, 1)`. The test asserts that metadata directly.

Local check run:

```text
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -p no:cacheprovider tests/test_curvytron_two_seat_native_replay_bridge.py
2 passed, 1 skipped
```

The skipped half is expected locally when `lzero` is not installed. The current
Gate 3 doc also says the Modal/LightZero runtime passed the native target
assertion for the tiny trace. So if Coach saw "missing support_scale", that was
a test-config completeness blocker, not evidence that the bridge semantics were
wrong.

## 3. Optimizer Relevance Now

Do not pivot Optimizer to this route now.

Even if the tiny native parity proof is accepted, it only proves the toy bridge
contract. It does not wire the bridge into the old custom two-seat trainer, and
it does not prove a learning curve.

Optimizer's useful work remains stock frozen-opponent profiling: collector
width, CPU/GPU split, env/render attribution, replay/learner timing, and safe
throughput knobs on the path that actually calls `train_muzero`.

## 4. Speed Notes If This Becomes Trusted Later

Remember these profiling implications:

- replay volume becomes two seat-local transitions per physical tick;
- collection still needs two seat decisions per live two-player env tick, though
  those decisions can be batched as active seat rows;
- GameSegment conversion, buffer push, native sampling, and target construction
  become real timing buckets again;
- old custom two-seat timings are not apples-to-apples because they bypassed
  native `GameSegment` / `MuZeroGameBuffer`;
- GPU may help policy/search/learner, but env/render/bridge/replay packaging
  can remain CPU-visible;
- profile reports must label physical ticks, seat-local transitions, MCTS roots,
  buffer samples, and learner updates separately.

Bottom line: keep this in the mental model, but leave it off the main Optimizer
critical path until Coach promotes an integrated native-bridge trainer gate.
