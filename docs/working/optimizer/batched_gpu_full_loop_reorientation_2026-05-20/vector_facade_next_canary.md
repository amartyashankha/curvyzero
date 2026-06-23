# Vector Facade Next Canary

Date: 2026-05-20

Status: first stock-boundary profile canary passed; matched A/B and larger
topology rows are running.

## Plain Goal

The direct GPU observation surface is now fast and semantically less broken:
`direct_gray64 + simple_symbols` keeps the 12 bonus symbols and passed an H100
two-view adversarial CPU-direct parity canary.

That does **not** mean the stock trainer is using it. The current stock
LightZero env wrapper is scalar: each env owns one `VectorMultiplayerEnv` row
and updates one stack at a time. If we put the fast renderer behind that scalar
wrapper, the GPU batch disappears and the optimization is mostly wasted.

The next canary should preserve this shape:

```text
one VectorMultiplayerEnv with B rows
-> one batched renderer call for B rows x 2 player views
-> row/player stacks [B,2,4,64,64]
-> scalar LightZero-shaped timesteps only at the outside boundary
```

## Current Evidence

- `block_704_gray64` surface canary: `0.144s` median surface step.
- corrected `direct_gray64` surface canary: `0.0339s` median surface step.
- stock full-loop CPU-oracle RND profile: about `457 steps/s`, H100 max GPU
  util about `17%`.

Plain read: direct GPU observation is a real local observation win. The next
question is whether the full loop can keep the batch and whether observation is
still the Amdahl bottleneck after search, collector, replay, learner, and RND
timers are included.

## First Local Slice

The first local slice now exists in
`src/curvyzero/training/source_state_batched_observation_mock_collector.py` as
`BatchedSourceStateTrainerProfileLoop`. It keeps one
`SourceStateMultiplayerTrainerSurface(batch_size=B, player_count=2)` batched
until `surface.step()` has produced row/player policy observations, then
materializes `MockBaseEnvTimestep` rows for payload/RND/profiler work.

This is still not stock LightZero integration. It is a shape and semantics
canary for the next A/B.

Tests added:

- profile-loop row/player order and observation materialization;
- renderer-backed stack FIFO: reset frame shifts correctly after one step;
- dynamic renderer fail-closed checks for partial row requests, wrong player
  order, wrong output shape, and wrong dtype.

2026-05-20 critique fix: the Modal surface-facade profiler now uses
`surface_step.policy_observation`, `surface_step.policy_env_row`, and
`surface_step.policy_player` when building the mock LightZero payload. It no
longer blindly flattens every seat from `step.observation`. It also exposes RND
metrics in the surface-facade result and resets terminal rows after measuring a
terminal step. This still needs partial-autoreset stress tests before a default
change.

2026-05-20 scalar-action bridge update: a local profile-only
`BatchedLightZeroScalarActionBridge` now exists in
`src/curvyzero/training/source_state_batched_observation_mock_collector.py`.
It accepts LightZero-style scalar actions keyed by `scalar_env_id`, validates
that exactly the current ready ids are present, converts them into one
`[batch, player]` joint action, commits one batched CurvyTron step, and then
returns scalar timestep rows keyed by env id. This is still not stock LightZero
integration, but it proves the next boundary shape in plain code:

```text
scalar LightZero actions -> one batched joint CurvyTron step -> scalar timesteps
```

Focused local tests passed for env-id mapping, row/player order, joint-action
commit, missing/extra action rejection, and invalid-action rejection.

2026-05-20 manager facade update: `BatchedLightZeroProfileEnvManager` now wraps
that bridge with the small env-manager-shaped surface we need next:
`env_num`, `ready_obs`, `reset`, `step`, `seed`, `close`, and
`last_reset_info`. The bridge now returns timesteps for the scalar env ids that
were actually stepped, not only the next live policy rows. That matters because
terminal rows are no longer policy-ready, but LightZero still needs their
`done=true` timestep. Local terminal-autoreset tests now prove terminal
timesteps keep `final_observation` before rows are reset.

Focused local test target:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_mock_collector.py \
  tests/test_multiplayer_source_state_trainer_surface.py \
  tests/test_source_state_batched_observation_boundary_profile.py

70 passed, 2 skipped
```

## Smallest Safe Modal Canary

Extend the profile-only vector facade or base-manager-like object. It should:

- own one `SourceStateMultiplayerTrainerSurface(batch_size=B, player_count=2)`;
- use `observation_stack_backend=renderer_backed_profile`;
- use the direct GPU renderer as an explicit renderer object;
- materialize `B * 2` LightZero-shaped observations after the batched render;
- report `ready_obs`/step outputs in normal env-id order;
- keep `calls_train_muzero=false` unless we intentionally wrap it into a stock
  LightZero profile later;
- write no live-run artifacts and touch no overnight run state.

First opponent scope should stay simple: fixed-straight or deterministic
proactive opponent. Frozen checkpoint opponents add policy inference inside the
environment boundary and can reintroduce CUDA/subprocess issues.

## Required Gates

- no hidden CPU fallback;
- row order: env id maps to the intended source row;
- player view: player 0 and player 1 observations are not swapped;
- stack FIFO order;
- reset row mask behavior;
- terminal `final_observation` captured before row reuse;
- partial autoreset stress;
- action mask shape and values;
- RND latest-frame extraction;
- timing buckets for env step, render pack, H2D, device render, D2H, stack
  update, scalar timestep construction, pickle/payload proxy, policy/MCTS,
  replay, learner, and RND.

## Do Not Do

- Do not set scalar `policy_observation_backend=jax_gpu` as the optimization.
- Do not make this a trainer/tournament default yet.
- Do not claim a full-loop speedup from the surface canary.
- Do not use `body_circles_fast` as the replacement recommendation.

## Next Decision

The manager facade now has prewarmed H100 rows:

- B128 no-RND: median/p95 manager step about `0.0438s` / `0.0557s`.
- B256 no-RND: median/p95 manager step about `0.0755s` / `0.0983s`.
- B128 CUDA RND update10: median/p95 manager step about `0.0427s` /
  `0.0493s`, but RND train median about `0.253s`.

Plain read: the profile-only manager facade is no longer blocked on renderer
p95 noise. The next proof should call stock `train_muzero` with a repo-owned
profile env manager. Success means `called_train_muzero=true`, no scalar env
instances, one batched CurvyTron surface inside the manager, and nonzero
collector/search/replay/learner counters. If the wiring collapses back to
scalar envs, stop and fix the vector boundary before making any trainer
recommendation.

2026-05-20 update: that first stock-boundary proof passed in
`opt-batched-stock-canary-20260520a/envmgr-b16-sim2e`.

It ran stock `lzero.entry.train_muzero` and reached:

- `env_steps_collected=16384`;
- `mcts_search_calls=1024`;
- `mcts_search_root_sum=16384`;
- `replay_sample_calls=1`;
- `learner_train_calls=1`;
- H100 max utilization about `77%`.

The speed was only about `150.19 steps/s`, so do not recommend it as a training
default yet. The current read is:

```text
integration: proven for tiny profile canary
speed: not proven
next question: matched CPU-oracle/base control and larger C64 batched topology
```

The bridge fixes required to reach this point were:

- DI-engine manager registry visibility for `curvyzero_batched_profile`;
- fail-closed refusal to fall back to scalar env managers;
- stock-shaped manager properties such as `action_space`, spaces, `ready_obs_id`,
  `env_ref`, replay no-op, and random action;
- scalar timestep `to_play`;
- plain `float`/`bool` reward and done;
- terminal `eval_episode_return` fallback.

Matched follow-up rows changed the read:

| row | path | C | steps/s | read |
|---|---|---:|---:|---|
| `cpuoracle-base-c16-sim2-control` | base CPU oracle | 16 | `98.01` | scalar base control |
| `envmgr-b16-sim2e` | batched GPU profile manager | 16 | `150.19` | integration proof plus small speed win over base |
| `envmgr-b64-sim2` | batched GPU profile manager | 64 | `416.89` | bigger root/env batches help |
| `cpuoracle-subproc-c64-sim2-control` | subprocess CPU oracle | 64 | `883.03` | current training-like control still wins |

Plain recommendation from these rows: do not promote the current batched GPU
manager to Coach training. It works, but it is a one-process profile manager.
The stock subprocess CPU-oracle path still parallelizes the environment work
better. The next useful canary is not another renderer microbenchmark; it is a
C64 rerun with direct batched-manager step timing so the collection gap is no
longer opaque.
